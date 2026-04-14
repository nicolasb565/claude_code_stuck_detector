"""Extract per-step features from parsed session steps.

Feature computation logic is ported from abstract_trajectory.py and kept
self-contained here so the new pipeline has no dependency on the old src/.
"""

import json
import math
import os
import re
import tempfile
import zlib
from datetime import datetime, timezone

# Schema 5: Phase 2 — adds 3 new feature dimensions on top of Schema 4's
# multi-slot output history. The new features address LLVM-style stuck
# patterns where the agent uses different commands to circle the same
# code area, which the original 7-feature set could not detect.
#
#   file_repeat_count_norm — count of prior steps touching any of the
#     file paths the current step references (extracted from cmd via
#     regex). Catches "agent reads VPlanTransforms.cpp 6 times across
#     different tools." Log1p-normalized so it stays in [0, 1].
#   cmd_hash_coarse — hash of the program name only (e.g. just "git",
#     just "grep"). Provides a SECOND has_prior_output binding at coarse
#     granularity, so the model gets independent signals for "this exact
#     command repeats" and "this command family repeats."
#   recent_token_jaccard — max Jaccard of the current command's token
#     set against the last K commands' token sets. Catches "semantically
#     related but technically different" exploration loops without any
#     hash binding at all.
#
# All three were validated in benchmarks/feature_experiments.py against
# Sonnet-labeled ground truth on the 10-task off-run: pooled LR AUC went
# from 0.7708 (schema 3 baseline) → 0.7826 (schema 4 multi-slot) →
# 0.8308 (schema 5 phase 2). Dominant new weight is file_repeat_count_norm.
SCHEMA_VERSION = 5
OUTPUT_HISTORY_SLOTS = 5
RECENT_TOKEN_HISTORY = 5

STEP_FEATURES = [
    "tool_idx",
    "cmd_hash",
    "file_hash",
    "output_similarity",
    "has_prior_output",
    "output_length",
    "is_error",
    "step_index_norm",
    # Phase 2:
    "file_repeat_count_norm",
    "cmd_hash_coarse",
    "recent_token_jaccard",
]

_CRC32_NORM = 1.0 / (1 << 32)  # map uint32 → [0, 1)

TOOL_NAMES = ["bash", "edit", "view", "search", "create", "submit", "other"]
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOL_NAMES)}

MAX_OUTPUT_LINES = 100

_SILENT_CMD_RE = re.compile(
    r"^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b"
)
_FILE_EXT_RE = re.compile(r"\.[a-zA-Z]{1,5}$")
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.I
)

# Phase 2 helpers
# Path-extraction regex: things in a command that look like file paths.
# Matches /abs/path/to/file.ext, ./relative/path, plain file.ext, dir/file.
_PATH_TOKEN_RE = re.compile(
    r"(?:/?[\w@.\-]+/)+[\w@.\-]+(?:\.[a-zA-Z0-9_]{1,8})?|[\w@.\-]+\.[a-zA-Z0-9_]{1,8}"
)
# Token splitter for recent_token_jaccard: word-like chunks of ≥2 chars.
_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z_][\w./\-]+|\b\d+\b")

ERROR_PATTERNS = re.compile(
    r"(error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied"
    r"|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError"
    r"|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError)",
    re.I,
)


def _cmd_semantic_key(cmd: str) -> str:
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not _SILENT_CMD_RE.match(p.strip())]
    if not real:
        return cmd.split()[0] if cmd.split() else ""
    first = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first.strip().split()
    if not tokens:
        return ""
    base = tokens[0].rsplit("/", 1)[-1]
    target = None
    for tok in tokens[1:]:
        if tok.startswith("-"):
            continue
        if _FILE_EXT_RE.search(tok) or "/" in tok:
            target = tok.rsplit("/", 1)[-1]
            break
    if target:
        return f"{base}:{target}"
    return base


def _normalize_to_set(output: str) -> frozenset:
    if not output:
        return frozenset()
    lines = output.strip().split("\n")[:MAX_OUTPUT_LINES]
    normalized = set()
    for line in lines:
        line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
        line = re.sub(r"pid[=: ]\d+", "pid=PID", line, flags=re.I)
        line = re.sub(r"/tmp/[^\s]+", "/tmp/TMPFILE", line)
        line = re.sub(r"\d+\.\d{3,}s", "N.NNNs", line)
        line = line.strip()
        if line:
            normalized.add(line)
    return frozenset(normalized)


def _jaccard(current_set: frozenset, previous_set: frozenset | None) -> float:
    if previous_set is None:
        return 0.0
    if not current_set and not previous_set:
        return 1.0
    union = current_set | previous_set
    return len(current_set & previous_set) / len(union) if union else 1.0


def _max_jaccard(current_set: frozenset, prior_sets: list | None) -> float:
    """Max Jaccard against any prior Set in the slot list. Mirrors
    maxJaccard() in proxy/features.mjs; both must agree exactly for
    train/inference parity."""
    if not prior_sets:
        return 0.0
    best = 0.0
    for p in prior_sets:
        j = _jaccard(current_set, p)
        if j > best:
            best = j
        if best >= 1.0:
            break
    return best


def _has_error_indicators(output: str) -> bool:
    if not output:
        return False
    return bool(ERROR_PATTERNS.search(output[:2000]))


def _strip_system_reminders(output: str) -> str:
    if not output or "<system-reminder" not in output:
        return output
    return _SYSTEM_REMINDER_RE.sub("", output)


def _extract_path_tokens(cmd: str) -> set[str]:
    """Pull file/path-like tokens out of a free-form command string."""
    if not cmd:
        return set()
    out = set()
    for m in _PATH_TOKEN_RE.finditer(cmd):
        tok = m.group(0).strip()
        if len(tok) < 2 or tok.isdigit():
            continue
        out.add(tok)
    return out


def _command_token_set(cmd: str) -> frozenset:
    """Token set for recent_token_jaccard. Strips flags and short tokens."""
    if not cmd:
        return frozenset()
    tokens = _TOKEN_SPLIT_RE.findall(cmd)
    return frozenset(t.lower() for t in tokens if not t.startswith("-") and len(t) >= 2)


def _coarse_program(cmd: str) -> str:
    """Just the program name from a bash command. 'git', 'grep', 'ninja', etc."""
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not _SILENT_CMD_RE.match(p.strip())]
    if not real:
        tokens = cmd.strip().split()
        return tokens[0].rsplit("/", 1)[-1] if tokens else ""
    first = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first.strip().split()
    if not tokens:
        return ""
    return tokens[0].rsplit("/", 1)[-1]


def compute_step_features(steps: list[dict]) -> list[dict]:
    """Compute per-step features from normalized step dicts.

    Args:
        steps: list of normalized step dicts (tool, cmd, file, output, thinking)

    Returns:
        list of feature dicts with exactly STEP_FEATURES fields
    """
    if not steps:
        return []

    total_steps = len(steps)
    result = []
    # cmd_hash_int → list of prior output sets, bounded to OUTPUT_HISTORY_SLOTS.
    output_history: dict[int, list] = {}
    # Phase 2 state
    file_touch_count: dict[str, int] = {}  # path → count of prior steps touching it
    recent_token_sets: list[frozenset] = []  # last RECENT_TOKEN_HISTORY token sets

    # Normalization constant for file_repeat_count_norm: log1p(50) ≈ 3.93.
    # Picked empirically — a step touching files seen 50 prior times is
    # close to the maximum we expect even on long sessions.
    _FILE_REPEAT_NORM = math.log1p(50)

    for i, step in enumerate(steps):
        tool = step.get("tool", "other")
        if tool not in TOOL_TO_IDX:
            tool = "other"

        file_path = step.get("file")
        cmd = step.get("cmd", "")
        output = step.get("output", "")

        # Compute hashes as unsigned 32-bit integers for identity comparisons
        file_hash_int = zlib.crc32(file_path.encode()) & 0xFFFFFFFF if file_path else None
        if tool == "bash" and cmd:
            cmd_key = _cmd_semantic_key(cmd)
            cmd_hash_int = (
                zlib.crc32(cmd_key.encode()) & 0xFFFFFFFF if cmd_key else None
            )
        else:
            cmd_key = f"{tool}:{cmd}" if cmd else None
            cmd_hash_int = (
                zlib.crc32(cmd_key.encode()) & 0xFFFFFFFF if cmd_key else None
            )

        output = _strip_system_reminders(output)

        is_edit_tool = tool in ("edit", "create", "submit")
        if is_edit_tool:
            output_set = frozenset()
            has_prior = False
            output_sim = 0.0
        else:
            output_set = _normalize_to_set(output)
            priors = output_history.get(cmd_hash_int) if cmd_hash_int is not None else None
            has_prior = bool(priors)
            output_sim = _max_jaccard(output_set, priors)

        # ── Phase 2 features ─────────────────────────────────────────────────
        cur_paths = _extract_path_tokens(cmd)
        if file_path:
            cur_paths.add(file_path)
        if cur_paths:
            repeat_sum = sum(file_touch_count.get(p, 0) for p in cur_paths)
            file_repeat_count_norm = min(1.0, math.log1p(repeat_sum) / _FILE_REPEAT_NORM)
        else:
            file_repeat_count_norm = 0.0

        if tool == "bash":
            coarse_str = _coarse_program(cmd)
        else:
            coarse_str = step.get("tool_name") or tool
        cmd_hash_coarse = (
            float((zlib.crc32(coarse_str.encode()) & 0xFFFFFFFF) * _CRC32_NORM)
            if coarse_str else 0.0
        )

        cur_tokens = _command_token_set(cmd)
        recent_token_jaccard = 0.0
        if cur_tokens and recent_token_sets:
            for prev in recent_token_sets:
                if not prev:
                    continue
                inter = len(cur_tokens & prev)
                union = len(cur_tokens | prev)
                if union > 0:
                    j = inter / union
                    if j > recent_token_jaccard:
                        recent_token_jaccard = j

        feat = {
            "tool_idx": TOOL_TO_IDX[tool],
            "cmd_hash": float(cmd_hash_int * _CRC32_NORM) if cmd_hash_int is not None else 0.0,
            "file_hash": float(file_hash_int * _CRC32_NORM) if file_hash_int is not None else 0.0,
            "output_similarity": float(output_sim),
            "has_prior_output": 1.0 if has_prior else 0.0,
            "output_length": float(math.log1p(output.count("\n"))),
            "is_error": 1.0 if _has_error_indicators(output) else 0.0,
            "step_index_norm": float(i) / float(max(total_steps - 1, 1)),
            "file_repeat_count_norm": float(file_repeat_count_norm),
            "cmd_hash_coarse": float(cmd_hash_coarse),
            "recent_token_jaccard": float(recent_token_jaccard),
        }
        result.append(feat)

        # Update state AFTER recording (don't count current step in its own features)
        if cmd_hash_int is not None and not is_edit_tool:
            slots = output_history.get(cmd_hash_int)
            if slots is None:
                output_history[cmd_hash_int] = [output_set]
            else:
                slots.append(output_set)
                if len(slots) > OUTPUT_HISTORY_SLOTS:
                    slots.pop(0)
        for p in cur_paths:
            file_touch_count[p] = file_touch_count.get(p, 0) + 1
        recent_token_sets.append(cur_tokens)
        if len(recent_token_sets) > RECENT_TOKEN_HISTORY:
            recent_token_sets.pop(0)

    return result


def _is_valid_feature_file(path: str, n_steps: int) -> bool:
    """Check if a feature file is valid and matches expected step count."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("schema_version") == SCHEMA_VERSION
            and data.get("n_steps") == n_steps
            and len(data.get("steps", [])) == n_steps
        )
    except (json.JSONDecodeError, OSError):
        return False


def extract_session(
    steps: list[dict],
    session_id: str,
    source: str,
    out_dir: str,
    force: bool = False,
) -> str:
    """Extract features for one session. Returns path to feature file.

    Idempotent: skip if valid feature file exists (unless force=True).
    Validates: JSON validity, len(steps)==n_steps, schema_version match.

    Args:
        steps: list of normalized step dicts
        session_id: session identifier
        source: source name
        out_dir: output directory for feature files
        force: re-extract even if valid file exists

    Returns:
        path to the feature file
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{session_id}_features.json")

    n_steps = len(steps)

    if not force and _is_valid_feature_file(out_path, n_steps):
        return out_path

    step_features = compute_step_features(steps)

    data = {
        "session_id": session_id,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "n_steps": n_steps,
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": step_features,
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=out_dir, delete=False, suffix=".tmp", encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return out_path
