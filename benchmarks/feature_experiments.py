#!/usr/bin/env python3
"""
Feature-extraction experiments: compare approaches to fixing the v5 MLP's
blind spots on the LLVM stuck pattern. Each variant produces a 7-feature
vector per step and the harness scores each variant's features against
Sonnet's per-step labels.

Variants:
  v0_current    — the current production features (mirror of features.mjs)
  v1_multi_slot — multi-slot output history (K=5), max Jaccard
  v2_token_hash — richer cmd_hash: sorted non-flag tokens for bash, full
                  structured input JSON for native tools
  v3_bash_parse — use bash-parser (via node subprocess) to extract
                  program+subcommand+args from the AST
  v4_combined   — v1 + v2
  v5_scope_key  — group commands by (tool, directory_scope) where scope
                  is the first absolute/relative path in the command,
                  truncated to 4 path components

For each variant:
  - recompute features on every step of a transcript
  - score each step with a naive heuristic (has_prior && output_sim>0.3,
    or high repetition signal) as a stand-in for "this variant would flag"
  - report precision/recall against Sonnet's STUCK labels
  - report the recall gain on Sonnet-STUCK / MLP-low steps (the 39 misses)

Usage:
  python benchmarks/feature_experiments.py
  python benchmarks/feature_experiments.py --task 03_llvm_loop_vec
  python benchmarks/feature_experiments.py --task all --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import zlib
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent

# ─── Shared helpers (ported from extract_features.py / features.mjs) ────────

SILENT_CMD_RE = re.compile(
    r"^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b"
)
FILE_EXT_RE = re.compile(r"\.[a-zA-Z]{1,5}$")
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.I)
ERROR_PATTERNS = re.compile(
    r"(error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied"
    r"|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError"
    r"|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError)",
    re.I,
)
PATH_IN_CMD = re.compile(r"(?:/[\w.\-]+){2,}")  # rough: /a/b/c
MAX_OUTPUT_LINES = 100
CRC32_NORM = 1.0 / (1 << 32)

TOOL_NAMES = ["bash", "edit", "view", "search", "create", "submit", "other"]
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOL_NAMES)}
TOOL_MAP = {
    "Bash": "bash", "bash": "bash",
    "Edit": "edit", "edit": "edit", "Write": "edit", "write": "edit", "MultiEdit": "edit",
    "Read": "view", "read": "view",
    "Grep": "search", "grep": "search", "Glob": "search", "glob": "search",
    "Agent": "other", "Task": "other", "TodoRead": "other", "TodoWrite": "other",
}
EDIT_TOOLS = {"edit", "create", "submit"}


def crc32_float(s: str) -> float:
    return (zlib.crc32(s.encode()) & 0xFFFFFFFF) * CRC32_NORM


def strip_system_reminders(text: str) -> str:
    if not text or "<system-reminder" not in text:
        return text
    return SYSTEM_REMINDER_RE.sub("", text)


def normalize_to_set(output: str) -> frozenset:
    if not output:
        return frozenset()
    lines = output.strip().split("\n")[:MAX_OUTPUT_LINES]
    out = set()
    for line in lines:
        line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
        line = re.sub(r"pid[=: ]\d+", "pid=PID", line, flags=re.I)
        line = re.sub(r"/tmp/[^\s]+", "/tmp/TMPFILE", line)
        line = re.sub(r"\d+\.\d{3,}s", "N.NNNs", line)
        line = line.strip()
        if line:
            out.add(line)
    return frozenset(out)


def jaccard(a: frozenset, b: frozenset | None) -> float:
    if b is None:
        return 0.0
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# ─── Step normalization (stream-json → {tool_name, input, output}) ──────────

def parse_transcript(path: Path) -> list[dict]:
    """Parse a stream-json transcript into an ordered list of step dicts."""
    tool_uses: dict[str, dict] = {}
    results: dict[str, str] = {}
    order = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", []) or []
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "tool_use":
                tool_uses[block.get("id")] = {
                    "name": block.get("name", "?"),
                    "input": block.get("input", {}) or {},
                    "_order": order,
                }
                order += 1
            elif t == "tool_result":
                c = block.get("content", "")
                if isinstance(c, list):
                    txt = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
                else:
                    txt = str(c or "")
                results[block.get("tool_use_id")] = txt
    steps: list[dict] = []
    for tid, tu in sorted(tool_uses.items(), key=lambda kv: kv[1]["_order"]):
        tool_class = TOOL_MAP.get(tu["name"], "other")
        inp = tu["input"]
        cmd = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("path") or ""
        file = inp.get("file_path") or inp.get("path") or None
        steps.append({
            "tool_name": tu["name"],
            "tool": tool_class,
            "input": inp,
            "cmd": str(cmd),
            "file": str(file) if file else None,
            "output": results.get(tid, ""),
        })
    return steps


# ─── Feature variants ──────────────────────────────────────────────────────

FEATURE_FIELDS = (
    "tool_idx", "cmd_hash", "file_hash", "output_similarity",
    "has_prior_output", "output_length", "is_error",
)
PHASE2_FIELDS = (
    "file_repeat_count_norm", "cmd_hash_coarse", "recent_token_jaccard",
)
# Path-extraction regex: things that look like file paths in a command.
# Catches /abs/path/to/file.ext, ./relative/path, plain file.ext, dir/file.
PATH_TOKEN_RE = re.compile(
    r"(?:/?[\w@.\-]+/)+[\w@.\-]+(?:\.[a-zA-Z0-9_]{1,8})?|[\w@.\-]+\.[a-zA-Z0-9_]{1,8}"
)
# Token splitter for token-jaccard feature: word-like chunks of >=2 chars
TOKEN_SPLIT_RE = re.compile(r"[A-Za-z_][\w./\-]+|\b\d+\b")


@dataclass
class Features:
    tool_idx: float
    cmd_hash: float
    file_hash: float
    output_similarity: float
    has_prior_output: float
    output_length: float
    is_error: float
    # Phase 2 candidate features. Default 0 so v0–v5 variants don't see them.
    file_repeat_count_norm: float = 0.0
    cmd_hash_coarse: float = 0.0
    recent_token_jaccard: float = 0.0
    _meta_key: str = ""  # for debugging


def _current_cmd_semantic_key(cmd: str) -> str:
    """The current production key — the buggy one."""
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not SILENT_CMD_RE.match(p.strip())]
    if not real:
        tokens = cmd.strip().split()
        return tokens[0] if tokens else ""
    first = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first.strip().split()
    if not tokens:
        return ""
    base = tokens[0].rsplit("/", 1)[-1]
    target = None
    for tok in tokens[1:]:
        if tok.startswith("-"):
            continue
        if FILE_EXT_RE.search(tok) or "/" in tok:
            target = tok.rsplit("/", 1)[-1]
            break
    return f"{base}:{target}" if target else base


def _token_set_key(cmd: str) -> str:
    """v2: Hash a canonicalized token set — all non-flag tokens sorted."""
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not SILENT_CMD_RE.match(p.strip())]
    if not real:
        tokens = cmd.strip().split()
        return tokens[0] if tokens else ""
    first = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = re.findall(r"\"[^\"]*\"|'[^']*'|\S+", first)
    non_flag = []
    for t in tokens:
        if t.startswith("-") and not re.match(r"^-?\d+(\.\d+)?$", t):  # keep numbers
            # strip flag=value forms to keep the value
            if "=" in t:
                _, _, v = t.partition("=")
                if v:
                    non_flag.append(v.strip('"').strip("'"))
            continue
        non_flag.append(t.strip('"').strip("'"))
    if not non_flag:
        return ""
    base = non_flag[0].rsplit("/", 1)[-1]
    rest = sorted(set(non_flag[1:]))
    return base + "|" + " ".join(rest)


def _scope_key(cmd: str) -> str:
    """v5: Group by (program, directory_scope). Scope = first /a/b/c/d path
    in the command, truncated to 4 components. Commands touching the same
    subtree get the same key regardless of exact invocation."""
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not SILENT_CMD_RE.match(p.strip())]
    first = re.split(r"\s*\|\s*", real[0].strip())[0] if real else cmd.strip()
    tokens = first.split()
    if not tokens:
        return ""
    base = tokens[0].rsplit("/", 1)[-1]
    # Find first path-like token
    path_match = PATH_IN_CMD.search(first)
    scope = ""
    if path_match:
        path = path_match.group(0)
        segments = path.split("/")
        scope = "/".join(segments[:5])  # /, a, b, c, d → first 4 components
    return f"{base}@{scope}" if scope else base


_BASH_PARSE_CACHE: dict[str, dict | None] = {}


def _bash_parse(cmd: str) -> dict | None:
    """v3: Use bash-parser (via node) to extract a structured key."""
    if not cmd:
        return None
    if cmd in _BASH_PARSE_CACHE:
        return _BASH_PARSE_CACHE[cmd]
    try:
        script = r"""
import('bash-parser').then(m => {
  const parse = m.default;
  process.stdin.resume();
  let buf = '';
  process.stdin.on('data', d => buf += d);
  process.stdin.on('end', () => {
    try {
      const ast = parse(buf);
      const cmds = [];
      const walk = (node) => {
        if (!node || typeof node !== 'object') return;
        if (node.type === 'Command' && node.name) {
          const name = node.name.text;
          const suffix = (node.suffix || []).map(s => s.text || '').filter(Boolean);
          cmds.push({ name, suffix });
        }
        for (const k of Object.keys(node)) {
          if (Array.isArray(node[k])) node[k].forEach(walk);
          else if (typeof node[k] === 'object') walk(node[k]);
        }
      };
      walk(ast);
      process.stdout.write(JSON.stringify(cmds));
    } catch (e) {
      process.stdout.write('null');
    }
  });
}).catch(e => { process.stdout.write('null'); });
"""
        result = subprocess.run(
            ["node", "-e", script],
            input=cmd,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(REPO / "proxy"),
        )
        parsed = json.loads(result.stdout) if result.stdout.strip() else None
        _BASH_PARSE_CACHE[cmd] = parsed
        return parsed
    except Exception:
        _BASH_PARSE_CACHE[cmd] = None
        return None


def _bash_parser_key(cmd: str) -> str:
    """Extract first command's (name, first non-flag arg) from bash-parser AST."""
    parsed = _bash_parse(cmd)
    if not parsed:
        return _current_cmd_semantic_key(cmd)  # fallback
    if len(parsed) == 0:
        return _current_cmd_semantic_key(cmd)
    first = parsed[0]
    name = first.get("name", "")
    if not name:
        return _current_cmd_semantic_key(cmd)
    base = name.rsplit("/", 1)[-1]
    # Find first subcommand (non-flag) and first file-ish token
    subcommand = None
    target = None
    for s in first.get("suffix", []):
        if s.startswith("-"):
            continue
        if subcommand is None:
            subcommand = s
        if FILE_EXT_RE.search(s) or "/" in s:
            target = s.rsplit("/", 1)[-1]
            break
    parts = [base]
    if subcommand and subcommand != target:
        parts.append(subcommand)
    if target:
        parts.append(target)
    return ":".join(parts)


# ─── Feature computation engines, one per variant ──────────────────────────

def _extract_path_tokens(text: str) -> set[str]:
    """Pull file/path-like tokens out of a free-form command string."""
    if not text:
        return set()
    out = set()
    for m in PATH_TOKEN_RE.finditer(text):
        tok = m.group(0).strip()
        # Drop things that are clearly NOT files: pure digits, single chars
        if len(tok) < 2:
            continue
        if tok.isdigit():
            continue
        # Normalize to bare basename for deduplication, but also keep the
        # full path so /a/b/foo.cpp and /c/d/foo.cpp are different files.
        out.add(tok)
    return out


def _command_token_set(cmd: str) -> frozenset[str]:
    """Token set for the recent-jaccard feature. Strips flags."""
    if not cmd:
        return frozenset()
    tokens = TOKEN_SPLIT_RE.findall(cmd)
    return frozenset(t.lower() for t in tokens if not t.startswith("-") and len(t) >= 2)


def _coarse_program(cmd: str) -> str:
    """Just the program name, no args, no path. 'git', 'grep', 'ninja', etc."""
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not SILENT_CMD_RE.match(p.strip())]
    if not real:
        tokens = cmd.strip().split()
        if not tokens:
            return ""
        return tokens[0].rsplit("/", 1)[-1]
    first = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first.strip().split()
    if not tokens:
        return ""
    return tokens[0].rsplit("/", 1)[-1]


def _make_engine(key_fn: Callable[[str], str], multi_slot: bool, phase2: bool = False):
    """Return a function that computes features for a list of steps."""

    def engine(steps: list[dict]) -> list[Features]:
        output_history: dict[str, deque | frozenset] = {}
        # Phase 2 state
        file_touch_count: dict[str, int] = {}  # path → count of prior steps touching it
        recent_token_sets: deque[frozenset[str]] = deque(maxlen=5)
        results: list[Features] = []
        for step in steps:
            tool = step["tool"]
            tool_idx = TOOL_TO_IDX.get(tool, TOOL_TO_IDX["other"])

            # cmd key
            if tool == "bash" and step["cmd"]:
                cmd_key = "bash:" + key_fn(step["cmd"])
            elif step["cmd"]:
                # native tools: full structured input for v2+; else just primary
                if key_fn.__name__ in ("_token_set_key", "_bash_parser_key"):
                    inp = step["input"]
                    inp_str = json.dumps(inp, sort_keys=True)[:512]
                    cmd_key = f"{step['tool_name']}:{inp_str}"
                else:
                    cmd_key = f"{tool}:{step['cmd']}"
            else:
                cmd_key = ""

            cmd_hash = crc32_float(cmd_key) if cmd_key else 0.0
            file_hash = crc32_float(step["file"]) if step["file"] else 0.0

            clean = strip_system_reminders(step["output"] or "")
            is_edit = tool in EDIT_TOOLS
            cur_set = frozenset() if is_edit else normalize_to_set(clean)

            prior = output_history.get(cmd_key)
            if is_edit or prior is None:
                has_prior = 0.0
                sim = 0.0
            else:
                has_prior = 1.0
                if multi_slot:
                    sim = max((jaccard(cur_set, p) for p in prior), default=0.0)
                else:
                    sim = jaccard(cur_set, prior)

            # ── Phase 2 features (always computed; only populated if phase2=True) ──
            f_repeat = 0.0
            f_coarse = 0.0
            f_recent_jacc = 0.0
            if phase2:
                # Collect path-like tokens from the command + file field
                cur_paths = _extract_path_tokens(step["cmd"])
                if step["file"]:
                    cur_paths.add(step["file"])
                # file_repeat_count: sum of prior touches across these paths
                if cur_paths:
                    repeat_sum = sum(file_touch_count.get(p, 0) for p in cur_paths)
                    f_repeat = math.log1p(repeat_sum) / math.log1p(50)  # normalize ~[0,1]
                # cmd_hash_coarse: program name only for bash; tool name for native
                if tool == "bash":
                    coarse = _coarse_program(step["cmd"])
                else:
                    coarse = step["tool_name"]
                f_coarse = crc32_float(coarse) if coarse else 0.0
                # recent_token_jaccard: token similarity to last K commands' tokens (max)
                cur_tokens = _command_token_set(step["cmd"])
                if cur_tokens and recent_token_sets:
                    f_recent_jacc = max(
                        (jaccard(cur_tokens, prev) for prev in recent_token_sets),
                        default=0.0,
                    )
                recent_token_sets.append(cur_tokens)
                # Update file_touch_count *after* using it (don't count the current step)
                for p in cur_paths:
                    file_touch_count[p] = file_touch_count.get(p, 0) + 1

            feat = Features(
                tool_idx=float(tool_idx),
                cmd_hash=float(cmd_hash),
                file_hash=float(file_hash),
                output_similarity=float(sim),
                has_prior_output=float(has_prior),
                output_length=float(math.log1p(clean.count("\n"))),
                is_error=1.0 if (clean and ERROR_PATTERNS.search(clean[:2000])) else 0.0,
                file_repeat_count_norm=float(f_repeat),
                cmd_hash_coarse=float(f_coarse),
                recent_token_jaccard=float(f_recent_jacc),
                _meta_key=cmd_key[:80],
            )
            results.append(feat)

            if cmd_key and not is_edit:
                if multi_slot:
                    if not isinstance(prior, deque):
                        output_history[cmd_key] = deque([cur_set], maxlen=5)
                    else:
                        prior.append(cur_set)
                else:
                    output_history[cmd_key] = cur_set

        return results

    return engine


VARIANTS: dict[str, Callable[[list[dict]], list[Features]]] = {
    "v0_current":    _make_engine(_current_cmd_semantic_key, multi_slot=False),
    "v1_multi_slot": _make_engine(_current_cmd_semantic_key, multi_slot=True),
    "v2_token_hash": _make_engine(_token_set_key,             multi_slot=False),
    "v3_bash_parse": _make_engine(_bash_parser_key,           multi_slot=False),
    "v4_combined":   _make_engine(_token_set_key,             multi_slot=True),
    "v5_scope_key":  _make_engine(_scope_key,                 multi_slot=True),
    # Phase 2: v1 base + 3 new feature dimensions
    "v6_phase2":     _make_engine(_current_cmd_semantic_key, multi_slot=True, phase2=True),
}

# Variants that include Phase 2 features (use 10-dim vector instead of 7)
PHASE2_VARIANTS = {"v6_phase2"}


# ─── Scoring against Sonnet labels ─────────────────────────────────────────

def _naive_stuck_score(f: Features) -> float:
    """Simple heuristic stand-in for 'variant thinks this is stuck'."""
    return 0.7 * f.output_similarity + 0.3 * f.has_prior_output


def feature_vector(f: Features, phase2: bool = False) -> list[float]:
    base = [
        f.tool_idx, f.cmd_hash, f.file_hash, f.output_similarity,
        f.has_prior_output, f.output_length, f.is_error,
    ]
    if phase2:
        base.extend([f.file_repeat_count_norm, f.cmd_hash_coarse, f.recent_token_jaccard])
    return base


def logreg_auc(features_by_task: dict[str, list[Features]],
               labels_by_task: dict[str, list[str]],
               phase2: bool = False) -> tuple[float, dict]:
    """Train a logistic regression on pooled features and report AUC."""
    X, y = [], []
    for t in features_by_task:
        for feat, lbl in zip(features_by_task[t], labels_by_task[t]):
            if lbl == "UNSURE":
                continue
            X.append(feature_vector(feat, phase2=phase2))
            y.append(1 if lbl == "STUCK" else 0)
    if not X:
        return 0.0, {}
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        Xn = np.array(X, dtype=float)
        yn = np.array(y, dtype=int)
        maxes = np.maximum(Xn.max(axis=0), 1e-9)
        Xn /= maxes
        if yn.sum() == 0 or yn.sum() == len(yn):
            return 0.0, {"note": "degenerate class distribution"}
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(Xn, yn)
        probs = lr.predict_proba(Xn)[:, 1]
        auc = roc_auc_score(yn, probs)
        names = list(FEATURE_FIELDS) + (list(PHASE2_FIELDS) if phase2 else [])
        weights = dict(zip(names, lr.coef_[0].tolist()))
        return float(auc), weights
    except ImportError:
        return 0.0, {"note": "sklearn not available"}


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)


def score_variant(features: list[Features], labels: list[str], threshold: float) -> Score:
    sc = Score()
    n = min(len(features), len(labels))
    for i in range(n):
        hot = _naive_stuck_score(features[i]) >= threshold
        stuck = labels[i] == "STUCK"
        if hot and stuck:
            sc.tp += 1
        elif hot and not stuck:
            sc.fp += 1
        elif (not hot) and stuck:
            sc.fn += 1
        else:
            sc.tn += 1
    return sc


def run_task(task_dir: Path, verbose: bool = False) -> dict:
    transcript = task_dir / "transcript_1.jsonl"
    labels_path = task_dir / "sonnet_labels.json"
    if not transcript.exists() or not labels_path.exists():
        return {}
    steps = parse_transcript(transcript)
    sonnet = json.loads(labels_path.read_text())
    labels = sonnet["labels"]
    n = min(len(steps), len(labels))
    steps, labels = steps[:n], labels[:n]

    task_result: dict = {"task": task_dir.name, "n_steps": n, "variants": {}}
    for name, engine in VARIANTS.items():
        features = engine(steps)
        sc = score_variant(features, labels, threshold=0.30)
        # Also dump the distribution of output_similarity on Sonnet-STUCK steps
        stuck_sims = [features[i].output_similarity for i in range(n) if labels[i] == "STUCK"]
        active = sum(1 for s in stuck_sims if s > 0)
        task_result["variants"][name] = {
            "p": sc.precision,
            "r": sc.recall,
            "f1": sc.f1,
            "tp": sc.tp, "fp": sc.fp, "fn": sc.fn, "tn": sc.tn,
            "sonnet_stuck_with_prior_sim": active,
            "sonnet_stuck_total": len(stuck_sims),
        }
        if verbose:
            print(f"  {name:15s} p={sc.precision:.3f} r={sc.recall:.3f} f1={sc.f1:.3f} "
                  f"tp={sc.tp} fp={sc.fp} fn={sc.fn} "
                  f"active_on_stuck={active}/{len(stuck_sims)}")
    return task_result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--run-dir", default="benchmarks/results/comparison_off")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.task == "all":
        task_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())
    else:
        task_dirs = [run_dir / args.task]

    results = []
    for td in task_dirs:
        if not (td / "sonnet_labels.json").exists():
            continue
        if args.verbose:
            print(f"\n=== {td.name} ===")
        r = run_task(td, verbose=args.verbose)
        if r:
            results.append(r)

    # Aggregate naive-threshold scores
    print("\n" + "=" * 90)
    print("AGGREGATE (naive threshold 0.3 on 0.7*sim + 0.3*has_prior)")
    print("=" * 90)
    agg: dict[str, Score] = {k: Score() for k in VARIANTS}
    agg_active: dict[str, tuple[int, int]] = {k: (0, 0) for k in VARIANTS}
    for r in results:
        for name, v in r["variants"].items():
            agg[name].tp += v["tp"]
            agg[name].fp += v["fp"]
            agg[name].fn += v["fn"]
            agg[name].tn += v["tn"]
            a, t = agg_active[name]
            agg_active[name] = (a + v["sonnet_stuck_with_prior_sim"],
                                t + v["sonnet_stuck_total"])

    print(f"\n{'variant':<16}{'prec':<10}{'recall':<10}{'f1':<10}"
          f"{'tp':<6}{'fp':<6}{'fn':<6}{'active/stuck':<15}")
    for name in VARIANTS:
        sc = agg[name]
        a, t = agg_active[name]
        print(f"{name:<16}{sc.precision:<10.3f}{sc.recall:<10.3f}{sc.f1:<10.3f}"
              f"{sc.tp:<6}{sc.fp:<6}{sc.fn:<6}{a}/{t}")

    # Per-variant logistic-regression AUC against Sonnet labels
    print("\n" + "=" * 90)
    print("LOGISTIC REGRESSION AUC on pooled Sonnet labels")
    print("=" * 90)
    for name in VARIANTS:
        feats_by_task: dict[str, list[Features]] = {}
        labels_by_task: dict[str, list[str]] = {}
        for td in task_dirs:
            if not (td / "sonnet_labels.json").exists():
                continue
            steps = parse_transcript(td / "transcript_1.jsonl")
            labels = json.loads((td / "sonnet_labels.json").read_text())["labels"]
            n = min(len(steps), len(labels))
            feats_by_task[td.name] = VARIANTS[name](steps[:n])
            labels_by_task[td.name] = labels[:n]
        is_p2 = name in PHASE2_VARIANTS
        auc, weights = logreg_auc(feats_by_task, labels_by_task, phase2=is_p2)
        ndim = 10 if is_p2 else 7
        print(f"{name:<16}AUC={auc:.4f}  (ndim={ndim})")
        if weights and "note" not in weights:
            ws = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
            print("  weights:", " ".join(f"{k}={v:+.2f}" for k, v in ws))

    # Save full results to JSON
    out_path = run_dir / "feature_experiments.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nfull per-task results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
