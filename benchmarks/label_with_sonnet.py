#!/usr/bin/env python3
"""
Ground-truth check: send each benchmark transcript to claude-sonnet-4-6 for
per-step stuck labeling, using the same prompt that produced the v5 MLP
training data. Compare Sonnet's labels against the simulator's per-turn
MLP scores to find disagreements.

Why:
  The v5 MLP classifier fires zero nudges on the clean off-run. That could
  mean (a) the tasks aren't actually stuck, or (b) the classifier is
  missing real stuck patterns. Sonnet-as-reviewer is our independent check.

Usage:
  python benchmarks/label_with_sonnet.py [run_dir]

  Default run_dir is benchmarks/results/comparison_off.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# Repo root is the parent of this file's directory
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.label_session import (  # noqa: E402
    LABELER_MODEL,
    SYSTEM_PROMPT,
    format_transcript,
    parse_csv_labels,
)

THRESHOLD = 0.5  # classifier firing threshold


def stream_json_to_steps(transcript_path: Path) -> list[dict]:
    """
    Convert a Claude Code --output-format stream-json transcript into the
    list-of-step-dicts format label_session.format_transcript expects.

    Each step dict has: tool_name, cmd, output.
    """
    tool_uses: dict[str, dict] = {}
    results: dict[str, str] = {}

    for line in transcript_path.read_text().splitlines():
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
                inp = block.get("input", {}) or {}
                cmd = (
                    inp.get("command")
                    or inp.get("file_path")
                    or inp.get("pattern")
                    or inp.get("path")
                    or ""
                )
                tool_uses[block.get("id")] = {
                    "tool_name": block.get("name", "?"),
                    "cmd": str(cmd)[:500],
                    "output": "",
                    "_order": len(tool_uses),
                }
            elif t == "tool_result":
                tid = block.get("tool_use_id")
                c = block.get("content", "")
                if isinstance(c, list):
                    txt = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
                else:
                    txt = str(c or "")
                results[tid] = txt

    # Fill in outputs, preserve original order
    steps: list[dict] = []
    for tid, tu in sorted(tool_uses.items(), key=lambda kv: kv[1]["_order"]):
        tu.pop("_order")
        tu["output"] = results.get(tid, "")
        steps.append(tu)
    return steps


def get_mlp_scores(transcript_path: Path) -> list[float]:
    """Run the simulator and return the per-turn MLP scores."""
    result = subprocess.run(
        ["node", str(REPO / "proxy" / "simulate.mjs"), str(transcript_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return [e["score"] for e in data.get("events", [])]


def label_with_sonnet(transcript: str, n_steps: int, client) -> list[str]:
    """Send formatted transcript to Sonnet, parse labels."""
    resp = client.messages.create(
        model=LABELER_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return parse_csv_labels(text, n_steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="benchmarks/results/comparison_off",
                    help="run directory containing per-task transcript_1.jsonl files")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    import anthropic
    client = anthropic.Anthropic()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"not a dir: {run_dir}", file=sys.stderr)
        return 1

    print(f"{'task':<22}{'steps':<7}{'P':<5}{'S':<5}{'U':<5}"
          f"{'mlp>=0.5':<10}{'agree':<8}{'disagree':<10}")
    print("-" * 78)

    all_disagreements: list[tuple[str, int, str, float]] = []

    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        t = task_dir / "transcript_1.jsonl"
        if not t.exists():
            continue

        steps = stream_json_to_steps(t)
        if not steps:
            continue
        transcript, n_steps = format_transcript(steps)

        try:
            labels = label_with_sonnet(transcript, n_steps, client)
        except Exception as e:
            print(f"{task_dir.name:<22} LABEL FAILED: {type(e).__name__}: {e}")
            continue

        try:
            mlp_scores = get_mlp_scores(t)
        except Exception as e:
            print(f"{task_dir.name:<22} MLP SCORES FAILED: {e}")
            continue

        if len(mlp_scores) != n_steps:
            print(f"{task_dir.name:<22} WARN: step count mismatch "
                  f"(labels={n_steps}, mlp={len(mlp_scores)}) — using min")
        n = min(n_steps, len(mlp_scores))

        p = labels[:n].count("PRODUCTIVE")
        s = labels[:n].count("STUCK")
        u = labels[:n].count("UNSURE")
        mlp_stuck = sum(1 for sc in mlp_scores[:n] if sc >= THRESHOLD)
        agree = sum(1 for i in range(n)
                    if (labels[i] == "STUCK") == (mlp_scores[i] >= THRESHOLD))
        disagree = n - agree

        print(f"{task_dir.name:<22}{n:<7}{p:<5}{s:<5}{u:<5}{mlp_stuck:<10}"
              f"{agree:<8}{disagree:<10}")

        # Collect disagreements: Sonnet says STUCK but MLP doesn't (or vice versa)
        for i in range(n):
            sonnet_stuck = labels[i] == "STUCK"
            mlp_hot = mlp_scores[i] >= THRESHOLD
            if sonnet_stuck != mlp_hot:
                all_disagreements.append(
                    (task_dir.name, i, labels[i], mlp_scores[i])
                )

        # Persist the per-task labeled output for later inspection
        out = task_dir / "sonnet_labels.json"
        out.write_text(json.dumps({
            "task": task_dir.name,
            "n_steps": n,
            "labels": labels[:n],
            "mlp_scores": mlp_scores[:n],
        }, indent=2))

    # Summary of disagreements where Sonnet > MLP (Sonnet thinks stuck but MLP didn't)
    missed = [d for d in all_disagreements if d[2] == "STUCK"]
    fp = [d for d in all_disagreements if d[2] == "PRODUCTIVE"]
    print()
    print(f"Total Sonnet-STUCK, MLP-low: {len(missed)}  (potential classifier misses)")
    print(f"Total Sonnet-PRODUCTIVE, MLP-high: {len(fp)}  (potential classifier FPs)")
    if missed[:10]:
        print("\nFirst 10 Sonnet-STUCK that MLP missed:")
        for task, idx, label, score in missed[:10]:
            print(f"  {task} step {idx} mlp_score={score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
