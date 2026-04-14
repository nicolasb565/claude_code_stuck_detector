#!/usr/bin/env python3
"""Compare two benchmark run dirs and print a per-task duration table.

Usage:
    python benchmarks/compare.py results/run_001 results/run_002

Each run dir contains per-task subdirs with summary_N.json files. We report
the median duration per task in each run, the delta, and (if proxy_events.jsonl
is present in the "on" run) the number of nudges fired per task.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


_METRIC_FIELDS = (
    "duration_seconds",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "total_cost_usd",
    "num_turns",
)


def load_task_metrics(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Returns {task_id: {metric_name: [value, ...]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        bucket: dict[str, list[float]] = {m: [] for m in _METRIC_FIELDS}
        for summary in sorted(task_dir.glob("summary_*.json")):
            try:
                data = json.loads(summary.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for m in _METRIC_FIELDS:
                v = data.get(m)
                if isinstance(v, (int, float)):
                    bucket[m].append(float(v))
        if any(bucket.values()):
            out[task_id] = bucket
    return out


def count_nudges_per_task(run_dir: Path) -> dict[str, int]:
    """
    Attribute nudge_injected events to tasks by matching the event's
    `sessionKeyPrefix` (first 64 chars of the agent's first user message)
    against each task.md's first 64 chars.

    Looks for events in both the per-run proxy_logs/ dir (new layout,
    isolated by LOG_DIR) and a legacy `proxy_events.jsonl` file (older
    runs that pre-dated LOG_DIR isolation).
    """
    tasks_dir = Path(__file__).parent / "tasks"
    prefix_to_task: dict[str, str] = {}
    if tasks_dir.exists():
        for task_md in tasks_dir.glob("*/task.md"):
            prefix = task_md.read_text()[:64]
            prefix_to_task[prefix] = task_md.parent.name

    def _attribute(prefix: str) -> str | None:
        if not prefix:
            return None
        for p, tid in prefix_to_task.items():
            if prefix.startswith(p) or p.startswith(prefix):
                return tid
        return None

    event_files: list[Path] = []
    plog = run_dir / "proxy_logs"
    if plog.is_dir():
        event_files.extend(sorted(plog.glob("events-*.jsonl")))
    legacy = run_dir / "proxy_events.jsonl"
    if legacy.exists():
        event_files.append(legacy)

    nudges: dict[str, int] = {}
    for f in event_files:
        for line in f.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "nudge_injected":
                continue
            task_id = _attribute(ev.get("sessionKeyPrefix", ""))
            if task_id:
                nudges[task_id] = nudges.get(task_id, 0) + 1
    return nudges


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def fmt_dur(x: float) -> str:
    if x != x:
        return "—"
    if x < 60:
        return f"{x:.0f}s"
    return f"{x/60:.1f}m"


def fmt_int(x: float) -> str:
    if x != x:
        return "—"
    if x >= 1000:
        return f"{x/1000:.0f}k"
    return f"{x:.0f}"


def fmt_usd(x: float) -> str:
    if x != x:
        return "—"
    return f"${x:.2f}"


def delta_pct(before: float, after: float) -> str:
    if before != before or after != after or before == 0:
        return "—"
    return f"{(after - before) / before * 100:+.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("off_dir", type=Path, help="baseline run dir (proxy off)")
    ap.add_argument("on_dir", type=Path, help="treatment run dir (proxy on)")
    args = ap.parse_args()

    for d in (args.off_dir, args.on_dir):
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 1

    off = load_task_metrics(args.off_dir)
    on = load_task_metrics(args.on_dir)
    nudges = count_nudges_per_task(args.on_dir)

    tasks = sorted(set(off) | set(on))
    if not tasks:
        print("no tasks found in either run dir", file=sys.stderr)
        return 1

    # Single wide table: duration, total tokens (in+out+cache), cost, turns, nudges
    cols = [
        ("task",       22, lambda t: t),
        ("dur off",    10, lambda t: fmt_dur(median(off.get(t, {}).get("duration_seconds", [])))),
        ("dur on",     10, lambda t: fmt_dur(median(on.get(t, {}).get("duration_seconds", [])))),
        ("Δdur",       8,  lambda t: delta_pct(
            median(off.get(t, {}).get("duration_seconds", [])),
            median(on.get(t, {}).get("duration_seconds", [])))),
        ("out tok off", 12, lambda t: fmt_int(median(off.get(t, {}).get("output_tokens", [])))),
        ("out tok on",  12, lambda t: fmt_int(median(on.get(t, {}).get("output_tokens", [])))),
        ("Δout",       8,  lambda t: delta_pct(
            median(off.get(t, {}).get("output_tokens", [])),
            median(on.get(t, {}).get("output_tokens", [])))),
        ("$ off",      8,  lambda t: fmt_usd(median(off.get(t, {}).get("total_cost_usd", [])))),
        ("$ on",       8,  lambda t: fmt_usd(median(on.get(t, {}).get("total_cost_usd", [])))),
        ("turns",      6,  lambda t: fmt_int(median(on.get(t, {}).get("num_turns", [])))),
        ("nudges",     7,  lambda t: str(nudges.get(t, 0))),
    ]

    header = "".join(f"{name:<{w}}" for name, w, _ in cols)
    print(header)
    print("-" * len(header))
    for t in tasks:
        row = "".join(f"{fn(t):<{w}}" for _, w, fn in cols)
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
