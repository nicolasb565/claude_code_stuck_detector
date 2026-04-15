#!/usr/bin/env python3
"""
Apply N-consecutive hindsight relabeling to the training dataset.

For each session's multi-turn chat, rewrite the assistant label from S to
P if the step isn't at least the Nth consecutive S in the original label
sequence. This creates a training set where "stuck" means "sustained
stuck" and the labels are causally learnable (a classifier looking at
past-only context can in principle see that N consecutive similar actions
have happened and commit to S).

Rationale and simulation in project_causal_ceiling.md. Simulation on
03_llvm_loop_vec: relabeling doesn't automatically improve causal
Sonnet's F1 on existing labels, but the hypothesis is that a classifier
trained specifically on the relabeled data will converge to a cleaner,
consistent function since the training signal no longer mixes "start-of-
loop (unreachable)" with "middle-of-loop (reachable)" cases.

Usage:
  .venv/bin/python benchmarks/relabel_train_data.py --n 5
  .venv/bin/python benchmarks/relabel_train_data.py --n 3 --inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def extract_labels(messages: list[dict]) -> list[str]:
    """Return the assistant labels in order."""
    labels = []
    for m in messages:
        if m.get("role") == "assistant":
            c = (m.get("content") or "").strip()
            if c:
                labels.append(c[0])  # first char: P / S / U
    return labels


def relabel_consecutive(labels: list[str], n: int) -> list[str]:
    """Relabel: position i stays S iff the last N positions ending at i are all S.
    Everything else becomes P. U labels are left unchanged (dropped at
    training time anyway).
    """
    new = []
    for i in range(len(labels)):
        old = labels[i]
        if old == "U":
            new.append("U")
            continue
        start = max(0, i - n + 1)
        window = labels[start:i + 1]
        if len(window) >= n and all(l == "S" for l in window):
            new.append("S")
        else:
            new.append("P")
    return new


def rewrite_messages(messages: list[dict], new_labels: list[str]) -> list[dict]:
    """Return a new messages list with assistant labels replaced by new_labels."""
    out = []
    idx = 0
    for m in messages:
        if m.get("role") == "assistant" and idx < len(new_labels):
            out.append({**m, "content": new_labels[idx]})
            idx += 1
        else:
            out.append(m)
    return out


def relabel_jsonl(in_path: Path, out_path: Path, n: int, inspect: int = 0):
    """Read a JSONL file of sessions, apply N-consecutive relabeling, write
    out a new JSONL. Returns stats.
    """
    stats = {
        "n_sessions": 0,
        "orig_labels": Counter(),
        "new_labels": Counter(),
        "n_flipped_S_to_P": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inspected = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sess = json.loads(line)
            msgs = sess.get("messages", [])
            old_labels = extract_labels(msgs)
            new_labels = relabel_consecutive(old_labels, n)

            stats["n_sessions"] += 1
            for l in old_labels:
                stats["orig_labels"][l] += 1
            for l in new_labels:
                stats["new_labels"][l] += 1
            for o, nw in zip(old_labels, new_labels):
                if o == "S" and nw == "P":
                    stats["n_flipped_S_to_P"] += 1

            # Update messages and n_labeled count
            sess["messages"] = rewrite_messages(msgs, new_labels)
            # Keep n_labeled as count of P+S (drop U), matches training data prep
            sess["n_labeled"] = sum(1 for l in new_labels if l in ("P", "S"))
            fout.write(json.dumps(sess) + "\n")

            if inspected < inspect:
                print(f"\n--- session {sess.get('session_id')} ---")
                print(f"  old labels: {''.join(old_labels[:30])}"
                      f"{'...' if len(old_labels) > 30 else ''}")
                print(f"  new labels: {''.join(new_labels[:30])}"
                      f"{'...' if len(new_labels) > 30 else ''}")
                inspected += 1

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                    help="N consecutive S required for a step to stay S")
    ap.add_argument("--train-in", default="data/generated/finetune_train.jsonl")
    ap.add_argument("--val-in", default="data/generated/finetune_val.jsonl")
    ap.add_argument("--train-out", default=None)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--inspect", type=int, default=3,
                    help="print N session examples for sanity check")
    args = ap.parse_args()

    if args.train_out is None:
        args.train_out = args.train_in.replace(".jsonl", f"_relabel_n{args.n}.jsonl")
    if args.val_out is None:
        args.val_out = args.val_in.replace(".jsonl", f"_relabel_n{args.n}.jsonl")

    print(f"N = {args.n}")
    print(f"train: {args.train_in} → {args.train_out}")
    print(f"val:   {args.val_in} → {args.val_out}")
    print()

    train_stats = relabel_jsonl(
        REPO / args.train_in, REPO / args.train_out, args.n, args.inspect
    )
    val_stats = relabel_jsonl(
        REPO / args.val_in, REPO / args.val_out, args.n, 0
    )

    def pct(c, total):
        return f"{c} ({100*c/max(total,1):.1f}%)"

    print("\n=== train ===")
    orig = train_stats["orig_labels"]
    new = train_stats["new_labels"]
    orig_total = sum(orig.values())
    new_total = sum(new.values())
    print(f"  sessions: {train_stats['n_sessions']}")
    print(f"  orig labels: P={pct(orig['P'],orig_total)} "
          f"S={pct(orig['S'],orig_total)} "
          f"U={pct(orig['U'],orig_total)}")
    print(f"  new labels:  P={pct(new['P'],new_total)} "
          f"S={pct(new['S'],new_total)} "
          f"U={pct(new['U'],new_total)}")
    print(f"  S→P flipped: {train_stats['n_flipped_S_to_P']}")

    print("\n=== val ===")
    orig = val_stats["orig_labels"]
    new = val_stats["new_labels"]
    orig_total = sum(orig.values())
    new_total = sum(new.values())
    print(f"  sessions: {val_stats['n_sessions']}")
    print(f"  orig labels: P={pct(orig['P'],orig_total)} "
          f"S={pct(orig['S'],orig_total)} "
          f"U={pct(orig['U'],orig_total)}")
    print(f"  new labels:  P={pct(new['P'],new_total)} "
          f"S={pct(new['S'],new_total)} "
          f"U={pct(new['U'],new_total)}")
    print(f"  S→P flipped: {val_stats['n_flipped_S_to_P']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
