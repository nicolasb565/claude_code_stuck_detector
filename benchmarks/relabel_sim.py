#!/usr/bin/env python3
"""
Simulate N-consecutive hindsight relabeling on a session with both original
and causal Sonnet labels.

Idea (user's proposal, 2026-04-15): the current ground-truth labels include
steps at the START of stuck loops which are information-theoretically
unreachable for any causal classifier. Instead of matching Sonnet's full-
transcript labels exactly, relabel each step as S only if the last N
consecutive steps (including the current one) were originally-S. That is:
  new_label[i] = S  iff  all(original_label[i-k] == S for k in 0..N-1)
  new_label[i] = P  otherwise

Effects of this relabeling:
  - First N-1 steps of any stuck loop flip from S to P
  - Dense middles of stuck loops stay S
  - Isolated / sporadic S labels disappear (they can't sustain N in a row)
  - Intuitively: "only call it stuck when the agent has been stuck for N
    steps consecutively — we don't need to detect the start, just the
    persistence."

This script tests whether the new labels become causally reachable, using
the existing causal Sonnet labels as a proxy for what a causal classifier
could achieve.

Usage:
  .venv/bin/python benchmarks/relabel_sim.py --task 03_llvm_loop_vec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def relabel_consecutive(labels: list[str], n: int, stuck_label: str = "STUCK") -> list[str]:
    """Return new labels where step i is S iff original[i-N+1..i] are all S.
    Labels are assumed to be strings like 'STUCK' / 'PRODUCTIVE' / 'UNSURE'.
    """
    new = []
    for i in range(len(labels)):
        start = max(0, i - n + 1)
        window = labels[start:i + 1]
        if len(window) >= n and all(l == stuck_label for l in window):
            new.append("STUCK")
        else:
            new.append("PRODUCTIVE")
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="03_llvm_loop_vec")
    ap.add_argument("--n-values", nargs="+", type=int, default=[2, 3, 4, 5])
    args = ap.parse_args()

    causal_path = REPO / f"data/generated/causal_label_{args.task}.json"
    if not causal_path.exists():
        print(f"ERROR: {causal_path} not found")
        print(f"Run: benchmarks/causal_label.py --task {args.task}")
        return 1

    d = json.loads(causal_path.read_text())
    original = d["original_labels"]  # list of "PRODUCTIVE"/"STUCK"/"UNSURE"
    per_step = d["per_step"]
    causal = [r["causal"] for r in per_step]  # list of "P"/"S"/"U"
    n_steps = len(original)

    # Map causal to full name for comparison
    causal_full = ["STUCK" if c == "S" else ("PRODUCTIVE" if c == "P" else "UNSURE")
                   for c in causal]

    n_orig_s = original.count("STUCK")
    n_orig_p = original.count("PRODUCTIVE")
    n_causal_s = causal_full.count("STUCK")

    print(f"=== {args.task}: {n_steps} steps ===")
    print(f"Original labels: P={n_orig_p} S={n_orig_s}")
    print(f"Causal labels:   P={causal_full.count('PRODUCTIVE')} S={n_causal_s}")
    print()

    # Baseline agreement: how well do causal labels match original labels?
    agree_orig = sum(1 for o, c in zip(original, causal_full) if o == c)
    print(f"Causal vs ORIGINAL agreement: {agree_orig}/{n_steps} = {100*agree_orig/n_steps:.1f}%")

    # Baseline STUCK recall: fraction of original STUCK that causal also called STUCK
    tp_orig = sum(1 for o, c in zip(original, causal_full) if o == "STUCK" and c == "STUCK")
    print(f"STUCK recall against ORIGINAL: {tp_orig}/{n_orig_s} = {100*tp_orig/max(n_orig_s,1):.1f}%")
    print()

    # Sweep N
    print(f"{'N':>4}{'new_S':>8}{'fraction':>12}{'causal_vs_new_agree':>22}{'S_recall':>12}{'S_prec':>10}{'F1':>8}")
    for n in args.n_values:
        new = relabel_consecutive(original, n)
        new_s_count = new.count("STUCK")

        # Causal classifier "predictions" = causal_full
        # Target = new (relabeled)
        tp = sum(1 for nl, cl in zip(new, causal_full) if nl == "STUCK" and cl == "STUCK")
        fp = sum(1 for nl, cl in zip(new, causal_full) if nl == "PRODUCTIVE" and cl == "STUCK")
        fn = sum(1 for nl, cl in zip(new, causal_full) if nl == "STUCK" and cl == "PRODUCTIVE")
        # UNSURE in causal → count as wrong (neither TP nor FP)
        agree = sum(1 for nl, cl in zip(new, causal_full) if nl == cl)

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"{n:>4}{new_s_count:>8}{100*new_s_count/n_steps:>11.1f}%"
              f"{100*agree/n_steps:>21.1f}%"
              f"{100*recall:>11.1f}%{100*precision:>9.1f}%{f1:>8.3f}")

    # Also show what happens side-by-side on a few specific steps
    # where causal and original disagreed
    print(f"\n=== Per-step detail: original STUCK regions ===")
    print(f"(row = step; cols = original | causal | relabeled-N=3 | relabeled-N=5)")
    new_3 = relabel_consecutive(original, 3)
    new_5 = relabel_consecutive(original, 5)
    for i in range(n_steps):
        if original[i] == "STUCK" or i > 0 and original[i-1] == "STUCK":
            c = causal_full[i][0]
            o = original[i][0]
            n3 = new_3[i][0]
            n5 = new_5[i][0]
            mark = ""
            if n3 != o:
                mark = "← N=3 flipped"
            if n5 != n3:
                mark += " | N=5 flipped"
            print(f"  step {i:3d}: orig={o} causal={c} N3={n3} N5={n5}  {mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
