#!/usr/bin/env python3
"""
v9 feature ablation study: which features help or hurt OOD generalization?

Approach:
  1. Load both datasets:
     - In-dist: the v6 training JSONL rows (sampled to ~20k for speed)
     - OOD: the benchmark transcripts with Sonnet labels (n=680)
  2. Fit LR on each set with the full 34 features → baseline AUC
  3. For each feature (or feature group), ablate it (set to 0 in both
     sets) and refit LR → ablated AUC
  4. Report Δ_indist, Δ_ood for each ablation. Features where:
     - Δ_indist < 0 AND Δ_ood > 0: useless in-dist, actively harmful OOD
     - Δ_indist > 0 AND Δ_ood > 0: genuinely useful (ablating hurts both)
     - Δ_indist > 0 AND Δ_ood < 0: in-dist signal that doesn't transfer
     - Δ_indist < 0 AND Δ_ood < 0: bad feature everywhere

The key metric is Δ_ood - Δ_indist: how much more does OOD suffer from
ablation than in-dist? Positive → feature transfers well.

Usage:
  .venv/bin/python benchmarks/v9_ablation.py
  .venv/bin/python benchmarks/v9_ablation.py --sample-size 50000
  .venv/bin/python benchmarks/v9_ablation.py --groups  # ablate by group
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.extract_features import V9_FEATURE_NAMES, compute_step_features  # noqa: E402
from src.pipeline.parsers.nlile import parse_session  # noqa: E402


def load_indist_sample(sample_size: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load a balanced sample from the schema-6 v6 training JSONL files."""
    rng = random.Random(seed)
    paths = [
        "data/generated/nlile_v6.jsonl",
        "data/generated/dataclaw_claude_v6.jsonl",
        "data/generated/masterclass_v6.jsonl",
        "data/generated/claudeset_v6.jsonl",
    ]
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    rng.shuffle(rows)
    # Take balanced sample: half stuck, half productive
    stuck = [r for r in rows if r["label"] >= 0.9][: sample_size // 2]
    prod = [r for r in rows if r["label"] <= 0.1][: sample_size // 2]
    balanced = stuck + prod
    rng.shuffle(balanced)
    X = np.array(
        [[float(r[k]) for k in V9_FEATURE_NAMES] for r in balanced],
        dtype=np.float32,
    )
    y = np.array([1 if r["label"] >= 0.9 else 0 for r in balanced], dtype=np.int32)
    return X, y


def load_ood() -> tuple[np.ndarray, np.ndarray]:
    """Parse benchmark transcripts, compute v9 features, pair with Sonnet labels."""
    run_dir = Path("benchmarks/results/comparison_off")
    X, y = [], []
    for td in sorted(run_dir.iterdir()):
        if not td.is_dir():
            continue
        transcript = td / "transcript_1.jsonl"
        labels_path = td / "sonnet_labels.json"
        if not (transcript.exists() and labels_path.exists()):
            continue
        messages = []
        for line in transcript.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") in ("user", "assistant"):
                msg = ev.get("message", {})
                if isinstance(msg, dict):
                    messages.append(msg)
        steps = parse_session(messages)
        feats = compute_step_features(steps)
        labels = json.loads(labels_path.read_text())["labels"]
        n = min(len(feats), len(labels))
        for i in range(n):
            if labels[i] == "UNSURE":
                continue
            X.append([float(feats[i][k]) for k in V9_FEATURE_NAMES])
            y.append(1 if labels[i] == "STUCK" else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def fit_auc(X: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    maxes = np.maximum(np.abs(X).max(axis=0), 1e-9)
    Xn = X / maxes
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    lr.fit(Xn, y)
    probs = lr.predict_proba(Xn)[:, 1]
    return float(roc_auc_score(y, probs))


def mask_columns(X: np.ndarray, indices_to_zero: list[int]) -> np.ndarray:
    """Return a copy of X with the given column indices set to 0.0."""
    Xc = X.copy()
    for i in indices_to_zero:
        Xc[:, i] = 0.0
    return Xc


FEATURE_GROUPS = {
    "action_match_all": [i for i, n in enumerate(V9_FEATURE_NAMES) if "act_match" in n],
    "file_match_all":   [i for i, n in enumerate(V9_FEATURE_NAMES) if "file_match" in n],
    "scope_match_all":  [i for i, n in enumerate(V9_FEATURE_NAMES) if "scope_match" in n],
    "self_sim_all":     [i for i, n in enumerate(V9_FEATURE_NAMES) if "self_sim" in n],
    "p_out_len_all":    [i for i, n in enumerate(V9_FEATURE_NAMES) if n.startswith("v9_p") and "out_len" in n],
    "p_is_err_all":     [i for i, n in enumerate(V9_FEATURE_NAMES) if n.startswith("v9_p") and "is_err" in n],
    "cur_out_len":      [V9_FEATURE_NAMES.index("v9_cur_out_len")],
    "cur_is_err":       [V9_FEATURE_NAMES.index("v9_cur_is_err")],
    "cur_sim_vs_match": [V9_FEATURE_NAMES.index("v9_cur_sim_vs_match")],
    "cur_consec_match": [V9_FEATURE_NAMES.index("v9_cur_consec_match")],
}

# Per-slot ablations (remove only a specific history depth)
PER_SLOT_GROUPS = {
    f"slot_p{slot+1}_all": [
        i for i, n in enumerate(V9_FEATURE_NAMES)
        if n.startswith(f"v9_p{slot+1}_")
    ]
    for slot in range(5)
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=20000,
                    help="Balanced in-dist sample size (default 20k, half stuck)")
    ap.add_argument("--single", action="store_true",
                    help="Also run single-feature ablations (34 runs)")
    args = ap.parse_args()

    print(f"Loading in-dist sample of {args.sample_size} balanced rows...")
    X_ind, y_ind = load_indist_sample(args.sample_size)
    print(f"  {len(X_ind)} rows, {y_ind.sum()} stuck ({y_ind.mean()*100:.1f}%)")

    print("Loading OOD benchmark transcripts...")
    X_ood, y_ood = load_ood()
    print(f"  {len(X_ood)} rows, {y_ood.sum()} stuck ({y_ood.mean()*100:.1f}%)")
    print()

    baseline_ind = fit_auc(X_ind, y_ind)
    baseline_ood = fit_auc(X_ood, y_ood)
    print(f"BASELINE  in-dist AUC = {baseline_ind:.4f}   OOD AUC = {baseline_ood:.4f}")
    print()

    results = []
    for name, idxs in FEATURE_GROUPS.items():
        ablated_X_ind = mask_columns(X_ind, idxs)
        ablated_X_ood = mask_columns(X_ood, idxs)
        a_ind = fit_auc(ablated_X_ind, y_ind)
        a_ood = fit_auc(ablated_X_ood, y_ood)
        delta_ind = a_ind - baseline_ind
        delta_ood = a_ood - baseline_ood
        results.append({
            "group": name,
            "ablated_ind": a_ind,
            "ablated_ood": a_ood,
            "delta_ind": delta_ind,
            "delta_ood": delta_ood,
            "transfer_score": delta_ood - delta_ind,
            "n_features": len(idxs),
        })

    print("=" * 86)
    print(f"FEATURE GROUP ABLATION  (baseline in-dist {baseline_ind:.4f}, OOD {baseline_ood:.4f})")
    print("=" * 86)
    print(f"{'group':<22}{'#feat':>6}{'ind_auc':>10}{'ood_auc':>10}"
          f"{'Δ_ind':>10}{'Δ_ood':>10}{'transfer':>11}")
    print("-" * 86)
    # Sort by transfer score (positive = removing the feature helps OOD more than it helps indist)
    results.sort(key=lambda r: -r["transfer_score"])
    for r in results:
        print(f"{r['group']:<22}{r['n_features']:>6}"
              f"{r['ablated_ind']:>10.4f}{r['ablated_ood']:>10.4f}"
              f"{r['delta_ind']:>+10.4f}{r['delta_ood']:>+10.4f}"
              f"{r['transfer_score']:>+11.4f}")
    print()
    print("Reading the table:")
    print("  Δ_ind = AUC drop on in-distribution when this group is removed")
    print("    (negative = ablation hurts in-dist, so feature is load-bearing there)")
    print("  Δ_ood = AUC drop on OOD benchmark when this group is removed")
    print("    (positive = ablation HELPS OOD, so feature is hurting there)")
    print("  transfer = Δ_ood - Δ_ind")
    print("    (positive → feature hurts OOD more than it hurts in-dist → candidate for removal)")
    print()

    # Per-slot
    print("=" * 86)
    print("PER-SLOT ABLATION (remove all 6 features of a given history depth)")
    print("=" * 86)
    slot_results = []
    for name, idxs in PER_SLOT_GROUPS.items():
        a_ind = fit_auc(mask_columns(X_ind, idxs), y_ind)
        a_ood = fit_auc(mask_columns(X_ood, idxs), y_ood)
        slot_results.append({
            "group": name,
            "delta_ind": a_ind - baseline_ind,
            "delta_ood": a_ood - baseline_ood,
            "transfer_score": (a_ood - baseline_ood) - (a_ind - baseline_ind),
        })
    print(f"{'slot':<15}{'Δ_ind':>10}{'Δ_ood':>10}{'transfer':>11}")
    print("-" * 50)
    for r in slot_results:
        print(f"{r['group']:<15}{r['delta_ind']:>+10.4f}{r['delta_ood']:>+10.4f}"
              f"{r['transfer_score']:>+11.4f}")

    if args.single:
        print()
        print("=" * 86)
        print("SINGLE-FEATURE ABLATION")
        print("=" * 86)
        single_results = []
        for i, name in enumerate(V9_FEATURE_NAMES):
            a_ind = fit_auc(mask_columns(X_ind, [i]), y_ind)
            a_ood = fit_auc(mask_columns(X_ood, [i]), y_ood)
            single_results.append({
                "feature": name,
                "delta_ind": a_ind - baseline_ind,
                "delta_ood": a_ood - baseline_ood,
                "transfer_score": (a_ood - baseline_ood) - (a_ind - baseline_ind),
            })
        single_results.sort(key=lambda r: -r["transfer_score"])
        print(f"{'feature':<25}{'Δ_ind':>10}{'Δ_ood':>10}{'transfer':>11}")
        print("-" * 60)
        for r in single_results[:15]:
            print(f"{r['feature']:<25}{r['delta_ind']:>+10.4f}"
                  f"{r['delta_ood']:>+10.4f}{r['transfer_score']:>+11.4f}")
        print("  ... (full results available in --single mode)")
        print(f"\nBottom 5 (hurt OOD most when removed):")
        for r in sorted(single_results, key=lambda x: x["transfer_score"])[:5]:
            print(f"{r['feature']:<25}{r['delta_ind']:>+10.4f}"
                  f"{r['delta_ood']:>+10.4f}{r['transfer_score']:>+11.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
