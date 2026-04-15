#!/usr/bin/env python3
"""
Re-evaluate the LR baseline under the precision-first product framing.

Instead of reporting F1 (which forces the classifier to the precision-recall
midpoint), compute Recall at fixed-Precision thresholds. The product we
actually want to ship only fires on clearly-stuck-for-a-while — it should
hardly ever false-positive, even at the cost of missing the start of loops
(which we can't catch causally anyway per project_causal_ceiling.md).

Usage:
  .venv/bin/python benchmarks/precision_recall_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
    )

    cache_path = REPO / "data" / "generated" / "content_prototype.json"
    if not cache_path.exists():
        print(f"ERROR: cached content dataset not found at {cache_path}")
        print(f"Run: .venv/bin/python benchmarks/v9_content_features.py --train")
        return 1

    d = json.loads(cache_path.read_text())
    indist = d["indist"]
    ood = d["ood"]

    FEATS = [
        "match_ratio_5", "self_sim_max", "repeat_no_error",
        "cur_bash_and_match_ratio",
        "unique_err_sigs_6", "new_token_ratio_vs_5",
        "has_success_marker", "err_volume_ratio_vs_5",
    ]
    print(f"Features ({len(FEATS)}): {FEATS}")

    def build(rows):
        rows = [r for r in rows if r["label"] in (0.0, 1.0)]
        X = np.array([[r[k] for k in FEATS] for r in rows], dtype=np.float64)
        y = np.array([1 if r["label"] >= 0.9 else 0 for r in rows], dtype=np.int32)
        return X, y

    X_tr, y_tr = build(indist)
    X_ood, y_ood = build(ood)
    print(f"\ntrain: n={len(X_tr)} stuck={y_tr.sum()}")
    print(f"ood:   n={len(X_ood)} stuck={y_ood.sum()}")

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0).clip(min=1e-6)
    Xtr = (X_tr - mean) / std
    Xood = (X_ood - mean) / std

    num_pos = int(y_tr.sum())
    num_neg = len(y_tr) - num_pos
    pw = num_neg / max(num_pos, 1)
    lr = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: pw},
        max_iter=2000,
        solver="lbfgs",
    )
    lr.fit(Xtr, y_tr)

    # Predictions on OOD
    s_ood = lr.predict_proba(Xood)[:, 1]

    # Summary: standard metrics at threshold 0.5
    pred_05 = (s_ood >= 0.5).astype(int)
    tp5 = int(((pred_05 == 1) & (y_ood == 1)).sum())
    fp5 = int(((pred_05 == 1) & (y_ood == 0)).sum())
    fn5 = int(((pred_05 == 0) & (y_ood == 1)).sum())
    p5 = tp5 / max(tp5 + fp5, 1)
    r5 = tp5 / max(tp5 + fn5, 1)
    f15 = 2 * p5 * r5 / max(p5 + r5, 1e-9)
    ap = average_precision_score(y_ood, s_ood)
    auc = roc_auc_score(y_ood, s_ood)

    print(f"\n=== At default threshold 0.5 ===")
    print(f"  P={p5:.3f}  R={r5:.3f}  F1={f15:.3f}  TP={tp5} FP={fp5} FN={fn5}")
    print(f"\nROC AUC:  {auc:.4f}")
    print(f"Avg Prec: {ap:.4f}")

    # ── The new product metric: R @ precision thresholds ────────────────
    # For each precision target, find the highest-recall operating point
    # where precision ≥ target. precision_recall_curve returns precisions,
    # recalls, and thresholds sorted by ascending threshold.
    precisions, recalls, thresholds = precision_recall_curve(y_ood, s_ood)
    # precisions and recalls are len(thresholds)+1; the last entry is
    # (precision=1.0, recall=0.0) by convention.

    print(f"\n=== Recall at fixed Precision (new product metric) ===")
    print(f"  {'target_P':>10} {'actual_P':>10} {'recall':>10} "
          f"{'threshold':>12} {'TP':>5} {'FP':>5} {'FN':>5}")
    for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        # Find the best (max recall) operating point with P >= target
        valid = precisions >= target
        if not valid.any():
            print(f"  {target:>10.2f}  (no threshold achieves P ≥ {target})")
            continue
        # recalls is same length as precisions
        valid_recalls = np.where(valid, recalls, -1)
        best_idx = int(valid_recalls.argmax())
        best_p = precisions[best_idx]
        best_r = recalls[best_idx]
        # Threshold at this point: thresholds is 1 shorter than p/r
        if best_idx >= len(thresholds):
            best_thresh = 1.0  # the sentinel (p=1, r=0) point
        else:
            best_thresh = thresholds[best_idx]
        # Compute TP/FP/FN at this threshold
        pred = (s_ood >= best_thresh).astype(int)
        tp = int(((pred == 1) & (y_ood == 1)).sum())
        fp = int(((pred == 1) & (y_ood == 0)).sum())
        fn = int(((pred == 0) & (y_ood == 1)).sum())
        print(f"  {target:>10.2f} {best_p:>10.3f} {best_r:>10.3f} "
              f"{best_thresh:>12.4f} {tp:>5} {fp:>5} {fn:>5}")

    # Also show recall at specific debounced thresholds
    # (debouncing = require N consecutive positive predictions before firing)
    print(f"\n=== Top-scored FPs at default 0.5 (sanity check) ===")
    # Show the highest-scored false positives so we understand what the
    # model gets wrong
    fp_idx = np.where((y_ood == 0))[0]
    fp_scores = s_ood[fp_idx]
    top_fp = fp_idx[np.argsort(-fp_scores)[:5]]
    for i in top_fp:
        print(f"  idx={i:4d} score={s_ood[i]:.4f} true_label=P")

    return 0


if __name__ == "__main__":
    sys.exit(main())
