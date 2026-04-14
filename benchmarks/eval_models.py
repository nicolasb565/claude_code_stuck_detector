#!/usr/bin/env python3
"""
Head-to-head evaluation of multiple stuck-detector MLP checkpoints
against the Sonnet-labeled benchmark transcripts.

For each checkpoint:
  1. Parse each task's stream-json transcript into normalized step dicts
  2. Use extract_features.compute_step_features to produce features
  3. Build the ring-buffer input the model expects
  4. Score every step with the loaded MLP
  5. Compare to Sonnet's per-step labels

Reports:
  - Per-task: number of Sonnet-STUCK steps the model flagged hot vs missed
  - Per-task: number of Sonnet-PRODUCTIVE steps the model flagged hot (FPs)
  - Pooled AUC + precision/recall at threshold 0.5
  - Lift on the 55 Sonnet-disagreement steps from the original analysis

Usage:
  .venv/bin/python benchmarks/eval_models.py
  .venv/bin/python benchmarks/eval_models.py --models v5_baseline v6_phase2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.extract_features import (  # noqa: E402
    STEP_FEATURES,
    compute_step_features,
)


# ─── Models we know about ──────────────────────────────────────────────────

KNOWN_MODELS = {
    "v5_baseline": REPO / "proxy",                              # current production
    "v5_1_multi_slot": REPO / "proxy" / "experiments" / "v5_1_multi_slot",
    "v6_phase2": REPO / "proxy" / "experiments" / "v6_phase2",
}


class MLP(nn.Module):
    """Same architecture as StuckDetectorV5 in src/training/train.py."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def load_model(checkpoint_dir: Path) -> tuple[MLP, np.ndarray, np.ndarray, dict]:
    ckpt = torch.load(checkpoint_dir / "stuck_checkpoint.pt", weights_only=False)
    config = json.loads((checkpoint_dir / "stuck_config.json").read_text())
    input_dim = config["input_dim"]
    model = MLP(input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    std = np.array(ckpt["norm_std"], dtype=np.float32)
    return model, mean, std, config


def parse_transcript_to_steps(path: Path) -> list[dict]:
    """stream-json transcript → list of normalized message dicts → parser."""
    messages = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") not in ("user", "assistant"):
            continue
        msg = ev.get("message", {})
        if isinstance(msg, dict):
            messages.append(msg)
    return parse_session(messages)


def build_inputs(features: list[dict], config: dict) -> np.ndarray:
    """Build ring-buffer inputs from per-step features. Mirrors
    src/training/train.py:build_sequences but for one session."""
    excluded = set(config.get("excluded_features", []))
    kept_features = [f for f in config["step_features"]]
    # config['step_features'] already excludes the dropped ones
    n_kept = config["num_features"]
    n_history = config["n_history"]
    use_score_history = config["use_score_history"]
    input_dim = config["input_dim"]

    inputs = np.zeros((len(features), input_dim), dtype=np.float32)
    feat_buf = np.zeros((n_history, n_kept), dtype=np.float32)
    score_buf = np.zeros(n_history, dtype=np.float32)

    for i, row in enumerate(features):
        curr = np.array([float(row[f]) for f in kept_features], dtype=np.float32)
        if use_score_history:
            inp = np.concatenate([curr, feat_buf.flatten(), score_buf])
        else:
            inp = np.concatenate([curr, feat_buf.flatten()])
        inputs[i] = inp
        feat_buf = np.roll(feat_buf, 1, axis=0)
        feat_buf[0] = curr
    return inputs


def evaluate(model_name: str, model_dir: Path, run_dir: Path, verbose: bool):
    model, mean, std, config = load_model(model_dir)
    print(f"\n========== {model_name} (input_dim={config['input_dim']}, "
          f"features={config['num_features']}) ==========")

    all_scores: list[float] = []
    all_labels: list[int] = []
    per_task_summary = []

    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        labels_path = task_dir / "sonnet_labels.json"
        transcript = task_dir / "transcript_1.jsonl"
        if not (labels_path.exists() and transcript.exists()):
            continue

        steps = parse_transcript_to_steps(transcript)
        if not steps:
            continue
        features = compute_step_features(steps)
        sonnet = json.loads(labels_path.read_text())
        labels = sonnet["labels"]
        n = min(len(features), len(labels))
        features = features[:n]
        labels = labels[:n]

        inputs = build_inputs(features, config)
        inputs = (inputs - mean) / std
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.tensor(inputs, dtype=torch.float32))).numpy()

        # Compare to Sonnet labels
        threshold = config.get("threshold", 0.5)
        sonnet_stuck = sum(1 for l in labels if l == "STUCK")
        mlp_hot = sum(1 for s in scores if s >= threshold)
        agree_stuck = sum(
            1 for s, l in zip(scores, labels)
            if s >= threshold and l == "STUCK"
        )
        agree_prod = sum(
            1 for s, l in zip(scores, labels)
            if s < threshold and l == "PRODUCTIVE"
        )
        miss = sonnet_stuck - agree_stuck   # Sonnet says stuck, MLP didn't
        fp = mlp_hot - agree_stuck          # MLP fires, Sonnet says productive

        per_task_summary.append({
            "task": task_dir.name,
            "n": n,
            "sonnet_stuck": sonnet_stuck,
            "mlp_hot": mlp_hot,
            "agree_stuck": agree_stuck,
            "miss": miss,
            "fp": fp,
            "max_score": float(scores.max()) if len(scores) else 0.0,
        })

        # Pool for AUC/PR
        for s, l in zip(scores, labels):
            if l == "UNSURE":
                continue
            all_scores.append(float(s))
            all_labels.append(1 if l == "STUCK" else 0)

    print(f"{'task':<22}{'n':>5}{'son_stk':>8}{'mlp_hot':>9}{'agree':>7}"
          f"{'miss':>6}{'fp':>5}{'max':>8}")
    for r in per_task_summary:
        print(f"{r['task']:<22}{r['n']:>5}{r['sonnet_stuck']:>8}{r['mlp_hot']:>9}"
              f"{r['agree_stuck']:>7}{r['miss']:>6}{r['fp']:>5}{r['max_score']:>8.3f}")

    arr_s = np.array(all_scores)
    arr_l = np.array(all_labels)
    if 0 < arr_l.sum() < len(arr_l):
        auc = roc_auc_score(arr_l, arr_s)
    else:
        auc = float("nan")

    threshold = config.get("threshold", 0.5)
    pred = (arr_s >= threshold).astype(int)
    tp = int(((pred == 1) & (arr_l == 1)).sum())
    fp = int(((pred == 1) & (arr_l == 0)).sum())
    fn = int(((pred == 0) & (arr_l == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\nPOOLED  AUC={auc:.4f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"TP={tp} FP={fp} FN={fn}  (n={len(arr_l)})")

    return {
        "model": model_name,
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
        "per_task": per_task_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(KNOWN_MODELS.keys()))
    ap.add_argument("--run-dir", default="benchmarks/results/comparison_off")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 1

    results = []
    for name in args.models:
        if name not in KNOWN_MODELS:
            print(f"unknown model: {name}; valid: {list(KNOWN_MODELS)}", file=sys.stderr)
            return 1
        results.append(evaluate(name, KNOWN_MODELS[name], run_dir, args.verbose))

    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD")
    print("=" * 90)
    print(f"{'model':<22}{'AUC':>8}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    for r in results:
        print(f"{r['model']:<22}{r['auc']:>8.4f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}")

    out_path = run_dir / "eval_models.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
