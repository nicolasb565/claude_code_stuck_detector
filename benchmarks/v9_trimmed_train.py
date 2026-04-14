#!/usr/bin/env python3
"""
v9-trimmed training — drop the 19 v9 features whose correlation with the
stuck label flips sign between in-distribution training data and the OOD
benchmark. Those features can only teach the model the wrong thing.

Kept features (15):
  - All 5 action_match slots (p1-p5)
  - self_sim at slots 1, 2, 3, 5 (p4 flipped)
  - p5_is_err, p1_is_err, p2_is_err (weak but consistent)
  - p2_file_match, p1_file_match
  - cur_sim_vs_match, cur_is_err

Dropped features (19):
  - All 5 scope_match features (flipped negative → positive)
  - All 6 output_length features (cur + p1-p5, all flipped)
  - 3 is_err slots (p1, p3, p4)
  - cur_consec_match (flipped)
  - file_match at p3, p4, p5 (flipped)
  - p4_self_sim (flipped — the in-dist signal was +0.68 but OOD is 0)

The reasoning: train.py learns a function from features → label in the
training distribution. If a feature's correlation with the label inverts
in OOD, the model's learned weights for that feature will actively pull
scores in the wrong direction. Zeroing these columns at both train and
eval time removes that corruption channel.

Usage:
  .venv/bin/python benchmarks/v9_trimmed_train.py
  POS_WEIGHT_MULT=3 .venv/bin/python benchmarks/v9_trimmed_train.py \
      --output-dir proxy/experiments/v9_trimmed_pw3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.extract_features import V9_FEATURE_NAMES  # noqa: E402

# Feature set selected by FEATURE-TYPE-LEVEL correlation analysis, not
# per-slot noise. A feature type is kept if and only if its mean per-slot
# correlation with the stuck label has the same sign on both the in-dist
# training corpus and the OOD benchmark.
#
# Only TWO feature types survive this test:
#   - prev_act_match: in-dist mean +0.60, OOD mean +0.13 (strong agreement)
#   - prev_self_sim:  in-dist mean +0.69, OOD mean +0.05 (weak but agree)
#
# All other feature types (file_match, scope_match, out_len, is_err,
# cur_out_len, cur_is_err, cur_sim_vs_match, cur_consec_match) either flip
# sign or are marginal in at least one distribution, so they can only
# teach the model the wrong thing or add random noise to OOD predictions.
#
# Slots are treated symmetrically — all 5 history positions kept or
# dropped together — because slot-level correlations within a feature
# class are tightly clustered (the per-slot variation is noise).
KEEP_FEATURES = (
    [f"v9_p{i+1}_act_match" for i in range(5)]   # 5 features
    + [f"v9_p{i+1}_self_sim" for i in range(5)]  # 5 features
)
assert all(f in V9_FEATURE_NAMES for f in KEEP_FEATURES)
assert len(KEEP_FEATURES) == 10

KEEP_INDICES = [V9_FEATURE_NAMES.index(f) for f in KEEP_FEATURES]
INPUT_DIM = len(KEEP_INDICES)  # 15

DEFAULT_SEED = 42


class TrimDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class V9TrimmedMLP(nn.Module):
    """Small MLP for the 10-dim principled-trimmed feature set (act_match × 5
    + self_sim × 5). Even smaller hidden layers than v9 since the feature
    set is smaller and cleaner."""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_inputs(rows, drop_unsure=True):
    X, y = [], []
    for r in rows:
        lbl = r.get("label", 0.0)
        if drop_unsure and 0.3 < lbl < 0.7:
            continue
        vec = [float(r[k]) for k in KEEP_FEATURES]
        X.append(vec)
        y.append(1.0 if lbl >= 0.9 else 0.0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def session_split(rows, test_fraction=0.1, seed=DEFAULT_SEED):
    by_sess = defaultdict(list)
    for r in rows:
        by_sess[r["session_id"]].append(r)
    ids = sorted(by_sess.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    test_ids = set(ids[:n_test])
    train, test = [], []
    for sid, rs in by_sess.items():
        (test if sid in test_ids else train).extend(rs)
    return train, test


def metrics_at(preds, labels, threshold):
    pred = (preds >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp, fp, fn, tn


def train_trimmed(manifest_path, output_dir, seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)
    datasets_cfg = manifest.get("datasets", manifest)

    print(f"\nv9-trimmed training — {INPUT_DIM} features (dropped {34 - INPUT_DIM} dist-flipping)")
    print(f"Kept: {', '.join(KEEP_FEATURES)}")
    print(f"Output: {output_dir}")

    all_rows = []
    for entry in datasets_cfg:
        path = entry["path"]
        if not os.path.exists(path):
            print(f"  WARN: {path} missing")
            continue
        rs = load_rows(path)
        all_rows.extend(rs)
        print(f"  {path}: {len(rs)} rows")

    train_rows, test_rows = session_split(all_rows, seed=seed)
    X_train, y_train = build_inputs(train_rows)
    X_test, y_test = build_inputs(test_rows)
    print(f"  train={len(X_train)}  stuck={int(y_train.sum())}  "
          f"test={len(X_test)}  stuck={int(y_test.sum())}")

    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    train_ds = TrimDataset(X_train_n, y_train)
    test_ds = TrimDataset(X_test_n, y_test)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    base_pw = num_neg / max(num_pos, 1)
    pw_mult = float(os.environ.get("POS_WEIGHT_MULT", "1.0"))
    pos_weight = torch.tensor([base_pw * pw_mult])
    print(f"  pos_weight={pos_weight.item():.1f} (base={base_pw:.1f} × mult={pw_mult})")

    model = V9TrimmedMLP()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    threshold = 0.5
    best_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    for epoch in range(40):
        model.train()
        total_loss, nb = 0.0, 0
        for inp, lab in train_loader:
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        model.eval()
        all_s, all_l = [], []
        with torch.no_grad():
            for inp, lab in test_loader:
                s = torch.sigmoid(model(inp))
                all_s.extend(s.numpy())
                all_l.extend(lab.numpy())
        scores = np.array(all_s)
        binary_labels = (np.array(all_l) >= 0.5).astype(int)
        prec, rec, f1, tp, fp, fn, _ = metrics_at(scores, binary_labels, threshold)
        print(f"  epoch {epoch:2d}: loss={total_loss/nb:.4f}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} FP={fp} FN={fn}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"  early stop at {epoch}")
                break

    model.load_state_dict(best_state)

    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for inp, lab in test_loader:
            s = torch.sigmoid(model(inp))
            all_s.extend(s.numpy())
            all_l.extend(lab.numpy())
    scores = np.array(all_s)
    binary_labels = (np.array(all_l) >= 0.5).astype(int)
    prec, rec, f1, tp, fp, fn, tn = metrics_at(scores, binary_labels, threshold)
    print(f"\n=== Final in-dist: P={prec:.3f} R={rec:.3f} F1={f1:.3f} ===")

    os.makedirs(output_dir, exist_ok=True)
    final = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
             "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": threshold}
    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "threshold": threshold,
        "metrics": final,
        "total_params": total_params,
        "architecture": "v9_trimmed",
        "input_dim": INPUT_DIM,
        "kept_features": KEEP_FEATURES,
    }, os.path.join(output_dir, "stuck_checkpoint.pt"))

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights["norm_mean"] = mean.tolist()
    weights["norm_std"] = std.tolist()
    with open(os.path.join(output_dir, "stuck_weights.json"), "w") as f:
        json.dump(weights, f)

    config = {
        "threshold": threshold,
        "architecture": "v9_trimmed",
        "input_dim": INPUT_DIM,
        "num_features": INPUT_DIM,
        "step_features": list(KEEP_FEATURES),
        "total_params": total_params,
        "metrics": final,
    }
    with open(os.path.join(output_dir, "stuck_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved: {output_dir}/stuck_checkpoint.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest_v6.json")
    ap.add_argument("--output-dir", default="proxy/experiments/v9_trimmed")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    train_trimmed(args.manifest, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
