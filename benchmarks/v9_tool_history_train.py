#!/usr/bin/env python3
"""
v9_trimmed + tool_idx on prior slots AND current step.

Replaces the binary is_bash with full tool identity. Two variants:

  --variant scalar:   tool_idx as int [0,6] per slot — 5 prior + 1 cur = 6 extras
                      (10 + 6 = 16 features)
  --variant onehot:   tool_idx as 7-dim one-hot per slot — 5×7 + 7 = 42 extras
                      (10 + 42 = 52 features)

Since per-slot prior tool_idx isn't stored in the v6 feature jsonl, we
recompute it on-the-fly here: sort rows per session by step, then
look back N_HISTORY positions at each step.

Usage:
  .venv/bin/python benchmarks/v9_tool_history_train.py --variant scalar
  .venv/bin/python benchmarks/v9_tool_history_train.py --variant onehot
  POS_WEIGHT_MULT=3 .venv/bin/python benchmarks/v9_tool_history_train.py --variant scalar
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

from src.pipeline.extract_features import V9_FEATURE_NAMES, TOOL_TO_IDX  # noqa: E402

BASE_KEEP = (
    [f"v9_p{i+1}_act_match" for i in range(5)]
    + [f"v9_p{i+1}_self_sim" for i in range(5)]
)
N_HISTORY = 5
N_TOOLS = len(TOOL_TO_IDX)  # 7

DEFAULT_SEED = 42


def build_inputs(rows: list[dict], variant: str, drop_unsure: bool = True):
    """
    Group rows by session, sort by step, for each step compute:
      - the 10 base v9_trimmed features (from the row)
      - the tool_idx features for current + 5 prior slots
    Return (X, y).
    """
    by_sess = defaultdict(list)
    for r in rows:
        by_sess[r["session_id"]].append(r)

    X, y = [], []
    for sid, rs in by_sess.items():
        rs_sorted = sorted(rs, key=lambda r: r.get("step", 0))
        prior_tool_buf = []  # most-recent first: prior_tool_buf[0] = T-1, buf[1] = T-2, ...
        for r in rs_sorted:
            lbl = r.get("label", 0.0)
            if drop_unsure and 0.3 < lbl < 0.7:
                # Still update the tool buffer so ring buffer state stays valid,
                # then skip the row
                prior_tool_buf.insert(0, int(r.get("tool_idx", 6)))
                prior_tool_buf = prior_tool_buf[:N_HISTORY]
                continue

            cur_tool = int(r.get("tool_idx", 6))
            vec = [float(r[k]) for k in BASE_KEEP]

            # Prior-slot tool features
            for slot in range(N_HISTORY):
                if slot < len(prior_tool_buf):
                    t = prior_tool_buf[slot]
                else:
                    t = -1  # zero-pad sentinel
                if variant == "scalar":
                    vec.append(float(t) if t >= 0 else 0.0)
                elif variant == "onehot":
                    oh = [0.0] * N_TOOLS
                    if 0 <= t < N_TOOLS:
                        oh[t] = 1.0
                    vec.extend(oh)

            # Current-step tool features
            if variant == "scalar":
                vec.append(float(cur_tool))
            elif variant == "onehot":
                oh = [0.0] * N_TOOLS
                if 0 <= cur_tool < N_TOOLS:
                    oh[cur_tool] = 1.0
                vec.extend(oh)

            X.append(vec)
            y.append(1.0 if lbl >= 0.9 else 0.0)

            prior_tool_buf.insert(0, cur_tool)
            prior_tool_buf = prior_tool_buf[:N_HISTORY]

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def input_dim_for(variant: str) -> int:
    if variant == "scalar":
        return 10 + N_HISTORY + 1  # 16
    elif variant == "onehot":
        return 10 + N_HISTORY * N_TOOLS + N_TOOLS  # 52
    raise ValueError(variant)


class V9ToolHistMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim <= 20:
            h1, h2 = 20, 10
        else:
            h1, h2 = 48, 20
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class Dataset_(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def load_rows(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


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


def train_model(manifest_path, output_dir, variant, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = input_dim_for(variant)
    print(f"\nv9_trimmed + tool_history ({variant}) — input_dim={input_dim}")
    print(f"Output: {output_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    datasets_cfg = manifest.get("datasets", manifest)

    all_rows = []
    for entry in datasets_cfg:
        path = entry["path"]
        if not os.path.exists(path):
            continue
        all_rows.extend(load_rows(path))

    train_rows, test_rows = session_split(all_rows, seed=seed)
    X_train, y_train = build_inputs(train_rows, variant)
    X_test, y_test = build_inputs(test_rows, variant)
    print(f"  train={len(X_train)}  stuck={int(y_train.sum())}")
    print(f"  test={len(X_test)}  stuck={int(y_test.sum())}")

    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    train_loader = DataLoader(Dataset_(X_train_n, y_train), batch_size=512, shuffle=True)
    test_loader = DataLoader(Dataset_(X_test_n, y_test), batch_size=1024)

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    base_pw = num_neg / max(num_pos, 1)
    pw_mult = float(os.environ.get("POS_WEIGHT_MULT", "1.0"))
    pos_weight = torch.tensor([base_pw * pw_mult])
    print(f"  pos_weight={pos_weight.item():.1f}  mult={pw_mult}")

    model = V9ToolHistMLP(input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {total_params}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    for epoch in range(40):
        model.train()
        tot, nb = 0.0, 0
        for inp, lab in train_loader:
            opt.zero_grad()
            loss = crit(model(inp), lab)
            loss.backward()
            opt.step()
            tot += loss.item()
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
        prec, rec, f1, tp, fp, fn, _ = metrics_at(scores, binary_labels, 0.5)
        print(f"  epoch {epoch:2d}: loss={tot/nb:.4f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
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
    prec, rec, f1, tp, fp, fn, tn = metrics_at(scores, binary_labels, 0.5)
    print(f"\n=== Final: P={prec:.3f} R={rec:.3f} F1={f1:.3f} ===")

    os.makedirs(output_dir, exist_ok=True)
    final = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
             "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": 0.5}
    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "threshold": 0.5,
        "metrics": final,
        "total_params": total_params,
        "architecture": f"v9_tool_hist_{variant}",
        "input_dim": input_dim,
        "variant": variant,
    }, os.path.join(output_dir, "stuck_checkpoint.pt"))
    w = {k: v.detach().cpu().numpy().tolist() for k, v in model.named_parameters()}
    w["norm_mean"] = mean.tolist()
    w["norm_std"] = std.tolist()
    with open(os.path.join(output_dir, "stuck_weights.json"), "w") as f:
        json.dump(w, f)
    cfg = {
        "threshold": 0.5,
        "architecture": f"v9_tool_hist_{variant}",
        "variant": variant,
        "input_dim": input_dim,
        "num_features": input_dim,
        "total_params": total_params,
        "metrics": final,
    }
    with open(os.path.join(output_dir, "stuck_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest_v6.json")
    ap.add_argument("--variant", choices=["scalar", "onehot"], required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    out = args.output_dir or f"proxy/experiments/v9_trimmed_tool_hist_{args.variant}"
    train_model(args.manifest, out, args.variant, args.seed)


if __name__ == "__main__":
    main()
