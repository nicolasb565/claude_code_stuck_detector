"""Train per-step MLP for stuck detection using the new data format.

Input: training_manifest.json (list of JSONL files with weights)
Data format: per-step JSONL rows (each row has all STEP_FEATURES + label + session_id)
Architecture: per-step MLP (v5) — tool_embed(4) + 11 continuous → fc(32) → relu → fc(16) → relu → fc(1) → sigmoid

Usage:
  python train.py [--manifest training_manifest.json]
"""

import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
MODEL_DIR = "proxy"

NUM_TOOLS = 7
TOOL_EMBED_DIM = 4

STEP_FEATURES = [
    "tool_idx",
    "steps_since_same_tool",
    "steps_since_same_file",
    "steps_since_same_cmd",
    "tool_count_in_window",
    "file_count_in_window",
    "cmd_count_in_window",
    "output_similarity",
    "has_prior_output",
    "output_length",
    "is_error",
    "step_index_norm",
]

CONTINUOUS_FEATURES = [f for f in STEP_FEATURES if f != "tool_idx"]
NUM_CONTINUOUS = len(CONTINUOUS_FEATURES)  # 16


class StuckDetectorMLP(nn.Module):
    """Per-step MLP: tool_embed + 11 continuous → fc(32) → relu → fc(16) → relu → fc(1)."""

    def __init__(self):
        super().__init__()
        self.tool_embed = nn.Embedding(NUM_TOOLS, TOOL_EMBED_DIM)
        in_dim = TOOL_EMBED_DIM + NUM_CONTINUOUS
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, tool_idx: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        emb = self.tool_embed(tool_idx)
        x = torch.cat([emb, cont], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class StepDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.tool_idxs = torch.tensor([r["tool_idx"] for r in rows], dtype=torch.long)
        self.cont = torch.tensor(
            [[float(r[f]) for f in CONTINUOUS_FEATURES] for r in rows],
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            [float(r["label"]) for r in rows], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tool_idxs[i], self.cont[i], self.labels[i]


def load_manifest(manifest_path: str) -> list[dict]:
    """Load training manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def load_rows_from_jsonl(path: str) -> list[dict]:
    """Load per-step JSONL rows."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def session_split(
    rows: list[dict],
    test_fraction: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    """Split rows by session_id (90/10 split).

    Args:
        rows: list of step dicts with session_id
        test_fraction: fraction of sessions for test set

    Returns:
        (train_rows, test_rows)
    """
    session_ids = sorted({r["session_id"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(session_ids)

    n_test = max(1, int(len(session_ids) * test_fraction))
    test_sessions = set(session_ids[:n_test])
    train_sessions = set(session_ids[n_test:])

    train_rows = [r for r in rows if r["session_id"] in train_sessions]
    test_rows = [r for r in rows if r["session_id"] in test_sessions]

    n_train = len(train_sessions)
    n_test_s = len(test_sessions)

    train_stuck = sum(1 for r in train_rows if r["label"] >= 0.9) / max(
        len(train_rows), 1
    )
    test_stuck = sum(1 for r in test_rows if r["label"] >= 0.9) / max(len(test_rows), 1)

    print(f"Train/test split: {n_train} sessions train, {n_test_s} sessions test")
    print(f"  Train STUCK prevalence: {train_stuck * 100:.1f}%")
    print(f"  Test  STUCK prevalence: {test_stuck * 100:.1f}%")

    return train_rows, test_rows


def metrics_at(
    preds: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float, float, int, int, int, int]:
    pred = (preds >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp, fp, fn, tn


def train(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    manifest_path: str = "training_manifest.json",
) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    manifest = load_manifest(manifest_path)
    datasets_cfg = (
        manifest.get("datasets", manifest) if isinstance(manifest, dict) else manifest
    )

    print("Loading data from manifest...")
    all_rows: list[dict] = []
    for entry in datasets_cfg:
        path = entry["path"]
        weight = float(entry.get("weight", 1.0))
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        rows = load_rows_from_jsonl(path)
        # Oversample by weight (round to nearest int)
        copies = max(1, int(round(weight)))
        all_rows.extend(rows * copies)
        print(f"  {path}: {len(rows)} rows × {copies} = {len(rows) * copies}")

    if not all_rows:
        print("ERROR: no data loaded", file=sys.stderr)
        sys.exit(1)

    train_rows, test_rows = session_split(all_rows)

    # Shuffle training rows
    random.shuffle(train_rows)

    train_ds = StepDataset(train_rows)
    test_ds = StepDataset(test_rows)

    # Normalize continuous features using training stats
    mean = train_ds.cont.mean(dim=0)
    std = train_ds.cont.std(dim=0).clamp(min=1e-6)
    train_ds.cont = (train_ds.cont - mean) / std
    test_ds.cont = (test_ds.cont - mean) / std

    num_pos = (train_ds.labels >= 0.9).sum().item()
    num_neg = len(train_ds.labels) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)])
    print(
        f"  Class balance: pos={int(num_pos)} neg={int(num_neg)} pos_weight={pos_weight.item():.1f}"
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    model = StuckDetectorMLP()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    best_state = None
    no_improve = 0
    patience = 5
    threshold = 0.5

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for tool_idx, cont, lab in train_loader:
            logits = model(tool_idx, cont)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        all_s, all_l = [], []
        with torch.no_grad():
            for tool_idx, cont, lab in test_loader:
                s = torch.sigmoid(model(tool_idx, cont))
                all_s.extend(s.numpy())
                all_l.extend(lab.numpy())
        scores = np.array(all_s)
        labels_arr = np.array(all_l)
        # Binarize soft labels for metric computation
        binary_labels = (labels_arr >= 0.9).astype(int)
        prec, rec, f1, tp, fp, fn, _ = metrics_at(scores, binary_labels, threshold)
        print(
            f"  Epoch {epoch:2d}: loss={total_loss/n_batches:.4f}  "
            f"t={threshold} P={prec:.3f} R={rec:.3f} F1={f1:.3f} FP={fp} FN={fn}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for tool_idx, cont, lab in test_loader:
            s = torch.sigmoid(model(tool_idx, cont))
            all_s.extend(s.numpy())
            all_l.extend(lab.numpy())
    scores = np.array(all_s)
    binary_labels = (np.array(all_l) >= 0.9).astype(int)

    prec, rec, f1, tp, fp, fn, tn = metrics_at(scores, binary_labels, threshold)
    print(f"\n=== Final test metrics at threshold={threshold} ===")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    stuck_scores = scores[binary_labels == 1]
    neg_scores = scores[binary_labels == 0]
    pcts = [50, 75, 90, 95, 99]
    print("\n=== Score distribution ===")
    print(f"  STUCK  (n={len(stuck_scores)}): " + "  ".join(
        f"p{p}={np.percentile(stuck_scores, p):.3f}" for p in pcts
    ))
    print(f"  PRODUC (n={len(neg_scores)}): " + "  ".join(
        f"p{p}={np.percentile(neg_scores, p):.3f}" for p in pcts
    ))

    print("\n=== Threshold sweep ===")
    print(f"  {'t':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'FP':>6}  {'FN':>6}")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        p, r, f, _, fp, fn, _ = metrics_at(scores, binary_labels, t)
        print(f"  {t:>5.2f}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {fp:>6}  {fn:>6}")

    final_metrics = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_mean": mean.numpy().tolist(),
            "norm_std": std.numpy().tolist(),
            "threshold": threshold,
            "metrics": final_metrics,
            "total_params": total_params,
        },
        os.path.join(MODEL_DIR, "cnn_trimmed_checkpoint.pt"),
    )

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights["norm_mean"] = mean.numpy().tolist()
    weights["norm_std"] = std.numpy().tolist()
    with open(os.path.join(MODEL_DIR, "cnn_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f)

    config = {
        "threshold": threshold,
        "model_stage": 5,
        "total_params": total_params,
        "metrics": final_metrics,
        "tool_embed_dim": TOOL_EMBED_DIM,
        "num_continuous": NUM_CONTINUOUS,
        "continuous_features": CONTINUOUS_FEATURES,
        "step_features": STEP_FEATURES,
    }
    with open(os.path.join(MODEL_DIR, "cnn_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    size = os.path.getsize(os.path.join(MODEL_DIR, "cnn_weights.json"))
    print("\nSaved:")
    print(f"  {MODEL_DIR}/cnn_trimmed_checkpoint.pt")
    print(f"  {MODEL_DIR}/cnn_weights.json ({size / 1024:.1f} KB)")
    print(f"  {MODEL_DIR}/cnn_config.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training_manifest.json")
    _args = parser.parse_args()
    train(_args.manifest)
