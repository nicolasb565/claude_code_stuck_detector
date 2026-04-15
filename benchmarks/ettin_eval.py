#!/usr/bin/env python3
"""
Evaluate a fine-tuned Ettin classifier on the OOD stuck-detection benchmark.

Mirrors slm_stuck.py's windowed framing so results are directly comparable
to the zero-shot SLM runs and the LR baseline.

Usage:
  python benchmarks/ettin_eval.py \
      --model proxy/experiments/ettin_400m_stuck
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="proxy/experiments/ettin_400m_stuck",
                    help="path to saved Ettin checkpoint")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--context-steps", type=int, default=5)
    ap.add_argument("--max-output-chars", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--relabel-n", type=int, default=0,
                    help="also report metrics against N-consecutive relabeled "
                         "ground truth (0 = skip)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.pipeline.parsers.nlile import parse_session
    from src.pipeline.label_session import _render_step

    RUN_DIR = REPO / "benchmarks" / "results" / "comparison_off"

    model_path = REPO / args.model
    print(f"torch: {torch.__version__}", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"loading: {model_path}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    print(f"loaded  vram={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    LABEL_TO_ID = {"P": 0, "S": 1, "U": 2}
    # Match ettin_train.py's build_text — share code if the files get more entangled
    SYSTEM_PROMPT = """\
You are evaluating a Claude Code coding session for stuck detection.

Classify the TARGET step (the last step shown) as exactly one letter:
  P - productive: the agent is making progress (new approach, first-time \
action, finding new bugs, legitimate iterative build/test loops with fixes \
in between)
  S - stuck: the agent is in a loop (same command/error/edit repeated \
without progress, trying the same thing from different angles)
  U - unsure: genuine ambiguity that cannot be resolved from the context

Rules:
- First attempt at any command → P
- Legitimate build/test iteration where fixes are being made between runs → P
- Same command with same error twice or more, no visible change → S
- Different approach or tool after a failure → P (first step of the new approach)
- Agent hitting the same underlying error from different files/angles → S"""

    def render_step(step, i):
        rendered = _render_step(step, i)
        lines = rendered.split("\n")
        out = []
        for ln in lines:
            if ln.startswith("  → ") and len(ln) > args.max_output_chars:
                ln = ln[: args.max_output_chars] + " [...]"
            out.append(ln)
        return "\n".join(out)

    def build_text(context_steps, target_step, target_index):
        parts = [SYSTEM_PROMPT, ""]
        if context_steps:
            parts.append("Context (prior steps):")
            start = target_index - len(context_steps)
            for offset, s in enumerate(context_steps):
                parts.append(render_step(s, start + offset))
        parts.append("")
        parts.append("TARGET step:")
        parts.append(render_step(target_step, target_index))
        return "\n".join(parts)

    def parse_transcript(path):
        messages = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") in ("user", "assistant"):
                m = ev.get("message", {})
                if isinstance(m, dict):
                    messages.append(m)
        return parse_session(messages)

    # ── Build the OOD flat set: (text, gold_label, task_name) ────────────
    print(f"\nbuilding OOD benchmark inputs", flush=True)
    texts = []
    golds = []  # "STUCK" / "PRODUCTIVE" / "UNSURE"
    tasks_for_step = []

    for td in sorted(RUN_DIR.iterdir()):
        if not td.is_dir():
            continue
        t = td / "transcript_1.jsonl"
        lp = td / "sonnet_labels.json"
        if not (t.exists() and lp.exists()):
            continue
        steps = parse_transcript(t)
        labels = json.loads(lp.read_text())["labels"]
        n = min(len(steps), len(labels))
        for i in range(n):
            context = steps[max(0, i - args.context_steps): i]
            text = build_text(context, steps[i], i)
            texts.append(text)
            golds.append(labels[i])
            tasks_for_step.append(td.name)

    print(f"  total steps: {len(texts)}", flush=True)

    # ── Batch through the model ──────────────────────────────────────────
    preds = []  # predicted class id
    s_probs = []  # P(S) per step
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i: i + args.batch_size]
            enc = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to("cuda:0") for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.float()
            probs = F.softmax(logits, dim=-1)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            s_probs.extend(probs[:, 1].cpu().tolist())
            if (i // args.batch_size) % 10 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = (i + len(batch_texts)) / elapsed
                eta = (len(texts) - i - len(batch_texts)) / rate
                print(f"  {i + len(batch_texts)}/{len(texts)}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"  total: {total_time:.0f}s  ({1000*total_time/len(texts):.0f}ms/step)",
          flush=True)

    # ── Metrics ──────────────────────────────────────────────────────────
    # Per-task
    from collections import defaultdict
    per_task_stats = defaultdict(
        lambda: {"n": 0, "stk": 0, "tp": 0, "fp": 0, "fn": 0}
    )
    all_preds_bin = []  # 1 if S else 0
    all_labels_bin = []  # 1 if STUCK else 0
    all_scores = []  # probability of S

    for pred, gold, task_name, s_prob in zip(preds, golds, tasks_for_step, s_probs):
        st = per_task_stats[task_name]
        st["n"] += 1
        if gold == "STUCK":
            st["stk"] += 1
        pred_is_s = pred == 1
        gold_is_s = gold == "STUCK"
        if pred_is_s and gold_is_s:
            st["tp"] += 1
        elif pred_is_s and not gold_is_s:
            st["fp"] += 1
        elif not pred_is_s and gold_is_s:
            st["fn"] += 1
        # Pooled (exclude UNSURE)
        if gold == "UNSURE":
            continue
        all_preds_bin.append(1 if pred_is_s else 0)
        all_labels_bin.append(1 if gold_is_s else 0)
        all_scores.append(s_prob)

    print(f"\n{'task':<22}{'n':>5}{'stk':>5}{'tp':>5}{'fp':>5}{'fn':>5}")
    for task in sorted(per_task_stats.keys()):
        r = per_task_stats[task]
        print(f"{task:<22}{r['n']:>5}{r['stk']:>5}{r['tp']:>5}"
              f"{r['fp']:>5}{r['fn']:>5}")

    arr_p = np.array(all_preds_bin)
    arr_l = np.array(all_labels_bin)
    arr_sc = np.array(all_scores)

    if 0 < arr_l.sum() < len(arr_l):
        auc_bin = roc_auc_score(arr_l, arr_p)
        auc_cont = roc_auc_score(arr_l, arr_sc)
    else:
        auc_bin = auc_cont = float("nan")

    tp = int(((arr_p == 1) & (arr_l == 1)).sum())
    fp = int(((arr_p == 1) & (arr_l == 0)).sum())
    fn = int(((arr_p == 0) & (arr_l == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\n=== Ettin pooled on OOD (ORIGINAL labels) ===")
    print(f"  AUC (binary argmax): {auc_bin:.4f}")
    print(f"  AUC (S-prob score):  {auc_cont:.4f}")
    print(f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"\n  LR baseline:               F1=0.326  AUC=0.736")
    print(f"  Zero-shot best (llama3.1): F1=0.165")

    # ── Recall @ fixed Precision (product-aligned metric) ────────────────
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(arr_l, arr_sc)
    print(f"\n=== Recall @ Precision (ORIGINAL labels) ===")
    print(f"  {'target_P':>10} {'actual_P':>10} {'recall':>10} "
          f"{'threshold':>12} {'TP':>5} {'FP':>5} {'FN':>5}")
    for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        valid = precisions >= target
        if not valid.any():
            print(f"  {target:>10.2f}  (no threshold achieves P ≥ {target})")
            continue
        valid_recalls = np.where(valid, recalls, -1)
        best_idx = int(valid_recalls.argmax())
        best_p = precisions[best_idx]
        best_r = recalls[best_idx]
        best_thresh = 1.0 if best_idx >= len(thresholds) else thresholds[best_idx]
        pr = (arr_sc >= best_thresh).astype(int)
        t_tp = int(((pr == 1) & (arr_l == 1)).sum())
        t_fp = int(((pr == 1) & (arr_l == 0)).sum())
        t_fn = int(((pr == 0) & (arr_l == 1)).sum())
        print(f"  {target:>10.2f} {best_p:>10.3f} {best_r:>10.3f} "
              f"{best_thresh:>12.4f} {t_tp:>5} {t_fp:>5} {t_fn:>5}")

    # ── Optionally re-evaluate against N-consecutive relabeled ground truth ──
    if args.relabel_n > 0:
        n = args.relabel_n
        print(f"\n=== Re-evaluating against N={n} consecutive-stuck relabeling ===")
        # Rebuild per-session step labels: need the task-grouped step order.
        # We have tasks_for_step[i] = which OOD task step i belongs to. Golds
        # are in the same order.
        from collections import defaultdict as _dd
        by_task = _dd(list)
        for i, tn in enumerate(tasks_for_step):
            by_task[tn].append((i, golds[i]))

        # Apply N-consecutive relabeling per task, produce relabeled_golds
        relabeled_golds = [None] * len(golds)
        for tn, entries in by_task.items():
            task_labels = [g for _, g in entries]
            new_labels = []
            for idx_in_task in range(len(task_labels)):
                start = max(0, idx_in_task - n + 1)
                window = task_labels[start:idx_in_task + 1]
                if len(window) >= n and all(l == "STUCK" for l in window):
                    new_labels.append("STUCK")
                else:
                    # Keep UNSURE as UNSURE; else PRODUCTIVE
                    orig = task_labels[idx_in_task]
                    new_labels.append("UNSURE" if orig == "UNSURE" else "PRODUCTIVE")
            for (orig_idx, _), new_l in zip(entries, new_labels):
                relabeled_golds[orig_idx] = new_l

        # Build binary arrays under relabeled ground truth
        re_preds = []
        re_labels = []
        re_scores = []
        for pred, gold, s_prob in zip(preds, relabeled_golds, s_probs):
            if gold == "UNSURE":
                continue
            re_preds.append(1 if pred == 1 else 0)
            re_labels.append(1 if gold == "STUCK" else 0)
            re_scores.append(s_prob)
        re_preds_arr = np.array(re_preds)
        re_labels_arr = np.array(re_labels)
        re_scores_arr = np.array(re_scores)

        rtp = int(((re_preds_arr == 1) & (re_labels_arr == 1)).sum())
        rfp = int(((re_preds_arr == 1) & (re_labels_arr == 0)).sum())
        rfn = int(((re_preds_arr == 0) & (re_labels_arr == 1)).sum())
        rp = rtp / max(rtp + rfp, 1)
        rr = rtp / max(rtp + rfn, 1)
        rf1 = 2 * rp * rr / max(rp + rr, 1e-9)
        if 0 < re_labels_arr.sum() < len(re_labels_arr):
            r_auc_cont = roc_auc_score(re_labels_arr, re_scores_arr)
        else:
            r_auc_cont = float("nan")

        print(f"  Relabeled STUCK count: {int(re_labels_arr.sum())} "
              f"(vs {int(arr_l.sum())} in original)")
        print(f"  AUC (S-prob score):  {r_auc_cont:.4f}")
        print(f"  P={rp:.3f}  R={rr:.3f}  F1={rf1:.3f}")
        print(f"  TP={rtp}  FP={rfp}  FN={rfn}")

        r_precisions, r_recalls, r_thresholds = precision_recall_curve(
            re_labels_arr, re_scores_arr)
        print(f"\n=== Recall @ Precision (N={n} relabeled) ===")
        print(f"  {'target_P':>10} {'actual_P':>10} {'recall':>10} "
              f"{'threshold':>12} {'TP':>5} {'FP':>5} {'FN':>5}")
        for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            valid = r_precisions >= target
            if not valid.any():
                print(f"  {target:>10.2f}  (no threshold achieves P ≥ {target})")
                continue
            valid_recalls = np.where(valid, r_recalls, -1)
            best_idx = int(valid_recalls.argmax())
            best_p = r_precisions[best_idx]
            best_r = r_recalls[best_idx]
            best_thresh = 1.0 if best_idx >= len(r_thresholds) else r_thresholds[best_idx]
            pr = (re_scores_arr >= best_thresh).astype(int)
            t_tp = int(((pr == 1) & (re_labels_arr == 1)).sum())
            t_fp = int(((pr == 1) & (re_labels_arr == 0)).sum())
            t_fn = int(((pr == 0) & (re_labels_arr == 1)).sum())
            print(f"  {target:>10.2f} {best_p:>10.3f} {best_r:>10.3f} "
                  f"{best_thresh:>12.4f} {t_tp:>5} {t_fp:>5} {t_fn:>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
