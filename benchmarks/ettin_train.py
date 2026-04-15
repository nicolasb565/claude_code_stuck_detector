#!/usr/bin/env python3
"""
Fine-tune Ettin-encoder-400m (jhu-clsp/ettin-encoder-400m) as a 3-way
(P/S/U) sequence classifier on the stuck-detection dataset.

Why encoder-only sequence classification instead of causal LM fine-tuning:
  - Our task is fixed-input, fixed-output classification. Decoder-only
    causal LM training is known to collapse on short-label classification
    because the gradient signal on the single label token is tiny relative
    to the sequence loss. We hit this collapse on phi4-mini despite LoRA,
    class balance, and gradient checkpointing.
  - Encoder-only classifiers train the label prediction directly through
    a classification head. No wasted capacity on token prediction; no
    collapse to majority class; much simpler recipe.
  - Ettin beats ModernBERT-large on GLUE (90.8 vs 90.4) at matched size
    and has active SDPA support (ModernBERT's eager path is broken).

Input format: concatenation of system prompt + 5 context steps + target
step, rendered via src.pipeline.label_session._render_step for train/serve
parity with the zero-shot benchmark.

Output: 3-class logits for P (0), S (1), U (2), cross-entropy with
inverse-frequency class weights computed from the training distribution.

Usage (inside the rocm/pytorch container):
  python benchmarks/ettin_train.py --smoke    # 500 examples, 1 epoch
  python benchmarks/ettin_train.py            # full, 3 epochs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/ettin-encoder-400m")
    ap.add_argument("--train-file", default="data/generated/finetune_train.jsonl")
    ap.add_argument("--val-file", default="data/generated/finetune_val.jsonl")
    ap.add_argument("--output-dir", default="proxy/experiments/ettin_400m_stuck")
    # NOTE on defaults: at seq 2048 × batch 8, attention activation memory
    # at ~22 layers × 16 heads is ~18 GB on Ettin/ModernBERT — larger than
    # the RX 9070 XT's 16 GB — because torch-rocm 7.2's SDPA path on gfx1201
    # materializes the full attention matrix (no memory-efficient kernel
    # yet). This causes the system to become unresponsive. Default here is
    # the largest config that empirically fits.
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--context-steps", type=int, default=5)
    ap.add_argument("--max-output-chars", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    from src.pipeline.parsers.nlile import parse_session  # noqa: F401
    from src.pipeline.label_session import _render_step

    print(f"torch: {torch.__version__}", flush=True)
    print(f"hip:   {torch.version.hip}", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"vram total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
          flush=True)

    # Pre-flight VRAM check. Refuse to start on a dirty GPU — on ROCm / gfx1201
    # a crashed process can leave a zombie KFD allocation that starves
    # subsequent runs and, when compounded, makes the whole system unresponsive
    # enough that we had to hit the reset button. Fail loud + early.
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb = free_b / 1e9
    total_gb = total_b / 1e9
    print(f"vram free:  {free_gb:.1f} GB / {total_gb:.1f} GB", flush=True)
    # 400M bf16 encoder + AdamW state + activations at seq 2048 batch 8 should
    # fit in ~10-12 GB. Require >= 13 GB free so we have headroom.
    if free_gb < 13.0:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB VRAM free (need >= 13 GB).\n"
            f"Something is holding VRAM — likely a zombie from a crashed\n"
            f"previous run. Recovery options:\n"
            f"  - sudo rocm-smi --gpureset --device 0\n"
            f"  - kill stale python processes\n"
            f"  - reboot"
        )
    # Cap our own budget so if *this* process leaks, the leak is contained.
    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    print(f"  per-process cap: 90% ({total_gb * 0.9:.1f} GB)", flush=True)

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"\nloading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"  vocab: {tok.vocab_size}  max_len: {tok.model_max_length}",
          flush=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"loading model {args.model} bf16 as 3-class classifier", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # fall back to eager if SDPA errors
    ).to("cuda:0")
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params/1e6:.0f}M total  {n_trainable/1e6:.0f}M trainable",
          flush=True)
    print(f"  vram after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # Label map: P=0, S=1, U=2
    LABEL_TO_ID = {"P": 0, "S": 1, "U": 2}
    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

    # ── Input rendering: same as slm_stuck.py windowed mode ───────────────
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

    def build_text(messages):
        """Build one classification input from a session's messages.

        Re-packages the multi-turn chat format into a flat text: system
        prompt + rendered prior context + target step. Each labeled turn
        is a separate classification example — we unroll the session into
        per-turn inputs.

        messages is the raw session messages list [system, user, assistant, ...]
        Returns a list of (input_text, label_char) per assistant turn.
        """
        examples = []
        # Strip leading system message; messages[1:] alternates user, assistant
        msgs = [m for m in messages if m["role"] != "system"]
        # Pair up: (user_0, assistant_0), (user_1, assistant_1), ...
        pairs = []
        i = 0
        while i < len(msgs) - 1:
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
                i += 2
            else:
                i += 1

        # For each pair, build a windowed input: system + last N context + target
        for idx, (target_user, target_label) in enumerate(pairs):
            ctx_start = max(0, idx - args.context_steps)
            ctx_pairs = pairs[ctx_start:idx]
            parts = [SYSTEM_PROMPT, ""]
            if ctx_pairs:
                parts.append("Context (prior steps):")
                for j, (ctx_user, _) in enumerate(ctx_pairs):
                    parts.append(f"[{ctx_start + j}]")
                    parts.append(ctx_user)
            parts.append("")
            parts.append("TARGET step:")
            parts.append(f"[{idx}]")
            parts.append(target_user)
            text = "\n".join(parts)
            examples.append((text, target_label.strip()[:1]))
        return examples

    # ── Load + convert data ───────────────────────────────────────────────
    def load_jsonl(path):
        sessions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sessions.append(json.loads(line))
        return sessions

    print("\nloading jsonl", flush=True)
    train_raw = load_jsonl(REPO / args.train_file)
    val_raw = load_jsonl(REPO / args.val_file)
    if args.smoke:
        train_raw = train_raw[:500]
        val_raw = val_raw[:50]
        # Smoke config: tight to verify the training path actually works
        # before committing GPU time. If this crashes we learn cheaply
        # instead of locking up the system.
        args.batch_size = min(args.batch_size, 2)
        args.max_length = min(args.max_length, 512)
        args.grad_accum = 1
        args.epochs = 1
        print(f"  [smoke] forcing batch={args.batch_size} "
              f"max_length={args.max_length} epochs=1", flush=True)
    print(f"  train sessions: {len(train_raw)}  val sessions: {len(val_raw)}",
          flush=True)

    print("expanding sessions → per-turn examples", flush=True)
    t0 = time.time()
    train_ex = []
    for sess in train_raw:
        for text, label in build_text(sess["messages"]):
            if label in LABEL_TO_ID:
                train_ex.append((text, LABEL_TO_ID[label]))
    val_ex = []
    for sess in val_raw:
        for text, label in build_text(sess["messages"]):
            if label in LABEL_TO_ID:
                val_ex.append((text, LABEL_TO_ID[label]))
    print(f"  train examples: {len(train_ex)}  val examples: {len(val_ex)}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Class distribution
    train_labels = Counter(lbl for _, lbl in train_ex)
    val_labels = Counter(lbl for _, lbl in val_ex)
    print(f"  train dist: "
          f"P={train_labels[0]} ({100*train_labels[0]/len(train_ex):.1f}%) "
          f"S={train_labels[1]} ({100*train_labels[1]/len(train_ex):.1f}%) "
          f"U={train_labels[2]} ({100*train_labels[2]/len(train_ex):.1f}%)",
          flush=True)
    print(f"  val dist:   "
          f"P={val_labels[0]} ({100*val_labels[0]/len(val_ex):.1f}%) "
          f"S={val_labels[1]} ({100*val_labels[1]/len(val_ex):.1f}%) "
          f"U={val_labels[2]} ({100*val_labels[2]/len(val_ex):.1f}%)",
          flush=True)

    # ── Inverse-frequency class weights ───────────────────────────────────
    # Skip class U if it has 0 examples.
    total = len(train_ex)
    weights = torch.zeros(3, dtype=torch.float32)
    for c in range(3):
        n_c = train_labels[c] or 1  # avoid div-by-zero
        weights[c] = total / (3 * n_c)  # balanced reweight
    # Cap U at 0 if no examples (but we drop U during training anyway)
    weights = weights.to("cuda:0", dtype=torch.bfloat16)
    print(f"  class weights: P={weights[0].item():.2f} "
          f"S={weights[1].item():.2f} U={weights[2].item():.2f}", flush=True)

    # ── Dataset + collate ─────────────────────────────────────────────────
    class ExDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            text, label = self.examples[idx]
            return text, label

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(
        ExDataset(train_ex),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        ExDataset(val_ex),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # ── Attention probe: single forward+backward at the configured shape ─
    # This is a cheap way to detect OOM on the attention activations BEFORE
    # committing to a training run. At seq=2048 batch=8 on Ettin/gfx1201
    # this will OOM because torch-rocm 7.2 lacks a memory-efficient SDPA
    # kernel for this target; we'd rather see the OOM in a controlled probe
    # than have it corrupt our VRAM pool during the training loop.
    print(f"\n=== Attention probe (batch={args.batch_size} seq={args.max_length}) ===",
          flush=True)
    import torch.nn.functional as F
    probe_input_ids = torch.randint(
        0, tok.vocab_size, (args.batch_size, args.max_length),
        device="cuda:0", dtype=torch.long,
    )
    probe_attn_mask = torch.ones_like(probe_input_ids)
    probe_labels = torch.zeros(args.batch_size, device="cuda:0", dtype=torch.long)
    try:
        probe_out = model(
            input_ids=probe_input_ids,
            attention_mask=probe_attn_mask,
            labels=probe_labels,
        )
        probe_out.loss.backward()
        vram_after = torch.cuda.memory_allocated() / 1e9
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  probe OK — loss={probe_out.loss.item():.4f} "
              f"vram_now={vram_after:.1f} GB peak={peak_vram:.1f} GB",
              flush=True)
        if peak_vram > 14.0:
            print(f"  WARNING: peak VRAM {peak_vram:.1f} GB is >14 GB, "
                  f"training headroom is thin; consider smaller batch/seq",
                  flush=True)
        model.zero_grad(set_to_none=True)
        del probe_input_ids, probe_attn_mask, probe_labels, probe_out
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"\n\nProbe OOM at batch={args.batch_size} seq={args.max_length}. "
            f"Training would have crashed too.\n"
            f"Reduce --batch-size or --max-length and retry.\n\n"
            f"VRAM free at probe time: "
            f"{torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB\n"
            f"Original error: {e}"
        ) from e

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )
    total_opt_steps = max(
        len(train_loader) * args.epochs // args.grad_accum, 1
    )
    warmup_steps = int(total_opt_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )
    print(f"\ntraining: {len(train_ex)} × {args.epochs} epochs / "
          f"({args.batch_size} × {args.grad_accum}) "
          f"= {total_opt_steps} opt steps  warmup={warmup_steps}", flush=True)

    # ── Eval helper ───────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        n_batches = 0
        preds_per_class = Counter()
        confusion = [[0] * 3 for _ in range(3)]  # confusion[true][pred]
        for batch in val_loader:
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.float()
            loss = F.cross_entropy(logits, batch["labels"], weight=weights.float())
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=-1)
            for true, pred in zip(batch["labels"].cpu().tolist(),
                                  preds.cpu().tolist()):
                confusion[true][pred] += 1
                preds_per_class[pred] += 1
        # Macro F1
        f1s = []
        for c in (0, 1):  # just P and S (skip U since rare)
            tp = confusion[c][c]
            fp = sum(confusion[r][c] for r in range(3) if r != c)
            fn = sum(confusion[c][r] for r in range(3) if r != c)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s)
        model.train()
        return {
            "loss": total_loss / max(n_batches, 1),
            "macro_f1": macro_f1,
            "confusion": confusion,
            "preds": dict(preds_per_class),
        }

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n=== Training ===", flush=True)
    model.train()
    global_step = 0
    running_loss = 0.0
    running_n = 0
    t_train = time.time()
    eval_every = max(total_opt_steps // 6, 50)
    val_history = []
    best_f1 = -1.0
    best_step = 0

    # Note: crash isolation is provided by the Docker container teardown,
    # not Python-level try/finally — container exit releases HIP/KFD cleanly
    # even on segfault. See benchmarks/ettin_docker.sh.
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.float()
            loss = F.cross_entropy(logits, batch["labels"], weight=weights.float())
            loss = loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            running_n += 1

            if running_n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 20 == 0:
                    avg = running_loss / running_n
                    elapsed = time.time() - t_train
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  e{epoch} step {global_step}/{total_opt_steps} "
                          f"loss={avg:.4f} lr={lr_now:.2e} "
                          f"elapsed={elapsed:.0f}s "
                          f"vram={torch.cuda.memory_allocated()/1e9:.1f}GB",
                          flush=True)
                    running_loss = 0.0
                    running_n = 0

                if global_step % eval_every == 0:
                    stats = evaluate()
                    val_history.append((global_step, stats))
                    conf = stats["confusion"]
                    print(f"  [VAL @ {global_step}] loss={stats['loss']:.4f} "
                          f"macro_f1={stats['macro_f1']:.4f}", flush=True)
                    print(f"    preds: {stats['preds']}", flush=True)
                    print(f"    conf[true→pred]: "
                          f"P→P={conf[0][0]} P→S={conf[0][1]} "
                          f"S→P={conf[1][0]} S→S={conf[1][1]}",
                          flush=True)
                    if stats["macro_f1"] > best_f1:
                        best_f1 = stats["macro_f1"]
                        best_step = global_step
                        if not args.smoke:
                            out_dir = REPO / args.output_dir
                            out_dir.mkdir(parents=True, exist_ok=True)
                            model.save_pretrained(str(out_dir))
                            tok.save_pretrained(str(out_dir))
                            print(f"    saved best to {out_dir}", flush=True)

    # Final eval
    stats = evaluate()
    val_history.append((global_step, stats))
    print(f"\nfinal: loss={stats['loss']:.4f} macro_f1={stats['macro_f1']:.4f}",
          flush=True)

    print(f"\n=== val history ===", flush=True)
    print(f"{'step':>8}{'loss':>10}{'macro_f1':>10}", flush=True)
    for s, st in val_history:
        print(f"{s:>8}{st['loss']:>10.4f}{st['macro_f1']:>10.4f}", flush=True)
    print(f"\nbest macro_f1: {best_f1:.4f} at step {best_step}", flush=True)

    if not args.smoke:
        print(f"best model saved: {REPO / args.output_dir}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
