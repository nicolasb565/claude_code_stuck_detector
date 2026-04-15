#!/usr/bin/env python3
"""
Causal Sonnet labeler experiment.

Hypothesis: the "ceiling" we keep hitting with causal classifiers (LR ~F1
0.33, Ettin TBD) may be partly information-theoretic. The ground-truth
labels were produced by Sonnet reading the WHOLE transcript — so it could
look ahead to decide whether step i was stuck. A causal classifier
looking only at past+current cannot reproduce labels that depend on
future information, no matter how good the model is.

This script tests that directly. For a chosen OOD benchmark session:
  - Walk through steps 0..N-1 in order
  - At each step i, show Sonnet ONLY steps 0..i (no future)
  - Get a commit-now P/S/U label
  - Compare to the original full-transcript Sonnet label at step i

If the two Sonnet runs diverge on many steps, that's direct evidence that
the original labels encode future information and a causal classifier
can never match them.

Cost note: we use prompt caching on the system prompt + transcript prefix.
At 81 steps × average ~4k input tokens ≈ $0.50-1.50 for 33_geometry on
Claude Sonnet 4.6 with caching.

Usage:
  .venv/bin/python benchmarks/causal_label.py --task 33_geometry
  .venv/bin/python benchmarks/causal_label.py --task 04_sqlite_cte --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.label_session import _render_step  # noqa: E402

RUN_DIR = REPO / "benchmarks" / "results" / "comparison_off"
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3

CAUSAL_SYSTEM_PROMPT = """\
You are an ONLINE causal labeler for a Claude Code coding session. You
will be shown tool-call steps one at a time, in order. After each new
step is shown you MUST commit to a single-letter label for that step
BEFORE seeing any later steps. Your prediction is final.

Classify the LAST step shown as exactly one letter:
  P - productive: the agent is making progress (new approach, first
    attempt, or iteration that the past alone shows is making change)
  S - stuck: the agent is in a loop that is already visible in the
    past — the same command/error/edit has been repeated without
    progress in the steps you have already seen
  U - unsure: genuine ambiguity that you cannot resolve from the past

CRITICAL CONSTRAINT: You are making a REAL-TIME judgment. You do not
know what happens next. A step that LOOKS like it could go either way
must be labeled based only on what the past shows. If the agent just
ran a command that failed once, you cannot label it S yet — one failure
is not a loop. If the same command with the same error has already
happened two or three times in the shown past, you can label S.

Rules (evaluated strictly against the past only):
- First attempt at any command → P
- Different file/flag/approach from what you have already seen → P
- Exactly the same command + error has appeared at least twice in the
  past → this new occurrence is S
- Reading or searching for something you have already read/searched → S
- Genuinely cannot decide from the past alone → U (use sparingly)

Output: one character — P, S, or U — nothing else."""


def parse_transcript(path: Path) -> list[dict]:
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


def render_all(steps: list[dict]) -> list[str]:
    """Render each step as the exact same text Sonnet saw during the
    original full-transcript labeling. Returns a list of per-step text
    blocks that can be concatenated as needed.
    """
    return [_render_step(s, i) for i, s in enumerate(steps)]


def _call_sonnet(client, system_blocks, user_text, max_retries=MAX_RETRIES):
    """Call Sonnet with the given system (cached) + user message."""
    import anthropic  # pylint: disable=import-outside-toplevel

    attempt = 0
    while True:
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4,
                system=system_blocks,
                messages=[
                    {"role": "user", "content": user_text},
                ],
            )
            return resp
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            backoff = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries} after {backoff}s]: {e}",
                  flush=True)
            time.sleep(backoff)


def _parse_label(text: str) -> str:
    """Extract P/S/U from a response. Returns 'U' on parse failure."""
    if not text:
        return "U"
    for ch in text.upper():
        if ch in ("P", "S", "U"):
            return ch
    return "U"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="33_geometry",
                    help="OOD benchmark task dir name under benchmarks/results/comparison_off")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="limit to first N steps (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print cost estimate, do not call API")
    ap.add_argument("--out", default=None,
                    help="output file (default: data/generated/causal_label_<task>.json)")
    args = ap.parse_args()

    task_dir = RUN_DIR / args.task
    if not task_dir.is_dir():
        print(f"ERROR: {task_dir} not found", file=sys.stderr)
        return 1

    transcript_path = task_dir / "transcript_1.jsonl"
    labels_path = task_dir / "sonnet_labels.json"
    if not (transcript_path.exists() and labels_path.exists()):
        print(f"ERROR: missing transcript_1.jsonl or sonnet_labels.json in {task_dir}",
              file=sys.stderr)
        return 1

    # Parse + render
    steps = parse_transcript(transcript_path)
    original_labels = json.loads(labels_path.read_text())["labels"]
    n = min(len(steps), len(original_labels))
    if args.max_steps:
        n = min(n, args.max_steps)
    steps = steps[:n]
    original_labels = original_labels[:n]
    rendered = render_all(steps)

    print(f"task: {args.task}")
    print(f"n steps: {n}")
    print(f"original label distribution: "
          f"P={original_labels.count('PRODUCTIVE')} "
          f"S={original_labels.count('STUCK')} "
          f"U={original_labels.count('UNSURE')}")

    # Cost estimate — rough, assumes caching reads are 1/10 cost of writes
    # and most of the prefix is served from cache after step 2.
    total_prefix_chars = sum(len(r) + 5 for r in rendered)  # +5 for newlines
    system_chars = len(CAUSAL_SYSTEM_PROMPT)
    # With caching: step i costs full system+prefix[0:i] ONCE (first call
    # warms cache), then subsequent calls re-read the cached prefix at ~10%
    # cost and pay full price only on the new step text. Very rough model.
    full_prefix_tokens = (system_chars + total_prefix_chars) // 4
    avg_step_tokens = (sum(len(r) for r in rendered) // max(n, 1)) // 4
    # Assume: first call pays system + step0 at full cost; each subsequent
    # call pays 10% of the prior prefix (cached) + 100% of the new step.
    # So total ≈ full_prefix*0.1*(n-1) + n*avg_step + system*n*(0.1 if cached else 1)
    # Simplification: total input ≈ 0.15 × full_prefix × n with caching
    est_input_tokens = int(0.15 * full_prefix_tokens * n + 500 * n)
    est_output_tokens = 4 * n  # ~1 token + overhead
    # Claude Sonnet 4.6 pricing: $3/Mtok input, $15/Mtok output, cache write 1.25×, cache read 0.1×
    est_cost = est_input_tokens / 1_000_000 * 3.0 + est_output_tokens / 1_000_000 * 15.0
    print(f"\nCost estimate (with prompt caching):")
    print(f"  avg step:         ~{avg_step_tokens} tokens")
    print(f"  full prefix:      ~{full_prefix_tokens} tokens")
    print(f"  est input tokens: ~{est_input_tokens:,}")
    print(f"  est output tokens: ~{est_output_tokens:,}")
    print(f"  est cost:         ~${est_cost:.2f}")

    if args.dry_run:
        print("\n[dry-run, exiting]")
        return 0

    # Load .env and get client
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    import anthropic  # pylint: disable=import-outside-toplevel
    client = anthropic.Anthropic(api_key=api_key)

    # Causal walk: at step i, build prompt from system + all rendered
    # steps[0..i], ask model to label the last step shown.
    print(f"\n=== Causal labeling {args.task} ===")
    causal_labels = []
    results = []  # list of dicts with per-step detail
    t0 = time.time()

    for i in range(n):
        # Build system blocks with prompt caching on the transcript prefix.
        # Two blocks: the fixed system prompt, and the growing transcript
        # prefix up through step i. Anthropic caching keys on exact prefix
        # match, so only the suffix that changes between calls is billed
        # at full rate on subsequent calls.
        prefix_text = "\n\n".join(rendered[:i + 1])
        system_blocks = [
            {
                "type": "text",
                "text": CAUSAL_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Session transcript so far (steps 0..{i}):\n\n{prefix_text}",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        user_text = (
            f"Classify step [{i}] — the LAST step shown — using only the "
            f"information visible in the transcript above. You have no "
            f"knowledge of what comes next. Output a single letter: P, S, or U."
        )

        t_call = time.time()
        try:
            resp = _call_sonnet(client, system_blocks, user_text)
        except Exception as e:
            print(f"  step {i}: API error after retries: {e}", flush=True)
            causal_labels.append("U")
            results.append({"step": i, "causal": "U", "error": str(e)})
            continue

        call_dur = time.time() - t_call
        # Extract text
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
        label = _parse_label(text)
        causal_labels.append(label)

        # Pull cache/usage stats
        usage = resp.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_write = getattr(usage, "cache_creation_input_tokens", 0)
        in_tokens = usage.input_tokens
        out_tokens = usage.output_tokens

        orig = original_labels[i][0] if original_labels[i] else "?"
        agree = "✓" if label == orig else "✗"
        results.append({
            "step": i,
            "causal": label,
            "original": original_labels[i],
            "agree": label == orig,
            "raw_response": text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "dur_s": round(call_dur, 2),
        })
        print(f"  step {i:3d}: causal={label} original={orig} {agree}  "
              f"(in={in_tokens} cr={cache_read} cw={cache_write} "
              f"{call_dur:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ===")

    # Compute summary
    n_agree = sum(1 for r in results if r.get("agree"))
    total_in = sum(r.get("input_tokens", 0) for r in results)
    total_out = sum(r.get("output_tokens", 0) for r in results)
    total_cr = sum(r.get("cache_read", 0) for r in results)
    total_cw = sum(r.get("cache_write", 0) for r in results)

    # Confusion
    conf = {"orig_PRODUCTIVE": {"P": 0, "S": 0, "U": 0},
            "orig_STUCK":      {"P": 0, "S": 0, "U": 0},
            "orig_UNSURE":     {"P": 0, "S": 0, "U": 0}}
    for r in results:
        if "original" not in r:
            continue
        key = f"orig_{r['original']}"
        if key in conf:
            conf[key][r["causal"]] += 1

    print(f"\nAgreement: {n_agree}/{n} ({100*n_agree/n:.1f}%)")
    print(f"\nConfusion (row = original Sonnet label, col = causal Sonnet label):")
    print(f"  {'':<18} {'causal_P':>10} {'causal_S':>10} {'causal_U':>10}")
    for orig in ("PRODUCTIVE", "STUCK", "UNSURE"):
        row = conf[f"orig_{orig}"]
        print(f"  {orig:<18} {row['P']:>10} {row['S']:>10} {row['U']:>10}")

    # Breakdown: of the original STUCK labels, how many did causal Sonnet
    # also call S (i.e., were detectable from past alone)?
    orig_stuck = [r for r in results if r.get("original") == "STUCK"]
    orig_prod = [r for r in results if r.get("original") == "PRODUCTIVE"]
    causal_recovered_stuck = sum(1 for r in orig_stuck if r["causal"] == "S")
    causal_false_stuck = sum(1 for r in orig_prod if r["causal"] == "S")

    if orig_stuck:
        recall = causal_recovered_stuck / len(orig_stuck)
        print(f"\nSTUCK recall under causal labeling: {causal_recovered_stuck}/"
              f"{len(orig_stuck)} = {100*recall:.1f}%")
        print(f"  (This is an UPPER BOUND on what any causal classifier can achieve.)")
    if orig_prod:
        print(f"PRODUCTIVE → causal S: {causal_false_stuck}/{len(orig_prod)} "
              f"(false stuck under causal framing)")

    print(f"\nAPI usage: in={total_in} out={total_out} "
          f"cache_read={total_cr} cache_write={total_cw}")
    actual_cost = (
        (total_in - total_cr) / 1_000_000 * 3.0
        + total_cr / 1_000_000 * 0.3
        + total_cw / 1_000_000 * 3.75
        + total_out / 1_000_000 * 15.0
    )
    print(f"Actual cost: ~${actual_cost:.3f}")

    # Save
    out_path = args.out or f"data/generated/causal_label_{args.task}.json"
    out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "task": args.task,
            "n_steps": n,
            "causal_labels": causal_labels,
            "original_labels": original_labels,
            "per_step": results,
            "agreement": n_agree / n,
            "confusion": conf,
            "usage": {
                "input": total_in, "output": total_out,
                "cache_read": total_cr, "cache_write": total_cw,
            },
            "cost_usd": round(actual_cost, 4),
            "model": MODEL,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
