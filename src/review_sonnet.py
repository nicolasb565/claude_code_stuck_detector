"""Apply Sonnet review decisions to candidate windows.

Reads batch files and their corresponding Sonnet results, then:
  STUCK / PRODUCTIVE → extract _full_window, set label, append to labeled file
  UNCLEAR            → escalate to Opus (written to data/review/escalated/)

Run after Sonnet review agents have processed all batches for the source.

Directory layout:
  data/review/batches/{source}_batch_*.jsonl        — candidate items (from label_sessions.py)
  data/review/results/sonnet/{source}_batch_*.jsonl — Sonnet decisions
  data/review/escalated/{source}_batch_*.jsonl      — escalated to Opus

Usage:
  python src/review_sonnet.py <source>
  python src/review_sonnet.py <source> --status    # check progress without merging
"""

import json
import os
import sys
from collections import Counter

SOURCES_DIR  = 'data/sources'
BATCHES_DIR  = 'data/review/batches'
RESULTS_DIR  = 'data/review/results/sonnet'
ESCALATE_DIR = 'data/review/escalated'
BATCH_SIZE   = 50


def load_batches(source):
    """Load all candidate items keyed by id."""
    items = {}
    batch_files = sorted(
        f for f in os.listdir(BATCHES_DIR)
        if f.startswith(f'{source}_batch_') and f.endswith('.jsonl')
    )
    for fname in batch_files:
        with open(os.path.join(BATCHES_DIR, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    items[item['id']] = item
    return items, batch_files


def status(source):
    _, batch_files = load_batches(source)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = pending = 0
    for fname in batch_files:
        result = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(result):
            done += 1
        else:
            pending += 1
    print(f"[{source}] Sonnet review: {done}/{len(batch_files)} batches done, {pending} pending")
    if pending:
        print(f"  Pending batches in: {BATCHES_DIR}/")
        print(f"  Results expected in: {RESULTS_DIR}/")


def merge(source):
    os.makedirs(ESCALATE_DIR, exist_ok=True)

    items, batch_files = load_batches(source)
    if not items:
        print(f"No candidate batches found for [{source}] in {BATCHES_DIR}/")
        return

    result_files = sorted(
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(f'{source}_batch_') and f.endswith('.jsonl')
    )
    if not result_files:
        print(f"No Sonnet results found for [{source}] in {RESULTS_DIR}/")
        print(f"Run Sonnet review agents first.")
        return

    decisions  = Counter()
    resolved   = []   # (label, full_window)
    escalated  = []   # items where Sonnet said UNCLEAR

    for fname in result_files:
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid   = r.get('id', '')
                label = r.get('label', 'UNCLEAR').upper()
                decisions[label] += 1

                if label in ('STUCK', 'PRODUCTIVE') and rid in items:
                    full_window = items[rid].get('_full_window', {})
                    full_window['label'] = label
                    resolved.append(full_window)
                elif label == 'UNCLEAR' and rid in items:
                    item = items[rid].copy()
                    item['sonnet_reason'] = r.get('reason', '')
                    escalated.append(item)

    print(f"Sonnet decisions: {dict(decisions)}")
    print(f"  Resolved: {len(resolved)}  Escalated to Opus: {len(escalated)}")

    # Append resolved windows to labeled file
    labeled_file = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl')
    with open(labeled_file, 'a') as f:
        for w in resolved:
            f.write(json.dumps(w) + '\n')
    print(f"  Appended {len(resolved)} windows to {labeled_file}")

    # Write escalated items for Opus
    if escalated:
        n_batches = 0
        for i in range(0, len(escalated), BATCH_SIZE):
            batch = escalated[i:i + BATCH_SIZE]
            out   = os.path.join(ESCALATE_DIR, f'{source}_batch_{n_batches:04d}.jsonl')
            with open(out, 'w') as f:
                for item in batch:
                    f.write(json.dumps(item) + '\n')
            n_batches += 1
        print(f"  Wrote {n_batches} Opus batches to {ESCALATE_DIR}/")
        print(f"\nNext: run Opus review agents on {ESCALATE_DIR}/{source}_batch_*.jsonl")
        print(f"      Results expected in: data/review/results/opus/{source}_batch_*.jsonl")
        print(f"Then: python src/review_opus.py {source}")
    else:
        print(f"\nNext: gzip -k {labeled_file}")
        print(f"      python src/merge_sources.py --force")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/review_sonnet.py <source> [--status]")
        sys.exit(1)

    source = sys.argv[1]
    if '--status' in sys.argv:
        status(source)
    else:
        merge(source)


if __name__ == '__main__':
    main()
