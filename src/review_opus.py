"""Apply Opus review decisions for items Sonnet could not resolve.

Reads escalated items and their Opus results, then:
  STUCK / PRODUCTIVE → extract _full_window, set label, append to labeled file
  UNCLEAR            → dropped (Opus is the final arbiter)

Run after review_sonnet.py has written escalated items and Opus agents
have processed them.

Directory layout:
  data/review/escalated/{source}_batch_*.jsonl          — items escalated by review_sonnet.py
  data/review/results/opus/{source}_batch_*.jsonl       — Opus decisions

Usage:
  python src/review_opus.py <source>
  python src/review_opus.py <source> --status
"""

import json
import os
import sys
from collections import Counter

SOURCES_DIR  = 'data/sources'
ESCALATE_DIR = 'data/review/escalated'
RESULTS_DIR  = 'data/review/results/opus'


def load_escalated(source):
    items = {}
    if not os.path.isdir(ESCALATE_DIR):
        return items
    for fname in sorted(os.listdir(ESCALATE_DIR)):
        if fname.startswith(f'{source}_batch_') and fname.endswith('.jsonl'):
            with open(os.path.join(ESCALATE_DIR, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        items[item['id']] = item
    return items


def status(source):
    items = load_escalated(source)
    if not items:
        print(f"[{source}] No escalated items found in {ESCALATE_DIR}/")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch_files = sorted(
        f for f in os.listdir(ESCALATE_DIR)
        if f.startswith(f'{source}_batch_') and f.endswith('.jsonl')
    )
    done = pending = 0
    for fname in batch_files:
        result = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(result):
            done += 1
        else:
            pending += 1
    print(f"[{source}] Opus review: {done}/{len(batch_files)} batches done, {pending} pending")
    print(f"  {len(items)} total escalated items")


def merge(source):
    items = load_escalated(source)
    if not items:
        print(f"No escalated items found for [{source}] in {ESCALATE_DIR}/")
        return

    if not os.path.isdir(RESULTS_DIR):
        print(f"No Opus results directory found: {RESULTS_DIR}/")
        return

    result_files = sorted(
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(f'{source}_batch_') and f.endswith('.jsonl')
    )
    if not result_files:
        print(f"No Opus results found for [{source}] in {RESULTS_DIR}/")
        return

    decisions = Counter()
    resolved  = []

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
                # UNCLEAR → dropped (no further escalation)

    dropped = decisions.get('UNCLEAR', 0)
    print(f"Opus decisions: {dict(decisions)}")
    print(f"  Resolved: {len(resolved)}  Dropped (still UNCLEAR): {dropped}")

    labeled_file = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl')
    with open(labeled_file, 'a') as f:
        for w in resolved:
            f.write(json.dumps(w) + '\n')
    print(f"  Appended {len(resolved)} windows to {labeled_file}")

    print(f"\nNext: gzip -k {labeled_file}")
    print(f"      python src/merge_sources.py --force")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/review_opus.py <source> [--status]")
        sys.exit(1)

    source = sys.argv[1]
    if '--status' in sys.argv:
        status(source)
    else:
        merge(source)


if __name__ == '__main__':
    main()
