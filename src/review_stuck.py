"""Remove Sonnet-confirmed false positives from source labeled file.

After running extract_stuck_for_review.py and Sonnet review agents:
- Sonnet says PRODUCTIVE or UNCLEAR → remove from labeled file (was a FP)
- Sonnet says STUCK → keep as-is

Usage:
    python src/review_stuck.py <source>

Reads results from: data/cc_sonnet_results/result_{source}_stuck_*.jsonl
Rewrites: data/sources/{source}_labeled.jsonl.gz
"""

import gzip
import json
import os
import sys
from collections import Counter

SOURCES_DIR = 'data/sources'
SONNET_RESULTS_DIR = 'data/cc_sonnet_results'


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/review_stuck.py <source>")
        sys.exit(1)

    source = sys.argv[1]
    labeled_gz = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl.gz')

    # Load Sonnet decisions for stuck review
    fp_ids = set()    # confirmed false positives → remove
    kept_ids = set()  # confirmed STUCK → keep

    result_files = sorted(
        f for f in os.listdir(SONNET_RESULTS_DIR)
        if f.startswith(f'result_{source}_stuck_') and f.endswith('.jsonl')
    )
    if not result_files:
        print(f"No stuck review results found for [{source}].")
        print(f"Expected: {SONNET_RESULTS_DIR}/result_{source}_stuck_*.jsonl")
        sys.exit(1)

    decisions = Counter()
    for rf in result_files:
        with open(os.path.join(SONNET_RESULTS_DIR, rf)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = r.get('label', 'STUCK')
                rid = r.get('id', '')
                decisions[label] += 1
                if label in ('PRODUCTIVE', 'UNCLEAR'):
                    fp_ids.add(rid)
                else:
                    kept_ids.add(rid)

    print(f"Sonnet decisions: {dict(decisions)}")
    print(f"  Confirmed STUCK (keep): {len(kept_ids)}")
    print(f"  False positives (remove): {len(fp_ids)}")

    # Rewrite labeled file, dropping FPs
    kept = []
    removed = 0
    with gzip.open(labeled_gz, 'rt') as f:
        for line in f:
            w = json.loads(line)
            if w['label'] == 'STUCK':
                wid = f"{w['trajectory_id']}_w{w['window_start']}"
                if wid in fp_ids:
                    removed += 1
                    continue
            kept.append(line.rstrip())

    # Write back
    tmp = labeled_gz + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        for line in kept:
            f.write(line + '\n')
    os.replace(tmp, labeled_gz)

    print(f"\nRemoved {removed} false positives from {labeled_gz}")
    print(f"Remaining windows: {len(kept)}")
    print(f"\nRun merge_sources.py --force to rebuild train/test splits.")


if __name__ == '__main__':
    main()
