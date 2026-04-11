"""Add has_prior_output feature to existing labeled source files.

Re-runs abstract_trajectory on raw sessions to compute the has_prior_output
flag for each step, then patches the existing labeled .jsonl.gz files in-place.

This migration is needed because has_prior_output was added after initial
labeling. Windows already in the labeled files (including Sonnet-reviewed ones)
need the new feature before retraining.

Usage:
    python src/migrate_add_prior_output.py dataclaw
    python src/migrate_add_prior_output.py nlile
    python src/migrate_add_prior_output.py all
"""

import gc
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import abstract_trajectory, WINDOW_SIZE
from parse_dataclaw import parse_dataclaw_session

SOURCES_DIR = 'data/sources'

TOOL_TO_IDX_NAME = {
    'Bash': 'bash', 'bash': 'bash',
    'Read': 'view', 'read': 'view',
    'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
    'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
    'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
}


def build_prior_lookup_dataclaw():
    """Build {trajectory_id: {window_start: [has_prior_output per step]}} for dataclaw."""
    path = 'data/separate/dataclaw/woctordho/conversations.jsonl'
    lookup = {}
    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            tid = f"dc_{sess['session_id']}"
            parsed = parse_dataclaw_session(sess['messages'])
            if len(parsed) < WINDOW_SIZE:
                continue
            abstract = abstract_trajectory(parsed)
            per_window = {}
            for start in range(0, len(abstract) - WINDOW_SIZE + 1, 5):
                window = abstract[start:start + WINDOW_SIZE]
                per_window[start] = [1.0 if s.get('has_prior_output') else 0.0 for s in window]
            lookup[tid] = per_window
    print(f"  Built prior lookup for {len(lookup)} dataclaw sessions")
    return lookup


def parse_nlile_session(messages):
    steps = []
    pending = {}
    last_thinking = ''
    for msg in messages:
        if not isinstance(msg.get('content'), list):
            continue
        for block in msg['content']:
            btype = block.get('type', '')
            if btype == 'thinking':
                last_thinking = block.get('thinking', block.get('text', ''))
            elif btype == 'tool_use':
                name = block.get('name', '')
                inp = block.get('input', {})
                tool_id = block.get('id', '')
                tool = TOOL_TO_IDX_NAME.get(name, 'other')
                cmd = inp.get('command', inp.get('file_path', inp.get('pattern', '')))
                file_path = inp.get('file_path', inp.get('path', None))
                pending[tool_id] = {'tool': tool, 'cmd': cmd, 'file': file_path, 'thinking': last_thinking}
                last_thinking = ''
            elif btype == 'tool_result':
                tool_id = block.get('tool_use_id', '')
                if tool_id in pending:
                    tu = pending.pop(tool_id)
                    output = block.get('content', '')
                    if isinstance(output, list):
                        output = '\n'.join(b.get('text', '') for b in output
                                           if isinstance(b, dict) and b.get('type') == 'text')
                    tu['output'] = str(output) if output else ''
                    steps.append(tu)
    for tu in pending.values():
        tu['output'] = ''
        steps.append(tu)
    return steps


def build_prior_lookup_nlile():
    """Build prior lookup for nlile from parquet files."""
    import pyarrow.parquet as pq

    parquet_dir = 'data/separate/nlile_parquet/data'
    if not os.path.isdir(parquet_dir):
        print(f"  nlile parquet dir not found: {parquet_dir}")
        return {}

    lookup = {}
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))
    for fname in files:
        pf = pq.read_table(os.path.join(parquet_dir, fname))
        for i in range(len(pf)):
            row_id = pf.column('id')[i].as_py()
            tid = f"nlile_{row_id}"
            msgs_raw = pf.column('messages_json')[i].as_py()
            if not msgs_raw:
                continue
            msgs = json.loads(msgs_raw)
            parsed = parse_nlile_session(msgs)
            if len(parsed) < WINDOW_SIZE:
                continue
            abstract = abstract_trajectory(parsed)
            per_window = {}
            for start in range(0, len(abstract) - WINDOW_SIZE + 1, 5):
                window = abstract[start:start + WINDOW_SIZE]
                per_window[start] = [1.0 if s.get('has_prior_output') else 0.0 for s in window]
            lookup[tid] = per_window
        del pf
        gc.collect()
        print(f"  {fname}: {len(lookup)} sessions so far")

    print(f"  Built prior lookup for {len(lookup)} nlile sessions")
    return lookup


def patch_labeled_file(source, lookup):
    """Patch has_prior_output into each step of the labeled file."""
    labeled_gz = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl.gz')
    patched = 0
    skipped = 0
    total = 0

    lines_out = []
    with gzip.open(labeled_gz, 'rt') as f:
        for line in f:
            w = json.loads(line)
            total += 1
            tid = w['trajectory_id']
            ws = w['window_start']
            if tid in lookup and ws in lookup[tid]:
                priors = lookup[tid][ws]
                for j, step in enumerate(w['steps']):
                    step['has_prior_output'] = priors[j] if j < len(priors) else 0.0
                patched += 1
            else:
                # Window not found in lookup (session too short after re-parse, etc.)
                # Default to 0.0 — conservative, marks all as first-occurrence
                for step in w['steps']:
                    step.setdefault('has_prior_output', 0.0)
                skipped += 1
            lines_out.append(json.dumps(w))

    tmp = labeled_gz + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        for line in lines_out:
            f.write(line + '\n')
    os.replace(tmp, labeled_gz)

    print(f"  {source}: {total} windows — patched={patched}, defaulted={skipped}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/migrate_add_prior_output.py <source|all>")
        sys.exit(1)

    sources = ['nlile', 'dataclaw'] if sys.argv[1] == 'all' else [sys.argv[1]]

    for source in sources:
        print(f"\n=== Migrating {source} ===")
        if source == 'dataclaw':
            lookup = build_prior_lookup_dataclaw()
        elif source == 'nlile':
            lookup = build_prior_lookup_nlile()
        else:
            print(f"Unknown source: {source}")
            continue
        patch_labeled_file(source, lookup)
        print(f"Done. {source}_labeled.jsonl.gz updated.")


if __name__ == '__main__':
    main()
