"""Extract deterministic STUCK windows with raw text for Sonnet review.

Reads a source's labeled .jsonl.gz, finds STUCK windows, re-parses the raw
sessions to recover step text, and writes Sonnet review batches.

After Sonnet review, use review_stuck.py to remove confirmed false positives
from the labeled file.

Usage:
    python src/extract_stuck_for_review.py dataclaw
    python src/extract_stuck_for_review.py nlile

Results go to: data/cc_stuck_review_batches/{source}_stuck_batch_XXXX.jsonl
Review results expected in: data/cc_sonnet_results/result_{source}_stuck_{num}.jsonl
"""

import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import abstract_trajectory, WINDOW_SIZE, TOOL_NAMES
from parse_dataclaw import parse_dataclaw_session

SOURCES_DIR = 'data/sources'
OUTPUT_DIR = 'data/cc_stuck_review_batches'
BATCH_SIZE = 50

TOOL_TO_IDX_NAME = {
    'Bash': 'bash', 'bash': 'bash',
    'Read': 'view', 'read': 'view',
    'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
    'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
    'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
}


def load_stuck_windows(source):
    """Load STUCK window locations from labeled file."""
    labeled_gz = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl.gz')
    stuck = []
    with gzip.open(labeled_gz, 'rt') as f:
        for line in f:
            w = json.loads(line)
            if w['label'] == 'STUCK':
                stuck.append({
                    'trajectory_id': w['trajectory_id'],
                    'window_start': w['window_start'],
                    'window_features': w.get('window_features', {}),
                })
    print(f"[{source}] Found {len(stuck)} deterministic STUCK windows")
    return stuck


def build_review_item(stuck_meta, parsed_steps, abstract_steps):
    """Build a Sonnet review item for a STUCK window."""
    start = stuck_meta['window_start']
    window_parsed = parsed_steps[start:start + WINDOW_SIZE]
    window_abstract = abstract_steps[start:start + WINDOW_SIZE]

    review_steps = []
    for j, (raw, abst) in enumerate(zip(window_parsed, window_abstract)):
        tool = abst.get('tool', 'other')
        step_data = {
            'tool': tool,
            'since_cmd': round(abst['steps_since_same_cmd'], 2),
            'since_file': round(abst['steps_since_same_file'], 2),
            'out_sim': round(abst['output_similarity'], 2),
            'has_prior': int(abst.get('has_prior_output', False)),
            'cmd_count': round(abst['cmd_count_in_window'], 2),
            'error': 1 if abst['is_error'] else 0,
            'out_len': round(abst['output_length'], 2),
        }
        if raw.get('cmd'):
            step_data['cmd'] = raw['cmd'][:200]
        if raw.get('file'):
            step_data['file'] = raw['file']
        if raw.get('output'):
            lines = raw['output'].strip().split('\n')
            if len(lines) <= 10:
                step_data['output_snippet'] = raw['output'][:500]
            else:
                head = '\n'.join(lines[:5])
                tail = '\n'.join(lines[-5:])
                step_data['output_snippet'] = f"{head}\n... ({len(lines)} lines) ...\n{tail}"[:500]
        if raw.get('thinking'):
            step_data['thinking_snippet'] = raw['thinking'][:300]
        review_steps.append(step_data)

    tid = stuck_meta['trajectory_id']
    ws = stuck_meta['window_start']
    return {
        'id': f"{tid}_w{ws}",
        'trajectory_id': tid,
        'window_start': ws,
        'steps': review_steps,
        'window_features': stuck_meta['window_features'],
        'review_type': 'stuck_validation',
    }


def process_dataclaw(stuck_windows):
    """Re-parse dataclaw sessions and build review items."""
    path = 'data/separate/dataclaw/woctordho/conversations.jsonl'

    # Index stuck windows by trajectory_id
    by_traj = defaultdict(list)
    for sw in stuck_windows:
        # dataclaw trajectory_ids are dc_{session_id}
        by_traj[sw['trajectory_id']].append(sw)

    items = []
    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            tid = f"dc_{sess['session_id']}"
            if tid not in by_traj:
                continue
            parsed = parse_dataclaw_session(sess['messages'])
            if len(parsed) < WINDOW_SIZE:
                continue
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                continue
            for sw in by_traj[tid]:
                start = sw['window_start']
                if start + WINDOW_SIZE > len(abstract):
                    continue
                item = build_review_item(sw, parsed, abstract)
                items.append(item)

    return items


def process_nlile(stuck_windows):
    """Re-parse nlile parquet sessions and build review items."""
    import pyarrow.parquet as pq

    parquet_dir = 'data/separate/nlile_parquet/data'
    if not os.path.isdir(parquet_dir):
        print(f"  nlile parquet dir not found: {parquet_dir}")
        return []

    # Index stuck windows by trajectory_id
    by_traj = defaultdict(list)
    for sw in stuck_windows:
        by_traj[sw['trajectory_id']].append(sw)

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

    items = []
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))
    for fname in files:
        pf = pq.read_table(os.path.join(parquet_dir, fname))
        for i in range(len(pf)):
            row_id = pf.column('id')[i].as_py()
            tid = f"nlile_{row_id}"
            if tid not in by_traj:
                continue
            msgs_raw = pf.column('messages_json')[i].as_py()
            if not msgs_raw:
                continue
            msgs = json.loads(msgs_raw)
            parsed = parse_nlile_session(msgs)
            if len(parsed) < WINDOW_SIZE:
                continue
            abstract = abstract_trajectory(parsed)
            for sw in by_traj[tid]:
                start = sw['window_start']
                if start + WINDOW_SIZE > len(abstract):
                    continue
                item = build_review_item(sw, parsed, abstract)
                items.append(item)

    return items


def write_batches(source, items):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_batches = 0
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        out = os.path.join(OUTPUT_DIR, f'{source}_stuck_batch_{n_batches:04d}.jsonl')
        with open(out, 'w') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
        n_batches += 1
    print(f"Wrote {n_batches} batches ({len(items)} items) to {OUTPUT_DIR}/")
    print(f"\nNext: run Sonnet review agents on {OUTPUT_DIR}/{source}_stuck_batch_*.jsonl")
    print(f"      Results expected in data/cc_sonnet_results/result_{source}_stuck_XXXX.jsonl")
    print(f"Then: python src/review_stuck.py {source}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/extract_stuck_for_review.py <source>")
        print("  source: dataclaw, nlile")
        sys.exit(1)

    source = sys.argv[1]
    stuck_windows = load_stuck_windows(source)

    if source == 'dataclaw':
        items = process_dataclaw(stuck_windows)
    elif source == 'nlile':
        items = process_nlile(stuck_windows)
    else:
        print(f"Unknown source: {source}. Add a parser in this script.")
        sys.exit(1)

    print(f"Built {len(items)} review items (missed {len(stuck_windows) - len(items)} — short sessions)")
    write_batches(source, items)


if __name__ == '__main__':
    main()
