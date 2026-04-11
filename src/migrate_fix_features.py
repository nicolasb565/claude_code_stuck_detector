"""Patch stored feature values after three abstract_trajectory fixes:

  Fix #1 — edit/create/submit output_similarity forced to 0.0, has_prior_output=False
  Fix #2 — non-bash cmd_hash now prefixed with tool name (edit:foo.py vs view:foo.py)
  Fix #3 — system-reminder blocks stripped from output before is_error / output_similarity

Re-runs abstract_trajectory (with fixes) on every raw session, then patches the
numeric feature values in the existing labeled .jsonl.gz files in-place.
Labels (STUCK/PRODUCTIVE) are preserved — only feature values change.

Usage:
    python src/migrate_fix_features.py dataclaw
    python src/migrate_fix_features.py nlile
    python src/migrate_fix_features.py work_embedded_c
    python src/migrate_fix_features.py all
"""

import gc
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import abstract_trajectory, WINDOW_SIZE

SOURCES_DIR = 'data/sources'

# Features to patch per step (all numeric features that abstract_trajectory produces)
STEP_FEATURES = [
    'steps_since_same_tool', 'steps_since_same_file', 'steps_since_same_cmd',
    'tool_count_in_window', 'file_count_in_window', 'cmd_count_in_window',
    'output_similarity', 'has_prior_output', 'output_length', 'is_error',
    'step_index_norm', 'false_start', 'strategy_change', 'circular_lang',
    'thinking_length', 'self_similarity',
]


def build_lookup(parsed_by_id):
    """Given {traj_id: parsed_steps}, return {traj_id: {window_start: [step_feature_dicts]}}."""
    lookup = {}
    for tid, parsed in parsed_by_id.items():
        if len(parsed) < WINDOW_SIZE:
            continue
        abstract = abstract_trajectory(parsed)
        per_window = {}
        for start in range(0, len(abstract) - WINDOW_SIZE + 1, 5):
            window = abstract[start:start + WINDOW_SIZE]
            per_window[start] = [
                {f: (1.0 if v is True else (0.0 if v is False else v))
                 for f, v in s.items() if f in STEP_FEATURES}
                for s in window
            ]
        lookup[tid] = per_window
    return lookup


def parse_nlile_session(messages):
    from abstract_trajectory import TOOL_TO_IDX
    TOOL_TO_NAME = {
        'Bash': 'bash', 'bash': 'bash',
        'Read': 'view', 'read': 'view',
        'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
        'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
        'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
    }
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
                tool = TOOL_TO_NAME.get(name, 'other')
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


def build_lookup_nlile():
    import pyarrow.parquet as pq
    parquet_dir = 'data/separate/nlile_parquet/data'
    if not os.path.isdir(parquet_dir):
        print(f"  nlile parquet dir not found: {parquet_dir}")
        return {}

    lookup = {}
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))
    for fname in files:
        pf = pq.read_table(os.path.join(parquet_dir, fname))
        batch = {}
        for i in range(len(pf)):
            row_id = pf.column('id')[i].as_py()
            tid = f"nlile_{row_id}"
            msgs_raw = pf.column('messages_json')[i].as_py()
            if not msgs_raw:
                continue
            msgs = json.loads(msgs_raw)
            parsed = parse_nlile_session(msgs)
            if len(parsed) >= WINDOW_SIZE:
                batch[tid] = parsed
        lookup.update(build_lookup(batch))
        del pf
        gc.collect()
        print(f"  {fname}: {len(lookup)} sessions so far")

    print(f"  Built lookup for {len(lookup)} nlile sessions")
    return lookup


def build_lookup_dataclaw():
    from parse_dataclaw import parse_dataclaw_session
    path = 'data/separate/dataclaw/woctordho/conversations.jsonl'
    batch = {}
    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            tid = f"dc_{sess['session_id']}"
            parsed = parse_dataclaw_session(sess['messages'])
            if len(parsed) >= WINDOW_SIZE:
                batch[tid] = parsed
    lookup = build_lookup(batch)
    print(f"  Built lookup for {len(lookup)} dataclaw sessions")
    return lookup


def build_lookup_work_embedded_c():
    gz = 'data/separate/work_embedded_c_sessions.jsonl.gz'
    if not os.path.exists(gz):
        print(f"  {gz} not found")
        return {}
    TOOL_TO_NAME = {
        'Bash': 'bash', 'bash': 'bash',
        'Read': 'view', 'read': 'view',
        'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
        'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
        'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
    }
    batch = {}
    with gzip.open(gz, 'rt') as f:
        for line in f:
            sess = json.loads(line)
            tid = sess.get('session_id', '')
            if not tid.startswith('work_embedded_c_'):
                tid = f"work_embedded_c_{tid}"
            msgs = sess.get('messages', [])
            parsed = parse_nlile_session(msgs)  # same format
            if len(parsed) >= WINDOW_SIZE:
                batch[tid] = parsed
    lookup = build_lookup(batch)
    print(f"  Built lookup for {len(lookup)} work_embedded_c sessions")
    return lookup


def patch_labeled_file(source, lookup):
    labeled_gz = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl.gz')
    patched = skipped = total = 0
    lines_out = []

    with gzip.open(labeled_gz, 'rt') as f:
        for line in f:
            w = json.loads(line)
            total += 1
            tid = w['trajectory_id']
            ws = w['window_start']
            if tid in lookup and ws in lookup[tid]:
                new_steps = lookup[tid][ws]
                for j, step in enumerate(w['steps']):
                    if j < len(new_steps):
                        step.update(new_steps[j])
                patched += 1
            else:
                skipped += 1
            lines_out.append(json.dumps(w))

    tmp = labeled_gz + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        for line in lines_out:
            f.write(line + '\n')
    os.replace(tmp, labeled_gz)

    print(f"  {source}: {total} windows — patched={patched}, skipped={skipped}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/migrate_fix_features.py <source|all>")
        sys.exit(1)

    arg = sys.argv[1]
    sources = ['nlile', 'dataclaw', 'work_embedded_c'] if arg == 'all' else [arg]

    for source in sources:
        print(f"\n=== Migrating {source} ===")
        if source == 'nlile':
            lookup = build_lookup_nlile()
        elif source == 'dataclaw':
            lookup = build_lookup_dataclaw()
        elif source == 'work_embedded_c':
            lookup = build_lookup_work_embedded_c()
        else:
            print(f"Unknown source: {source}")
            continue
        patch_labeled_file(source, lookup)
        print(f"Done. {source}_labeled.jsonl.gz updated.")


if __name__ == '__main__':
    main()
