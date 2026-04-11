"""Parse raw sessions, extract features, and label windows for training.

Heuristics emit only two labels:
  PRODUCTIVE — written directly to the labeled file (high-precision rules)
  CANDIDATE  — batched for Sonnet review; becomes STUCK or PRODUCTIVE after review

No STUCK label is ever written directly by this script. All STUCK labels in the
final training data come from Sonnet/Opus review of CANDIDATE windows.

Directory layout:
  data/sources/{source}_labeled.jsonl   — PRODUCTIVE windows (grows after reviews)
  data/review/batches/{source}_batch_NNNN.jsonl — CANDIDATE items for Sonnet

After labeling, run:
  python src/review_sonnet.py <source>
  python src/review_opus.py <source>    # only if Sonnet escalated anything
  python src/merge_sources.py --force

Usage:
  python src/label_sessions.py nlile
  python src/label_sessions.py dataclaw
  python src/label_sessions.py <source> <path/to/sessions.jsonl>
"""

import gc
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import (
    abstract_trajectory, compute_window_features, precompute_review_counts,
    WINDOW_SIZE, STRIDE,
)
from parse_dataclaw import parse_dataclaw_session, has_outputs

SOURCES_DIR  = 'data/sources'
BATCHES_DIR  = 'data/review/batches'
BATCH_SIZE   = 50

TOOL_NAMES = ['bash', 'edit', 'view', 'search', 'create', 'submit', 'other']

TOOL_MAP = {
    'Bash': 'bash', 'bash': 'bash',
    'Read': 'view', 'read': 'view',
    'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
    'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
    'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
}


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

def classify(precomputed):
    """
    Return (label, reason) where label is PRODUCTIVE or CANDIDATE.

    PRODUCTIVE rules are high-precision: a window matching these is
    extremely unlikely to be a stuck loop.

    Everything else is CANDIDATE — sent to Sonnet for STUCK/PRODUCTIVE
    determination. This includes what the old pipeline called STUCK and UNCLEAR.
    """
    tight     = precomputed['tight_loop_steps']
    diverse   = precomputed['diverse_steps']
    errors    = precomputed['error_steps']
    has_submit = precomputed.get('has_submit', False)
    ctx = f"tight={tight} diverse={diverse} errors={errors}"

    if tight == 0:
        return 'PRODUCTIVE', f"{ctx} → no tight loop"
    if diverse >= tight + 3:
        return 'PRODUCTIVE', f"{ctx} → diversity dominates"
    if diverse >= 6:
        return 'PRODUCTIVE', f"{ctx} → high diversity"
    if has_submit and diverse >= 2:
        return 'PRODUCTIVE', f"{ctx} → submitted with diversity"

    return 'CANDIDATE', f"{ctx} → needs review"


# ---------------------------------------------------------------------------
# Window creation
# ---------------------------------------------------------------------------

def _step_features(s):
    return {
        'tool_idx':              s['tool_idx'],
        'steps_since_same_tool': s['steps_since_same_tool'],
        'steps_since_same_file': s['steps_since_same_file'],
        'steps_since_same_cmd':  s['steps_since_same_cmd'],
        'tool_count_in_window':  s['tool_count_in_window'],
        'file_count_in_window':  s['file_count_in_window'],
        'cmd_count_in_window':   s['cmd_count_in_window'],
        'output_similarity':     s['output_similarity'],
        'has_prior_output':      1.0 if s.get('has_prior_output') else 0.0,
        'output_length':         s['output_length'],
        'is_error':              1.0 if s['is_error'] else 0.0,
        'step_index_norm':       s['step_index_norm'],
        'false_start':           1.0 if s['false_start'] else 0.0,
        'strategy_change':       1.0 if s['strategy_change'] else 0.0,
        'circular_lang':         1.0 if s['circular_lang'] else 0.0,
        'thinking_length':       s['thinking_length'],
        'self_similarity':       s['self_similarity'],
    }


def _review_step(abstract_step, raw_step):
    """Build a human-readable review step with truncated raw text."""
    s = abstract_step
    tool = TOOL_NAMES[s['tool_idx']] if isinstance(s['tool_idx'], int) else s.get('tool', 'other')
    item = {
        'tool':       tool,
        'since_cmd':  round(s['steps_since_same_cmd'], 2),
        'since_file': round(s['steps_since_same_file'], 2),
        'out_sim':    round(s['output_similarity'], 2),
        'has_prior':  int(s.get('has_prior_output', False)),
        'cmd_count':  round(s['cmd_count_in_window'], 2),
        'error':      1 if s['is_error'] else 0,
        'out_len':    round(s['output_length'], 2),
    }
    if raw_step:
        if raw_step.get('cmd'):
            item['cmd'] = raw_step['cmd'][:200]
        if raw_step.get('file'):
            item['file'] = raw_step['file']
        if raw_step.get('output'):
            lines = raw_step['output'].strip().split('\n')
            if len(lines) <= 10:
                item['output_snippet'] = raw_step['output'][:500]
            else:
                head = '\n'.join(lines[:5])
                tail = '\n'.join(lines[-5:])
                item['output_snippet'] = f"{head}\n... ({len(lines)} lines) ...\n{tail}"[:500]
        if raw_step.get('thinking'):
            item['thinking_snippet'] = raw_step['thinking'][:300]
    return item


def create_windows(abstract_seq, trajectory_id, parsed_steps=None):
    """
    Slide window over abstract_seq.
    Returns (productive_windows, candidate_items).

    candidate_items embed the full training window (_full_window) so
    review_sonnet.py can recover it without re-parsing the source.
    """
    productive = []
    candidates = []

    for start in range(0, len(abstract_seq) - WINDOW_SIZE + 1, STRIDE):
        window  = abstract_seq[start:start + WINDOW_SIZE]
        win_feats   = compute_window_features(window)
        precomputed = precompute_review_counts(window)
        label, reason = classify(precomputed)

        full_window = {
            'trajectory_id': trajectory_id,
            'window_start':  start,
            'label':         label,          # placeholder; overwritten by reviewer
            'label_source':  'heuristic',    # overwritten by review scripts
            'steps':         [_step_features(s) for s in window],
            'window_features': win_feats,
        }

        if label == 'PRODUCTIVE':
            productive.append(full_window)
        else:
            raw_window = parsed_steps[start:start + WINDOW_SIZE] if parsed_steps else []
            review_steps = [
                _review_step(window[j], raw_window[j] if j < len(raw_window) else None)
                for j in range(len(window))
            ]
            wid = f"{trajectory_id}_w{start}"
            candidates.append({
                'id':            wid,
                'trajectory_id': trajectory_id,
                'window_start':  start,
                'steps':         review_steps,
                'precomputed':   precomputed,
                'window_features': win_feats,
                'reason':        reason,
                '_full_window':  full_window,   # self-contained for reviewer
            })

    return productive, candidates


# ---------------------------------------------------------------------------
# Session parsers
# ---------------------------------------------------------------------------

def parse_nlile_session(messages):
    """Parse Anthropic API format (nlile / Claude Code transcripts)."""
    steps   = []
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
                inp  = block.get('input', {})
                tid  = block.get('id', '')
                pending[tid] = {
                    'tool':     TOOL_MAP.get(block.get('name', ''), 'other'),
                    'cmd':      inp.get('command', inp.get('file_path', inp.get('pattern', ''))),
                    'file':     inp.get('file_path', inp.get('path')),
                    'thinking': last_thinking,
                }
                last_thinking = ''
            elif btype == 'tool_result':
                tid = block.get('tool_use_id', '')
                if tid in pending:
                    tu  = pending.pop(tid)
                    out = block.get('content', '')
                    if isinstance(out, list):
                        out = '\n'.join(
                            b.get('text', '') for b in out
                            if isinstance(b, dict) and b.get('type') == 'text'
                        )
                    tu['output'] = str(out) if out else ''
                    steps.append(tu)

    for tu in pending.values():   # flush tool calls without results
        tu['output'] = ''
        steps.append(tu)
    return steps


# ---------------------------------------------------------------------------
# Source processors
# ---------------------------------------------------------------------------

def process_nlile(source_name):
    import pyarrow.parquet as pq

    parquet_dir = 'data/separate/nlile_parquet/data'
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))

    productive, candidates = [], []
    n_ok = n_skip = 0

    for fname in files:
        pf = pq.read_table(os.path.join(parquet_dir, fname))
        for i in range(len(pf)):
            msgs_raw = pf.column('messages_json')[i].as_py()
            if not msgs_raw:
                n_skip += 1; continue
            msgs = json.loads(msgs_raw)
            has_tools = any(
                isinstance(m.get('content'), list) and
                any(b.get('type') == 'tool_use' for b in m['content'] if isinstance(b, dict))
                for m in msgs
            )
            if not has_tools:
                n_skip += 1; continue
            row_id = pf.column('id')[i].as_py()
            parsed = parse_nlile_session(msgs)
            if len(parsed) < WINDOW_SIZE:
                n_skip += 1; continue
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                n_skip += 1; continue
            p, c = create_windows(abstract, f"nlile_{row_id}", parsed)
            productive.extend(p); candidates.extend(c)
            n_ok += 1
        del pf; gc.collect()
        print(f"  {fname}: {n_ok} sessions", flush=True)

    print(f"nlile: {n_ok} sessions processed, {n_skip} skipped")
    return productive, candidates


def process_dataclaw(source_name):
    path = 'data/separate/dataclaw/woctordho/conversations.jsonl'
    productive, candidates = [], []
    n_ok = n_skip = 0

    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            if not has_outputs(sess['messages']):
                n_skip += 1; continue
            parsed = parse_dataclaw_session(sess['messages'])
            if len(parsed) < WINDOW_SIZE:
                n_skip += 1; continue
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                n_skip += 1; continue
            p, c = create_windows(abstract, f"dc_{sess['session_id']}", parsed)
            productive.extend(p); candidates.extend(c)
            n_ok += 1

    print(f"dataclaw: {n_ok} sessions processed, {n_skip} skipped")
    return productive, candidates


def process_jsonl(source_name, path):
    """Generic processor for JSONL files of Claude Code sessions."""
    productive, candidates = [], []
    n_ok = n_skip = 0

    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            msgs = sess.get('messages', sess if isinstance(sess, list) else [])
            parsed = parse_nlile_session(msgs)
            if len(parsed) < WINDOW_SIZE:
                n_skip += 1; continue
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                n_skip += 1; continue
            sid = sess.get('session_id', sess.get('id', str(n_ok)))
            p, c = create_windows(abstract, f"{source_name}_{sid}", parsed)
            productive.extend(p); candidates.extend(c)
            n_ok += 1

    print(f"{source_name}: {n_ok} sessions processed, {n_skip} skipped")
    return productive, candidates


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(source_name, productive, candidates):
    os.makedirs(SOURCES_DIR, exist_ok=True)
    os.makedirs(BATCHES_DIR, exist_ok=True)

    labeled_file = os.path.join(SOURCES_DIR, f'{source_name}_labeled.jsonl')
    with open(labeled_file, 'w') as f:
        for w in productive:
            f.write(json.dumps(w) + '\n')

    n_batches = 0
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        out   = os.path.join(BATCHES_DIR, f'{source_name}_batch_{n_batches:04d}.jsonl')
        with open(out, 'w') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
        n_batches += 1

    print(f"\n{source_name}:")
    print(f"  {len(productive)} PRODUCTIVE → {labeled_file}")
    print(f"  {len(candidates)} CANDIDATE  → {n_batches} batches in {BATCHES_DIR}/")
    print(f"\nNext steps:")
    print(f"  1. Run Sonnet review agents on {BATCHES_DIR}/{source_name}_batch_*.jsonl")
    print(f"     Results expected in: data/review/results/sonnet/{source_name}_batch_*.jsonl")
    print(f"  2. python src/review_sonnet.py {source_name}")
    print(f"  3. python src/review_opus.py {source_name}  # if anything was escalated")
    print(f"  4. gzip -k {labeled_file}")
    print(f"  5. python src/merge_sources.py --force")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCES = {
    'nlile':    process_nlile,
    'dataclaw': process_dataclaw,
}


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/label_sessions.py nlile")
        print("  python src/label_sessions.py dataclaw")
        print("  python src/label_sessions.py <source_name> <path/to/sessions.jsonl>")
        sys.exit(1)

    source = sys.argv[1]

    if source in SOURCES:
        productive, candidates = SOURCES[source](source)
    elif len(sys.argv) >= 3:
        productive, candidates = process_jsonl(source, sys.argv[2])
    else:
        print(f"Unknown source '{source}'. Either register it in SOURCES or pass a path:")
        print(f"  python src/label_sessions.py {source} /path/to/sessions.jsonl")
        sys.exit(1)

    write_output(source, productive, candidates)


if __name__ == '__main__':
    main()
