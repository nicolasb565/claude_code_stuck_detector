"""Submit and poll Anthropic Message Batches API for stuck-detection labeling.

Usage:
  python src/pipeline/batch_label.py datasets/nlile/ [datasets/dataclaw_claude/]
         [--max-sessions N] [--dry-run-estimate]
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

from src.pipeline.label_session import (
    LABELER_MODEL,
    SYSTEM_PROMPT,
    format_transcript,
    parse_csv_labels,
    validate_label_file,
    write_label_file,
)

_RETRY_DELAYS = [1, 2, 4, 8]
_RETRY_STATUS = {429, 500, 529}
_ABORT_STATUS = {400, 401, 402}

# Cost per million tokens (batch pricing)
_INPUT_COST_PER_MTOK = 1.50
_OUTPUT_COST_PER_MTOK = 7.50


def _get_client():
    import anthropic  # pylint: disable=import-outside-toplevel

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def _retry_call(fn, *args, **kwargs):
    """Call fn with exponential backoff retries."""
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            status = getattr(exc, "status_code", None)
            if status in _ABORT_STATUS:
                print(
                    f"ERROR: HTTP {status} — aborting (no retry): {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if status in _RETRY_STATUS or status is None:
                print(
                    f"WARNING: attempt {attempt + 1} failed (HTTP {status}): {exc}",
                    file=sys.stderr,
                )
                last_exc = exc
                continue
            raise
    print(
        f"ERROR: All retries exhausted. Last error: {last_exc}",
        file=sys.stderr,
    )
    sys.exit(1)


def submit_batch(
    transcripts: list[tuple[str, str, int]],
    source: str,
    labels_dir: str,
) -> str:
    """Submit one batch. Returns batch_id. Saves pending_batch.json.

    Args:
        transcripts: list of (session_id, transcript_text, n_steps)
        source: source name for labeling
        labels_dir: directory to save pending_batch.json

    Returns:
        batch_id string
    """
    client = _get_client()

    requests = []
    for session_id, transcript_text, _n_steps in transcripts:
        requests.append(
            {
                "custom_id": session_id,
                "params": {
                    "model": LABELER_MODEL,
                    "max_tokens": 1024,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": transcript_text}],
                },
            }
        )

    batch = _retry_call(
        client.messages.batches.create,
        requests=requests,
    )
    batch_id = batch.id

    os.makedirs(labels_dir, exist_ok=True)
    pending_path = os.path.join(labels_dir, "pending_batch.json")
    # Store n_steps per session so resume can reconstruct transcripts_by_id
    # even if the caller's session list has changed since submission.
    session_n_steps = {sid: n for sid, _, n in transcripts}
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_id": batch_id,
                "source": source,
                "session_n_steps": session_n_steps,
            },
            f,
        )

    print(f"Submitted batch {batch_id} ({len(requests)} requests)")
    return batch_id


def poll_and_retrieve(  # pylint: disable=too-many-branches
    batch_id: str,
    source: str,
    labels_dir: str,
    transcripts_by_id: dict[str, tuple[str, int]],
) -> dict[str, list[str] | None]:
    """Poll until done. Returns {session_id: labels_or_None}.

    Args:
        batch_id: Anthropic batch ID
        source: source name
        labels_dir: directory for label files
        transcripts_by_id: {session_id: (transcript_text, n_steps)}

    Returns:
        {session_id: list of label strings, or None if failed}
    """
    client = _get_client()
    poll_interval = 30

    while True:
        batch = _retry_call(client.messages.batches.retrieve, batch_id)
        status = batch.processing_status

        if status == "expired":
            print(f"WARNING: batch {batch_id} expired", file=sys.stderr)
            pending_path = os.path.join(labels_dir, "pending_batch.json")
            if os.path.exists(pending_path):
                os.unlink(pending_path)
            break

        if status == "ended":
            break

        print(f"Batch {batch_id} status: {status} — waiting {poll_interval}s...")
        time.sleep(poll_interval)

    # Collect results
    results: dict[str, list[str] | None] = {sid: None for sid in transcripts_by_id}
    non_recoverable_found = False

    for result in _retry_call(client.messages.batches.results, batch_id):
        session_id = result.custom_id
        if session_id not in transcripts_by_id:
            continue

        _transcript_text, n_steps = transcripts_by_id[session_id]

        if result.result.type == "succeeded":
            content = result.result.message.content
            csv_text = ""
            for block in content:
                if hasattr(block, "text"):
                    csv_text = block.text.strip()
                    break
            try:
                labels = parse_csv_labels(csv_text, n_steps)
                label_path = os.path.join(labels_dir, f"{session_id}_labels.json")
                write_label_file(label_path, session_id, source, labels, n_steps)
                results[session_id] = labels
            except ValueError as exc:
                print(
                    f"WARNING: could not parse labels for {session_id}: {exc}",
                    file=sys.stderr,
                )
        elif result.result.type == "errored":
            err = result.result.error
            err_type = getattr(err, "type", "unknown")
            if err_type in ("overloaded_error",):
                print(
                    f"WARNING: recoverable error for {session_id}: {err_type}",
                    file=sys.stderr,
                )
            elif err_type in (
                "invalid_request_error",
                "authentication_error",
                "billing_error",
            ):
                print(
                    f"ERROR: non-recoverable error for {session_id}: {err_type}. "
                    "Check your API key and billing.",
                    file=sys.stderr,
                )
                non_recoverable_found = True
            else:
                print(
                    f"WARNING: error for {session_id}: {err_type}",
                    file=sys.stderr,
                )

    # Clean up pending file
    pending_path = os.path.join(labels_dir, "pending_batch.json")
    if os.path.exists(pending_path):
        os.unlink(pending_path)

    if non_recoverable_found:
        sys.exit(1)

    return results


def run_batch_label(  # pylint: disable=too-many-positional-arguments
    source_dir: str,
    raw_sessions: list[dict],
    labels_dir: str,
    max_sessions: int | None = None,
    dry_run_estimate: bool = False,
    force: bool = False,
) -> dict[str, list[str] | None]:
    """Main entry point for batch labeling.

    Args:
        source_dir: path to source dataset directory (for fetch/filter config)
        raw_sessions: list of {session_id: str, steps: list[dict]}
        labels_dir: directory for label files
        max_sessions: cap number of sessions (None = no cap)
        dry_run_estimate: print cost estimate and exit without submitting
        force: re-label already-labeled sessions

    Returns:
        {session_id: labels_or_None}
    """
    os.makedirs(labels_dir, exist_ok=True)

    if not force:
        to_label = []
        for sess in raw_sessions:
            sid = sess["session_id"]
            steps = sess["steps"]
            # Count only real steps — CompactBlocks (claudeset) are not labeled.
            n_real = sum(1 for s in steps if s.get("type") != "compact")
            label_path = os.path.join(labels_dir, f"{sid}_labels.json")
            if validate_label_file(label_path, n_real):
                continue
            to_label.append(sess)
    else:
        to_label = list(raw_sessions)

    if max_sessions is not None:
        to_label = to_label[:max_sessions]

    if not to_label:
        print("All sessions already labeled.")
        return {}

    transcripts: list[tuple[str, str, int]] = []
    for sess in to_label:
        transcript_text, n_steps = format_transcript(sess["steps"])
        transcripts.append((sess["session_id"], transcript_text, n_steps))

    source_name = os.path.basename(source_dir.rstrip("/"))

    if dry_run_estimate:
        system_tokens = len(SYSTEM_PROMPT) // 4
        avg_transcript_tokens = (
            sum(len(t) for _, t, _ in transcripts) // max(len(transcripts), 1) // 4
        )
        total_input_tokens = (system_tokens + avg_transcript_tokens) * len(transcripts)
        output_tokens_per = 50
        total_output_tokens = output_tokens_per * len(transcripts)
        total_tokens = total_input_tokens + total_output_tokens

        cost_in = (total_input_tokens / 1_000_000) * _INPUT_COST_PER_MTOK
        cost_out = (total_output_tokens / 1_000_000) * _OUTPUT_COST_PER_MTOK
        total_cost = cost_in + cost_out

        print(
            f"Cost estimate for {source_name}: {len(transcripts)} sessions, "
            f"~{avg_transcript_tokens} tokens each, ~{total_tokens} total"
        )
        print(
            f"  Input:  ~{total_input_tokens} tokens "
            f"x ${_INPUT_COST_PER_MTOK:.2f}/MTok (batch) = ${cost_in:.4f}"
        )
        print(
            f"  Output: ~{total_output_tokens} tokens "
            f"x ${_OUTPUT_COST_PER_MTOK:.2f}/MTok (batch) = ${cost_out:.4f}"
        )
        print(f"  Total:  ~${total_cost:.4f}")
        return {}

    pending_path = os.path.join(labels_dir, "pending_batch.json")
    if os.path.exists(pending_path):
        with open(pending_path, encoding="utf-8") as f:
            pending = json.load(f)
        batch_id = pending["batch_id"]
        print(f"Resuming batch {batch_id}...")
        # Reconstruct transcripts_by_id from the saved n_steps so it matches
        # what was actually submitted — not the current (possibly different) session list.
        saved_n_steps: dict[str, int] = pending.get("session_n_steps", {})
        transcripts_by_id = {
            sid: (text, saved_n_steps.get(sid, n_steps))
            for sid, text, n_steps in transcripts
            if sid in saved_n_steps
        }
        # Include any sessions in the saved batch that are no longer in the current
        # transcripts list (they were submitted but filtered out on resume).
        current_ids = {sid for sid, _, _ in transcripts}
        for sid, n in saved_n_steps.items():
            if sid not in current_ids:
                transcripts_by_id[sid] = ("", n)
    else:
        batch_id = submit_batch(transcripts, source_name, labels_dir)
        transcripts_by_id = {sid: (text, n_steps) for sid, text, n_steps in transcripts}

    return poll_and_retrieve(batch_id, source_name, labels_dir, transcripts_by_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch label sessions using Anthropic API"
    )
    parser.add_argument("source_dirs", nargs="+", help="Dataset directories to process")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--dry-run-estimate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for source_dir in args.source_dirs:
        fetch_path = os.path.join(source_dir, "fetch.json")
        if not os.path.exists(fetch_path):
            print(f"WARNING: no fetch.json in {source_dir}, skipping", file=sys.stderr)
            continue

        with open(fetch_path, encoding="utf-8") as f:
            fetch_cfg = json.load(f)

        source_name = os.path.basename(source_dir.rstrip("/"))
        print(f"Processing {source_name}...")
        print(
            f"  Use generate.py to load raw sessions and call run_batch_label().\n"
            f"  fetch.json type: {fetch_cfg.get('type', 'unknown')}"
        )


if __name__ == "__main__":
    main()
