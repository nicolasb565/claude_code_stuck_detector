"""Tests for src/pipeline/batch_label.py — all API calls mocked."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from src.pipeline.label_session import write_label_file


def _make_transcripts(n=3):
    return [(f"sess_{i}", f"transcript {i}", 5) for i in range(n)]


def _make_mock_client():
    client = MagicMock()
    return client


class TestParseCsvToLabels:
    """CSV mapping → PRODUCTIVE/STUCK/UNSURE."""

    def test_csv_maps_correctly(self):
        from src.pipeline.label_session import parse_csv_labels

        assert parse_csv_labels("P,S,U", 3) == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_all_productive(self):
        from src.pipeline.label_session import parse_csv_labels

        assert parse_csv_labels("P,P,P,P,P", 5) == ["PRODUCTIVE"] * 5


class TestSubmitBatch:
    def test_submit_saves_pending_batch(self):
        mock_batch = MagicMock()
        mock_batch.id = "batch_abc123"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()
                mock_client.messages.batches.create.return_value = mock_batch
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import submit_batch

                transcripts = _make_transcripts(2)
                batch_id = submit_batch(transcripts, "nlile", tmpdir)

                assert batch_id == "batch_abc123"
                pending_path = os.path.join(tmpdir, "pending_batch.json")
                assert os.path.exists(pending_path)
                with open(pending_path) as f:
                    data = json.load(f)
                assert data["batch_id"] == "batch_abc123"

    def test_submit_saves_session_n_steps_in_pending(self):
        """pending_batch.json must record session_n_steps so resume works correctly
        even if the caller's session list changes between submission and resume."""
        mock_batch = MagicMock()
        mock_batch.id = "batch_nsteps"

        transcripts = [("sess_A", "transcript A", 10), ("sess_B", "transcript B", 25)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()
                mock_client.messages.batches.create.return_value = mock_batch
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import submit_batch

                submit_batch(transcripts, "nlile", tmpdir)

                pending_path = os.path.join(tmpdir, "pending_batch.json")
                with open(pending_path) as f:
                    data = json.load(f)

                assert data["session_n_steps"] == {"sess_A": 10, "sess_B": 25}


class TestResumeFromPending:
    def test_resume_uses_saved_n_steps_not_current_list(self):
        """On resume, n_steps must come from pending_batch.json (what was submitted),
        not from the current session list (which may have changed).

        Bug: before the fix, if the session list changed between submission and resume,
        poll_and_retrieve received wrong n_steps and would fail label validation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # pending_batch.json records sess_0 with n_steps=20
            pending_path = os.path.join(tmpdir, "pending_batch.json")
            with open(pending_path, "w") as f:
                json.dump(
                    {
                        "batch_id": "resume_batch",
                        "source": "nlile",
                        "session_n_steps": {"sess_0": 20},
                    },
                    f,
                )

            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()
                mock_status = MagicMock()
                mock_status.processing_status = "ended"
                mock_client.messages.batches.retrieve.return_value = mock_status

                # Batch returns 20 labels for sess_0
                mock_result = MagicMock()
                mock_result.custom_id = "sess_0"
                mock_result.result.type = "succeeded"
                mock_block = MagicMock()
                mock_block.text = ",".join(["P"] * 20)
                mock_result.result.message.content = [mock_block]
                mock_client.messages.batches.results.return_value = [mock_result]
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import run_batch_label

                # Current session list has sess_0 with n_steps=5 (DIFFERENT from submitted)
                raw_sessions = [{"session_id": "sess_0", "steps": [{}] * 5}]
                result = run_batch_label(
                    source_dir="datasets/nlile/",
                    raw_sessions=raw_sessions,
                    labels_dir=tmpdir,
                )

                # Labels should be written using saved n_steps=20, not current n_steps=5.
                # With the fix: label file written with 20 labels → result is not None.
                # Without the fix: n_steps=5 would cause parse_csv_labels to fail
                # (20 labels for 5 steps) → session left unlabeled.
                label_path = os.path.join(tmpdir, "sess_0_labels.json")
                assert os.path.exists(
                    label_path
                ), "label file should be written on resume"
                with open(label_path) as f:
                    label_data = json.load(f)
                assert label_data["n_steps"] == 20

    def test_pending_batch_skips_submission(self):
        """If pending_batch.json exists, poll without resubmitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a pending batch file
            pending_path = os.path.join(tmpdir, "pending_batch.json")
            with open(pending_path, "w") as f:
                json.dump({"batch_id": "existing_batch_id", "source": "nlile"}, f)

            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()

                # Batch is already ended
                mock_batch_status = MagicMock()
                mock_batch_status.processing_status = "ended"
                mock_client.messages.batches.retrieve.return_value = mock_batch_status
                mock_client.messages.batches.results.return_value = []
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import run_batch_label

                # Sessions that are already labeled — so nothing to submit
                with tempfile.TemporaryDirectory() as tmpdir2:
                    # Pre-write labels for all sessions
                    for i in range(2):
                        sid = f"sess_{i}"
                        write_label_file(
                            os.path.join(tmpdir, f"{sid}_labels.json"),
                            sid,
                            "nlile",
                            ["PRODUCTIVE"] * 5,
                            5,
                        )

                    raw_sessions = [
                        {"session_id": f"sess_{i}", "steps": [{}] * 5} for i in range(2)
                    ]
                    result = run_batch_label(
                        source_dir="datasets/nlile/",
                        raw_sessions=raw_sessions,
                        labels_dir=tmpdir,
                    )

                # Should not have called create (all already labeled)
                mock_client.messages.batches.create.assert_not_called()


class TestBatchExpiry:
    def test_expired_batch_deletes_pending_and_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pending_path = os.path.join(tmpdir, "pending_batch.json")
            with open(pending_path, "w") as f:
                json.dump({"batch_id": "expiring_batch", "source": "nlile"}, f)

            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()

                mock_batch_status = MagicMock()
                mock_batch_status.processing_status = "expired"
                mock_client.messages.batches.retrieve.return_value = mock_batch_status
                mock_client.messages.batches.results.return_value = []
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import poll_and_retrieve

                transcripts_by_id = {"sess_0": ("transcript", 5)}
                results = poll_and_retrieve(
                    "expiring_batch", "nlile", tmpdir, transcripts_by_id
                )

                # pending_batch.json should be deleted after expiry
                assert not os.path.exists(pending_path)
                # Session result should be None (not labeled)
                assert results.get("sess_0") is None


class TestRetryBackoff:
    def test_529_retries_with_backoff(self):
        """HTTP 529 triggers up to 4 retries with exponential backoff."""
        with patch("src.pipeline.batch_label._get_client") as mock_get_client:
            mock_client = _make_mock_client()

            # Simulate overloaded error then success
            error_429 = Exception("overloaded")
            error_429.status_code = 529

            success_batch = MagicMock()
            success_batch.id = "batch_ok"
            mock_client.messages.batches.create.side_effect = [
                error_429,
                success_batch,
            ]
            mock_get_client.return_value = mock_client

            with patch("src.pipeline.batch_label.time.sleep"):
                from src.pipeline.batch_label import submit_batch

                with tempfile.TemporaryDirectory() as tmpdir:
                    batch_id = submit_batch(_make_transcripts(1), "nlile", tmpdir)
                    assert batch_id == "batch_ok"
                    assert mock_client.messages.batches.create.call_count == 2

    def test_400_no_retry_exits(self):
        """HTTP 400 aborts immediately without retry."""
        with patch("src.pipeline.batch_label._get_client") as mock_get_client:
            mock_client = _make_mock_client()

            error_400 = Exception("bad request")
            error_400.status_code = 400
            mock_client.messages.batches.create.side_effect = error_400
            mock_get_client.return_value = mock_client

            with patch("src.pipeline.batch_label.time.sleep"):
                from src.pipeline.batch_label import submit_batch

                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(SystemExit) as exc_info:
                        submit_batch(_make_transcripts(1), "nlile", tmpdir)
                    assert exc_info.value.code == 1
                    # Should not retry at all
                    assert mock_client.messages.batches.create.call_count == 1

    def test_401_no_retry_exits(self):
        """HTTP 401 aborts immediately."""
        with patch("src.pipeline.batch_label._get_client") as mock_get_client:
            mock_client = _make_mock_client()

            error_401 = Exception("unauthorized")
            error_401.status_code = 401
            mock_client.messages.batches.create.side_effect = error_401
            mock_get_client.return_value = mock_client

            with patch("src.pipeline.batch_label.time.sleep"):
                from src.pipeline.batch_label import submit_batch

                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(SystemExit) as exc_info:
                        submit_batch(_make_transcripts(1), "nlile", tmpdir)
                    assert exc_info.value.code == 1


class TestDryRunEstimate:
    def test_dry_run_does_not_exit_process(self):
        """--dry-run-estimate must return {} without calling sys.exit.

        Bug: run_batch_label called sys.exit(0) after printing the estimate,
        so multi-source dry-runs exited after the first source.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = [
                {"session_id": f"sess_{i}", "steps": [{}] * 35} for i in range(3)
            ]
            from src.pipeline.batch_label import run_batch_label

            # Should return {} without raising SystemExit
            result = run_batch_label(
                source_dir="datasets/nlile/",
                raw_sessions=sessions,
                labels_dir=tmpdir,
                dry_run_estimate=True,
            )
            assert result == {}
