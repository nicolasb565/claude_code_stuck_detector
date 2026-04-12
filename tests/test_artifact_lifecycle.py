"""Tests for artifact (labeled .gz) lifecycle operations."""

import gzip
import json
import os
import tempfile

from generate import _update_gz_artifact


def _read_gz(path: str) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_row(session_id: str, step: int = 0) -> dict:
    return {
        "session_id": session_id,
        "step": step,
        "tool_idx": 0,
        "label": 0.0,
        "schema_version": 1,
    }


class TestArtifactLifecycle:
    def test_new_session_appended(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl.gz")

            # Start with one session
            existing_rows = [_make_row("sess_A", 0), _make_row("sess_A", 1)]
            with gzip.open(path, "wt") as f:
                for row in existing_rows:
                    f.write(json.dumps(row) + "\n")

            # Append a new session
            new_rows = [_make_row("sess_B", 0), _make_row("sess_B", 1)]
            _update_gz_artifact(path, new_rows)

            result = _read_gz(path)
            session_ids = [r["session_id"] for r in result]
            assert "sess_A" in session_ids
            assert "sess_B" in session_ids
            assert len(result) == 4

    def test_existing_session_skipped(self):
        """A session already in the artifact is not duplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl.gz")

            existing_rows = [_make_row("sess_X", 0)]
            with gzip.open(path, "wt") as f:
                for row in existing_rows:
                    f.write(json.dumps(row) + "\n")

            # Try to add same session again
            _update_gz_artifact(path, [_make_row("sess_X", 0)])

            result = _read_gz(path)
            # Should still have only 1 row
            assert len(result) == 1

    def test_creates_new_artifact_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new.jsonl.gz")
            assert not os.path.exists(path)

            new_rows = [_make_row("sess_new", 0)]
            _update_gz_artifact(path, new_rows)

            assert os.path.exists(path)
            result = _read_gz(path)
            assert len(result) == 1
            assert result[0]["session_id"] == "sess_new"

    def test_atomic_write_uses_rename(self):
        """Verify the artifact is written atomically (no partial state visible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "atomic.jsonl.gz")

            rows = [_make_row("sess_atomic", i) for i in range(5)]
            _update_gz_artifact(path, rows)

            # No temp files should remain
            tmp_files = [f for f in os.listdir(tmpdir) if ".tmp" in f]
            assert not tmp_files

            result = _read_gz(path)
            assert len(result) == 5
