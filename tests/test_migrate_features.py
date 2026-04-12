"""Tests for src/pipeline/migrate_features.py."""

import gzip
import json
import os
import sys
import tempfile

import pytest

from src.pipeline.migrate_features import CURRENT_VERSION, migrate_artifact


def _write_gz(path: str, rows: list[dict]) -> None:
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_gz(path: str) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_row(session_id="sess_001", n_steps=3, schema_version=2) -> dict:
    steps = [{"tool_idx": i % 7, "output_similarity": 0.0} for i in range(n_steps)]
    return {
        "session_id": session_id,
        "source": "test",
        "schema_version": schema_version,
        "n_steps": n_steps,
        "steps": steps,
        "label": 0.0,
    }


class TestVerify:
    def test_valid_artifact_passes(self):
        rows = [_make_row("s1", 3), _make_row("s2", 5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl.gz")
            _write_gz(path, rows)
            # Should not raise or exit
            migrate_artifact(path, verify=True)

    def test_inconsistent_n_steps_detected(self):
        """Verify should detect rows of same session with different n_steps."""
        row1 = _make_row("sess_bad", 3)
        row2 = _make_row("sess_bad", 5)  # same session_id, different n_steps!
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.jsonl.gz")
            _write_gz(path, [row1, row2])
            with pytest.raises(SystemExit) as exc_info:
                migrate_artifact(path, verify=True)
            assert exc_info.value.code == 1

    def test_inconsistent_schema_versions_detected(self):
        """Verify should detect mixed schema versions for same session."""
        row1 = _make_row("sess_mixed", 3, schema_version=1)
        row2 = _make_row("sess_mixed", 3, schema_version=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mixed.jsonl.gz")
            _write_gz(path, [row1, row2])
            with pytest.raises(SystemExit) as exc_info:
                migrate_artifact(path, verify=True)
            assert exc_info.value.code == 1


class TestMigration:
    def test_already_current_version_no_change(self):
        rows = [_make_row("s1"), _make_row("s2")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "current.jsonl.gz")
            _write_gz(path, rows)
            migrate_artifact(path, to_version=CURRENT_VERSION)
            result = _read_gz(path)
            assert len(result) == 2

    def test_no_migration_path_raises(self):
        """Migration from v1 to v99 should fail gracefully."""
        rows = [_make_row("s1", schema_version=1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old.jsonl.gz")
            _write_gz(path, rows)
            with pytest.raises(ValueError, match="No migration path"):
                migrate_artifact(path, to_version=99)
