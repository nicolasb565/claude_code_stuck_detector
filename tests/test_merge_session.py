"""Tests for src/pipeline/merge_session.py."""

import json
import os
import tempfile

import pytest

from src.pipeline.extract_features import SCHEMA_VERSION, extract_session
from src.pipeline.label_session import write_label_file
from src.pipeline.merge_session import merge_session


def _make_steps(n=5) -> list[dict]:
    return [
        {
            "tool": "bash",
            "tool_name": "Bash",
            "cmd": f"cmd_{i}",
            "file": None,
            "output": f"output {i}",
            "thinking": "",
        }
        for i in range(n)
    ]


class TestMergeSession:
    def test_produces_correct_jsonl_rows(self):
        n = 5
        steps = _make_steps(n)
        labels = ["PRODUCTIVE", "STUCK", "UNSURE", "PRODUCTIVE", "STUCK"]

        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path = extract_session(steps, "sess_merge_001", "test", tmpdir)
            label_path = os.path.join(tmpdir, "sess_merge_001_labels.json")
            write_label_file(label_path, "sess_merge_001", "test", labels, n)

            out_path = os.path.join(tmpdir, "out.jsonl")
            rows_written = merge_session(label_path, feat_path, out_path)

            assert rows_written == n
            with open(out_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]

            assert len(rows) == n
            for i, row in enumerate(rows):
                assert row["session_id"] == "sess_merge_001"
                assert row["step"] == i
                assert "tool_idx" in row
                assert "label" in row

    def test_label_encoding(self):
        n = 3
        steps = _make_steps(n)
        labels = ["PRODUCTIVE", "STUCK", "UNSURE"]

        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path = extract_session(steps, "sess_enc", "test", tmpdir)
            label_path = os.path.join(tmpdir, "sess_enc_labels.json")
            write_label_file(label_path, "sess_enc", "test", labels, n)

            out_path = os.path.join(tmpdir, "out.jsonl")
            merge_session(label_path, feat_path, out_path)

            with open(out_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]

            assert rows[0]["label"] == 0.0  # PRODUCTIVE
            assert rows[1]["label"] == 1.0  # STUCK
            assert rows[2]["label"] == 0.5  # UNSURE

    def test_appends_to_existing_file(self):
        n = 3
        steps = _make_steps(n)
        labels = ["PRODUCTIVE"] * n

        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path = extract_session(steps, "sess_append", "test", tmpdir)
            label_path = os.path.join(tmpdir, "sess_append_labels.json")
            write_label_file(label_path, "sess_append", "test", labels, n)

            out_path = os.path.join(tmpdir, "out.jsonl")
            # Write twice — should append
            merge_session(label_path, feat_path, out_path)
            merge_session(label_path, feat_path, out_path)

            with open(out_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            assert len(rows) == 2 * n

    def test_mismatched_n_steps_raises(self):
        n = 5
        steps = _make_steps(n)

        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path = extract_session(steps, "sess_mismatch", "test", tmpdir)
            # Write label file with wrong count
            label_path = os.path.join(tmpdir, "sess_mismatch_labels.json")
            write_label_file(label_path, "sess_mismatch", "test", ["PRODUCTIVE"] * 3, 3)

            out_path = os.path.join(tmpdir, "out.jsonl")
            with pytest.raises(ValueError, match="n_steps mismatch"):
                merge_session(label_path, feat_path, out_path)

    def test_wrong_schema_version_raises(self):
        n = 3
        steps = _make_steps(n)
        labels = ["PRODUCTIVE"] * n

        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path = extract_session(steps, "sess_badver", "test", tmpdir)

            # Corrupt schema version
            with open(feat_path) as f:
                data = json.load(f)
            data["schema_version"] = 99
            with open(feat_path, "w") as f:
                json.dump(data, f)

            label_path = os.path.join(tmpdir, "sess_badver_labels.json")
            write_label_file(label_path, "sess_badver", "test", labels, n)

            out_path = os.path.join(tmpdir, "out.jsonl")
            with pytest.raises(ValueError, match="schema_version"):
                merge_session(label_path, feat_path, out_path)
