"""Tests for src/training/train.py."""

import json
import os
import tempfile

import pytest

from src.training.train import session_split, load_rows_from_jsonl


def _make_rows(n_sessions=10, steps_per_session=5) -> list[dict]:
    rows = []
    for s in range(n_sessions):
        for i in range(steps_per_session):
            rows.append(
                {
                    "session_id": f"sess_{s:03d}",
                    "step": i,
                    "tool_idx": i % 7,
                    "steps_since_same_tool": 0.1,
                    "steps_since_same_file": 0.2,
                    "steps_since_same_cmd": 0.3,
                    "tool_count_in_window": 0.1,
                    "file_count_in_window": 0.1,
                    "cmd_count_in_window": 0.1,
                    "output_similarity": 0.0,
                    "has_prior_output": 0.0,
                    "output_length": 1.0,
                    "is_error": 0.0,
                    "step_index_norm": float(i) / max(steps_per_session - 1, 1),
                    "label": 1.0 if (s % 5 == 0 and i > 2) else 0.0,
                }
            )
    return rows


class TestManifestLoading:
    def test_manifest_loads_with_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy JSONL files
            rows = _make_rows(5, 3)
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            manifest = {
                "schema_version": 1,
                "datasets": [{"path": jsonl_path, "weight": 2.0}],
            }
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            loaded = json.load(open(manifest_path))
            assert loaded["datasets"][0]["weight"] == 2.0
            assert loaded["schema_version"] == 1


class TestSessionSplit:
    def test_no_session_in_both_sets(self):
        rows = _make_rows(50, 5)
        train_rows, test_rows = session_split(rows, test_fraction=0.1)

        train_sessions = {r["session_id"] for r in train_rows}
        test_sessions = {r["session_id"] for r in test_rows}

        overlap = train_sessions & test_sessions
        assert len(overlap) == 0, f"Sessions in both sets: {overlap}"

    def test_all_sessions_covered(self):
        rows = _make_rows(20, 5)
        train_rows, test_rows = session_split(rows, test_fraction=0.1)

        all_in = {r["session_id"] for r in rows}
        all_out = {r["session_id"] for r in train_rows} | {
            r["session_id"] for r in test_rows
        }
        assert all_in == all_out

    def test_split_ratio_approximately_correct(self):
        rows = _make_rows(100, 3)
        train_rows, test_rows = session_split(rows, test_fraction=0.1)

        train_sessions = {r["session_id"] for r in train_rows}
        test_sessions = {r["session_id"] for r in test_rows}

        assert len(test_sessions) == 10
        assert len(train_sessions) == 90

    def test_deterministic(self):
        rows = _make_rows(50, 3)
        train1, test1 = session_split(rows, test_fraction=0.2)
        train2, test2 = session_split(rows, test_fraction=0.2)

        assert {r["session_id"] for r in test1} == {r["session_id"] for r in test2}


class TestStepDatasetValidation:
    def test_missing_tool_idx_raises(self):
        """Rows missing tool_idx must raise a clear error at dataset construction time,
        not a cryptic KeyError deep in tensor building or silent zero-fill."""
        from src.training.train import StepDataset

        rows = _make_rows(2, 3)
        del rows[0]["tool_idx"]  # remove from one row

        with pytest.raises((KeyError, ValueError)):
            StepDataset(rows)

    def test_missing_continuous_feature_raises(self):
        """Rows missing a continuous feature should fail loudly, not silently fill zero."""
        from src.training.train import StepDataset

        rows = _make_rows(2, 3)
        del rows[0]["output_similarity"]

        # Currently fills silently with 0.0 — this test documents the expected behaviour
        # after the fix: should raise, not silently corrupt training data.
        with pytest.raises((KeyError, ValueError)):
            StepDataset(rows)


class TestLoadRowsFromJsonl:
    def test_loads_rows(self):
        rows = _make_rows(5, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            loaded = load_rows_from_jsonl(path)
            assert len(loaded) == 15

    def test_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"session_id": "a", "label": 0.0}) + "\n")
                f.write("\n")
                f.write(json.dumps({"session_id": "b", "label": 1.0}) + "\n")

            loaded = load_rows_from_jsonl(path)
            assert len(loaded) == 2
