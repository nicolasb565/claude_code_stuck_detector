"""Tests for src/pipeline/extract_features.py."""

import json
import os
import tempfile

import pytest

from src.pipeline.extract_features import (
    SCHEMA_VERSION,
    STEP_FEATURES,
    compute_step_features,
    extract_session,
)


def _make_steps(n=5) -> list[dict]:
    steps = []
    tools = ["bash", "view", "edit", "search", "other"]
    for i in range(n):
        steps.append(
            {
                "tool": tools[i % len(tools)],
                "tool_name": ["Bash", "Read", "Edit", "Grep", "Agent"][i % 5],
                "cmd": f"cmd_{i}",
                "file": f"src/file_{i % 3}.c" if i % 2 == 0 else None,
                "output": (
                    f"output line {i}\nerror: something failed"
                    if i % 3 == 0
                    else f"ok {i}"
                ),
                "thinking": "let me reconsider" if i % 4 == 0 else "",
            }
        )
    return steps


class TestComputeStepFeatures:
    def test_returns_correct_count(self):
        steps = _make_steps(10)
        feats = compute_step_features(steps)
        assert len(feats) == 10

    def test_all_step_feature_keys_present(self):
        steps = _make_steps(5)
        feats = compute_step_features(steps)
        for feat in feats:
            for key in STEP_FEATURES:
                assert key in feat, f"Missing feature: {key}"

    def test_tool_idx_is_int(self):
        steps = _make_steps(5)
        feats = compute_step_features(steps)
        for feat in feats:
            assert isinstance(feat["tool_idx"], int)

    def test_continuous_features_are_float(self):
        steps = _make_steps(5)
        feats = compute_step_features(steps)
        for feat in feats:
            for key in STEP_FEATURES:
                if key != "tool_idx":
                    assert isinstance(feat[key], float), f"{key} should be float"

    def test_no_extra_fields(self):
        steps = _make_steps(3)
        feats = compute_step_features(steps)
        for feat in feats:
            extra = set(feat.keys()) - set(STEP_FEATURES)
            assert not extra, f"Extra fields: {extra}"

    def test_empty_steps_returns_empty(self):
        feats = compute_step_features([])
        assert feats == []

    def test_output_length_empty_is_zero(self):
        """Empty output must produce output_length=0.0 (log1p(0) = 0)."""
        steps = [{"tool": "bash", "tool_name": "Bash", "cmd": "ls", "file": None,
                  "output": "", "thinking": ""}]
        feats = compute_step_features(steps)
        assert feats[0]["output_length"] == 0.0

    def test_output_length_is_log1p_of_lines(self):
        """output_length must be log1p(n_newlines), not log1p(n_newlines + 1).

        Bug: formula was log1p(lines + 1) which gives log(lines + 2) — off by one
        inside the log. A 1-line output (0 newlines) should give log1p(0)=0, not
        log1p(1)=0.693.
        """
        import math
        # 1-line output: 0 newlines → log1p(0) = 0.0
        steps_1line = [{"tool": "bash", "tool_name": "Bash", "cmd": "ls", "file": None,
                        "output": "one line", "thinking": ""}]
        feats = compute_step_features(steps_1line)
        assert feats[0]["output_length"] == pytest.approx(math.log1p(0), abs=1e-6)

        # 3-line output: 2 newlines → log1p(2) ≈ 1.099
        steps_3line = [{"tool": "bash", "tool_name": "Bash", "cmd": "ls", "file": None,
                        "output": "line1\nline2\nline3", "thinking": ""}]
        feats = compute_step_features(steps_3line)
        assert feats[0]["output_length"] == pytest.approx(math.log1p(2), abs=1e-6)

    def test_is_error_detected(self):
        steps = [
            {
                "tool": "bash",
                "tool_name": "Bash",
                "cmd": "make",
                "file": None,
                "output": "error: compilation failed",
                "thinking": "",
            }
        ]
        feats = compute_step_features(steps)
        assert feats[0]["is_error"] == 1.0


class TestExtractSession:
    def test_creates_feature_file(self):
        steps = _make_steps(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = extract_session(steps, "sess_001", "nlile", tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["session_id"] == "sess_001"
            assert data["source"] == "nlile"
            assert data["schema_version"] == SCHEMA_VERSION
            assert data["n_steps"] == 10
            assert len(data["steps"]) == 10

    def test_idempotent_skips_existing(self):
        steps = _make_steps(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = extract_session(steps, "sess_idem", "nlile", tmpdir)
            mtime1 = os.path.getmtime(path1)
            # Small sleep to ensure mtime would differ if file was rewritten
            import time

            time.sleep(0.05)
            path2 = extract_session(steps, "sess_idem", "nlile", tmpdir)
            mtime2 = os.path.getmtime(path2)
            assert path1 == path2
            assert mtime1 == mtime2  # file was NOT rewritten

    def test_force_reextracts(self):
        """force=True causes re-extraction even if file exists."""
        steps = _make_steps(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = extract_session(steps, "sess_force", "nlile", tmpdir)
            # Corrupt the extracted_at timestamp to detect re-extraction
            with open(path1) as f:
                data = json.load(f)
            data["extracted_at"] = "2000-01-01T00:00:00Z"
            with open(path1, "w") as f:
                json.dump(data, f)
            # Without force, should skip (keep corrupted timestamp)
            extract_session(steps, "sess_force", "nlile", tmpdir, force=False)
            with open(path1) as f:
                data_no_force = json.load(f)
            assert data_no_force["extracted_at"] == "2000-01-01T00:00:00Z"
            # With force, should re-extract (fresh timestamp)
            extract_session(steps, "sess_force", "nlile", tmpdir, force=True)
            with open(path1) as f:
                data_force = json.load(f)
            assert data_force["extracted_at"] != "2000-01-01T00:00:00Z"

    def test_two_runs_identical_output(self):
        steps = _make_steps(8)
        with tempfile.TemporaryDirectory() as tmpdir1:
            path1 = extract_session(steps, "sess_same", "nlile", tmpdir1)
            with open(path1) as f:
                data1 = json.load(f)

        with tempfile.TemporaryDirectory() as tmpdir2:
            path2 = extract_session(steps, "sess_same", "nlile", tmpdir2)
            with open(path2) as f:
                data2 = json.load(f)

        assert data1["steps"] == data2["steps"]
        assert data1["n_steps"] == data2["n_steps"]


class TestMultiSlotOutputHistory:
    """Schema 4: output_history keeps last K=5 predecessors and reports
    max Jaccard. This catches long-range repetition patterns that the
    single-slot schema 3 silently dropped."""

    def _run(self, *bash_outputs):
        steps = [
            {"tool": "bash", "cmd": "ls", "file": None, "output": o, "thinking": ""}
            for o in bash_outputs
        ]
        return compute_step_features(steps)

    def test_matches_older_predecessor_not_just_most_recent(self):
        # Pattern: A(X), A(Y), A(X again). Schema 3 compared step 3 to step 2
        # only → similarity 0. Schema 4 finds the earlier X match → similarity 1.
        feats = self._run("alpha\nbeta", "gamma\ndelta", "alpha\nbeta")
        assert feats[2]["output_similarity"] == 1.0
        assert feats[2]["has_prior_output"] == 1.0

    def test_max_over_all_priors(self):
        feats = self._run("aaa", "bbb", "ccc", "aaa")
        assert feats[3]["output_similarity"] == 1.0

    def test_fifo_eviction_after_k_plus_one(self):
        # After pushing 6 distinct outputs with K=5 slots, the oldest is
        # evicted. A 7th call matching the oldest should score 0.
        feats = self._run("a", "b", "c", "d", "e", "f", "a")
        assert feats[6]["output_similarity"] == 0.0
        # But matching the most recent (f) still works.
        feats2 = self._run("a", "b", "c", "d", "e", "f", "f")
        assert feats2[6]["output_similarity"] == 1.0

    def test_has_prior_unaffected_by_multi_slot_for_first_call(self):
        feats = self._run("anything")
        assert feats[0]["has_prior_output"] == 0.0
        assert feats[0]["output_similarity"] == 0.0

    def test_partial_overlap_with_older_slot(self):
        # Step 3 shares half its lines with step 1 (evicted? no — K=5, not
        # evicted) and nothing with step 2. Max Jaccard should catch step 1.
        feats = self._run("x\ny", "z\nw", "x\nq")
        # step 3 vs step 1: {x,y} ∩ {x,q} = {x}; union = {x,y,q} → 1/3
        # step 3 vs step 2: {z,w} ∩ {x,q} = {} → 0
        # max = 1/3
        assert abs(feats[2]["output_similarity"] - 1.0 / 3.0) < 1e-9
