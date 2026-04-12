"""Tests for session filtering logic in generate.py."""

from generate import _apply_filters


def _make_session(session_id: str, n_steps: int, path: str = "") -> dict:
    return {
        "session_id": session_id,
        "steps": [{}] * n_steps,
        "_source_path": path,
    }


class TestFilters:
    def test_min_steps_filter(self):
        sessions = [
            _make_session("s1", 10),
            _make_session("s2", 30),
            _make_session("s3", 50),
        ]
        filtered = _apply_filters(sessions, {"min_steps": 30}, "test")
        ids = [s["session_id"] for s in filtered]
        assert "s1" not in ids
        assert "s2" in ids
        assert "s3" in ids

    def test_max_steps_filter(self):
        sessions = [
            _make_session("s1", 50),
            _make_session("s2", 200),
            _make_session("s3", 300),
        ]
        filtered = _apply_filters(sessions, {"max_steps": 200}, "test")
        ids = [s["session_id"] for s in filtered]
        assert "s1" in ids
        assert "s2" in ids
        assert "s3" not in ids

    def test_min_and_max_steps_combined(self):
        sessions = [
            _make_session("too_short", 10),
            _make_session("just_right", 50),
            _make_session("too_long", 300),
        ]
        filtered = _apply_filters(sessions, {"min_steps": 30, "max_steps": 200}, "test")
        ids = [s["session_id"] for s in filtered]
        assert ids == ["just_right"]

    def test_max_sessions_cap(self):
        sessions = [_make_session(f"s{i}", 50) for i in range(100)]
        filtered = _apply_filters(sessions, {"max_sessions": 10}, "test")
        assert len(filtered) == 10

    def test_max_sessions_deterministic(self):
        """Same seed → same selection."""
        sessions = [_make_session(f"s{i:03d}", 50) for i in range(100)]
        filtered1 = _apply_filters(sessions, {"max_sessions": 20}, "test")
        filtered2 = _apply_filters(sessions, {"max_sessions": 20}, "test")
        ids1 = [s["session_id"] for s in filtered1]
        ids2 = [s["session_id"] for s in filtered2]
        assert ids1 == ids2

    def test_no_filters_returns_all(self):
        sessions = [_make_session(f"s{i}", 50) for i in range(10)]
        filtered = _apply_filters(sessions, {}, "test")
        assert len(filtered) == 10

    def test_folder_limits(self):
        sessions = [
            _make_session("s1", 50, "data/rust/file1.parquet"),
            _make_session("s2", 50, "data/rust/file2.parquet"),
            _make_session("s3", 50, "data/rust/file3.parquet"),
            _make_session("s4", 50, "data/python/file1.parquet"),
        ]
        filter_cfg = {"folder_limits": [{"pattern": "data/rust/*", "max": 2}]}
        filtered = _apply_filters(sessions, filter_cfg, "test")
        rust_sessions = [s for s in filtered if "rust" in s["_source_path"]]
        python_sessions = [s for s in filtered if "python" in s["_source_path"]]
        assert len(rust_sessions) == 2
        assert len(python_sessions) == 1
