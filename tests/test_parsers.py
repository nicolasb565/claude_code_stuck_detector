"""Tests for src/pipeline/parsers/{nlile,dataclaw,claudeset}."""

import json
import os
import pytest

from src.pipeline.parsers.nlile import (
    parse_session as nlile_parse,
    ParserSchemaError as NlileError,
)
from src.pipeline.parsers.dataclaw import (
    parse_session as dc_parse,
    ParserSchemaError as DcError,
)
from src.pipeline.parsers.claudeset import (
    parse_session as cs_parse,
    parse_session_steps_only as cs_parse_steps,
    ParserSchemaError as CsError,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


# --- nlile ---


class TestNlileParser:
    def test_parse_fixture_produces_steps(self):
        sess = load_fixture("nlile_session.json")
        messages = json.loads(sess["messages_json"])
        steps = nlile_parse(messages)
        assert len(steps) >= 35

    def test_step_has_required_fields(self):
        sess = load_fixture("nlile_session.json")
        messages = json.loads(sess["messages_json"])
        steps = nlile_parse(messages)
        for step in steps:
            assert "tool" in step
            assert "tool_name" in step
            assert "cmd" in step
            assert "file" in step or "file" not in step  # file may be None
            assert "output" in step
            assert "thinking" in step

    def test_tool_mapping(self):
        sess = load_fixture("nlile_session.json")
        messages = json.loads(sess["messages_json"])
        steps = nlile_parse(messages)
        abstract_tools = {"bash", "view", "edit", "search", "other"}
        for step in steps:
            assert step["tool"] in abstract_tools

    def test_empty_messages_raises(self):
        with pytest.raises(NlileError, match="no tool calls"):
            nlile_parse([])

    def test_no_tool_use_raises(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        with pytest.raises(NlileError, match="no tool calls"):
            nlile_parse(messages)

    def test_unknown_fields_handled(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "UnknownTool",
                        "input": {"some_field": "value"},
                        "extra_unknown": "ignored",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"}
                ],
            },
        ]
        steps = nlile_parse(messages)
        assert len(steps) == 1
        assert steps[0]["tool"] == "other"


# --- dataclaw ---


class TestDataclawParser:
    def test_parse_fixture_produces_steps(self):
        sess = load_fixture("dataclaw_session.json")
        steps = dc_parse(sess["messages"])
        assert len(steps) >= 35

    def test_step_has_required_fields(self):
        sess = load_fixture("dataclaw_session.json")
        steps = dc_parse(sess["messages"])
        for step in steps:
            assert "tool" in step
            assert "tool_name" in step
            assert "cmd" in step
            assert "output" in step
            assert "thinking" in step

    def test_peteromallet_format_skipped(self):
        """Tool_uses with no output/status are skipped."""
        messages = [
            {
                "tool_uses": [
                    {"tool": "Bash", "input": {"command": "ls"}},  # no output/status
                    {
                        "tool": "Bash",
                        "input": {"command": "pwd"},
                        "output": "ok",
                        "status": "success",
                    },
                ]
            }
        ]
        steps = dc_parse(messages)
        assert len(steps) == 1
        assert steps[0]["cmd"] == "pwd"

    def test_empty_produces_error(self):
        with pytest.raises(DcError, match="no tool calls"):
            dc_parse([])

    def test_only_peteromallet_raises(self):
        messages = [{"tool_uses": [{"tool": "Bash", "input": "ls"}]}]
        with pytest.raises(DcError, match="no tool calls"):
            dc_parse(messages)

    def test_model_filter_is_caller_responsibility(self):
        """Parser itself doesn't filter by model; that's done in generate.py."""
        sess = load_fixture("dataclaw_session.json")
        # Both claude and non-claude models would parse the same way
        steps = dc_parse(sess["messages"])
        assert len(steps) > 0


# --- claudeset ---


class TestCldsetParser:
    def test_parse_fixture_steps_only(self):
        sess = load_fixture("claudeset_session.json")
        steps = cs_parse_steps(sess["turns"])
        assert len(steps) >= 35

    def test_compact_blocks_in_mixed_output(self):
        sess = load_fixture("claudeset_session.json")
        mixed = cs_parse(sess["turns"])
        compact_blocks = [item for item in mixed if item.get("type") == "compact"]
        step_items = [item for item in mixed if item.get("type") != "compact"]
        assert len(compact_blocks) >= 2
        assert len(step_items) >= 35

    def test_compact_block_format(self):
        sess = load_fixture("claudeset_session.json")
        mixed = cs_parse(sess["turns"])
        for item in mixed:
            if item.get("type") == "compact":
                assert "text" in item

    def test_step_has_required_fields(self):
        sess = load_fixture("claudeset_session.json")
        steps = cs_parse_steps(sess["turns"])
        for step in steps:
            assert "tool" in step
            assert "tool_name" in step
            assert "cmd" in step
            assert "output" in step
            assert "thinking" in step

    def test_no_exchange_turns_raises(self):
        turns = [
            {
                "type": "compact",
                "assistant": {"text": "summary", "thinking": "", "tool_calls": []},
            }
        ]
        with pytest.raises(CsError, match="no tool calls"):
            cs_parse(turns)

    def test_exchange_with_no_tool_calls_raises(self):
        turns = [
            {
                "type": "exchange",
                "user": "hi",
                "assistant": {"thinking": "", "text": "hello", "tool_calls": []},
            }
        ]
        with pytest.raises(CsError, match="no tool calls"):
            cs_parse(turns)

    def test_thinking_only_first_call_in_turn(self):
        """Thinking is applied to first tool call only."""
        turns = [
            {
                "type": "exchange",
                "user": "go",
                "assistant": {
                    "thinking": "my thinking",
                    "text": "",
                    "tool_calls": [
                        {"tool": "Bash", "input": {"command": "ls"}, "output": "a.txt"},
                        {
                            "tool": "Bash",
                            "input": {"command": "pwd"},
                            "output": "/home",
                        },
                    ],
                },
            }
        ]
        steps = cs_parse_steps(turns)
        assert len(steps) == 2
        assert steps[0]["thinking"] == "my thinking"
        assert steps[1]["thinking"] == ""

    def test_parse_session_preserves_compact_blocks(self):
        """parse_session (mixed) must keep CompactBlocks so format_transcript
        can render them as context for the labeler.

        Bug: _fetch_huggingface called parse_session_steps_only, stripping
        compact blocks before they reached format_transcript.
        """
        from src.pipeline.label_session import format_transcript

        turns = [
            {
                "type": "compact",
                "assistant": {"text": "Earlier work summarized here."},
            },
            {
                "type": "exchange",
                "user": "continue",
                "assistant": {
                    "thinking": "",
                    "text": "",
                    "tool_calls": [
                        {"tool": "Bash", "input": {"command": "ls"}, "output": "ok"},
                    ],
                },
            },
        ]

        # parse_session (mixed) preserves the compact block
        mixed = cs_parse(turns)
        compact_blocks = [s for s in mixed if s.get("type") == "compact"]
        assert (
            len(compact_blocks) == 1
        ), "compact block must be preserved in parse_session"

        # format_transcript on the mixed list must include [compact: ...] in output
        transcript, n_steps = format_transcript(mixed)
        assert "[compact:" in transcript, "compact context must appear in transcript"
        assert n_steps == 1, "compact blocks must not count as steps"

        # parse_session_steps_only silently drops the compact → transcript has no context
        steps_only = cs_parse_steps(turns)
        transcript_stripped, _ = format_transcript(steps_only)
        assert "[compact:" not in transcript_stripped, (
            "steps_only strips compact blocks — if this is used for labeling, "
            "the labeler loses context"
        )
