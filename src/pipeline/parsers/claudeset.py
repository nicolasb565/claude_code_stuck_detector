"""Parse lelouch0110/claudeset-community dataset into normalized step dicts."""

from typing import Union

TOOL_MAP = {
    "Bash": "bash",
    "bash": "bash",
    "Read": "view",
    "read": "view",
    "Edit": "edit",
    "edit": "edit",
    "Write": "edit",
    "write": "edit",
    "MultiEdit": "edit",
    "Grep": "search",
    "grep": "search",
    "Glob": "search",
    "glob": "search",
    "Agent": "other",
    "Task": "other",
    "TodoRead": "other",
    "TodoWrite": "other",
}


class ParserSchemaError(Exception):
    pass


# Type alias for CompactBlock
CompactBlock = dict  # {"type": "compact", "text": str}


def _make_compact_block(text: str) -> CompactBlock:
    return {"type": "compact", "text": text}


def _extract_cmd_and_file(_tool_name: str, inp: dict) -> tuple[str, str | None]:
    """Extract cmd and file from tool input dict."""
    if not isinstance(inp, dict):
        return "", None
    cmd = inp.get("command", inp.get("pattern", ""))
    file_path = inp.get("file_path", inp.get("path", None))
    if not cmd and file_path:
        cmd = file_path
    elif not cmd and not file_path:
        cmd = inp.get("description", inp.get("prompt", inp.get("todos", "")))[:200]
    return str(cmd) if cmd else "", file_path


def parse_session(turns: list[dict]) -> list[Union[dict, CompactBlock]]:
    """Parse claudeset turns into normalized step dicts and CompactBlocks.

    Turn format:
        {"type": "exchange"|"compact", "user": ...,
         "assistant": {"thinking": str, "text": str, "tool_calls": [...]}}

    - compact turns: yield a CompactBlock marker (not counted as steps)
    - exchange turns: parse tool_calls into normalized step dicts

    Args:
        turns: list of turn dicts

    Returns:
        list of step dicts and CompactBlocks (mixed)

    Raises:
        ParserSchemaError: if no exchange turns have tool calls
    """
    result: list[Union[dict, CompactBlock]] = []
    has_tool_calls = False

    for turn in turns:
        turn_type = turn.get("type", "exchange")

        if turn_type == "compact":
            assistant = turn.get("assistant", {})
            text = ""
            if isinstance(assistant, dict):
                text = assistant.get("text", "")
            elif isinstance(assistant, str):
                text = assistant
            result.append(_make_compact_block(text))
            continue

        # exchange turn
        assistant = turn.get("assistant", {})
        if not isinstance(assistant, dict):
            continue

        thinking = assistant.get("thinking", "")
        tool_calls = assistant.get("tool_calls", [])

        if not tool_calls:
            continue

        for tc in tool_calls:
            tool_name = tc.get("tool", "other")
            inp = tc.get("input", {})
            output = tc.get("output", "")

            cmd, file_path = _extract_cmd_and_file(tool_name, inp)

            result.append(
                {
                    "tool": TOOL_MAP.get(tool_name, "other"),
                    "tool_name": tool_name,
                    "cmd": cmd,
                    "file": file_path,
                    "output": str(output) if output else "",
                    "thinking": thinking,
                }
            )
            has_tool_calls = True
            # Only use thinking for first tool call in turn
            thinking = ""

    if not has_tool_calls:
        raise ParserSchemaError("no tool calls")

    return result


def parse_session_steps_only(turns: list[dict]) -> list[dict]:
    """Parse claudeset turns, returning only step dicts (no CompactBlocks).

    Args:
        turns: list of turn dicts

    Returns:
        list of normalized step dicts only

    Raises:
        ParserSchemaError: if no exchange turns have tool calls
    """
    mixed = parse_session(turns)
    return [item for item in mixed if item.get("type") != "compact"]
