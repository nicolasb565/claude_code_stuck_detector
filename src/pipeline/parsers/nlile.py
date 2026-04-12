"""Parse Anthropic API format (nlile / Claude Code transcripts) into normalized step dicts."""

TOOL_MAP = {
    "Bash": "bash",
    "bash": "bash",
    "Read": "view",
    "view": "view",
    "Edit": "edit",
    "Write": "edit",
    "MultiEdit": "edit",
    "Grep": "search",
    "Glob": "search",
    "Agent": "other",
    "Task": "other",
    "TodoRead": "other",
    "TodoWrite": "other",
}


class ParserSchemaError(Exception):
    pass


def parse_session(
    messages: list[dict],
) -> list[dict]:
    """Parse Anthropic API messages into normalized step dicts.

    Args:
        messages: list of message dicts with role and content fields

    Returns:
        list of normalized step dicts with keys: tool, tool_name, cmd, file, output, thinking

    Raises:
        ParserSchemaError: if the session has no tool calls
    """
    # pylint: disable=too-many-branches
    steps: list[dict] = []
    pending: dict[str, dict] = {}
    last_thinking = ""

    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "thinking":
                last_thinking = block.get("thinking", block.get("text", ""))
            elif btype == "tool_use":
                inp = block.get("input", {})
                tid = block.get("id", "")
                tool_name = block.get("name", "")
                cmd = inp.get("command", inp.get("file_path", inp.get("pattern", "")))
                if not cmd:
                    if isinstance(inp, dict):
                        cmd = inp.get("description", inp.get("prompt", ""))[:200]
                pending[tid] = {
                    "tool": TOOL_MAP.get(tool_name, "other"),
                    "tool_name": tool_name,
                    "cmd": str(cmd) if cmd else "",
                    "file": (
                        inp.get("file_path", inp.get("path"))
                        if isinstance(inp, dict)
                        else None
                    ),
                    "thinking": last_thinking,
                }
                last_thinking = ""
            elif btype == "tool_result":
                tid = block.get("tool_use_id", "")
                if tid in pending:
                    tu = pending.pop(tid)
                    out = block.get("content", "")
                    if isinstance(out, list):
                        out = "\n".join(
                            b.get("text", "")
                            for b in out
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    tu["output"] = str(out) if out else ""
                    steps.append(tu)

    # Flush tool calls without results
    for tu in pending.values():
        tu["output"] = ""
        steps.append(tu)

    if not steps:
        raise ParserSchemaError("no tool calls")

    return steps
