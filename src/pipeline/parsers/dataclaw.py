"""Parse DataClaw conversations into normalized step dicts."""

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


def _extract_output(tu: dict) -> str:
    """Extract output text from a tool_use dict."""
    out = tu.get("output", "")
    if isinstance(out, dict):
        return out.get("text", str(out))
    return str(out) if out else ""


def _extract_input_fields(_tool_name: str, inp) -> tuple[str, str | None]:
    """Extract cmd and file_path from tool input."""
    if isinstance(inp, dict):
        cmd = inp.get("command", inp.get("pattern", ""))
        file_path = inp.get("file_path", inp.get("path", None))
        if not cmd and file_path:
            cmd = file_path
        elif not cmd and not file_path:
            cmd = inp.get("description", inp.get("prompt", ""))[:200]
        return str(cmd), file_path
    return str(inp)[:200] if inp else "", None


def parse_session(messages: list[dict]) -> list[dict]:
    """Parse a DataClaw session's messages into normalized step dicts.

    Skips tool_uses with no output/status (peteromallet format).

    Args:
        messages: list of message dicts

    Returns:
        list of normalized step dicts with keys: tool, tool_name, cmd, file, output, thinking

    Raises:
        ParserSchemaError: if the session has no tool calls with outputs
    """
    steps: list[dict] = []
    pending_thinking: str | None = None

    for msg in messages:
        if "thinking" in msg and msg["thinking"]:
            pending_thinking = msg["thinking"]
            continue

        if "tool_uses" not in msg:
            continue

        for tu in msg["tool_uses"]:
            tool_name = tu.get("tool", "other")

            # Skip if no output field at all (peteromallet format)
            if "output" not in tu and "status" not in tu:
                continue

            inp = tu.get("input", "")
            cmd, file_path = _extract_input_fields("", inp)
            output = _extract_output(tu)

            status = tu.get("status", "")
            if status == "error" and output and not output.startswith("Error"):
                output = f"Error: {output}"

            steps.append(
                {
                    "tool": TOOL_MAP.get(tool_name, "other"),
                    "tool_name": tool_name,
                    "cmd": cmd,
                    "file": file_path,
                    "output": output,
                    "thinking": pending_thinking or "",
                }
            )
            pending_thinking = None

    if not steps:
        raise ParserSchemaError("no tool calls")

    return steps
