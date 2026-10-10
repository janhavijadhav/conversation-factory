"""Convert our Message objects to OpenAI-style message dicts."""
from __future__ import annotations

import json

from src.factory.models import Message


def to_openai_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
        elif m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def user_view_transcript(messages: list[Message]) -> str:
    """What the user actually sees: no tool calls, no raw tool results."""
    lines = []
    for m in messages:
        if m.role == "user":
            lines.append(f"You: {m.content}")
        elif m.role == "assistant" and m.content:
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)
