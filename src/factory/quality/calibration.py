"""
Judge calibration with controlled corruptions.

Take conversations we believe are good, inject one known flaw at a time, and
check whether the judge fails them. Unmodified originals are controls that
should pass. This measures the judge's detection rate per error type and its
false alarm rate, with no hand labeling.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from src.factory.graph.tool_graph import is_id_param
from src.factory.models import Conversation, Message, Scenario

NUMBER = re.compile(r"\d+(?:\.\d+)?")
Corrupted = Optional[tuple[Conversation, str]]


# ---- loading saved conversations -------------------------------------------

def record_to_conversation(record: dict) -> Conversation | None:
    """Rebuild a Conversation from a saved JSON record. None if it isn't usable."""
    messages = [Message(**m) for m in record.get("messages", [])]
    if not messages:
        return None
    first_user = next((m.content for m in messages if m.role == "user"), None)
    tools = [tc.name for m in messages for tc in m.tool_calls]
    last = messages[-1]
    if not first_user or not tools or last.role != "assistant" or not last.content or last.tool_calls:
        return None
    return Conversation(
        scenario=Scenario(
            user_goal=record.get("goal") or first_user,
            tool_chain=tools,
            category=record.get("category", "travel"),
            persona=record.get("persona", "a regular user"),
        ),
        messages=messages,
        stop_reason=record.get("stop_reason", "user_satisfied"),
    )


def load_conversations(paths: list[str]) -> list[tuple[str, Conversation]]:
    out = []
    for path in map(Path, paths):
        text = path.read_text().strip()
        if not text:
            continue
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = [json.loads(text)]
        for k, record in enumerate(records):
            conv = record_to_conversation(record)
            if conv:
                out.append((f"{path.name}#{k}", conv))
    return out


# ---- helpers ---------------------------------------------------------------

def _replace_final(conv: Conversation, text: str) -> Conversation:
    msgs = list(conv.messages)
    msgs[-1] = msgs[-1].model_copy(update={"content": text})
    return conv.model_copy(update={"messages": msgs})


def _final_text(conv: Conversation) -> str:
    return conv.messages[-1].content or ""


# ---- the corruptions -------------------------------------------------------

def wrong_number(conv: Conversation) -> Corrupted:
    """Change a number in the final reply that came from a tool result (prices first)."""
    text = _final_text(conv)
    tool_text = " ".join(m.content or "" for m in conv.messages if m.role == "tool")
    candidates = [m for m in NUMBER.finditer(text)
                  if m.group(0) in tool_text and len(m.group(0).replace(".", "")) >= 2]
    if not candidates:
        return None
    candidates.sort(key=lambda m: "." not in m.group(0))      # decimals (prices) first
    m = candidates[0]
    token = m.group(0)
    new = f"{float(token) * 1.37:.2f}" if "." in token else str(int(token) + 7)
    return _replace_final(conv, text[:m.start()] + new + text[m.end():]), f"changed {token} to {new}"


FALSE_CLAIMS = {
    "send_email": "I've also emailed you a confirmation with all the details.",
    "create_event": "I've also added this to your calendar so you don't forget.",
    "book_hotel": "I've also reserved a hotel room for you nearby.",
}


def false_action(conv: Conversation) -> Corrupted:
    """Claim an action that no tool performed."""
    used = {tc.name for tc in conv.tool_calls}
    for tool, claim in FALSE_CLAIMS.items():
        if tool not in used:
            return _replace_final(conv, _final_text(conv).rstrip() + " " + claim), f"claims {tool} that never ran"
    return None


def invented_detail(conv: Conversation) -> Corrupted:
    """Add a fact that appears in no tool result."""
    extra = "Good news: a 15% loyalty discount was applied automatically."
    return _replace_final(conv, _final_text(conv).rstrip() + " " + extra), "invented a 15% discount"


def wrong_id_argument(conv: Conversation) -> Corrupted:
    """Swap an ID argument for one that was never returned by any tool."""
    msgs = list(conv.messages)
    for i, m in enumerate(msgs):
        for j, tc in enumerate(m.tool_calls):
            for key, value in tc.arguments.items():
                if is_id_param(key) and isinstance(value, str):
                    fake = f"{key[:-3]}_ffffff"
                    calls = list(m.tool_calls)
                    calls[j] = tc.model_copy(update={"arguments": {**tc.arguments, key: fake}})
                    msgs[i] = m.model_copy(update={"tool_calls": calls})
                    return conv.model_copy(update={"messages": msgs}), f"{tc.name}.{key} set to {fake}"
    return None


def missing_step(conv: Conversation) -> Corrupted:
    """Delete the last tool call and its result, but keep the reply that relies on it."""
    msgs = conv.messages
    call_positions = [i for i, m in enumerate(msgs) if m.role == "assistant" and m.tool_calls]
    if len(call_positions) < 2:
        return None
    pos = call_positions[-1]
    removed_ids = {tc.id for tc in msgs[pos].tool_calls}
    names = ", ".join(tc.name for tc in msgs[pos].tool_calls)
    kept = [m for k, m in enumerate(msgs)
            if k != pos and not (m.role == "tool" and m.tool_call_id in removed_ids)]
    return conv.model_copy(update={"messages": kept}), f"removed {names} but kept the reply"


def unhelpful_ending(conv: Conversation) -> Corrupted:
    """Replace the final reply with something that doesn't finish the job."""
    return _replace_final(conv, "Okay, noted."), "final reply replaced with 'Okay, noted.'"


CORRUPTIONS: dict[str, Callable[[Conversation], Corrupted]] = {
    "wrong_number": wrong_number,
    "false_action": false_action,
    "invented_detail": invented_detail,
    "wrong_id_argument": wrong_id_argument,
    "missing_step": missing_step,
    "unhelpful_ending": unhelpful_ending,
}


# ---- running ---------------------------------------------------------------

async def calibrate(judge, passes: Callable, bases: list[tuple[str, Conversation]]) -> list[dict]:
    """Judge each original (control) and each corrupted copy. Returns one row per case."""
    rows = []
    for name, conv in bases:
        cases = [("control", conv, "unchanged original")]
        for kind, fn in CORRUPTIONS.items():
            result = fn(conv)
            if result:
                cases.append((kind, result[0], result[1]))

        for kind, case, description in cases:
            r = await judge.judge(case)
            ok = passes(r.scores)
            rows.append({
                "base": name,
                "kind": kind,
                "description": description,
                "passed": ok,
                "correct_verdict": ok if kind == "control" else not ok,
                "scores": {d: getattr(r.scores, d) for d in
                           ("correctness", "naturalness", "groundedness", "completeness")},
                "feedback": r.scores.feedback,
            })
            print(f"  {name:<22} {kind:<18} {'PASS' if ok else 'FAIL'}"
                  f"  {'ok' if rows[-1]['correct_verdict'] else '<-- judge got this wrong'}")
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for kind in ["control"] + list(CORRUPTIONS):
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        correct = sum(r["correct_verdict"] for r in subset)
        avg = {d: round(sum(r["scores"][d] for r in subset) / len(subset), 2)
               for d in ("correctness", "naturalness", "groundedness", "completeness")}
        summary[kind] = {"cases": len(subset), "correct": correct,
                         "rate": round(correct / len(subset), 2), "avg_scores": avg}
    return summary
