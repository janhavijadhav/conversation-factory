"""
Quality level 2: LLM judge.

Scores a finished conversation on 4 dimensions (1-5) and lists concrete
issues with message numbers, so the repair loop knows what to fix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.factory import config
from src.factory.models import Conversation, QualityScores
from src.factory.tools.registry import ToolRegistry

DIMENSIONS = ("correctness", "naturalness", "groundedness", "completeness")

JUDGE_PROMPT = """You are a strict reviewer of synthetic training data for tool-using AI assistants.
Bad examples teach a model bad habits, so be critical.

Tools available in this conversation (? = optional argument):
{tools}

User's goal: {goal}

Conversation (numbered):
{transcript}

Automatic checks found:
{auto_notes}

Score each dimension from 1 (very bad) to 5 (excellent):
- correctness: right tools, right arguments, sensible order, no pointless or missing calls.
- naturalness: the user sounds like a real person; the assistant is clear, concise and
  polite; no internal IDs shown to the user when listing options.
- groundedness: every fact the assistant states (prices, times, names, actions taken)
  matches the tool results. Nothing invented. No claims of actions a tool did not do.
- completeness: the user's goal was achieved AND the user was clearly told the outcome.
  The final assistant message must tell the user what was done, with the key details
  (for example confirmation numbers, totals, dates or times). If the tools did the work
  but the final reply is vague or empty (like "Okay" or "Noted"), completeness is at
  most 2, because the user never learns what happened. If the goal could not be
  achieved, the assistant must clearly explain why.

List each concrete issue with its message number. Set "fixable_by_rewrite" to true ONLY
if every issue can be fixed by rewriting the assistant's text replies, without changing
any tool call or tool result.

Return ONLY a JSON object:
{"correctness": 1-5, "naturalness": 1-5, "groundedness": 1-5, "completeness": 1-5,
 "issues": [{"message": 4, "problem": "..."}], "fixable_by_rewrite": false,
 "feedback": "one or two sentences"}"""


def format_transcript(conv: Conversation, max_tool_chars: int = 600) -> str:
    """Numbered transcript; numbers match message positions (1-based)."""
    lines = []
    for i, m in enumerate(conv.messages, start=1):
        if m.role == "user":
            lines.append(f"[{i}] USER: {m.content}")
        elif m.role == "assistant":
            parts = [m.content] if m.content else []
            parts += [f"<calls {tc.name}({json.dumps(tc.arguments)})>" for tc in m.tool_calls]
            lines.append(f"[{i}] ASSISTANT: " + " ".join(parts))
        elif m.role == "tool":
            text = m.content or ""
            if len(text) > max_tool_chars:
                text = text[:max_tool_chars] + "..."
            lines.append(f"[{i}] TOOL {m.name} -> {text}")
    return "\n".join(lines)


def _clamp(value: Any) -> float:
    try:
        return float(min(5.0, max(1.0, float(value))))
    except (TypeError, ValueError):
        return 1.0


@dataclass
class JudgeResult:
    scores: QualityScores
    issues: list[dict[str, Any]] = field(default_factory=list)
    fixable_by_rewrite: bool = False


class LLMJudge:
    def __init__(self, registry: ToolRegistry, llm):
        self.registry = registry
        self.llm = llm

    def _tool_lines(self, conv: Conversation) -> str:
        categories = {self.registry.get(t).category
                      for t in conv.scenario.tool_chain if self.registry.has(t)}
        lines = []
        for t in self.registry.all():
            if t.category in categories:
                params = ", ".join(p.name + ("" if p.required else "?") for p in t.parameters) or "none"
                lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    @staticmethod
    def _auto_notes(conv: Conversation) -> str:
        notes = [f"grounding fixed {f['tool']}.{f['param']} ({f['reason']})" for f in conv.grounding_fixes]
        notes += [f"validator blocked {v['tool']}: {v['message']}" for v in conv.validation_errors]
        return "\n".join(f"- {n}" for n in notes) or "- none"

    async def judge(self, conv: Conversation) -> JudgeResult:
        prompt = (JUDGE_PROMPT
                  .replace("{tools}", self._tool_lines(conv))
                  .replace("{goal}", conv.scenario.user_goal)
                  .replace("{transcript}", format_transcript(conv))
                  .replace("{auto_notes}", self._auto_notes(conv)))
        data = await self.llm.chat_json([{"role": "user", "content": prompt}],
                                        temperature=config.JUDGE_TEMPERATURE)

        scores = QualityScores(
            **{d: _clamp(data.get(d)) for d in DIMENSIONS},
            feedback=str(data.get("feedback", "")),
        )
        issues = [i for i in data.get("issues", []) if isinstance(i, dict)]
        return JudgeResult(scores, issues, bool(data.get("fixable_by_rewrite", False)))
