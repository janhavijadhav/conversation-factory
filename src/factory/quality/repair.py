"""
Quality level 3: repair loop.

For each conversation:
  1. free structural checks (no LLM)  -> fail = regenerate
  2. LLM judge                         -> pass = done
  3. repair:
       wording-only issues  -> rewrite the assistant's text (cheap, structure untouched)
       first regeneration   -> same scenario, new attempt
       second regeneration  -> fresh scenario (the old one may be impossible)
  ...up to MAX_REPAIR_ATTEMPTS times.
"""
from __future__ import annotations

from typing import Any

from src.factory import config
from src.factory.models import Conversation, QualityScores
from src.factory.quality.judge import LLMJudge, format_transcript

REWRITE_PROMPT = """You fix assistant replies in a training conversation for a tool-using AI.

Conversation (numbered):
{transcript}

Problems a reviewer found:
{issues}

Rewrite ONLY the assistant text replies that have problems.
- Every fact must match the tool results exactly. Remove anything invented.
- Never claim an action that no tool performed.
- Don't mention internal IDs when listing options.
- Keep the same helpful tone and roughly the same length.

Return ONLY a JSON object:
{"rewrites": [{"message": 4, "content": "new reply text"}]}"""


# ---- level 3a: free structural checks --------------------------------------

def structural_problems(conv: Conversation, max_blocked: int = 2) -> list[str]:
    problems: list[str] = []
    msgs = conv.messages
    if not msgs:
        return ["empty conversation"]
    if msgs[0].role != "user":
        problems.append("does not start with a user message")
    last = msgs[-1]
    if last.role != "assistant" or last.tool_calls or not last.content:
        problems.append("does not end with an assistant text reply")
    if conv.num_tool_calls == 0:
        problems.append("no tool calls")
    if conv.stop_reason == "max_tool_rounds":
        problems.append("hit the tool-round limit")

    call_ids = {tc.id for tc in conv.tool_calls}
    result_ids = {m.tool_call_id for m in msgs if m.role == "tool"}
    if call_ids - result_ids:
        problems.append("a tool call has no result")
    if len(conv.validation_errors) > max_blocked:
        problems.append(f"too many blocked calls ({len(conv.validation_errors)})")
    return problems


# ---- level 3b: rewriter ----------------------------------------------------

class Rewriter:
    def __init__(self, llm):
        self.llm = llm

    async def rewrite(self, conv: Conversation, issues: list[dict[str, Any]]) -> Conversation:
        issue_text = "\n".join(f"- message {i.get('message')}: {i.get('problem')}" for i in issues)
        prompt = (REWRITE_PROMPT
                  .replace("{transcript}", format_transcript(conv))
                  .replace("{issues}", issue_text))
        data = await self.llm.chat_json([{"role": "user", "content": prompt}],
                                        temperature=config.REWRITE_TEMPERATURE)

        messages = list(conv.messages)
        changed = 0
        for r in data.get("rewrites", []):
            try:
                idx = int(r.get("message")) - 1
            except (TypeError, ValueError):
                continue
            new_text = r.get("content")
            # Only assistant text may change. Tool calls and results are never touched.
            if (0 <= idx < len(messages) and messages[idx].role == "assistant"
                    and messages[idx].content and isinstance(new_text, str) and new_text.strip()):
                messages[idx] = messages[idx].model_copy(update={"content": new_text.strip()})
                changed += 1

        metadata = {**conv.metadata,
                    "rewritten_messages": conv.metadata.get("rewritten_messages", 0) + changed}
        return conv.model_copy(update={"messages": messages, "metadata": metadata})


# ---- the loop --------------------------------------------------------------

class QualityPipeline:
    def __init__(self, generator, judge: LLMJudge, rewriter: Rewriter,
                 threshold: float = config.JUDGE_PASS_THRESHOLD,
                 min_dimension: float = config.MIN_DIMENSION_SCORE,
                 max_attempts: int = config.MAX_REPAIR_ATTEMPTS):
        self.generator = generator
        self.judge = judge
        self.rewriter = rewriter
        self.threshold = threshold
        self.min_dimension = min_dimension
        self.max_attempts = max_attempts

    def passes(self, s: QualityScores) -> bool:
        lowest = min(s.correctness, s.naturalness, s.groundedness, s.completeness)
        return s.mean >= self.threshold and lowest >= self.min_dimension

    async def run(self, chain, seed: int | None = None) -> Conversation:
        conv, _, _ = await self.generator.generate(chain, seed=seed)
        history: list[dict[str, Any]] = []
        passed = False
        attempt = 0
        regenerations = 0

        for attempt in range(self.max_attempts + 1):
            problems = structural_problems(conv)
            result = None
            if problems:
                entry: dict[str, Any] = {"attempt": attempt, "stage": "structural", "problems": problems}
            else:
                result = await self.judge.judge(conv)
                conv.scores = result.scores
                entry = {"attempt": attempt, "stage": "judge",
                         "mean": round(result.scores.mean, 2), "issues": result.issues}
            history.append(entry)

            if result and self.passes(result.scores):
                passed = True
                break
            if attempt == self.max_attempts:
                break

            if result and result.fixable_by_rewrite and result.issues:
                conv = await self.rewriter.rewrite(conv, result.issues)
                entry["action"] = "rewrite"
            else:
                regenerations += 1
                new_seed = None if seed is None else seed * 1000 + attempt + 1
                # First regeneration keeps the scenario. After that, the scenario itself
                # may be the problem (e.g. it asks for something no tool can provide).
                keep = conv.scenario if regenerations == 1 else None
                conv, _, _ = await self.generator.generate(chain, seed=new_seed, scenario=keep)
                entry["action"] = "regenerate"
                entry["new_scenario"] = keep is None

        conv.passed = passed
        conv.repair_attempts = attempt
        conv.metadata = {**conv.metadata, "quality_history": history}
        return conv
