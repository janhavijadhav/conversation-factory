import asyncio

from src.factory.models import Conversation, Message, Scenario, ToolCall
from src.factory.quality.judge import LLMJudge, format_transcript
from src.factory.quality.repair import QualityPipeline, Rewriter, structural_problems
from src.factory.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.from_directory()

GOOD = {"correctness": 5, "naturalness": 4, "groundedness": 5, "completeness": 5,
        "issues": [], "fixable_by_rewrite": False, "feedback": "good"}
BAD_FIXABLE = {"correctness": 5, "naturalness": 4, "groundedness": 2, "completeness": 4,
               "issues": [{"message": 4, "problem": "claims a booking that never happened"}],
               "fixable_by_rewrite": True, "feedback": "hallucinated action"}
BAD_NOT_FIXABLE = {"correctness": 2, "naturalness": 4, "groundedness": 4, "completeness": 2,
                   "issues": [{"message": 2, "problem": "wrong tool"}],
                   "fixable_by_rewrite": False, "feedback": "wrong tool"}


def make_conv(final_text="I found 3 flights for you.", end_with_tool=False):
    tc = ToolCall(id="call_1", name="search_flights",
                  arguments={"origin": "BWI", "destination": "SFO", "date": "2026-12-18"})
    msgs = [
        Message(role="user", content="Find me flights BWI to SFO on Dec 18"),
        Message(role="assistant", tool_calls=[tc]),
        Message(role="tool", tool_call_id="call_1", name="search_flights", content='{"results": []}'),
    ]
    if not end_with_tool:
        msgs.append(Message(role="assistant", content=final_text))
    return Conversation(
        scenario=Scenario(user_goal="find a flight", tool_chain=["search_flights"], category="travel"),
        messages=msgs, stop_reason="user_satisfied",
    )


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def chat_json(self, messages, temperature=0.7):
        self.calls += 1
        return self.replies.pop(0)


class FakeGenerator:
    def __init__(self, convs):
        self.convs = list(convs)
        self.calls = 0

    async def generate(self, chain, seed=None, scenario=None):
        self.calls += 1
        return self.convs.pop(0), None, "user_satisfied"


def pipeline(gen, judge_replies, rewrite_replies=()):
    judge_llm = FakeLLM(judge_replies)
    p = QualityPipeline(gen, LLMJudge(REGISTRY, judge_llm), Rewriter(FakeLLM(rewrite_replies)),
                        threshold=3.5, min_dimension=3, max_attempts=2)
    return p, judge_llm


def test_structural_ok():
    assert structural_problems(make_conv()) == []


def test_structural_catches_missing_final_reply():
    problems = structural_problems(make_conv(end_with_tool=True))
    assert "does not end with an assistant text reply" in problems


def test_transcript_numbering():
    assert "[4] ASSISTANT: I found 3 flights" in format_transcript(make_conv())


def test_judge_clamps_and_defaults():
    llm = FakeLLM([{"correctness": 9, "naturalness": "4", "groundedness": 0, "issues": []}])
    result = asyncio.run(LLMJudge(REGISTRY, llm).judge(make_conv()))
    assert result.scores.correctness == 5
    assert result.scores.naturalness == 4
    assert result.scores.groundedness == 1
    assert result.scores.completeness == 1        # missing -> lowest score


def test_passes_first_time():
    gen = FakeGenerator([make_conv()])
    p, _ = pipeline(gen, [GOOD])
    conv = asyncio.run(p.run(chain=None, seed=1))
    assert conv.passed and conv.repair_attempts == 0 and gen.calls == 1


def test_rewrite_fixes_wording():
    gen = FakeGenerator([make_conv(final_text="Your flight is booked!")])
    p, _ = pipeline(gen, [BAD_FIXABLE, GOOD],
                    rewrite_replies=[{"rewrites": [{"message": 4, "content": "I found 3 flights. Want me to book one?"}]}])
    conv = asyncio.run(p.run(chain=None, seed=1))
    assert conv.passed and conv.repair_attempts == 1
    assert conv.messages[-1].content.startswith("I found 3 flights")
    assert conv.metadata["quality_history"][0]["action"] == "rewrite"
    assert gen.calls == 1                          # no regeneration needed


def test_regenerate_when_not_fixable():
    gen = FakeGenerator([make_conv(), make_conv()])
    p, _ = pipeline(gen, [BAD_NOT_FIXABLE, GOOD])
    conv = asyncio.run(p.run(chain=None, seed=1))
    assert conv.passed and gen.calls == 2
    assert conv.metadata["quality_history"][0]["action"] == "regenerate"


def test_gives_up_after_max_attempts():
    gen = FakeGenerator([make_conv(), make_conv(), make_conv()])
    p, _ = pipeline(gen, [BAD_NOT_FIXABLE] * 3)
    conv = asyncio.run(p.run(chain=None, seed=1))
    assert not conv.passed
    assert conv.repair_attempts == 2


def test_structural_failure_skips_judge():
    gen = FakeGenerator([make_conv(end_with_tool=True), make_conv()])
    p, judge_llm = pipeline(gen, [GOOD])
    conv = asyncio.run(p.run(chain=None, seed=1))
    assert conv.passed
    assert judge_llm.calls == 1                    # judge only ran on the good one
    assert conv.metadata["quality_history"][0]["stage"] == "structural"
