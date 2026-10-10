import asyncio
import json
from types import SimpleNamespace

from src.factory.agents.orchestrator import ConversationGenerator
from src.factory.graph.sampler import SampledChain
from src.factory.llm import parse_json
from src.factory.tools.registry import ToolRegistry


class FakeLLM:
    """Returns scripted replies in order. chat_json for planner/user, chat for assistant."""
    def __init__(self, json_replies, chat_replies):
        self.json_replies = list(json_replies)
        self.chat_replies = list(chat_replies)
        self.calls = 0

    async def chat_json(self, messages, temperature=0.7):
        self.calls += 1
        return self.json_replies.pop(0)

    async def chat(self, messages, tools=None, temperature=0.7):
        self.calls += 1
        return self.chat_replies.pop(0)


def tool_call_reply(name, args, call_id="call_1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id=call_id, function=fn)])


def text_reply(text):
    return SimpleNamespace(content=text, tool_calls=None)


PLAN = {"user_goal": "Find a flight", "persona": "busy nurse", "notes": "BWI to SFO, Dec 18"}


def test_full_flow_with_fake_llm():
    llm = FakeLLM(
        json_replies=[
            PLAN,
            {"message": "Can you find me flights from BWI to SFO on Dec 18?", "done": False},
            {"message": "Perfect, thanks!", "done": True},
        ],
        chat_replies=[
            tool_call_reply("search_flights", {"origin": "BWI", "destination": "SFO", "date": "2026-12-18"}),
            text_reply("I found 3 flights. The cheapest is $77."),
        ],
    )
    gen = ConversationGenerator(ToolRegistry.from_directory(), llm)
    chain = SampledChain(tools=["search_flights"], categories=["travel"])

    conv, session, reason = asyncio.run(gen.generate(chain, seed=1))

    roles = [m.role for m in conv.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert conv.num_tool_calls == 1
    assert reason == "user_satisfied"
    assert conv.validation_errors == []
    assert len(session.live_ids("flight_id")) == 3
    assert llm.calls == 5


def test_bad_call_is_blocked_and_logged():
    llm = FakeLLM(
        json_replies=[
            PLAN,
            {"message": "Book flight XYZ for me please, name A, email a@b.com", "done": False},
            {"message": "Okay, thanks.", "done": True},
        ],
        chat_replies=[
            tool_call_reply("book_flight", {"flight_id": "flight_fake", "passenger_name": "A", "email": "a@b.com"}),
            text_reply("Sorry, I couldn't find that flight. Want me to search first?"),
        ],
    )
    gen = ConversationGenerator(ToolRegistry.from_directory(), llm)
    chain = SampledChain(tools=["search_flights"], categories=["travel"])

    conv, session, _ = asyncio.run(gen.generate(chain, seed=1))

    rules = [e["rule"] for e in conv.validation_errors]
    assert "unknown_id" in rules
    tool_msg = next(m for m in conv.messages if m.role == "tool")
    assert "Invalid tool call" in tool_msg.content
    assert session.live_ids("booking_id") == []        # nothing was booked


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"done": true} hope that helps') == {"done": True}
