import asyncio

from src.factory.agents.planner import ScenarioPlanner
from src.factory.agents.user_agent import UserAgent, looks_like_assistant
from src.factory.graph.sampler import SampledChain
from src.factory.models import Conversation, Message, Scenario, ToolCall
from src.factory.quality.judge import LLMJudge
from src.factory.quality.repair import QualityPipeline, Rewriter
from src.factory.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.from_directory()
SCENARIO = Scenario(user_goal="find a flight", tool_chain=["search_flights"], category="travel",
                    persona="Ines Duarte, 29")
BAD = {"correctness": 2, "naturalness": 4, "groundedness": 4, "completeness": 2,
       "issues": [{"message": 2, "problem": "impossible goal"}], "fixable_by_rewrite": False}
GOOD = {"correctness": 5, "naturalness": 5, "groundedness": 5, "completeness": 5,
        "issues": [], "fixable_by_rewrite": False}


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def chat_json(self, messages, temperature=0.7):
        self.calls += 1
        return self.replies.pop(0)


def good_conv():
    tc = ToolCall(id="c1", name="search_flights",
                  arguments={"origin": "BWI", "destination": "SFO", "date": "2026-12-18"})
    return Conversation(scenario=SCENARIO, stop_reason="user_satisfied", messages=[
        Message(role="user", content="flights BWI to SFO Dec 18?"),
        Message(role="assistant", tool_calls=[tc]),
        Message(role="tool", tool_call_id="c1", name="search_flights", content='{"results": []}'),
        Message(role="assistant", content="I found 3 flights."),
    ])


def test_looks_like_assistant():
    assert looks_like_assistant("Sure, I can pull the transactions for you.")
    assert looks_like_assistant("Your JetBlue flight has been booked for 6am.")
    assert looks_like_assistant("Hi! How can I help you today?")
    assert not looks_like_assistant("Sure, here are the details: name Jordan, email j@x.com")
    assert not looks_like_assistant("Can you find me a flight to Seattle?")


def test_user_agent_retries_role_drift():
    llm = FakeLLM([{"message": "Sure, I can pull the transactions for you.", "done": False},
                   {"message": "Can you check my recent charges?", "done": False}])
    agent = UserAgent(llm)
    reply = asyncio.run(agent.respond(SCENARIO, []))
    assert reply.message == "Can you check my recent charges?"
    assert llm.calls == 2 and agent.role_drift_fixes == 1


def test_second_regeneration_plans_fresh_scenario():
    class RecordingGenerator:
        def __init__(self):
            self.scenarios = []

        async def generate(self, chain, seed=None, scenario=None):
            self.scenarios.append(scenario)
            return good_conv(), None, "user_satisfied"

    gen = RecordingGenerator()
    pipeline = QualityPipeline(gen, LLMJudge(REGISTRY, FakeLLM([BAD, BAD, GOOD])), Rewriter(None),
                               threshold=3.5, min_dimension=3, max_attempts=2)
    conv = asyncio.run(pipeline.run(chain=None, seed=1))
    assert conv.passed
    assert gen.scenarios[0] is None           # original generation plans its own scenario
    assert gen.scenarios[1] is not None       # 1st regeneration keeps the scenario
    assert gen.scenarios[2] is None           # 2nd regeneration plans a fresh one
    assert conv.metadata["quality_history"][1]["new_scenario"] is True


def test_planner_lists_return_fields():
    planner = ScenarioPlanner(REGISTRY, llm=None)
    lines = planner._tool_lines(SampledChain(tools=["search_flights"], categories=["travel"]))
    assert "Returns: flight_id, airline, price, departure_time" in lines
