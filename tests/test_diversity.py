import asyncio
from collections import Counter

from src.factory.agents.planner import ScenarioPlanner
from src.factory.graph.sampler import ChainSampler, SampledChain
from src.factory.graph.tool_graph import ToolGraph
from src.factory.models import ToolCall
from src.factory.quality.diversity import DiversityTracker, persona_name
from src.factory.state.session import SessionState
from src.factory.tools.mock_executor import MockExecutor
from src.factory.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.from_directory()
GRAPH = ToolGraph(REGISTRY)


def test_persona_name():
    assert persona_name("Tomás Herrera, 52, retired teacher") == "Tomás Herrera"
    assert persona_name("a busy nurse who travels") == ""


def test_tracker_counts():
    t = DiversityTracker(REGISTRY)
    t.add_record("travel", ["search_flights", "book_flight", "send_email"], "Maya Patel, 30")
    t.add_record("banking", ["list_accounts"], "Maya Shah, 41")
    s = t.stats()
    assert s["conversations"] == 2
    assert s["mean_tool_calls"] == 2.0
    assert s["multi_step_ratio"] == 0.5
    assert s["long_chain_ratio"] == 0.5
    assert s["repeated_first_names"] == {"Maya": 2}


def test_least_used_category_first():
    t = DiversityTracker(REGISTRY)
    for _ in range(3):
        t.add_record("travel", ["search_flights", "book_flight"])
    t.add_record("ecommerce", ["search_products", "get_product"])
    plan = t.steering()
    assert plan.categories[-1] == "travel"
    assert plan.categories[0] in {"banking", "calendar"}


def test_min_len_raised_when_chains_short():
    t = DiversityTracker(REGISTRY, target_long_ratio=0.5)
    for _ in range(5):
        t.add_record("travel", ["search_flights", "book_flight"])
    assert t.steering().min_len == 3


def test_unused_tools_boosted_and_names_avoided():
    t = DiversityTracker(REGISTRY)
    t.add_record("travel", ["search_flights", "book_flight"], "Maya Patel, 30")
    plan = t.steering()
    assert plan.tool_boosts["transfer_funds"] == 3.0
    assert "search_flights" not in plan.tool_boosts
    assert plan.avoid_names == ["Maya"]


def test_boosts_shift_sampling():
    plain = ChainSampler(GRAPH, seed=11)
    boosted = ChainSampler(GRAPH, seed=11)
    count = lambda s, b: Counter(t for _ in range(400) for t in s.sample(category="banking", boosts=b).tools)
    before = count(plain, None)["transfer_funds"]
    after = count(boosted, {"transfer_funds": 10.0})["transfer_funds"]
    assert after > before


def test_boosted_chains_still_valid():
    sampler = ChainSampler(GRAPH, seed=3)
    boosts = {t: 5.0 for t in REGISTRY.names()[:10]}
    for _ in range(200):
        chain = sampler.sample(min_len=1, boosts=boosts)
        assert sampler.validate(chain.tools) == []


def test_search_first_result_matches_query():
    ex = MockExecutor(REGISTRY, SessionState(), seed=0)
    out = ex.execute(ToolCall(name="search_events", arguments={"query": "dentist"})).output
    assert out["results"][0]["title"] == "Dentist"


def test_planner_receives_avoid_names():
    class CaptureLLM:
        prompt = ""
        async def chat_json(self, messages, temperature=0.7):
            CaptureLLM.prompt = messages[0]["content"]
            return {"user_goal": "x", "persona": "Ines Duarte, 29", "notes": "y"}

    planner = ScenarioPlanner(REGISTRY, CaptureLLM())
    planner.avoid_names = ["Maya", "Jordan"]
    asyncio.run(planner.plan(SampledChain(tools=["search_flights"], categories=["travel"])))
    assert "Maya, Jordan" in CaptureLLM.prompt
