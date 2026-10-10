import pytest

from src.factory.graph.sampler import ChainSampler
from src.factory.graph.tool_graph import ToolGraph
from src.factory.tools.registry import ToolRegistry


@pytest.fixture(scope="module")
def graph():
    return ToolGraph(ToolRegistry.from_directory())


def test_same_seed_same_chain(graph):
    a = ChainSampler(graph, seed=42).sample()
    b = ChainSampler(graph, seed=42).sample()
    assert a.tools == b.tools


def test_every_sampled_chain_is_valid(graph):
    sampler = ChainSampler(graph, seed=1)
    for _ in range(300):
        chain = sampler.sample(min_len=1, max_len=5)
        assert chain is not None
        assert sampler.validate(chain.tools) == [], chain.tools


def test_chains_start_with_entry_tool(graph):
    sampler = ChainSampler(graph, seed=2)
    entries = set(graph.entry_tools())
    for _ in range(50):
        assert sampler.sample().tools[0] in entries


def test_category_filter(graph):
    sampler = ChainSampler(graph, seed=3)
    for _ in range(30):
        chain = sampler.sample(category="banking")
        assert graph.registry.get(chain.tools[0]).category == "banking"


def test_validate_catches_missing_id(graph):
    sampler = ChainSampler(graph)
    errors = sampler.validate(["search_flights", "cancel_booking"])
    assert any("booking_id" in e for e in errors)


def test_validate_catches_bad_start(graph):
    sampler = ChainSampler(graph)
    errors = sampler.validate(["book_flight"])
    assert any("cannot start" in e for e in errors)


def test_validate_accepts_good_chain(graph):
    sampler = ChainSampler(graph)
    assert sampler.validate(["search_flights", "book_flight", "cancel_booking"]) == []


def test_cannot_use_id_after_delete(graph):
    sampler = ChainSampler(graph)
    errors = sampler.validate(["create_event", "delete_event", "get_event"])
    assert any("event_id" in e for e in errors)


def test_every_tool_is_reachable(graph):
    sampler = ChainSampler(graph, seed=5)
    used = set()
    for _ in range(1000):
        used.update(sampler.sample(min_len=1, max_len=5).tools)
    assert used == set(graph.registry.names())
