import pytest

from src.factory.graph.tool_graph import ToolGraph, is_id_param
from src.factory.tools.registry import ToolRegistry


@pytest.fixture(scope="module")
def graph():
    return ToolGraph(ToolRegistry.from_directory())


def test_all_tools_are_nodes(graph):
    assert graph.g.number_of_nodes() == 26


def test_data_flow_edge(graph):
    e = graph.edge("search_flights", "book_flight")
    assert e is not None
    assert "data_flow" in e.types
    assert e.flow_fields == ["flight_id"]


def test_operation_rank_edge(graph):
    e = graph.edge("get_booking", "cancel_booking")
    assert "operation_rank" in e.types


def test_cross_category_shared_param(graph):
    e = graph.edge("book_flight", "send_email")
    assert e is not None
    assert "shared_param" in e.types
    assert "email" in e.shared_params


def test_generic_params_dont_link_categories(graph):
    assert graph.edge("search_flights", "search_events") is None
    assert graph.edge("search_flights", "search_products") is None


def test_no_self_loops(graph):
    assert all(a != b for a, b in graph.g.edges())


def test_entry_tools_need_no_ids(graph):
    entries = graph.entry_tools()
    assert "search_flights" in entries
    assert "list_accounts" in entries
    assert "book_flight" not in entries      # needs flight_id
    assert "cancel_order" not in entries     # needs order_id


def test_successors_sorted_by_weight(graph):
    succ = graph.successors("search_flights")
    weights = [e.weight for e in succ]
    assert weights == sorted(weights, reverse=True)
    assert succ[0].target in {"book_flight", "get_flight_details"}


def test_is_id_param():
    assert is_id_param("booking_id")
    assert not is_id_param("email")
