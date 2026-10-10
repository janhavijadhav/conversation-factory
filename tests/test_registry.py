import pytest

from src.factory.models import Tool
from src.factory.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry.from_directory()


def test_loads_all_tools(registry):
    assert len(registry) == 26
    assert registry.categories() == ["banking", "calendar", "ecommerce", "travel"]


def test_lookup(registry):
    tool = registry.get("book_flight")
    assert tool.category == "travel"
    assert tool.operation.value == "create"


def test_unknown_tool_raises(registry):
    with pytest.raises(KeyError):
        registry.get("teleport_user")


def test_producers_and_consumers(registry):
    producers = {t.name for t in registry.producers_of("flight_id")}
    consumers = {t.name for t in registry.consumers_of("flight_id")}
    assert "search_flights" in producers
    assert "book_flight" in consumers


def test_duplicate_names_rejected(registry):
    tool = registry.get("search_flights")
    with pytest.raises(ValueError):
        ToolRegistry([tool, tool])
