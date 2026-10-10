import pytest
from pydantic import ValidationError

from src.factory.models import (
    Conversation, Message, OperationType, Scenario, Tool,
    ToolCall, ToolParameter, ReturnField,
)


def make_tool():
    return Tool(
        name="search_flights",
        category="travel",
        description="Find flights between two cities",
        operation=OperationType.SEARCH,
        parameters=[
            ToolParameter(name="origin", type="string"),
            ToolParameter(name="destination", type="string"),
            ToolParameter(name="max_price", type="number", required=False),
        ],
        returns=[ReturnField(name="flight_id", type="string")],
    )


def test_tool_builds():
    tool = make_tool()
    assert len(tool.required_params()) == 2
    assert tool.operation.rank == 0


def test_openai_schema():
    schema = make_tool().to_openai_schema()
    assert schema["function"]["name"] == "search_flights"
    assert schema["function"]["parameters"]["required"] == ["origin", "destination"]


def test_bad_param_type_rejected():
    with pytest.raises(ValidationError):
        ToolParameter(name="x", type="banana")


def test_conversation_counts_tool_calls():
    conv = Conversation(
        scenario=Scenario(user_goal="book a flight", tool_chain=["search_flights"], category="travel"),
        messages=[
            Message(role="user", content="Find me a flight to NYC"),
            Message(role="assistant", tool_calls=[ToolCall(name="search_flights", arguments={"origin": "BWI", "destination": "NYC"})]),
        ],
    )
    assert conv.num_tool_calls == 1
    assert conv.is_multi_step is False
    assert conv.to_training_record()["category"] == "travel"
