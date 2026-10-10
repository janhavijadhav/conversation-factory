from src.factory.models import Conversation, Message, Scenario, ToolCall
from src.factory.quality.calibration import (
    false_action, invented_detail, missing_step, record_to_conversation,
    unhelpful_ending, wrong_id_argument, wrong_number,
)


def make_conv():
    search = ToolCall(id="c1", name="search_flights",
                      arguments={"origin": "BWI", "destination": "SFO", "date": "2026-12-18"})
    book = ToolCall(id="c2", name="book_flight",
                    arguments={"flight_id": "flight_aaa111", "passenger_name": "A", "email": "a@b.com"})
    return Conversation(
        scenario=Scenario(user_goal="book a flight", tool_chain=["search_flights", "book_flight"],
                          category="travel"),
        messages=[
            Message(role="user", content="Book me the cheapest flight BWI to SFO on Dec 18"),
            Message(role="assistant", tool_calls=[search]),
            Message(role="tool", tool_call_id="c1", name="search_flights",
                    content='{"results": [{"flight_id": "flight_aaa111", "price": 226.21}]}'),
            Message(role="assistant", tool_calls=[book]),
            Message(role="tool", tool_call_id="c2", name="book_flight",
                    content='{"booking_id": "booking_bbb222", "total_price": 226.21}'),
            Message(role="assistant", content="Booked! Your total is $226.21."),
        ],
    )


def test_wrong_number():
    conv, desc = wrong_number(make_conv())
    assert "309.91" in conv.messages[-1].content
    assert "226.21" in desc


def test_false_action():
    conv, desc = false_action(make_conv())
    assert "emailed you" in conv.messages[-1].content


def test_invented_detail():
    conv, _ = invented_detail(make_conv())
    assert "15% loyalty discount" in conv.messages[-1].content


def test_wrong_id_argument():
    conv, _ = wrong_id_argument(make_conv())
    assert conv.tool_calls[1].arguments["flight_id"] == "flight_ffffff"


def test_missing_step():
    conv, desc = missing_step(make_conv())
    assert conv.num_tool_calls == 1
    assert all(m.tool_call_id != "c2" for m in conv.messages)
    assert conv.messages[-1].content == "Booked! Your total is $226.21."


def test_unhelpful_ending():
    conv, _ = unhelpful_ending(make_conv())
    assert conv.messages[-1].content == "Okay, noted."


def test_original_is_never_modified():
    original = make_conv()
    for fn in (wrong_number, false_action, invented_detail, wrong_id_argument, missing_step, unhelpful_ending):
        fn(original)
    assert original.messages[-1].content == "Booked! Your total is $226.21."
    assert original.tool_calls[1].arguments["flight_id"] == "flight_aaa111"
    assert original.num_tool_calls == 2


def test_record_round_trip():
    record = make_conv().to_training_record()
    rebuilt = record_to_conversation(record)
    assert rebuilt is not None and rebuilt.num_tool_calls == 2
