from src.factory.models import ToolCall
from src.factory.state.grounding import ConversationGrounding
from src.factory.state.session import SessionState
from src.factory.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.from_directory()


def make_session(flights=(("flight_aaa111", "Delta"), ("flight_bbb222", "United"))):
    s = SessionState()
    for fid, airline in flights:
        s._register("flight_id", fid)
        s.entities[fid].update({"airline": airline, "price": 100.0})
    return s


def ground(session, name, **args):
    return ConversationGrounding(REGISTRY, session).ground(ToolCall(name=name, arguments=args))


def test_correct_call_needs_no_fixes():
    call, fixes = ground(make_session(), "get_flight_details", flight_id="flight_aaa111")
    assert fixes == []
    assert call.arguments["flight_id"] == "flight_aaa111"


def test_name_instead_of_id():
    call, fixes = ground(make_session(), "get_flight_details", flight_id="Delta")
    assert call.arguments["flight_id"] == "flight_aaa111"
    assert "matched description" in fixes[0].reason


def test_placeholder_replaced_with_most_recent():
    call, fixes = ground(make_session(), "get_flight_details", flight_id="<flight_id>")
    assert call.arguments["flight_id"] == "flight_bbb222"
    assert "ambiguous" in fixes[0].reason


def test_unknown_id_with_single_option():
    session = make_session(flights=(("flight_only01", "Delta"),))
    call, fixes = ground(session, "get_flight_details", flight_id="flight_madeup")
    assert call.arguments["flight_id"] == "flight_only01"


def test_unknown_id_with_several_options_left_alone():
    call, fixes = ground(make_session(), "get_flight_details", flight_id="flight_zzz999")
    assert call.arguments["flight_id"] == "flight_zzz999"
    assert fixes == []


def test_deleted_id_left_alone():
    session = make_session()
    session.mark_deleted("flight_aaa111")
    call, fixes = ground(session, "get_flight_details", flight_id="flight_aaa111")
    assert call.arguments["flight_id"] == "flight_aaa111"
    assert fixes == []


def test_missing_id_filled_when_only_one():
    session = make_session(flights=(("flight_only01", "Delta"),))
    call, fixes = ground(session, "book_flight", passenger_name="A", email="a@b.com")
    assert call.arguments["flight_id"] == "flight_only01"


def test_type_coercion():
    call, fixes = ground(SessionState(), "search_flights",
                         origin="BWI", destination="SFO", date="2026-12-18", max_price="$350")
    assert call.arguments["max_price"] == 350.0


def test_enum_normalization():
    session = SessionState()
    session._register("transaction_id", "transaction_abc123")
    session.entities["transaction_abc123"].update({"merchant": "Amazon"})
    call, _ = ground(session, "dispute_transaction",
                     transaction_id="transaction_abc123", reason="Not Received")
    assert call.arguments["reason"] == "not_received"
