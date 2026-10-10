from src.factory.models import ToolCall
from src.factory.quality.validators import ToolCallValidator
from src.factory.state.session import SessionState
from src.factory.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.from_directory()


def make_session():
    s = SessionState()
    s._register("flight_id", "flight_aaa111")
    s.entities["flight_aaa111"].update({"airline": "Delta", "price": 100.0})
    s._register("transaction_id", "transaction_abc123")
    return s


def check(name, session=None, allowed=None, **args):
    v = ToolCallValidator(REGISTRY, session or make_session())
    return [x.rule for x in v.validate(ToolCall(name=name, arguments=args), allowed)]


def test_valid_call_passes():
    assert check("search_flights", origin="BWI", destination="SFO", date="2026-12-18") == []


def test_unknown_tool():
    assert check("teleport", x=1) == ["unknown_tool"]


def test_tool_not_offered():
    rules = check("list_accounts", allowed={"search_flights"})
    assert "tool_not_offered" in rules


def test_missing_required():
    assert "missing_required" in check("search_flights", origin="BWI", destination="SFO")


def test_unexpected_argument():
    rules = check("get_flight_details", flight_id="flight_aaa111", seat="12A")
    assert "unexpected_argument" in rules


def test_wrong_type():
    rules = check("search_flights", origin="BWI", destination="SFO", date="2026-12-18", max_price="cheap")
    assert "wrong_type" in rules


def test_invalid_enum():
    assert "invalid_enum" in check("dispute_transaction", transaction_id="transaction_abc123", reason="angry")


def test_unknown_id():
    assert "unknown_id" in check("get_flight_details", flight_id="flight_madeup")


def test_deleted_id():
    session = make_session()
    session.mark_deleted("flight_aaa111")
    assert "deleted_id" in check("get_flight_details", session=session, flight_id="flight_aaa111")


def test_bad_date():
    assert "bad_date" in check("search_flights", origin="BWI", destination="SFO", date="next Friday")


def test_checkout_before_checkin():
    s = make_session()
    s._register("hotel_id", "hotel_xyz789")
    rules = check("book_hotel", session=s, hotel_id="hotel_xyz789", guest_name="A",
                  check_in="2027-02-15", check_out="2027-02-12", email="a@b.com")
    assert "date_order" in rules


def test_bad_email():
    rules = check("book_flight", flight_id="flight_aaa111", passenger_name="A", email="not-an-email")
    assert "bad_email" in rules


def test_non_positive_quantity():
    s = make_session()
    s._register("product_id", "product_aaa111")
    assert "not_positive" in check("add_to_cart", session=s, product_id="product_aaa111", quantity=0)


def test_duplicate_action_blocked_but_search_allowed():
    v = ToolCallValidator(REGISTRY, make_session())
    book = ToolCall(name="book_flight",
                    arguments={"flight_id": "flight_aaa111", "passenger_name": "A", "email": "a@b.com"})
    v.remember(book)
    assert "duplicate_action" in [x.rule for x in v.validate(book)]

    search = ToolCall(name="search_flights",
                      arguments={"origin": "BWI", "destination": "SFO", "date": "2026-12-18"})
    v.remember(search)
    assert v.validate(search) == []
