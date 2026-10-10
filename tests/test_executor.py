import pytest

from src.factory.models import ToolCall
from src.factory.state.session import SessionState
from src.factory.tools.mock_executor import MockExecutor
from src.factory.tools.registry import ToolRegistry


@pytest.fixture
def setup():
    registry = ToolRegistry.from_directory()
    session = SessionState()
    executor = MockExecutor(registry, session, seed=0)
    return executor, session


def run(executor, name, **args):
    return executor.execute(ToolCall(name=name, arguments=args))


def test_search_returns_multiple_results(setup):
    executor, session = setup
    r = run(executor, "search_flights", origin="BWI", destination="JFK", date="2026-11-20")
    assert r.success
    assert len(r.output["results"]) == 3
    assert len(session.live_ids("flight_id")) == 3


def test_values_stay_consistent(setup):
    executor, session = setup
    flight = run(executor, "search_flights", origin="BWI", destination="JFK", date="2026-11-20").output["results"][0]

    details = run(executor, "get_flight_details", flight_id=flight["flight_id"]).output
    assert details["price"] == flight["price"]
    assert details["airline"] == flight["airline"]

    booking = run(executor, "book_flight", flight_id=flight["flight_id"],
                  passenger_name="Janhavi J", email="j@example.com").output
    assert booking["total_price"] == flight["price"]

    lookup = run(executor, "get_booking", booking_id=booking["booking_id"]).output
    assert lookup["flight_id"] == flight["flight_id"]


def test_fake_id_rejected(setup):
    executor, _ = setup
    r = run(executor, "book_flight", flight_id="flight_madeup", passenger_name="A", email="a@b.com")
    assert not r.success
    assert "not found" in r.error


def test_missing_argument_rejected(setup):
    executor, _ = setup
    flight = run(executor, "search_flights", origin="BWI", destination="JFK", date="2026-11-20").output["results"][0]
    r = run(executor, "book_flight", flight_id=flight["flight_id"], email="a@b.com")
    assert not r.success
    assert "passenger_name" in r.error


def test_delete_sticks(setup):
    executor, session = setup
    flight = run(executor, "search_flights", origin="BWI", destination="JFK", date="2026-11-20").output["results"][0]
    booking = run(executor, "book_flight", flight_id=flight["flight_id"],
                  passenger_name="A", email="a@b.com").output
    cancel = run(executor, "cancel_booking", booking_id=booking["booking_id"])
    assert cancel.success
    assert cancel.output["refund_amount"] == booking["total_price"]

    assert not session.exists(booking["booking_id"])
    again = run(executor, "get_booking", booking_id=booking["booking_id"])
    assert not again.success


def test_update_persists(setup):
    executor, _ = setup
    event = run(executor, "create_event", title="Standup", start_time="2026-11-01T10:00:00",
                duration_minutes=15).output
    run(executor, "update_event", event_id=event["event_id"], start_time="2026-11-02T15:00:00")
    fetched = run(executor, "get_event", event_id=event["event_id"]).output
    assert fetched["start_time"] == "2026-11-02T15:00:00"
    assert fetched["title"] == "Standup"


def test_same_seed_same_output():
    registry = ToolRegistry.from_directory()
    a = MockExecutor(registry, SessionState(), seed=9)
    b = MockExecutor(registry, SessionState(), seed=9)
    args = dict(origin="BWI", destination="JFK", date="2026-11-20")
    assert run(a, "search_flights", **args).output == run(b, "search_flights", **args).output


def test_max_price_respected(setup):
    executor, _ = setup
    hotels = run(executor, "search_hotels", city="Denver", check_in="2026-12-15",
                 check_out="2026-12-18", max_price=200).output["results"]
    assert all(h["price_per_night"] <= 200 for h in hotels)

    flights = run(executor, "search_flights", origin="DEN", destination="SFO",
                  date="2026-12-18", max_price=350).output["results"]
    assert all(f["price"] <= 350 for f in flights)


def test_min_amount_respected(setup):
    executor, session = setup
    run(executor, "list_accounts")
    account = session.latest("account_id")
    txns = run(executor, "search_transactions", account_id=account, min_amount=100).output["results"]
    assert all(t["amount"] >= 100 for t in txns)
