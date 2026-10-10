"""Walk through a sampled chain by hand, using SessionState to fill in real IDs."""
import json

from src.factory.models import ToolCall
from src.factory.state.session import SessionState
from src.factory.tools.mock_executor import MockExecutor
from src.factory.tools.registry import ToolRegistry

registry = ToolRegistry.from_directory()
session = SessionState()
executor = MockExecutor(registry, session, seed=3)


def call(name, **args):
    result = executor.execute(ToolCall(name=name, arguments=args))
    status = "OK " if result.success else "ERR"
    print(f"\n[{status}] {name}({args})")
    print(json.dumps(result.output if result.success else result.error, indent=2))
    return result


call("search_flights", origin="BWI", destination="SFO", date="2026-12-18")
flight_id = session.latest("flight_id")

call("book_flight", flight_id=flight_id, passenger_name="Janhavi Jadhav", email="janhavi@example.com")
booking_id = session.latest("booking_id")

call("send_email", email="janhavi@example.com", subject="Your flight", body=f"Booking {booking_id} confirmed")
call("cancel_booking", booking_id=booking_id)
call("get_booking", booking_id=booking_id)          # should fail: it was cancelled
call("book_flight", flight_id="flight_fake99", passenger_name="X", email="x@y.com")  # should fail

print("\n--- What the world looks like now ---")
print(session.describe())
