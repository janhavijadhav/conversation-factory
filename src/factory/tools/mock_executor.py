"""
Mock Executor: pretends to be every tool.

- Rejects unknown tools, missing required args, and IDs that don't exist.
- Builds realistic fake outputs, reusing known values so the story stays consistent.
- Respects filter arguments like max_price and min_amount.
- Search tools return several results; the first one matches the query, like real search.
- Records everything into SessionState.
"""
from __future__ import annotations

import random
from typing import Any

from src.factory.graph.tool_graph import is_id_param
from src.factory.models import OperationType, Tool, ToolCall, ToolResult
from src.factory.state.session import SessionState
from src.factory.tools.registry import ToolRegistry

SEARCH_RESULTS = 3

AIRLINES = ["Delta", "United", "American", "JetBlue", "Southwest"]
HOTELS = ["Harbor View Inn", "The Grand Plaza", "Maple Suites", "Cityline Hotel"]
MERCHANTS = ["Amazon", "Whole Foods", "Shell", "Netflix", "Uber", "Target"]
EVENT_TITLES = ["Team sync", "Project review", "Dentist appointment", "Lunch with Sam"]
NAMES = ["Alex Kim", "Priya Shah", "Jordan Lee", "Maria Garcia"]
STATUS_BY_OP = {
    OperationType.CREATE: "confirmed",
    OperationType.UPDATE: "updated",
    OperationType.DELETE: "cancelled",
    OperationType.READ: "active",
    OperationType.SEARCH: "available",
}
PRICE_FIELDS = {"price", "price_per_night"}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MockExecutor:
    def __init__(self, registry: ToolRegistry, session: SessionState, seed: int | None = None):
        self.registry = registry
        self.session = session
        self.rng = random.Random(seed)

    # ---- public ------------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        if not self.registry.has(call.name):
            return self._fail(call, f"Unknown tool '{call.name}'")
        tool = self.registry.get(call.name)
        args = call.arguments

        for p in tool.required_params():
            if p.name not in args:
                return self._fail(call, f"Missing required argument '{p.name}'")

        for p in tool.parameters:
            if is_id_param(p.name) and p.name in args and not self.session.exists(args[p.name]):
                return self._fail(call, f"{p.name} '{args[p.name]}' not found")

        context = self._context(args)
        if tool.operation == OperationType.SEARCH:
            output = {"results": [self._build_item(tool, args, context, i) for i in range(SEARCH_RESULTS)]}
        else:
            output = self._build_item(tool, args, context)

        result = ToolResult(call_id=call.id, name=tool.name, success=True, output=output)
        self.session.record(tool, args, result)
        return result

    # ---- internals ---------------------------------------------------------

    def _fail(self, call: ToolCall, message: str) -> ToolResult:
        result = ToolResult(call_id=call.id, name=call.name, success=False, error=message)
        self.session.history.append(result)
        return result

    def _context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Known details of every entity the arguments point to."""
        ctx: dict[str, Any] = {}
        for k, v in args.items():
            if is_id_param(k) and self.session.exists(v):
                ctx.update(self.session.get(v))
        return ctx

    def _build_item(self, tool: Tool, args: dict[str, Any], ctx: dict[str, Any],
                    index: int = 0) -> dict[str, Any]:
        item = {}
        for r in tool.returns:
            if r.name == "status":
                item[r.name] = ctx.get("status", "active") if tool.operation == OperationType.READ \
                    else STATUS_BY_OP[tool.operation]
            elif r.name in args:                         # echo what was passed in
                item[r.name] = args[r.name]
            elif r.name in ctx:                          # reuse known values
                item[r.name] = ctx[r.name]
            else:
                item[r.name] = self._generate(r.name, r.type, args, ctx, index)
        return item

    def _generate(self, field: str, ftype: str, args: dict[str, Any], ctx: dict[str, Any],
                  index: int = 0) -> Any:
        rng = self.rng

        if is_id_param(field):
            return f"{field[:-3]}_{rng.getrandbits(24):06x}"

        if ftype == "boolean":
            return True

        if ftype == "integer":
            if field == "item_count":
                return args.get("quantity", rng.randint(1, 4))
            return rng.randint(1, 50)

        if ftype == "number":
            # respect filters the user asked for
            max_price = _as_float(args.get("max_price"))
            if field in PRICE_FIELDS and max_price and max_price > 0:
                return round(rng.uniform(max_price * 0.4, max_price), 2)
            min_amount = _as_float(args.get("min_amount"))
            if field == "amount" and min_amount is not None:
                return round(rng.uniform(min_amount, min_amount * 4 + 20), 2)

            if field == "total_price" and "price" in ctx:
                return ctx["price"]
            if "refund" in field:
                for key in ("total_price", "total", "price", "amount"):
                    if key in ctx:
                        return ctx[key]
            if field == "balance":
                return round(rng.uniform(500, 8000), 2)
            return round(rng.uniform(20, 900), 2)

        # strings
        date = args.get("date") or args.get("check_in") or "2026-11-15"
        query = str(args.get("query", "")).strip()
        if field == "title":
            if index == 0 and query:                     # best match first, like real search
                return query[:1].upper() + query[1:]
            return rng.choice(EVENT_TITLES)
        if field == "airline":
            return rng.choice(AIRLINES)
        if field == "hotel_name":
            return rng.choice(HOTELS)
        if field == "merchant":
            return rng.choice(MERCHANTS)
        if field == "product_name":
            base = (query or "item").title()
            return f"{base} {rng.choice(['Pro', 'Lite', 'Max', 'Classic'])}"
        if field in ("departure_time", "start_time"):
            return f"{date}T{rng.randint(6, 21):02d}:{rng.choice(['00', '15', '30', '45'])}:00"
        if field == "estimated_delivery":
            return "2026-10-20"
        if field == "account_type":
            return rng.choice(["checking", "savings"])
        if field == "email":
            return "alex.kim@example.com"
        if field.endswith("name"):
            return rng.choice(NAMES)
        return f"{field}_value"
