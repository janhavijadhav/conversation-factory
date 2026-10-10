"""
Quality level 1: pre-execution validators.

Runs after grounding, before the executor. Blocks any tool call that breaks
the tool's schema, refers to dead IDs, has badly formatted values, or repeats
an action that already happened.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from src.factory.graph.tool_graph import is_id_param
from src.factory.models import OperationType, Tool, ToolCall
from src.factory.state.session import SessionState
from src.factory.tools.registry import ToolRegistry

DATE_PARAMS = {"date", "check_in", "check_out"}
DATETIME_PARAMS = {"start_time"}
EMAIL_PARAMS = {"email"}
POSITIVE_PARAMS = {"amount", "max_price", "quantity", "duration_minutes"}
NON_NEGATIVE_PARAMS = {"min_amount"}
STATEFUL_OPS = {OperationType.CREATE, OperationType.UPDATE, OperationType.DELETE}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


@dataclass
class Violation:
    tool: str
    param: str | None
    rule: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class ToolCallValidator:
    def __init__(self, registry: ToolRegistry, session: SessionState):
        self.registry = registry
        self.session = session
        self._completed: set[str] = set()   # successful stateful calls, to catch duplicates

    # ---- public ------------------------------------------------------------

    @staticmethod
    def _key(call: ToolCall) -> str:
        return call.name + "|" + json.dumps(call.arguments, sort_keys=True, default=str)

    def remember(self, call: ToolCall) -> None:
        """Call after a successful execution."""
        self._completed.add(self._key(call))

    def validate(self, call: ToolCall, allowed_tools: set[str] | None = None) -> list[Violation]:
        if not self.registry.has(call.name):
            return [Violation(call.name, None, "unknown_tool", f"tool '{call.name}' does not exist")]

        out: list[Violation] = []
        tool = self.registry.get(call.name)
        args = call.arguments

        if allowed_tools is not None and call.name not in allowed_tools:
            out.append(Violation(tool.name, None, "tool_not_offered",
                                 f"tool '{call.name}' was not available in this conversation"))

        for p in tool.required_params():
            if p.name not in args:
                out.append(Violation(tool.name, p.name, "missing_required",
                                     f"missing required argument '{p.name}'"))

        for key in args:
            if tool.param(key) is None:
                out.append(Violation(tool.name, key, "unexpected_argument",
                                     f"'{key}' is not an argument of {tool.name}"))

        for p in tool.parameters:
            if p.name not in args:
                continue
            value = args[p.name]
            if not TYPE_CHECKS[p.type](value):
                out.append(Violation(tool.name, p.name, "wrong_type",
                                     f"'{p.name}' should be {p.type}, got {type(value).__name__}"))
                continue
            if p.enum and value not in p.enum:
                out.append(Violation(tool.name, p.name, "invalid_enum",
                                     f"'{p.name}' must be one of {p.enum}, got {value!r}"))
            if is_id_param(p.name):
                self._check_id(tool, p.name, value, out)
            self._check_format(tool, p.name, value, out)

        self._check_date_order(tool, args, out)

        if tool.operation in STATEFUL_OPS and self._key(call) in self._completed:
            out.append(Violation(tool.name, None, "duplicate_action",
                                 f"identical {tool.name} call already succeeded"))
        return out

    # ---- rules -------------------------------------------------------------

    def _check_id(self, tool: Tool, name: str, value: Any, out: list[Violation]) -> None:
        if self.session.exists(value):
            return
        if value in self.session.deleted:
            out.append(Violation(tool.name, name, "deleted_id",
                                 f"{name} '{value}' refers to something that was deleted"))
        else:
            out.append(Violation(tool.name, name, "unknown_id",
                                 f"{name} '{value}' was never returned by any tool"))

    @staticmethod
    def _check_format(tool: Tool, name: str, value: Any, out: list[Violation]) -> None:
        if name in DATE_PARAMS:
            ok = isinstance(value, str) and bool(DATE_RE.match(value))
            if ok:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    ok = False
            if not ok:
                out.append(Violation(tool.name, name, "bad_date", f"'{name}' must be YYYY-MM-DD, got {value!r}"))

        elif name in DATETIME_PARAMS:
            try:
                datetime.fromisoformat(str(value))
            except ValueError:
                out.append(Violation(tool.name, name, "bad_datetime",
                                     f"'{name}' must be ISO datetime, got {value!r}"))

        elif name in EMAIL_PARAMS and not EMAIL_RE.match(str(value)):
            out.append(Violation(tool.name, name, "bad_email", f"'{name}' is not a valid email: {value!r}"))

        elif name in POSITIVE_PARAMS and isinstance(value, (int, float)) and value <= 0:
            out.append(Violation(tool.name, name, "not_positive", f"'{name}' must be greater than 0"))

        elif name in NON_NEGATIVE_PARAMS and isinstance(value, (int, float)) and value < 0:
            out.append(Violation(tool.name, name, "negative", f"'{name}' cannot be negative"))

    @staticmethod
    def _check_date_order(tool: Tool, args: dict, out: list[Violation]) -> None:
        if "check_in" in args and "check_out" in args:
            try:
                if date.fromisoformat(args["check_out"]) <= date.fromisoformat(args["check_in"]):
                    out.append(Violation(tool.name, "check_out", "date_order",
                                         "check_out must be after check_in"))
            except (TypeError, ValueError):
                pass   # already reported as bad_date
