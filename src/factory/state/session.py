"""
SessionState: the "world" a conversation lives in.

Tracks every entity (flight, booking, order, event...) that tools have
produced, keyed by its ID for O(1) lookups, plus which IDs were deleted.
Kept separate from the message history on purpose.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.factory.graph.tool_graph import is_id_param
from src.factory.models import OperationType, Tool, ToolResult


class SessionState:
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}          # id value -> attributes
        self.ids_by_field: dict[str, list[str]] = defaultdict(list)  # "flight_id" -> [ids]
        self.deleted: set[str] = set()
        self.history: list[ToolResult] = []

    # ---- lookups (all O(1) or close) ---------------------------------------

    def exists(self, id_value: Any) -> bool:
        """Is this ID known and not deleted?"""
        return isinstance(id_value, str) and id_value in self.entities and id_value not in self.deleted

    def get(self, id_value: str) -> dict[str, Any] | None:
        return self.entities.get(id_value) if self.exists(id_value) else None

    def live_ids(self, field: str) -> list[str]:
        return [v for v in self.ids_by_field.get(field, []) if v not in self.deleted]

    def latest(self, field: str) -> str | None:
        live = self.live_ids(field)
        return live[-1] if live else None

    # ---- updates -----------------------------------------------------------

    def _register(self, field: str, value: str) -> None:
        if value not in self.ids_by_field[field]:
            self.ids_by_field[field].append(value)
        self.entities.setdefault(value, {})

    def mark_deleted(self, id_value: str) -> None:
        self.deleted.add(id_value)

    def record(self, tool: Tool, args: dict[str, Any], result: ToolResult) -> None:
        """Update the world after a tool runs."""
        self.history.append(result)
        if not result.success:
            return

        items = result.output.get("results", [result.output])
        id_fields = [r.name for r in tool.returns if is_id_param(r.name)]

        for item in items:
            for f in id_fields:
                if f in item:
                    self._register(f, item[f])
            # attach details to the item's main ID (first ID field the tool returns)
            if id_fields and id_fields[0] in item:
                main_id = item[id_fields[0]]
                self.entities[main_id].update({**args, **item})

        if tool.operation == OperationType.DELETE:
            for p in tool.required_params():
                if is_id_param(p.name) and p.name in args:
                    self.mark_deleted(args[p.name])

    # ---- for prompts later -------------------------------------------------

    def describe(self, max_lines: int = 20) -> str:
        """Readable list of live entities, used to ground the assistant later."""
        lines = []
        for field, values in self.ids_by_field.items():
            for v in values:
                if v in self.deleted:
                    continue
                attrs = {k: val for k, val in self.entities.get(v, {}).items()
                         if k != field and not is_id_param(k)}
                lines.append(f"{field}={v} {attrs}")
        return "\n".join(lines[-max_lines:]) or "(nothing yet)"
