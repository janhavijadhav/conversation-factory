"""
ConversationGrounding: fixes the assistant's tool arguments before they run.

Safe fixes only:
  - type coercion        "2" -> 2, "350" -> 350.0
  - enum normalization   "Not Received" -> "not_received"
  - placeholder IDs      "<flight_id>" -> a real flight_id from SessionState
  - name instead of ID   "The Grand Plaza" -> that hotel's hotel_id
  - unknown ID when exactly one real option exists
  - missing required ID when exactly one real option exists

Never "fixes" an ID that was deleted, or a made-up ID when several real
options exist: those are real mistakes and should fail visibly.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.factory.graph.tool_graph import is_id_param
from src.factory.models import Tool, ToolCall, ToolParameter
from src.factory.state.session import SessionState
from src.factory.tools.registry import ToolRegistry

PLACEHOLDER_PATTERN = re.compile(r"^\s*(<.*>|\{.*\}|\[.*\]|\$\{.*\})\s*$")
PLACEHOLDER_WORDS = {"", "id", "unknown", "none", "null", "n/a", "tbd", "placeholder", "xxx", "?"}
SKIP_MATCH_FIELDS = {"status"}


@dataclass
class GroundingFix:
    tool: str
    param: str
    old: Any
    new: Any
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class ConversationGrounding:
    def __init__(self, registry: ToolRegistry, session: SessionState):
        self.registry = registry
        self.session = session

    # ---- public ------------------------------------------------------------

    def ground(self, call: ToolCall) -> tuple[ToolCall, list[GroundingFix]]:
        if not self.registry.has(call.name):
            return call, []
        tool = self.registry.get(call.name)
        args = dict(call.arguments)
        fixes: list[GroundingFix] = []

        for p in tool.parameters:
            if p.name not in args:
                self._fill_missing_id(tool, p, args, fixes)
                continue

            value = args[p.name]
            new_value, reason = self._coerce(p, value)
            if reason:
                fixes.append(GroundingFix(tool.name, p.name, value, new_value, reason))
                args[p.name] = value = new_value

            if is_id_param(p.name):
                new_value, reason = self._resolve_id(p.name, value)
                if reason:
                    fixes.append(GroundingFix(tool.name, p.name, value, new_value, reason))
                    args[p.name] = new_value

        return call.model_copy(update={"arguments": args}), fixes

    # ---- missing IDs -------------------------------------------------------

    def _fill_missing_id(self, tool: Tool, p: ToolParameter, args: dict, fixes: list) -> None:
        if not (p.required and is_id_param(p.name)):
            return
        live = self.session.live_ids(p.name)
        if len(live) == 1:
            args[p.name] = live[0]
            fixes.append(GroundingFix(tool.name, p.name, None, live[0],
                                      "filled missing ID (only one available)"))

    # ---- types and enums ---------------------------------------------------

    @staticmethod
    def _coerce(p: ToolParameter, value: Any) -> tuple[Any, str | None]:
        if p.type == "integer" and isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip()), "string -> integer"
        if p.type == "integer" and isinstance(value, float) and value.is_integer():
            return int(value), "float -> integer"
        if p.type == "number" and isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            try:
                return float(cleaned), "string -> number"
            except ValueError:
                return value, None
        if p.type == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true", "string -> boolean"
        if p.type == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), "number -> string"

        if p.enum and isinstance(value, str) and value not in p.enum:
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            for option in p.enum:
                if str(option).lower() == normalized:
                    return option, "normalized enum value"
        return value, None

    # ---- IDs ---------------------------------------------------------------

    def _resolve_id(self, field: str, value: Any) -> tuple[Any, str | None]:
        if self.session.exists(value):
            return value, None                       # already correct
        if isinstance(value, str) and value in self.session.deleted:
            return value, None                       # real mistake: let it fail

        live = self.session.live_ids(field)
        if not live:
            return value, None

        text = str(value).strip()

        matches = [i for i in live if self._mentions(i, text)]
        if len(matches) == 1:
            return matches[0], f"matched description '{text}'"

        if len(live) == 1:
            return live[0], "unknown ID replaced (only one available)"

        if self._is_placeholder(text, field):
            return live[-1], "placeholder replaced with most recent (ambiguous)"

        return value, None                           # several options, can't be sure

    def _mentions(self, id_value: str, text: str) -> bool:
        """Does `text` describe this entity (e.g. its name)?"""
        if len(text) < 3:
            return False
        t = text.lower()
        for key, v in (self.session.get(id_value) or {}).items():
            if key in SKIP_MATCH_FIELDS or is_id_param(key) or not isinstance(v, str):
                continue
            s = v.lower()
            if t == s or (len(s) >= 4 and s in t) or (len(t) >= 4 and t in s):
                return True
        return False

    @staticmethod
    def _is_placeholder(text: str, field: str) -> bool:
        low = text.lower()
        return (
            bool(PLACEHOLDER_PATTERN.match(text))
            or low in PLACEHOLDER_WORDS
            or low in {field, field.replace("_id", ""), field.replace("_", "")}
            or "placeholder" in low
            or (text.isupper() and "_" in text)
        )
