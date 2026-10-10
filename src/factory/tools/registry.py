"""
Loads tool definitions from JSON files and validates them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.factory.models import Tool

DEFAULT_TOOLS_DIR = Path(__file__).resolve().parents[3] / "data" / "tools"


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    # ---- loading -----------------------------------------------------------

    @classmethod
    def from_directory(cls, path: Path | str = DEFAULT_TOOLS_DIR) -> "ToolRegistry":
        path = Path(path)
        tools: list[Tool] = []
        for file in sorted(path.glob("*.json")):
            raw = json.loads(file.read_text())
            for item in raw:
                tools.append(Tool.model_validate(item))   # fails loudly if malformed
        if not tools:
            raise ValueError(f"No tools found in {path}")
        return cls(tools)

    # ---- lookups -----------------------------------------------------------

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def categories(self) -> list[str]:
        return sorted({t.category for t in self._tools.values()})

    def by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]

    def producers_of(self, field_name: str) -> list[Tool]:
        """Tools whose output includes this field (e.g. who gives a flight_id?)."""
        return [t for t in self._tools.values()
                if any(r.name == field_name for r in t.returns)]

    def consumers_of(self, field_name: str) -> list[Tool]:
        """Tools that take this field as input."""
        return [t for t in self._tools.values()
                if any(p.name == field_name for p in t.parameters)]

    def __len__(self) -> int:
        return len(self._tools)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for t in self._tools.values():
            counts[t.category] += 1
        return dict(counts)
