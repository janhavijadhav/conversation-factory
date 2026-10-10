"""
Samples structurally valid tool chains by walking the tool graph.

A chain is valid when:
  1. the first tool is an entry tool (needs no IDs),
  2. every consecutive pair is connected by a graph edge,
  3. every required ID input was produced by an earlier tool in the chain,
  4. no ID is used after the thing it points to was deleted,
  5. no tool repeats.

Optional `boosts` ({tool_name: multiplier}) let diversity steering favor
under-used tools without ever breaking rules 1-5.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from src.factory import config
from src.factory.graph.tool_graph import EdgeInfo, ToolGraph, is_id_param
from src.factory.models import OperationType, Tool

Field = tuple[str, str]   # (field name, type)


@dataclass
class SampledChain:
    tools: list[str]
    edges: list[EdgeInfo] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tools)

    @property
    def main_category(self) -> str:
        return Counter(self.categories).most_common(1)[0][0]

    def pretty(self) -> str:
        if not self.edges:
            return self.tools[0]
        parts = [self.tools[0]]
        for e in self.edges:
            label = ",".join(e.flow_fields) if e.flow_fields else e.types[0]
            parts.append(f"--[{label}]--> {e.target}")
        return " ".join(parts)


class ChainSampler:
    def __init__(self, graph: ToolGraph, seed: int | None = None):
        self.graph = graph
        self.registry = graph.registry
        self.rng = random.Random(seed)

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _outputs(tool: Tool) -> set[Field]:
        return {(r.name, r.type) for r in tool.returns}

    @staticmethod
    def missing_ids(tool: Tool, available: set[Field]) -> list[str]:
        """Required ID inputs of `tool` that aren't currently available."""
        return [
            p.name for p in tool.required_params()
            if is_id_param(p.name) and (p.name, p.type) not in available
        ]

    def _apply(self, tool: Tool, available: set[Field]) -> set[Field]:
        """Update the pool of usable fields after `tool` runs."""
        updated = available | self._outputs(tool)
        if tool.operation == OperationType.DELETE:
            deleted = {(p.name, p.type) for p in tool.required_params() if is_id_param(p.name)}
            updated -= deleted                     # deleted things can't be used again
        return updated

    # ---- sampling ----------------------------------------------------------

    def sample(
        self,
        min_len: int = config.MIN_CHAIN_LENGTH,
        max_len: int = config.MAX_CHAIN_LENGTH,
        category: str | None = None,
        start_tool: str | None = None,
        boosts: dict[str, float] | None = None,
        max_attempts: int = 25,
    ) -> SampledChain | None:
        """Try several walks; return the first that reaches min_len."""
        for _ in range(max_attempts):
            target_len = self.rng.randint(min_len, max(min_len, max_len))
            chain = self._walk(target_len, category, start_tool, boosts or {})
            if chain and len(chain) >= min_len:
                return chain
        return None

    def _walk(self, target_len: int, category: str | None, start_tool: str | None,
              boosts: dict[str, float]) -> SampledChain | None:
        entries = self.graph.entry_tools()
        if start_tool:
            entries = [start_tool] if start_tool in entries else []
        elif category:
            entries = [t for t in entries if self.registry.get(t).category == category]
        if not entries:
            return None

        current = self.rng.choices(entries, weights=[boosts.get(t, 1.0) for t in entries], k=1)[0]
        first = self.registry.get(current)
        tools = [current]
        edges: list[EdgeInfo] = []
        categories = [first.category]
        available = self._apply(first, set())

        while len(tools) < target_len:
            candidates = [
                e for e in self.graph.successors(current)
                if e.target not in tools
                and not self.missing_ids(self.registry.get(e.target), available)
            ]
            if not candidates:
                break                                        # dead end, stop early

            weights = [e.weight * boosts.get(e.target, 1.0) for e in candidates]
            edge = self.rng.choices(candidates, weights=weights, k=1)[0]
            nxt = self.registry.get(edge.target)
            tools.append(nxt.name)
            edges.append(edge)
            categories.append(nxt.category)
            available = self._apply(nxt, available)
            current = nxt.name

        return SampledChain(tools, edges, categories)

    # ---- validation (reused later by the quality system) -------------------

    def validate(self, tools: list[str]) -> list[str]:
        """Return a list of problems. Empty list = valid chain."""
        errors: list[str] = []
        if not tools:
            return ["empty chain"]

        for name in tools:
            if not self.registry.has(name):
                errors.append(f"unknown tool: {name}")
        if errors:
            return errors

        if len(set(tools)) != len(tools):
            errors.append("chain repeats a tool")

        if tools[0] not in self.graph.entry_tools():
            errors.append(f"{tools[0]} cannot start a conversation (needs an ID)")

        available: set[Field] = set()
        for i, name in enumerate(tools):
            tool = self.registry.get(name)
            if i > 0 and self.graph.edge(tools[i - 1], name) is None:
                errors.append(f"no edge {tools[i - 1]} -> {name}")
            for missing in self.missing_ids(tool, available):
                errors.append(f"{name} needs {missing} but it is not available (never produced or already deleted)")
            available = self._apply(tool, available)

        return errors
