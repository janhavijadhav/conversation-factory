"""
Directed tool graph.

Nodes  = tools.
Edges  = "tool B makes sense after tool A", with 4 edge types:
    data_flow       A returns a field that B requires         (strongest)
    shared_param    A and B take the same parameter
    operation_rank  same category, B is a later operation
    category        same category                             (weakest)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

from src.factory.models import Tool
from src.factory.tools.registry import ToolRegistry

EDGE_WEIGHTS = {
    "data_flow": 3.0,
    "shared_param": 1.5,
    "operation_rank": 1.0,
    "category": 0.5,
}

# Parameters so common they don't signal a real connection between categories.
# "date" in flights and "date" in calendar events doesn't mean they belong together.
GENERIC_PARAMS = {"query", "date", "max_price", "min_amount"}


def is_id_param(name: str) -> bool:
    """IDs are values only a tool can produce, never something a user just knows."""
    return name.endswith("_id")


@dataclass
class EdgeInfo:
    source: str
    target: str
    types: list[str]
    weight: float
    flow_fields: list[str] = field(default_factory=list)    # fields passed A -> B
    shared_params: list[str] = field(default_factory=list)


class ToolGraph:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.g = nx.DiGraph()
        self._build()

    # ---- building ----------------------------------------------------------

    def _build(self) -> None:
        tools = self.registry.all()
        for t in tools:
            self.g.add_node(
                t.name,
                category=t.category,
                operation=t.operation.value,
                rank=t.operation.rank,
            )

        for a in tools:
            for b in tools:
                if a.name == b.name:
                    continue                      # no self-loops
                info = self._compute_edge(a, b)
                if info.types:
                    self.g.add_edge(
                        a.name, b.name,
                        types=info.types,
                        weight=info.weight,
                        flow_fields=info.flow_fields,
                        shared_params=info.shared_params,
                    )

    @staticmethod
    def _data_flow_fields(a: Tool, b: Tool) -> list[str]:
        """Required inputs of B that A's output provides (same name AND same type)."""
        a_returns = {(r.name, r.type) for r in a.returns}
        return sorted(p.name for p in b.required_params() if (p.name, p.type) in a_returns)

    def _compute_edge(self, a: Tool, b: Tool) -> EdgeInfo:
        types: list[str] = []
        a_rank, b_rank = a.operation.rank, b.operation.rank
        same_category = a.category == b.category

        # 1. data flow
        flow = self._data_flow_fields(a, b)
        if flow:
            types.append("data_flow")

        # 2. shared parameters (don't go "backwards" in operation order)
        shared = {p.name for p in a.parameters} & {p.name for p in b.parameters}
        if not same_category:
            shared -= GENERIC_PARAMS            # prune noisy cross-category links
        shared = sorted(shared)
        if shared and a_rank <= b_rank:
            types.append("shared_param")

        # 3. operation rank inside a category
        if same_category and a_rank < b_rank:
            types.append("operation_rank")

        # 4. same category
        if same_category:
            types.append("category")

        weight = sum(EDGE_WEIGHTS[t] for t in types)
        return EdgeInfo(a.name, b.name, types, weight, flow, shared)

    # ---- queries -----------------------------------------------------------

    def entry_tools(self) -> list[str]:
        """Tools a conversation can start with: no required ID inputs."""
        return sorted(
            t.name for t in self.registry.all()
            if not any(is_id_param(p.name) for p in t.required_params())
        )

    def successors(self, name: str) -> list[EdgeInfo]:
        """All tools reachable in one step from `name`, strongest first."""
        out = []
        for _, target, data in self.g.out_edges(name, data=True):
            out.append(EdgeInfo(
                name, target, data["types"], data["weight"],
                data["flow_fields"], data["shared_params"],
            ))
        return sorted(out, key=lambda e: e.weight, reverse=True)

    def edge(self, a: str, b: str) -> EdgeInfo | None:
        if not self.g.has_edge(a, b):
            return None
        d = self.g.edges[a, b]
        return EdgeInfo(a, b, d["types"], d["weight"], d["flow_fields"], d["shared_params"])

    def stats(self) -> dict:
        type_counts: Counter = Counter()
        cross_category = 0
        for a, b, d in self.g.edges(data=True):
            type_counts.update(d["types"])
            if self.g.nodes[a]["category"] != self.g.nodes[b]["category"]:
                cross_category += 1
        n = self.g.number_of_nodes()
        e = self.g.number_of_edges()
        return {
            "nodes": n,
            "edges": e,
            "avg_out_degree": round(e / n, 2) if n else 0,
            "edge_types": dict(type_counts),
            "cross_category_edges": cross_category,
            "entry_tools": len(self.entry_tools()),
        }
