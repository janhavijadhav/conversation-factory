"""
Diversity steering.

The tracker counts what has been generated so far. Before each batch it
produces a SteeringPlan that pushes the next batch toward what's missing:
under-used categories, under-used tools, longer chains, and new names.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.factory import config
from src.factory.models import Conversation
from src.factory.tools.registry import ToolRegistry


def persona_name(persona: str) -> str:
    """'Tomás Herrera, 52, teacher' -> 'Tomás Herrera'. Empty if it doesn't look like a name."""
    head = (persona or "").split(",")[0].strip()
    words = head.split()
    if 2 <= len(words) <= 4 and all(w[:1].isupper() for w in words):
        return " ".join(words)
    return ""


@dataclass
class SteeringPlan:
    categories: list[str]                  # least-used first
    min_len: int
    tool_boosts: dict[str, float] = field(default_factory=dict)
    avoid_names: list[str] = field(default_factory=list)

    def category_for(self, i: int) -> str | None:
        return self.categories[i % len(self.categories)] if self.categories else None


class DiversityTracker:
    def __init__(self, registry: ToolRegistry, target_long_ratio: float = 0.5,
                 boost_unused: float = 3.0, boost_rare: float = 2.0, max_avoid_names: int = 15):
        self.registry = registry
        self.target_long_ratio = target_long_ratio
        self.boost_unused = boost_unused
        self.boost_rare = boost_rare
        self.max_avoid_names = max_avoid_names

        self.n = 0
        self.total_calls = 0
        self.multi = 0                      # conversations with 2+ tool calls
        self.long = 0                       # conversations with 3+ tool calls
        self.category_counts: Counter = Counter()
        self.tool_counts: Counter = Counter()
        self.first_name_counts: Counter = Counter()

    # ---- recording ---------------------------------------------------------

    def add_record(self, category: str, tool_names: list[str], persona: str = "") -> None:
        self.n += 1
        calls = len(tool_names)
        self.total_calls += calls
        self.multi += calls >= 2
        self.long += calls >= 3
        self.category_counts[category] += 1
        self.tool_counts.update(tool_names)
        name = persona_name(persona)
        if name:
            self.first_name_counts[name.split()[0]] += 1

    def add(self, conv: Conversation) -> None:
        self.add_record(conv.scenario.category, [tc.name for tc in conv.tool_calls], conv.scenario.persona)

    # ---- reporting ---------------------------------------------------------

    def stats(self) -> dict:
        n = max(self.n, 1)
        cats = self.registry.categories()
        shares = {c: round(self.category_counts[c] / n, 3) for c in cats}
        top = max(shares.values()) if shares else 0
        used = [t for t in self.registry.names() if self.tool_counts[t] > 0]
        return {
            "conversations": self.n,
            "mean_tool_calls": round(self.total_calls / n, 2),
            "multi_step_ratio": round(self.multi / n, 3),
            "long_chain_ratio": round(self.long / n, 3),
            "category_share": shares,
            "category_balance": round(min(shares.values()) / top, 3) if top else 0.0,
            "tool_coverage": round(len(used) / len(self.registry), 3),
            "repeated_first_names": {k: v for k, v in self.first_name_counts.most_common(5) if v > 1},
        }

    # ---- steering ----------------------------------------------------------

    def steering(self) -> SteeringPlan:
        categories = sorted(self.registry.categories(), key=lambda c: (self.category_counts[c], c))

        long_ratio = self.long / self.n if self.n else 0.0
        min_len = 3 if self.n and long_ratio < self.target_long_ratio else config.MIN_CHAIN_LENGTH

        boosts: dict[str, float] = {}
        if self.n:
            names = self.registry.names()
            expected = sum(self.tool_counts.values()) / len(names)
            for t in names:
                count = self.tool_counts[t]
                if count == 0:
                    boosts[t] = self.boost_unused
                elif count < 0.5 * expected:
                    boosts[t] = self.boost_rare

        avoid = [name for name, _ in self.first_name_counts.most_common(self.max_avoid_names)]
        return SteeringPlan(categories, min_len, boosts, avoid)
