"""Scenario Planner: turns a sampled tool chain into a believable user + goal."""
from __future__ import annotations

from src.factory import config
from src.factory.agents.prompts import PLANNER_PROMPT
from src.factory.graph.sampler import SampledChain
from src.factory.models import Scenario
from src.factory.tools.registry import ToolRegistry


class ScenarioPlanner:
    def __init__(self, registry: ToolRegistry, llm):
        self.registry = registry
        self.llm = llm
        self.avoid_names: list[str] = []      # set by diversity steering before each batch

    def _tool_lines(self, chain: SampledChain) -> str:
        lines = []
        for i, name in enumerate(chain.tools, start=1):
            t = self.registry.get(name)
            inputs = ", ".join(
                p.name + ("" if p.required else " (optional)") for p in t.parameters
            ) or "none"
            returns = ", ".join(r.name for r in t.returns) or "nothing"
            lines.append(f"{i}. {t.name}: {t.description}. Inputs: {inputs}. Returns: {returns}")
        return "\n".join(lines)

    async def plan(self, chain: SampledChain) -> Scenario:
        prompt = (PLANNER_PROMPT
                  .replace("{tool_lines}", self._tool_lines(chain))
                  .replace("{avoid_names}", ", ".join(self.avoid_names) or "(none yet)"))
        data = await self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=config.PLANNER_TEMPERATURE,
        )
        notes = data.get("notes", "")
        if isinstance(notes, list):
            notes = "; ".join(str(n) for n in notes)
        return Scenario(
            user_goal=data.get("user_goal", "Get help with a task"),
            persona=data.get("persona", "a regular user"),
            notes=str(notes),
            tool_chain=chain.tools,
            category=chain.main_category,
        )
