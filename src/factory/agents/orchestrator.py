"""
Orchestrator: wires the agents together with LangGraph.

ConversationState = the dialogue (messages, turn counters, quality logs).
SessionState      = the world (entities, IDs), referenced but kept separate.

Every tool call goes: grounding -> validator -> executor.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.factory import config
from src.factory.agents.assistant_agent import AssistantAgent
from src.factory.agents.planner import ScenarioPlanner
from src.factory.agents.user_agent import UserAgent
from src.factory.graph.sampler import SampledChain
from src.factory.models import Conversation, Message, Scenario
from src.factory.quality.validators import ToolCallValidator
from src.factory.state.grounding import ConversationGrounding
from src.factory.state.session import SessionState
from src.factory.tools.mock_executor import MockExecutor
from src.factory.tools.registry import ToolRegistry


class ConversationState(TypedDict, total=False):
    chain: SampledChain
    scenario: Scenario
    messages: list[Message]
    session: SessionState
    executor: MockExecutor
    grounding: ConversationGrounding
    validator: ToolCallValidator
    grounding_fixes: list[dict[str, Any]]
    validation_errors: list[dict[str, Any]]
    user_turns: int
    tool_rounds: int
    done: bool
    stop_reason: str


class ConversationGenerator:
    def __init__(self, registry: ToolRegistry, llm, simulator_llm=None):
        """llm drives the assistant (the training target).
        simulator_llm drives the planner and simulated user; defaults to llm."""
        self.registry = registry
        self.llm = llm
        simulator = simulator_llm or llm
        self.planner = ScenarioPlanner(registry, simulator)
        self.user = UserAgent(simulator)
        self.assistant = AssistantAgent(registry, llm)
        self.app = self._build_graph()

    # ---- graph -------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(ConversationState)
        g.add_node("plan", self._plan)
        g.add_node("user", self._user)
        g.add_node("assistant", self._assistant)
        g.add_node("tools", self._tools)

        g.add_edge(START, "plan")
        g.add_edge("plan", "user")
        g.add_conditional_edges("user", self._after_user, {"assistant": "assistant", "end": END})
        g.add_conditional_edges("assistant", self._after_assistant,
                                {"tools": "tools", "user": "user", "end": END})
        g.add_edge("tools", "assistant")
        return g.compile()

    def _allowed_tools(self, scenario: Scenario) -> set[str]:
        return {s["function"]["name"] for s in self.assistant.tool_schemas(scenario)}

    # ---- nodes -------------------------------------------------------------

    async def _plan(self, state: ConversationState) -> dict:
        if state.get("scenario"):
            return {}
        return {"scenario": await self.planner.plan(state["chain"])}

    async def _user(self, state: ConversationState) -> dict:
        if state.get("user_turns", 0) >= config.MAX_USER_TURNS:
            return {"done": True, "stop_reason": "max_user_turns"}

        reply = await self.user.respond(state["scenario"], state["messages"])
        if reply.done and state["messages"]:
            # End on the assistant's last reply, which makes a cleaner training example.
            return {"done": True, "stop_reason": "user_satisfied"}

        return {
            "messages": state["messages"] + [Message(role="user", content=reply.message)],
            "user_turns": state.get("user_turns", 0) + 1,
        }

    async def _assistant(self, state: ConversationState) -> dict:
        msg = await self.assistant.respond(state["messages"], state["session"], state["scenario"])
        return {"messages": state["messages"] + [msg]}

    async def _tools(self, state: ConversationState) -> dict:
        last = state["messages"][-1]
        allowed = self._allowed_tools(state["scenario"])
        grounded_calls, tool_messages = [], []
        new_fixes: list[dict[str, Any]] = []
        new_errors: list[dict[str, Any]] = []

        for tc in last.tool_calls:
            # 1. grounding: fix safe mistakes
            grounded, fixes = state["grounding"].ground(tc)
            new_fixes.extend(f.to_dict() for f in fixes)
            grounded_calls.append(grounded)

            # 2. validator: block anything still wrong
            violations = state["validator"].validate(grounded, allowed)
            if violations:
                new_errors.extend(v.to_dict() for v in violations)
                payload = {"error": "Invalid tool call: " + "; ".join(v.message for v in violations)}
            else:
                # 3. executor: run it
                result = state["executor"].execute(grounded)
                if result.success:
                    state["validator"].remember(grounded)
                payload = result.output if result.success else {"error": result.error}

            tool_messages.append(Message(
                role="tool", tool_call_id=grounded.id, name=grounded.name, content=json.dumps(payload),
            ))

        # The transcript keeps the corrected calls, so training data shows the right behavior.
        fixed_last = last.model_copy(update={"tool_calls": grounded_calls})
        return {
            "messages": state["messages"][:-1] + [fixed_last] + tool_messages,
            "grounding_fixes": state.get("grounding_fixes", []) + new_fixes,
            "validation_errors": state.get("validation_errors", []) + new_errors,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    # ---- routing -----------------------------------------------------------

    @staticmethod
    def _after_user(state: ConversationState) -> str:
        return "end" if state.get("done") else "assistant"

    @staticmethod
    def _after_assistant(state: ConversationState) -> str:
        last = state["messages"][-1]
        if last.tool_calls:
            if state.get("tool_rounds", 0) >= config.MAX_TOOL_ROUNDS:
                return "end"
            return "tools"
        return "user"

    # ---- public ------------------------------------------------------------

    async def generate(self, chain: SampledChain, seed: int | None = None,
                       scenario: Scenario | None = None) -> tuple[Conversation, SessionState, str]:
        session = SessionState()
        initial: ConversationState = {
            "chain": chain,
            "messages": [],
            "session": session,
            "executor": MockExecutor(self.registry, session, seed=seed),
            "grounding": ConversationGrounding(self.registry, session),
            "validator": ToolCallValidator(self.registry, session),
            "grounding_fixes": [],
            "validation_errors": [],
            "user_turns": 0,
            "tool_rounds": 0,
            "done": False,
        }
        if scenario:
            initial["scenario"] = scenario

        final = await self.app.ainvoke(initial, config={"recursion_limit": 80})
        stop_reason = final.get("stop_reason", "max_tool_rounds")
        conversation = Conversation(
            scenario=final["scenario"],
            messages=final["messages"],
            grounding_fixes=final.get("grounding_fixes", []),
            validation_errors=final.get("validation_errors", []),
            stop_reason=stop_reason,
        )
        return conversation, session, stop_reason
