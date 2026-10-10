"""Assistant Agent: the model behavior we want to capture as training data."""
from __future__ import annotations

import json

from src.factory import config
from src.factory.agents.formatting import to_openai_messages
from src.factory.agents.prompts import ASSISTANT_PROMPT
from src.factory.models import Message, Scenario, ToolCall
from src.factory.state.session import SessionState
from src.factory.tools.registry import ToolRegistry


class AssistantAgent:
    def __init__(self, registry: ToolRegistry, llm):
        self.registry = registry
        self.llm = llm

    def tool_schemas(self, scenario: Scenario) -> list[dict]:
        categories = {self.registry.get(t).category for t in scenario.tool_chain}
        tools = [t for t in self.registry.all() if t.category in categories]
        return [t.to_openai_schema() for t in tools]

    async def respond(self, messages: list[Message], session: SessionState,
                      scenario: Scenario) -> Message:
        system = ASSISTANT_PROMPT.replace("{session}", session.describe())
        api_messages = [{"role": "system", "content": system}] + to_openai_messages(messages)

        reply = await self.llm.chat(
            api_messages,
            tools=self.tool_schemas(scenario),
            temperature=config.ASSISTANT_TEMPERATURE,
        )

        calls: list[ToolCall] = []
        for tc in reply.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            extra = {"id": tc.id} if tc.id else {}
            calls.append(ToolCall(name=tc.function.name, arguments=args, **extra))

        return Message(role="assistant", content=reply.content or None, tool_calls=calls)
