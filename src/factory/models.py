"""
Core data shapes for the Conversation Factory.
Everything else in the project passes these objects around.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class OperationType(str, Enum):
    """What kind of action a tool performs. Used for the 'operation rank' edge."""
    SEARCH = "search"   # find things        (rank 0)
    READ = "read"       # get one thing      (rank 1)
    CREATE = "create"   # make a new thing   (rank 2)
    UPDATE = "update"   # change a thing     (rank 3)
    DELETE = "delete"   # remove a thing     (rank 4)

    @property
    def rank(self) -> int:
        order = ["search", "read", "create", "update", "delete"]
        return order.index(self.value)


ParamType = Literal["string", "integer", "number", "boolean", "array", "object"]


class ToolParameter(BaseModel):
    """One input argument a tool accepts."""
    name: str
    type: ParamType
    description: str = ""
    required: bool = True
    enum: Optional[list[Any]] = None   # allowed values, if restricted


class ReturnField(BaseModel):
    """One field a tool gives back. Used to build data-flow edges."""
    name: str
    type: ParamType
    description: str = ""


class Tool(BaseModel):
    """A single API the assistant can call."""
    name: str
    category: str                      # e.g. "travel", "finance"
    description: str
    operation: OperationType
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: list[ReturnField] = Field(default_factory=list)

    def required_params(self) -> list[ToolParameter]:
        return [p for p in self.parameters if p.required]

    def param(self, name: str) -> Optional[ToolParameter]:
        return next((p for p in self.parameters if p.name == name), None)

    def to_openai_schema(self) -> dict:
        """Format the tool the way OpenAI-style APIs expect."""
        properties = {}
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [p.name for p in self.required_params()],
                },
            },
        }


# ---------------------------------------------------------------------------
# Tool calls and results
# ---------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class ToolCall(BaseModel):
    """The assistant asking to run a tool."""
    id: str = Field(default_factory=lambda: _new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """What the (mock) tool returned."""
    call_id: str
    name: str
    success: bool = True
    output: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Messages and conversations
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One turn in a conversation."""
    role: Role
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None   # set only when role == "tool"
    name: Optional[str] = None           # tool name, when role == "tool"


class Scenario(BaseModel):
    """The plan for one conversation, made by the Scenario Planner."""
    user_goal: str                       # what the user wants, in plain words
    tool_chain: list[str]                # ordered tool names sampled from the graph
    category: str
    persona: str = "a regular user"
    notes: str = ""


class QualityScores(BaseModel):
    """LLM judge scores, each 1-5."""
    correctness: float = 0.0     # right tools, right arguments
    naturalness: float = 0.0     # sounds like a real person/assistant
    groundedness: float = 0.0    # uses real IDs/values from earlier results
    completeness: float = 0.0    # user's goal actually got done
    feedback: str = ""

    @property
    def mean(self) -> float:
        return (self.correctness + self.naturalness
                + self.groundedness + self.completeness) / 4


class Conversation(BaseModel):
    """A full generated conversation plus its metadata."""
    id: str = Field(default_factory=lambda: _new_id("conv"))
    scenario: Scenario
    messages: list[Message] = Field(default_factory=list)
    scores: Optional[QualityScores] = None
    grounding_fixes: list[dict[str, Any]] = Field(default_factory=list)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    repair_attempts: int = 0
    passed: bool = False

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for m in self.messages for tc in m.tool_calls]

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def is_multi_step(self) -> bool:
        return self.num_tool_calls >= 2

    def to_training_record(self) -> dict:
        """One line of the final JSONL file."""
        return {
            "id": self.id,
            "category": self.scenario.category,
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "num_tool_calls": self.num_tool_calls,
            "num_grounding_fixes": len(self.grounding_fixes),
            "num_validation_errors": len(self.validation_errors),
            "stop_reason": self.stop_reason,
            "scores": self.scores.model_dump() if self.scores else None,
        }
