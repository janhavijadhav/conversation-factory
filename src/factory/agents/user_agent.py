"""User Agent: plays the person in the scenario, with a guard against role drift."""
from __future__ import annotations

import re

from pydantic import BaseModel

from src.factory import config
from src.factory.agents.formatting import user_view_transcript
from src.factory.agents.prompts import USER_PROMPT
from src.factory.models import Message, Scenario

# Phrases a user almost never says, but an assistant often does.
ASSISTANT_PATTERNS = [
    r"\bhow can i (help|assist)\b",
    r"\b(i can|i'd be happy to|happy to) help\b",
    r"\blet me (check|look|pull|find|search)\b",
    r"\bhere (is|are) your\b",
    # "your booking is confirmed", "your JetBlue flight has been booked" (up to 3 words between)
    r"\byour\s+(?:\w+\s+){0,3}(booking|reservation|order|flight|event|transfer|dispute)\b"
    r".*\b(is|has been) (confirmed|booked|cancelled|scheduled)\b",
    r"^(sure|certainly|of course)[,!.]?\s+(i can|i'll|let me)\b",
]
ASSISTANT_LIKE = re.compile("|".join(ASSISTANT_PATTERNS), re.IGNORECASE)


def looks_like_assistant(text: str) -> bool:
    return bool(ASSISTANT_LIKE.search(text or ""))


class UserReply(BaseModel):
    message: str
    done: bool = False


class UserAgent:
    def __init__(self, llm, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts
        self.role_drift_fixes = 0

    async def respond(self, scenario: Scenario, messages: list[Message]) -> UserReply:
        system = (USER_PROMPT
                  .replace("{persona}", scenario.persona)
                  .replace("{goal}", scenario.user_goal)
                  .replace("{notes}", scenario.notes))

        transcript = user_view_transcript(messages)
        if transcript:
            ask = f"Conversation so far:\n{transcript}\n\nWrite your next message."
        else:
            ask = "The conversation has not started. Write your opening message."

        reply = UserReply(message="")
        for attempt in range(self.max_attempts):
            prompt = ask if attempt == 0 else (
                ask + "\n\nYour last draft sounded like the ASSISTANT. Write ONLY what the "
                      "USER says: no offers to help, no reporting results you weren't told."
            )
            data = await self.llm.chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=config.USER_TEMPERATURE,
            )
            reply = UserReply(message=str(data.get("message", "")).strip(),
                              done=bool(data.get("done", False)))
            if not looks_like_assistant(reply.message):
                return reply
            self.role_drift_fixes += 1
        return reply          # still drifting after retries: the judge will catch it	
