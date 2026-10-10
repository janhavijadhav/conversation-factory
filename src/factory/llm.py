"""
Thin async wrapper around any OpenAI-compatible API.
Adds: request timeout, pacing, concurrency limit, retry with backoff, JSON parsing, call counting.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from openai import (
    APIConnectionError, APITimeoutError, AsyncOpenAI,
    InternalServerError, RateLimitError,
)

from src.factory import config

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
WAIT_PATTERN = re.compile(r"(?:retry|try again) in (?:(\d+)m)?([\d.]+)(ms|s)", re.IGNORECASE)
REQUEST_TIMEOUT_SECONDS = 90


class DailyQuotaExceeded(RuntimeError):
    """The provider's daily limit is used up. Retrying won't help until it resets."""


def parse_json(text: str) -> dict[str, Any]:
    """Pull the first {...} object out of a reply, ignoring <think> blocks, fences or chatter."""
    cleaned = THINK_BLOCK.sub("", text or "")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {cleaned[:200]!r}")
    return json.loads(match.group(0))


def suggested_wait(error: Exception) -> float | None:
    """Read the provider's suggested wait, e.g. 'retry in 37.5s', 'try again in 2m5.4s', 'in 650ms'."""
    match = WAIT_PATTERN.search(str(error))
    if not match:
        return None
    minutes, amount, unit = match.groups()
    seconds = float(amount) / 1000 if unit.lower() == "ms" else float(amount)
    seconds += 60 * int(minutes) if minutes else 0
    return seconds + 1


def limit_type(error: Exception) -> str:
    text = str(error).lower()
    for label in ("tokens per minute", "requests per minute", "tokens per day", "requests per day"):
        if label in text:
            return label
    if isinstance(error, APITimeoutError):
        return "request timeout"
    return "rate limit"


def is_daily_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "perday" in text or "per day" in text


class LLMClient:
    def __init__(self, model: str | None = None, max_retries: int = 8,
                 requests_per_minute: int | None = None):
        # max_retries=0: the library's silent retries are off; our loop below retries and logs.
        self.client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                                  timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
        self.model = model or config.MODEL_NAME
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(config.CONCURRENCY)

        rpm = config.REQUESTS_PER_MINUTE if requests_per_minute is None else requests_per_minute
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._pace_lock = asyncio.Lock()
        self._last_call = 0.0
        self.calls = 0
        self.rate_limit_waits = 0

    async def _pace(self) -> None:
        """Wait so calls are spaced at least _min_interval seconds apart."""
        if not self._min_interval:
            return
        async with self._pace_lock:
            wait = self._last_call + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   temperature: float = 0.7):
        """Return the raw assistant message object (has .content and .tool_calls)."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools

        delay = 10.0
        for attempt in range(1, self.max_retries + 1):
            try:
                await self._pace()
                async with self._semaphore:
                    response = await self.client.chat.completions.create(**kwargs)
                self.calls += 1
                return response.choices[0].message
            except RETRYABLE as e:
                if is_daily_limit(e):
                    raise DailyQuotaExceeded(
                        f"Daily quota used up for {self.model} ({limit_type(e)}). "
                        "Switch MODEL_NAME/provider in .env or wait for the reset."
                    ) from e
                if attempt == self.max_retries:
                    raise
                self.rate_limit_waits += 1
                wait = suggested_wait(e) or delay
                print(f"  [llm] {self.model}: {limit_type(e)}, waiting {wait:.1f}s "
                      f"(attempt {attempt}/{self.max_retries})")
                await asyncio.sleep(wait)
                delay = min(delay * 2, 90)

    async def chat_json(self, messages: list[dict], temperature: float = 0.7,
                        parse_retries: int = 2) -> dict[str, Any]:
        """Ask for JSON and parse it; retry if the model returns something unparseable."""
        last_error: Exception | None = None
        for _ in range(parse_retries + 1):
            msg = await self.chat(messages, temperature=temperature)
            try:
                return parse_json(msg.content)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
        raise ValueError(f"Could not get valid JSON: {last_error}")
