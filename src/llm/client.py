"""OpenRouter LLM client for the Steps 5/6/7 roles.

Uses the OpenAI-compatible chat-completions API pointed at OpenRouter
(models.openrouter_base_url from evolver.yaml). Model IDs are locked in the
run configuration (deepseek/deepseek-v3 for research + evaluation, a ~30B
class model for the implementation loop); the exact IDs used are surfaced on
every call so a run can never silently use a different model than was planned.

Real calls only — no mock mode. Failures retry with exponential backoff up to
``max_retries``; a final failure raises, and the orchestrator treats that
candidate as failed (Step 6 stage 3 semantics) rather than halting the run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from openai import OpenAI

from src.config.env import load_env

log = logging.getLogger("llm")


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of a model response (tolerates markdown fences
    and trailing prose — the response is a chat message, not a guaranteed
    machine-readable payload)."""
    if text is None:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


class OpenRouterClient:
    """Minimal chat client recording which model produced each completion."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout_s: int = 120,
        temperature: float = 0.4,
    ) -> None:
        load_env()
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,  # we implement retries/backoff ourselves
        )

    def chat(self, messages: list[dict], model: str, temperature: float | None = None,
             max_tokens: int = 2048) -> str:
        """One chat completion. Raises LLMError after exhausting retries.

        An empty completion is a *generation*-level outcome (the request
        succeeded, the model produced nothing), not a transport fault. Retrying
        the identical request is wasted work; return "" so callers that expect
        code (CoderTeam) can re-prompt with a corrective nudge, and callers
        that expect JSON (chat_json) hit their parse-retry path.
        """
        temp = self.temperature if temperature is None else temperature
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                if content is None or not content.strip():
                    return ""
                return content
            except Exception as exc:  # noqa: BLE001 - surface any transport error
                last_err = exc
                if attempt < self.max_retries:
                    backoff = 2.0 * (2 ** attempt)
                    log.warning("LLM call to %s failed (%s); retrying in %.1fs",
                                model, exc, backoff)
                    time.sleep(backoff)
        raise LLMError(f"LLM call to {model} failed after {self.max_retries + 1} "
                       f"attempts: {last_err}")

    def chat_json(self, messages: list[dict], model: str,
                  temperature: float | None = None,
                  max_tokens: int = 2048) -> dict:
        """Chat completion parsed as JSON; retries once on parse failure."""
        for _ in range(2):
            text = self.chat(messages, model, temperature=temperature, max_tokens=max_tokens)
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
        raise LLMError(f"model {model} produced unparseable JSON: {text[:400]!r}")
