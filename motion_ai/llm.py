"""Thin Claude wrapper returning schema-validated JSON. Returns None on any failure so callers fall back to rules."""
from __future__ import annotations

import json
import logging

import anthropic

from .config import Settings

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, settings: Settings):
        self.model = settings.anthropic_model
        self.effort = settings.llm_effort
        self.client = anthropic.AsyncAnthropic(timeout=60.0, max_retries=2)

    async def json(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict | None:
        try:
            resp = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                # Server-side refusal fallback: reruns on a fallback model if this one declines.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError:
            log.warning("LLM rate limited; using rule-based fallback")
            return None
        except anthropic.APIStatusError as exc:
            log.warning("LLM API error %s: %s", exc.status_code, exc.message)
            return None
        except anthropic.APIConnectionError as exc:
            log.warning("LLM connection error: %s", exc)
            return None
        except (anthropic.AnthropicError, TypeError) as exc:  # e.g. no API key configured
            log.warning("LLM unavailable (%s); using rule-based fallback", exc)
            return None

        if resp.stop_reason in ("refusal", "max_tokens"):
            log.warning("LLM stopped with %s; using fallback", resp.stop_reason)
            return None
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON")
            return None
