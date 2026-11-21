"""Thin Claude wrapper returning schema-validated JSON. Returns None on any failure so callers fall back to rules.

Two backends, picked by LLM_BACKEND (env or .env):
  api         Anthropic API via the SDK. Needs ANTHROPIC_API_KEY.
  claude-cli  The Claude Code CLI (`claude -p`), signed in with the user's Claude account; no API key.
              ~5 s per call, fine for commentary windows and curation.
  auto        (default) api when ANTHROPIC_API_KEY is set, else claude-cli when `claude` is on PATH.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import anthropic

from .config import Settings

log = logging.getLogger(__name__)

CLI_TIMEOUT_S = 120.0


def _env(name: str, env_file: str = ".env") -> str:
    """Value from the process env, else from .env (pydantic Settings does not export .env to os.environ)."""
    if os.environ.get(name):
        return os.environ[name].strip()
    p = Path(env_file)
    if p.exists():
        for line in p.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == name:
                return v.split("#", 1)[0].strip().strip("\"'")
    return ""


def pick_backend() -> str:
    choice = (_env("LLM_BACKEND") or "auto").lower()
    if choice in ("api", "claude-cli"):
        return choice
    if _env("ANTHROPIC_API_KEY"):
        return "api"
    return "claude-cli" if shutil.which("claude") else "api"


class LLM:
    def __init__(self, settings: Settings):
        self.model = settings.anthropic_model
        self.effort = settings.llm_effort
        self.backend = pick_backend()
        self.calls = 0
        self.failures = 0
        log.info("LLM backend: %s (model %s)", self.backend, self.model)
        if self.backend == "api":
            key = _env("ANTHROPIC_API_KEY") or None
            self.client = anthropic.AsyncAnthropic(api_key=key, timeout=60.0, max_retries=2)

    async def json(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict | None:
        self.calls += 1
        if self.backend == "claude-cli":
            out = await self._cli_json(system, user, schema)
        else:
            out = await self._api_json(system, user, schema, max_tokens)
        if out is None:
            self.failures += 1
        return out

    async def _cli_json(self, system: str, user: str, schema: dict) -> dict | None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use the signed-in account
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.model, "--effort", self.effort,
               "--tools", "", "--no-session-persistence",
               "--system-prompt", system, "--json-schema", json.dumps(schema)]
        try:
            # Neutral cwd so the CLI does not pick up this repo's CLAUDE.md / project settings.
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=tempfile.gettempdir(), env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(proc.communicate(user.encode()), CLI_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError) as exc:
            log.warning("claude CLI unavailable (%s); using rule-based fallback", exc)
            return None
        try:
            res = json.loads(stdout)
        except json.JSONDecodeError:
            log.warning("claude CLI returned non-JSON (exit %s): %s", proc.returncode, stderr.decode()[-300:])
            return None
        if res.get("is_error") or res.get("subtype") != "success":
            log.warning("claude CLI error: %s", str(res.get("result") or res.get("subtype"))[:300])
            return None
        out = res.get("structured_output")
        if not isinstance(out, dict):
            log.warning("claude CLI gave no structured output")
            return None
        return out

    async def _api_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict | None:
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
