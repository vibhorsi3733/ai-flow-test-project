"""Thin Claude wrapper returning schema-validated JSON. Returns None on any failure so callers fall back to rules.

Two backends, picked by LLM_BACKEND (env or .env):
  api         Anthropic API via the SDK. Needs ANTHROPIC_API_KEY.
  claude-cli  The Claude Code CLI (`claude -p`), signed in with the user's Claude account; no API key.
              ~5 s per call, fine for commentary windows and curation.
  auto        (default) api when ANTHROPIC_API_KEY is set, else claude-cli when `claude` is on PATH.

Usage accounting: every call appends one record to `LLM.usage_log` (stage, backend, model, input/output/
cache-read/cache-creation tokens, latency_s, ok, cost_usd) and `LLM.usage_summary()` aggregates it per stage
for the package manifest (`llm_usage`). Tokens come only from what the backend reports; when it reports
nothing (call failed before a response, CLI JSON without usage) they are None, never estimated.
  api         response.usage of the SDK's BetaMessage (input_tokens, output_tokens, cache_read_input_tokens,
              cache_creation_input_tokens, cache_creation.ephemeral_5m/1h_input_tokens); model = response.model
              (can differ from the requested one after a server-side fallback).
  claude-cli  `claude -p --output-format json` result: `modelUsage` {model: {inputTokens, outputTokens,
              cacheReadInputTokens, cacheCreationInputTokens, costUSD}} summed over models (the CLI may add a
              small helper-model call), else top-level `usage` (API field names); `total_cost_usd` is kept as
              cli_reported_cost_usd. The CLI is billed to the Claude Code plan, so cost_usd is an
              API-EQUIVALENT figure, labelled with cost_basis.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import anthropic

from .config import Settings

log = logging.getLogger(__name__)

CLI_TIMEOUT_S = 120.0

# USD per 1M tokens, Anthropic first-party API list prices (cached 2026-11-03).
# claude-haiku-5-5: cache-read price not published in our table -> priced as plain input (overstates cost).
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.10},
}
PRICE_NOTES = ["claude-haiku-5-5 cache read: price unknown, charged at the input price",
               "cache writes: 1.25x input (5-minute TTL), 2x input (1-hour TTL), standard API multipliers"]
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.0
COST_BASIS = {"api": "api (billed to ANTHROPIC_API_KEY)",
              "claude-cli": "api-equivalent (claude-cli: billed to the Claude Code plan, not the API)"}
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def price_for(model: str | None) -> dict[str, float] | None:
    """Table price for a model id, tolerating a date/alias suffix (claude-opus-5-5-20261001)."""
    if not model:
        return None
    return next((v for k, v in PRICES.items() if model == k or model.startswith(k + "-")), None)


def cost_usd(model: str | None, input_tokens: int | None, output_tokens: int | None,
             cache_read: int | None = 0, cache_write: int | None = 0, cache_write_1h: int | None = 0) -> float | None:
    """API cost of one call from the price table; None when the model is not priced or no tokens are known.
    cache_write = all cache-creation tokens, cache_write_1h = the part of them written with a 1-hour TTL."""
    p = price_for(model)
    if p is None or (input_tokens is None and output_tokens is None):
        return None
    cw, cw1h = cache_write or 0, min(cache_write_1h or 0, cache_write or 0)
    dollars = ((input_tokens or 0) * p["input"] + (output_tokens or 0) * p["output"]
               + (cache_read or 0) * p["cache_read"]
               + (cw - cw1h) * p["input"] * CACHE_WRITE_5M + cw1h * p["input"] * CACHE_WRITE_1H)
    return round(dollars / 1e6, 6)


def _get(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def usage_from_api(resp) -> dict:
    """Token fields of an SDK response (BetaMessage.usage / BetaUsage). Missing usage -> None values."""
    u = _get(resp, "usage")
    model = _get(resp, "model")
    if u is None:
        return {"model": model, **{k: None for k in TOKEN_KEYS}, "cost_usd": None}
    out = {"model": model, **{k: _get(u, k) for k in TOKEN_KEYS}}
    cc = _get(u, "cache_creation")
    w1h = _get(cc, "ephemeral_1h_input_tokens") if cc is not None else 0
    out["cost_usd"] = cost_usd(model, out["input_tokens"], out["output_tokens"], out["cache_read_input_tokens"],
                               out["cache_creation_input_tokens"], w1h)
    return out


def usage_from_cli(res: dict, model: str | None) -> dict:
    """Token fields of a `claude -p --output-format json` result. Defensive: any field may be missing.

    Prefers `modelUsage` (per model, camelCase, includes helper-model calls the CLI makes), else top-level
    `usage` (API names). cost_usd = API-equivalent from the price table; a model not in the table uses the
    CLI's own per-model costUSD (list price); if neither is known the call's cost is None."""
    out: dict = {"model": model, **{k: None for k in TOKEN_KEYS}, "cost_usd": None,
                 "cli_reported_cost_usd": res.get("total_cost_usd") if isinstance(res, dict) else None}
    if not isinstance(res, dict):
        return out
    top = res.get("usage") if isinstance(res.get("usage"), dict) else {}
    top_1h = (top.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
    mu = res.get("modelUsage")
    if isinstance(mu, dict) and mu:
        camel = {"input_tokens": "inputTokens", "output_tokens": "outputTokens",
                 "cache_read_input_tokens": "cacheReadInputTokens",
                 "cache_creation_input_tokens": "cacheCreationInputTokens"}
        totals = {k: None for k in TOKEN_KEYS}
        cost, cost_known, models = 0.0, True, {}
        for name, m in mu.items():
            if not isinstance(m, dict):
                continue
            vals = {k: m.get(c) for k, c in camel.items()}
            for k, v in vals.items():
                if isinstance(v, (int, float)):
                    totals[k] = (totals[k] or 0) + int(v)
            # the 1-hour cache split is only reported for the main model (top-level usage)
            main = name == model or (model and name.startswith(model + "-"))
            c = cost_usd(name, vals["input_tokens"], vals["output_tokens"], vals["cache_read_input_tokens"],
                         vals["cache_creation_input_tokens"], top_1h if main else 0)
            if c is None and isinstance(m.get("costUSD"), (int, float)):
                c = round(float(m["costUSD"]), 6)
            if c is None:
                cost_known = False
            else:
                cost += c
            models[name] = {**vals, "cost_usd": c}
        out |= totals
        out["cost_usd"] = round(cost, 6) if cost_known and models else None
        out["models"] = models
        return out
    if top:
        out |= {k: top.get(k) for k in TOKEN_KEYS}
        out["cost_usd"] = cost_usd(model, out["input_tokens"], out["output_tokens"], out["cache_read_input_tokens"],
                                   out["cache_creation_input_tokens"], top_1h)
    return out


def summarize_usage(records: list[dict], backend: str | None = None) -> dict:
    """Per-stage and total aggregates of usage records (as stored in the manifest's `llm_usage`)."""
    def agg(rs: list[dict]) -> dict:
        a = {"calls": len(rs), "ok": sum(1 for r in rs if r.get("ok")),
             "calls_without_usage": sum(1 for r in rs if r.get("input_tokens") is None
                                        and r.get("output_tokens") is None)}
        for k in TOKEN_KEYS:
            a[k] = sum(r.get(k) or 0 for r in rs)
        costs = [r.get("cost_usd") for r in rs]
        a["cost_usd"] = round(sum(c for c in costs if c is not None), 6)
        a["cost_complete"] = all(c is not None for c in costs)
        cli = [r.get("cli_reported_cost_usd") for r in rs if r.get("backend") == "claude-cli"]
        if cli:
            a["cli_reported_cost_usd"] = round(sum(c for c in cli if c is not None), 6)
        a["latency_s"] = round(sum(r.get("latency_s") or 0 for r in rs), 2)
        return a

    stages: dict[str, list[dict]] = {}
    for r in records:
        stages.setdefault(r.get("stage") or "other", []).append(r)
    backends = sorted({r.get("backend") for r in records if r.get("backend")}) or ([backend] if backend else [])
    return {
        "backend": backends[0] if len(backends) == 1 else backends,
        "models": sorted({r.get("model") for r in records if r.get("model")}),
        "cost_basis": "; ".join(COST_BASIS.get(b, b) for b in backends),
        "prices_per_mtok": PRICES, "price_notes": PRICE_NOTES,
        "stages": {s: agg(rs) for s, rs in stages.items()},
        "total": agg(records),
        "calls": records,
    }


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
        self.usage_log: list[dict] = []   # one record per call, see module docstring
        log.info("LLM backend: %s (model %s)", self.backend, self.model)
        if self.backend == "api":
            key = _env("ANTHROPIC_API_KEY") or None
            self.client = anthropic.AsyncAnthropic(api_key=key, timeout=60.0, max_retries=2)

    async def json(self, system: str, user: str, schema: dict, max_tokens: int = 8000,
                   stage: str = "other") -> dict | None:
        """stage labels the call for usage accounting: "commentary", "curator", else "other"."""
        self.calls += 1
        usage: dict = {}
        t0 = time.monotonic()
        if self.backend == "claude-cli":
            out = await self._cli_json(system, user, schema, usage)
        else:
            out = await self._api_json(system, user, schema, max_tokens, usage)
        if out is None:
            self.failures += 1
        rec = {"stage": stage, "backend": self.backend, "model": usage.get("model") or self.model,
               **{k: usage.get(k) for k in TOKEN_KEYS}, "latency_s": round(time.monotonic() - t0, 3),
               "ok": out is not None, "cost_usd": usage.get("cost_usd"),
               "cost_basis": COST_BASIS.get(self.backend, self.backend)}
        if self.backend == "claude-cli":
            rec["cli_reported_cost_usd"] = usage.get("cli_reported_cost_usd")
        self.usage_log.append(rec)
        return out

    def usage_summary(self) -> dict:
        """Aggregates for the package manifest (`llm_usage`)."""
        return summarize_usage(self.usage_log, self.backend)

    async def _cli_json(self, system: str, user: str, schema: dict, usage: dict | None = None) -> dict | None:
        usage = {} if usage is None else usage
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
        if not isinstance(res, dict):
            log.warning("claude CLI returned unexpected JSON: %s", str(res)[:300])
            return None
        usage.update(usage_from_cli(res, self.model))
        if res.get("is_error") or res.get("subtype") != "success":
            log.warning("claude CLI error: %s", str(res.get("result") or res.get("subtype"))[:300])
            return None
        out = res.get("structured_output")
        if not isinstance(out, dict):
            log.warning("claude CLI gave no structured output")
            return None
        return out

    async def _api_json(self, system: str, user: str, schema: dict, max_tokens: int,
                        usage: dict | None = None) -> dict | None:
        usage = {} if usage is None else usage
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

        usage.update(usage_from_api(resp))
        if resp.stop_reason in ("refusal", "max_tokens"):
            log.warning("LLM stopped with %s; using fallback", resp.stop_reason)
            return None
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON")
            return None
