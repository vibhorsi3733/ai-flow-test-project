"""Claude usage accounting: API/CLI usage parsing, aggregation + cost math, manifest wiring, expense.txt."""
import asyncio
import json
import os
import stat
from types import SimpleNamespace

import pytest

from motion_ai import expense
from motion_ai import llm as llm_mod
from motion_ai.highlights.pipeline import HighlightPipeline
from motion_ai.replay import replay_usage

SETTINGS = SimpleNamespace(anthropic_model="claude-opus-5-5", llm_effort="low")

# Shape of a real `claude -p --output-format json` result (CLI 2.1.284), trimmed.
CLI_RESULT = {
    "type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.0123,
    "structured_output": {"events": []},
    "usage": {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 3000,
              "cache_creation": {"ephemeral_1h_input_tokens": 3000, "ephemeral_5m_input_tokens": 0}},
    "modelUsage": {
        "claude-opus-5-5": {"inputTokens": 1000, "outputTokens": 200, "cacheReadInputTokens": 0,
                            "cacheCreationInputTokens": 3000, "costUSD": 0.03},
        "claude-haiku-4-5-20251001": {"inputTokens": 900, "outputTokens": 12, "cacheReadInputTokens": 0,
                                      "cacheCreationInputTokens": 0, "costUSD": 0.000956},
    },
}


def fake_claude(tmp_path, monkeypatch, payload):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    script = tmp_path / "claude"
    script.write_text(f"#!/bin/sh\ncat > /dev/null\ncat <<'EOF'\n{body}\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LLM_BACKEND", "claude-cli")


def ask(llm, stage="other"):
    return asyncio.run(llm.json("system", "user", {"type": "object"}, stage=stage))


# ---- cost math -------------------------------------------------------------------------------------------
def test_cost_math_price_table():
    # 1M in + 1M out + 1M cache read on Opus: 4 + 20 + 0.20
    assert llm_mod.cost_usd("claude-opus-5-5", 10**6, 10**6, 10**6) == pytest.approx(24.20)
    assert llm_mod.cost_usd("claude-sonnet-5-5", 10**6, 10**6, 10**6) == pytest.approx(12.10)
    # haiku cache read priced as input (documented)
    assert llm_mod.cost_usd("claude-haiku-5-5", 0, 0, 10**6) == pytest.approx(0.10)
    # cache writes: 5m at 1.25x, 1h at 2x input
    assert llm_mod.cost_usd("claude-opus-5-5", 0, 0, 0, 10**6, 0) == pytest.approx(5.0)
    assert llm_mod.cost_usd("claude-opus-5-5", 0, 0, 0, 10**6, 10**6) == pytest.approx(8.0)
    assert llm_mod.cost_usd("claude-opus-5-5-20261001", 10**6, 0) == pytest.approx(4.0)  # dated id
    assert llm_mod.cost_usd("gpt-x", 10, 10) is None
    assert llm_mod.cost_usd("claude-opus-5-5", None, None) is None


# ---- API path --------------------------------------------------------------------------------------------
def api_resp(text='{"a": 1}', stop="end_turn", model="claude-opus-5-5", usage=True):
    u = SimpleNamespace(input_tokens=1200, output_tokens=500, cache_read_input_tokens=300,
                        cache_creation_input_tokens=None, cache_creation=None) if usage else None
    return SimpleNamespace(stop_reason=stop, model=model, usage=u,
                           content=[SimpleNamespace(type="text", text=text)])


def api_llm(monkeypatch, resp=None, exc=None):
    monkeypatch.setenv("LLM_BACKEND", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = llm_mod.LLM(SETTINGS)

    async def create(**kw):
        if exc:
            raise exc
        return resp
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return llm


def test_api_usage_fields_exist_on_sdk_class():
    from anthropic.types.beta import BetaUsage
    for k in llm_mod.TOKEN_KEYS:
        assert k in BetaUsage.model_fields


def test_api_usage_recorded(monkeypatch):
    llm = api_llm(monkeypatch, api_resp())
    assert ask(llm, "curator") == {"a": 1}
    rec = llm.usage_log[0]
    assert rec["stage"] == "curator" and rec["backend"] == "api" and rec["ok"] is True
    assert (rec["input_tokens"], rec["output_tokens"], rec["cache_read_input_tokens"]) == (1200, 500, 300)
    assert rec["cache_creation_input_tokens"] is None
    assert rec["cost_usd"] == pytest.approx((1200 * 4 + 500 * 20 + 300 * 0.2) / 1e6)
    assert rec["latency_s"] >= 0 and rec["cost_basis"].startswith("api")


def test_api_refusal_keeps_usage_and_fallback_model(monkeypatch):
    llm = api_llm(monkeypatch, api_resp(stop="refusal", model="claude-sonnet-5-5"))
    assert ask(llm) is None
    rec = llm.usage_log[0]
    assert rec["ok"] is False and rec["model"] == "claude-sonnet-5-5" and rec["input_tokens"] == 1200


def test_api_exception_records_null_tokens(monkeypatch):
    import anthropic
    llm = api_llm(monkeypatch, exc=anthropic.APIConnectionError(request=None))
    assert ask(llm, "commentary") is None
    rec = llm.usage_log[0]
    assert rec["ok"] is False and rec["input_tokens"] is None and rec["cost_usd"] is None
    assert rec["model"] == "claude-opus-5-5"


# ---- CLI path --------------------------------------------------------------------------------------------
def test_cli_usage_from_model_usage(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch, CLI_RESULT)
    llm = llm_mod.LLM(SETTINGS)
    assert ask(llm, "commentary") == {"events": []}
    rec = llm.usage_log[0]
    assert rec["backend"] == "claude-cli" and rec["stage"] == "commentary" and rec["ok"]
    assert rec["input_tokens"] == 1900 and rec["output_tokens"] == 212
    assert rec["cache_creation_input_tokens"] == 3000
    opus = (1000 * 4 + 200 * 20 + 3000 * 4 * 2.0) / 1e6   # 1h cache write on the main model
    assert rec["cost_usd"] == pytest.approx(opus + 0.000956)  # helper model not in table -> CLI costUSD
    assert rec["cli_reported_cost_usd"] == 0.0123
    assert "api-equivalent" in rec["cost_basis"]


def test_cli_top_level_usage_only():
    res = {k: v for k, v in CLI_RESULT.items() if k != "modelUsage"}
    u = llm_mod.usage_from_cli(res, "claude-opus-5-5")
    assert (u["input_tokens"], u["output_tokens"]) == (1000, 200)
    assert u["cost_usd"] == pytest.approx((1000 * 4 + 200 * 20 + 3000 * 8) / 1e6)


@pytest.mark.parametrize("res", [
    {"type": "result", "subtype": "success", "is_error": False, "structured_output": {}},   # no usage at all
    {"modelUsage": {"claude-opus-5-5": {}}},                                                # empty entries
    {"usage": "garbage", "modelUsage": []},
])
def test_cli_missing_usage_is_null_not_invented(res):
    u = llm_mod.usage_from_cli(res, "claude-opus-5-5")
    assert u["input_tokens"] is None and u["output_tokens"] is None
    assert u["cost_usd"] is None


def test_cli_unpriced_model_without_cost_is_incomplete():
    u = llm_mod.usage_from_cli({"modelUsage": {"mystery": {"inputTokens": 5, "outputTokens": 1}}}, "mystery")
    assert u["input_tokens"] == 5 and u["cost_usd"] is None


def test_cli_error_still_records_call(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch, "not json")
    llm = llm_mod.LLM(SETTINGS)
    assert ask(llm) is None
    assert llm.usage_log[0]["ok"] is False and llm.usage_log[0]["input_tokens"] is None


# ---- aggregation -----------------------------------------------------------------------------------------
def rec(stage, i, o, cr=0, cost=0.01, ok=True, backend="api"):
    return {"stage": stage, "backend": backend, "model": "claude-opus-5-5", "input_tokens": i, "output_tokens": o,
            "cache_read_input_tokens": cr, "cache_creation_input_tokens": None, "latency_s": 1.5, "ok": ok,
            "cost_usd": cost}


def test_summary_per_stage_and_total():
    recs = [rec("commentary", 100, 50), rec("commentary", 200, 60, cr=10),
            rec("commentary", None, None, cost=None, ok=False), rec("curator", 1000, 400, cost=0.5)]
    s = llm_mod.summarize_usage(recs)
    c = s["stages"]["commentary"]
    assert (c["calls"], c["ok"], c["calls_without_usage"]) == (3, 2, 1)
    assert (c["input_tokens"], c["output_tokens"], c["cache_read_input_tokens"]) == (300, 110, 10)
    assert c["cost_usd"] == pytest.approx(0.02) and c["cost_complete"] is False
    assert s["stages"]["curator"]["cost_complete"] is True
    assert s["total"]["calls"] == 4 and s["total"]["cost_usd"] == pytest.approx(0.52)
    assert s["total"]["latency_s"] == 6.0
    assert s["backend"] == "api" and s["models"] == ["claude-opus-5-5"]
    assert len(s["calls"]) == 4


def test_summary_empty():
    s = llm_mod.summarize_usage([], "claude-cli")
    assert s["total"]["calls"] == 0 and s["total"]["cost_usd"] == 0 and s["stages"] == {}


# ---- manifest wiring -------------------------------------------------------------------------------------
def test_pipeline_manifest_has_llm_usage(tmp_path, monkeypatch):
    llm = api_llm(monkeypatch, api_resp())
    ask(llm, "commentary")
    s = SimpleNamespace(work_dir=tmp_path, video_encoder="libx264")
    p = HighlightPipeline(s, buffer=None, llm=llm, match={})
    p.stats = lambda: {"ocr_read_failures": 0}
    m = p.manifest_stats()
    assert m["ocr_read_failures"] == 0 and m["clip_failures"] == 0
    assert m["llm_usage"]["run"] == "live"
    assert m["llm_usage"]["stages"]["commentary"]["input_tokens"] == 1200
    json.dumps(m)   # serialisable into the package json


def test_pipeline_without_llm_has_no_usage(tmp_path):
    p = HighlightPipeline(SimpleNamespace(work_dir=tmp_path, video_encoder="libx264"), buffer=None, llm=None,
                          match={})
    assert "llm_usage" not in p.manifest_stats()


def test_replay_usage_is_labelled(monkeypatch):
    llm = api_llm(monkeypatch, api_resp())
    ask(llm, "curator")
    u = replay_usage(llm)
    assert u["run"] == "replay" and "not the live" in u["note"]
    assert u["stages"]["curator"]["calls"] == 1
    assert replay_usage(None)["total"]["calls"] == 0


# ---- expense.txt -----------------------------------------------------------------------------------------
def test_expense_rewrite_idempotent(tmp_path, capsys):
    usage = llm_mod.summarize_usage([rec("commentary", 100, 50), rec("curator", 1000, 400, cost=0.5)]) | \
        {"run": "live"}
    pkg = tmp_path / "pkg.json"
    pkg.write_text(json.dumps({"title": "x", "llm_usage": usage}))
    exp = tmp_path / "expense.txt"
    exp.write_text("ESTIMATES\nA) old section\n")
    expense.main([str(pkg), "--expense-file", str(exp)])
    first = exp.read_text()
    assert first.startswith(expense.BEGIN)
    assert "MEASURED (from pkg.json)" in first and "ESTIMATES\nA) old section\n" in first
    assert "commentary" in first and "curator" in first and "$0.5100" in first
    expense.main([str(pkg), "--expense-file", str(exp)])
    assert exp.read_text() == first
    assert first.count(expense.BEGIN) == 1
    assert "TOTAL" in capsys.readouterr().out


def test_expense_missing_usage_exits(tmp_path):
    pkg = tmp_path / "pkg.json"
    pkg.write_text(json.dumps({"title": "x"}))
    with pytest.raises(SystemExit):
        expense.main([str(pkg), "--no-write"])
