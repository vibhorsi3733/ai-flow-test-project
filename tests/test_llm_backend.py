import asyncio
import json
import os
import stat
from types import SimpleNamespace

import pytest

from motion_ai import llm as llm_mod

SETTINGS = SimpleNamespace(anthropic_model="claude-opus-5-5", llm_effort="low")


def fake_claude(tmp_path, monkeypatch, payload: dict | str, exit_code: int = 0):
    """Put a fake `claude` on PATH that prints `payload` and records its stdin."""
    body = payload if isinstance(payload, str) else json.dumps(payload)
    script = tmp_path / "claude"
    script.write_text(f"#!/bin/sh\ncat > {tmp_path}/stdin.txt\ncat <<'EOF'\n{body}\nEOF\nexit {exit_code}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LLM_BACKEND", "claude-cli")


def ask(llm):
    return asyncio.run(llm.json("system", "user text", {"type": "object"}))


def test_cli_returns_structured_output(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch, {"type": "result", "subtype": "success", "is_error": False,
                                        "structured_output": {"events": [{"type": "six", "line": 2}]}})
    llm = llm_mod.LLM(SETTINGS)
    assert llm.backend == "claude-cli"
    assert ask(llm) == {"events": [{"type": "six", "line": 2}]}
    assert (tmp_path / "stdin.txt").read_text() == "user text"


@pytest.mark.parametrize("payload", [
    {"type": "result", "subtype": "error_max_turns", "is_error": True},
    {"type": "result", "subtype": "success", "is_error": False},          # no structured_output
    "not json at all",
])
def test_cli_failures_return_none(tmp_path, monkeypatch, payload):
    fake_claude(tmp_path, monkeypatch, payload)
    llm = llm_mod.LLM(SETTINGS)
    assert ask(llm) is None
    assert llm.failures == 1


def test_auto_prefers_api_when_key_set(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm_mod.pick_backend() == "api"


def test_auto_uses_cli_without_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)            # no .env here
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "auto")
    (tmp_path / "claude").write_text("#!/bin/sh\n")
    (tmp_path / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert llm_mod.pick_backend() == "claude-cli"
