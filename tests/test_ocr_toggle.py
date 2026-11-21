import sys

import pytest

from motion_ai.config import Settings
from motion_ai.detectors import build_detectors
from motion_ai.llm import LLM


def test_ocr_off_never_loads_ocr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    s = Settings(srt_url="srt://x", ocr_available=False, _env_file=None)
    names = [d.name for d in build_detectors(s, LLM(s), {})]
    assert names == ["audio", "commentary"]
    assert "motion_ai.detectors.ocr_scoreboard" not in sys.modules


def test_ocr_on_without_setup_fails_fast(tmp_path):
    s = Settings(srt_url="srt://x", ocr_available=True,
                 ocr_config_path=tmp_path / "missing.json", _env_file=None)
    with pytest.raises(SystemExit, match="OCR_AVAILABLE=true"):
        s.validate_ocr_setup()


def test_env_flag_parsing(monkeypatch):
    monkeypatch.setenv("SRT_URL", "srt://x")
    monkeypatch.setenv("OCR_AVAILABLE", "true")
    assert Settings(_env_file=None).ocr_available is True
    monkeypatch.setenv("OCR_AVAILABLE", "false")
    assert Settings(_env_file=None).ocr_available is False
