"""Detector registry — the single place that decides which sources run.

    OCR_AVAILABLE=false  ->  [audio excitement, commentary ASR+LLM]
    OCR_AVAILABLE=true   ->  [audio excitement, commentary ASR+LLM, scoreboard OCR]

The OCR module (and paddleocr) is imported only when enabled, so a box without OCR
dependencies runs fine.
"""
from __future__ import annotations

import logging

from ..config import Settings
from ..llm import LLM
from .audio_excitement import AudioExcitementDetector
from .base import Detector
from .commentary import CommentaryDetector

log = logging.getLogger(__name__)


def build_detectors(settings: Settings, llm: LLM, match: dict) -> list[Detector]:
    detectors: list[Detector] = [
        AudioExcitementDetector(settings),
        CommentaryDetector(settings, llm, match),
    ]
    if settings.ocr_available:
        from .ocr_scoreboard import ScoreboardOCRDetector

        detectors.append(ScoreboardOCRDetector(settings))
        log.info("OCR_AVAILABLE=true -> scoreboard OCR is the primary source for runs/wickets/overs")
    else:
        log.info("OCR_AVAILABLE=false -> commentary + crowd audio are the event sources")
    return detectors


__all__ = ["Detector", "build_detectors"]
