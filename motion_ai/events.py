"""Event model shared by all detectors, the fuser and the highlight pipeline.

All times are *stream seconds*: seconds since ingest started (the master clock).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    TOSS = "toss"
    INNINGS_START = "innings_start"
    FOUR = "four"
    SIX = "six"
    WICKET = "wicket"
    DROPPED_CATCH = "dropped_catch"
    REVIEW = "review"
    MILESTONE = "milestone"
    OVER_COMPLETE = "over_complete"
    POWERPLAY_END = "powerplay_end"
    TARGET = "target"          # chase target read from the scoreboard
    MATCH_END = "match_end"    # chasing side reached the target
    INNINGS_END = "innings_end"  # confirmed innings total (scorecard footer), e.g. 179/7 in 20.0 overs
    EXCITEMENT = "excitement"  # crowd/commentary peak with no identified cricket event


class Source(str, Enum):
    OCR = "ocr"
    ASR = "asr"        # commentary speech-to-text + LLM
    AUDIO = "audio"    # crowd / loudness excitement
    MANUAL = "manual"
    TIMER = "timer"


# Events that become clips in the highlights package
CLIP_EVENTS = {
    EventType.TOSS, EventType.INNINGS_START, EventType.FOUR, EventType.SIX,
    EventType.WICKET, EventType.DROPPED_CATCH, EventType.REVIEW,
    EventType.MILESTONE, EventType.EXCITEMENT, EventType.MATCH_END,
}

# Ball-outcome / scoreboard facts the scoreboard OCR can settle authoritatively
OCR_AUTHORITATIVE = {
    EventType.FOUR, EventType.SIX, EventType.WICKET,
    EventType.OVER_COMPLETE, EventType.POWERPLAY_END, EventType.INNINGS_START,
    EventType.TARGET, EventType.MATCH_END, EventType.INNINGS_END,
}


@dataclass
class Event:
    """A raw detection from a single source."""
    type: EventType
    t: float
    source: Source
    confidence: float
    detail: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class FusedEvent:
    """A final, de-duplicated event after combining sources."""
    type: EventType
    t: float
    confidence: float
    sources: set[Source]
    detail: dict = field(default_factory=dict)
    needs_review: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Why a human must check this before publishing (empty = auto-publishable).
    # run_out | review | asr_only | no_call_anchor | no_cut_snap | slow_confirm | ocr_low_conf | over_cap
    # | multi_wicket_gap | innings_inferred | gap_unanchored
    hold_reasons: list[str] = field(default_factory=list)

    def hold(self, reason: str) -> None:
        if reason not in self.hold_reasons:
            self.hold_reasons.append(reason)
        self.needs_review = True
