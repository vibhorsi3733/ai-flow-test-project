"""Commentary → events: faster-whisper transcribes each segment; a cheap keyword filter
decides when to ask Claude to turn the last ~30s of transcript into structured events.

Works in Hindi, English and Hinglish. Player names from config/match.json are passed
to Whisper as a prompt so they are spelled correctly.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from ..config import Settings
from ..events import Event, EventType, Source
from ..ingest.srt_ingest import Segment
from ..llm import LLM
from ..media import read_pcm
from .base import Detector

log = logging.getLogger(__name__)

TRIGGER = re.compile(
    r"\b(six|sixes|maximum|four|boundary|out|bowled|caught|lbw|stumped|run out|wicket|"
    r"review|drs|dropped|put down|toss|elected|power ?play|fifty|half.century|direct hit|third umpire|"
    r"chhakka|chakka|chauka|chaukaa)\b|विकेट|छक्का|चौका|आउट|टॉस|पावरप्ले|कैच",
    re.IGNORECASE,
)
FOLLOW_UP_S = 6.0      # wait for the rest of the commentator's sentence
CONTEXT_S = 30.0       # transcript window sent to the LLM

LLM_TYPES = [
    "toss", "innings_start", "four", "six", "wicket", "dropped_catch",
    "review", "milestone", "powerplay_end",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": LLM_TYPES},
                    "line": {"type": "integer", "description": "Index of the transcript line where it happens"},
                    "confidence": {"type": "number"},
                    "summary": {"type": "string", "description": "One short English sentence"},
                    "players": {"type": "array", "items": {"type": "string"}},
                    "toss_winner": {"type": "string"},
                    "toss_decision": {"type": "string", "enum": ["bat", "bowl", ""]},
                },
                "required": ["type", "line", "confidence", "summary", "players",
                             "toss_winner", "toss_decision"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}

SYSTEM = """You read live IPL TV commentary transcripts (Hindi, English or mixed) and extract cricket events
that HAPPENED in this window. Rules:
- Only report events the commentators describe as happening now. Ignore replays being discussed,
  past overs, statistics, predictions and "nearly a six" style near-misses (a near-miss catch that was
  put down IS a dropped_catch).
- 'line' is the index of the line where the event is first called.
- confidence: 0.9+ when explicit ("that's out!", "six!"), 0.5-0.8 when implied, below 0.5 if unsure.
- toss_winner/toss_decision only for type=toss; otherwise empty strings.
- powerplay_end only when they say the powerplay has just finished.
- Return an empty list if nothing happened. Players: use the exact spellings from the squad list."""


@dataclass
class Line:
    t0: float
    t1: float
    text: str


class CommentaryDetector(Detector):
    name = "commentary"

    def __init__(self, settings: Settings, llm: LLM, match: dict):
        super().__init__()
        self.settings = settings
        self.llm = llm
        self.match = match
        self.lines: list[Line] = []
        self._pending_trigger: float | None = None
        self._last_checked_until = 0.0
        # (event type, line start) already reported: the next window re-sends ~5 s of context lines, which
        # made the same call come back twice (e.g. wicket at 105.12 reported in two windows).
        self._reported: set[tuple[str, float]] = set()
        players = [p for squad in match.get("teams", {}).values() for p in squad]
        self.prompt = f"IPL cricket commentary. {match.get('match', '')}. Players: {', '.join(players)}."
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            import os

            threads = getattr(self.settings, "asr_cpu_threads", 0) or (os.cpu_count() or 4)
            self._model = WhisperModel(self.settings.asr_model, device=self.settings.asr_device,
                                       compute_type=self.settings.asr_compute_type, cpu_threads=threads)
        return self._model

    def _transcribe(self, pcm) -> list[tuple[float, float, str]]:
        segments, _ = self._load_model().transcribe(
            pcm, language=self.settings.asr_language, vad_filter=True,
            initial_prompt=self.prompt, beam_size=1, condition_on_previous_text=False,
        )
        return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]

    async def process(self, seg: Segment) -> list[Event]:
        pcm = await read_pcm(seg.path)
        if len(pcm):
            for t0, t1, text in await asyncio.to_thread(self._transcribe, pcm):
                self.lines.append(Line(seg.start + t0, seg.start + t1, text))
                if TRIGGER.search(text) and self._pending_trigger is None:
                    self._pending_trigger = seg.start + t0
        self.lines = [ln for ln in self.lines if ln.t1 > seg.end - 300]  # keep 5 min

        events: list[Event] = []
        if self._pending_trigger is not None and seg.end >= self._pending_trigger + FOLLOW_UP_S:
            events = self._new_only(
                await self._extract(max(self._pending_trigger - CONTEXT_S, self._last_checked_until - 5), seg.end))
            self._last_checked_until = seg.end
            self._pending_trigger = None
        self.watermark = seg.end
        return events

    def _new_only(self, events: list[Event]) -> list[Event]:
        """Drop events already reported from the same transcript line (context overlap between windows)."""
        fresh = []
        for ev in events:
            key = (ev.type.value, round(ev.t, 2))
            if key not in self._reported:
                self._reported.add(key)
                fresh.append(ev)
        return fresh

    async def _extract(self, t0: float, t1: float) -> list[Event]:
        window = [ln for ln in self.lines if ln.t1 >= t0 and ln.t0 <= t1]
        if not window:
            return []
        transcript = "\n".join(f"[{i}] {ln.text}" for i, ln in enumerate(window))
        squads = "; ".join(f"{team}: {', '.join(p)}" for team, p in self.match.get("teams", {}).items())
        result = await self.llm.json(SYSTEM, f"Squads: {squads}\n\nTranscript:\n{transcript}", SCHEMA)
        if result is None:
            return self._keyword_fallback(window)

        events = []
        for e in result.get("events", []):
            idx = min(max(e["line"], 0), len(window) - 1)
            detail = {"summary": e["summary"], "players": e["players"], "quote": window[idx].text}
            if e["type"] == "toss":
                detail |= {"toss_winner": e["toss_winner"], "toss_decision": e["toss_decision"]}
            events.append(Event(EventType(e["type"]), window[idx].t0, Source.ASR,
                                float(e["confidence"]), detail))
        if events:
            log.info("Commentary events: %s", [(ev.type.value, round(ev.t, 1)) for ev in events])
        return events

    @staticmethod
    def _keyword_fallback(window: list[Line]) -> list[Event]:
        """If the LLM is unavailable: low-confidence events from obvious keywords only."""
        rules = [
            (re.compile(r"\b(sixes|six|maximum|into the stands|out of the park|chhakka)\b|छक्का", re.I), EventType.SIX),
            (re.compile(r"\b(fours|four|boundary|chauka)\b|चौका", re.I), EventType.FOUR),
            # "wicket" alone is too common in commentary ("wicket keeper", "good wicket to bat on")
            (re.compile(r"\b(bowled him|clean bowled|caught|lbw|stumped|run out|run-out|that's out|"
                        r"he's out|he is out|gone|wicket down|first wicket|second wicket|"
                        r"big wicket|takes the wicket|direct hit)\b|विकेट|आउट", re.I), EventType.WICKET),
            (re.compile(r"\b(review|third umpire|decision pending|upstairs)\b", re.I), EventType.REVIEW),
            (re.compile(r"won the toss|elected to|टॉस", re.I), EventType.TOSS),
        ]
        keeper = re.compile(r"wicket[\s-]?keep", re.I)
        events = []
        for ln in window:
            text = keeper.sub("keeper", ln.text)
            for rx, et in rules:
                if rx.search(text):
                    events.append(Event(et, ln.t0, Source.ASR, 0.4, {"quote": ln.text, "fallback": True}))
                    break
        return events
