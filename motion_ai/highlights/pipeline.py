"""Match flow: cut clips as events arrive; on powerplay end (or end of input in match scope), curate + render.

HIGHLIGHT_SCOPE=match: POWERPLAY_END is only logged; nothing is dropped after it; the package is built when
the input ends (or on the manual trigger file). No safety timer.

Powerplay-end triggers (first one wins):
  1. Scoreboard OCR reads over 6.0          (only when OCR_AVAILABLE=true)
  2. Commentary says the powerplay is over   (needs the innings to have started)
  3. Manual: create the file  <WORK_DIR>/TRIGGER_POWERPLAY_END
  4. Safety timer: POWERPLAY_MAX_MINUTES after innings start
"""
from __future__ import annotations

import asyncio
import logging

from ..config import Settings
from ..events import CLIP_EVENTS, EventType, FusedEvent, Source
from ..ingest.buffer import SegmentBuffer
from ..llm import LLM
from .clips import Clip, ClipCutter
from .curator import curate
from .renderer import render

log = logging.getLogger(__name__)

AFTER_PP_GRACE_S = 30.0   # include the last ball's reaction/replay after the trigger
LATE_EVENT_WAIT_S = 50.0  # keep accepting events this long after powerplay end (late wicket confirmations)


FACT_EVENTS = {EventType.INNINGS_START, EventType.TARGET, EventType.MATCH_END, EventType.OVER_COMPLETE,
               EventType.WICKET, EventType.POWERPLAY_END}

BALL_EVENTS = {EventType.FOUR, EventType.SIX, EventType.WICKET, EventType.REVIEW,
               EventType.DROPPED_CATCH, EventType.MILESTONE, EventType.EXCITEMENT}


def before_first_ball(ev: FusedEvent, innings_start_t: float | None) -> bool:
    """Ball events can't happen before the innings starts (pre-match chat mentions 'review', 'six'...)."""
    return innings_start_t is not None and ev.type in BALL_EVENTS and ev.t < innings_start_t - 5


class HighlightPipeline:
    def __init__(self, settings: Settings, buffer: SegmentBuffer, llm: LLM, match: dict):
        self.settings = settings
        self.llm = llm
        self.match = match
        self.cutter = ClipCutter(buffer, settings.work_dir / "clips", settings.video_encoder)
        self.clip_tasks: list[asyncio.Task[Clip | None]] = []
        self.innings_start_t: float | None = None
        self.trigger_file = settings.work_dir / "TRIGGER_POWERPLAY_END"
        self.trigger_file.unlink(missing_ok=True)
        self.finished = asyncio.Event()
        self._finishing = False
        self.result = None
        # Only claim "Powerplay" when a powerplay-end signal actually fired
        self.title = "Match highlights"
        self.pp_end_t: float | None = None
        self.stats = lambda: {}   # extra manifest fields (e.g. OCR read failures), set by main/replay
        self.match_scope = getattr(settings, "match_scope", False)
        self.facts: list[FusedEvent] = []   # innings / target / match-end / over events (cards, wicket audit)
        if self.match_scope:
            self.title = "Match highlights"

    def on_event(self, ev: FusedEvent) -> None:
        if self._finishing:
            return
        if ev.type in FACT_EVENTS:
            self.facts.append(ev)
        if ev.type == EventType.INNINGS_START and self.innings_start_t is None:
            self.innings_start_t = ev.t
            log.info("Innings started at t=%.1f", ev.t)
        if ev.type == EventType.POWERPLAY_END and self.match_scope:
            log.info("Powerplay end at t=%.1f (innings %s) — match scope, continuing", ev.t, ev.detail.get("innings"))
            return
        if ev.type == EventType.POWERPLAY_END:
            if self.innings_start_t is None and Source.MANUAL not in ev.sources:
                log.warning("Ignoring powerplay_end before innings start (t=%.1f)", ev.t)
                return
            if self.pp_end_t is None:
                # Don't finish yet: a wicket on the last ball is confirmed ~30s later. Finish at +50s (tick).
                self.title = "Powerplay highlights"
                self.pp_end_t = ev.t
                log.info("Powerplay end detected at t=%.1f (%s); finishing at t=%.1f", ev.t,
                         sorted(s.value for s in ev.sources), ev.t + LATE_EVENT_WAIT_S)
            return
        if self.pp_end_t is not None and ev.t > self.pp_end_t + AFTER_PP_GRACE_S:
            return
        if ev.type in CLIP_EVENTS:
            self.clip_tasks.append(asyncio.create_task(self.cutter.cut(ev)))

    def tick(self, now: float) -> None:
        if self._finishing:
            return
        if self.match_scope:
            if self.trigger_file.exists():
                self._start_finish(now, why="manual trigger file")
            return
        if self.pp_end_t is not None and now >= self.pp_end_t + LATE_EVENT_WAIT_S:
            self._start_finish(self.pp_end_t, why="powerplay end + late-event wait")
        elif self.trigger_file.exists():
            self._start_finish(now, why="manual trigger file")
        elif self.innings_start_t is not None and now - self.innings_start_t > self.settings.powerplay_max_minutes * 60:
            self._start_finish(now, why="safety timer")

    def end_of_input(self, t: float) -> None:
        """Recorded file finished: build the package from whatever was found."""
        if not self._finishing:
            self._start_finish(self.pp_end_t if self.pp_end_t is not None else t, why="end of input file")

    def _start_finish(self, t: float, why: str) -> None:
        self._finishing = True
        log.info("%s at t=%.1f (%s) — building package", "INPUT END" if self.match_scope else "POWERPLAY END", t, why)
        asyncio.create_task(self._finish(t))   # never block the fusion loop (clips need new video)

    async def _finish(self, t_end: float) -> None:
        try:
            clips = [c for c in await asyncio.gather(*self.clip_tasks) if c is not None]
            clips = [c for c in clips if (self.match_scope or c.event.t <= t_end + AFTER_PP_GRACE_S)
                     and not before_first_ball(c.event, self.innings_start_t)]
            log.info("%d clips available for curation", len(clips))
            if self.match_scope:
                self.result = await build_match_package(clips, self.facts, self.settings, self.match,
                                                        self.cutter.buffer, self.stats())
                return
            plan = await curate(self.llm, clips, self.settings.highlight_target_seconds, self.match, self.title)
            self.result = await render(plan, self.settings.output_dir, self.stats())
        finally:
            self.finished.set()


async def build_match_package(clips: list[Clip], fused_facts: list[FusedEvent], settings: Settings, match: dict,
                              buffer: SegmentBuffer, extra: dict | None = None):
    """HIGHLIGHT_SCOPE=match: curate both innings, render with cards, then run the binding quality gate."""
    from pathlib import Path

    from ..media import out_size, probe
    from ..quality import gate
    from .curator import MatchFacts, match_plan
    from .renderer import render_match

    facts = MatchFacts.from_events(fused_facts)
    teams = list((match.get("teams") or {}).keys())
    title = f"{' v '.join(teams)}: Match Highlights" if len(teams) == 2 else "Match highlights"
    mp = match_plan(clips, settings.highlight_target_seconds, facts, title)
    src_path = Path(settings.srt_url)
    source = probe(src_path if src_path.exists() else buffer.segments[0].path)
    size = out_size(source.get("width", 1920), source.get("height", 1080))
    from ..media import video_opts
    extra = (extra or {}) | {"source": source, "encoder_args": video_opts(settings.video_encoder)}
    video = await render_match(mp, facts, match, settings.output_dir, settings.work_dir, settings.video_encoder,
                               size, source.get("fps") or 25.0, extra, buffer.covering)
    if video is not None:
        await gate(video.with_suffix(".json"), settings.work_dir / "clips", buffer.covering, source)
    return video
