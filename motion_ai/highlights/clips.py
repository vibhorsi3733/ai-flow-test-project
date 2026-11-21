"""Plan and cut one clip per final event (panel spec: guidance_final).

Wickets (OCR on): call = earliest commentary call, confirm = scoreboard change.
  start  = earliest camera cut in [call-12, call-3] that opens a shot >=2s, else call-10 (+hold no_cut_snap)
  range1 = [start, call+3]                                   delivery + live hit
  range2 = [confirm-10, end]                                 replay + big-screen OUT + reaction
  one range if the gap between them is < 3s
Boundaries: start = earliest cut in [ref-8, ref-3] else ref-7; single range to end.
End rule (R = confirm for wickets, ref for boundaries):
  peak = loudest 0.5s window in [R, R+3]; end = first time after R+3 where loudness stays below
  peak/2 for 1s (2 consecutive windows); clamp [R+3, R+8] boundary / [R+3, R+7] wicket;
  if a camera cut falls in the last 2s, end just before it (no frames of the next shot).
Caps: run-out/review 25s (target 22) · other wicket 20s · boundary 15s.
Trim order: 1) end earlier (not before R+3)  2) start later (not past call-8 / ref-5)  3) hold over_cap.
Caption: '{LABEL} · {runs}/{wkts} ({over} ov)', burned in, shown from confirm (boundary: ref+1).
Gap wickets with no commentary call (fm/guidance_v1): anchor = loudest 0.5s in [confirm-20, confirm-2],
  start = earliest cut in [anchor-10, anchor-4]; neither -> [confirm-16, confirm+4] + hold gap_unanchored. Cap 20s.
Winning runs (MATCH_END) are planned like a boundary.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..events import EventType, FusedEvent
from ..ingest.buffer import SegmentBuffer
from ..media import loudness, render_clip, scene_cuts

log = logging.getLogger(__name__)

BOUNDARIES = {EventType.FOUR, EventType.SIX, EventType.MATCH_END}
WICKETISH = {EventType.WICKET, EventType.REVIEW}
CAP_RUNOUT, TARGET_RUNOUT, CAP_WICKET, CAP_BOUNDARY = 25.0, 22.0, 20.0, 15.0
FRAME = 0.04
DIRECT_HIT_RX = re.compile(r"direct hit", re.I)
RUN_OUT_RX = re.compile(r"run[\s-]?out", re.I)

# Fixed windows for other event types (no snapping): (pre, post)
WINDOWS: dict[EventType, tuple[float, float]] = {
    EventType.TOSS: (4, 16),
    EventType.INNINGS_START: (2, 8),
    EventType.FOUR: (6, 6),
    EventType.SIX: (6, 8),
    EventType.WICKET: (8, 12),
    EventType.DROPPED_CATCH: (6, 8),
    EventType.REVIEW: (4, 14),
    EventType.MILESTONE: (4, 8),
    EventType.EXCITEMENT: (6, 6),
}


@dataclass
class Clip:
    event: FusedEvent
    ranges: list[tuple[float, float]]
    path: Path
    caption: str = ""
    caption_t: float | None = None          # stream time the caption appears
    meta: dict = field(default_factory=dict)

    @property
    def t0(self) -> float:
        return self.ranges[0][0]

    @property
    def t1(self) -> float:
        return self.ranges[-1][1]

    @property
    def duration(self) -> float:
        return sum(b - a for a, b in self.ranges)

    def offset_of(self, t: float) -> float:
        """Map a stream time to time inside the clip (gap -> start of next range)."""
        off = 0.0
        for a, b in self.ranges:
            if t < a:
                return off
            if t <= b:
                return off + (t - a)
            off += b - a
        return max(0.0, off - 1.0)


def caption_for(ev: FusedEvent) -> str:
    d = ev.detail
    quotes = " ".join(q or "" for q in d.get("quotes", []) + [d.get("quote", "")])
    if ev.type == EventType.WICKET:
        if DIRECT_HIT_RX.search(quotes):
            label = "RUN OUT · DIRECT HIT"
        elif RUN_OUT_RX.search(quotes):
            label = "RUN OUT"
        else:
            label = "WICKET"
    elif ev.type == EventType.TOSS and d.get("toss_winner"):
        return f"{d['toss_winner']} win the toss, choose to {d.get('toss_decision') or 'play'}"
    else:
        label = {EventType.FOUR: "FOUR", EventType.SIX: "SIX", EventType.REVIEW: "REVIEW",
                 EventType.DROPPED_CATCH: "DROPPED",
                 EventType.INNINGS_START: "2nd INNINGS" if d.get("innings", 1) >= 2 else "FIRST OVERS",
                 EventType.EXCITEMENT: "BIG MOMENT", EventType.MILESTONE: "MILESTONE",
                 EventType.MATCH_END: "WINNING RUNS"}.get(ev.type, ev.type.value.upper())
    if not d.get("score"):
        return "KEY MOMENT" if ev.needs_review else label
    runs_wkts = d["score"]
    return f"{label} · {runs_wkts} ({d['overs']} ov)" if d.get("overs") else f"{label} · {runs_wkts}"


# Range 1 tail: a transcript line's start is when the commentator BEGINS the call; on the demo the call
# line starts at 70.5s while the throw is still in the air until ~72s. +3s keeps the hit (spec said +1).
RANGE1_TAIL_S = 3.0
MIN_SHOT_S = 2.0   # a "shot" shorter than this is a transition/sting flash, not where a clip should open


def snap_start(cuts: list[float], lo: float, hi: float) -> float | None:
    """Earliest camera cut in [lo, hi] that opens a real shot (next cut >= 2s later)."""
    for i, c in enumerate(cuts):
        if lo <= c <= hi:
            nxt = cuts[i + 1] if i + 1 < len(cuts) else None
            if nxt is None or nxt - c >= MIN_SHOT_S:
                return c
    return None


def end_by_roar(loud: list[tuple[float, float]], R: float, hi: float, cuts: list[float]) -> float:
    """End rule: 1s below half of the [R, R+3] peak, clamped to [R+3, R+hi], snapped before a late cut."""
    peak = max((v for t, v in loud if R <= t <= R + 2.5), default=0.0)
    end = R + hi
    if peak > 0:
        by_t = sorted(loud)
        for i, (t, v) in enumerate(by_t):
            if t < R + 3 or t > R + hi:
                continue
            nxt = [v2 for t2, v2 in by_t if abs(t2 - (t + 0.5)) < 0.051]
            if v < peak / 2 and nxt and nxt[0] < peak / 2:
                end = t
                break
    end = min(max(end, R + 3), R + hi)
    late = [c for c in cuts if end - 2 <= c <= end]
    if late:
        end = late[0] - FRAME
    return round(end, 2)


def plan_ranges(ev: FusedEvent, cuts: list[float], loud: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], float | None]:
    """Pure planning (unit-testable). Returns (ranges, caption_t) and adds hold reasons to ev."""
    d = ev.detail
    if (ev.type == EventType.WICKET and d.get("gap") and d.get("call_t") is None
            and d.get("confirm_t") is not None):
        return _plan_gap_wicket(ev, cuts, loud)
    if ev.type in WICKETISH:
        call = d.get("call_t") if d.get("call_t") is not None else ev.t
        confirm = d.get("confirm_t")
        R = confirm if confirm is not None else call
        snap = snap_start(cuts, call - 12, call - 3)
        start = snap if snap is not None else call - 10
        if snap is None:
            ev.hold("no_cut_snap")
        end = end_by_roar(loud, R, 7, cuts)
        if confirm is not None and (confirm - 10) - (call + RANGE1_TAIL_S) >= 3:
            ranges = [[start, call + RANGE1_TAIL_S], [confirm - 10, end]]
        else:
            ranges = [[start, end]]
        runout = bool({"run_out", "review"} & set(ev.hold_reasons))
        cap = CAP_RUNOUT if runout else CAP_WICKET
        start_limit, caption_t = call - 8, (confirm if confirm is not None else call)
    elif ev.type in BOUNDARIES:
        ref = ev.t
        R = ref
        snap = snap_start(cuts, ref - 8, ref - 3)
        start = snap if snap is not None else ref - 7
        if snap is None:
            ev.hold("no_cut_snap")
        end = end_by_roar(loud, R, 8, cuts)
        ranges = [[start, end]]
        cap, start_limit, caption_t = CAP_BOUNDARY, ref - 5, ref + 1
    else:
        pre, post = WINDOWS.get(ev.type, (6, 6))
        return [(max(0.0, ev.t - pre), ev.t + post)], ev.t

    total = lambda: sum(b - a for a, b in ranges)  # noqa: E731
    over = total() - cap
    if over > 0:   # 1) end earlier, not before R+3 (replay + OUT are before that)
        ranges[-1][1] = round(max(R + 3, ranges[-1][1] - over), 2)
        over = total() - cap
    if over > 0:   # 2) start later, never past call-8 / ref-5
        ranges[0][0] = round(min(start_limit, ranges[0][0] + over), 2)
        over = total() - cap
    if over > 0.05:
        ev.hold("over_cap")
    return [(round(a, 2), round(b, 2)) for a, b in ranges], caption_t


GAP_PEAK_LO, GAP_PEAK_HI = 20.0, 2.0


def _plan_gap_wicket(ev: FusedEvent, cuts: list[float], loud: list[tuple[float, float]]):
    """Scoreboard jumped several balls (highlight edit) and nobody called it: anchor on the crowd."""
    confirm = ev.detail["confirm_t"]
    window = [(t, v) for t, v in loud if confirm - GAP_PEAK_LO <= t <= confirm - GAP_PEAK_HI]
    anchor = max(window, key=lambda x: x[1])[0] if window else None
    start = snap_start(cuts, anchor - 10, anchor - 4) if anchor is not None else None
    if start is None:
        ev.hold("gap_unanchored")
        ranges = [(confirm - 16, confirm + 4)]
    else:
        ev.detail["gap_anchor_t"] = round(anchor, 2)
        ranges = [(start, min(confirm + 4, start + CAP_WICKET))]
    ev.hold("no_call_anchor")
    return [(round(a, 2), round(b, 2)) for a, b in ranges], confirm


class ClipCutter:
    def __init__(self, buffer: SegmentBuffer, out_dir: Path, encoder: str, max_parallel: int = 2):
        self.buffer = buffer
        self.out_dir = out_dir
        self.encoder = encoder
        self._sem = asyncio.Semaphore(max_parallel)
        self.cache_path = out_dir.parent / "cuts.json"
        try:
            self._cuts_cache: dict[str, list[float]] = json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            self._cuts_cache = {}

    def _segs(self, t0: float, t1: float):
        return self.buffer.covering(t0, t1)

    async def _cuts(self, t0: float, t1: float) -> list[float]:
        key = f"{t0:.1f}-{t1:.1f}"
        if key not in self._cuts_cache:
            segs = self._segs(t0, t1)
            if not segs:
                return []
            self._cuts_cache[key] = await scene_cuts([s.path for s in segs], segs[0].start, t0, t1, self.out_dir)
            self.cache_path.write_text(json.dumps(self._cuts_cache))
        return self._cuts_cache[key]

    async def plan(self, ev: FusedEvent) -> Clip | None:
        d = ev.detail
        loud_lo = None
        if ev.type == EventType.WICKET and d.get("gap") and d.get("call_t") is None and d.get("confirm_t"):
            R = d["confirm_t"]
            lo, hi, loud_lo = R - GAP_PEAK_LO - 11, R + 8, R - GAP_PEAK_LO
        elif ev.type in WICKETISH:
            call = d.get("call_t") if d.get("call_t") is not None else ev.t
            R = d.get("confirm_t") if d.get("confirm_t") is not None else call
            lo, hi = call - 13, R + 8
        elif ev.type in BOUNDARIES:
            R, lo, hi = ev.t, ev.t - 9, ev.t + 9
        else:
            ranges, cap_t = plan_ranges(ev, [], [])
            return Clip(ev, ranges, self._path(ev), caption_for(ev), cap_t)
        if not await self.buffer.wait_until(hi + 1):
            log.warning("Video for %s at %.1f never arrived; skipping clip", ev.type.value, ev.t)
            return None
        lo, hi = max(0.0, lo), min(hi, self.buffer.latest)
        segs = self._segs(lo, hi)
        if not segs:
            return None
        cuts = await self._cuts(lo, hi)
        l0 = max(lo, loud_lo) if loud_lo is not None else R
        loud = await loudness([s.path for s in segs], segs[0].start, l0, min(R + 9, hi), self.out_dir)
        ranges, cap_t = plan_ranges(ev, cuts, loud)
        return Clip(ev, ranges, self._path(ev), caption_for(ev), cap_t, {"cuts": cuts})

    def _path(self, ev: FusedEvent) -> Path:
        return self.out_dir / f"{int(ev.t):06d}_{ev.type.value}_{ev.id}.mp4"

    async def render(self, clip: Clip) -> Clip | None:
        lo, hi = clip.t0, min(clip.t1, self.buffer.latest)
        clip.ranges = [(a, min(b, hi)) for a, b in clip.ranges if a < hi]
        segs = self._segs(lo, hi)
        if not segs or clip.duration < 2:
            return None
        cap_off = clip.offset_of(clip.caption_t) if clip.caption_t is not None else 0.0
        async with self._sem:
            try:
                await render_clip([s.path for s in segs], segs[0].start, clip.ranges, clip.path,
                                  clip.caption, cap_off, self.encoder)
            except RuntimeError as exc:
                log.error("Clip render failed for %s: %s", clip.event.type.value, exc)
                return None
        clip.path.with_suffix(".json").write_text(json.dumps(sidecar(clip), indent=1, default=str))
        log.info("Clip ready: %s %s (%.1fs) holds=%s", clip.path.name, clip.ranges, clip.duration,
                 clip.event.hold_reasons)
        return clip

    async def cut(self, ev: FusedEvent) -> Clip | None:
        clip = await self.plan(ev)
        return await self.render(clip) if clip else None


def sidecar(clip: Clip) -> dict:
    ev = clip.event
    return {"id": ev.id, "type": ev.type.value, "t": ev.t, "confidence": ev.confidence,
            "sources": sorted(s.value for s in ev.sources), "hold_reasons": ev.hold_reasons,
            "detail": {k: v for k, v in ev.detail.items() if not k.startswith("_")},
            "segments": clip.ranges, "caption": clip.caption, "caption_t": clip.caption_t,
            "caption_offset": clip.offset_of(clip.caption_t) if clip.caption_t is not None else None,
            "duration": round(clip.duration, 2)}
