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

Edited-source rules (caption check report, scratchpad/caption_check/report.md — the demo is a highlights edit, so
the scoreboard change confirming an event can come several balls after it):
  Gap wicket, no call, prev_change_t known: the dismissal is shown right after the PREVIOUS change.
    start = latest cut in [prev-1.5, prev+0.3], else prev-0.2 (prev-1.5 after the innings-start read)
      (+hold no_cut_snap)
    end   = confirm - 0.12 (the next ball follows the board change), capped at start+25, before a late cut
    caption_t = loudest 0.5s in [prev+1, prev+15] + 1s, not before prev+8 (the WICKET graphic comes
      prev+8.7..10.6: 139/4 275.1, 31/1 443, 104/2 484; catches end by prev+8)
  Gap wicket with a call: ranges as above but range 2 ends at confirm-0.12; caption at range-2 start
    (dismissal card / replay; 116/3: card at 224.4, the 14.1 ball runs live from 234.4).
  Any wicket: never start more than 0.5s before prev_change_t (the previous ball's aftermath).
  Non-gap wicket whose runs moved with the wicket (score_basis "unknown"): play resumed before the board
    changed, so end <= confirm + 0.24 (179/6: board 362.06, the next ball released 363.1).
  OCR-only boundary (no commentary call before the board change): shot = loudest 0.5s in
    [max(confirm-16, prev+0.5), confirm-6]; when the previous change is within 16s of the confirm the shot is
    the first ball after it: start = latest cut in [prev-1.5, prev+0.3], else prev-0.2; otherwise start snapped
    from the shot as before; end <= confirm - 0.04;
    caption at shot + 2.5 (161/4: shot 288.6, ball still running at 292). ASR-anchored boundary: caption at ref + 2.5 (6/0: call 56.8, rope 58.5-59.5).
  avoid_ranges (config/match.json, stream seconds): no frame inside one is used (SUBSCRIBE pop-up 494-497.5).
  Caption: clamped into the clip, >= 3s before its end (CAPTION_MIN_S; fix round 2: 178/6 and the 177/5 six
    showed 1.5s). Where the planned time is too late, the end moves out (boundaries) or the caption moves to
    end - CAPTION_HOLD_S (wickets confirmed at the clip end).
  Non-gap wicket with prev_change_t, no earlier call: the dismissal is the first ball(s) after the previous
    change, so when [prev, end] fits the cap the clip starts at the previous change like a gap wicket
    (178/6: change 339.42, wide bowled 341.5, run out at the bowler's end ~345, WICKET ticker 355, replay
    356-361, board 362.06; the old start 345.83 opened on the umpire signalling the wide). Dead time is cut:
    range 1 = [start, first cut in [start+7, start+10]] (live ball + outcome), range 2 = from the first cut in
    [confirm-10, confirm-6] (OUT / replay) when the gap between them is >= 3s.
  Wicket confirmed only by the INNINGS_END scorecard footer (no board change): the full-screen card fades in
    INNINGS_CARD_LEAD_S before the footer read, so end <= innings_end_t - 3.3 (179/7: stumps broken 397.4,
    card from 397.75, footer read 400.96).
  OCR-only boundary on the first ball after the previous change: hit = min(crowd peak, prev + 1.5) (a later
    peak is the celebration: winning runs peak 601.4, hit 595.0); caption at hit + 2.5 (four: ball reaches
    the rope; winning runs rope ~597-598.5) or hit + 0.5 for a SIX (177/5: hit 582.5, in the stands 583.4);
    the end moves out towards caption + 6s, never past the board change (the next ball cannot start
    earlier), before a late cut (177/5: 589.3; roar rule alone ended it 587.56).
  MATCH_END label: "WINNING RUNS · FOUR"/"· SIX" from the event's own runs (not a gap), decided here so the
    burned caption, manifest and .srt agree (the curator fold ran after the burn).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..events import EventType, FusedEvent, Source
from ..fusion.fuser import annotate_dismissal
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


# On-screen captions carry NO numbers or data-derived names: OCR scores/overs were wrong too often
# (e.g. "WICKET · 31/1 (2.5 ov)" over a 69/1 (7.4) board). Score/overs stay in the manifest as data.
# Set True to bring "LABEL · score (overs ov)", ordinal wickets and the toss winner back on screen.
CAPTION_SHOW_SCORE = False
_DIGIT_WORD_RX = re.compile(r"\S*\d\S*")


def strip_numbers(caption: str) -> str:
    """Drop every " · " part that contains a digit (score, overs, ordinal), then any leftover digit word."""
    if CAPTION_SHOW_SCORE:
        return caption
    parts = [p.strip() for p in caption.split("·")]
    kept = [p for p in parts if p and not any(ch.isdigit() for ch in p)]
    text = " · ".join(kept)
    return " ".join(_DIGIT_WORD_RX.sub("", text).split())


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
        if not CAPTION_SHOW_SCORE:
            return "TOSS"
        return f"{d['toss_winner']} win the toss, choose to {d.get('toss_decision') or 'play'}"
    else:
        label = {EventType.FOUR: "FOUR", EventType.SIX: "SIX", EventType.REVIEW: "REVIEW",
                 EventType.DROPPED_CATCH: "DROPPED",
                 EventType.INNINGS_START: "FIRST BALL",
                 EventType.EXCITEMENT: "BIG MOMENT", EventType.MILESTONE: "MILESTONE",
                 EventType.MATCH_END: "WINNING RUNS"}.get(ev.type, ev.type.value.upper())
    if ev.type == EventType.WICKET:
        annotate_dismissal(ev)                       # data side (event_score, holds) stays either way
        if d.get("score_basis") == "unknown":       # never print a score that may be wrong
            n = d.get("event_wicket_no")
            if not n or not CAPTION_SHOW_SCORE:
                return label
            return f"{ordinal(n)} WICKET" if label == "WICKET" else f"{label} · {ordinal(n)} WICKET"
    elif ev.type in BOUNDARIES and d.get("gap"):
        return label
    if ev.type == EventType.MATCH_END:
        label = winning_label(d)                                 # board jumped several balls: score not of this shot
    score = d.get("event_score") or d.get("score")
    overs = d.get("event_overs") or d.get("overs")
    if not score:
        return "KEY MOMENT" if ev.needs_review else label
    if not CAPTION_SHOW_SCORE:
        return label
    return f"{label} · {score} ({overs} ov)" if overs else f"{label} · {score}"


WINNING_SHOT = {4: "FOUR", 6: "SIX"}


def winning_label(d: dict) -> str:
    """'WINNING RUNS · FOUR' when the winning board change was one boundary ball (not a gap), else 'WINNING RUNS'."""
    try:
        shot = None if d.get("gap") else WINNING_SHOT.get(int(d.get("runs")))
    except (TypeError, ValueError):
        shot = None
    return f"WINNING RUNS · {shot}" if shot else "WINNING RUNS"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


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


PREV_LEAD_S = 0.5        # never start more than this before the previous scoreboard change: the next delivery
                         # can come up to ~0.4s before the change is read (94/2: delivery 151.5, change 151.84)
GAP_CAP = CAP_RUNOUT     # gap wicket: delivery + dismissal + first replay
BOARD_NEXT_BALL_S = 0.12  # end this long before a board change that is followed by the next ball
UNKNOWN_SCORE_TAIL_S = 0.24
SHOT_LO, SHOT_HI = 16.0, 6.0   # OCR-only boundary: shot is the crowd peak in [confirm-16, confirm-6]
ASR_CAPTION_S = 2.5      # boundary call -> ball reaches the rope (6/0: call 56.8, rope 58.5-59.5)
CAPTION_MIN_S = 3.0      # every caption is on screen at least this long inside the clip
CAPTION_TAIL_S = CAPTION_MIN_S   # (old name)
CAPTION_HOLD_S = 4.0     # preferred hold when a caption has to move back from the clip end
CAPTION_PREF_S = 6.0     # boundary on the first ball after a change: tail kept until caption + 6s (<= board change)
LIVE_S = 7.0             # first ball after a change: run-up + delivery + live outcome (178/6: 339.4 -> run out ~345)
SIX_CAPTION_S = 0.5      # hit -> ball in the stands (177/5: hit 582.5, stands 583.4)
NEAR_PREV_HIT_S = 1.5    # first ball after a change is hit by prev + ~1.0-1.4 (161/4 288.6, 177/5 582.5, 181/5 595.0)
INNINGS_CARD_LEAD_S = 3.3   # full-screen batting card fades in this long before its footer is read (397.75 / 400.96)
FIRST_BALL_LEAD_S = 0.2   # no cut found: start this long before the change (earlier frames show the previous
                          # state: 139/4 at 265.9 still shows "116-3 14 / DECISION UPHELD", change 266.36)
INNINGS_START_LEAD_S = 1.5   # ... unless the previous change is the (slow-confirmed) innings-start read
GAP_SNAP_LO, GAP_SNAP_HI = 1.5, 0.3   # "first ball after the change" starts on the latest cut in [prev-1.5, prev+0.3]
                                     # (177/5: a cut at prev-2.9 still shows the previous ball, bar 161-4 18.1)


def loudest(loud: list[tuple[float, float]], lo: float, hi: float) -> float | None:
    """Start time of the loudest 0.5s window in [lo, hi]."""
    w = [(t, v) for t, v in loud if lo <= t <= hi]
    return max(w, key=lambda x: x[1])[0] if w else None


def _prev_floor(d: dict) -> float | None:
    p = d.get("prev_change_t")
    return None if p is None else p - PREV_LEAD_S


def _first_ball_lead(d: dict) -> float:
    return INNINGS_START_LEAD_S if d.get("prev_is_innings_start") else FIRST_BALL_LEAD_S


def _is_gap_wicket(ev: FusedEvent) -> bool:
    d = ev.detail
    return ev.type == EventType.WICKET and bool(d.get("gap")) and d.get("confirm_t") is not None


def _ocr_only_boundary(ev: FusedEvent) -> bool:
    """No commentary call before the board change: ev.t is just confirm - OCR lag (12-14s after the shot)."""
    d = ev.detail
    confirm = d.get("confirm_t")
    if confirm is None:
        return False
    return Source.ASR not in ev.sources or ev.t >= confirm - 2.0


def plan_ranges(ev: FusedEvent, cuts: list[float], loud: list[tuple[float, float]],
                avoid: list[tuple[float, float]] | tuple = ()) -> tuple[list[tuple[float, float]], float | None]:
    """Pure planning (unit-testable). Returns (ranges, caption_t) and adds hold reasons to ev.

    avoid ranges are cut out and the caption is clamped into the final clip (always visible)."""
    ranges, cap_t = _plan(ev, cuts, loud)
    ranges = apply_avoid(ranges, avoid)
    return ranges, clamp_caption(ranges, cap_t)


def _plan(ev: FusedEvent, cuts: list[float], loud: list[tuple[float, float]]):
    d = ev.detail
    if ev.type == EventType.WICKET:
        annotate_dismissal(ev)
    if _is_gap_wicket(ev) and d.get("prev_change_t") is not None and (
            d.get("call_t") is None or not ({"run_out", "review"} & set(ev.hold_reasons))):
        # Gap wicket: the live dismissal follows the previous board change. A commentary call here is
        # usually spoken over the replay (104/2: catch 475-481, call 493.4; 31/1: catch 432-439, call
        # 444.7), so it must not move the clip — it only decides the label. Run-outs / reviews keep the
        # call branch: their decision (big-screen OUT) comes ~10 s before the board change (116/3).
        return _plan_gap_wicket_prev(ev, cuts, loud)
    if _is_gap_wicket(ev) and d.get("call_t") is None:
        return _plan_gap_wicket(ev, cuts, loud)
    floor = _prev_floor(d)
    if ev.type in WICKETISH:
        call = d.get("call_t") if d.get("call_t") is not None else ev.t
        confirm = d.get("confirm_t")
        gap = _is_gap_wicket(ev)
        R = confirm if confirm is not None else call
        snap = snap_start(cuts, call - 12, call - 3)
        start = snap if snap is not None else call - 10
        if snap is None:
            ev.hold("no_cut_snap")
        if floor is not None and floor < call - 1.5:
            start = max(start, floor)
        if gap:   # the edit goes straight from the replay to the next ball at the board change
            end = round(confirm - BOARD_NEXT_BALL_S, 2)
        else:
            end = end_by_roar(loud, R, 7, cuts)
            if confirm is not None and d.get("score_basis") == "unknown":
                end = min(end, round(confirm + UNKNOWN_SCORE_TAIL_S, 2))
            if confirm is None and d.get("innings_end_t") is not None:   # last ball: before the batting card
                end = min(end, round(d["innings_end_t"] - INNINGS_CARD_LEAD_S, 2))
                R = min(R, end - CAPTION_MIN_S)
        runout = bool({"run_out", "review"} & set(ev.hold_reasons))
        cap = CAP_RUNOUT if runout or gap else CAP_WICKET
        start_limit = call - 8
        prev = d.get("prev_change_t")
        from_prev = None
        if (not gap and prev is not None and confirm is not None
                and (d.get("call_t") is None or d["call_t"] >= prev)):
            in_win = [c for c in cuts if prev - GAP_SNAP_LO <= c <= prev + GAP_SNAP_HI]
            from_prev = in_win[-1] if in_win else prev - _first_ball_lead(d)
            if end - from_prev > cap:
                from_prev = None
        if from_prev is not None:   # the ball after the previous change (live) + OUT / replay before the board
            ranges = [[from_prev, end]]
            start_limit = from_prev
            live = [c for c in cuts if from_prev + LIVE_S <= c <= from_prev + LIVE_S + 3]
            live_end = round(live[0] - FRAME, 2) if live else round(from_prev + LIVE_S + 1, 2)
            back = [c for c in cuts if confirm - 10 <= c <= confirm - 6]
            r2 = back[0] if back else confirm - 10
            if r2 - live_end >= 3:   # skip the dead time between (178/6: umpire signals the wide 348.8-353.6)
                ranges = [[from_prev, live_end], [r2, end]]
        elif confirm is not None and (confirm - 10) - (call + RANGE1_TAIL_S) >= 3:
            ranges = [[start, call + RANGE1_TAIL_S], [confirm - 10, end]]
        else:
            ranges = [[start, end]]
        if gap:   # caption on the dismissal card / replay, not on a board change that is off screen
            caption_t = ranges[1][0] + 0.1 if len(ranges) == 2 else call + 1.5
            R = min(R, end - 3)
        else:
            caption_t = confirm if confirm is not None else call
            out = _trim(ev, ranges, cap, R, start_limit)
            end = out[-1][1]
            if caption_t > end - CAPTION_MIN_S + 1e-6:   # confirmed at the very end: hold it on the replay
                caption_t = max(out[0][0], end - (CAPTION_HOLD_S if confirm is not None else CAPTION_MIN_S))
            elif confirm is None and d.get("innings_end_t") is not None:
                # last ball, cut before the batting card: as close to the stumps breaking as 3s allows
                # (179/7: call 394.2, stumps 397.4, end 397.66 -> 394.66)
                caption_t = max(caption_t, end - CAPTION_MIN_S)
            return out, caption_t
    elif ev.type in BOUNDARIES:
        confirm = d.get("confirm_t")
        R = ev.t
        prev = d.get("prev_change_t")
        ocr_only = _ocr_only_boundary(ev)
        # OCR-only and the previous change is close: the shot is the first ball after it (161/4: change
        # 287.2, run-up 287.3, shot 288.6, board 300.16; winning runs: change 593.56, delivery 594.0)
        near_prev = ocr_only and prev is not None and not d.get("gap") and confirm - prev <= SHOT_LO
        if ocr_only:
            lo = max(confirm - SHOT_LO, prev + PREV_LEAD_S) if near_prev else confirm - SHOT_LO
            shot = loudest(loud, lo, confirm - SHOT_HI)
            ref = shot if shot is not None else ev.t
            if shot is not None:
                d["shot_t"] = round(shot, 2)
        else:
            ref = ev.t
        if near_prev:   # the boundary ball follows the change; a later crowd peak is the celebration
            ref = min(ref, prev + NEAR_PREV_HIT_S)
            d["hit_t"] = round(ref, 2)
        caption_t = ref + (SIX_CAPTION_S if near_prev and ev.type == EventType.SIX else ASR_CAPTION_S)
        if near_prev:
            in_win = [c for c in cuts if prev - GAP_SNAP_LO <= c <= prev + GAP_SNAP_HI]
            start = in_win[-1] if in_win else prev - _first_ball_lead(d)
            if not in_win:
                ev.hold("no_cut_snap")
            start_limit = start            # never trim the delivery away
        else:
            snap = snap_start(cuts, ref - 8, ref - 3)
            start = snap if snap is not None else ref - 7
            if snap is None:
                ev.hold("no_cut_snap")
            start_limit = ref - 5
        end = end_by_roar(loud, R, 8, cuts)
        if near_prev:   # nothing new starts before the board change: give the caption CAPTION_PREF_S if it can
            want = round(min(confirm - FRAME, caption_t + CAPTION_PREF_S), 2)
            if want > end:
                late = [c for c in cuts if want - 2 <= c <= want and c - caption_t >= CAPTION_HOLD_S]
                end = round(late[0] - FRAME, 2) if late else want
        if ocr_only:
            end = min(end, round(confirm - FRAME, 2))   # 161/4: the next ball starts at the change (300.2)
            R = min(R, confirm - SHOT_HI)
        R = min(R, end - 3)
        ranges = [[start, end]]
        cap = CAP_BOUNDARY
    elif ev.type == EventType.INNINGS_START:
        return _plan_first_ball(ev, cuts)
    else:
        pre, post = WINDOWS.get(ev.type, (6, 6))
        return [(max(0.0, ev.t - pre), ev.t + post)], ev.t

    return _trim(ev, ranges, cap, R, start_limit), caption_t


# First ball of each innings (mandatory, shown right after the innings card).
# With a commentary call ("Starting the first ball coming up", 412.72) the delivery follows the call: the clip
# runs FIRST_BALL_LEN_S from just after it (chase: delivery 413, shot 414, chase + throw 415-421, back at the
# pitch 422-424). Without a call, the event time is the first board read minus the OCR lag; the first read
# lags the first delivery by ~10 s (innings 1: run-up 45, delivery 46.2, shot 47-48, read "2-0 0.2" at 56.08),
# so the clip starts FIRST_BALL_BEFORE_READ_S before the read and ends on the last cut before the read.
FIRST_BALL_LEN_S = 11.5
FIRST_BALL_BEFORE_READ_S = 11.0
FIRST_BALL_MIN_S = 5.0


def _plan_first_ball(ev: FusedEvent, cuts: list[float]):
    d = ev.detail
    read = d.get("confirm_t")
    if Source.ASR in ev.sources:
        start = round(ev.t + 0.2, 2)
        end = start + FIRST_BALL_LEN_S
        if read is not None and read > start:
            end = min(end, read - FRAME)
    else:
        read = read if read is not None else ev.t + 6.0
        start = round(max(0.0, read - FIRST_BALL_BEFORE_READ_S), 2)
        after = [c for c in cuts if start + FIRST_BALL_MIN_S <= c <= read - 1.0]
        end = after[0] - FRAME if after else read - 4.0
    end = round(max(end, start + FIRST_BALL_MIN_S), 2)
    return [(start, end)], round(min(start + 3.0, end - CAPTION_MIN_S), 2)


def _trim(ev: FusedEvent, ranges: list[list[float]], cap: float, R: float, start_limit: float):
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
    return [(round(a, 2), round(b, 2)) for a, b in ranges]


GAP_PEAK_LO, GAP_PEAK_HI = 20.0, 2.0
GAP_CAPTION_LO, GAP_CAPTION_HI, GAP_CAPTION_MIN = 1.0, 15.0, 8.0


def _plan_gap_wicket_prev(ev: FusedEvent, cuts: list[float], loud: list[tuple[float, float]]):
    """Gap wicket without a call: the dismissal is the first ball after the previous board change.

    Evidence (stream s): 69/1 catch 432-439 (prev 433.0); 112/2 catch 475-481 (prev 473.36); 127/3 catch
    507-514 (prev 502.52); 155/4 catch 537.5-541 (prev 533.12); 171/5 bowled 564 (prev 562.68). The replay
    ends just before the change (69/1: replay to 460.5, change 461.2, next ball 462.8)."""
    d = ev.detail
    prev, confirm = d["prev_change_t"], d["confirm_t"]
    in_win = [c for c in cuts if prev - GAP_SNAP_LO <= c <= prev + GAP_SNAP_HI]
    if in_win:
        start = in_win[-1]
    else:
        start = prev - _first_ball_lead(d)
        ev.hold("no_cut_snap")
    end = min(confirm - BOARD_NEXT_BALL_S, start + GAP_CAP)
    late = [c for c in cuts if end - 2 <= c <= end and c - start > 10]
    if late:
        end = late[0] - FRAME
    peak = loudest(loud, prev + GAP_CAPTION_LO, min(prev + GAP_CAPTION_HI, end))
    caption_t = max(prev + GAP_CAPTION_MIN, (peak + 1) if peak is not None else prev + 8)
    d["gap_anchor_t"] = round(caption_t - 1, 2)
    ev.hold("no_call_anchor")
    return [(round(start, 2), round(end, 2))], round(caption_t, 2)


def _plan_gap_wicket(ev: FusedEvent, cuts: list[float], loud: list[tuple[float, float]]):
    """Fallback (events without prev_change_t): scoreboard jumped several balls and nobody called it —
    anchor on the crowd."""
    confirm = ev.detail["confirm_t"]
    window = [(t, v) for t, v in loud if confirm - GAP_PEAK_LO <= t <= confirm - GAP_PEAK_HI]
    anchor = max(window, key=lambda x: x[1])[0] if window else None
    start = snap_start(cuts, anchor - 10, anchor - 4) if anchor is not None else None
    if start is None:
        ev.hold("gap_unanchored")
        ranges = [(confirm - 16, confirm + 4)]
    else:
        ev.detail["gap_anchor_t"] = round(anchor, 2)
        end = min(confirm + 4, start + CAP_WICKET)
        if end < confirm + CAPTION_MIN_S:   # caption (at the board change) >= 3s, within the gap cap
            end = min(confirm + CAPTION_MIN_S, start + GAP_CAP)
        ranges = [(start, end)]
    ev.hold("no_call_anchor")
    return [(round(a, 2), round(b, 2)) for a, b in ranges], confirm


MIN_PIECE_S = 2.0   # a range piece shorter than this after removing an avoid range is dropped


def apply_avoid(ranges, avoid) -> list[tuple[float, float]]:
    """Remove every avoid range [a0, a1] (source artefacts) from the planned ranges; drop pieces < 2s.

    E.g. the YouTube SUBSCRIBE pop-up burned into the demo source at 494-497.5."""
    out = [(float(a), float(b)) for a, b in ranges]
    for a0, a1 in avoid or ():
        nxt = []
        for a, b in out:
            if b <= a0 or a >= a1:
                nxt.append((a, b))
                continue
            if a < a0:
                nxt.append((a, round(a0 - FRAME, 2)))
            if b > a1:
                nxt.append((round(a1 + FRAME, 2), b))
        out = nxt
    kept = [(round(a, 2), round(b, 2)) for a, b in out if b - a >= MIN_PIECE_S]
    return kept or [(round(a, 2), round(b, 2)) for a, b in out if b > a][:1]


def clamp_caption(ranges, t: float | None, tail: float = CAPTION_TAIL_S) -> float | None:
    """Move the caption into the clip so it shows for >= `tail` s: a time before the clip or in a gap goes
    to the start of the next range; a time too close to the end goes back to (duration - tail)."""
    if t is None or not ranges:
        return t
    dur = sum(b - a for a, b in ranges)
    off, target = 0.0, None
    for a, b in ranges:
        if t <= b:
            target = off + max(0.0, t - a)
            break
        off += b - a
    if target is None:
        target = dur
    target = max(0.0, min(target, dur - tail))
    off = 0.0
    for i, (a, b) in enumerate(ranges):
        last = i == len(ranges) - 1
        if target < off + (b - a) - 1e-9 or (last and target <= off + (b - a) + 1e-9):
            return round(a + (target - off), 2)
        off += b - a
    return round(ranges[-1][1], 2)


MATCH_CONFIG = Path("config/match.json")   # module default; ClipCutter(match_config=...) overrides


def load_avoid_ranges(path: Path) -> list[tuple[float, float]]:
    """`"avoid_ranges": [[494.0, 497.5]]` (stream seconds) from the match config; missing/invalid -> []."""
    try:
        raw = json.loads(Path(path).read_text()).get("avoid_ranges") or []
        return sorted((float(a), float(b)) for a, b in raw if float(b) > float(a))
    except (OSError, ValueError, TypeError, AttributeError):
        return []


class ClipCutter:
    def __init__(self, buffer: SegmentBuffer, out_dir: Path, encoder: str, max_parallel: int = 2,
                 avoid_ranges: list[tuple[float, float]] | None = None, match_config: Path | None = None):
        self.buffer = buffer
        self.avoid = avoid_ranges if avoid_ranges is not None else load_avoid_ranges(match_config or MATCH_CONFIG)
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
        if ev.type == EventType.WICKET:
            annotate_dismissal(ev)
        if _is_gap_wicket(ev) and d.get("call_t") is None and d.get("prev_change_t") is not None:
            prev, R = d["prev_change_t"], d["confirm_t"]
            lo, hi, loud_lo = prev - GAP_SNAP_LO - 1, min(R + 1, prev + GAP_CAP + 4), prev
        elif _is_gap_wicket(ev) and d.get("call_t") is None:
            R = d["confirm_t"]
            lo, hi, loud_lo = R - GAP_PEAK_LO - 11, R + 8, R - GAP_PEAK_LO
        elif ev.type in WICKETISH:
            call = d.get("call_t") if d.get("call_t") is not None else ev.t
            R = d.get("confirm_t") if d.get("confirm_t") is not None else call
            lo, hi = call - 13, R + 8
        elif ev.type in BOUNDARIES:
            R, lo, hi = ev.t, ev.t - 9, ev.t + 9
            if _ocr_only_boundary(ev):
                confirm = d["confirm_t"]
                loud_lo = confirm - SHOT_LO
                lo, hi = min(lo, loud_lo - 9), max(hi, confirm + 1)
        elif ev.type == EventType.INNINGS_START:
            read = d.get("confirm_t") or ev.t + 6.0
            R, lo, hi = ev.t, min(ev.t, read) - FIRST_BALL_BEFORE_READ_S - 2, max(ev.t + FIRST_BALL_LEN_S, read) + 1
        else:
            ranges, cap_t = plan_ranges(ev, [], [], self.avoid)
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
        l1 = min(max(R + 9, d.get("confirm_t") or 0), hi)
        loud = await loudness([s.path for s in segs], segs[0].start, l0, l1, self.out_dir)
        ranges, cap_t = plan_ranges(ev, cuts, loud, self.avoid)
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
