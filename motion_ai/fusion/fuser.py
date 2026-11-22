"""Combine raw events from all detectors into final, de-duplicated events.

Policy
------
OCR ON  (OCR_AVAILABLE=true)
  * Runs / wickets / overs / powerplay end: the scoreboard is the source of truth.
  * Commentary call + scoreboard agree  -> high confidence, clip anchored on the commentary time
                                           (the scoreboard updates a few seconds after the ball).
  * Scoreboard only                     -> anchor = OCR time - OCR_LAG_SECONDS.
  * Commentary only                     -> kept, but lower confidence + needs_review
                                           (scoreboard did not confirm it).
OCR OFF (OCR_AVAILABLE=false)
  * Commentary (ASR+LLM) is primary; a crowd-noise peak nearby raises confidence.
  * Low-confidence calls are flagged needs_review.
Both
  * Dropped catches / reviews / milestones / toss always come from commentary.
  * A very loud crowd moment with no identified event becomes an EXCITEMENT clip.
  * Events are only finalized once EVERY detector has processed past (t + settle),
    so a slow detector can never cause a missed confirmation.
"""
from __future__ import annotations

import logging
import re

from ..events import OCR_AUTHORITATIVE, Event, EventType, FusedEvent, Source

log = logging.getLogger(__name__)

DEDUPE_S = 20.0
# A dismissal is followed by replays, DRS reviews and "confirmed" calls for up to ~90s — one event.
DEDUPE_BY_TYPE = {EventType.WICKET: 90.0, EventType.REVIEW: 90.0, EventType.TOSS: 120.0}
REVIEW_BELOW = 0.7            # OCR confidence below this is held (ocr_low_conf)
EXCITEMENT_NEAR_S = 12.0
STANDALONE_EXCITEMENT_MIN_CONF = 0.75

# Wicket anchoring (OCR on): the scoreboard change CONFIRMS a wicket; the earliest commentary call
# between the previous scoreboard change and this one ANCHORS it (delivery is just before the call).
CALL_TYPES = {EventType.WICKET, EventType.REVIEW}
WICKET_LOOKBACK_S = 45.0      # max gap call -> scoreboard confirmation (third-umpire reviews take ~30s)
WICKET_ABSORB_AFTER_S = 25.0  # later calls ("first wicket down", replays) fold into the same wicket
ASR_CALL_EXTRA_WAIT_S = 20.0  # commentary calls wait settle+20s so a slow confirmation can arrive
SLOW_CONFIRM_S = 20.0
# Wickets the scoreboard bar never showed (fusion report: 179/7 last-ball run-out). The confirmed INNINGS_END
# footer reports unseen_wickets = footer wickets - last bar wickets. Each one still becomes a WICKET clip:
#   1. a commentary wicket call in (last board change, footer] (and not absorbed as a replay call of the last
#      confirmed wicket) is CONFIRMED by the footer (OCR stays the source
#      of truth for the count; the call only anchors the clip), score at the dismissal from the footer;
#   2. wickets no call covers become one inferred WICKET just before the footer (t = footer - LEAD, never before
#      the last board change), held innings_inferred + no_call_anchor so a human checks the moment.
INNINGS_END_CALL_S = 120.0    # max gap call -> footer for a call the footer may confirm
INFERRED_WICKET_LEAD_S = 10.0  # inferred anchor: this long before the footer (last ball, then the footer)
INFERRED_AFTER_PREV_S = 2.0    # ...but at least this long after the last board change / claiming call
RUN_OUT_RX = re.compile(r"run[\s-]?out|direct hit", re.I)
REVIEW_RX = re.compile(r"review|third umpire|decision pending|upstairs", re.I)


# ---------------------------------------------------------------------------------------------------------
# Score AT the dismissal (caption check report, scratchpad/caption_check/report.md)
#
# The demo source is an edited highlights video: balls between dismissals are cut, so the scoreboard change
# that confirms a wicket can belong to a ball several deliveries LATER ("gap" change). The live dismissal is
# shown right after the PREVIOUS change, with the bar still showing that previous state (catch 432-439 over
# "31-0 2.4"; the bar jumps to 69-1 7.4 at 461). Rules:
#   gap wicket, prev_score/prev_overs known -> wicket on the next legal ball with no runs:
#       event_score = prev runs / prev wkts+1, event_overs = next ball (ball 6 -> next over .1),
#       score_basis "inferred_next_ball" + hold score_inferred.
#       Ground truth: 69/1 7.4 <- 31/0 2.4 => 31/1 (2.5); 116/3 14.0 <- 99/2 12.0 => 99/3 (12.1);
#       157/4 18.0 <- 139/3 16.1 => 139/4 (16.2); 177/5 19.4 <- 177/4 19.2 => 177/5 (19.3).
#   non-gap wicket whose change is explained by ONE delivery (runs +0, wickets +1) -> the board is the
#       score at the dismissal (score_basis "board").
#   anything else (runs moved with the wicket, two wickets in one change, gap without prev_score) -> the
#       score cannot be inferred reliably: score_basis "unknown", no score in the caption, hold score_inferred.
#       Evidence: 179/6 19.5 <- 177/5 19.4 was a run-out off a WIDE (178/6 19.4) + another wide + a dot.
# ---------------------------------------------------------------------------------------------------------
def parse_score(score) -> tuple[int, int] | None:
    """'31/0' -> (31, 0)."""
    m = re.fullmatch(r"\s*(\d+)\s*[/-]\s*(\d+)\s*", str(score or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def next_ball(overs) -> str | None:
    """Over notation of the next legal ball: '2.4' -> '2.5', '12.0' -> '12.1', '11.6' -> '12.1'."""
    m = re.fullmatch(r"\s*(\d+)(?:\.(\d))?\s*", str(overs or ""))
    if not m:
        return None
    o, b = int(m.group(1)), int(m.group(2) or 0)
    return f"{o + 1}.1" if b >= 6 else f"{o}.{b + 1}"


def _balls(overs) -> int | None:
    """'19.6' and '20.0' -> 120 (legal balls bowled)."""
    m = re.fullmatch(r"\s*(\d+)(?:\.(\d))?\s*", str(overs or ""))
    return int(m.group(1)) * 6 + int(m.group(2) or 0) if m else None


def dismissal_score(detail: dict) -> dict:
    """Fields to merge into a scoreboard-confirmed wicket's detail (see rules above)."""
    board = parse_score(detail.get("score"))
    if board is None:
        return {}
    prev = parse_score(detail.get("prev_score"))
    wd = detail.get("wickets") or 1
    if detail.get("gap"):
        nb = next_ball(detail.get("prev_overs"))
        if prev is not None and nb is not None:
            return {"score_basis": "inferred_next_ball", "event_score": f"{prev[0]}/{prev[1] + 1}",
                    "event_overs": nb, "event_wicket_no": prev[1] + 1}
        return {"score_basis": "unknown", "event_wicket_no": prev[1] + 1 if prev else board[1] - (wd - 1)}
    runs = detail.get("runs")
    if runs is None and prev is not None:
        runs = board[0] - prev[0]
    if not runs and wd == 1:
        return {"score_basis": "board"}
    return {"score_basis": "unknown", "event_wicket_no": prev[1] + 1 if prev else board[1] - (wd - 1)}


def annotate_dismissal(f: FusedEvent) -> None:
    """Idempotent: set score_basis / event_score / event_overs on an OCR wicket and hold score_inferred."""
    if f.type != EventType.WICKET or Source.OCR not in f.sources or "score_basis" in f.detail:
        return
    extra = dismissal_score(f.detail)
    f.detail.update(extra)
    if extra.get("score_basis") in ("inferred_next_ball", "unknown"):
        f.hold("score_inferred")


class Fuser:
    def __init__(self, ocr_enabled: bool, settle_seconds: float, ocr_lag_seconds: float):
        self.ocr_enabled = ocr_enabled
        self.settle = settle_seconds
        self.ocr_lag = ocr_lag_seconds
        # Same-type events within this window after the first one are treated as the same moment
        self.group_window = (-5.0, 25.0) if ocr_enabled else (-5.0, 20.0)
        self.pending: list[Event] = []
        self.excitement: list[Event] = []
        self.recent: list[FusedEvent] = []
        self._board_changes: list[tuple[float, int]] = []   # (t, innings) of every OCR score event (fallback)
        self._end_claimed: dict[str, int] = {}               # raw INNINGS_END id -> unseen wickets accounted for
        self._end_last_call: dict[str, float] = {}           # raw INNINGS_END id -> latest call that claimed one

    def add(self, ev: Event) -> None:
        if ev.source == Source.OCR and ev.type != EventType.INNINGS_END and ev.detail.get("score"):
            self._board_changes.append((ev.t, int(ev.detail.get("innings") or 1)))
        (self.excitement if ev.type == EventType.EXCITEMENT else self.pending).append(ev)

    def _wait(self, e: Event) -> float:
        extra = ASR_CALL_EXTRA_WAIT_S if (self.ocr_enabled and e.source == Source.ASR and e.type in CALL_TYPES) else 0
        return self.settle + extra

    def finalize(self, now: float) -> list[FusedEvent]:
        out: list[FusedEvent] = []
        self.pending.sort(key=lambda e: e.t)
        lo, hi = self.group_window
        while self.pending and now - self.pending[0].t >= self._wait(self.pending[0]):
            anchor = self.pending.pop(0)
            if self.ocr_enabled and anchor.type in CALL_TYPES:
                fused = self._fuse_wicket(anchor)
            else:
                group = [anchor] + [e for e in self.pending
                                    if e.type == anchor.type and lo <= e.t - anchor.t <= hi]
                self._remove(group)
                fused = self._fuse(group)
                if self.ocr_enabled and anchor.source == Source.OCR and anchor.type == EventType.INNINGS_END:
                    for w in self._unseen_wickets(anchor):   # before the footer, in time order
                        if not self._is_duplicate(w):
                            out.append(w)
                            self.recent.append(w)
            self._mark_prev_innings_start(fused)
            if not self._is_duplicate(fused):
                out.append(fused)
                self.recent.append(fused)

        out.extend(self._standalone_excitement(now))
        self.recent = [f for f in self.recent if now - f.t < 600]
        self.excitement = [e for e in self.excitement if now - e.t < 600]
        for f in out:
            log.info("FINAL %-14s t=%7.1f conf=%.2f src=%s holds=%s", f.type.value, f.t, f.confidence,
                     sorted(s.value for s in f.sources), f.hold_reasons)
        return sorted(out, key=lambda f: f.t)

    def _remove(self, events: list[Event]) -> None:
        ids = {id(e) for e in events}
        self.pending = [e for e in self.pending if id(e) not in ids]

    # ------------------------------------------------------------------
    def _fuse_wicket(self, anchor: Event) -> FusedEvent:
        """OCR on: one wicket = scoreboard confirmation + every commentary call about it."""
        if anchor.source == Source.OCR and anchor.type == EventType.WICKET:
            ocr = anchor
        else:
            cands = [e for e in self.pending
                     if e.source == Source.OCR and e.type == EventType.WICKET
                     and 0 <= e.t - anchor.t <= WICKET_LOOKBACK_S
                     and (e.detail.get("prev_change_t") is None or anchor.t > e.detail["prev_change_t"])]
            ocr = cands[0] if cands else None

        if ocr is None and anchor.type == EventType.WICKET and (end := self._footer_for_call(anchor)) is not None:
            return self._footer_wicket(end, anchor)

        if ocr is None:   # commentary only — scoreboard never confirmed it
            group = [anchor] + [e for e in self.pending if e.source == Source.ASR and e.type in CALL_TYPES
                                and 0 <= e.t - anchor.t <= WICKET_ABSORB_AFTER_S
                                and self._owner(e) is None]   # never steal a confirmed wicket's call
            self._remove(group)
            etype = EventType.WICKET if any(e.type == EventType.WICKET for e in group) else EventType.REVIEW
            quotes = [e.detail.get("quote", "") for e in group]
            f = FusedEvent(etype, anchor.t, round(max(e.confidence for e in group), 3), {Source.ASR},
                           {"call_t": anchor.t, "quotes": quotes} | anchor.detail)
            f.hold("asr_only")
            self._call_holds(f, group, quotes)
            return f

        prev = ocr.detail.get("prev_change_t")
        prev = float("-inf") if prev is None else prev
        calls = [] if anchor is ocr else [anchor]
        # Calls after the confirmation (replays, "direct hit") are absorbed only until the next scoreboard
        # change, and never when they are the pre-call of a LATER confirmed wicket (highlight edits jump
        # straight to the next dismissal).
        nxt = min((e.t for e in self.pending if e.source == Source.OCR and e.t > ocr.t and e.detail.get("score")),
                  default=float("inf"))
        calls += [e for e in self.pending if e.source == Source.ASR and e.type in CALL_TYPES
                  and prev < e.t <= min(ocr.t + WICKET_ABSORB_AFTER_S, nxt)
                  and (e.t <= ocr.t or self._owner(e) in (None, ocr))]
        self._remove(calls + [ocr])
        before = sorted(c.t for c in calls if c.t <= ocr.t)
        call_t = before[0] if before else None
        quotes = [c.detail.get("quote", "") for c in calls]
        detail = dict(ocr.detail) | {"call_t": call_t, "confirm_t": ocr.t, "quotes": quotes}
        sources = {Source.OCR} | ({Source.ASR} if calls else set())
        t = call_t if call_t is not None else ocr.t - self.ocr_lag
        f = FusedEvent(EventType.WICKET, t, round(ocr.confidence, 3), sources, detail)   # honest: OCR conf only
        if call_t is None:
            f.hold("no_call_anchor")
        elif ocr.t - call_t > SLOW_CONFIRM_S:
            f.hold("slow_confirm")
        if ocr.confidence < REVIEW_BELOW:
            f.hold("ocr_low_conf")
        if (ocr.detail.get("wickets") or 1) >= 2:
            f.hold("multi_wicket_gap")   # two dismissals behind one scoreboard change, only one clip
        self._call_holds(f, calls, quotes)
        annotate_dismissal(f)
        return f

    # ------------------------------------------------------------------
    # Wickets only the INNINGS_END footer saw (see INNINGS_END_CALL_S above)
    def _last_change_t(self, end: Event) -> float | None:
        """Time of the last scoreboard change of the footer's innings (score tracker field, else recorded reads)."""
        if end.detail.get("prev_change_t") is not None:
            return float(end.detail["prev_change_t"])
        inn = int(end.detail.get("innings") or 1)
        ts = [t for t, i in self._board_changes if i == inn and t < end.t]
        return max(ts) if ts else None

    def _open_unseen(self, end: Event) -> int:
        return max(0, int(end.detail.get("unseen_wickets") or 0) - self._end_claimed.get(end.id, 0))

    def _call_in_window(self, call: Event, end: Event) -> bool:
        prev = self._last_change_t(end)
        return (prev is None or call.t > prev) and 0 <= end.t - call.t <= INNINGS_END_CALL_S

    def _footer_for_call(self, call: Event) -> Event | None:
        """A pending INNINGS_END footer with an unaccounted wicket that this commentary call can stand for."""
        ends = [e for e in self.pending if e.source == Source.OCR and e.type == EventType.INNINGS_END
                and self._open_unseen(e) > 0 and self._call_in_window(call, e)]
        return min(ends, key=lambda e: e.t) if ends else None

    def _footer_wicket(self, end: Event, anchor: Event) -> FusedEvent:
        """The footer confirms the call: an OCR wicket anchored on the call (and calls absorbed with it)."""
        group = [anchor] + [e for e in self.pending if e.source == Source.ASR and e.type in CALL_TYPES
                            and 0 <= e.t - anchor.t <= WICKET_ABSORB_AFTER_S and e.t <= end.t
                            and self._owner(e) is None]
        self._remove(group)
        quotes = [e.detail.get("quote", "") for e in group]
        n = self._end_claimed.get(end.id, 0) + 1
        self._end_claimed[end.id] = n
        self._end_last_call[end.id] = anchor.t
        f = FusedEvent(EventType.WICKET, anchor.t, round(end.confidence, 3), {Source.OCR, Source.ASR},
                       self._footer_detail(end, n, 1) | {"call_t": anchor.t, "quotes": quotes})
        self._call_holds(f, group, quotes)
        if f.detail["score_basis"] == "unknown":
            f.hold("score_inferred")
        log.info("Footer %s confirms commentary wicket call at t=%.1f (wicket %s)", end.detail.get("score"),
                 anchor.t, f.detail.get("event_wicket_no"))
        return f

    def _unseen_wickets(self, end: Event) -> list[FusedEvent]:
        """At the footer: unseen wickets no commentary call claimed become ONE inferred wicket before it.
        (Calls in the window are always fused first: pending events finalize in time order and a call waits
        longer than the footer, so it holds the footer back until it has claimed its wicket.)"""
        out: list[FusedEvent] = []
        k = self._open_unseen(end)
        if k == 0:
            return out
        n = self._end_claimed.get(end.id, 0) + k
        self._end_claimed[end.id] = n
        # after the last board change and after any call that claimed an earlier unseen wicket
        floor = max([t for t in (self._last_change_t(end), self._end_last_call.get(end.id)) if t is not None],
                    default=None)
        t = end.t - INFERRED_WICKET_LEAD_S
        if floor is not None:
            t = max(t, min(floor + INFERRED_AFTER_PREV_S, end.t - self.ocr_lag))
        f = FusedEvent(EventType.WICKET, t, round(end.confidence, 3), {Source.OCR},
                       self._footer_detail(end, n, k) | {"call_t": None, "quotes": []})
        f.hold("innings_inferred")      # the moment is a guess: nobody called it, the bar never showed it
        f.hold("no_call_anchor")
        if k >= 2:
            f.hold("multi_wicket_gap")
        if f.detail["score_basis"] == "unknown":
            f.hold("score_inferred")
        log.info("Footer %s: %d wicket(s) never shown or called — inferred wicket at t=%.1f (held)",
                 end.detail.get("score"), k, t)
        return out + [f]

    def _footer_detail(self, end: Event, n: int, k: int) -> dict:
        """Detail of the n-th unseen wicket (k wickets in this event). score = the footer total for the last one,
        otherwise the last bar runs with the wicket count so far (runs unknown -> score_basis unknown)."""
        d = end.detail
        prev, tot = parse_score(d.get("prev_score")), parse_score(d.get("score"))
        unseen = int(d.get("unseen_wickets") or 0)
        last = n >= unseen
        wkt_no = (prev[1] + n) if prev else (tot[1] - unseen + n if tot else None)
        if last or prev is None:
            score = d.get("score")
        else:
            score = f"{prev[0]}/{wkt_no}"
        one_ball = next_ball(d.get("prev_overs")) is not None and _balls(next_ball(d.get("prev_overs"))) == _balls(d.get("overs"))
        exact = last and unseen == 1 and k == 1 and int(d.get("unseen_runs") or 0) == 0 and one_ball
        out = {"score": score, "overs": d.get("overs") if last else None, "wickets": k,
               "innings": d.get("innings", 1), "prev_score": d.get("prev_score"),
               "prev_overs": d.get("prev_overs"), "prev_change_t": self._last_change_t(end),
               "confirm_source": "innings_end", "innings_end_t": end.t, "innings_end_id": end.id,
               "event_wicket_no": wkt_no, "score_basis": "board" if exact else "unknown"}
        return out

    def _mark_prev_innings_start(self, f: FusedEvent) -> None:
        """Flag events whose previous scoreboard change is the innings-start read. That read is confirmed only
        after 6 reads over >= 3 s, so the board was on screen earlier (31/0 bar visible from 431.3, read 433.0;
        the 69/1 dismissal ball's run-up starts 432.1) and the clip planner must lead in further."""
        prev = f.detail.get("prev_change_t")
        if prev is not None and any(r.type == EventType.INNINGS_START and r.detail.get("confirm_t") is not None
                                    and abs(r.detail["confirm_t"] - prev) < 0.05 for r in self.recent):
            f.detail["prev_is_innings_start"] = True

    def _owner(self, call: Event) -> Event | None:
        """The OCR wicket this commentary call precedes (call after that wicket's previous scoreboard change)."""
        for o in sorted((e for e in self.pending if e.source == Source.OCR and e.type == EventType.WICKET),
                        key=lambda e: e.t):
            prev = o.detail.get("prev_change_t")
            if 0 <= o.t - call.t <= WICKET_LOOKBACK_S and (prev is None or call.t > prev):
                return o
        return None

    @staticmethod
    def _call_holds(f: FusedEvent, calls: list[Event], quotes: list[str]) -> None:
        if any(RUN_OUT_RX.search(q or "") for q in quotes):
            f.hold("run_out")
        if any(c.type == EventType.REVIEW for c in calls) or any(REVIEW_RX.search(q or "") for q in quotes):
            f.hold("review")

    def _fuse(self, group: list[Event]) -> FusedEvent:
        etype = group[0].type
        by = {s: [e for e in group if e.source == s] for s in Source}
        ocr, asr = by[Source.OCR], by[Source.ASR]
        detail: dict = {}
        for e in asr + ocr:              # OCR detail (score) wins on conflict
            detail |= e.detail
        sources = {e.source for e in group}

        if by[Source.MANUAL]:
            return FusedEvent(etype, by[Source.MANUAL][0].t, 1.0, sources, detail)
        if by[Source.TIMER]:
            f = FusedEvent(etype, by[Source.TIMER][0].t, 0.5, sources, detail)
            f.hold("asr_only")
            return f

        holds: list[str] = []
        if ocr:
            detail["confirm_t"] = ocr[0].t
        if self.ocr_enabled and etype in OCR_AUTHORITATIVE and ocr:
            conf = ocr[0].confidence                       # honest: the scoreboard reading's own confidence
            t = asr[0].t if asr else ocr[0].t - self.ocr_lag
            if conf < REVIEW_BELOW:
                holds.append("ocr_low_conf")
        else:
            t = (asr or group)[0].t
            conf = max(e.confidence for e in group)
            if asr:
                holds.append("asr_only")                   # commentary only, no scoreboard confirmation
        if etype == EventType.REVIEW:
            holds.append("review")

        ex = self._nearest_excitement(t)
        if ex is not None:
            detail["excitement_z"] = ex.detail.get("z")
            sources.add(Source.AUDIO)
            if not ocr:
                conf = min(1.0, conf + 0.15)
        f = FusedEvent(etype, t, round(conf, 3), sources, detail)
        for h in holds:
            f.hold(h)
        return f

    def _nearest_excitement(self, t: float) -> Event | None:
        near = [e for e in self.excitement if abs(e.t - t) <= EXCITEMENT_NEAR_S]
        return max(near, key=lambda e: e.confidence) if near else None

    def _is_duplicate(self, f: FusedEvent) -> bool:
        window = DEDUPE_BY_TYPE.get(f.type, DEDUPE_S)
        if self.ocr_enabled and f.type == EventType.WICKET and Source.OCR in f.sources:
            # Scoreboard counted a new wicket — it is a different dismissal even if close in time
            # Key = (innings, score): a 6/1 in the chase is not the 6/1 of the first innings.
            key = (f.detail.get("innings", 1), f.detail.get("score"))
            return any(r.type == f.type and (r.detail.get("innings", 1), r.detail.get("score")) == key
                       for r in self.recent)
        if self.ocr_enabled and f.type == EventType.WICKET:
            # Commentary-only wicket: same as an earlier commentary-only call within 90 s, or as a confirmed
            # wicket only inside its replay window (confirm + 25 s). A later call is a different ball and must
            # reach the wicket audit (e.g. a 7th wicket the scoreboard never showed before the innings break).
            return any(r.type == f.type and (
                (Source.OCR not in r.sources and abs(r.t - f.t) < window)
                or (Source.OCR in r.sources and -WICKET_LOOKBACK_S <= f.t - (r.detail.get("confirm_t") or r.t)
                    <= WICKET_ABSORB_AFTER_S)) for r in self.recent)
        return any(r.type == f.type and abs(r.t - f.t) < window for r in self.recent)

    def _standalone_excitement(self, now: float) -> list[FusedEvent]:
        out = []
        ready = [e for e in self.excitement if now - e.t >= self.settle and not e.detail.get("_done")]
        for ex in ready:
            ex.detail["_done"] = True
            if ex.confidence < STANDALONE_EXCITEMENT_MIN_CONF:
                continue
            if any(abs(r.t - ex.t) <= 15 for r in self.recent) or any(abs(p.t - ex.t) <= 15 for p in self.pending):
                continue
            f = FusedEvent(EventType.EXCITEMENT, ex.t, ex.confidence, {Source.AUDIO},
                           {"excitement_z": ex.detail.get("z")})
            self.recent.append(f)
            out.append(f)
        return out
