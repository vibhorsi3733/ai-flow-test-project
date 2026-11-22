"""Pure logic: turn noisy per-frame scoreboard readings into cricket events.

No OCR dependency here so it can be unit-tested. Guards against misreads:
  * consensus — a value is accepted only after N identical consecutive readings
  * plausibility — score never decreases, overs never go backwards
  * gaps — if the scoreboard was hidden (replay/ad), the before/after delta is still classified
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..events import Event, EventType, Source

SCORE_RX = re.compile(r"(\d{1,3})\s*[/\-]\s*(\d{1,2})")
SCORELINE_RX = re.compile(r"(\d{1,3})\s*[/\-]\s*(\d)")
# "11.2" or just "14" (broadcasters often drop ".0" at the end of an over)
OVERS_RX = re.compile(r"(?<!\d)(\d{1,2})(?:\s*[.,]\s*([0-6]))?(?!\d)")


@dataclass(frozen=True)
class ScoreState:
    runs: int
    wickets: int
    over: int
    ball: int

    @property
    def balls(self) -> int:
        return self.over * 6 + self.ball


# "0.50" / "19.2P": a sharper (1080p) crop picks up a trailing glyph after the overs — take the first d.d
DOTTED_OVERS_RX = re.compile(r"(?<!\d)(\d{1,2})\s*[.,]\s*([0-6])")


def parse_reading(score_text: str, overs_text: str) -> ScoreState | None:
    s = SCORE_RX.search(score_text or "")
    o = DOTTED_OVERS_RX.search(overs_text or "") or OVERS_RX.search(overs_text or "")
    if not s:
        return None
    runs, wkts = int(s.group(1)), int(s.group(2))
    if wkts > 10:
        return None
    over, ball = (int(o.group(1)), int(o.group(2) or 0)) if o else (-1, -1)
    if over > 20 and o.group(2) is None and 10 <= over and over % 10 <= 5:
        over, ball = over // 10, over % 10      # dot dropped by OCR: "24" can only be 2.4
    if over > 20:
        return None
    return ScoreState(runs, wkts, over, ball)


def parse_scoreline(text: str) -> ScoreState | None:
    """Score and overs read from one box, e.g. '157-4 18.1' or '2-0 0.2 P'.

    OCR often drops the space ('157-418.1'), so wickets are taken as a single digit
    (an all-out 10 never matters inside a powerplay window)."""
    s = SCORELINE_RX.search(text or "")
    if not s:
        return None
    return parse_reading(f"{s.group(1)}/{s.group(2)}", text[s.end():])


TARGET_RX = re.compile(r"TARGET\D{0,4}(\d{2,3})", re.I)

# New innings (full-match highlight videos join the chase mid-over, e.g. 179/6 (19.5) -> 31/0 (2.4)):
NEW_INNINGS_RUN_DROP = 30
NEW_INNINGS_BALL_DROP = 30
NEW_INNINGS_MIN_OLD_BALLS = 60
NEW_INNINGS_READS = 6
NEW_INNINGS_HOLD_S = 3.0
TARGET_CONFIRM_GAP_S = 1.0   # two matching TARGET reads at least this far apart

# Innings total from the broadcaster's scorecard footer "EXTRAS 11  OVERS 20  TOTAL 179-7".
# Evidence (demo 1080p): the last-ball run-out of innings 1 (~364-380 s, DRS) never reached the score bar
# (last bar read 179/6 19.5); the batting + bowling cards at 398-412.5 s read "OVERS20TOTAL179-7" on every
# frame (conf 0.93-0.96), and the innings-2 cards at 617-635.5 s read "OVERS19.2TOTAL181-5".
FOOTER_RX = re.compile(r"OVERS\s*(\d{1,2})(?:\s*[.,]\s*([0-5]))?\s*TOTAL\s*(\d{1,3})\s*[-/]\s*(\d{1,2})(?!\d)", re.I)
INNINGS_END_CONFIRM_GAP_S = 1.0   # two matching footer reads at least this far apart
INNINGS_END_MIN_OVER = 18         # only look for the total once the bar is this deep into the innings...
INNINGS_END_MIN_WICKETS = 8       # ...or this many wickets down (or the chase is won)


def parse_target(text: str) -> int | None:
    m = TARGET_RX.search(text or "")
    return int(m.group(1)) if m else None


def parse_footer(text: str) -> ScoreState | None:
    """Scorecard footer, e.g. 'OVERS20TOTAL179-7' or 'OVERS 19.2TOTAL181-5' (OCR drops spaces)."""
    m = FOOTER_RX.search(text or "")
    if not m:
        return None
    over, ball, runs, wkts = int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), int(m.group(4))
    if wkts > 10 or over > 20 or (over == 20 and ball):
        return None
    return ScoreState(runs, wkts, over, ball)


def fmt_score(st: ScoreState | None) -> str | None:
    return f"{st.runs}/{st.wickets}" if st else None


def fmt_overs(st: ScoreState | None) -> str | None:
    return f"{st.over}.{st.ball}" if st else None


class ScoreTracker:
    def __init__(self, powerplay_overs: int = 6, consensus_frames: int = 3):
        self.powerplay_overs = powerplay_overs
        self.consensus = consensus_frames
        self.state: ScoreState | None = None
        self._candidate: ScoreState | None = None
        self._candidate_t: float = 0.0
        self._candidate_n = 0
        self._powerplay_done = False
        self._last_change_t: float | None = None   # time of the previous accepted scoreboard change
        self.innings = 1
        self.innings_last: dict[int, tuple[float, ScoreState]] = {}   # last committed (confirmed) read
        self.target: int | None = None
        self._target_seen: tuple[int, float] | None = None
        self._match_end_sent = False
        self.innings_end: dict[int, ScoreState] = {}            # confirmed innings totals
        self._footer_seen: tuple[ScoreState, float] | None = None

    def _ev(self, etype: EventType, t: float, conf: float, detail: dict,
            prev: ScoreState | None = None, use_prev: bool = False) -> Event:
        """Every scoreboard event carries prev_score / prev_overs = the board state before this change
        (the current state for events that do not change it, e.g. TARGET; None before the first read)."""
        p = prev if use_prev else self.state
        return Event(etype, t, Source.OCR, conf,
                     {"prev_score": fmt_score(p), "prev_overs": fmt_overs(p)} | detail | {"innings": self.innings})

    def wants_innings_total(self) -> bool:
        """Cheap gate for reading the scorecard footer: only near an innings' end, once per innings."""
        st = self.state
        if st is None or self.innings in self.innings_end:
            return False
        return st.over >= INNINGS_END_MIN_OVER or st.wickets >= INNINGS_END_MIN_WICKETS or self._match_end_sent

    def update_innings_total(self, t: float, text: str) -> list[Event]:
        """Scorecard footer 'OVERS 20 TOTAL 179-7' -> INNINGS_END for the current innings.

        Accepted only if the total (a) is a completed innings: 20 overs, 10 wickets, or the chase reached the
        target; (b) is consistent with the last confirmed bar read (never lower runs/wickets/balls, at most
        7 runs per ball in between); (c) is not a previous innings' total (a recap card); and (d) is read
        twice, identically, >= 1 s apart. Never guessed from the bar alone."""
        tot = parse_footer(text)
        st = self.state
        if tot is None or st is None or self.innings in self.innings_end:
            return []
        complete = (tot.over == 20 and tot.ball == 0) or tot.wickets == 10 or (
            self.innings >= 2 and self.target is not None and tot.runs >= self.target)
        d_balls = tot.balls - st.balls if st.over >= 0 else 0
        consistent = (tot.runs >= st.runs and tot.wickets >= st.wickets and d_balls >= 0
                      and tot.runs - st.runs <= 7 * max(d_balls, 1))
        if not complete or not consistent or tot in self.innings_end.values():
            return []
        if self._footer_seen and self._footer_seen[0] == tot and t - self._footer_seen[1] >= INNINGS_END_CONFIRM_GAP_S:
            self.innings_end[self.innings] = tot
            self._footer_seen = None
            ev = self._ev(EventType.INNINGS_END, t, 0.95,
                          {"score": fmt_score(tot), "overs": fmt_overs(tot), "source": "scorecard_footer",
                           "unseen_wickets": tot.wickets - st.wickets, "unseen_runs": tot.runs - st.runs,
                           # last bar change: a wicket the bar never showed fell after it (fuser)
                           "prev_change_t": self._last_change_t})
            # The bar may flash the final state later (e.g. 179-7 20.0) — do not report that as a new wicket.
            self.state = tot
            self.innings_last[self.innings] = (t, tot)
            return [ev]
        if not self._footer_seen or self._footer_seen[0] != tot:
            self._footer_seen = (tot, t)
        return []

    def update(self, t: float, reading: ScoreState | None) -> list[Event]:
        if reading is None:          # scoreboard hidden / unreadable — skip frame
            return []
        if reading == self._candidate:
            self._candidate_n += 1
        else:
            self._candidate, self._candidate_t, self._candidate_n = reading, t, 1
        if self._candidate_n < self.consensus or reading == self.state:
            return []
        if self._is_new_innings(self.state, reading) and (
                self._candidate_n < NEW_INNINGS_READS or t - self._candidate_t < NEW_INNINGS_HOLD_S):
            return []   # a new innings needs a longer, steadier read
        return self._commit(self._candidate_t, reading)

    def update_target(self, t: float, text: str) -> list[Event]:
        """'TARGET - 180' box (chase only). Confirmed by two matching reads >= 1 s apart."""
        val = parse_target(text)
        if val is None or self.target is not None or self.innings < 2:
            return []
        if self._target_seen and self._target_seen[0] == val and t - self._target_seen[1] >= TARGET_CONFIRM_GAP_S:
            self.target = val
            return [self._ev(EventType.TARGET, t, 0.95, {"target": val})]
        if not self._target_seen or self._target_seen[0] != val:
            self._target_seen = (val, t)
        return []

    @staticmethod
    def _is_new_innings(old: ScoreState | None, new: ScoreState) -> bool:
        if old is None or old.over < 0 or new.over < 0:
            return False
        return (old.runs - new.runs >= NEW_INNINGS_RUN_DROP and old.balls - new.balls >= NEW_INNINGS_BALL_DROP
                and old.balls >= NEW_INNINGS_MIN_OLD_BALLS)

    def _commit(self, t: float, new: ScoreState) -> list[Event]:
        old = self.state

        def ev(*a) -> Event:   # prev_score / prev_overs = the state before this change
            return self._ev(*a, prev=old, use_prev=True)

        if old is None:
            self.state = new
            self._last_change_t = t
            self.innings_last[self.innings] = (t, new)
            if 0 <= new.balls < self.powerplay_overs * 6 or (new.runs == 0 and new.wickets == 0):
                # 0/0 = true start; otherwise we joined an innings already in progress (still in the powerplay)
                conf = 0.9 if new.runs == 0 and new.wickets == 0 else 0.7
                return [ev(EventType.INNINGS_START, t, conf,
                                 {"score": f"{new.runs}/{new.wickets}", "overs": f"{new.over}.{new.ball}"})]
            return []

        if self._is_new_innings(old, new) or (new.runs == 0 and new.wickets == 0 and old.balls > 0):
            self.innings += 1
            self.state = new
            self._last_change_t = t
            self._powerplay_done = new.balls >= self.powerplay_overs * 6
            self.innings_last[self.innings] = (t, new)
            joined = f"{new.runs}/{new.wickets} ({new.over}.{new.ball})"
            return [ev(EventType.INNINGS_START, t, 0.8,
                             {"score": f"{new.runs}/{new.wickets}", "overs": f"{new.over}.{new.ball}",
                              "joined": joined, "prev_innings_last": f"{old.runs}/{old.wickets} ({old.over}.{old.ball})"})]

        # Plausibility: reject going backwards (misread).
        if new.runs < old.runs or new.wickets < old.wickets or (new.over >= 0 and old.over >= 0 and new.balls < old.balls):
            return []

        d_runs, d_wkts = new.runs - old.runs, new.wickets - old.wickets
        gap = new.over >= 0 and old.over >= 0 and new.balls - old.balls > 1
        if d_runs > 7 and not gap:   # implausible single-ball jump without a gap → misread
            return []
        if gap and new.balls - old.balls > 30 and d_runs < 5:   # e.g. "1.4" misread as "14"
            return []

        self.state = new
        self.innings_last[self.innings] = (t, new)
        detail = {"score": f"{new.runs}/{new.wickets}", "overs": f"{new.over}.{new.ball}",
                  "runs": d_runs, "gap": gap, "prev_change_t": self._last_change_t}
        self._last_change_t = t
        conf = 0.75 if gap else 0.95
        events: list[Event] = []
        if d_wkts > 0:
            events.append(ev(EventType.WICKET, t, conf, detail | {"wickets": d_wkts}))
        small_gap = gap and new.balls - old.balls <= 3   # scoreboard hidden for a replay of 1-3 balls
        if d_wkts == 0 and (not gap or small_gap):
            # Small gap: the runs may span 2-3 balls, so only a candidate — needs commentary to confirm.
            if d_runs in (6, 7):     # 7 = six + wide/no-ball
                events.append(ev(EventType.SIX, t, 0.6 if gap else conf, detail))
            elif d_runs in (4, 5):   # 5 = four + no-ball/overthrow
                events.append(ev(EventType.FOUR, t, 0.6 if gap else 0.85, detail))
        if new.ball == 0 and new.over > old.over and old.over >= 0:
            events.append(ev(EventType.OVER_COMPLETE, t, 0.95, detail))
        pp_balls = self.powerplay_overs * 6
        if old.balls >= 0 and old.balls < pp_balls <= new.balls and not self._powerplay_done:
            self._powerplay_done = True
            events.append(ev(EventType.POWERPLAY_END, t, 0.95 if not gap else 0.8, detail))
        if (self.innings >= 2 and self.target is not None and new.runs >= self.target
                and not self._match_end_sent):
            self._match_end_sent = True
            events.append(ev(EventType.MATCH_END, t, 0.95, detail | {"target": self.target}))
        return events
