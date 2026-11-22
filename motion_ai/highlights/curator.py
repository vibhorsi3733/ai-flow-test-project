"""Choose which clips go into the package, in what order, with captions.

Claude curates; hard rules are enforced afterwards so the output is always safe:
  * every confident wicket and the toss are always included
  * the total length stays near the target
  * clips run in match order
If the LLM call fails, a rule-based plan is used.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..config import OversFilter
from ..events import EventType, Source
from ..llm import LLM
from .clips import Clip, caption_for

log = logging.getLogger(__name__)

PRIORITY = {
    EventType.WICKET: 100, EventType.TOSS: 95, EventType.SIX: 80, EventType.REVIEW: 70,
    EventType.DROPPED_CATCH: 65, EventType.MILESTONE: 60, EventType.FOUR: 50,
    EventType.INNINGS_START: 40, EventType.EXCITEMENT: 30,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "caption": {"type": "string"}},
                "required": ["id", "caption"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "clips"],
    "additionalProperties": False,
}

SYSTEM = """You are the highlights editor for a live IPL broadcast. From the candidate clips (toss to the end
of the powerplay), choose and caption the clips for a highlights package.
- Total duration must stay within the target. Always keep the toss and every wicket.
- Prefer sixes, reviews and dropped catches over fours; skip clips marked needs_review unless they
  clearly add to the story.
- Captions: short on-screen text in English, max 8 words, e.g. "SIX! Rohit clears long-on".
  Use only facts given in the clip data; never invent players or scores.
- Title: one punchy line summarising the powerplay."""


@dataclass
class Plan:
    title: str
    items: list[tuple[Clip, str]]
    status: str = "auto_ok"   # or "hold_for_review"
    withheld: list[dict] = field(default_factory=list)   # clips left out on purpose (e.g. not_selected_over)

    @property
    def duration(self) -> float:
        return sum(c.duration for c, _ in self.items)


# ---------------------------------------------------------------------------------------------------------
# HIGHLIGHT_OVERS — keep only the selected overs of each batting team
_FROM_SETTINGS = object()          # sentinel: read HIGHLIGHT_OVERS from Settings


def over_number(overs) -> int | None:
    """1-based over a scoreboard "o.b" reading belongs to.

    "o.b" with b > 0 is a ball of over o+1 ("0.1" -> 1, "5.6" -> 6, "19.2" -> 20). "o.0" is shown right
    after the ball that completed over o, so it is over o ("6.0" -> 6, "20.0" -> 20); "0.0" (nothing bowled
    yet, e.g. a wide off the first ball) counts as over 1. Unreadable -> None (never filtered).
    """
    m = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", str(overs)) if overs not in (None, "") else None
    if not m:
        return None
    o, b = int(m[1]), int(m[2] or 0)
    return max(1, o) if b == 0 else o + 1


def event_over(detail: dict) -> int | None:
    """Over of the ball itself: event_overs (the dismissal ball of a gap wicket) before the board's overs."""
    return over_number(detail.get("event_overs") or detail.get("overs"))


def resolve_overs_filter(overs=_FROM_SETTINGS) -> OversFilter | None:
    """An explicit OversFilter/None wins; otherwise HIGHLIGHT_OVERS from Settings (validated at startup)."""
    if overs is not _FROM_SETTINGS:
        return overs
    try:
        from ..config import Settings
        return Settings().overs_filter()
    except Exception as exc:          # no SRT_URL etc. (tests, tools): fall back to all overs
        log.debug("HIGHLIGHT_OVERS not loaded (%s); keeping all overs", exc)
        return None


def _withheld_entry(c: Clip, innings: int, reason: str) -> dict:
    return {"id": c.event.id, "type": c.event.type.value, "match_time_s": round(c.event.t, 1),
            "innings": innings, "score": c.event.detail.get("score"),
            "overs": c.event.detail.get("event_overs") or c.event.detail.get("overs"), "reason": reason}


NOT_BALL_EVENTS = {EventType.TOSS, EventType.INNINGS_START}   # never filtered by over (cards / pre-match)


def split_by_overs(clips: list[Clip], flt: OversFilter | None, innings: dict[str, int],
                   over_at=None) -> tuple[list[Clip], list[dict]]:
    """(clips in the selected overs, withheld[] entries with reason not_selected_over).

    A clip without its own over (commentary-only call, excitement) uses over_at(t, innings) when given.
    """
    if flt is None:
        return list(clips), []
    kept, out = [], []
    for c in clips:
        i = innings.get(c.event.id, 1)
        over = event_over(c.event.detail)
        if over is None and over_at is not None:
            over = over_at(c.event.t, i)
        if c.event.type in NOT_BALL_EVENTS or flt.allows(i, over):
            kept.append(c)
        else:
            out.append(_withheld_entry(c, i, "not_selected_over"))
    return kept, out


def default_caption(c: Clip) -> str:
    return c.caption or caption_for(c.event)


def _must_keep(c: Clip) -> bool:
    """Toss and every scoreboard-confirmed wicket are always in (held or not — a hold never drops a clip)."""
    return c.event.type == EventType.TOSS or (
        c.event.type == EventType.WICKET and (Source.OCR in c.event.sources or not c.event.needs_review))


def _overlaps(a: Clip, b: Clip) -> bool:
    inter = min(a.t1, b.t1) - max(a.t0, b.t0)
    return inter > 0.5 * min(a.duration, b.duration)


def _status(items: list[Clip]) -> str:
    return "hold_for_review" if any(c.event.needs_review for c in items) else "auto_ok"


def rule_based_plan(clips: list[Clip], target: float, title: str = "Powerplay highlights",
                    overs=_FROM_SETTINGS) -> Plan:
    """Held clips are KEPT (the package is held instead). Budget is shared round-robin across types."""
    clips, not_selected = split_by_overs(clips, resolve_overs_filter(overs),
                                         {c.event.id: innings_of(c.event, None) for c in clips})
    chosen: list[Clip] = []

    def fits(c: Clip) -> bool:
        return (not any(_overlaps(c, k) for k in chosen)
                and (_must_keep(c) or sum(k.duration for k in chosen) + c.duration <= target))

    for c in clips:                           # 1) toss + confirmed wickets always
        if _must_keep(c) and fits(c):
            chosen.append(c)
    by_type: dict[EventType, list[Clip]] = {}
    for c in sorted(clips, key=lambda c: (not c.event.needs_review, c.event.confidence), reverse=True):
        if c not in chosen:
            by_type.setdefault(c.event.type, []).append(c)
    order = sorted(by_type, key=lambda t: PRIORITY.get(t, 0), reverse=True)
    while any(by_type[t] for t in order):     # 2) round-robin across event types
        for t in order:
            if by_type[t]:
                c = by_type[t].pop(0)
                if fits(c):
                    chosen.append(c)
    chosen.sort(key=lambda c: c.t0)
    return Plan(title, [(c, default_caption(c)) for c in chosen], _status(chosen), not_selected)


async def curate(llm: LLM, clips: list[Clip], target: float, match: dict,
                 title: str = "Powerplay highlights", overs=_FROM_SETTINGS) -> Plan:
    """Claude chooses and orders clips; captions stay the fact-based burned-in ones."""
    flt = resolve_overs_filter(overs)
    clips, not_selected = split_by_overs(clips, flt, {c.event.id: innings_of(c.event, None) for c in clips})
    if not clips:
        return Plan(title, [], "hold_for_review", not_selected)
    by_id = {c.event.id: c for c in clips}
    candidates = [{
        "id": c.event.id, "type": c.event.type.value, "match_time_s": round(c.event.t),
        "duration_s": round(c.duration), "confidence": c.event.confidence,
        "hold_reasons": c.event.hold_reasons, "caption": c.caption,
        "facts": {k: v for k, v in c.event.detail.items() if not k.startswith("_")},
    } for c in clips]
    user = (f"Match: {match.get('match', 'IPL')}\nTarget duration: {int(target)}s\n"
            f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=1, default=str)}")
    result = await llm.json(SYSTEM, user, SCHEMA)
    if result is None:
        log.warning("Curator LLM unavailable; using rule-based plan")
        plan = rule_based_plan(clips, target, title, overs=None)
        plan.withheld = not_selected + plan.withheld
        return plan

    chosen = {i["id"]: by_id[i["id"]] for i in result["clips"] if i["id"] in by_id}
    for c in clips:                           # enforce must-keep
        if _must_keep(c):
            chosen.setdefault(c.event.id, c)
    items = sorted(chosen.values(), key=lambda c: c.t0)
    while sum(c.duration for c in items) > target * 1.15:
        optional = [c for c in items if not _must_keep(c)]
        if not optional:
            break
        items.remove(min(optional, key=lambda c: (PRIORITY.get(c.event.type, 0), c.event.confidence)))
    return Plan(result["title"] or title, [(c, default_caption(c)) for c in items], _status(items), not_selected)


# ---------------------------------------------------------------------------------------------------------
# Full-match scope (HIGHLIGHT_SCOPE=match) — panel spec fm/guidance_v1 §1 Budget, §4 Holds, IPL head conditions
HARD_MAX_S = 360.0
CARD_SECONDS = {"title": 3.0, "innings1": 2.0, "break": 3.0, "innings2": 2.0, "result": 3.0, "outro": 2.0}
BOUNDARY_ORDER = {EventType.SIX: 0, EventType.FOUR: 1}


def _score(detail: dict) -> tuple[int, int] | None:
    try:
        r, w = str(detail["score"]).split("/")
        return int(r), int(w)
    except (KeyError, ValueError):
        return None


def innings_of(ev, innings2_t: float | None) -> int:
    if ev.detail.get("innings"):
        return int(ev.detail["innings"])
    return 2 if innings2_t is not None and ev.t >= innings2_t - 5 else 1


@dataclass
class MatchFacts:
    innings2_t: float | None = None
    innings1_runs: int | None = None        # last confirmed first-innings runs
    innings1_last: str | None = None
    target: int | None = None               # confirmed TARGET read (two reads >= 1 s apart)
    result_score: tuple[int, int] | None = None   # confirmed chase read >= target
    wicket_changes: dict[int, list[dict]] = None  # innings -> OCR wicket score changes
    innings_end: dict[int, dict] = None           # innings -> confirmed INNINGS_END {"score", "overs", ...}
    board: list[tuple[float, int, str]] = None    # (t, innings, overs) of every OCR reading, in time order

    @classmethod
    def from_events(cls, events) -> "MatchFacts":
        f = cls(wicket_changes={1: [], 2: []}, innings_end={})
        f.board = sorted((e.t, innings_of(e, f.innings2_t), str(e.detail["overs"])) for e in events
                         if Source.OCR in e.sources and e.detail.get("overs") not in (None, ""))
        inn2 = [e for e in events if e.type == EventType.INNINGS_START and e.detail.get("innings", 1) >= 2]
        f.innings2_t = inn2[0].t if inn2 else None
        if inn2 and inn2[0].detail.get("prev_innings_last"):
            f.innings1_last = inn2[0].detail["prev_innings_last"]
        runs1 = [s[0] for e in events if innings_of(e, f.innings2_t) == 1
                 and Source.OCR in e.sources and (s := _score(e.detail))]
        if f.innings1_last and (s := _score({"score": f.innings1_last.split(" ")[0]})):
            runs1.append(s[0])
        f.innings1_runs = max(runs1) if runs1 else None
        tg = [e for e in events if e.type == EventType.TARGET]
        f.target = tg[0].detail.get("target") if tg else None
        me = [e for e in events if e.type == EventType.MATCH_END]
        if me and f.target is not None and (s := _score(me[0].detail)) and s[0] >= f.target:
            f.result_score = s
        for e in events:
            if e.type == EventType.WICKET and Source.OCR in e.sources:
                f.wicket_changes.setdefault(innings_of(e, f.innings2_t), []).append(
                    {"t": round(e.t, 1), "score": e.detail.get("score"), "overs": e.detail.get("overs"),
                     "wickets": e.detail.get("wickets") or 1, "id": e.id,
                     "event_score": e.detail.get("event_score"), "event_overs": e.detail.get("event_overs")})
            elif e.type.value == "innings_end" and (s := _score(e.detail)):
                i = int(e.detail.get("innings") or innings_of(e, f.innings2_t))
                old = f.innings_end.get(i)
                if old is None or s[1] >= (_score(old) or (0, 0))[1]:
                    f.innings_end[i] = dict(e.detail)
        return f

    def over_at(self, t: float, innings: int) -> int | None:
        """Over of a ball bowled at t: the next ball after the last board reading of that innings before t
        ("6.0" or "6.3" on the board -> over 7). Used only for clips without their own overs."""
        prev = [o for bt, i, o in (self.board or []) if i == innings and bt <= t]
        if not prev or not (m := re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", prev[-1])):
            return None
        o, b = int(m[1]), int(m[2] or 0)
        return min(20, o + 1) if b < 6 else min(20, o + 2)

    @property
    def target_matches_innings1(self) -> bool:
        return self.target is not None and self.innings1_runs is not None and self.target == self.innings1_runs + 1


@dataclass
class MatchPlan:
    plan: Plan
    innings: dict[str, int]                 # clip id -> innings
    withheld: list[dict]
    wicket_audit: dict
    package_holds: list[str]
    budget_note: str | None = None


def match_plan(clips: list[Clip], target: float, facts: MatchFacts, title: str,
               overs=_FROM_SETTINGS) -> MatchPlan:
    card_total = sum(CARD_SECONDS.values())
    inn = {c.event.id: innings_of(c.event, facts.innings2_t) for c in clips}
    flt = resolve_overs_filter(overs)
    if not facts.target_matches_innings1:   # chase not confirmed by a TARGET read = innings-1 runs + 1
        for c in clips:
            if inn[c.event.id] >= 2:
                c.event.hold("innings_inferred")

    clips, withheld = split_by_overs(clips, flt, inn, facts.over_at)   # HIGHLIGHT_OVERS: before must-keep
    chosen: list[Clip] = []

    def exempt(c: Clip) -> bool:
        return _must_keep(c) or c.event.type == EventType.MATCH_END


    def skip(c: Clip, reason: str) -> None:
        withheld.append(_withheld_entry(c, inn[c.event.id], reason))

    # The winning ball is one clip: a boundary with the same final score folds into WINNING RUNS.
    for me in [c for c in clips if c.event.type == EventType.MATCH_END]:
        same = [c for c in clips if c.event.type in BOUNDARY_ORDER and c.event.detail.get("score") == me.event.detail.get("score")]
        for b in same:
            label = "FOUR" if b.event.type == EventType.FOUR else "SIX"
            me.caption = me.caption.replace("WINNING RUNS", f"WINNING RUNS · {label}", 1)
            clips = [c for c in clips if c is not b]
            inn[b.event.id] = inn.get(b.event.id, 2)
            withheld.append({"id": b.event.id, "type": b.event.type.value, "match_time_s": round(b.event.t, 1),
                             "innings": inn[b.event.id], "score": b.event.detail.get("score"),
                             "reason": "same_ball_as_winning_runs"})
    for c in sorted([c for c in clips if exempt(c)], key=lambda c: (c.event.type != EventType.MATCH_END, c.t0)):
        if any(_overlaps(c, k) for k in chosen):
            skip(c, "overlap")
        else:
            chosen.append(c)
    budget = target - card_total
    left = budget - sum(c.duration for c in chosen)
    note = None
    if sum(c.duration for c in chosen) + card_total > HARD_MAX_S:
        note = "over_budget"
        log.warning("Exempt clips alone exceed the %ss hard max — keeping all of them", HARD_MAX_S)

    unconfirmed = [c for c in clips if not exempt(c) and Source.OCR not in c.event.sources
                   and c.event.type in (EventType.FOUR, EventType.SIX, EventType.WICKET, EventType.REVIEW)]
    optional = [c for c in clips if not exempt(c) and c not in unconfirmed
                and c.event.type not in (EventType.INNINGS_START, EventType.TOSS)]
    for c in clips:
        if c.event.type == EventType.INNINGS_START:
            skip(c, "replaced_by_card")
        elif c in unconfirmed:
            skip(c, "not_on_scoreboard")   # commentary-only call: never shown without scoreboard confirmation

    def rank(c: Clip):
        return (BOUNDARY_ORDER.get(c.event.type, 2), -c.event.t)   # sixes, then fours (last over first), then rest

    shares = {1: max(0.0, left) / 2, 2: max(0.0, left) / 2}
    pending: list[Clip] = []
    for i in (1, 2):
        for c in sorted([c for c in optional if inn[c.event.id] == i], key=rank):
            if any(_overlaps(c, k) for k in chosen):
                skip(c, "overlap")
            elif c.duration <= shares[i]:
                chosen.append(c)
                shares[i] -= c.duration
            else:
                pending.append(c)
    spare = shares[1] + shares[2]                          # unused share of one innings goes to the other
    for c in sorted(pending, key=rank):
        if c.duration <= spare and not any(_overlaps(c, k) for k in chosen):
            chosen.append(c)
            spare -= c.duration
        else:
            skip(c, "budget")

    chosen.sort(key=lambda c: c.t0)
    audit, missing = {}, []
    clipped_ids = {c.event.id for c in chosen}
    for i in (1, 2):
        calls = [c for c in unconfirmed if c.event.type == EventType.WICKET and inn[c.event.id] == i]
        a, gaps = wicket_audit(facts, i, clipped_ids, flt)
        a["wicket_calls_not_on_scoreboard"] = [
            {"match_time_s": round(c.event.t, 1),
             "quote": (c.event.detail.get("quotes") or [c.event.detail.get("quote", "")])[0]} for c in calls]
        audit[f"innings_{i}"] = a
        missing += gaps
        if calls:
            missing.append(f"innings {i}: {len(calls)} commentary wicket call(s) never confirmed on the scoreboard")
    holds = ["wicket_missing"] if missing else []
    if missing:
        audit["wicket_missing"] = missing
    status = "hold_for_review" if (holds or any(c.event.needs_review for c in chosen)) else "auto_ok"
    plan = Plan(title, [(c, default_caption(c)) for c in chosen], status)
    return MatchPlan(plan, {c.event.id: inn[c.event.id] for c in chosen}, withheld, audit, holds, note)


def _label(w: dict) -> str:
    """Audit text for a wicket change: the score/over AT the dismissal (event_*), board reading if different."""
    score, overs = w.get("event_score") or w.get("score"), w.get("event_overs") or w.get("overs")
    text = f"{score} ({overs} ov)" if overs else f"{score}"
    return text + (f" [board {w['score']}]" if w.get("event_score") and w["event_score"] != w.get("score") else "")


def wicket_audit(facts: MatchFacts, i: int, clipped_ids: set[str], flt: OversFilter | None
                 ) -> tuple[dict, list[str]]:
    """wickets_on_board vs wickets_clipped for innings i, plus the wicket_missing lines.

    wickets_on_board = the highest of: OCR wicket changes, the last first-innings reading, and a confirmed
    INNINGS_END read (authoritative: the 7th first-innings wicket, 179/7 off the last ball, never showed as
    a score change in the demo — the board went straight to the innings-break graphic). A wicket number
    that no scoreboard change covers is named ("innings 1: 179/7 not clipped"). Wickets outside the
    HIGHLIGHT_OVERS selection are listed under not_selected_over and never count as missing.
    """
    changes = facts.wicket_changes.get(i, [])
    end = (facts.innings_end or {}).get(i)
    on_board = max([(_score({"score": w["score"]}) or (0, 0))[1] for w in changes] or [0])
    if i == 1 and facts.innings1_last and (s := _score({"score": facts.innings1_last.split(" ")[0]})):
        on_board = max(on_board, s[1])
    if end and (s := _score(end)):
        on_board = max(on_board, s[1])

    def selected(over) -> bool:
        return flt is None or flt.allows(i, over)

    covered: dict[int, dict] = {}                 # wicket number -> the change that showed it
    for w in changes:
        if s := _score({"score": w["score"]}):
            for n in range(max(1, s[1] - int(w.get("wickets") or 1) + 1), s[1] + 1):
                covered.setdefault(n, w)
    sel = [w for w in changes if selected(event_over(w))]
    not_sel = [w for w in changes if w not in sel]
    clipped = sum(1 for w in sel if w["id"] in clipped_ids)
    not_clipped = [w for w in sel if w["id"] not in clipped_ids]

    unseen, unseen_not_sel = [], []               # wickets on the board that no change showed
    for n in range(1, on_board + 1):
        if n in covered:
            continue
        later = [w for w in changes if (_score({"score": w["score"]}) or (0, 0))[1] > n]
        if later:                                 # inside the sequence: belongs to the next change's ball
            ref = min(later, key=lambda w: w["t"])
            text, over = f"wicket {n} before {_label(ref)}", event_over(ref)
        elif end and (s := _score(end)) and s[1] >= n:
            text, over = f"{s[0]}/{n}" + (f" ({end.get('overs')} ov)" if end.get("overs") else ""), \
                over_number(end.get("overs"))
        else:                                     # only the last first-innings reading "179/6 (19.5)"
            last = str(facts.innings1_last or "")
            m = re.search(r"\(([\d.]+)\)", last)
            text, over = f"wicket {n} (last reading {last})", over_number(m[1]) if m else None
        (unseen if selected(over) else unseen_not_sel).append(text)

    excluded = sum(int(w.get("wickets") or 1) for w in not_sel) + len(unseen_not_sel)
    expected = max(0, on_board - excluded)
    audit = {"wickets_on_board": on_board, "wickets_clipped": clipped,
             "not_clipped": [_label(w) for w in not_clipped] + [f"{t} (no scoreboard change)" for t in unseen]}
    if flt is not None:
        audit["wickets_in_selected_overs"] = expected
        audit["not_selected_over"] = [_label(w) for w in not_sel] + unseen_not_sel
    if end:
        audit["innings_end"] = {k: end.get(k) for k in ("score", "overs", "source") if end.get(k) is not None}
    lines = []
    if clipped < expected:
        lines.append(f"innings {i}: {clipped}/{expected} wickets clipped")
        lines += [f"innings {i}: {_label(w)} not clipped" for w in not_clipped]
        lines += [f"innings {i}: {t} not clipped" for t in unseen]
    return audit, lines
