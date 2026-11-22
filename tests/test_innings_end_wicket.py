"""Wickets only the INNINGS_END footer saw become clips (last package: innings 1 179/7, wicket 7 not clipped).

Recorded demo (runtime_1080/events.jsonl): bar 179/6 (19.5) at 362.06, then straight to the footer
179/7 (20.0) at 400.96 (unseen_wickets 1). The last-ball run-out was called at 394.2 ("one more easy run
out") and was withheld as not_on_scoreboard.
"""
from pathlib import Path

from motion_ai.events import Event, EventType, Source
from motion_ai.fusion.fuser import Fuser
from motion_ai.highlights.clips import Clip
from motion_ai.highlights.curator import MatchFacts, match_plan


def ev(t, type_, src, conf=0.95, **detail):
    return Event(type_, t, src, conf, detail)


def wicket6():
    return ev(362.06, EventType.WICKET, Source.OCR, prev_score="177/5", prev_overs="19.4", score="179/6",
              overs="19.5", runs=2, gap=False, prev_change_t=339.42, wickets=1, innings=1)


def footer(unseen=1, score="179/7", prev_score="179/6", prev_overs="19.5", **kw):
    return ev(400.96, EventType.INNINGS_END, Source.OCR, prev_score=prev_score, prev_overs=prev_overs,
              score=score, overs="20.0", source="scorecard_footer", unseen_wickets=unseen, unseen_runs=0,
              innings=1, **kw)


def call(t, quote="And one more easy run out."):
    return ev(t, EventType.WICKET, Source.ASR, 0.4, quote=quote)


def wickets(out):
    return [e for e in out if e.type == EventType.WICKET]


def fuse(events, now=float("inf")):
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    for e in events:
        f.add(e)
    return f.finalize(now)


def test_footer_confirms_last_ball_commentary_call():
    out = fuse([wicket6(), call(394.2), footer()])     # no prev_change_t: recorded events (fallback = 362.06)
    w7 = [w for w in wickets(out) if w.detail.get("score") == "179/7"]
    assert len(w7) == 1, [(w.t, w.sources, w.detail.get("score")) for w in wickets(out)]
    (w7,) = w7
    assert w7.t == 394.2 and w7.detail["call_t"] == 394.2
    assert Source.OCR in w7.sources and "asr_only" not in w7.hold_reasons
    assert "run_out" in w7.hold_reasons
    assert w7.detail["event_wicket_no"] == 7 and w7.detail["score_basis"] == "board"
    assert w7.detail["prev_change_t"] == 362.06 and w7.detail["innings_end_t"] == 400.96
    assert not [w for w in wickets(out) if Source.OCR not in w.sources]     # the call is not left over


def test_live_order_footer_finalizes_before_the_call():
    """Live: the call waits settle + 20 s and holds the footer back (time order); nothing is emitted early."""
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    for e in (wicket6(), call(394.2), footer(prev_change_t=362.06)):
        f.add(e)
    first = f.finalize(431.0)                          # footer settled (400.96 + 30), call not yet (394.2 + 50)
    assert [w.detail.get("score") for w in wickets(first)] == ["179/6"]
    assert not [e for e in first if e.type == EventType.INNINGS_END]
    later = f.finalize(445.0)
    (w7,) = wickets(later)
    assert w7.t == 394.2 and Source.OCR in w7.sources and w7.detail["score"] == "179/7"
    assert [e.type for e in later].count(EventType.INNINGS_END) == 1


def test_no_call_infers_a_held_wicket_before_the_footer():
    out = fuse([wicket6(), footer(prev_change_t=362.06)])
    (w7,) = [w for w in wickets(out) if w.detail.get("score") == "179/7"]
    assert 362.06 < w7.t < 400.96 and w7.t == 400.96 - 10
    assert {"innings_inferred", "no_call_anchor"} <= set(w7.hold_reasons) and w7.needs_review
    assert Source.OCR in w7.sources and w7.detail["wickets"] == 1


def test_replay_call_of_the_last_wicket_is_not_reused():
    """A call within 25 s after 179/6's confirmation is that wicket's replay: wicket 7 is inferred."""
    out = fuse([wicket6(), call(380.0), footer(prev_change_t=362.06)])
    (w7,) = [w for w in wickets(out) if w.detail.get("score") == "179/7"]
    assert "innings_inferred" in w7.hold_reasons


def test_call_before_the_last_board_change_is_not_used():
    """A call before 179/6 belongs to wicket 6 or earlier; wicket 7 is inferred instead."""
    out = fuse([call(340.0), wicket6(), footer(prev_change_t=362.06)])
    (w7,) = [w for w in wickets(out) if w.detail.get("score") == "179/7"]
    assert w7.t != 340.0 and "innings_inferred" in w7.hold_reasons


def test_two_unseen_wickets_one_call_one_inferred():
    out = fuse([ev(362.06, EventType.WICKET, Source.OCR, prev_score="177/4", prev_overs="19.4", score="179/5",
                   overs="19.5", runs=2, gap=False, prev_change_t=339.42, wickets=1, innings=1),
                call(392.0),            # > 25 s after 179/5: not one of its replay calls
                footer(unseen=2, prev_score="179/5", prev_change_t=362.06)])
    ws = sorted([w for w in wickets(out) if w.detail.get("confirm_source") == "innings_end"], key=lambda w: w.t)
    assert [w.detail["event_wicket_no"] for w in ws] == [6, 7]
    assert ws[0].t == 392.0 and "innings_inferred" not in ws[0].hold_reasons
    assert ws[1].detail["score"] == "179/7" and "innings_inferred" in ws[1].hold_reasons
    assert all(w.detail["score_basis"] == "unknown" for w in ws)


def test_no_unseen_wickets_changes_nothing():
    out = fuse([wicket6(), call(394.2), footer(unseen=0, score="179/6")])
    assert [w.detail.get("score") for w in wickets(out) if Source.OCR in w.sources] == ["179/6"]


def test_audit_clips_all_seven_innings_1_wickets():
    """The package audit: 179/7 is clipped, never named missing; the call is not withheld."""
    out = fuse([wicket6(), call(394.2), footer()])
    facts = MatchFacts.from_events(out)
    clips = [Clip(e, [(e.t - 5, e.t + 5)], Path(f"/tmp/{e.id}.mp4")) for e in wickets(out)]
    mp = match_plan(clips, 300, facts, "t", overs=None)
    a = mp.wicket_audit["innings_1"]
    assert a["wickets_on_board"] == 7
    assert a["wickets_clipped"] == 2                                 # 179/6 and 179/7 (only these in the test)
    assert not any("179/7" in t for t in a["not_clipped"]) and a["wicket_calls_not_on_scoreboard"] == []
    assert not any("179/7" in m or "commentary" in m for m in mp.wicket_audit.get("wicket_missing", []))
    assert not [w for w in mp.withheld if w["type"] == "wicket"]
