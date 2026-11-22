"""Recorded 1080p demo run (runtime_1080/events.jsonl) -> fusion -> MatchFacts, no media calls.

Fuses exactly like motion_ai/replay.py fuse_recorded (OCR on: settle 30 s, lag 6 s) and checks the facts
the package cards and the wicket audit are built from. IPL head condition 2: every innings-2 wicket change
is present. Broadcaster scorecard: 179-7 (FoW 6, 94, 99, 139, 177, 178, 179), chase 181/5 in 19.2 ov.
The 7th first-innings wicket (179/7, last-ball run-out) never showed as a score change: only the confirmed
INNINGS_END read can carry it into the audit.
"""
from pathlib import Path

import pytest

from motion_ai.events import EventType
from motion_ai.fusion.fuser import Fuser
from motion_ai.highlights.curator import MatchFacts, match_plan
from motion_ai.recorder import load_events

RUN = Path(__file__).resolve().parent.parent / "runtime_1080"

pytestmark = pytest.mark.skipif(not (RUN / "events.jsonl").exists(), reason="runtime_1080/events.jsonl missing")


@pytest.fixture(scope="module")
def fused():
    fuser = Fuser(True, 30.0, 6.0)          # Settings(ocr_available=True).fusion_settle_seconds / ocr_lag_seconds
    for e in sorted(load_events(RUN), key=lambda e: e.t):
        fuser.add(e)
    return fuser.finalize(float("inf"))


@pytest.fixture(scope="module")
def facts(fused):
    return MatchFacts.from_events(fused)


def test_innings_2_starts_at_the_31_0_join(fused, facts):
    (start,) = [e for e in fused if e.type == EventType.INNINGS_START and e.detail.get("innings") == 2]
    assert 425 <= facts.innings2_t <= 440
    assert 428 <= start.detail.get("confirm_t", start.t) <= 440            # confirmed ~433 s
    assert facts.innings1_runs == 179


def test_target_and_result_confirmed(facts):
    assert facts.target == 180 and facts.target_matches_innings1
    assert facts.result_score == (181, 5)


def test_ocr_wicket_changes_both_innings(facts):
    assert [w["score"] for w in facts.wicket_changes[1]] == ["6/1", "94/2", "116/3", "157/4", "177/5", "179/6"]
    assert [w["score"] for w in facts.wicket_changes[2]] == ["69/1", "112/2", "127/3", "155/4", "171/5"]


def test_innings_end_179_7_is_read(facts):
    assert facts.innings_end[1]["score"] == "179/7" and facts.innings_end[1]["overs"] == "20.0"
    assert facts.innings_end[1].get("unseen_wickets") == 1          # the last-ball run-out, never on the bar
    assert facts.innings_end[2]["score"] == "181/5" and facts.innings_end[2]["overs"] == "19.2"


def test_every_wicket_change_is_eligible_for_the_plan(fused, facts):
    """No clip media: the audit only needs the wicket ids, so plan with one synthetic 10 s clip per wicket."""
    from motion_ai.highlights.clips import Clip
    clips = [Clip(e, [(e.t - 5, e.t + 5)], Path(f"/tmp/{e.id}.mp4")) for e in fused
             if e.type == EventType.WICKET and e.id in {w["id"] for i in (1, 2) for w in facts.wicket_changes[i]}]
    mp = match_plan(clips, 300, facts, "t", overs=None)
    for i, n in ((1, 6), (2, 5)):
        assert mp.wicket_audit[f"innings_{i}"]["wickets_clipped"] == n
    # 179/7 (last-ball run-out) was never a score change: INNINGS_END makes it 7 on board and names it
    assert mp.wicket_audit["innings_1"]["wickets_on_board"] == 7
    assert "innings 1: 179/7 (20.0 ov) not clipped" in mp.wicket_audit["wicket_missing"]
    assert mp.wicket_audit["innings_2"]["wickets_on_board"] == 5
