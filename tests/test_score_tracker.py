from motion_ai.detectors.score_tracker import ScoreTracker, parse_reading
from motion_ai.events import EventType


def feed(tracker, t, score, overs, n=3):
    """Feed n identical frames (0.5s apart) and return emitted event types."""
    out = []
    for i in range(n):
        out += tracker.update(t + i * 0.5, parse_reading(score, overs))
    return [e.type for e in out]


def test_parse():
    s = parse_reading("MI 45/1", "4.3")
    assert (s.runs, s.wickets, s.over, s.ball) == (45, 1, 4, 3)
    assert parse_reading("", "4.3") is None
    assert parse_reading("45/13", "4.3") is None  # impossible wickets


def test_six_four_wicket_and_powerplay_end():
    tr = ScoreTracker(powerplay_overs=6)
    assert feed(tr, 0, "0/0", "0.0") == [EventType.INNINGS_START]
    assert feed(tr, 10, "6/0", "0.1") == [EventType.SIX]
    assert feed(tr, 20, "10/0", "0.2") == [EventType.FOUR]
    assert feed(tr, 30, "10/1", "0.3") == [EventType.WICKET]
    feed(tr, 40, "40/1", "5.5")  # gap (scoreboard hidden) — no six/four guessed
    assert feed(tr, 50, "41/1", "6.0") == [EventType.OVER_COMPLETE, EventType.POWERPLAY_END]
    assert EventType.POWERPLAY_END not in feed(tr, 60, "42/1", "7.0")  # only once


def test_flicker_and_misread_rejected():
    tr = ScoreTracker()
    feed(tr, 0, "20/0", "2.1")
    assert feed(tr, 5, "26/0", "2.2", n=2) == []          # not enough consensus
    assert feed(tr, 10, "80/0", "2.2") == []              # +60 in one ball → misread
    assert feed(tr, 15, "2/0", "2.2") == []               # score went down → misread
    assert feed(tr, 20, "26/0", "2.2") == [EventType.SIX]


def test_scoreline_real_ocr_readings():
    from motion_ai.detectors.score_tracker import parse_scoreline
    # Actual PaddleOCR outputs from the demo feed (spaces dropped, "P" powerplay badge)
    cases = {"2-00.2": (2, 0, 0, 2), "6-00.4": (6, 0, 0, 4), "86-19.3": (86, 1, 9, 3),
             "116-314": (116, 3, 14, 0), "157-418.1": (157, 4, 18, 1), "2-0 0.2 P": (2, 0, 0, 2)}
    for text, want in cases.items():
        s = parse_scoreline(text)
        assert (s.runs, s.wickets, s.over, s.ball) == want, text
    assert parse_scoreline("NEED3") is None


def test_powerplay_end_on_jump_and_joined_in_progress():
    tr = ScoreTracker(powerplay_overs=6)
    assert feed(tr, 0, "2/0", "0.2") == [EventType.INNINGS_START]     # joined mid-innings
    types = feed(tr, 10, "55/1", "6.1")                                # highlights reel jump past 6.0
    assert EventType.POWERPLAY_END in types


def test_misread_overs_jump_rejected():
    tr = ScoreTracker(powerplay_overs=6)
    feed(tr, 0, "10/0", "1.3")
    assert feed(tr, 10, "10/0", "14") == []    # "1.4" read as "14" with no runs → rejected


# --- prev_score / prev_overs on every scoreboard event (fix round contract) ------------------------------------
def _events(tracker, t, score, overs, n=3):
    out = []
    for i in range(n):
        out += tracker.update(t + i * 0.5, parse_reading(score, overs))
    return out


def test_prev_score_and_prev_overs_on_every_event():
    tr = ScoreTracker(powerplay_overs=6)
    start = _events(tr, 0, "2/0", "0.2")[0]
    assert start.detail["prev_score"] is None and start.detail["prev_overs"] is None
    six = _events(tr, 10, "8/0", "0.3")[0]
    assert (six.detail["prev_score"], six.detail["prev_overs"]) == ("2/0", "0.2")
    # gap wicket (edited highlights: 31-0 2.4 -> 69-1 7.4): prev_* is the state BEFORE the change
    _events(tr, 20, "31/0", "2.4")
    evs = _events(tr, 30, "69/1", "7.4")
    assert {e.type for e in evs} == {EventType.WICKET, EventType.POWERPLAY_END}
    for e in evs:
        assert (e.detail["prev_score"], e.detail["prev_overs"]) == ("31/0", "2.4")
        assert (e.detail["score"], e.detail["overs"]) == ("69/1", "7.4")


def test_prev_score_on_new_innings_and_target():
    from motion_ai.detectors.score_tracker import ScoreState
    tr = ScoreTracker(6, 3)
    for i in range(3):
        tr.update(i * 0.5, ScoreState(179, 6, 19, 5))
    evs = []
    for i in range(7):
        evs += tr.update(432 + i * 0.5, ScoreState(31, 0, 2, 4))
    assert evs[0].type == EventType.INNINGS_START and evs[0].detail["prev_score"] == "179/6"
    tr.update_target(436, "TARGET - 180")
    tgt = tr.update_target(437.5, "TARGET - 180")[0]
    assert tgt.detail["prev_score"] == "31/0" and tgt.detail["prev_overs"] == "2.4"


# --- INNINGS_END from the scorecard footer ---------------------------------------------------------------------
def test_parse_footer_real_ocr_readings():
    from motion_ai.detectors.score_tracker import ScoreState, parse_footer
    # Actual PaddleOCR outputs on the 'footer' region of the demo feed (398-412 s and 617-635 s)
    assert parse_footer("OVERS20TOTAL179-7") == ScoreState(179, 7, 20, 0)
    assert parse_footer("OVERS19.2TOTAL181-5") == ScoreState(181, 5, 19, 2)
    assert parse_footer("OVERS 19.2TOTAL181-5") == ScoreState(181, 5, 19, 2)
    for junk in ("OVERS200TOTAA79977", "179-619.5VE4WD", "BY5WICKES", "VERS20", "", "OVERS20TOTAL179-12"):
        assert parse_footer(junk) is None, junk


def _innings1_at(state):
    tr = ScoreTracker(6, 3)
    for i in range(3):
        tr.update(i * 0.5, state)
    return tr


def test_last_ball_wicket_confirmed_by_footer():
    """Demo: bar's last read 179/6 (19.5); the final-ball run-out only shows on the scorecard (179-7, 20 ov)."""
    from motion_ai.detectors.score_tracker import ScoreState
    tr = _innings1_at(ScoreState(179, 6, 19, 5))
    assert tr.wants_innings_total()
    assert tr.update_innings_total(398.0, "OVERS20TOTAL179-7") == []
    assert tr.update_innings_total(398.5, "OVERS20TOTAL179-7") == []          # < 1 s apart
    evs = tr.update_innings_total(399.0, "OVERS20TOTAL179-7")
    assert [e.type for e in evs] == [EventType.INNINGS_END]
    d = evs[0].detail
    assert (d["innings"], d["score"], d["overs"], d["source"]) == (1, "179/7", "20.0", "scorecard_footer")
    assert (d["prev_score"], d["prev_overs"], d["unseen_wickets"]) == ("179/6", "19.5", 1)
    assert d["prev_change_t"] == 0.0          # last bar change: the unseen wicket fell after it (fuser window)
    assert EventType.INNINGS_END in __import__("motion_ai.events", fromlist=["x"]).OCR_AUTHORITATIVE
    assert not tr.wants_innings_total()
    assert tr.update_innings_total(405.0, "OVERS20TOTAL179-7") == []          # once per innings
    # the bar flashing the final state later is not a new wicket
    assert [e for i in range(3) for e in tr.update(410 + i * 0.5, ScoreState(179, 7, 20, 0))] == []
    # and the chase still starts innings 2
    evs = []
    for i in range(7):
        evs += tr.update(432 + i * 0.5, ScoreState(31, 0, 2, 4))
    assert [e.type for e in evs] == [EventType.INNINGS_START] and tr.innings == 2


def test_chase_total_after_match_end():
    from motion_ai.detectors.score_tracker import ScoreState
    tr = _innings1_at(ScoreState(179, 6, 19, 5))
    tr.update_innings_total(398, "OVERS20TOTAL179-7")
    tr.update_innings_total(399, "OVERS20TOTAL179-7")
    for i in range(7):
        tr.update(432 + i * 0.5, ScoreState(31, 0, 2, 4))
    tr.update_target(436, "TARGET - 180")
    tr.update_target(437.5, "TARGET - 180")
    for i in range(3):
        tr.update(600 + i * 0.5, ScoreState(177, 5, 19, 0))
    assert tr.wants_innings_total()
    assert tr.update_innings_total(605, "OVERS20TOTAL179-7") == []    # innings-1 recap card is ignored
    ends = []
    for i in range(3):
        ends += tr.update(608.8 + i * 0.5, ScoreState(181, 5, 19, 2))
    assert EventType.MATCH_END in [e.type for e in ends]
    tr.update_innings_total(617.0, "OVERS19.2TOTAL181-5")
    evs = tr.update_innings_total(618.0, "OVERS19.2TOTAL181-5")
    assert [(e.type, e.detail["innings"], e.detail["score"], e.detail["overs"]) for e in evs] == [
        (EventType.INNINGS_END, 2, "181/5", "19.2")]
    assert tr.innings_end == {1: ScoreState(179, 7, 20, 0), 2: ScoreState(181, 5, 19, 2)}


def test_footer_never_guessed():
    from motion_ai.detectors.score_tracker import ScoreState
    # mid-innings card (not a completed innings): ignored
    tr = _innings1_at(ScoreState(150, 4, 18, 0))
    assert tr.update_innings_total(1, "OVERS18TOTAL150-4") == [] and tr.update_innings_total(3, "OVERS18TOTAL150-4") == []
    # lower than the bar (misread): ignored
    tr = _innings1_at(ScoreState(179, 6, 19, 5))
    assert tr.update_innings_total(1, "OVERS20TOTAL170-7") == [] and tr.update_innings_total(3, "OVERS20TOTAL170-7") == []
    # two different reads do not confirm each other
    assert tr.update_innings_total(5, "OVERS20TOTAL179-7") == [] and tr.update_innings_total(7, "OVERS20TOTAL179-8") == []
    # early in the innings the footer is not even read
    assert not _innings1_at(ScoreState(60, 2, 8, 0)).wants_innings_total()
