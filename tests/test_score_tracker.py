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
