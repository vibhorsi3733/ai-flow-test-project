"""Captions + clip ranges on the edited demo source (times = stream seconds from the caption check report,
scratchpad/caption_check/report.md). Cuts are the ones measured by the planner on runtime_1080 (cuts.json)."""
import json

from motion_ai.events import Event, EventType, FusedEvent, Source
from motion_ai.fusion.fuser import Fuser, dismissal_score, next_ball
from motion_ai.highlights import clips as clips_mod
from motion_ai.highlights.clips import (Clip, apply_avoid, caption_for, clamp_caption, load_avoid_ranges,
                                        plan_ranges, strip_numbers)
from motion_ai.highlights.curator import default_caption

AVOID = [(494.0, 497.5)]   # YouTube SUBSCRIBE pop-up burned into the source


def loud(t0, t1, peak_t=None, base=0.1, peak=0.9):
    out = [(round(t0 + i * 0.1, 2), base) for i in range(int((t1 - t0) / 0.1))]
    return [(t, peak if peak_t is not None and abs(t - peak_t) < 0.05 else v) for t, v in out]


def gap_wicket(confirm, prev_t, score, overs, prev_score, prev_overs, call_t=None, quotes=(), holds=()):
    ev = FusedEvent(EventType.WICKET, call_t or confirm - 6, 0.95, {Source.OCR} | ({Source.ASR} if call_t else set()),
                    {"score": score, "overs": overs, "gap": True, "prev_change_t": prev_t, "confirm_t": confirm,
                     "call_t": call_t, "prev_score": prev_score, "prev_overs": prev_overs, "wickets": 1,
                     "quotes": list(quotes), "innings": 1})
    for h in holds:
        ev.hold(h)
    return ev


def inside(ranges, t):
    return any(a <= t <= b for a, b in ranges)


def caption_visible(ranges, cap_t):
    c = Clip(None, ranges, None, "", cap_t)
    return c.duration - c.offset_of(cap_t) >= 3.0 - 1e-6 and inside(ranges, cap_t)   # every caption >= 3s


# --- score at the dismissal ---------------------------------------------------------------------------------
def test_next_ball_handles_over_end():
    assert next_ball("2.4") == "2.5" and next_ball("12.0") == "12.1" and next_ball("11.6") == "12.1"
    assert next_ball("19.2") == "19.3" and next_ball(None) is None


def test_gap_wicket_captions_have_no_score_but_data_keeps_it():
    cases = [  # board after the gap, previous board, score AT the dismissal (report fixes 4, 5, 7, 9-13)
        ("69/1", "7.4", "31/0", "2.4", "31/1"),
        ("112/2", "12.2", "104/1", "11.2", "104/2"),
        ("127/3", "14.2", "112/2", "12.2", "112/3"),
        ("155/4", "17.4", "127/3", "14.2", "127/4"),
        ("171/5", "18.5", "155/4", "17.4", "155/5"),
        ("157/4", "18.0", "139/3", "16.1", "139/4"),
        ("177/5", "19.4", "177/4", "19.2", "177/5"),
    ]
    for score, overs, ps, po, event_score in cases:
        ev = gap_wicket(500, 480, score, overs, ps, po)
        assert caption_for(ev) == "WICKET"
        assert "score_inferred" in ev.hold_reasons
        assert ev.detail["event_score"] == event_score      # manifest data unchanged


def test_gap_run_out_keeps_commentary_label():
    ev = gap_wicket(234.42, 186.02, "116/3", "14.0", "99/2", "12.0", call_t=191.24,
                    quotes=["That's a quick single, that's a direct hit as well."], holds=("run_out", "review"))
    assert caption_for(ev) == "RUN OUT · DIRECT HIT"
    assert ev.detail["event_overs"] == "12.1"


def test_run_out_off_a_wide_gets_a_safe_caption():
    # 177/5 19.4 -> 179/6 19.5 (not a gap): truth 178/6 (19.4), run out off a wide, then a wide and a dot
    ev = FusedEvent(EventType.WICKET, 343.24, 0.95, {Source.OCR, Source.ASR},
                    {"score": "179/6", "overs": "19.5", "runs": 2, "wickets": 1, "gap": False,
                     "prev_score": "177/5", "prev_overs": "19.4", "prev_change_t": 339.42, "confirm_t": 362.06,
                     "call_t": 343.24, "quotes": ["Surya runs, doesn't matter, Makran will be run out."]})
    assert caption_for(ev) == "RUN OUT"
    assert "score_inferred" in ev.hold_reasons and "event_score" not in ev.detail


def test_one_delivery_wicket_trusts_the_board_and_old_events_fall_back():
    d = {"score": "6/1", "overs": "0.5", "runs": 0, "wickets": 1, "gap": False}
    assert dismissal_score(d) == {"score_basis": "board"}
    # gap without prev_score (older events): never print the later ball's score
    assert dismissal_score({"score": "69/1", "overs": "7.4", "gap": True, "wickets": 1}) == {
        "score_basis": "unknown", "event_wicket_no": 1}
    ev = FusedEvent(EventType.WICKET, 455, 0.95, {Source.OCR}, {"score": "69/1", "overs": "7.4", "gap": True})
    assert caption_for(ev) == "WICKET"


def test_gap_boundary_has_no_score():
    ev = FusedEvent(EventType.FOUR, 177.6, 0.6, {Source.OCR, Source.ASR}, {"score": "99/2", "overs": "12.0",
                                                                           "gap": True, "confirm_t": 186.02})
    assert caption_for(ev) == "FOUR"


# --- gap wicket ranges ---------------------------------------------------------------------------------------
def test_69_1_catch_is_in_the_clip_and_caption_after_it():
    ev = gap_wicket(461.2, 433.0, "69/1", "7.4", "31/0", "2.4")
    ev.detail["prev_is_innings_start"] = True       # 31/0 bar on screen from 431.3, innings-start read 433.0
    ranges, cap_t = plan_ranges(ev, [434.49, 442.83, 444.81, 448.33], loud(433, 449, peak_t=443.8))
    (a, b), = ranges
    assert a <= 432.1 and b < 460.8 and b - a <= 25.0          # run-up 432.1-433, delivery 433.3; before the board
    assert 439 <= cap_t <= 446 and caption_visible(ranges, cap_t)


def test_112_2_avoids_the_subscribe_popup():
    ev = gap_wicket(502.52, 473.36, "112/2", "12.2", "104/1", "11.2")
    cuts = [472.31, 475.02, 477.46, 481.89, 486.16, 487.84, 496.18]
    ranges, cap_t = plan_ranges(ev, cuts, loud(473, 489, peak_t=481.8), AVOID)
    assert ranges[0][0] <= 475.0 and inside(ranges, 481.0)           # live catch 475-481
    assert not any(a < 497.5 and b > 494.0 for a, b in ranges)      # no frame of the pop-up
    assert all(b < 502.52 for _, b in ranges)                        # 12.3 ball bowled 502.8
    assert 481 <= cap_t <= 485 and caption_visible(ranges, cap_t)


def test_127_3_155_4_171_5_157_4_cover_delivery_and_end_before_next_ball():
    cases = [  # confirm, prev, board, prev board, cuts, peak, must contain, end before
        (533.12, 502.52, "127/3", "14.2", "112/2", "12.2", [503.74, 515.42, 516.9], 510.4, [502.8, 507, 514], 535),
        (562.68, 533.12, "155/4", "17.4", "127/3", "14.2", [534.96, 536.44, 543.17, 545.65], 540.3,
         [537.5, 541], 563.0),
        (581.56, 562.68, "171/5", "18.5", "155/4", "17.4", [], 572.8, [563.0, 564.0], 581.5),
        (287.2, 266.36, "157/4", "18.0", "139/3", "16.1", [268.35, 276.09, 278.09], 270.2, [267, 275.1], 287.3),
    ]
    for confirm, prev, s, o, ps, po, cuts, peak, contain, before in cases:
        ev = gap_wicket(confirm, prev, s, o, ps, po)
        ranges, cap_t = plan_ranges(ev, cuts, loud(prev, prev + 16, peak_t=peak), AVOID)
        assert all(inside(ranges, t) for t in contain), (s, ranges)
        assert ranges[-1][1] < before and sum(b - a for a, b in ranges) <= 25.0 + 1e-6, (s, ranges)
        assert cap_t >= prev + 1.3 and caption_visible(ranges, cap_t), (s, cap_t)
        assert ranges[0][0] >= prev - 1.5, (s, ranges)              # no frames of the previous ball


def test_177_5_caption_is_visible():
    # old plan: caption_t == clip end (339.42) -> zero-length SRT cue
    ev = gap_wicket(339.42, 312.0, "177/5", "19.4", "177/4", "19.2")
    ranges, cap_t = plan_ranges(ev, [309.08, 319.42, 323.39, 326.99, 328.19], loud(312, 328, peak_t=317.1))
    assert 311.4 <= ranges[0][0] <= 312.0 and 336.0 <= ranges[-1][1] < 339.42   # bar 161-4 until ~312; replay catch 334-335
    assert caption_visible(ranges, cap_t) and cap_t >= 320.0


def test_116_3_gap_run_out_with_call():
    ev = gap_wicket(234.42, 186.02, "116/3", "14.0", "99/2", "12.0", call_t=191.24,
                    quotes=["direct hit", "review for a runout"], holds=("run_out", "review", "slow_confirm"))
    ranges, cap_t = plan_ranges(ev, [184.92, 187.46, 193.41, 238.03], loud(234, 243))
    assert ranges[0][0] <= 185.9 and ranges[0][1] >= 191.24                # delivery 185.9, direct hit
    assert ranges[1] == (224.42, 234.3)                                    # report fix 4: replay, not the 14.1 ball
    assert 224.4 <= cap_t <= 225.0 and caption_visible(ranges, cap_t)


def test_179_6_no_replay_repeat_and_no_half_last_ball():
    ev = FusedEvent(EventType.WICKET, 343.24, 0.95, {Source.OCR, Source.ASR},
                    {"score": "179/6", "overs": "19.5", "runs": 2, "wickets": 1, "gap": False,
                     "prev_score": "177/5", "prev_overs": "19.4", "prev_change_t": 339.42, "confirm_t": 362.06,
                     "call_t": 343.24, "quotes": ["Makran will be run out."]})
    ev.hold("run_out")
    cuts = [334.32, 335.44, 344.13, 345.89, 348.89, 353.69, 365.95]
    ranges, cap_t = plan_ranges(ev, cuts, loud(362, 371))
    assert ranges[0][0] >= 338.9                       # clip 7 (177/5) footage 335.4-339.4 not repeated
    assert ranges[-1][1] <= 362.3                      # 19.6 ball released 363.1
    assert caption_visible(ranges, cap_t)


# --- boundaries ---------------------------------------------------------------------------------------------
def test_ocr_only_four_starts_before_the_shot():
    ev = FusedEvent(EventType.FOUR, 294.16, 0.95, {Source.OCR},
                    {"score": "161/4", "overs": "18.1", "gap": False, "prev_change_t": 287.2, "confirm_t": 300.16})
    ranges, cap_t = plan_ranges(ev, [289.23, 295.71, 299.19, 301.69], loud(284, 303, peak_t=289.5))
    (a, b), = ranges
    assert 287.0 <= a <= 288.5 and b <= 300.16           # shot 288.6-288.9; next ball at 300.2
    assert 289.0 <= cap_t <= 296.2 and caption_visible(ranges, cap_t)


def test_winning_runs_include_delivery():
    ev = FusedEvent(EventType.MATCH_END, 602.8, 0.95, {Source.OCR},
                    {"score": "181/5", "overs": "19.2", "gap": False, "prev_change_t": 593.56, "confirm_t": 608.8})
    ranges, cap_t = plan_ranges(ev, [590.71, 593.57, 596.11, 602.59, 604.47, 606.49], loud(592, 618, peak_t=601.4))
    (a, b), = ranges
    assert a <= 593.8 and b >= 604                        # delivery 594.0, shot 595.0
    assert cap_t > 595.0 and caption_visible(ranges, cap_t)


def test_asr_four_caption_when_ball_reaches_rope():
    ev = FusedEvent(EventType.FOUR, 56.8, 0.85, {Source.OCR, Source.ASR},
                    {"score": "6/0", "overs": "0.3", "confirm_t": 63.1, "prev_change_t": 56.08})
    ranges, cap_t = plan_ranges(ev, [51.48, 56.88, 61.56, 63.24], loud(56.8, 66))
    assert ranges[0][0] <= 55.5 and 58.5 <= cap_t <= 59.5   # delivery 55.5; rope 58.5-59.5


# --- helpers ------------------------------------------------------------------------------------------------
def test_apply_avoid_splits_and_drops_slivers():
    assert apply_avoid([(480, 510)], AVOID) == [(480, 493.96), (497.54, 510)]
    assert apply_avoid([(480, 498.5)], AVOID) == [(480, 493.96)]        # 0.96s tail dropped
    assert apply_avoid([(400, 420)], AVOID) == [(400, 420)]


def test_clamp_caption():
    assert clamp_caption([(319.42, 339.42)], 339.42) == 336.42          # >= 3s on screen (fix round 2)
    assert clamp_caption([(319.42, 339.42)], 339.42, tail=1.5) == 337.92
    assert clamp_caption([(10, 20), (30, 40)], 25) == 30
    assert clamp_caption([(10, 20)], 5) == 10
    assert clamp_caption([(10, 20)], 15) == 15


def test_avoid_ranges_loaded_from_match_config(tmp_path):
    p = tmp_path / "match.json"
    p.write_text(json.dumps({"teams": {}, "avoid_ranges": [[494.0, 497.5]]}))
    assert load_avoid_ranges(p) == [(494.0, 497.5)]
    assert load_avoid_ranges(tmp_path / "missing.json") == []


# --- fuser: each gap wicket owns its own calls ----------------------------------------------------------------
def _e(t, type_, src, **detail):
    return Event(type_, t, src, 0.9 if src == Source.ASR else 0.95, detail)


def test_gap_wickets_get_their_own_calls():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(_e(191.24, EventType.WICKET, Source.ASR, quote="that's a direct hit as well."))
    f.add(_e(195.04, EventType.REVIEW, Source.ASR, quote="We have a review for a runout at the strikers end."))
    f.add(_e(234.42, EventType.WICKET, Source.OCR, score="116/3", overs="14.0", runs=17, gap=True,
             prev_change_t=186.02, prev_score="99/2", prev_overs="12.0", wickets=1, innings=1))
    f.add(_e(339.42, EventType.WICKET, Source.OCR, score="177/5", overs="19.4", runs=0, gap=True,
             prev_change_t=312.0, prev_score="177/4", prev_overs="19.2", wickets=1, innings=1))
    f.add(_e(343.24, EventType.WICKET, Source.ASR, quote="Surya runs, doesn't matter, Makran will be run out."))
    f.add(_e(362.06, EventType.WICKET, Source.OCR, score="179/6", overs="19.5", runs=2, gap=False,
             prev_change_t=339.42, prev_score="177/5", prev_overs="19.4", wickets=1, innings=1))
    out = {e.detail["score"]: e for e in f.finalize(1000) if e.type == EventType.WICKET}
    assert out["116/3"].detail["call_t"] == 191.24 and caption_for(out["116/3"]) == "RUN OUT · DIRECT HIT"
    assert out["177/5"].detail["call_t"] is None and caption_for(out["177/5"]) == "WICKET"
    assert out["179/6"].detail["call_t"] == 343.24 and caption_for(out["179/6"]) == "RUN OUT"
    assert "run_out" not in out["177/5"].hold_reasons


def test_first_change_after_innings_start_is_flagged():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(_e(433.0, EventType.INNINGS_START, Source.OCR, score="31/0", overs="2.4", innings=2))
    f.add(_e(461.2, EventType.WICKET, Source.OCR, score="69/1", overs="7.4", runs=38, gap=True, prev_change_t=433.0,
             prev_score="31/0", prev_overs="2.4", wickets=1, innings=2))
    f.add(_e(502.52, EventType.WICKET, Source.OCR, score="112/2", overs="12.2", runs=8, gap=True,
             prev_change_t=473.36, prev_score="104/1", prev_overs="11.2", wickets=1, innings=2))
    w = {e.detail["score"]: e for e in f.finalize(1000) if e.type == EventType.WICKET}
    assert w["69/1"].detail.get("prev_is_innings_start") and not w["112/2"].detail.get("prev_is_innings_start")


# --- captions carry no numbers ----------------------------------------------------------------------------------
def test_no_caption_contains_a_digit():
    d = {"score": "69/1", "overs": "7.4", "gap": False, "prev_score": "68/0", "prev_overs": "7.3", "wickets": 1,
         "innings": 2, "runs": 1}
    evs = [FusedEvent(t, 100.0, 0.95, {Source.OCR}, dict(d)) for t in EventType]
    evs.append(FusedEvent(EventType.TOSS, 1.0, 0.95, {Source.ASR},
                          {"toss_winner": "Mumbai Indians", "toss_decision": "bowl"}))
    evs += [gap_wicket(500, 480, "69/1", "7.4", "31/0", "2.4"),
            FusedEvent(EventType.WICKET, 455, 0.95, {Source.OCR}, {"score": "69/1", "overs": "7.4", "gap": True}),
            gap_wicket(234.42, 186.02, "116/3", "14.0", "99/2", "12.0", call_t=191.24, quotes=["direct hit"])]
    for ev in evs:
        cap = caption_for(ev)
        assert cap and not any(ch.isdigit() for ch in cap), (ev.type, cap)
        assert "Mumbai" not in cap


def test_final_caption_strips_scores_from_any_source():
    assert strip_numbers("WINNING RUNS · FOUR · 181/5 (19.2 ov)") == "WINNING RUNS · FOUR"
    assert strip_numbers("RUN OUT · 6th WICKET") == "RUN OUT"
    ev = FusedEvent(EventType.SIX, 100.0, 0.95, {Source.OCR}, {"score": "112/3", "overs": "12.3"})
    assert default_caption(Clip(ev, [(90, 105)], None, "SIX · 112/3 (12.3 ov)", 100)) == "SIX"   # old manifest
    assert default_caption(Clip(ev, [(90, 105)], None, "1st WICKET", 100)) == "SIX"              # nothing left


def test_scores_can_be_turned_back_on(monkeypatch):
    monkeypatch.setattr(clips_mod, "CAPTION_SHOW_SCORE", True)
    ev = gap_wicket(500, 480, "69/1", "7.4", "31/0", "2.4")
    assert caption_for(ev) == "WICKET · 31/1 (2.5 ov)"


def test_late_commentary_call_does_not_move_a_gap_wicket_clip():
    """Claude calls are often spoken over the replay: 104/2 catch live 475-481 s, call 493.4 s.
    The clip must still start at the previous board change (472.3 s), not after the catch."""
    from motion_ai.events import EventType, FusedEvent, Source
    from motion_ai.highlights.clips import plan_ranges
    ev = FusedEvent(EventType.WICKET, 493.4, 0.75, {Source.OCR, Source.ASR},
                    {"score": "112/2", "overs": "12.2", "gap": True, "innings": 2, "wickets": 1,
                     "prev_score": "104/1", "prev_overs": "11.2", "prev_change_t": 473.36,
                     "confirm_t": 502.52, "call_t": 493.4, "quotes": ["And holds out into the deep"]})
    ranges, cap_t = plan_ranges(ev, [472.31, 481.84, 497.54], [])
    assert ranges[0][0] <= 475.0 and ranges[-1][1] >= 481.0       # the live catch is inside the clip
    assert ranges[0][0] <= cap_t <= ranges[-1][1]


# --- fix round 2: caption >= 3s at/after the outcome (ground truth from source frames) -----------------------
def test_178_6_run_out_off_a_wide_includes_the_live_ball():
    # change 339.42; run-up 340.0, wide released 341.5, keeper 342-343.5, run out at the bowler's end ~345,
    # umpire signals wide 347-353, WICKET ticker 355, replay 356-361.5, board 362.06, next ball 363.1.
    # The only call ("Also going to result in another runout") is spoken over the replay at 356.06.
    ev = FusedEvent(EventType.WICKET, 356.06, 0.95, {Source.OCR, Source.ASR},
                    {"score": "179/6", "overs": "19.5", "runs": 2, "wickets": 1, "gap": False, "innings": 1,
                     "prev_score": "177/5", "prev_overs": "19.4", "prev_change_t": 339.42, "confirm_t": 362.06,
                     "call_t": None, "quotes": ["Also going to result in another runout."]})
    ev.hold("run_out")
    cuts = [330.92, 334.32, 335.44, 344.13, 345.89, 348.89, 353.69, 365.95]
    ranges, cap_t = plan_ranges(ev, cuts, loud(339, 371))
    assert 338.9 <= ranges[0][0] <= 340.0                       # live run-up, not the umpire signalling
    assert all(inside(ranges, t) for t in (341.5, 345.0, 356.5, 361.0))   # wide, run out, replay
    assert ranges[-1][1] <= 362.3 and sum(b - a for a, b in ranges) <= 25.0
    assert cap_t >= 356.0 and caption_visible(ranges, cap_t)    # after the outcome, on the replay


def test_179_7_last_ball_run_out_ends_before_the_batting_card():
    # delivery 384.5, dive 386.5-388, throw relay, stumps broken 397.4, card fades in 397.75, footer read 400.96
    ev = FusedEvent(EventType.WICKET, 394.2, 0.9, {Source.ASR, Source.OCR},
                    {"score": "179/7", "overs": "20.0", "wickets": 1, "innings": 1, "prev_score": "179/6",
                     "prev_overs": "19.5", "prev_change_t": 362.06, "confirm_source": "innings_end",
                     "innings_end_t": 400.96, "call_t": 394.2, "quotes": ["And one more easy run out."]})
    ev.hold("run_out")
    ranges, cap_t = plan_ranges(ev, [], loud(394, 403))
    assert ranges[0][0] <= 384.5 and 397.45 <= ranges[-1][1] <= 397.7
    assert cap_t >= 394.2 and caption_visible(ranges, cap_t)
    assert ranges[-1][1] - cap_t <= 3.05                          # as close to the break as 3s allows


def test_177_5_six_caption_when_the_ball_lands():
    # change 581.56, hit 582.5, in the stands 583.0-583.5, seats until 588.5, batter 589-590.3, board 590.56
    ev = FusedEvent(EventType.SIX, 584.56, 0.95, {Source.OCR},
                    {"score": "177/5", "overs": "19.0", "runs": 6, "gap": False, "innings": 2,
                     "prev_score": "171/5", "prev_overs": "18.5", "prev_change_t": 581.56, "confirm_t": 590.56})
    ranges, cap_t = plan_ranges(ev, [582.95, 589.36, 590.76], loud(565, 592, peak_t=584.36))
    (a, b), = ranges
    assert a <= 581.5 and 589.0 <= b < 590.56
    assert 583.0 <= cap_t <= 584.0 and caption_visible(ranges, cap_t)


def test_winning_runs_label_is_decided_before_the_burn():
    d = {"score": "181/5", "overs": "19.2", "runs": 4, "gap": False, "prev_change_t": 593.56,
         "confirm_t": 608.8, "target": 180, "innings": 2}
    ev = FusedEvent(EventType.MATCH_END, 602.8, 0.95, {Source.OCR}, dict(d))
    assert caption_for(ev) == "WINNING RUNS · FOUR"
    assert caption_for(FusedEvent(EventType.MATCH_END, 1, 0.95, {Source.OCR}, d | {"runs": 6})) == "WINNING RUNS · SIX"
    assert caption_for(FusedEvent(EventType.MATCH_END, 1, 0.95, {Source.OCR}, d | {"runs": 2})) == "WINNING RUNS"
    assert caption_for(FusedEvent(EventType.MATCH_END, 1, 0.95, {Source.OCR}, d | {"gap": True})) == "WINNING RUNS"
    # delivery 594.0, hit 595.0, ball to the rope ~597-598.5; crowd peak 601.4 is the celebration
    ranges, cap_t = plan_ranges(ev, [596.11, 602.63, 604.51, 606.53], loud(578, 618, peak_t=601.4))
    assert ranges[0][0] <= 594.0 and 596.5 <= cap_t <= 598.5 and caption_visible(ranges, cap_t)


def test_winning_runs_fold_never_relabels_a_burned_clip(tmp_path):
    from motion_ai.highlights.curator import MatchFacts, match_plan
    def clip(etype, path, caption):
        ev = FusedEvent(etype, 602.8, 0.95, {Source.OCR}, {"score": "181/5", "overs": "19.2", "innings": 2,
                                                         "runs": 4, "gap": False})
        return Clip(ev, [(593.6, 606.5)], path, caption, 597.6)
    burned = tmp_path / "me.mp4"
    burned.write_bytes(b"x")
    for path, cap, want in [(burned, "WINNING RUNS", "WINNING RUNS"),               # burned: keep what is on screen
                            (tmp_path / "new.mp4", "WINNING RUNS", "WINNING RUNS · FOUR"),
                            (burned, "WINNING RUNS · FOUR", "WINNING RUNS · FOUR")]:  # never doubled
        me, four = clip(EventType.MATCH_END, path, cap), clip(EventType.FOUR, None, "FOUR")
        mp = match_plan([me, four], 300, MatchFacts(wicket_changes={1: [], 2: []}, innings_end={}), "t", overs=None)
        assert [c for c, _ in mp.plan.items] == [me] and mp.plan.items[0][1] == want and me.caption == want
