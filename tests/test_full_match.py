"""Full-match scope (panel spec fm/guidance_v1 §5 + IPL head conditions)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

from motion_ai.detectors.score_tracker import ScoreState, ScoreTracker, parse_scoreline
from motion_ai.events import Event, EventType, FusedEvent, Source
from motion_ai.fusion.fuser import Fuser
from motion_ai.highlights.clips import Clip, plan_ranges
from motion_ai.highlights.curator import MatchFacts, match_plan
from motion_ai.highlights.pipeline import HighlightPipeline
from motion_ai.highlights.renderer import card_texts, crowd_windows
from motion_ai.media import scale_filter, video_opts


def feed(tr: ScoreTracker, t0: float, st: ScoreState, n: int, step: float = 0.5):
    out = []
    for i in range(n):
        out += tr.update(t0 + i * step, st)
    return out


def first_innings_tracker() -> ScoreTracker:
    tr = ScoreTracker(6, 3)
    feed(tr, 0, ScoreState(2, 0, 0, 2), 3)
    feed(tr, 300, ScoreState(179, 6, 19, 5), 3)
    return tr


# 1 ---------------------------------------------------------------------------------------------------------
def test_chase_joining_at_31_0_starts_innings_2_then_scores():
    tr = first_innings_tracker()
    evs = feed(tr, 432, ScoreState(31, 0, 2, 4), 7)          # 7 reads over 3 s
    starts = [e for e in evs if e.type == EventType.INNINGS_START]
    assert len(starts) == 1 and starts[0].detail["innings"] == 2
    assert starts[0].detail["joined"] == "31/0 (2.4)"
    evs = feed(tr, 440, ScoreState(35, 0, 2, 5), 3)
    four = [e for e in evs if e.type == EventType.FOUR]
    assert four and four[0].detail["innings"] == 2


# 2 ---------------------------------------------------------------------------------------------------------
def test_misreads_do_not_start_a_new_innings():
    tr = first_innings_tracker()
    assert feed(tr, 400, ScoreState(77, 6, 19, 5), 8) == []     # 179/6 -> 77/6 same overs: overs did not drop
    assert feed(tr, 410, ScoreState(17, 6, 19, 5), 8) == []     # 179 -> 17
    assert feed(tr, 420, ScoreState(31, 0, 2, 4), 2) == []      # a 2-read blip
    feed(tr, 422, ScoreState(179, 6, 19, 5), 1)
    assert feed(tr, 425, ScoreState(31, 0, 2, 4), 5) == []      # 5 reads < 6 required
    assert tr.innings == 1


def test_overs_with_dropped_dot_are_read():
    assert parse_scoreline("31-024") == ScoreState(31, 0, 2, 4)
    assert parse_scoreline("73-175") == ScoreState(73, 1, 7, 5)
    assert parse_scoreline("161-418.1") == ScoreState(161, 4, 18, 1)
    assert parse_scoreline("6-10.50") == ScoreState(6, 1, 0, 5)      # trailing glyph on a sharp 1080p crop
    assert parse_scoreline("6-00.4") == ScoreState(6, 0, 0, 4)


# 3 / 4 -----------------------------------------------------------------------------------------------------
def _pipeline(tmp_path, scope_match: bool):
    s = SimpleNamespace(work_dir=tmp_path, video_encoder="libx264", powerplay_max_minutes=45,
                        output_dir=tmp_path, highlight_target_seconds=300, match_scope=scope_match)
    p = HighlightPipeline(s, buffer=None, llm=None, match={})
    finished = []
    p._start_finish = lambda t, why: finished.append((t, why))
    p.cutter.cut = lambda ev: asyncio.sleep(0, result=None)
    return p, finished


def test_match_scope_keeps_clips_after_powerplay_and_finishes_at_end_of_input(tmp_path):
    async def go():
        p, finished = _pipeline(tmp_path, True)
        p.on_event(FusedEvent(EventType.INNINGS_START, 10, 0.9, {Source.OCR}))
        p.on_event(FusedEvent(EventType.POWERPLAY_END, 131, 0.95, {Source.OCR}))
        p.tick(2000)
        p.on_event(FusedEvent(EventType.WICKET, 331, 0.95, {Source.OCR}))   # pp_end + 200 s
        assert finished == [] and len(p.clip_tasks) == 2            # innings start + the late wicket
        p.end_of_input(673)
        assert finished == [(673, "end of input file")]
    asyncio.run(go())


def test_powerplay_scope_unchanged(tmp_path):
    async def go():
        p, finished = _pipeline(tmp_path, False)
        p.on_event(FusedEvent(EventType.INNINGS_START, 10, 0.9, {Source.OCR}))
        p.on_event(FusedEvent(EventType.POWERPLAY_END, 131, 0.95, {Source.OCR}))
        p.on_event(FusedEvent(EventType.WICKET, 331, 0.95, {Source.OCR}))
        assert len(p.clip_tasks) == 1                                 # innings start only; wicket dropped
        p.tick(181)
        assert finished and finished[0][0] == 131
    asyncio.run(go())


# 5 ---------------------------------------------------------------------------------------------------------
def test_same_score_wicket_in_innings_2_is_kept():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(Event(EventType.WICKET, 101, Source.OCR, 0.95, {"score": "6/1", "overs": "0.5", "innings": 1}))
    f.add(Event(EventType.WICKET, 450, Source.OCR, 0.95, {"score": "6/1", "overs": "1.2", "innings": 2}))
    out = [e for e in f.finalize(10_000) if e.type == EventType.WICKET]
    assert len(out) == 2


# 6 ---------------------------------------------------------------------------------------------------------
def _clip(etype, t, dur, innings, score, conf=0.95, wickets=1, src=Source.OCR):
    ev = FusedEvent(etype, t, conf, {src}, {"score": score, "overs": "1.0", "innings": innings,
                                            "wickets": wickets})
    return Clip(ev, [(t - dur / 2, t + dur / 2)], Path(f"/tmp/{t}.mp4"))


def _facts(clips, target=180, innings2_t=432, result=None):
    evs = [c.event for c in clips]
    evs.append(FusedEvent(EventType.INNINGS_START, innings2_t, 0.8, {Source.OCR},
                          {"innings": 2, "score": "31/0", "prev_innings_last": "179/6 (19.5)"}))
    if target:
        evs.append(FusedEvent(EventType.TARGET, innings2_t + 5, 0.95, {Source.OCR}, {"target": target, "innings": 2}))
    if result:
        evs.append(FusedEvent(EventType.MATCH_END, 630, 0.95, {Source.OCR}, {"score": result, "innings": 2}))
    return MatchFacts.from_events(evs)


def test_all_known_wickets_kept_at_180s_and_both_innings_get_clips():
    wkts = [_clip(EventType.WICKET, t, 20, 1, f"{r}/{w}") for t, r, w in
            [(101, 6, 1), (181, 94, 2), (234, 116, 3), (287, 157, 4), (340, 177, 5), (362, 179, 6)]]
    fours = [_clip(EventType.SIX, 200, 12, 1, "100/2"), _clip(EventType.SIX, 500, 12, 2, "90/1"),
             _clip(EventType.FOUR, 560, 12, 2, "120/2"), _clip(EventType.FOUR, 600, 12, 2, "150/3")]
    win = _clip(EventType.MATCH_END, 630, 12, 2, "181/5")
    clips = wkts + fours + [win]
    mp = match_plan(clips, 180, _facts(clips, result="181/5"), "t")
    chosen = {c.event.id for c, _ in mp.plan.items}
    assert all(w.event.id in chosen for w in wkts)
    assert win.event.id in chosen                                        # winning runs forced
    assert any(mp.innings[i] == 2 for i in chosen)
    budget = [w for w in mp.withheld if w["reason"] == "budget"]
    assert budget, "clips dropped for length must be listed in withheld[] with reason budget"


# 7 ---------------------------------------------------------------------------------------------------------
def test_holds_multi_wicket_gap_and_innings_inferred():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(Event(EventType.WICKET, 287, Source.OCR, 0.75, {"score": "157/4", "wickets": 2, "gap": True, "innings": 1}))
    (w,) = [e for e in f.finalize(10_000) if e.type == EventType.WICKET]
    assert "multi_wicket_gap" in w.hold_reasons

    c2 = _clip(EventType.SIX, 500, 12, 2, "90/1")
    mp = match_plan([c2], 300, _facts([c2], target=None), "t")         # no TARGET read
    assert "innings_inferred" in c2.event.hold_reasons and mp.plan.status == "hold_for_review"
    c3 = _clip(EventType.SIX, 500, 12, 2, "90/1")
    match_plan([c3], 300, _facts([c3], target=180), "t")               # 180 == 179 + 1 -> confirmed
    assert "innings_inferred" not in c3.event.hold_reasons


def test_missing_wicket_holds_the_package():
    a = _clip(EventType.WICKET, 101, 20, 1, "6/1")
    b = _clip(EventType.WICKET, 110, 20, 1, "6/2")                      # overlaps a -> cannot both be clipped
    mp = match_plan([a, b], 300, _facts([a, b]), "t")
    assert mp.package_holds == ["wicket_missing"]
    assert mp.wicket_audit["innings_1"]["wickets_clipped"] < mp.wicket_audit["innings_1"]["wickets_on_board"]


def test_gap_wicket_without_call_anchors_on_crowd_peak():
    ev = FusedEvent(EventType.WICKET, 228, 0.75, {Source.OCR},
                    {"score": "116/3", "gap": True, "confirm_t": 234.4, "call_t": None})
    loud = [(214.4 + i * 0.1, 0.1) for i in range(180)]
    loud[100] = (224.4, 0.9)                                            # crowd roar at 224.4
    ranges, cap_t = plan_ranges(ev, [216.0, 219.0, 230.0], loud)
    assert ranges[0][0] == 216.0 and cap_t == 234.4
    assert "gap_unanchored" not in ev.hold_reasons
    ev2 = FusedEvent(EventType.WICKET, 228, 0.75, {Source.OCR},
                     {"score": "116/3", "gap": True, "confirm_t": 234.4, "call_t": None})
    ranges2, _ = plan_ranges(ev2, [], [])
    assert ranges2 == [(218.4, 238.4)] and "gap_unanchored" in ev2.hold_reasons


# 8 ---------------------------------------------------------------------------------------------------------
def test_cards_show_numbers_only_from_confirmed_reads():
    match = {"teams": {"Triumph Knights": [], "Eagle Thane Strikers": []}, "batting_first": "Triumph Knights"}
    no_target = card_texts(_facts([], target=None), match)
    assert no_target["break"] == [("INNINGS BREAK", 96)]                # overs < 20 and no target: no score
    assert no_target["result"] is None
    assert not any("TARGET" in t for t, _ in no_target["innings2"])
    full = card_texts(_facts([], target=180, result="181/5"), match)
    assert any("179" in t for t, _ in full["break"])
    assert any("TARGET 180" in t for t, _ in full["innings2"])
    assert full["result"][1][0] == "BY 5 WICKETS"
    wrong = card_texts(_facts([], target=200), match)                    # target != innings-1 runs + 1
    assert not any("179" in t for t, _ in wrong["break"])


def test_one_target_read_is_not_confirmed():
    tr = first_innings_tracker()
    feed(tr, 432, ScoreState(31, 0, 2, 4), 7)
    assert tr.update_target(440, "TARGET-180") == []
    assert tr.update_target(440.5, "TARGET-180") == []                 # < 1 s apart
    (t,) = tr.update_target(441.2, "TArget-180")
    assert t.type == EventType.TARGET and t.detail["target"] == 180


# 9 ---------------------------------------------------------------------------------------------------------
def test_video_opts_and_no_upscale():
    assert video_opts("libx264") == [
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-level", "4.1",
        "-pix_fmt", "yuv420p", "-g", "50", "-bf", "3",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    assert scale_filter(1920, 1080) is None
    assert scale_filter(1280, 720) is None                                # never upscale
    assert scale_filter(3840, 2160) == "scale=1920:1080:flags=lanczos"


# Caption ownership (IPL head condition 8) — real sequences from the 1080p demo ------------------------------
def _w(t, score, prev, gap=False, innings=1):
    return Event(EventType.WICKET, t, Source.OCR, 0.95, {"score": score, "overs": "x", "prev_change_t": prev,
                                                         "gap": gap, "innings": innings, "wickets": 1})


def _call(t, quote, etype=EventType.WICKET):
    return Event(etype, t, Source.ASR, 0.4, {"quote": quote})


def test_next_balls_direct_hit_is_not_absorbed_into_a_caught_wicket():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    for e in [_call(151.8, "Thane Strikers, this is gone."),
              _call(157.5, "Straight to hand and that's the wicket they were looking for."),
              _w(181.1, "94/2", 151.84),
              Event(EventType.FOUR, 186.0, Source.OCR, 0.6, {"score": "99/2", "prev_change_t": 181.1}),
              _call(191.2, "That's a quick single, that's a direct hit as well."),
              _call(195.0, "We have a review for a runout", EventType.REVIEW),
              _w(234.4, "116/3", 186.02, gap=True)]:
        f.add(e)
    by_score = {e.detail.get("score"): e for e in f.finalize(10_000) if e.type == EventType.WICKET}
    caught, runout = by_score["94/2"], by_score["116/3"]
    assert caught.detail["call_t"] == 157.5 and "run_out" not in caught.hold_reasons
    assert runout.detail["call_t"] == 191.2 and "run_out" in runout.hold_reasons


def test_run_out_call_after_a_confirm_belongs_to_the_next_wicket():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    for e in [_w(339.4, "177/5", 312.0, gap=True), _call(343.2, "Makran will be run out."),
              _w(362.1, "179/6", 339.42)]:
        f.add(e)
    by_score = {e.detail.get("score"): e for e in f.finalize(10_000) if e.type == EventType.WICKET}
    assert "run_out" not in by_score["177/5"].hold_reasons
    assert "run_out" in by_score["179/6"].hold_reasons and by_score["179/6"].detail["call_t"] == 343.2


def test_winning_boundary_folds_into_one_winning_runs_clip():
    me = _clip(EventType.MATCH_END, 602.8, 10, 2, "181/5")
    me.caption = "WINNING RUNS · 181/5 (19.2 ov)"
    four = _clip(EventType.FOUR, 608.8, 7, 2, "181/5")
    mp = match_plan([me, four], 300, _facts([me, four], result="181/5"), "t")
    assert [c.event.id for c, _ in mp.plan.items] == [me.event.id]
    assert mp.plan.items[0][1] == "WINNING RUNS · FOUR · 181/5 (19.2 ov)"
    assert any(w["reason"] == "same_ball_as_winning_runs" for w in mp.withheld)


def test_later_commentary_wicket_call_is_not_swallowed_by_dedupe():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    for e in [_call(343.2, "Makran will be run out."), _w(362.1, "179/6", 339.42),
              _call(394.2, "And one more easy run out.")]:
        f.add(e)
    wk = [e for e in f.finalize(10_000) if e.type == EventType.WICKET]
    assert any(Source.OCR not in e.sources and e.t == 394.2 for e in wk)   # reaches the wicket audit


def test_card_crowd_bed_windows():
    sec = {"title": 3.0, "innings1": 2.0, "break": 3.0, "innings2": 2.0, "result": 3.0, "outro": 2.0}
    layout = ["title", "innings1", (40.0, 52.0), (100.0, 110.0), "break", "innings2", (300.0, 312.0),
              "result", "outro"]
    w = crowd_windows(layout, sec, limits=(0.0, 316.0))
    # opening cards: one continuous stretch just before the first clip (title 35-38, innings1 38-40)
    assert w["title"] == 35.0 and w["innings1"] == 38.0
    # later cards: right after the previous clip (the reaction), back-to-back cards continue the stretch
    assert w["break"] == 110.0 and w["innings2"] == 113.0
    # end of the buffer: shifted back so the whole 5 s run stays inside the recorded stream
    assert w["result"] == 311.0 and w["outro"] == 314.0
    # first clip at the very start of the stream: clamped to the stream start
    assert crowd_windows(["title", (1.0, 9.0)], sec, limits=(0.0, 50.0)) == {"title": 0.0}
    # no clip on either side -> no bed (silent card)
    assert crowd_windows(["title", "outro"], sec) == {}
