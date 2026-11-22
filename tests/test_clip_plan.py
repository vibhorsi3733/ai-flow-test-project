"""Clip planning rules from the panel spec, using the cuts measured on the demo feed."""
from motion_ai.events import EventType, FusedEvent, Source
from motion_ai.highlights.clips import Clip, caption_for, end_by_roar, plan_ranges

DEMO_CUTS = [51.48, 56.88, 61.56, 63.24, 64.88, 67.28, 73.12, 76.32, 79.6, 108.2]


def flat(t0, t1, level=0.1):
    return [(round(t0 + i * 0.1, 2), level) for i in range(int((t1 - t0) / 0.1))]


def runout_event():
    ev = FusedEvent(EventType.WICKET, 74, 0.95, {Source.OCR, Source.ASR},
                    {"call_t": 74, "confirm_t": 101, "score": "6/1", "overs": "0.5",
                     "quotes": ["direct hit", "review for run out"]})
    for h in ("run_out", "review", "slow_confirm"):
        ev.hold(h)
    return ev


def test_runout_ranges_match_spec():
    ev = runout_event()
    ranges, cap_t = plan_ranges(ev, DEMO_CUTS, flat(101, 110))
    # earliest real cut, call+3 (keeps the hit), trims: end down to R+3=104, then start +0.12
    assert ranges == [(65.0, 77.0), (91.0, 104.0)]
    assert abs(sum(b - a for a, b in ranges) - 25.0) < 0.05
    assert cap_t == 101 and "over_cap" not in ev.hold_reasons


def test_caption_direct_hit_and_offset_at_confirmation():
    ev = runout_event()
    ranges, cap_t = plan_ranges(ev, DEMO_CUTS, flat(101, 110))
    clip = Clip(ev, ranges, None, caption_for(ev), cap_t)
    assert clip.caption == "RUN OUT · DIRECT HIT · 6/1 (0.5 ov)"
    assert abs(clip.offset_of(101) - (12.0 + 10.0)) < 0.01    # appears at the OUT, not before


def test_no_cut_falls_back_and_holds():
    ev = runout_event()
    ranges, _ = plan_ranges(ev, [], flat(101, 110))
    assert 64.0 <= ranges[0][0] <= 66.0 and "no_cut_snap" in ev.hold_reasons   # call-10, then cap trim


def test_boundary_starts_on_earliest_cut_and_respects_cap():
    ev = FusedEvent(EventType.FOUR, 56.8, 0.85, {Source.OCR, Source.ASR}, {"score": "6/0", "overs": "0.3"})
    ranges, cap_t = plan_ranges(ev, DEMO_CUTS, flat(56.8, 66))
    (a, b), = ranges
    assert a == 51.48 and b - a <= 15 and cap_t == 59.3   # call 56.8 + 2.5: ball reaches the rope 58.5-59.5
    assert caption_for(ev) == "FOUR · 6/0 (0.3 ov)"


def test_end_needs_one_full_second_of_quiet():
    R = 100.0
    loud = [(round(R + i * 0.1, 2), 1.0) for i in range(90)]
    # single 0.5s dip at 104.0 (a word gap) must NOT end the clip
    loud = [(t, 0.1 if 104.0 <= t < 104.5 else v) for t, v in loud]
    assert end_by_roar(loud, R, 7, []) == 107.0
    # a real 1s fade from 105.0 does
    loud2 = [(t, 0.1 if t >= 105.0 else v) for t, v in loud]
    assert end_by_roar(loud2, R, 7, []) == 105.0


def test_end_snaps_before_late_cut():
    assert end_by_roar(flat(100, 110), 100, 7, [106.5]) == 106.46


def test_snap_skips_transition_flash():
    from motion_ai.highlights.clips import snap_start
    # 63.24 opens a 1.64s flash before the 64.88 cut to the bowler close-up
    assert snap_start(DEMO_CUTS, 62, 71) == 64.88
