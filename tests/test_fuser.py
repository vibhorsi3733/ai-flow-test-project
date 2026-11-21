from motion_ai.events import Event, EventType, Source
from motion_ai.fusion.fuser import Fuser


def ev(t, type_, src, conf=0.9, **detail):
    return Event(type_, t, src, conf, detail)


def test_ocr_on_confirms_commentary_and_uses_commentary_time():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(100, EventType.SIX, Source.ASR, 0.8))
    f.add(ev(108, EventType.SIX, Source.OCR, 0.95, score="52/1"))
    out = f.finalize(200)
    assert len(out) == 1
    assert out[0].t == 100 and not out[0].needs_review
    assert out[0].sources == {Source.ASR, Source.OCR}
    assert out[0].detail["score"] == "52/1"


def test_ocr_on_unconfirmed_commentary_is_flagged():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(100, EventType.SIX, Source.ASR, 0.9))
    (out,) = f.finalize(200)
    assert out.needs_review and "asr_only" in out.hold_reasons


def test_ocr_on_ocr_only_wicket_shifts_back_and_is_held():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(108, EventType.WICKET, Source.OCR, 0.95))
    (out,) = f.finalize(200)
    assert out.t == 102 and out.hold_reasons == ["no_call_anchor"]   # start edge unknown


def test_ocr_off_commentary_primary_boosted_by_crowd():
    f = Fuser(ocr_enabled=False, settle_seconds=20, ocr_lag_seconds=6)
    f.add(ev(100, EventType.FOUR, Source.ASR, 0.5))
    f.add(ev(103, EventType.EXCITEMENT, Source.AUDIO, 0.8, z=4.0))
    (out,) = f.finalize(200)
    assert out.type == EventType.FOUR
    assert out.confidence == 0.65 and out.needs_review   # below the 0.7 review threshold
    assert Source.AUDIO in out.sources


def test_one_wicket_event_despite_replay_calls():
    f = Fuser(ocr_enabled=False, settle_seconds=20, ocr_lag_seconds=6)
    f.add(ev(74, EventType.WICKET, Source.ASR, 0.9))
    f.add(ev(105, EventType.WICKET, Source.ASR, 0.9))   # "first wicket down" after the DRS review
    assert len(f.finalize(300)) == 1


def test_waits_for_settle_and_dedupes():
    f = Fuser(ocr_enabled=False, settle_seconds=20, ocr_lag_seconds=6)
    f.add(ev(100, EventType.WICKET, Source.ASR))
    assert f.finalize(110) == []                      # not settled yet
    f.add(ev(105, EventType.WICKET, Source.ASR))      # same wicket called again
    assert len(f.finalize(130)) == 1


def test_standalone_crowd_peak_becomes_moment():
    f = Fuser(ocr_enabled=False, settle_seconds=20, ocr_lag_seconds=6)
    f.add(ev(300, EventType.EXCITEMENT, Source.AUDIO, 0.9, z=5))
    (out,) = f.finalize(400)
    assert out.type == EventType.EXCITEMENT


# ---- panel spec (guidance_final): wicket anchoring, honest confidence, hold reasons ----

def test_round2_runout_becomes_one_wicket_anchored_on_live_call():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(74, EventType.WICKET, Source.ASR, 0.4, quote="direct hit, that could be the first wicket"))
    f.add(ev(84.4, EventType.REVIEW, Source.ASR, 0.4, quote="I have a review for run out"))
    f.add(ev(101, EventType.WICKET, Source.OCR, 0.95, score="6/1", overs="0.5", prev_change_t=60))
    f.add(ev(112, EventType.WICKET, Source.ASR, 0.4, quote="first wicket down"))   # absorbed (<= ocr+25)
    out = f.finalize(400)
    assert len(out) == 1
    w = out[0]
    assert (w.t, w.detail["call_t"], w.detail["confirm_t"]) == (74, 74, 101)
    assert w.confidence == 0.95                     # OCR confidence only, no +0.1 inflation
    assert {"run_out", "review", "slow_confirm"} <= set(w.hold_reasons)


def test_wicket_lookback_floored_at_previous_scoreboard_change():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(50, EventType.WICKET, Source.ASR, 0.4, quote="caught"))          # previous ball
    f.add(ev(80, EventType.WICKET, Source.OCR, 0.95, score="20/1", prev_change_t=60))
    out = sorted(f.finalize(400), key=lambda e: e.t)
    ocr_w = [e for e in out if Source.OCR in e.sources][0]
    assert "no_call_anchor" in ocr_w.hold_reasons and ocr_w.detail["call_t"] is None


def test_asr_call_waits_for_slow_confirmation():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(74, EventType.WICKET, Source.ASR, 0.4, quote="run out"))
    assert f.finalize(74 + 35) == []                # would have finalized alone before the fix
    f.add(ev(101, EventType.WICKET, Source.OCR, 0.95, score="6/1", prev_change_t=60))
    (w,) = f.finalize(74 + 50)
    assert w.detail["confirm_t"] == 101


def test_boundary_confidence_not_inflated():
    f = Fuser(ocr_enabled=True, settle_seconds=30, ocr_lag_seconds=6)
    f.add(ev(56.8, EventType.FOUR, Source.ASR, 0.4))
    f.add(ev(62, EventType.FOUR, Source.OCR, 0.85, score="6/0", overs="0.3"))
    (b,) = f.finalize(200)
    assert b.confidence == 0.85 and b.hold_reasons == [] and b.detail["confirm_t"] == 62
