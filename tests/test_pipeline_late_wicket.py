import asyncio
from types import SimpleNamespace

from motion_ai.events import EventType, FusedEvent, Source
from motion_ai.highlights.pipeline import HighlightPipeline


def test_waits_50s_after_powerplay_end_for_late_wicket(tmp_path):
    async def go():
        s = SimpleNamespace(work_dir=tmp_path, video_encoder="libx264", powerplay_max_minutes=45,
                            output_dir=tmp_path, highlight_target_seconds=180)
        p = HighlightPipeline(s, buffer=None, llm=None, match={})
        finished = []
        p._start_finish = lambda t, why: finished.append((t, why))
        p.on_event(FusedEvent(EventType.INNINGS_START, 10, 0.9, {Source.OCR}))
        p.on_event(FusedEvent(EventType.POWERPLAY_END, 1000, 0.95, {Source.OCR}))
        p.tick(1030)
        assert finished == []                    # still accepting the ball-5.6 wicket confirmation
        p.tick(1050)
        assert finished and finished[0][0] == 1000
    asyncio.run(go())


def test_ball_events_before_innings_start_are_dropped():
    from motion_ai.highlights.pipeline import before_first_ball
    pre = FusedEvent(EventType.REVIEW, 19.2, 0.4, {Source.ASR})
    assert before_first_ball(pre, 50.6)
    assert not before_first_ball(FusedEvent(EventType.TOSS, 19.2, 0.4, {Source.ASR}), 50.6)
    assert not before_first_ball(FusedEvent(EventType.FOUR, 56.8, 0.85, {Source.OCR}), 50.6)
