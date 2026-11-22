"""Live-feed robustness (code review: structure 6/10, robustness 4.5). Fakes only, no network; ffmpeg lavfi
for tiny synthetic segments."""
import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from motion_ai import media
from motion_ai.events import Event, EventType, FusedEvent, Source
from motion_ai.highlights import pipeline as pl
from motion_ai.ingest.srt_ingest import Segment, SrtIngest
from motion_ai.main import LagMonitor, resume_run
from motion_ai.recorder import Recorder, load_events, load_segments

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def _fresh_media_caches():
    media.reset_caches()
    yield
    media.reset_caches()


# ------------------------------------------------------------------ 1. one failed clip must not lose the package
def _settings(tmp_path, **kw):
    return SimpleNamespace(work_dir=tmp_path, video_encoder="libx264", powerplay_max_minutes=45,
                           output_dir=tmp_path, highlight_target_seconds=180, match_scope=True, **kw)


def test_one_failed_clip_is_counted_and_package_still_built(tmp_path, monkeypatch):
    built = {}

    async def fake_build(clips, facts, settings, match, buffer, extra=None):
        built["clips"], built["extra"] = clips, extra
        return tmp_path / "package.mp4"

    monkeypatch.setattr(pl, "build_match_package", fake_build)

    async def go():
        p = pl.HighlightPipeline(_settings(tmp_path), buffer=None, llm=None, match={})
        p.stats = lambda: {"ocr_read_failures": 0}

        async def cut(ev):
            if ev.t == 200:
                raise RuntimeError("ffmpeg failed (1): loudness")   # e.g. media.loudness() on a broken segment
            return SimpleNamespace(event=ev, path=tmp_path / f"{ev.t}.mp4")

        p.cutter.cut = cut
        for t in (100, 200, 300):
            p.on_event(FusedEvent(EventType.FOUR, t, 0.9, {Source.OCR}))
        p.end_of_input(400)
        assert p._finish_task is not None            # the finish task is kept, not fire-and-forget
        await asyncio.wait_for(p.finished.wait(), 5)
        return p

    p = asyncio.run(go())
    assert p.result == tmp_path / "package.mp4" and p.error is None
    assert [c.event.t for c in built["clips"]] == [100, 300]
    assert built["extra"]["clip_failures"] == 1 and built["extra"]["ocr_read_failures"] == 0


def test_package_build_failure_is_logged_and_kept(tmp_path, monkeypatch, caplog):
    async def boom(*a, **k):
        raise ValueError("render_match blew up")

    monkeypatch.setattr(pl, "build_match_package", boom)

    async def go():
        p = pl.HighlightPipeline(_settings(tmp_path), buffer=None, llm=None, match={})
        p.end_of_input(10)
        await asyncio.wait_for(p.finished.wait(), 5)
        await asyncio.sleep(0)
        return p

    with caplog.at_level(logging.ERROR):
        p = asyncio.run(go())
    assert isinstance(p.error, ValueError) and p.finished.is_set()
    assert "PACKAGE BUILD FAILED" in caplog.text


def test_collect_clips_survives_raised_tasks():
    async def ok():
        return "clip"

    async def bad():
        raise RuntimeError("x")

    async def none():
        return None

    async def go():
        return await pl.collect_clips([asyncio.create_task(c()) for c in (ok, bad, none, ok)])

    assert asyncio.run(go()) == (["clip", "clip"], 1)


# ------------------------------------------------------------------ 2. ingest stall watchdog + detector lag
class FakeProc:
    """ffmpeg that is alive but writes nothing (SRT socket open, no data)."""

    def __init__(self):
        self.returncode = None
        self.pid = 4242
        self.killed = False
        self._done = asyncio.Event()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode


def test_stall_watchdog_kills_ffmpeg_when_no_segment_arrives(tmp_path):
    csv = tmp_path / "r00_list.csv"
    csv.write_text("r00_000000.ts,0.0,2.0\n")
    stalls = []

    async def go():
        proc = FakeProc()
        got = [x async for x in SrtIngest._tail_csv(csv, proc, 0.5, lambda: stalls.append(1))]
        return proc, got

    proc, got = asyncio.run(asyncio.wait_for(go(), 5))
    assert got == [("r00_000000.ts", 0.0, 2.0)]
    assert proc.killed and stalls == [1]          # -> SrtIngest.segments() reconnects with a new run


def test_watchdog_does_not_fire_while_segments_keep_coming(tmp_path):
    csv = tmp_path / "r00_list.csv"
    csv.write_text("")

    async def go():
        proc = FakeProc()

        async def writer():
            for i in range(6):
                await asyncio.sleep(0.15)
                with csv.open("a") as f:
                    f.write(f"r00_{i:06d}.ts,{2 * i}.0,{2 * i + 2}.0\n")
            await asyncio.sleep(0.1)
            proc.returncode = 0          # ffmpeg exits normally
            proc._done.set()

        w = asyncio.create_task(writer())
        got = [x async for x in SrtIngest._tail_csv(csv, proc, 0.5)]
        await w
        return proc, got

    proc, got = asyncio.run(asyncio.wait_for(go(), 5))
    assert len(got) == 6 and not proc.killed


def test_stall_seconds_from_settings(tmp_path):
    ing = SrtIngest(SimpleNamespace(srt_url="srt://x:9000", work_dir=tmp_path, ingest_stall_seconds=7))
    assert ing.stall_s == 7.0
    assert SrtIngest(SimpleNamespace(srt_url="srt://x:9000", work_dir=tmp_path)).stall_s == 20.0


def test_lag_monitor_warns_with_lag_seconds_and_rate_limits(caplog):
    lag = LagMonitor(warn_segments=10, warn_every_s=30)
    for i in range(30):
        lag.ingested(Segment(i, Path(f"{i}.ts"), 2.0 * i, 2.0 * i + 2))
    with caplog.at_level(logging.WARNING):
        assert lag.processed("ocr", Segment(5, Path("5.ts"), 10, 12), backlog=24, now=100.0)
        assert not lag.processed("ocr", Segment(6, Path("6.ts"), 12, 14), backlog=23, now=110.0)   # rate-limited
        assert lag.processed("ocr", Segment(7, Path("7.ts"), 14, 16), backlog=22, now=131.0)
        assert not lag.processed("audio", Segment(29, Path("29.ts"), 58, 60), backlog=0, now=131.0)
    assert "ocr is 24 segments (48 s of video) behind live" in caplog.text
    snap = lag.snapshot()
    assert snap["ocr"] == {"lag_s": 44.0, "max_lag_s": 48.0, "backlog_segments": 22}
    assert snap["audio"]["lag_s"] == 0.0


# ------------------------------------------------------------------ 3. crash / restart keeps recorded data
def _ev(t):
    return Event(EventType.FOUR, t, Source.OCR, 0.9, {"score": "4/0"}, f"e{t}")


def test_fresh_start_archives_previous_run_instead_of_truncating(tmp_path):
    r = Recorder(tmp_path)
    r.event(_ev(1.0))
    r.segment(Segment(0, tmp_path / "segments/r00_000000.ts", 0.0, 2.0, 1.4))
    r2 = Recorder(tmp_path)                       # restart without RESUME
    assert r2.archived is not None
    assert load_events(r2.archived)[0].id == "e1.0"
    assert load_segments(tmp_path) == [] and load_events(tmp_path) == []


def test_resume_appends_and_reloads(tmp_path):
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()
    r = Recorder(tmp_path)
    for i in range(3):
        (seg_dir / f"r00_{i:06d}.ts").write_bytes(b"x")
        r.segment(Segment(i, seg_dir / f"r00_{i:06d}.ts", 2.0 * i, 2.0 * i + 2, 1.4 + 2 * i))
    r.event(_ev(3.0))
    with r.events_path.open("a") as f:
        f.write('{"type": "four", "t": 5')     # crash mid-write: torn last line
    r2 = Recorder(tmp_path, resume=True)
    assert r2.archived is None
    r2.event(_ev(7.0))
    assert [e.id for e in load_events(tmp_path)] == ["e3.0", "e7.0"]
    assert [s.pts for s in load_segments(tmp_path)] == [1.4, 3.4, 5.4]

    settings = SimpleNamespace(srt_url="srt://x:9000", work_dir=tmp_path)
    ingest = SrtIngest(settings)
    added = []
    buffer = SimpleNamespace(segments=[])
    n = resume_run(settings, buffer, SimpleNamespace(add=added.append), ingest)
    assert n == 2 and len(added) == 2 and len(buffer.segments) == 3
    assert ingest._next_index == 3 and ingest._clock_offset == 6.0
    assert ingest.first_free_run() == 1           # new files are r01_*: r00_* is never overwritten


def test_old_segments_jsonl_without_pts_still_loads(tmp_path):
    (tmp_path / "segments.jsonl").write_text(
        '{"index": 0, "path": "runtime_1080/segments/r00_000000.ts", "start": 0.0, "end": 4.68}\n')
    seg = load_segments(tmp_path)[0]
    assert seg.pts is None and seg.end == 4.68
    rec = Recorder(tmp_path, resume=True)
    rec.segment(Segment(1, Path("a.ts"), 4.68, 8.08))
    assert "pts" not in json.loads((tmp_path / "segments.jsonl").read_text().splitlines()[-1])


# ------------------------------------------------------------------ 6. PTS continuity (pure)
def test_pts_continuity_rules():
    # runtime_1080 r00_000000..2: PTS 1.44, 6.08, 9.48; CSV durations 4.68, 3.40 (1-frame mismatch ok)
    assert media.pts_continuity([1.44, 6.08, 9.48], [4.68, 3.40, 4.76])[0]
    ok, why = media.pts_continuity([1.44, 6.08, 1.40], [4.68, 3.40, 4.76])   # ad-break splice: PTS reset
    assert not ok and "segment 2" in why
    assert not media.pts_continuity([1.4, 1.4, 1.4], [2.0, 2.0, 2.0])[0]    # per-segment reset timestamps
    assert not media.pts_continuity([1.4, None], [2.0, 2.0])[0]
    assert media.pts_continuity([3600.0], [None])[0]
    assert not media.pts_continuity([10.0, 95.0], [None, None])[0]          # big jump without durations
    assert media.pts_continuity([10.0, 12.0], [None, None])[0]


# ------------------------------------------------------------------ synthetic media (ffmpeg lavfi)
def _make_segments(out_dir: Path, rate: int, seconds: int = 6, offset: float | None = None) -> list[Segment]:
    """Tiny 25 fps 160x120 + sine source cut like SrtIngest (segment muxer, -c copy, csv list)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "src.ts"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration={seconds}",
                    "-c:v", "libx264", "-g", "25", "-preset", "ultrafast", "-c:a", "aac", "-ac", "2",
                    *(["-output_ts_offset", str(offset)] if offset is not None else []),
                    "-f", "mpegts", str(src)], check=True)
    csv = out_dir / "list.csv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0",
                    "-map", "0:a:0", "-c", "copy", "-f", "segment", "-segment_time", "2", "-segment_format",
                    "mpegts", "-segment_list", str(csv), "-segment_list_type", "csv",
                    str(out_dir / "r00_%06d.ts")], check=True)
    segs = []
    for i, ln in enumerate(csv.read_text().split()):
        name, a, b = ln.split(",")
        p = out_dir / name
        pts = media.probe(p).get("video_start")
        segs.append(Segment(i, p, float(a), float(b), pts))
        media.remember_segment(p, pts, float(b) - float(a))
    return segs


@pytest.fixture(scope="module")
def seg48(tmp_path_factory):
    return _make_segments(tmp_path_factory.mktemp("seg48"), 48000)


def _no_blocking_subprocess(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("blocking subprocess.run inside async render path")

    monkeypatch.setattr(subprocess, "run", forbidden)


# ------------------------------------------------------------------ 4 + 5: async probes, source audio rate
@needs_ffmpeg
def test_render_clip_keeps_48k_source_rate_and_never_blocks(seg48, tmp_path, monkeypatch, caplog):
    for s in seg48:
        media.remember_segment(s.path, s.pts, s.duration)
    _no_blocking_subprocess(monkeypatch)
    out = tmp_path / "clip.mp4"
    with caplog.at_level(logging.WARNING):
        asyncio.run(media.render_clip([s.path for s in seg48], seg48[0].start, [(0.5, 4.5)], out, "FOUR", 1.0))
    assert "demuxer fallback" not in caplog.text            # continuous PTS -> frame-exact byte join
    monkeypatch.undo()
    info = media.probe(out)
    assert info["audio_hz"] == 48000 and abs(info["duration"] - 4.0) < 0.1


@needs_ffmpeg
def test_render_card_uses_package_rate_for_silence_and_crowd(seg48, tmp_path, monkeypatch):
    for s in seg48:
        media.remember_segment(s.path, s.pts, s.duration)
    _no_blocking_subprocess(monkeypatch)
    silent, crowd = tmp_path / "silent.mp4", tmp_path / "crowd.mp4"

    async def go():
        await media.render_card([("TITLE", 20)], 1.0, silent, (160, 120), 25, audio_rate=48000)
        await media.render_card([("BREAK", 20)], 1.5, crowd, (160, 120), 25,
                                crowd=([s.path for s in seg48], seg48[0].start, 1.0))

    asyncio.run(go())
    monkeypatch.undo()
    assert media.probe(silent)["audio_hz"] == 48000
    assert media.probe(crowd)["audio_hz"] == 48000           # pinned from the first segment probed


def test_unknown_audio_rate_defaults_to_48k():
    assert asyncio.run(media.source_audio_rate(None)) == 48000
    assert media.audio_opts()[media.audio_opts().index("-ar") + 1] == "48000"


@needs_ffmpeg
def test_probe_async_is_cached_per_path(seg48, monkeypatch):
    calls = []
    real = asyncio.create_subprocess_exec

    async def counting(*a, **k):
        calls.append(a)
        return await real(*a, **k)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", counting)

    async def go():
        a = await media.probe_async(seg48[0].path)
        b = await media.probe_async(seg48[0].path)
        c = await media.probe_async(seg48[0].path, cache=False)
        return a, b, c

    a, b, c = asyncio.run(go())
    assert a == b == c and len(calls) == 2 and a["width"] == 160


# ------------------------------------------------------------------ 6. splice -> concat demuxer for that clip
@needs_ffmpeg
def test_pts_splice_falls_back_to_demuxer(seg48, tmp_path_factory, caplog):
    other = _make_segments(tmp_path_factory.mktemp("spliced"), 48000, seconds=4, offset=500.0)
    for s in seg48 + other:
        media.remember_segment(s.path, s.pts, s.duration)
    spliced = [seg48[0].path, other[0].path]                 # ad break: PTS jumps by ~500 s
    with caplog.at_level(logging.WARNING):
        args, t_in, lst = asyncio.run(media.continuous_input(spliced, tmp_path_factory.mktemp("t"), "x"))
    assert args[:2] == ["-f", "concat"] and t_in == 0.0 and lst is not None
    assert "PTS discontinuity" in caplog.text
    args, t_in, lst = asyncio.run(media.continuous_input([s.path for s in seg48], Path("/tmp"), "y"))
    assert args[0] == "-copyts" and t_in == seg48[0].pts and lst is None


@needs_ffmpeg
def test_ingest_records_first_video_pts(seg48, tmp_path):
    from motion_ai.ingest.srt_ingest import first_video_pts
    assert asyncio.run(first_video_pts(seg48[1].path)) == pytest.approx(seg48[1].pts)
    assert asyncio.run(first_video_pts(tmp_path / "missing.ts")) is None


def test_sliver_segment_pts_inferred_from_neighbours():
    """Segment 57 of the demo is 0.16 s with no SPS/PPS: its PTS is inferred, the join stays frame-exact."""
    from motion_ai.media import fill_unknown_pts, pts_continuity
    pts = [283.44, None, 283.6, 290.56]           # real values around segments 56-59 (+1.4 s TS offset)
    durs = [3.0, 0.16, 6.96, 6.4]
    filled = fill_unknown_pts([280.44] + pts[1:], durs)
    assert filled[1] == 283.44 and pts_continuity(filled, durs)[0]
    assert fill_unknown_pts([None, 10.0], [2.0, 2.0]) == [8.0, 10.0]
    assert fill_unknown_pts([None, None], [None, 2.0]) == [None, None]   # never invent without a neighbour


def test_commentary_does_not_report_the_same_line_twice():
    """Windows overlap by ~5 s of context; an event on an already-reported line must not come back."""
    from types import SimpleNamespace
    from motion_ai.detectors.commentary import CommentaryDetector
    from motion_ai.events import Event, EventType, Source
    d = CommentaryDetector(SimpleNamespace(), llm=None, match={})
    e1 = Event(EventType.WICKET, 105.12, Source.ASR, 0.9, {"quote": "first wicket down"})
    e2 = Event(EventType.WICKET, 105.12, Source.ASR, 0.9, {"quote": "first wicket down"})
    e3 = Event(EventType.FOUR, 105.12, Source.ASR, 0.9, {"quote": "and four"})
    assert d._new_only([e1]) == [e1]
    assert d._new_only([e2, e3]) == [e3]
