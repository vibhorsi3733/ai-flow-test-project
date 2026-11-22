"""Orchestrator.

    SRT ─► ingest ─► segment buffer ─┬─► audio excitement ─┐
                                     ├─► commentary ASR+LLM ┼─► fuser ─► clip cutter ─► (powerplay end) ─► curator ─► renderer
                                     └─► scoreboard OCR ────┘                 (only if OCR_AVAILABLE=true)

Detector backlog: each detector has its own unbounded queue of Segment references (a few hundred bytes each,
so memory is not the risk; lateness is). LagMonitor measures each detector's lag = newest ingested stream time
- end of the segment it just finished, logs a WARNING (rate-limited) once a detector is more than
LAG_WARN_SEGMENTS behind, and exposes current/max lag in the manifest (pipeline.stats -> "detector_lag").
Nothing is dropped: OCR is the source of truth for the score, and commentary is NOT skipped ahead either —
its calls/quotes anchor wickets and captions, and in match scope the package is built at input end, so a late
commentary event is still used; dropping it would silently lose a call. (fusion_loop paces on the slowest
detector, so a lagging detector delays finalisation, which the lag warning makes visible.)

RESUME=true (settings.resume via getattr): continue a crashed live run — recorded segments go back into the
buffer, recorded events into the fuser, ingest continues after the last segment (see recorder.py).
"""
from __future__ import annotations

import asyncio
import logging
import time

from .config import Settings, load_match
from .detectors import Detector, build_detectors
from .fusion.fuser import Fuser
from .highlights.pipeline import HighlightPipeline
from .ingest.buffer import SegmentBuffer
from .ingest.srt_ingest import Segment, SrtIngest
from .llm import LLM
from .recorder import Recorder, load_events, load_segments

log = logging.getLogger("motion_ai")


LAG_WARN_SEGMENTS = 10      # ~20-50 s of video behind live
LAG_WARN_EVERY_S = 30.0     # at most one warning per detector per 30 s


class LagMonitor:
    """How far each detector is behind the newest ingested segment (stream seconds)."""

    def __init__(self, warn_segments: int = LAG_WARN_SEGMENTS, warn_every_s: float = LAG_WARN_EVERY_S):
        self.warn_segments, self.warn_every_s = warn_segments, warn_every_s
        self.latest_end = 0.0
        self.lag: dict[str, float] = {}
        self.max_lag: dict[str, float] = {}
        self.backlog: dict[str, int] = {}
        self._last_warn: dict[str, float] = {}

    def ingested(self, seg: Segment) -> None:
        self.latest_end = max(self.latest_end, seg.end)

    def processed(self, name: str, seg: Segment, backlog: int, now: float | None = None) -> bool:
        """Record that `name` finished `seg` with `backlog` segments still queued. True if a warning was logged."""
        now = time.monotonic() if now is None else now
        lag = max(0.0, self.latest_end - seg.end)
        self.lag[name] = lag
        self.max_lag[name] = max(self.max_lag.get(name, 0.0), lag)
        self.backlog[name] = backlog
        if backlog > self.warn_segments and now - self._last_warn.get(name, float("-inf")) >= self.warn_every_s:
            self._last_warn[name] = now
            log.warning("%s is %d segments (%.0f s of video) behind live", name, backlog, lag)
            return True
        return False

    def snapshot(self) -> dict:
        return {n: {"lag_s": round(self.lag[n], 1), "max_lag_s": round(self.max_lag[n], 1),
                    "backlog_segments": self.backlog[n]} for n in self.lag}


async def run_detector(det: Detector, queue: asyncio.Queue[Segment], fuser: Fuser,
                       recorder: Recorder | None = None, lag: LagMonitor | None = None) -> None:
    while True:
        seg = await queue.get()
        try:
            for ev in await det.process(seg):
                fuser.add(ev)
                if recorder:
                    recorder.event(ev)
        except Exception:  # one bad segment must never stop the match
            log.exception("%s failed on segment %d", det.name, seg.index)
            det.watermark = seg.end
        finally:
            queue.task_done()
        if lag is not None:
            lag.processed(det.name, seg, queue.qsize())


async def fusion_loop(detectors: list[Detector], fuser: Fuser, pipeline: HighlightPipeline) -> None:
    while not pipeline.finished.is_set():
        now = min(d.watermark for d in detectors)   # slowest detector sets the pace
        for ev in fuser.finalize(now):
            pipeline.on_event(ev)
        pipeline.tick(now)
        await asyncio.sleep(1.0)


async def run(settings: Settings) -> None:
    settings.validate_ocr_setup()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    match = load_match(settings.match_config_path)
    llm = LLM(settings)
    detectors = build_detectors(settings, llm, match)
    setup_logging()   # PaddleOCR reconfigures logging on import/init — restore ours
    resume = bool(getattr(settings, "resume", False))
    recorder = Recorder(settings.work_dir, resume=resume)
    fuser = Fuser(settings.ocr_available, settings.fusion_settle_seconds, settings.ocr_lag_seconds)
    buffer = SegmentBuffer()
    pipeline = HighlightPipeline(settings, buffer, llm, match)
    lag = LagMonitor()
    pipeline.stats = lambda: {"ocr_read_failures": sum(getattr(d, "read_failures", 0) for d in detectors),
                              "detector_lag": lag.snapshot(), "ingest_stalls": ingest.stalls,
                              "resumed": resume}
    ingest = SrtIngest(settings)
    if resume:
        resume_run(settings, buffer, fuser, ingest)

    queues: list[asyncio.Queue[Segment]] = [asyncio.Queue() for _ in detectors]
    tasks = [asyncio.create_task(run_detector(d, q, fuser, recorder, lag)) for d, q in zip(detectors, queues)]
    tasks.append(asyncio.create_task(fusion_loop(detectors, fuser, pipeline)))

    async def feed() -> None:
        async for seg in ingest.segments():
            await buffer.add(seg)
            recorder.segment(seg)
            lag.ingested(seg)
            for q in queues:
                q.put_nowait(seg)
        # Only reached for a local file (live SRT reconnects forever).
        await buffer.close()
        log.info("Input ended; waiting for detectors to finish the backlog")
        for q in queues:
            await q.join()
        for ev in fuser.finalize(float("inf")):
            pipeline.on_event(ev)
        pipeline.end_of_input(buffer.latest)

    tasks.append(asyncio.create_task(feed()))
    log.info("Running: %s | OCR %s | match: %s", settings.srt_url,
             "ON" if settings.ocr_available else "OFF", match.get("match"))
    try:
        await pipeline.finished.wait()
    finally:
        for t in tasks:
            t.cancel()
    if pipeline.error is not None:
        raise RuntimeError("Package build failed (see traceback above)") from pipeline.error
    log.info("Done. Output: %s", pipeline.result)


def resume_run(settings: Settings, buffer: SegmentBuffer, fuser: Fuser, ingest: SrtIngest) -> int:
    """RESUME: reload the crashed run's segments (files still on disk) and events; ingest continues after them."""
    segs = load_segments(settings.work_dir)
    buffer.segments = [s for s in segs if s.path.exists()]
    events = load_events(settings.work_dir)
    for e in events:
        fuser.add(e)
    if segs:
        ingest.continue_after(max(segs, key=lambda s: s.index))
    log.info("RESUME: %d segments (%d on disk) and %d events reloaded", len(segs), len(buffer.segments), len(events))
    return len(events)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", force=True)
    for noisy in ("ppocr", "httpx", "httpx2", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    asyncio.run(run(Settings()))


if __name__ == "__main__":
    main()
