"""Orchestrator.

    SRT ─► ingest ─► segment buffer ─┬─► audio excitement ─┐
                                     ├─► commentary ASR+LLM ┼─► fuser ─► clip cutter ─► (powerplay end) ─► curator ─► renderer
                                     └─► scoreboard OCR ────┘                 (only if OCR_AVAILABLE=true)
"""
from __future__ import annotations

import asyncio
import logging

from .config import Settings, load_match
from .detectors import Detector, build_detectors
from .fusion.fuser import Fuser
from .highlights.pipeline import HighlightPipeline
from .ingest.buffer import SegmentBuffer
from .ingest.srt_ingest import Segment, SrtIngest
from .llm import LLM
from .recorder import Recorder

log = logging.getLogger("motion_ai")


async def run_detector(det: Detector, queue: asyncio.Queue[Segment], fuser: Fuser,
                       recorder: Recorder | None = None) -> None:
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
        if queue.qsize() > 10 and seg.index % 50 == 0:
            log.warning("%s is %d segments behind live", det.name, queue.qsize())


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
    recorder = Recorder(settings.work_dir)
    fuser = Fuser(settings.ocr_available, settings.fusion_settle_seconds, settings.ocr_lag_seconds)
    buffer = SegmentBuffer()
    pipeline = HighlightPipeline(settings, buffer, llm, match)
    pipeline.stats = lambda: {"ocr_read_failures": sum(getattr(d, "read_failures", 0) for d in detectors)}
    ingest = SrtIngest(settings)

    queues: list[asyncio.Queue[Segment]] = [asyncio.Queue() for _ in detectors]
    tasks = [asyncio.create_task(run_detector(d, q, fuser, recorder)) for d, q in zip(detectors, queues)]
    tasks.append(asyncio.create_task(fusion_loop(detectors, fuser, pipeline)))

    async def feed() -> None:
        async for seg in ingest.segments():
            await buffer.add(seg)
            recorder.segment(seg)
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
    log.info("Done. Output: %s", pipeline.result)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", force=True)
    for noisy in ("ppocr", "httpx", "httpx2", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    asyncio.run(run(Settings()))


if __name__ == "__main__":
    main()
