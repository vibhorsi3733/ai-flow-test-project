"""Scoreboard OCR detector. Imported ONLY when OCR_AVAILABLE=true.

Crops the score boxes from config/scoreboard.json, reads them with PaddleOCR
(recognition only — the boxes are already known), and feeds ScoreTracker.
"""
from __future__ import annotations

import asyncio
import json
import logging

from ..config import Settings
from ..events import Event
from ..ingest.srt_ingest import Segment
from ..media import read_frames
from .base import Detector
from .score_tracker import ScoreTracker, parse_reading, parse_scoreline

log = logging.getLogger(__name__)

FOOTER_READS_PER_SEGMENT = 3


def _spread(items: list, n: int) -> list:
    """Up to n items evenly spread (first, middle, last) — keeps reads >= 1 s apart on 2 fps frames."""
    if len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else items[:1]


class ScoreboardOCRDetector(Detector):
    name = "ocr"

    def __init__(self, settings: Settings):
        super().__init__()
        cfg = json.loads(settings.ocr_config_path.read_text())
        self.width, self.height = cfg["resolution"]
        self.regions = cfg["regions"]
        self.fps = settings.ocr_fps
        self.tracker = ScoreTracker(settings.powerplay_overs, cfg.get("consensus_frames", 3))
        self.read_failures = 0

        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=settings.ocr_use_gpu, show_log=False)
        log.info("Scoreboard OCR ready (%s regions)", list(self.regions))

    def _read_region(self, frame, name: str) -> str:
        import cv2

        r = self.regions.get(name)
        if not r:
            return ""
        crop = frame[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
        scale = r.get("scale", 3)   # wide boxes (footer) need less upscaling
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0).apply(gray)
        result = self.ocr.ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), det=False, cls=False)
        # rec-only output: [[(text, score), ...]]
        try:
            text, score = result[0][0]
            return text if score > 0.6 else ""
        except (IndexError, TypeError, ValueError):
            return ""

    def _read_frame(self, frame):
        if "scoreline" in self.regions:
            # Score and overs in one box (their positions shift with score width): "157-4 18.1"
            st = parse_scoreline(self._read_region(frame, "scoreline"))
            if st is None and "scoreline_alt" in self.regions:
                # compact layout (result / announcement banner in the centre): score moves to the left box
                st = parse_scoreline(self._read_region(frame, "scoreline_alt"))
            return st
        return parse_reading(self._read_region(frame, "score"), self._read_region(frame, "overs"))

    async def process(self, seg: Segment) -> list[Event]:
        frames = None
        for attempt in (1, 2):   # retry once; some keyframe-cut segments fail to decode the first time
            try:
                frames = await read_frames(seg.path, self.fps, self.width, self.height)
                break
            except RuntimeError as exc:
                if attempt == 2:
                    self.read_failures += 1
                    log.warning("OCR frame read failed twice on segment %d (%.1f-%.1fs): %s",
                                seg.index, seg.start, seg.end, str(exc)[-200:])
        if frames is None:
            self.watermark = seg.end
            return []
        readings = await asyncio.to_thread(lambda: [(off, self._read_frame(f)) for off, f in frames])
        events: list[Event] = []
        for off, reading in readings:
            events.extend(self.tracker.update(seg.start + off, reading))
        if "target" in self.regions and self.tracker.innings >= 2 and self.tracker.target is None:
            # The chase shows "TARGET - 180" — read it on two frames per segment until confirmed
            picks = [frames[0], frames[-1]] if len(frames) > 1 else frames
            texts = await asyncio.to_thread(lambda: [(off, self._read_region(f, "target")) for off, f in picks])
            for off, text in texts:
                events.extend(self.tracker.update_target(seg.start + off, text))
        if "footer" in self.regions and self.tracker.wants_innings_total():
            # Innings total from the scorecard footer ("OVERS 20 TOTAL 179-7"): the last ball of an innings
            # often never reaches the bar. Only near an innings' end, only on frames where the bar is
            # unreadable (cards cover it), at most FOOTER_READS_PER_SEGMENT frames per segment.
            blank = [frames[i] for i, (_, r) in enumerate(readings) if r is None]
            picks = _spread(blank, FOOTER_READS_PER_SEGMENT)
            texts = await asyncio.to_thread(lambda: [(off, self._read_region(f, "footer")) for off, f in picks])
            for off, text in texts:
                events.extend(self.tracker.update_innings_total(seg.start + off, text))
        if events:
            log.info("OCR events: %s", [(e.type.value, e.detail.get("score") or e.detail.get("target")) for e in events])
        self.watermark = seg.end
        return events
