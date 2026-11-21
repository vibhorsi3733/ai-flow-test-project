"""Crowd/commentary excitement from loudness, relative to a rolling baseline.

Robust z-score (median/MAD) over the last ~10 minutes, so it adapts to each stadium
and to the broadcaster's audio levels.
"""
from __future__ import annotations

from collections import deque

from ..config import Settings
from ..events import Event, EventType, Source
from ..ingest.srt_ingest import Segment
from ..media import read_pcm
from .base import Detector

WINDOW_S = 0.5
SAMPLE_RATE = 16000
COOLDOWN_S = 10.0


class AudioExcitementDetector(Detector):
    name = "audio"

    def __init__(self, settings: Settings):
        super().__init__()
        self.z_threshold = settings.excitement_z
        self.history: deque[float] = deque(maxlen=int(600 / WINDOW_S))
        self._last_peak_t = -1e9

    async def process(self, seg: Segment) -> list[Event]:
        import numpy as np

        pcm = await read_pcm(seg.path, SAMPLE_RATE)
        events: list[Event] = []
        win = int(WINDOW_S * SAMPLE_RATE)
        for i in range(len(pcm) // win):
            chunk = pcm[i * win:(i + 1) * win]
            db = 20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
            t = seg.start + i * WINDOW_S
            if len(self.history) >= 120:  # need ~1 min of baseline first
                hist = np.fromiter(self.history, float)
                med = np.median(hist)
                mad = np.median(np.abs(hist - med)) * 1.4826 + 1e-6
                z = (db - med) / mad
                if z >= self.z_threshold and t - self._last_peak_t > COOLDOWN_S:
                    self._last_peak_t = t
                    events.append(Event(EventType.EXCITEMENT, t, Source.AUDIO,
                                        confidence=min(1.0, z / (2 * self.z_threshold)),
                                        detail={"z": round(float(z), 2)}))
            self.history.append(float(db))
        self.watermark = seg.end
        return events
