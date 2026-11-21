from __future__ import annotations

from abc import ABC, abstractmethod

from ..events import Event
from ..ingest.srt_ingest import Segment


class Detector(ABC):
    """One independent signal source. Each runs in its own task with its own queue,
    so a slow detector (ASR) never blocks a fast one (audio)."""

    name: str = "detector"

    def __init__(self) -> None:
        # Stream time up to which this detector has finished processing.
        # The fuser only finalizes events older than the slowest detector's watermark.
        self.watermark: float = 0.0

    @abstractmethod
    async def process(self, seg: Segment) -> list[Event]:
        ...
