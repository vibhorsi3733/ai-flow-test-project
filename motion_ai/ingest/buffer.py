"""Rolling buffer of recorded segments, so clips can be cut after the fact."""
from __future__ import annotations

import asyncio
import logging

from .srt_ingest import Segment

log = logging.getLogger(__name__)


class SegmentBuffer:
    def __init__(self, keep_seconds: float = 3 * 3600):
        # Toss → powerplay end is ~1h; keep 3h so nothing is deleted before the package is built.
        self.keep_seconds = keep_seconds
        self.segments: list[Segment] = []
        self.closed = False  # input ended (file finished) — no more video will arrive
        self._cond = asyncio.Condition()

    @property
    def latest(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    async def add(self, seg: Segment) -> None:
        async with self._cond:
            self.segments.append(seg)
            self._prune()
            self._cond.notify_all()

    async def close(self) -> None:
        async with self._cond:
            self.closed = True
            self._cond.notify_all()

    async def wait_until(self, t: float, timeout: float = 120.0) -> bool:
        """Wait until the buffer contains video up to stream time t (or the input has ended)."""
        try:
            async with self._cond:
                await asyncio.wait_for(self._cond.wait_for(lambda: self.latest >= t or self.closed), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def covering(self, t0: float, t1: float) -> list[Segment]:
        return [s for s in self.segments if s.end > t0 and s.start < t1]

    def _prune(self) -> None:
        cutoff = self.latest - self.keep_seconds
        while self.segments and self.segments[0].end < cutoff:
            old = self.segments.pop(0)
            old.path.unlink(missing_ok=True)
