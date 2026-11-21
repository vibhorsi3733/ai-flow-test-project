"""SRT → ~2s MPEG-TS segments on disk, yielded with exact stream times.

FFmpeg's segment muxer writes a CSV list (filename,start,end) as each segment closes;
we tail that list. If the SRT connection drops, FFmpeg is restarted and the stream
clock continues from where it stopped.

Stall watchdog: an SRT socket can stay open while no data flows (encoder frozen, network black hole) and
FFmpeg then neither exits nor writes segments. If no new segment is listed for STALL_SECONDS (setting
`ingest_stall_seconds`, read via getattr; default 20 s = ~10 segments) while FFmpeg is alive, it is killed and
the normal reconnect loop starts a new run.

Each segment's first video PTS is probed when it closes and kept on the Segment (persisted in segments.jsonl)
so media.continuous_input can tell continuous segments from an ad-break splice.

Run numbers: each FFmpeg run writes rNN_*.ts. A new process starts at the first run number not on disk, so a
restart never overwrites the segments of an earlier run (recorder.py, RESUME).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from .. import media
from ..config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    index: int
    path: Path
    start: float  # stream seconds
    end: float
    pts: float | None = None   # first video PTS in the file (None: unknown / old segments.jsonl)

    @property
    def duration(self) -> float:
        return self.end - self.start


STALL_SECONDS = 20.0


class SrtIngest:
    SEGMENT_SECONDS = 2

    def __init__(self, settings: Settings):
        self.url = settings.srt_url
        self.seg_dir = settings.work_dir / "segments"
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self._next_index = 0
        self._clock_offset = 0.0
        # A local file (e.g. a recorded match) is played at real-time speed to simulate live.
        self.is_file = "://" not in self.url
        self.realtime = self.is_file and getattr(settings, "simulate_live", True)
        self.stall_s = float(getattr(settings, "ingest_stall_seconds", STALL_SECONDS) or STALL_SECONDS)
        self.stalls = 0

    def continue_after(self, last: Segment) -> None:
        """RESUME: keep segment indexes and the stream clock going from the last recorded segment."""
        self._next_index = last.index + 1
        self._clock_offset = last.end
        log.info("Resuming ingest after segment %d (stream t=%.1f)", last.index, last.end)

    def first_free_run(self) -> int:
        used = {int(p.name[1:p.name.index("_")]) for p in self.seg_dir.glob("r*_*")
                if p.name[1:p.name.index("_")].isdigit()}
        return max(used) + 1 if used else 0

    def _ffmpeg_cmd(self, run: int, csv: Path) -> list[str]:
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            *(["-re"] if self.realtime else []),
            "-i", self.url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.SEGMENT_SECONDS),
            "-segment_format", "mpegts",
            "-segment_list", str(csv),
            "-segment_list_type", "csv",
            # no -reset_timestamps: segments keep the source's continuous timestamps so clips can be
            # byte-joined frame-exactly (media.continuous_input)
            str(self.seg_dir / f"r{run:02d}_%06d.ts"),
        ]

    async def segments(self) -> AsyncIterator[Segment]:
        run = self.first_free_run()
        while True:
            csv = self.seg_dir / f"r{run:02d}_list.csv"
            csv.unlink(missing_ok=True)
            proc = await asyncio.create_subprocess_exec(*self._ffmpeg_cmd(run, csv))
            log.info("Ingest run %d started (pid %s)", run, proc.pid)
            last_end = 0.0
            async for name, start, end in self._tail_csv(csv, proc, self.stall_s, self._on_stall):
                last_end = end
                path = self.seg_dir / name
                pts = await first_video_pts(path)
                seg = Segment(self._next_index, path, self._clock_offset + start, self._clock_offset + end, pts)
                media.remember_segment(path, pts, seg.duration)
                self._next_index += 1
                yield seg
            if self.is_file:
                log.info("Finished reading file %s", self.url)
                return
            log.warning("Ingest run %d ended (code %s); reconnecting in 2s", run, proc.returncode)
            self._clock_offset += last_end
            run += 1
            await asyncio.sleep(2)

    def _on_stall(self) -> None:
        self.stalls += 1

    @staticmethod
    async def _tail_csv(csv: Path, proc, stall_s: float = STALL_SECONDS, on_stall=None):
        """Yield (name, start, end) per closed segment; kill FFmpeg if nothing new arrives for stall_s."""
        pos = 0
        buf = ""
        loop = asyncio.get_running_loop()
        last_progress = loop.time()
        while True:
            if csv.exists():
                with csv.open("r") as f:
                    f.seek(pos)
                    buf += f.read()
                    pos = f.tell()
                *lines, buf = buf.split("\n")
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        last_progress = loop.time()
                        yield parts[0], float(parts[1]), float(parts[2])
            if proc.returncode is not None:
                return
            if loop.time() - last_progress > stall_s:
                log.warning("Ingest stalled: no new segment for %.0fs while ffmpeg (pid %s) is alive — "
                            "killing it to reconnect", stall_s, getattr(proc, "pid", "?"))
                if on_stall:
                    on_stall()
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                return
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass


async def first_video_pts(path: Path) -> float | None:
    """First video PTS of a closed segment (async; None if it cannot be read)."""
    try:
        return (await media.probe_async(path)).get("video_start")
    except (OSError, RuntimeError) as exc:
        log.warning("Could not read PTS of %s: %s", path.name, exc)
        return None
