"""Record raw detector events and segments to JSONL so runs can be replayed without re-processing.

    <WORK_DIR>/events.jsonl    one raw detector Event per line
    <WORK_DIR>/segments.jsonl  one Segment per line (index, path, stream start/end, optional first video "pts")

`python -m motion_ai.replay` rebuilds the package from these in seconds.

Crash / restart (the files are never truncated any more):
  * RESUME=true (settings.resume, read via getattr): the same live run continues. Both files are appended to;
    main.py reloads the recorded segments into the buffer and the recorded events into the fuser, and ingest
    continues the segment index and stream clock after the last recorded segment. New .ts files use the next
    free run number (rNN_), so nothing on disk is overwritten.
  * Fresh start (default): non-empty events.jsonl / segments.jsonl of the previous run are MOVED to
    <WORK_DIR>/archive/<timestamp>/ (with their .ts files kept on disk under their own run prefix) and new empty
    files are started. Nothing is deleted.
Either way the live files stay at <WORK_DIR>/events.jsonl and segments.jsonl in the same line format, so
replay / redetect / quality on runtime_1080 work unchanged. "pts" is optional: old files load as pts=None.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from . import media
from .events import Event, EventType, Source
from .ingest.srt_ingest import Segment

log = logging.getLogger(__name__)


def event_to_dict(e: Event) -> dict:
    return {"type": e.type.value, "t": e.t, "source": e.source.value,
            "confidence": e.confidence, "detail": e.detail, "id": e.id}


def event_from_dict(d: dict) -> Event:
    return Event(EventType(d["type"]), d["t"], Source(d["source"]), d["confidence"], d["detail"], d["id"])


def segment_to_dict(s: Segment) -> dict:
    d = {"index": s.index, "path": str(s.path), "start": s.start, "end": s.end}
    if s.pts is not None:
        d["pts"] = s.pts
    return d


def archive_previous(work_dir: Path) -> Path | None:
    """Move a previous run's non-empty events/segments files into archive/<stamp>/ (fresh start, no data loss)."""
    files = [work_dir / n for n in ("events.jsonl", "segments.jsonl")]
    files = [p for p in files if p.exists() and p.stat().st_size > 0]
    if not files:
        return None
    base = dest = work_dir / "archive" / time.strftime("%Y%m%d_%H%M%S")
    n = 1
    while dest.exists():
        dest = base.with_name(f"{base.name}_{n}")
        n += 1
    dest.mkdir(parents=True)
    for p in files:
        shutil.move(str(p), dest / p.name)
    log.warning("Previous run's recordings moved to %s (set RESUME=true to continue a crashed run instead)", dest)
    return dest


class Recorder:
    def __init__(self, work_dir: Path, resume: bool = False):
        work_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = work_dir / "events.jsonl"
        self.segments_path = work_dir / "segments.jsonl"
        self.archived = None if resume else archive_previous(work_dir)
        for p in (self.events_path, self.segments_path):
            p.touch()   # append mode: never truncate
            _end_with_newline(p)   # a crash mid-write left a torn line: don't glue the next record onto it

    def event(self, e: Event) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(event_to_dict(e), ensure_ascii=False, default=str) + "\n")

    def segment(self, s: Segment) -> None:
        with self.segments_path.open("a") as f:
            f.write(json.dumps(segment_to_dict(s)) + "\n")


def _end_with_newline(path: Path) -> None:
    if path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            f.write(b"\n")


def _lines(path: Path) -> list[str]:
    """Non-empty lines; a torn last line (crash mid-write) is skipped, not fatal."""
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            json.loads(ln)
        except ValueError:
            log.warning("Skipping unreadable line in %s: %.80s", path.name, ln)
            continue
        out.append(ln)
    return out


def load_events(work_dir: Path) -> list[Event]:
    return [event_from_dict(json.loads(ln)) for ln in _lines(work_dir / "events.jsonl")]


def load_segments(work_dir: Path) -> list[Segment]:
    out = []
    for ln in _lines(work_dir / "segments.jsonl"):
        d = json.loads(ln)
        seg = Segment(d["index"], Path(d["path"]), d["start"], d["end"], d.get("pts"))
        media.remember_segment(seg.path, seg.pts, seg.duration)
        out.append(seg)
    return out
