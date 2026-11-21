"""Rebuild the highlights package from clips already rendered in <WORK_DIR>/clips (reads each clip's
sidecar .json) — no re-processing, no re-encoding of clips.

    python -m motion_ai.rebuild
"""
from __future__ import annotations

import asyncio
import json
import logging

from .config import Settings, load_match
from .events import EventType, FusedEvent, Source
from .highlights.clips import Clip
from .highlights.curator import curate
from .highlights.renderer import render
from .llm import LLM


def load_clips(settings: Settings) -> list[Clip]:
    clips = []
    for side in sorted((settings.work_dir / "clips").glob("*.json")):
        d = json.loads(side.read_text())
        mp4 = side.with_suffix(".mp4")
        if not mp4.exists():
            continue
        ev = FusedEvent(EventType(d["type"]), d["t"], d["confidence"], {Source(s) for s in d["sources"]},
                        d.get("detail", {}), needs_review=bool(d["hold_reasons"]), id=d["id"],
                        hold_reasons=list(d["hold_reasons"]))
        clips.append(Clip(ev, [tuple(r) for r in d["segments"]], mp4, d["caption"], d["caption_t"]))
    return clips


async def rebuild(settings: Settings) -> None:
    clips = load_clips(settings)
    logging.info("Rebuilding from %d existing clips", len(clips))
    plan = await curate(LLM(settings), clips, settings.highlight_target_seconds,
                        load_match(settings.match_config_path))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    await render(plan, settings.output_dir, {"rebuilt": True})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    asyncio.run(rebuild(Settings()))
