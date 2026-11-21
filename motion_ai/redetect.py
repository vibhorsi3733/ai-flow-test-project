"""Re-run ONE detector over a recorded run (after a parser / region / model change), keeping the others' events.

    python -m motion_ai.redetect ocr              # -> <WORK_DIR>/events_ocr.jsonl
    python -m motion_ai.redetect commentary       # -> <WORK_DIR>/events_commentary.jsonl
    python -m motion_ai.redetect ocr --apply      # swap that detector's events into <WORK_DIR>/events.jsonl

Then `python -m motion_ai.replay` rebuilds the package.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from .config import Settings, load_match
from .recorder import event_to_dict, load_segments

log = logging.getLogger("motion_ai.redetect")
SOURCE_OF = {"ocr": "ocr", "commentary": "asr", "audio": "audio"}


async def redetect(settings: Settings, name: str) -> None:
    from .detectors import build_detectors
    from .llm import LLM
    from .main import setup_logging

    dets = [d for d in build_detectors(settings, LLM(settings), load_match(settings.match_config_path)) if d.name == name]
    setup_logging()
    if not dets:
        raise SystemExit(f"No detector named {name!r} (OCR_AVAILABLE must be true for ocr)")
    det = dets[0]
    out = settings.work_dir / f"events_{name}.jsonl"
    segs = load_segments(settings.work_dir)
    n, t0 = 0, time.time()
    with out.open("w") as f:
        for i, seg in enumerate(segs):
            for ev in await det.process(seg):
                f.write(json.dumps(event_to_dict(ev), ensure_ascii=False, default=str) + "\n")
                n += 1
            if i % 20 == 0:
                log.info("%s: segment %d/%d (%.0fs of video) after %.0fs", name, i, len(segs), seg.end, time.time() - t0)
    log.info("Re-ran %s on %d segments in %.0fs: %d events -> %s", name, len(segs), time.time() - t0, n, out)


def apply(settings: Settings, name: str) -> None:
    src = SOURCE_OF[name]
    ev_path, new_path = settings.work_dir / "events.jsonl", settings.work_dir / f"events_{name}.jsonl"
    keep = [ln for ln in ev_path.read_text().splitlines() if ln.strip() and json.loads(ln)["source"] != src]
    new = [ln for ln in new_path.read_text().splitlines() if ln.strip()]
    ev_path.with_name(f"events.before_{name}.jsonl").write_text(ev_path.read_text())
    ev_path.write_text("\n".join(keep + new) + "\n")
    log.info("events.jsonl: kept %d other events, %d %s events from %s", len(keep), len(new), src, new_path.name)


def main() -> None:
    from .main import setup_logging

    ap = argparse.ArgumentParser()
    ap.add_argument("detector", choices=sorted(SOURCE_OF))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    setup_logging()
    settings = Settings()
    if args.apply:
        apply(settings, args.detector)
    else:
        asyncio.run(redetect(settings, args.detector))


if __name__ == "__main__":
    main()
