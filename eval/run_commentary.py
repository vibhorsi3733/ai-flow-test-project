"""Re-run the commentary detector on a recorded run WITHOUT writing into the run folder.

    python -m eval.run_commentary                          # reads runtime_1080, writes eval/results/
    python -m eval.run_commentary --work-dir runtime --out eval/results/x.jsonl

Transcripts are cached in eval/results/transcripts_<run>.json, so a second run (e.g. after a prompt change)
skips Whisper and only re-asks Claude. Then score it:

    python -m eval.score --events eval/results/events_commentary_runtime_1080.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from motion_ai.config import Settings, load_match
from motion_ai.detectors.commentary import CommentaryDetector
from motion_ai.llm import LLM
from motion_ai.recorder import event_to_dict, load_segments

log = logging.getLogger("eval.run_commentary")


class CachedTranscripts:
    """Wraps CommentaryDetector._transcribe with a per-segment cache keyed by segment index."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, list] = json.loads(path.read_text()) if path.exists() else {}

    def wrap(self, det: CommentaryDetector, seg_index_ref: list[int]):
        real = det._transcribe

        def cached(pcm):
            key = str(seg_index_ref[0])
            if key not in self.data:
                self.data[key] = [list(x) for x in real(pcm)]
            return [tuple(x) for x in self.data[key]]

        det._transcribe = cached

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False))


async def run(work_dir: Path, out: Path) -> None:
    settings = Settings()
    llm = LLM(settings)
    det = CommentaryDetector(settings, llm, load_match(settings.match_config_path))
    cache = CachedTranscripts(out.parent / f"transcripts_{work_dir.name}.json")
    seg_ref = [0]
    cache.wrap(det, seg_ref)

    segs = load_segments(work_dir)
    n, t0 = 0, time.time()
    with out.open("w") as f:
        for i, seg in enumerate(segs):
            seg_ref[0] = seg.index
            for ev in await det.process(seg):
                f.write(json.dumps(event_to_dict(ev), ensure_ascii=False, default=str) + "\n")
                n += 1
            if i % 20 == 0:
                log.info("segment %d/%d after %.0fs (LLM calls %d, failed %d)",
                         i, len(segs), time.time() - t0, llm.calls, llm.failures)
                cache.save()
    cache.save()
    log.info("%d commentary events -> %s in %.0fs | backend %s, LLM calls %d, failed %d",
             n, out, time.time() - t0, llm.backend, llm.calls, llm.failures)


def main() -> None:
    from motion_ai.main import setup_logging

    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="runtime_1080")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    setup_logging()
    work_dir = Path(a.work_dir)
    out = Path(a.out or f"eval/results/events_commentary_{work_dir.name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(work_dir, out))


if __name__ == "__main__":
    main()
