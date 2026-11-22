"""Replay a recorded run (runtime/events.jsonl + segments.jsonl) through fusion → planning → render.

    python -m motion_ai.replay --plan-only   # fusion + clip plans as JSON, no encoding (seconds)
    python -m motion_ai.replay               # full package render from cached segments

Lets fusion/clip rules be changed and checked without re-running speech-to-text and OCR.

Claude usage (manifest `llm_usage`): a replay re-runs only the curator call (powerplay scope; match scope has
no LLM call at all). Commentary events come from the recording, so replay usage is NOT the live usage of the
video; take per-video numbers from the live run's package.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .config import Settings, load_match
from .events import CLIP_EVENTS, EventType
from .fusion.fuser import Fuser
from .highlights.clips import ClipCutter, sidecar
from .highlights.curator import curate
from .highlights.pipeline import AFTER_PP_GRACE_S, before_first_ball, build_match_package
from .highlights.renderer import render
from .ingest.buffer import SegmentBuffer
from .llm import LLM
from .recorder import load_events, load_segments


def fuse_recorded(settings: Settings):
    fuser = Fuser(settings.ocr_available, settings.fusion_settle_seconds, settings.ocr_lag_seconds)
    for e in sorted(load_events(settings.work_dir), key=lambda e: e.t):
        fuser.add(e)
    fused = fuser.finalize(float("inf"))
    innings = next((f.t for f in fused if f.type == EventType.INNINGS_START), None)
    pp_end = None if settings.match_scope else next(
        (f.t for f in fused if f.type == EventType.POWERPLAY_END and innings is not None), None)
    clip_events = [f for f in fused if f.type in CLIP_EVENTS
                   and (pp_end is None or f.t <= pp_end + AFTER_PP_GRACE_S)
                   and not before_first_ball(f, innings)]
    return fused, clip_events, pp_end


REPLAY_USAGE_NOTE = ("replay: commentary events were read from the recording (no commentary calls); "
                     "only the curator call was re-run, so this is not the live per-video usage")


def replay_usage(llm: LLM | None) -> dict:
    """`llm_usage` for a replayed package, labelled so it is never mistaken for a live run."""
    from .llm import summarize_usage
    summary = llm.usage_summary() if llm is not None else summarize_usage([])
    return summary | {"run": "replay", "note": REPLAY_USAGE_NOTE}


async def replay(settings: Settings, plan_only: bool) -> None:
    buffer = SegmentBuffer()
    buffer.segments = load_segments(settings.work_dir)
    buffer.closed = True
    fused, clip_events, pp_end = fuse_recorded(settings)
    cutter = ClipCutter(buffer, settings.work_dir / "clips", settings.video_encoder,
                         match_config=settings.match_config_path)
    plans = [c for c in [await cutter.plan(ev) for ev in clip_events] if c]

    if plan_only:
        out = {"powerplay_end_t": pp_end, "clips": [sidecar(c) for c in plans]}
        if settings.match_scope:
            from .highlights.curator import MatchFacts, match_plan
            facts = MatchFacts.from_events(fused)
            mp = match_plan(plans, settings.highlight_target_seconds, facts, "plan")
            out |= {"chosen": [c.event.id for c, _ in mp.plan.items],
                    "chosen_seconds": round(mp.plan.duration, 1), "withheld": mp.withheld,
                    "wicket_audit": mp.wicket_audit, "package_holds": mp.package_holds,
                    "facts": {"innings2_t": facts.innings2_t, "innings1_runs": facts.innings1_runs,
                              "target": facts.target, "result": facts.result_score}}
        print(json.dumps(out, indent=1, default=str))
        return
    clips = [c for c in [await cutter.render(p) for p in plans] if c]
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if settings.match_scope:
        await build_match_package(clips, fused, settings, load_match(settings.match_config_path), buffer,
                                  {"replayed": True, "llm_usage": replay_usage(None)})
        return
    title = "Powerplay highlights" if pp_end is not None else "Match highlights"
    llm = LLM(settings)
    plan = await curate(llm, clips, settings.highlight_target_seconds,
                        load_match(settings.match_config_path), title)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    await render(plan, settings.output_dir, {"replayed": True, "llm_usage": replay_usage(llm)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    asyncio.run(replay(Settings(), args.plan_only))


if __name__ == "__main__":
    main()
