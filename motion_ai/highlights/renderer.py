"""Join the chosen clips into the final package + captions (.srt) + manifest.json.

Captions are burned into each clip at render time (readable on mute); the .srt mirrors them.
The manifest carries the review/hold decision for the publishing step.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from ..branding import active_template
from ..events import EventType
from ..media import concat_clips
from .curator import Plan

log = logging.getLogger(__name__)

APPROVE_WINDOW_S = 600   # approve_by = finish + 10 min
TIMEOUT_MIN_CLIPS, TIMEOUT_MIN_SECONDS = 2, 20.0


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def timeout_outcome(plan: Plan) -> tuple[str, list[str]]:
    """What happens if nobody approves by approve_by (IPL head's rule)."""
    withheld = [c.event.id for c, _ in plan.items if c.event.needs_review]
    if plan.status != "hold_for_review":
        return "not_required", withheld
    ok = [c for c, _ in plan.items if not c.event.needs_review]
    meets = (len(ok) >= TIMEOUT_MIN_CLIPS and sum(c.duration for c in ok) >= TIMEOUT_MIN_SECONDS
             and any(c.event.type in (EventType.WICKET, EventType.SIX) for c in ok))
    return ("pending" if meets else "below_minimum"), withheld


def manifest_score(detail: dict) -> dict:
    """Manifest score fields of a clip. `score`/`overs` = the score AT the event: the dismissal score
    (event_score/event_overs) when the fuser derived one, else the board read itself. When the dismissal score
    could not be inferred (score_basis "unknown", e.g. a run-out off a wide behind a gap change) score/overs are
    None, never the later board read. `board_score`/`board_overs` = the scoreboard read that confirmed the event
    (gap wicket 139/4 (16.2) was confirmed by the 157/4 (18.0) read)."""
    board_score, board_overs = detail.get("score"), detail.get("overs")
    basis = detail.get("score_basis")
    if detail.get("event_score"):
        score, overs = detail["event_score"], detail.get("event_overs")
    elif basis == "unknown":
        score, overs = None, None
    else:
        score, overs = board_score, board_overs
    out = {"score": score, "overs": overs, "board_score": board_score, "board_overs": board_overs}
    if basis:
        out["score_basis"] = basis
    return out


async def render(plan: Plan, out_dir: Path, extra: dict | None = None) -> Path | None:
    if not plan.items:
        log.error("Nothing to render — no clips selected")
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video = out_dir / f"powerplay_highlights_{stamp}.mp4"
    await concat_clips([c.path for c, _ in plan.items], video)

    srt, manifest, cursor = [], [], 0.0
    for i, (clip, caption) in enumerate(plan.items, 1):
        off = clip.offset_of(clip.caption_t) if clip.caption_t is not None else 0.0
        srt.append(f"{i}\n{_ts(cursor + off)} --> {_ts(cursor + clip.duration)}\n{caption}\n")
        manifest.append({
            "start_in_package": round(cursor, 2), "duration": round(clip.duration, 2),
            "type": clip.event.type.value, "caption": caption,
            "caption_t": clip.caption_t, "caption_in_package": round(cursor + off, 2),
            "segments": clip.ranges, "confidence": clip.event.confidence,
            "hold_reasons": clip.event.hold_reasons, "needs_review": clip.event.needs_review,
            "sources": sorted(s.value for s in clip.event.sources),
            "match_time_s": round(clip.event.t, 1), "clip_file": clip.path.name,
        })
        cursor += clip.duration

    on_timeout_result, withheld = timeout_outcome(plan)
    package = {
        "title": plan.title, "status": plan.status, "duration": round(cursor, 1),
        "approve_by": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + APPROVE_WINDOW_S)),
        "on_timeout": "publish_non_held",
        "timeout_minimum": {"clips": TIMEOUT_MIN_CLIPS, "seconds": TIMEOUT_MIN_SECONDS,
                            "needs_one_of": ["wicket", "six"]},
        "on_timeout_result": on_timeout_result, "withheld": withheld,
        "left_out": plan.withheld,          # clips left out on purpose (e.g. not_selected_over)
        "publish_channel_allowed": False,   # until the IPL template + approve step exist
        **(extra or {}),
        "clips": manifest,
    }
    video.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    video.with_suffix(".json").write_text(json.dumps(package, indent=2, ensure_ascii=False, default=str))
    log.info("PACKAGE READY: %s (%.0fs, %d clips, %s) — %s", video, cursor, len(plan.items), plan.status, plan.title)
    return video


# ---------------------------------------------------------------------------------------------------------
# Full-match package: cards + both innings (fm/guidance_v1 §3). Cards carry a soft crowd bed from the feed.
def card_texts(facts, match: dict, sizes: dict[str, int] | None = None) -> dict[str, list[tuple[str, int]] | None]:
    """Card lines. A number appears only when it comes from a confirmed read (never guessed).
    `sizes`: font sizes by role (default: the branding template's font.sizes)."""
    sz = sizes or active_template().sizes
    H, SUB, STAT, BODY, SMALL = sz["headline"], sz["subhead"], sz["stat"], sz["body"], sz["small"]
    teams = list((match.get("teams") or {}).keys())
    first = match.get("batting_first") or (teams[0] if teams else "1st innings")
    second = next((t for t in teams if t != first), "2nd innings")
    vs = f"{first.upper()} v {second.upper()}" if teams else (match.get("match") or "MATCH").upper()
    cards: dict[str, list[tuple[str, int]] | None] = {
        "title": [(vs, BODY), ("MATCH HIGHLIGHTS", H)],
        "innings1": [("1st INNINGS", H), (first.upper(), BODY)],
        "break": [("INNINGS BREAK", H)],
        "innings2": [("2nd INNINGS", H), (second.upper(), BODY)],
        "result": None,
        "outro": [("MATCH HIGHLIGHTS", BODY), (vs, SMALL)],
    }
    if facts.target_matches_innings1:   # TARGET read confirms the first-innings total (target - 1)
        cards["break"].append((f"{first.upper()}  {facts.innings1_runs}", STAT))
        cards["innings2"].append((f"TARGET {facts.target}", STAT))
    elif facts.target is not None:
        cards["innings2"].append((f"TARGET {facts.target}", STAT))
    if facts.result_score is not None:
        r, w = facts.result_score
        cards["result"] = [(f"{second.upper()} WIN", H), (f"BY {10 - w} WICKETS", SUB), (f"{r}/{w}", BODY)]
    return cards


def crowd_windows(layout: list[str | tuple[float, float]], seconds: dict[str, float],
                  limits: tuple[float, float] | None = None) -> dict[str, float]:
    """Stream time each card's crowd bed starts at. `layout` = the package order: card names and clip spans
    (t0, t1). A run of back-to-back cards shares one continuous stretch, taken right after the previous clip
    (the reaction / celebration, never audio the viewer just heard) or, for the opening cards, just before the
    first clip. Clamped into `limits` (the buffered stream). Cards with no clip either side get no bed."""
    out: dict[str, float] = {}
    i = 0
    while i < len(layout):
        if not isinstance(layout[i], str):
            i += 1
            continue
        j = i
        while j < len(layout) and isinstance(layout[j], str):
            j += 1
        run = layout[i:j]
        total = sum(seconds[n] for n in run)
        if i > 0:
            t = layout[i - 1][1]
        elif j < len(layout):
            t = layout[j][0] - total
        else:
            i = j
            continue
        if limits:
            t = max(limits[0], min(t, limits[1] - total))
        for n in run:
            out[n] = round(t, 3)
            t += seconds[n]
        i = j
    return out


async def render_match(mp, facts, match: dict, out_dir: Path, work_dir: Path, encoder: str,
                       size: tuple[int, int], fps: float, extra: dict | None = None,
                       covering: Callable[[float, float], list] | None = None,
                       audio_rate: int | None = None) -> Path | None:
    """`covering(t0, t1)` -> ingest segments (SegmentBuffer.covering): source of the cards' crowd bed.
    `audio_rate`: package audio rate (= the clips' source rate) for the cards, so the -c copy join works."""
    from ..media import render_card
    from .curator import CARD_SECONDS

    plan = mp.plan
    if not plan.items:
        log.error("Nothing to render — no clips selected")
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video = out_dir / f"match_highlights_{stamp}.mp4"
    tpl = active_template()
    seconds = {**CARD_SECONDS, **tpl.card_seconds}   # card lengths from the branding template
    texts = card_texts(facts, match, tpl.sizes)
    card_dir = work_dir / "cards"

    inn1 = [(c, cap) for c, cap in plan.items if mp.innings.get(c.event.id, 1) == 1]
    inn2 = [(c, cap) for c, cap in plan.items if mp.innings.get(c.event.id, 1) >= 2]
    has2 = bool(inn2) or facts.innings2_t is not None
    layout: list = ["title", "innings1", *inn1]
    if has2:
        layout += ["break", "innings2", *inn2, "result"]
    layout = [x for x in layout + ["outro"] if not isinstance(x, str) or texts.get(x)]
    buffered = covering(float("-inf"), float("inf")) if covering else []
    spans = [x if isinstance(x, str) else (x[0].t0, x[0].t1) for x in layout]
    bed_at = crowd_windows(spans, seconds, (buffered[0].start, buffered[-1].end)) if buffered else {}

    async def card(name: str):
        lines, sec = texts[name], seconds[name]
        path = card_dir / f"{name}.mp4"
        crowd = None
        if name in bed_at:
            segs = covering(bed_at[name], bed_at[name] + sec)
            crowd = ([s.path for s in segs], segs[0].start, bed_at[name]) if segs else None
        await render_card(lines, sec, path, size, fps, encoder, crowd, audio_rate, tpl)
        return {"name": name, "path": path, "duration": sec, "text": " / ".join(t for t, _ in lines),
                "crowd_bed_t": bed_at.get(name)}

    sequence = [await card(x) if isinstance(x, str) else x for x in layout]

    await concat_clips([s["path"] if isinstance(s, dict) else s[0].path for s in sequence], video)

    srt, manifest, cards, cursor, n = [], [], [], 0.0, 0
    for s in sequence:
        if isinstance(s, dict):
            cards.append({"card": s["name"], "text": s["text"], "start_in_package": round(cursor, 2),
                          "duration": s["duration"], "crowd_bed_t": s["crowd_bed_t"]})
            cursor += s["duration"]
            continue
        clip, caption = s
        n += 1
        off = clip.offset_of(clip.caption_t) if clip.caption_t is not None else 0.0
        srt.append(f"{n}\n{_ts(cursor + off)} --> {_ts(cursor + clip.duration)}\n{caption}\n")
        manifest.append({
            "start_in_package": round(cursor, 2), "duration": round(clip.duration, 2),
            "innings": mp.innings.get(clip.event.id, 1),
            "type": clip.event.type.value, "caption": caption,
            "caption_t": clip.caption_t, "caption_in_package": round(cursor + off, 2),
            "caption_offset_in_clip": round(off, 2),
            "segments": clip.ranges, "confidence": clip.event.confidence,
            "hold_reasons": clip.event.hold_reasons, "needs_review": clip.event.needs_review,
            "sources": sorted(s.value for s in clip.event.sources),
            **manifest_score(clip.event.detail),
            "quotes": clip.event.detail.get("quotes") or ([clip.event.detail["quote"]] if clip.event.detail.get("quote") else []),
            "match_time_s": round(clip.event.t, 1), "clip_file": clip.path.name, "id": clip.event.id,
        })
        cursor += clip.duration

    on_timeout_result, held_ids = timeout_outcome(plan)
    status = "hold_for_review" if (plan.status == "hold_for_review" or mp.package_holds) else plan.status
    package = {
        "title": plan.title, "scope": "match", "status": status, "package_holds": mp.package_holds,
        "duration": round(cursor, 1),
        "approve_by": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + APPROVE_WINDOW_S)),
        "on_timeout": "publish_non_held",
        "timeout_minimum": {"clips": TIMEOUT_MIN_CLIPS, "seconds": TIMEOUT_MIN_SECONDS,
                            "needs_one_of": ["wicket", "six"]},
        "on_timeout_result": on_timeout_result,
        "held_clip_ids": held_ids,
        "withheld": mp.withheld,
        "wicket_audit": mp.wicket_audit,
        "facts": {"innings2_start_t": facts.innings2_t, "innings1_runs": facts.innings1_runs,
                  "innings1_last_read": facts.innings1_last, "target_confirmed": facts.target,
                  "target_matches_innings1": facts.target_matches_innings1,
                  "result_score": facts.result_score},
        "budget_note": mp.budget_note,
        "publish_channel_allowed": False,
        "source_rights": "unverified_youtube_reupload",
        **(extra or {}),
        "cards": cards,
        "clips": manifest,
    }
    video.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    video.with_suffix(".json").write_text(json.dumps(package, indent=2, ensure_ascii=False, default=str))
    log.info("PACKAGE READY: %s (%.0fs, %d clips + %d cards, %s)", video, cursor, n, len(cards), status)
    return video
