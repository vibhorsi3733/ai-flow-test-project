"""Propose scoreboard OCR regions (config/scoreboard.json format) for a new broadcaster / video.

    python -m motion_ai.calibrate <video> [--at 100 300 ...] [--out config/scoreboard.proposed.json] [--gpu]

1. Grabs frames at the given times (default: 5 evenly spaced in the middle of the video), native resolution.
2. Runs PaddleOCR detection + recognition over the bottom ~25% of each frame.
3. Looks for a scoreline ("73-1" + overs, possibly two boxes), "TARGET n" and the "OVERS x TOTAL r-w" footer,
   using the project's own parsers (score_tracker.parse_scoreline / parse_target / parse_footer).
4. Proposes padded regions that are stable across frames, writes them as JSON, saves previews under
   runtime/calibration/<video>/ and re-reads every region with ScoreboardOCRDetector to validate.

The pure parts (box merging, clustering, union/padding, choosing the scoreline) need no OCR and are unit-tested.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .detectors.score_tracker import parse_footer, parse_scoreline, parse_target

BOTTOM_FRACTION = 0.25
# Max horizontal gap between neighbouring boxes of one item, in text heights. The footer's
# "OVERS 20" and "TOTAL 179-7" sit far apart on the same card line.
MAX_GAP = {"scoreline": 1.5, "target": 2.0, "footer": 12.0}
MIN_FRAMES = {"scoreline": 2, "target": 1, "footer": 1}
PAD = {"target": (1.5, 0.4)}   # (x, y) in text heights; default 0.5 / 0.25. TARGET text is small: pad wider
SCALE = {"footer": 2}   # wide boxes need less upscaling (ScoreboardOCRDetector default is 3)
DENSIFY = (5, 9, 17)   # default sample counts tried in turn until a stable scoreline is found


@dataclass(frozen=True)
class Box:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float = 1.0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


# ------------------------------------------------------------------------------------------- pure helpers
def is_scoreline(text: str) -> bool:
    """A score WITH overs ('73-1 7.5', '6-00.4'); a bare '73-1' is not enough to place the region."""
    st = parse_scoreline(text)
    return st is not None and st.over >= 0


PREDICATES: dict[str, Callable[[str], bool]] = {
    "scoreline": is_scoreline,
    "target": lambda t: parse_target(t) is not None,
    "footer": lambda t: parse_footer(t) is not None,
}


def _v_overlap(a: Box, b: Box) -> float:
    """Vertical overlap as a fraction of the shorter box."""
    ov = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(ov, 0) / max(min(a.h, b.h), 1)


def _h_overlap(a: Box, b: Box) -> float:
    ov = min(a.x1, b.x1) - max(a.x0, b.x0)
    return max(ov, 0) / max(min(a.w, b.w), 1)


def group_lines(boxes: list[Box], min_v_overlap: float = 0.5) -> list[list[Box]]:
    """Boxes on the same text line (vertical overlap >= min_v_overlap), each line sorted left to right."""
    lines: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: (b.y0 + b.y1) / 2):
        for line in lines:
            if any(_v_overlap(b, o) >= min_v_overlap for o in line):
                line.append(b)
                break
        else:
            lines.append([b])
    return [sorted(line, key=lambda b: b.x0) for line in lines]


def merge_boxes(boxes: list[Box]) -> Box:
    """One box spanning `boxes` (left to right), texts joined with a space."""
    bs = sorted(boxes, key=lambda b: b.x0)
    return Box(" ".join(b.text for b in bs), min(b.x0 for b in bs), min(b.y0 for b in bs),
               max(b.x1 for b in bs), max(b.y1 for b in bs), min(b.conf for b in bs))


def find_spans(boxes: list[Box], predicate: Callable[[str], bool], max_gap: float,
               max_boxes: int = 3) -> list[Box]:
    """Smallest runs of neighbouring boxes on one line whose joined text satisfies `predicate`.

    Neighbours are merged only while the horizontal gap is <= max_gap x text height
    (e.g. score '6-0' + overs '0.4' as separate boxes). A run is kept only if no shorter run inside it matches,
    so 'SIDHAANT 42 6-0 0.4' yields just '6-0 0.4'."""
    out: list[Box] = []
    for line in group_lines(boxes):
        found: list[range] = []
        for n in range(1, max_boxes + 1):
            for i in range(len(line) - n + 1):
                run = line[i:i + n]
                if any(r.start >= i and r.stop <= i + n for r in found):
                    continue   # a shorter match lies inside
                if any(b.x0 - a.x1 > max_gap * max(min(a.h, b.h), 1) for a, b in zip(run, run[1:])):
                    continue
                if predicate(" ".join(b.text for b in run)):
                    found.append(range(i, i + n))
                    out.append(merge_boxes(run))
    return out


def cluster(cands: list[tuple[int, Box]], min_overlap: float = 0.5) -> list[list[tuple[int, Box]]]:
    """Group (frame_index, box) candidates that sit in the same place on screen (overlap both ways)."""
    groups: list[list[tuple[int, Box]]] = []
    for fi, b in cands:
        for g in groups:
            u = union_box([x for _, x in g])
            if _v_overlap(b, u) >= min_overlap and _h_overlap(b, u) >= min_overlap:
                g.append((fi, b))
                break
        else:
            groups.append([(fi, b)])
    return groups


def union_box(boxes: list[Box]) -> Box:
    return Box("", min(b.x0 for b in boxes), min(b.y0 for b in boxes),
               max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def pad_region(box: Box, text_h: float, width: int, height: int,
               pad_x: float = 0.5, pad_y: float = 0.25) -> dict:
    """{x, y, w, h} around `box`, padded by pad_x / pad_y text heights (min 6 / 4 px), clamped to the frame.

    Horizontal padding is generous: the score's width changes (6-0 -> 181-5) and moves the overs."""
    px, py = max(round(pad_x * text_h), 6), max(round(pad_y * text_h), 4)
    x0, y0 = max(box.x0 - px, 0), max(box.y0 - py, 0)
    x1, y1 = min(box.x1 + px, width), min(box.y1 + py, height)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _n_frames(g: list[tuple[int, Box]]) -> int:
    return len({fi for fi, _ in g})


def _text_h(g: list[tuple[int, Box]]) -> int:
    hs = sorted(b.h for _, b in g)
    return hs[len(hs) // 2]


def rank_clusters(groups: list[list[tuple[int, Box]]], min_frames: int = 2) -> list[list[tuple[int, Box]]]:
    """Clusters seen on >= min_frames frames, best first: most frames, ties to the taller text (the main score
    bar is bigger than e.g. the bowler's figures '0-6 0.4', which also parse as a scoreline)."""
    ok = [g for g in groups if _n_frames(g) >= min_frames]
    return sorted(ok, key=lambda g: (_n_frames(g), _text_h(g)), reverse=True)


def choose_cluster(groups: list[list[tuple[int, Box]]], min_frames: int = 2) -> list[tuple[int, Box]] | None:
    ranked = rank_clusters(groups, min_frames)
    return ranked[0] if ranked else None


def choose_alt(groups: list[list[tuple[int, Box]]], main: list[tuple[int, Box]], min_frames: int = 2,
               min_rel_h: float = 0.75) -> list[tuple[int, Box]] | None:
    """A second scoreline position (compact bar shown with banners -> 'scoreline_alt'): another stable cluster
    with text about as tall as the main bar (bowler figures are much smaller) that never shows on the same
    frame as the main bar."""
    main_frames = {fi for fi, _ in main}
    for g in rank_clusters(groups, min_frames):
        if g is main or _text_h(g) < min_rel_h * _text_h(main):
            continue
        if not main_frames & {fi for fi, _ in g}:
            return g
    return None


def near(a: Box, b: Box, max_dy: float) -> bool:
    """a and b overlap horizontally and are at most max_dy px apart vertically (e.g. TARGET above the score)."""
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return min(a.x1, b.x1) > max(a.x0, b.x0) and dy <= max_dy


def propose(per_frame: list[list[Box]], width: int, height: int,
            min_frames: dict[str, int] | None = None) -> tuple[dict, dict]:
    """Per-frame OCR boxes -> (regions, info). info[name] = {frames, texts, text_h} for the report."""
    min_frames = MIN_FRAMES | (min_frames or {})
    regions, info = {}, {}
    main: list[tuple[int, Box]] = []

    def add(name: str, g: list[tuple[int, Box]]) -> None:
        bs = [b for _, b in g]
        regions[name] = pad_region(union_box(bs), _text_h(g), width, height, *PAD.get(name, (0.5, 0.25)))
        if name in SCALE:
            regions[name]["scale"] = SCALE[name]
        info[name] = {"frames": sorted({fi for fi, _ in g}), "texts": [b.text for b in bs], "text_h": _text_h(g)}

    for name, pred in PREDICATES.items():
        cands = [(fi, b) for fi, boxes in enumerate(per_frame) for b in find_spans(boxes, pred, MAX_GAP[name])]
        if name == "target":
            # the detector reads TARGET at one place: next to the main score bar (not the compact layout's)
            if "scoreline" not in info:
                continue
            sl = union_box([b for _, b in main])
            cands = [(fi, b) for fi, b in cands if near(b, sl, 3 * info["scoreline"]["text_h"])]
        groups = cluster(cands)
        need = min(min_frames[name], max(len(per_frame), 1))
        g = choose_cluster(groups, need)
        if not g:
            continue
        add(name, g)
        if name == "scoreline":
            main = g
        if name == "scoreline" and (alt := choose_alt(groups, g, need)):
            add("scoreline_alt", alt)
    return regions, info


def default_times(duration: float, n: int = 5) -> list[float]:
    """n times evenly spread over the middle 70% of the video (skips intro / outro)."""
    if n == 1:
        return [round(duration / 2, 1)]
    # 5 -> 9 -> 17 keeps every earlier time, so densifying only grabs the new ones
    return [round(duration * (0.15 + 0.7 * i / (n - 1)), 1) for i in range(n)]


def build_config(regions: dict, width: int, height: int, video: str) -> dict:
    return {
        "_comment": f"PROPOSAL from `python -m motion_ai.calibrate {video}` - review the previews under "
                    "runtime/calibration/ and the validation table, then copy to config/scoreboard.json. "
                    "'scoreline' = score + overs read together; 'target' = chase-only TARGET line; 'footer' = "
                    "'OVERS x TOTAL r-w' scorecard footer (innings totals); 'scoreline_alt' = a second score-bar position "
                    "(read only when 'scoreline' is empty).",
        "season": f"calibrated-{Path(video).stem}",
        "resolution": [width, height],
        "regions": regions,
        "consensus_frames": 3,
    }


# ------------------------------------------------------------------------------------------- OCR / IO
def probe(video: Path) -> tuple[int, int, float]:
    """(width, height, duration_s) at native resolution."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return w, h, duration


def grab_frames(video: Path, times: list[float]):
    """[(t, RGB frame)] at native resolution."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            out.append((t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            print(f"  ! no frame at {t}s")
    cap.release()
    return out


def detect_boxes(ocr, frame) -> list[Box]:
    """PaddleOCR det+rec over the bottom BOTTOM_FRACTION of the frame, in full-frame pixel coordinates."""
    import cv2

    y_off = int(frame.shape[0] * (1 - BOTTOM_FRACTION))
    res = ocr.ocr(cv2.cvtColor(frame[y_off:], cv2.COLOR_RGB2BGR), cls=False)
    boxes = []
    for quad, (text, conf) in (res[0] or []) if res else []:
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        boxes.append(Box(text, int(min(xs)), int(min(ys)) + y_off, int(max(xs)), int(max(ys)) + y_off, float(conf)))
    return boxes


COLORS = {"scoreline": (0, 220, 0), "scoreline_alt": (220, 0, 220), "target": (255, 160, 0),
          "footer": (0, 160, 255)}


def save_previews(out_dir: Path, frames, per_frame: list[list[Box]], regions: dict, info: dict) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):   # previews of an earlier run
        old.unlink()
    for fi, (t, frame) in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        for b in per_frame[fi]:
            cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), (160, 160, 160), 1)
        for name, r in regions.items():
            c = COLORS[name][::-1]
            cv2.rectangle(img, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), c, 2)
            cv2.putText(img, name, (r["x"], max(r["y"] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.imwrite(str(out_dir / f"frame_{t:07.1f}s.png"), img)
    for name, r in regions.items():
        t, frame = frames[info[name]["frames"][0]]
        crop = frame[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out_dir / f"region_{name}_{t:.1f}s.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def validate(config_path: Path, frames, use_gpu: bool) -> None:
    """Re-read each proposed region with the project's own detector + parsers and print a table."""
    from .config import Settings
    from .detectors.ocr_scoreboard import ScoreboardOCRDetector

    det = ScoreboardOCRDetector(Settings(srt_url="calibrate", ocr_available=True, ocr_use_gpu=use_gpu,
                                         ocr_config_path=config_path, _env_file=None))
    parsers = {"scoreline": parse_scoreline, "scoreline_alt": parse_scoreline, "target": parse_target,
               "footer": parse_footer}
    print(f"\n{'time':>8}  {'region':<13} {'raw text':<28} parsed")
    for t, frame in frames:
        for name in det.regions:
            raw = det._read_region(frame, name)
            print(f"{t:>7.1f}s  {name:<13} {raw!r:<28} {parsers[name](raw) if name in parsers else ''}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m motion_ai.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--at", type=float, nargs="+", help="frame times in seconds (default: 5 in the middle)")
    ap.add_argument("--out", type=Path, default=Path("config/scoreboard.proposed.json"))
    ap.add_argument("--preview-dir", type=Path, default=Path("runtime/calibration"))
    ap.add_argument("--gpu", action="store_true", help="run PaddleOCR on the GPU")
    args = ap.parse_args(argv)

    w, h, duration = probe(args.video)
    print(f"{args.video}: {w}x{h}, {duration:.0f}s")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=args.gpu, show_log=False)
    frames, per_frame = [], []
    # Explicit --at: those times only. Default: 5 middle times, densified (9, 17) while no stable scoreline
    # (replays / ads / highlight-reel cuts hide the score bar on many frames).
    for times in ([args.at] if args.at else [default_times(duration, n) for n in DENSIFY]):
        seen = {t for t, _ in frames}
        new = grab_frames(args.video, [t for t in times if t not in seen])
        for t, f in new:
            boxes = detect_boxes(ocr, f)
            print(f"  {t:>7.1f}s  {[b.text for b in boxes]}")
            frames.append((t, f))
            per_frame.append(boxes)
        regions, info = propose(per_frame, w, h)
        if "scoreline" in regions:
            break
    if not frames:
        sys.exit("no frames read")
    for name in ("scoreline", "scoreline_alt", "target", "footer"):
        if name == "scoreline_alt" and name not in regions:
            continue
        if name in regions:
            i = info[name]
            print(f"  {name:<13} {regions[name]}  seen on {len(i['frames'])}/{len(frames)} frames: {i['texts']}")
        else:
            print(f"  {name:<13} not found (need {MIN_FRAMES[name]}+ frames; try --at times where it is on screen)")
    if "scoreline" not in regions:
        sys.exit("no stable scoreline found - pass --at times where the score bar is visible")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_config(regions, w, h, str(args.video)), indent=2) + "\n")
    prev_dir = args.preview_dir / args.video.stem
    save_previews(prev_dir, frames, per_frame, regions, info)
    print(f"wrote {args.out}; previews in {prev_dir}/")
    validate(args.out, frames, args.gpu)


if __name__ == "__main__":
    main()
