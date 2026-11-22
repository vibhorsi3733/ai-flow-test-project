"""Quality gate (panel spec fm/guidance_v1 §2, IPL head condition 5): binding for the whole package.

Per clip, the part BEFORE the caption appears (captions/cards would skew the score) is compared with the
same moment of the ORIGINAL source file (not the ingest segments: re-joining 2 s .ts segments shifts a frame at
some boundaries, which would score the reference, not our encode). The stream-clock -> file-time offset is
found once by SSIM search (±0.2 s in 1-frame steps). Live SRT (no file): falls back to the ingest segments.
Thresholds:  VMAF mean >= 95, VMAF 5th percentile >= 90, SSIM >= 0.985.
Package: ffprobe 1920x1080 (or the native source size, never upscaled), source fps, H.264 High, and video
bitrate >= source bitrate. One failure fails the package; the table is written to the manifest either way.

    python -m motion_ai.quality output/match_highlights_*.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from .media import _run, probe, probe_async

log = logging.getLogger(__name__)

VMAF_MEAN_MIN, VMAF_P5_MIN, SSIM_MIN = 95.0, 90.0, 0.985
MIN_COMPARE_S = 1.0


def _src_args(src: Path) -> list[str]:
    return ["-f", "concat", "-safe", "0", "-i", str(src)] if src.suffix == ".txt" else ["-i", str(src)]


async def ssim_only(clip: Path, src: Path, src_start: float, seconds: float) -> float:
    graph = (f"[0:v]trim=0:{seconds:.3f},setpts=PTS-STARTPTS[d];"
             f"[1:v]trim=start={src_start:.3f}:duration={seconds:.3f},setpts=PTS-STARTPTS[r];[d][r]ssim")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(clip), *_src_args(src), "-filter_complex", graph,
        "-f", "null", "-", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    m = re.search(r"SSIM .*All:([0-9.]+)", err.decode(errors="ignore"))
    return float(m.group(1)) if m else 0.0


async def find_offset(clip: Path, src: Path, t: float, fps: float = 25.0) -> tuple[float, float]:
    """File time = stream time + offset; offset is a whole number of frames in [-0.2, 0.2]. Returns (offset, ssim)."""
    frame = 1.0 / fps
    scores = [(round(k * frame, 3), await ssim_only(clip, src, max(0.0, t + k * frame), 1.0)) for k in range(-5, 6)]
    return max(scores, key=lambda x: x[1])


async def compare(clip: Path, src_list: Path, src_start: float, seconds: float, threads: int = 12) -> dict:
    """VMAF + SSIM of clip[0:seconds] against source[src_start : src_start+seconds]."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "vmaf.json"
        graph = (f"[0:v]trim=0:{seconds:.3f},setpts=PTS-STARTPTS,split[d1][d2];"
                 f"[1:v]trim=start={src_start:.3f}:end={src_start + seconds:.3f},setpts=PTS-STARTPTS,split[r1][r2];"
                 f"[d1][r1]libvmaf=log_fmt=json:log_path={log_path}:n_threads={threads};"
                 f"[d2][r2]ssim")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(clip), *_src_args(src_list),
            "-filter_complex", graph, "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        text = err.decode(errors="ignore")
        if proc.returncode != 0 or not log_path.exists():
            raise RuntimeError(text[-500:])
        frames = json.loads(log_path.read_text())["frames"]
        v = np.array([f["metrics"]["vmaf"] for f in frames])
        m = re.search(r"SSIM .*All:([0-9.]+)", text)
        return {"vmaf_mean": round(float(v.mean()), 2), "vmaf_p5": round(float(np.percentile(v, 5)), 2),
                "ssim": round(float(m.group(1)), 4) if m else None, "frames": len(frames),
                "compared_s": round(seconds, 2)}


def segments_list(segments: list[Path], tmp: Path) -> Path:
    lst = tmp / "src.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    return lst


async def gate(manifest_path: Path, clip_dir: Path, covering, source: dict) -> dict:
    """covering(t0, t1) -> list of segments (with .path/.start) for the source trim of each clip."""
    man = json.loads(manifest_path.read_text())
    video = manifest_path.with_suffix(".mp4")
    rows, failures = [], []
    src_file = Path(source.get("path", "")) if source.get("path") else None
    use_file = src_file is not None and src_file.exists() and src_file.suffix != ".ts"
    offset, reference = 0.0, "ingest segments"
    with tempfile.TemporaryDirectory() as tmp:
        if use_file and man["clips"]:
            c0 = man["clips"][0]
            offset, s0 = await find_offset(clip_dir / c0["clip_file"], src_file, c0["segments"][0][0],
                                           source.get("fps") or 25.0)
            reference = f"original file {src_file} (offset {offset:+.2f}s, probe SSIM {s0:.4f})"
            log.info("Quality reference: %s", reference)
        for c in man["clips"]:
            a, b = c["segments"][0]
            seconds = min(b - a, c.get("caption_offset_in_clip") or (b - a))
            row = {"clip": c["clip_file"], "type": c["type"], "caption": c["caption"]}
            if seconds < MIN_COMPARE_S:
                row["skipped"] = "caption at start"
                rows.append(row)
                continue
            try:
                if use_file:
                    row |= await compare(clip_dir / c["clip_file"], src_file, a + offset, seconds)
                else:
                    segs = covering(a, a + seconds)
                    lst = segments_list([s.path for s in segs], Path(tmp))
                    row |= await compare(clip_dir / c["clip_file"], lst, a - segs[0].start, seconds)
            except RuntimeError as exc:
                row["error"] = str(exc)[-200:]
                failures.append(f"{c['clip_file']}: compare failed")
                rows.append(row)
                continue
            ok = (row["vmaf_mean"] >= VMAF_MEAN_MIN and row["vmaf_p5"] >= VMAF_P5_MIN
                  and (row["ssim"] or 0) >= SSIM_MIN)
            row["pass"] = ok
            if not ok:
                failures.append(f"{c['clip_file']}: vmaf {row['vmaf_mean']}/{row['vmaf_p5']} ssim {row['ssim']}")
            rows.append(row)
            log.info("Quality %s: VMAF %.2f (p5 %.2f) SSIM %s", c["clip_file"], row["vmaf_mean"], row["vmaf_p5"], row["ssim"])

    out = await probe_async(video, cache=False)
    exp_w = min(source.get("width", 1920), 1920)
    exp_h = min(source.get("height", 1080), 1080)
    checks = {
        "resolution": f"{out.get('width')}x{out.get('height')}",
        "resolution_ok": (out.get("width"), out.get("height")) == (exp_w, exp_h),
        "fps": out.get("fps"), "fps_ok": abs((out.get("fps") or 0) - (source.get("fps") or 25)) < 0.05,
        "profile": out.get("profile"), "profile_ok": out.get("profile") == "High",
        "video_kbps": out.get("video_kbps"), "source_video_kbps": source.get("video_kbps"),
        "bitrate_ok": (out.get("video_kbps") or 0) >= (source.get("video_kbps") or 0),
    }
    for k in ("resolution_ok", "fps_ok", "profile_ok", "bitrate_ok"):
        if not checks[k]:
            failures.append(f"package {k.removesuffix('_ok')}: {checks}")
    result = {"pass": not failures, "reference": reference, "thresholds": {"vmaf_mean": VMAF_MEAN_MIN, "vmaf_p5": VMAF_P5_MIN,
                                                    "ssim": SSIM_MIN, "bitrate": ">= source"},
              "package": checks, "clips": rows, "failures": failures}
    man["quality_gate"] = result
    if failures:
        man["status"] = "hold_for_review"
        man.setdefault("package_holds", []).append("quality_gate_failed")
    manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str))
    log.info("QUALITY GATE %s (%d clips, %d failures)", "PASS" if not failures else "FAIL", len(rows), len(failures))
    return result


def main() -> None:
    """Re-run the gate on an existing package (uses <WORK_DIR>/segments.jsonl for the source)."""
    from .config import Settings
    from .ingest.buffer import SegmentBuffer
    from .recorder import load_segments

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    settings = Settings()
    buf = SegmentBuffer()
    buf.segments = load_segments(settings.work_dir)
    man = Path(sys.argv[1])
    res = asyncio.run(gate(man, settings.work_dir / "clips", buf.covering, probe(settings.srt_url)))
    print(json.dumps({k: res[k] for k in ("pass", "package", "failures")}, indent=1))


if __name__ == "__main__":
    main()
