"""FFmpeg helpers: decode audio / frames from a segment, cut clips."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .branding import (DEFAULT_FONT, active_template, caption_draw, card_background, card_draws, card_fade,
                       logo_chain, overlay_xy)

log = logging.getLogger(__name__)


async def _run(cmd: list[str]) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err.decode(errors='ignore')[-500:]}")
    return out


async def read_pcm(path: Path, sample_rate: int = 16000):
    """Mono float32 PCM for a segment (empty array if no audio)."""
    import numpy as np

    out = await _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ])
    return np.frombuffer(out, dtype=np.float32)


async def read_frames(path: Path, fps: float, width: int, height: int):
    """List of (offset_seconds, HxWx3 uint8 RGB frame) sampled at `fps`."""
    import numpy as np

    out = await _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    frame_bytes = width * height * 3
    n = len(out) // frame_bytes
    return [
        (i / fps, np.frombuffer(out, np.uint8, frame_bytes, i * frame_bytes).reshape(height, width, 3))
        for i in range(n)
    ]


# ---------------------------------------------------------------- output quality (panel spec fm/guidance_v1 §2)
OUT_MAX_W, OUT_MAX_H = 1920, 1080
# Audio is encoded at the SOURCE rate (no resample = no quality loss). The rate is pinned once per process from
# the first segment probed (source_audio_rate) and used for every clip AND card, so the final -c copy join works
# even if the feed's rate changes mid-match (the AAC encoder then resamples that stretch). IPL feeds are usually
# 48 kHz; the demo recording is 44.1 kHz. Unknown -> 48 kHz.
DEFAULT_AUDIO_RATE = 48000
AUDIO_RATE = DEFAULT_AUDIO_RATE   # fallback only; kept for old imports


def audio_opts(rate: int | None = None) -> list[str]:
    return ["-c:a", "aac", "-b:a", "192k", "-ar", str(rate or DEFAULT_AUDIO_RATE), "-ac", "2"]


AUDIO_OPTS = audio_opts()   # fallback only; clips/cards use audio_opts(source rate)


def video_opts(encoder: str = "libx264") -> list[str]:
    """Identical for every clip and card, so the package joins with -c copy (one encode in total)."""
    if encoder != "libx264":
        return ["-c:v", encoder, "-preset", "p6", "-cq", "17", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-level", "4.1",
            "-pix_fmt", "yuv420p", "-g", "50", "-bf", "3",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def scale_filter(src_w: int, src_h: int) -> str | None:
    """None when the source is already <= 1920x1080 (never upscale); else downscale with lanczos."""
    if src_w <= OUT_MAX_W and src_h <= OUT_MAX_H:
        return None
    return f"scale={OUT_MAX_W}:{OUT_MAX_H}:flags=lanczos"


def out_size(src_w: int, src_h: int) -> tuple[int, int]:
    return (src_w, src_h) if scale_filter(src_w, src_h) is None else (OUT_MAX_W, OUT_MAX_H)


def parse_probe(err: str, path: Path | str) -> dict:
    """width, height, fps, video kb/s, profile, codec, audio rate, first video PTS from `ffmpeg -i` stderr."""
    import re as _re

    info: dict = {"path": str(path)}
    line = next((ln for ln in err.splitlines() if "Video:" in ln), "")
    v = _re.search(r"Video: (\w+)(?: \(([^)]*)\))?.*?, (\d{2,5})x(\d{2,5})", line)
    if v:
        kb = _re.search(r", (\d+) kb/s", line)
        fps = _re.search(r", ([\d.]+) tbr", line) or _re.search(r", ([\d.]+) fps", line)   # tbr = nominal rate
        info |= {"codec": v.group(1), "profile": v.group(2), "width": int(v.group(3)), "height": int(v.group(4)),
                 "video_kbps": int(kb.group(1)) if kb else None, "fps": float(fps.group(1)) if fps else None}
    vs = _re.search(r"start (-?[\d.]+)", line)
    if vs:
        info["video_start"] = float(vs.group(1))
    a = _re.search(r"Audio: \w+.*?, (\d+) Hz", err)
    if a:
        info["audio_hz"] = int(a.group(1))
    d = _re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if d:
        info["duration"] = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    b = _re.search(r"Duration:.*?bitrate: (\d+) kb/s", err)
    if b:
        info["total_kbps"] = int(b.group(1))
    return info


def probe(path: Path | str) -> dict:
    """Blocking probe (CLI / tests / sync code only). Async code uses probe_async."""
    import subprocess

    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    return parse_probe(err, path)


# Probe cache per source path: segments and the source file never change once written, and probing every
# segment on every clip/card/quality call was a blocking subprocess inside the event loop (review issue).
_PROBE_CACHE: dict[str, dict] = {}
_PROBE_CACHE_MAX = 50_000          # ~28 h of 2 s segments; cleared (not LRU) beyond that
_SEG_INFO: dict[str, tuple[float | None, float | None]] = {}   # path -> (first video PTS, stream duration)
_SOURCE_AUDIO_RATE: int | None = None


def _key(path: Path | str) -> str:
    return str(Path(path).resolve())


async def probe_async(path: Path | str, cache: bool = True) -> dict:
    """Non-blocking probe; cached per path (pass cache=False for files that are rewritten, e.g. the package)."""
    k = _key(path)
    if cache and k in _PROBE_CACHE:
        return _PROBE_CACHE[k]
    proc = await asyncio.create_subprocess_exec("ffmpeg", "-hide_banner", "-i", str(path),
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    info = parse_probe(err.decode(errors="ignore"), path)
    if cache:
        if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
            _PROBE_CACHE.clear()
        _PROBE_CACHE[k] = info
    return info


def remember_segment(path: Path | str, pts: float | None, duration: float | None) -> None:
    """Ingest / load_segments register each segment's first video PTS and stream duration (continuity check)."""
    _SEG_INFO[_key(path)] = (pts, duration)


async def segment_pts(path: Path | str) -> float | None:
    """First video PTS: the value recorded at ingest if any, else probed (cached)."""
    pts = _SEG_INFO.get(_key(path), (None, None))[0]
    return pts if pts is not None else (await probe_async(path)).get("video_start")


async def source_audio_rate(path: Path | str | None = None) -> int:
    """The package audio rate: the source rate from the first file probed, pinned for the process."""
    global _SOURCE_AUDIO_RATE
    if _SOURCE_AUDIO_RATE is None and path is not None:
        hz = (await probe_async(path)).get("audio_hz")
        if hz:
            _SOURCE_AUDIO_RATE = int(hz)
            log.info("Package audio rate: %d Hz (source %s)", hz, Path(path).name)
    return _SOURCE_AUDIO_RATE or DEFAULT_AUDIO_RATE


def reset_caches() -> None:
    """Tests / a new match in the same process."""
    global _SOURCE_AUDIO_RATE
    _PROBE_CACHE.clear()
    _SEG_INFO.clear()
    _SOURCE_AUDIO_RATE = None


# True-peak safety on every clip and card: the broadcast mix hits 0.0 dBFS and the AAC encode overshoots on top
# of that. A look-ahead limiter (no auto-gain, delay compensated so A/V sync is kept) runs at 4x the sample rate
# (oversampled = inter-sample peaks are caught, i.e. a true-peak limiter) with a ceiling below -1 dBTP that leaves
# room for the AAC overshoot. It only touches the rare peaks (fast attack, 50 ms release = no audible pumping);
# programme loudness is unchanged. Output rate / channels / codec stay the same so the -c copy join still works.
TRUE_PEAK_DBTP = -1.0              # package target (measured after the AAC encode, ebur128 true peak)
LIMITER_CEILING_DB = -3.0          # limiter ceiling at 4x oversampling; margin for the AAC encode (+1.3 dBTP seen at -2)
LIMITER_OVERSAMPLE = 4


def peak_limiter(rate: int = DEFAULT_AUDIO_RATE) -> str:
    """Audio filter chain: oversampled look-ahead true-peak limiter, back to `rate`, stereo fltp."""
    ceiling = 10 ** (LIMITER_CEILING_DB / 20)
    return (f"aresample={rate * LIMITER_OVERSAMPLE},"
            f"alimiter=limit={ceiling:.4f}:attack=1:release=50:level=0:latency=1,"
            f"aresample={rate},aformat=sample_fmts=fltp:channel_layouts=stereo")


# Crowd bed under the text cards: a stretch of this match's own feed, band-limited to 80-400 Hz (commentary
# becomes an unintelligible murmur, the crowd roar stays), loudness-normalised to -27 LUFS and faded at both ends
# so the -c copy joins don't click. Measured on the demo match: ~-31 dB mean (volumedetect) vs clips ~-23 dB, i.e.
# clearly present but well under the programme (was -36 LUFS = ~-39 dB mean, which read as dead air).
CROWD_BED_LUFS = -27
CROWD_BED_TP = -3
CROWD_BED_FADE = 0.4


def crowd_bed_filter(start: float, seconds: float, rate: int = DEFAULT_AUDIO_RATE) -> str:
    """-af chain: [start, start + seconds] of the input -> soft crowd bed exactly `seconds` long."""
    fade = min(CROWD_BED_FADE, seconds / 4)
    return (f"atrim=start={start:.3f}:end={start + seconds:.3f},asetpts=PTS-STARTPTS,"
            "highpass=f=80,lowpass=f=400,lowpass=f=400,"
            f"loudnorm=I={CROWD_BED_LUFS}:TP={CROWD_BED_TP}:LRA=5,aresample={rate},"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"apad,atrim=end={seconds:.3f},"
            f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={seconds - fade:.3f}:d={fade:.2f},"
            + peak_limiter(rate))


def card_video_args(tpl, lines: list[tuple[str, int]], seconds: float, size: tuple[int, int], fps: float,
                    textfiles: list) -> tuple[list[str], list[str]]:
    """(input args, filter + map args) for a card's picture. Input 0 = background, 1 = audio, 2 = logo (if any)."""
    vin = ["-f", "lavfi", "-i", card_background(tpl, size, fps, seconds)]
    draws = card_draws(tpl, lines, size, textfiles)
    spec = tpl.logo("card")
    if not spec:
        return vin, ["-vf", ",".join(draws + [card_fade(tpl, seconds)]), "-map", "0:v"]
    x, y = overlay_xy(spec["position"], tpl.margins)
    graph = (f"[0:v]{','.join(draws)}[base];{logo_chain(spec, '2:v', 'lg')};"
             f"[base][lg]overlay=x={x}:y={y}:shortest=1,{card_fade(tpl, seconds)}[vout]")
    return vin, ["-filter_complex", graph, "-map", "[vout]"]


async def render_card(lines: list[tuple[str, int]], seconds: float, out: Path, size: tuple[int, int],
                      fps: float, encoder: str = "libx264",
                      crowd: tuple[list[Path], float, float] | None = None,
                      audio_rate: int | None = None, template=None) -> Path:
    """Full-screen text card (title / innings / result), same encode as the clips.

    Audio: a soft crowd bed from the match feed when `crowd` = (segments, base, t0) is given (segments cover
    stream time [t0, t0 + seconds], base = stream time of segments[0]); silence otherwise or if reading it fails.
    `audio_rate`: the package rate (default: source_audio_rate, pinned from the first segment) so the -c copy
    join with the clips works; the crowd bed and the silent anullsrc both use it.
    `template`: branding (colours, font, logo; default: the active config/template[.example].json)."""
    tpl = template or active_template()
    out.parent.mkdir(parents=True, exist_ok=True)
    tfs = []
    for i, (text, _) in enumerate(lines):
        tf = out.with_suffix(f".l{i}.txt")
        tf.write_text(text, encoding="utf-8")
        tfs.append(tf)
    video_in, video_args = card_video_args(tpl, lines, seconds, size, fps, tfs)

    rate = audio_rate or await source_audio_rate(crowd[0][0] if crowd and crowd[0] else None)

    logo_in = ["-loop", "1", "-i", tpl.logo_file] if tpl.logo("card") else []

    async def encode(audio_in: list[str], af: list[str]) -> None:
        await _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *video_in, *audio_in, *logo_in,
            *video_args, *af, "-t", f"{seconds}", "-map", "1:a",
            *video_opts(encoder), *audio_opts(rate), "-movflags", "+faststart", str(out)])

    silence = ["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=stereo"]
    lst = None
    try:
        if crowd and crowd[0]:
            segments, base, t0 = crowd
            try:
                in_args, t_in, lst = await continuous_input(segments, out.parent, out.stem)
                await encode(in_args, ["-af", crowd_bed_filter(t_in + t0 - base, seconds, rate)])
                return out
            except (RuntimeError, OSError) as exc:
                log.warning("Card %s: crowd bed unavailable (%s) — silent card", out.name, str(exc)[-200:])
        else:
            log.warning("Card %s: no source audio for a crowd bed — silent card", out.name)
        await encode(silence, [])
    finally:
        if lst is not None:
            lst.unlink(missing_ok=True)
        for i in range(len(lines)):
            out.with_suffix(f".l{i}.txt").unlink(missing_ok=True)
    return out


async def cut_clip(segments: list[Path], seg_start: float, t0: float, t1: float,
                   out: Path, encoder: str = "libx264") -> Path:
    """Cut [t0, t1] (stream seconds) out of consecutive segments starting at seg_start.

    All clips are encoded with identical settings so the final package can be concatenated without re-encoding.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    concat_list = out.with_suffix(".txt")
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    src = await probe_async(segments[0])
    sf = scale_filter(src.get("width", 1920), src.get("height", 1080))
    rate = await source_audio_rate(segments[0])
    await _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-ss", f"{max(0.0, t0 - seg_start):.3f}", "-t", f"{t1 - t0:.3f}",
        *(["-vf", sf] if sf else []), "-af", peak_limiter(rate), *video_opts(encoder), *audio_opts(rate),
        "-movflags", "+faststart", str(out),
    ])
    concat_list.unlink(missing_ok=True)
    return out


def _concat_list(segments: list[Path], out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / f"_{tag}.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    return lst


async def scene_cuts(segments: list[Path], base: float, t0: float, t1: float, tmp: Path,
                     threshold: float = 6.0) -> list[float]:
    """Camera cuts (stream seconds) in [t0, t1] using ffmpeg scdet. `base` = stream time of segments[0].

    Threshold 6 measured on broadcast footage: in-shot scores <=1.9, real cuts 6.6-34.
    Cuts closer than 1s are merged (first kept)."""
    import re as _re

    lst = _concat_list(segments, tmp, f"cuts_{int(t0 * 10)}")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-nostats", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-t", f"{t1 - base:.3f}", "-an", "-vf", f"scale=640:-2,scdet=threshold={threshold}",
        "-f", "null", "-", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    lst.unlink(missing_ok=True)
    times = [base + float(m) for m in _re.findall(r"lavfi\.scd\.time:\s*([0-9.]+)", err.decode(errors="ignore"))]
    merged: list[float] = []
    for c in sorted(times):
        if t0 <= c <= t1 and (not merged or c - merged[-1] >= 1.0):
            merged.append(round(c, 2))
    return merged


async def loudness(segments: list[Path], base: float, t0: float, t1: float, tmp: Path,
                   win: float = 0.5, step: float = 0.1) -> list[tuple[float, float]]:
    """(window_start_stream_time, linear RMS) for 0.5s windows on a 0.1s step over [t0, t1]."""
    import numpy as np

    sr = 16000
    lst = _concat_list(segments, tmp, f"loud_{int(t0 * 10)}")
    out = await _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-ss", f"{max(0.0, t0 - base):.3f}", "-t", f"{t1 - t0:.3f}",
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"])
    lst.unlink(missing_ok=True)
    pcm = np.frombuffer(out, dtype=np.float32)
    w, s = int(win * sr), int(step * sr)
    return [(t0 + i / sr, float(np.sqrt(np.mean(pcm[i:i + w] ** 2)) + 1e-9))
            for i in range(0, max(0, len(pcm) - w + 1), s)]


FONT = DEFAULT_FONT   # kept for callers; styling now comes from motion_ai.branding


def video_start_pts(path: Path) -> float | None:
    """First video PTS of a file, blocking (CLI/tests). Async code uses segment_pts / probe_async."""
    return probe(path).get("video_start")


MAX_SEGMENT_S = 30.0       # without a recorded duration, a PTS step outside (0, 30] s is a splice
PTS_TOLERANCE_S = 0.5      # recording: CSV duration vs PTS step differ by <= 1 frame (seg 0: 4.68 vs 4.64)


def pts_continuity(pts: list[float | None], durations: list[float | None],
                   tol: float = PTS_TOLERANCE_S) -> tuple[bool, str]:
    """Can these consecutive segments be byte-joined (concat protocol + -copyts)?

    Segment i continues segment i-1 when its first video PTS = PTS[i-1] + duration[i-1] (+-tol). An ad-break
    splice, encoder restart or reconnect resets / jumps the PTS; per-segment reset timestamps (every segment
    starting ~1.4 s) show as a ~0 step. Any of those -> False (use the concat demuxer, which re-times each file).
    Replaces the old "v0 > 0.5 s and same r00_ prefix" heuristic (review issue 4)."""
    if not pts or any(p is None for p in pts):
        return False, "first video PTS unknown"
    for i in range(1, len(pts)):
        step = pts[i] - pts[i - 1]
        dur = durations[i - 1] if i - 1 < len(durations) else None
        if dur is not None:
            if abs(step - dur) > tol:
                return False, f"PTS discontinuity before segment {i} (step {step:+.3f}s, duration {dur:.3f}s)"
        elif not 0 < step <= MAX_SEGMENT_S:
            return False, f"PTS discontinuity before segment {i} (step {step:+.3f}s)"
    return True, "continuous"


def fill_unknown_pts(pts: list[float | None], durations: list[float | None]) -> list[float | None]:
    """A sliver segment (e.g. 0.16 s cut between two keyframes) has no SPS/PPS of its own, so `ffmpeg -i` cannot
    report its first PTS. Inside a byte-join it decodes fine with the previous segment's parameter sets, so infer
    its PTS from a known neighbour: prev PTS + prev duration (or next PTS - own duration). Never invents a value
    without a known neighbour and a known duration; pts_continuity still checks every step."""
    out = list(pts)
    for i in range(1, len(out)):                       # forward from known predecessors
        if out[i] is None and out[i - 1] is not None and durations[i - 1] is not None:
            out[i] = round(out[i - 1] + durations[i - 1], 6)
    for i in range(len(out) - 2, -1, -1):              # backward for a leading unknown
        if out[i] is None and out[i + 1] is not None and durations[i] is not None:
            out[i] = round(out[i + 1] - durations[i], 6)
    return out


async def continuous_input(segments: list[Path], tmp_dir: Path, tag: str) -> tuple[list[str], float, Path | None]:
    """ffmpeg input args + the input time of segments[0]'s start.

    Segments with continuous timestamps (checked with the first video PTS recorded at ingest, see
    pts_continuity) are byte-joined with the concat PROTOCOL and read with -copyts: frame-exact, original A/V
    alignment. (The concat DEMUXER re-times every file from its earliest stream — audio starts ~20 ms before
    video in each segment — which dropped/duplicated a frame at some segment joins.) Anything else (PTS splice
    at an ad break, reconnect, unknown PTS) falls back to the demuxer for this clip, logged."""
    pts = [await segment_pts(p) for p in segments]
    durations = [_SEG_INFO.get(_key(p), (None, None))[1] for p in segments]
    pts = fill_unknown_pts(pts, durations)
    ok, why = pts_continuity(pts, durations)
    if ok:
        return ["-copyts", "-i", "concat:" + "|".join(str(p.resolve()) for p in segments)], pts[0], None
    log.warning("%s: %s — concat demuxer fallback (not frame-exact at that join)", tag, why)
    lst = _concat_list(segments, tmp_dir, tag)
    return ["-f", "concat", "-safe", "0", "-i", str(lst)], 0.0, lst


async def render_clip(segments: list[Path], base: float, ranges: list[tuple[float, float]], out: Path,
                      caption: str | None = None, caption_offset: float = 0.0,
                      encoder: str = "libx264", fade: float = 0.25, audio_rate: int | None = None,
                      template=None) -> Path:
    """One ffmpeg pass: trim each range, bake 0.25s audio fades at every clip edge and join, concat,
    burn in the caption, encode once (no AAC gaps or clicks at joins). Audio at the source rate
    (`audio_rate`, default source_audio_rate pinned from the first segment) — no resample.
    Caption styling / optional corner logo bug come from `template` (default: the active branding template)."""
    tpl = template or active_template()
    out.parent.mkdir(parents=True, exist_ok=True)
    rate = audio_rate or await source_audio_rate(segments[0])
    in_args, t_in, lst = await continuous_input(segments, out.parent, out.stem)
    parts, labels = [], []
    for i, (a, b) in enumerate(ranges):
        sa, sb, d = t_in + a - base, t_in + b - base, b - a
        parts.append(f"[0:v]trim=start={sa:.3f}:end={sb:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={sa:.3f}:end={sb:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:st=0:d={fade},afade=t=out:st={max(0.0, d - fade):.3f}:d={fade}[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(ranges)}:v=1:a=1[vc][acat]")
    parts.append(f"[acat]{peak_limiter(rate)}[ac]")
    src = await probe_async(segments[0])
    sw, sh = src.get("width", 1920), src.get("height", 1080)
    sf = scale_filter(sw, sh)
    vchain = "[vc]" + (sf or "null")
    if caption and tpl.caption.get("enabled", True):
        txt = out.with_suffix(".caption.txt")
        txt.write_text(caption, encoding="utf-8")
        vchain += "," + caption_draw(tpl, txt, caption_offset, height=out_size(sw, sh)[1],
                                     end=sum(b - a for a, b in ranges))
    bug = tpl.logo("clip_bug")
    if bug:
        x, y = overlay_xy(bug["position"], tpl.margins)
        logo_idx = in_args.count("-i")
        in_args = [*in_args, "-loop", "1", "-i", tpl.logo_file]
        parts.append(vchain + "[vcap]")
        parts.append(logo_chain(bug, f"{logo_idx}:v", "bug"))
        parts.append(f"[vcap][bug]overlay=x={x}:y={y}:shortest=1[vout]")
    else:
        parts.append(vchain + "[vout]")
    try:
        await _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *in_args,
            "-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[ac]",
            *video_opts(encoder), *audio_opts(rate),
            "-movflags", "+faststart", str(out)])
    finally:
        if lst is not None:
            lst.unlink(missing_ok=True)
        out.with_suffix(".caption.txt").unlink(missing_ok=True)
    return out


async def concat_clips(clips: list[Path], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in clips))
    await _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
        "-movflags", "+faststart", str(out),
    ])
    lst.unlink(missing_ok=True)
    return out
