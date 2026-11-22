"""Caption look (brand box, upper frame, scaled with frame height, fade) and audio levels (true-peak limiter,
card crowd bed). Fix round 2, issues 6 + 7."""
import asyncio
import json
import shutil
import subprocess

import pytest

from motion_ai import branding, media
from motion_ai.branding import TemplateError, load_template

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def example():
    return load_template(branding.EXAMPLE_PATH)


def test_example_caption_is_a_solid_brand_box_in_the_upper_frame(example):
    c = example.caption
    assert c["box"] and c["border_width"] == 0
    assert example.colors["caption_box"] == example.colors["accent"]       # brand gold box
    assert example.colors["caption_text"] == example.colors["background"]  # brand navy letters
    assert example.colors["caption_box"].startswith("#") and len(example.colors["caption_box"]) == 7  # opaque
    assert example.sizes["caption"] >= 84                                   # larger than before (72)
    # whole box (text top - padding .. text top + font + padding) inside the clean band at 1080p:
    # below the corner logos (~150 px tall... box centred, so only the top edge matters) and far above the
    # broadcaster score bar (y 900-1024)
    top = c["y"] * 1080 - c["box_padding"]
    bottom = c["y"] * 1080 + example.sizes["caption"] + c["box_padding"]
    assert 100 <= top and bottom < 0.5 * 1080
    assert c["fade"] > 0


def test_caption_scales_with_frame_height(example):
    hd = branding.caption_draw(example, "/x/c.txt", 2.0, height=1080)
    sd = branding.caption_draw(example, "/x/c.txt", 2.0, height=720)
    assert "fontsize=88:" in hd and "boxborderw=22|44:" in hd
    assert "fontsize=59:" in sd and "boxborderw=15|29:" in sd
    assert "y=h*0.15:" in sd                                    # position is already a fraction of the height


def test_caption_fade_in_and_out_at_clip_end(example):
    s = branding.caption_draw(example, "/x/c.txt", 9.5, end=25.0)
    assert "alpha='max(0,min(1,min((t-9.50)/0.3,(25.00-t)/0.3)))'" in s
    assert s.endswith("enable='gte(t,9.50)'")                  # timing logic unchanged: from the offset on


@pytest.mark.parametrize("caption, msg", [
    ({"y": 0.85}, "broadcaster score bar"),
    ({"fade": -1}, "caption.fade must be"),
    ({"box_padding_x": "wide"}, "caption.box_padding_x must be a number"),
    ({"ref_height": 10}, "caption.ref_height must be"),
])
def test_bad_caption_keys(tmp_path, caption, msg):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"caption": caption}))
    with pytest.raises(TemplateError) as exc:
        load_template(p)
    assert msg in str(exc.value)


def test_nested_comment_keys_are_ignored(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"caption": {"_comment": "x", "y": 0.2}}))
    assert load_template(p).caption["y"] == 0.2


# --- audio ---------------------------------------------------------------------------------------------------
def test_peak_limiter_is_oversampled_no_autogain_delay_compensated():
    f = media.peak_limiter(44100)
    ceiling = 10 ** (media.LIMITER_CEILING_DB / 20)
    assert f.startswith(f"aresample=176400,alimiter=limit={ceiling:.4f}:")
    assert "level=0" in f and "latency=1" in f
    assert f.endswith("aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo")
    assert media.LIMITER_CEILING_DB < media.TRUE_PEAK_DBTP


def test_crowd_bed_louder_and_limited():
    f = media.crowd_bed_filter(10.0, 3.0, 48000)
    assert f"loudnorm=I={media.CROWD_BED_LUFS}:" in f and -28 <= media.CROWD_BED_LUFS <= -26
    assert f.endswith(media.peak_limiter(48000))


def test_render_clip_limits_programme_audio(monkeypatch, tmp_path):
    cmds = []

    async def fake_run(cmd):
        cmds.append(cmd)
        return b""

    async def fake_input(segments, tmp_dir, tag):
        return ["-i", "src.ts"], 0.0, None

    async def fake_probe(path, cache=True):
        return {"width": 1920, "height": 1080}

    monkeypatch.setattr(media, "_run", fake_run)
    monkeypatch.setattr(media, "continuous_input", fake_input)
    monkeypatch.setattr(media, "probe_async", fake_probe)
    import copy
    from motion_ai import branding
    d = copy.deepcopy(branding.DEFAULTS)
    d["caption"]["enabled"] = True
    asyncio.run(media.render_clip([tmp_path / "s.ts"], 0.0, [(1.0, 3.0), (5.0, 6.0)], tmp_path / "c.mp4", "SIX",
                                  0.5, audio_rate=48000, template=branding.Template(d)))
    graph = cmds[0][cmds[0].index("-filter_complex") + 1]
    assert f"concat=n=2:v=1:a=1[vc][acat];[acat]{media.peak_limiter(48000)}[ac]" in graph
    assert "(3.00-t)/0.3" in graph                               # caption fades out at the clip end (2 + 1 s)
    assert cmds[0][cmds[0].index("-map") + 3] == "[ac]"


@needs_ffmpeg
@pytest.mark.parametrize("src, gain", [("sine=f=997:r=48000:d=2", "24dB"),          # 2x full scale
                                       ("anoisesrc=a=1:c=pink:seed=7:d=2:r=48000", "3dB")])
def test_limiter_keeps_true_peak_under_minus_1_dbtp(tmp_path, src, gain):
    """Signals driven well over full scale through the limiter + AAC encode stay <= -1 dBTP."""
    out = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src,
                    "-af", f"volume={gain},aformat=channel_layouts=stereo," + media.peak_limiter(48000),
                    *media.audio_opts(48000), str(out)], check=True)
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(out), "-af",
                          "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    peak = float(err.split("True peak:")[1].split("Peak:")[1].split()[0])
    assert peak <= media.TRUE_PEAK_DBTP


def test_caption_off_by_default_burns_nothing(monkeypatch, tmp_path):
    """caption.enabled false (current default): no drawtext; the label stays in the .srt / manifest."""
    cmds = []

    async def fake_run(cmd):
        cmds.append(cmd)
        return b""

    async def fake_input(segments, tmp_dir, tag):
        return ["-i", "src.ts"], 0.0, None

    async def fake_probe(path, cache=True):
        return {"width": 1920, "height": 1080}

    monkeypatch.setattr(media, "_run", fake_run)
    monkeypatch.setattr(media, "continuous_input", fake_input)
    monkeypatch.setattr(media, "probe_async", fake_probe)
    from motion_ai import branding
    asyncio.run(media.render_clip([tmp_path / "s.ts"], 0.0, [(1.0, 3.0)], tmp_path / "c.mp4", "SIX", 0.5,
                                  audio_rate=48000, template=branding.Template(branding.DEFAULTS)))
    graph = cmds[0][cmds[0].index("-filter_complex") + 1]
    assert "drawtext" not in graph
