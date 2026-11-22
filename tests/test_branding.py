"""Branding template: loading, validation, fallbacks, and that the example template reproduces today's ffmpeg args."""
import asyncio
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from motion_ai import branding, media
from motion_ai.branding import DEFAULT_FONT, DEFAULTS, TemplateError, load_template
from motion_ai.highlights.curator import CARD_SECONDS

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def _write(tmp_path: Path, data, name="template.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return p


@pytest.fixture
def example():
    return load_template(branding.EXAMPLE_PATH)


@pytest.fixture
def logo_png(tmp_path):
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not installed")
    p = tmp_path / "brand" / "logo.png"
    p.parent.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red@0.5:s=64x32,format=rgba",
                    "-frames:v", "1", str(p)], check=True)
    return p


# --- loading + fallbacks -----------------------------------------------------------------------------------
def test_example_template_equals_builtin_defaults(example):
    data = dict(example.data, name=DEFAULTS["name"])
    assert data == DEFAULTS
    assert example.font == DEFAULT_FONT and example.logo_file is None
    assert example.logo("card") is None and example.logo("clip_bug") is None


def test_card_seconds_match_curator_budget(example):
    assert example.card_seconds == CARD_SECONDS        # the curator budgets with the same lengths


def test_template_json_wins_over_example(tmp_path, monkeypatch):
    real = _write(tmp_path, {"name": "brand"})
    ex = _write(tmp_path, {"name": "example"}, "template.example.json")
    monkeypatch.setattr(branding, "TEMPLATE_PATH", real)
    monkeypatch.setattr(branding, "EXAMPLE_PATH", ex)
    assert load_template().data["name"] == "brand"
    real.unlink()
    assert load_template().data["name"] == "example"
    ex.unlink()
    assert load_template().data == DEFAULTS


def test_missing_keys_fall_back_to_defaults(tmp_path):
    t = load_template(_write(tmp_path, {"colors": {"background": "#112233"}, "font": {"sizes": {"caption": 60}},
                                        "_comment": "ignored"}))
    assert t.color("background") == "0x112233"
    assert t.colors["text"] == "white"
    assert t.sizes["caption"] == 60 and t.sizes["headline"] == 96
    assert t.cards == DEFAULTS["cards"]


def test_relative_font_and_logo_paths_resolve_next_to_template(tmp_path, logo_png):
    font = tmp_path / "brand" / "Brand.ttf"
    font.write_bytes(Path(DEFAULT_FONT).read_bytes() if Path(DEFAULT_FONT).exists() else b"x")
    t = load_template(_write(tmp_path, {"font": {"file": "brand/Brand.ttf"}, "logo": {"file": "brand/logo.png"}}))
    assert t.font == str(font) and t.logo_file == str(logo_png)
    assert t.logo("card")["position"] == "top-right"
    assert t.logo("clip_bug") is None                    # corner bug is opt-in


# --- validation --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("data, msg", [
    ({"colors": {"accent": "#12345"}}, "colors.accent must be"),
    ({"colors": {"text": "chartreuse-ish"}}, "colors.text must be"),
    ({"colours": {}}, "unknown key 'colours'"),
    ({"caption": {"y_pos": 0.5}}, "unknown key caption.'y_pos'"),
    ({"font": {"file": "nope.ttf"}}, "font.file: file not found"),
    ({"logo": {"file": "missing.png"}}, "logo.file: file not found"),
    ({"logo": {"card": {"position": "middle"}}}, "logo.card.position must be one of"),
    ({"logo": {"clip_bug": {"opacity": 0}}}, "logo.clip_bug.opacity must be > 0"),
    ({"font": {"sizes": {"headline": "big"}}}, "font.sizes.headline must be a number"),
    ({"cards": {"seconds": {"title": 0}}}, "cards.seconds.title must be > 0"),
    ({"cards": {"seconds": {"intro": 2}}}, "unknown key cards.seconds.'intro'"),
    ({"caption": {"y": 1.5}}, "caption.y must be"),
    ({"caption": {"box": "yes"}}, "caption.box must be true or false"),
    ({"colors": "blue"}, "colors must be an object"),
])
def test_bad_templates_raise_clear_errors(tmp_path, data, msg):
    p = _write(tmp_path, data)
    with pytest.raises(TemplateError, match=None) as exc:
        load_template(p)
    assert msg in str(exc.value) and str(p) in str(exc.value)


def test_invalid_json_and_missing_file(tmp_path):
    with pytest.raises(TemplateError, match="not valid JSON"):
        load_template(_write(tmp_path, "{nope"))
    with pytest.raises(TemplateError, match="template not found"):
        load_template(tmp_path / "absent.json")


def test_logo_must_be_png_and_font_ttf(tmp_path):
    (tmp_path / "logo.jpg").write_bytes(b"x")
    (tmp_path / "font.woff").write_bytes(b"x")
    with pytest.raises(TemplateError, match="PNG"):
        load_template(_write(tmp_path, {"logo": {"file": "logo.jpg"}}))
    with pytest.raises(TemplateError, match=".ttf or .otf"):
        load_template(_write(tmp_path, {"font": {"file": "font.woff"}}))


# --- ffmpeg args from the example template == today's hard-coded args --------------------------------------
TODAY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def test_card_args_from_example_equal_todays(example):
    lines, seconds, (w, h), fps = [("A v B", 64), ("MATCH HIGHLIGHTS", 96)], 3.0, (1920, 1080), 25.0
    tfs = ["/x/title.l0.txt", "/x/title.l1.txt"]
    # today's render_card body, verbatim
    total_h = sum(fs * 1.5 for _, fs in lines)
    y = (h - total_h) / 2
    draws = []
    for tf, (text, fs) in zip(tfs, lines):
        draws.append(f"drawtext=fontfile={TODAY_FONT}:textfile='{tf}':fontsize={fs}:fontcolor=white:"
                     f"x=(w-text_w)/2:y={y:.0f}")
        y += fs * 1.5
    vf = ",".join(draws + [f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(0.0, seconds - 0.3):.2f}:d=0.3"])
    color = f"color=c=0x0b1d3a:s={w}x{h}:r={fps:g}:d={seconds}"

    vin, vargs = media.card_video_args(example, lines, seconds, (w, h), fps, tfs)
    assert vin == ["-f", "lavfi", "-i", color]
    assert vargs == ["-vf", vf, "-map", "0:v"]


def test_caption_from_example_equals_todays(example):
    today = (f"drawtext=fontfile={TODAY_FONT}:textfile='/x/c.txt':fontsize=72:fontcolor=white:"
             f"borderw=4:bordercolor=black:x=(w-text_w)/2:y=h*0.72:enable='gte(t,{1.234:.2f})'")
    assert branding.caption_draw(example, "/x/c.txt", 1.234) == today


def test_card_texts_sizes_from_example_equal_todays(example):
    from types import SimpleNamespace
    from motion_ai.highlights.renderer import card_texts
    facts = SimpleNamespace(target_matches_innings1=True, innings1_runs=179, target=180, result_score=(181, 5))
    match = {"teams": {"A": [], "B": []}, "batting_first": "A"}
    c = card_texts(facts, match, example.sizes)
    assert c["title"] == [("A v B", 64), ("MATCH HIGHLIGHTS", 96)]
    assert c["break"] == [("INNINGS BREAK", 96), ("A  179", 72)]
    assert c["result"] == [("B WIN", 96), ("BY 5 WICKETS", 80), ("181/5", 64)]
    assert c["outro"] == [("MATCH HIGHLIGHTS", 64), ("A v B", 48)]
    big = card_texts(facts, match, dict(example.sizes, headline=120))
    assert big["break"][0] == ("INNINGS BREAK", 120)


def _capture_render_clip(monkeypatch, tpl, tmp_path):
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
    asyncio.run(media.render_clip([tmp_path / "s.ts"], 0.0, [(1.0, 3.0)], tmp_path / "c.mp4", "SIX", 0.5,
                                  audio_rate=48000, template=tpl))
    return cmds[0]


def test_render_clip_caption_unchanged_and_bug_optional(monkeypatch, tmp_path, example, logo_png):
    cmd = _capture_render_clip(monkeypatch, example, tmp_path)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.endswith(f"[vc]null,drawtext=fontfile={TODAY_FONT}:textfile='{tmp_path / 'c.caption.txt'}':"
                          "fontsize=72:fontcolor=white:borderw=4:bordercolor=black:x=(w-text_w)/2:y=h*0.72:"
                          "enable='gte(t,0.50)'[vout]")
    assert cmd.count("-i") == 1

    d = copy.deepcopy(DEFAULTS)
    d["logo"].update(file=str(logo_png), clip_bug={"enabled": True, "position": "bottom-left", "width": 100,
                                                   "opacity": 0.5})
    tpl = branding.Template(d)
    cmd = _capture_render_clip(monkeypatch, tpl, tmp_path)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert cmd[cmd.index("-loop"):cmd.index("-loop") + 4] == ["-loop", "1", "-i", str(logo_png)]
    assert "[1:v]scale=100:-1,format=rgba,colorchannelmixer=aa=0.5[bug]" in graph
    assert "[vcap][bug]overlay=x=64:y=H-h-48:shortest=1[vout]" in graph


def test_overlay_positions():
    m = {"x": 10, "y": 20}
    assert branding.overlay_xy("top-left", m) == ("10", "20")
    assert branding.overlay_xy("top-right", m) == ("W-w-10", "20")
    assert branding.overlay_xy("bottom-center", m) == ("(W-w)/2", "H-h-20")


def test_styling_options_change_filters(tmp_path):
    t = load_template(_write(tmp_path, {"colors": {"text": "#fafafa", "accent": "#ff000080"},
                                        "cards": {"accent_bar_height": 12},
                                        "caption": {"box": True, "border_width": 0}}))
    draws = branding.card_draws(t, [("X", 50)], (640, 360), ["/x/a.txt"])
    assert "fontcolor=0xfafafa" in draws[0]
    assert draws[-1] == "drawbox=x=0:y=ih-12:w=iw:h=12:color=0xff000080:t=fill"
    cap = branding.caption_draw(t, "/x/c.txt", 0)
    assert ":borderw" not in cap and "box=1:boxcolor=0x00000099:boxborderw=16:" in cap


@needs_ffmpeg
def test_card_with_logo_renders(tmp_path, logo_png):
    d = copy.deepcopy(DEFAULTS)
    d["logo"]["file"] = str(logo_png)
    d["cards"]["accent_bar_height"] = 6
    out = tmp_path / "card.mp4"
    asyncio.run(media.render_card([("TITLE", 20)], 1.0, out, (160, 120), 25, audio_rate=48000,
                                  template=branding.Template(d)))
    info = media.probe(out)
    assert info["audio_hz"] == 48000 and abs(info["duration"] - 1.0) < 0.1
