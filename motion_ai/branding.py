"""Branding template: card / caption / logo styling read from config/template.json.

config/template.json (git-ignored, the licensed brand files) wins; otherwise config/template.example.json (a
neutral placeholder). Every key is optional: a missing key falls back to DEFAULTS, which reproduce the look the
renderer had before templates existed. The template is validated on load (TemplateError says what is wrong).
Relative file paths (font, logo) are resolved against the template file's own folder.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
TEMPLATE_PATH = CONFIG_DIR / "template.json"
EXAMPLE_PATH = CONFIG_DIR / "template.example.json"
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center")
CARD_NAMES = ("title", "innings1", "break", "innings2", "result", "outro")

# Today's values. Keys here are the full schema: anything else in a template is rejected as a typo.
DEFAULTS: dict = {
    "name": "default",
    "colors": {
        "background": "#0b1d3a",     # card background
        "accent": "#f2b705",         # card accent bar (only drawn when cards.accent_bar_height > 0)
        "text": "white",             # card text
        "caption_text": "#0b1d3a",   # caption letters (brand navy on the brand-gold box)
        "caption_border": "black",   # outline around caption letters (only when caption.border_width > 0)
        "caption_box": "#f2b705",    # solid box behind the caption (only when caption.box is true)
    },
    "font": {
        "file": None,                # null = DEFAULT_FONT
        "sizes": {"headline": 96, "subhead": 80, "stat": 72, "body": 64, "small": 48, "caption": 88},
    },
    "logo": {
        "file": None,                # PNG with alpha; null = no logo anywhere
        "card": {"enabled": True, "position": "top-right", "width": 220, "opacity": 1.0},
        "clip_bug": {"enabled": False, "position": "top-right", "width": 120, "opacity": 0.8},
    },
    "cards": {
        "seconds": {"title": 3.0, "innings1": 2.0, "break": 3.0, "innings2": 2.0, "result": 3.0, "outro": 2.0},
        "line_spacing": 1.5,
        "fade": 0.3,
        "accent_bar_height": 0,      # px, bottom of the card; 0 = no bar
    },
    "caption": {
        "enabled": False,            # off for now (user decision 2026-11-22); true = burn the caption in (the .srt / manifest still carry the labels)
        # Our caption must read as ours, not as the broadcaster's ticker: a solid brand-coloured box, top centre,
        # clear of the broadcaster score bar (y 900-1024 at 1080p) and the corner logos (y < ~150 at 1080p).
        "y": 0.15,                   # top of the caption TEXT as a fraction of frame height (box adds padding)
        "border_width": 0,           # px outline around the letters; 0 = none
        "box": True,
        "box_padding": 22,           # px above / below the text
        "box_padding_x": 44,         # px left / right of the text
        "fade": 0.3,                 # s fade in at the caption start and out at the clip end; 0 = hard cut
        "ref_height": 1080,          # frame height the px values (font size, padding, border) are designed for
    },
    "safe_margins": {"x": 64, "y": 48},  # px from the frame edge for the logo / corner bug
}

_HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})")
_NAMED = frozenset({"white", "black", "red", "green", "blue", "yellow", "orange", "gray", "grey", "navy",
                    "gold", "silver", "transparent"})


class TemplateError(ValueError):
    """The branding template is invalid."""


def ffmpeg_color(value: str) -> str:
    """'#rrggbb' / '#rrggbbaa' -> '0xrrggbb[aa]'; plain names pass through."""
    return "0x" + value[1:] if value.startswith("#") else value


def _merge(base: dict, over: dict, where: str) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):            # "_comment" keys are allowed at any level
            continue
        if k not in base:
            raise TemplateError(f"unknown key {where}{k!r} (known: {', '.join(sorted(base))})")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TemplateError(f"{where}{k} must be an object, got {type(v).__name__}")
            out[k] = _merge(base[k], v, f"{where}{k}.")
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class Template:
    data: dict
    source: Path | None = None

    # --- convenience accessors -----------------------------------------------------------------------
    @property
    def font(self) -> str:
        return self.data["font"]["file"] or DEFAULT_FONT

    @property
    def sizes(self) -> dict[str, int]:
        return self.data["font"]["sizes"]

    @property
    def colors(self) -> dict[str, str]:
        return self.data["colors"]

    def color(self, key: str) -> str:
        return ffmpeg_color(self.data["colors"][key])

    @property
    def cards(self) -> dict:
        return self.data["cards"]

    @property
    def card_seconds(self) -> dict[str, float]:
        return self.data["cards"]["seconds"]

    @property
    def caption(self) -> dict:
        return self.data["caption"]

    @property
    def logo_file(self) -> str | None:
        return self.data["logo"]["file"]

    def logo(self, kind: str) -> dict | None:
        """Logo placement for 'card' or 'clip_bug', or None when no logo should be drawn there."""
        spec = self.data["logo"][kind]
        return spec if self.logo_file and spec["enabled"] else None

    @property
    def margins(self) -> dict[str, int]:
        return self.data["safe_margins"]


def _num(v, where: str, lo: float = 0.0, hi: float | None = None, allow_lo: bool = True) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TemplateError(f"{where} must be a number, got {v!r}")
    if (v < lo or (v == lo and not allow_lo)) or (hi is not None and v > hi):
        rng = f"{'>=' if allow_lo else '>'} {lo:g}" + (f" and <= {hi:g}" if hi is not None else "")
        raise TemplateError(f"{where} must be {rng}, got {v!r}")


def _file(v, where: str, base: Path) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise TemplateError(f"{where} must be a file path or null, got {v!r}")
    p = Path(v).expanduser()
    if not p.is_absolute():
        p = base / p
    if not p.is_file():
        raise TemplateError(f"{where}: file not found: {p}")
    if any(c in str(p) for c in "'\\"):
        raise TemplateError(f"{where}: path may not contain quotes or backslashes (ffmpeg filter): {p}")
    return str(p)


def validate(raw: dict, base_dir: Path) -> dict:
    """Merge `raw` over DEFAULTS and check every value. Returns the resolved dict."""
    if not isinstance(raw, dict):
        raise TemplateError("template must be a JSON object")
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}   # "_comment" keys are allowed
    d = _merge(DEFAULTS, raw, "")
    for k, v in d["colors"].items():
        if not isinstance(v, str) or not (_HEX.fullmatch(v) or v.lower() in _NAMED):
            raise TemplateError(f"colors.{k} must be '#rrggbb', '#rrggbbaa' or one of "
                                f"{', '.join(sorted(_NAMED))}; got {v!r}")
    for k, v in d["font"]["sizes"].items():
        _num(v, f"font.sizes.{k}", 1, 1000)
        d["font"]["sizes"][k] = int(v)
    d["font"]["file"] = _file(d["font"]["file"], "font.file", base_dir)
    if d["font"]["file"] and not d["font"]["file"].lower().endswith((".ttf", ".otf")):
        raise TemplateError(f"font.file must be a .ttf or .otf file, got {d['font']['file']}")
    d["logo"]["file"] = _file(d["logo"]["file"], "logo.file", base_dir)
    if d["logo"]["file"] and not d["logo"]["file"].lower().endswith(".png"):
        raise TemplateError(f"logo.file must be a PNG (with alpha), got {d['logo']['file']}")
    for kind in ("card", "clip_bug"):
        spec = d["logo"][kind]
        if not isinstance(spec["enabled"], bool):
            raise TemplateError(f"logo.{kind}.enabled must be true or false")
        if spec["position"] not in POSITIONS:
            raise TemplateError(f"logo.{kind}.position must be one of {', '.join(POSITIONS)}; "
                                f"got {spec['position']!r}")
        _num(spec["width"], f"logo.{kind}.width", 1, 4000)
        _num(spec["opacity"], f"logo.{kind}.opacity", 0, 1, allow_lo=False)
    for k, v in d["cards"]["seconds"].items():
        _num(v, f"cards.seconds.{k}", 0, 60, allow_lo=False)
    _num(d["cards"]["line_spacing"], "cards.line_spacing", 0.5, 5)
    _num(d["cards"]["fade"], "cards.fade", 0, 5)
    _num(d["cards"]["accent_bar_height"], "cards.accent_bar_height", 0, 1000)
    _num(d["caption"]["y"], "caption.y", 0, 1)
    _num(d["caption"]["border_width"], "caption.border_width", 0, 50)
    if not isinstance(d["caption"]["enabled"], bool):
        raise TemplateError("caption.enabled must be true or false")
    if not isinstance(d["caption"]["box"], bool):
        raise TemplateError("caption.box must be true or false")
    _num(d["caption"]["box_padding"], "caption.box_padding", 0, 200)
    _num(d["caption"]["box_padding_x"], "caption.box_padding_x", 0, 400)
    _num(d["caption"]["fade"], "caption.fade", 0, 2)
    _num(d["caption"]["ref_height"], "caption.ref_height", 100, 4320)
    if d["caption"]["box"] and d["caption"]["y"] > 0.75:
        raise TemplateError(f"caption.y {d['caption']['y']:g} puts the caption box over the broadcaster score bar "
                            "(keep it <= 0.75; the upper part of the frame is recommended)")
    for k, v in d["safe_margins"].items():
        _num(v, f"safe_margins.{k}", 0, 1000)
    if not isinstance(d["name"], str):
        raise TemplateError("name must be a string")
    return d


def load_template(path: Path | str | None = None) -> Template:
    """Load + validate a template. No path: config/template.json if present, else the example, else DEFAULTS."""
    if path is None:
        path = next((p for p in (TEMPLATE_PATH, EXAMPLE_PATH) if p.is_file()), None)
        if path is None:
            return Template(copy.deepcopy(DEFAULTS), None)
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TemplateError(f"template not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise TemplateError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from None
    try:
        return Template(validate(raw, path.parent), path)
    except TemplateError as exc:
        raise TemplateError(f"{path}: {exc}") from None


@lru_cache(maxsize=1)
def active_template() -> Template:
    """The template the renderer uses (loaded once per process)."""
    return load_template()


# --- ffmpeg filter pieces ------------------------------------------------------------------------------
def overlay_xy(position: str, margins: dict[str, int]) -> tuple[str, str]:
    """overlay x/y expressions (main = W/H, overlay = w/h) for a corner position inside the safe margins."""
    mx, my = int(margins["x"]), int(margins["y"])
    x = {"left": f"{mx}", "right": f"W-w-{mx}", "center": "(W-w)/2"}[position.split("-")[1]]
    y = {"top": f"{my}", "bottom": f"H-h-{my}"}[position.split("-")[0]]
    return x, y


def logo_chain(spec: dict, label_in: str, label_out: str) -> str:
    """[label_in] logo input -> scaled, opacity-applied [label_out]."""
    chain = f"[{label_in}]scale={int(spec['width'])}:-1,format=rgba"
    if spec["opacity"] < 1:
        chain += f",colorchannelmixer=aa={spec['opacity']:g}"
    return chain + f"[{label_out}]"


def card_background(tpl: Template, size: tuple[int, int], fps: float, seconds: float) -> str:
    w, h = size
    return f"color=c={tpl.color('background')}:s={w}x{h}:r={fps:g}:d={seconds}"


def card_draws(tpl: Template, lines: list[tuple[str, int]], size: tuple[int, int],
               textfiles: list[Path | str]) -> list[str]:
    """drawtext (and accent bar) filters for one card, vertically centred."""
    w, h = size
    spacing = tpl.cards["line_spacing"]
    total_h = sum(fs * spacing for _, fs in lines)
    y = (h - total_h) / 2
    draws = []
    for tf, (_, fs) in zip(textfiles, lines):
        draws.append(f"drawtext=fontfile={tpl.font}:textfile='{tf}':fontsize={fs}:fontcolor={tpl.color('text')}:"
                     f"x=(w-text_w)/2:y={y:.0f}")
        y += fs * spacing
    bar = int(tpl.cards["accent_bar_height"])
    if bar > 0:
        draws.append(f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color={tpl.color('accent')}:t=fill")
    return draws


def card_fade(tpl: Template, seconds: float) -> str:
    f = tpl.cards["fade"]
    return f"fade=t=in:st=0:d={f:g},fade=t=out:st={max(0.0, seconds - f):.2f}:d={f:g}"


def caption_draw(tpl: Template, textfile: Path | str, offset: float, height: int = 1080,
                 end: float | None = None) -> str:
    """drawtext for the burned clip caption, shown from `offset` (clip seconds) to the clip end.

    Pixel values in the template (font size, padding, border) are for caption.ref_height and are scaled to the
    output frame `height`. With caption.fade > 0 the caption (text and box) fades in at `offset` and, when the
    clip length `end` is given, out over the last caption.fade seconds of the clip."""
    c = tpl.caption
    k = height / float(c["ref_height"])
    px = lambda v: max(0, int(round(v * k)))  # noqa: E731
    s = (f"drawtext=fontfile={tpl.font}:textfile='{textfile}':fontsize={max(1, px(tpl.sizes['caption']))}:"
         f"fontcolor={tpl.color('caption_text')}:")
    if c["border_width"]:
        s += f"borderw={max(1, px(c['border_width']))}:bordercolor={tpl.color('caption_border')}:"
    if c["box"]:
        pad_y, pad_x = px(c["box_padding"]), px(c["box_padding_x"])
        s += f"box=1:boxcolor={tpl.color('caption_box')}:boxborderw={pad_y}|{pad_x}:"
    f = float(c["fade"])
    if f > 0:
        fade_in = f"(t-{offset:.2f})/{f:g}"
        expr = f"min({fade_in},({end:.2f}-t)/{f:g})" if end is not None else fade_in
        s += f"alpha='max(0,min(1,{expr}))':"
    return s + f"x=(w-text_w)/2:y=h*{c['y']:g}:enable='gte(t,{offset:.2f})'"
