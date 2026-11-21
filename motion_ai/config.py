"""Runtime settings, loaded from environment / .env."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

MAX_OVERS = 20


def parse_overs_spec(spec) -> frozenset[int]:
    """Overs for one team: "1-6", [1, 2, 16], or a mix like ["1-6", 20]. Every over must be 1..20."""
    items = spec if isinstance(spec, list) else [spec]
    out: set[int] = set()
    for it in items:
        if isinstance(it, bool):
            raise ValueError(f"over {it!r} is not a number")
        if isinstance(it, int):
            lo = hi = it
        elif isinstance(it, str) and (m := re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", it)):
            lo, hi = int(m[1]), int(m[2] or m[1])
        else:
            raise ValueError(f"over {it!r} must be an int or an 'a-b' range string")
        if not (1 <= lo <= hi <= MAX_OVERS):
            raise ValueError(f"overs {it!r} must lie within 1..{MAX_OVERS} (and a <= b)")
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError("an empty overs list selects nothing; drop the team instead")
    return frozenset(out)


def parse_highlight_overs(raw: str | list | None) -> dict[str, frozenset[int]]:
    """HIGHLIGHT_OVERS JSON -> {team: overs}. Empty/unset = {} (all overs, the default)."""
    if raw in (None, "", []):
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ValueError(f"HIGHLIGHT_OVERS is not valid JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(data, list):
        raise ValueError('HIGHLIGHT_OVERS must be a JSON array like [{"team": "...", "overs": "1-6"}]')
    out: dict[str, frozenset[int]] = {}
    for entry in data:
        if not isinstance(entry, dict) or "team" not in entry or "overs" not in entry:
            raise ValueError(f"HIGHLIGHT_OVERS entry {entry!r} needs \"team\" and \"overs\"")
        team = str(entry["team"])
        try:
            overs = parse_overs_spec(entry["overs"])
        except ValueError as exc:
            raise ValueError(f"HIGHLIGHT_OVERS team {team!r}: {exc}") from None
        out[team] = out.get(team, frozenset()) | overs
    return out


@dataclass(frozen=True)
class OversFilter:
    """Which overs of which batting team go into the package (HIGHLIGHT_OVERS).

    batting_order[0] bats in innings 1, batting_order[1] in innings 2. A team not listed in the
    selection keeps all its overs; an event without a known over is never filtered.
    """
    selection: dict[str, frozenset[int]]
    batting_order: tuple[str, ...] = field(default_factory=tuple)

    def team_for_innings(self, innings: int) -> str | None:
        i = int(innings or 1) - 1
        return self.batting_order[i] if 0 <= i < len(self.batting_order) else None

    def allows(self, innings: int, over: int | None) -> bool:
        if over is None:
            return True
        team = self.team_for_innings(innings)
        return team not in self.selection or over in self.selection[team]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Source
    srt_url: str
    work_dir: Path = Path("runtime")
    output_dir: Path = Path("output")
    match_config_path: Path = Path("config/match.json")
    # For a local file: true = play at real-time speed (like live), false = process as fast as possible
    simulate_live: bool = True

    # OCR (optional)
    ocr_available: bool = False
    ocr_config_path: Path = Path("config/scoreboard.json")
    ocr_fps: float = 2.0
    ocr_use_gpu: bool = True
    ocr_lag_seconds: float = 6.0

    # Speech-to-text
    asr_model: str = "large-v3-turbo"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_language: str | None = None
    asr_cpu_threads: int = 0          # 0 = all CPU cores (CTranslate2 otherwise defaults to 4 or fewer)

    # LLM
    anthropic_model: str = "claude-opus-5-5"
    llm_effort: str = "low"

    # Highlights
    # powerplay = toss -> end of powerplay (finish right after it); match = whole input, both innings
    highlight_scope: str = "powerplay"
    powerplay_overs: int = 6
    highlight_target_seconds: int = 240
    powerplay_max_minutes: int = 45
    excitement_z: float = 2.5
    video_encoder: str = "libx264"
    # JSON array: [{"team": "Triumph Knights", "overs": "1-6"}, {"team": "...", "overs": [1, 2, "16-20"]}]
    # Empty = every over of both innings.
    highlight_overs: str = ""

    _overs: dict[str, frozenset[int]] = PrivateAttr(default_factory=dict)

    @field_validator("asr_language", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return v or None

    @field_validator("highlight_scope")
    @classmethod
    def _scope(cls, v):
        v = (v or "powerplay").lower()
        if v not in ("powerplay", "match"):
            raise ValueError("HIGHLIGHT_SCOPE must be 'powerplay' or 'match'")
        return v

    @model_validator(mode="after")
    def _check_highlight_overs(self):
        """Startup error for a malformed HIGHLIGHT_OVERS or a team that is not in match.json."""
        overs = parse_highlight_overs(self.highlight_overs)
        if overs:
            teams = list(load_match(self.match_config_path).get("teams", {}))
            unknown = [t for t in overs if t not in teams]
            if unknown:
                raise ValueError(f"HIGHLIGHT_OVERS names unknown team(s) {unknown}; "
                                 f"teams in {self.match_config_path}: {teams}")
        self._overs = overs
        return self

    @property
    def overs_selection(self) -> dict[str, frozenset[int]]:
        return self._overs

    def overs_filter(self) -> OversFilter | None:
        """None when HIGHLIGHT_OVERS is empty (all overs). Innings 1 = match.json "batting_first"."""
        if not self._overs:
            return None
        match = load_match(self.match_config_path)
        return OversFilter(self._overs, batting_order(match))

    @property
    def match_scope(self) -> bool:
        return self.highlight_scope == "match"

    @property
    def fusion_settle_seconds(self) -> float:
        # With OCR we wait longer so the (late) scoreboard update can confirm the commentary call.
        return 30.0 if self.ocr_available else 20.0

    def validate_ocr_setup(self) -> None:
        """Fail fast with a clear message if OCR is switched on but not set up, or needed but off."""
        if self.match_scope and not self.ocr_available:
            # Full-match curation only shows scoreboard-confirmed boundaries/wickets: without OCR every
            # clip would be withheld as not_on_scoreboard and the package would be empty.
            raise SystemExit(
                "HIGHLIGHT_SCOPE=match needs the scoreboard: set OCR_AVAILABLE=true (with "
                "config/scoreboard.json), or use HIGHLIGHT_SCOPE=powerplay.")
        if not self.ocr_available:
            return
        if not self.ocr_config_path.exists():
            raise SystemExit(
                f"OCR_AVAILABLE=true but {self.ocr_config_path} not found. "
                "Copy config/scoreboard.example.json and measure the score boxes, "
                "or set OCR_AVAILABLE=false."
            )
        try:
            import paddleocr  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "OCR_AVAILABLE=true but paddleocr is not installed. "
                "Run `pip install -r requirements-ocr.txt` or set OCR_AVAILABLE=false."
            ) from exc


def load_match(path: Path) -> dict:
    """Teams/players for this match. Used to spell names correctly in ASR + captions."""
    if not path.exists():
        return {"match": "IPL match", "teams": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def batting_order(match: dict) -> tuple[str, ...]:
    """(innings-1 team, innings-2 team) from match.json "batting_first"; first team key if missing."""
    teams = list(match.get("teams", {}))
    first = match.get("batting_first")
    if first not in teams:
        if teams:
            log.warning('match.json has no valid "batting_first" (%r); assuming %r batted first', first, teams[0])
        first = teams[0] if teams else None
    return tuple([first] + [t for t in teams if t != first]) if first else ()
