"""Map manifest-referenced media to safe URL keys.

The server never turns a request path into a filesystem path. Every servable
file is enumerated up front from the manifest and looked up by key
("package" or a clip id); anything else is a 404.
"""

from __future__ import annotations

import re
from pathlib import Path

_CLIP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.(mp4|m4v|webm|mov)$")


def safe_basename(name) -> bool:
    """A plain media file name: no directories, no '..', no hidden files."""
    return (isinstance(name, str) and bool(_SAFE_NAME.match(name))
            and ".." not in name and "/" not in name and "\\" not in name)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def default_clips_dir(manifest_path: Path, manifest: dict) -> Path:
    """Guess where per-clip mp4s live (pipeline writes them to <work_dir>/clips)."""
    base = manifest_path.resolve().parent
    candidates = [base.parent / "runtime_1080" / "clips", base.parent / "runtime" / "clips",
                  base / "clips", base]
    names = [c.get("clip_file") for c in manifest.get("clips") or [] if c.get("clip_file")]
    for cand in candidates:
        if cand.is_dir() and any((cand / n).is_file() for n in names if safe_basename(n)):
            return cand
    return candidates[0]


class MediaIndex:
    """Whitelist of servable files, keyed by 'package' or clip id."""

    def __init__(self, manifest_path: Path, manifest: dict, clips_dir: Path):
        self.clips_dir = Path(clips_dir).resolve()
        self._files: dict[str, Path] = {}
        manifest_path = Path(manifest_path).resolve()
        pkg = manifest_path.with_suffix(".mp4")
        if pkg.is_file():
            self._files["package"] = pkg.resolve()
        for clip in manifest.get("clips") or []:
            cid, name = clip.get("id"), clip.get("clip_file")
            if not (isinstance(cid, str) and _CLIP_ID.match(cid) and cid != "package"):
                continue
            if not safe_basename(name):
                continue
            path = self.clips_dir / name
            if path.is_file() and _inside(path, self.clips_dir):
                self._files[cid] = path.resolve()

    def has(self, key: str) -> bool:
        return key in self._files

    def resolve(self, key: str) -> Path | None:
        """Path for a whitelisted key, or None. Never derived from the key's text."""
        return self._files.get(key) if isinstance(key, str) else None

    def url(self, key: str) -> str | None:
        return f"/media/{key}" if key in self._files else None
