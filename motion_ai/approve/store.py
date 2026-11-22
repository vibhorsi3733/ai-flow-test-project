"""Read/write ``<package>.approvals.json``. The manifest itself is never written."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .decision import DECISIONS, compute_publish_decision

MAX_NOTE = 2000
MAX_NAME = 100


class ApprovalError(ValueError):
    pass


def approvals_path(manifest_path: Path) -> Path:
    p = Path(manifest_path)
    return p.with_name(p.stem + ".approvals.json")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty_approvals(manifest_path: Path) -> dict:
    return {"manifest": Path(manifest_path).name, "manifest_sha256": file_sha256(manifest_path),
            "package": None, "clips": {}, "history": []}


def load_approvals(manifest_path: Path) -> dict:
    path = approvals_path(manifest_path)
    if not path.is_file():
        return empty_approvals(manifest_path)
    data = json.loads(path.read_text())
    data.setdefault("package", None)
    data.setdefault("clips", {})
    data.setdefault("history", [])
    return data


def save_approvals(manifest_path: Path, manifest: dict, approvals: dict) -> dict:
    approvals["manifest"] = Path(manifest_path).name
    approvals["manifest_sha256"] = file_sha256(manifest_path)
    approvals["publish_decision"] = compute_publish_decision(manifest, approvals)
    approvals["updated_at"] = _now()
    path = approvals_path(manifest_path)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(approvals, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return approvals


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _entry(decision, reviewer, note) -> dict:
    if decision not in DECISIONS:
        raise ApprovalError(f"decision must be one of {DECISIONS}")
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ApprovalError("reviewer name is required")
    if len(reviewer) > MAX_NAME:
        raise ApprovalError("reviewer name too long")
    note = (note or "").strip()
    if len(note) > MAX_NOTE:
        raise ApprovalError("note too long")
    return {"decision": decision, "reviewer": reviewer, "note": note, "decided_at": _now()}


def record_clip(manifest: dict, approvals: dict, clip_id, decision, reviewer, note="") -> dict:
    ids = {c.get("id") for c in manifest.get("clips") or []}
    if clip_id not in ids:
        raise ApprovalError("unknown clip id (withheld clips cannot be decided)")
    entry = _entry(decision, reviewer, note)
    approvals["clips"][clip_id] = entry
    approvals["history"].append({"scope": "clip", "clip_id": clip_id, **entry})
    return entry


def record_package(approvals: dict, decision, reviewer, note="") -> dict:
    entry = _entry(decision, reviewer, note)
    approvals["package"] = entry
    approvals["history"].append({"scope": "package", **entry})
    return entry
