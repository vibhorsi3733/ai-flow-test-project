"""Pure publish-decision logic. No I/O: takes the manifest and approvals dicts."""

from __future__ import annotations

import re

# Package holds a reviewer may only override by naming them in the package note.
HARD_BLOCKER_HOLDS = ("quality_gate_failed", "wicket_missing")

DECISIONS = ("approve", "reject")


def _norm(text: str) -> str:
    """Lowercase and collapse '_', '-' and whitespace to single spaces."""
    return re.sub(r"[\s_\-]+", " ", (text or "").lower()).strip()


def hold_acknowledged(hold: str, note: str) -> bool:
    """True when the note names the hold, e.g. 'quality_gate_failed' or 'quality gate failed'."""
    return _norm(hold) in _norm(note)


def compute_publish_decision(manifest: dict, approvals: dict | None) -> dict:
    """Decide whether the package can be marked approved.

    Rules: the reviewer approved the package, every held clip has a decision,
    and every hard-blocker package hold is named in the package approval note.
    Clips that are not held and have no decision ride along with the package
    (they passed the automatic checks). Withheld clips are never included.
    """
    approvals = approvals or {}
    clip_decisions = approvals.get("clips") or {}
    pkg = approvals.get("package") or None

    clip_ids = [c.get("id") for c in manifest.get("clips") or [] if c.get("id")]
    clip_set = set(clip_ids)
    held = [cid for cid in manifest.get("held_clip_ids") or [] if cid]
    # A clip with its own hold_reasons counts as held even if the list missed it.
    for c in manifest.get("clips") or []:
        if c.get("hold_reasons") and c.get("id") and c["id"] not in held:
            held.append(c["id"])
    held_set = set(held)

    def decision_of(cid):
        d = clip_decisions.get(cid) or {}
        return d.get("decision") if d.get("decision") in DECISIONS else None

    approved = [cid for cid in clip_ids if decision_of(cid) == "approve"]
    rejected = [cid for cid in clip_ids if decision_of(cid) == "reject"]
    undecided_held = [cid for cid in held if decision_of(cid) is None]
    default_included = [cid for cid in clip_ids
                        if cid not in held_set and decision_of(cid) is None]
    unknown = sorted(cid for cid in clip_decisions if cid not in clip_set)

    package_holds = list(manifest.get("package_holds") or [])
    hard_holds = [h for h in package_holds if h in HARD_BLOCKER_HOLDS]
    pkg_decision = pkg.get("decision") if pkg and pkg.get("decision") in DECISIONS else None
    pkg_note = (pkg or {}).get("note") or ""
    unacknowledged = [h for h in hard_holds if not hold_acknowledged(h, pkg_note)]

    blockers = []
    if pkg_decision is None:
        blockers.append("package_not_decided")
    elif pkg_decision == "reject":
        blockers.append("package_rejected")
    elif not (pkg.get("reviewer") or "").strip():
        blockers.append("package_reviewer_missing")
    for cid in undecided_held:
        blockers.append(f"held_clip_undecided:{cid}")
    for h in unacknowledged:
        blockers.append(f"hard_hold_not_acknowledged:{h}")
    publish_ids = [cid for cid in clip_ids if cid in set(approved) | set(default_included)]
    if not publish_ids:
        blockers.append("no_clips_to_publish")

    can_approve = not blockers
    return {
        "package_decision": pkg_decision,
        "package_approved": can_approve,
        "status": "approved" if can_approve else
                  ("rejected" if pkg_decision == "reject" else "pending"),
        "approved_clip_ids": approved,
        "rejected_clip_ids": rejected,
        "default_included_clip_ids": default_included,
        "publish_clip_ids": publish_ids if can_approve else [],
        "undecided_held_clip_ids": undecided_held,
        "hard_holds": hard_holds,
        "unacknowledged_hard_holds": unacknowledged,
        "unknown_clip_ids": unknown,
        "blockers": blockers,
        # Publishing is out of scope for this step; never flip this here.
        "publish_channel_allowed": False,
    }
