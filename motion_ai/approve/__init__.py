"""Human approval step for highlight packages.

Reads a package manifest (never modifies it), serves a local review page and
records reviewer decisions in ``<package>.approvals.json`` next to the manifest.
Publishing itself is out of scope: ``publish_channel_allowed`` stays false.
"""

from .decision import HARD_BLOCKER_HOLDS, compute_publish_decision
from .media import MediaIndex

__all__ = ["HARD_BLOCKER_HOLDS", "compute_publish_decision", "MediaIndex"]
