"""Shared utilities — no heavy dependencies."""

from __future__ import annotations

import re


def _sanitize(name: str) -> str:
    """Sanitize a name for use as a directory component."""
    name = name.strip().lower()
    # Remove characters not safe for filenames
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or "unknown"
