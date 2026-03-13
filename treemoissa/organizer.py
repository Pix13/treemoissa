"""File organization — copy images into brand/model/color directory tree."""

from __future__ import annotations

import shutil
from pathlib import Path


def build_output_path(
    output_dir: Path,
    brand: str,
    model: str,
    color: str,
) -> Path:
    """Build and create the output directory: output_dir/brand/model/color/."""
    target = output_dir / brand / model / color
    target.mkdir(parents=True, exist_ok=True)
    return target


def copy_image(
    source: Path,
    output_dir: Path,
    brand: str,
    model: str,
    color: str,
) -> Path:
    """Copy an image to the appropriate brand/model/color subdirectory.

    Uses shutil.copy2 (no hardlinks) for NFS compatibility.
    Returns the destination path.
    """
    target_dir = build_output_path(output_dir, brand, model, color)
    dest = target_dir / source.name

    # Handle name collisions by appending a counter
    if dest.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1
        while dest.exists():
            dest = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(source, dest)
    return dest
