"""Dominant color extraction from car crops using HSV analysis."""

from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image


# HSV-based color mapping: (hue_min, hue_max, sat_min, val_min) -> name
# Hue is in [0, 180] (OpenCV convention), Sat/Val in [0, 255]
_COLOR_RANGES: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]] = [
    # name, (hue_lo, hue_hi), (sat_lo, sat_hi), (val_lo, val_hi)
    ("red",     (0, 10),    (70, 255),  (50, 255)),
    ("red",     (170, 180), (70, 255),  (50, 255)),
    ("orange",  (10, 25),   (70, 255),  (50, 255)),
    ("yellow",  (25, 35),   (70, 255),  (50, 255)),
    ("green",   (35, 85),   (40, 255),  (40, 255)),
    ("blue",    (85, 130),  (40, 255),  (40, 255)),
    ("purple",  (130, 170), (40, 255),  (40, 255)),
    ("white",   (0, 180),   (0, 40),    (200, 255)),
    ("silver",  (0, 180),   (0, 40),    (130, 200)),
    ("gray",    (0, 180),   (0, 40),    (60, 130)),
    ("black",   (0, 180),   (0, 255),   (0, 60)),
]


def _rgb_to_hsv_array(img_array: np.ndarray) -> np.ndarray:
    """Convert RGB [0-255] image array to HSV with H in [0,180], S/V in [0,255]."""
    img_float = img_array.astype(np.float32) / 255.0
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(delta)
    mask = delta > 0
    # Red is max
    m = mask & (cmax == r)
    hue[m] = 30.0 * (((g[m] - b[m]) / delta[m]) % 6)
    # Green is max
    m = mask & (cmax == g)
    hue[m] = 30.0 * (((b[m] - r[m]) / delta[m]) + 2)
    # Blue is max
    m = mask & (cmax == b)
    hue[m] = 30.0 * (((r[m] - g[m]) / delta[m]) + 4)

    # Saturation
    sat = np.where(cmax > 0, (delta / cmax) * 255.0, 0)

    # Value
    val = cmax * 255.0

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    return hsv


def _classify_pixel(h: int, s: int, v: int) -> str:
    """Classify a single HSV pixel into a color name."""
    for name, (h_lo, h_hi), (s_lo, s_hi), (v_lo, v_hi) in _COLOR_RANGES:
        if h_lo <= h <= h_hi and s_lo <= s <= s_hi and v_lo <= v <= v_hi:
            return name
    return "unknown"


def extract_dominant_color(crop: Image.Image, sample_size: int = 5000) -> str:
    """Extract the dominant color from a car crop image.

    Uses center-weighted sampling in HSV space to determine the most
    common color, ignoring background pixels at the edges.
    """
    # Focus on the center 60% of the crop to avoid background
    w, h = crop.size
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    center = crop.crop((margin_x, margin_y, w - margin_x, h - margin_y))

    # Resize for speed if needed
    center_small = center.resize((min(center.width, 200), min(center.height, 200)))

    arr = np.array(center_small)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return "unknown"

    hsv = _rgb_to_hsv_array(arr)
    pixels = hsv.reshape(-1, 3)

    # Sample if too many pixels
    if len(pixels) > sample_size:
        indices = np.random.default_rng(42).choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Classify each pixel
    counter: Counter[str] = Counter()
    for px in pixels:
        color = _classify_pixel(int(px[0]), int(px[1]), int(px[2]))
        if color != "unknown":
            counter[color] += 1

    if not counter:
        return "unknown"

    return counter.most_common(1)[0][0]
