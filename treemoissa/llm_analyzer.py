"""Analyze car photos using a vision LLM served by llama.cpp."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from treemoissa.utils import _sanitize

DEFAULT_URL = "http://localhost:8080"

_SYSTEM_PROMPT = """\
You are a car identification expert. Analyze the provided photo and identify ALL cars visible.

For each car, provide: brand (manufacturer), model, and dominant color.

Respond ONLY with a JSON array. Each element must have exactly these keys:
- "brand": manufacturer name (e.g. "porsche", "toyota", "ford")
- "model": model name (e.g. "911", "supra", "mustang")
- "color": dominant color (e.g. "red", "silver", "black", "white", "blue")

If no car is visible, respond with an empty array: []

Examples:
- Single car: [{"brand": "porsche", "model": "911", "color": "red"}]
- Two cars: [{"brand": "porsche", "model": "911", "color": "red"}, \
{"brand": "toyota", "model": "supra", "color": "silver"}]
- No car: []

Respond with ONLY the JSON array, no other text."""


@dataclass
class LLMCarResult:
    """A car identified by the LLM."""

    brand: str
    model: str
    color: str


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Encode image to base64 data URI. Returns (base64_data, media_type)."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".webp": "image/webp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return data, media_type


def _parse_response(text: str) -> list[dict]:
    """Parse the LLM response into a list of car dicts."""
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def analyze_image(
    image_path: Path,
    server_url: str = DEFAULT_URL,
    timeout: float = 120.0,
) -> list[LLMCarResult]:
    """Send an image to the LLM server and get car identifications.

    Returns a list of LLMCarResult. Empty list means no cars detected.
    """
    b64_data, media_type = _encode_image(image_path)

    payload = {
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Identify all cars in this photo.",
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{server_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"]

    raw_cars = _parse_response(text)
    results = []
    for car in raw_cars:
        if not isinstance(car, dict):
            continue
        brand = _sanitize(str(car.get("brand", "unknown")))
        model = _sanitize(str(car.get("model", "unknown")))
        color = _sanitize(str(car.get("color", "unknown")))
        results.append(LLMCarResult(brand=brand, model=model, color=color))

    return results
