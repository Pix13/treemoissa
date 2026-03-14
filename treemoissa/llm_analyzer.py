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
You are a car identification expert working with photos from car trackdays and automotive shows.
Do NOT use thinking mode. Do NOT output <think> tags. Respond immediately with the JSON array.

Context: every photo in this batch was taken at a motorsport or car show event. \
Each photo almost certainly contains at least one car, even if it is partially visible, \
shot from an unusual angle (rear, top, close-up of a detail), motion-blurred, or taken \
in difficult lighting.

For each car visible, provide: brand (manufacturer), model, and dominant color.

IMPORTANT rules:
- Always attempt identification. Never use "unknown" for brand or model if a car is \
visible — provide your best estimate based on body shape, badges, lights, or any \
recognizable feature.
- Only return an empty array [] if you are certain there is no car in the image at all \
(e.g. a photo of the sky, crowd, or pit lane equipment with no vehicle).
- If you can identify the brand but not the exact model, use the closest model family \
(e.g. "911" for a Porsche sports car you cannot identify more precisely).

Respond ONLY with a JSON array. Each element must have exactly these keys:
- "brand": manufacturer name in lowercase (e.g. "porsche", "toyota", "ford")
- "model": model name in lowercase (e.g. "911", "supra", "mustang")
- "color": dominant color in lowercase (e.g. "red", "silver", "black", "white", "blue")

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


async def analyze_image(
    image_path: Path,
    *,
    client: httpx.AsyncClient,
    server_url: str = DEFAULT_URL,
) -> tuple[list[LLMCarResult], str]:
    """Send an image to the LLM server and get car identifications.

    Returns (results, raw_response_text).
    results is an empty list if the LLM detected no cars.
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
                        "text": "Identify all cars in this photo. /no_think",
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        # Disable Qwen3 thinking/reasoning mode — prevents <think> blocks from
        # consuming all tokens before the JSON output is generated.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = await client.post(f"{server_url}/v1/chat/completions", json=payload)
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

    return results, text
