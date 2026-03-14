"""Tests for treemoissa.llm_analyzer (async)."""

import json

import httpx
import pytest

from treemoissa.llm_analyzer import LLMCarResult, _parse_response, analyze_image


def test_parse_response_valid_json():
    text = '[{"brand": "porsche", "model": "911", "color": "red"}]'
    result = _parse_response(text)
    assert len(result) == 1
    assert result[0]["brand"] == "porsche"


def test_parse_response_embedded_json():
    text = 'Here are the cars: [{"brand": "toyota", "model": "supra", "color": "blue"}] done.'
    result = _parse_response(text)
    assert len(result) == 1


def test_parse_response_empty_array():
    assert _parse_response("[]") == []


def test_parse_response_garbage():
    assert _parse_response("no cars here") == []


@pytest.mark.asyncio
async def test_analyze_image_single_car(tmp_path):
    """Test analyze_image with a mocked server response."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    response_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps([
                        {"brand": "Ferrari", "model": "F40", "color": "red"},
                    ])
                }
            }
        ]
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        results = await analyze_image(img_path, client=client, server_url="http://fake:8080")

    assert len(results) == 1
    assert results[0].brand == "ferrari"
    assert results[0].model == "f40"
    assert results[0].color == "red"


@pytest.mark.asyncio
async def test_analyze_image_no_cars(tmp_path):
    """Test analyze_image when LLM returns empty array."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="white")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    response_body = {
        "choices": [{"message": {"content": "[]"}}]
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        results = await analyze_image(img_path, client=client, server_url="http://fake:8080")

    assert results == []
