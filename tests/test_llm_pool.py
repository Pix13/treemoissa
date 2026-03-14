# tests/test_llm_pool.py
"""Tests for treemoissa.llm_pool."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from treemoissa.llm_pool import LLMPool, ServerConfig


def _make_ok_transport(cars=None):
    """Create a mock transport that returns a valid LLM response."""
    if cars is None:
        cars = [{"brand": "porsche", "model": "911", "color": "red"}]
    body = {"choices": [{"message": {"content": json.dumps(cars)}}]}
    return httpx.MockTransport(lambda req: httpx.Response(200, json=body))


def _make_failing_transport():
    """Create a transport that always returns 500."""
    return httpx.MockTransport(
        lambda req: httpx.Response(500, text="Internal Server Error")
    )


def _make_test_images(tmp_path, count=3):
    """Create tiny test images."""
    from PIL import Image
    paths = []
    for i in range(count):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (10, 10), color="red").save(p)
        paths.append(p)
    return paths


class TestServerConfig:
    def test_parse_single(self):
        servers = ServerConfig.parse("192.168.1.10:8080")
        assert len(servers) == 1
        assert servers[0].url == "http://192.168.1.10:8080"

    def test_parse_multiple(self):
        servers = ServerConfig.parse("10.0.0.1:8080,10.0.0.2:9090")
        assert len(servers) == 2
        assert servers[0].url == "http://10.0.0.1:8080"
        assert servers[1].url == "http://10.0.0.2:9090"

    def test_parse_default_localhost(self):
        servers = ServerConfig.parse("localhost:8080")
        assert servers[0].url == "http://localhost:8080"


class TestLLMPool:
    @pytest.mark.asyncio
    async def test_process_single_image(self, tmp_path):
        images = _make_test_images(tmp_path, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transport = _make_ok_transport()
        pool = LLMPool(
            servers=ServerConfig.parse("localhost:8080"),
            concurrency=1,
            output_dir=output_dir,
            transport=transport,
        )

        stats = await pool.run(images)
        assert stats["total_images"] == 1
        assert stats["total_cars"] == 1
        assert stats["copies"] == 1

    @pytest.mark.asyncio
    async def test_process_multiple_images(self, tmp_path):
        images = _make_test_images(tmp_path, count=5)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transport = _make_ok_transport()
        pool = LLMPool(
            servers=ServerConfig.parse("localhost:8080"),
            concurrency=2,
            output_dir=output_dir,
            transport=transport,
        )

        stats = await pool.run(images)
        assert stats["total_images"] == 5
        assert stats["copies"] == 5

    @pytest.mark.asyncio
    async def test_retry_then_fallback(self, tmp_path):
        """First server always fails, second succeeds."""
        images = _make_test_images(tmp_path, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        call_count = {"fail": 0, "ok": 0}

        def routing_handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "10.0.0.1" in url:
                call_count["fail"] += 1
                return httpx.Response(500, text="down")
            call_count["ok"] += 1
            body = {"choices": [{"message": {"content": json.dumps(
                [{"brand": "toyota", "model": "supra", "color": "blue"}]
            )}}]}
            return httpx.Response(200, json=body)

        transport = httpx.MockTransport(routing_handler)
        pool = LLMPool(
            servers=ServerConfig.parse("10.0.0.1:8080,10.0.0.2:8080"),
            concurrency=1,
            output_dir=output_dir,
            transport=transport,
        )

        stats = await pool.run(images)
        assert stats["copies"] == 1
        # 3 attempts on first server (1 + 2 retries), then 1 on second
        assert call_count["fail"] == 3
        assert call_count["ok"] == 1

    @pytest.mark.asyncio
    async def test_all_servers_fail_goes_to_unknown(self, tmp_path):
        images = _make_test_images(tmp_path, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transport = _make_failing_transport()
        pool = LLMPool(
            servers=ServerConfig.parse("10.0.0.1:8080"),
            concurrency=1,
            output_dir=output_dir,
            transport=transport,
        )

        stats = await pool.run(images)
        assert stats["no_car"] == 1
        assert stats["copies"] == 1
        # File should be in unknown/unknown/unknown
        assert (output_dir / "unknown" / "unknown" / "unknown").exists()

    @pytest.mark.asyncio
    async def test_no_cars_detected(self, tmp_path):
        images = _make_test_images(tmp_path, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transport = _make_ok_transport(cars=[])
        pool = LLMPool(
            servers=ServerConfig.parse("localhost:8080"),
            concurrency=1,
            output_dir=output_dir,
            transport=transport,
        )

        stats = await pool.run(images)
        assert stats["no_car"] == 1
        assert stats["copies"] == 1
