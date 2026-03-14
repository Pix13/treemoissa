# LLM-only Default Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make treemoissa installable without PyTorch, with LLM as default mode and async multi-server support.

**Architecture:** Split dependencies into core (lightweight) and `[ml]` extras. LLM mode becomes default, with `--llm-host ip:port,ip:port` for multi-server. New `llm_pool.py` orchestrates async workers with per-server semaphores. Extract `_sanitize()` into `utils.py` to break transitive torch imports.

**Tech Stack:** Python 3.10+, asyncio, httpx (AsyncClient), Rich progress

**Spec:** `docs/superpowers/specs/2026-03-14-llm-only-default-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `treemoissa/utils.py` | Create | `_sanitize()` function — dependency-free |
| `treemoissa/llm_pool.py` | Create | Async worker pool, per-server semaphores, retry logic |
| `treemoissa/llm_analyzer.py` | Modify | Convert to async, accept `httpx.AsyncClient`, remove `_sanitize` import from classifier |
| `treemoissa/classifier.py` | Modify | Import `_sanitize` from `utils.py` instead of defining it |
| `treemoissa/main.py` | Modify | Remove `--llm`/`--llm-url`, add `--llm-host`/`--llm-concurrency`, lazy ML imports |
| `pyproject.toml` | Modify | Split deps into core / `[ml]` extra |
| `README.md` | Modify | Update docs for new CLI and install modes |
| `tests/test_utils.py` | Create | Tests for `_sanitize()` |
| `tests/test_llm_pool.py` | Create | Tests for pool, retry, fallback logic |
| `tests/test_llm_analyzer.py` | Create | Tests for async `analyze_image` |
| `tests/test_main_cli.py` | Create | Tests for CLI argument parsing and mode detection |

---

## Chunk 1: Extract `_sanitize` and split dependencies

### Task 1: Create `utils.py` with `_sanitize()`

**Files:**
- Create: `treemoissa/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` and `tests/test_utils.py`:

```python
# tests/__init__.py
# (empty)
```

```python
# tests/test_utils.py
"""Tests for treemoissa.utils."""

from treemoissa.utils import _sanitize


def test_sanitize_lowercases():
    assert _sanitize("Porsche") == "porsche"


def test_sanitize_replaces_spaces():
    assert _sanitize("Aston Martin") == "aston_martin"


def test_sanitize_removes_unsafe_chars():
    assert _sanitize("911/turbo") == "911turbo"


def test_sanitize_strips_whitespace():
    assert _sanitize("  supra  ") == "supra"


def test_sanitize_empty_returns_unknown():
    assert _sanitize("") == "unknown"


def test_sanitize_only_unsafe_chars_returns_unknown():
    assert _sanitize("///") == "unknown"


def test_sanitize_preserves_hyphens():
    assert _sanitize("Rolls-Royce") == "rolls-royce"


def test_sanitize_mixed_special_chars():
    assert _sanitize("911 GT3 (RS)") == "911_gt3_rs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'treemoissa.utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# treemoissa/utils.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_utils.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add treemoissa/utils.py tests/__init__.py tests/test_utils.py
git commit -m "feat: extract _sanitize into utils.py (dependency-free)"
```

---

### Task 2: Update `classifier.py` to import from `utils.py`

**Files:**
- Modify: `treemoissa/classifier.py:95-101` (remove `_sanitize` definition)
- Modify: `treemoissa/classifier.py:1-10` (add import)

- [ ] **Step 1: Modify `classifier.py`**

Remove the `_sanitize` function definition (lines 95-101) and add the import at the top:

Add the import after the third-party imports block (after line 9 `from transformers import ...`), following PEP 8 import ordering (stdlib → third-party → first-party):

```python
from treemoissa.utils import _sanitize
```

Delete the `_sanitize` function (lines 95-101).

All existing references to `_sanitize` in `classifier.py` (lines 86, 91, 92) continue to work via the import.

- [ ] **Step 2: Verify the module imports correctly**

Run: `python -c "from treemoissa.classifier import _sanitize, parse_brand_model; print('OK')"`
Expected: `OK` (this requires ML deps, so skip if not installed — the import chain is what matters)

- [ ] **Step 3: Commit**

```bash
git add treemoissa/classifier.py
git commit -m "refactor: import _sanitize from utils in classifier"
```

---

### Task 3: Update `llm_analyzer.py` to import from `utils.py`

**Files:**
- Modify: `treemoissa/llm_analyzer.py:13` (change import source)

- [ ] **Step 1: Change the import**

In `llm_analyzer.py`, replace line 13:

```python
# Old:
from treemoissa.classifier import _sanitize
# New:
from treemoissa.utils import _sanitize
```

- [ ] **Step 2: Verify the import works without ML deps**

Run: `python -c "from treemoissa.llm_analyzer import analyze_image; print('OK')"`
Expected: `OK` — this should now work without torch/ultralytics/transformers installed.

- [ ] **Step 3: Commit**

```bash
git add treemoissa/llm_analyzer.py
git commit -m "refactor: import _sanitize from utils in llm_analyzer"
```

---

### Task 4: Split dependencies in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:6-16`

- [ ] **Step 1: Update `pyproject.toml`**

Replace the `dependencies` section:

```toml
[project]
name = "treemoissa"
version = "0.1.0"
description = "Car trackdays and shows photo triage tool — sort car photos by brand, model, and color"
requires-python = ">=3.10"
dependencies = [
    "Pillow>=10.0.0",
    "rich>=13.0.0",
    "numpy>=1.24.0",
    "huggingface-hub>=0.20.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
ml = [
    "ultralytics>=8.3.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "transformers>=4.40.0",
]
```

- [ ] **Step 2: Verify install**

Run: `pip install -e .` (reinstall without ml extras — should succeed without torch)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: split deps into core and [ml] optional extras"
```

---

## Chunk 2: Async LLM analyzer and pool

### Task 5: Convert `llm_analyzer.py` to async

**Files:**
- Modify: `treemoissa/llm_analyzer.py:85-137`
- Test: `tests/test_llm_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_analyzer.py
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

    # Create a tiny test image
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
```

- [ ] **Step 2: Install test dependencies**

Run: `pip install pytest pytest-asyncio`

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analyzer.py -v`
Expected: FAIL — `analyze_image` is not async and doesn't accept `client` parameter.

- [ ] **Step 4: Rewrite `llm_analyzer.py` as async**

Replace the full file:

```python
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


async def analyze_image(
    image_path: Path,
    *,
    client: httpx.AsyncClient,
    server_url: str = DEFAULT_URL,
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

    return results
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_analyzer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add treemoissa/llm_analyzer.py tests/test_llm_analyzer.py
git commit -m "feat: convert llm_analyzer to async with client injection"
```

---

### Task 6: Create `llm_pool.py` — async worker pool

**Files:**
- Create: `treemoissa/llm_pool.py`
- Test: `tests/test_llm_pool.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'treemoissa.llm_pool'`

- [ ] **Step 3: Write `llm_pool.py`**

```python
"""Async worker pool for distributing LLM requests across servers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from treemoissa.llm_analyzer import LLMCarResult, analyze_image
from treemoissa.organizer import copy_image


@dataclass
class ServerConfig:
    """A single LLM server endpoint."""

    host: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def parse(hosts_str: str) -> list[ServerConfig]:
        """Parse 'ip:port,ip:port' into a list of ServerConfig."""
        servers = []
        for part in hosts_str.split(","):
            part = part.strip()
            if ":" in part:
                host, port_str = part.rsplit(":", 1)
                servers.append(ServerConfig(host=host, port=int(port_str)))
            else:
                servers.append(ServerConfig(host=part, port=8080))
        return servers


MAX_RETRIES = 2  # retries on same server (total 3 attempts)


@dataclass
class LLMPool:
    """Manages async workers distributing photos across LLM servers."""

    servers: list[ServerConfig]
    concurrency: int
    output_dir: Path
    transport: httpx.AsyncBaseTransport | None = None
    on_progress: Callable[[], None] | None = None
    _server_index: int = 0  # rotating index for round-robin

    async def _analyze_with_retry(
        self,
        image_path: Path,
        client: httpx.AsyncClient,
        semaphores: dict[str, asyncio.Semaphore],
        server_order: list[ServerConfig],
    ) -> list[LLMCarResult] | None:
        """Try to analyze an image, with retries and server fallback.

        Returns list of results, or None if all servers failed.
        """
        primary = server_order[0]
        fallbacks = server_order[1:]

        # Try primary server with retries
        async with semaphores[primary.url]:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    return await analyze_image(
                        image_path, client=client, server_url=primary.url,
                    )
                except (httpx.HTTPStatusError, httpx.RequestError):
                    if attempt < MAX_RETRIES:
                        continue

        # Fallback: try each remaining server once
        for server in fallbacks:
            async with semaphores[server.url]:
                try:
                    return await analyze_image(
                        image_path, client=client, server_url=server.url,
                    )
                except (httpx.HTTPStatusError, httpx.RequestError):
                    continue

        return None

    async def _worker(
        self,
        queue: asyncio.Queue[Path | None],
        client: httpx.AsyncClient,
        semaphores: dict[str, asyncio.Semaphore],
        stats: dict[str, int],
        brand_counts: dict[str, int],
        stats_lock: asyncio.Lock,
        stop_event: asyncio.Event,
    ) -> None:
        """Worker coroutine: pull photos from queue, analyze, copy."""
        while not stop_event.is_set():
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            image_path = item
            # Rotate server order for round-robin load distribution
            idx = self._server_index
            self._server_index = (idx + 1) % len(self.servers)
            server_order = self.servers[idx:] + self.servers[:idx]

            results = await self._analyze_with_retry(
                image_path, client, semaphores, server_order,
            )

            # Copy files outside the lock to avoid serializing I/O
            if results is None or not results:
                await asyncio.to_thread(
                    copy_image, image_path, self.output_dir,
                    "unknown", "unknown", "unknown",
                )
                async with stats_lock:
                    stats["no_car"] += 1
                    stats["copies"] += 1
            else:
                seen: set[tuple[str, str, str]] = set()
                copies_made: list[str] = []
                for car in results:
                    key = (car.brand, car.model, car.color)
                    if key in seen:
                        continue
                    seen.add(key)
                    await asyncio.to_thread(
                        copy_image, image_path, self.output_dir,
                        car.brand, car.model, car.color,
                    )
                    copies_made.append(car.brand)
                async with stats_lock:
                    stats["total_cars"] += len(results)
                    stats["copies"] += len(copies_made)
                    for brand in copies_made:
                        brand_counts[brand] = brand_counts.get(brand, 0) + 1

            if self.on_progress:
                self.on_progress()

            queue.task_done()

    async def run(
        self,
        images: list[Path],
        on_progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Process all images through the worker pool.

        Returns stats dict with total_images, total_cars, copies, no_car keys.
        """
        if on_progress:
            self.on_progress = on_progress

        stats: dict[str, int] = {
            "total_images": len(images),
            "total_cars": 0,
            "copies": 0,
            "no_car": 0,
        }
        brand_counts: dict[str, int] = {}
        stats_lock = asyncio.Lock()
        stop_event = asyncio.Event()

        semaphores = {s.url: asyncio.Semaphore(self.concurrency) for s in self.servers}
        num_workers = len(self.servers) * self.concurrency

        queue: asyncio.Queue[Path | None] = asyncio.Queue()
        for img in images:
            queue.put_nowait(img)
        # Sentinel values to stop workers
        for _ in range(num_workers):
            queue.put_nowait(None)

        timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        client_kwargs: dict = {"timeout": timeout}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport

        async with httpx.AsyncClient(**client_kwargs) as client:
            workers = [
                asyncio.create_task(
                    self._worker(
                        queue, client, semaphores, stats, brand_counts,
                        stats_lock, stop_event,
                    )
                )
                for _ in range(num_workers)
            ]

            await asyncio.gather(*workers)

        stats["brand_counts"] = brand_counts
        return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_pool.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add treemoissa/llm_pool.py tests/test_llm_pool.py
git commit -m "feat: add async LLM worker pool with retry and fallback"
```

---

## Chunk 3: Refactor `main.py` CLI and update docs

### Task 7: Refactor `main.py` — lazy imports and new CLI

**Files:**
- Modify: `treemoissa/main.py`
- Test: `tests/test_main_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_main_cli.py
"""Tests for treemoissa CLI argument parsing and mode detection."""

import sys
from unittest.mock import patch

import pytest


def test_default_mode_is_llm():
    """Without --model, LLM mode should be used."""
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir"])
    assert args.model is None
    assert args.llm_host == "localhost:8080"


def test_llm_host_single():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--llm-host", "10.0.0.1:9090"])
    assert args.llm_host == "10.0.0.1:9090"


def test_llm_host_multiple():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--llm-host", "10.0.0.1:8080,10.0.0.2:8080"])
    assert args.llm_host == "10.0.0.1:8080,10.0.0.2:8080"


def test_llm_concurrency():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--llm-concurrency", "3"])
    assert args.llm_concurrency == 3


def test_model_flag_selects_ml_mode():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--model", "yolov8l"])
    assert args.model == "yolov8l"


def test_old_llm_flag_removed():
    """--llm should no longer be accepted."""
    from treemoissa.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["input_dir", "output_dir", "--llm"])


def test_old_llm_url_removed():
    """--llm-url should no longer be accepted."""
    from treemoissa.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["input_dir", "output_dir", "--llm-url", "http://x:8080"])


def test_default_concurrency():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir"])
    assert args.llm_concurrency == 1


def test_default_confidence():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir"])
    assert args.confidence == 0.35


def test_invalid_model_rejected():
    from treemoissa.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["input_dir", "output_dir", "--model", "invalid"])


def test_ml_import_error(monkeypatch):
    """--model with missing ML deps should print error and exit."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    from treemoissa.main import _run_ml_pipeline
    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(SystemExit):
        _run_ml_pipeline(Path("/tmp"), Path("/tmp"), 0.35, "yolov8l")


def test_main_imports_without_torch():
    """main.py must import cleanly without torch."""
    import importlib
    import treemoissa.main
    # If we got here, no ImportError was raised at module level
    assert hasattr(treemoissa.main, "_parse_args")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main_cli.py -v`
Expected: FAIL — `_parse_args` doesn't exist, old flags still present.

- [ ] **Step 3: Rewrite `main.py`**

```python
"""CLI entrypoint for treemoissa."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from treemoissa.organizer import copy_image

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Keep in sync with detector.AVAILABLE_MODELS keys
_ML_MODEL_CHOICES = ["yolov8m", "yolov8l", "rtdetr"]


def gather_images(input_dir: Path) -> list[Path]:
    """Collect all image files from the input directory (non-recursive)."""
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def _print_summary(stats: dict, brand_counts: dict[str, int]) -> None:
    """Print results summary and brand breakdown tables."""
    console.print()
    table = Table(title="Results Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Images processed", str(stats["total_images"]))
    table.add_row("Cars detected", str(stats["total_cars"]))
    table.add_row("Files copied", str(stats["copies"]))
    table.add_row("Images with no car", str(stats["no_car"]))
    console.print(table)

    if brand_counts:
        console.print()
        brand_table = Table(title="Cars by Brand")
        brand_table.add_column("Brand", style="bold")
        brand_table.add_column("Count", justify="right")
        for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
            brand_table.add_row(brand, str(count))
        console.print(brand_table)


def _make_progress() -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _run_llm_pipeline(
    input_dir: Path,
    output_dir: Path,
    llm_host: str,
    concurrency: int,
) -> dict:
    """Run the async LLM-based pipeline."""
    from treemoissa.llm_pool import LLMPool, ServerConfig

    images = gather_images(input_dir)
    if not images:
        console.print("[bold red]No images found in input directory.")
        return {"total_images": 0, "total_cars": 0, "copies": 0, "no_car": 0}

    servers = ServerConfig.parse(llm_host)
    server_list = ", ".join(s.url for s in servers)
    console.print(f"[bold]Mode:[/bold] LLM vision")
    console.print(f"[bold]Servers:[/bold] {server_list}")
    console.print(f"[bold]Concurrency:[/bold] {concurrency} per server")
    console.print(f"[bold]Found {len(images)} images to process.[/bold]\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    pool = LLMPool(
        servers=servers,
        concurrency=concurrency,
        output_dir=output_dir,
    )

    with _make_progress() as progress:
        task_id = progress.add_task("Processing images", total=len(images))

        def on_progress():
            progress.advance(task_id)

        try:
            stats = asyncio.run(pool.run(images, on_progress=on_progress))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — partial results below.[/yellow]")
            return {"total_images": len(images), "total_cars": 0, "copies": 0, "no_car": 0}

    brand_counts = stats.pop("brand_counts", {})
    _print_summary(stats, brand_counts)
    return stats


def _run_ml_pipeline(
    input_dir: Path,
    output_dir: Path,
    conf: float,
    model_key: str,
) -> dict:
    """Run the YOLO+ViT pipeline (requires treemoissa[ml])."""
    try:
        import torch
        from treemoissa.classifier import classify_car, load_classifier, parse_brand_model
        from treemoissa.color import extract_dominant_color
        from treemoissa.detector import AVAILABLE_MODELS, detect_cars, load_detector
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] --model requires ML dependencies.\n"
            "Install with: [bold]pip install treemoissa\\[ml][/bold]"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold]Device:[/bold] {device}")
    if device == "cuda":
        console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0)}")

    model_info = AVAILABLE_MODELS[model_key]

    # Load models
    with console.status(f"[bold green]Loading detection model ({model_info['name']})..."):
        detector = load_detector(model_key=model_key, device=device)

    with console.status("[bold green]Loading classification model..."):
        processor, classifier = load_classifier(device=device)

    # Gather images
    images = gather_images(input_dir)
    if not images:
        console.print("[bold red]No images found in input directory.")
        return {"total_images": 0, "total_cars": 0, "copies": 0, "no_car": 0}

    console.print(f"[bold]Found {len(images)} images to process.[/bold]\n")

    stats = {"total_images": len(images), "total_cars": 0, "copies": 0, "no_car": 0}
    brand_counts: dict[str, int] = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    with _make_progress() as progress:
        task = progress.add_task("Processing images", total=len(images))

        for image_path in images:
            progress.update(task, description=f"[cyan]{image_path.name}")

            try:
                detections = detect_cars(detector, image_path, conf=conf)
            except Exception as e:
                console.print(f"[yellow]Warning: failed to process {image_path.name}: {e}")
                progress.advance(task)
                continue

            if not detections:
                stats["no_car"] += 1
                progress.advance(task)
                continue

            stats["total_cars"] += len(detections)

            seen: set[tuple[str, str, str]] = set()

            for det in detections:
                label, _conf = classify_car(processor, classifier, det.crop, device=device)
                brand, model = parse_brand_model(label)
                color = extract_dominant_color(det.crop)

                key = (brand, model, color)
                if key in seen:
                    continue
                seen.add(key)

                copy_image(image_path, output_dir, brand, model, color)
                stats["copies"] += 1
                brand_counts[brand] = brand_counts.get(brand, 0) + 1

            progress.advance(task)

    _print_summary(stats, brand_counts)
    return stats


def _prompt_model_selection(available_models: dict) -> str:
    """Display an interactive menu to select the detection model."""
    console.print("\n[bold]Select a detection model:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", style="bold", width=3)
    table.add_column("Model", style="bold")
    table.add_column("Description")

    keys = list(available_models.keys())
    for idx, key in enumerate(keys, 1):
        info = available_models[key]
        table.add_row(str(idx), info["name"], info["desc"])

    console.print(table)
    console.print()

    while True:
        choice = console.input(f"[bold]Choice [1-{len(keys)}] (default: 1):[/bold] ").strip()
        if choice == "":
            return keys[0]
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        console.print(f"[red]Please enter a number between 1 and {len(keys)}.[/red]")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="treemoissa",
        description="Sort car photos into brand/model/color directory tree.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Flat directory containing car photos",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the organized tree",
    )
    parser.add_argument(
        "--llm-host",
        type=str,
        default="localhost:8080",
        help="LLM server(s) as ip:port,ip:port (default: localhost:8080)",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Concurrent requests per server (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=_ML_MODEL_CHOICES,
        default=None,
        help="Use YOLO+ViT pipeline with this model (requires treemoissa[ml])",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence for ML mode (default: 0.35)",
    )

    return parser.parse_args(argv)


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    conf: float = 0.35,
    model_key: str | None = None,
    *,
    llm_host: str = "localhost:8080",
    llm_concurrency: int = 1,
) -> dict:
    """Run the pipeline — LLM by default, ML if --model specified."""
    if model_key is not None:
        return _run_ml_pipeline(input_dir, output_dir, conf, model_key)
    return _run_llm_pipeline(input_dir, output_dir, llm_host, llm_concurrency)


def main() -> None:
    """Main CLI entrypoint."""
    args = _parse_args()

    if not args.input_dir.is_dir():
        console.print(f"[bold red]Error: {args.input_dir} is not a directory.")
        sys.exit(1)

    run_pipeline(
        args.input_dir,
        args.output_dir,
        conf=args.confidence,
        model_key=args.model,
        llm_host=args.llm_host,
        llm_concurrency=args.llm_concurrency,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main_cli.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add treemoissa/main.py tests/test_main_cli.py
git commit -m "feat: LLM default mode, lazy ML imports, new --llm-host CLI"
```

---

### Task 8: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Key changes:
- Feature list: LLM is default, YOLO+ViT is optional
- Requirements: remove "NVIDIA GPU with CUDA toolkit (recommended)" from base requirements
- Installation: show `pip install treemoissa` vs `pip install treemoissa[ml]`
- Usage: default examples use LLM, `--model` documented as optional
- LLM Mode section: remove `--llm` flag references, add `--llm-host` examples with multi-server
- Remove old `--llm-url` references

Replace the full README content with updated version that reflects:

```
## Installation

### Lightweight (LLM only)
pip install treemoissa

### With ML pipeline (YOLO + ViT)
pip install treemoissa[ml]
```

Usage section:
```
## Usage

# Default — uses local LLM server
treemoissa <input_dir> <output_dir>

# Multiple LLM servers in parallel
treemoissa <input_dir> <output_dir> --llm-host 10.0.0.1:8080,10.0.0.2:8080

# YOLO+ViT mode (requires treemoissa[ml])
treemoissa <input_dir> <output_dir> --model yolov8l
```

CLI arguments table:
```
| --llm-host       | LLM server(s) ip:port,ip:port (default: localhost:8080) |
| --llm-concurrency| Concurrent requests per server (default: 1)             |
| --model          | Use YOLO+ViT: yolov8m, yolov8l, or rtdetr (needs [ml]) |
| --confidence     | Min detection confidence for ML mode (default: 0.35)    |
```

- [ ] **Step 2: Verify README renders correctly**

Quick visual check of the markdown structure.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for LLM-default mode and multi-server support"
```

---

### Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Add note that LLM mode is now default and ML pipeline is optional via `[ml]` extras. No other changes needed — the project overview description stays the same.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for LLM-default architecture"
```

---

### Task 10: Final integration test

- [ ] **Step 1: Verify clean import without ML deps**

Run: `python -c "from treemoissa.main import _parse_args; print('CLI OK')"`
Expected: `CLI OK` (no torch import errors)

Run: `python -c "from treemoissa.llm_analyzer import analyze_image; print('LLM OK')"`
Expected: `LLM OK`

Run: `python -c "from treemoissa.llm_pool import LLMPool; print('Pool OK')"`
Expected: `Pool OK`

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Verify `--model` without ML deps gives clear error**

Run: `python -m treemoissa /tmp /tmp --model yolov8l`
Expected: `Error: --model requires ML dependencies. Install with: pip install treemoissa[ml]` (only if torch is not installed)

- [ ] **Step 4: Commit any fixes if needed**
