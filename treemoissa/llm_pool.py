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
