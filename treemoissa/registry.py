"""Thread-safe LRU registry of identified cars for context injection."""

from __future__ import annotations

import asyncio
from collections import OrderedDict


class CarRegistry:
    """Maintains the last N unique (brand, model, color) tuples seen during a run.

    Front of the OrderedDict = newest (index 0 in snapshot).
    Tail = oldest (evicted first when full).

    add() is asyncio-safe via Lock.
    snapshot() needs no lock: it has no await, so the asyncio event loop
    cannot interleave it with add() on the same thread.
    """

    def __init__(self, max_size: int = 25) -> None:
        self._max_size = max_size
        self._data: OrderedDict[tuple[str, str, str], None] = OrderedDict()
        self._lock = asyncio.Lock()

    async def add(self, brand: str, model: str, color: str) -> None:
        """Add or bump a car. Ignores entries with unknown brand or model."""
        if brand == "unknown" or model == "unknown":
            return
        key = (brand, model, color)
        async with self._lock:
            if key in self._data:
                # Move to front (most recently seen)
                self._data.move_to_end(key, last=False)
            else:
                if len(self._data) >= self._max_size:
                    # Evict oldest (tail)
                    self._data.popitem(last=True)
                self._data[key] = None
                self._data.move_to_end(key, last=False)

    def snapshot(self) -> list[tuple[str, str, str]]:
        """Return a copy of the registry, newest first. No lock needed (no await)."""
        return list(self._data.keys())
