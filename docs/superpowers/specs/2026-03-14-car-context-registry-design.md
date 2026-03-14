# Design: Car Context Registry

**Date:** 2026-03-14
**Feature:** Inject previously identified cars as context into LLM prompts to improve recognition of repeated vehicles from different angles.

---

## Summary

Maintain a thread-safe LRU registry of the last N unique `(brand, model, color)` tuples identified during a run. Before each LLM call, inject the current registry contents into the prompt. After each successful identification, update the registry. This helps the LLM recognise the same car from a different angle (e.g. rear shot after a front shot).

Scope: LLM mode only. ML mode is unaffected.

---

## Architecture

### 1. `CarRegistry` — `treemoissa/registry.py`

Single-responsibility component: maintain a bounded, deduplicated, ordered list of identified cars.

```python
class CarRegistry:
    def __init__(self, max_size: int = 25) -> None: ...
    async def add(self, brand: str, model: str, color: str) -> None: ...
    def snapshot(self) -> list[tuple[str, str, str]]: ...
```

**`add(brand, model, color)`:**
- Protected by `asyncio.Lock`.
- If the tuple already exists, move it to the front (LRU bump — recently seen cars stay visible).
- If not present and registry is full, evict the oldest entry (tail), then prepend the new one.
- If not present and registry is not full, prepend.
- Entries with `brand == "unknown"` or `model == "unknown"` are silently ignored (partial identifications do not pollute the context). `color == "unknown"` is accepted — a known brand+model with unknown color is still useful context.

**`snapshot()`:**
- Synchronous (no `await`). In asyncio, coroutines only yield at `await` points. Because `snapshot()` has no `await`, it cannot be interleaved with `add()` on the event loop — no lock is needed. This is **asyncio-specific** reasoning; it would be unsafe in a threaded context.
- Returns a shallow copy (list) of the registry keys, newest-first (index 0 = most recently seen).

**Internal structure:** `collections.OrderedDict` keyed by `(brand, model, color)` tuple, values unused.

**LRU direction:** front = index 0 = newest (most recently seen or added). Tail = last index = oldest. `OrderedDict.move_to_end(key, last=False)` moves to front. Eviction removes the tail (`last=True` in `popitem`).

### 2. Prompt injection — `treemoissa/llm_analyzer.py`

`analyze_image` gains an optional parameter:

```python
async def analyze_image(
    image_path: Path,
    *,
    client: httpx.AsyncClient,
    server_url: str = DEFAULT_URL,
    context: list[tuple[str, str, str]] | None = None,
) -> tuple[list[LLMCarResult], str]:
```

If `context` is non-empty, a block is prepended to the user message text (before `/no_think`):

```
Previously identified cars at this event:
- porsche 911 (red)
- bmw m3 (white)
- renault megane rs (blue)

Identify all cars in this photo. /no_think
```

If `context` is `None` or empty, the user message is unchanged. The system prompt is never modified — backends may cache its tokenisation, avoiding redundant GPU work.

### 3. Wiring — `treemoissa/llm_pool.py`

**In `run()`**, after `_server_stats` initialisation:
```python
from treemoissa.registry import CarRegistry
registry = CarRegistry(max_size=25)
```

`registry` is passed to each worker via a new parameter.

**`_analyze_with_retry` gains a `context` parameter:**
```python
async def _analyze_with_retry(
    self, image_path, client, semaphores, server_order, stats_lock,
    context: list[tuple[str, str, str]],
) -> tuple[list[LLMCarResult] | None, str]:
```
It forwards `context` to every `analyze_image` call. No other logic changes.

**In `_worker()`**, before calling `_analyze_with_retry`:
```python
context = registry.snapshot()
```

`context` flows: `_worker` → `_analyze_with_retry` → `analyze_image`.

**After a successful identification** (results non-empty), in `_worker()`:
```python
for car in results:
    if car.brand != "unknown" and car.model != "unknown":
        await registry.add(car.brand, car.model, car.color)
```

The registry is not updated for photos that go to `unknown/unknown/unknown`.

---

## Data Flow

```
_worker
  ├─ context = registry.snapshot()          # read: no lock
  ├─ results = await _analyze_with_retry(context=context)
  │    └─ analyze_image(context=context)
  │         └─ injects context into user message if non-empty
  └─ if results and not unknown:
       await registry.add(...)              # write: asyncio.Lock
```

---

## Edge Cases

| Case | Behaviour |
|------|-----------|
| First images (registry empty) | `context=[]` → no injection, prompt unchanged |
| All results unknown | Registry not updated |
| brand or model == "unknown" | Not added (partial ID excluded) |
| color == "unknown" but brand+model known | Added (useful context even without color) |
| Registry at capacity | Oldest entry evicted, new entry prepended |
| Same car seen again | Moved to front (LRU bump), no duplicate |
| Concurrent workers adding same car | Lock ensures only one entry added |
| ML mode | `CarRegistry` not instantiated; `analyze_image` signature is backwards-compatible (`context` defaults to `None`) |

---

## Files Changed

| File | Change |
|------|--------|
| `treemoissa/registry.py` | **New file** — `CarRegistry` class |
| `treemoissa/llm_analyzer.py` | Add `context` param to `analyze_image`; inject block into user message |
| `treemoissa/llm_pool.py` | Instantiate `CarRegistry` in `run()`; pass to `_worker`; add `context` param to `_analyze_with_retry`; snapshot before each call; update after success |

---

## Non-Goals

- No persistence of the registry across runs
- No registry for ML mode
- No configurable `max_size` via CLI (hardcoded 25)
- No weighting by frequency (LRU is sufficient)
