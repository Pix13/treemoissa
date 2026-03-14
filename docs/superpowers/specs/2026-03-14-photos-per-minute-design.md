# Design: Photos-per-Minute Throughput Metrics

**Date:** 2026-03-14
**Feature:** Add live and per-server photos/minute metrics to the treemoissa CLI

---

## Summary

Add throughput visibility to treemoissa:
- **Live**: overall photos/minute displayed in the Rich progress bar (both LLM and ML modes)
- **Final summary (LLM mode only)**: photos/minute broken down per server in a dedicated table

ML mode shows no additional throughput stats in the final summary.

---

## Architecture

### 1. Live Progress Bar — `PhotosPerMinuteColumn`

**File:** `treemoissa/main.py`

**New imports:**
- Add `ProgressColumn` to the existing `rich.progress` import block
- Add `from rich.text import Text`

`progress.advance(task_id)` is called with `advance=1` per image (default, existing behavior unchanged), so `task.speed` (items/second as tracked by Rich) equals images/second. The column multiplies by 60: `rate = task.speed * 60` to display images/minute.

```python
class PhotosPerMinuteColumn(ProgressColumn):
    def render(self, task) -> Text:
        if task.speed is None:
            return Text("-- img/min", style="progress.data.speed")
        rate = task.speed * 60   # task.speed is img/sec; multiply by 60 → img/min
        return Text(f"{rate:.1f} img/min", style="progress.data.speed")
```

- `task.speed is None` at startup until Rich has data; guard prevents `TypeError`.
- `task.speed == 0.0` during pipeline drain displays `"0.0 img/min"` — accurate.
- Added to `_make_progress()` after `MofNCompleteColumn`. Appears in both pipelines.

### 2. Per-Server Tracking — `LLMPool`

**File:** `treemoissa/llm_pool.py`

**New import:** `import time`

#### Data structure

`_server_stats` is internal state only (`init=False`):

```python
_server_stats: dict[str, dict] = field(default_factory=dict, init=False)
```

Note: the existing `_server_index: int = 0` field is `init=True` (a latent inconsistency in the existing code — it should be `init=False` but fixing it is out of scope). The new field uses `init=False` explicitly.

At the top of `run()`, before workers start:

```python
run_start = time.monotonic()
self._server_stats = {
    s.url: {"count": 0, "start": run_start}
    for s in self.servers
}
```

**Pre-seeding all URLs:** `_server_stats` is initialized with every URL in `self.servers`. Fallback servers also come from `self.servers` (see `server_order = self.servers[idx:] + self.servers[:idx]`), so all URLs that `_analyze_with_retry` may access as `responding_url` are guaranteed to be present in `_server_stats` — no `KeyError` is possible.

**Metric definition:** `elapsed = now - run_start` is the total run wall time (same for all servers). This is a deliberate simplification: img/min in the summary means "images processed by this server per minute of total run duration". A server that processes 10 images in the first 10 seconds of a 60-second run will show `10 img/min`, not `60 img/min`. This tradeoff is communicated to users via the column label `img/min (avg)` in the table, which signals that the rate is an average over the full run rather than a peak rate.

#### `_analyze_with_retry` restructuring

**Signature change:** adds `stats_lock: asyncio.Lock` as a new parameter (the same lock object already held by `_worker` — forwarded, not a new lock).

**Return type:** unchanged — `list[LLMCarResult] | None`.

The current code returns directly inside `async with semaphores[...]`. The restructuring captures the result and breaks out of the semaphore before updating stats:

```python
result = None
responding_url = None

# Primary server
async with semaphores[primary.url]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await analyze_image(image_path, client=client, server_url=primary.url)
            responding_url = primary.url
            break
        except (httpx.HTTPStatusError, httpx.RequestError):
            if attempt < MAX_RETRIES:
                continue

if responding_url is not None:
    async with stats_lock:
        self._server_stats[responding_url]["count"] += 1
    return result

# Fallback servers
for server in fallbacks:
    async with semaphores[server.url]:
        try:
            result = await analyze_image(image_path, client=client, server_url=server.url)
            responding_url = server.url
            break
        except (httpx.HTTPStatusError, httpx.RequestError):
            continue

if responding_url is not None:
    async with stats_lock:
        self._server_stats[responding_url]["count"] += 1
    return result

return None
```

**Lock order:** semaphore is always released (exiting `async with`) before `stats_lock` is acquired. `_worker` acquires `stats_lock` without holding any semaphore. This ordering is consistent and deadlock-free.

#### Call-site update: `_worker`

**`_worker` call site changes** — it forwards the existing `stats_lock` as a new argument:

```python
results = await self._analyze_with_retry(
    image_path, client, semaphores, server_order, stats_lock,   # stats_lock added
)
```

The return type and all downstream code in `_worker` (result handling, copy_image calls, stats updates) are unchanged.

#### Elapsed computation in `run()`

After `await asyncio.gather(*workers)`:

```python
now = time.monotonic()
server_stats_out = {
    url: {"count": d["count"], "elapsed": now - d["start"]}
    for url, d in self._server_stats.items()
    if d["count"] > 0
}
stats["server_stats"] = server_stats_out
```

Servers with `count == 0` are excluded. This is intentional — zero-image servers provide no useful throughput data.

**`run_pipeline()` return contract:** `server_stats` (and `brand_counts`) are popped by `_run_llm_pipeline` before `stats` is returned. The dict returned by `run_pipeline()` contains only: `total_images`, `total_cars`, `copies`, `no_car`. Neither `server_stats` nor `brand_counts` is present in the return value — this is unchanged from current behavior for `brand_counts` and new behavior for `server_stats`.

### 3. Final Summary — `_print_summary` update

**File:** `treemoissa/main.py`

Updated signature with explicit `*` separator (makes `server_stats` keyword-only; fully backwards-compatible):

```python
def _print_summary(
    stats: dict,
    brand_counts: dict[str, int],
    *,
    server_stats: dict | None = None,
) -> None:
```

If `server_stats` is present and non-empty, a third Rich table is rendered after the brand table, rows sorted descending by count:

```
┌──────────────────────────────────────┐
│        Throughput by Server          │
├──────────────────┬────────┬──────────┤
│ Server           │ Images │img/min(avg)│
├──────────────────┼────────┼──────────┤
│ localhost:8080   │    142 │   18.3   │
│ 192.168.1.10:8080│     98 │   12.7   │
└──────────────────┴────────┴──────────┘
```

Rate computation (guards evaluated before division):

```python
elapsed = d["elapsed"]
if elapsed == 0.0:
    rate_str = "--"      # ZeroDivisionError guard
elif elapsed < 1.0:
    rate_str = "--"      # data-quality guard: sub-second runs produce unreliable rates
else:
    rate_str = f"{d['count'] / elapsed * 60:.1f}"
```

Since `elapsed` is total run wall time, a sub-second run means all servers show `"--"`. This is intentional.

#### `_run_llm_pipeline` changes

Early-return path (no images): **unchanged** — `_print_summary` not called.

Normal path:

```python
brand_counts = stats.pop("brand_counts", {})
server_stats = stats.pop("server_stats", None)
_print_summary(stats, brand_counts, server_stats=server_stats)
```

`_run_ml_pipeline` is **unchanged**.

---

## Edge Cases

| Case | Behavior |
|------|----------|
| Server with `count == 0` | Excluded from table (intentional) |
| `elapsed == 0.0` | `"--"` (ZeroDivisionError guard, before division) |
| `elapsed < 1.0` (total run < 1s) | All servers `"--"` (intentional) |
| `task.speed is None` | `"-- img/min"` (no crash) |
| `task.speed == 0.0` | `"0.0 img/min"` (accurate) |
| KeyboardInterrupt | No change to existing behavior |
| Single server | Table renders with one row |
| Primary succeeds on retry (attempt 1 or 2) | Credited to `primary.url` |
| Fallback handles image | Credited to responding fallback URL |
| No images found (early return) | `_print_summary` not called; unchanged |
| `server_stats` not in stats | `pop(..., None)` → `None`; table omitted |
| `run_pipeline()` return value | Contains only `total_images`, `total_cars`, `copies`, `no_car` |

---

## Files Changed

| File | Change |
|------|--------|
| `treemoissa/main.py` | Add `ProgressColumn` + `Text` imports; add `PhotosPerMinuteColumn`; update `_make_progress()`; add `*` to `_print_summary()` signature; update `_run_llm_pipeline()` |
| `treemoissa/llm_pool.py` | Add `import time`; add `_server_stats` field (`init=False`); restructure `_analyze_with_retry()` (semaphore exit before stats update, new `stats_lock` param); update `_worker` call site (add `stats_lock` arg); compute `server_stats` in `run()` |

---

## Non-Goals

- No per-model tracking for ML mode
- No historical throughput logging to file
- No `--no-stats` flag to suppress the new table
- No fix for `_server_index` `init=True` inconsistency (deferred)
