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

A custom Rich `ProgressColumn` subclass reads `task.speed` (items/second, maintained by Rich internally) and multiplies by 60 to produce photos/minute.

```python
class PhotosPerMinuteColumn(ProgressColumn):
    def render(self, task) -> Text:
        if task.speed is None:
            return Text("-- img/min", style="progress.data.speed")
        rate = task.speed * 60
        return Text(f"{rate:.1f} img/min", style="progress.data.speed")
```

This column is added to `_make_progress()` after `MofNCompleteColumn`. It appears in both LLM and ML pipelines since both use `_make_progress()`.

### 2. Per-Server Tracking — `LLMPool`

**File:** `treemoissa/llm_pool.py`

Add a `_server_stats` dict to `LLMPool` tracking per-server activity:

```python
_server_stats: dict[str, dict] = field(default_factory=dict)
```

Initialized in `run()` before workers start:
```python
self._server_stats = {s.url: {"count": 0, "start": None, "elapsed": 0.0} for s in self.servers}
```

In `_analyze_with_retry`, when a server successfully analyzes an image:
- Record `start` time on first use (`time.monotonic()` if `start is None`)
- Increment `count`

At the end of `run()`, compute `elapsed = time.monotonic() - start` per server and include in returned stats:
```python
stats["server_stats"] = {
    url: {"count": d["count"], "elapsed": d["elapsed"]}
    for url, d in self._server_stats.items()
    if d["count"] > 0
}
```

### 3. Final Summary — `_print_summary` update

**File:** `treemoissa/main.py`

`_print_summary` receives an optional `server_stats` argument. If present and non-empty, a third Rich table is rendered:

```
┌──────────────────────────────────────┐
│        Throughput by Server          │
├──────────────────┬────────┬──────────┤
│ Server           │ Images │  img/min │
├──────────────────┼────────┼──────────┤
│ localhost:8080   │    142 │   18.3   │
│ 192.168.1.10:8080│     98 │   12.7   │
└──────────────────┴────────┴──────────┘
```

The per-server rate is computed as `count / elapsed * 60`. Servers with `elapsed < 1.0` display `--` to avoid misleading values from very short runs.

`_run_llm_pipeline` extracts `server_stats` from the returned stats dict and passes it to `_print_summary`. `_run_ml_pipeline` does not pass `server_stats`.

---

## Edge Cases

- **Server with zero images**: excluded from the server stats table (filtered by `count > 0`).
- **Elapsed < 1 second**: display `--` instead of a potentially misleading rate.
- **KeyboardInterrupt**: partial stats are discarded as per existing behavior — no change.
- **Single server**: table still renders (one row), consistent with multi-server.

---

## Files Changed

| File | Change |
|------|--------|
| `treemoissa/main.py` | Add `PhotosPerMinuteColumn`; update `_make_progress()`; update `_print_summary()` signature; update `_run_llm_pipeline()` to pass server stats |
| `treemoissa/llm_pool.py` | Add `_server_stats` tracking in `_analyze_with_retry()` and `run()` |

---

## Non-Goals

- No throughput tracking for ML mode per-model (not requested)
- No historical throughput logging to file
- No `--no-stats` flag to suppress the new table
