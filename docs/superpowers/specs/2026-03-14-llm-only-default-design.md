# Design: LLM-only default mode with parallel multi-server support

**Date:** 2026-03-14
**Status:** Approved

## Context

treemoissa currently requires heavy ML dependencies (PyTorch, ultralytics, transformers) even when using the LLM pipeline. The `--llm` flag is required to opt into LLM mode. Users want a lightweight install that only needs the LLM server, and the ability to distribute work across multiple LLM servers.

## Design

### 1. Dependencies split

`pyproject.toml` is split into two profiles:

```
pip install treemoissa          # LLM-only, lightweight
pip install treemoissa[ml]      # + YOLO+ViT pipeline
```

**Core (always installed):** `Pillow`, `numpy`, `rich`, `httpx`
**Optional `[ml]`:** `torch`, `torchvision`, `ultralytics`, `transformers`, `huggingface-hub`

If the user specifies `--model yolov8l` without `[ml]` installed, treemoissa prints a clear error and exits.

### 2. CLI interface

**Removed:** `--llm`, `--llm-url`
**Added:** `--llm-host` (default: `localhost:8080`), `--llm-concurrency` (default: `1`)
**Preserved:** `--model`, `--confidence` (reserved for `[ml]` mode)

```bash
# LLM mode (default) — local server
treemoissa /photos /sorted

# LLM on a remote machine
treemoissa /photos /sorted --llm-host 192.168.1.10:8080

# Multiple servers in parallel
treemoissa /photos /sorted --llm-host 192.168.1.10:8080,192.168.1.11:8080

# YOLO+ViT pipeline (requires treemoissa[ml])
treemoissa /photos /sorted --model yolov8l
```

The presence of `--model` switches to the YOLO+ViT pipeline. Without `--model`, it is always LLM mode.

### 3. Async parallel LLM pipeline

An `asyncio.Semaphore` per server limits concurrent requests (configurable via `--llm-concurrency`, default: `1`). A pool of async coroutines processes photos from an `asyncio.Queue`.

```
Photos → asyncio.Queue
              ↓
    [worker-1] [worker-2] [worker-3] ...
         ↓          ↓          ↓
    srv:8080    srv:8081    srv:8080   ← semaphore per server
```

**Number of workers** = `nb_servers × llm_concurrency`

**Progress:** Rich progress bar updated as completions arrive (not at the end).

**Retry strategy:**

1. Send to chosen server
2. Failure → retry 1 on same server
3. Failure → retry 2 on same server
4. Failure → try next server in the list (round-robin through remaining servers)
5. If all servers fail → copy to `unknown/unknown/unknown/` + warning

### 4. Code changes

| File | Change |
|---|---|
| `pyproject.toml` | Split deps into core / `[ml]` extra |
| `main.py` | Remove `--llm`/`--llm-url`, add `--llm-host`/`--llm-concurrency`, auto-detect mode based on `--model` presence |
| `llm_analyzer.py` | Refactor to async (`analyze_image` → `async def`), add retry + server fallback logic |
| `detector.py` | Guard torch import behind `try/except ImportError` with clear message |
| `classifier.py` | Same guard |

**New file:** `llm_pool.py` — manages the server pool, per-server semaphores, photo queue, and orchestrates async workers. Isolates all parallelism logic from `main.py`.

### 5. README update

- LLM mode presented as the default mode
- Examples with `--llm-host` (single and multi-server)
- Installation section: `pip install treemoissa` (default) vs `pip install treemoissa[ml]`
- `--model` documented as an alternative requiring `[ml]`
- Remove `--llm` and `--llm-url` references
