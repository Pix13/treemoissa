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

`runserver` remains in core: `huggingface-hub` is moved to core dependencies since it is lightweight (~2 MB) and required for the `runserver` command which is part of the LLM workflow.

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

### 3. Lazy imports for ML dependencies

`main.py` currently has top-level `import torch` and imports from `detector.py` / `classifier.py` which also import `torch`, `ultralytics`, and `transformers` at module level. This would crash on startup without `[ml]`.

**Solution:** All ML-related imports become lazy, loaded only when `--model` is specified:

- `main.py`: remove top-level `import torch` and imports from `detector`/`classifier`. Move them inside the `_run_ml_pipeline()` function (renamed from current YOLO+ViT path).
- `AVAILABLE_MODELS` dict: duplicated as a plain constant in `main.py` for argparse `--model` choices (no dependency on `detector.py` at import time).
- `pick_device()` (uses `torch.cuda.is_available()`): only called from within ML pipeline, not at module level.
- `detector.py` / `classifier.py`: keep their top-level imports as-is since they are only imported lazily.

### 4. Async parallel LLM pipeline

An `asyncio.Semaphore` per server limits concurrent requests (configurable via `--llm-concurrency`, default: `1`). A pool of async coroutines processes photos from an `asyncio.Queue`.

```
Photos → asyncio.Queue
              ↓
    [worker-1] [worker-2] [worker-3] ...
         ↓          ↓          ↓
    srv:8080    srv:8081    srv:8080   ← semaphore per server
```

**Number of workers** = `nb_servers × llm_concurrency`

**Server assignment:** Workers dynamically acquire any server's semaphore. Each worker loops: dequeue a photo, try to acquire any available server semaphore (trying servers in round-robin order from a rotating index), process, release semaphore, repeat.

**Progress:** Rich progress bar updated as completions arrive (not at the end).

**Retry strategy:**

1. Send to acquired server
2. Failure → retry 1 on same server
3. Failure → retry 2 on same server
4. Failure → release semaphore, try each remaining server once (acquire semaphore, single attempt, release)
5. If all servers fail → copy to `unknown/unknown/unknown/` + warning

**Timeouts:**

- Connect timeout: 5 seconds (fail fast on unreachable servers)
- Read timeout: 120 seconds (LLM inference can be slow)

**HTTP client lifecycle:** One `httpx.AsyncClient` shared across all workers, created at pool startup, closed at shutdown. Connection pooling handled by httpx internally.

**Blocking I/O:** File copies from async workers use `await asyncio.to_thread(shutil.copy2, ...)` to avoid blocking the event loop.

**Async integration:** `main.py` calls `asyncio.run(llm_pool.run(photos, servers, concurrency, progress_callback))`. The `llm_pool.run()` function is the top-level async entry point. Rich progress bar is passed via callback.

**Graceful shutdown:** On `KeyboardInterrupt`, workers drain the queue (no new work), cancel in-flight requests, and print a partial summary of what was processed.

### 5. Code changes

| File | Change |
|---|---|
| `pyproject.toml` | Split deps into core / `[ml]` extra, move `huggingface-hub` to core |
| `main.py` | Remove `--llm`/`--llm-url`, add `--llm-host`/`--llm-concurrency`, lazy ML imports, auto-detect mode based on `--model` presence, call `asyncio.run()` for LLM pipeline. When `--model` is used without `[ml]` installed, catch `ImportError` and print: `"Error: --model requires ML dependencies. Install with: pip install treemoissa[ml]"` then `sys.exit(1)`. |
| `llm_analyzer.py` | Refactor to async (`analyze_image` → `async def`), accept `httpx.AsyncClient` as parameter instead of creating its own client, add retry + server fallback logic |
| `detector.py` | No changes (imported lazily) |
| `classifier.py` | Import `_sanitize` from `utils.py` instead of defining locally (imported lazily) |
| `color.py` | No changes (only uses PIL/numpy which are core deps) |

**New file:** `utils.py` — extract `_sanitize()` from `classifier.py` into this dependency-free module. Both `classifier.py` and `llm_analyzer.py` import from here. This breaks the transitive `torch` import chain that would otherwise pull ML deps in LLM-only mode.

**New file:** `llm_pool.py` — manages the server pool, per-server semaphores, photo queue, and orchestrates async workers. Isolates all parallelism logic from `main.py`. Handles graceful shutdown on cancellation (workers check a `stop` event, in-flight httpx requests are cancelled via `asyncio.Task.cancel()`).

### 6. README update

- LLM mode presented as the default mode
- Installation section: `pip install treemoissa` (default) vs `pip install treemoissa[ml]`
- Examples with `--llm-host` (single and multi-server)
- `--model` documented as an alternative requiring `[ml]`
- Remove `--llm` and `--llm-url` references
- WSL2 mirrored networking note preserved
