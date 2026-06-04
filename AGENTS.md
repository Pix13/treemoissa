# AGENTS.md

Non-obvious context for agents working in the treemoissa repository.

## Essential Commands

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests (requires `pytest-asyncio`) |
| `ruff check .` | Lint (E, F, W, I rules; line length 99) |
| `ruff format .` | Format code |
| `pip install -e .` | Install in editable mode (core deps only) |
| `pip install -e ".[ml]"` | Install with ML deps (torch, ultralytics, transformers) |
| `runserver` | Download llama.cpp + Qwen3.5 GGUF model, launch local LLM server |
| `treemoissa INPUT_DIR OUTPUT_DIR` | Run LLM mode (default, needs `runserver` in another shell) |
| `treemoissa INPUT_DIR OUTPUT_DIR --model yolov8m` | Run ML mode (YOLO+ViT, needs `treemoissa[ml]`) |

## Architecture Overview

Two independent pipelines share the same CLI but use completely different code paths:

**LLM Mode** (default — `treemoissa/llm_analyzer.py`, `llm_pool.py`):
- Sends each image as base64 to a llama.cpp server running Qwen3.5 (vision model)
- LLM returns JSON with brand/model/color per car
- Async worker pool (`LLMPool`) distributes images across multiple servers with round-robin and retry/fallback
- `CarRegistry` maintains a sliding window of recently seen cars, injected as context into subsequent LLM prompts for consistency
- The `chat_template_kwargs: {enable_thinking: False}` flag is critical — without it, Qwen3's thinking mode consumes all tokens before producing JSON output

**ML Mode** (`--model` flag — `detector.py`, `classifier.py`, `color.py`):
- YOLO (ultralytics) detects car bounding boxes → crops them
- ViT (transformers, Stanford Cars dataset) classifies each crop to brand+model
- HSV-based color extraction on the center 60% of each crop
- Sequential, synchronous, single-image-at-a-time processing

Both pipelines write to the same output structure: `output_dir/brand/model/color/image.jpg`

## Module Responsibilities

| Module | Role |
|--------|------|
| `main.py` | CLI entrypoint; dispatches to LLM or ML pipeline |
| `llm_analyzer.py` | Single-image LLM call (base64 encode, HTTP request, JSON parse) |
| `llm_pool.py` | Async worker pool; server round-robin, retry, fallback, stats |
| `registry.py` | Thread-safe LRU cache of identified cars for context injection |
| `organizer.py` | `copy_image()` — creates `brand/model/color/` dirs and copies files |
| `detector.py` | YOLO-based car detection (COCO classes 2, 5, 7) |
| `classifier.py` | ViT classification + Stanford Cars label parsing |
| `color.py` | HSV-based dominant color extraction |
| `runserver.py` | Auto-download llama.cpp + GGUF model, launch server |
| `utils.py` | `_sanitize()`, `is_wsl()`, `wsl_keep_awake()` |

## Key Gotchas

### No Hardlinks (NFS Compatibility)
`organizer.copy_image()` always uses `shutil.copy2` (not `os.link`). The tool must work over NFS-mounted directories, so hardlinks are never an option. If a photo contains multiple cars, it is **copied** to each `brand/model/color/` subdirectory.

### WSL2 Path Translation
Under WSL2, `runserver` uses the **Windows** `llama-server.exe` (not a Linux binary). Paths to the GGUF model and mmproj must be converted via `wslpath -w` before passing to the Windows process. The server binds to `0.0.0.0` (not `127.0.0.1`) so it's accessible from both WSL and Windows.

### WSL2 Sleep Prevention
`main()` wraps the pipeline in `wsl_keep_awake()` which launches a PowerShell process calling `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`. This prevents Windows from sleeping during long batch runs. Only active when `is_wsl()` is true.

### Thinking Mode Must Be Disabled
The LLM payload includes `chat_template_kwargs: {enable_thinking: False}` AND the user prompt ends with `/no_think`. Both are needed — the system prompt also explicitly says "Do NOT use thinking mode." Without this, Qwen3 produces `<thinking>` blocks that consume the entire 512-token budget before any JSON output is generated.

### Server Fallback Order
`LLMPool` rotates the primary server on each image (round-robin via `_server_index`). If the primary fails after 2 retries, it tries each remaining server once. The server that responds gets credit in `_server_stats`.

### CarRegistry Locking
`CarRegistry.add()` uses `asyncio.Lock` but `snapshot()` does not — it relies on the fact that `snapshot()` has no `await` points, so the event loop cannot interleave it with `add()` on the same coroutine. This is a documented optimization, not a bug.

### ML Model Auto-Download
YOLO weights and the ViT model are auto-downloaded on first use (ultralytics and transformers handle caching). No pre-download step needed for ML mode.

### Image Scanning is Non-Recursive
`gather_images()` only scans the top-level input directory — subdirectories are ignored. Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`.

## Testing Patterns

- Tests use `httpx.MockTransport` to mock LLM server responses (no real server needed)
- `pytest.mark.asyncio` decorates all async tests
- Test images are created with `PIL.Image.new()` — tiny 10×10 pixel placeholders
- `test_main_cli.py` verifies CLI arg parsing and that `main.py` imports without torch installed
- `test_llm_pool.py` covers retry logic, server fallback, and round-robin distribution

## Code Style

- Python 3.10+, `from __future__ import annotations` in every module
- PEP8 strict, ruff line-length 99
- Type hints on all public functions
- `_prefix` for private functions and module-level constants
- Dataclasses for simple data carriers (`LLMCarResult`, `DetectedCar`, `ServerConfig`)
- Rich console for all user-facing output (progress bars, tables, status messages)
