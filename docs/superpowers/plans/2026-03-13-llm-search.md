# LLM Search — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--llm` mode to treemoissa that sends full photos to a local Qwen3.5-9B vision model (served by llama.cpp) to identify all cars with brand/model/color in a single call, replacing the YOLO+ViT+HSV pipeline.

**Architecture:** Two new components: (1) `runserver` CLI tool that downloads llama.cpp + GGUF model and launches the server, (2) `llm_analyzer` module that calls the llama.cpp OpenAI-compatible API with base64 images. The existing `main.py` gains a `--llm` flag that switches to this new pipeline while reusing the organizer and progress display.

**Tech Stack:** llama.cpp server (pre-built binary from GitHub releases), huggingface-hub (GGUF download), httpx (HTTP client for API calls), unsloth/Qwen3.5-9B-GGUF Q4_1.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `treemoissa/runserver.py` | Create | Download llama.cpp + GGUF, launch server |
| `treemoissa/llm_analyzer.py` | Create | Send images to llama.cpp API, parse JSON response |
| `treemoissa/main.py` | Modify | Add `--llm` flag, LLM pipeline branch |
| `treemoissa/classifier.py` | Modify | Extract `_sanitize` to shared util or re-import |
| `pyproject.toml` | Modify | Add deps (huggingface-hub, httpx), add `runserver` script |
| `README.md` | Modify | Document `--llm` mode and `runserver` |

---

## Chunk 1: runserver tool

### Task 1: Create `runserver.py` — download logic

**Files:**
- Create: `treemoissa/runserver.py`

- [ ] **Step 1: Create runserver.py with download helpers**

```python
"""Download llama.cpp and GGUF model, then launch llama-server."""

from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download
from rich.console import Console

console = Console()

CACHE_DIR = Path.home() / ".cache" / "treemoissa"
LLAMA_DIR = CACHE_DIR / "llama-server"
MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE = "Qwen3.5-9B-Q4_1.gguf"
DEFAULT_PORT = 8080
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _get_llama_server_path() -> Path:
    """Return path to llama-server binary, downloading if absent."""
    # Look for existing binary
    server_bin = LLAMA_DIR / "llama-server"
    if server_bin.exists():
        console.print(f"[green]llama-server found:[/green] {server_bin}")
        return server_bin

    console.print("[bold]Downloading latest llama.cpp release...[/bold]")
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch latest release info
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        resp = client.get(GITHUB_API, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        release = resp.json()

    tag = release["tag_name"]
    # Find the ubuntu x64 asset
    asset_name = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    asset_url = None
    for asset in release["assets"]:
        if asset["name"] == asset_name:
            asset_url = asset["browser_download_url"]
            break

    if asset_url is None:
        console.print(f"[bold red]Could not find asset {asset_name} in release {tag}[/bold red]")
        console.print("Available assets:")
        for asset in release["assets"]:
            if "ubuntu" in asset["name"]:
                console.print(f"  - {asset['name']}")
        sys.exit(1)

    console.print(f"[bold]Downloading:[/bold] {asset_name} ({tag})")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        resp = client.get(asset_url)
        resp.raise_for_status()

    # Extract llama-server binary from tarball
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("/llama-server") or member.name == "llama-server":
                member.name = "llama-server"
                tar.extract(member, path=LLAMA_DIR)
                break
        else:
            # Try alternate: binary might be named differently
            names = [m.name for m in tar.getmembers()]
            console.print(f"[bold red]llama-server not found in tarball.[/bold red]")
            console.print(f"Contents: {names[:20]}")
            sys.exit(1)

    # Make executable
    server_bin.chmod(server_bin.stat().st_mode | stat.S_IEXEC)
    console.print(f"[green]llama-server installed:[/green] {server_bin}")
    return server_bin


def _get_model_path() -> Path:
    """Return path to GGUF model, downloading if absent."""
    console.print(f"[bold]Checking model:[/bold] {MODEL_REPO} / {MODEL_FILE}")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=CACHE_DIR / "models",
    )
    console.print(f"[green]Model ready:[/green] {path}")
    return Path(path)


def main() -> None:
    """Main entrypoint for runserver CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="runserver",
        description="Download and launch llama.cpp server with Qwen3.5-9B vision model.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port for the llama-server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--gpu-layers", "-ngl", type=int, default=99,
        help="Number of layers to offload to GPU (default: 99 = all)",
    )
    parser.add_argument(
        "--ctx-size", "-c", type=int, default=4096,
        help="Context size (default: 4096)",
    )
    args = parser.parse_args()

    server_bin = _get_llama_server_path()
    model_path = _get_model_path()

    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "--port", str(args.port),
        "-ngl", str(args.gpu_layers),
        "-c", str(args.ctx_size),
    ]

    console.print(f"\n[bold green]Starting llama-server on port {args.port}...[/bold green]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")

    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test runserver imports and arg parsing**

Run: `cd /home/pix/treemoissa-workdir && python -c "from treemoissa.runserver import main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add treemoissa/runserver.py
git commit -m "feat: add runserver tool for llama.cpp + GGUF download"
```

---

### Task 2: Register `runserver` CLI and add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Add to `[project.scripts]`:
```toml
runserver = "treemoissa.runserver:main"
```

Add to `dependencies`:
```toml
"huggingface-hub>=0.20.0",
"httpx>=0.27.0",
```

- [ ] **Step 2: Reinstall package**

Run: `cd /home/pix/treemoissa-workdir && pip install -e .`
Expected: successful install

- [ ] **Step 3: Verify CLI entry**

Run: `runserver --help`
Expected: shows help with `--port`, `--gpu-layers`, `--ctx-size` flags

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: register runserver CLI and add httpx/huggingface-hub deps"
```

---

## Chunk 2: LLM analyzer module

### Task 3: Create `llm_analyzer.py`

**Files:**
- Create: `treemoissa/llm_analyzer.py`

- [ ] **Step 1: Create llm_analyzer.py**

```python
"""Analyze car photos using a vision LLM served by llama.cpp."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from treemoissa.classifier import _sanitize

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
- Multiple cars: [{"brand": "porsche", "model": "911", "color": "red"}, {"brand": "toyota", "model": "supra", "color": "silver"}]
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
    # Try to extract JSON array from response
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


def analyze_image(
    image_path: Path,
    server_url: str = DEFAULT_URL,
    timeout: float = 120.0,
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

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{server_url}/v1/chat/completions", json=payload)
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

- [ ] **Step 2: Verify imports**

Run: `cd /home/pix/treemoissa-workdir && python -c "from treemoissa.llm_analyzer import analyze_image, LLMCarResult; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add treemoissa/llm_analyzer.py
git commit -m "feat: add LLM analyzer module for vision-based car identification"
```

---

## Chunk 3: Integrate `--llm` into main pipeline

### Task 4: Modify `main.py` to support `--llm` flag

**Files:**
- Modify: `treemoissa/main.py`

- [ ] **Step 1: Add `--llm` and `--llm-url` arguments to argparse**

After the `--model` argument block, add:

```python
parser.add_argument(
    "--llm",
    action="store_true",
    default=False,
    help="Use LLM vision model instead of YOLO+ViT (requires running runserver)",
)
parser.add_argument(
    "--llm-url",
    type=str,
    default="http://localhost:8080",
    help="URL of the llama.cpp server (default: http://localhost:8080)",
)
```

- [ ] **Step 2: Add LLM pipeline branch in `run_pipeline`**

Modify `run_pipeline` signature to accept `llm` and `llm_url` parameters:

```python
def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    conf: float = 0.35,
    model_key: str | None = None,
    *,
    llm: bool = False,
    llm_url: str = "http://localhost:8080",
) -> dict:
```

At the start of `run_pipeline`, before device/model selection, add an LLM branch:

```python
if llm:
    return _run_llm_pipeline(input_dir, output_dir, llm_url)
```

- [ ] **Step 3: Implement `_run_llm_pipeline` function**

Add before `run_pipeline`:

```python
def _run_llm_pipeline(
    input_dir: Path,
    output_dir: Path,
    llm_url: str,
) -> dict:
    """Run the LLM-based pipeline (no YOLO/ViT/HSV)."""
    from treemoissa.llm_analyzer import analyze_image

    images = gather_images(input_dir)
    if not images:
        console.print("[bold red]No images found in input directory.")
        return {"total_images": 0, "total_cars": 0, "copies": 0}

    console.print(f"[bold]Mode:[/bold] LLM vision ({llm_url})")
    console.print(f"[bold]Found {len(images)} images to process.[/bold]\n")

    stats = {"total_images": len(images), "total_cars": 0, "copies": 0, "no_car": 0}
    brand_counts: dict[str, int] = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Processing images", total=len(images))

        for image_path in images:
            progress.update(task, description=f"[cyan]{image_path.name}")

            try:
                cars = analyze_image(image_path, server_url=llm_url)
            except Exception as e:
                console.print(f"[yellow]Warning: failed to process {image_path.name}: {e}")
                progress.advance(task)
                continue

            if not cars:
                # No car detected → copy to unknown
                copy_image(image_path, output_dir, "unknown", "unknown", "unknown")
                stats["no_car"] += 1
                stats["copies"] += 1
                progress.advance(task)
                continue

            stats["total_cars"] += len(cars)

            seen: set[tuple[str, str, str]] = set()
            for car in cars:
                key = (car.brand, car.model, car.color)
                if key in seen:
                    continue
                seen.add(key)

                copy_image(image_path, output_dir, car.brand, car.model, car.color)
                stats["copies"] += 1
                brand_counts[car.brand] = brand_counts.get(car.brand, 0) + 1

            progress.advance(task)

    # Summary tables (same as standard pipeline)
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

    return stats
```

- [ ] **Step 4: Update `main()` to pass `--llm` flags**

Change the `run_pipeline` call in `main()`:

```python
run_pipeline(
    args.input_dir,
    args.output_dir,
    conf=args.confidence,
    model_key=args.model,
    llm=args.llm,
    llm_url=args.llm_url,
)
```

- [ ] **Step 5: Verify CLI help**

Run: `cd /home/pix/treemoissa-workdir && python -m treemoissa.main --help`
Expected: shows `--llm` and `--llm-url` flags

- [ ] **Step 6: Commit**

```bash
git add treemoissa/main.py
git commit -m "feat: add --llm flag to use vision LLM instead of YOLO+ViT pipeline"
```

---

## Chunk 4: Documentation update

### Task 5: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Features section**

Add to the features list:
```markdown
- **LLM mode** (`--llm`): send full photos to a local Qwen3.5-9B vision model for brand/model/color identification in a single pass
- Includes `runserver` tool to auto-download llama.cpp and the GGUF model
```

- [ ] **Step 2: Add LLM Mode section after Usage**

Add a new `## LLM Mode` section documenting `runserver` and `--llm` usage:

```markdown
## LLM Mode

As an alternative to the YOLO + ViT pipeline, treemoissa can use a local vision LLM
to identify cars. This sends each photo to a Qwen3.5-9B model served by llama.cpp.

### 1. Start the server

In a dedicated terminal:

\`\`\`bash
runserver
\`\`\`

On first run, this downloads:
- **llama.cpp** server binary (~latest release from GitHub)
- **Qwen3.5-9B Q4_1** GGUF model (~5.8 GB from HuggingFace)

Files are cached in `~/.cache/treemoissa/`.

| Option | Description |
|---|---|
| `--port` | Server port (default: `8080`) |
| `--gpu-layers` / `-ngl` | GPU layers to offload (default: `99` = all) |
| `--ctx-size` / `-c` | Context size (default: `4096`) |

### 2. Run treemoissa with `--llm`

In another terminal:

\`\`\`bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --llm
\`\`\`

| Option | Description |
|---|---|
| `--llm` | Use the LLM vision model instead of YOLO + ViT |
| `--llm-url` | Server URL (default: `http://localhost:8080`) |

In LLM mode, photos with no detected car are copied to `unknown/unknown/unknown`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add LLM mode and runserver documentation"
```

---

## Chunk 5: Final verification and push

### Task 6: End-to-end verification

- [ ] **Step 1: Lint check**

Run: `cd /home/pix/treemoissa-workdir && ruff check .`
Expected: no errors

- [ ] **Step 2: Import check all modules**

Run: `python -c "from treemoissa.runserver import main; from treemoissa.llm_analyzer import analyze_image; from treemoissa.main import run_pipeline; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Verify both CLIs**

Run: `treemoissa --help && runserver --help`
Expected: both show help with correct flags

- [ ] **Step 4: Push branch**

```bash
git push -u origin llm-search
```
