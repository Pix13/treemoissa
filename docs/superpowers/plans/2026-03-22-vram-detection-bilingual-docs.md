# VRAM Auto-Detection & Bilingual Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add VRAM auto-detection for model selection in `runserver`, and rewrite README as bilingual French-first documentation with WSL2/CUDA setup and two-shell workflow.

**Architecture:** Two independent changes — (1) `runserver.py` gains `_detect_vram_mb()` and `_select_best_model()` functions plus a `--quant` CLI flag; static `MODEL_REPO`/`MODEL_FILE` constants become dynamic. (2) `README.md` is rewritten as bilingual French/English with new WSL2 setup and two-shell workflow sections.

**Tech Stack:** Python 3.10+, subprocess (`nvidia-smi`), argparse, huggingface_hub, pytest

---

### Task 1: VRAM detection function

**Files:**
- Modify: `treemoissa/runserver.py:1-27` (add function after imports)
- Test: `tests/test_runserver.py` (create)

- [ ] **Step 1: Write the failing test for `_detect_vram_mb`**

Create `tests/test_runserver.py`:

```python
"""Tests for runserver VRAM detection and model selection."""

from unittest.mock import patch, MagicMock
import subprocess

from treemoissa.runserver import _detect_vram_mb


def test_detect_vram_parses_nvidia_smi():
    """_detect_vram_mb returns VRAM in MB when nvidia-smi succeeds."""
    mock_result = MagicMock()
    mock_result.stdout = "8192\n"
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = _detect_vram_mb()
        assert result == 8192
        mock_run.assert_called_once()


def test_detect_vram_returns_none_on_failure():
    """_detect_vram_mb returns None when nvidia-smi is not available."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = _detect_vram_mb()
        assert result is None


def test_detect_vram_returns_none_on_error():
    """_detect_vram_mb returns None when nvidia-smi returns an error."""
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
        result = _detect_vram_mb()
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runserver.py -v`
Expected: FAIL — `_detect_vram_mb` does not exist yet

- [ ] **Step 3: Implement `_detect_vram_mb`**

Add to `treemoissa/runserver.py` after the existing constants block (after line 26):

```python
def _detect_vram_mb() -> int | None:
    """Detect total GPU VRAM in MB via nvidia-smi. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runserver.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_runserver.py treemoissa/runserver.py
git commit -m "feat: add VRAM detection function via nvidia-smi"
```

---

### Task 2: Model selection function

**Files:**
- Modify: `treemoissa/runserver.py` (add model table + selection function)
- Test: `tests/test_runserver.py` (add tests)

- [ ] **Step 1: Write failing tests for `_select_best_model`**

Append to `tests/test_runserver.py`:

```python
from treemoissa.runserver import _select_best_model, MODEL_CANDIDATES


def test_select_model_8gb_picks_9b():
    """8 GB VRAM should select the 9B model."""
    repo, filename, display = _select_best_model(8192, "Q4_K_M")
    assert "9B" in repo
    assert "Q4_K_M" in filename


def test_select_model_24gb_picks_27b():
    """24 GB VRAM should select the 27B model."""
    repo, filename, display = _select_best_model(24576, "Q4_K_M")
    assert "27B" in repo


def test_select_model_3gb_picks_4b():
    """4 GB VRAM should select the 4B model."""
    repo, filename, display = _select_best_model(4096, "Q4_K_M")
    assert "4B" in repo


def test_select_model_1gb_picks_smallest():
    """2 GB VRAM should select the 0.8B model."""
    repo, filename, display = _select_best_model(2048, "Q4_K_M")
    assert "0.8B" in repo


def test_select_model_none_vram_defaults_9b():
    """When VRAM is None (detection failed), default to 9B."""
    repo, filename, display = _select_best_model(None, "Q4_K_M")
    assert "9B" in repo


def test_select_model_quant_override():
    """--quant should change the filename pattern."""
    repo, filename, display = _select_best_model(8192, "Q8_0")
    assert "Q8_0" in filename
    assert "9B" in repo


def test_select_model_tiny_vram_still_returns():
    """Even very small VRAM should return the smallest model."""
    repo, filename, display = _select_best_model(512, "Q4_K_M")
    assert "0.8B" in repo
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runserver.py -v`
Expected: FAIL — `_select_best_model` and `MODEL_CANDIDATES` do not exist

- [ ] **Step 3: Implement model candidates table and `_select_best_model`**

Add to `treemoissa/runserver.py`, replacing the old `MODEL_REPO` and `MODEL_FILE` constants:

```python
# Replace these two lines:
#   MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
#   MODEL_FILE = "Qwen3.5-9B-Q4_1.gguf"
# With:

MMPROJ_FILE = "mmproj-BF16.gguf"
DEFAULT_QUANT = "Q4_K_M"

# Ordered largest to smallest. min_vram_mb includes ~1.2 GB overhead for mmproj + KV cache.
MODEL_CANDIDATES = [
    {"size": "27B", "repo": "unsloth/Qwen3.5-27B-GGUF", "min_vram_mb": 18_000},
    {"size": "9B",  "repo": "unsloth/Qwen3.5-9B-GGUF",  "min_vram_mb": 7_000},
    {"size": "4B",  "repo": "unsloth/Qwen3.5-4B-GGUF",  "min_vram_mb": 4_000},
    {"size": "2B",  "repo": "unsloth/Qwen3.5-2B-GGUF",  "min_vram_mb": 3_000},
    {"size": "0.8B","repo": "unsloth/Qwen3.5-0.8B-GGUF", "min_vram_mb": 2_000},
]

DEFAULT_MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"


def _select_best_model(vram_mb: int | None, quant: str) -> tuple[str, str, str]:
    """Select the best Qwen3.5 model that fits in available VRAM.

    Returns (repo_id, model_filename, display_name).
    """
    if vram_mb is None:
        # Fallback: use 9B when detection fails
        repo = DEFAULT_MODEL_REPO
        for c in MODEL_CANDIDATES:
            if c["repo"] == repo:
                size = c["size"]
                break
        filename = f"Qwen3.5-{size}-{quant}.gguf"
        return repo, filename, f"Qwen3.5-{size}-{quant} (default, VRAM unknown)"

    for candidate in MODEL_CANDIDATES:
        if vram_mb >= candidate["min_vram_mb"]:
            size = candidate["size"]
            filename = f"Qwen3.5-{size}-{quant}.gguf"
            return candidate["repo"], filename, f"Qwen3.5-{size}-{quant}"

    # Even smallest doesn't fit — use smallest anyway
    smallest = MODEL_CANDIDATES[-1]
    size = smallest["size"]
    filename = f"Qwen3.5-{size}-{quant}.gguf"
    return smallest["repo"], filename, f"Qwen3.5-{size}-{quant} (warning: may exceed VRAM)"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runserver.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add treemoissa/runserver.py tests/test_runserver.py
git commit -m "feat: add model selection based on VRAM detection"
```

---

### Task 3: Wire VRAM detection into `main()` and add `--quant` CLI arg

**Files:**
- Modify: `treemoissa/runserver.py:179-239` (main function)
- Modify: `treemoissa/runserver.py:156-176` (`_get_model_paths` to accept dynamic repo/file)

- [ ] **Step 1: Update `_get_model_paths` to accept repo and filename as arguments**

Change signature from:
```python
def _get_model_paths() -> tuple[Path, Path]:
```
To:
```python
def _get_model_paths(model_repo: str, model_file: str) -> tuple[Path, Path]:
```

Replace references to `MODEL_REPO` and `MODEL_FILE` inside the function with the parameters.

- [ ] **Step 2: Add `--quant` argument and wire VRAM detection into `main()`**

In `main()`, add the `--quant` arg:

```python
ALLOWED_QUANTS = [
    "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S",
    "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "BF16",
]

parser.add_argument(
    "--quant", type=str, default=DEFAULT_QUANT,
    choices=ALLOWED_QUANTS,
    help=f"Model quantization (default: {DEFAULT_QUANT})",
)
```

After parsing args, add VRAM detection and model selection:

```python
vram_mb = _detect_vram_mb()
if vram_mb is not None:
    console.print(f"[bold]VRAM detected:[/bold] {vram_mb} MB")
else:
    console.print("[yellow]Could not detect VRAM — using default model.[/yellow]")

model_repo, model_file, display_name = _select_best_model(vram_mb, args.quant)
console.print(f"[bold]Selected model:[/bold] {display_name}")
if args.quant == DEFAULT_QUANT:
    console.print(f"[dim]Use --quant to override (e.g. --quant Q8_0)[/dim]")

server_bin = _get_llama_server_path()
model_path, mmproj_path = _get_model_paths(model_repo, model_file)
```

Also update the parser description:
```python
description="Download and launch llama.cpp server with auto-selected Qwen3.5 vision model.",
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add treemoissa/runserver.py
git commit -m "feat: add --quant flag and wire VRAM auto-detection into runserver"
```

---

### Task 4: Rewrite README as bilingual French/English

**Files:**
- Modify: `README.md` (full rewrite)

- [ ] **Step 1: Write the bilingual README**

Rewrite `README.md` with the following structure:

```markdown
# treemoissa

[Francais](#francais) | [English](#english)

---

# Francais

(Full French documentation including:)
- Description du projet
- Prerequis WSL2 + CUDA
  - Activation WSL2 (wsl --install, redemarrage)
  - Installation driver NVIDIA pour WSL (lien nvidia.com/cuda/wsl)
  - Verification (nvidia-smi dans WSL)
  - Configuration reseau miroir (.wslconfig)
- Installation (clone, venv, pip install)
- Utilisation avec deux terminaux WSL
  - ASCII diagram: Terminal 1 = runserver, Terminal 2 = treemoissa
  - Explication que runserver est bloquant
- Options runserver (--port, --gpu-layers, --ctx-size, --quant)
  - Mention de la detection automatique VRAM
- Options treemoissa (--llm-host, --llm-concurrency, --model, --confidence)
- Pipeline ML (optionnel)
- Formats supportes
- Notes
- Developpement

---

# English

(Same content in English, updated with:)
- WSL2 + CUDA setup section
- Two-shell workflow with ASCII diagram
- Updated runserver section mentioning VRAM auto-detection and --quant
- All existing content preserved
```

The French section translates all existing English content and adds the new sections. The English section mirrors the French additions.

- [ ] **Step 2: Review the README renders correctly**

Run: `head -50 README.md` to verify structure.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as bilingual FR/EN with WSL2 setup and VRAM detection"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Run linter**

Run: `ruff check treemoissa/ tests/`
Expected: No errors

- [ ] **Step 3: Verify runserver --help shows new --quant option**

Run: `python -m treemoissa.runserver --help`
Expected: Shows `--quant` with choices listed

- [ ] **Step 4: Final commit if any fixups needed**
