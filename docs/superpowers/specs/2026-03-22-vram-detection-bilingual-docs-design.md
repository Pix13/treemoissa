# Design: VRAM Auto-Detection & Bilingual Documentation

**Date:** 2026-03-22
**Status:** Approved

## Overview

Four changes in one spec:
1. Bilingual README (French first, English second, single file)
2. WSL2 activation + CUDA WSL driver installation instructions
3. Two-shell workflow documentation (runserver + treemoissa)
4. Automatic VRAM detection and Qwen3.5 model selection in `runserver`

## 1. Bilingual README Structure

Single `README.md` with this layout:

```
# treemoissa

[Français](#français) | [English](#english)

---

## Français
(full French content including WSL2/CUDA setup, two-shell workflow)

---

## English
(full English content, updated to match)
```

French section appears first. Both sections contain identical information.

## 2. WSL2 + CUDA Documentation

New section in both languages covering:

1. **Enable WSL2**: `wsl --install` from admin PowerShell, reboot, verify with `wsl --version`
2. **CUDA driver for WSL**: Download the standard NVIDIA Windows driver (not the Linux driver — WSL2 accesses the GPU through the Windows host driver). Link to https://developer.nvidia.com/cuda/wsl
3. **Verify**: Run `nvidia-smi` inside WSL to confirm GPU visibility
4. **Mirrored networking** (already documented): `.wslconfig` with `networkingMode=mirrored`

## 3. Two-Shell Workflow

Dedicated section with ASCII diagram:

```
Terminal 1 (LLM server):           Terminal 2 (photo sorting):
$ runserver                         $ treemoissa /input /output
  Server ready on port 8080...        Processing images...
```

Explains that `runserver` is a blocking process that must stay running during the sort.

## 4. VRAM Auto-Detection in `runserver.py`

### New function: `_detect_vram_mb() -> int | None`

- Runs `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
- Parses total VRAM in MB
- Returns `None` if `nvidia-smi` fails (fallback to 9B-Q4_K_M default)

### New function: `_select_best_model(vram_mb: int | None, quant: str) -> tuple[str, str, str]`

Returns `(repo_id, model_filename, display_name)`.

Model candidates table (largest to smallest):

| Model | Repo | File (Q4_K_M) | Est. Size | Min VRAM |
|---|---|---|---|---|
| 27B | unsloth/Qwen3.5-27B-GGUF | Qwen3.5-27B-Q4_K_M.gguf | ~16 GB | 18 GB |
| 9B | unsloth/Qwen3.5-9B-GGUF | Qwen3.5-9B-Q4_K_M.gguf | ~5.8 GB | 7 GB |
| 4B | unsloth/Qwen3.5-4B-GGUF | Qwen3.5-4B-Q4_K_M.gguf | ~2.8 GB | 4 GB |
| 2B | unsloth/Qwen3.5-2B-GGUF | Qwen3.5-2B-Q4_K_M.gguf | ~1.5 GB | 3 GB |
| 0.8B | unsloth/Qwen3.5-0.8B-GGUF | Qwen3.5-0.8B-Q4_K_M.gguf | ~0.6 GB | 2 GB |

Min VRAM = estimated model size + ~1.2 GB overhead (mmproj BF16 + llama.cpp KV cache + runtime).

Selection logic: iterate largest to smallest, pick the first where `min_vram <= detected_vram`.

### New CLI argument: `--quant`

- Allowed values: `Q3_K_M`, `Q3_K_S`, `Q4_0`, `Q4_1`, `Q4_K_M` (default), `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `BF16`
- Overrides the default Q4_K_M quantization
- When specified, the model filename pattern changes (e.g., `Qwen3.5-9B-Q8_0.gguf`)

### Constants change

`MODEL_REPO` and `MODEL_FILE` become dynamic — determined by `_select_best_model()` at runtime. `MMPROJ_FILE` stays constant (`mmproj-BF16.gguf`, same across all repos).

### Display at startup

```
VRAM detected: 8192 MB (NVIDIA GeForce RTX 4060 Ti)
Selected model: Qwen3.5-9B-Q4_K_M (~5.8 GB)
Use --quant to override (e.g. --quant Q8_0)
```

## Files Modified

- `README.md` — full rewrite (bilingual structure)
- `treemoissa/runserver.py` — VRAM detection, model selection, `--quant` flag
