# treemoissa

Car trackdays and shows photo triage tool — automatically sorts a flat directory of car photos into a `brand/model/color` directory tree using AI.

## Features

- **LLM vision mode (default):** sends photos to a local Qwen3.5-9B vision model for brand/model/color identification in a single pass
- **Multi-server support:** distribute work across multiple LLM servers in parallel
- **ML pipeline (optional):** detect cars with YOLOv8 / RT-DETR, classify with a Vision Transformer, extract color via HSV analysis
- If a photo contains multiple cars, it is copied into each matching subdirectory (no hardlinks — NFS-safe)
- Includes `runserver` tool to auto-download llama.cpp and the GGUF model
- Rich progress display and summary table

## Output structure

```
output_dir/
└── Toyota/
│   └── Supra/
│       └── red/
│           └── IMG_0042.jpg
└── Porsche/
    └── 911/
        └── silver/
            └── IMG_0042.jpg   ← same photo copied if two cars visible
```

## Requirements

- Python 3.10+
- WSL2 or Linux
- NVIDIA GPU with CUDA toolkit (recommended for `runserver` — GeForce RTX 4060 Ti or similar)

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd treemoissa
```

### 2. Create a virtual environment

Using `uv` (recommended):

```bash
uv venv
source .venv/bin/activate
```

Or with standard `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install treemoissa

**Lightweight (LLM only — recommended):**

```bash
pip install -e .
```

**With ML pipeline (YOLO + ViT):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[ml]"
```

Check your CUDA version with `nvcc --version` or `nvidia-smi`, then visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact PyTorch install command.

## Usage

```
treemoissa <input_dir> <output_dir> [options]
```

| Argument | Description |
|---|---|
| `input_dir` | Flat directory containing your car photos |
| `output_dir` | Destination directory for the organized tree (created if absent) |
| `--llm-host` | LLM server(s) as `ip:port,ip:port` (default: `localhost:8080`) |
| `--llm-concurrency` | Concurrent requests per server (default: `1`) |
| `--model` | Use YOLO+ViT pipeline: `yolov8m`, `yolov8l`, or `rtdetr` (requires `treemoissa[ml]`) |
| `--confidence` | Minimum detection confidence for ML mode (default: `0.35`) |

### Examples

Basic usage (local LLM server):

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted
```

Multiple LLM servers in parallel:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --llm-host 10.0.0.1:8080,10.0.0.2:8080
```

Remote LLM server with higher concurrency:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --llm-host 192.168.1.10:8080 --llm-concurrency 2
```

Use YOLO+ViT ML pipeline (requires `treemoissa[ml]`):

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model rtdetr
```

Input directory over NFS:

```bash
treemoissa //nas/photos/events /home/pix/sorted
```

## LLM Server Setup

Before using treemoissa, start the LLM server in a dedicated terminal:

```bash
runserver
```

On first run, this downloads:
- **llama.cpp** server binary (latest release from GitHub)
- **Qwen3.5-9B Q4_1** GGUF model (~5.8 GB from HuggingFace)

Files are cached in `~/.cache/treemoissa/`.

| Option | Description |
|---|---|
| `--port` | Server port (default: `8080`) |
| `--gpu-layers` / `-ngl` | GPU layers to offload (default: `99` = all) |
| `--ctx-size` / `-c` | Context size (default: `4096`) |

Then in another terminal, run treemoissa:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted
```

Photos with no detected car are copied to `unknown/unknown/unknown`.

## ML Pipeline (optional)

If you installed with `pip install -e ".[ml]"`, you can use the YOLO + ViT pipeline instead of the LLM:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model yolov8l
```

On first run, detection weights are downloaded automatically by `ultralytics` into `~/.cache/ultralytics/`:

| Model | Size | Notes |
|---|---|---|
| YOLOv8m | ~50 MB | Fast, good for large batches |
| YOLOv8l | ~87 MB | Better accuracy |
| RT-DETR-l | ~65 MB | Transformer-based, highest accuracy |

The classification model (`therealcyberlord/stanford-car-vit-patch16`) is downloaded from HuggingFace on first run.

### Supported image formats

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff` `.tif`

## WSL2 Configuration

For the LLM server to be reachable via `localhost` from within WSL2, WSL2 must be configured in **mirrored networking mode**.

Add the following to `%USERPROFILE%\.wslconfig` on Windows:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL2:

```powershell
wsl --shutdown
```

Without this setting, `localhost` inside WSL2 does not map to the Windows loopback interface and the `runserver` endpoint will not be reachable at `http://localhost:8080`.

## Notes

- Photos are **copied**, never moved — your original directory is untouched.
- Copies use `shutil.copy2` (no hardlinks) so the tool works over NFS mounts.
- In LLM mode, photos with no detected car are copied to `unknown/unknown/unknown`.
- In ML mode, photos with no detected car are silently skipped and counted in the summary.
- The tool processes the input directory non-recursively (flat directory expected).

## Development

```bash
# Run linter
ruff check .

# Run tests
pytest
```
