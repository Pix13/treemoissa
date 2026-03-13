# treemoissa

Car trackdays and shows photo triage tool — automatically sorts a flat directory of car photos into a `brand/model/color` directory tree using AI.

## Features

- Detects cars in photos using **YOLOv8** or **RT-DETR** (auto-downloaded on first run)
- Interactive model selection menu with 3 options: YOLOv8m (fast), YOLOv8l (balanced), RT-DETR (high accuracy)
- Classifies brand and model using a **Vision Transformer** fine-tuned on Stanford Cars
- Extracts dominant color from each detected car (HSV-based, 11 color categories)
- If a photo contains multiple cars, it is copied into each matching subdirectory (no hardlinks — NFS-safe)
- **LLM mode** (`--llm`): send full photos to a local Qwen3.5-9B vision model for brand/model/color identification in a single pass
- Includes `runserver` tool to auto-download llama.cpp and the GGUF model
- GPU-accelerated (CUDA) with CPU fallback
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
- NVIDIA GPU with CUDA toolkit (recommended — GeForce RTX 4060 Ti or similar)
- WSL2 or Linux

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

### 3. Install PyTorch with CUDA support

Before installing the project dependencies, install the correct PyTorch build for your CUDA version.
Check your CUDA version with `nvcc --version` or `nvidia-smi`, then visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact install command. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install treemoissa

```bash
pip install -e .
```

On first run, detection weights are downloaded automatically by `ultralytics` into `~/.cache/ultralytics/`:

| Model | Size | Notes |
|---|---|---|
| YOLOv8m | ~50 MB | Fast, good for large batches |
| YOLOv8l | ~87 MB | Better accuracy, default recommendation |
| RT-DETR-l | ~65 MB | Transformer-based, highest accuracy |

The classification model (`therealcyberlord/stanford-car-vit-patch16`) is downloaded from HuggingFace on first run.

## Usage

```
treemoissa <input_dir> <output_dir> [--confidence THRESHOLD] [--model {yolov8m,yolov8l,rtdetr}]
```

| Argument | Description |
|---|---|
| `input_dir` | Flat directory containing your car photos |
| `output_dir` | Destination directory for the organized tree (created if absent) |
| `--confidence` | Minimum detection confidence, 0–1 (default: `0.35`) |
| `--model` | Detection model to use: `yolov8m`, `yolov8l`, or `rtdetr` (skips interactive menu) |

If `--model` is not specified, an interactive menu lets you choose the detection model at startup.

### Examples

Basic usage:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted
```

Stricter detection (fewer false positives):

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --confidence 0.55
```

Use RT-DETR for best accuracy:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model rtdetr
```

Input directory over NFS:

```bash
treemoissa //nas/photos/events /home/pix/sorted
```

## LLM Mode

As an alternative to the YOLO + ViT pipeline, treemoissa can use a local vision LLM
to identify cars. This sends each photo to a Qwen3.5-9B model served by llama.cpp.

### 1. Start the server

In a dedicated terminal:

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

### 2. Run treemoissa with `--llm`

In another terminal:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --llm
```

| Option | Description |
|---|---|
| `--llm` | Use the LLM vision model instead of YOLO + ViT |
| `--llm-url` | Server URL (default: `http://localhost:8080`) |

In LLM mode, photos with no detected car are copied to `unknown/unknown/unknown`.

### Supported image formats

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff` `.tif`

## Notes

- Photos are **copied**, never moved — your original directory is untouched.
- Copies use `shutil.copy2` (no hardlinks) so the tool works over NFS mounts.
- If no car is detected in a photo, it is silently skipped and counted in the summary.
- The tool processes the input directory non-recursively (flat directory expected).

## Development

```bash
# Run linter
ruff check .

# Run tests
pytest
```
