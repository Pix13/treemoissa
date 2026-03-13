# treemoissa

Car trackdays and shows photo triage tool — automatically sorts a flat directory of car photos into a `brand/model/color` directory tree using AI.

## Features

- Detects cars in photos using **YOLOv8x** (auto-downloaded on first run)
- Classifies brand and model using a **HuggingFace vision-language model**
- Extracts dominant color from each detected car
- If a photo contains multiple cars, it is copied into each matching subdirectory (no hardlinks — NFS-safe)
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

On first run, **YOLOv8x weights** (~140 MB) are downloaded automatically by `ultralytics` into `~/.cache/ultralytics/`.

## Usage

```
treemoissa <input_dir> <output_dir> [--confidence THRESHOLD]
```

| Argument | Description |
|---|---|
| `input_dir` | Flat directory containing your car photos |
| `output_dir` | Destination directory for the organized tree (created if absent) |
| `--confidence` | Minimum detection confidence, 0–1 (default: `0.35`) |

### Examples

Basic usage:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted
```

Stricter detection (fewer false positives):

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --confidence 0.55
```

Input directory over NFS:

```bash
treemoissa //nas/photos/events /home/pix/sorted
```

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
