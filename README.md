# treemoissa

[Francais](#francais) | [English](#english)

---

# Francais

Outil de tri de photos de voitures (trackdays, rassemblements) — trie automatiquement un repertoire de photos en arborescence `marque/modele/couleur` grace a l'IA.

## Fonctionnalites

- **Mode vision LLM (par defaut) :** envoie les photos a un modele Qwen3.5 local pour identifier marque/modele/couleur en une seule passe
- **Detection automatique de la VRAM :** selectionne automatiquement le meilleur modele Qwen3.5 (0.8B a 27B) en fonction de la memoire GPU disponible
- **Support multi-serveurs :** repartit le travail sur plusieurs serveurs LLM en parallele
- **Pipeline ML (optionnel) :** detection via YOLOv8 / RT-DETR, classification par Vision Transformer, extraction de couleur par analyse HSV
- Si une photo contient plusieurs voitures, elle est copiee dans chaque sous-repertoire correspondant (pas de liens durs — compatible NFS)
- Barre de progression et tableau recapitulatif

## Structure de sortie

```
output_dir/
└── Toyota/
│   └── Supra/
│       └── red/
│           └── IMG_0042.jpg
└── Porsche/
    └── 911/
        └── silver/
            └── IMG_0042.jpg   <- meme photo copiee si deux voitures visibles
```

## Prerequis : configuration WSL2 et CUDA

treemoissa est prevu pour fonctionner sous **WSL2** (Windows Subsystem for Linux). Voici les etapes de configuration initiale.

### 1. Activer WSL2

Ouvrir un **PowerShell en administrateur** et executer :

```powershell
wsl --install
```

Redemarrer l'ordinateur, puis verifier l'installation :

```powershell
wsl --version
```

S'assurer que la version par defaut est bien WSL 2.

### 2. Installer le driver NVIDIA pour WSL

WSL2 accede au GPU via le driver **Windows** — il ne faut **pas** installer de driver Linux NVIDIA dans WSL.

1. Telecharger et installer le dernier driver NVIDIA pour Windows depuis :
   https://developer.nvidia.com/cuda/wsl
2. Le driver Windows fournit automatiquement le support CUDA dans WSL2.

### 3. Verifier que le GPU est visible dans WSL

Ouvrir un terminal WSL et executer :

```bash
nvidia-smi
```

Cette commande doit afficher le nom du GPU, la VRAM disponible et la version du driver. Si `nvidia-smi` n'est pas trouve, le driver Windows n'est pas correctement installe.

### 4. Configurer le reseau en mode miroir

Pour que le serveur LLM soit accessible via `localhost` depuis WSL2, le reseau WSL2 doit etre en **mode miroir**.

Ajouter dans `%USERPROFILE%\.wslconfig` sur Windows :

```ini
[wsl2]
networkingMode=mirrored
```

Puis redemarrer WSL2 :

```powershell
wsl --shutdown
```

## Installation

### 1. Cloner le depot

```bash
git clone <repo-url>
cd treemoissa
```

### 2. Creer un environnement virtuel

Avec `uv` (recommande) :

```bash
uv venv
source .venv/bin/activate
```

Ou avec `venv` standard :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer treemoissa

**Leger (LLM uniquement — recommande) :**

```bash
pip install -e .
```

**Avec pipeline ML (YOLO + ViT) :**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[ml]"
```

Verifier la version CUDA avec `nvcc --version` ou `nvidia-smi`, puis consulter [pytorch.org](https://pytorch.org/get-started/locally/) pour la commande exacte d'installation de PyTorch.

## Utilisation : deux terminaux WSL necessaires

treemoissa necessite **deux terminaux WSL** ouverts simultanement :

```
Terminal 1 (serveur LLM) :              Terminal 2 (tri des photos) :
┌──────────────────────────┐            ┌──────────────────────────────┐
│ $ runserver              │            │ $ treemoissa /input /output  │
│   VRAM detected: 8192 MB │            │   Processing images...       │
│   Selected: Qwen3.5-9B   │            │   ████████████░░░ 75/100    │
│   Server ready on :8080  │            │                              │
│   (bloquant — ne pas     │            │                              │
│    fermer ce terminal)   │            │                              │
└──────────────────────────┘            └──────────────────────────────┘
```

**`runserver` est un processus bloquant** — il doit rester actif pendant toute la duree du tri. Ne pas fermer ce terminal tant que le tri n'est pas termine.

### Serveur LLM (Terminal 1)

```bash
runserver
```

Au premier lancement, cela telecharge automatiquement :
- Le binaire **llama.cpp** (derniere version depuis GitHub)
- Le **modele GGUF Qwen3.5** adapte a votre GPU (selection automatique selon la VRAM)

Les fichiers sont mis en cache dans `~/.cache/treemoissa/`.

| Option | Description |
|---|---|
| `--port` | Port du serveur (defaut : `8080`) |
| `--gpu-layers` / `-ngl` | Couches a decharger sur le GPU (defaut : `99` = toutes) |
| `--ctx-size` / `-c` | Taille du contexte (defaut : `4096`) |
| `--quant` | Quantisation du modele (defaut : `Q4_K_M`). Valeurs : `Q3_K_M`, `Q3_K_S`, `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `BF16` |

La selection automatique du modele fonctionne ainsi :

| VRAM disponible | Modele selectionne |
|---|---|
| >= 18 Go | Qwen3.5-27B |
| >= 7 Go | Qwen3.5-9B |
| >= 4 Go | Qwen3.5-4B |
| >= 3 Go | Qwen3.5-2B |
| >= 2 Go | Qwen3.5-0.8B |

### Tri des photos (Terminal 2)

```bash
treemoissa <repertoire_entree> <repertoire_sortie> [options]
```

| Argument | Description |
|---|---|
| `repertoire_entree` | Repertoire plat contenant les photos de voitures |
| `repertoire_sortie` | Repertoire de destination pour l'arborescence (cree si absent) |
| `--llm-host` | Serveur(s) LLM au format `ip:port,ip:port` (defaut : `localhost:8080`) |
| `--llm-concurrency` | Requetes simultanees par serveur (defaut : `1`) |
| `--model` | Utiliser le pipeline YOLO+ViT : `yolov8m`, `yolov8l` ou `rtdetr` (necessite `treemoissa[ml]`) |
| `--confidence` | Confiance minimale de detection en mode ML (defaut : `0.35`) |

### Exemples

Usage basique (serveur LLM local) :

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted
```

Plusieurs serveurs LLM en parallele :

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --llm-host 10.0.0.1:8080,10.0.0.2:8080
```

Pipeline ML YOLO+ViT (necessite `treemoissa[ml]`) :

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model rtdetr
```

## Pipeline ML (optionnel)

Avec `pip install -e ".[ml]"`, le pipeline YOLO + ViT est disponible :

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model yolov8l
```

Au premier lancement, les poids de detection sont telecharges automatiquement dans `~/.cache/ultralytics/` :

| Modele | Taille | Notes |
|---|---|---|
| YOLOv8m | ~50 Mo | Rapide, adapte aux gros lots |
| YOLOv8l | ~87 Mo | Meilleure precision |
| RT-DETR-l | ~65 Mo | Base Transformer, meilleure precision |

Le modele de classification (`therealcyberlord/stanford-car-vit-patch16`) est telecharge depuis HuggingFace au premier lancement.

### Formats d'image supportes

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff` `.tif`

## Notes

- Les photos sont **copiees**, jamais deplacees — le repertoire source reste intact.
- Les copies utilisent `shutil.copy2` (pas de liens durs) pour fonctionner sur des montages NFS.
- En mode LLM, les photos sans voiture detectee sont copiees dans `unknown/unknown/unknown`.
- En mode ML, les photos sans voiture sont ignorees et comptabilisees dans le recapitulatif.
- L'outil traite le repertoire d'entree de maniere non recursive (repertoire plat attendu).

## Developpement

```bash
# Lancer le linter
ruff check .

# Lancer les tests
pytest
```

---

# English

Car trackdays and shows photo triage tool — automatically sorts a flat directory of car photos into a `brand/model/color` directory tree using AI.

## Features

- **LLM vision mode (default):** sends photos to a local Qwen3.5 vision model for brand/model/color identification in a single pass
- **Automatic VRAM detection:** automatically selects the best Qwen3.5 model (0.8B to 27B) based on available GPU memory
- **Multi-server support:** distribute work across multiple LLM servers in parallel
- **ML pipeline (optional):** detect cars with YOLOv8 / RT-DETR, classify with a Vision Transformer, extract color via HSV analysis
- If a photo contains multiple cars, it is copied into each matching subdirectory (no hardlinks — NFS-safe)
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
            └── IMG_0042.jpg   <- same photo copied if two cars visible
```

## Prerequisites: WSL2 and CUDA setup

treemoissa is designed to run under **WSL2** (Windows Subsystem for Linux). Here are the initial setup steps.

### 1. Enable WSL2

Open an **admin PowerShell** and run:

```powershell
wsl --install
```

Reboot, then verify:

```powershell
wsl --version
```

Make sure the default version is WSL 2.

### 2. Install the NVIDIA driver for WSL

WSL2 accesses the GPU through the **Windows** driver — do **not** install a Linux NVIDIA driver inside WSL.

1. Download and install the latest NVIDIA Windows driver from:
   https://developer.nvidia.com/cuda/wsl
2. The Windows driver automatically provides CUDA support inside WSL2.

### 3. Verify GPU visibility in WSL

Open a WSL terminal and run:

```bash
nvidia-smi
```

This should display your GPU name, available VRAM, and driver version. If `nvidia-smi` is not found, the Windows driver is not correctly installed.

### 4. Configure mirrored networking

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

## Usage: two WSL terminals required

treemoissa requires **two WSL terminals** open simultaneously:

```
Terminal 1 (LLM server):                Terminal 2 (photo sorting):
┌──────────────────────────┐            ┌──────────────────────────────┐
│ $ runserver              │            │ $ treemoissa /input /output  │
│   VRAM detected: 8192 MB │            │   Processing images...       │
│   Selected: Qwen3.5-9B   │            │   ████████████░░░ 75/100    │
│   Server ready on :8080  │            │                              │
│   (blocking — do not     │            │                              │
│    close this terminal)  │            │                              │
└──────────────────────────┘            └──────────────────────────────┘
```

**`runserver` is a blocking process** — it must stay running during the entire sort. Do not close this terminal until sorting is complete.

### LLM Server (Terminal 1)

```bash
runserver
```

On first run, this automatically downloads:
- The **llama.cpp** server binary (latest release from GitHub)
- The **Qwen3.5 GGUF model** best suited to your GPU (automatically selected based on VRAM)

Files are cached in `~/.cache/treemoissa/`.

| Option | Description |
|---|---|
| `--port` | Server port (default: `8080`) |
| `--gpu-layers` / `-ngl` | GPU layers to offload (default: `99` = all) |
| `--ctx-size` / `-c` | Context size (default: `4096`) |
| `--quant` | Model quantization (default: `Q4_K_M`). Values: `Q3_K_M`, `Q3_K_S`, `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `BF16` |

Automatic model selection works as follows:

| Available VRAM | Selected model |
|---|---|
| >= 18 GB | Qwen3.5-27B |
| >= 7 GB | Qwen3.5-9B |
| >= 4 GB | Qwen3.5-4B |
| >= 3 GB | Qwen3.5-2B |
| >= 2 GB | Qwen3.5-0.8B |

### Photo sorting (Terminal 2)

```bash
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

YOLO+ViT ML pipeline (requires `treemoissa[ml]`):

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model rtdetr
```

## ML Pipeline (optional)

If you installed with `pip install -e ".[ml]"`, you can use the YOLO + ViT pipeline:

```bash
treemoissa /mnt/photos/trackday_2024 /mnt/photos/sorted --model yolov8l
```

On first run, detection weights are downloaded automatically into `~/.cache/ultralytics/`:

| Model | Size | Notes |
|---|---|---|
| YOLOv8m | ~50 MB | Fast, good for large batches |
| YOLOv8l | ~87 MB | Better accuracy |
| RT-DETR-l | ~65 MB | Transformer-based, highest accuracy |

The classification model (`therealcyberlord/stanford-car-vit-patch16`) is downloaded from HuggingFace on first run.

### Supported image formats

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff` `.tif`

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
