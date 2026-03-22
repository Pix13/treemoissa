"""Download llama.cpp and GGUF model, then launch llama-server."""

from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download
from rich.console import Console

console = Console()

CACHE_DIR = Path.home() / ".cache" / "treemoissa"
LLAMA_DIR = CACHE_DIR / "llama-server"
MMPROJ_FILE = "mmproj-BF16.gguf"
DEFAULT_PORT = 8080
DEFAULT_QUANT = "Q4_K_M"
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

ALLOWED_QUANTS = [
    "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S",
    "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "BF16",
]

# Ordered largest to smallest. min_vram_mb includes ~1.2 GB overhead for mmproj + KV cache.
MODEL_CANDIDATES = [
    {"size": "27B", "repo": "unsloth/Qwen3.5-27B-GGUF", "min_vram_mb": 18_000},
    {"size": "9B",  "repo": "unsloth/Qwen3.5-9B-GGUF",  "min_vram_mb": 7_000},
    {"size": "4B",  "repo": "unsloth/Qwen3.5-4B-GGUF",  "min_vram_mb": 4_000},
    {"size": "2B",  "repo": "unsloth/Qwen3.5-2B-GGUF",  "min_vram_mb": 3_000},
    {"size": "0.8B", "repo": "unsloth/Qwen3.5-0.8B-GGUF", "min_vram_mb": 2_000},
]

DEFAULT_MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"


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


def _select_best_model(vram_mb: int | None, quant: str) -> tuple[str, str, str]:
    """Select the best Qwen3.5 model that fits in available VRAM.

    Returns (repo_id, model_filename, display_name).
    """
    if vram_mb is None:
        size = "9B"
        filename = f"Qwen3.5-{size}-{quant}.gguf"
        return DEFAULT_MODEL_REPO, filename, f"Qwen3.5-{size}-{quant} (default, VRAM unknown)"

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


def _is_wsl() -> bool:
    """Detect if running under WSL2."""
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _wsl_win_path(posix_path: Path) -> str:
    """Convert a WSL posix path to a Windows path."""
    result = subprocess.run(
        ["wslpath", "-w", str(posix_path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _download_asset(release: dict, name: str) -> bytes:
    """Find and download a release asset by name."""
    url = None
    for asset in release["assets"]:
        if asset["name"] == name:
            url = asset["browser_download_url"]
            break

    if url is None:
        console.print(f"[bold red]Could not find asset {name}[/bold red]")
        console.print("Available assets:")
        for asset in release["assets"]:
            console.print(f"  - {asset['name']}")
        sys.exit(1)

    console.print(f"[bold]Downloading:[/bold] {name}")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return resp.content


def _get_llama_server_path() -> Path:
    """Return path to llama-server binary, downloading if absent."""
    wsl = _is_wsl()
    server_name = "llama-server.exe" if wsl else "llama-server"
    server_bin = LLAMA_DIR / server_name

    if server_bin.exists():
        console.print(f"[green]llama-server found:[/green] {server_bin}")
        return server_bin

    console.print("[bold]Downloading latest llama.cpp release...[/bold]")
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True, timeout=30) as client:
        resp = client.get(GITHUB_API, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        release = resp.json()

    tag = release["tag_name"]

    if wsl:
        _download_win_cuda(release, tag)
    else:
        _download_linux_vulkan(release, tag)

    if not server_bin.exists():
        console.print(f"[bold red]{server_name} not found after extraction.[/bold red]")
        sys.exit(1)

    server_bin.chmod(server_bin.stat().st_mode | stat.S_IEXEC)
    console.print(f"[green]llama-server installed:[/green] {server_bin}")
    return server_bin


def _download_win_cuda(release: dict, tag: str) -> None:
    """Download Windows CUDA build (main + cudart) for WSL2."""
    # Main CUDA build
    main_asset = f"llama-{tag}-bin-win-cuda-12.4-x64.zip"
    console.print("[bold cyan]WSL2 detected:[/bold cyan] using Windows CUDA build")
    data = _download_asset(release, main_asset)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            basename = Path(member).name
            if not basename:
                continue
            if basename.endswith((".exe", ".dll")):
                with zf.open(member) as src, open(LLAMA_DIR / basename, "wb") as dst:
                    dst.write(src.read())
                console.print(f"  [dim]extracted {basename}[/dim]")

    # CUDA runtime DLLs
    cudart_asset = "cudart-llama-bin-win-cuda-12.4-x64.zip"
    data = _download_asset(release, cudart_asset)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            basename = Path(member).name
            if not basename:
                continue
            if basename.endswith(".dll"):
                with zf.open(member) as src, open(LLAMA_DIR / basename, "wb") as dst:
                    dst.write(src.read())
                console.print(f"  [dim]extracted cudart: {basename}[/dim]")


def _download_linux_vulkan(release: dict, tag: str) -> None:
    """Download Linux Vulkan build."""
    asset_name = f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"
    data = _download_asset(release, asset_name)

    found_server = False
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename == "llama-server" or basename.endswith(".so") or ".so." in basename:
                member.name = basename
                tar.extract(member, path=LLAMA_DIR)
                if basename == "llama-server":
                    found_server = True
                    console.print(f"  [dim]extracted {basename}[/dim]")
                else:
                    console.print(f"  [dim]extracted lib: {basename}[/dim]")
    if not found_server:
        console.print("[bold red]llama-server not found in tarball.[/bold red]")
        sys.exit(1)


def _get_model_paths(model_repo: str, model_file: str) -> tuple[Path, Path]:
    """Return paths to GGUF model and mmproj, downloading if absent."""
    models_dir = CACHE_DIR / "models"

    console.print(f"[bold]Checking model:[/bold] {model_repo} / {model_file}")
    model_path = Path(hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        cache_dir=models_dir,
    ))
    console.print(f"[green]Model ready:[/green] {model_path}")

    console.print(f"[bold]Checking mmproj:[/bold] {model_repo} / {MMPROJ_FILE}")
    mmproj_path = Path(hf_hub_download(
        repo_id=model_repo,
        filename=MMPROJ_FILE,
        cache_dir=models_dir,
    ))
    console.print(f"[green]mmproj ready:[/green] {mmproj_path}")

    return model_path, mmproj_path


def main() -> None:
    """Main entrypoint for runserver CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="runserver",
        description="Download and launch llama.cpp server with auto-selected Qwen3.5 model.",
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
    parser.add_argument(
        "--quant", type=str, default=DEFAULT_QUANT,
        choices=ALLOWED_QUANTS,
        help=f"Model quantization (default: {DEFAULT_QUANT})",
    )
    args = parser.parse_args()

    vram_mb = _detect_vram_mb()
    if vram_mb is not None:
        console.print(f"[bold]VRAM detected:[/bold] {vram_mb} MB")
    else:
        console.print("[yellow]Could not detect VRAM — using default model.[/yellow]")

    model_repo, model_file, display_name = _select_best_model(vram_mb, args.quant)
    console.print(f"[bold]Selected model:[/bold] {display_name}")
    if args.quant == DEFAULT_QUANT:
        console.print("[dim]Use --quant to override (e.g. --quant Q8_0)[/dim]")

    server_bin = _get_llama_server_path()
    model_path, mmproj_path = _get_model_paths(model_repo, model_file)

    wsl = _is_wsl()

    # Under WSL2, the Windows .exe needs Windows-style paths
    if wsl:
        model_arg = _wsl_win_path(model_path)
        mmproj_arg = _wsl_win_path(mmproj_path)
    else:
        model_arg = str(model_path)
        mmproj_arg = str(mmproj_path)

    cmd = [
        str(server_bin),
        "-m", model_arg,
        "--mmproj", mmproj_arg,
        "--host", "0.0.0.0" if wsl else "127.0.0.1",
        "--port", str(args.port),
        "-ngl", str(args.gpu_layers),
        "-c", str(args.ctx_size),
    ]

    env = {**os.environ}
    if not wsl:
        env["LD_LIBRARY_PATH"] = str(LLAMA_DIR)

    console.print(f"\n[bold green]Starting llama-server on port {args.port}...[/bold green]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")

    try:
        proc = subprocess.run(cmd, env=env)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    main()
