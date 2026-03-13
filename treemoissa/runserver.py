"""Download llama.cpp and GGUF model, then launch llama-server."""

from __future__ import annotations

import io
import os
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
MMPROJ_FILE = "mmproj-BF16.gguf"
DEFAULT_PORT = 8080
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _get_llama_server_path() -> Path:
    """Return path to llama-server binary, downloading if absent."""
    server_bin = LLAMA_DIR / "llama-server"
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
    asset_name = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    asset_url = None
    for asset in release["assets"]:
        if asset["name"] == asset_name:
            asset_url = asset["browser_download_url"]
            break

    if asset_url is None:
        console.print(f"[bold red]Could not find asset {asset_name} in release {tag}[/bold red]")
        console.print("Available ubuntu assets:")
        for asset in release["assets"]:
            if "ubuntu" in asset["name"]:
                console.print(f"  - {asset['name']}")
        sys.exit(1)

    console.print(f"[bold]Downloading:[/bold] {asset_name} ({tag})")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        resp = client.get(asset_url)
        resp.raise_for_status()

    found_server = False
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            # Extract llama-server binary and all shared libraries
            if basename == "llama-server" or basename.endswith(".so") or ".so." in basename:
                member.name = basename
                tar.extract(member, path=LLAMA_DIR)
                if basename == "llama-server":
                    found_server = True
                    console.print(f"  [dim]extracted {basename}[/dim]")
                else:
                    console.print(f"  [dim]extracted lib: {basename}[/dim]")
        if not found_server:
            names = [m.name for m in tar.getmembers()]
            console.print("[bold red]llama-server not found in tarball.[/bold red]")
            console.print(f"Contents: {names[:20]}")
            sys.exit(1)

    server_bin.chmod(server_bin.stat().st_mode | stat.S_IEXEC)
    console.print(f"[green]llama-server installed:[/green] {server_bin}")
    return server_bin


def _get_model_paths() -> tuple[Path, Path]:
    """Return paths to GGUF model and mmproj, downloading if absent."""
    models_dir = CACHE_DIR / "models"

    console.print(f"[bold]Checking model:[/bold] {MODEL_REPO} / {MODEL_FILE}")
    model_path = Path(hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=models_dir,
    ))
    console.print(f"[green]Model ready:[/green] {model_path}")

    console.print(f"[bold]Checking mmproj:[/bold] {MODEL_REPO} / {MMPROJ_FILE}")
    mmproj_path = Path(hf_hub_download(
        repo_id=MODEL_REPO,
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
    model_path, mmproj_path = _get_model_paths()

    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--port", str(args.port),
        "-ngl", str(args.gpu_layers),
        "-c", str(args.ctx_size),
    ]

    env = {**os.environ, "LD_LIBRARY_PATH": str(LLAMA_DIR)}
    console.print(f"\n[bold green]Starting llama-server on port {args.port}...[/bold green]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")

    try:
        proc = subprocess.run(cmd, env=env)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    main()
