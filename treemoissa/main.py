"""CLI entrypoint for treemoissa."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from treemoissa.organizer import copy_image

console = Console()


class PhotosPerMinuteColumn(ProgressColumn):
    """Rich progress column displaying photos processed per minute."""

    def render(self, task) -> Text:
        if task.speed is None:
            return Text("-- img/min", style="progress.data.speed")
        rate = task.speed * 60  # task.speed is img/sec; multiply by 60 → img/min
        return Text(f"{rate:.1f} img/min", style="progress.data.speed")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Keep in sync with detector.AVAILABLE_MODELS keys
_ML_MODEL_CHOICES = ["yolov8m", "yolov8l", "rtdetr"]


def gather_images(input_dir: Path) -> list[Path]:
    """Collect all image files from the input directory (non-recursive)."""
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def _print_summary(
    stats: dict,
    brand_counts: dict[str, int],
    *,
    server_stats: dict | None = None,
) -> None:
    """Print results summary and brand breakdown tables."""
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

    if server_stats:
        console.print()
        server_table = Table(title="Throughput by Server")
        server_table.add_column("Server", style="bold")
        server_table.add_column("Images", justify="right")
        server_table.add_column("img/min (avg)", justify="right")
        for url, d in sorted(server_stats.items(), key=lambda x: -x[1]["count"]):
            elapsed = d["elapsed"]
            if elapsed == 0.0:
                rate_str = "--"
            elif elapsed < 1.0:
                rate_str = "--"
            else:
                rate_str = f"{d['count'] / elapsed * 60:.1f}"
            server_table.add_row(url, str(d["count"]), rate_str)
        console.print(server_table)


def _make_progress() -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        PhotosPerMinuteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _run_llm_pipeline(
    input_dir: Path,
    output_dir: Path,
    llm_host: str,
    concurrency: int,
) -> dict:
    """Run the async LLM-based pipeline."""
    from treemoissa.llm_pool import LLMPool, ServerConfig

    images = gather_images(input_dir)
    if not images:
        console.print("[bold red]No images found in input directory.")
        return {"total_images": 0, "total_cars": 0, "copies": 0, "no_car": 0}

    servers = ServerConfig.parse(llm_host)
    server_list = ", ".join(s.url for s in servers)
    console.print(f"[bold]Mode:[/bold] LLM vision")
    console.print(f"[bold]Servers:[/bold] {server_list}")
    console.print(f"[bold]Concurrency:[/bold] {concurrency} per server")
    console.print(f"[bold]Found {len(images)} images to process.[/bold]\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    pool = LLMPool(
        servers=servers,
        concurrency=concurrency,
        output_dir=output_dir,
    )

    with _make_progress() as progress:
        task_id = progress.add_task("Processing images", total=len(images))

        def on_progress():
            progress.advance(task_id)

        try:
            stats = asyncio.run(pool.run(images, on_progress=on_progress))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — partial results below.[/yellow]")
            return {"total_images": len(images), "total_cars": 0, "copies": 0, "no_car": 0}

    brand_counts = stats.pop("brand_counts", {})
    server_stats = stats.pop("server_stats", None)
    _print_summary(stats, brand_counts, server_stats=server_stats)
    return stats


def _run_ml_pipeline(
    input_dir: Path,
    output_dir: Path,
    conf: float,
    model_key: str,
) -> dict:
    """Run the YOLO+ViT pipeline (requires treemoissa[ml])."""
    try:
        import torch
        from treemoissa.classifier import classify_car, load_classifier, parse_brand_model
        from treemoissa.color import extract_dominant_color
        from treemoissa.detector import AVAILABLE_MODELS, detect_cars, load_detector
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] --model requires ML dependencies.\n"
            "Install with: [bold]pip install treemoissa\\[ml][/bold]"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold]Device:[/bold] {device}")
    if device == "cuda":
        console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0)}")

    model_info = AVAILABLE_MODELS[model_key]

    # Load models
    with console.status(f"[bold green]Loading detection model ({model_info['name']})..."):
        detector = load_detector(model_key=model_key, device=device)

    with console.status("[bold green]Loading classification model..."):
        processor, classifier = load_classifier(device=device)

    # Gather images
    images = gather_images(input_dir)
    if not images:
        console.print("[bold red]No images found in input directory.")
        return {"total_images": 0, "total_cars": 0, "copies": 0, "no_car": 0}

    console.print(f"[bold]Found {len(images)} images to process.[/bold]\n")

    stats = {"total_images": len(images), "total_cars": 0, "copies": 0, "no_car": 0}
    brand_counts: dict[str, int] = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    with _make_progress() as progress:
        task = progress.add_task("Processing images", total=len(images))

        for image_path in images:
            progress.update(task, description=f"[cyan]{image_path.name}")

            try:
                detections = detect_cars(detector, image_path, conf=conf)
            except Exception as e:
                console.print(f"[yellow]Warning: failed to process {image_path.name}: {e}")
                progress.advance(task)
                continue

            if not detections:
                stats["no_car"] += 1
                progress.advance(task)
                continue

            stats["total_cars"] += len(detections)

            seen: set[tuple[str, str, str]] = set()

            for det in detections:
                label, _conf = classify_car(processor, classifier, det.crop, device=device)
                brand, model = parse_brand_model(label)
                color = extract_dominant_color(det.crop)

                key = (brand, model, color)
                if key in seen:
                    continue
                seen.add(key)

                copy_image(image_path, output_dir, brand, model, color)
                stats["copies"] += 1
                brand_counts[brand] = brand_counts.get(brand, 0) + 1

            progress.advance(task)

    _print_summary(stats, brand_counts)
    return stats


def _prompt_model_selection(available_models: dict) -> str:
    """Display an interactive menu to select the detection model."""
    console.print("\n[bold]Select a detection model:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", style="bold", width=3)
    table.add_column("Model", style="bold")
    table.add_column("Description")

    keys = list(available_models.keys())
    for idx, key in enumerate(keys, 1):
        info = available_models[key]
        table.add_row(str(idx), info["name"], info["desc"])

    console.print(table)
    console.print()

    while True:
        choice = console.input(f"[bold]Choice [1-{len(keys)}] (default: 1):[/bold] ").strip()
        if choice == "":
            return keys[0]
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        console.print(f"[red]Please enter a number between 1 and {len(keys)}.[/red]")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="treemoissa",
        description="Sort car photos into brand/model/color directory tree.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Flat directory containing car photos",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the organized tree",
    )
    parser.add_argument(
        "--llm-host",
        type=str,
        default="localhost:8080",
        help="LLM server(s) as ip:port,ip:port (default: localhost:8080)",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Concurrent requests per server (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=_ML_MODEL_CHOICES,
        default=None,
        help="Use YOLO+ViT pipeline with this model (requires treemoissa[ml])",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence for ML mode (default: 0.35)",
    )

    return parser.parse_args(argv)


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    conf: float = 0.35,
    model_key: str | None = None,
    *,
    llm_host: str = "localhost:8080",
    llm_concurrency: int = 1,
) -> dict:
    """Run the pipeline — LLM by default, ML if --model specified."""
    if model_key is not None:
        return _run_ml_pipeline(input_dir, output_dir, conf, model_key)
    return _run_llm_pipeline(input_dir, output_dir, llm_host, llm_concurrency)


def main() -> None:
    """Main CLI entrypoint."""
    args = _parse_args()

    if not args.input_dir.is_dir():
        console.print(f"[bold red]Error: {args.input_dir} is not a directory.")
        sys.exit(1)

    run_pipeline(
        args.input_dir,
        args.output_dir,
        conf=args.confidence,
        model_key=args.model,
        llm_host=args.llm_host,
        llm_concurrency=args.llm_concurrency,
    )


if __name__ == "__main__":
    main()
