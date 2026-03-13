"""CLI entrypoint for treemoissa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from treemoissa.classifier import classify_car, load_classifier, parse_brand_model
from treemoissa.color import extract_dominant_color
from treemoissa.detector import (
    AVAILABLE_MODELS,
    detect_cars,
    load_detector,
)
from treemoissa.organizer import copy_image

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def gather_images(input_dir: Path) -> list[Path]:
    """Collect all image files from the input directory (non-recursive)."""
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def pick_device() -> str:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def prompt_model_selection() -> str:
    """Display an interactive menu to select the detection model."""
    console.print("\n[bold]Select a detection model:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", style="bold", width=3)
    table.add_column("Model", style="bold")
    table.add_column("Description")

    keys = list(AVAILABLE_MODELS.keys())
    for idx, key in enumerate(keys, 1):
        info = AVAILABLE_MODELS[key]
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


def _print_summary(stats: dict, brand_counts: dict[str, int]) -> None:
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


def _make_progress() -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


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

    with _make_progress() as progress:
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

    _print_summary(stats, brand_counts)
    return stats


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    conf: float = 0.35,
    model_key: str | None = None,
    *,
    llm: bool = False,
    llm_url: str = "http://localhost:8080",
) -> dict:
    """Run the full detection → classification → organization pipeline."""
    if llm:
        return _run_llm_pipeline(input_dir, output_dir, llm_url)

    device = pick_device()

    console.print(f"[bold]Device:[/bold] {device}")
    if device == "cuda":
        console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0)}")

    # Model selection
    if model_key is None:
        model_key = prompt_model_selection()

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
        return {"total_images": 0, "total_cars": 0, "copies": 0}

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

            # Track unique destinations to avoid duplicate copies per image
            seen: set[tuple[str, str, str]] = set()

            for det in detections:
                # Classify brand/model
                label, _conf = classify_car(processor, classifier, det.crop, device=device)
                brand, model = parse_brand_model(label)

                # Extract color
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


def main() -> None:
    """Main CLI entrypoint."""
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
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence (default: 0.35)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        default=None,
        help="Detection model to use (skip interactive menu)",
    )
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

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        console.print(f"[bold red]Error: {args.input_dir} is not a directory.")
        sys.exit(1)

    run_pipeline(
        args.input_dir,
        args.output_dir,
        conf=args.confidence,
        model_key=args.model,
        llm=args.llm,
        llm_url=args.llm_url,
    )


if __name__ == "__main__":
    main()
