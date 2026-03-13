"""Car detection using YOLO / RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


# COCO class IDs for vehicles
_CAR_CLASS_IDS = {2, 5, 7}  # car, bus, truck
_CAR_LABEL = {2: "car", 5: "bus", 7: "truck"}

# Minimum confidence threshold
_DEFAULT_CONF = 0.35

# Available detection models (ultralytics auto-downloads missing weights)
AVAILABLE_MODELS: dict[str, dict[str, str]] = {
    "yolov8m": {
        "file": "yolov8m.pt",
        "name": "YOLOv8 Medium",
        "desc": "Fast, good accuracy (~50 MB)",
    },
    "yolov8l": {
        "file": "yolov8l.pt",
        "name": "YOLOv8 Large",
        "desc": "Slower, better accuracy (~87 MB)",
    },
    "rtdetr": {
        "file": "rtdetr-l.pt",
        "name": "RT-DETR Large",
        "desc": "Transformer-based, high accuracy (~65 MB)",
    },
}

DEFAULT_MODEL = "yolov8m"


@dataclass
class DetectedCar:
    """A single car detected in an image."""

    crop: Image.Image
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    vehicle_type: str


def load_detector(model_key: str = DEFAULT_MODEL, device: str = "cuda") -> YOLO:
    """Load detection model (auto-downloads weights on first run)."""
    info = AVAILABLE_MODELS[model_key]
    model = YOLO(info["file"])
    model.to(device)
    return model


def detect_cars(
    model: YOLO,
    image_path: Path,
    conf: float = _DEFAULT_CONF,
) -> list[DetectedCar]:
    """Detect all cars in an image, return cropped regions."""
    results = model(str(image_path), conf=conf, verbose=False)
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    detections: list[DetectedCar] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in _CAR_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Clamp to image bounds
            h, w = img_array.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img.crop((x1, y1, x2, y2))

            detections.append(DetectedCar(
                crop=crop,
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                vehicle_type=_CAR_LABEL.get(cls_id, "car"),
            ))

    return detections
