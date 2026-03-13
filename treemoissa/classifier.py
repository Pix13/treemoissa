"""Car make/model classification using a fine-tuned ViT on Stanford Cars."""

from __future__ import annotations

import re

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


_MODEL_ID = "therealcyberlord/stanford-car-vit-patch16"


def load_classifier(device: str = "cuda") -> tuple:
    """Load the car brand/model classifier (auto-downloads from HuggingFace)."""
    processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(_MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


def classify_car(
    processor,
    model,
    crop: Image.Image,
    device: str = "cuda",
) -> tuple[str, float]:
    """Classify a car crop, return (label, confidence).

    The label is the raw model output (typically brand name).
    """
    inputs = processor(images=crop, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    top_idx = probs.argmax(dim=-1).item()
    confidence = probs[0, top_idx].item()
    label = model.config.id2label[top_idx]

    return label, confidence


# Multi-word brand names from Stanford Cars dataset
_MULTI_WORD_BRANDS = {
    "aston martin", "land rover", "rolls-royce", "rolls royce",
    "am general",
}

# Known body types to strip from the label tail
_BODY_TYPES = {
    "sedan", "coupe", "convertible", "suv", "hatchback", "wagon",
    "minivan", "van", "pickup", "cab", "truck", "roadster",
    "crew cab", "regular cab", "extended cab", "super cab",
    "supercrew", "type-s",
}


def parse_brand_model(label: str) -> tuple[str, str]:
    """Parse a Stanford Cars label into (brand, model).

    Labels follow the pattern: 'Brand Model BodyType Year'
    e.g. 'Audi S5 Convertible 2012' → ('audi', 's5')
         'Rolls-Royce Phantom Sedan 2012' → ('rolls-royce', 'phantom')
    """
    label = label.strip()

    # Strip trailing year (4-digit number at end)
    label = re.sub(r"\s+\d{4}\s*$", "", label)

    # Strip body type suffixes (from right)
    lower = label.lower()
    for bt in sorted(_BODY_TYPES, key=len, reverse=True):
        if lower.endswith(" " + bt):
            label = label[: -(len(bt) + 1)]
            lower = label.lower()

    # Try multi-word brands first
    for brand in _MULTI_WORD_BRANDS:
        if lower.startswith(brand + " "):
            rest = label[len(brand):].strip()
            return _sanitize(brand), _sanitize(rest) if rest else "unknown"

    # Default: first word is brand, rest is model
    parts = label.split(None, 1)
    if len(parts) == 2:
        return _sanitize(parts[0]), _sanitize(parts[1])
    return _sanitize(parts[0]), "unknown"


def _sanitize(name: str) -> str:
    """Sanitize a name for use as a directory component."""
    name = name.strip().lower()
    # Remove characters not safe for filenames
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or "unknown"
