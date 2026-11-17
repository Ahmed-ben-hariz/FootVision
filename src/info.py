"""Small read-only helpers for the dashboard's information pages."""
from __future__ import annotations

import json
from pathlib import Path

from src.data_quality import ANN_NAME

SPLITS = ("train", "valid", "test")


def dataset_summary(dataset_dir: str | Path, class_aliases: dict[str, str]):
    """Image and annotation counts per split, or None if the dataset is missing."""
    root = Path(dataset_dir)
    if not root.is_dir():
        return None
    aliases = {k.lower(): v for k, v in class_aliases.items()}
    out: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = root / split / ANN_NAME
        if not path.is_file():
            continue
        try:
            coco = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        cats = {c["id"]: aliases.get(c["name"].lower()) for c in coco.get("categories", [])}
        counts = {"player": 0, "ball": 0}
        for ann in coco.get("annotations", []):
            target = cats.get(ann.get("category_id"))
            if target:
                counts[target] += 1
        out[split] = {"images": len(coco.get("images", [])), **counts}
    return out or None


def read_json(path: str | Path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def read_text(path: str | Path):
    path = Path(path)
    return path.read_text() if path.is_file() else None
