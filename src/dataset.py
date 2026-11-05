"""Custom PyTorch Dataset for COCO-format football detection data.

Model labels: 0 = background (reserved by Torchvision), 1 = player, 2 = ball.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)

LABELS = {"player": 1, "ball": 2}
Target = dict[str, torch.Tensor]
Transform = Callable[[torch.Tensor, Target], tuple[torch.Tensor, Target]]


class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir: str | Path, annotation_file: str | Path,
                 class_aliases: Mapping[str, str],
                 transform: Transform | None = None) -> None:
        self.images_dir = Path(images_dir)
        ann_path = Path(annotation_file)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        with open(ann_path) as f:
            coco = json.load(f)
        for key in ("images", "annotations", "categories"):
            if key not in coco:
                raise ValueError(f"{ann_path} is not valid COCO JSON: missing '{key}'")

        aliases = {k.lower(): v for k, v in class_aliases.items()}
        unknown_targets = set(aliases.values()) - set(LABELS)
        if unknown_targets:
            raise ValueError(f"class_aliases map to unknown classes: {unknown_targets}")

        self.cat_to_label: dict[int, int] = {}
        ignored = []
        for cat in coco["categories"]:
            name = cat["name"].lower()
            if name in aliases:
                self.cat_to_label[cat["id"]] = LABELS[aliases[name]]
            else:
                ignored.append(cat["name"])
        if ignored:
            logger.warning("Ignoring categories not in class_aliases: %s", ignored)

        self.transform = transform
        self.images = sorted(coco["images"], key=lambda im: im["id"])
        self._anns: dict[int, list[dict]] = {im["id"]: [] for im in self.images}
        for ann in coco["annotations"]:
            if ann["category_id"] in self.cat_to_label and ann["image_id"] in self._anns:
                self._anns[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Target]:
        info = self.images[idx]
        path = self.images_dir / info["file_name"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Image listed in annotations but missing on disk: {path}")
        image = Image.open(path).convert("RGB")
        width, height = image.size

        boxes, labels = [], []
        for ann in self._anns[info["id"]]:
            x, y, w, h = ann["bbox"]                       # COCO: x, y, width, height
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 - x1 < 1 or y2 - y1 < 1:                 # degenerate after clipping
                logger.warning("Dropping degenerate box %s in %s", ann["bbox"], path.name)
                continue
            boxes.append([x1, y1, x2, y2])                 # Torchvision: x1, y1, x2, y2
            labels.append(self.cat_to_label[ann["category_id"]])

        box_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target: Target = {
            "boxes": box_t,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]]),
            "area": (box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        tensor = F.to_tensor(image)  # float32 [0,1]; the model normalizes internally
        if self.transform is not None:
            tensor, target = self.transform(tensor, target)
        return tensor, target


def collate_fn(batch):
    """Images have different numbers of boxes, so return tuples, not stacked tensors."""
    return tuple(zip(*batch))
