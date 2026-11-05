"""Draw ground-truth boxes on dataset samples to verify annotations by eye.

Usage: python -m src.visualize --split data/raw/<dataset>/train --n 8
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.data_quality import ANN_NAME
from src.dataset import CocoDetectionDataset

COLORS = {1: (0, 200, 0), 2: (0, 0, 255)}   # BGR: player green, ball red
NAMES = {1: "player", 2: "ball"}


def draw_boxes(image_rgb: np.ndarray, boxes, labels) -> np.ndarray:
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, COLORS[int(lab)], 2)
        cv2.putText(out, NAMES[int(lab)], (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[int(lab)], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="Split folder with images + annotations")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/samples")
    args = ap.parse_args()

    cfg = load_config()
    split = Path(args.split)
    ds = CocoDetectionDataset(split, split / ANN_NAME, cfg["data"]["class_aliases"])
    random.seed(cfg["seed"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in random.sample(range(len(ds)), min(args.n, len(ds))):
        img, tgt = ds[idx]
        rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        drawn = draw_boxes(rgb, tgt["boxes"].tolist(), tgt["labels"].tolist())
        path = out_dir / f"sample_{idx}.jpg"
        cv2.imwrite(str(path), drawn)
        print(f"{path}: {int((tgt['labels'] == 1).sum())} players, "
              f"{int((tgt['labels'] == 2).sum())} balls")


if __name__ == "__main__":
    main()
