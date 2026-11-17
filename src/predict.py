"""Inference: image -> detections, counts, annotated image.

Usage:
    python -m src.predict --image path/to/image.jpg [--mode single|tiled] [--threshold 0.5]
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.ops import batched_nms

from src.config import ROOT, load_config
from src.device import get_device
from src.model import load_checkpoint

LABEL_NAMES = {1: "player", 2: "ball"}
COLORS = {1: (0, 200, 0), 2: (0, 0, 255)}  # BGR


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original image pixels
    label: int
    score: float


@dataclass
class PredictionResult:
    detections: list[Detection]
    image_size: tuple[int, int]               # (width, height)

    def count(self, label: int) -> int:
        return sum(d.label == label for d in self.detections)

    @property
    def n_players(self) -> int:
        return self.count(1)

    @property
    def n_balls(self) -> int:
        return self.count(2)


def load_predictor(checkpoint: str | Path, device: torch.device):
    """Load a checkpoint and restore the input scale it was trained at."""
    path = Path(checkpoint)
    path = path if path.is_absolute() else ROOT / path
    model, meta = load_checkpoint(path, device)
    if "min_size" in meta:
        model.transform.min_size = (meta["min_size"],)
        model.transform.max_size = meta["max_size"]
    return model, meta


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not read image {path}: {e}") from e


@torch.no_grad()
def _infer_square(model, image: Image.Image, device, square_size: int):
    """Stretch to a square (as the dataset did), run the model, map boxes back."""
    w, h = image.size
    resized = image.resize((square_size, square_size), Image.BILINEAR)
    model.eval()
    out = model([F.to_tensor(resized).to(device)])[0]
    boxes = out["boxes"].detach().cpu().clone()
    boxes[:, [0, 2]] *= w / square_size
    boxes[:, [1, 3]] *= h / square_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes, out["labels"].detach().cpu(), out["scores"].detach().cpu()


def _to_result(boxes, labels, scores, size) -> PredictionResult:
    dets = [Detection(tuple(float(v) for v in b), int(l), float(s))
            for b, l, s in zip(boxes.tolist(), labels.tolist(), scores.tolist())]
    dets.sort(key=lambda d: -d.score)
    return PredictionResult(dets, size)


def detect_single(model, image: Image.Image, device, score_thr: float,
                  square_size: int = 1280) -> PredictionResult:
    boxes, labels, scores = _infer_square(model, image, device, square_size)
    keep = scores >= score_thr
    return _to_result(boxes[keep], labels[keep], scores[keep], image.size)


def detect_tiled(model, image: Image.Image, device, score_thr: float,
                 square_size: int = 1280, grid: int = 2, overlap: float = 0.1,
                 nms_iou: float = 0.5) -> PredictionResult:
    """Run the model on an overlapping grid of tiles and merge with per-class NMS."""
    w, h = image.size
    tw, th = w / grid, h / grid
    px, py = overlap * tw, overlap * th
    all_b, all_l, all_s = [], [], []
    for r in range(grid):
        for c in range(grid):
            x0, x1 = max(0, int(c * tw - px)), min(w, int((c + 1) * tw + px))
            y0, y1 = max(0, int(r * th - py)), min(h, int((r + 1) * th + py))
            b, l, s = _infer_square(model, image.crop((x0, y0, x1, y1)), device, square_size)
            keep = s >= score_thr
            b, l, s = b[keep].clone(), l[keep], s[keep]
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            all_b.append(b); all_l.append(l); all_s.append(s)
    boxes, labels, scores = torch.cat(all_b), torch.cat(all_l), torch.cat(all_s)
    keep = batched_nms(boxes, scores, labels, nms_iou)
    return _to_result(boxes[keep], labels[keep], scores[keep], image.size)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    thick = max(1, round(max(arr.shape[:2]) / 700))
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = COLORS[d.label]
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, thick)
        cv2.putText(arr, f"{LABEL_NAMES[d.label]} {d.score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 + 0.1 * thick, color, max(1, thick - 1))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def predict_image(model, image: Image.Image, device, cfg_inf: dict,
                  mode: str | None = None, threshold: float | None = None) -> PredictionResult:
    """Run detection with settings from config (overridable)."""
    mode = mode or cfg_inf["mode"]
    thr = cfg_inf["confidence_threshold"] if threshold is None else threshold
    if mode == "single":
        return detect_single(model, image, device, thr, cfg_inf["square_size"])
    if mode == "tiled":
        return detect_tiled(model, image, device, thr, cfg_inf["square_size"],
                            cfg_inf["tile_grid"], cfg_inf["tile_overlap"], cfg_inf["nms_iou"])
    raise ValueError(f"Unknown mode {mode!r}; use 'single' or 'tiled'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--mode", choices=["single", "tiled"])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out", default="results/predictions")
    args = ap.parse_args()

    cfg = load_config()
    inf = cfg["inference"]
    device = get_device(cfg["device"])
    model, meta = load_predictor(args.checkpoint or inf["checkpoint"], device)
    image = load_image(args.image)
    mode = args.mode or inf["mode"]
    result = predict_image(model, image, device, inf, mode, args.threshold)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_{mode}.jpg"
    draw_detections(image, result.detections).save(out_path)

    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]}), mode={mode}, device={device}")
    print(f"Players detected: {result.n_players}")
    print(f"Footballs detected: {result.n_balls}")
    print("Confidences:", ", ".join(f"{LABEL_NAMES[d.label]}={d.score:.2f}"
                                      for d in result.detections) or "none")
    print(f"Output image: {out_path}")


if __name__ == "__main__":
    main()
