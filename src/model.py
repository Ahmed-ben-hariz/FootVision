"""Faster R-CNN (MobileNetV3-Large FPN) fine-tuning model.

The backbone is pretrained on ImageNet and the detector on COCO (torchvision
default weights). We keep that knowledge and replace only the final classifier,
so the model predicts 3 labels: 0 = background, 1 = player, 2 = ball.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 3  # background + player + ball
ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                min_size: int = 800, max_size: int = 1333) -> torch.nn.Module:
    """Return a detection model ready for fine-tuning.

    pretrained=True downloads COCO weights (needs internet the first time).
    pretrained=False gives random weights (used in unit tests).
    min_size/max_size set the internal resize applied by the model itself.
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_checkpoint(path: str | Path, model: torch.nn.Module, **metadata) -> None:
    """Save weights plus the information needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "architecture": ARCHITECTURE,
        "num_classes": NUM_CLASSES,
        **metadata,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu",
                    min_size: int = 800, max_size: int = 1333
                    ) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model, load weights, return (model in eval mode, metadata)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Unexpected architecture in {path}: {ckpt.get('architecture')}")
    model = build_model(ckpt["num_classes"], pretrained=False,
                        min_size=min_size, max_size=max_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    return model, meta
