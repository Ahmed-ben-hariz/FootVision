"""Detection-safe augmentations: every geometric change also updates the boxes."""
import random

import torch
import torchvision.transforms.functional as F

Target = dict[str, torch.Tensor]


def hflip(image: torch.Tensor, target: Target) -> tuple[torch.Tensor, Target]:
    """Flip a [C,H,W] image left-right and mirror the boxes (x1,y1,x2,y2)."""
    width = image.shape[-1]
    boxes = target["boxes"].clone()
    if len(boxes):
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return image.flip(-1), {**target, "boxes": boxes}


class TrainAugment:
    """Horizontal flip + mild brightness/contrast jitter. No vertical flip or
    rotation: they make football images unrealistic."""

    def __init__(self, flip_prob: float = 0.5, brightness: float = 0.2,
                 contrast: float = 0.2) -> None:
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: torch.Tensor, target: Target):
        if random.random() < self.flip_prob:
            image, target = hflip(image, target)
        image = F.adjust_brightness(
            image, 1 + random.uniform(-self.brightness, self.brightness))
        image = F.adjust_contrast(
            image, 1 + random.uniform(-self.contrast, self.contrast))
        return image.clamp(0.0, 1.0), target
