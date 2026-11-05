import math

import torch
from torch.utils.data import DataLoader

from src.dataset import CocoDetectionDataset, collate_fn
from src.model import build_model
from src.preprocessing import TrainAugment
from src.train import make_subset, train_one_epoch, validate

ANN = "_annotations.coco.json"


def test_one_epoch_and_validate(split_dir, aliases):
    ds = CocoDetectionDataset(split_dir, split_dir / ANN, aliases, transform=TrainAugment())
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    model = build_model(pretrained=False, min_size=128, max_size=256)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss = train_one_epoch(model, loader, opt, torch.device("cpu"), 0, log_every=1000)
    assert math.isfinite(loss)
    metrics = validate(model, loader, torch.device("cpu"))
    assert {"map", "map_50", "mar_100", "map_per_class"} <= set(metrics)


def test_subset_is_reproducible(split_dir, aliases):
    ds = CocoDetectionDataset(split_dir, split_dir / ANN, aliases)
    a, b = make_subset(ds, 2, seed=1), make_subset(ds, 2, seed=1)
    assert a.indices == b.indices and len(a) == 2
    assert make_subset(ds, None, 1) is ds
