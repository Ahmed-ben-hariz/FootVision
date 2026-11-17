"""Fine-tune Faster R-CNN on the football dataset.

Usage:
    python -m src.train                                  # settings from configs/config.yaml
    python -m src.train --epochs 1 --max-train 20 --max-val 10   # quick smoke test
"""
from __future__ import annotations

import os

# Must be set before torch touches MPS: unsupported ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection import MeanAveragePrecision

from src.config import ROOT, load_config
from src.data_quality import ANN_NAME
from src.dataset import CocoDetectionDataset, collate_fn
from src.device import get_device
from src.model import build_model, save_checkpoint
from src.preprocessing import TrainAugment

CLASS_NAMES = {1: "player", 2: "ball"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_subset(ds, n: int | None, seed: int):
    """Seeded random subset (or the full dataset if n is None / too large)."""
    if n is None or n >= len(ds):
        return ds
    return Subset(ds, sorted(random.Random(seed).sample(range(len(ds)), n)))


def train_one_epoch(model, loader, optimizer, device, epoch: int,
                    log_every: int = 50) -> float:
    """One pass over the training data. Returns the mean total loss."""
    model.train()
    warmup = None
    if epoch == 0 and len(loader) > 1:  # linear LR warmup during the first epoch
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=min(200, len(loader) - 1))
    total, n, t0 = 0.0, 0, time.time()
    for i, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)        # dict of 4 detection losses in train mode
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch} iter {i}: "
                f"{ {k: float(v) for k, v in losses.items()} }. "
                "Try a lower learning rate.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if warmup is not None:
            warmup.step()
        total += loss.item()
        n += 1
        if (i + 1) % log_every == 0:
            print(f"  epoch {epoch + 1} iter {i + 1}/{len(loader)} "
                  f"loss {total / n:.4f} ({(time.time() - t0) / (i + 1):.2f}s/iter)")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    """Compute COCO-style mAP/mAR on a loader (not a loss)."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        preds = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    r = metric.compute()
    out = {"map": float(r["map"]), "map_50": float(r["map_50"]),
           "map_75": float(r["map_75"]), "mar_100": float(r["mar_100"])}
    classes = torch.atleast_1d(r["classes"]).tolist()
    per_class = torch.atleast_1d(r["map_per_class"]).tolist()
    out["map_per_class"] = {CLASS_NAMES.get(c, str(c)): v for c, v in zip(classes, per_class)}
    return out


def save_plots(history: list[dict], plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    plt.figure(); plt.plot(ep, [h["train_loss"] for h in history], marker="o")
    plt.xlabel("epoch"); plt.ylabel("mean training loss"); plt.title("Training loss")
    plt.savefig(plots_dir / "train_loss.png", dpi=120, bbox_inches="tight"); plt.close()
    plt.figure()
    for key in ("map_50", "map", "mar_100"):
        plt.plot(ep, [h["val"][key] for h in history], marker="o", label=key)
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Validation metrics"); plt.legend()
    plt.savefig(plots_dir / "val_metrics.png", dpi=120, bbox_inches="tight"); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--max-val", type=int)
    ap.add_argument("--min-size", type=int)
    ap.add_argument("--max-size", type=int)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--anchors", help="comma-separated anchor sizes, e.g. 16,32,64,128,256")
    args = ap.parse_args()

    cfg = load_config()
    tc = cfg["train"]
    if args.min_size:
        tc["min_size"] = args.min_size
        tc["max_size"] = args.max_size or max(args.min_size, tc["max_size"])
    epochs = args.epochs or tc["epochs"]
    seed = cfg["seed"]
    seed_everything(seed)
    device = get_device(cfg["device"])

    root = ROOT / cfg["data"]["dataset_dir"]
    aliases = cfg["data"]["class_aliases"]
    train_full = CocoDetectionDataset(root / "train", root / "train" / ANN_NAME,
                                      aliases, transform=TrainAugment())
    val_full = CocoDetectionDataset(root / "valid", root / "valid" / ANN_NAME, aliases)
    train_ds = make_subset(train_full, args.max_train or tc.get("max_train_images"), seed)
    val_ds = make_subset(val_full, args.max_val or tc.get("max_val_images"), seed)

    workers = cfg["data"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=workers, collate_fn=collate_fn,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                            num_workers=workers, collate_fn=collate_fn)
    print(f"Device: {device} | train images: {len(train_ds)} | val images: {len(val_ds)}")

    anchor_sizes = tuple(int(x) for x in args.anchors.split(",")) if args.anchors else None
    model = build_model(pretrained=True, min_size=tc["min_size"], max_size=tc["max_size"],
                        anchor_sizes=anchor_sizes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=tc["lr"], momentum=tc["momentum"],
                                weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=tc["lr_step_size"], gamma=tc["lr_gamma"])

    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    results_dir = ROOT / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = -1.0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val = validate(model, val_loader, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val": val,
                        "seconds": time.time() - t0})
        meta = {"epoch": epoch + 1, "val_metrics": val, "history": history,
                "device": str(device), "min_size": tc["min_size"],
                "max_size": tc["max_size"], "n_train_images": len(train_ds),
                "n_val_images": len(val_ds), "dataset_dir": cfg["data"]["dataset_dir"],
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "class_names": CLASS_NAMES, "anchor_sizes": anchor_sizes}
        save_checkpoint(ckpt_dir / f"{args.run_name}_last.pt", model, **meta)
        if val["map_50"] > best:
            best = val["map_50"]
            save_checkpoint(ckpt_dir / f"{args.run_name}_best.pt", model, **meta)
        (results_dir / f"history_{args.run_name}.json").write_text(json.dumps(history, indent=2))
        save_plots(history, ROOT / cfg["paths"]["plots"] / args.run_name)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss {train_loss:.4f} | "
              f"mAP@0.5 {val['map_50']:.3f} | mAP {val['map']:.3f} | "
              f"mAR@100 {val['mar_100']:.3f} | per-class mAP {val['map_per_class']} | "
              f"{history[-1]['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
