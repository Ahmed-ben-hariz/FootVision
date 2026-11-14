"""Evaluate a checkpoint: mAP, precision/recall at a threshold, counting MAE,
and saved error examples.

Usage:
    python -m src.evaluate --split valid [--checkpoint models/best.pt] [--max-images N]
Use --split test only for the final report; never tune on it.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from src.config import ROOT, load_config
from src.data_quality import ANN_NAME
from src.dataset import CocoDetectionDataset, collate_fn
from src.device import get_device
from src.model import load_checkpoint
from src.train import make_subset

NAMES = {1: "player", 2: "ball"}


def match_detections(pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                     gt_boxes: torch.Tensor, iou_thr: float = 0.5):
    """Greedy one-to-one matching, highest score first.

    Returns (tp, gt_matched): tp[i] is True if prediction i matched a ground
    truth box with IoU >= iou_thr; gt_matched[j] is True if ground truth j was found.
    """
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    tp = np.zeros(n_p, dtype=bool)
    gt_matched = np.zeros(n_g, dtype=bool)
    if n_p and n_g:
        iou = box_iou(pred_boxes, gt_boxes).numpy()
        for p in np.argsort(-pred_scores.numpy()):
            free = np.where(~gt_matched)[0]
            if len(free) == 0:
                break
            g = free[np.argmax(iou[p, free])]
            if iou[p, g] >= iou_thr:
                tp[p], gt_matched[g] = True, True
    return tp, gt_matched


def _ratio(num: int, den: int):
    return num / den if den else None


@torch.no_grad()
def evaluate(model, ds, device, score_thr: float, iou_thr: float, batch_size: int):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    stats = {c: dict(tp=0, fp=0, fn=0, abs_err=0, pred_total=0, gt_total=0) for c in NAMES}
    records, offset = [], 0
    for images, targets in loader:
        outs = model([im.to(device) for im in images])
        for j, (out, tgt) in enumerate(zip(outs, targets)):
            out = {k: v.detach().cpu() for k, v in out.items()}
            metric.update([out], [{"boxes": tgt["boxes"], "labels": tgt["labels"]}])
            keep = out["scores"] >= score_thr
            pb, pl, ps = out["boxes"][keep], out["labels"][keep], out["scores"][keep]
            rec = {"idx": offset + j, "pb": pb, "pl": pl, "gb": tgt["boxes"],
                   "gl": tgt["labels"], "tp": np.zeros(len(pb), bool),
                   "gm": np.zeros(len(tgt["boxes"]), bool), "errors": 0}
            for c in NAMES:
                pm, gm = pl == c, tgt["labels"] == c
                tp, gt_matched = match_detections(pb[pm], ps[pm], tgt["boxes"][gm], iou_thr)
                s = stats[c]
                s["tp"] += int(tp.sum()); s["fp"] += int((~tp).sum())
                s["fn"] += int((~gt_matched).sum())
                s["abs_err"] += abs(int(pm.sum()) - int(gm.sum()))
                s["pred_total"] += int(pm.sum()); s["gt_total"] += int(gm.sum())
                rec["tp"][pm.numpy()] = tp
                rec["gm"][gm.numpy()] = gt_matched
                rec["errors"] += int((~tp).sum()) + int((~gt_matched).sum())
            records.append(rec)
        offset += len(images)
    r = metric.compute()
    n = len(records)
    result = {"n_images": n, "score_threshold": score_thr, "iou_threshold": iou_thr,
              "map": float(r["map"]), "map_50": float(r["map_50"]),
              "mar_100": float(r["mar_100"]), "classes": {}}
    for c, name in NAMES.items():
        s = stats[c]
        result["classes"][name] = {
            "precision": _ratio(s["tp"], s["tp"] + s["fp"]),
            "recall": _ratio(s["tp"], s["tp"] + s["fn"]),
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "count_mae": s["abs_err"] / n if n else None,
            "gt_total": s["gt_total"], "pred_total": s["pred_total"]}
    return result, records


def save_error_images(ds, records, out_dir: Path, top_n: int) -> int:
    """Blue = ground truth, green = correct detection, red = false detection,
    orange (thick) = missed ground truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted((r for r in records if r["errors"] > 0),
                   key=lambda r: -r["errors"])[:top_n]
    for r in worst:
        img, _ = ds[r["idx"]]
        im = cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                          cv2.COLOR_RGB2BGR)
        for b in r["gb"]:
            cv2.rectangle(im, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]),
                          (255, 100, 0), 1)
        for b, ok in zip(r["gb"], r["gm"]):
            if not ok:
                cv2.rectangle(im, tuple(int(v) for v in b[:2]),
                              tuple(int(v) for v in b[2:]), (0, 140, 255), 3)
        for b, ok in zip(r["pb"], r["tp"]):
            cv2.rectangle(im, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]),
                          (0, 200, 0) if ok else (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"err_{r['idx']}_n{r['errors']}.jpg"), im)
    return len(worst)


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def write_report(path: Path, split: str, res: dict, meta: dict, ckpt: str,
                 n_errors_saved: int) -> None:
    lines = [
        f"# Evaluation report ({split} split)", "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Checkpoint: `{ckpt}` (epoch {meta.get('epoch')}, trained {meta.get('trained_at')})",
        f"- Model: Faster R-CNN MobileNetV3-Large FPN, COCO-pretrained, fine-tuned on "
        f"{meta.get('n_train_images')} training images, min_size={meta.get('min_size')}",
        f"- Dataset: `{meta.get('dataset_dir')}`; evaluated images: {res['n_images']}",
        f"- Score threshold {res['score_threshold']}, IoU threshold {res['iou_threshold']}",
        "", "## Detection quality (all classes)",
        f"- mAP@0.5: {fmt(res['map_50'])}",
        f"- mAP@[0.5:0.95]: {fmt(res['map'])}",
        f"- mAR@100: {fmt(res['mar_100'])}",
        "", "## Per class at the chosen threshold",
        "| Class | Precision | Recall | TP | FP | FN | Count MAE (per image) |",
        "|---|---|---|---|---|---|---|"]
    for name, c in res["classes"].items():
        lines.append(f"| {name} | {fmt(c['precision'])} | {fmt(c['recall'])} | {c['tp']} | "
                     f"{c['fp']} | {c['fn']} | {fmt(c['count_mae'])} |")
    lines += ["", f"Error examples saved: {n_errors_saved} (results/errors/).",
              "", "## Known limitations",
              "- Counts are per dataset image (tiles/crops), not per full match frame.",
              "- Source frames of the dataset may share matches or adjacent video frames "
              "across splits; this cannot be verified from the dataset documentation, so "
              "scores may be optimistic.",
              "- Precision, recall and counts depend on the score threshold above."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--checkpoint", default="models/best.pt")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--errors", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    device = get_device(cfg["device"])
    model, meta = load_checkpoint(ROOT / args.checkpoint, device)
    if "min_size" in meta:  # evaluate at the same scale the model was trained at
        model.transform.min_size = (meta["min_size"],)
        model.transform.max_size = meta["max_size"]

    root = ROOT / cfg["data"]["dataset_dir"] / args.split
    full = CocoDetectionDataset(root, root / ANN_NAME, cfg["data"]["class_aliases"])
    ds = make_subset(full, args.max_images, cfg["seed"])
    res, records = evaluate(model, ds, device, cfg["inference"]["confidence_threshold"],
                            0.5, cfg["train"]["batch_size"])
    saved = save_error_images(ds, records, ROOT / "results" / "errors", args.errors)
    out = ROOT / "results" / ("evaluation_report.md" if args.split == "test"
                              else f"evaluation_{args.split}_{Path(args.checkpoint).stem}.md")
    write_report(out, args.split, res, meta, args.checkpoint, saved)
    (ROOT / "results" / f"metrics_{args.split}_{Path(args.checkpoint).stem}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
