"""Dataset validation for COCO-format splits, with a Markdown quality report.

Usage: python -m src.data_quality --root data/raw/<dataset-folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ANN_NAME = "_annotations.coco.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class QualityReport:
    split: str
    n_images: int = 0
    n_annotations: int = 0
    n_empty_images: int = 0
    class_counts: Counter = field(default_factory=Counter)
    issues: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @property
    def n_issues(self) -> int:
        return sum(len(v) for v in self.issues.values())


def check_coco(images_dir: str | Path, ann_path: str | Path,
               split: str = "dataset") -> QualityReport:
    images_dir, ann_path = Path(images_dir), Path(ann_path)
    report = QualityReport(split=split)
    if not ann_path.is_file():
        report.issues["missing_annotation_file"].append(str(ann_path))
        return report
    try:
        coco = json.loads(ann_path.read_text())
    except json.JSONDecodeError as e:
        report.issues["invalid_annotation_json"].append(f"{ann_path}: {e}")
        return report

    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images = {im["id"]: im for im in coco.get("images", [])}
    report.n_images = len(images)

    sizes: dict[int, tuple[int, int]] = {}
    for im_id, im in images.items():
        path = images_dir / im["file_name"]
        if not path.is_file():
            report.issues["missing_image_file"].append(str(path))
            continue
        try:
            with Image.open(path) as img:
                size = img.size
        except Exception as e:  # unreadable/corrupt file
            report.issues["unreadable_image"].append(f"{path}: {e}")
            continue
        sizes[im_id] = size
        if size != (im.get("width"), im.get("height")):
            report.issues["size_mismatch"].append(
                f"{path.name}: file {size} vs json {(im.get('width'), im.get('height'))}")

    seen: set = set()
    annotated: set = set()
    for ann in coco.get("annotations", []):
        report.n_annotations += 1
        label = f"ann {ann.get('id')}"
        im_id = ann.get("image_id")
        if im_id not in images:
            report.issues["unknown_image_id"].append(f"{label}: image_id={im_id}")
            continue
        annotated.add(im_id)
        cat = ann.get("category_id")
        if cat not in cats:
            report.issues["unknown_category_id"].append(f"{label}: category_id={cat}")
        else:
            report.class_counts[cats[cat]] += 1
        bbox = ann.get("bbox")
        valid = (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and
                 all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox))
        if not valid or bbox[2] <= 0 or bbox[3] <= 0:
            report.issues["invalid_bbox"].append(f"{label}: bbox={bbox}")
            continue
        x, y, w, h = bbox
        if im_id in sizes:
            W, H = sizes[im_id]
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                report.issues["out_of_bounds_bbox"].append(
                    f"{label}: bbox={bbox} image={W}x{H}")
        key = (im_id, cat, tuple(round(v, 1) for v in bbox))
        if key in seen:
            report.issues["duplicate_annotation"].append(f"{label}: {key}")
        seen.add(key)

    report.n_empty_images = len(set(images) - annotated)
    return report


def find_cross_split_duplicates(split_dirs: dict[str, Path]) -> list[list[str]]:
    """Exact (byte-identical) image duplicates that appear in more than one split.
    Augmented copies are NOT byte-identical, so this is a minimum check only."""
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, d in split_dirs.items():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                by_hash[hashlib.sha256(p.read_bytes()).hexdigest()].append((split, p.name))
    return [[f"{s}/{n}" for s, n in group] for group in by_hash.values()
            if len({s for s, _ in group}) > 1]


def write_markdown(reports: list[QualityReport], duplicates: list[list[str]],
                   out_path: Path, max_examples: int = 10) -> None:
    lines = ["# Dataset Quality Report", ""]
    lines += ["| Split | Images | Annotations | Empty images | Class counts | Issues |",
              "|---|---|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r.split} | {r.n_images} | {r.n_annotations} | {r.n_empty_images} "
                     f"| {dict(r.class_counts)} | {r.n_issues} |")
    for r in reports:
        if not r.issues:
            continue
        lines += ["", f"## Issues in `{r.split}`"]
        for kind, items in r.issues.items():
            lines.append(f"- **{kind}**: {len(items)}")
            lines += [f"    - {x}" for x in items[:max_examples]]
    lines += ["", "## Cross-split exact duplicates", f"Groups found: {len(duplicates)}"]
    lines += [f"- {', '.join(g)}" for g in duplicates[:max_examples]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Folder holding split subfolders")
    ap.add_argument("--out", default="results/dataset_quality_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    split_dirs = {d.name: d for d in sorted(root.iterdir())
                  if d.is_dir() and (d / ANN_NAME).is_file()}
    if not split_dirs:
        print(f"No split folders containing {ANN_NAME} found under {root}", file=sys.stderr)
        return 2
    reports = [check_coco(d, d / ANN_NAME, split=name) for name, d in split_dirs.items()]
    duplicates = find_cross_split_duplicates(split_dirs)
    write_markdown(reports, duplicates, Path(args.out))
    for r in reports:
        print(f"{r.split}: {r.n_images} images, {r.n_annotations} annotations, "
              f"{r.n_issues} issues, classes={dict(r.class_counts)}")
    print(f"Cross-split duplicate groups: {len(duplicates)}\nReport: {args.out}")
    return 1 if (any(r.n_issues for r in reports) or duplicates) else 0


if __name__ == "__main__":
    sys.exit(main())
