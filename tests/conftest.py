import json

import numpy as np
import pytest
from PIL import Image

ALIASES = {"player": "player", "football": "ball"}


def _make_split(root, name="train", n=3, w=200, h=100):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"{name}_img{i}.png")
    coco = {
        "images": [{"id": i, "file_name": f"{name}_img{i}.png", "width": w, "height": h}
                   for i in range(n)],
        "categories": [{"id": 0, "name": "dataset"}, {"id": 1, "name": "football"},
                       {"id": 2, "name": "player"}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [10, 20, 30, 40],
             "area": 1200, "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [100, 50, 8, 8],
             "area": 64, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0, 0, 50, 50],
             "area": 2500, "iscrowd": 0},
        ],  # image 2 has no annotations on purpose
    }
    (d / "_annotations.coco.json").write_text(json.dumps(coco))
    return d


@pytest.fixture
def aliases():
    return ALIASES


@pytest.fixture
def split_dir(tmp_path):
    return _make_split(tmp_path)


@pytest.fixture
def make_split(tmp_path):
    return lambda name, **kw: _make_split(tmp_path, name, **kw)
