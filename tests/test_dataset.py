import pytest
import torch

from src.dataset import CocoDetectionDataset, collate_fn
from src.preprocessing import TrainAugment

ANN = "_annotations.coco.json"


def make(split_dir, aliases, **kw):
    return CocoDetectionDataset(split_dir, split_dir / ANN, aliases, **kw)


def test_length_and_sample_format(split_dir, aliases):
    ds = make(split_dir, aliases)
    assert len(ds) == 3
    img, tgt = ds[0]
    assert img.shape == (3, 100, 200) and img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert tgt["boxes"].shape == (2, 4) and tgt["boxes"].dtype == torch.float32
    assert set(tgt["labels"].tolist()) == {1, 2}
    # COCO [10,20,30,40] (x,y,w,h) -> [10,20,40,60] (x1,y1,x2,y2)
    assert tgt["boxes"][0].tolist() == [10, 20, 40, 60]
    assert tgt["area"][0].item() == 1200


def test_empty_annotations(split_dir, aliases):
    _, tgt = make(split_dir, aliases)[2]
    assert tgt["boxes"].shape == (0, 4) and tgt["labels"].shape == (0,)


def test_missing_image_raises(split_dir, aliases):
    (split_dir / "train_img1.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing on disk"):
        make(split_dir, aliases)[1]


def test_missing_annotation_file(split_dir, aliases):
    with pytest.raises(FileNotFoundError):
        CocoDetectionDataset(split_dir, split_dir / "nope.json", aliases)


def test_bad_alias_target(split_dir):
    with pytest.raises(ValueError):
        make(split_dir, {"player": "referee"})


def test_collate(split_dir, aliases):
    ds = make(split_dir, aliases)
    images, targets = collate_fn([ds[0], ds[2]])
    assert len(images) == 2 and len(targets) == 2


def test_hflip_updates_boxes(split_dir, aliases):
    aug = TrainAugment(flip_prob=1.0, brightness=0.0, contrast=0.0)
    _, tgt = make(split_dir, aliases, transform=aug)[0]
    # [10,20,40,60] in a 200px-wide image -> [160,20,190,60]
    assert tgt["boxes"][0].tolist() == [160, 20, 190, 60]
