import numpy as np
import pytest
import torch
from PIL import Image

from src.model import build_model
from src.predict import (detect_single, detect_tiled, draw_detections,
                         load_image, predict_image)

CPU = torch.device("cpu")
INF = {"mode": "single", "confidence_threshold": 0.0, "square_size": 256,
       "tile_grid": 2, "tile_overlap": 0.1, "nms_iou": 0.5}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False, min_size=128, max_size=256)


@pytest.fixture
def image():
    arr = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def check_valid(result, w, h):
    for d in result.detections:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h
        assert 0.0 <= d.score <= 1.0
        assert d.label in (1, 2)


@pytest.mark.parametrize("mode", ["single", "tiled"])
def test_valid_coordinates_scores_and_counts(model, image, mode):
    res = predict_image(model, image, CPU, INF, mode=mode)
    check_valid(res, *image.size)
    assert res.n_players + res.n_balls == len(res.detections)
    assert res.n_players == sum(d.label == 1 for d in res.detections)


@pytest.mark.parametrize("fn", [detect_single, detect_tiled])
def test_zero_detections_are_handled(model, image, fn):
    res = fn(model, image, CPU, 1.01, 256)
    assert res.detections == [] and res.n_players == 0 and res.n_balls == 0
    assert draw_detections(image, res.detections).size == image.size


def test_draw_with_detections(model, image):
    res = detect_single(model, image, CPU, 0.0, 256)
    assert draw_detections(image, res.detections).size == image.size


def test_missing_image_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(tmp_path / "nope.jpg")


def test_unknown_mode(model, image):
    with pytest.raises(ValueError):
        predict_image(model, image, CPU, INF, mode="bogus")


def test_suppress_duplicates():
    from src.predict import Detection, suppress_duplicates
    big = Detection((0, 0, 100, 200), 1, 0.9)
    inside = Detection((10, 20, 60, 150), 1, 0.6)            # mostly inside big: duplicate
    neighbour = Detection((120, 0, 220, 200), 1, 0.8)        # separate player: keep
    other_class = Detection((10, 20, 60, 150), 2, 0.7)       # different class: keep
    kept = suppress_duplicates([big, inside, neighbour, other_class], 0.7)
    assert inside not in kept
    assert big in kept and neighbour in kept and other_class in kept
    assert suppress_duplicates([], 0.7) == []
