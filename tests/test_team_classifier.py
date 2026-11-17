import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.team_classifier import (TEAM_A, TEAM_B, UNKNOWN, apply_overrides,
                                 classify_teams, dominant_feature, torso_crop)

GRASS = (30, 140, 30)
BLUE, WHITE, YELLOW = (0, 0, 255), (255, 255, 255), (255, 255, 0)


def scene(colors):
    img = Image.new("RGB", (60 * len(colors) + 20, 200), GRASS)
    draw, boxes = ImageDraw.Draw(img), []
    for i, c in enumerate(colors):
        x = 10 + i * 60
        draw.rectangle([x, 20, x + 40, 120], fill=c)
        boxes.append((x, 20, x + 40, 120))
    return img, boxes


def test_two_teams_and_referee():
    img, boxes = scene([BLUE] * 3 + [WHITE] * 3 + [YELLOW])
    res = classify_teams(img, boxes)
    assert res.separable
    assert len(set(res.labels[:3])) == 1 and len(set(res.labels[3:6])) == 1
    assert res.labels[0] != res.labels[3] and res.labels[0] in (TEAM_A, TEAM_B)
    assert res.labels[6] == UNKNOWN
    assert res.counts()[UNKNOWN] == 1 and set(res.colors) == {TEAM_A, TEAM_B}


def test_team_a_is_the_larger_group():
    img, boxes = scene([BLUE] * 4 + [WHITE] * 2)
    res = classify_teams(img, boxes)
    assert res.labels[:4] == [TEAM_A] * 4 and res.labels[4:] == [TEAM_B] * 2


def test_similar_colors_are_not_forced_into_teams():
    img, boxes = scene([(200, 0, 0)] * 3 + [(210, 10, 10)] * 3)
    res = classify_teams(img, boxes)
    assert not res.separable and set(res.labels) == {UNKNOWN} and "similar" in res.note


def test_too_few_players():
    img, boxes = scene([BLUE])
    res = classify_teams(img, boxes)
    assert res.labels == [UNKNOWN] and not res.separable
    assert classify_teams(img, []).labels == []


def test_pitch_pixels_are_ignored():
    green = np.full((20, 20, 3), GRASS, dtype=np.uint8)
    assert dominant_feature(green) is None
    img, boxes = scene([BLUE, BLUE, WHITE, WHITE])
    boxes.append((10, 150, 50, 190))                 # box over bare grass
    assert classify_teams(img, boxes).labels[-1] == UNKNOWN


def test_torso_crop_clamps_and_handles_degenerate_boxes():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    assert torso_crop(arr, (-50, -50, 20, 20)).ndim == 3
    assert torso_crop(arr, (200, 200, 300, 300)).shape[0] == 0
    assert dominant_feature(torso_crop(arr, (30, 30, 30, 30))) is None


def test_overrides():
    out = apply_overrides([TEAM_A, UNKNOWN], {1: TEAM_B})
    assert out == [TEAM_A, TEAM_B]
    with pytest.raises(ValueError):
        apply_overrides([TEAM_A], {0: "Team C"})
    with pytest.raises(IndexError):
        apply_overrides([TEAM_A], {5: TEAM_B})
