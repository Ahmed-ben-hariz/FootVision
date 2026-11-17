import numpy as np
from PIL import Image, ImageDraw

from src.analysis import analyze, render
from src.info import dataset_summary, read_json, read_text
from src.predict import Detection

GRASS = (30, 140, 30)
CFG = {"max_distance": 0.35, "min_separation": 0.2}


def scene():
    colors = [(0, 0, 255)] * 3 + [(255, 255, 255)] * 3
    img = Image.new("RGB", (60 * len(colors) + 20, 200), GRASS)
    draw, dets = ImageDraw.Draw(img), []
    for i, c in enumerate(colors):
        x = 10 + i * 60
        draw.rectangle([x, 20, x + 40, 120], fill=c)
        dets.append(Detection((x, 20, x + 40, 120), 1, 0.9 - i * 0.05))
    return img, dets


def test_threshold_filters_players_and_balls():
    img, dets = scene()
    dets.append(Detection((5, 150, 15, 160), 2, 0.6))
    a = analyze(img, dets, 0.5, CFG)
    assert len(a.players) == 6 and len(a.balls) == 1 and len(a.labels) == 6
    b = analyze(img, dets, 0.78, CFG)
    assert len(b.players) == 3 and len(b.balls) == 0


def test_teams_are_grouped():
    img, dets = scene()
    a = analyze(img, dets, 0.5, CFG)
    assert a.teams.separable
    assert a.labels[0] != a.labels[3] and a.team_counts()["Unknown"] == 0


def test_render_draws_and_handles_empty():
    img, dets = scene()
    out = render(img, analyze(img, dets, 0.5, CFG))
    assert out.size == img.size and not np.array_equal(np.array(out), np.array(img))
    empty = analyze(img, [], 0.5, CFG)
    assert empty.players == [] and empty.labels == []
    assert np.array_equal(np.array(render(img, empty)), np.array(img))


def test_dataset_summary(make_split, aliases, tmp_path):
    make_split("train")
    assert dataset_summary(tmp_path, aliases) == {
        "train": {"images": 3, "player": 2, "ball": 1}}
    assert dataset_summary(tmp_path / "missing", aliases) is None
    assert dataset_summary(tmp_path / "train", aliases) is None  # no split folders


def test_read_helpers(tmp_path):
    assert read_json(tmp_path / "x.json") is None and read_text(tmp_path / "x.md") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert read_json(tmp_path / "bad.json") is None
    (tmp_path / "ok.json").write_text('{"a": 1}')
    assert read_json(tmp_path / "ok.json") == {"a": 1}
