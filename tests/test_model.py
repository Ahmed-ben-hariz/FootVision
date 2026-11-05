import pytest
import torch

from src.model import NUM_CLASSES, build_model, load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False, min_size=320, max_size=512)


def test_head_has_three_outputs(model):
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES


def test_train_mode_returns_losses(model):
    model.train()
    images = [torch.rand(3, 240, 320)]
    targets = [{"boxes": torch.tensor([[10.0, 20.0, 100.0, 120.0]]),
                "labels": torch.tensor([1])}]
    losses = model(images, targets)
    assert {"loss_classifier", "loss_box_reg", "loss_objectness",
            "loss_rpn_box_reg"} <= set(losses)
    assert all(torch.isfinite(v) for v in losses.values())


def test_train_mode_accepts_empty_target(model):
    model.train()
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}]
    losses = model([torch.rand(3, 240, 320)], targets)
    assert all(torch.isfinite(v) for v in losses.values())


def test_eval_output_format(model):
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 240, 320)])
    assert len(out) == 1
    assert set(out[0]) == {"boxes", "labels", "scores"}
    assert out[0]["boxes"].shape[1] == 4
    assert len(out[0]["boxes"]) == len(out[0]["labels"]) == len(out[0]["scores"])


def test_checkpoint_roundtrip(model, tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, epoch=3, note="test")
    loaded, meta = load_checkpoint(path, "cpu", min_size=320, max_size=512)
    assert meta["epoch"] == 3 and meta["num_classes"] == NUM_CLASSES
    assert not loaded.training
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pt")
