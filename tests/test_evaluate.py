import torch

from src.evaluate import match_detections

BOX = torch.tensor([[0.0, 0.0, 10.0, 10.0]])


def test_perfect_match():
    tp, gm = match_detections(BOX, torch.tensor([0.9]), BOX)
    assert tp.tolist() == [True] and gm.tolist() == [True]


def test_false_positive_and_missed_object():
    far = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    tp, gm = match_detections(far, torch.tensor([0.9]), BOX)
    assert tp.tolist() == [False] and gm.tolist() == [False]


def test_duplicate_prediction_counts_once():
    preds = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0]])
    tp, gm = match_detections(preds, torch.tensor([0.9, 0.8]), BOX)
    assert tp.tolist() == [True, False] and gm.tolist() == [True]


def test_empty_inputs():
    tp, gm = match_detections(torch.zeros((0, 4)), torch.zeros(0), torch.zeros((0, 4)))
    assert len(tp) == 0 and len(gm) == 0
