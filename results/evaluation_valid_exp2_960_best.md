# Evaluation report (valid split)

- Generated: 2026-11-17T15:32:46+00:00
- Checkpoint: `models/exp2_960_best.pt` (epoch 3, trained 2026-11-14T21:06:00.843491+00:00)
- Model: Faster R-CNN MobileNetV3-Large FPN, COCO-pretrained, fine-tuned on 1500 training images, min_size=960
- Dataset: `data/raw/Football-Player-Detection.v8-resized1280_tile2x2_aug3x.coco`; evaluated images: 972
- Score threshold 0.5, IoU threshold 0.5

## Detection quality (all classes)
- mAP@0.5: 0.592
- mAP@[0.5:0.95]: 0.292
- mAR@100: 0.356

## Per class at the chosen threshold
| Class | Precision | Recall | TP | FP | FN | Count MAE (per image) |
|---|---|---|---|---|---|---|
| player | 0.851 | 0.865 | 4438 | 780 | 694 | 0.780 |
| ball | 0.527 | 0.320 | 88 | 79 | 187 | 0.253 |

Error examples saved: 20 (results/errors/).

## Known limitations
- Counts are per dataset image (tiles/crops), not per full match frame.
- Source frames of the dataset may share matches or adjacent video frames across splits; this cannot be verified from the dataset documentation, so scores may be optimistic.
- Precision, recall and counts depend on the score threshold above.
