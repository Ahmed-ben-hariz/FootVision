# Evaluation report (valid split)

- Generated: 2026-11-05T22:52:58+00:00
- Checkpoint: `models/best.pt` (epoch 3, trained 2026-11-05T19:37:34.330207+00:00)
- Model: Faster R-CNN MobileNetV3-Large FPN, COCO-pretrained, fine-tuned on 1500 training images, min_size=640
- Dataset: `data/raw/Football-Player-Detection.v8-resized1280_tile2x2_aug3x.coco`; evaluated images: 972
- Score threshold 0.5, IoU threshold 0.5

## Detection quality (all classes)
- mAP@0.5: 0.494
- mAP@[0.5:0.95]: 0.217
- mAR@100: 0.277

## Per class at the chosen threshold
| Class | Precision | Recall | TP | FP | FN | Count MAE (per image) |
|---|---|---|---|---|---|---|
| player | 0.822 | 0.788 | 4046 | 879 | 1086 | 0.962 |
| ball | 0.456 | 0.189 | 52 | 62 | 223 | 0.283 |

Error examples saved: 20 (results/errors/).

## Known limitations
- Counts are per dataset image (tiles/crops), not per full match frame.
- Source frames of the dataset may share matches or adjacent video frames across splits; this cannot be verified from the dataset documentation, so scores may be optimistic.
- Precision, recall and counts depend on the score threshold above.
