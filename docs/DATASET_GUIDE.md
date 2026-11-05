# Dataset Guide

## Task classes

| Model label | Class      |
|-------------|------------|
| 0           | Background (reserved by Torchvision) |
| 1           | Player     |
| 2           | Ball       |

Goalkeepers and referees are not separate classes in v1. Datasets that contain
them must be remapped or filtered explicitly (see `src/preprocessing.py`).

## Candidate sources

Only datasets whose page I have personally opened and checked are listed as
"verified". Fill in the fields below after checking the dataset page.

| Field | Value |
|---|---|
| Name | Football-Player-Detection (Augmented Startups) |
| URL | https://universe.roboflow.com/augmented-startups/football-player-detection-kucab |
| License | TODO: copy from the dataset page |
| Classes | football, player (as listed on the page) |
| Images / instances | TODO: copy from the dataset page |
| Export format used | COCO JSON |
| Access | TODO: note whether a Roboflow account was needed |
| Date downloaded | TODO |

Other candidates considered: SoccerNet-Tracking (12 Swiss Super League matches,
MOT-style annotations; access terms must be checked on the official site before
use).

## Known risks

1. **Data leakage.** Roboflow exports may contain augmented copies of the same
   frame, and adjacent video frames. `src/data_quality.py` checks for duplicates
   across splits. Where the source match is unknown, this limitation is stated in
   `results/evaluation_report.md`.
2. **Class mismatch.** Source datasets may use different class names; all are
   mapped to the two classes above.
3. **Annotation quality is not guaranteed.** We visually inspect samples and run
   the validation script before training.

## Splits

Target 70/15/15, reproducible via the seed in `configs/config.yaml`. If the
source already provides splits, we keep them and report the sizes honestly.
