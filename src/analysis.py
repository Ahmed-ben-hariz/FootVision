"""UI-independent analysis: filter detections, classify teams, render the result."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.predict import Detection, PredictionResult, predict_image
from src.team_classifier import (TEAM_A, TEAM_B, UNKNOWN, TeamAssignment,
                                 classify_teams)

DETECTION_FLOOR = 0.05  # run the model once at this score; filter higher in the UI
DISPLAY_COLORS = {      # RGB; fixed display colors, not the shirt colors
    TEAM_A: (255, 120, 0), TEAM_B: (0, 120, 255),
    UNKNOWN: (170, 170, 170), "ball": (255, 0, 0)}


@dataclass
class Analysis:
    players: list[Detection]
    balls: list[Detection]
    teams: TeamAssignment
    labels: list[str]          # final team label per player (after manual edits)

    def team_counts(self) -> dict[str, int]:
        return {k: self.labels.count(k) for k in (TEAM_A, TEAM_B, UNKNOWN)}


def run_detection(model, image: Image.Image, device, inf_cfg: dict,
                  mode: str) -> PredictionResult:
    """Run the detector once at a low score floor."""
    return predict_image(model, image, device, inf_cfg, mode=mode,
                         threshold=DETECTION_FLOOR)


def analyze(image: Image.Image, detections: list[Detection], threshold: float,
            team_cfg: dict) -> Analysis:
    """Filter by score, split players/balls, and group players into teams."""
    kept = [d for d in detections if d.score >= threshold]
    players = [d for d in kept if d.label == 1]
    balls = [d for d in kept if d.label == 2]
    teams = classify_teams(image, [d.box for d in players], **team_cfg)
    return Analysis(players, balls, teams, list(teams.labels))


def render(image: Image.Image, analysis: Analysis) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    thick = max(1, round(max(arr.shape[:2]) / 700))
    font = 0.4 + 0.1 * thick

    def draw(det: Detection, name: str, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, thick)
        cv2.putText(arr, f"{name} {det.score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, color, max(1, thick - 1))

    for det, label in zip(analysis.players, analysis.labels):
        draw(det, label, DISPLAY_COLORS[label])
    for det in analysis.balls:
        draw(det, "ball", DISPLAY_COLORS["ball"])
    return Image.fromarray(arr)
