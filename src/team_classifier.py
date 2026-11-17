"""Color-based team grouping. A heuristic on shirt colors, NOT team identity.

Players whose shirt is far from both team colors (referee, goalkeeper) or whose
crop has too few non-pitch pixels are labelled Unknown rather than forced into a team.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

UNKNOWN = "Unknown"
TEAM_A, TEAM_B = "Team A", "Team B"
Box = tuple[float, float, float, float]


@dataclass
class TeamAssignment:
    labels: list[str]                                   # one per input box
    colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # RGB
    separable: bool = False
    note: str = ""

    def counts(self) -> dict[str, int]:
        return {k: self.labels.count(k) for k in (TEAM_A, TEAM_B, UNKNOWN)}


def torso_crop(image_rgb: np.ndarray, box: Box,
               x_frac=(0.25, 0.75), y_frac=(0.2, 0.55)) -> np.ndarray:
    """Approximate torso region of a player box (clamped to the image)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return image_rgb[int(y1 + bh * y_frac[0]): int(y1 + bh * y_frac[1]),
                     int(x1 + bw * x_frac[0]): int(x1 + bw * x_frac[1])]


def dominant_feature(crop_rgb: np.ndarray, min_pixels: int = 30) -> np.ndarray | None:
    """Median shirt color as a 3-vector (s*cos(hue), s*sin(hue), v), or None.

    Grass-like pixels (green hue, saturated) are removed first. Hue is circular, so
    hue and saturation are mapped to Cartesian coordinates.
    """
    if crop_rgb.ndim != 3 or crop_rgb.shape[0] == 0 or crop_rgb.shape[1] == 0:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2HSV)
    hsv = hsv.reshape(-1, 3).astype(np.float64)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    pitch = (h >= 35) & (h <= 85) & (s >= 50) & (v >= 40)   # OpenCV hue is 0-179
    keep = ~pitch
    if keep.sum() < min_pixels:
        return None
    ang = h[keep] * 2 * np.pi / 180
    feats = np.stack([s[keep] / 255 * np.cos(ang), s[keep] / 255 * np.sin(ang),
                      v[keep] / 255], axis=1)
    return np.median(feats, axis=0)


def feature_to_rgb(f: np.ndarray) -> tuple[int, int, int]:
    """Convert a feature vector back to an RGB color for display."""
    sat = float(np.hypot(f[0], f[1]))
    hue = (np.arctan2(f[1], f[0]) % (2 * np.pi)) * 180 / (2 * np.pi)
    hsv = np.array([[[min(hue, 179), min(sat, 1) * 255, min(max(f[2], 0), 1) * 255]]],
                   dtype=np.uint8)
    r, g, b = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


def classify_teams(image: Image.Image, boxes: list[Box], max_distance: float = 0.35,
                   min_separation: float = 0.2, seed: int = 42) -> TeamAssignment:
    """Group player boxes into Team A / Team B / Unknown by torso color."""
    arr = np.array(image.convert("RGB"))
    feats = [dominant_feature(torso_crop(arr, b)) for b in boxes]
    labels = [UNKNOWN] * len(boxes)
    valid = [i for i, f in enumerate(feats) if f is not None]
    if len(valid) < 2:
        return TeamAssignment(labels, note="Fewer than 2 players with usable shirt pixels.")

    X = np.array([feats[i] for i in valid])
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
    centers = km.cluster_centers_
    dist = np.linalg.norm(X[:, None, :] - centers[None], axis=2).min(axis=1)
    inliers = dist <= max_distance
    if inliers.sum() >= 2 and not inliers.all():      # refit without odd shirts
        centers = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X[inliers]).cluster_centers_

    if np.linalg.norm(centers[0] - centers[1]) < min_separation:
        return TeamAssignment(labels, note="Team colors are too similar to separate.")

    d_all = np.linalg.norm(X[:, None, :] - centers[None], axis=2)
    nearest, dmin = d_all.argmin(axis=1), d_all.min(axis=1)
    assigned = dmin <= max_distance
    sizes = [int(((nearest == c) & assigned).sum()) for c in (0, 1)]
    order = sorted((0, 1), key=lambda c: -sizes[c])   # Team A = the larger group
    names = {order[0]: TEAM_A, order[1]: TEAM_B}
    for j, i in enumerate(valid):
        if assigned[j]:
            labels[i] = names[int(nearest[j])]
    colors = {names[c]: feature_to_rgb(centers[c]) for c in (0, 1)}
    return TeamAssignment(labels, colors, True)


def apply_overrides(labels: list[str], overrides: dict[int, str]) -> list[str]:
    """Manual corrections: {player_index: 'Team A' | 'Team B' | 'Unknown'}."""
    out = list(labels)
    for idx, name in overrides.items():
        if name not in (TEAM_A, TEAM_B, UNKNOWN):
            raise ValueError(f"Invalid team name: {name!r}")
        if not 0 <= idx < len(out):
            raise IndexError(f"Player index out of range: {idx}")
        out[idx] = name
    return out
