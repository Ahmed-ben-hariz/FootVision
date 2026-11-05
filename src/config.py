"""Load the single project configuration file."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
