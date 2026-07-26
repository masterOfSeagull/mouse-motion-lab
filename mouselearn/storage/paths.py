from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_path


DATA_ROOT_ENV = "MOUSE_MOTION_LAB_DATA_ROOT"


def data_root() -> Path:
    override = os.environ.get(DATA_ROOT_ENV)
    return Path(override).expanduser().resolve() if override else user_data_path("MouseMotionLab", appauthor=False)


def initialize_data_root(root: Path | None = None) -> Path:
    root = root or data_root()
    for name in ("logs", "raw_sessions", "datasets", "experiments", "models", "exports", "cache", "temp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def database_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "app.db"
