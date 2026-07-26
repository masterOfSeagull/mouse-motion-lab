from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("MOUSE_MOTION_LAB_DATA_ROOT", str(root))
    return root
