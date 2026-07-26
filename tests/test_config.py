from __future__ import annotations

import pytest
from pydantic import ValidationError

from mouselearn.domain.config import AppConfig, load_config


def test_defaults_are_resolved() -> None:
    config = load_config()
    assert config.schema_version == 1
    assert config.collector.sampling_hz == 125
    assert not config.training.enabled


def test_unknown_config_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"schema_version": 1, "collector": {"unexpected": True}})
