"""Strict, versioned configuration. Future sections are inert in milestone 1."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CollectorConfig(StrictModel):
    sampling_hz: int = Field(default=125, ge=1, le=1000)
    include_mouse_moves: bool = True
    include_clicks: bool = True


class PreprocessingConfig(StrictModel):
    resample_hz: int = Field(default=125, ge=1, le=1000)
    normalize_coordinates: bool = True


class TrainingConfig(StrictModel):
    enabled: bool = False
    random_seed: int = 42
    backend: Literal["reserved"] = "reserved"


class GenerationConfig(StrictModel):
    enabled: bool = False
    trajectory_points: int = Field(default=128, ge=2)


class PlaybackConfig(StrictModel):
    enabled: bool = False
    emergency_stop_key: str = "F8"


class AppConfig(StrictModel):
    schema_version: Literal[1] = 1
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    playback: PlaybackConfig = Field(default_factory=PlaybackConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load a YAML config; an omitted path produces the fully resolved defaults."""
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return AppConfig.model_validate(raw)


__all__ = ["AppConfig", "ValidationError", "load_config"]
