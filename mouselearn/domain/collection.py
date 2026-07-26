"""Typed persistence contracts for the Milestone 2 collection pipeline."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from .config import StrictModel


CollectionMode = Literal["standard", "balanced_coverage", "repeated_condition", "manual", "validation_only"]
SessionState = Literal["planned", "active", "paused", "completed", "failed", "cancelled"]
TrialState = Literal["target_visible", "tracking", "completed", "failed", "cancelled"]
TrialEndReason = Literal["valid_click", "timeout", "cancelled", "window_focus_lost", "capture_failure", "user_paused"]
CaptureSeverity = Literal["info", "warning", "error"]


class TargetCondition(StrictModel):
    """Realized geometry is authoritative; requested scheduler values retain provenance."""
    """The reproducible condition selected for one target trial."""

    distance_px: float = Field(ge=0)
    radius_px: float = Field(gt=0)
    angle_degrees: float = Field(ge=0, lt=360)
    screen_region: Literal["center", "left", "right", "top", "bottom", "corner"]
    difficulty_band: str = Field(min_length=1, max_length=80)
    target_x: int
    target_y: int
    monitor_id: str = Field(min_length=1, max_length=256)
    requested_distance_px: float | None = Field(default=None, ge=0)
    requested_radius_px: float | None = Field(default=None, gt=0)
    requested_angle_degrees: float | None = Field(default=None, ge=0, lt=360)
    requested_screen_region: Literal["center", "left", "right", "top", "bottom", "corner"] | None = None
    requested_corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] | None = None
    realized_corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] | None = None
    collection_protocol_version: int = Field(default=1, ge=1)
    target_sampling_strategy: str = Field(default="legacy_unknown", min_length=1, max_length=80)
    reaction_time_confidence: Literal["high", "legacy_render_unconfirmed"] = "legacy_render_unconfirmed"


class CollectionSessionPlan(StrictModel):
    display_name: str = Field(min_length=1, max_length=160)
    mode: CollectionMode = "standard"
    planned_trials: int = Field(ge=1, le=100_000)
    random_seed: int = Field(ge=0)
    config: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)


class TrialPlan(StrictModel):
    session_id: UUID
    condition: TargetCondition
    target_appeared_ns: int = Field(ge=0)
    start_screen_x: int
    start_screen_y: int


class ClickRecord(StrictModel):
    timestamp_ns: int = Field(ge=0)
    screen_x: int
    screen_y: int
    button: Literal["left", "right", "middle", "x1", "x2"] = "left"
    is_valid: bool


class TrialFinalization(StrictModel):
    state: Literal["completed", "failed", "cancelled"]
    end_reason: TrialEndReason
    ended_ns: int = Field(ge=0)
    clicks: tuple[ClickRecord, ...] = ()

    @model_validator(mode="after")
    def completed_trials_need_a_valid_click(self) -> "TrialFinalization":
        if self.state == "completed" and self.end_reason == "valid_click" and not any(click.is_valid for click in self.clicks):
            raise ValueError("a valid_click completion requires a valid click record")
        return self


class RawEventFileReference(StrictModel):
    """Metadata for a bounded Parquet batch, never individual raw events."""

    relative_path: str = Field(min_length=1, max_length=1024)
    event_count: int = Field(ge=0)
    first_timestamp_ns: int | None = Field(default=None, ge=0)
    last_timestamp_ns: int | None = Field(default=None, ge=0)
    qpc_frequency_hz: int = Field(gt=0)
    byte_count: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def require_a_safe_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or normalized.startswith("/"):
            raise ValueError("raw event file path must be relative to raw_sessions")
        return path.as_posix()

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> "RawEventFileReference":
        if self.first_timestamp_ns is not None and self.last_timestamp_ns is not None and self.last_timestamp_ns < self.first_timestamp_ns:
            raise ValueError("last_timestamp_ns must not precede first_timestamp_ns")
        return self


class CaptureHealthRecord(StrictModel):
    severity: CaptureSeverity
    code: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    occurred_at_ns: int = Field(ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)


class CollectionPhaseMarker(StrictModel):
    phase: Literal[
        "inter_trial", "target_visible", "trial_completed", "trial_cancelled", "trial_failed", "trial_timed_out",
        "session_completed", "session_cancelled", "session_failed",
    ]
    timestamp_ns: int = Field(ge=0)
    screen_x: int | None = None
    screen_y: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coordinates_are_paired(self) -> "CollectionPhaseMarker":
        if (self.screen_x is None) != (self.screen_y is None):
            raise ValueError("phase marker screen coordinates must be present together")
        return self


__all__ = [
    "CaptureHealthRecord",
    "CollectionPhaseMarker",
    "ClickRecord",
    "CollectionSessionPlan",
    "RawEventFileReference",
    "TargetCondition",
    "TrialFinalization",
    "TrialPlan",
]
