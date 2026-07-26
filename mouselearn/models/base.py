"""Shared contracts and decoding for every movement generator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


POSITION_COUNT = 64
OUTPUT_SIZE = POSITION_COUNT * 2 + 1
CONDITION_FEATURE_NAMES = (
    "log_distance", "log_radius", "log_effective_difficulty", "radius_distance_ratio",
    "sin_direction", "cos_direction", "start_x", "start_y", "target_x", "target_y",
    "start_left", "start_right", "start_top", "start_bottom",
    "target_left", "target_right", "target_top", "target_bottom",
    "previous_velocity_x", "previous_velocity_y", "dpi_scale",
)


@dataclass(frozen=True)
class GenerationRequest:
    start_x: float
    start_y: float
    target_center_x: float
    target_center_y: float
    target_radius: float
    virtual_desktop_left: float
    virtual_desktop_top: float
    virtual_desktop_width: float
    virtual_desktop_height: float
    previous_velocity_x: float = 0.0
    previous_velocity_y: float = 0.0
    dpi_scale: float = 1.0
    click_requested: bool = False
    random_seed: int = 0
    output_rate_hz: int = 250
    solver_steps: int = 16

    def __post_init__(self) -> None:
        values = (
            self.start_x, self.start_y, self.target_center_x, self.target_center_y, self.target_radius,
            self.virtual_desktop_left, self.virtual_desktop_top, self.virtual_desktop_width,
            self.virtual_desktop_height, self.previous_velocity_x, self.previous_velocity_y, self.dpi_scale,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("generation request values must be finite")
        if self.target_radius <= 0 or self.virtual_desktop_width <= 0 or self.virtual_desktop_height <= 0:
            raise ValueError("target radius and virtual desktop dimensions must be positive")
        if self.dpi_scale <= 0 or self.output_rate_hz <= 0 or self.solver_steps <= 0 or self.random_seed < 0:
            raise ValueError("DPI scale, output rate, solver steps, and seed must be valid")


@dataclass(frozen=True)
class TrajectorySample:
    relative_time_ns: int
    x: float
    y: float


@dataclass(frozen=True)
class GeneratedParameterBatch:
    outputs: np.ndarray
    nearest_distances: np.ndarray
    out_of_distribution: np.ndarray
    source_indices: np.ndarray | None = None


@dataclass(frozen=True)
class GenerationResult:
    samples: tuple[TrajectorySample, ...]
    movement_duration_ns: int
    endpoint_x: float
    endpoint_y: float
    click_requested: bool
    out_of_distribution: bool
    condition_distance_score: float
    seed: int
    endpoint_projected: bool
    desktop_clipped_point_count: int

    @property
    def path_length(self) -> float:
        return sum(
            math.hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(self.samples, self.samples[1:], strict=False)
        )

    @property
    def peak_speed(self) -> float:
        speeds = []
        for left, right in zip(self.samples, self.samples[1:], strict=False):
            duration_seconds = (right.relative_time_ns - left.relative_time_ns) / 1_000_000_000
            if duration_seconds > 0:
                speeds.append(math.hypot(right.x - left.x, right.y - left.y) / duration_seconds)
        return max(speeds, default=0.0)


@dataclass(frozen=True)
class ProcessedDataset:
    conditions: np.ndarray
    outputs: np.ndarray
    requests: tuple[GenerationRequest, ...]
    splits: tuple[str, ...]
    trial_ids: tuple[str, ...]
    snapshot_id: str = "synthetic"
    preprocessing_run_id: str = "synthetic"

    def __post_init__(self) -> None:
        conditions = np.asarray(self.conditions, dtype=np.float64)
        outputs = np.asarray(self.outputs, dtype=np.float64)
        count = len(conditions)
        if conditions.ndim != 2 or conditions.shape[1] != len(CONDITION_FEATURE_NAMES):
            raise ValueError("processed conditions have the wrong shape")
        if outputs.shape != (count, OUTPUT_SIZE) or not all(len(items) == count for items in (self.requests, self.splits, self.trial_ids)):
            raise ValueError("processed dataset arrays and metadata do not align")
        if count == 0 or not np.isfinite(conditions).all() or not np.isfinite(outputs).all():
            raise ValueError("processed dataset must contain finite samples")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "outputs", outputs)

    def subset(self, split: str) -> "ProcessedDataset":
        indices = [index for index, value in enumerate(self.splits) if value == split]
        if not indices:
            raise ValueError(f"dataset has no {split} samples")
        return ProcessedDataset(
            self.conditions[indices], self.outputs[indices], tuple(self.requests[index] for index in indices),
            tuple(self.splits[index] for index in indices), tuple(self.trial_ids[index] for index in indices),
            self.snapshot_id, self.preprocessing_run_id,
        )


@runtime_checkable
class MovementGenerator(Protocol):
    model_type: str

    def fit(self, dataset: ProcessedDataset) -> "MovementGenerator": ...
    def generate_batch(self, conditions: np.ndarray, seeds: np.ndarray) -> GeneratedParameterBatch: ...
    def save(self, destination: Path) -> None: ...


def condition_vector(request: GenerationRequest) -> np.ndarray:
    dx = request.target_center_x - request.start_x
    dy = request.target_center_y - request.start_y
    distance = math.hypot(dx, dy)
    safe_distance = max(distance, 1e-9)
    angle = math.atan2(dy, dx) if distance else 0.0
    difficulty = math.log2(distance / (2 * request.target_radius) + 1.0)
    left, top = request.virtual_desktop_left, request.virtual_desktop_top
    width, height = request.virtual_desktop_width, request.virtual_desktop_height
    right, bottom = left + width, top + height
    return np.asarray((
        math.log(safe_distance), math.log(request.target_radius), math.log1p(difficulty),
        request.target_radius / safe_distance, math.sin(angle), math.cos(angle),
        (request.start_x - left) / width, (request.start_y - top) / height,
        (request.target_center_x - left) / width, (request.target_center_y - top) / height,
        (request.start_x - left) / width, (right - request.start_x) / width,
        (request.start_y - top) / height, (bottom - request.start_y) / height,
        (request.target_center_x - left) / width, (right - request.target_center_x) / width,
        (request.target_center_y - top) / height, (bottom - request.target_center_y) / height,
        request.previous_velocity_x / width, request.previous_velocity_y / height, request.dpi_scale,
    ), dtype=np.float64)


def constrain_parameter_output(output: np.ndarray, radius_distance_ratio: float, safety_margin: float = 0.99) -> np.ndarray:
    """Keep the exact start and endpoint valid before the runtime safety decoder."""
    values = np.asarray(output, dtype=np.float64).copy()
    if values.shape != (OUTPUT_SIZE,) or not np.isfinite(values).all() or radius_distance_ratio <= 0:
        raise ValueError("parameter output and target-radius ratio must be valid")
    values[:2] = 0.0
    endpoint = values[-3:-1]
    offset = endpoint - np.asarray([1.0, 0.0])
    norm = float(np.linalg.norm(offset))
    allowed = radius_distance_ratio * safety_margin
    if norm > allowed:
        values[-3:-1] = np.asarray([1.0, 0.0]) + offset * (allowed / max(norm, 1e-12))
    return values


def seeded_normal_source(seed: int, count: int) -> np.ndarray:
    """Portable SplitMix64/Box-Muller source shared with the C++ runtime."""
    mask = (1 << 64) - 1
    state = seed & mask
    def uniform() -> float:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return ((value >> 11) + 0.5) / 9007199254740992.0
    result: list[float] = []
    while len(result) < count:
        radius = math.sqrt(-2.0 * math.log(uniform()))
        angle = 2.0 * math.pi * uniform()
        result.append(radius * math.cos(angle))
        if len(result) < count:
            result.append(radius * math.sin(angle))
    return np.asarray(result, dtype=np.float32)


def decode_output(
    output: np.ndarray, request: GenerationRequest, *, condition_distance: float = 0.0,
    out_of_distribution: bool = False, endpoint_safety_margin: float = 0.995,
) -> GenerationResult:
    """Decode one 129-value model vector into a valid, equal-time screen trajectory."""
    values = np.asarray(output, dtype=np.float64)
    if values.shape != (OUTPUT_SIZE,) or not np.isfinite(values).all():
        raise ValueError("generated output must contain 129 finite values")
    canonical = values[:-1].reshape(POSITION_COUNT, 2).copy()
    canonical[0] = 0.0
    duration = int(round(math.exp(float(np.clip(values[-1], math.log(1_000_000), math.log(60_000_000_000))))))
    dx = request.target_center_x - request.start_x
    dy = request.target_center_y - request.start_y
    distance = math.hypot(dx, dy)
    endpoint_projected = False
    if distance <= 1e-9:
        screen = np.repeat(np.asarray([[request.start_x, request.start_y]]), POSITION_COUNT, axis=0)
    else:
        radius_ratio = request.target_radius / distance
        endpoint_offset = canonical[-1] - np.asarray([1.0, 0.0])
        endpoint_norm = float(np.linalg.norm(endpoint_offset))
        allowed = radius_ratio * endpoint_safety_margin
        if endpoint_norm > allowed:
            canonical[-1] = np.asarray([1.0, 0.0]) + endpoint_offset * (allowed / max(endpoint_norm, 1e-12))
            endpoint_projected = True
        angle = math.atan2(dy, dx)
        cosine, sine = math.cos(angle), math.sin(angle)
        x = canonical[:, 0] * distance
        y = canonical[:, 1] * distance
        screen = np.column_stack((
            request.start_x + cosine * x - sine * y,
            request.start_y + sine * x + cosine * y,
        ))
    screen[0] = (request.start_x, request.start_y)
    left, top = request.virtual_desktop_left, request.virtual_desktop_top
    right = left + request.virtual_desktop_width
    bottom = top + request.virtual_desktop_height
    clipped = screen.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], left, right)
    clipped[:, 1] = np.clip(clipped[:, 1], top, bottom)
    clipped_count = int(np.count_nonzero(np.any(clipped != screen, axis=1)))
    timestamps = [round(duration * index / (POSITION_COUNT - 1)) for index in range(POSITION_COUNT)]
    timestamps[0], timestamps[-1] = 0, duration
    samples = tuple(TrajectorySample(timestamp, float(point[0]), float(point[1])) for timestamp, point in zip(timestamps, clipped, strict=True))
    return GenerationResult(
        samples, duration, samples[-1].x, samples[-1].y, request.click_requested, out_of_distribution,
        float(condition_distance), request.random_seed, endpoint_projected, clipped_count,
    )


__all__ = [
    "CONDITION_FEATURE_NAMES", "OUTPUT_SIZE", "POSITION_COUNT", "GeneratedParameterBatch", "GenerationRequest",
    "GenerationResult", "MovementGenerator", "ProcessedDataset", "TrajectorySample", "condition_vector",
    "constrain_parameter_output", "decode_output", "seeded_normal_source",
]
