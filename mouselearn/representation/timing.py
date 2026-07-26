"""Monotonic time-to-spline-progress representation with fixed interval count."""
from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]


@dataclass(frozen=True)
class TimingRepresentation:
    interval_logits: tuple[float, ...]
    progress_knots: tuple[float, ...]


def _softplus(value: float) -> float:
    return value if value > 30 else math.log1p(math.exp(value))


def _inverse_softplus(value: float) -> float:
    return value if value > 30 else math.log(math.expm1(value))


def decode_timing_logits(logits: tuple[float, ...], epsilon: float = 1e-6) -> tuple[float, ...]:
    if not logits:
        raise ValueError("at least one timing interval is required")
    positives = [_softplus(value) + epsilon for value in logits]
    total = sum(positives)
    cumulative, result = 0.0, []
    for value in positives:
        cumulative += value / total
        result.append(cumulative)
    result[-1] = 1.0
    return tuple(result)


def _progress_at_time(time_fraction: float, times: list[float], progress: list[float]) -> float:
    for index in range(1, len(times)):
        if times[index] >= time_fraction:
            span = times[index] - times[index - 1]
            if span <= 1e-12:
                return progress[index]
            weight = (time_fraction - times[index - 1]) / span
            return progress[index - 1] + weight * (progress[index] - progress[index - 1])
    return 1.0


def fit_timing_representation(timestamps_ns: list[int], points: list[Point], interval_count: int = 12) -> TimingRepresentation:
    if len(timestamps_ns) != len(points) or len(points) < 2:
        raise ValueError("timing requires matching timestamped points")
    if interval_count < 1:
        raise ValueError("interval_count must be positive")
    if any(second <= first for first, second in zip(timestamps_ns, timestamps_ns[1:], strict=False)):
        raise ValueError("timestamps must be strictly increasing")
    start, duration = timestamps_ns[0], timestamps_ns[-1] - timestamps_ns[0]
    if duration <= 0:
        raise ValueError("timestamps must span a positive duration")
    times = [(value - start) / duration for value in timestamps_ns]
    distances = [0.0]
    for first, second in zip(points, points[1:], strict=False):
        distances.append(distances[-1] + math.dist(first, second))
    total_distance = distances[-1]
    progress = [index / (len(points) - 1) for index in range(len(points))] if total_distance <= 1e-12 else [value / total_distance for value in distances]
    knots = [_progress_at_time(index / interval_count, times, progress) for index in range(1, interval_count + 1)]
    knots[-1] = 1.0
    intervals = [knots[0], *[right - left for left, right in zip(knots, knots[1:], strict=False)]]
    logits = tuple(_inverse_softplus(max(1e-6, value) - 1e-6 + 1e-9) for value in intervals)
    return TimingRepresentation(logits, decode_timing_logits(logits))
