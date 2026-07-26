"""Focused Milestone-4 correctness and baseline-distribution validation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import MovementGenerator, ProcessedDataset, decode_output


def _quantile_wasserstein(first: np.ndarray, second: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, max(len(first), len(second), 2))
    return float(np.mean(np.abs(np.quantile(first, quantiles) - np.quantile(second, quantiles))))


def _trajectory_features(result, request) -> dict[str, float]:
    dx, dy = request.target_center_x - request.start_x, request.target_center_y - request.start_y
    distance = max(math.hypot(dx, dy), 1e-9)
    ux, uy = dx / distance, dy / distance
    progress, lateral = [], []
    for sample in result.samples:
        px, py = sample.x - request.start_x, sample.y - request.start_y
        progress.append((px * ux + py * uy) / distance)
        lateral.append((-px * uy + py * ux) / distance)
    reversals = sum((right - left) < -1e-4 for left, right in zip(progress, progress[1:], strict=False))
    return {
        "path_length_ratio": result.path_length / distance,
        "peak_speed": result.peak_speed,
        "maximum_perpendicular_deviation": max((abs(value) for value in lateral), default=0.0),
        "signed_average_lateral_deviation": float(np.mean(lateral)),
        "endpoint_radius_ratio": math.hypot(result.endpoint_x - request.target_center_x, result.endpoint_y - request.target_center_y) / request.target_radius,
        "overshoot": float(max(progress, default=0.0) > 1.0 + request.target_radius / distance),
        "direction_reversals": float(reversals),
    }


def validate_baseline(generator: MovementGenerator, dataset: ProcessedDataset) -> dict[str, Any]:
    """Validate unseen held-out requests, including deterministic seeded generation."""
    held_out_indices = [index for index, split in enumerate(dataset.splits) if split in {"validation", "test"}]
    if not held_out_indices:
        raise ValueError("baseline validation requires validation or test samples")
    conditions = dataset.conditions[held_out_indices]
    seeds = np.arange(10_000, 10_000 + len(held_out_indices), dtype=np.uint64)
    generated = generator.generate_batch(conditions, seeds)
    repeated = generator.generate_batch(conditions[:1], seeds[:1])
    deterministic = bool(np.array_equal(generated.outputs[:1], repeated.outputs))
    correctness_failures: list[dict[str, Any]] = []
    projected = clipped = 0
    generated_durations, real_durations = [], []
    generated_features: dict[str, list[float]] = {}
    real_features: dict[str, list[float]] = {}
    for output_index, dataset_index in enumerate(held_out_indices):
        original = dataset.requests[dataset_index]
        request = original.__class__(**{**original.__dict__, "random_seed": int(seeds[output_index])})
        try:
            result = decode_output(
                generated.outputs[output_index], request,
                condition_distance=float(generated.nearest_distances[output_index]),
                out_of_distribution=bool(generated.out_of_distribution[output_index]),
            )
            endpoint_distance = math.hypot(result.endpoint_x - request.target_center_x, result.endpoint_y - request.target_center_y)
            valid = (
                len(result.samples) == 64 and result.samples[0].x == request.start_x and result.samples[0].y == request.start_y
                and endpoint_distance <= request.target_radius + 1e-6 and result.movement_duration_ns > 0
                and all(right.relative_time_ns > left.relative_time_ns for left, right in zip(result.samples, result.samples[1:], strict=False))
                and all(math.isfinite(value) for sample in result.samples for value in (sample.x, sample.y))
            )
            if not valid:
                correctness_failures.append({"trial_id": dataset.trial_ids[dataset_index], "reason": "hard correctness check failed"})
            projected += int(result.endpoint_projected)
            clipped += result.desktop_clipped_point_count
            generated_durations.append(result.movement_duration_ns)
            real_result = decode_output(dataset.outputs[dataset_index], request)
            real_durations.append(real_result.movement_duration_ns)
            for name, value in _trajectory_features(result, request).items():
                generated_features.setdefault(name, []).append(value)
            for name, value in _trajectory_features(real_result, request).items():
                real_features.setdefault(name, []).append(value)
        except Exception as exc:
            correctness_failures.append({"trial_id": dataset.trial_ids[dataset_index], "reason": str(exc)})
    total = len(held_out_indices)
    report = {
        "schema_version": 1, "snapshot_id": dataset.snapshot_id, "preprocessing_run_id": dataset.preprocessing_run_id,
        "model_type": generator.model_type, "held_out_sample_count": total,
        "hard_correctness": {"passed": not correctness_failures, "failure_count": len(correctness_failures), "failures": correctness_failures[:25]},
        "deterministic_same_seed": deterministic,
        "endpoint_projection_count": projected, "endpoint_projection_rate": projected / total,
        "desktop_clipped_point_count": clipped,
        "out_of_distribution_count": int(np.count_nonzero(generated.out_of_distribution)),
        "movement_duration": {
            "generated_mean_ns": float(np.mean(generated_durations)), "real_mean_ns": float(np.mean(real_durations)),
            "wasserstein_ns": _quantile_wasserstein(np.asarray(generated_durations), np.asarray(real_durations)),
        },
        "distribution_metrics": {},
        "condition_distance_percentiles": {
            "in_distribution_max": float(np.quantile(generated.nearest_distances, 0.90)),
            "sparse_max": float(np.quantile(generated.nearest_distances, 0.99)),
        },
    }
    for name in sorted(generated_features):
        generated_values, real_values = np.asarray(generated_features[name]), np.asarray(real_features[name])
        report["distribution_metrics"][name] = {
            "generated_mean": float(np.mean(generated_values)), "real_mean": float(np.mean(real_values)),
            "wasserstein": _quantile_wasserstein(generated_values, real_values),
        }
    report["path_length_ratio"] = report["distribution_metrics"]["path_length_ratio"]
    report["passed"] = bool(report["hard_correctness"]["passed"] and deterministic)
    return report


__all__ = ["validate_baseline"]
