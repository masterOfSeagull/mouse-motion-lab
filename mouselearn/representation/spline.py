"""Numerically stable clamped cubic B-spline fitting with explicit diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class SplineSpec:
    degree: int = 3
    control_point_count: int = 16
    smoothing: float = 1e-4

    def __post_init__(self) -> None:
        if self.degree != 3:
            raise ValueError("Milestone 3 uses clamped cubic splines")
        if self.control_point_count <= self.degree:
            raise ValueError("control point count must exceed spline degree")
        if self.smoothing < 0:
            raise ValueError("smoothing must not be negative")


@dataclass(frozen=True)
class SplineFit:
    spec: SplineSpec
    control_points: tuple[Point, ...]
    parameters: tuple[float, ...]
    rank: int
    condition_number: float

    def evaluate(self, parameter: float) -> Point:
        basis = bspline_basis(parameter, self.spec)
        return (
            sum(weight * point[0] for weight, point in zip(basis, self.control_points, strict=True)),
            sum(weight * point[1] for weight, point in zip(basis, self.control_points, strict=True)),
        )

    def residual_controls(self) -> tuple[Point, ...]:
        start, endpoint = self.control_points[0], self.control_points[-1]
        count = len(self.control_points) - 1
        return tuple(
            (point[0] - (start[0] + (endpoint[0] - start[0]) * index / count),
             point[1] - (start[1] + (endpoint[1] - start[1]) * index / count))
            for index, point in enumerate(self.control_points[1:-1], start=1)
        )


def _open_uniform_knots(spec: SplineSpec) -> tuple[float, ...]:
    internal_count = spec.control_point_count - spec.degree - 1
    internal = tuple(index / (internal_count + 1) for index in range(1, internal_count + 1))
    return (0.0,) * (spec.degree + 1) + internal + (1.0,) * (spec.degree + 1)


def bspline_basis(parameter: float, spec: SplineSpec) -> tuple[float, ...]:
    parameter = min(1.0, max(0.0, parameter))
    knots = _open_uniform_knots(spec)
    count = spec.control_point_count
    values = [0.0] * count
    if parameter == 1.0:
        values[-1] = 1.0
        return tuple(values)
    for index in range(count):
        values[index] = 1.0 if knots[index] <= parameter < knots[index + 1] else 0.0
    for degree in range(1, spec.degree + 1):
        next_values = [0.0] * count
        for index in range(count):
            left_denom = knots[index + degree] - knots[index]
            left = (parameter - knots[index]) * values[index] / left_denom if left_denom else 0.0
            right = 0.0
            if index + 1 < count:
                right_denom = knots[index + degree + 1] - knots[index + 1]
                right = (knots[index + degree + 1] - parameter) * values[index + 1] / right_denom if right_denom else 0.0
            next_values[index] = left + right
        values = next_values
    return tuple(values)


def chord_parameters(points: list[Point]) -> tuple[float, ...]:
    distances = [0.0]
    for first, second in zip(points, points[1:], strict=False):
        distances.append(distances[-1] + math.dist(first, second))
    total = distances[-1]
    if total <= 1e-12:
        return tuple(index / (len(points) - 1) for index in range(len(points)))
    return tuple(distance / total for distance in distances)


def _regularization_rows(spec: SplineSpec, first: float, last: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-difference rows, adjusted for fixed start/end controls."""
    free_count, control_count = spec.control_point_count - 2, spec.control_point_count
    matrix = np.zeros((control_count - 2, free_count), dtype=np.float64)
    target = np.zeros(control_count - 2, dtype=np.float64)
    for row, center in enumerate(range(1, control_count - 1)):
        fixed = 0.0
        for control, value in ((center - 1, 1.0), (center, -2.0), (center + 1, 1.0)):
            if control == 0:
                fixed += value * first
            elif control == control_count - 1:
                fixed += value * last
            else:
                matrix[row, control - 1] = value
        target[row] = -fixed
    return matrix, target


def _fit_coordinate(points: list[Point], parameters: tuple[float, ...], spec: SplineSpec, coordinate: int) -> tuple[np.ndarray, int, float]:
    control_count = spec.control_point_count
    first, last = points[0][coordinate], points[-1][coordinate]
    basis = np.asarray([bspline_basis(parameter, spec) for parameter in parameters], dtype=np.float64)
    design = basis[:, 1:-1]
    values = np.asarray([point[coordinate] for point in points], dtype=np.float64)
    target = values - basis[:, 0] * first - basis[:, -1] * last
    if spec.smoothing:
        smooth_design, smooth_target = _regularization_rows(spec, first, last)
        scale = math.sqrt(spec.smoothing)
        design = np.vstack((design, smooth_design * scale))
        target = np.concatenate((target, smooth_target * scale))
    solution, _residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    condition = float("inf") if not len(singular_values) or singular_values[-1] <= np.finfo(float).eps else float(singular_values[0] / singular_values[-1])
    return np.concatenate(([first], solution, [last])), int(rank), condition


def fit_clamped_spline(points: list[Point], spec: SplineSpec = SplineSpec()) -> SplineFit:
    if len(points) < 2:
        raise ValueError("at least two points are required for spline fitting")
    parameters = chord_parameters(points)
    x_values, x_rank, x_condition = _fit_coordinate(points, parameters, spec, 0)
    y_values, y_rank, y_condition = _fit_coordinate(points, parameters, spec, 1)
    control_points = tuple((float(x), float(y)) for x, y in zip(x_values, y_values, strict=True))
    return SplineFit(spec, control_points, parameters, min(x_rank, y_rank), max(x_condition, y_condition))
