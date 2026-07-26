"""Clamped cubic B-spline fitting with fixed endpoints and smoothness regularization."""
from __future__ import annotations

import math
from dataclasses import dataclass


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
    """Cox-de Boor basis for an open clamped knot vector."""
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
            left = 0.0
            left_denom = knots[index + degree] - knots[index]
            if left_denom:
                left = (parameter - knots[index]) * values[index] / left_denom
            right = 0.0
            if index + 1 < count:
                right_denom = knots[index + degree + 1] - knots[index + 1]
                if right_denom:
                    right = (knots[index + degree + 1] - parameter) * values[index + 1] / right_denom
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


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            matrix[column][column] += 1e-8
            pivot = column
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            vector[column], vector[pivot] = vector[pivot], vector[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        vector[column] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            if factor:
                matrix[row] = [value - factor * pivot_value for value, pivot_value in zip(matrix[row], matrix[column], strict=True)]
                vector[row] -= factor * vector[column]
    return vector


def _fit_coordinate(points: list[Point], parameters: tuple[float, ...], spec: SplineSpec, coordinate: int) -> list[float]:
    count, free_count = spec.control_point_count, spec.control_point_count - 2
    first, last = points[0][coordinate], points[-1][coordinate]
    matrix = [[0.0] * free_count for _ in range(free_count)]
    vector = [0.0] * free_count
    for point, parameter in zip(points, parameters, strict=True):
        basis = bspline_basis(parameter, spec)
        adjusted = point[coordinate] - basis[0] * first - basis[-1] * last
        for row in range(free_count):
            coefficient = basis[row + 1]
            vector[row] += coefficient * adjusted
            for column in range(free_count):
                matrix[row][column] += coefficient * basis[column + 1]
    if spec.smoothing:
        for index in range(1, count - 1):
            terms = ((index - 1, 1.0), (index, -2.0), (index + 1, 1.0))
            fixed = sum(value * (first if control == 0 else last) for control, value in terms if control in {0, count - 1})
            free_terms = [(control - 1, value) for control, value in terms if 0 < control < count - 1]
            for row, left in free_terms:
                vector[row] -= spec.smoothing * left * fixed
                for column, right in free_terms:
                    matrix[row][column] += spec.smoothing * left * right
    solved = _solve(matrix, vector)
    return [first, *solved, last]


def fit_clamped_spline(points: list[Point], spec: SplineSpec = SplineSpec()) -> SplineFit:
    if len(points) < 2:
        raise ValueError("at least two points are required for spline fitting")
    parameters = chord_parameters(points)
    x_values = _fit_coordinate(points, parameters, spec, 0)
    y_values = _fit_coordinate(points, parameters, spec, 1)
    return SplineFit(spec, tuple(zip(x_values, y_values, strict=True)), parameters)
