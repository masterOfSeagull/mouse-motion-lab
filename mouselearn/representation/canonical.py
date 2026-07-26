"""Shared exact forward/inverse geometry and valid-by-construction endpoint mapping."""
from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]


@dataclass(frozen=True)
class CanonicalTransform:
    start: Point
    target_center: Point
    target_radius: float
    distance: float
    angle_radians: float

    @classmethod
    def from_start_target(cls, start: Point, target_center: Point, target_radius: float) -> "CanonicalTransform":
        dx, dy = target_center[0] - start[0], target_center[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= 0:
            raise ValueError("movement start and target center must be distinct")
        if target_radius <= 0:
            raise ValueError("target radius must be positive")
        return cls(start, target_center, target_radius, distance, math.atan2(dy, dx))

    def forward(self, point: Point) -> Point:
        dx, dy = point[0] - self.start[0], point[1] - self.start[1]
        cosine, sine = math.cos(self.angle_radians), math.sin(self.angle_radians)
        return ((cosine * dx + sine * dy) / self.distance, (-sine * dx + cosine * dy) / self.distance)

    def inverse(self, point: Point) -> Point:
        x, y = point[0] * self.distance, point[1] * self.distance
        cosine, sine = math.cos(self.angle_radians), math.sin(self.angle_radians)
        return (self.start[0] + cosine * x - sine * y, self.start[1] + sine * x + cosine * y)

    @property
    def canonical_target_radius(self) -> float:
        return self.target_radius / self.distance


def decode_endpoint(latent: Point, epsilon: float = 1e-9) -> Point:
    """Map an unconstrained latent vector to a point strictly inside the unit disk."""
    norm = math.hypot(*latent)
    if norm <= epsilon:
        return (0.0, 0.0)
    radius = math.tanh(norm)
    return (radius * latent[0] / norm, radius * latent[1] / norm)


def encode_endpoint(point: Point, epsilon: float = 1e-9) -> Point:
    """Invert the endpoint transform after clipping numerically invalid source points."""
    norm = math.hypot(*point)
    if norm <= epsilon:
        return (0.0, 0.0)
    clipped = min(norm, 1.0 - epsilon)
    magnitude = math.atanh(clipped)
    return (magnitude * point[0] / norm, magnitude * point[1] / norm)
