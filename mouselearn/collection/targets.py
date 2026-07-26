"""Target selection policies for the collection game."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, log2, radians, sin
import random


@dataclass(frozen=True)
class ScheduledTarget:
    x: int
    y: int
    radius: int
    distance_px: float
    angle_degrees: float
    screen_region: str
    difficulty_band: str
    requested_distance_px: float | None
    requested_radius_px: float | None
    requested_angle_degrees: float | None
    requested_screen_region: str | None
    requested_corner: str | None
    realized_corner: str | None
    sampling_strategy: str


@dataclass(frozen=True)
class _ConditionCell:
    distance: int
    radius: int
    angle: float
    region: str
    corner: str | None


class BalancedTargetScheduler:
    """Shuffled Cartesian condition cells, realized from the cursor at target onset."""

    _distances = (80, 180, 350, 600)
    _radii = (8, 14, 24, 40)
    _angles = tuple(index * 22.5 for index in range(16))
    _regions = ("center", "left", "right", "top", "bottom", "corner")
    _corners = ("top_left", "top_right", "bottom_left", "bottom_right")

    def __init__(self, seed: int):
        self._random = random.Random(seed)
        self._cells = [
            _ConditionCell(distance, radius, angle, region, corner)
            for distance in self._distances
            for radius in self._radii
            for angle in self._angles
            for region in self._regions
            for corner in (self._corners if region == "corner" else (None,))
        ]
        self._deck: list[_ConditionCell] = []

    def _replenish(self) -> None:
        if not self._deck:
            self._deck = list(self._cells)
            self._random.shuffle(self._deck)

    def next(self, width: int, height: int, cursor_x: float | None = None, cursor_y: float | None = None) -> ScheduledTarget:
        if width < 100 or height < 100:
            raise ValueError("collection canvas must be at least 100 by 100 pixels")
        cursor_x = width / 2 if cursor_x is None else cursor_x
        cursor_y = height / 2 if cursor_y is None else cursor_y
        # Once every feasible cell for this cursor has been consumed, reshuffle a fresh Cartesian cycle.
        for _cycle in range(2):
            self._replenish()
            rejected: list[_ConditionCell] = []
            while self._deck:
                cell = self._deck.pop()
                target = self._realize(cell, width, height, cursor_x, cursor_y)
                if target is not None:
                    self._deck.extend(rejected)
                    return target
                rejected.append(cell)
            self._deck = []
        raise ValueError("no scheduled target cell fits the cursor position and collection canvas")

    def _realize(self, cell: _ConditionCell, width: int, height: int, cursor_x: float, cursor_y: float) -> ScheduledTarget | None:
        margin = cell.radius + 20
        x = round(cursor_x + cell.distance * cos(radians(cell.angle)))
        y = round(cursor_y + cell.distance * sin(radians(cell.angle)))
        if not (margin <= x <= width - margin and margin <= y <= height - margin):
            return None
        region, corner = self._region_of(x, y, width, height)
        if region != cell.region or (region == "corner" and corner != cell.corner):
            return None
        distance = hypot(x - cursor_x, y - cursor_y)
        angle = degrees(atan2(y - cursor_y, x - cursor_x)) % 360
        difficulty = log2(distance / (2 * cell.radius) + 1) if distance else 0.0
        return ScheduledTarget(
            x=x, y=y, radius=cell.radius, distance_px=distance, angle_degrees=angle,
            screen_region=region, difficulty_band=self._difficulty_band(difficulty),
            requested_distance_px=cell.distance, requested_radius_px=cell.radius,
            requested_angle_degrees=cell.angle, requested_screen_region=cell.region,
            requested_corner=cell.corner, realized_corner=corner, sampling_strategy="balanced_cartesian_v2",
        )

    @staticmethod
    def _region_of(x: float, y: float, width: int, height: int) -> tuple[str, str | None]:
        horizontal = "left" if x < width / 3 else "right" if x > width * 2 / 3 else "center"
        vertical = "top" if y < height / 3 else "bottom" if y > height * 2 / 3 else "center"
        if horizontal == "center" and vertical == "center":
            return "center", None
        if vertical == "center":
            return horizontal, None
        if horizontal == "center":
            return vertical, None
        return "corner", f"{vertical}_{horizontal}"

    @staticmethod
    def _difficulty_band(index_of_difficulty: float) -> str:
        if index_of_difficulty < 2:
            return "low"
        if index_of_difficulty < 4:
            return "medium"
        return "high"


class ContinuousUniformTargetScheduler:
    """Protocol-3 target sampler: uniform feasible target centers and radii."""

    radius_min = 12
    radius_max = 36
    edge_margin = 12

    def __init__(self, seed: int):
        self._random = random.Random(seed)

    def next(self, width: int, height: int, cursor_x: float | None = None, cursor_y: float | None = None) -> ScheduledTarget:
        required = 2 * (self.radius_max + self.edge_margin) + 1
        if width < required or height < required:
            raise ValueError(f"collection canvas must be at least {required} by {required} pixels")
        # The renderer uses integer logical pixels; sampling is uniform over every feasible pixel center.
        radius = self._random.randint(self.radius_min, self.radius_max)
        minimum, maximum_x = radius + self.edge_margin, width - radius - self.edge_margin
        maximum_y = height - radius - self.edge_margin
        x = self._random.randint(minimum, maximum_x)
        y = self._random.randint(minimum, maximum_y)
        cursor_x = width / 2 if cursor_x is None else cursor_x
        cursor_y = height / 2 if cursor_y is None else cursor_y
        distance = hypot(x - cursor_x, y - cursor_y)
        angle = degrees(atan2(y - cursor_y, x - cursor_x)) % 360 if distance else 0.0
        region, corner = BalancedTargetScheduler._region_of(x, y, width, height)
        difficulty = log2(distance / (2 * radius) + 1) if distance else 0.0
        return ScheduledTarget(
            x=x, y=y, radius=radius, distance_px=distance, angle_degrees=angle,
            screen_region=region, difficulty_band=BalancedTargetScheduler._difficulty_band(difficulty),
            requested_distance_px=None, requested_radius_px=radius, requested_angle_degrees=None,
            requested_screen_region=None, requested_corner=None, realized_corner=corner,
            sampling_strategy="continuous_uniform_feasible_v3",
        )
