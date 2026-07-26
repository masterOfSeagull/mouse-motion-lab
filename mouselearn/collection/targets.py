"""Reproducible, balanced target selection for the collection game."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, log2
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


class BalancedTargetScheduler:
    """Cycles every condition axis evenly while keeping targets within the canvas."""

    _distances = (80, 180, 350, 600)
    _radii = (8, 14, 24, 40)
    _regions = ("center", "left", "right", "top", "bottom", "corner")

    def __init__(self, seed: int):
        self._random = random.Random(seed)
        self._index = 0
        self._distance_offset = self._random.randrange(len(self._distances))
        self._radius_offset = self._random.randrange(len(self._radii))
        self._angle_offset = self._random.randrange(16)
        self._region_offset = self._random.randrange(len(self._regions))

    def next(self, width: int, height: int) -> ScheduledTarget:
        if width < 100 or height < 100:
            raise ValueError("collection canvas must be at least 100 by 100 pixels")
        index = self._index
        self._index += 1
        radius = self._radii[(index + self._radius_offset) % len(self._radii)]
        margin = radius + 20
        min_x, max_x = margin, max(margin, width - margin)
        min_y, max_y = margin, max(margin, height - margin)
        requested_distance = self._distances[(index + self._distance_offset) % len(self._distances)]
        angle = ((index + self._angle_offset) % 16) * 22.5
        region = self._regions[(index + self._region_offset) % len(self._regions)]
        center_x, center_y = width / 2, height / 2
        region_x, region_y = self._region_anchor(region, min_x, max_x, min_y, max_y, center_x, center_y)
        # Keep the prescribed direction/distance as far as the selected region and canvas permit.
        from math import cos, radians, sin
        x = round(min(max(region_x + requested_distance * cos(radians(angle)), min_x), max_x))
        y = round(min(max(region_y + requested_distance * sin(radians(angle)), min_y), max_y))
        distance = hypot(x - center_x, y - center_y)
        actual_angle = degrees(atan2(y - center_y, x - center_x)) % 360
        index_of_difficulty = log2(distance / (2 * radius) + 1) if distance else 0.0
        return ScheduledTarget(
            x=x, y=y, radius=radius, distance_px=distance, angle_degrees=actual_angle,
            screen_region=region, difficulty_band=self._difficulty_band(index_of_difficulty),
        )

    @staticmethod
    def _region_anchor(region: str, min_x: int, max_x: int, min_y: int, max_y: int, center_x: float, center_y: float) -> tuple[float, float]:
        quarter_x, quarter_y = (min_x + max_x) / 4, (min_y + max_y) / 4
        if region == "left": return min_x + quarter_x / 2, center_y
        if region == "right": return max_x - quarter_x / 2, center_y
        if region == "top": return center_x, min_y + quarter_y / 2
        if region == "bottom": return center_x, max_y - quarter_y / 2
        if region == "corner":
            return (min_x + quarter_x / 2, min_y + quarter_y / 2) if (int(center_x + center_y) % 2 == 0) else (max_x - quarter_x / 2, max_y - quarter_y / 2)
        return center_x, center_y

    @staticmethod
    def _difficulty_band(index_of_difficulty: float) -> str:
        if index_of_difficulty < 2: return "low"
        if index_of_difficulty < 4: return "medium"
        return "high"
