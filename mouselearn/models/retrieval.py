"""Seeded nearest-neighbour baseline over standardized condition vectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalConfig:
    neighbor_count: int = 8
    temperature: float = 0.15
    maximum_neighbor_distance: float = 6.0


@dataclass(frozen=True)
class RetrievalResult:
    output: np.ndarray
    source_index: int
    nearest_distance: float
    out_of_distribution: bool


class RetrievalGenerator:
    """A reproducible conditional baseline; output rows may be any fixed model vector."""

    def __init__(self, config: RetrievalConfig = RetrievalConfig()):
        if config.neighbor_count < 1 or config.temperature <= 0 or config.maximum_neighbor_distance <= 0:
            raise ValueError("retrieval configuration values must be positive")
        self.config = config
        self._conditions: np.ndarray | None = None
        self._outputs: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, conditions: np.ndarray, outputs: np.ndarray) -> "RetrievalGenerator":
        conditions = np.asarray(conditions, dtype=np.float64)
        outputs = np.asarray(outputs, dtype=np.float64)
        if conditions.ndim != 2 or outputs.ndim != 2 or len(conditions) != len(outputs) or len(conditions) == 0:
            raise ValueError("conditions and outputs must be non-empty matching 2D arrays")
        self._mean = conditions.mean(axis=0)
        self._scale = np.maximum(conditions.std(axis=0), 1e-8)
        self._conditions, self._outputs = conditions, outputs
        return self

    def generate(self, condition: np.ndarray, seed: int) -> RetrievalResult:
        if self._conditions is None or self._outputs is None or self._mean is None or self._scale is None:
            raise RuntimeError("fit must be called before generate")
        value = np.asarray(condition, dtype=np.float64)
        if value.shape != (self._conditions.shape[1],):
            raise ValueError("condition shape does not match fitted conditions")
        distances = np.linalg.norm((self._conditions - value) / self._scale, axis=1)
        count = min(self.config.neighbor_count, len(distances))
        indices = np.argsort(distances, kind="stable")[:count]
        weights = np.exp(-(distances[indices] - distances[indices][0]) / self.config.temperature)
        weights /= weights.sum()
        source = int(np.random.default_rng(seed).choice(indices, p=weights))
        return RetrievalResult(self._outputs[source].copy(), source, float(distances[indices][0]), bool(distances[indices][0] > self.config.maximum_neighbor_distance))
