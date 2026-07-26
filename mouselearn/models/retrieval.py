"""Seeded nearest-neighbour baseline over standardized condition vectors."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import GeneratedParameterBatch, ProcessedDataset, constrain_parameter_output


@dataclass(frozen=True)
class RetrievalConfig:
    neighbor_count: int = 8
    temperature: float = 0.15
    maximum_neighbor_distance: float = 6.0
    distance_weights: tuple[float, ...] = ()
    allow_blending: bool = False
    maximum_blend_distance: float = 0.20


@dataclass(frozen=True)
class RetrievalResult:
    output: np.ndarray
    source_index: int
    nearest_distance: float
    out_of_distribution: bool


class RetrievalGenerator:
    """A reproducible conditional baseline; output rows may be any fixed model vector."""

    model_type = "retrieval"

    def __init__(self, config: RetrievalConfig = RetrievalConfig()):
        if config.neighbor_count < 1 or min(config.temperature, config.maximum_neighbor_distance, config.maximum_blend_distance) <= 0:
            raise ValueError("retrieval configuration values must be positive")
        if config.distance_weights and any(value <= 0 for value in config.distance_weights):
            raise ValueError("retrieval distance weights must be positive")
        self.config = config
        self._conditions: np.ndarray | None = None
        self._outputs: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, conditions: np.ndarray | ProcessedDataset, outputs: np.ndarray | None = None) -> "RetrievalGenerator":
        if isinstance(conditions, ProcessedDataset):
            dataset = conditions.subset("train")
            conditions, outputs = dataset.conditions, dataset.outputs
        if outputs is None:
            raise ValueError("outputs are required when fitting from condition arrays")
        conditions = np.asarray(conditions, dtype=np.float64)
        outputs = np.asarray(outputs, dtype=np.float64)
        if conditions.ndim != 2 or outputs.ndim != 2 or len(conditions) != len(outputs) or len(conditions) == 0:
            raise ValueError("conditions and outputs must be non-empty matching 2D arrays")
        self._mean = conditions.mean(axis=0)
        self._scale = np.maximum(conditions.std(axis=0), 1e-8)
        if self.config.distance_weights and len(self.config.distance_weights) != conditions.shape[1]:
            raise ValueError("retrieval distance weights must match condition feature count")
        self._conditions, self._outputs = conditions, outputs
        return self

    def generate(self, condition: np.ndarray, seed: int) -> RetrievalResult:
        if self._conditions is None or self._outputs is None or self._mean is None or self._scale is None:
            raise RuntimeError("fit must be called before generate")
        value = np.asarray(condition, dtype=np.float64)
        if value.shape != (self._conditions.shape[1],):
            raise ValueError("condition shape does not match fitted conditions")
        weights_by_feature = np.asarray(self.config.distance_weights or np.ones(self._conditions.shape[1]), dtype=np.float64)
        distances = np.linalg.norm(((self._conditions - value) / self._scale) * weights_by_feature, axis=1)
        count = min(self.config.neighbor_count, len(distances))
        indices = np.argsort(distances, kind="stable")[:count]
        weights = np.exp(-(distances[indices] - distances[indices][0]) / self.config.temperature)
        weights /= weights.sum()
        compatible = distances[indices] <= distances[indices][0] + self.config.maximum_blend_distance
        if self.config.allow_blending and int(np.count_nonzero(compatible)) > 1:
            blend_indices = indices[compatible]
            blend_weights = weights[compatible]
            blend_weights /= blend_weights.sum()
            output = np.average(self._outputs[blend_indices], axis=0, weights=blend_weights)
            source = -1
        else:
            source = int(np.random.default_rng(seed).choice(indices, p=weights))
            output = self._outputs[source].copy()
        return RetrievalResult(output, source, float(distances[indices][0]), bool(distances[indices][0] > self.config.maximum_neighbor_distance))

    def generate_batch(self, conditions: np.ndarray, seeds: np.ndarray) -> GeneratedParameterBatch:
        conditions = np.asarray(conditions, dtype=np.float64)
        seeds = np.asarray(seeds, dtype=np.uint64)
        if conditions.ndim != 2 or seeds.shape != (len(conditions),):
            raise ValueError("conditions and seeds must be aligned batches")
        results = [self.generate(condition, int(seed)) for condition, seed in zip(conditions, seeds, strict=True)]
        return GeneratedParameterBatch(
            np.stack([constrain_parameter_output(result.output, float(condition[3])) for result, condition in zip(results, conditions, strict=True)]),
            np.asarray([result.nearest_distance for result in results]),
            np.asarray([result.out_of_distribution for result in results], dtype=bool),
            np.asarray([result.source_index for result in results], dtype=np.int64),
        )

    def save(self, destination: Path) -> None:
        if any(value is None for value in (self._conditions, self._outputs, self._mean, self._scale)):
            raise RuntimeError("fit must be called before save")
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination / "weights.npz", conditions=self._conditions, outputs=self._outputs, mean=self._mean, scale=self._scale)
        (destination / "model.json").write_text(json.dumps({
            "schema_version": 1, "model_type": self.model_type,
            "config": {"neighbor_count": self.config.neighbor_count, "temperature": self.config.temperature,
                       "maximum_neighbor_distance": self.config.maximum_neighbor_distance,
                       "distance_weights": self.config.distance_weights, "allow_blending": self.config.allow_blending,
                       "maximum_blend_distance": self.config.maximum_blend_distance},
        }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, source: Path) -> "RetrievalGenerator":
        manifest = json.loads((source / "model.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("model_type") != cls.model_type:
            raise ValueError("unsupported retrieval model manifest")
        model = cls(RetrievalConfig(**manifest["config"]))
        with np.load(source / "weights.npz") as weights:
            model._conditions = weights["conditions"].copy()
            model._outputs = weights["outputs"].copy()
            model._mean = weights["mean"].copy()
            model._scale = weights["scale"].copy()
        return model
