"""PCA plus a condition-binned diagonal Gaussian mixture baseline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import GeneratedParameterBatch, ProcessedDataset, constrain_parameter_output


@dataclass(frozen=True)
class PcaMixtureConfig:
    retained_variance: float = 0.97
    latent_dimension: int = 24
    mixture_component_count: int = 6
    condition_temperature: float = 1.0
    covariance_floor: float = 0.03
    maximum_neighbor_distance: float = 6.0
    covariance_type: str = "diagonal"

    def __post_init__(self) -> None:
        if not 0 < self.retained_variance <= 1 or min(self.latent_dimension, self.mixture_component_count) < 1:
            raise ValueError("PCA variance, latent dimension, and component count must be positive")
        if min(self.condition_temperature, self.covariance_floor, self.maximum_neighbor_distance) <= 0:
            raise ValueError("mixture scale values must be positive")
        if self.covariance_type != "diagonal":
            raise ValueError("the Milestone-4 PCA baseline supports diagonal covariance")


class PcaMixtureGenerator:
    model_type = "pca_mixture"

    def __init__(self, config: PcaMixtureConfig = PcaMixtureConfig()):
        self.config = config
        self._condition_mean: np.ndarray | None = None
        self._condition_scale: np.ndarray | None = None
        self._training_conditions: np.ndarray | None = None
        self._output_mean: np.ndarray | None = None
        self._output_scale: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._condition_centers: np.ndarray | None = None
        self._latent_means: np.ndarray | None = None
        self._latent_scales: np.ndarray | None = None
        self._component_priors: np.ndarray | None = None

    def fit(self, dataset: ProcessedDataset) -> "PcaMixtureGenerator":
        train = dataset.subset("train")
        conditions, outputs = train.conditions, train.outputs
        self._condition_mean = conditions.mean(axis=0)
        self._condition_scale = np.maximum(conditions.std(axis=0), 1e-8)
        standardized_conditions = (conditions - self._condition_mean) / self._condition_scale
        self._training_conditions = standardized_conditions
        self._output_mean = outputs.mean(axis=0)
        self._output_scale = np.maximum(outputs.std(axis=0), 1e-8)
        standardized_outputs = (outputs - self._output_mean) / self._output_scale
        _, singular, vectors = np.linalg.svd(standardized_outputs, full_matrices=False)
        variance = singular * singular
        cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
        retained = int(np.searchsorted(cumulative, self.config.retained_variance) + 1)
        dimension = min(max(1, retained), self.config.latent_dimension, vectors.shape[0])
        self._components = vectors[:dimension]
        latent = standardized_outputs @ self._components.T
        count = min(self.config.mixture_component_count, len(conditions))
        centers, assignments = _condition_kmeans(standardized_conditions, count)
        self._condition_centers = centers
        global_scale = np.maximum(latent.std(axis=0), self.config.covariance_floor)
        means, scales, priors = [], [], []
        for component in range(count):
            members = latent[assignments == component]
            means.append(members.mean(axis=0))
            scales.append(np.maximum(members.std(axis=0), global_scale * self.config.covariance_floor))
            priors.append(len(members) / len(latent))
        self._latent_means = np.asarray(means)
        self._latent_scales = np.asarray(scales)
        self._component_priors = np.asarray(priors)
        return self

    def generate_batch(self, conditions: np.ndarray, seeds: np.ndarray) -> GeneratedParameterBatch:
        arrays = (
            self._condition_mean, self._condition_scale, self._training_conditions, self._output_mean,
            self._output_scale, self._components, self._condition_centers, self._latent_means,
            self._latent_scales, self._component_priors,
        )
        if any(array is None for array in arrays):
            raise RuntimeError("fit must be called before generate")
        conditions = np.asarray(conditions, dtype=np.float64)
        seeds = np.asarray(seeds, dtype=np.uint64)
        if conditions.ndim != 2 or seeds.shape != (len(conditions),) or conditions.shape[1] != self._condition_mean.shape[0]:
            raise ValueError("conditions and seeds must be aligned batches")
        normalized = (conditions - self._condition_mean) / self._condition_scale
        outputs, nearest = [], []
        for condition, seed in zip(normalized, seeds, strict=True):
            training_distances = np.linalg.norm(self._training_conditions - condition, axis=1)
            nearest.append(float(training_distances.min()))
            component_distances = np.linalg.norm(self._condition_centers - condition, axis=1)
            weights = self._component_priors * np.exp(-(component_distances - component_distances.min()) / self.config.condition_temperature)
            weights /= weights.sum()
            random = np.random.default_rng(int(seed))
            component = int(random.choice(len(weights), p=weights))
            latent = self._latent_means[component] + self._latent_scales[component] * random.normal(size=self._latent_means.shape[1])
            standardized = latent @ self._components
            outputs.append(constrain_parameter_output(standardized * self._output_scale + self._output_mean, float(conditions[len(outputs), 3])))
        distances = np.asarray(nearest)
        return GeneratedParameterBatch(np.asarray(outputs), distances, distances > self.config.maximum_neighbor_distance)

    def save(self, destination: Path) -> None:
        arrays = {
            "condition_mean": self._condition_mean, "condition_scale": self._condition_scale,
            "training_conditions": self._training_conditions, "output_mean": self._output_mean,
            "output_scale": self._output_scale, "components": self._components,
            "condition_centers": self._condition_centers, "latent_means": self._latent_means,
            "latent_scales": self._latent_scales, "component_priors": self._component_priors,
        }
        if any(value is None for value in arrays.values()):
            raise RuntimeError("fit must be called before save")
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination / "weights.npz", **arrays)
        (destination / "model.json").write_text(json.dumps({
            "schema_version": 1, "model_type": self.model_type,
            "config": self.config.__dict__, "latent_dimension": int(self._components.shape[0]),
        }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, source: Path) -> "PcaMixtureGenerator":
        manifest = json.loads((source / "model.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("model_type") != cls.model_type:
            raise ValueError("unsupported PCA mixture manifest")
        model = cls(PcaMixtureConfig(**manifest["config"]))
        with np.load(source / "weights.npz") as weights:
            for name in weights.files:
                setattr(model, f"_{name}", weights[name].copy())
        return model


def _condition_kmeans(values: np.ndarray, count: int, iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic farthest-first condition bins with stable empty-bin recovery."""
    centers = [values[int(np.argmin(np.linalg.norm(values, axis=1)))]]
    while len(centers) < count:
        distance = np.min(np.stack([np.linalg.norm(values - center, axis=1) for center in centers]), axis=0)
        centers.append(values[int(np.argmax(distance))])
    centers_array = np.asarray(centers).copy()
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        distances = np.stack([np.linalg.norm(values - center, axis=1) for center in centers_array], axis=1)
        next_assignments = np.argmin(distances, axis=1)
        if np.array_equal(next_assignments, assignments) and _ > 0:
            break
        assignments = next_assignments
        nearest = distances[np.arange(len(values)), assignments]
        for index in range(count):
            members = values[assignments == index]
            centers_array[index] = members.mean(axis=0) if len(members) else values[int(np.argmax(nearest))]
    return centers_array, assignments


__all__ = ["PcaMixtureConfig", "PcaMixtureGenerator"]
