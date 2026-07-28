"""Hash-verified portable PCA-mixture export and numerical reference runtime."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

from mouselearn.models import PcaMixtureGenerator


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_pca_mixture(model_directory: Path, destination: Path) -> dict[str, Any]:
    model = PcaMixtureGenerator.load(model_directory)
    arrays = (
        model._condition_mean, model._condition_scale, model._training_conditions,
        model._output_mean, model._output_scale, model._components, model._condition_centers,
        model._latent_means, model._latent_scales, model._component_priors,
    )
    if any(value is None for value in arrays):
        raise ValueError("PCA mixture is missing learned arrays")
    source_manifest = json.loads((model_directory / "model.json").read_text(encoding="utf-8"))
    provenance_path = model_directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {
        key: source_manifest.get(key) for key in (
            "model_id", "dataset_snapshot_id", "preprocessing_run_id", "code_revision", "training_seed"
        )
    }
    condition_size = int(model._condition_mean.shape[0])
    output_size = int(model._output_mean.shape[0])
    training_count = int(model._training_conditions.shape[0])
    latent_size = int(model._components.shape[0])
    mixture_count = int(model._condition_centers.shape[0])
    position_count = int((output_size - 1) // 2)
    if output_size != position_count * 2 + 1:
        raise ValueError("PCA output must contain x/y positions followed by log duration")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    try:
        payload = destination / "pca.bin"
        with payload.open("wb") as stream:
            stream.write(struct.pack(
                "<8sIIIIII2d", b"MMLPCA1\0", condition_size, output_size, training_count,
                latent_size, mixture_count, position_count,
                model.config.condition_temperature, model.config.maximum_neighbor_distance,
            ))
            for values in arrays:
                stream.write(np.asarray(values, dtype="<f8").reshape(-1).tobytes())
        manifest = {
            "schema_version": 1,
            "format": "mousemotionlab-pca-mixture",
            "model_id": source_manifest.get("model_id", "unregistered"),
            "model_type": "pca_mixture",
            "condition_size": condition_size,
            "output_size": output_size,
            "position_count": position_count,
            "latent_dimension": latent_size,
            "mixture_component_count": mixture_count,
            "training_condition_count": training_count,
            "config": source_manifest.get("config", model.config.__dict__),
            "provenance": provenance,
            "files": {"pca.bin": _digest(payload)},
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


class _PortableRandom:
    def __init__(self, seed: int):
        self.state = seed & ((1 << 64) - 1)
        self.spare: float | None = None

    def _splitmix(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)

    def uniform(self) -> float:
        return ((self._splitmix() >> 11) + 0.5) / 9007199254740992.0

    def normal(self) -> float:
        if self.spare is not None:
            value, self.spare = self.spare, None
            return value
        radius = np.sqrt(-2.0 * np.log(self.uniform()))
        angle = 2.0 * np.pi * self.uniform()
        self.spare = float(radius * np.sin(angle))
        return float(radius * np.cos(angle))


class PortablePcaRuntime:
    """Pure-Python reference for the dependency-free native PCA runtime."""

    def __init__(self, source: Path):
        self.source = source.resolve()
        self.manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("format") != "mousemotionlab-pca-mixture":
            raise ValueError("unsupported PCA export manifest")
        for name, expected in self.manifest.get("files", {}).items():
            if _digest(self.source / name) != expected:
                raise ValueError(f"export artifact hash changed: {name}")
        with (self.source / "pca.bin").open("rb") as stream:
            header = stream.read(struct.calcsize("<8sIIIIII2d"))
            magic, c, o, n, k, m, p, self.condition_temperature, self.maximum_neighbor_distance = struct.unpack("<8sIIIIII2d", header)
            if magic != b"MMLPCA1\0" or o != p * 2 + 1 or min(c, n, k, m, p) < 1:
                raise ValueError("unsupported PCA payload dimensions")
            def take(count: int) -> np.ndarray:
                data = stream.read(count * 8)
                if len(data) != count * 8:
                    raise ValueError("PCA payload is truncated")
                return np.frombuffer(data, dtype="<f8").copy()
            self.condition_mean = take(c)
            self.condition_scale = take(c)
            self.training_conditions = take(n * c).reshape(n, c)
            self.output_mean = take(o)
            self.output_scale = take(o)
            self.components = take(k * o).reshape(k, o)
            self.condition_centers = take(m * c).reshape(m, c)
            self.latent_means = take(m * k).reshape(m, k)
            self.latent_scales = take(m * k).reshape(m, k)
            self.component_priors = take(m)
            if stream.read(1):
                raise ValueError("PCA payload has trailing data")

    def generate_parameters(self, condition: np.ndarray, seed: int, *, exact_endpoint: bool = False) -> tuple[np.ndarray, float, bool]:
        normalized = (np.asarray(condition, dtype=np.float64) - self.condition_mean) / self.condition_scale
        nearest = float(np.linalg.norm(self.training_conditions - normalized, axis=1).min())
        distances = np.linalg.norm(self.condition_centers - normalized, axis=1)
        weights = self.component_priors * np.exp(-(distances - distances.min()) / self.condition_temperature)
        weights /= weights.sum()
        random = _PortableRandom(seed)
        pick, total, component = random.uniform(), 0.0, len(weights) - 1
        for index, weight in enumerate(weights):
            total += float(weight)
            if pick < total:
                component = index
                break
        latent = self.latent_means[component] + self.latent_scales[component] * np.asarray(
            [random.normal() for _ in range(self.components.shape[0])]
        )
        output = (latent @ self.components) * self.output_scale + self.output_mean
        output[0:2] = 0.0
        if exact_endpoint:
            endpoint = output[-3:-1].copy()
            norm_squared = float(endpoint @ endpoint)
            if norm_squared < 1e-18:
                raise ValueError("raw PCA endpoint is too close to zero")
            points = output[:-1].reshape(-1, 2)
            x, y = points[:, 0].copy(), points[:, 1].copy()
            points[:, 0] = (x * endpoint[0] + y * endpoint[1]) / norm_squared
            points[:, 1] = (-x * endpoint[1] + y * endpoint[0]) / norm_squared
            points[0] = 0.0
            points[-1] = (1.0, 0.0)
        return output, nearest, nearest > self.maximum_neighbor_distance


__all__ = ["PortablePcaRuntime", "export_pca_mixture"]
