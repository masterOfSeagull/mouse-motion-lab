"""Conditional-flow ONNX export and Python ONNX Runtime integration."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

from mouselearn.models import (
    ConditionalFlowGenerator, GenerationRequest, condition_vector, constrain_parameter_output, decode_output,
    seeded_normal_source,
)


class _VelocityWrapper:
    @staticmethod
    def create(network):
        import torch
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__(); self.inner = inner
            def forward(self, state, condition, time):
                return self.inner(torch.cat((state, condition, time), dim=1))
        return Wrapper(network)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_conditional_flow(model_directory: Path, destination: Path) -> dict[str, Any]:
    import onnx
    import torch
    model = ConditionalFlowGenerator.load(model_directory)
    if model.config.source_mode != "gaussian":
        raise ValueError("portable export does not yet support training-sample flow sources")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    try:
        wrapper = _VelocityWrapper.create(model._network).eval()
        output_size, condition_size = len(model._output_mean), len(model._condition_mean)
        onnx_path = destination / "velocity.onnx"
        torch.onnx.export(
            wrapper, (torch.zeros(1, output_size), torch.zeros(1, condition_size), torch.zeros(1, 1)),
            onnx_path, input_names=["state", "condition", "time"], output_names=["velocity"],
            dynamic_axes={name: {0: "batch"} for name in ("state", "condition", "time", "velocity")},
            opset_version=17, dynamo=False,
        )
        onnx.checker.check_model(onnx.load(onnx_path))
        shutil.copy2(model_directory / "normalization.npz", destination / "normalization.npz")
        normalization_path = destination / "normalization.bin"
        arrays = (
            model._condition_mean, model._condition_scale, model._training_conditions.reshape(-1),
            model._output_mean, model._output_scale,
        )
        with normalization_path.open("wb") as stream:
            stream.write(struct.pack("<8sIII", b"MMLNORM1", condition_size, output_size, len(model._training_conditions)))
            for values in arrays:
                stream.write(np.asarray(values, dtype="<f8").tobytes())
        source_manifest = json.loads((model_directory / "model.json").read_text(encoding="utf-8"))
        manifest = {
            "schema_version": 1, "format": "mousemotionlab-onnx-flow", "model_id": source_manifest.get("model_id", "unregistered"),
            "dataset_snapshot_id": source_manifest.get("dataset_snapshot_id", "synthetic"),
            "preprocessing_run_id": source_manifest.get("preprocessing_run_id", "synthetic"),
            "solver": model.config.solver, "solver_steps": model.config.solver_steps,
            "condition_mode": model.config.condition_mode,
            "condition_size": condition_size, "output_size": output_size, "position_count": 64,
            "files": {"velocity.onnx": _digest(onnx_path), "normalization.npz": _digest(destination / "normalization.npz"),
                      "normalization.bin": _digest(normalization_path)},
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


class OnnxFlowRuntime:
    def __init__(self, source: Path):
        import onnxruntime as ort
        self.source = source.resolve()
        self.manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("format") != "mousemotionlab-onnx-flow":
            raise ValueError("unsupported export manifest")
        for name, expected in self.manifest["files"].items():
            if _digest(self.source / name) != expected:
                raise ValueError(f"export artifact hash changed: {name}")
        with np.load(self.source / "normalization.npz") as values:
            self.condition_mean = values["condition_mean"].copy()
            self.condition_scale = values["condition_scale"].copy()
            self.output_mean = values["output_mean"].copy()
            self.output_scale = values["output_scale"].copy()
            self.training_conditions = values["training_conditions"].copy()
        self.session = ort.InferenceSession(str(self.source / "velocity.onnx"), providers=["CPUExecutionProvider"])

    def velocity(self, state: np.ndarray, condition: np.ndarray, time: np.ndarray) -> np.ndarray:
        return self.session.run(["velocity"], {
            "state": np.asarray(state, dtype=np.float32), "condition": np.asarray(condition, dtype=np.float32),
            "time": np.asarray(time, dtype=np.float32),
        })[0]

    def integrate_source(self, normalized_condition: np.ndarray, source: np.ndarray) -> np.ndarray:
        condition = np.asarray(normalized_condition, dtype=np.float32)[None, :]
        state = np.asarray(source, dtype=np.float32)[None, :]
        steps = int(self.manifest["solver_steps"]); step = 1.0 / steps
        for index in range(steps):
            time = np.full((1, 1), index * step, dtype=np.float32)
            velocity = self.velocity(state, condition, time)
            if self.manifest["solver"] == "heun":
                predicted = state + step * velocity
                next_velocity = self.velocity(predicted, condition, np.full((1, 1), (index + 1) * step, dtype=np.float32))
                state = state + step * (velocity + next_velocity) / 2
            else:
                state = state + step * velocity
        return state[0].astype(np.float64) * self.output_scale + self.output_mean

    @staticmethod
    def normal_source(seed: int, count: int) -> np.ndarray:
        return seeded_normal_source(seed, count)

    def generate(self, request: GenerationRequest):
        raw_condition = condition_vector(request)
        normalized = (
            np.zeros_like(raw_condition) if self.manifest.get("condition_mode", "full") == "zero"
            else (raw_condition - self.condition_mean) / self.condition_scale
        )
        output = self.integrate_source(normalized, self.normal_source(request.random_seed, len(self.output_mean)))
        output = constrain_parameter_output(output, float(raw_condition[3]))
        distances = np.linalg.norm(self.training_conditions - normalized, axis=1)
        nearest = float(distances.min())
        return decode_output(output, request, condition_distance=nearest, out_of_distribution=nearest > 6.0)


__all__ = ["OnnxFlowRuntime", "export_conditional_flow"]
