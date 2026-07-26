"""Small internal conditional flow-matching model with deterministic CPU inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import GeneratedParameterBatch, ProcessedDataset, constrain_parameter_output, seeded_normal_source


@dataclass(frozen=True)
class ConditionalFlowConfig:
    hidden_size: int = 192
    hidden_layers: int = 3
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 0.0003
    seed: int = 42
    solver: str = "heun"
    solver_steps: int = 16
    checkpoint_every: int = 25
    maximum_neighbor_distance: float = 6.0

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.hidden_layers, self.epochs, self.batch_size, self.solver_steps, self.checkpoint_every) < 1:
            raise ValueError("flow integer settings must be positive")
        if self.learning_rate <= 0 or self.maximum_neighbor_distance <= 0 or self.seed < 0:
            raise ValueError("flow scale and seed settings must be valid")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("flow solver must be euler or heun")


class ConditionalFlowGenerator:
    model_type = "conditional_flow"

    def __init__(self, config: ConditionalFlowConfig = ConditionalFlowConfig()):
        self.config = config
        self._condition_mean: np.ndarray | None = None
        self._condition_scale: np.ndarray | None = None
        self._training_conditions: np.ndarray | None = None
        self._output_mean: np.ndarray | None = None
        self._output_scale: np.ndarray | None = None
        self._network = None
        self.history: list[dict[str, float]] = []

    def _make_network(self, condition_size: int, output_size: int):
        import torch.nn as nn
        layers: list[Any] = []
        input_size = output_size + condition_size + 1
        for _ in range(self.config.hidden_layers):
            layers.extend((nn.Linear(input_size, self.config.hidden_size), nn.SiLU()))
            input_size = self.config.hidden_size
        layers.append(nn.Linear(input_size, output_size))
        return nn.Sequential(*layers)

    def fit(
        self, dataset: ProcessedDataset, callback: Callable[[int, dict[str, float]], None] | None = None,
        checkpoint_directory: Path | None = None,
    ) -> "ConditionalFlowGenerator":
        import torch
        train = dataset.subset("train")
        validation = dataset.subset("validation") if "validation" in dataset.splits else train
        self._condition_mean = train.conditions.mean(axis=0)
        self._condition_scale = np.maximum(train.conditions.std(axis=0), 1e-8)
        self._training_conditions = (train.conditions - self._condition_mean) / self._condition_scale
        self._output_mean = train.outputs.mean(axis=0)
        self._output_scale = np.maximum(train.outputs.std(axis=0), 1e-6)
        train_c = torch.tensor(self._training_conditions, dtype=torch.float32)
        train_y = torch.tensor((train.outputs - self._output_mean) / self._output_scale, dtype=torch.float32)
        validation_c = torch.tensor((validation.conditions - self._condition_mean) / self._condition_scale, dtype=torch.float32)
        validation_y = torch.tensor((validation.outputs - self._output_mean) / self._output_scale, dtype=torch.float32)
        torch.manual_seed(self.config.seed)
        torch.use_deterministic_algorithms(True)
        self._network = self._make_network(train_c.shape[1], train_y.shape[1])
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.config.learning_rate)
        random = torch.Generator().manual_seed(self.config.seed)
        best_loss = float("inf")
        best_state = None
        if checkpoint_directory is not None:
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, self.config.epochs + 1):
            self._network.train()
            permutation = torch.randperm(len(train_y), generator=random)
            losses = []
            for offset in range(0, len(train_y), self.config.batch_size):
                indices = permutation[offset:offset + self.config.batch_size]
                target, condition = train_y[indices], train_c[indices]
                source = torch.randn(target.shape, generator=random)
                time = torch.rand((len(indices), 1), generator=random)
                state = (1 - time) * source + time * target
                velocity = self._network(torch.cat((state, condition, time), dim=1))
                loss = torch.mean((velocity - (target - source)) ** 2)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            self._network.eval()
            with torch.no_grad():
                validation_random = torch.Generator().manual_seed(self.config.seed + epoch)
                source = torch.randn(validation_y.shape, generator=validation_random)
                time = torch.rand((len(validation_y), 1), generator=validation_random)
                state = (1 - time) * source + time * validation_y
                validation_loss = float(torch.mean((self._network(torch.cat((state, validation_c, time), dim=1)) - (validation_y - source)) ** 2))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {name: value.detach().clone() for name, value in self._network.state_dict().items()}
            metrics = {"training_loss": float(np.mean(losses)), "validation_loss": validation_loss, "best_validation_loss": best_loss}
            self.history.append({"epoch": float(epoch), **metrics})
            if checkpoint_directory is not None and (epoch % self.config.checkpoint_every == 0 or epoch == self.config.epochs):
                torch.save({"epoch": epoch, "model": self._network.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics}, checkpoint_directory / f"epoch-{epoch:04d}.pt")
            if callback:
                callback(epoch, metrics)
        if best_state is not None:
            self._network.load_state_dict(best_state)
        return self

    def generate_batch(self, conditions: np.ndarray, seeds: np.ndarray) -> GeneratedParameterBatch:
        import torch
        if any(value is None for value in (self._condition_mean, self._condition_scale, self._training_conditions, self._output_mean, self._output_scale, self._network)):
            raise RuntimeError("fit or load must be called before generate")
        conditions = np.asarray(conditions, dtype=np.float64)
        seeds = np.asarray(seeds, dtype=np.uint64)
        normalized = (conditions - self._condition_mean) / self._condition_scale
        if conditions.ndim != 2 or seeds.shape != (len(conditions),):
            raise ValueError("conditions and seeds must be aligned batches")
        self._network.eval()
        results, nearest = [], []
        with torch.no_grad():
            for index, (condition, seed) in enumerate(zip(normalized, seeds, strict=True)):
                nearest.append(float(np.min(np.linalg.norm(self._training_conditions - condition, axis=1))))
                source = seeded_normal_source(int(seed), len(self._output_mean))
                output = self.integrate_source(condition, source)
                results.append(constrain_parameter_output(output, float(conditions[index, 3])))
        distances = np.asarray(nearest)
        return GeneratedParameterBatch(np.asarray(results), distances, distances > self.config.maximum_neighbor_distance)

    def integrate_source(self, normalized_condition: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Integrate one explicit source vector for cross-runtime parity fixtures."""
        import torch
        if self._network is None or self._output_mean is None or self._output_scale is None:
            raise RuntimeError("fit or load must be called before integration")
        condition_tensor = torch.tensor(np.asarray(normalized_condition, dtype=np.float32)[None, :])
        state = torch.tensor(np.asarray(source, dtype=np.float32)[None, :])
        step = 1.0 / self.config.solver_steps
        self._network.eval()
        with torch.no_grad():
            for step_index in range(self.config.solver_steps):
                time = torch.full((1, 1), step_index * step)
                velocity = self._network(torch.cat((state, condition_tensor, time), dim=1))
                if self.config.solver == "heun":
                    predicted = state + step * velocity
                    next_time = torch.full((1, 1), (step_index + 1) * step)
                    next_velocity = self._network(torch.cat((predicted, condition_tensor, next_time), dim=1))
                    state = state + step * (velocity + next_velocity) / 2
                else:
                    state = state + step * velocity
        return state.numpy()[0] * self._output_scale + self._output_mean

    def save(self, destination: Path) -> None:
        import torch
        if self._network is None:
            raise RuntimeError("fit must be called before save")
        destination.mkdir(parents=True, exist_ok=False)
        torch.save(self._network.state_dict(), destination / "weights.pt")
        np.savez_compressed(destination / "normalization.npz", condition_mean=self._condition_mean,
                            condition_scale=self._condition_scale, training_conditions=self._training_conditions,
                            output_mean=self._output_mean, output_scale=self._output_scale)
        (destination / "model.json").write_text(json.dumps({"schema_version": 1, "model_type": self.model_type, "config": self.config.__dict__}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, source: Path) -> "ConditionalFlowGenerator":
        import torch
        manifest = json.loads((source / "model.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("model_type") != cls.model_type:
            raise ValueError("unsupported conditional-flow manifest")
        model = cls(ConditionalFlowConfig(**manifest["config"]))
        with np.load(source / "normalization.npz") as values:
            for name in values.files:
                setattr(model, f"_{name}", values[name].copy())
        model._network = model._make_network(len(model._condition_mean), len(model._output_mean))
        model._network.load_state_dict(torch.load(source / "weights.pt", map_location="cpu", weights_only=True))
        model._network.eval()
        return model


__all__ = ["ConditionalFlowConfig", "ConditionalFlowGenerator"]
