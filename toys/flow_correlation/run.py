"""Test whether the project's flow learns joint outputs while ignoring nuisance inputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mouselearn.models.base import seeded_normal_source
from mouselearn.models.conditional_flow import ConditionalFlowConfig, ConditionalFlowGenerator


BASIS = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
NONZERO = np.asarray([0, 1, 3, 4])
CONDITION_SIZE = 21


def _two_sample_ks(first: np.ndarray, second: np.ndarray) -> float:
    points = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(np.sort(first), points, side="right") / len(first)
    second_cdf = np.searchsorted(np.sort(second), points, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _conditions(random: np.random.Generator, count: int, mode: str) -> np.ndarray:
    if mode == "constant":
        return np.zeros((count, CONDITION_SIZE), dtype=np.float64)
    if mode == "garbage":
        return random.normal(size=(count, CONDITION_SIZE))
    raise ValueError(f"unknown condition mode: {mode}")


def train_case(
    config: ConditionalFlowConfig, training_size: int, sample_count: int, condition_mode: str,
) -> tuple[dict, list[dict]]:
    import torch

    data_random = np.random.default_rng(7_000 + training_size)
    training_constants = np.linspace(-2.0, 2.0, training_size, dtype=np.float64)
    data_random.shuffle(training_constants)
    validation_size = max(64, training_size // 4)
    validation_constants = data_random.uniform(-2.0, 2.0, validation_size)
    test_constants = np.linspace(-2.0, 2.0, sample_count, dtype=np.float64)
    training_conditions = _conditions(data_random, training_size, condition_mode)
    validation_conditions = _conditions(data_random, validation_size, condition_mode)
    test_conditions = _conditions(data_random, sample_count, condition_mode)

    training_targets = training_constants[:, None] * BASIS[None, :]
    validation_targets = validation_constants[:, None] * BASIS[None, :]
    condition_mean = training_conditions.mean(axis=0)
    condition_scale = np.maximum(training_conditions.std(axis=0), 1e-8)
    output_mean = training_targets.mean(axis=0)
    output_scale = np.maximum(training_targets.std(axis=0), 1e-6)
    train_c = torch.tensor((training_conditions - condition_mean) / condition_scale, dtype=torch.float32)
    validation_c = torch.tensor((validation_conditions - condition_mean) / condition_scale, dtype=torch.float32)
    test_c = torch.tensor((test_conditions - condition_mean) / condition_scale, dtype=torch.float32)
    train_y = torch.tensor((training_targets - output_mean) / output_scale, dtype=torch.float32)
    validation_y = torch.tensor((validation_targets - output_mean) / output_scale, dtype=torch.float32)

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    network = ConditionalFlowGenerator(config)._make_network(CONDITION_SIZE, len(BASIS))
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
    random = torch.Generator().manual_seed(config.seed)
    best_validation_loss = math.inf
    best_epoch = 0
    best_state = None
    final_training_loss = math.inf
    final_validation_loss = math.inf
    for epoch in range(1, config.epochs + 1):
        network.train()
        permutation = torch.randperm(len(train_y), generator=random)
        losses: list[float] = []
        for offset in range(0, len(train_y), config.batch_size):
            indices = permutation[offset:offset + config.batch_size]
            target, condition = train_y[indices], train_c[indices]
            source = torch.randn(target.shape, generator=random)
            time = torch.rand((len(indices), 1), generator=random)
            state = (1 - time) * source + time * target
            velocity = network(torch.cat((state, condition, time), dim=1))
            loss = torch.mean((velocity - (target - source)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_training_loss = float(np.mean(losses))
        network.eval()
        with torch.no_grad():
            validation_random = torch.Generator().manual_seed(config.seed + epoch)
            source = torch.randn(validation_y.shape, generator=validation_random)
            time = torch.rand((validation_size, 1), generator=validation_random)
            state = (1 - time) * source + time * validation_y
            velocity = network(torch.cat((state, validation_c, time), dim=1))
            final_validation_loss = float(torch.mean((velocity - (validation_y - source)) ** 2))
        if final_validation_loss < best_validation_loss:
            best_validation_loss = final_validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
    if best_state is not None:
        network.load_state_dict(best_state)

    source = np.stack([seeded_normal_source(10_000 + index, len(BASIS)) for index in range(sample_count)])
    state = torch.tensor(source, dtype=torch.float32)
    step = 1.0 / config.solver_steps
    network.eval()
    with torch.no_grad():
        for step_index in range(config.solver_steps):
            time = torch.full((sample_count, 1), step_index * step)
            velocity = network(torch.cat((state, test_c, time), dim=1))
            if config.solver == "heun":
                predicted = state + step * velocity
                next_time = torch.full((sample_count, 1), (step_index + 1) * step)
                next_velocity = network(torch.cat((predicted, test_c, next_time), dim=1))
                state = state + step * (velocity + next_velocity) / 2
            else:
                state = state + step * velocity
    predictions = state.numpy().astype(np.float64) * output_scale + output_mean

    scalar = predictions @ BASIS / float(BASIS @ BASIS)
    projected = scalar[:, None] * BASIS[None, :]
    residual = predictions - projected
    centered = predictions - predictions.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank_one_fraction = float(singular[0] ** 2 / np.sum(singular ** 2))
    correlation = np.corrcoef(predictions[:, NONZERO], rowvar=False)
    expected_sign = np.sign(BASIS[NONZERO, None] * BASIS[None, NONZERO])
    correlation_error = float(np.mean(np.abs(correlation - expected_sign)))
    target_std = float(test_constants.std())
    predicted_std = float(scalar.std())
    quantile_levels = np.asarray([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    target_quantiles = np.quantile(test_constants, quantile_levels)
    predicted_quantiles = np.quantile(scalar, quantile_levels)
    wasserstein = float(np.mean(np.abs(np.sort(scalar) - np.sort(test_constants))))
    centered_scalar = scalar - scalar.mean()
    predicted_skewness = float(np.mean(centered_scalar**3) / max(predicted_std**3, 1e-12))
    predicted_excess_kurtosis = float(np.mean(centered_scalar**4) / max(predicted_std**4, 1e-12) - 3.0)
    prediction_rms = float(np.sqrt(np.mean(predictions**2)))
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    input_dependence: dict | None = None
    if condition_mode == "garbage":
        correlations = np.asarray([
            np.corrcoef(test_conditions[:, index], scalar)[0, 1] for index in range(CONDITION_SIZE)
        ])
        design = np.column_stack((np.ones(sample_count), test_conditions))
        coefficients = np.linalg.lstsq(design, scalar, rcond=None)[0]
        fitted = design @ coefficients
        total_sum = float(np.sum((scalar - scalar.mean()) ** 2))
        input_dependence = {
            "max_absolute_input_c_correlation": float(np.max(np.abs(correlations))),
            "input_c_correlations": correlations.tolist(),
            "linear_r_squared": 1.0 - float(np.sum((scalar - fitted) ** 2)) / max(total_sum, 1e-12),
        }

    rows = []
    for index in range(sample_count):
        row = {
            "sample_index": index,
            "seed": 10_000 + index,
            "x0": float(predictions[index, 0]),
            "x1": float(predictions[index, 1]),
            "x2": float(predictions[index, 2]),
            "x3": float(predictions[index, 3]),
            "x4": float(predictions[index, 4]),
            "c_hat": float(scalar[index]),
            "projection_residual_norm": float(np.linalg.norm(residual[index])),
        }
        row.update({f"input_{column}": float(test_conditions[index, column]) for column in range(CONDITION_SIZE)})
        rows.append(row)

    return {
        "condition_mode": condition_mode,
        "condition_size": CONDITION_SIZE,
        "training_size": training_size,
        "validation_size": validation_size,
        "test_size": sample_count,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss,
        "target_scalar": {"mean": float(test_constants.mean()), "std": target_std, "range": [-2.0, 2.0]},
        "predicted_scalar": {
            "mean": float(scalar.mean()),
            "std": predicted_std,
            "std_ratio_to_target": predicted_std / target_std,
            "skewness": predicted_skewness,
            "excess_kurtosis": predicted_excess_kurtosis,
            "outside_training_range_fraction": float(np.mean((scalar < -2.0) | (scalar > 2.0))),
            "quantile_levels": quantile_levels.tolist(),
            "quantiles": predicted_quantiles.tolist(),
        },
        "target_vs_predicted_scalar": {
            "target_quantiles": target_quantiles.tolist(),
            "quantile_errors": (predicted_quantiles - target_quantiles).tolist(),
            "empirical_wasserstein_1": wasserstein,
            "empirical_ks_distance": _two_sample_ks(test_constants, scalar),
        },
        "input_dependence": input_dependence,
        "predicted_mean_vector": [float(value) for value in predictions.mean(axis=0)],
        "predicted_std_vector": [float(value) for value in predictions.std(axis=0)],
        "example_predictions": [
            {
                "vector": [float(value) for value in predictions[index]],
                "projected_scalar": float(scalar[index]),
                "projection_residual_norm": float(np.linalg.norm(residual[index])),
                "input_first_five": [float(value) for value in test_conditions[index, :5]],
            }
            for index in range(min(8, sample_count))
        ],
        "nonzero_coordinate_correlation": correlation.tolist(),
        "mean_correlation_sign_error": correlation_error,
        "rank_one_variance_fraction": rank_one_fraction,
        "projection_relative_rmse": residual_rms / max(prediction_rms, 1e-12),
        "prediction_rms": prediction_rms,
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("build/toy-flow-correlation/results.json"))
    parser.add_argument("--vectors", type=Path, default=Path("build/toy-flow-correlation/generated-vectors-garbage-input.csv"))
    parser.add_argument("--training-size", type=int, help="run one targeted case instead of the default comparison suite")
    parser.add_argument("--preset", choices=("small", "standard"), default="standard")
    parser.add_argument("--condition-mode", choices=("constant", "garbage"), default="constant")
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("--samples must be at least 100")
    small = ConditionalFlowConfig(hidden_size=96, hidden_layers=2, epochs=120, checkpoint_every=20)
    standard = ConditionalFlowConfig()
    if args.training_size is not None:
        if args.training_size < 10:
            parser.error("--training-size must be at least 10")
        config = small if args.preset == "small" else standard
        cases = [(f"{args.condition_mode}-{args.preset}-{args.training_size}", config, args.training_size, args.condition_mode)]
    else:
        cases = [
            (f"{mode}-small-67", small, 67, mode)
            for mode in ("constant", "garbage")
        ] + [
            (f"{mode}-standard-{size}", standard, size, mode)
            for size in (67, 512)
            for mode in ("constant", "garbage")
        ]
    reports: dict[str, dict] = {}
    vector_rows: list[dict] = []
    for name, config, size, mode in cases:
        report, rows = train_case(config, size, args.samples, mode)
        reports[name] = report
        vector_rows.extend({"case": name, **row} for row in rows)
    result = {
        "schema_version": 2,
        "hypothesis": "targets are c * [-2,-1,0,1,2]; conditions are either constant or independent 21-D Gaussian garbage",
        "split_protocol": "separate training and validation sets; best validation checkpoint; unseen test conditions and c distribution",
        "cases": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    with args.vectors.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(vector_rows[0]))
        writer.writeheader()
        writer.writerows(vector_rows)
    for name, case in result["cases"].items():
        dependence = case["input_dependence"]
        nuisance = "" if dependence is None else f" nuisance_r2={dependence['linear_r_squared']:.4f}"
        print(
            f"{name}: rank1={case['rank_one_variance_fraction']:.6f} "
            f"projection_error={case['projection_relative_rmse']:.6f} "
            f"correlation_error={case['mean_correlation_sign_error']:.6f} "
            f"scalar_std_ratio={case['predicted_scalar']['std_ratio_to_target']:.3f}" + nuisance
        )
    print(args.output.resolve())
    print(args.vectors.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
