"""Compare a conditional flow-matching toy with EDM continuous diffusion."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mouselearn.models.base import seeded_normal_source


BASIS = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
NONZERO = np.asarray([0, 1, 3, 4])
CONDITION_SIZE = 21
OUTPUT_SIZE = 5
ALGORITHMS = ("flow", "edm")
CHECKPOINT_CATEGORIES = ("parity_250", "converged")
DEFAULT_SIZES = (67, 128, 256, 512, 1000, 2000, 4000)
DEFAULT_SEEDS = (42, 43, 44)
QUANTILES = np.asarray([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])


@dataclass(frozen=True)
class ExperimentConfig:
    hidden_size: int = 192
    hidden_layers: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    parity_epochs: int = 250
    epoch_block: int = 250
    maximum_epochs: int = 1000
    continuation_window: int = 50
    continuation_improvement: float = 0.01
    validation_corruptions: int = 4
    flow_steps: int = 16
    edm_steps: int = 18
    edm_p_mean: float = -1.2
    edm_p_std: float = 1.2
    edm_sigma_data: float = 1.0
    edm_sigma_min: float = 0.002
    edm_sigma_max: float = 80.0
    edm_rho: float = 7.0


@dataclass(frozen=True)
class ToySplit:
    train_conditions: np.ndarray
    train_targets: np.ndarray
    validation_conditions: np.ndarray
    validation_targets: np.ndarray
    condition_mean: np.ndarray
    condition_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    split_hash: str


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def parse_algorithm_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result or any(item not in ALGORITHMS for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"expected unique comma-separated algorithms from: {','.join(ALGORITHMS)}")
    return result


def make_split(training_size: int, seed: int) -> ToySplit:
    """Create the one held-out split shared by both algorithms for a run pair."""
    random = np.random.default_rng(7_000_000 + 10_000 * seed + training_size)
    train_c = np.linspace(-2.0, 2.0, training_size, dtype=np.float64)
    random.shuffle(train_c)
    validation_size = max(64, training_size // 4)
    validation_c = random.uniform(-2.0, 2.0, validation_size)
    train_conditions = np.zeros((training_size, CONDITION_SIZE), dtype=np.float64)
    validation_conditions = np.zeros((validation_size, CONDITION_SIZE), dtype=np.float64)
    train_targets = train_c[:, None] * BASIS[None, :]
    validation_targets = validation_c[:, None] * BASIS[None, :]
    condition_mean = train_conditions.mean(axis=0)
    condition_scale = np.maximum(train_conditions.std(axis=0), 1e-8)
    output_mean = train_targets.mean(axis=0)
    output_scale = np.maximum(train_targets.std(axis=0), 1e-6)
    normalized = (
        (train_conditions - condition_mean) / condition_scale,
        (train_targets - output_mean) / output_scale,
        (validation_conditions - condition_mean) / condition_scale,
        (validation_targets - output_mean) / output_scale,
    )
    digest = hashlib.sha256()
    for values in normalized:
        digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return ToySplit(*normalized, condition_mean, condition_scale, output_mean, output_scale, digest.hexdigest())


def make_network(config: ExperimentConfig):
    import torch.nn as nn

    layers: list[Any] = []
    input_size = OUTPUT_SIZE + CONDITION_SIZE + 1
    for _ in range(config.hidden_layers):
        layers.extend((nn.Linear(input_size, config.hidden_size), nn.SiLU()))
        input_size = config.hidden_size
    layers.append(nn.Linear(input_size, OUTPUT_SIZE))
    return nn.Sequential(*layers)


def parameter_count(network: Any) -> int:
    return sum(parameter.numel() for parameter in network.parameters())


def edm_preconditioning(sigma: Any, sigma_data: float = 1.0) -> tuple[Any, Any, Any, Any]:
    """Return EDM c_skip, c_out, c_in, and c_noise coefficients."""
    import torch

    sigma = torch.as_tensor(sigma)
    denominator = sigma.square() + sigma_data**2
    c_skip = sigma_data**2 / denominator
    c_out = sigma * sigma_data / torch.sqrt(denominator)
    c_in = torch.rsqrt(denominator)
    c_noise = torch.log(sigma) / 4.0
    return c_skip, c_out, c_in, c_noise


def edm_denoise(network: Any, noisy: Any, conditions: Any, sigma: Any, sigma_data: float = 1.0) -> Any:
    import torch

    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, sigma_data)
    network_output = network(torch.cat((c_in * noisy, conditions, c_noise), dim=1))
    return c_skip * noisy + c_out * network_output


def edm_weight(sigma: Any, sigma_data: float = 1.0) -> Any:
    return (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()


def karras_schedule(config: ExperimentConfig, *, dtype: Any = None, device: Any = None) -> Any:
    import torch

    dtype = dtype or torch.float32
    ramp = torch.linspace(0, 1, config.edm_steps, dtype=dtype, device=device)
    minimum = config.edm_sigma_min ** (1 / config.edm_rho)
    maximum = config.edm_sigma_max ** (1 / config.edm_rho)
    schedule = (maximum + ramp * (minimum - maximum)) ** config.edm_rho
    return torch.cat((schedule, torch.zeros_like(schedule[:1])))


def expected_nfe(algorithm: str, config: ExperimentConfig) -> int:
    if algorithm == "flow":
        return config.flow_steps * 2
    if algorithm == "edm":
        return config.edm_steps * 2 - 1
    raise ValueError(f"unknown algorithm: {algorithm}")


def _fixed_validation_draws(algorithm: str, validation_targets: Any, seed: int, config: ExperimentConfig) -> dict[str, Any]:
    import torch

    count = len(validation_targets) * config.validation_corruptions
    targets = validation_targets.repeat_interleave(config.validation_corruptions, dim=0)
    random = torch.Generator().manual_seed(900_000 + seed)
    if algorithm == "flow":
        return {
            "targets": targets,
            "source": torch.randn(targets.shape, generator=random),
            "time": torch.rand((count, 1), generator=random),
        }
    standard = torch.randn((count, 1), generator=random)
    sigma = torch.exp(standard * config.edm_p_std + config.edm_p_mean)
    return {"targets": targets, "sigma": sigma, "noise": torch.randn(targets.shape, generator=random)}


def _validation_loss(network: Any, algorithm: str, conditions: Any, draws: dict[str, Any], config: ExperimentConfig) -> float:
    import torch

    target = draws["targets"]
    condition = conditions.repeat_interleave(config.validation_corruptions, dim=0)
    with torch.no_grad():
        if algorithm == "flow":
            source, interpolation_time = draws["source"], draws["time"]
            state = (1 - interpolation_time) * source + interpolation_time * target
            prediction = network(torch.cat((state, condition, interpolation_time), dim=1))
            loss = torch.mean((prediction - (target - source)).square())
        else:
            sigma, noise = draws["sigma"], draws["noise"]
            denoised = edm_denoise(network, target + sigma * noise, condition, sigma, config.edm_sigma_data)
            loss = torch.mean(edm_weight(sigma, config.edm_sigma_data) * (denoised - target).square())
    return float(loss)


def _training_batch_loss(
    network: Any, algorithm: str, target: Any, condition: Any, random: Any, config: ExperimentConfig,
) -> Any:
    import torch

    if algorithm == "flow":
        source = torch.randn(target.shape, generator=random)
        interpolation_time = torch.rand((len(target), 1), generator=random)
        state = (1 - interpolation_time) * source + interpolation_time * target
        prediction = network(torch.cat((state, condition, interpolation_time), dim=1))
        return torch.mean((prediction - (target - source)).square())
    standard = torch.randn((len(target), 1), generator=random)
    sigma = torch.exp(standard * config.edm_p_std + config.edm_p_mean)
    noise = torch.randn(target.shape, generator=random)
    denoised = edm_denoise(network, target + sigma * noise, condition, sigma, config.edm_sigma_data)
    return torch.mean(edm_weight(sigma, config.edm_sigma_data) * (denoised - target).square())


def _should_continue(history: list[dict[str, float]], best_epoch: int, config: ExperimentConfig) -> tuple[bool, dict[str, Any]]:
    end_epoch = int(history[-1]["epoch"])
    window = config.continuation_window
    latest = [item["validation_loss"] for item in history[-window:]]
    preceding = [item["validation_loss"] for item in history[-2 * window:-window]]
    latest_median = float(statistics.median(latest))
    preceding_median = float(statistics.median(preceding))
    improvement = (preceding_median - latest_median) / max(abs(preceding_median), 1e-12)
    recent_best = best_epoch > end_epoch - window
    improved = improvement >= config.continuation_improvement
    decision = recent_best or improved
    return decision, {
        "after_epoch": end_epoch,
        "best_checkpoint_in_latest_50": recent_best,
        "latest_50_median_validation_loss": latest_median,
        "preceding_50_median_validation_loss": preceding_median,
        "relative_median_improvement": improvement,
        "continued": decision and end_epoch < config.maximum_epochs,
    }


def train_algorithm(algorithm: str, split: ToySplit, seed: int, config: ExperimentConfig) -> tuple[Any, Any, dict[str, Any]]:
    """Train one arm and return strict-parity state, converged state, and its full record."""
    import torch

    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algorithm}")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = make_network(config)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
    random = torch.Generator().manual_seed(seed)
    train_c = torch.tensor(split.train_conditions, dtype=torch.float32)
    train_y = torch.tensor(split.train_targets, dtype=torch.float32)
    validation_c = torch.tensor(split.validation_conditions, dtype=torch.float32)
    validation_y = torch.tensor(split.validation_targets, dtype=torch.float32)
    validation_draws = _fixed_validation_draws(algorithm, validation_y, seed, config)
    history: list[dict[str, float]] = []
    decisions: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(network.state_dict())
    parity_state = None
    parity_epoch = 0
    parity_loss = math.inf
    start = time.perf_counter()
    parity_seconds = 0.0

    for epoch in range(1, config.maximum_epochs + 1):
        network.train()
        permutation = torch.randperm(len(train_y), generator=random)
        batch_losses: list[float] = []
        for offset in range(0, len(train_y), config.batch_size):
            indices = permutation[offset:offset + config.batch_size]
            loss = _training_batch_loss(network, algorithm, train_y[indices], train_c[indices], random, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        network.eval()
        validation_loss = _validation_loss(network, algorithm, validation_c, validation_draws, config)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
        history.append({
            "epoch": epoch,
            "training_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
        })
        if epoch == config.parity_epochs:
            parity_state = copy.deepcopy(best_state)
            parity_epoch = best_epoch
            parity_loss = best_loss
            parity_seconds = time.perf_counter() - start
        if epoch % config.epoch_block == 0:
            should_continue, decision = _should_continue(history, best_epoch, config)
            decisions.append(decision)
            if epoch >= config.maximum_epochs or not should_continue:
                break

    if parity_state is None:
        raise RuntimeError("training stopped before the strict-parity checkpoint")
    total_seconds = time.perf_counter() - start
    converged_state = copy.deepcopy(best_state)
    return parity_state, converged_state, {
        "history": history,
        "continuation_decisions": decisions,
        "strict_parity": {"epoch": parity_epoch, "validation_loss": parity_loss},
        "converged": {"epoch": best_epoch, "validation_loss": best_loss},
        "epochs_trained": len(history),
        "training_seconds_to_250": parity_seconds,
        "training_seconds_total": total_seconds,
    }


def generated_sources(sample_seeds: np.ndarray) -> np.ndarray:
    return np.stack([seeded_normal_source(int(seed), OUTPUT_SIZE) for seed in sample_seeds])


def sample_network(
    network: Any, algorithm: str, normalized_conditions: np.ndarray, sources: np.ndarray, config: ExperimentConfig,
) -> tuple[np.ndarray, int]:
    import torch

    condition = torch.tensor(normalized_conditions, dtype=torch.float32)
    source = torch.tensor(sources, dtype=torch.float32)
    network.eval()
    evaluations = 0
    with torch.no_grad():
        if algorithm == "flow":
            state = source.clone()
            step_size = 1.0 / config.flow_steps
            for index in range(config.flow_steps):
                current_time = torch.full((len(state), 1), index * step_size)
                velocity = network(torch.cat((state, condition, current_time), dim=1))
                evaluations += 1
                predicted = state + step_size * velocity
                next_time = torch.full((len(state), 1), (index + 1) * step_size)
                next_velocity = network(torch.cat((predicted, condition, next_time), dim=1))
                evaluations += 1
                state = state + step_size * (velocity + next_velocity) / 2
        elif algorithm == "edm":
            schedule = karras_schedule(config, dtype=source.dtype, device=source.device)
            state = source * schedule[0]
            for index, (current_sigma, next_sigma) in enumerate(zip(schedule[:-1], schedule[1:], strict=True)):
                sigma = current_sigma.expand(len(state), 1)
                denoised = edm_denoise(network, state, condition, sigma, config.edm_sigma_data)
                evaluations += 1
                derivative = (state - denoised) / current_sigma
                predicted = state + (next_sigma - current_sigma) * derivative
                if index < config.edm_steps - 1:
                    next_batch_sigma = next_sigma.expand(len(state), 1)
                    next_denoised = edm_denoise(network, predicted, condition, next_batch_sigma, config.edm_sigma_data)
                    evaluations += 1
                    next_derivative = (predicted - next_denoised) / next_sigma
                    state = state + (next_sigma - current_sigma) * (derivative + next_derivative) / 2
                else:
                    state = predicted
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
    return state.numpy().astype(np.float64), evaluations


def _two_sample_ks(first: np.ndarray, second: np.ndarray) -> float:
    points = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(np.sort(first), points, side="right") / len(first)
    second_cdf = np.searchsorted(np.sort(second), points, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def calculate_metrics(predictions: np.ndarray, reference_c: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    scalar = predictions @ BASIS / float(BASIS @ BASIS)
    projected = scalar[:, None] * BASIS[None, :]
    residual_norms = np.linalg.norm(predictions - projected, axis=1)
    centered = predictions - predictions.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    singular_energy = float(np.sum(singular**2))
    selected = predictions[:, NONZERO]
    selected_centered = selected - selected.mean(axis=0)
    covariance = selected_centered.T @ selected_centered / max(len(selected), 1)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0))
    correlation = covariance / np.maximum(standard_deviation[:, None] * standard_deviation[None, :], 1e-12)
    expected_sign = np.sign(BASIS[NONZERO, None] * BASIS[None, NONZERO])
    predicted_quantiles = np.quantile(scalar, QUANTILES)
    reference_quantiles = np.quantile(reference_c, QUANTILES)
    sorted_scalar = np.sort(scalar)
    sorted_reference = np.sort(reference_c)
    if len(sorted_reference) != len(sorted_scalar):
        positions = (np.arange(len(sorted_scalar)) + 0.5) / len(sorted_scalar)
        sorted_reference = np.quantile(reference_c, positions)
    metrics = {
        "rank_one_variance_fraction": float(singular[0] ** 2 / max(singular_energy, 1e-12)),
        "coordinate_correlation_error": float(np.mean(np.abs(correlation - expected_sign))),
        "clean_vector_deviation_global_rms": float(np.sqrt(np.mean(residual_norms**2))),
        "clean_vector_deviation_median": float(np.median(residual_norms)),
        "clean_vector_deviation_p90": float(np.quantile(residual_norms, 0.90)),
        "c_wasserstein_1": float(np.mean(np.abs(sorted_scalar - sorted_reference))),
        "c_ks_distance": _two_sample_ks(scalar, reference_c),
        "c_std_ratio": float(np.std(scalar) / max(np.std(reference_c), 1e-12)),
        "c_out_of_range_rate": float(np.mean((scalar < -2.0) | (scalar > 2.0))),
        "c_mean": float(np.mean(scalar)),
        "c_std": float(np.std(scalar)),
        "c_quantiles": {f"q{int(level * 100):02d}": float(value) for level, value in zip(QUANTILES, predicted_quantiles, strict=True)},
        "c_quantile_errors": {f"q{int(level * 100):02d}": float(value) for level, value in zip(QUANTILES, predicted_quantiles - reference_quantiles, strict=True)},
    }
    return metrics, scalar, residual_norms


def _checkpoint_result(
    network: Any, state: Any, algorithm: str, category: str, split: ToySplit, sample_seeds: np.ndarray,
    sources: np.ndarray, reference_c: np.ndarray, checkpoint: dict[str, Any], config: ExperimentConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    network.load_state_dict(state)
    conditions = np.zeros((len(sample_seeds), CONDITION_SIZE), dtype=np.float64)
    normalized_conditions = (conditions - split.condition_mean) / split.condition_scale
    started = time.perf_counter()
    normalized_predictions, evaluations = sample_network(network, algorithm, normalized_conditions, sources, config)
    sampling_seconds = time.perf_counter() - started
    predictions = normalized_predictions * split.output_scale + split.output_mean
    metrics, scalar, residual_norms = calculate_metrics(predictions, reference_c)
    if evaluations != expected_nfe(algorithm, config):
        raise RuntimeError(f"{algorithm} sampler used {evaluations} evaluations")
    rows = [{
        "algorithm": algorithm,
        "training_size": len(split.train_targets),
        "training_seed": "",  # filled by the caller
        "checkpoint_category": category,
        "sample_index": index,
        "sample_seed": int(sample_seed),
        **{f"x{coordinate}": float(predictions[index, coordinate]) for coordinate in range(OUTPUT_SIZE)},
        "projected_c": float(scalar[index]),
        "projection_residual_norm": float(residual_norms[index]),
    } for index, sample_seed in enumerate(sample_seeds)]
    return {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_loss": float(checkpoint["validation_loss"]),
        "sampling_seconds": sampling_seconds,
        "network_evaluations": evaluations,
        "metrics": metrics,
    }, rows


def run_one(algorithm: str, training_size: int, seed: int, sample_count: int, config: ExperimentConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    split = make_split(training_size, seed)
    parity_state, converged_state, training = train_algorithm(algorithm, split, seed, config)
    network = make_network(config)
    sample_seeds = np.arange(10_000, 10_000 + sample_count, dtype=np.uint64)
    sources = generated_sources(sample_seeds)
    reference_c = np.linspace(-2.0, 2.0, sample_count, dtype=np.float64)
    checkpoints: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for category, state, checkpoint in (
        ("parity_250", parity_state, training["strict_parity"]),
        ("converged", converged_state, training["converged"]),
    ):
        result, category_rows = _checkpoint_result(
            network, state, algorithm, category, split, sample_seeds, sources, reference_c, checkpoint, config,
        )
        checkpoints[category] = result
        for row in category_rows:
            row["training_seed"] = seed
        rows.extend(category_rows)
    del parity_state, converged_state, network
    if hasattr(torch, "clear_autocast_cache"):
        torch.clear_autocast_cache()
    return {
        "algorithm": algorithm,
        "training_size": training_size,
        "training_seed": seed,
        "condition_size": CONDITION_SIZE,
        "output_size": OUTPUT_SIZE,
        "training_split_hash": split.split_hash,
        "training_count": len(split.train_targets),
        "validation_count": len(split.validation_targets),
        "parameter_count": parameter_count(make_network(config)),
        "config": asdict(config),
        "training": training,
        "checkpoints": checkpoints,
    }, rows


SUMMARY_METRICS = (
    "rank_one_variance_fraction", "coordinate_correlation_error", "clean_vector_deviation_global_rms",
    "clean_vector_deviation_median", "clean_vector_deviation_p90", "c_wasserstein_1", "c_ks_distance",
    "c_std_ratio", "c_out_of_range_rate", "training_seconds", "sampling_seconds", "checkpoint_epoch",
    "network_evaluations", "parameter_count",
) + tuple(f"c_q{int(level * 100):02d}" for level in QUANTILES)


def _flat_summary_record(run: dict[str, Any], category: str) -> dict[str, float]:
    checkpoint = run["checkpoints"][category]
    training_seconds = (
        run["training"]["training_seconds_to_250"] if category == "parity_250"
        else run["training"]["training_seconds_total"]
    )
    record = {
        **checkpoint["metrics"],
        "training_seconds": training_seconds,
        "sampling_seconds": checkpoint["sampling_seconds"],
        "checkpoint_epoch": checkpoint["checkpoint_epoch"],
        "network_evaluations": checkpoint["network_evaluations"],
        "parameter_count": run["parameter_count"],
    }
    record.update({f"c_{name}": value for name, value in checkpoint["metrics"]["c_quantiles"].items()})
    return record


def build_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(run["algorithm"], run["training_size"], category) for run in runs for category in CHECKPOINT_CATEGORIES})
    for algorithm, size, category in keys:
        records = [_flat_summary_record(run, category) for run in runs if run["algorithm"] == algorithm and run["training_size"] == size]
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "training_size": size,
            "checkpoint_category": category,
            "seed_count": len(records),
        }
        for metric in SUMMARY_METRICS:
            values = [float(record[metric]) for record in records]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def _format(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def build_comparison(summary: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    algorithms_present = {run["algorithm"] for run in runs}
    subject = "Both models" if algorithms_present == set(ALGORITHMS) else f"The requested {', '.join(sorted(algorithms_present))} arm"
    fixed_horizon = all(
        run["config"]["epoch_block"] == run["config"]["maximum_epochs"]
        for run in runs
    )
    long_title = "Best checkpoint across fixed training horizon" if fixed_horizon else "Validation-driven convergence"
    long_label = "Across the fixed training horizon" if fixed_horizon else "After validation-driven convergence"
    lines = [
        "# EDM diffusion vs. conditional flow", "",
        f"{subject} used the normalized held-out split, constant 21-dimensional zero condition, "
        "27-192-192-192-5 SiLU MLP, AdamW settings, generated seeds, and validation/checkpoint protocol. "
        "No composite score is used.", "",
    ]
    for category, title in (("parity_250", "Strict 250-epoch parity"), ("converged", long_title)):
        lines.extend([
            f"## {title}", "",
            "| Train n | Algorithm | Best epoch | Rank-1 frac. | Corr. error | Residual RMS | c W1 | c KS | c std ratio | Train s | Sample s | NFE |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (item for item in summary if item["checkpoint_category"] == category):
            lines.append(
                f"| {row['training_size']} | {row['algorithm']} | {_format(row['checkpoint_epoch_mean'])} | "
                f"{_format(row['rank_one_variance_fraction_mean'])} | {_format(row['coordinate_correlation_error_mean'])} | "
                f"{_format(row['clean_vector_deviation_global_rms_mean'])} | {_format(row['c_wasserstein_1_mean'])} | "
                f"{_format(row['c_ks_distance_mean'])} | {_format(row['c_std_ratio_mean'])} | "
                f"{_format(row['training_seconds_mean'])} | {_format(row['sampling_seconds_mean'])} | "
                f"{_format(row['network_evaluations_mean'])} |"
            )
        lines.append("")

    lines.extend(["## Interpretation", ""])
    for category, label in (("parity_250", "At strict parity"), ("converged", long_label)):
        category_rows = [row for row in summary if row["checkpoint_category"] == category]
        size_count = len({row["training_size"] for row in category_rows})
        if algorithms_present != set(ALGORITHMS):
            lines.append(f"- {label}: only {', '.join(sorted(algorithms_present))} was requested, so cross-algorithm wins are unavailable.")
            continue
        comparisons: list[str] = []
        for metric, phrase, higher_is_better in (
            ("rank_one_variance_fraction_mean", "rank-one structure", True),
            ("coordinate_correlation_error_mean", "coordinate correlation", False),
            ("clean_vector_deviation_global_rms_mean", "clean-line deviation", False),
            ("c_wasserstein_1_mean", "scalar-distribution W1", False),
        ):
            wins = {algorithm: 0 for algorithm in ALGORITHMS}
            for size in sorted({row["training_size"] for row in category_rows}):
                by_algorithm = {row["algorithm"]: row[metric] for row in category_rows if row["training_size"] == size}
                if len(by_algorithm) == 2 and not math.isclose(by_algorithm["flow"], by_algorithm["edm"], rel_tol=1e-9, abs_tol=1e-12):
                    winner = max(by_algorithm, key=by_algorithm.get) if higher_is_better else min(by_algorithm, key=by_algorithm.get)
                    wins[winner] += 1
            comparisons.append(f"{phrase}: flow {wins['flow']}, EDM {wins['edm']} of {size_count} sizes")
        lines.append(f"- {label}: " + "; ".join(comparisons) + ".")
    flow_nfe = expected_nfe("flow", ExperimentConfig())
    edm_nfe = expected_nfe("edm", ExperimentConfig())
    lines.extend([
        f"- Sampler work is fixed at {flow_nfe} network evaluations for flow and {edm_nfe} for EDM. "
        "Wall-clock latency is reported separately because NFE alone does not capture runtime overhead.",
        "- A lower deviation or distribution distance is better; a rank-one fraction and c standard-deviation ratio closer to 1 are better. "
        "Checkpoint selection used only fixed-corruption held-out validation loss, never these generated-sample metrics.",
        "",
        "The complete epoch histories, per-seed values, continuation decisions, timings, and configuration are in `runs.json`; "
        "all generated vectors are in `generated-vectors.csv`.", "",
    ])
    return "\n".join(lines)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def verify_results(runs: list[dict[str, Any]], vector_rows: list[dict[str, Any]], summary: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    expected_rows = len(runs) * len(CHECKPOINT_CATEGORIES) * sample_count
    if len(vector_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} generated vectors, found {len(vector_rows)}")
    if not _all_finite(runs) or not _all_finite(vector_rows) or not _all_finite(summary):
        raise RuntimeError("a non-finite experiment result was found")
    for size_seed in {(run["training_size"], run["training_seed"]) for run in runs}:
        hashes = {run["training_split_hash"] for run in runs if (run["training_size"], run["training_seed"]) == size_seed}
        counts = {run["parameter_count"] for run in runs if (run["training_size"], run["training_seed"]) == size_seed}
        if len(hashes) != 1 or len(counts) != 1:
            raise RuntimeError(f"algorithm parity failed for size/seed {size_seed}")
    rebuilt = build_summary(runs)
    if json.dumps(summary, sort_keys=True) != json.dumps(rebuilt, sort_keys=True):
        raise RuntimeError("summary values do not match run records")
    return {
        "all_values_finite": True,
        "generated_vector_row_count": len(vector_rows),
        "expected_generated_vector_row_count": expected_rows,
        "identical_pair_splits": True,
        "identical_parameter_counts": True,
        "summary_matches_run_records": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_int_list, default=DEFAULT_SIZES)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--algorithms", type=parse_algorithm_list, default=ALGORITHMS)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("build/toy-diffusion-vs-flow"))
    parser.add_argument(
        "--fixed-epochs", type=int,
        help="train every requested arm for exactly this many epochs; still retain the best first-250 checkpoint",
    )
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    config = ExperimentConfig()
    if args.fixed_epochs is not None:
        if args.fixed_epochs < config.parity_epochs:
            parser.error("--fixed-epochs must be at least 250")
        config = replace(config, epoch_block=args.fixed_epochs, maximum_epochs=args.fixed_epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(args.sizes) * len(args.seeds) * len(args.algorithms)
    for size in args.sizes:
        for seed in args.seeds:
            for algorithm in args.algorithms:
                print(f"[{len(runs) + 1}/{total}] {algorithm} n={size} seed={seed}", flush=True)
                run, rows = run_one(algorithm, size, seed, args.samples, config)
                runs.append(run)
                vector_rows.extend(rows)
                partial = {
                    "schema_version": 1,
                    "status": "in_progress",
                    "completed_runs": len(runs),
                    "expected_runs": total,
                    "runs": runs,
                }
                (args.output_dir / "runs.partial.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
                print(
                    f"  epochs={run['training']['epochs_trained']} parity={run['training']['strict_parity']['epoch']} "
                    f"converged={run['training']['converged']['epoch']} time={run['training']['training_seconds_total']:.2f}s",
                    flush=True,
                )
    summary = build_summary(runs)
    verification = verify_results(runs, vector_rows, summary, args.samples)
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "EDM diffusion vs. conditional flow on c * (-2, -1, 0, 1, 2)",
        "condition": "constant 21-dimensional zero vector",
        "sizes": list(args.sizes),
        "seeds": list(args.seeds),
        "samples_per_checkpoint": args.samples,
        "algorithm_order": list(args.algorithms),
        "config": asdict(config),
        "total_wall_seconds": time.perf_counter() - started,
        "verification": verification,
        "runs": runs,
    }
    (args.output_dir / "runs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "generated-vectors.csv", vector_rows)
    (args.output_dir / "comparison.md").write_text(build_comparison(summary, runs), encoding="utf-8")
    partial_path = args.output_dir / "runs.partial.json"
    partial_path.unlink(missing_ok=True)
    print(f"Completed {len(runs)} runs and {len(vector_rows)} generated vectors in {result['total_wall_seconds']:.2f}s")
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
