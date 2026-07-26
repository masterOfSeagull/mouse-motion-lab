"""Fixed-horizon and residual-matched interpolation study for the flow/EDM toy."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toys.diffusion_vs_flow.run import (
    ALGORITHMS, BASIS, CHECKPOINT_CATEGORIES, CONDITION_SIZE, ExperimentConfig, OUTPUT_SIZE,
    _fixed_validation_draws, _training_batch_loss, _validation_loss, calculate_metrics,
    generated_sources, make_network, make_split, parameter_count, sample_network,
)


DISTANCE_EDGES = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, math.inf])
DISTANCE_LABELS = ("0-.10", ".10-.25", ".25-.50", ".50-.75", ".75-1", "1-2", ">2")
RESIDUAL_THRESHOLDS = (0.25, 0.50, 1.00)


def train_exact(
    algorithm: str, training_size: int, seed: int, epochs: int, probe_every: int,
    probe_sources: np.ndarray, config: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, float]]]:
    """Train to an exact epoch, returning that state and a held-out probe curve."""
    import torch

    split = make_split(training_size, seed)
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
    probe_conditions = np.zeros((len(probe_sources), CONDITION_SIZE), dtype=np.float64)
    curve: list[dict[str, float]] = []
    started = time.perf_counter()
    latest_training_loss = math.inf
    for epoch in range(1, epochs + 1):
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
        latest_training_loss = float(np.mean(batch_losses))
        if epoch % probe_every == 0 or epoch == epochs:
            network.eval()
            normalized, nfe = sample_network(network, algorithm, probe_conditions, probe_sources, config)
            predictions = normalized * split.output_scale + split.output_mean
            probe_metrics, _, _ = calculate_metrics(predictions, np.linspace(-2, 2, len(predictions)))
            validation_loss = _validation_loss(network, algorithm, validation_c, validation_draws, config)
            curve.append({
                "epoch": epoch,
                "training_loss": latest_training_loss,
                "validation_loss": validation_loss,
                "probe_residual_rms": probe_metrics["clean_vector_deviation_global_rms"],
                "probe_rank_one_fraction": probe_metrics["rank_one_variance_fraction"],
                "probe_nfe": nfe,
            })
    return copy.deepcopy(network.state_dict()), {
        "algorithm": algorithm,
        "training_size": training_size,
        "training_seed": seed,
        "epochs_trained": epochs,
        "optimizer_steps": epochs * math.ceil(training_size / config.batch_size),
        "training_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count(network),
        "split_hash": split.split_hash,
        "final_training_loss": latest_training_loss,
    }, curve


def interpolation_metrics(
    predictions: np.ndarray, training_outputs: np.ndarray, reference_c: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base, scalar, residual_norm = calculate_metrics(predictions, reference_c)
    sorted_outputs = training_outputs[np.argsort(training_outputs @ BASIS)]
    training_c = np.sort(training_outputs @ BASIS / float(BASIS @ BASIS))
    vector_gap = float(np.linalg.norm(sorted_outputs[1] - sorted_outputs[0]))
    c_gap = float(training_c[1] - training_c[0])
    half_vector_gap = vector_gap / 2
    full_distance = np.min(np.linalg.norm(predictions[:, None, :] - training_outputs[None, :, :], axis=2), axis=1)
    projected_c_distance = np.min(np.abs(scalar[:, None] - training_c[None, :]), axis=1)
    projected_vector_distance = projected_c_distance * float(np.linalg.norm(BASIS))
    full_normalized = full_distance / half_vector_gap
    projected_normalized = projected_c_distance / (c_gap / 2)
    full_bin = np.clip(np.digitize(full_normalized, DISTANCE_EDGES[1:-1], right=False), 0, len(DISTANCE_LABELS) - 1)
    projected_bin = np.clip(np.digitize(projected_normalized, DISTANCE_EDGES[1:-1], right=False), 0, len(DISTANCE_LABELS) - 1)
    inside = (scalar >= training_c[0]) & (scalar <= training_c[-1])
    far_half = projected_normalized >= 0.5
    novelty: dict[str, Any] = {
        "training_vector_gap": vector_gap,
        "half_training_vector_gap": half_vector_gap,
        "training_c_gap": c_gap,
        "nearest_training_vector_distance_mean": float(np.mean(full_distance)),
        "nearest_training_vector_distance_median": float(np.median(full_distance)),
        "nearest_training_vector_distance_p90": float(np.quantile(full_distance, 0.9)),
        "projected_gap_position_mean": float(np.mean(np.minimum(projected_normalized, 1.0))),
        "projected_in_far_half_rate": float(np.mean(inside & far_half)),
        "full_distance_bin_rates": {
            label: float(np.mean(full_bin == index)) for index, label in enumerate(DISTANCE_LABELS)
        },
        "projected_distance_bin_rates": {
            label: float(np.mean(projected_bin == index)) for index, label in enumerate(DISTANCE_LABELS)
        },
    }
    for threshold in RESIDUAL_THRESHOLDS:
        consistent = residual_norm <= threshold * half_vector_gap
        novelty[f"consistent_inbetween_rate_r{threshold:.2f}"] = float(np.mean(inside & far_half & consistent))
        novelty[f"consistent_copy_like_rate_r{threshold:.2f}"] = float(
            np.mean(inside & (projected_normalized <= 0.1) & consistent)
        )
    return {**base, **novelty}, {
        "scalar": scalar,
        "residual_norm": residual_norm,
        "full_distance": full_distance,
        "projected_distance": projected_vector_distance,
        "full_normalized": full_normalized,
        "projected_normalized": projected_normalized,
        "full_bin": full_bin,
        "projected_bin": projected_bin,
        "inside": inside,
    }


def evaluate_state(
    state: dict[str, Any], algorithm: str, epoch: int, seed: int, comparison: str,
    sample_sources: np.ndarray, sample_seeds: np.ndarray, config: ExperimentConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = make_split(32, seed)
    network = make_network(config)
    network.load_state_dict(state)
    conditions = np.zeros((len(sample_sources), CONDITION_SIZE), dtype=np.float64)
    started = time.perf_counter()
    normalized, nfe = sample_network(network, algorithm, conditions, sample_sources, config)
    sample_seconds = time.perf_counter() - started
    predictions = normalized * split.output_scale + split.output_mean
    training_outputs = split.train_targets * split.output_scale + split.output_mean
    metrics, values = interpolation_metrics(predictions, training_outputs, np.linspace(-2, 2, len(predictions)))
    rows: list[dict[str, Any]] = []
    for index, sample_seed in enumerate(sample_seeds):
        rows.append({
            "comparison": comparison,
            "algorithm": algorithm,
            "training_seed": seed,
            "model_epoch": epoch,
            "sample_index": index,
            "sample_seed": int(sample_seed),
            **{f"x{coordinate}": float(predictions[index, coordinate]) for coordinate in range(OUTPUT_SIZE)},
            "projected_c": float(values["scalar"][index]),
            "projection_residual_norm": float(values["residual_norm"][index]),
            "nearest_training_vector_distance": float(values["full_distance"][index]),
            "nearest_projected_training_vector_distance": float(values["projected_distance"][index]),
            "nearest_distance_half_gap_units": float(values["full_normalized"][index]),
            "projected_distance_half_gap_units": float(values["projected_normalized"][index]),
            "nearest_distance_bin": DISTANCE_LABELS[int(values["full_bin"][index])],
            "projected_distance_bin": DISTANCE_LABELS[int(values["projected_bin"][index])],
            "inside_training_range": bool(values["inside"][index]),
            **{
                f"consistent_inbetween_r{threshold:.2f}": bool(
                    values["inside"][index]
                    and values["projected_normalized"][index] >= 0.5
                    and values["residual_norm"][index] <= threshold * metrics["half_training_vector_gap"]
                ) for threshold in RESIDUAL_THRESHOLDS
            },
        })
    return {
        "comparison": comparison,
        "algorithm": algorithm,
        "training_seed": seed,
        "model_epoch": epoch,
        "sampling_seconds": sample_seconds,
        "network_evaluations": nfe,
        "metrics": metrics,
    }, rows


SUMMARY_FIELDS = (
    "model_epoch", "rank_one_variance_fraction", "coordinate_correlation_error",
    "clean_vector_deviation_global_rms", "c_wasserstein_1", "c_ks_distance", "c_std_ratio",
    "c_out_of_range_rate",
    "nearest_training_vector_distance_mean", "nearest_training_vector_distance_median",
    "nearest_training_vector_distance_p90", "projected_in_far_half_rate",
    "consistent_inbetween_rate_r0.25", "consistent_inbetween_rate_r0.50",
    "consistent_inbetween_rate_r1.00", "consistent_copy_like_rate_r0.25",
)


def summarize(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in ("same_epoch_30000", "residual_matched"):
        for algorithm in ALGORITHMS:
            selected = [item for item in evaluations if item["comparison"] == comparison and item["algorithm"] == algorithm]
            row: dict[str, Any] = {"comparison": comparison, "algorithm": algorithm, "seed_count": len(selected)}
            for field in SUMMARY_FIELDS:
                values = [float(item[field] if field == "model_epoch" else item["metrics"][field]) for item in selected]
                row[f"{field}_mean"] = float(np.mean(values))
                row[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            for label in DISTANCE_LABELS:
                values = [item["metrics"]["full_distance_bin_rates"][label] for item in selected]
                row[f"full_bin_{label}_mean"] = float(np.mean(values))
                row[f"full_bin_{label}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            rows.append(row)
    return rows


def comparison_markdown(summary: list[dict[str, Any]], matches: list[dict[str, Any]]) -> str:
    lines = [
        "# n=32 interpolation: EDM versus flow", "",
        "Training targets are 32 equally spaced vectors on `c * [-2,-1,0,1,2]`. The exact 30,000-epoch "
        "comparison uses equal optimizer-step budgets. Residual matching uses separate 512-vector probe seeds; "
        "the reported 2,048-vector evaluation seeds never select an epoch.", "",
    ]
    for comparison, title in (("same_epoch_30000", "Same epoch: 30,000"), ("residual_matched", "Residual-matched")):
        lines.extend([
            f"## {title}", "",
            "| Algorithm | Epoch | Residual RMS | Rank-1 | c W1 | Out of range | Nearest distance | Far-half rate | Consistent in-between strict | Copy-like strict |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (item for item in summary if item["comparison"] == comparison):
            lines.append(
                f"| {row['algorithm']} | {row['model_epoch_mean']:.0f} | {row['clean_vector_deviation_global_rms_mean']:.5f} | "
                f"{row['rank_one_variance_fraction_mean']:.6f} | {row['c_wasserstein_1_mean']:.5f} | {row['c_out_of_range_rate_mean']:.3f} | "
                f"{row['nearest_training_vector_distance_mean_mean']:.5f} | {row['projected_in_far_half_rate_mean']:.3f} | "
                f"{row['consistent_inbetween_rate_r0.25_mean']:.3f} | {row['consistent_copy_like_rate_r0.25_mean']:.3f} |"
            )
        lines.append("")
    same = {row["algorithm"]: row for row in summary if row["comparison"] == "same_epoch_30000"}
    matched = {row["algorithm"]: row for row in summary if row["comparison"] == "residual_matched"}
    lines.extend([
        "## Findings", "",
        f"- At 30,000 epochs, EDM has lower residual RMS ({same['edm']['clean_vector_deviation_global_rms_mean']:.5f} vs. "
        f"{same['flow']['clean_vector_deviation_global_rms_mean']:.5f}) and therefore a much higher strict consistent-in-between rate "
        f"({same['edm']['consistent_inbetween_rate_r0.25_mean']:.3f} vs. {same['flow']['consistent_inbetween_rate_r0.25_mean']:.3f}).",
        f"- Gap-position novelty alone is similar: the far-half rates are {same['flow']['projected_in_far_half_rate_mean']:.3f} for flow "
        f"and {same['edm']['projected_in_far_half_rate_mean']:.3f} for EDM. EDM's equal-epoch advantage is primarily consistency, not a stronger tendency to choose gap centers.",
        f"- At matched residual, strict consistent-in-between rates are nearly equal ({matched['flow']['consistent_inbetween_rate_r0.25_mean']:.3f} flow, "
        f"{matched['edm']['consistent_inbetween_rate_r0.25_mean']:.3f} EDM). Flow retains better scalar-distribution fidelity "
        f"(W1 {matched['flow']['c_wasserstein_1_mean']:.5f} vs. {matched['edm']['c_wasserstein_1_mean']:.5f}) and fewer out-of-range values "
        f"({matched['flow']['c_out_of_range_rate_mean']:.3f} vs. {matched['edm']['c_out_of_range_rate_mean']:.3f}).",
        "",
    ])
    lines.extend(["## Per-seed residual matches", "", "| Seed | Fixed model | Fixed epoch | Matched model | Matched epoch | Probe target | Probe matched |", "|---:|:---|---:|:---|---:|---:|---:|"])
    for item in matches:
        lines.append(
            f"| {item['seed']} | {item['fixed_algorithm']} | 30000 | {item['matched_algorithm']} | {item['matched_epoch']} | "
            f"{item['target_probe_residual_rms']:.6f} | {item['matched_probe_residual_rms']:.6f} |"
        )
    lines.extend([
        "", "## Distance bins", "",
        "Nearest-vector distances are normalized by half the spacing between adjacent training output vectors. "
        "A clean midpoint has distance 1; a copied training output has distance 0. Full bin distributions are in `summary.csv`.", "",
        "Strict consistent in-between means: projected c is inside the training range, its projected distance is at least "
        "0.5 half-gap, and off-line residual is at most 0.25 half-gap. Rates at 0.50 and 1.00 residual thresholds are also in `summary.csv`.", "",
    ])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30_000)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--probe-samples", type=int, default=512)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("build/toy-diffusion-vs-flow-n32-interpolation"))
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if args.epochs < args.probe_every or args.epochs % args.probe_every:
        parser.error("--epochs must be a positive multiple of --probe-every")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(ExperimentConfig(), parity_epochs=args.epochs, epoch_block=args.epochs, maximum_epochs=args.epochs)
    probe_seeds = np.arange(50_000, 50_000 + args.probe_samples, dtype=np.uint64)
    probe_sources = generated_sources(probe_seeds)
    sample_seeds = np.arange(10_000, 10_000 + args.samples, dtype=np.uint64)
    sample_sources = generated_sources(sample_seeds)
    runs: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        final_states: dict[str, dict[str, Any]] = {}
        curves: dict[str, list[dict[str, float]]] = {}
        for algorithm in ALGORITHMS:
            print(f"training {algorithm} n=32 seed={seed} to epoch {args.epochs}", flush=True)
            state, record, curve = train_exact(algorithm, 32, seed, args.epochs, args.probe_every, probe_sources, config)
            final_states[algorithm], curves[algorithm] = state, curve
            runs.append({**record, "probe_curve": curve})
            evaluation, rows = evaluate_state(state, algorithm, args.epochs, seed, "same_epoch_30000", sample_sources, sample_seeds, config)
            evaluations.append(evaluation); vector_rows.extend(rows)
        final_residual = {algorithm: curves[algorithm][-1]["probe_residual_rms"] for algorithm in ALGORITHMS}
        fixed_algorithm = max(final_residual, key=final_residual.get)
        matched_algorithm = min(final_residual, key=final_residual.get)
        target = final_residual[fixed_algorithm]
        candidates = [item for item in curves[matched_algorithm] if item["probe_residual_rms"] <= target]
        match_point = candidates[0] if candidates else min(curves[matched_algorithm], key=lambda item: abs(item["probe_residual_rms"] - target))
        match_epoch = int(match_point["epoch"])
        if match_epoch == args.epochs:
            matched_state = final_states[matched_algorithm]
        else:
            print(f"replaying {matched_algorithm} seed={seed} to matched epoch {match_epoch}", flush=True)
            matched_state, replay_record, _ = train_exact(
                matched_algorithm, 32, seed, match_epoch, max(match_epoch, 1), probe_sources, config,
            )
            replay_record["purpose"] = "recover_residual_matched_exact_state"
            runs.append(replay_record)
        matches.append({
            "seed": seed,
            "fixed_algorithm": fixed_algorithm,
            "matched_algorithm": matched_algorithm,
            "matched_epoch": match_epoch,
            "target_probe_residual_rms": target,
            "matched_probe_residual_rms": match_point["probe_residual_rms"],
            "selection_probe_seed_start": int(probe_seeds[0]),
            "selection_probe_count": len(probe_seeds),
        })
        for algorithm, epoch, state in (
            (fixed_algorithm, args.epochs, final_states[fixed_algorithm]),
            (matched_algorithm, match_epoch, matched_state),
        ):
            evaluation, rows = evaluate_state(state, algorithm, epoch, seed, "residual_matched", sample_sources, sample_seeds, config)
            evaluations.append(evaluation); vector_rows.extend(rows)
    summary = summarize(evaluations)
    expected_rows = len(seeds) * len(ALGORITHMS) * 2 * args.samples
    if len(vector_rows) != expected_rows or not all(np.isfinite(float(row[f"x{i}"])) for row in vector_rows for i in range(5)):
        raise RuntimeError("generated-vector verification failed")
    result = {
        "schema_version": 1,
        "status": "complete",
        "training_size": 32,
        "training_outputs": "32 equally spaced c values in [-2,2] times [-2,-1,0,1,2]",
        "config": asdict(config),
        "epochs": args.epochs,
        "seeds": list(seeds),
        "samples": args.samples,
        "probe_samples": args.probe_samples,
        "probe_every": args.probe_every,
        "total_wall_seconds": time.perf_counter() - started,
        "verification": {"all_generated_values_finite": True, "generated_rows": len(vector_rows), "expected_rows": expected_rows},
        "matches": matches,
        "training_runs": runs,
        "evaluations": evaluations,
    }
    (args.output_dir / "runs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "generated-vectors.csv", vector_rows)
    write_csv(args.output_dir / "probe-curves.csv", [
        {"algorithm": run["algorithm"], "training_seed": run["training_seed"], **point}
        for run in runs if "probe_curve" in run for point in run["probe_curve"]
    ])
    (args.output_dir / "comparison.md").write_text(comparison_markdown(summary, matches), encoding="utf-8")
    print(f"completed in {result['total_wall_seconds']:.2f}s with {len(vector_rows)} generated rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
