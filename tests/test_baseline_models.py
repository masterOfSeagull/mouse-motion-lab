from __future__ import annotations

import math
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mouselearn.models import (
    ConditionalFlowConfig, ConditionalFlowGenerator, GenerationRequest, PcaMixtureConfig, PcaMixtureGenerator, ProcessedDataset,
    RetrievalConfig, RetrievalGenerator, condition_vector, constrain_parameter_output, decode_output, validate_baseline,
)
from mouselearn.export import OnnxFlowRuntime, PortablePcaRuntime, export_conditional_flow, export_pca_mixture
from mouselearn.cli import main as cli_main


def _dataset(position_count: int = 64) -> ProcessedDataset:
    requests, conditions, outputs, splits, trial_ids = [], [], [], [], []
    for index in range(36):
        angle = 2 * math.pi * index / 36
        start_x, start_y = 400.0 + (index % 3) * 10, 300.0 + (index % 2) * 8
        distance = 180.0 + (index % 7) * 15
        request = GenerationRequest(
            start_x, start_y, start_x + math.cos(angle) * distance, start_y + math.sin(angle) * distance,
            18.0 + index % 5, 0.0, 0.0, 1920.0, 1080.0, random_seed=index,
        )
        canonical = []
        bend = ((index % 3) - 1) * 0.035
        for position in range(position_count):
            progress = position / (position_count - 1)
            canonical.extend((progress, bend * math.sin(math.pi * progress)))
        canonical[-2] = 1.0 + 0.02 * math.cos(angle)
        canonical[-1] = 0.02 * math.sin(angle)
        requests.append(request)
        conditions.append(condition_vector(request))
        outputs.append([*canonical, math.log(220_000_000 + index * 4_000_000)])
        splits.append("train" if index < 24 else "validation" if index < 30 else "test")
        trial_ids.append(f"trial-{index}")
    return ProcessedDataset(
        np.asarray(conditions), np.asarray(outputs), tuple(requests), tuple(splits), tuple(trial_ids), "snapshot", "run",
    )


def test_condition_vector_and_decoder_produce_valid_equal_time_trajectory() -> None:
    dataset = _dataset()
    request = dataset.requests[0]
    deliberately_invalid_endpoint = dataset.outputs[0].copy()
    deliberately_invalid_endpoint[-3:-1] = (2.0, 2.0)
    result = decode_output(deliberately_invalid_endpoint, request)
    assert len(condition_vector(request)) == 21
    assert len(result.samples) == 64
    assert (result.samples[0].x, result.samples[0].y) == (request.start_x, request.start_y)
    assert math.dist((result.endpoint_x, result.endpoint_y), (request.target_center_x, request.target_center_y)) < request.target_radius
    assert result.endpoint_projected
    assert all(right.relative_time_ns > left.relative_time_ns for left, right in zip(result.samples, result.samples[1:]))


def test_retrieval_common_interface_round_trips_artifact(tmp_path: Path) -> None:
    dataset = _dataset()
    model = RetrievalGenerator(RetrievalConfig(neighbor_count=4)).fit(dataset)
    first = model.generate_batch(dataset.conditions[24:26], np.asarray([7, 8], dtype=np.uint64))
    destination = tmp_path / "retrieval"
    model.save(destination)
    loaded = RetrievalGenerator.load(destination)
    second = loaded.generate_batch(dataset.conditions[24:26], np.asarray([7, 8], dtype=np.uint64))
    assert np.array_equal(first.outputs, second.outputs)
    report = validate_baseline(loaded, dataset)
    assert report["passed"]
    assert report["held_out_sample_count"] == 12


def test_pca_mixture_is_seeded_valid_and_round_trips_artifact(tmp_path: Path) -> None:
    dataset = _dataset()
    model = PcaMixtureGenerator(PcaMixtureConfig(latent_dimension=8, mixture_component_count=4)).fit(dataset)
    seeds = np.asarray([99, 100], dtype=np.uint64)
    first = model.generate_batch(dataset.conditions[24:26], seeds)
    repeated = model.generate_batch(dataset.conditions[24:26], seeds)
    assert np.array_equal(first.outputs, repeated.outputs)
    destination = tmp_path / "pca"
    model.save(destination)
    loaded = PcaMixtureGenerator.load(destination)
    restored = loaded.generate_batch(dataset.conditions[24:26], seeds)
    assert np.array_equal(first.outputs, restored.outputs)
    assert validate_baseline(loaded, dataset)["passed"]


def test_pca_mixture_supports_128_point_artifacts(tmp_path: Path) -> None:
    dataset = _dataset(position_count=128)
    model = PcaMixtureGenerator(PcaMixtureConfig(latent_dimension=12, mixture_component_count=4)).fit(dataset)
    destination = tmp_path / "pca-128"
    model.save(destination)
    manifest = json.loads((destination / "model.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["position_count"] == 128
    loaded = PcaMixtureGenerator.load(destination)
    generated = loaded.generate_batch(dataset.conditions[24:25], np.asarray([123], dtype=np.uint64))
    result = decode_output(generated.outputs[0], dataset.requests[24])
    assert len(result.samples) == 128
    report = validate_baseline(loaded, dataset)
    assert report["passed"]
    assert report["position_count"] == 128


def test_portable_pca_export_is_hashed_dynamic_and_exact_endpoint(tmp_path: Path) -> None:
    dataset = _dataset(position_count=128)
    source = tmp_path / "source"
    PcaMixtureGenerator(PcaMixtureConfig(latent_dimension=12, mixture_component_count=4)).fit(dataset).save(source)
    destination = tmp_path / "portable"
    manifest = export_pca_mixture(source, destination)
    assert manifest["position_count"] == 128
    assert manifest["output_size"] == 257
    runtime = PortablePcaRuntime(destination)
    first, distance, ood = runtime.generate_parameters(dataset.conditions[24], 123, exact_endpoint=True)
    repeated, repeated_distance, repeated_ood = runtime.generate_parameters(dataset.conditions[24], 123, exact_endpoint=True)
    assert np.array_equal(first, repeated)
    assert (distance, ood) == (repeated_distance, repeated_ood)
    assert np.array_equal(first[:2], np.zeros(2))
    assert np.allclose(first[-3:-1], (1.0, 0.0), atol=1e-14)
    points = first[:-1].reshape(-1, 2)
    raw, *_ = PortablePcaRuntime(destination).generate_parameters(dataset.conditions[24], 123)
    raw_points = raw[:-1].reshape(-1, 2)
    # A single similarity transform preserves all pairwise distance ratios.
    assert np.allclose(
        np.linalg.norm(points[1:] - points[:-1], axis=1) / np.linalg.norm(points[-1] - points[0]),
        np.linalg.norm(raw_points[1:] - raw_points[:-1], axis=1) / np.linalg.norm(raw_points[-1] - raw_points[0]),
    )
    if sys.platform == "win32":
        cli = Path(__file__).parents[1] / "build" / "native" / "Release" / "mousegen_pca_cli.exe"
        assert cli.is_file(), "build native targets before running PCA cross-runtime parity"
        request = dataset.requests[24]
        native = json.loads(subprocess.run([
            str(cli), str(destination), str(request.start_x), str(request.start_y),
            str(request.target_center_x), str(request.target_center_y), str(request.target_radius), "123",
            str(request.virtual_desktop_left), str(request.virtual_desktop_top),
            str(request.virtual_desktop_width), str(request.virtual_desktop_height), "exact",
        ], check=True, capture_output=True, text=True).stdout)
        distance_px = math.dist((request.start_x, request.start_y), (request.target_center_x, request.target_center_y))
        angle = math.atan2(request.target_center_y - request.start_y, request.target_center_x - request.start_x)
        cosine, sine = math.cos(angle), math.sin(angle)
        expected = []
        duration = round(math.exp(np.clip(first[-1], math.log(1e6), math.log(6e10))))
        for index, point in enumerate(points):
            x = request.start_x + distance_px * (cosine * point[0] - sine * point[1])
            y = request.start_y + distance_px * (sine * point[0] + cosine * point[1])
            expected.append((round(duration * index / 127), np.clip(x, 0, 1919), np.clip(y, 0, 1079)))
        expected[0] = (0, request.start_x, request.start_y)
        expected[-1] = (duration, request.target_center_x, request.target_center_y)
        assert len(native["points"]) == 128
        assert np.allclose(native["points"], expected, atol=1e-9)
    damaged = bytearray((destination / "pca.bin").read_bytes())
    damaged[-1] ^= 1
    (destination / "pca.bin").write_bytes(damaged)
    with pytest.raises(ValueError, match="hash changed"):
        PortablePcaRuntime(destination)


def test_zero_distance_request_returns_positive_settling_trajectory() -> None:
    request = GenerationRequest(10, 20, 10, 20, 5, 0, 0, 100, 100)
    output = np.zeros(129)
    output[-1] = math.log(10_000_000)
    result = decode_output(output, request)
    assert result.movement_duration_ns == 10_000_000
    assert all((sample.x, sample.y) == (10, 20) for sample in result.samples)


def test_small_conditional_flow_trains_checkpoints_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset()
    config = ConditionalFlowConfig(
        hidden_size=32, hidden_layers=2, epochs=20, batch_size=12, learning_rate=0.001,
        checkpoint_every=10, solver_steps=4,
    )
    model = ConditionalFlowGenerator(config).fit(dataset, checkpoint_directory=tmp_path / "checkpoints")
    assert (tmp_path / "checkpoints" / "epoch-0020.pt").is_file()
    assert min(item["validation_loss"] for item in model.history) < model.history[0]["validation_loss"]
    conditions = dataset.conditions[24:26]
    seeds = np.asarray([7, 8], dtype=np.uint64)
    first = model.generate_batch(conditions, seeds)
    assert np.array_equal(first.outputs, model.generate_batch(conditions, seeds).outputs)
    assert not np.array_equal(first.outputs[0], first.outputs[1])
    variants = model.generate_batch(np.repeat(conditions[:1], 32, axis=0), np.arange(32, dtype=np.uint64)).outputs
    lateral_midpoints = variants[:, 65]
    assert np.any(lateral_midpoints < 0) and np.any(lateral_midpoints > 0)
    destination = tmp_path / "flow"
    model.save(destination)
    loaded = ConditionalFlowGenerator.load(destination)
    assert np.allclose(first.outputs, loaded.generate_batch(conditions, seeds).outputs)
    assert validate_baseline(loaded, dataset)["passed"]
    export_conditional_flow(destination, tmp_path / "onnx")
    runtime = OnnxFlowRuntime(tmp_path / "onnx")
    normalized = (conditions[0] - loaded._condition_mean) / loaded._condition_scale
    source = np.random.default_rng(123).standard_normal(129)
    torch_output = loaded.integrate_source(normalized, source)
    onnx_output = runtime.integrate_source(normalized, source)
    assert np.max(np.abs(torch_output - onnx_output)) < 1e-4
    monkeypatch.setenv("MOUSE_MOTION_LAB_DATA_ROOT", str(tmp_path / "cli-data"))
    generated_json = tmp_path / "trajectory.json"
    assert cli_main([
        "generate", "--model", str(tmp_path / "onnx"), "--start", "400,300", "--target", "600,350",
        "--radius", "24", "--seed", "7", "--desktop", "0,0,1920,1080", "--output", str(generated_json),
    ]) == 0
    assert len(json.loads(generated_json.read_text(encoding="utf-8"))["samples"]) == 64
    if sys.platform == "win32":
        cli = Path(__file__).parents[1] / "build" / "native" / "Release" / "mousegen_cli.exe"
        assert cli.is_file(), "build native targets before running cross-runtime parity"
        request = replace(dataset.requests[24], random_seed=7)
        torch_parameters = loaded.generate_batch(condition_vector(request)[None, :], np.asarray([7], dtype=np.uint64))
        torch_result = decode_output(torch_parameters.outputs[0], request, condition_distance=float(torch_parameters.nearest_distances[0]))
        onnx_result = runtime.generate(request)
        native = json.loads(subprocess.run([
            str(cli), str(tmp_path / "onnx"), str(request.start_x), str(request.start_y),
            str(request.target_center_x), str(request.target_center_y), str(request.target_radius), "7",
            str(request.virtual_desktop_left), str(request.virtual_desktop_top),
            str(request.virtual_desktop_width), str(request.virtual_desktop_height),
        ], check=True, capture_output=True, text=True).stdout)
        assert len(native["points"]) == 64
        for torch_sample, onnx_sample, native_sample in zip(torch_result.samples, onnx_result.samples, native["points"], strict=True):
            assert abs(torch_sample.relative_time_ns - onnx_sample.relative_time_ns) <= 1
            assert abs(torch_sample.relative_time_ns - native_sample[0]) <= 1
            assert abs(torch_sample.x - onnx_sample.x) < 1e-3
            assert abs(torch_sample.y - onnx_sample.y) < 1e-3
            assert abs(torch_sample.x - native_sample[1]) < 1e-3
            assert abs(torch_sample.y - native_sample[2]) < 1e-3
        normalization = tmp_path / "onnx" / "normalization.bin"
        damaged = bytearray(normalization.read_bytes()); damaged[-1] ^= 1; normalization.write_bytes(damaged)
        rejected = subprocess.run([
            str(cli), str(tmp_path / "onnx"), "400", "300", "600", "350", "24", "7", "0", "0", "1920", "1080",
        ], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "hash changed" in rejected.stderr


def test_zero_condition_flow_uses_every_row_and_ignores_request_conditions(tmp_path: Path) -> None:
    dataset = _dataset()
    config = ConditionalFlowConfig(
        hidden_size=16, hidden_layers=1, epochs=2, batch_size=36, learning_rate=0.001,
        checkpoint_every=1, solver_steps=2, training_scope="all", validation_mode="none",
        condition_mode="zero",
    )
    model = ConditionalFlowGenerator(config).fit(dataset)
    assert model._training_conditions.shape == (36, 21)
    assert np.count_nonzero(model._training_conditions) == 0
    assert all("validation_loss" not in item for item in model.history)

    conditions = dataset.conditions[[0, 20]].copy()
    conditions[1, 3] = conditions[0, 3]
    generated = model.generate_batch(conditions, np.asarray([7, 7], dtype=np.uint64))
    assert np.array_equal(generated.outputs[0], generated.outputs[1])
    assert np.array_equal(generated.nearest_distances, np.zeros(2))

    destination = tmp_path / "zero-condition-flow"
    model.save(destination)
    loaded = ConditionalFlowGenerator.load(destination)
    assert loaded.config.condition_mode == "zero"
    assert np.array_equal(generated.outputs, loaded.generate_batch(conditions, np.asarray([7, 7], dtype=np.uint64)).outputs)
    manifest = export_conditional_flow(destination, tmp_path / "zero-condition-onnx")
    assert manifest["condition_mode"] == "zero"


def test_training_sample_flow_starts_from_seeded_recorded_path_and_exposes_trace(tmp_path: Path) -> None:
    dataset = _dataset()
    config = ConditionalFlowConfig(
        hidden_size=16, hidden_layers=1, epochs=2, batch_size=36, learning_rate=0.001,
        checkpoint_every=1, solver_steps=2, training_scope="all", validation_mode="none",
        condition_mode="zero", source_mode="training_sample",
    )
    model = ConditionalFlowGenerator(config).fit(dataset)
    condition = dataset.conditions[:1]
    seed = 7
    generated = model.generate_batch(condition, np.asarray([seed], dtype=np.uint64))
    expected_source_index = int(np.random.default_rng(seed).integers(len(dataset.outputs)))
    assert generated.source_indices.tolist() == [expected_source_index]

    source = model._source_outputs[expected_source_index]
    trace = model.integrate_source_trace(np.zeros(21), source)
    assert trace.shape == (config.solver_steps + 1, 129)
    assert np.allclose(trace[0], dataset.outputs[expected_source_index])
    assert np.allclose(generated.outputs[0], constrain_parameter_output(trace[-1], condition[0, 3]))

    destination = tmp_path / "training-sample-flow"
    model.save(destination)
    loaded = ConditionalFlowGenerator.load(destination)
    assert np.array_equal(
        generated.outputs, loaded.generate_batch(condition, np.asarray([seed], dtype=np.uint64)).outputs,
    )
    with pytest.raises(ValueError, match="does not yet support training-sample"):
        export_conditional_flow(destination, tmp_path / "unsupported-export")


def test_training_sample_source_can_amplify_canonical_vertical_axis() -> None:
    dataset = _dataset()
    config = ConditionalFlowConfig(
        hidden_size=16, hidden_layers=1, epochs=1, batch_size=36, checkpoint_every=1,
        solver_steps=2, training_scope="all", validation_mode="none", condition_mode="zero",
        source_mode="training_sample", source_vertical_scale=3.0,
    )
    model = ConditionalFlowGenerator(config).fit(dataset)
    source_index = int(np.random.default_rng(7).integers(len(dataset.outputs)))
    source = model.integrate_source_trace(np.zeros(21), model._source_outputs[source_index])[0]
    expected = dataset.outputs[source_index].copy()
    expected_positions = expected[:-1].reshape(64, 2)
    expected_positions[:, 1] *= 3
    assert np.allclose(source, expected)
    assert source[-1] == pytest.approx(dataset.outputs[source_index, -1])
    unamplified = ConditionalFlowGenerator(replace(config, source_vertical_scale=1.0)).fit(dataset)
    assert model.history[0]["training_loss"] != pytest.approx(unamplified.history[0]["training_loss"])
    with pytest.raises(ValueError, match="requires training-sample"):
        ConditionalFlowConfig(source_vertical_scale=3.0)
