from __future__ import annotations

import argparse
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toys.diffusion_vs_flow.run import (
    CONDITION_SIZE,
    OUTPUT_SIZE,
    ExperimentConfig,
    _training_batch_loss,
    edm_denoise,
    edm_preconditioning,
    expected_nfe,
    karras_schedule,
    make_network,
    make_split,
    parameter_count,
    parse_algorithm_list,
    sample_network,
    train_algorithm,
)


def test_edm_preconditioning_has_expected_limits_and_reconstruction() -> None:
    sigma = torch.tensor([[0.5], [2.0]], dtype=torch.float64)
    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, sigma_data=1.0)
    assert torch.allclose(c_skip, 1 / (sigma.square() + 1))
    assert torch.allclose(c_out, sigma / torch.sqrt(sigma.square() + 1))
    assert torch.allclose(c_in, torch.rsqrt(sigma.square() + 1))
    assert torch.allclose(c_noise, torch.log(sigma) / 4)

    class Zero(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(values), OUTPUT_SIZE), dtype=values.dtype)

    noisy = torch.arange(OUTPUT_SIZE * 2, dtype=torch.float64).reshape(2, OUTPUT_SIZE)
    conditions = torch.zeros((2, CONDITION_SIZE), dtype=torch.float64)
    assert torch.allclose(edm_denoise(Zero(), noisy, conditions, sigma), c_skip * noisy)


def test_both_losses_are_finite_and_parameter_counts_match() -> None:
    config = ExperimentConfig()
    target = torch.randn((8, OUTPUT_SIZE), generator=torch.Generator().manual_seed(1))
    condition = torch.zeros((8, CONDITION_SIZE))
    counts = []
    for algorithm in ("flow", "edm"):
        torch.manual_seed(10)
        network = make_network(config)
        counts.append(parameter_count(network))
        loss = _training_batch_loss(
            network, algorithm, target, condition, torch.Generator().manual_seed(2), config,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    assert counts[0] == counts[1]


def test_algorithms_receive_identical_deterministic_splits() -> None:
    flow_split = make_split(67, 42)
    edm_split = make_split(67, 42)
    assert flow_split.split_hash == edm_split.split_hash
    assert np.array_equal(flow_split.train_targets, edm_split.train_targets)
    assert np.array_equal(flow_split.validation_targets, edm_split.validation_targets)
    assert not np.shares_memory(flow_split.train_targets, flow_split.validation_targets)


@pytest.mark.parametrize("algorithm,expected", [("flow", 32), ("edm", 35)])
def test_sampling_is_deterministic_has_expected_shape_and_nfe(algorithm: str, expected: int) -> None:
    config = ExperimentConfig()
    torch.manual_seed(123)
    network = make_network(config)
    conditions = np.zeros((7, CONDITION_SIZE), dtype=np.float64)
    sources = np.random.default_rng(9).normal(size=(7, OUTPUT_SIZE)).astype(np.float32)
    first, first_nfe = sample_network(network, algorithm, conditions, sources, config)
    second, second_nfe = sample_network(network, algorithm, conditions, sources, config)
    assert first.shape == (7, OUTPUT_SIZE)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert first_nfe == second_nfe == expected_nfe(algorithm, config) == expected


def test_karras_schedule_has_configured_endpoints_and_zero_terminal() -> None:
    config = ExperimentConfig()
    schedule = karras_schedule(config, dtype=torch.float64)
    assert len(schedule) == config.edm_steps + 1
    assert schedule[0].item() == pytest.approx(config.edm_sigma_max)
    assert schedule[-2].item() == pytest.approx(config.edm_sigma_min)
    assert schedule[-1].item() == 0
    assert torch.all(schedule[:-2] > schedule[1:-1])


def test_training_and_fixed_validation_corruptions_are_reproducible() -> None:
    config = ExperimentConfig(
        hidden_size=8, hidden_layers=1, batch_size=8, parity_epochs=2,
        epoch_block=2, maximum_epochs=2, continuation_window=1,
    )
    split = make_split(10, 42)
    first_parity, first_converged, first_record = train_algorithm("edm", split, 42, config)
    second_parity, second_converged, second_record = train_algorithm("edm", split, 42, config)
    assert first_record["history"] == second_record["history"]
    for first, second in ((first_parity, second_parity), (first_converged, second_converged)):
        assert first.keys() == second.keys()
        assert all(torch.equal(first[name], second[name]) for name in first)


def test_algorithm_list_parser_accepts_selected_arms_and_rejects_duplicates() -> None:
    assert parse_algorithm_list("flow") == ("flow",)
    assert parse_algorithm_list("edm,flow") == ("edm", "flow")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_algorithm_list("flow,flow")
