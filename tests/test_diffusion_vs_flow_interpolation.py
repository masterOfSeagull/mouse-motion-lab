from __future__ import annotations

import numpy as np

from toys.diffusion_vs_flow.run import BASIS
from toys.diffusion_vs_flow.run_interpolation import interpolation_metrics


def test_interpolation_metrics_separate_copies_midpoints_and_off_line_error() -> None:
    training_c = np.linspace(-2, 2, 32)
    training = training_c[:, None] * BASIS[None, :]
    midpoint_c = (training_c[:-1] + training_c[1:]) / 2
    midpoint = midpoint_c[:, None] * BASIS[None, :]
    metrics, values = interpolation_metrics(midpoint, training, midpoint_c)
    assert np.allclose(values["projected_normalized"], 1)
    assert np.allclose(values["residual_norm"], 0, atol=1e-12)
    assert metrics["consistent_inbetween_rate_r0.25"] == 1
    assert metrics["consistent_copy_like_rate_r0.25"] == 0

    metrics, values = interpolation_metrics(training, training, training_c)
    assert np.allclose(values["full_distance"], 0, atol=1e-12)
    assert metrics["consistent_inbetween_rate_r0.25"] == 0
    assert metrics["consistent_copy_like_rate_r0.25"] == 1
