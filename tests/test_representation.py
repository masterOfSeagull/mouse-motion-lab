from __future__ import annotations

import math

import pytest

from mouselearn.representation.canonical import CanonicalTransform, decode_endpoint, encode_endpoint
from mouselearn.representation.spline import SplineSpec, fit_clamped_spline
from mouselearn.representation.timing import decode_timing_logits, fit_timing_representation


def test_canonical_round_trip_and_endpoint_disk_mapping() -> None:
    transform = CanonicalTransform.from_start_target((100.0, 50.0), (400.0, 250.0), 30.0)
    point = (265.5, 132.25)
    assert transform.inverse(transform.forward(point)) == pytest.approx(point)
    assert transform.forward(transform.target_center) == pytest.approx((1.0, 0.0))
    latent = encode_endpoint((0.3, -0.4))
    assert decode_endpoint(latent) == pytest.approx((0.3, -0.4))
    assert math.hypot(*decode_endpoint((100.0, 100.0))) < 1.0


def test_clamped_spline_reconstructs_a_smooth_trajectory() -> None:
    points = [(index / 30, 0.12 * math.sin(index / 30 * math.pi)) for index in range(31)]
    fit = fit_clamped_spline(points, SplineSpec(control_point_count=16, smoothing=1e-5))
    reconstructed = [fit.evaluate(parameter) for parameter in fit.parameters]
    errors = [math.dist(actual, predicted) for actual, predicted in zip(points, reconstructed, strict=True)]
    assert fit.control_points[0] == pytest.approx(points[0])
    assert fit.control_points[-1] == pytest.approx(points[-1])
    assert max(errors) < 0.02
    assert len(fit.residual_controls()) == 14


def test_timing_logits_decode_to_monotonic_progress() -> None:
    timestamps = [index * 10_000_000 for index in range(13)]
    points = [(float(index), float(index * index) / 12) for index in range(13)]
    representation = fit_timing_representation(timestamps, points)
    decoded = decode_timing_logits(representation.interval_logits)
    assert len(decoded) == 12
    assert decoded[-1] == 1.0
    assert all(right > left for left, right in zip(decoded, decoded[1:], strict=False))
