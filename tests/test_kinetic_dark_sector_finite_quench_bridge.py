"""Focused invariants for the analytic finite quench bridge."""

from __future__ import annotations

import math

import pytest

from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
    compact_c1_bump,
    compact_c1_bump_derivative,
    compact_c1_cumulative,
)


def _bridge(**overrides: float) -> FiniteQuenchBridge:
    values = dict(
        n_star=-4.0,
        half_width=0.5,
        omega_prod0=0.12,
        reservoir_present_density=0.21,
        w_reservoir=0.0,
        w_open=2.1767e-4,
    )
    values.update(overrides)
    return FiniteQuenchBridge(FiniteQuenchBridgeConfig(**values))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_star", math.nan),
        ("half_width", 0.0),
        ("half_width", -0.1),
        ("n_star", 0.0),
        ("omega_prod0", -1.0),
        ("reservoir_present_density", -1.0),
        ("w_reservoir", -1.001),
        ("w_open", -1.0),
    ],
)
def test_config_invalid_inputs_fail_closed(field: str, value: float) -> None:
    values = dict(n_star=-2.0, half_width=0.2, omega_prod0=0.1, reservoir_present_density=0.2, w_reservoir=0.0, w_open=0.0)
    values[field] = value
    with pytest.raises(ValueError):
        FiniteQuenchBridgeConfig(**values)


@pytest.mark.parametrize("value", [None, "-2", True, math.inf])
def test_generic_inputs_fail_closed(value: object) -> None:
    bridge = _bridge()
    with pytest.raises(ValueError):
        bridge.source(value)
    with pytest.raises(ValueError):
        compact_c1_bump(-4.0, value, 0.5)


def test_compact_c1_bump_support_normalization_and_endpoint_slope() -> None:
    center, width = -4.0, 0.5
    left, right = center - width, center + width
    assert compact_c1_bump(left, center, width) == 0.0
    assert compact_c1_bump(right, center, width) == 0.0
    assert compact_c1_bump(center - 2.0 * width, center, width) == 0.0
    assert compact_c1_cumulative(left, center, width) == 0.0
    assert compact_c1_cumulative(right, center, width) == 1.0
    # Composite Simpson quadrature checks the bump independently of cumulative().
    steps = 4096
    spacing = (right - left) / steps
    samples = [
        compact_c1_bump(left + index * spacing, center, width)
        for index in range(steps + 1)
    ]
    quadrature = (spacing / 3.0) * math.fsum(
        (
            samples[0],
            samples[-1],
            4.0 * math.fsum(samples[1:-1:2]),
            2.0 * math.fsum(samples[2:-1:2]),
        )
    )
    assert quadrature == pytest.approx(1.0, abs=2.0e-13)
    eps = 1.0e-6
    left_slope = (compact_c1_bump(left + eps, center, width) - 0.0) / eps
    right_slope = (0.0 - compact_c1_bump(right - eps, center, width)) / eps
    assert abs(left_slope) < 1.0e-4
    assert abs(right_slope) < 1.0e-4


def test_pre_during_post_production_behavior_and_exact_present_abundance() -> None:
    bridge = _bridge()
    c = bridge.config
    assert bridge.production_density(c.n_minus - 0.1) == 0.0
    assert bridge.production_density(c.n_star) > 0.0
    n_post = c.n_plus + 0.1
    assert bridge.production_density(n_post) == pytest.approx(c.omega_prod0 * math.exp(-3.0 * n_post))
    assert bridge.production_density(0.0) == c.omega_prod0
    assert bridge.certificate().present_abundance_residual == 0.0


def test_analytic_derivatives_and_total_source_cancellation() -> None:
    bridge = _bridge(w_reservoir=0.2)
    for n in (-4.35, -4.0, -3.65, 0.0):
        production_scale = max(
            1.0,
            abs(bridge.production_derivative(n)),
            abs(3.0 * bridge.production_density(n)),
            abs(bridge.source(n)),
        )
        assert (
            abs(bridge.production_continuity_residual(n)) / production_scale
            < 2.0e-15
        )
        assert bridge.total_continuity_relative_residual(n) < 2.0e-15


def test_reservoir_pays_is_positive_and_has_present_minimum() -> None:
    bridge = _bridge(w_reservoir=0.0)
    c = bridge.config
    before = bridge.reservoir_density(c.n_minus)
    after = bridge.reservoir_density(c.n_plus)
    assert before > after >= c.reservoir_present_density
    certificate = bridge.certificate()
    assert certificate.early_reservoir_density == pytest.approx(before)
    assert certificate.min_reservoir_density == c.reservoir_present_density
    assert certificate.present_reservoir_density == c.reservoir_present_density


def test_cold_bound_and_zero_abundance_identity() -> None:
    bridge = _bridge()
    assert bridge.cold_density_error_bound() == pytest.approx(1.0 - math.exp(-1.5 * 2.1767e-4))
    empty = _bridge(omega_prod0=0.0)
    for n in (empty.config.n_initial, -4.0, 0.0):
        assert empty.source(n) == 0.0
        assert empty.production_density(n) == 0.0
        expected = empty.config.reservoir_present_density * math.exp(-3.0 * (1.0 + empty.config.w_reservoir) * n)
        assert empty.reservoir_density(n) == pytest.approx(expected)


def test_compact_zero_avoids_outside_support_overflow() -> None:
    bridge = _bridge()
    empty = _bridge(omega_prod0=0.0)
    assert bridge.source(-1000.0) == 0.0
    assert bridge.production_density(-1000.0) == 0.0
    assert empty.source(-1000.0) == 0.0
    assert empty.production_density(-1000.0) == 0.0
    with pytest.raises(ValueError):
        bridge.reservoir_density(bridge.config.n_initial - 1.0)


def test_cold_bound_requires_declared_nonrelativistic_envelope() -> None:
    values = dict(
        n_star=-4.0,
        half_width=0.5,
        omega_prod0=0.12,
        reservoir_present_density=0.21,
        w_reservoir=0.0,
        w_open=2.1767e-4,
        cold_envelope="constant_w",
    )
    with pytest.raises(ValueError):
        FiniteQuenchBridgeConfig(**values)


def test_weighted_source_integral_avoids_large_exponent_cancellation() -> None:
    bridge = FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-1.0,
            half_width=1.0,
            omega_prod0=0.1,
            reservoir_present_density=0.2,
            w_reservoir=1000.0,
            w_open=0.0,
            n_initial=-2.0,
        )
    )
    rate = 3000.0
    expected = (
        0.1
        * (15.0 / 16.0)
        * (8.0 / rate**3 - 24.0 / rate**4 + 24.0 / rate**5)
    )
    weighted = bridge._weighted_source_integral(-2.0, 0.0)
    assert math.isfinite(weighted)
    assert weighted == pytest.approx(expected, rel=2.0e-15)


def test_analytic_derivatives_crosscheck_centered_finite_differences() -> None:
    bridge = _bridge(w_reservoir=0.15)
    h = 1.0e-5
    for n in (-4.25, -4.0, -3.75):
        production_fd = (bridge.production_density(n + h) - bridge.production_density(n - h)) / (2.0 * h)
        reservoir_fd = (bridge.reservoir_density(n + h) - bridge.reservoir_density(n - h)) / (2.0 * h)
        assert production_fd == pytest.approx(bridge.production_derivative(n), rel=2.0e-9, abs=2.0e-10)
        assert reservoir_fd == pytest.approx(bridge.reservoir_derivative(n), rel=2.0e-9, abs=2.0e-10)


def test_bump_and_source_derivatives_crosscheck_finite_differences() -> None:
    bridge = _bridge()
    center = bridge.config.n_star
    width = bridge.config.half_width
    h = 1.0e-6
    for n in (-4.25, -4.0, -3.75):
        bump_fd = (
            compact_c1_bump(n + h, center, width)
            - compact_c1_bump(n - h, center, width)
        ) / (2.0 * h)
        source_fd = (
            bridge.source(n + h) - bridge.source(n - h)
        ) / (2.0 * h)
        assert bump_fd == pytest.approx(
            compact_c1_bump_derivative(n, center, width),
            rel=3.0e-9,
            abs=3.0e-9,
        )
        assert source_fd == pytest.approx(
            bridge.source_derivative(n),
            rel=3.0e-9,
            abs=3.0e-7,
        )
    for endpoint in (bridge.config.n_minus, bridge.config.n_plus):
        assert compact_c1_bump_derivative(endpoint, center, width) == 0.0
        assert bridge.source_derivative(endpoint) == 0.0
