from __future__ import annotations

import math

import pytest

from examples.physics.causal_light_geometry_audit import (
    conformal_counterexample,
    counting_volume_audit,
    estimate_myrheim_meyer_dimension,
    expected_ordering_fraction,
    expected_relation_density,
    lattice_directional_split,
    massive_carrier_speed_ratio,
    ordering_fraction,
    sprinkle_minkowski_diamond,
    square_lattice_angular_frequency,
)


def test_conformal_null_order_does_not_fix_volume_or_curvature() -> None:
    result = conformal_counterexample()

    assert result.causal_order_identical
    assert math.isclose(result.minkowski_normalized_four_volume, 1.0)
    assert math.isclose(result.de_sitter_normalized_four_volume, 7.0 / 24.0)
    assert result.minkowski_normalized_ricci_scalar == 0.0
    assert result.de_sitter_normalized_ricci_scalar == 12.0


def test_ordering_fraction_and_relation_density_conventions_are_distinct() -> None:
    assert math.isclose(expected_ordering_fraction(2.0), 0.5)
    assert math.isclose(expected_ordering_fraction(4.0), 0.1)
    assert math.isclose(expected_relation_density(4.0), 0.05)
    assert math.isclose(estimate_myrheim_meyer_dimension(0.1), 4.0)


def test_calibrated_event_counts_recover_the_missing_volume_ratio() -> None:
    result = counting_volume_audit(seed=20260828)

    assert abs(result.recovered_volume_ratio - result.expected_volume_ratio) < 0.01


def test_single_seed_has_no_dimension_estimator() -> None:
    with pytest.raises(ValueError, match="at least two events"):
        ordering_fraction([(0.0, (0.0, 0.0, 0.0))])


def test_manifoldlike_four_dimensional_sprinkling_recovers_dimension() -> None:
    events = sprinkle_minkowski_diamond(4, 800, seed=20260828)
    observed = ordering_fraction(events)
    estimated = estimate_myrheim_meyer_dimension(observed)

    assert abs(observed - 0.1) < 0.015
    assert abs(estimated - 4.0) < 0.25


def test_lattice_maximum_speed_does_not_guarantee_exact_lorentz_symmetry() -> None:
    low_k = 0.01
    low_frequency = square_lattice_angular_frequency((low_k, 0.0))

    assert math.isclose(low_frequency / low_k, 1.0, rel_tol=5.0e-6)
    assert lattice_directional_split(low_k) < 2.2e-6
    assert lattice_directional_split(1.5) > 0.02


def test_null_frontier_does_not_force_every_record_to_move_at_c() -> None:
    assert massive_carrier_speed_ratio(0.0) == 0.0
    assert math.isclose(massive_carrier_speed_ratio(1.0), 1.0 / math.sqrt(2.0))
    assert massive_carrier_speed_ratio(100.0) < 1.0

    with pytest.raises(ValueError, match="finite and non-negative"):
        massive_carrier_speed_ratio(-1.0)
