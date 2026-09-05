from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal.causal_light_geometry import (
    P0,
    canonical_section,
    certificate,
    conformal_counterexample,
    conformal_sign,
    counting_volume_audit,
    density,
    estimate_myrheim_meyer_dimension,
    expected_ordering_fraction,
    expected_relation_density,
    lattice_directional_split,
    luders_zero,
    massive_carrier_speed_ratio,
    ordering_fraction,
    posterior_zero,
    quotient_coordinate,
    sprinkle_minkowski_diamond,
    square_lattice_angular_frequency,
    volume_recovery,
    z2_cycle,
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


def test_rank_one_readout_has_exact_same_record_and_distinct_priors() -> None:
    result = certificate()
    assert result.identical_subnormalised_readouts
    assert result.identical_posteriors
    assert result.distinct_priors
    assert all(math.isclose(sum(values), 1.0) and min(values) >= 0.0 for values in result.prior_eigenvalues)
    assert np.allclose(luders_zero(density(result.p, result.coherences[2])), result.p * P0)
    assert np.allclose(posterior_zero(density(result.p, result.coherences[1])), P0)


def test_section_roundtrip_and_conditional_quotient_theorem_certificate() -> None:
    result = certificate()
    state = canonical_section(result.p)
    assert result.section_roundtrip
    assert quotient_coordinate(state) == result.p
    assert all(result.quotient_homeomorphism_conditions.values())
    assert result.controls["posterior_sample_satisfies_p_ge_epsilon"]
    assert not result.status["closed_posterior_domain_constructed"]


def test_existing_conformal_counterexample_and_supplied_scale_law() -> None:
    result = certificate(omega=2.0, n=4)
    conformal = result.conformal
    assert conformal["existing_counterexample_causal_order_identical"]
    assert conformal["existing_minkowski_volume"] == 1.0
    assert math.isclose(conformal["existing_de_sitter_volume"], 7.0 / 24.0)
    assert conformal["existing_de_sitter_ricci"] == 12.0
    assert conformal["volume_ratio"] == 16.0
    assert conformal["recovered_Omega"] == 2.0
    assert conformal["causal_signs_unchanged"]


def test_conformal_signs_and_metric_connection_nonidentifiability() -> None:
    assert conformal_sign((2.0, 1.0), 3.0) < 0.0
    assert conformal_sign((1.0, 1.0), 3.0) == 0.0
    assert conformal_sign((1.0, 2.0), 3.0) > 0.0
    hidden = certificate().z2_hidden_connection
    assert hidden["plus"]["holonomy"] == 1
    assert hidden["minus"]["holonomy"] == -1
    assert hidden["fixed_holonomy_different_perimeter"]


def test_fail_closed_inputs_and_controls() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        density(0.4, 1.0)
    with pytest.raises(ValueError, match="epsilon"):
        certificate(epsilon=0.5)
    with pytest.raises(ValueError, match="Omega"):
        certificate(omega=0.0)
    with pytest.raises(ValueError, match="at least two"):
        volume_recovery(1.0, n=1)
    with pytest.raises(ValueError, match="integer spacetime"):
        volume_recovery(1.0, n=4.0)
    with pytest.raises(ValueError, match="integer spacetime"):
        certificate(n=True)
    with pytest.raises(ValueError, match="interval components"):
        conformal_sign((float("nan"), 1.0), 2.0)
    with pytest.raises(ValueError, match="Z2"):
        z2_cycle((1, 0, 1, 1), (1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        z2_cycle((1, 1, 1, 1), (1.0, 0.0, 1.0, 1.0))


def test_status_ceiling_and_accounting_absence() -> None:
    result = certificate()
    assert not result.status["full_map_injective"]
    assert result.status["induced_quotient_homeomorphism_conditional"]
    assert not result.status["homeomorphism_determines_metric"]
    assert result.status["same_causal_order_different_full_metric_witness"]
    assert not result.status["continuum_causal_order_to_conformal_theorem_proved"]
    assert not result.status["distinguishing_continuum_assumptions_supplied"]
    assert result.status["volume_scale_recovered_for_supplied_toy"]
    assert not result.status["gr_lensing_backreaction_derived"]
    assert not result.status["success_gates_5_to_8_complete"]
    assert result.z2_hidden_connection["supplied_regular_bundle_control"]
    assert not result.z2_hidden_connection["instrument_connection_derived"]
    assert not result.status["instrument_fibers_global_bundle_derived"]
    assert not result.status["quotient_smooth_manifold_derived"]
    assert not result.status["metric_tensor_pullback_derived"]
    assert not result.accounting["rn_weighting_used"]
    assert not result.accounting["energy_or_stress_accounting_present"]
