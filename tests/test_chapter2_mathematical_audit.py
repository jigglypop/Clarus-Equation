from __future__ import annotations

import math

import pytest

from examples.physics.chapter2_mathematical_audit import (
    build_audit,
    canonical_chain,
    critical_raw_depth,
    low_poisson_extinction,
    regularized_fixed_point_potential,
    track_b_positive_roots,
    two_channel_path_kernel,
    validate,
)


def test_chapter2_counterexample_ledger_is_reproducible() -> None:
    audit = build_audit()
    validate(audit)

    assert math.isclose(audit.poisson_q_at_two, 0.20318786998, abs_tol=5e-12)
    assert audit.reducible_extinction_vector == (audit.poisson_q_at_two, 1.0)
    assert audit.canonical_x_endpoint_half_span > 0.00001
    assert audit.two_channel_paths_preserved
    assert abs(audit.two_channel_total_mass_residual) < 1e-15
    assert abs(audit.two_channel_self_consistency_residual) < 1e-15
    assert math.isclose(
        audit.two_channel_surviving_fraction,
        audit.canonical_x,
        abs_tol=1e-15,
    )
    assert abs(audit.conditional_flat_energy_identity_residual) < 1e-15
    assert math.isfinite(audit.regularized_d0_potential)
    assert abs(audit.regularized_fixed_point_gradient) < 1e-15
    assert audit.sample_lyapunov_rate < 0.0
    assert audit.hodge_bivector_vector_dimensions == (3,)
    assert abs(audit.ger_response_reparameterization_residual) < 1e-15
    assert abs(audit.ger_complement_log_action_residual) < 1e-15
    assert abs(audit.ger_weight_reparameterization_residual) < 1e-15
    assert abs(audit.koide_geometry_residual) < 1e-15
    assert audit.tensor_ratio_relative_gap > 0.15
    assert audit.portal_to_quartic_ratio > 47_000.0
    expected_slope_ratio = (
        2.0
        * 11.0974588093**2
        * (3.0 + 0.4904868132 * 11.0974588093**2)
        / 1.3434991214e-10
    )
    assert math.isclose(audit.einstein_slope_ratio_per_c6, expected_slope_ratio)
    assert math.isclose(
        audit.c6_unit_slope_tolerance_bound,
        1.0 / expected_slope_ratio,
    )


def test_pi_candidates_are_not_exact_physical_constants() -> None:
    audit = build_audit()

    assert abs(audit.inverse_alpha_pull_sigma) > 14_000.0
    assert abs(audit.proton_electron_pull_sigma) > 1_000_000.0


def test_track_b_has_exactly_two_positive_convex_branches() -> None:
    roots = track_b_positive_roots(1.0 / 127.95)

    assert len(roots) == 2
    assert math.isclose(roots[0], 0.0528678687, abs_tol=5e-11)
    assert math.isclose(roots[1], 0.1173186647, abs_tol=5e-11)
    assert roots[1] < 1.0 / (2.0 * math.sqrt(2.0))
    with pytest.raises(ValueError):
        track_b_positive_roots(0.0)


@pytest.mark.parametrize("bad_mean", [-1.0, math.nan, math.inf])
def test_extinction_rejects_invalid_mean_offspring(bad_mean: float) -> None:
    with pytest.raises(ValueError):
        low_poisson_extinction(bad_mean)


def test_extinction_solver_is_stable_near_criticality_and_at_large_mean() -> None:
    near_mean = 1.0 + 1e-12
    near_q = low_poisson_extinction(near_mean)
    near_survival = 1.0 - near_q
    assert 0.0 < near_survival < 1e-10
    assert abs(math.log1p(-near_survival) + near_mean * near_survival) < 1e-22

    adjacent_mean = math.nextafter(1.0, 2.0)
    adjacent_survival = 1.0 - low_poisson_extinction(adjacent_mean)
    assert adjacent_survival > 0.0
    assert math.isclose(
        adjacent_survival,
        2.0 * (adjacent_mean - 1.0),
        rel_tol=1e-12,
    )

    q_700 = low_poisson_extinction(700.0)
    assert math.isclose(q_700, math.exp(-700.0), rel_tol=1e-12)
    assert low_poisson_extinction(1000.0) == 0.0


@pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, math.nan, math.inf, 1.0])
def test_registered_chain_rejects_inputs_outside_its_domain(bad_alpha: float) -> None:
    with pytest.raises(ValueError):
        canonical_chain(bad_alpha)


@pytest.mark.parametrize(
    ("measure", "depth"),
    [
        ({"a": 0.0}, 1.0),
        ({"a": math.nan}, 1.0),
        ({"a": math.inf}, 1.0),
        ({"a": 1.0}, math.nan),
        ({"a": 1.0}, math.inf),
        ({"a": 1.0}, -1.0),
    ],
)
def test_path_kernel_rejects_nonfinite_or_zero_measure(
    measure: dict[str, float], depth: float
) -> None:
    with pytest.raises(ValueError):
        two_channel_path_kernel(measure, effective_depth=depth, fixed_point=0.5)


def test_operational_depth_and_potential_domains_are_explicit() -> None:
    assert critical_raw_depth(0.5) == 2.0
    with pytest.raises(ValueError):
        critical_raw_depth(0.0)
    with pytest.raises(ValueError):
        regularized_fixed_point_potential(math.nan, 0.5)
