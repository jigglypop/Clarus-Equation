"""Focused tests for the exact two-variable reduced ODE closure."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_reduced_ode_closure import (
    FiniteQuenchReducedODEClosure,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(
    *,
    omega: float = 0.12,
    reservoir: float = 0.21,
    w_reservoir: float = 0.1,
) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=omega,
            reservoir_present_density=reservoir,
            w_reservoir=w_reservoir,
            w_open=2.1767e-4,
        )
    )


def _gate(bridge: FiniteQuenchBridge | None = None):
    if bridge is None:
        bridge = _bridge()
    return bridge, FiniteQuenchReducedODEClosure(
        bridge,
        n_reference=-4.0,
        kappa_reference=2.0,
    )


def _construct(*, n: float = -4.0, clock: float = 0.25, total_u: float = 0.005):
    bridge, gate = _gate()
    return bridge, gate, gate.construct(
        n=n,
        scalar_clock_shift=clock,
        total_momentum_density=total_u,
    )


def test_reduced_rhs_matches_full_chain_and_reconstructs_effective_solution() -> None:
    _, _, receipt = _construct()
    assert receipt.reference_kappa_state_matches
    assert receipt.parent_second_tangent_holds
    assert receipt.parent_constraint_propagation_holds
    assert receipt.parent_algebraic_metric_tangent_holds
    assert receipt.reduced_clock_rhs_matches_full_system
    assert receipt.reduced_momentum_rhs_matches_full_system
    assert receipt.pointwise_effective_force_closure_holds
    assert receipt.conditional_global_reduced_solution_exists_uniquely
    assert receipt.conditional_effective_full_reconstruction_holds
    assert not receipt.failure_reasons


def test_matrix_coefficients_are_the_derived_common_clock_system() -> None:
    _, _, receipt = _construct()
    node = receipt.common_clock_second_tangent.common_clock_tangent.gr_linear_node
    background = node.background
    coupling = background.gravity_constraint_coupling
    enthalpy = background.total_enthalpy
    h = background.hubble_log_derivative
    kappa_squared = receipt.k_over_a_h**2
    expected = (
        3.0 * coupling * enthalpy / kappa_squared,
        coupling
        + 3.0 * coupling / kappa_squared
        - kappa_squared / (3.0 * enthalpy),
        -3.0 * coupling * enthalpy**2 / kappa_squared
        - _bridge().config.w_reservoir
        * background.reservoir_density_derivative,
        -(3.0 - h) - 3.0 * coupling * enthalpy / kappa_squared,
    )
    assert receipt.matrix_a11 == pytest.approx(expected[0])
    assert receipt.matrix_a12 == pytest.approx(expected[1])
    assert receipt.matrix_a21 == pytest.approx(expected[2])
    assert receipt.matrix_a22 == pytest.approx(expected[3])


def test_matrix_multiplication_reproduces_both_derivatives() -> None:
    _, _, receipt = _construct()
    expected_clock = (
        receipt.matrix_a11 * receipt.scalar_clock_shift
        + receipt.matrix_a12 * receipt.total_momentum_density
    )
    expected_u = (
        receipt.matrix_a21 * receipt.scalar_clock_shift
        + receipt.matrix_a22 * receipt.total_momentum_density
    )
    assert receipt.reduced_clock_log_derivative == pytest.approx(expected_clock)
    assert receipt.reduced_total_momentum_density_derivative == pytest.approx(
        expected_u
    )
    assert receipt.full_clock_log_derivative == pytest.approx(expected_clock)
    assert receipt.full_total_momentum_density_derivative == pytest.approx(
        expected_u
    )


def test_kappa_reference_evolution_has_correct_log_derivative() -> None:
    bridge, gate = _gate()
    for n in (-5.0, -4.0, -3.0):
        epsilon = 1.0e-6
        numerical = (
            math.log(gate.k_over_a_h(n + epsilon))
            - math.log(gate.k_over_a_h(n - epsilon))
        ) / (2.0 * epsilon)
        background = gate.construct(
            n=n,
            scalar_clock_shift=0.0,
            total_momentum_density=0.0,
        ).common_clock_second_tangent.common_clock_tangent.gr_linear_node.background
        assert numerical == pytest.approx(
            -1.0 - background.hubble_log_derivative,
            rel=2.0e-8,
            abs=1.0e-9,
        )
    assert gate.k_over_a_h(-4.0) == pytest.approx(2.0)
    assert bridge.config.n_initial < -5.0


def test_domain_receipt_proves_positive_bounds_and_continuous_linear_system() -> None:
    _, gate = _gate()
    domain = gate.domain_receipt()
    assert domain.present_total_density_lower_bound == pytest.approx(0.33)
    assert domain.present_total_enthalpy_lower_bound == pytest.approx(0.33)
    assert domain.kappa_positive_lower_bound > 0.0
    assert domain.kappa_at_n_final >= domain.kappa_at_n_initial
    assert domain.hubble_log_derivative_lower_bound == -3.0
    assert domain.hubble_log_derivative_upper_bound == -1.5
    assert domain.kappa_log_derivative_lower_bound == 0.5
    assert domain.kappa_log_derivative_upper_bound == 2.0
    assert domain.source_endpoint_values == (0.0, 0.0)
    assert domain.source_endpoint_derivatives == (0.0, 0.0)
    assert domain.source_regularity_derived_from_piecewise_analytic_matching
    assert domain.background_regularity_derived_from_continuity_odes
    assert domain.compact_source_is_c1
    assert domain.background_is_at_least_c2
    assert domain.reduced_matrix_is_continuous
    assert domain.effective_force_closure_is_continuous
    assert domain.all_denominators_have_positive_domain_bounds
    assert domain.global_reduced_linear_ode_existence_uniqueness_proven


def test_initial_source_edges_center_and_today_share_one_regular_vector_field() -> None:
    bridge, gate = _gate()
    states = (
        (bridge.config.n_initial, 1.0e-5, 1.0e-7),
        (-4.5, 0.01, 0.001),
        (-4.0, 0.25, 0.005),
        (-3.5, 0.01, 0.001),
        (0.0, 0.01, 0.001),
    )
    for n, clock, total_u in states:
        receipt = gate.construct(
            n=n,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
        )
        assert receipt.conditional_effective_full_reconstruction_holds


@pytest.mark.parametrize("w_reservoir", [0.0, 1.0])
def test_causal_barotrope_boundaries_and_reservoir_only_branch_are_regular(
    w_reservoir: float,
) -> None:
    bridge = _bridge(
        omega=0.0,
        reservoir=0.21,
        w_reservoir=w_reservoir,
    )
    _, gate = _gate(bridge)
    domain = gate.domain_receipt()
    receipt = gate.construct(
        n=-4.0,
        scalar_clock_shift=0.01,
        total_momentum_density=0.001,
    )
    assert domain.all_denominators_have_positive_domain_bounds
    assert domain.global_reduced_linear_ode_existence_uniqueness_proven
    assert receipt.conditional_effective_full_reconstruction_holds


@pytest.mark.parametrize("n_reference", [-5.0, -4.0, -2.0])
def test_multiple_reference_nodes_define_the_same_regular_kind_of_domain(
    n_reference: float,
) -> None:
    bridge = _bridge()
    gate = FiniteQuenchReducedODEClosure(
        bridge,
        n_reference=n_reference,
        kappa_reference=1.3,
    )
    assert gate.k_over_a_h(n_reference) == pytest.approx(1.3)
    assert gate.domain_receipt().global_reduced_linear_ode_existence_uniqueness_proven
    assert gate.construct(
        n=n_reference,
        scalar_clock_shift=0.01,
        total_momentum_density=0.001,
    ).conditional_effective_full_reconstruction_holds


def test_reduced_system_and_effective_force_are_homogeneous_linear() -> None:
    _, gate = _gate()
    base = gate.construct(
        n=-4.0,
        scalar_clock_shift=0.13,
        total_momentum_density=-0.004,
    )
    factor = -2.5
    scaled = gate.construct(
        n=-4.0,
        scalar_clock_shift=factor * 0.13,
        total_momentum_density=factor * -0.004,
    )
    assert scaled.reduced_clock_log_derivative == pytest.approx(
        factor * base.reduced_clock_log_derivative
    )
    assert scaled.reduced_total_momentum_density_derivative == pytest.approx(
        factor * base.reduced_total_momentum_density_derivative
    )
    assert scaled.required_effective_produced_intrinsic_force == pytest.approx(
        factor * base.required_effective_produced_intrinsic_force
    )


def test_zero_reduced_state_reconstructs_zero_perturbations_and_force() -> None:
    _, gate = _gate()
    receipt = gate.construct(
        n=-4.0,
        scalar_clock_shift=0.0,
        total_momentum_density=0.0,
    )
    assert receipt.reduced_clock_log_derivative == pytest.approx(0.0)
    assert receipt.reduced_total_momentum_density_derivative == pytest.approx(0.0)
    assert receipt.required_effective_produced_intrinsic_force == pytest.approx(0.0)
    assert receipt.conditional_effective_full_reconstruction_holds


def test_wrong_reduced_rhs_candidates_are_falsified() -> None:
    _, gate, receipt = _construct()
    bad_clock = gate.audit(
        algebraic_metric_tangent=receipt.algebraic_metric_tangent,
        scalar_clock_log_derivative=receipt.full_clock_log_derivative + 0.5,
        total_momentum_density_derivative=(
            receipt.full_total_momentum_density_derivative
        ),
    )
    assert not bad_clock.reduced_clock_rhs_matches_full_system
    assert not bad_clock.conditional_effective_full_reconstruction_holds
    bad_u = gate.audit(
        algebraic_metric_tangent=receipt.algebraic_metric_tangent,
        scalar_clock_log_derivative=receipt.full_clock_log_derivative,
        total_momentum_density_derivative=(
            receipt.full_total_momentum_density_derivative + 0.5
        ),
    )
    assert not bad_u.reduced_momentum_rhs_matches_full_system


def test_forged_metric_parent_boolean_is_raw_recomputed() -> None:
    _, gate, receipt = _construct()
    metric = receipt.algebraic_metric_tangent
    forged = replace(
        metric,
        provided_curvature_potential_log_derivative=(
            metric.provided_curvature_potential_log_derivative + 0.25
        ),
        algebraic_curvature_first_tangent_holds=True,
        local_algebraic_metric_second_tangent_holds=True,
    )
    bad = gate.audit(
        algebraic_metric_tangent=forged,
        scalar_clock_log_derivative=receipt.full_clock_log_derivative,
        total_momentum_density_derivative=(
            receipt.full_total_momentum_density_derivative
        ),
    )
    assert not bad.parent_algebraic_metric_tangent_holds
    assert not bad.conditional_effective_full_reconstruction_holds


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0"])
def test_nonfinite_or_nonreal_reference_or_state_is_rejected(bad) -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="finite real"):
        FiniteQuenchReducedODEClosure(
            bridge,
            n_reference=-4.0,
            kappa_reference=bad,
        )


def test_zero_present_density_and_out_of_domain_inputs_are_rejected() -> None:
    empty = _bridge(omega=0.0, reservoir=0.0)
    with pytest.raises(ValueError, match="positive present density"):
        FiniteQuenchReducedODEClosure(
            empty,
            n_reference=-4.0,
            kappa_reference=2.0,
        )
    _, gate = _gate()
    with pytest.raises(ValueError, match="outside"):
        gate.construct(
            n=0.1,
            scalar_clock_shift=0.0,
            total_momentum_density=0.0,
        )


@pytest.mark.parametrize("w_reservoir", [-0.1, 1.1])
def test_noncausal_or_negative_sound_speed_barotropes_are_rejected(
    w_reservoir: float,
) -> None:
    bridge = _bridge(w_reservoir=w_reservoir)
    with pytest.raises(ValueError, match="0 <= w_R <= 1"):
        FiniteQuenchReducedODEClosure(
            bridge,
            n_reference=-4.0,
            kappa_reference=2.0,
        )


def test_receipt_separates_effective_mathematical_closure_from_physics() -> None:
    _, _, receipt = _construct()
    assert receipt.domain.force_closure_status.startswith("ADOPTED_EFFECTIVE")
    assert not receipt.domain.microphysical_covariant_force_law_proven
    assert not receipt.numerical_finite_step_solution_certified
    assert not receipt.interval_enclosure_proven
    assert not receipt.microphysical_covariant_transfer_law_proven
    assert not receipt.observed_initial_spectrum_supplied
    assert not receipt.observable_prediction_proven
    assert "GLOBAL_LINEAR_REDUCED_ODE" in receipt.role
    assert "NOT_MICROPHYSICAL_COVARIANT" in receipt.role
