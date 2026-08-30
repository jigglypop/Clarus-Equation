"""Focused tests for local common-clock second-tangent closure."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_common_clock_second_tangent import (
    FiniteQuenchCommonClockSecondTangent,
)
from examples.physics.finite_quench_common_clock_tangent import (
    FiniteQuenchCommonClockTangent,
)
from examples.physics.finite_quench_einstein_trace_evolution import (
    FiniteQuenchEinsteinTraceEvolution,
)
from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


N = -4.0
KAPPA = 2.0
CLOCK = 0.25
TOTAL_U = 0.005


def _bridge() -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=0.1,
            w_open=2.1767e-4,
        )
    )


def _construct(
    *,
    n: float = N,
    total_momentum: float = TOTAL_U,
):
    bridge = _bridge()
    receipt = FiniteQuenchCommonClockSecondTangent(bridge).construct(
        n=n,
        k_over_a_h=KAPPA,
        scalar_clock_shift=CLOCK,
        total_momentum_density=total_momentum,
    )
    return bridge, receipt


def test_constructed_second_tangent_closes_all_local_residuals() -> None:
    _, receipt = _construct()
    assert receipt.parent_first_tangent_holds
    assert receipt.parent_metric_trace_holds
    assert receipt.trace_and_tangent_same_node
    assert receipt.total_momentum_derivative_holds
    assert receipt.scalar_clock_second_derivative_holds
    assert receipt.locally_required_intrinsic_force_holds
    assert receipt.component_momentum_derivatives_hit_second_tangent_targets
    assert receipt.produced_second_tangent_holds
    assert receipt.reservoir_second_tangent_holds
    assert receipt.total_second_tangent_holds
    assert receipt.local_common_clock_second_tangent_holds
    assert not receipt.failure_reasons


def test_required_force_is_the_eq21_value_for_the_split_target() -> None:
    bridge, receipt = _construct()
    node = receipt.common_clock_tangent.gr_linear_node
    background = node.background
    closure = node.closure
    einstein = node.einstein_constraint
    total_velocity = (
        einstein.total_momentum_density / background.total_enthalpy
    )
    expected = (
        receipt.required_produced_momentum_density_derivative
        + (3.0 - background.hubble_log_derivative)
        * einstein.produced_momentum_density
        + (background.produced_density + background.produced_pressure)
        * einstein.lapse_potential
        + closure.produced_pressure_perturbation
        - bridge.source(background.n) * total_velocity
    )
    assert receipt.required_produced_intrinsic_momentum_potential == (
        pytest.approx(expected)
    )
    assert receipt.provided_produced_intrinsic_momentum_potential == (
        pytest.approx(expected)
    )


def test_clock_second_derivative_is_derivative_of_summed_clock_equation() -> None:
    _, receipt = _construct()
    tangent = receipt.common_clock_tangent
    trace = receipt.einstein_trace_evolution
    einstein = tangent.gr_linear_node.einstein_constraint
    background = tangent.gr_linear_node.background
    expected = (
        -trace.provided_curvature_potential_second_log_derivative
        - (
            receipt.kappa_squared_derivative
            * einstein.total_momentum_density
            + einstein.k_over_a_h_squared
            * receipt.required_total_momentum_density_derivative
        )
        / (3.0 * background.total_enthalpy)
        + einstein.k_over_a_h_squared
        * einstein.total_momentum_density
        * receipt.total_enthalpy_derivative
        / (3.0 * background.total_enthalpy**2)
    )
    assert receipt.required_scalar_clock_second_log_derivative == (
        pytest.approx(expected)
    )


def test_eq21_component_derivatives_hit_differentiated_split_targets() -> None:
    _, receipt = _construct()
    assert receipt.provided_produced_momentum_density_derivative == (
        pytest.approx(
            receipt.required_produced_momentum_density_derivative
        )
    )
    assert receipt.provided_reservoir_momentum_density_derivative == (
        pytest.approx(
            receipt.required_reservoir_momentum_density_derivative
        )
    )
    assert (
        receipt.required_produced_momentum_density_derivative
        + receipt.required_reservoir_momentum_density_derivative
    ) == pytest.approx(receipt.required_total_momentum_density_derivative)


def test_background_derivatives_obey_exact_summed_identities() -> None:
    _, receipt = _construct()
    node = receipt.common_clock_tangent.gr_linear_node
    h = node.background.hubble_log_derivative
    kappa_squared = node.einstein_constraint.k_over_a_h_squared
    assert receipt.total_density_second_derivative == pytest.approx(
        -3.0 * receipt.total_enthalpy_derivative
    )
    assert receipt.kappa_squared_derivative == pytest.approx(
        -2.0 * (1.0 + h) * kappa_squared
    )


def test_hubble_second_log_derivative_matches_independent_centered_difference() -> None:
    bridge, receipt = _construct()
    background = FiniteQuenchTwoFluidFlatGRBackground(bridge)
    epsilon = 1.0e-6
    finite_difference = (
        background.construct(N + epsilon).hubble_log_derivative
        - background.construct(N - epsilon).hubble_log_derivative
    ) / (2.0 * epsilon)
    assert receipt.hubble_second_log_derivative == pytest.approx(
        finite_difference,
        rel=2.0e-8,
        abs=1.0e-9,
    )


def test_source_off_empty_produced_limit_needs_no_fitted_force() -> None:
    _, receipt = _construct(n=-5.0)
    einstein = receipt.common_clock_tangent.gr_linear_node.einstein_constraint
    assert einstein.produced_momentum_density == pytest.approx(0.0)
    assert receipt.required_produced_intrinsic_momentum_potential == (
        pytest.approx(0.0)
    )
    assert receipt.local_common_clock_second_tangent_holds


def test_wrong_intrinsic_force_is_falsified_by_second_tangent() -> None:
    bridge, receipt = _construct()
    wrong_force = (
        receipt.required_produced_intrinsic_momentum_potential + 1.0
    )
    tangent = FiniteQuenchCommonClockTangent(bridge).construct(
        n=N,
        k_over_a_h=KAPPA,
        scalar_clock_shift=CLOCK,
        total_momentum_density=TOTAL_U,
        produced_intrinsic_momentum_potential=wrong_force,
    )
    trace = FiniteQuenchEinsteinTraceEvolution(bridge).construct(
        gr_linear_node=tangent.gr_linear_node
    )
    bad = FiniteQuenchCommonClockSecondTangent(bridge).audit(
        common_clock_tangent=tangent,
        einstein_trace_evolution=trace,
        scalar_clock_second_log_derivative=(
            receipt.required_scalar_clock_second_log_derivative
        ),
    )
    assert not bad.locally_required_intrinsic_force_holds
    assert not bad.component_momentum_derivatives_hit_second_tangent_targets
    assert not bad.produced_second_tangent_holds
    assert not bad.local_common_clock_second_tangent_holds
    assert "LOCALLY_REQUIRED_INTRINSIC_FORCE_FAILED" in bad.failure_reasons


def test_wrong_clock_second_derivative_is_falsified() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchCommonClockSecondTangent(bridge).audit(
        common_clock_tangent=receipt.common_clock_tangent,
        einstein_trace_evolution=receipt.einstein_trace_evolution,
        scalar_clock_second_log_derivative=(
            receipt.required_scalar_clock_second_log_derivative + 0.5
        ),
    )
    assert not bad.scalar_clock_second_derivative_holds
    assert not bad.produced_second_tangent_holds
    assert not bad.reservoir_second_tangent_holds
    assert not bad.total_second_tangent_holds


def test_forged_trace_booleans_do_not_hide_wrong_psi_second_derivative() -> None:
    bridge, receipt = _construct()
    trace = receipt.einstein_trace_evolution
    forged_trace = replace(
        trace,
        provided_curvature_potential_second_log_derivative=(
            trace.provided_curvature_potential_second_log_derivative + 0.25
        ),
        general_spatial_trace_holds=True,
        reduced_zero_stress_trace_holds=True,
        one_node_metric_second_derivative_holds=True,
    )
    bad = FiniteQuenchCommonClockSecondTangent(bridge).audit(
        common_clock_tangent=receipt.common_clock_tangent,
        einstein_trace_evolution=forged_trace,
        scalar_clock_second_log_derivative=(
            receipt.required_scalar_clock_second_log_derivative
        ),
    )
    assert not bad.parent_metric_trace_holds
    assert not bad.local_common_clock_second_tangent_holds
    assert "PARENT_METRIC_TRACE_FAILED" in bad.failure_reasons


def test_trace_from_another_node_is_rejected() -> None:
    bridge, receipt = _construct()
    other = FiniteQuenchCommonClockSecondTangent(bridge).construct(
        n=N,
        k_over_a_h=KAPPA,
        scalar_clock_shift=CLOCK,
        total_momentum_density=TOTAL_U + 0.002,
    )
    bad = FiniteQuenchCommonClockSecondTangent(bridge).audit(
        common_clock_tangent=receipt.common_clock_tangent,
        einstein_trace_evolution=other.einstein_trace_evolution,
        scalar_clock_second_log_derivative=(
            receipt.required_scalar_clock_second_log_derivative
        ),
    )
    assert not bad.trace_and_tangent_same_node
    assert not bad.local_common_clock_second_tangent_holds
    assert "TRACE_TANGENT_NODE_MISMATCH" in bad.failure_reasons


def test_forged_first_tangent_booleans_are_raw_recomputed() -> None:
    bridge, receipt = _construct()
    tangent = receipt.common_clock_tangent
    forged_tangent = replace(
        tangent,
        scalar_clock_log_derivative=(
            tangent.scalar_clock_log_derivative + 0.25
        ),
        total_clock_derivative_holds=True,
        produced_tangent_equation_holds=True,
        reservoir_tangent_equation_holds=True,
        local_common_clock_first_tangent_holds=True,
    )
    bad = FiniteQuenchCommonClockSecondTangent(bridge).audit(
        common_clock_tangent=forged_tangent,
        einstein_trace_evolution=receipt.einstein_trace_evolution,
        scalar_clock_second_log_derivative=(
            receipt.required_scalar_clock_second_log_derivative
        ),
    )
    assert not bad.parent_first_tangent_holds
    assert not bad.local_common_clock_second_tangent_holds
    assert "PARENT_FIRST_TANGENT_FAILED" in bad.failure_reasons


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0"])
def test_nonfinite_or_nonreal_clock_second_candidate_is_rejected(bad) -> None:
    bridge, receipt = _construct()
    with pytest.raises(ValueError, match="finite real"):
        FiniteQuenchCommonClockSecondTangent(bridge).audit(
            common_clock_tangent=receipt.common_clock_tangent,
            einstein_trace_evolution=receipt.einstein_trace_evolution,
            scalar_clock_second_log_derivative=bad,
        )


def test_wrong_parent_receipt_types_are_rejected() -> None:
    bridge, receipt = _construct()
    gate = FiniteQuenchCommonClockSecondTangent(bridge)
    with pytest.raises(ValueError, match="CommonClockTangentReceipt"):
        gate.audit(
            common_clock_tangent=object(),
            einstein_trace_evolution=receipt.einstein_trace_evolution,
            scalar_clock_second_log_derivative=0.0,
        )
    with pytest.raises(ValueError, match="EinsteinTraceEvolutionReceipt"):
        gate.audit(
            common_clock_tangent=receipt.common_clock_tangent,
            einstein_trace_evolution=object(),
            scalar_clock_second_log_derivative=0.0,
        )


def test_receipt_does_not_overclaim_fitted_force_or_time_propagation() -> None:
    _, receipt = _construct()
    assert receipt.fitted_node_force_is_not_a_force_law
    assert not receipt.unreduced_qmu_second_derivative_cancellation_proven
    assert not receipt.microphysical_covariant_transfer_law_proven
    assert not receipt.finite_step_constraint_propagation_proven
    assert "FITTED_FORCE" in receipt.role
    assert "NOT_MICROPHYSICAL_FORCE_LAW" in receipt.role
    assert receipt.free_declared_inputs == (
        "scalar_clock_shift",
        "total_momentum_density",
    )
    roles = dict(receipt.dimensionless_roles)
    assert roles["fhat_A"] == "a f_A/rho_unit"
