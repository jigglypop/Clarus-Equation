"""Focused tests for the one-node spatial-trace Einstein evolution."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_einstein_trace_evolution import (
    FiniteQuenchEinsteinTraceEvolution,
)
from examples.physics.finite_quench_gr_linear_node import (
    FiniteQuenchGRLinearNode,
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


def _node(
    bridge: FiniteQuenchBridge,
    *,
    n: float = -4.0,
    kappa: float = 2.0,
    clock: float = 0.25,
    momentum_p: float = 0.002,
    momentum_r: float = 0.003,
    intrinsic_f_p: float = 0.001,
):
    return FiniteQuenchGRLinearNode(bridge).construct(
        n=n,
        k_over_a_h=kappa,
        scalar_clock_shift=clock,
        produced_momentum_density=momentum_p,
        reservoir_momentum_density=momentum_r,
        produced_intrinsic_momentum_potential=intrinsic_f_p,
    )


def _construct():
    bridge = _bridge()
    node = _node(bridge)
    return bridge, FiniteQuenchEinsteinTraceEvolution(bridge).construct(
        gr_linear_node=node
    )


def test_constructed_trace_solves_both_general_and_reduced_forms() -> None:
    _, receipt = _construct()
    assert receipt.parent_gr_linear_node_holds
    assert receipt.functional_zero_anisotropic_stress_declared
    assert receipt.lapse_curvature_derivative_identity_holds
    assert receipt.general_spatial_trace_holds
    assert receipt.reduced_zero_stress_trace_holds
    assert receipt.one_node_metric_second_derivative_holds
    assert not receipt.failure_reasons


def test_required_second_derivative_is_raw_zero_stress_trace_formula() -> None:
    _, receipt = _construct()
    node = receipt.gr_linear_node
    background = node.background
    einstein = node.einstein_constraint
    expected = (
        background.gravity_constraint_coupling
        * receipt.total_pressure_perturbation
        - (4.0 + background.hubble_log_derivative)
        * einstein.curvature_potential_log_derivative
        - (3.0 + 2.0 * background.hubble_log_derivative)
        * einstein.curvature_potential
    )
    assert receipt.required_curvature_potential_second_log_derivative == (
        pytest.approx(expected)
    )
    assert receipt.provided_curvature_potential_second_log_derivative == (
        pytest.approx(expected)
    )


def test_general_and_reduced_residuals_agree_on_functional_zero_stress() -> None:
    _, receipt = _construct()
    assert receipt.general_spatial_trace_residual == pytest.approx(
        receipt.reduced_zero_stress_trace_residual
    )


def test_pure_radiation_limit_has_h_minus_two_coefficients() -> None:
    bridge = _bridge(omega=0.0, reservoir=0.21, w_reservoir=1.0 / 3.0)
    node = _node(
        bridge,
        momentum_p=0.0,
        momentum_r=0.005,
        intrinsic_f_p=0.0,
    )
    receipt = FiniteQuenchEinsteinTraceEvolution(bridge).construct(
        gr_linear_node=node
    )
    einstein = receipt.gr_linear_node.einstein_constraint
    background = receipt.gr_linear_node.background
    assert background.hubble_log_derivative == pytest.approx(-2.0)
    expected = (
        background.gravity_constraint_coupling
        * receipt.total_pressure_perturbation
        - 2.0 * einstein.curvature_potential_log_derivative
        + einstein.curvature_potential
    )
    assert receipt.required_curvature_potential_second_log_derivative == (
        pytest.approx(expected)
    )


def test_pure_dust_limit_removes_the_undifferentiated_potential_term() -> None:
    bridge = _bridge(omega=0.12, reservoir=0.0, w_reservoir=0.0)
    node = _node(
        bridge,
        n=0.0,
        momentum_p=0.005,
        momentum_r=0.0,
        intrinsic_f_p=0.0,
    )
    receipt = FiniteQuenchEinsteinTraceEvolution(bridge).construct(
        gr_linear_node=node
    )
    einstein = receipt.gr_linear_node.einstein_constraint
    assert receipt.gr_linear_node.background.hubble_log_derivative == (
        pytest.approx(-1.5)
    )
    assert receipt.total_pressure_perturbation == pytest.approx(0.0)
    assert receipt.required_curvature_potential_second_log_derivative == (
        pytest.approx(-2.5 * einstein.curvature_potential_log_derivative)
    )


def test_wrong_second_derivative_is_falsified() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchEinsteinTraceEvolution(bridge).audit(
        gr_linear_node=receipt.gr_linear_node,
        lapse_potential_log_derivative=receipt.lapse_potential_log_derivative,
        curvature_potential_second_log_derivative=(
            receipt.required_curvature_potential_second_log_derivative + 0.25
        ),
    )
    assert not bad.general_spatial_trace_holds
    assert not bad.reduced_zero_stress_trace_holds
    assert not bad.one_node_metric_second_derivative_holds
    assert "GENERAL_SPATIAL_TRACE_FAILED" in bad.failure_reasons


def test_wrong_lapse_derivative_is_falsified_independently() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchEinsteinTraceEvolution(bridge).audit(
        gr_linear_node=receipt.gr_linear_node,
        lapse_potential_log_derivative=(
            receipt.lapse_potential_log_derivative + 0.25
        ),
        curvature_potential_second_log_derivative=(
            receipt.required_curvature_potential_second_log_derivative
        ),
    )
    assert not bad.lapse_curvature_derivative_identity_holds
    assert not bad.general_spatial_trace_holds
    assert not bad.one_node_metric_second_derivative_holds


def test_forged_parent_booleans_do_not_bypass_raw_recomputation() -> None:
    bridge, receipt = _construct()
    closure = receipt.gr_linear_node.closure
    forged_closure = replace(
        closure,
        reservoir_pressure_perturbation=(
            closure.reservoir_pressure_perturbation + 0.2
        ),
        pressure_closure_holds=True,
        all_strict_barotropic_constraints_hold=True,
    )
    forged_node = replace(
        receipt.gr_linear_node,
        closure=forged_closure,
        strict_barotropic_closure_holds=True,
        full_declared_gr_linear_node_holds=True,
    )
    audited = FiniteQuenchEinsteinTraceEvolution(bridge).construct(
        gr_linear_node=forged_node
    )
    assert not audited.parent_gr_linear_node_holds
    assert not audited.one_node_metric_second_derivative_holds
    assert "PARENT_GR_LINEAR_NODE_FAILED" in audited.failure_reasons


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0"])
def test_nonfinite_or_nonreal_derivative_candidates_are_rejected(bad) -> None:
    bridge, receipt = _construct()
    with pytest.raises(ValueError, match="finite real"):
        FiniteQuenchEinsteinTraceEvolution(bridge).audit(
            gr_linear_node=receipt.gr_linear_node,
            lapse_potential_log_derivative=bad,
            curvature_potential_second_log_derivative=0.0,
        )


def test_wrong_parent_type_is_rejected() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="GRLinearNodeReceipt"):
        FiniteQuenchEinsteinTraceEvolution(bridge).construct(
            gr_linear_node=object()
        )


def test_receipt_keeps_exact_scope_and_dimensionless_contract() -> None:
    _, receipt = _construct()
    assert receipt.zero_stress_is_constitutive_not_pointwise
    assert not receipt.finite_step_constraint_propagation_proven
    assert "ONE_NODE" in receipt.role
    assert "NOT_TIME_INTEGRATION" in receipt.role
    assert receipt.source.startswith("Ma_Bertschinger_1995")
    roles = dict(receipt.dimensionless_roles)
    assert roles["kappa"] == "k/(aH)"
    assert "rho_unit" in roles["Delta_P"]
