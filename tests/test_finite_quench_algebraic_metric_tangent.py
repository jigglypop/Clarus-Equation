"""Focused tests for algebraic metric tangent consistency."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_algebraic_metric_tangent import (
    FiniteQuenchAlgebraicMetricTangent,
)
from examples.physics.finite_quench_common_clock_second_tangent import (
    FiniteQuenchCommonClockSecondTangent,
)
from examples.physics.finite_quench_einstein_constraint_propagation import (
    FiniteQuenchEinsteinConstraintPropagation,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


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


def _construct(*, n: float = -4.0, total_u: float = 0.005):
    bridge = _bridge()
    second = FiniteQuenchCommonClockSecondTangent(bridge).construct(
        n=n,
        k_over_a_h=2.0,
        scalar_clock_shift=0.25,
        total_momentum_density=total_u,
    )
    propagation = FiniteQuenchEinsteinConstraintPropagation(
        bridge
    ).construct(common_clock_second_tangent=second)
    receipt = FiniteQuenchAlgebraicMetricTangent(bridge).construct(
        constraint_propagation=propagation
    )
    return bridge, receipt


def test_constructed_algebraic_metric_has_first_and_second_tangents() -> None:
    _, receipt = _construct()
    assert receipt.parent_constraint_propagation_holds
    assert receipt.functional_zero_stress_branch_holds
    assert receipt.algebraic_curvature_first_tangent_holds
    assert receipt.algebraic_curvature_second_tangent_holds
    assert receipt.local_algebraic_metric_second_tangent_holds
    assert not receipt.failure_reasons


def test_first_derivative_is_full_quotient_product_rule() -> None:
    _, receipt = _construct()
    propagation = receipt.constraint_propagation
    second = propagation.common_clock_second_tangent
    node = second.common_clock_tangent.gr_linear_node
    coupling = node.background.gravity_constraint_coupling
    kappa_squared = node.einstein_constraint.k_over_a_h_squared
    expected = math.fsum(
        (
            propagation.gravity_coupling_derivative
            * receipt.density_momentum_numerator
            / kappa_squared,
            coupling
            * receipt.density_momentum_numerator_derivative
            / kappa_squared,
            -coupling
            * receipt.density_momentum_numerator
            * propagation.kappa_squared_derivative
            / kappa_squared**2,
        )
    )
    assert receipt.direct_algebraic_curvature_potential_log_derivative == (
        pytest.approx(expected)
    )


def test_second_derivative_is_differentiated_momentum_constraint() -> None:
    _, receipt = _construct()
    propagation = receipt.constraint_propagation
    second = propagation.common_clock_second_tangent
    node = second.common_clock_tangent.gr_linear_node
    coupling = node.background.gravity_constraint_coupling
    einstein = node.einstein_constraint
    momentum = node.momentum_equation
    expected = (
        -receipt.direct_algebraic_curvature_potential_log_derivative
        - propagation.gravity_coupling_derivative
        * einstein.total_momentum_density
        - coupling * momentum.provided_total_momentum_density_derivative
    )
    assert (
        receipt.direct_algebraic_curvature_potential_second_log_derivative
        == pytest.approx(expected)
    )


def test_wrong_first_derivative_candidate_is_falsified() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchAlgebraicMetricTangent(bridge).audit(
        constraint_propagation=receipt.constraint_propagation,
        curvature_potential_log_derivative=(
            receipt.provided_curvature_potential_log_derivative + 0.5
        ),
        curvature_potential_second_log_derivative=(
            receipt.provided_curvature_potential_second_log_derivative
        ),
    )
    assert not bad.algebraic_curvature_first_tangent_holds
    assert not bad.local_algebraic_metric_second_tangent_holds
    assert "ALGEBRAIC_CURVATURE_FIRST_TANGENT_FAILED" in bad.failure_reasons


def test_wrong_second_derivative_candidate_is_falsified() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchAlgebraicMetricTangent(bridge).audit(
        constraint_propagation=receipt.constraint_propagation,
        curvature_potential_log_derivative=(
            receipt.provided_curvature_potential_log_derivative
        ),
        curvature_potential_second_log_derivative=(
            receipt.provided_curvature_potential_second_log_derivative + 0.5
        ),
    )
    assert not bad.algebraic_curvature_second_tangent_holds
    assert not bad.local_algebraic_metric_second_tangent_holds


def test_forged_parent_boolean_is_raw_recomputed() -> None:
    bridge, receipt = _construct()
    propagation = receipt.constraint_propagation
    forged = replace(
        propagation,
        provided_energy_constraint_log_derivative=(
            propagation.provided_energy_constraint_log_derivative + 0.25
        ),
        energy_derivative_candidate_matches_direct=True,
        energy_constraint_derivative_vanishes=True,
        local_first_derivative_constraint_propagation_holds=True,
    )
    bad = FiniteQuenchAlgebraicMetricTangent(bridge).construct(
        constraint_propagation=forged
    )
    assert not bad.parent_constraint_propagation_holds
    assert not bad.local_algebraic_metric_second_tangent_holds
    assert "PARENT_CONSTRAINT_PROPAGATION_FAILED" in bad.failure_reasons


def test_source_off_and_multiple_momentum_amplitudes_hold() -> None:
    for n, total_u in ((-5.0, 0.005), (-4.0, -0.01), (-4.0, 0.02)):
        _, receipt = _construct(n=n, total_u=total_u)
        assert receipt.local_algebraic_metric_second_tangent_holds


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0"])
def test_nonfinite_or_nonreal_metric_derivative_is_rejected(bad) -> None:
    bridge, receipt = _construct()
    with pytest.raises(ValueError, match="finite real"):
        FiniteQuenchAlgebraicMetricTangent(bridge).audit(
            constraint_propagation=receipt.constraint_propagation,
            curvature_potential_log_derivative=bad,
            curvature_potential_second_log_derivative=0.0,
        )


def test_wrong_parent_type_is_rejected() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="EinsteinConstraintPropagationReceipt"):
        FiniteQuenchAlgebraicMetricTangent(bridge).construct(
            constraint_propagation=object()
        )


def test_receipt_keeps_metric_tangent_separate_from_finite_integration() -> None:
    _, receipt = _construct()
    assert not receipt.finite_step_metric_evolution_proven
    assert not receipt.interval_certified
    assert "LOCAL_ALGEBRAIC_METRIC_SECOND_TANGENT" in receipt.role
    assert "NOT_FINITE_STEP" in receipt.role
    roles = dict(receipt.dimensionless_roles)
    assert roles["psi_n"] == "d psi/d ln(a)"
