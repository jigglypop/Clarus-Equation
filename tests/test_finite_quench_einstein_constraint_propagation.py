"""Focused tests for differentiated scalar Einstein constraints."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

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
    receipt = FiniteQuenchEinsteinConstraintPropagation(bridge).construct(
        common_clock_second_tangent=second
    )
    return bridge, receipt


def test_constructed_constraint_derivatives_vanish_and_identify_propagation() -> None:
    _, receipt = _construct()
    assert receipt.parent_second_tangent_holds
    assert receipt.zero_stress_functional_branch_holds
    assert receipt.energy_derivative_candidate_matches_direct
    assert receipt.momentum_derivative_candidate_matches_direct
    assert receipt.energy_propagation_identity_holds
    assert receipt.momentum_propagation_identity_holds
    assert receipt.energy_constraint_derivative_vanishes
    assert receipt.momentum_constraint_derivative_vanishes
    assert receipt.local_first_derivative_constraint_propagation_holds
    assert not receipt.failure_reasons


def test_direct_derivatives_recompute_product_rules_from_raw_equations() -> None:
    _, receipt = _construct()
    second = receipt.common_clock_second_tangent
    node = second.common_clock_tangent.gr_linear_node
    trace = second.einstein_trace_evolution
    einstein = node.einstein_constraint
    energy = node.energy_equation
    momentum = node.momentum_equation
    coupling = node.background.gravity_constraint_coupling
    kappa_squared = einstein.k_over_a_h_squared
    expected_energy = math.fsum(
        (
            receipt.kappa_squared_derivative
            * einstein.curvature_potential,
            kappa_squared * einstein.curvature_potential_log_derivative,
            3.0
            * trace.provided_curvature_potential_second_log_derivative,
            3.0 * trace.lapse_potential_log_derivative,
            receipt.gravity_coupling_derivative
            * einstein.total_density_perturbation,
            coupling * energy.provided_total_density_perturbation_derivative,
        )
    )
    expected_momentum = math.fsum(
        (
            trace.provided_curvature_potential_second_log_derivative,
            trace.lapse_potential_log_derivative,
            receipt.gravity_coupling_derivative
            * einstein.total_momentum_density,
            coupling * momentum.provided_total_momentum_density_derivative,
        )
    )
    assert receipt.direct_energy_constraint_log_derivative == pytest.approx(
        expected_energy
    )
    assert receipt.direct_momentum_constraint_log_derivative == pytest.approx(
        expected_momentum
    )


def test_closed_constraint_identities_have_exact_coefficients_and_signs() -> None:
    _, receipt = _construct()
    node = receipt.common_clock_second_tangent.common_clock_tangent.gr_linear_node
    h = node.background.hubble_log_derivative
    kappa_squared = node.einstein_constraint.k_over_a_h_squared
    expected_energy = (
        3.0 * receipt.parent_trace_evolution_residual
        - (3.0 + 2.0 * h) * receipt.parent_energy_constraint_residual
        + kappa_squared * receipt.parent_momentum_constraint_residual
    )
    expected_momentum = (
        receipt.parent_trace_evolution_residual
        - (3.0 + h) * receipt.parent_momentum_constraint_residual
    )
    assert receipt.identity_energy_constraint_log_derivative == (
        pytest.approx(expected_energy)
    )
    assert receipt.identity_momentum_constraint_log_derivative == (
        pytest.approx(expected_momentum)
    )


def test_coupling_and_kappa_derivatives_match_flat_gr_background() -> None:
    _, receipt = _construct()
    node = receipt.common_clock_second_tangent.common_clock_tangent.gr_linear_node
    background = node.background
    kappa_squared = node.einstein_constraint.k_over_a_h_squared
    assert receipt.gravity_coupling_derivative == pytest.approx(
        -2.0
        * background.hubble_log_derivative
        * background.gravity_constraint_coupling
    )
    assert receipt.kappa_squared_derivative == pytest.approx(
        -2.0
        * (1.0 + background.hubble_log_derivative)
        * kappa_squared
    )


def test_wrong_energy_derivative_candidate_is_falsified() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchEinsteinConstraintPropagation(bridge).audit(
        common_clock_second_tangent=receipt.common_clock_second_tangent,
        energy_constraint_log_derivative=(
            receipt.direct_energy_constraint_log_derivative + 0.5
        ),
        momentum_constraint_log_derivative=(
            receipt.direct_momentum_constraint_log_derivative
        ),
    )
    assert not bad.energy_derivative_candidate_matches_direct
    assert not bad.energy_constraint_derivative_vanishes
    assert not bad.local_first_derivative_constraint_propagation_holds
    assert "ENERGY_DERIVATIVE_CANDIDATE_FAILED" in bad.failure_reasons


def test_wrong_momentum_derivative_candidate_is_falsified() -> None:
    bridge, receipt = _construct()
    bad = FiniteQuenchEinsteinConstraintPropagation(bridge).audit(
        common_clock_second_tangent=receipt.common_clock_second_tangent,
        energy_constraint_log_derivative=(
            receipt.direct_energy_constraint_log_derivative
        ),
        momentum_constraint_log_derivative=(
            receipt.direct_momentum_constraint_log_derivative + 0.5
        ),
    )
    assert not bad.momentum_derivative_candidate_matches_direct
    assert not bad.momentum_constraint_derivative_vanishes
    assert not bad.local_first_derivative_constraint_propagation_holds


def test_forged_parent_boolean_is_recomputed_from_raw_second_tangent() -> None:
    bridge, receipt = _construct()
    second = receipt.common_clock_second_tangent
    forged = replace(
        second,
        provided_scalar_clock_second_log_derivative=(
            second.provided_scalar_clock_second_log_derivative + 0.25
        ),
        scalar_clock_second_derivative_holds=True,
        produced_second_tangent_holds=True,
        reservoir_second_tangent_holds=True,
        total_second_tangent_holds=True,
        local_common_clock_second_tangent_holds=True,
    )
    bad = FiniteQuenchEinsteinConstraintPropagation(bridge).construct(
        common_clock_second_tangent=forged
    )
    assert not bad.parent_second_tangent_holds
    assert not bad.local_first_derivative_constraint_propagation_holds
    assert "PARENT_SECOND_TANGENT_FAILED" in bad.failure_reasons


def test_source_off_node_also_propagates_constraints() -> None:
    _, receipt = _construct(n=-5.0)
    assert receipt.local_first_derivative_constraint_propagation_holds
    assert receipt.direct_energy_constraint_log_derivative == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert receipt.direct_momentum_constraint_log_derivative == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_different_total_momentum_amplitudes_keep_the_identity() -> None:
    for total_u in (-0.01, 0.0, 0.02):
        _, receipt = _construct(total_u=total_u)
        assert receipt.energy_propagation_identity_holds
        assert receipt.momentum_propagation_identity_holds
        assert receipt.local_first_derivative_constraint_propagation_holds


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0"])
def test_nonfinite_or_nonreal_constraint_derivative_is_rejected(bad) -> None:
    bridge, receipt = _construct()
    with pytest.raises(ValueError, match="finite real"):
        FiniteQuenchEinsteinConstraintPropagation(bridge).audit(
            common_clock_second_tangent=receipt.common_clock_second_tangent,
            energy_constraint_log_derivative=bad,
            momentum_constraint_log_derivative=0.0,
        )


def test_wrong_parent_type_is_rejected() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="CommonClockSecondTangentReceipt"):
        FiniteQuenchEinsteinConstraintPropagation(bridge).construct(
            common_clock_second_tangent=object()
        )


def test_receipt_keeps_continuous_identity_separate_from_numerical_steps() -> None:
    _, receipt = _construct()
    assert not receipt.finite_step_constraint_propagation_proven
    assert not receipt.interval_certified
    assert not receipt.microphysical_covariant_transfer_law_proven
    assert "LOCAL_FIRST_DERIVATIVE" in receipt.role
    assert "NOT_FINITE_STEP" in receipt.role
    roles = dict(receipt.dimensionless_roles)
    assert roles["D_n"] == "derivative with respect to ln(a)"
