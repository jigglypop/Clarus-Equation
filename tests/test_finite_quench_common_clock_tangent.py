"""Focused tests for local common-clock first-tangent closure."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_common_clock_tangent import (
    FiniteQuenchCommonClockTangent,
)
from examples.physics.finite_quench_gr_linear_node import (
    FiniteQuenchGRLinearNode,
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


def _construct(
    *,
    n: float = -4.0,
    kappa: float = 2.0,
    clock: float = 0.25,
    total_momentum: float = 0.005,
    intrinsic_f_p: float = 0.003,
):
    bridge = _bridge()
    return FiniteQuenchCommonClockTangent(bridge).construct(
        n=n,
        k_over_a_h=kappa,
        scalar_clock_shift=clock,
        total_momentum_density=total_momentum,
        produced_intrinsic_momentum_potential=intrinsic_f_p,
    )


def test_constructed_tangent_solves_clock_derivative_and_momentum_split() -> None:
    receipt = _construct()
    assert receipt.parent_gr_linear_node_holds
    assert receipt.total_clock_derivative_holds
    assert receipt.produced_tangent_equation_holds
    assert receipt.reservoir_tangent_equation_holds
    assert receipt.momentum_split_holds
    assert receipt.density_tangent_derivatives_match_energy_equations
    assert receipt.local_common_clock_first_tangent_holds
    assert not receipt.failure_reasons
    assert receipt.scalar_clock_log_derivative == pytest.approx(
        receipt.required_scalar_clock_log_derivative
    )


def test_solved_component_momenta_sum_to_declared_total() -> None:
    total = 0.017
    receipt = _construct(total_momentum=total)
    einstein = receipt.gr_linear_node.einstein_constraint
    assert einstein.total_momentum_density == pytest.approx(total)
    assert (
        receipt.required_produced_momentum_density
        + receipt.required_reservoir_momentum_density
    ) == pytest.approx(total)


def test_energy_derivatives_are_actual_clock_tangent_derivatives() -> None:
    receipt = _construct()
    energy = receipt.gr_linear_node.energy_equation
    assert energy.provided_produced_density_perturbation_derivative == (
        pytest.approx(
            receipt.expected_produced_density_perturbation_derivative
        )
    )
    assert energy.provided_reservoir_density_perturbation_derivative == (
        pytest.approx(
            receipt.expected_reservoir_density_perturbation_derivative
        )
    )


def test_arbitrary_momentum_split_can_close_one_node_but_fail_clock_tangent() -> None:
    bridge = _bridge()
    node = FiniteQuenchGRLinearNode(bridge).construct(
        n=-4.0,
        k_over_a_h=2.0,
        scalar_clock_shift=0.25,
        produced_momentum_density=0.01,
        reservoir_momentum_density=-0.005,
        produced_intrinsic_momentum_potential=0.003,
    )
    assert node.full_declared_gr_linear_node_holds
    tangent = FiniteQuenchCommonClockTangent(bridge)
    clock_prime = tangent.required_clock_log_derivative(node)
    audited = tangent.audit(
        gr_linear_node=node,
        scalar_clock_log_derivative=clock_prime,
    )
    assert audited.total_clock_derivative_holds
    assert not audited.momentum_split_holds
    assert not audited.local_common_clock_first_tangent_holds


def test_wrong_clock_derivative_fails_total_and_both_component_tangents() -> None:
    receipt = _construct()
    audited = FiniteQuenchCommonClockTangent(_bridge()).audit(
        gr_linear_node=receipt.gr_linear_node,
        scalar_clock_log_derivative=(
            receipt.scalar_clock_log_derivative + 0.1
        ),
    )
    assert not audited.total_clock_derivative_holds
    assert not audited.produced_tangent_equation_holds
    assert not audited.reservoir_tangent_equation_holds
    assert not audited.local_common_clock_first_tangent_holds


def test_q_prime_root_keeps_q_phi_plus_h_t_term_in_tangent() -> None:
    bridge = _bridge()
    n_root = -4.0 - 1.0 / 6.0
    receipt = FiniteQuenchCommonClockTangent(bridge).construct(
        n=n_root,
        k_over_a_h=2.0,
        scalar_clock_shift=0.25,
        total_momentum_density=0.005,
        produced_intrinsic_momentum_potential=0.003,
    )
    assert bridge.source(n_root) > 0.0
    assert abs(bridge.source_derivative(n_root)) <= (
        1.0e-12 * bridge.source(n_root)
    )
    assert receipt.local_common_clock_first_tangent_holds


def test_empty_produced_component_is_handled_without_density_derivative_division() -> None:
    bridge = _bridge()
    receipt = FiniteQuenchCommonClockTangent(bridge).construct(
        n=-5.0,
        k_over_a_h=2.0,
        scalar_clock_shift=0.25,
        total_momentum_density=0.005,
        produced_intrinsic_momentum_potential=0.003,
    )
    background = receipt.gr_linear_node.background
    assert background.produced_density == 0.0
    assert background.produced_density_derivative == 0.0
    assert receipt.required_produced_momentum_density == 0.0
    assert receipt.produced_tangent_equation_holds
    assert receipt.local_common_clock_first_tangent_holds


def test_forged_parent_einstein_derivative_is_reaudited_and_rejected() -> None:
    receipt = _construct()
    node = receipt.gr_linear_node
    forged_einstein = replace(
        node.einstein_constraint,
        curvature_potential_log_derivative=(
            node.einstein_constraint.curvature_potential_log_derivative + 0.1
        ),
        all_declared_scalar_constraints_hold=True,
    )
    forged_node = replace(
        node,
        einstein_constraint=forged_einstein,
        full_declared_gr_linear_node_holds=True,
    )
    audited = FiniteQuenchCommonClockTangent(_bridge()).audit(
        gr_linear_node=forged_node,
        scalar_clock_log_derivative=receipt.scalar_clock_log_derivative,
    )
    assert not audited.parent_gr_linear_node_holds
    assert not audited.local_common_clock_first_tangent_holds


@pytest.mark.parametrize("n", [-5.0, -4.0, -3.0, 0.0])
def test_source_on_and_off_nodes_have_a_local_first_tangent(n: float) -> None:
    receipt = _construct(n=n)
    assert receipt.local_common_clock_first_tangent_holds


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kappa=0.0),
        dict(kappa=math.inf),
        dict(clock=math.nan),
        dict(total_momentum=True),
        dict(total_momentum=math.inf),
        dict(intrinsic_f_p=math.nan),
    ],
)
def test_tangent_inputs_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _construct(**kwargs)


def test_receipt_reduces_split_to_total_momentum_but_denies_propagation() -> None:
    receipt = _construct()
    assert receipt.free_declared_inputs == (
        "scalar_clock_shift",
        "total_momentum_density",
        "produced_intrinsic_momentum_potential",
    )
    assert not receipt.momentum_split_tangent_preservation_proven
    assert not receipt.finite_step_constraint_propagation_proven
    assert receipt.role.endswith(
        "NOT_SECOND_TANGENT_TIME_INTEGRATION_OR_PROPAGATION_PROOF"
    )
