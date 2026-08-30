"""Focused tests for the cross-receipt linear perturbation node gate."""

from __future__ import annotations

from dataclasses import replace

import pytest

from examples.physics.finite_quench_linear_energy_equation import (
    FiniteQuenchLinearEnergyEquation,
)
from examples.physics.finite_quench_linear_momentum_equation import (
    FiniteQuenchLinearMomentumEquation,
)
from examples.physics.finite_quench_linear_system_gate import (
    FiniteQuenchLinearSystemNodeGate,
)
from examples.physics.finite_quench_qmu_projection_ledger import (
    FiniteQuenchLowerQmuProjectionLedger,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(*, omega_prod0: float = 0.12) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=omega_prod0,
            reservoir_present_density=0.21,
            w_reservoir=0.1,
            w_open=2.1767e-4,
        )
    )


def _base_projection(bridge: FiniteQuenchBridge, **overrides: object):
    values = dict(
        n=-4.0,
        k_over_a_h=2.0,
        scalar_clock_shift=0.25,
        hubble_log_derivative=-1.2,
        lapse_potential=0.03,
        normalized_total_velocity_potential=0.4,
        produced_intrinsic_momentum_potential=0.07,
    )
    values.update(overrides)
    return FiniteQuenchLowerQmuProjectionLedger(
        bridge
    ).construct_common_clock(**values)


def _energy(
    bridge: FiniteQuenchBridge,
    projection,
    momentum=None,
    *,
    produced_theta_offset: float = 0.0,
):
    if momentum is None:
        momentum = _momentum(bridge, projection)
    kappa_squared = projection.k_over_a_h**2
    theta_p = (
        0.0
        if momentum.produced_background_enthalpy == 0.0
        else -kappa_squared
        * momentum.produced_momentum_density
        / momentum.produced_background_enthalpy
    )
    theta_r = (
        0.0
        if momentum.reservoir_background_enthalpy == 0.0
        else -kappa_squared
        * momentum.reservoir_momentum_density
        / momentum.reservoir_background_enthalpy
    )
    return FiniteQuenchLinearEnergyEquation(
        bridge
    ).construct_required_derivatives(
        transfer_projection=projection,
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=0.015,
        metric_curvature_log_derivative=-0.02,
        produced_theta_over_a_h=theta_p + produced_theta_offset,
        reservoir_theta_over_a_h=theta_r,
    )


def _momentum(
    bridge: FiniteQuenchBridge,
    projection,
    *,
    reservoir_pressure_perturbation: float = 0.015,
):
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    momentum_p = 0.2
    momentum_r = equation.reservoir_momentum_for_total_energy_frame(
        transfer_projection=projection,
        produced_momentum_density=momentum_p,
    )
    return equation.construct_required_derivatives(
        transfer_projection=projection,
        produced_momentum_density=momentum_p,
        reservoir_momentum_density=momentum_r,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=reservoir_pressure_perturbation,
        produced_normalized_anisotropic_stress=0.0,
        reservoir_normalized_anisotropic_stress=0.0,
    )


def _gate(bridge: FiniteQuenchBridge, projection, energy=None, momentum=None):
    if momentum is None:
        momentum = _momentum(bridge, projection)
    if energy is None:
        energy = _energy(bridge, projection, momentum)
    return FiniteQuenchLinearSystemNodeGate(bridge).audit(
        transfer_projection=projection,
        energy_equation=energy,
        momentum_equation=momentum,
    )


def test_consistent_nondegenerate_common_clock_node_passes_cross_gate() -> None:
    bridge = _bridge()
    receipt = _gate(bridge, _base_projection(bridge))
    assert receipt.cross_receipt_consistency_holds
    assert receipt.projection_all_component_pairs_hold
    assert receipt.energy_equations_and_exchange_hold
    assert receipt.momentum_equations_and_exchange_hold
    assert receipt.projection_common_physical_clock_holds
    assert receipt.nondegenerate_total_energy_frame_holds
    assert receipt.full_declared_nondegenerate_node_holds
    assert receipt.failure_reasons == ()
    assert receipt.role.endswith(
        "NOT_INTEGRATED_EINSTEIN_BOLTZMANN_OR_OBSERVABLE_SOLUTION"
    )


def test_q_prime_only_pair_passes_algebra_but_fails_common_clock_gate() -> None:
    bridge = _bridge()
    qmu = FiniteQuenchLowerQmuProjectionLedger(bridge)
    n = -4.0
    clock = 0.25
    naive = bridge.source_derivative(n) * clock
    projection = qmu.audit(
        n=n,
        k_over_a_h=2.0,
        scalar_clock_shift=clock,
        hubble_log_derivative=-1.2,
        lapse_potential=0.03,
        normalized_total_velocity_potential=0.4,
        produced_physical_energy_perturbation=naive,
        reservoir_physical_energy_perturbation=-naive,
        produced_intrinsic_momentum_potential=0.07,
        reservoir_intrinsic_momentum_potential=-0.07,
    )
    receipt = _gate(bridge, projection)
    assert receipt.algebraic_energy_momentum_node_holds
    assert not receipt.projection_common_physical_clock_holds
    assert not receipt.common_clock_energy_momentum_node_holds
    assert not receipt.full_declared_nondegenerate_node_holds
    assert "COMMON_PHYSICAL_CLOCK_FAILED" in receipt.failure_reasons


def test_unpaired_delta_q_cannot_hide_behind_momentum_success() -> None:
    bridge = _bridge()
    paired = _base_projection(bridge)
    projection = FiniteQuenchLowerQmuProjectionLedger(bridge).audit(
        n=paired.n,
        k_over_a_h=paired.k_over_a_h,
        scalar_clock_shift=paired.scalar_clock_shift,
        hubble_log_derivative=paired.hubble_log_derivative,
        lapse_potential=paired.lapse_potential,
        normalized_total_velocity_potential=(
            paired.normalized_total_velocity_potential
        ),
        produced_physical_energy_perturbation=(
            paired.produced_physical_energy_perturbation
        ),
        reservoir_physical_energy_perturbation=0.0,
        produced_intrinsic_momentum_potential=(
            paired.produced_intrinsic_momentum_potential
        ),
        reservoir_intrinsic_momentum_potential=(
            paired.reservoir_intrinsic_momentum_potential
        ),
    )
    receipt = _gate(bridge, projection)
    assert receipt.momentum_equations_and_exchange_hold
    assert not receipt.energy_equations_and_exchange_hold
    assert not receipt.projection_physical_energy_pair_holds
    assert not receipt.full_declared_nondegenerate_node_holds


def test_unpaired_f_cannot_hide_behind_energy_success() -> None:
    bridge = _bridge()
    paired = _base_projection(bridge)
    projection = FiniteQuenchLowerQmuProjectionLedger(bridge).audit(
        n=paired.n,
        k_over_a_h=paired.k_over_a_h,
        scalar_clock_shift=paired.scalar_clock_shift,
        hubble_log_derivative=paired.hubble_log_derivative,
        lapse_potential=paired.lapse_potential,
        normalized_total_velocity_potential=(
            paired.normalized_total_velocity_potential
        ),
        produced_physical_energy_perturbation=(
            paired.produced_physical_energy_perturbation
        ),
        reservoir_physical_energy_perturbation=(
            paired.reservoir_physical_energy_perturbation
        ),
        produced_intrinsic_momentum_potential=0.07,
        reservoir_intrinsic_momentum_potential=0.0,
    )
    receipt = _gate(bridge, projection)
    assert receipt.energy_equations_and_exchange_hold
    assert not receipt.momentum_equations_and_exchange_hold
    assert not receipt.projection_intrinsic_momentum_pair_holds
    assert not receipt.full_declared_nondegenerate_node_holds


def test_k_zero_keeps_algebra_but_is_not_promoted_to_nondegenerate_node() -> None:
    bridge = _bridge()
    projection = _base_projection(bridge, k_over_a_h=0.0)
    receipt = _gate(bridge, projection)
    assert receipt.algebraic_energy_momentum_node_holds
    assert receipt.common_clock_energy_momentum_node_holds
    assert not receipt.nondegenerate_total_energy_frame_holds
    assert not receipt.full_declared_nondegenerate_node_holds
    assert "NONDEGENERATE_TOTAL_ENERGY_FRAME_FAILED" in receipt.failure_reasons


def test_receipts_from_different_lapse_nodes_fail_cross_consistency() -> None:
    bridge = _bridge()
    projection = _base_projection(bridge, lapse_potential=0.03)
    other_projection = _base_projection(bridge, lapse_potential=0.04)
    receipt = _gate(
        bridge,
        projection,
        energy=_energy(bridge, other_projection),
        momentum=_momentum(bridge, projection),
    )
    assert not receipt.cross_receipt_consistency_holds
    assert not receipt.algebraic_energy_momentum_node_holds
    assert "CROSS_RECEIPT_MISMATCH" in receipt.failure_reasons


def test_receipts_fail_against_a_different_bridge() -> None:
    source_bridge = _bridge(omega_prod0=0.12)
    projection = _base_projection(source_bridge)
    receipt = _gate(
        _bridge(omega_prod0=0.13),
        projection,
        energy=_energy(source_bridge, projection),
        momentum=_momentum(source_bridge, projection),
    )
    assert not receipt.cross_receipt_consistency_holds
    assert not receipt.full_declared_nondegenerate_node_holds


def test_different_pressure_perturbations_fail_same_state_gate() -> None:
    bridge = _bridge()
    projection = _base_projection(bridge)
    momentum = _momentum(
        bridge,
        projection,
        reservoir_pressure_perturbation=0.025,
    )
    receipt = _gate(
        bridge,
        projection,
        energy=_energy(bridge, projection, _momentum(bridge, projection)),
        momentum=momentum,
    )
    assert not receipt.cross_receipt_consistency_holds
    assert not receipt.full_declared_nondegenerate_node_holds
    assert "CROSS_RECEIPT_MISMATCH" in receipt.failure_reasons


def test_different_velocity_states_fail_same_state_gate() -> None:
    bridge = _bridge()
    projection = _base_projection(bridge)
    momentum = _momentum(bridge, projection)
    energy = _energy(
        bridge,
        projection,
        momentum,
        produced_theta_offset=0.1,
    )
    receipt = _gate(
        bridge,
        projection,
        energy=energy,
        momentum=momentum,
    )
    assert not receipt.cross_receipt_consistency_holds
    assert not receipt.full_declared_nondegenerate_node_holds


def test_current_bridge_q_prime_and_clock_are_recomputed() -> None:
    bridge = _bridge()
    projection = _base_projection(bridge)
    forged_q_prime = projection.produced_background_q_derivative + 1.0
    forged_delta_q = (
        forged_q_prime
        + projection.hubble_log_derivative * projection.produced_background_q
    ) * projection.scalar_clock_shift
    forged_projection = replace(
        projection,
        produced_background_q_derivative=forged_q_prime,
        produced_physical_energy_perturbation=forged_delta_q,
        reservoir_physical_energy_perturbation=-forged_delta_q,
        common_clock_physical_source_holds=True,
        all_declared_lower_component_constraints_hold=True,
    )
    receipt = _gate(bridge, forged_projection)
    assert not receipt.cross_receipt_consistency_holds
    assert not receipt.projection_common_physical_clock_holds
    assert not receipt.full_declared_nondegenerate_node_holds


def test_forged_energy_success_boolean_does_not_bypass_recomputation() -> None:
    bridge = _bridge()
    projection = _base_projection(bridge)
    momentum = _momentum(bridge, projection)
    energy = _energy(bridge, projection, momentum)
    forged_energy = replace(
        energy,
        provided_produced_density_perturbation_derivative=(
            energy.provided_produced_density_perturbation_derivative + 0.5
        ),
        produced_energy_equation_holds=True,
        both_energy_equations_hold=True,
        energy_equations_and_exchange_hold=True,
    )
    receipt = _gate(
        bridge,
        projection,
        energy=forged_energy,
        momentum=momentum,
    )
    assert not receipt.energy_equations_and_exchange_hold
    assert not receipt.full_declared_nondegenerate_node_holds


@pytest.mark.parametrize(
    "field",
    ["transfer_projection", "energy_equation", "momentum_equation"],
)
def test_gate_receipt_types_fail_closed(field: str) -> None:
    bridge = _bridge()
    projection = _base_projection(bridge)
    values = dict(
        transfer_projection=projection,
        energy_equation=_energy(bridge, projection),
        momentum_equation=_momentum(bridge, projection),
    )
    values[field] = object()
    with pytest.raises(ValueError):
        FiniteQuenchLinearSystemNodeGate(bridge).audit(**values)
