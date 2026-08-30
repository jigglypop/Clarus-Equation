"""Focused tests for the normalized one-node linear energy equation."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_linear_energy_equation import (
    FiniteQuenchLinearEnergyEquation,
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


def _projection(bridge: FiniteQuenchBridge | None = None, **overrides: object):
    if bridge is None:
        bridge = _bridge()
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


def _state() -> dict[str, float]:
    return dict(
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=0.015,
        metric_curvature_log_derivative=-0.02,
        produced_theta_over_a_h=0.3,
        reservoir_theta_over_a_h=-0.25,
    )


def _constructed():
    bridge = _bridge()
    projection = _projection(bridge)
    receipt = FiniteQuenchLinearEnergyEquation(
        bridge
    ).construct_required_derivatives(
        transfer_projection=projection,
        **_state(),
    )
    return bridge, projection, receipt


def test_constructed_derivatives_close_both_energy_equations() -> None:
    _bridge_value, _projection_value, receipt = _constructed()
    assert receipt.produced_energy_equation_residual == 0.0
    assert receipt.reservoir_energy_equation_residual == 0.0
    assert receipt.summed_energy_equation_residual == 0.0
    assert receipt.transfer_source_pair_cancels
    assert receipt.energy_equations_and_exchange_hold
    assert receipt.common_clock_energy_branch_holds
    assert receipt.source.endswith("Eq_20")
    assert receipt.role.endswith("NOT_INTEGRATED_EINSTEIN_BOLTZMANN_SOLUTION")


def test_required_derivative_offset_is_detected_with_its_sign() -> None:
    bridge, projection, exact = _constructed()
    offset = 0.125
    audited = FiniteQuenchLinearEnergyEquation(bridge).audit(
        transfer_projection=projection,
        **_state(),
        produced_density_perturbation_derivative=(
            exact.required_produced_density_perturbation_derivative + offset
        ),
        reservoir_density_perturbation_derivative=(
            exact.required_reservoir_density_perturbation_derivative
        ),
    )
    assert audited.produced_energy_equation_residual == pytest.approx(offset)
    assert audited.reservoir_energy_equation_residual == 0.0
    assert audited.summed_energy_equation_residual == pytest.approx(offset)
    assert not audited.both_energy_equations_hold


def test_paired_transfer_disappears_only_from_the_summed_equation() -> None:
    _bridge_value, _projection_value, receipt = _constructed()
    assert receipt.produced_energy_transfer_source != 0.0
    assert receipt.reservoir_energy_transfer_source == pytest.approx(
        -receipt.produced_energy_transfer_source
    )
    assert receipt.total_energy_transfer_source_residual == 0.0
    assert receipt.required_total_density_perturbation_derivative == (
        pytest.approx(
            receipt.required_produced_density_perturbation_derivative
            + receipt.required_reservoir_density_perturbation_derivative
        )
    )


def test_produced_required_derivative_has_eq20_signs() -> None:
    _bridge_value, _projection_value, receipt = _constructed()
    expected = (
        receipt.produced_energy_transfer_source
        - 3.0
        * (
            receipt.produced_density_perturbation
            + receipt.produced_pressure_perturbation
        )
        + 3.0
        * receipt.produced_background_enthalpy
        * receipt.metric_curvature_log_derivative
        - receipt.produced_background_enthalpy
        * receipt.produced_theta_over_a_h
    )
    assert receipt.required_produced_density_perturbation_derivative == (
        pytest.approx(expected)
    )


def test_q_prime_only_sources_can_balance_but_fail_common_clock_branch() -> None:
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
    receipt = FiniteQuenchLinearEnergyEquation(
        bridge
    ).construct_required_derivatives(
        transfer_projection=projection,
        **_state(),
    )
    assert receipt.transfer_source_pair_cancels
    assert receipt.energy_equations_and_exchange_hold
    assert not receipt.transfer_projection_common_clock_holds
    assert not receipt.common_clock_energy_branch_holds


def test_unpaired_delta_q_breaks_summed_exchange_not_individual_equations() -> None:
    bridge = _bridge()
    exact = _projection(bridge)
    projection = FiniteQuenchLowerQmuProjectionLedger(bridge).audit(
        n=exact.n,
        k_over_a_h=exact.k_over_a_h,
        scalar_clock_shift=exact.scalar_clock_shift,
        hubble_log_derivative=exact.hubble_log_derivative,
        lapse_potential=exact.lapse_potential,
        normalized_total_velocity_potential=(
            exact.normalized_total_velocity_potential
        ),
        produced_physical_energy_perturbation=(
            exact.produced_physical_energy_perturbation
        ),
        reservoir_physical_energy_perturbation=0.0,
        produced_intrinsic_momentum_potential=(
            exact.produced_intrinsic_momentum_potential
        ),
        reservoir_intrinsic_momentum_potential=(
            exact.reservoir_intrinsic_momentum_potential
        ),
    )
    receipt = FiniteQuenchLinearEnergyEquation(
        bridge
    ).construct_required_derivatives(
        transfer_projection=projection,
        **_state(),
    )
    assert receipt.both_energy_equations_hold
    assert not receipt.transfer_source_pair_cancels
    assert not receipt.energy_equations_and_exchange_hold
    assert receipt.total_energy_transfer_source_residual != 0.0


def test_pressure_and_velocity_terms_change_required_derivative_separately() -> None:
    bridge = _bridge()
    projection = _projection(bridge)
    equation = FiniteQuenchLinearEnergyEquation(bridge)
    base_state = _state()
    base = equation.construct_required_derivatives(
        transfer_projection=projection,
        **base_state,
    )
    changed_state = dict(base_state)
    changed_state["reservoir_pressure_perturbation"] += 0.02
    changed_state["reservoir_theta_over_a_h"] += 0.1
    changed = equation.construct_required_derivatives(
        transfer_projection=projection,
        **changed_state,
    )
    expected_change = -3.0 * 0.02 - 0.1 * base.reservoir_background_enthalpy
    assert (
        changed.required_reservoir_density_perturbation_derivative
        - base.required_reservoir_density_perturbation_derivative
    ) == pytest.approx(expected_change)


def test_source_off_reduces_to_the_uncoupled_energy_equation() -> None:
    bridge = _bridge()
    projection = _projection(
        bridge,
        n=-5.0,
        produced_intrinsic_momentum_potential=0.0,
    )
    receipt = FiniteQuenchLinearEnergyEquation(
        bridge
    ).construct_required_derivatives(
        transfer_projection=projection,
        **_state(),
    )
    assert receipt.produced_energy_transfer_source == 0.0
    assert receipt.reservoir_energy_transfer_source == 0.0
    assert receipt.transfer_source_pair_cancels
    assert receipt.common_clock_energy_branch_holds


def test_projection_from_a_different_bridge_is_rejected() -> None:
    projection = _projection(_bridge(omega_prod0=0.12))
    equation = FiniteQuenchLinearEnergyEquation(_bridge(omega_prod0=0.13))
    with pytest.raises(ValueError, match="does not match"):
        equation.construct_required_derivatives(
            transfer_projection=projection,
            **_state(),
        )


def test_non_receipt_projection_is_rejected() -> None:
    with pytest.raises(ValueError, match="LowerQmuProjectionReceipt"):
        FiniteQuenchLinearEnergyEquation(
            _bridge()
        ).construct_required_derivatives(
            transfer_projection=object(),
            **_state(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("produced_density_perturbation", math.nan),
        ("reservoir_pressure_perturbation", math.inf),
        ("metric_curvature_log_derivative", True),
        ("produced_theta_over_a_h", 1.0e308),
    ],
)
def test_energy_state_inputs_fail_closed(field: str, value: object) -> None:
    bridge = _bridge()
    state = _state()
    state[field] = value
    with pytest.raises(ValueError):
        FiniteQuenchLinearEnergyEquation(
            bridge
        ).construct_required_derivatives(
            transfer_projection=_projection(bridge),
            **state,
        )
