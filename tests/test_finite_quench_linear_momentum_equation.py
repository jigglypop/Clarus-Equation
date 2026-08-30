"""Focused tests for the normalized one-node linear momentum equation."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_linear_momentum_equation import (
    FiniteQuenchLinearMomentumEquation,
)
from examples.physics.finite_quench_qmu_projection_ledger import (
    FiniteQuenchLowerQmuProjectionLedger,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(
    *,
    omega_prod0: float = 0.12,
    w_reservoir: float = 0.1,
) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=omega_prod0,
            reservoir_present_density=0.21,
            w_reservoir=w_reservoir,
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


def _state(
    equation: FiniteQuenchLinearMomentumEquation,
    projection,
) -> dict[str, float]:
    momentum_p = 0.2
    momentum_r = equation.reservoir_momentum_for_total_energy_frame(
        transfer_projection=projection,
        produced_momentum_density=momentum_p,
    )
    return dict(
        produced_momentum_density=momentum_p,
        reservoir_momentum_density=momentum_r,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=0.015,
        produced_normalized_anisotropic_stress=0.02,
        reservoir_normalized_anisotropic_stress=-0.01,
    )


def _constructed(**projection_overrides: object):
    bridge = _bridge()
    projection = _projection(bridge, **projection_overrides)
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    receipt = equation.construct_required_derivatives(
        transfer_projection=projection,
        **_state(equation, projection),
    )
    return bridge, projection, equation, receipt


def test_constructed_momentum_derivatives_close_all_nondegenerate_gates() -> None:
    _bridge_value, _projection_value, _equation, receipt = _constructed()
    assert receipt.produced_momentum_equation_residual == 0.0
    assert receipt.reservoir_momentum_equation_residual == 0.0
    assert receipt.summed_momentum_equation_residual == 0.0
    assert receipt.total_momentum_transfer_source_residual == 0.0
    frame_scale = max(
        1.0,
        abs(receipt.produced_momentum_density),
        abs(receipt.reservoir_momentum_density),
        abs(
            receipt.total_background_enthalpy
            * receipt.normalized_total_velocity_potential
        ),
    )
    assert abs(receipt.total_energy_frame_momentum_residual) <= (
        64.0 * math.ulp(frame_scale)
    )
    assert receipt.momentum_equations_and_exchange_hold
    assert receipt.total_energy_frame_momentum_branch_holds
    assert receipt.source.endswith("Eq_21")
    assert receipt.role.endswith("NOT_INTEGRATED_EINSTEIN_BOLTZMANN_SOLUTION")


def test_required_produced_derivative_has_eq21_signs() -> None:
    _bridge_value, _projection_value, _equation, receipt = _constructed()
    expected = (
        receipt.produced_momentum_transfer_source
        - (3.0 - receipt.hubble_log_derivative)
        * receipt.produced_momentum_density
        - receipt.produced_background_enthalpy * receipt.lapse_potential
        - receipt.produced_pressure_perturbation
        + (2.0 / 3.0)
        * receipt.k_over_a_h**2
        * receipt.produced_normalized_anisotropic_stress
    )
    assert receipt.required_produced_momentum_density_derivative == (
        pytest.approx(expected)
    )


def test_common_total_velocity_and_intrinsic_pairs_cancel_transfer() -> None:
    _bridge_value, _projection_value, _equation, receipt = _constructed()
    assert receipt.produced_momentum_transfer_source != 0.0
    assert receipt.reservoir_momentum_transfer_source == pytest.approx(
        -receipt.produced_momentum_transfer_source
    )
    assert receipt.transfer_source_pair_cancels


def test_total_energy_frame_mismatch_is_not_hidden_by_equation_residuals() -> None:
    bridge, projection, equation, exact = _constructed()
    state = _state(equation, projection)
    state["reservoir_momentum_density"] += 0.25
    receipt = equation.construct_required_derivatives(
        transfer_projection=projection,
        **state,
    )
    assert receipt.both_momentum_equations_hold
    assert receipt.transfer_source_pair_cancels
    assert not receipt.total_energy_frame_relation_holds
    assert not receipt.total_energy_frame_momentum_branch_holds
    assert exact.total_energy_frame_momentum_branch_holds


def test_unpaired_intrinsic_transfer_breaks_summed_exchange() -> None:
    bridge = _bridge()
    paired = _projection(bridge)
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
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    receipt = equation.construct_required_derivatives(
        transfer_projection=projection,
        **_state(equation, projection),
    )
    assert receipt.both_momentum_equations_hold
    assert not receipt.transfer_source_pair_cancels
    assert not receipt.momentum_equations_and_exchange_hold
    assert not receipt.total_energy_frame_momentum_branch_holds


def test_derivative_offset_is_detected_with_its_sign() -> None:
    bridge, projection, equation, exact = _constructed()
    offset = 0.125
    receipt = equation.audit(
        transfer_projection=projection,
        **_state(equation, projection),
        produced_momentum_density_derivative=(
            exact.required_produced_momentum_density_derivative + offset
        ),
        reservoir_momentum_density_derivative=(
            exact.required_reservoir_momentum_density_derivative
        ),
    )
    assert receipt.produced_momentum_equation_residual == pytest.approx(offset)
    assert receipt.reservoir_momentum_equation_residual == 0.0
    assert receipt.summed_momentum_equation_residual == pytest.approx(offset)
    assert not receipt.both_momentum_equations_hold


def test_pressure_anisotropic_stress_and_hubble_terms_have_distinct_signs() -> None:
    bridge, projection, equation, base = _constructed()
    state = _state(equation, projection)
    state["reservoir_pressure_perturbation"] += 0.02
    state["reservoir_normalized_anisotropic_stress"] += 0.03
    changed = equation.construct_required_derivatives(
        transfer_projection=projection,
        **state,
    )
    expected_change = (
        -0.02 + (2.0 / 3.0) * projection.k_over_a_h**2 * 0.03
    )
    assert (
        changed.required_reservoir_momentum_density_derivative
        - base.required_reservoir_momentum_density_derivative
    ) == pytest.approx(expected_change)

    changed_h_projection = _projection(bridge, hubble_log_derivative=-1.1)
    changed_h_state = _state(equation, changed_h_projection)
    changed_h = equation.construct_required_derivatives(
        transfer_projection=changed_h_projection,
        **changed_h_state,
    )
    assert (
        changed_h.required_produced_momentum_density_derivative
        - base.required_produced_momentum_density_derivative
    ) == pytest.approx(0.1 * base.produced_momentum_density)


def test_k_zero_is_regular_but_not_promoted_to_physical_scalar_branch() -> None:
    _bridge_value, _projection_value, _equation, receipt = _constructed(
        k_over_a_h=0.0
    )
    assert receipt.homogeneous_fourier_mode_degenerate
    assert receipt.momentum_equations_and_exchange_hold
    assert receipt.total_energy_frame_relation_holds
    assert not receipt.total_energy_frame_momentum_branch_holds


def test_zero_total_enthalpy_is_regular_but_velocity_is_not_identifiable() -> None:
    bridge = _bridge(w_reservoir=-1.0)
    projection = _projection(
        bridge,
        n=-5.0,
        produced_intrinsic_momentum_potential=0.0,
    )
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    state = dict(
        produced_momentum_density=0.0,
        reservoir_momentum_density=0.0,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=0.0,
        produced_normalized_anisotropic_stress=0.0,
        reservoir_normalized_anisotropic_stress=0.0,
    )
    receipt = equation.construct_required_derivatives(
        transfer_projection=projection,
        **state,
    )
    assert receipt.total_background_enthalpy == 0.0
    assert not receipt.total_velocity_identifiable_from_total_enthalpy
    assert receipt.total_energy_frame_relation_holds
    assert receipt.momentum_equations_and_exchange_hold
    assert not receipt.total_energy_frame_momentum_branch_holds


def test_source_off_with_zero_intrinsic_transfer_is_uncoupled() -> None:
    bridge = _bridge()
    projection = _projection(
        bridge,
        n=-5.0,
        produced_intrinsic_momentum_potential=0.0,
    )
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    receipt = equation.construct_required_derivatives(
        transfer_projection=projection,
        **_state(equation, projection),
    )
    assert receipt.produced_momentum_transfer_source == 0.0
    assert receipt.reservoir_momentum_transfer_source == 0.0
    assert receipt.transfer_source_pair_cancels


def test_projection_from_a_different_bridge_is_rejected() -> None:
    projection = _projection(_bridge(omega_prod0=0.12))
    equation = FiniteQuenchLinearMomentumEquation(_bridge(omega_prod0=0.13))
    with pytest.raises(ValueError, match="does not match"):
        equation.construct_required_derivatives(
            transfer_projection=projection,
            **_state(
                FiniteQuenchLinearMomentumEquation(_bridge(omega_prod0=0.12)),
                projection,
            ),
        )


def test_non_receipt_projection_is_rejected() -> None:
    with pytest.raises(ValueError, match="LowerQmuProjectionReceipt"):
        FiniteQuenchLinearMomentumEquation(
            _bridge()
        ).construct_required_derivatives(
            transfer_projection=object(),
            produced_momentum_density=0.0,
            reservoir_momentum_density=0.0,
            produced_pressure_perturbation=0.0,
            reservoir_pressure_perturbation=0.0,
            produced_normalized_anisotropic_stress=0.0,
            reservoir_normalized_anisotropic_stress=0.0,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("produced_momentum_density", math.nan),
        ("reservoir_pressure_perturbation", math.inf),
        ("produced_normalized_anisotropic_stress", True),
        ("reservoir_momentum_density", 1.0e308),
    ],
)
def test_momentum_state_inputs_fail_closed(field: str, value: object) -> None:
    bridge = _bridge()
    projection = _projection(bridge)
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    state = _state(equation, projection)
    state[field] = value
    with pytest.raises(ValueError):
        equation.construct_required_derivatives(
            transfer_projection=projection,
            **state,
        )


def test_huge_kappa_anisotropic_term_fails_closed() -> None:
    bridge = _bridge()
    projection = _projection(
        bridge,
        n=-5.0,
        k_over_a_h=1.0e308,
        normalized_total_velocity_potential=0.0,
        produced_intrinsic_momentum_potential=0.0,
    )
    equation = FiniteQuenchLinearMomentumEquation(bridge)
    with pytest.raises(ValueError, match="kappa squared"):
        equation.construct_required_derivatives(
            transfer_projection=projection,
            produced_momentum_density=0.0,
            reservoir_momentum_density=0.0,
            produced_pressure_perturbation=0.0,
            reservoir_pressure_perturbation=0.0,
            produced_normalized_anisotropic_stress=1.0,
            reservoir_normalized_anisotropic_stress=-1.0,
        )
