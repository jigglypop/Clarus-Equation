"""Focused tests for the finite-quench lower-Qmu projection ledger."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_qmu_projection_ledger import (
    FiniteQuenchLowerQmuProjectionLedger,
    LowerQmuProjectionAxioms,
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


def _ledger() -> FiniteQuenchLowerQmuProjectionLedger:
    return FiniteQuenchLowerQmuProjectionLedger(_bridge())


def _construct(**overrides: object):
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
    return _ledger().construct_common_clock(**values)


def test_constructed_lower_components_close_all_declared_pairs() -> None:
    receipt = _construct()
    assert receipt.background_q_pair_residual == 0.0
    assert receipt.produced_physical_energy_clock_residual == 0.0
    assert receipt.reservoir_physical_energy_clock_residual == 0.0
    assert receipt.physical_energy_perturbation_pair_residual == 0.0
    assert receipt.intrinsic_momentum_pair_residual == 0.0
    assert receipt.normalized_lower_time_component_sum_residual == 0.0
    assert receipt.normalized_lower_spatial_component_sum_residual == 0.0
    assert receipt.all_declared_lower_component_constraints_hold
    assert receipt.source.endswith("Eqs_15_to_19")
    assert receipt.role.endswith(
        "NOT_MICROPHYSICAL_QMU_OR_DYNAMICAL_SOLUTION"
    )


def test_physical_rate_clock_contains_hubble_normalization_term() -> None:
    receipt = _construct()
    expected_missing = (
        receipt.hubble_log_derivative
        * receipt.produced_background_q
        * receipt.scalar_clock_shift
    )
    assert receipt.missing_hubble_normalization_term == pytest.approx(
        expected_missing
    )
    assert receipt.missing_hubble_normalization_term != 0.0
    assert receipt.q_prime_only_physical_clock_residual == pytest.approx(
        -receipt.missing_hubble_normalization_term
    )


def test_q_prime_only_pair_conserves_but_fails_physical_clock() -> None:
    ledger = _ledger()
    n = -4.0
    clock = 0.25
    naive = ledger.bridge.source_derivative(n) * clock
    receipt = ledger.audit(
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
    assert receipt.physical_energy_perturbation_pair_cancels
    assert receipt.lower_time_component_pair_cancels
    assert receipt.lower_spatial_component_pair_cancels
    assert not receipt.common_clock_physical_source_holds
    assert not receipt.all_declared_lower_component_constraints_hold


def test_q_prime_only_is_sufficient_when_hubble_is_constant() -> None:
    receipt = _construct(hubble_log_derivative=0.0)
    assert receipt.missing_hubble_normalization_term == 0.0
    assert receipt.q_prime_only_physical_clock_residual == 0.0
    assert receipt.common_clock_physical_source_holds


def test_lower_time_component_includes_the_lapse_term() -> None:
    with_lapse = _construct(lapse_potential=0.03)
    without_lapse = _construct(lapse_potential=0.0)
    expected_difference = -0.03 * with_lapse.produced_background_q
    assert (
        with_lapse.produced_normalized_lower_time_component
        - without_lapse.produced_normalized_lower_time_component
    ) == pytest.approx(expected_difference)
    assert with_lapse.lower_time_component_pair_cancels


def test_spatial_projection_contains_energy_carried_by_total_velocity() -> None:
    receipt = _construct(
        k_over_a_h=2.0,
        normalized_total_velocity_potential=0.4,
        produced_intrinsic_momentum_potential=0.0,
    )
    assert receipt.produced_normalized_spatial_bracket == pytest.approx(
        receipt.produced_background_q * 0.4
    )
    assert receipt.produced_normalized_lower_spatial_fourier_scalar == (
        pytest.approx(2.0 * receipt.produced_normalized_spatial_bracket)
    )
    assert receipt.lower_spatial_component_pair_cancels


def test_zero_wavenumber_removes_component_not_the_spatial_bracket() -> None:
    receipt = _construct(k_over_a_h=0.0)
    assert receipt.produced_normalized_spatial_bracket != 0.0
    assert receipt.produced_normalized_lower_spatial_fourier_scalar == 0.0
    assert receipt.reservoir_normalized_lower_spatial_fourier_scalar == 0.0
    assert receipt.lower_spatial_component_pair_cancels


def test_unpaired_energy_perturbation_breaks_source_and_time_pairs() -> None:
    exact = _construct()
    receipt = _ledger().audit(
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
    assert not receipt.physical_energy_perturbation_pair_cancels
    assert not receipt.lower_time_component_pair_cancels
    assert not receipt.all_declared_lower_component_constraints_hold


def test_unpaired_intrinsic_momentum_breaks_nonzero_k_spatial_pair() -> None:
    exact = _construct()
    receipt = _ledger().audit(
        n=exact.n,
        k_over_a_h=2.0,
        scalar_clock_shift=exact.scalar_clock_shift,
        hubble_log_derivative=exact.hubble_log_derivative,
        lapse_potential=exact.lapse_potential,
        normalized_total_velocity_potential=(
            exact.normalized_total_velocity_potential
        ),
        produced_physical_energy_perturbation=(
            exact.produced_physical_energy_perturbation
        ),
        reservoir_physical_energy_perturbation=(
            exact.reservoir_physical_energy_perturbation
        ),
        produced_intrinsic_momentum_potential=0.07,
        reservoir_intrinsic_momentum_potential=0.0,
    )
    assert not receipt.intrinsic_momentum_pair_cancels
    assert not receipt.lower_spatial_component_pair_cancels
    assert not receipt.all_declared_lower_component_constraints_hold


def test_zero_k_does_not_hide_an_unpaired_intrinsic_momentum_input() -> None:
    exact = _construct(k_over_a_h=0.0)
    receipt = _ledger().audit(
        n=exact.n,
        k_over_a_h=0.0,
        scalar_clock_shift=exact.scalar_clock_shift,
        hubble_log_derivative=exact.hubble_log_derivative,
        lapse_potential=exact.lapse_potential,
        normalized_total_velocity_potential=(
            exact.normalized_total_velocity_potential
        ),
        produced_physical_energy_perturbation=(
            exact.produced_physical_energy_perturbation
        ),
        reservoir_physical_energy_perturbation=(
            exact.reservoir_physical_energy_perturbation
        ),
        produced_intrinsic_momentum_potential=0.07,
        reservoir_intrinsic_momentum_potential=0.0,
    )
    assert receipt.lower_spatial_component_pair_cancels
    assert not receipt.intrinsic_momentum_pair_cancels
    assert not receipt.all_declared_lower_component_constraints_hold


def test_source_off_clock_projection_is_regular() -> None:
    receipt = _construct(
        n=-5.0,
        produced_intrinsic_momentum_potential=0.0,
    )
    assert receipt.produced_background_q == 0.0
    assert receipt.produced_background_q_derivative == 0.0
    assert receipt.produced_physical_energy_perturbation == 0.0
    assert receipt.produced_normalized_lower_time_component == 0.0
    assert receipt.produced_normalized_lower_spatial_fourier_scalar == 0.0
    assert receipt.all_declared_lower_component_constraints_hold


@pytest.mark.parametrize(
    "overrides",
    [
        dict(n=0.1),
        dict(k_over_a_h=-0.1),
        dict(scalar_clock_shift=True),
        dict(hubble_log_derivative=math.nan),
        dict(lapse_potential=math.inf),
        dict(normalized_total_velocity_potential=1.0e308),
        dict(hubble_log_derivative=1.0e308),
    ],
)
def test_projection_inputs_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _construct(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gauge="newtonian"),
        dict(metric_signature="plus_minus_minus_minus"),
        dict(density_normalization="time_dependent_rho_unit"),
        dict(source_identification="q_equals_aQ_over_H"),
        dict(time_shift_convention="opposite_sign"),
        dict(fourier_convention="exp_minus_i_k_dot_x"),
    ],
)
def test_projection_axioms_fail_closed(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        LowerQmuProjectionAxioms(**kwargs)
