"""Focused tests for source-off super-horizon regular variables and modes."""

from __future__ import annotations

from fractions import Fraction
import math

import pytest

from examples.physics.finite_quench_superhorizon_regularity import (
    FiniteQuenchSuperhorizonRegularity,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(*, w_reservoir: float = 0.1) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=w_reservoir,
            w_open=2.1767e-4,
        )
    )


def _mode(*, w: float = 0.1, kappa: float = 0.05, amplitude: float = 1.0e-5):
    bridge = _bridge(w_reservoir=w)
    gate = FiniteQuenchSuperhorizonRegularity(bridge)
    return bridge, gate, gate.construct_regular_mode(
        n=-5.0,
        k_over_a_h=kappa,
        primordial_potential_amplitude=amplitude,
    )


def test_regularized_system_and_past_bounded_mode_close_full_chain() -> None:
    _, _, receipt = _mode()
    system = receipt.regular_system
    assert system.source_off_pure_reservoir_holds
    assert system.raw_numerator_to_curvature_identity_holds
    assert system.transformed_clock_equation_holds
    assert system.transformed_curvature_equation_holds
    assert system.perfect_fluid_potential_equation_holds
    assert system.k_zero_eigenpairs_hold
    assert system.no_forward_growing_superhorizon_mode
    assert system.bounded_past_mode_dimension == 1
    assert system.produced_intrinsic_force_vanishes
    assert system.full_superhorizon_regular_system_holds
    assert receipt.past_bounded_regular_series_holds
    assert receipt.zero_decaying_mode_selected
    assert receipt.constraint_compatible_mode_holds
    assert receipt.full_regular_mode_holds
    assert not receipt.failure_reasons


def test_transformed_matrix_has_no_inverse_kappa_powers() -> None:
    _, _, receipt = _mode()
    system = receipt.regular_system
    g = system.gravity_enthalpy_coupling
    k = system.k_over_a_h_squared
    assert system.transformed_matrix_b11 == pytest.approx(-g + k / 3.0)
    assert system.transformed_matrix_b12 == pytest.approx(
        1.0 + k / 3.0 - k**2 / (9.0 * g)
    )
    assert system.transformed_matrix_b21 == pytest.approx(g)
    assert system.transformed_matrix_b22 == pytest.approx(-(1.0 + k / 3.0))


@pytest.mark.parametrize(
    ("w", "g", "lambda_decay", "kappa_power"),
    [
        (0.0, 1.5, -2.5, 5.0),
        (0.1, 1.65, -2.65, 53.0 / 13.0),
        (1.0 / 3.0, 2.0, -3.0, 3.0),
        (1.0, 3.0, -4.0, 2.0),
    ],
)
def test_k_zero_modes_match_standard_perfect_fluid_exponents(
    w: float,
    g: float,
    lambda_decay: float,
    kappa_power: float,
) -> None:
    _, _, receipt = _mode(w=w)
    system = receipt.regular_system
    assert system.gravity_enthalpy_coupling == pytest.approx(g)
    assert system.adiabatic_limit_eigenvalue == 0.0
    assert system.decaying_limit_eigenvalue == pytest.approx(lambda_decay)
    assert system.decaying_kappa_power == pytest.approx(kappa_power)
    assert system.k_zero_eigenpairs_hold
    assert system.no_forward_growing_superhorizon_mode


@pytest.mark.parametrize("w", [0.0, 0.1, 1.0 / 3.0, 1.0])
def test_past_bounded_series_solves_full_potential_equation(w: float) -> None:
    _, _, receipt = _mode(w=w)
    assert receipt.series_potential_equation_residual == pytest.approx(
        0.0,
        abs=2.0e-20,
    )
    assert receipt.full_regular_mode_holds


def test_radiation_regular_series_starts_with_minus_kappa_squared_over_30() -> None:
    _, _, receipt = _mode(w=1.0 / 3.0, amplitude=1.0)
    assert receipt.first_kappa_squared_series_coefficient == pytest.approx(
        -1.0 / 30.0
    )
    k = receipt.regular_system.k_over_a_h_squared
    assert receipt.series_curvature_potential == pytest.approx(
        1.0 - k / 30.0,
        rel=0.0,
        abs=2.0 * k**2,
    )


def test_dust_regular_series_is_exactly_constant() -> None:
    _, _, receipt = _mode(w=0.0, amplitude=0.7)
    assert receipt.first_kappa_squared_series_coefficient == 0.0
    assert receipt.series_curvature_potential == pytest.approx(0.7)
    assert receipt.series_curvature_log_derivative == pytest.approx(0.0)
    assert receipt.series_curvature_second_log_derivative == pytest.approx(0.0)


def test_exact_regular_initial_enclosure_has_rational_tail_proof() -> None:
    _, gate, floating = _mode()
    exact_8 = gate.construct_exact_regular_initial_enclosure(
        n=-5.0,
        k_over_a_h=0.05,
        primordial_potential_amplitude=1.0e-5,
        highest_partial_sum_order=8,
    )
    exact_16 = gate.construct_exact_regular_initial_enclosure(
        n=-5.0,
        k_over_a_h=0.05,
        primordial_potential_amplitude=1.0e-5,
        highest_partial_sum_order=16,
    )

    assert exact_16.curvature_interval[0] >= exact_8.curvature_interval[0]
    assert exact_16.curvature_interval[1] <= exact_8.curvature_interval[1]
    assert (
        exact_16.curvature_prime_interval[0]
        >= exact_8.curvature_prime_interval[0]
    )
    assert (
        exact_16.curvature_prime_interval[1]
        <= exact_8.curvature_prime_interval[1]
    )
    assert exact_16.curvature_tail_abs_upper_bound > 0
    assert exact_16.curvature_prime_tail_abs_upper_bound > 0
    assert (
        exact_16.curvature_tail_ratio_upper_bound
        < exact_8.curvature_tail_ratio_upper_bound
        < 1
    )
    assert floating.next_series_term_bound == 0.0
    assert exact_16.exact_binary_float_inputs_frozen
    assert exact_16.source_off_pure_reservoir_series_equation_proven
    assert exact_16.exact_series_recurrence_proven
    assert exact_16.tail_ratios_monotone_and_strictly_below_one
    assert exact_16.exact_rational_tail_enclosures_proven
    assert exact_16.unique_past_bounded_regular_mode_enclosed
    assert exact_16.normalized_dimensionless_series_proven
    assert exact_16.potential_amplitude_is_free_initial_data
    assert not exact_16.physical_primordial_amplitude_supplied
    assert not exact_16.scalar_clock_initial_interval_enclosed


def test_exact_regular_recurrence_first_term_is_algebraically_exact() -> None:
    _, gate, _ = _mode()
    exact = gate.construct_exact_regular_initial_enclosure(
        n=-5.0,
        k_over_a_h=0.05,
        primordial_potential_amplitude=1.0e-5,
        highest_partial_sum_order=1,
    )
    amplitude = exact.primordial_potential_amplitude
    coupling = (
        exact.reservoir_equation_of_state
        * exact.kappa_initial_squared
    )
    expected_first = -(
        amplitude
        * coupling
        / (
            exact.exponential_rate
            * (
                exact.exponential_rate
                + exact.potential_friction
            )
        )
    )
    assert exact.curvature_partial_sum == amplitude + expected_first
    assert (
        exact.curvature_prime_partial_sum
        == exact.exponential_rate * expected_first
    )


def test_exact_dust_regular_initial_interval_is_a_point() -> None:
    _, gate, _ = _mode(w=0.0, amplitude=0.7)
    exact = gate.construct_exact_regular_initial_enclosure(
        n=-5.0,
        k_over_a_h=0.05,
        primordial_potential_amplitude=0.7,
    )
    amplitude = Fraction.from_float(0.7)
    assert exact.curvature_interval == (amplitude, amplitude)
    assert exact.curvature_prime_interval == (Fraction(0), Fraction(0))
    assert exact.curvature_tail_abs_upper_bound == 0
    assert exact.curvature_prime_tail_abs_upper_bound == 0


def test_regular_numerator_cancels_to_kappa_squared_over_coupling() -> None:
    _, _, receipt = _mode()
    system = receipt.regular_system
    coupling = system.background.gravity_constraint_coupling
    expected_j = (
        system.k_over_a_h_squared
        * receipt.provided_curvature_potential
        / (3.0 * coupling)
    )
    assert system.cancellation_numerator_j == pytest.approx(expected_j)
    assert receipt.regular_numerator_residual == pytest.approx(0.0, abs=1.0e-12)


def test_arbitrary_raw_t_u_basis_exhibits_coordinate_one_over_kappa_squared() -> None:
    bridge = _bridge()
    gate = FiniteQuenchSuperhorizonRegularity(bridge)
    coarse = gate.construct_system(
        n=-5.0,
        k_over_a_h=0.1,
        scalar_clock_shift=0.0,
        total_momentum_density=1.0e-6,
    )
    fine = gate.construct_system(
        n=-5.0,
        k_over_a_h=0.05,
        scalar_clock_shift=0.0,
        total_momentum_density=1.0e-6,
    )
    assert fine.curvature_potential == pytest.approx(
        4.0 * coarse.curvature_potential
    )
    assert coarse.full_superhorizon_regular_system_holds
    assert fine.full_superhorizon_regular_system_holds


def test_regular_series_is_linear_in_the_free_potential_amplitude() -> None:
    _, _, base = _mode(amplitude=2.0e-5)
    factor = -3.0
    _, _, scaled = _mode(amplitude=factor * 2.0e-5)
    assert scaled.series_curvature_potential == pytest.approx(
        factor * base.series_curvature_potential
    )
    assert scaled.required_scalar_clock_shift == pytest.approx(
        factor * base.required_scalar_clock_shift
    )
    assert scaled.required_total_momentum_density == pytest.approx(
        factor * base.required_total_momentum_density
    )


def test_wrong_clock_candidate_is_falsified_against_regular_mode() -> None:
    _, gate, receipt = _mode()
    bad = gate.audit_regular_mode(
        n=receipt.regular_system.n,
        k_over_a_h=receipt.regular_system.k_over_a_h,
        primordial_potential_amplitude=receipt.primordial_potential_amplitude,
        scalar_clock_shift=receipt.required_scalar_clock_shift + 0.01,
        total_momentum_density=receipt.required_total_momentum_density,
        curvature_potential=receipt.series_curvature_potential,
        curvature_potential_log_derivative=(
            receipt.series_curvature_log_derivative
        ),
        curvature_potential_second_log_derivative=(
            receipt.series_curvature_second_log_derivative
        ),
    )
    assert not bad.constraint_compatible_mode_holds
    assert not bad.full_regular_mode_holds
    assert "REGULAR_CLOCK_MODE_FAILED" in bad.failure_reasons


def test_wrong_metric_second_candidate_is_falsified_against_series() -> None:
    _, gate, receipt = _mode()
    bad = gate.audit_regular_mode(
        n=receipt.regular_system.n,
        k_over_a_h=receipt.regular_system.k_over_a_h,
        primordial_potential_amplitude=receipt.primordial_potential_amplitude,
        scalar_clock_shift=receipt.required_scalar_clock_shift,
        total_momentum_density=receipt.required_total_momentum_density,
        curvature_potential=receipt.series_curvature_potential,
        curvature_potential_log_derivative=(
            receipt.series_curvature_log_derivative
        ),
        curvature_potential_second_log_derivative=(
            receipt.series_curvature_second_log_derivative + 0.01
        ),
    )
    assert not bad.constraint_compatible_mode_holds
    assert "REGULAR_METRIC_SERIES_FAILED" in bad.failure_reasons


def test_source_on_or_post_source_nodes_are_rejected() -> None:
    gate = FiniteQuenchSuperhorizonRegularity(_bridge())
    for n in (-4.0, -3.0):
        with pytest.raises(ValueError, match="pre-source"):
            gate.construct_regular_mode(
                n=n,
                k_over_a_h=0.05,
                primordial_potential_amplitude=1.0e-5,
            )


@pytest.mark.parametrize("kappa", [0.0, -0.1, 0.11, math.inf, math.nan, True])
def test_non_superhorizon_or_nonfinite_kappa_is_rejected(kappa) -> None:
    gate = FiniteQuenchSuperhorizonRegularity(_bridge())
    with pytest.raises(ValueError):
        gate.construct_regular_mode(
            n=-5.0,
            k_over_a_h=kappa,
            primordial_potential_amplitude=1.0e-5,
        )


def test_receipt_does_not_claim_source_matching_or_primordial_prediction() -> None:
    _, _, receipt = _mode()
    system = receipt.regular_system
    assert not system.source_on_matching_proven
    assert not system.subhorizon_stability_proven
    assert not system.primordial_spectrum_supplied
    assert not system.microphysical_covariant_transfer_law_proven
    assert receipt.potential_amplitude_is_free_initial_data
    assert not receipt.primordial_amplitude_predicted
    assert not receipt.finite_time_source_on_evolution_certified
    assert "PAST_BOUNDED_SUPERHORIZON_ADIABATIC_MODE" in receipt.role
