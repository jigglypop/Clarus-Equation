"""Focused tests for compact-bin primordial-curvature normalization."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    construct_compact_kappa_bin_weyl_transfer_enclosure,
)
from examples.physics.finite_quench_primordial_curvature_normalization import (
    _source_off_normalized_comoving_curvature_interval,
    normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature,
)
from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.finite_quench_source_plane_harmonic_transfer import (
    project_compact_kappa_bin_to_source_plane_harmonic_transfer,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    FiniteQuenchTraceEndpointEnclosure,
    _RationalInterval,
    _interval_divide,
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


@pytest.fixture(scope="module")
def compact_bin():
    evolution = FiniteQuenchRegularMetricEvolution(
        _bridge(),
        n_initial=-5.0,
        kappa_initial=0.05,
    )
    return construct_compact_kappa_bin_weyl_transfer_enclosure(
        FiniteQuenchTraceEndpointEnclosure(evolution),
        initial_kappa_lower=Fraction(49, 1000),
        initial_kappa_upper=Fraction(51, 1000),
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
    )


@pytest.fixture(scope="module")
def harmonic_transfer(compact_bin):
    return project_compact_kappa_bin_to_source_plane_harmonic_transfer(
        compact_bin,
        ell=2,
    )


@pytest.fixture(scope="module")
def normalized_receipt(compact_bin, harmonic_transfer):
    return normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
        compact_bin,
        harmonic_transfer,
    )


def test_matter_and_radiation_superhorizon_normalizations_are_exact() -> None:
    unit = _RationalInterval(Fraction(1), Fraction(1))
    zero = _RationalInterval(Fraction(0), Fraction(0))

    matter = _source_off_normalized_comoving_curvature_interval(
        normalized_curvature=unit,
        normalized_curvature_prime=zero,
        hubble_log_derivative=Fraction(-3, 2),
    )
    radiation = _source_off_normalized_comoving_curvature_interval(
        normalized_curvature=unit,
        normalized_curvature_prime=zero,
        hubble_log_derivative=Fraction(-2),
    )

    assert matter.lower <= Fraction(5, 3) <= matter.upper
    assert radiation.lower <= Fraction(3, 2) <= radiation.upper
    with pytest.raises(ValueError, match="negative hubble log derivative"):
        _source_off_normalized_comoving_curvature_interval(
            normalized_curvature=unit,
            normalized_curvature_prime=zero,
            hubble_log_derivative=Fraction(3, 2),
        )


@pytest.mark.parametrize("kappa_squared", [Fraction(0), Fraction(1, 7), Fraction(9, 2)])
def test_source_off_constraints_cancel_explicit_kappa_term(
    kappa_squared: Fraction,
) -> None:
    psi = Fraction(7, 5)
    psi_prime = Fraction(-2, 9)
    w = Fraction(1, 10)
    g = 3 * (1 + w) / 2
    clock = (psi_prime + (1 + kappa_squared / 3) * psi) / g
    velocity = kappa_squared * psi / (3 * g) - clock
    direct = psi - velocity
    reduced = psi + (psi_prime + psi) / g

    assert direct == reduced


def test_default_bin_normalizes_harmonic_transfer_to_positive_R(
    normalized_receipt,
) -> None:
    receipt = normalized_receipt
    lower, upper = receipt.compact_bin_normalized_comoving_curvature_interval
    transfer_lower, transfer_upper = (
        receipt
        .normalized_convergence_harmonic_transfer_per_comoving_curvature_interval
    )

    assert receipt.source_off_hubble_log_derivative == (
        -3 * (1 + receipt.reservoir_equation_of_state) / 2
    )
    assert float(lower) == pytest.approx(
        1.605939349917946,
        rel=0,
        abs=5.0e-16,
    )
    assert float(upper) == pytest.approx(
        1.6059487676443964,
        rel=0,
        abs=5.0e-16,
    )
    assert receipt.compact_bin_normalized_comoving_curvature_certified_sign == 1
    assert float(transfer_lower) == pytest.approx(
        -0.18826492056862554,
        rel=0,
        abs=5.0e-17,
    )
    assert float(transfer_upper) == pytest.approx(
        -0.05532901417812093,
        rel=0,
        abs=5.0e-17,
    )
    assert (
        receipt
        .normalized_convergence_harmonic_transfer_per_comoving_curvature_sign
        == -1
    )
    assert receipt.zero_k_potential_to_comoving_curvature_ratio == (
        3 * (1 + receipt.reservoir_equation_of_state)
        / (5 + 3 * receipt.reservoir_equation_of_state)
    )
    assert receipt.matter_era_potential_to_comoving_curvature_ratio == Fraction(
        3,
        5,
    )


def test_initial_parameter_radius_widens_both_state_components(
    normalized_receipt,
) -> None:
    receipt = normalized_receipt
    psi_radius = (
        receipt
        .normalized_initial_curvature_parameter_variation_radius_upper_bound
    )
    prime_radius = (
        receipt
        .normalized_initial_curvature_prime_parameter_variation_radius_upper_bound
    )
    psi_center = receipt.central_normalized_curvature_interval
    psi_bin = receipt.compact_bin_normalized_curvature_interval
    prime_center = receipt.central_normalized_curvature_prime_interval
    prime_bin = receipt.compact_bin_normalized_curvature_prime_interval

    assert psi_radius > 0
    assert prime_radius > 0
    assert psi_bin[0] <= psi_center[0] - psi_radius
    assert psi_bin[1] >= psi_center[1] + psi_radius
    assert prime_bin[0] <= prime_center[0] - prime_radius
    assert prime_bin[1] >= prime_center[1] + prime_radius


def test_signed_interval_quotient_is_ordered_and_rejects_zero_denominator() -> None:
    quotient = _interval_divide(
        _RationalInterval(Fraction(-3), Fraction(-1)),
        _RationalInterval(Fraction(2), Fraction(4)),
    )

    assert quotient.lower <= Fraction(-3, 2)
    assert quotient.upper >= Fraction(-1, 4)
    assert quotient.upper < 0
    with pytest.raises(ValueError, match="division crossed zero"):
        _interval_divide(
            _RationalInterval(Fraction(-3), Fraction(-1)),
            _RationalInterval(Fraction(-1), Fraction(1)),
        )


def test_receipt_locks_conventions_proofs_and_nonclaims(normalized_receipt) -> None:
    receipt = normalized_receipt

    assert receipt.newtonian_gauge_metric_minus_two_psi_spatial_convention_adopted
    assert receipt.velocity_divergence_theta_equals_minus_k_squared_v_adopted
    assert receipt.cmb_positive_primordial_comoving_curvature_convention_adopted
    assert receipt.hubble_log_derivative_sign_and_definition_locked
    assert receipt.source_off_constraint_explicit_kappa_term_cancellation_proven
    assert receipt.exact_regular_source_off_adiabatic_mode_reused
    assert receipt.initial_superhorizon_compact_bin_proven
    assert receipt.initial_parameter_sensitivity_component_enclosure_reused
    assert (
        receipt
        .coordinatewise_initial_parameter_sensitivity_reconstructed_from_series
    )
    assert receipt.finite_initial_comoving_curvature_normalization_enclosed
    assert receipt.matter_era_three_fifths_normalization_recovered
    assert receipt.exact_rational_outward_interval_operations_used
    assert receipt.numerator_denominator_correlation_discarded_conservatively
    assert receipt.dimensionless_normalization_proven
    assert receipt.compact_bin_harmonic_transfer_per_comoving_curvature_enclosed
    assert not receipt.physical_wavenumber_bin_calibrated
    assert not receipt.primordial_curvature_power_spectrum_supplied
    assert not receipt.inflationary_state_or_spectrum_derived
    assert not receipt.all_k_einstein_boltzmann_transfer_enclosed
    assert not receipt.source_population_distribution_supplied
    assert not receipt.post_born_or_relativistic_corrections_enclosed
    assert not receipt.angular_power_spectrum_enclosed
    assert not receipt.covariance_or_likelihood_enclosed


def test_zero_amplitude_and_incomplete_proofs_fail_closed(
    compact_bin,
    harmonic_transfer,
) -> None:
    with pytest.raises(ValueError, match="nonzero A"):
        normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
            replace(compact_bin, primordial_potential_amplitude=Fraction(0)),
            harmonic_transfer,
        )
    with pytest.raises(ValueError, match="proof prerequisites"):
        normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
            replace(
                compact_bin,
                exact_regular_series_parameter_derivative_conservatively_enclosed=(
                    False
                ),
            ),
            harmonic_transfer,
        )


def test_mismatched_harmonic_provenance_fails_closed(
    compact_bin,
    harmonic_transfer,
) -> None:
    mismatched = replace(
        harmonic_transfer,
        initial_kappa_interval=(Fraction(1, 20), Fraction(51, 1000)),
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
            compact_bin,
            mismatched,
        )


def test_zero_crossing_comoving_curvature_fails_closed(
    compact_bin,
    harmonic_transfer,
) -> None:
    bridge = compact_bin.central_trace_receipt.regular_initial_bridge
    exact = replace(
        bridge.regular_mode_enclosure,
        curvature_interval=(Fraction(0), Fraction(0)),
        curvature_prime_interval=(Fraction(0), Fraction(0)),
    )
    changed_bridge = replace(bridge, regular_mode_enclosure=exact)
    changed_trace = replace(
        compact_bin.central_trace_receipt,
        regular_initial_bridge=changed_bridge,
    )
    zero_curvature = replace(compact_bin, central_trace_receipt=changed_trace)

    with pytest.raises(ValueError, match="crossed zero"):
        normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
            zero_curvature,
            harmonic_transfer,
        )
