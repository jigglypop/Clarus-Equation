"""Focused tests for the compact-bin full-sky source-plane transfer."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math

import pytest

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    construct_compact_kappa_bin_weyl_transfer_enclosure,
)
from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.finite_quench_source_plane_harmonic_transfer import (
    _spherical_bessel_over_x_interval,
    project_compact_kappa_bin_to_source_plane_harmonic_transfer,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    FiniteQuenchTraceEndpointEnclosure,
    _RationalInterval,
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


def _spherical_bessel(ell: int, value: float) -> float:
    if value == 0:
        return 1.0 if ell == 0 else 0.0
    odd_double_factorial = math.prod(range(1, 2 * ell + 2, 2))
    term = value**ell / odd_double_factorial
    terms = [term]
    for order in range(1, 64):
        term *= -(value * value) / (
            2 * order * (2 * ell + 2 * order + 1)
        )
        terms.append(term)
    return math.fsum(terms)


@pytest.mark.parametrize("ell", [2, 3, 4])
@pytest.mark.parametrize("argument", [Fraction(1, 2), Fraction(1), Fraction(2)])
def test_exact_bessel_ratio_interval_contains_reference(
    ell: int,
    argument: Fraction,
) -> None:
    interval, tail, ratio = _spherical_bessel_over_x_interval(
        ell=ell,
        argument=_RationalInterval(argument, argument),
        highest_partial_sum_order=16,
    )
    reference = _spherical_bessel(ell, float(argument)) / float(argument)
    float_roundoff = 8 * math.ulp(reference)

    assert float(interval.lower) - float_roundoff <= reference
    assert reference <= float(interval.upper) + float_roundoff
    assert interval.lower >= 0
    assert tail >= 0
    assert ratio <= Fraction(2, 7)


def test_default_ell_two_transfer_is_finite_negative_interval(
    compact_bin,
) -> None:
    receipt = project_compact_kappa_bin_to_source_plane_harmonic_transfer(
        compact_bin,
        ell=2,
    )

    assert receipt.maximum_dimensionless_bessel_argument_upper_bound < 2
    assert all(
        lower >= 0
        for lower, _ in receipt.spherical_bessel_over_x_cell_intervals
    )
    assert receipt.dimensionless_source_fraction_cell_intervals[0][0] == 0
    assert receipt.dimensionless_source_fraction_cell_intervals[-1][1] == 1
    lower, upper = receipt.normalized_convergence_harmonic_transfer_interval
    assert float(lower) == pytest.approx(-0.30234204415033217)
    assert float(upper) == pytest.approx(-0.08885556209436223)
    assert receipt.normalized_convergence_harmonic_transfer_certified_sign == -1


def test_harmonic_receipt_locks_orientation_factors_and_nonclaims(
    compact_bin,
) -> None:
    receipt = project_compact_kappa_bin_to_source_plane_harmonic_transfer(
        compact_bin,
        ell=2,
    )

    assert receipt.four_pi_i_ell_plane_wave_harmonic_convention_adopted
    assert receipt.source_to_observer_mesh_orientation_fixed
    assert receipt.positive_conformal_measure_reversed_to_observer_radial_integral
    assert receipt.lensing_potential_minus_two_convention_adopted
    assert receipt.convergence_minus_half_angular_laplacian_adopted
    assert receipt.lensing_factor_two_cancellation_proven
    assert receipt.observer_inverse_distance_singularity_removed_by_bessel_ratio
    assert receipt.all_bessel_arguments_within_exact_series_domain
    assert receipt.exact_rational_decreasing_alternating_bessel_series_enclosed
    assert receipt.spherical_bessel_over_x_nonnegative_on_certified_domain_proven
    assert receipt.binwide_not_nodewise_weyl_envelopes_used
    assert receipt.compact_kappa_bin_source_plane_harmonic_transfer_enclosed
    assert receipt.normalized_by_free_potential_amplitude
    assert not receipt.physical_wavenumber_bin_calibrated
    assert not receipt.primordial_curvature_to_potential_normalization_supplied
    assert not receipt.primordial_curvature_power_spectrum_supplied
    assert not receipt.all_k_einstein_boltzmann_transfer_enclosed
    assert not receipt.source_population_distribution_supplied
    assert not receipt.post_born_or_relativistic_corrections_enclosed
    assert not receipt.angular_power_spectrum_enclosed
    assert not receipt.covariance_or_likelihood_enclosed


def test_invalid_harmonic_and_bessel_domain_fail_closed(compact_bin) -> None:
    with pytest.raises(ValueError, match="ell >= 2"):
        project_compact_kappa_bin_to_source_plane_harmonic_transfer(
            compact_bin,
            ell=1,
        )
    with pytest.raises(ValueError, match="partial-sum order"):
        project_compact_kappa_bin_to_source_plane_harmonic_transfer(
            compact_bin,
            ell=2,
            bessel_highest_partial_sum_order=65,
        )

    overwide = replace(
        compact_bin,
        initial_kappa_interval=(Fraction(49, 1000), Fraction(1, 10)),
    )
    with pytest.raises(ValueError, match="Bessel argument domain"):
        project_compact_kappa_bin_to_source_plane_harmonic_transfer(
            overwide,
            ell=2,
        )


def test_zero_amplitude_normalization_cannot_be_projected(compact_bin) -> None:
    unnormalized = replace(
        compact_bin,
        normalized_compact_bin_weyl_average_cell_intervals=None,
    )
    with pytest.raises(ValueError, match="nonzero amplitude normalization"):
        project_compact_kappa_bin_to_source_plane_harmonic_transfer(
            unnormalized,
            ell=2,
        )


def test_unproven_compact_bin_cannot_be_projected(compact_bin) -> None:
    unproven = replace(
        compact_bin,
        cellwise_duhamel_sensitivity_recurrence_proven=False,
    )
    with pytest.raises(ValueError, match="proof prerequisites"):
        project_compact_kappa_bin_to_source_plane_harmonic_transfer(
            unproven,
            ell=2,
        )
