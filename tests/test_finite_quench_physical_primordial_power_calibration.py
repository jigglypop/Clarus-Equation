"""Focused tests for theorem-38 physical-k primordial power calibration."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    construct_compact_kappa_bin_weyl_transfer_enclosure,
)
from examples.physics.finite_quench_physical_primordial_power_calibration import (
    ANGULAR_COMOVING_FOURIER_CONVENTION,
    HUBBLE_REFERENCE_UNITS,
    INTERNAL_DENSITY_HUBBLE_NORMALIZATION,
    PHYSICAL_WAVENUMBER_UNITS,
    SPEED_OF_LIGHT_KM_S,
    ExternalPhysicalWavenumberPivotCalibration,
    ExternalScalarPowerLawPrimordialSpectrum,
    construct_physically_calibrated_primordial_power_cell,
    enclose_positive_rational_signed_log,
    enclose_rational_exponential,
)
from examples.physics.finite_quench_primordial_curvature_normalization import (
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
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


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
def curvature_transfer():
    evolution = FiniteQuenchRegularMetricEvolution(
        _bridge(),
        n_initial=-5.0,
        kappa_initial=0.05,
    )
    compact = construct_compact_kappa_bin_weyl_transfer_enclosure(
        FiniteQuenchTraceEndpointEnclosure(evolution),
        initial_kappa_lower=Fraction(49, 1000),
        initial_kappa_upper=Fraction(51, 1000),
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
    )
    harmonic = project_compact_kappa_bin_to_source_plane_harmonic_transfer(
        compact,
        ell=2,
    )
    return normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
        compact,
        harmonic,
    )


def _calibration(curvature_transfer):
    q_bin = (
        curvature_transfer.source_plane_harmonic_transfer_receipt
        .dimensionless_fixed_wavenumber_interval
    )
    return ExternalPhysicalWavenumberPivotCalibration.freeze(
        dimensionless_fixed_wavenumber_interval=q_bin,
        hubble_reference_km_s_mpc_interval=(
            Fraction(337, 5),
            Fraction(337, 5),
        ),
        pivot_wavenumber_mpc_inverse_interval=(
            Fraction(1, 20),
            Fraction(1, 20),
        ),
        calibration_reference="illustrative external H_ref=67.4; not CE",
        hubble_reference_units=HUBBLE_REFERENCE_UNITS,
        pivot_wavenumber_units=PHYSICAL_WAVENUMBER_UNITS,
        fourier_convention=ANGULAR_COMOVING_FOURIER_CONVENTION,
        density_hubble_normalization=INTERNAL_DENSITY_HUBBLE_NORMALIZATION,
        scale_factor_today_equals_one_adopted=True,
        fixed_mode_independent_hubble_reference_adopted=True,
        internal_density_hubble_normalization_adopted=True,
    )


def _primordial_parameters():
    return ExternalScalarPowerLawPrimordialSpectrum.freeze(
        scalar_amplitude_interval=(
            Fraction(21, 10_000_000_000),
            Fraction(21, 10_000_000_000),
        ),
        scalar_spectral_index_interval=(
            Fraction(193, 200),
            Fraction(193, 200),
        ),
        pivot_wavenumber_mpc_inverse_interval=(
            Fraction(1, 20),
            Fraction(1, 20),
        ),
        parameter_reference="illustrative external A_s,n_s; not CE",
        dimensionless_mathcal_p_r_convention_adopted=True,
        zero_running_power_law_model_adopted=True,
        parameters_are_binwide_not_nodewise=True,
    )


@pytest.fixture(scope="module")
def calibrated_cell(curvature_transfer):
    return construct_physically_calibrated_primordial_power_cell(
        curvature_transfer,
        _calibration(curvature_transfer),
        _primordial_parameters(),
    )


@pytest.mark.parametrize(
    "argument",
    (Fraction(1, 100), Fraction(1), Fraction(51, 49), Fraction(100)),
)
def test_signed_log_encloses_high_precision_reference(argument: Fraction) -> None:
    receipt = enclose_positive_rational_signed_log(argument)
    lower, upper = receipt.logarithm_interval
    with localcontext() as context:
        context.prec = 120
        reference = _decimal(argument).ln()
        assert _decimal(lower) <= reference <= _decimal(upper)
    assert lower <= upper
    assert receipt.monotonic_signed_log_enclosed


@pytest.mark.parametrize(
    "argument",
    (Fraction(-5), Fraction(-1, 5), Fraction(0), Fraction(1, 5), Fraction(5)),
)
def test_rational_exponential_encloses_high_precision_reference(
    argument: Fraction,
) -> None:
    receipt = enclose_rational_exponential(argument)
    lower, upper = receipt.exponential_interval
    with localcontext() as context:
        context.prec = 120
        reference = _decimal(argument).exp()
        assert _decimal(lower) <= reference <= _decimal(upper)
    assert 0 < lower <= upper
    assert receipt.positive_taylor_remainder_enclosed
    assert receipt.repeated_squaring_exact


def test_physical_k_and_pivot_log_are_calibrated_with_explicit_c(
    calibrated_cell,
) -> None:
    q_lower, q_upper = calibrated_cell.dimensionless_fixed_wavenumber_interval
    k_lower, k_upper = calibrated_cell.physical_wavenumber_mpc_inverse_interval
    assert (k_lower, k_upper) == (
        q_lower * Fraction(337, 5) / SPEED_OF_LIGHT_KM_S,
        q_upper * Fraction(337, 5) / SPEED_OF_LIGHT_KM_S,
    )
    assert 0 < k_lower < k_upper < Fraction(1, 20)

    x_lower, x_upper = calibrated_cell.log_wavenumber_over_pivot_interval
    with localcontext() as context:
        context.prec = 120
        reference_lower = (_decimal(k_lower) / Decimal("0.05")).ln()
        reference_upper = (_decimal(k_upper) / Decimal("0.05")).ln()
        assert _decimal(x_lower) <= reference_lower
        assert reference_upper <= _decimal(x_upper)
    assert x_upper < 0


def test_external_power_law_is_enclosed_binwide_with_negative_tilt(
    calibrated_cell,
) -> None:
    x_lower, x_upper = calibrated_cell.log_wavenumber_over_pivot_interval
    power_lower, power_upper = (
        calibrated_cell.dimensionless_primordial_curvature_power_interval
    )
    assert calibrated_cell.scalar_spectral_tilt_interval == (
        Fraction(-7, 200),
        Fraction(-7, 200),
    )
    with localcontext() as context:
        context.prec = 120
        amplitude = Decimal("2.1e-9")
        beta = Decimal("-0.035")
        reference_lower = amplitude * (beta * _decimal(x_upper)).exp()
        reference_upper = amplitude * (beta * _decimal(x_lower)).exp()
        assert _decimal(power_lower) <= reference_lower
        assert reference_upper <= _decimal(power_upper)
    assert Fraction(5, 2_000_000_000) < power_lower < power_upper


def test_calibrated_power_cell_reuses_theorem37_nonnegative_integral(
    calibrated_cell,
) -> None:
    assert (
        calibrated_cell.generated_power_certificate
        .primordial_curvature_power_interval
        == calibrated_cell.dimensionless_primordial_curvature_power_interval
    )
    assert (
        calibrated_cell.reduced_angular_power_cell_interval
        == calibrated_cell.compact_power_cell.reduced_angular_power_cell_interval
    )
    lower, upper = calibrated_cell.reduced_angular_power_cell_interval
    assert 0 < lower <= upper


def test_tilt_interval_crossing_zero_uses_all_rectangle_corners(
    curvature_transfer,
) -> None:
    parameters = ExternalScalarPowerLawPrimordialSpectrum.freeze(
        scalar_amplitude_interval=(Fraction(2, 10**9), Fraction(2, 10**9)),
        scalar_spectral_index_interval=(Fraction(9, 10), Fraction(11, 10)),
        pivot_wavenumber_mpc_inverse_interval=(Fraction(1, 20), Fraction(1, 20)),
        parameter_reference="synthetic sign-crossing tilt interval",
        dimensionless_mathcal_p_r_convention_adopted=True,
        zero_running_power_law_model_adopted=True,
        parameters_are_binwide_not_nodewise=True,
    )
    receipt = construct_physically_calibrated_primordial_power_cell(
        curvature_transfer,
        _calibration(curvature_transfer),
        parameters,
    )
    exponent_lower, exponent_upper = receipt.power_law_exponent_interval
    assert exponent_lower < 0 < exponent_upper
    power_lower, power_upper = (
        receipt.dimensionless_primordial_curvature_power_interval
    )
    assert power_lower < Fraction(2, 10**9) < power_upper


def test_contract_status_keeps_external_inputs_and_nonclaims(calibrated_cell) -> None:
    assert calibrated_cell.physical_k_and_pivot_algebra_verified
    assert calibrated_cell.exact_rational_log_and_exponential_enclosures_used
    assert calibrated_cell.tilt_sign_crossing_safe_rectangle_bound_used
    assert calibrated_cell.binwide_power_law_mathcal_p_r_enclosed
    assert calibrated_cell.one_compact_bin_reduced_angular_power_enclosed
    assert calibrated_cell.external_inputs_not_ce_predictions
    assert not calibrated_cell.primordial_spectrum_derived_from_ce
    assert not calibrated_cell.running_or_features_enclosed
    assert not calibrated_cell.multiple_or_all_k_bins_enclosed
    assert not calibrated_cell.exterior_tail_integrals_enclosed
    assert not calibrated_cell.full_angular_power_spectrum_enclosed
    assert not calibrated_cell.covariance_or_likelihood_enclosed


def test_invalid_transcendental_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        enclose_positive_rational_signed_log(0)
    with pytest.raises(ValueError, match="must be an integer"):
        enclose_rational_exponential(1, highest_partial_sum_order=True)
    with pytest.raises(ValueError, match="lie in"):
        enclose_rational_exponential(1, highest_partial_sum_order=3)
    with pytest.raises(ValueError, match="absolute value exceeds"):
        enclose_rational_exponential(257)
    with pytest.raises(ValueError, match="representation exceeds"):
        enclose_rational_exponential(Fraction(1, 1 << 16_385))


def test_units_flags_references_and_provenance_fail_closed(
    curvature_transfer,
) -> None:
    q_bin = (
        curvature_transfer.source_plane_harmonic_transfer_receipt
        .dimensionless_fixed_wavenumber_interval
    )
    common = dict(
        dimensionless_fixed_wavenumber_interval=q_bin,
        hubble_reference_km_s_mpc_interval=(Fraction(337, 5), Fraction(337, 5)),
        pivot_wavenumber_mpc_inverse_interval=(Fraction(1, 20), Fraction(1, 20)),
        calibration_reference="synthetic calibration",
        hubble_reference_units=HUBBLE_REFERENCE_UNITS,
        pivot_wavenumber_units=PHYSICAL_WAVENUMBER_UNITS,
        fourier_convention=ANGULAR_COMOVING_FOURIER_CONVENTION,
        density_hubble_normalization=INTERNAL_DENSITY_HUBBLE_NORMALIZATION,
        scale_factor_today_equals_one_adopted=True,
        fixed_mode_independent_hubble_reference_adopted=True,
        internal_density_hubble_normalization_adopted=True,
    )
    with pytest.raises(ValueError, match="Fourier convention mismatch"):
        ExternalPhysicalWavenumberPivotCalibration.freeze(
            **{**common, "fourier_convention": "cycles per Mpc"}
        )
    with pytest.raises(ValueError, match="flags are incomplete"):
        ExternalPhysicalWavenumberPivotCalibration.freeze(
            **{
                **common,
                "fixed_mode_independent_hubble_reference_adopted": False,
            }
        )
    with pytest.raises(ValueError, match="requires a reference"):
        ExternalPhysicalWavenumberPivotCalibration.freeze(
            **{**common, "calibration_reference": ""}
        )

    calibration = _calibration(curvature_transfer)
    wrong_q = replace(
        calibration,
        dimensionless_fixed_wavenumber_interval=(
            q_bin[0],
            q_bin[1] + Fraction(1, 1000),
        ),
    )
    with pytest.raises(ValueError, match="q-bin provenance mismatch"):
        construct_physically_calibrated_primordial_power_cell(
            curvature_transfer,
            wrong_q,
            _primordial_parameters(),
        )
    falsified = replace(
        calibration,
        internal_density_hubble_normalization_adopted=False,
    )
    with pytest.raises(ValueError, match="prerequisites fail"):
        construct_physically_calibrated_primordial_power_cell(
            curvature_transfer,
            falsified,
            _primordial_parameters(),
        )
    wrong_c = replace(calibration, speed_of_light_km_s=Fraction(1))
    with pytest.raises(ValueError, match="speed-of-light"):
        construct_physically_calibrated_primordial_power_cell(
            curvature_transfer,
            wrong_c,
            _primordial_parameters(),
        )

    parameters = _primordial_parameters()
    wrong_model = replace(
        parameters,
        zero_running_power_law_model_adopted=False,
    )
    with pytest.raises(ValueError, match="power-law prerequisites fail"):
        construct_physically_calibrated_primordial_power_cell(
            curvature_transfer,
            calibration,
            wrong_model,
        )
    wrong_pivot = replace(
        parameters,
        pivot_wavenumber_mpc_inverse_interval=(
            Fraction(1, 10),
            Fraction(1, 10),
        ),
    )
    with pytest.raises(ValueError, match="pivot provenance mismatch"):
        construct_physically_calibrated_primordial_power_cell(
            curvature_transfer,
            calibration,
            wrong_pivot,
        )
    negative_amplitude = replace(
        parameters,
        scalar_amplitude_interval=(Fraction(-1), Fraction(1)),
    )
    with pytest.raises(ValueError, match="primordial power interval is invalid"):
        construct_physically_calibrated_primordial_power_cell(
            curvature_transfer,
            calibration,
            negative_amplitude,
        )
