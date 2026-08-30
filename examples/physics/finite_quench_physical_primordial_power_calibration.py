"""Physical-k and pivot calibration for one finite-quench compact bin.

The internal coordinate is dimensionless.  Restoring c on the adopted
fixed-slice normalization gives

    q = c k / H_ref,
    k[Mpc^-1] = q H_ref[km s^-1 Mpc^-1] / c[km s^-1].

The Fourier convention, H_ref, pivot, A_s, and n_s are versioned external
inputs.  This module proves only their algebraic conversion and an
exact-rational bin enclosure of

    x = ln(k/k_pivot),
    mathcal P_R(k) = A_s exp((n_s - 1) x).

It does not derive H_ref or the primordial spectrum from CE and it does not
claim multiple-bin, all-k, tail, covariance, or likelihood closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

from examples.physics.finite_quench_curvature_power_cell import (
    CertifiedPrimordialCurvaturePowerKappaBinEnvelope,
    CompactKappaBinCurvaturePowerCellReceipt,
    ExactRationalPositiveLogRatioEnclosure,
    construct_compact_kappa_bin_curvature_power_cell,
    enclose_positive_rational_log_ratio,
)
from examples.physics.finite_quench_harmonic_power_enclosure import (
    _exact_fraction,
    _ordered_interval,
)
from examples.physics.finite_quench_primordial_curvature_normalization import (
    PrimordialCurvatureNormalizedHarmonicTransferReceipt,
)


SPEED_OF_LIGHT_KM_S = Fraction(299_792_458, 1000)
MAXIMUM_ABSOLUTE_CERTIFIED_EXPONENT = Fraction(256)
MAXIMUM_CERTIFIED_RATIONAL_BITS = 16_384
HUBBLE_REFERENCE_UNITS = "km s^-1 Mpc^-1"
PHYSICAL_WAVENUMBER_UNITS = "Mpc^-1"
ANGULAR_COMOVING_FOURIER_CONVENTION = (
    "exp(i k dot x); angular comoving wavenumber; no cycles or 2pi factor"
)
INTERNAL_DENSITY_HUBBLE_NORMALIZATION = (
    "H(n)^2/H_ref^2 = rho_total(n); q = c k/H_ref"
)


def _validated_order(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    order = int(value)
    if not 4 <= order <= 64:
        raise ValueError(f"{name} must lie in [4, 64]")
    return order


@dataclass(frozen=True)
class ExactRationalSignedLogEnclosure:
    argument: Fraction
    logarithm_interval: tuple[Fraction, Fraction]
    positive_ratio_receipt: (
        ExactRationalPositiveLogRatioEnclosure | None
    ) = field(repr=False)
    reciprocal_reduction_used: bool
    exact_zero_logarithm: bool
    monotonic_signed_log_enclosed: bool


def enclose_positive_rational_signed_log(
    argument: object,
    *,
    highest_partial_sum_order: object = 24,
) -> ExactRationalSignedLogEnclosure:
    """Enclose ln(argument) exactly for every positive rational argument."""

    value = _exact_fraction(argument, "logarithm argument")
    order = _validated_order(
        highest_partial_sum_order,
        "logarithm partial-sum order",
    )
    if value <= 0:
        raise ValueError("logarithm argument must be positive")
    if value == 1:
        return ExactRationalSignedLogEnclosure(
            value,
            (Fraction(0), Fraction(0)),
            None,
            False,
            True,
            True,
        )
    if value > 1:
        receipt = enclose_positive_rational_log_ratio(
            numerator=value.numerator,
            denominator=value.denominator,
            highest_partial_sum_order=order,
        )
        interval = receipt.logarithm_interval
        reciprocal = False
    else:
        receipt = enclose_positive_rational_log_ratio(
            numerator=value.denominator,
            denominator=value.numerator,
            highest_partial_sum_order=order,
        )
        lower, upper = receipt.logarithm_interval
        interval = (-upper, -lower)
        reciprocal = True
    return ExactRationalSignedLogEnclosure(
        value,
        interval,
        receipt,
        reciprocal,
        False,
        True,
    )


def _nonnegative_exp_series(
    argument: Fraction,
    order: int,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if argument < 0:
        raise ValueError("exponential helper requires a nonnegative argument")
    term = Fraction(1)
    partial = Fraction(1)
    for index in range(1, order + 1):
        term *= argument / index
        partial += term
    first_omitted = term * argument / (order + 1)
    ratio = argument / (order + 2)
    if ratio >= 1:
        raise ValueError("exponential Taylor tail left its certified domain")
    tail = first_omitted / (1 - ratio)
    return partial, tail, partial, partial + tail


@dataclass(frozen=True)
class ExactRationalExponentialEnclosure:
    argument: Fraction
    binary_range_reduction_exponent: int
    reduced_absolute_argument: Fraction
    highest_partial_sum_order: int
    reduced_partial_sum_lower_bound: Fraction
    reduced_positive_tail_upper_bound: Fraction
    exponential_interval: tuple[Fraction, Fraction]
    reciprocal_identity_used: bool
    positive_taylor_remainder_enclosed: bool
    repeated_squaring_exact: bool


def enclose_rational_exponential(
    argument: object,
    *,
    highest_partial_sum_order: object = 24,
) -> ExactRationalExponentialEnclosure:
    """Enclose exp(argument) with rational Taylor and range bounds."""

    value = _exact_fraction(argument, "exponential argument")
    order = _validated_order(
        highest_partial_sum_order,
        "exponential partial-sum order",
    )
    reduced = abs(value)
    if reduced > MAXIMUM_ABSOLUTE_CERTIFIED_EXPONENT:
        raise ValueError("exponential argument absolute value exceeds 256")
    if (
        value.numerator.bit_length() > MAXIMUM_CERTIFIED_RATIONAL_BITS
        or value.denominator.bit_length() > MAXIMUM_CERTIFIED_RATIONAL_BITS
    ):
        raise ValueError("exponential rational representation exceeds 16384 bits")
    reduction = 0
    while reduced > Fraction(1, 2):
        reduced /= 2
        reduction += 1
        if reduction > 256:
            raise ValueError("exponential range reduction exceeds 256 squarings")
    partial, tail, lower, upper = _nonnegative_exp_series(reduced, order)
    for _ in range(reduction):
        lower *= lower
        upper *= upper
    reciprocal = value < 0
    interval = (
        (Fraction(1, upper), Fraction(1, lower))
        if reciprocal
        else (lower, upper)
    )
    return ExactRationalExponentialEnclosure(
        argument=value,
        binary_range_reduction_exponent=reduction,
        reduced_absolute_argument=reduced,
        highest_partial_sum_order=order,
        reduced_partial_sum_lower_bound=partial,
        reduced_positive_tail_upper_bound=tail,
        exponential_interval=interval,
        reciprocal_identity_used=reciprocal,
        positive_taylor_remainder_enclosed=True,
        repeated_squaring_exact=True,
    )


@dataclass(frozen=True)
class ExternalPhysicalWavenumberPivotCalibration:
    """External scale, units, Fourier convention, and pivot contract."""

    dimensionless_fixed_wavenumber_interval: tuple[Fraction, Fraction]
    hubble_reference_km_s_mpc_interval: tuple[Fraction, Fraction]
    pivot_wavenumber_mpc_inverse_interval: tuple[Fraction, Fraction]
    calibration_reference: str
    speed_of_light_km_s: Fraction
    fixed_mode_independent_hubble_reference_adopted: bool
    internal_density_hubble_normalization_adopted: bool
    angular_comoving_fourier_convention_adopted: bool
    scale_factor_today_equals_one_adopted: bool
    external_calibration_not_ce_prediction: bool

    @classmethod
    def freeze(
        cls,
        *,
        dimensionless_fixed_wavenumber_interval: object,
        hubble_reference_km_s_mpc_interval: object,
        pivot_wavenumber_mpc_inverse_interval: object,
        calibration_reference: object,
        hubble_reference_units: object,
        pivot_wavenumber_units: object,
        fourier_convention: object,
        density_hubble_normalization: object,
        scale_factor_today_equals_one_adopted: bool,
        fixed_mode_independent_hubble_reference_adopted: bool,
        internal_density_hubble_normalization_adopted: bool,
    ) -> "ExternalPhysicalWavenumberPivotCalibration":
        q_bin = _ordered_interval(
            dimensionless_fixed_wavenumber_interval,
            "dimensionless fixed-wavenumber bin",
        )
        hubble = _ordered_interval(
            hubble_reference_km_s_mpc_interval,
            "H_ref interval",
        )
        pivot = _ordered_interval(
            pivot_wavenumber_mpc_inverse_interval,
            "pivot-wavenumber interval",
        )
        if not Fraction(0) < q_bin[0] < q_bin[1]:
            raise ValueError("dimensionless q bin must be positive and nonempty")
        if hubble[0] <= 0 or pivot[0] <= 0:
            raise ValueError("H_ref and pivot intervals must be positive")
        if hubble_reference_units != HUBBLE_REFERENCE_UNITS:
            raise ValueError("H_ref units must be km s^-1 Mpc^-1")
        if pivot_wavenumber_units != PHYSICAL_WAVENUMBER_UNITS:
            raise ValueError("pivot units must be Mpc^-1")
        if fourier_convention != ANGULAR_COMOVING_FOURIER_CONVENTION:
            raise ValueError("physical-k Fourier convention mismatch")
        if density_hubble_normalization != INTERNAL_DENSITY_HUBBLE_NORMALIZATION:
            raise ValueError("internal density/H_ref normalization mismatch")
        required = (
            scale_factor_today_equals_one_adopted,
            fixed_mode_independent_hubble_reference_adopted,
            internal_density_hubble_normalization_adopted,
        )
        if any(flag is not True for flag in required):
            raise ValueError("physical calibration adoption flags are incomplete")
        if (
            not isinstance(calibration_reference, str)
            or not calibration_reference.strip()
        ):
            raise ValueError("physical calibration requires a reference")
        return cls(
            dimensionless_fixed_wavenumber_interval=q_bin,
            hubble_reference_km_s_mpc_interval=hubble,
            pivot_wavenumber_mpc_inverse_interval=pivot,
            calibration_reference=calibration_reference.strip(),
            speed_of_light_km_s=SPEED_OF_LIGHT_KM_S,
            fixed_mode_independent_hubble_reference_adopted=True,
            internal_density_hubble_normalization_adopted=True,
            angular_comoving_fourier_convention_adopted=True,
            scale_factor_today_equals_one_adopted=True,
            external_calibration_not_ce_prediction=True,
        )


@dataclass(frozen=True)
class ExternalScalarPowerLawPrimordialSpectrum:
    """External dimensionless A_s and n_s contract with zero running."""

    scalar_amplitude_interval: tuple[Fraction, Fraction]
    scalar_spectral_index_interval: tuple[Fraction, Fraction]
    pivot_wavenumber_mpc_inverse_interval: tuple[Fraction, Fraction]
    parameter_reference: str
    dimensionless_mathcal_p_r_convention_adopted: bool
    zero_running_power_law_model_adopted: bool
    parameters_are_binwide_not_nodewise: bool
    parameters_are_external_not_ce_predictions: bool

    @classmethod
    def freeze(
        cls,
        *,
        scalar_amplitude_interval: object,
        scalar_spectral_index_interval: object,
        pivot_wavenumber_mpc_inverse_interval: object,
        parameter_reference: object,
        dimensionless_mathcal_p_r_convention_adopted: bool,
        zero_running_power_law_model_adopted: bool,
        parameters_are_binwide_not_nodewise: bool,
    ) -> "ExternalScalarPowerLawPrimordialSpectrum":
        amplitude = _ordered_interval(
            scalar_amplitude_interval,
            "scalar-amplitude interval",
        )
        spectral_index = _ordered_interval(
            scalar_spectral_index_interval,
            "scalar spectral-index interval",
        )
        pivot = _ordered_interval(
            pivot_wavenumber_mpc_inverse_interval,
            "primordial pivot interval",
        )
        if amplitude[0] <= 0 or pivot[0] <= 0:
            raise ValueError("scalar amplitude and pivot must be positive")
        required = (
            dimensionless_mathcal_p_r_convention_adopted,
            zero_running_power_law_model_adopted,
            parameters_are_binwide_not_nodewise,
        )
        if any(flag is not True for flag in required):
            raise ValueError("primordial power-law adoption flags are incomplete")
        if (
            not isinstance(parameter_reference, str)
            or not parameter_reference.strip()
        ):
            raise ValueError("primordial parameters require a reference")
        return cls(
            scalar_amplitude_interval=amplitude,
            scalar_spectral_index_interval=spectral_index,
            pivot_wavenumber_mpc_inverse_interval=pivot,
            parameter_reference=parameter_reference.strip(),
            dimensionless_mathcal_p_r_convention_adopted=True,
            zero_running_power_law_model_adopted=True,
            parameters_are_binwide_not_nodewise=True,
            parameters_are_external_not_ce_predictions=True,
        )


@dataclass(frozen=True)
class PhysicalPrimordialPowerCalibratedCellReceipt:
    transfer_receipt: PrimordialCurvatureNormalizedHarmonicTransferReceipt = field(
        repr=False
    )
    calibration: ExternalPhysicalWavenumberPivotCalibration = field(repr=False)
    primordial_parameters: ExternalScalarPowerLawPrimordialSpectrum = field(
        repr=False
    )
    generated_power_certificate: (
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope
    ) = field(repr=False)
    compact_power_cell: CompactKappaBinCurvaturePowerCellReceipt = field(
        repr=False
    )
    ell: int
    dimensionless_fixed_wavenumber_interval: tuple[Fraction, Fraction]
    physical_wavenumber_mpc_inverse_interval: tuple[Fraction, Fraction]
    wavenumber_over_pivot_interval: tuple[Fraction, Fraction]
    log_wavenumber_over_pivot_interval: tuple[Fraction, Fraction]
    scalar_spectral_tilt_interval: tuple[Fraction, Fraction]
    power_law_exponent_interval: tuple[Fraction, Fraction]
    dimensionless_primordial_curvature_power_interval: tuple[Fraction, Fraction]
    reduced_angular_power_cell_interval: tuple[Fraction, Fraction]
    physical_k_and_pivot_algebra_verified: bool
    exact_rational_log_and_exponential_enclosures_used: bool
    tilt_sign_crossing_safe_rectangle_bound_used: bool
    binwide_power_law_mathcal_p_r_enclosed: bool
    one_compact_bin_reduced_angular_power_enclosed: bool
    external_inputs_not_ce_predictions: bool
    primordial_spectrum_derived_from_ce: bool = False
    running_or_features_enclosed: bool = False
    multiple_or_all_k_bins_enclosed: bool = False
    exterior_tail_integrals_enclosed: bool = False
    full_angular_power_spectrum_enclosed: bool = False
    covariance_or_likelihood_enclosed: bool = False


def construct_physically_calibrated_primordial_power_cell(
    transfer: PrimordialCurvatureNormalizedHarmonicTransferReceipt,
    calibration: ExternalPhysicalWavenumberPivotCalibration,
    primordial_parameters: ExternalScalarPowerLawPrimordialSpectrum,
    *,
    log_highest_partial_sum_order: object = 24,
    exp_highest_partial_sum_order: object = 24,
) -> PhysicalPrimordialPowerCalibratedCellReceipt:
    """Calibrate q and enclose a zero-running power-law primordial cell."""

    if not isinstance(
        transfer,
        PrimordialCurvatureNormalizedHarmonicTransferReceipt,
    ):
        raise ValueError("transfer has the wrong receipt type")
    if not isinstance(calibration, ExternalPhysicalWavenumberPivotCalibration):
        raise ValueError("calibration has the wrong external-contract type")
    if not isinstance(
        primordial_parameters,
        ExternalScalarPowerLawPrimordialSpectrum,
    ):
        raise ValueError("primordial parameters have the wrong external-contract type")
    if calibration.speed_of_light_km_s != SPEED_OF_LIGHT_KM_S:
        raise ValueError("speed-of-light calibration provenance mismatch")
    calibration_flags = (
        calibration.fixed_mode_independent_hubble_reference_adopted,
        calibration.internal_density_hubble_normalization_adopted,
        calibration.angular_comoving_fourier_convention_adopted,
        calibration.scale_factor_today_equals_one_adopted,
        calibration.external_calibration_not_ce_prediction,
    )
    power_flags = (
        primordial_parameters.dimensionless_mathcal_p_r_convention_adopted,
        primordial_parameters.zero_running_power_law_model_adopted,
        primordial_parameters.parameters_are_binwide_not_nodewise,
        primordial_parameters.parameters_are_external_not_ce_predictions,
    )
    if not all(calibration_flags):
        raise ValueError("physical calibration prerequisites fail")
    if not all(power_flags):
        raise ValueError("primordial power-law prerequisites fail")

    harmonic = transfer.source_plane_harmonic_transfer_receipt
    q_bin = harmonic.dimensionless_fixed_wavenumber_interval
    if calibration.dimensionless_fixed_wavenumber_interval != q_bin:
        raise ValueError("physical calibration q-bin provenance mismatch")
    if (
        primordial_parameters.pivot_wavenumber_mpc_inverse_interval
        != calibration.pivot_wavenumber_mpc_inverse_interval
    ):
        raise ValueError("physical and primordial pivot provenance mismatch")

    q_lower, q_upper = q_bin
    h_lower, h_upper = calibration.hubble_reference_km_s_mpc_interval
    pivot_lower, pivot_upper = (
        calibration.pivot_wavenumber_mpc_inverse_interval
    )
    c = calibration.speed_of_light_km_s
    physical_k = (q_lower * h_lower / c, q_upper * h_upper / c)
    k_over_pivot = (
        physical_k[0] / pivot_upper,
        physical_k[1] / pivot_lower,
    )
    lower_log = enclose_positive_rational_signed_log(
        k_over_pivot[0],
        highest_partial_sum_order=log_highest_partial_sum_order,
    )
    upper_log = enclose_positive_rational_signed_log(
        k_over_pivot[1],
        highest_partial_sum_order=log_highest_partial_sum_order,
    )
    log_interval = (
        lower_log.logarithm_interval[0],
        upper_log.logarithm_interval[1],
    )
    if log_interval[0] > log_interval[1]:
        raise ValueError("physical pivot log interval is reversed")

    ns_lower, ns_upper = primordial_parameters.scalar_spectral_index_interval
    beta = (ns_lower - 1, ns_upper - 1)
    corners = (
        beta[0] * log_interval[0],
        beta[0] * log_interval[1],
        beta[1] * log_interval[0],
        beta[1] * log_interval[1],
    )
    exponent_interval = (min(corners), max(corners))
    lower_exp = enclose_rational_exponential(
        exponent_interval[0],
        highest_partial_sum_order=exp_highest_partial_sum_order,
    )
    upper_exp = enclose_rational_exponential(
        exponent_interval[1],
        highest_partial_sum_order=exp_highest_partial_sum_order,
    )
    amplitude_lower, amplitude_upper = (
        primordial_parameters.scalar_amplitude_interval
    )
    primordial_power = (
        amplitude_lower * lower_exp.exponential_interval[0],
        amplitude_upper * upper_exp.exponential_interval[1],
    )
    if not Fraction(0) < primordial_power[0] <= primordial_power[1]:
        raise ValueError("dimensionless primordial power interval is invalid")

    generated = CertifiedPrimordialCurvaturePowerKappaBinEnvelope.freeze(
        initial_kappa_interval=transfer.initial_kappa_interval,
        dimensionless_fixed_wavenumber_interval=q_bin,
        primordial_curvature_power_interval=primordial_power,
        proof_reference=(
            "THEOREM38: "
            + calibration.calibration_reference
            + " | "
            + primordial_parameters.parameter_reference
        ),
        binwide_power_envelope_certified=True,
    )
    power_cell = construct_compact_kappa_bin_curvature_power_cell(
        transfer,
        generated,
    )
    return PhysicalPrimordialPowerCalibratedCellReceipt(
        transfer_receipt=transfer,
        calibration=calibration,
        primordial_parameters=primordial_parameters,
        generated_power_certificate=generated,
        compact_power_cell=power_cell,
        ell=transfer.ell,
        dimensionless_fixed_wavenumber_interval=q_bin,
        physical_wavenumber_mpc_inverse_interval=physical_k,
        wavenumber_over_pivot_interval=k_over_pivot,
        log_wavenumber_over_pivot_interval=log_interval,
        scalar_spectral_tilt_interval=beta,
        power_law_exponent_interval=exponent_interval,
        dimensionless_primordial_curvature_power_interval=primordial_power,
        reduced_angular_power_cell_interval=(
            power_cell.reduced_angular_power_cell_interval
        ),
        physical_k_and_pivot_algebra_verified=True,
        exact_rational_log_and_exponential_enclosures_used=True,
        tilt_sign_crossing_safe_rectangle_bound_used=True,
        binwide_power_law_mathcal_p_r_enclosed=True,
        one_compact_bin_reduced_angular_power_enclosed=True,
        external_inputs_not_ce_predictions=True,
    )
