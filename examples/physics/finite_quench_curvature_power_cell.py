"""One certified compact-kappa contribution to harmonic convergence power.

On one fixed initial slice the dimensionless fixed wavenumber is

    q = C * kappa_i,  C > 0,

with the same constant ``C`` for every mode in the compact bin.  Hence

    d ln(k) = d ln(q) = d ln(kappa_i)

and the exact bin width is ``ln(kappa_+ / kappa_-)`` even before an absolute
Mpc^-1 calibration is supplied.

For a positive rational ratio ``r``, set ``y = (r-1)/(r+1)``.  The identity

    ln(r) = 2 atanh(y)
          = 2 sum_{j >= 0} y^(2j+1)/(2j+1)

has an exact-rational positive remainder bound.  This module combines that
width with a theorem-36 bin-wide ``Delta_l^kappa/R`` interval and a separately
certified bin-wide dimensionless primordial-curvature spectrum
``mathcal P_R`` interval to enclose one contribution to
``C_l^(kappa kappa)/(4*pi)``.  The convention is

    <R(k) R(k')> = (2*pi)^3 delta^3(k+k') (2*pi^2/k^3) mathcal P_R(k),
    C_l^(kappa kappa) = 4*pi integral d ln(k) mathcal P_R(k)
                        |Delta_l^kappa(k)|^2.

Thus ``mathcal P_R``, ``R``, ``kappa``, ``Delta_l^kappa``, and ``C_l`` are
dimensionless; the field ``primordial_curvature_power_interval`` denotes an
envelope of ``mathcal P_R``, not a dimensional power density.

The primordial envelope is supplied on the same internal dimensionless-q
bin.  No physical Mpc^-1 scale, primordial pivot, all-k coverage, source
population, covariance, or likelihood is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

from examples.physics.finite_quench_harmonic_power_enclosure import (
    _exact_fraction,
    _ordered_interval,
    _square_interval,
)
from examples.physics.finite_quench_primordial_curvature_normalization import (
    PrimordialCurvatureNormalizedHarmonicTransferReceipt,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    _RationalInterval,
    _interval_multiply,
)


def _twice_atanh_partial_and_tail(
    *,
    argument: Fraction,
    highest_partial_sum_order: int,
) -> tuple[Fraction, Fraction]:
    """Return a lower partial sum and positive tail bound for 2*atanh(y)."""

    y = Fraction(argument)
    if not Fraction(0) <= y < Fraction(1):
        raise ValueError("atanh-series argument must lie in [0, 1)")
    if not 0 <= highest_partial_sum_order <= 128:
        raise ValueError("log partial-sum order must lie in [0, 128]")
    if y == 0:
        return Fraction(0), Fraction(0)

    y_squared = y * y
    power = y
    partial = Fraction(0)
    for order in range(highest_partial_sum_order + 1):
        if order:
            power *= y_squared
        partial += 2 * power / (2 * order + 1)
    first_omitted_denominator = 2 * highest_partial_sum_order + 3
    first_omitted_power = power * y_squared
    tail = (
        2
        * first_omitted_power
        / (first_omitted_denominator * (1 - y_squared))
    )
    return partial, tail


@dataclass(frozen=True)
class ExactRationalPositiveLogRatioEnclosure:
    """Exact-rational enclosure of ``ln(numerator/denominator) > 0``."""

    numerator: Fraction
    denominator: Fraction
    ratio: Fraction
    highest_partial_sum_order: int
    power_of_two_range_reduction_exponent: int
    reduced_ratio: Fraction
    reduced_ratio_atanh_argument: Fraction
    log_two_atanh_argument: Fraction
    partial_sum_lower_bound: Fraction
    positive_tail_upper_bound: Fraction
    logarithm_interval: tuple[Fraction, Fraction]
    ratio_strictly_above_one_proven: bool
    reduced_ratio_in_one_to_two_proven: bool
    atanh_arguments_in_certified_domain_proven: bool
    exact_rational_positive_series_remainder_proven: bool
    role: str = (
        "EXACT_RATIONAL_POSITIVE_LOG_RATIO_ENCLOSURE_BY_RANGE_REDUCED_"
        "TWICE_ATANH_SERIES"
    )


def enclose_positive_rational_log_ratio(
    *,
    numerator: object,
    denominator: object,
    highest_partial_sum_order: object = 16,
) -> ExactRationalPositiveLogRatioEnclosure:
    """Enclose ``ln(numerator/denominator)`` without floating logarithms."""

    top = _exact_fraction(numerator, "log-ratio numerator")
    bottom = _exact_fraction(denominator, "log-ratio denominator")
    if not Fraction(0) < bottom < top:
        raise ValueError(
            "positive log ratio requires 0 < denominator < numerator"
        )
    if (
        isinstance(highest_partial_sum_order, bool)
        or not isinstance(highest_partial_sum_order, Integral)
    ):
        raise ValueError("log partial-sum order must be an integer")
    order = int(highest_partial_sum_order)
    if not 0 <= order <= 128:
        raise ValueError("log partial-sum order must lie in [0, 128]")

    ratio = top / bottom
    exponent = max(
        0,
        ratio.numerator.bit_length() - ratio.denominator.bit_length(),
    )
    power_of_two = Fraction(1 << exponent)
    if ratio < power_of_two:
        exponent -= 1
        power_of_two /= 2
    while ratio >= 2 * power_of_two:
        exponent += 1
        power_of_two *= 2
    reduced = ratio / power_of_two
    if not Fraction(1) <= reduced < Fraction(2):
        raise ValueError("exact power-of-two log range reduction failed")

    reduced_argument = (reduced - 1) / (reduced + 1)
    log_two_argument = Fraction(1, 3)
    reduced_partial, reduced_tail = _twice_atanh_partial_and_tail(
        argument=reduced_argument,
        highest_partial_sum_order=order,
    )
    log_two_partial, log_two_tail = _twice_atanh_partial_and_tail(
        argument=log_two_argument,
        highest_partial_sum_order=order,
    )
    partial = exponent * log_two_partial + reduced_partial
    tail = exponent * log_two_tail + reduced_tail
    logarithm = (partial, partial + tail)
    return ExactRationalPositiveLogRatioEnclosure(
        numerator=top,
        denominator=bottom,
        ratio=ratio,
        highest_partial_sum_order=order,
        power_of_two_range_reduction_exponent=exponent,
        reduced_ratio=reduced,
        reduced_ratio_atanh_argument=reduced_argument,
        log_two_atanh_argument=log_two_argument,
        partial_sum_lower_bound=partial,
        positive_tail_upper_bound=tail,
        logarithm_interval=logarithm,
        ratio_strictly_above_one_proven=True,
        reduced_ratio_in_one_to_two_proven=True,
        atanh_arguments_in_certified_domain_proven=True,
        exact_rational_positive_series_remainder_proven=True,
    )


@dataclass(frozen=True)
class CertifiedPrimordialCurvaturePowerKappaBinEnvelope:
    """Supplied bin-wide dimensionless ``mathcal P_R`` interval on q/kappa."""

    initial_kappa_interval: tuple[Fraction, Fraction]
    dimensionless_fixed_wavenumber_interval: tuple[Fraction, Fraction]
    primordial_curvature_power_interval: tuple[Fraction, Fraction]
    proof_reference: str
    coordinate_is_internal_dimensionless_fixed_wavenumber_q: bool
    binwide_not_nodewise_primordial_power_envelope_certified: bool
    dimensionless_nonnegative_primordial_auto_power_proven: bool
    physical_wavenumber_mpc_inverse_calibrated: bool = False
    primordial_pivot_wavenumber_calibrated: bool = False
    primordial_spectrum_derived_from_ce: bool = False
    role: str = (
        "SUPPLIED_CERTIFIED_BINWIDE_PRIMORDIAL_CURVATURE_POWER_ENVELOPE_"
        "ON_INTERNAL_DIMENSIONLESS_Q_BIN_NOT_PHYSICAL_K_PIVOT_OR_CE_DERIVATION"
    )

    @classmethod
    def freeze(
        cls,
        *,
        initial_kappa_interval: object,
        dimensionless_fixed_wavenumber_interval: object,
        primordial_curvature_power_interval: object,
        proof_reference: object,
        binwide_power_envelope_certified: bool,
    ) -> "CertifiedPrimordialCurvaturePowerKappaBinEnvelope":
        kappa = _ordered_interval(initial_kappa_interval, "initial-kappa bin")
        if not Fraction(0) < kappa[0] < kappa[1]:
            raise ValueError("initial-kappa bin must be positive and nonempty")
        q_interval = _ordered_interval(
            dimensionless_fixed_wavenumber_interval,
            "dimensionless fixed-wavenumber bin",
        )
        if not Fraction(0) < q_interval[0] < q_interval[1]:
            raise ValueError(
                "dimensionless fixed-wavenumber bin must be positive and nonempty"
            )
        primordial = _ordered_interval(
            primordial_curvature_power_interval,
            "primordial curvature power interval",
        )
        if primordial[0] < 0:
            raise ValueError("primordial curvature auto-power must be nonnegative")
        if binwide_power_envelope_certified is not True:
            raise ValueError(
                "primordial power must be certified bin-wide; nodes are insufficient"
            )
        if not isinstance(proof_reference, str) or not proof_reference.strip():
            raise ValueError(
                "primordial-power certificate requires a proof reference"
            )
        return cls(
            initial_kappa_interval=kappa,
            dimensionless_fixed_wavenumber_interval=q_interval,
            primordial_curvature_power_interval=primordial,
            proof_reference=proof_reference.strip(),
            coordinate_is_internal_dimensionless_fixed_wavenumber_q=True,
            binwide_not_nodewise_primordial_power_envelope_certified=True,
            dimensionless_nonnegative_primordial_auto_power_proven=True,
        )


@dataclass(frozen=True)
class CompactKappaBinCurvaturePowerCellReceipt:
    """Conditional one-bin contribution using dimensionless ``mathcal P_R``."""

    curvature_normalized_transfer_receipt: (
        PrimordialCurvatureNormalizedHarmonicTransferReceipt
    ) = field(repr=False)
    primordial_power_certificate: (
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope
    ) = field(repr=False)
    logarithmic_width_enclosure: ExactRationalPositiveLogRatioEnclosure
    ell: int
    initial_efold: Fraction
    initial_kappa_interval: tuple[Fraction, Fraction]
    dimensionless_fixed_wavenumber_interval: tuple[Fraction, Fraction]
    logarithmic_kappa_and_wavenumber_width_interval: (
        tuple[Fraction, Fraction]
    )
    primordial_curvature_power_interval: tuple[Fraction, Fraction]
    convergence_transfer_per_comoving_curvature_interval: (
        tuple[Fraction, Fraction]
    )
    convergence_transfer_modulus_squared_interval: (
        tuple[Fraction, Fraction]
    )
    reduced_power_integrand_interval: tuple[Fraction, Fraction]
    reduced_angular_power_cell_interval: tuple[Fraction, Fraction]
    fixed_initial_slice_shared_wavenumber_scale_proven: bool
    d_log_k_equals_d_log_q_equals_d_log_kappa_proven: bool
    exact_rational_logarithmic_bin_width_enclosed: bool
    binwide_primordial_power_and_transfer_envelopes_used: bool
    signed_transfer_squared_before_power_aggregation: bool
    nonnegative_reduced_power_integrand_enclosed: bool
    four_pi_d_log_k_angular_power_identity_reused: bool
    four_pi_factor_kept_symbolic: bool
    single_compact_bin_reduced_angular_power_contribution_enclosed: bool
    physical_wavenumber_mpc_inverse_calibrated: bool = False
    primordial_pivot_wavenumber_calibrated: bool = False
    primordial_spectrum_derived_from_ce: bool = False
    all_k_compact_bin_coverage_enclosed: bool = False
    exterior_tail_integrals_enclosed: bool = False
    source_population_distribution_supplied: bool = False
    post_born_or_relativistic_corrections_enclosed: bool = False
    full_angular_power_spectrum_enclosed: bool = False
    covariance_or_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_ONE_COMPACT_KAPPA_BIN_CONTRIBUTION_TO_REDUCED_"
        "CONVERGENCE_HARMONIC_AUTO_POWER_NOT_PHYSICAL_K_PIVOT_ALL_K_"
        "SOURCE_POPULATION_FULL_POWER_COVARIANCE_OR_LIKELIHOOD"
    )


def construct_compact_kappa_bin_curvature_power_cell(
    transfer: PrimordialCurvatureNormalizedHarmonicTransferReceipt,
    primordial_power: CertifiedPrimordialCurvaturePowerKappaBinEnvelope,
    *,
    log_highest_partial_sum_order: object = 16,
) -> CompactKappaBinCurvaturePowerCellReceipt:
    """Enclose one theorem-36 bin contribution to ``C_l/(4*pi)``.

    The dimensionless ``mathcal P_R`` bin-wide enclosure remains a supplied
    proof obligation under the module-level Fourier-covariance convention.
    """

    if not isinstance(
        transfer,
        PrimordialCurvatureNormalizedHarmonicTransferReceipt,
    ):
        raise ValueError("transfer has the wrong receipt type")
    if not isinstance(
        primordial_power,
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope,
    ):
        raise ValueError("primordial_power has the wrong certificate type")
    if not (
        transfer.compact_bin_harmonic_transfer_per_comoving_curvature_enclosed
        and transfer.dimensionless_normalization_proven
        and transfer.initial_superhorizon_compact_bin_proven
    ):
        raise ValueError("curvature-normalized transfer proof prerequisites fail")
    if not (
        primordial_power.coordinate_is_internal_dimensionless_fixed_wavenumber_q
        and primordial_power
        .binwide_not_nodewise_primordial_power_envelope_certified
        and primordial_power.dimensionless_nonnegative_primordial_auto_power_proven
    ):
        raise ValueError("primordial-power certificate proof prerequisites fail")
    if not primordial_power.proof_reference.strip():
        raise ValueError("primordial-power certificate proof reference is empty")

    compact = transfer.compact_bin_receipt
    harmonic = transfer.source_plane_harmonic_transfer_receipt
    if not (
        transfer.ell == harmonic.ell
        and transfer.initial_kappa_interval == compact.initial_kappa_interval
        and transfer.initial_kappa_interval == harmonic.initial_kappa_interval
        and primordial_power.initial_kappa_interval
        == transfer.initial_kappa_interval
        and primordial_power.dimensionless_fixed_wavenumber_interval
        == harmonic.dimensionless_fixed_wavenumber_interval
    ):
        raise ValueError("power-cell receipt provenance mismatch")

    central_q = _RationalInterval(
        *compact.central_trace_receipt.fixed_mode_born_lensing_absolute_envelope
        .dimensionless_fixed_wavenumber_interval
    )
    kappa_lower, kappa_upper = transfer.initial_kappa_interval
    central_kappa = transfer.central_initial_kappa
    if not Fraction(0) < kappa_lower <= central_kappa <= kappa_upper:
        raise ValueError("power-cell initial-kappa provenance is invalid")
    kappa_ratio = _RationalInterval(
        kappa_lower / central_kappa,
        kappa_upper / central_kappa,
    )
    expected_q = _interval_multiply(central_q, kappa_ratio)
    if (expected_q.lower, expected_q.upper) != (
        harmonic.dimensionless_fixed_wavenumber_interval
    ):
        raise ValueError("fixed-slice q-to-kappa scaling provenance mismatch")

    log_width = enclose_positive_rational_log_ratio(
        numerator=kappa_upper,
        denominator=kappa_lower,
        highest_partial_sum_order=log_highest_partial_sum_order,
    )
    transfer_interval = _ordered_interval(
        transfer
        .normalized_convergence_harmonic_transfer_per_comoving_curvature_interval,
        "curvature-normalized convergence transfer",
    )
    modulus_squared = _square_interval(transfer_interval)
    primordial_interval = _ordered_interval(
        primordial_power.primordial_curvature_power_interval,
        "certified primordial curvature power",
    )
    if primordial_interval[0] < 0:
        raise ValueError("certified primordial curvature power is negative")
    integrand = (
        primordial_interval[0] * modulus_squared[0],
        primordial_interval[1] * modulus_squared[1],
    )
    if integrand[0] < 0 or integrand[0] > integrand[1]:
        raise ValueError("reduced power integrand lost nonnegativity")
    width_lower, width_upper = log_width.logarithm_interval
    power_cell = (
        width_lower * integrand[0],
        width_upper * integrand[1],
    )
    if power_cell[0] < 0 or power_cell[0] > power_cell[1]:
        raise ValueError("reduced angular-power cell interval is invalid")

    exact = compact.central_trace_receipt.regular_initial_bridge.regular_mode_enclosure
    return CompactKappaBinCurvaturePowerCellReceipt(
        curvature_normalized_transfer_receipt=transfer,
        primordial_power_certificate=primordial_power,
        logarithmic_width_enclosure=log_width,
        ell=transfer.ell,
        initial_efold=exact.n,
        initial_kappa_interval=transfer.initial_kappa_interval,
        dimensionless_fixed_wavenumber_interval=(
            harmonic.dimensionless_fixed_wavenumber_interval
        ),
        logarithmic_kappa_and_wavenumber_width_interval=(
            log_width.logarithm_interval
        ),
        primordial_curvature_power_interval=primordial_interval,
        convergence_transfer_per_comoving_curvature_interval=(
            transfer_interval
        ),
        convergence_transfer_modulus_squared_interval=modulus_squared,
        reduced_power_integrand_interval=integrand,
        reduced_angular_power_cell_interval=power_cell,
        fixed_initial_slice_shared_wavenumber_scale_proven=True,
        d_log_k_equals_d_log_q_equals_d_log_kappa_proven=True,
        exact_rational_logarithmic_bin_width_enclosed=True,
        binwide_primordial_power_and_transfer_envelopes_used=True,
        signed_transfer_squared_before_power_aggregation=True,
        nonnegative_reduced_power_integrand_enclosed=True,
        four_pi_d_log_k_angular_power_identity_reused=True,
        four_pi_factor_kept_symbolic=True,
        single_compact_bin_reduced_angular_power_contribution_enclosed=True,
    )
