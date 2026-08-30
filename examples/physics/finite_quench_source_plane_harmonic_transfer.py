"""Full-sky source-plane harmonic transfer for one compact CE kappa bin.

For the adopted flat-Born convention,

    Delta_l^kappa / A
      = -l(l+1) integral dchi (chi_s-chi)/(chi_s chi)
          (T_W/A) j_l(k chi).

Here ``Delta_l`` is the radial transfer in the plane-wave convention

    kappa_lm = 4 pi i^l integral d^3k/(2 pi)^3
               A(k) Delta_l(k) Y_lm^*(khat),

not the literal harmonic coefficient of one fixed real cosine mode.

Writing ``x = q * chibar`` removes the apparent observer singularity:

    dchi G j_l = dchibar * ((chibar_s-chibar)/chibar_s)
                 * q * (j_l(x)/x).

For l >= 2 and 0 <= x <= 2, ``j_l(x)/x`` has a decreasing alternating
series with exact-rational coefficients.  Cell-wide Bessel, Weyl, distance,
and measure intervals therefore give a rigorous compact-bin transfer
enclosure per free potential amplitude.  No primordial-curvature
normalization or angular power spectrum is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    CompactKappaBinWeylTransferEnclosureReceipt,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    _RationalInterval,
    _certified_component_sign,
    _interval_abs_upper,
    _interval_add,
    _interval_divide,
    _interval_multiply,
    _interval_scale,
    _outward_dyadic,
    _point_interval,
)


def _nonnegative_interval_power(
    interval: _RationalInterval,
    exponent: int,
) -> _RationalInterval:
    if interval.lower < 0 or exponent < 0:
        raise ValueError("nonnegative interval power has an invalid domain")
    return _outward_dyadic(
        _RationalInterval(
            interval.lower**exponent,
            interval.upper**exponent,
        )
    )


def _odd_double_factorial(value: int) -> int:
    if value < 1 or value % 2 == 0:
        raise ValueError("odd double factorial requires a positive odd integer")
    result = 1
    for factor in range(1, value + 1, 2):
        result *= factor
    return result


def _spherical_bessel_over_x_interval(
    *,
    ell: int,
    argument: _RationalInterval,
    highest_partial_sum_order: int,
) -> tuple[_RationalInterval, Fraction, Fraction]:
    """Enclose j_ell(x)/x by its decreasing alternating rational series."""

    if ell < 2:
        raise ValueError("regular observer limit requires ell >= 2")
    if argument.lower < 0 or argument.upper > 2:
        raise ValueError("Bessel argument interval must lie in [0, 2]")
    if not 0 <= highest_partial_sum_order <= 64:
        raise ValueError("Bessel partial-sum order must lie in [0, 64]")

    argument_squared = _nonnegative_interval_power(argument, 2)
    term = _interval_scale(
        _nonnegative_interval_power(argument, ell - 1),
        Fraction(1, _odd_double_factorial(2 * ell + 1)),
    )
    partial = term
    for order in range(1, highest_partial_sum_order + 1):
        denominator = 2 * order * (2 * ell + 2 * order + 1)
        term = _interval_multiply(
            term,
            _interval_scale(
                argument_squared,
                Fraction(-1, denominator),
            ),
        )
        partial = _interval_add(partial, term)

    omitted_order = highest_partial_sum_order + 1
    omitted_denominator = (
        2 * omitted_order * (2 * ell + 2 * omitted_order + 1)
    )
    first_omitted = _interval_multiply(
        term,
        _interval_scale(
            argument_squared,
            Fraction(-1, omitted_denominator),
        ),
    )
    first_omitted_abs_upper = _interval_abs_upper(first_omitted)
    maximum_successive_term_ratio = (
        argument.upper * argument.upper / (2 * (2 * ell + 3))
    )
    if maximum_successive_term_ratio >= 1:
        raise ValueError("Bessel alternating terms are not decreasing")
    enclosed = _interval_add(
        partial,
        _RationalInterval(
            -first_omitted_abs_upper,
            first_omitted_abs_upper,
        ),
    )
    if enclosed.upper < 0:
        raise ValueError("Bessel enclosure contradicted certified nonnegativity")
    enclosed = _RationalInterval(max(Fraction(0), enclosed.lower), enclosed.upper)
    return (
        enclosed,
        first_omitted_abs_upper,
        maximum_successive_term_ratio,
    )


@dataclass(frozen=True)
class CompactKappaBinSourcePlaneHarmonicTransferReceipt:
    """Conditional source-plane Delta_l^kappa/A interval on one kappa bin."""

    ell: int
    bessel_highest_partial_sum_order: int
    initial_kappa_interval: tuple[Fraction, Fraction]
    central_initial_kappa: Fraction
    dimensionless_fixed_wavenumber_interval: tuple[Fraction, Fraction]
    dimensionless_source_distance_interval: tuple[Fraction, Fraction]
    maximum_dimensionless_bessel_argument_upper_bound: Fraction
    dimensionless_bessel_argument_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    spherical_bessel_over_x_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    spherical_bessel_first_omitted_term_cell_abs_upper_bounds: (
        tuple[Fraction, ...]
    )
    spherical_bessel_successive_term_ratio_cell_upper_bounds: (
        tuple[Fraction, ...]
    )
    dimensionless_source_fraction_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    normalized_weyl_average_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    normalized_radial_integrand_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    normalized_radial_integral_interval: tuple[Fraction, Fraction]
    normalized_convergence_harmonic_transfer_interval: (
        tuple[Fraction, Fraction]
    )
    normalized_convergence_harmonic_transfer_certified_sign: int | None
    four_pi_i_ell_plane_wave_harmonic_convention_adopted: bool
    source_to_observer_mesh_orientation_fixed: bool
    positive_conformal_measure_reversed_to_observer_radial_integral: bool
    lensing_potential_minus_two_convention_adopted: bool
    convergence_minus_half_angular_laplacian_adopted: bool
    lensing_factor_two_cancellation_proven: bool
    observer_inverse_distance_singularity_removed_by_bessel_ratio: bool
    all_bessel_arguments_within_exact_series_domain: bool
    exact_rational_decreasing_alternating_bessel_series_enclosed: bool
    spherical_bessel_over_x_nonnegative_on_certified_domain_proven: bool
    binwide_not_nodewise_weyl_envelopes_used: bool
    compact_kappa_bin_source_plane_harmonic_transfer_enclosed: bool
    normalized_by_free_potential_amplitude: bool
    physical_wavenumber_bin_calibrated: bool = False
    primordial_curvature_to_potential_normalization_supplied: bool = False
    primordial_curvature_power_spectrum_supplied: bool = False
    all_k_einstein_boltzmann_transfer_enclosed: bool = False
    source_population_distribution_supplied: bool = False
    post_born_or_relativistic_corrections_enclosed: bool = False
    angular_power_spectrum_enclosed: bool = False
    covariance_or_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_COMPACT_KAPPA_BIN_SINGLE_SOURCE_PLANE_FULL_SKY_"
        "CONVERGENCE_HARMONIC_TRANSFER_PER_FREE_POTENTIAL_AMPLITUDE_NOT_"
        "PRIMORDIAL_ALL_K_SOURCE_POPULATION_POWER_OR_LIKELIHOOD"
    )


def project_compact_kappa_bin_to_source_plane_harmonic_transfer(
    compact_bin: CompactKappaBinWeylTransferEnclosureReceipt,
    *,
    ell: object,
    bessel_highest_partial_sum_order: object = 16,
) -> CompactKappaBinSourcePlaneHarmonicTransferReceipt:
    """Project a bin-wide Weyl tube through the exact full-sky Born kernel."""

    if not isinstance(compact_bin, CompactKappaBinWeylTransferEnclosureReceipt):
        raise ValueError("compact_bin has the wrong receipt type")
    if isinstance(ell, bool) or not isinstance(ell, Integral):
        raise ValueError("ell must be an integer")
    harmonic = int(ell)
    if harmonic < 2:
        raise ValueError("source-plane convergence transfer requires ell >= 2")
    if (
        isinstance(bessel_highest_partial_sum_order, bool)
        or not isinstance(bessel_highest_partial_sum_order, Integral)
    ):
        raise ValueError("Bessel partial-sum order must be an integer")
    bessel_order = int(bessel_highest_partial_sum_order)
    if not 0 <= bessel_order <= 64:
        raise ValueError("Bessel partial-sum order must lie in [0, 64]")
    if not (
        compact_bin.compact_kappa_bin_uniform_weyl_path_tube_enclosed
        and compact_bin.cellwise_duhamel_sensitivity_recurrence_proven
        and compact_bin
        .zero_anisotropic_stress_weyl_average_equals_curvature_adopted
    ):
        raise ValueError("compact-bin Weyl proof prerequisites are incomplete")

    normalized_weyl = (
        compact_bin.normalized_compact_bin_weyl_average_cell_intervals
    )
    if normalized_weyl is None:
        raise ValueError("harmonic transfer requires nonzero amplitude normalization")
    central = compact_bin.central_initial_kappa
    lower_kappa, upper_kappa = compact_bin.initial_kappa_interval
    if not 0 < lower_kappa <= central <= upper_kappa:
        raise ValueError("compact kappa-bin provenance is invalid")

    born = (
        compact_bin.central_trace_receipt
        .fixed_mode_born_lensing_absolute_envelope
    )
    if not (
        born.positive_conformal_cell_measure_enclosed
        and born.source_distance_identity_enclosed_by_intersection
        and born.exact_rational_dimensionless_fixed_wavenumber_enclosed
    ):
        raise ValueError("source-plane distance proof prerequisites are incomplete")
    cell_measures = tuple(
        _RationalInterval(*value)
        for value in born.dimensionless_conformal_cell_measure_intervals
    )
    source_side = tuple(
        _RationalInterval(*value)
        for value in born.dimensionless_source_side_distance_node_intervals
    )
    observer_side = tuple(
        _RationalInterval(*value)
        for value in born.dimensionless_observer_side_distance_node_intervals
    )
    source_distance = _RationalInterval(
        *born.dimensionless_source_distance_interval
    )
    cell_count = len(cell_measures)
    if not (
        len(normalized_weyl) == cell_count
        and len(source_side) == cell_count + 1
        and len(observer_side) == cell_count + 1
        and compact_bin.central_trace_receipt.refined_step_count == cell_count
    ):
        raise ValueError("harmonic transfer cell evidence is not aligned")

    central_q = _RationalInterval(
        *born.dimensionless_fixed_wavenumber_interval
    )
    kappa_ratio = _RationalInterval(
        lower_kappa / central,
        upper_kappa / central,
    )
    q_interval = _interval_multiply(central_q, kappa_ratio)
    maximum_observer_distance_upper = max(
        value.upper for value in observer_side
    )
    maximum_argument = (
        q_interval.upper * maximum_observer_distance_upper
    )
    if maximum_argument > 2:
        raise ValueError(
            "compact bin exceeds the certified Bessel argument domain x <= 2"
        )

    argument_cells: list[_RationalInterval] = []
    bessel_cells: list[_RationalInterval] = []
    bessel_tail_bounds: list[Fraction] = []
    bessel_ratio_bounds: list[Fraction] = []
    source_fraction_cells: list[_RationalInterval] = []
    integrand_cells: list[_RationalInterval] = []
    radial_integral = _point_interval(0)

    for index, (measure, weyl_pair) in enumerate(
        zip(cell_measures, normalized_weyl, strict=True)
    ):
        observer_distance_cell = _RationalInterval(
            observer_side[index + 1].lower,
            observer_side[index].upper,
        )
        argument = _interval_multiply(q_interval, observer_distance_cell)
        bessel_over_x, tail_bound, term_ratio = (
            _spherical_bessel_over_x_interval(
                ell=harmonic,
                argument=argument,
                highest_partial_sum_order=bessel_order,
            )
        )
        source_separation_cell = _RationalInterval(
            source_side[index].lower,
            source_side[index + 1].upper,
        )
        source_fraction_natural = _interval_divide(
            source_separation_cell,
            source_distance,
        )
        source_fraction = _RationalInterval(
            max(Fraction(0), source_fraction_natural.lower),
            min(Fraction(1), source_fraction_natural.upper),
        )
        if source_fraction.lower > source_fraction.upper:
            raise ValueError("source fraction identity intersection is empty")

        integrand = _interval_multiply(
            measure,
            _interval_multiply(
                source_fraction,
                _interval_multiply(
                    q_interval,
                    _interval_multiply(
                        bessel_over_x,
                        _RationalInterval(*weyl_pair),
                    ),
                ),
            ),
        )
        radial_integral = _interval_add(radial_integral, integrand)
        argument_cells.append(argument)
        bessel_cells.append(bessel_over_x)
        bessel_tail_bounds.append(tail_bound)
        bessel_ratio_bounds.append(term_ratio)
        source_fraction_cells.append(source_fraction)
        integrand_cells.append(integrand)

    convergence_transfer = _interval_scale(
        radial_integral,
        -harmonic * (harmonic + 1),
    )

    def pairs(
        values: list[_RationalInterval],
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        return tuple((value.lower, value.upper) for value in values)

    transfer_pair = (
        convergence_transfer.lower,
        convergence_transfer.upper,
    )
    return CompactKappaBinSourcePlaneHarmonicTransferReceipt(
        ell=harmonic,
        bessel_highest_partial_sum_order=bessel_order,
        initial_kappa_interval=(lower_kappa, upper_kappa),
        central_initial_kappa=central,
        dimensionless_fixed_wavenumber_interval=(
            q_interval.lower,
            q_interval.upper,
        ),
        dimensionless_source_distance_interval=(
            source_distance.lower,
            source_distance.upper,
        ),
        maximum_dimensionless_bessel_argument_upper_bound=(
            maximum_argument
        ),
        dimensionless_bessel_argument_cell_intervals=pairs(argument_cells),
        spherical_bessel_over_x_cell_intervals=pairs(bessel_cells),
        spherical_bessel_first_omitted_term_cell_abs_upper_bounds=tuple(
            bessel_tail_bounds
        ),
        spherical_bessel_successive_term_ratio_cell_upper_bounds=tuple(
            bessel_ratio_bounds
        ),
        dimensionless_source_fraction_cell_intervals=pairs(
            source_fraction_cells
        ),
        normalized_weyl_average_cell_intervals=tuple(normalized_weyl),
        normalized_radial_integrand_cell_intervals=pairs(integrand_cells),
        normalized_radial_integral_interval=(
            radial_integral.lower,
            radial_integral.upper,
        ),
        normalized_convergence_harmonic_transfer_interval=transfer_pair,
        normalized_convergence_harmonic_transfer_certified_sign=(
            _certified_component_sign(transfer_pair)
        ),
        four_pi_i_ell_plane_wave_harmonic_convention_adopted=True,
        source_to_observer_mesh_orientation_fixed=True,
        positive_conformal_measure_reversed_to_observer_radial_integral=True,
        lensing_potential_minus_two_convention_adopted=True,
        convergence_minus_half_angular_laplacian_adopted=True,
        lensing_factor_two_cancellation_proven=True,
        observer_inverse_distance_singularity_removed_by_bessel_ratio=True,
        all_bessel_arguments_within_exact_series_domain=True,
        exact_rational_decreasing_alternating_bessel_series_enclosed=True,
        spherical_bessel_over_x_nonnegative_on_certified_domain_proven=True,
        binwide_not_nodewise_weyl_envelopes_used=True,
        compact_kappa_bin_source_plane_harmonic_transfer_enclosed=True,
        normalized_by_free_potential_amplitude=True,
    )
