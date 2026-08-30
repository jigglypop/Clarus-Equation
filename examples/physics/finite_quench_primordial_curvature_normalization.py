"""Normalize one compact-bin CE harmonic transfer by comoving curvature.

The source-off regular mode uses

    ds^2 = a^2 [-(1 + 2 phi) d tau^2 + (1 - 2 psi) dx^2],
    theta = -k^2 v,

and the CMB-sign comoving-curvature convention

    R = psi - H_conformal v.

Writing ``h = d ln H / dn = -3(1+w)/2``, the source-off Einstein
constraints cancel their explicit ``K = kappa_i^2`` terms and give

    R = psi - (psi_n + psi) / h.

The exact regular-series intervals at the central ``u = kappa_i^2`` are
widened by the theorem-34 initial parameter-sensitivity bound.  Dividing the
theorem-35 interval ``Delta_l^kappa/A`` by the resulting sign-separated
``R/A`` interval then encloses ``Delta_l^kappa/R`` on the whole compact bin.

All quantities in this normalization are dimensionless.  The quotient
deliberately discards the correlation between numerator and denominator, so
it is rigorous but generally conservative.  This remains a conditional
single-bin, single-source-plane transfer: it does not derive a primordial
power spectrum, an all-k Einstein--Boltzmann solution, an angular power
spectrum, or a likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    CompactKappaBinWeylTransferEnclosureReceipt,
)
from examples.physics.finite_quench_source_plane_harmonic_transfer import (
    CompactKappaBinSourcePlaneHarmonicTransferReceipt,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    _RationalInterval,
    _certified_component_sign,
    _interval_add,
    _interval_divide,
    _interval_subtract,
    _point_interval,
)


def _source_off_normalized_comoving_curvature_interval(
    *,
    normalized_curvature: _RationalInterval,
    normalized_curvature_prime: _RationalInterval,
    hubble_log_derivative: Fraction,
) -> _RationalInterval:
    """Evaluate R/A = psi/A - (psi_n/A + psi/A)/h."""

    h = Fraction(hubble_log_derivative)
    if h >= 0:
        raise ValueError(
            "source-off CMB curvature convention requires negative "
            "hubble log derivative"
        )
    velocity = _interval_divide(
        _interval_add(normalized_curvature, normalized_curvature_prime),
        _point_interval(h),
    )
    return _interval_subtract(normalized_curvature, velocity)


@dataclass(frozen=True)
class PrimordialCurvatureNormalizedHarmonicTransferReceipt:
    """Conditional compact-bin ``Delta_l^kappa/R`` enclosure."""

    compact_bin_receipt: CompactKappaBinWeylTransferEnclosureReceipt = field(
        repr=False
    )
    source_plane_harmonic_transfer_receipt: (
        CompactKappaBinSourcePlaneHarmonicTransferReceipt
    ) = field(repr=False)
    ell: int
    initial_kappa_interval: tuple[Fraction, Fraction]
    central_initial_kappa: Fraction
    initial_kappa_squared_interval: tuple[Fraction, Fraction]
    central_initial_kappa_squared: Fraction
    maximum_initial_kappa_squared_distance_from_center: Fraction
    free_potential_amplitude: Fraction
    reservoir_equation_of_state: Fraction
    source_off_hubble_log_derivative: Fraction
    central_normalized_curvature_interval: tuple[Fraction, Fraction]
    central_normalized_curvature_prime_interval: tuple[Fraction, Fraction]
    normalized_initial_parameter_derivative_l2_upper_bound_via_l1: Fraction
    normalized_initial_curvature_parameter_derivative_abs_upper_bound: Fraction
    normalized_initial_curvature_prime_parameter_derivative_abs_upper_bound: (
        Fraction
    )
    normalized_initial_curvature_parameter_variation_radius_upper_bound: Fraction
    normalized_initial_curvature_prime_parameter_variation_radius_upper_bound: (
        Fraction
    )
    compact_bin_normalized_curvature_interval: tuple[Fraction, Fraction]
    compact_bin_normalized_curvature_prime_interval: tuple[Fraction, Fraction]
    compact_bin_normalized_comoving_curvature_interval: (
        tuple[Fraction, Fraction]
    )
    compact_bin_normalized_comoving_curvature_certified_sign: int
    zero_k_normalized_comoving_curvature_limit: Fraction
    zero_k_potential_to_comoving_curvature_ratio: Fraction
    matter_era_potential_to_comoving_curvature_ratio: Fraction
    normalized_convergence_harmonic_transfer_per_free_amplitude_interval: (
        tuple[Fraction, Fraction]
    )
    normalized_convergence_harmonic_transfer_per_comoving_curvature_interval: (
        tuple[Fraction, Fraction]
    )
    normalized_convergence_harmonic_transfer_per_comoving_curvature_sign: (
        int
    )
    newtonian_gauge_metric_minus_two_psi_spatial_convention_adopted: bool
    velocity_divergence_theta_equals_minus_k_squared_v_adopted: bool
    cmb_positive_primordial_comoving_curvature_convention_adopted: bool
    hubble_log_derivative_sign_and_definition_locked: bool
    source_off_constraint_explicit_kappa_term_cancellation_proven: bool
    exact_regular_source_off_adiabatic_mode_reused: bool
    initial_superhorizon_compact_bin_proven: bool
    initial_parameter_sensitivity_component_enclosure_reused: bool
    coordinatewise_initial_parameter_sensitivity_reconstructed_from_series: bool
    finite_initial_comoving_curvature_normalization_enclosed: bool
    matter_era_three_fifths_normalization_recovered: bool
    exact_rational_outward_interval_operations_used: bool
    numerator_denominator_correlation_discarded_conservatively: bool
    dimensionless_normalization_proven: bool
    compact_bin_harmonic_transfer_per_comoving_curvature_enclosed: bool
    physical_wavenumber_bin_calibrated: bool = False
    primordial_curvature_power_spectrum_supplied: bool = False
    inflationary_state_or_spectrum_derived: bool = False
    all_k_einstein_boltzmann_transfer_enclosed: bool = False
    source_population_distribution_supplied: bool = False
    post_born_or_relativistic_corrections_enclosed: bool = False
    angular_power_spectrum_enclosed: bool = False
    covariance_or_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_COMPACT_KAPPA_BIN_SINGLE_SOURCE_PLANE_FULL_SKY_"
        "CONVERGENCE_HARMONIC_TRANSFER_PER_PRIMORDIAL_COMOVING_CURVATURE_"
        "NOT_PRIMORDIAL_SPECTRUM_ALL_K_SOURCE_POPULATION_POWER_OR_LIKELIHOOD"
    )


def normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
    compact_bin: CompactKappaBinWeylTransferEnclosureReceipt,
    harmonic_transfer: CompactKappaBinSourcePlaneHarmonicTransferReceipt,
) -> PrimordialCurvatureNormalizedHarmonicTransferReceipt:
    """Convert theorem 35's ``Delta_l^kappa/A`` to ``Delta_l^kappa/R``."""

    if not isinstance(compact_bin, CompactKappaBinWeylTransferEnclosureReceipt):
        raise ValueError("compact_bin has the wrong receipt type")
    if not isinstance(
        harmonic_transfer,
        CompactKappaBinSourcePlaneHarmonicTransferReceipt,
    ):
        raise ValueError("harmonic_transfer has the wrong receipt type")
    if compact_bin.primordial_potential_amplitude == 0:
        raise ValueError("comoving-curvature normalization requires nonzero A")
    if not (
        compact_bin.compact_kappa_bin_uniform_weyl_path_tube_enclosed
        and compact_bin.cellwise_duhamel_sensitivity_recurrence_proven
        and compact_bin.initial_bin_is_superhorizon
        and compact_bin
        .exact_regular_series_parameter_derivative_conservatively_enclosed
    ):
        raise ValueError("compact-bin curvature proof prerequisites are incomplete")
    if not (
        harmonic_transfer.compact_kappa_bin_source_plane_harmonic_transfer_enclosed
        and harmonic_transfer.normalized_by_free_potential_amplitude
        and harmonic_transfer.four_pi_i_ell_plane_wave_harmonic_convention_adopted
    ):
        raise ValueError("harmonic-transfer proof prerequisites are incomplete")

    normalized_weyl = compact_bin.normalized_compact_bin_weyl_average_cell_intervals
    if (
        harmonic_transfer.initial_kappa_interval
        != compact_bin.initial_kappa_interval
        or harmonic_transfer.central_initial_kappa
        != compact_bin.central_initial_kappa
        or normalized_weyl is None
        or harmonic_transfer.normalized_weyl_average_cell_intervals
        != normalized_weyl
    ):
        raise ValueError("compact-bin and harmonic-transfer provenance mismatch")

    exact = (
        compact_bin.central_trace_receipt.regular_initial_bridge
        .regular_mode_enclosure
    )
    if not (
        exact.source_off_pure_reservoir_series_equation_proven
        and exact.exact_series_recurrence_proven
        and exact.exact_rational_tail_enclosures_proven
        and exact.unique_past_bounded_regular_mode_enclosed
        and exact.normalized_dimensionless_series_proven
    ):
        raise ValueError("exact regular-mode proof prerequisites are incomplete")
    amplitude = compact_bin.primordial_potential_amplitude
    if (
        exact.primordial_potential_amplitude != amplitude
        or exact.kappa_initial != compact_bin.central_initial_kappa
        or exact.kappa_initial_squared
        != compact_bin.central_initial_kappa_squared
    ):
        raise ValueError("regular-mode and compact-bin provenance mismatch")

    normalized_curvature = _interval_divide(
        _RationalInterval(*exact.curvature_interval),
        _point_interval(amplitude),
    )
    normalized_curvature_prime = _interval_divide(
        _RationalInterval(*exact.curvature_prime_interval),
        _point_interval(amplitude),
    )
    derivative_nodes = compact_bin.normalized_parameter_derivative_node_upper_bounds
    if not derivative_nodes or derivative_nodes[0] < 0:
        raise ValueError("initial parameter-sensitivity evidence is invalid")
    derivative_bound = derivative_nodes[0]
    if (
        derivative_bound
        != compact_bin
        .normalized_initial_parameter_derivative_l2_upper_bound_via_l1
    ):
        raise ValueError("initial parameter-sensitivity provenance mismatch")
    w = exact.reservoir_equation_of_state
    if not Fraction(0) <= w <= Fraction(1):
        raise ValueError("source-off reservoir equation of state is invalid")
    rate = 1 + 3 * w
    friction = (5 + 3 * w) / 2
    _, curvature_ratio, curvature_prime_ratio = (
        compact_bin.initial_series_ratio_upper_bounds
    )
    curvature_derivative_bound = (
        (w / (rate * (rate + friction))) / (1 - curvature_ratio)
    )
    curvature_prime_derivative_bound = (
        (w / (rate + friction)) / (1 - curvature_prime_ratio)
    )
    if curvature_derivative_bound + curvature_prime_derivative_bound != derivative_bound:
        raise ValueError("coordinatewise initial sensitivity provenance mismatch")
    delta_u = compact_bin.maximum_initial_kappa_squared_distance_from_center
    curvature_parameter_radius = delta_u * curvature_derivative_bound
    curvature_prime_parameter_radius = (
        delta_u * curvature_prime_derivative_bound
    )
    bin_curvature = _interval_add(
        normalized_curvature,
        _RationalInterval(
            -curvature_parameter_radius,
            curvature_parameter_radius,
        ),
    )
    bin_curvature_prime = _interval_add(
        normalized_curvature_prime,
        _RationalInterval(
            -curvature_prime_parameter_radius,
            curvature_prime_parameter_radius,
        ),
    )

    h = -3 * (1 + w) / 2
    normalized_comoving_curvature = (
        _source_off_normalized_comoving_curvature_interval(
            normalized_curvature=bin_curvature,
            normalized_curvature_prime=bin_curvature_prime,
            hubble_log_derivative=h,
        )
    )
    curvature_pair = (
        normalized_comoving_curvature.lower,
        normalized_comoving_curvature.upper,
    )
    curvature_sign = _certified_component_sign(curvature_pair)
    if curvature_sign is None:
        raise ValueError("normalized comoving-curvature interval crossed zero")

    amplitude_transfer = _RationalInterval(
        *harmonic_transfer.normalized_convergence_harmonic_transfer_interval
    )
    curvature_transfer = _interval_divide(
        amplitude_transfer,
        normalized_comoving_curvature,
    )
    curvature_transfer_pair = (
        curvature_transfer.lower,
        curvature_transfer.upper,
    )
    curvature_transfer_sign = _certified_component_sign(curvature_transfer_pair)
    if curvature_transfer_sign is None:
        raise ValueError("comoving-curvature harmonic transfer has no certified sign")

    zero_k_curvature = (5 + 3 * w) / (3 * (1 + w))
    zero_k_potential_ratio = 1 / zero_k_curvature
    return PrimordialCurvatureNormalizedHarmonicTransferReceipt(
        compact_bin_receipt=compact_bin,
        source_plane_harmonic_transfer_receipt=harmonic_transfer,
        ell=harmonic_transfer.ell,
        initial_kappa_interval=compact_bin.initial_kappa_interval,
        central_initial_kappa=compact_bin.central_initial_kappa,
        initial_kappa_squared_interval=compact_bin.initial_kappa_squared_interval,
        central_initial_kappa_squared=compact_bin.central_initial_kappa_squared,
        maximum_initial_kappa_squared_distance_from_center=(
            compact_bin.maximum_initial_kappa_squared_distance_from_center
        ),
        free_potential_amplitude=amplitude,
        reservoir_equation_of_state=w,
        source_off_hubble_log_derivative=h,
        central_normalized_curvature_interval=(
            normalized_curvature.lower,
            normalized_curvature.upper,
        ),
        central_normalized_curvature_prime_interval=(
            normalized_curvature_prime.lower,
            normalized_curvature_prime.upper,
        ),
        normalized_initial_parameter_derivative_l2_upper_bound_via_l1=(
            derivative_bound
        ),
        normalized_initial_curvature_parameter_derivative_abs_upper_bound=(
            curvature_derivative_bound
        ),
        normalized_initial_curvature_prime_parameter_derivative_abs_upper_bound=(
            curvature_prime_derivative_bound
        ),
        normalized_initial_curvature_parameter_variation_radius_upper_bound=(
            curvature_parameter_radius
        ),
        normalized_initial_curvature_prime_parameter_variation_radius_upper_bound=(
            curvature_prime_parameter_radius
        ),
        compact_bin_normalized_curvature_interval=(
            bin_curvature.lower,
            bin_curvature.upper,
        ),
        compact_bin_normalized_curvature_prime_interval=(
            bin_curvature_prime.lower,
            bin_curvature_prime.upper,
        ),
        compact_bin_normalized_comoving_curvature_interval=curvature_pair,
        compact_bin_normalized_comoving_curvature_certified_sign=curvature_sign,
        zero_k_normalized_comoving_curvature_limit=zero_k_curvature,
        zero_k_potential_to_comoving_curvature_ratio=zero_k_potential_ratio,
        matter_era_potential_to_comoving_curvature_ratio=Fraction(3, 5),
        normalized_convergence_harmonic_transfer_per_free_amplitude_interval=(
            amplitude_transfer.lower,
            amplitude_transfer.upper,
        ),
        normalized_convergence_harmonic_transfer_per_comoving_curvature_interval=(
            curvature_transfer.lower,
            curvature_transfer.upper,
        ),
        normalized_convergence_harmonic_transfer_per_comoving_curvature_sign=(
            curvature_transfer_sign
        ),
        newtonian_gauge_metric_minus_two_psi_spatial_convention_adopted=True,
        velocity_divergence_theta_equals_minus_k_squared_v_adopted=True,
        cmb_positive_primordial_comoving_curvature_convention_adopted=True,
        hubble_log_derivative_sign_and_definition_locked=True,
        source_off_constraint_explicit_kappa_term_cancellation_proven=True,
        exact_regular_source_off_adiabatic_mode_reused=True,
        initial_superhorizon_compact_bin_proven=True,
        initial_parameter_sensitivity_component_enclosure_reused=True,
        coordinatewise_initial_parameter_sensitivity_reconstructed_from_series=(
            True
        ),
        finite_initial_comoving_curvature_normalization_enclosed=True,
        matter_era_three_fifths_normalization_recovered=True,
        exact_rational_outward_interval_operations_used=True,
        numerator_denominator_correlation_discarded_conservatively=True,
        dimensionless_normalization_proven=True,
        compact_bin_harmonic_transfer_per_comoving_curvature_enclosed=True,
    )
