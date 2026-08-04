"""Cutoff-independent certificates for the explicit global throat targets.

The original inverse-designed geometry has a finite ADM mass and satisfies the
anisotropic conservation identity exactly.  It nevertheless retains an
``r^-3`` radial-pressure tail, so its radial affine ANEC is finite and negative
while its volume integral of ``rho + p_r`` diverges logarithmically.

The localized comparison keeps the same shape function and exact Casimir
throat data but matches the redshift to the Schwarzschild tail.  Its stress is
exponentially localized.  A positive throat scalar-kinetic reconstruction is
not enough, however: an explicit finite-radius counterexample keeps the global
healthy-nonminimal-scalar gate closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_SHAPE_LIMIT = 2.0 / 3.0
_GLOBAL_KINETIC_COUNTEREXAMPLE_RADIUS = 37.0 / 32.0


@dataclass(frozen=True)
class OriginalGlobalThroatCertificate:
    throat_shape: float
    throat_shape_derivative: float
    throat_redshift_derivative: float
    throat_density: float
    throat_radial_pressure: float
    throat_tangential_pressure: float
    throat_matches_ideal_casimir: bool
    shape_gap_at_throat: float
    shape_gap_derivative_infimum: float
    shape_gap_positive_for_every_x_above_one: bool
    metric_factor_throat_slope: float
    two_sided_extension_available: bool
    lapse_squared_infimum: float
    lapse_squared_supremum: float
    horizon_free_for_every_x: bool
    asymptotic_shape_limit: float
    asymptotically_flat_exact: bool
    adm_mass_per_end_in_throat_radii: float
    adm_mass_per_end_positive: bool
    bianchi_conservation_identity_exact: bool
    independent_matter_eom_derived: bool
    radial_nec_negative_for_every_x: bool
    radial_pressure_cubic_tail_coefficient: float
    tangential_pressure_cubic_tail_coefficient: float
    radial_affine_killing_energy_normalization: float
    radial_affine_anec_dimensionless: float
    radial_affine_anec_finite: bool
    radial_affine_anec_negative: bool
    coordinate_volume_nec_log_coefficient: float
    volume_nec_diverges_logarithmically: bool
    coordinate_volume_nec_finite: bool
    proper_volume_nec_finite: bool
    stress_l1_localized: bool


@dataclass(frozen=True)
class LocalizedPhiMatchCertificate:
    throat_shape: float
    throat_shape_derivative: float
    throat_redshift_derivative: float
    throat_redshift_second_derivative: float
    throat_density: float
    throat_radial_pressure: float
    throat_tangential_pressure: float
    throat_matches_ideal_casimir: bool
    shape_gap_positive_for_every_x_above_one: bool
    metric_factor_throat_slope: float
    lapse_squared_global_lower_bound: float
    horizon_free_for_every_x: bool
    asymptotic_shape_limit: float
    asymptotically_schwarzschild: bool
    adm_mass_per_end_in_throat_radii: float
    adm_mass_per_end_positive: bool
    bianchi_conservation_identity_exact: bool
    independent_matter_eom_derived: bool
    radial_nec_negative_for_every_x: bool
    radial_affine_killing_energy_normalization: float
    radial_affine_anec_dimensionless: float
    radial_affine_anec_finite: bool
    radial_affine_anec_negative: bool
    coordinate_volume_nec_dimensionless_per_end: float
    proper_volume_nec_dimensionless_per_end: float
    coordinate_volume_nec_finite: bool
    proper_volume_nec_finite: bool
    volume_nec_diverges_logarithmically: bool
    stress_tail_exponentially_localized: bool
    stress_l1_localized: bool
    local_nonminimal_kinetic_over_planck_factor: float
    local_nonminimal_kinetic_positive: bool
    global_kinetic_counterexample_radius: float
    global_kinetic_counterexample_value: float
    minimum_sampled_kinetic_over_planck_factor: float
    minimum_sampled_kinetic_radius: float
    global_nonminimal_kinetic_positive: bool
    potential_reconstructed: bool
    perturbative_stability_derived: bool


@dataclass(frozen=True)
class GlobalThroatExactCertificate:
    quadrature_order: int
    kinetic_sample_count: int
    kinetic_radial_cutoff: float
    original: OriginalGlobalThroatCertificate
    localized_phi_match: LocalizedPhiMatchCertificate


def _strict_integer(
    value: int,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _strict_kinetic_cutoff(value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("kinetic_radial_cutoff must be a real number")
    cutoff = float(value)
    if not math.isfinite(cutoff) or cutoff < 2.0 or cutoff > 1.0e4:
        raise ValueError("kinetic_radial_cutoff must be finite and between 2 and 10000")
    return cutoff


def _half_line_grid(quadrature_order: int) -> tuple[np.ndarray, ...]:
    """Return a Gauss--Legendre grid for ``x=1+u^2``, ``u=v/(1-v)``."""

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    v = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    u = v / (1.0 - v)
    u_squared = u * u
    x = 1.0 + u_squared
    exponential = np.exp(-u_squared)
    # This form remains accurate at the throat, where both terms vanish.
    metric_factor = (3.0 * u_squared - np.expm1(-u_squared)) / (3.0 * x)
    jacobian = 2.0 * u / (1.0 - v) ** 2
    return x, exponential, metric_factor, jacobian, weights


def _original_radial_nec(x: np.ndarray, exponential: np.ndarray) -> np.ndarray:
    numerator = 2.0 + exponential * (3.0 * x**2 - x + 1.0) - x * exponential**2
    return -numerator / (3.0 * x**3)


def _localized_radial_nec(
    x: np.ndarray,
    exponential: np.ndarray,
) -> np.ndarray:
    positive_bracket = 1.0 / 3.0 + 1.0 / (3.0 * x - 2.0) + (3.0 * x - 2.0 - exponential)
    return -exponential * positive_bracket / x**2


def _integrated_controls(quadrature_order: int) -> tuple[float, ...]:
    x, exponential, metric_factor, jacobian, weights = _half_line_grid(quadrature_order)
    square_root_metric = np.sqrt(metric_factor)

    original_nec = _original_radial_nec(x, exponential)
    original_redshift = exponential / 2.0
    original_affine_anec = 2.0 * float(
        np.sum(weights * original_nec * np.exp(-original_redshift) / square_root_metric * jacobian)
    )

    localized_nec = _localized_radial_nec(x, exponential)
    localized_redshift = 0.5 * np.log1p(-2.0 / (3.0 * x)) + 1.5 * exponential
    localized_affine_anec = 2.0 * float(
        np.sum(
            weights * localized_nec * np.exp(-localized_redshift) / square_root_metric * jacobian
        )
    )
    coordinate_volume_nec = float(np.sum(weights * x**2 * localized_nec * jacobian))
    proper_volume_nec = float(
        np.sum(weights * x**2 * localized_nec / square_root_metric * jacobian)
    )
    return (
        original_affine_anec,
        localized_affine_anec,
        coordinate_volume_nec,
        proper_volume_nec,
    )


def _localized_kinetic_over_planck(x: np.ndarray) -> np.ndarray:
    """Return the reconstructed ``K/F`` for the localized redshift target."""

    exponential = np.exp(1.0 - x)
    shape = _SHAPE_LIMIT + exponential / 3.0
    shape_first = -exponential / 3.0
    shape_second = exponential / 3.0

    metric_factor = 1.0 - shape / x
    metric_first = -shape_first / x + shape / x**2
    metric_second = -shape_second / x + 2.0 * shape_first / x**2 - 2.0 * shape / x**3

    redshift_first = _SHAPE_LIMIT / (2.0 * x * (x - _SHAPE_LIMIT)) - 1.5 * exponential
    redshift_second = (
        -_SHAPE_LIMIT * (2.0 * x - _SHAPE_LIMIT) / (2.0 * x**2 * (x - _SHAPE_LIMIT) ** 2)
        + 1.5 * exponential
    )
    redshift_third = (
        _SHAPE_LIMIT
        * (_SHAPE_LIMIT**2 - 3.0 * _SHAPE_LIMIT * x + 3.0 * x**2)
        / (x**3 * (x - _SHAPE_LIMIT) ** 3)
        - 1.5 * exponential
    )

    density = shape_first / x**2
    density_first = shape_second / x**2 - 2.0 * shape_first / x**3
    radial_pressure = (metric_factor - 1.0) / x**2 + 2.0 * metric_factor * redshift_first / x

    angular_core = redshift_second + redshift_first**2 + redshift_first / x
    tangential_pressure = (
        metric_factor * angular_core + 0.5 * metric_first * redshift_first + 0.5 * metric_first / x
    )
    angular_core_first = (
        redshift_third
        + 2.0 * redshift_first * redshift_second
        + redshift_second / x
        - redshift_first / x**2
    )
    tangential_first = (
        metric_first * angular_core
        + metric_factor * angular_core_first
        + 0.5 * metric_second * redshift_first
        + 0.5 * metric_first * redshift_second
        + 0.5 * metric_second / x
        - 0.5 * metric_first / x**2
    )

    slope_numerator = density + tangential_pressure
    slope_numerator_first = density_first + tangential_first
    denominator_core = 1.0 / x - redshift_first
    slope_denominator = metric_factor * denominator_core
    slope_denominator_first = metric_first * denominator_core + metric_factor * (
        -1.0 / x**2 - redshift_second
    )
    logarithmic_planck_slope = slope_numerator / slope_denominator
    logarithmic_planck_slope_first = (
        slope_numerator_first * slope_denominator - slope_numerator * slope_denominator_first
    ) / slope_denominator**2

    return (
        density
        + radial_pressure
        - 0.5 * metric_first * logarithmic_planck_slope
        - metric_factor * (logarithmic_planck_slope**2 + logarithmic_planck_slope_first)
        + metric_factor * redshift_first * logarithmic_planck_slope
    )


def global_throat_exact_certificate(
    *,
    quadrature_order: int = 128,
    kinetic_sample_count: int = 20_000,
    kinetic_radial_cutoff: float = 40.0,
) -> GlobalThroatExactCertificate:
    """Certify the original target and its localized-redshift replacement.

    Exact booleans follow from closed-form inequalities and limits, not from a
    radial cutoff.  Quadrature supplies reproducible magnitudes for the finite
    integrals.  The scalar reconstruction is refuted by the explicit point
    ``x=37/32``; the wider scan only locates the nearby numerical minimum.
    """

    order = _strict_integer(
        quadrature_order,
        "quadrature_order",
        minimum=32,
        maximum=1_024,
    )
    sample_count = _strict_integer(
        kinetic_sample_count,
        "kinetic_sample_count",
        minimum=1_024,
        maximum=1_000_000,
    )
    cutoff = _strict_kinetic_cutoff(kinetic_radial_cutoff)

    (
        original_affine_anec,
        localized_affine_anec,
        localized_coordinate_volume_nec,
        localized_proper_volume_nec,
    ) = _integrated_controls(order)

    kinetic_x = 1.0 + np.geomspace(1.0e-6, cutoff - 1.0, sample_count)
    sampled_kinetic = _localized_kinetic_over_planck(kinetic_x)
    minimum_index = int(np.argmin(sampled_kinetic))
    counterexample_value = float(
        _localized_kinetic_over_planck(np.asarray([_GLOBAL_KINETIC_COUNTEREXAMPLE_RADIUS]))[0]
    )

    original = OriginalGlobalThroatCertificate(
        throat_shape=1.0,
        throat_shape_derivative=-1.0 / 3.0,
        throat_redshift_derivative=-1.0 / 2.0,
        throat_density=-1.0 / 3.0,
        throat_radial_pressure=-1.0,
        throat_tangential_pressure=1.0 / 3.0,
        throat_matches_ideal_casimir=True,
        shape_gap_at_throat=0.0,
        shape_gap_derivative_infimum=1.0,
        shape_gap_positive_for_every_x_above_one=True,
        metric_factor_throat_slope=4.0 / 3.0,
        two_sided_extension_available=True,
        lapse_squared_infimum=1.0,
        lapse_squared_supremum=math.e,
        horizon_free_for_every_x=True,
        asymptotic_shape_limit=_SHAPE_LIMIT,
        asymptotically_flat_exact=True,
        adm_mass_per_end_in_throat_radii=1.0 / 3.0,
        adm_mass_per_end_positive=True,
        bianchi_conservation_identity_exact=True,
        independent_matter_eom_derived=False,
        radial_nec_negative_for_every_x=True,
        radial_pressure_cubic_tail_coefficient=-2.0 / 3.0,
        tangential_pressure_cubic_tail_coefficient=1.0 / 3.0,
        radial_affine_killing_energy_normalization=1.0,
        radial_affine_anec_dimensionless=original_affine_anec,
        radial_affine_anec_finite=math.isfinite(original_affine_anec),
        radial_affine_anec_negative=original_affine_anec < 0.0,
        coordinate_volume_nec_log_coefficient=-2.0 / 3.0,
        volume_nec_diverges_logarithmically=True,
        coordinate_volume_nec_finite=False,
        proper_volume_nec_finite=False,
        stress_l1_localized=False,
    )

    localized = LocalizedPhiMatchCertificate(
        throat_shape=1.0,
        throat_shape_derivative=-1.0 / 3.0,
        throat_redshift_derivative=-1.0 / 2.0,
        throat_redshift_second_derivative=-5.0 / 2.0,
        throat_density=-1.0 / 3.0,
        throat_radial_pressure=-1.0,
        throat_tangential_pressure=1.0 / 3.0,
        throat_matches_ideal_casimir=True,
        shape_gap_positive_for_every_x_above_one=True,
        metric_factor_throat_slope=4.0 / 3.0,
        lapse_squared_global_lower_bound=1.0 / 3.0,
        horizon_free_for_every_x=True,
        asymptotic_shape_limit=_SHAPE_LIMIT,
        asymptotically_schwarzschild=True,
        adm_mass_per_end_in_throat_radii=1.0 / 3.0,
        adm_mass_per_end_positive=True,
        bianchi_conservation_identity_exact=True,
        independent_matter_eom_derived=False,
        radial_nec_negative_for_every_x=True,
        radial_affine_killing_energy_normalization=1.0,
        radial_affine_anec_dimensionless=localized_affine_anec,
        radial_affine_anec_finite=math.isfinite(localized_affine_anec),
        radial_affine_anec_negative=localized_affine_anec < 0.0,
        coordinate_volume_nec_dimensionless_per_end=(localized_coordinate_volume_nec),
        proper_volume_nec_dimensionless_per_end=localized_proper_volume_nec,
        coordinate_volume_nec_finite=math.isfinite(localized_coordinate_volume_nec),
        proper_volume_nec_finite=math.isfinite(localized_proper_volume_nec),
        volume_nec_diverges_logarithmically=False,
        stress_tail_exponentially_localized=True,
        stress_l1_localized=True,
        local_nonminimal_kinetic_over_planck_factor=7.0 / 12.0,
        local_nonminimal_kinetic_positive=True,
        global_kinetic_counterexample_radius=(_GLOBAL_KINETIC_COUNTEREXAMPLE_RADIUS),
        global_kinetic_counterexample_value=counterexample_value,
        minimum_sampled_kinetic_over_planck_factor=float(sampled_kinetic[minimum_index]),
        minimum_sampled_kinetic_radius=float(kinetic_x[minimum_index]),
        global_nonminimal_kinetic_positive=counterexample_value >= 0.0,
        potential_reconstructed=False,
        perturbative_stability_derived=False,
    )
    return GlobalThroatExactCertificate(
        quadrature_order=order,
        kinetic_sample_count=sample_count,
        kinetic_radial_cutoff=cutoff,
        original=original,
        localized_phi_match=localized,
    )
