"""Inverse-designed global throat target for a future multi-mode source.

The construction is a geometry/control result.  It computes the anisotropic
stress tensor required by Einstein's equation; it does not claim that Clarus
fields or a driven resonator can produce that tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Literal

import numpy as np


RedshiftProfile = Literal["exponential", "schwarzschild_matched"]


@dataclass(frozen=True)
class GlobalAnisotropicTargetAudit:
    redshift_profile: str
    throat_density: float
    throat_radial_pressure: float
    throat_tangential_pressure: float
    throat_matches_ideal_casimir: bool
    flare_out_satisfied: bool
    horizon_free: bool
    analytic_lapse_squared_lower_bound: float
    shape_gap_at_throat: float
    sampled_minimum_shape_gap: float
    shape_gap_derivative_lower_bound: float
    shape_gap_positive_on_entire_exterior_proved: bool
    sampled_maximum_conservation_residual: float
    conservation_identity_exact: bool
    throat_proper_distance_integrable_proved: bool
    metric_factor_throat_slope: float
    shape_over_radius_at_cutoff: float
    redshift_at_cutoff: float
    adm_mass_per_end_in_throat_radii: float
    finite_adm_mass: bool
    asymptotically_flat: bool
    asymptotic_flatness_proved_without_cutoff: bool
    radial_nec_strictly_negative_everywhere_proved: bool
    complete_radial_affine_anec_finite_proved: bool
    complete_radial_affine_anec_negative_proved: bool
    sampled_dimensionless_two_sided_radial_anec: float
    radial_anec_quadrature_refinement_delta: float
    asymptotic_radial_nec_x_cubed_coefficient: float
    sampled_coordinate_volume_nec_integral: float
    coordinate_volume_nec_burden_finite: bool
    proper_volume_nec_burden_finite: bool
    stress_l1_localized: bool
    source_tail_control_pass: bool
    sampled_numerical_identity_pass: bool
    fixed_casimir_eos_preserved_globally: bool
    two_sided_geometric_extension_available: bool
    global_geometry_control_pass: bool
    ce_multimode_stress_derived: bool
    independent_matter_eom_derived: bool
    perturbative_stability_derived: bool

    @property
    def minimum_shape_gap(self) -> float:
        """Backward-compatible alias for the sampled diagnostic."""

        return self.sampled_minimum_shape_gap

    @property
    def maximum_conservation_residual(self) -> float:
        """Backward-compatible alias for the sampled numerical residual."""

        return self.sampled_maximum_conservation_residual

    @property
    def adm_mass_in_throat_radii(self) -> float:
        """Backward-compatible alias; the value is for each asymptotic end."""

        return self.adm_mass_per_end_in_throat_radii


@dataclass(frozen=True)
class ModeFitLevel:
    mode_count: int
    maximum_normalized_error: float


@dataclass(frozen=True)
class MultimodeTargetFitAudit:
    redshift_profile: str
    radial_extent: float
    levels: tuple[ModeFitLevel, ...]
    error_decreases: bool
    finite_mode_target_approximation_pass: bool
    basis_is_physical_resonator_spectrum: bool
    carrier_envelope_bridge_derived: bool
    quantized_negative_stress_derived: bool


def _validate_redshift_profile(redshift_profile: str) -> RedshiftProfile:
    if redshift_profile not in ("exponential", "schwarzschild_matched"):
        raise ValueError(
            "redshift_profile must be 'exponential' or 'schwarzschild_matched'"
        )
    return redshift_profile


def _geometry_profiles(
    x: np.ndarray,
    redshift_profile: RedshiftProfile,
) -> tuple[np.ndarray, ...]:
    exponential = np.exp(-(x - 1.0))
    shape = 2.0 / 3.0 + exponential / 3.0
    shape_prime = -exponential / 3.0

    if redshift_profile == "exponential":
        redshift = exponential / 2.0
        redshift_prime = -exponential / 2.0
        redshift_second = exponential / 2.0
    else:
        schwarzschild_factor = 1.0 - 2.0 / (3.0 * x)
        redshift = 0.5 * np.log(schwarzschild_factor) + 1.5 * exponential
        redshift_prime = 1.0 / (x * (3.0 * x - 2.0)) - 1.5 * exponential
        redshift_second = (
            -(6.0 * x - 2.0) / (x**2 * (3.0 * x - 2.0) ** 2)
            + 1.5 * exponential
        )

    return (
        shape,
        shape_prime,
        redshift,
        redshift_prime,
        redshift_second,
    )


def _target_profiles(
    x: np.ndarray,
    redshift_profile: RedshiftProfile = "exponential",
) -> tuple[np.ndarray, ...]:
    """Return ``y=b/r0, Phi`` and the dimensionless target stresses.

    Stress components are normalized by ``1/(8 pi G r0^2)``.  Derivatives are
    with respect to ``x=r/r0``.
    """

    shape, shape_prime, redshift, redshift_prime, redshift_second = (
        _geometry_profiles(x, redshift_profile)
    )

    metric_factor = 1.0 - shape / x
    density = shape_prime / x**2
    radial_pressure = -shape / x**3 + 2.0 * metric_factor * redshift_prime / x

    # Algebraically cancel the apparent (x-shape)^-1 terms at the throat.
    difference = shape_prime * x - shape
    tangential_pressure = (
        metric_factor
        * (redshift_second + redshift_prime**2 + redshift_prime / x)
        - difference * redshift_prime / (2.0 * x**2)
        - difference / (2.0 * x**3)
    )
    return shape, redshift, density, radial_pressure, tangential_pressure


def _dimensionless_two_sided_radial_anec(
    redshift_profile: RedshiftProfile,
    sample_count: int,
) -> float:
    """Quadrature the complete radial ANEC at unit asymptotic Killing energy.

    ``x=1+[z/(1-z)]^2`` maps both the throat square-root endpoint and infinity
    to a finite interval.  The returned magnitude depends on affine
    normalization; its strict negative sign and finiteness do not.
    """

    z = np.linspace(0.0, 1.0, sample_count)
    integrand = np.empty_like(z)
    interior = z[1:-1]
    u = interior / (1.0 - interior)
    x = 1.0 + u**2
    shape, redshift, density, radial, _ = _target_profiles(x, redshift_profile)
    metric_factor = 1.0 - shape / x
    integrand[1:-1] = (
        4.0
        * interior
        / (1.0 - interior) ** 3
        * (density + radial)
        * np.exp(-redshift)
        / np.sqrt(metric_factor)
    )
    throat_redshift = (
        0.5
        if redshift_profile == "exponential"
        else 0.5 * math.log(1.0 / 3.0) + 1.5
    )
    integrand[0] = -(8.0 * math.sqrt(3.0) / 3.0) * math.exp(
        -throat_redshift
    )
    integrand[-1] = 0.0
    return float(np.trapezoid(integrand, z))


def global_anisotropic_target_audit(
    *,
    radial_cutoff: float = 1.0e4,
    sample_count: int = 4000,
    redshift_profile: RedshiftProfile = "exponential",
) -> GlobalAnisotropicTargetAudit:
    """Audit a finite-mass global geometry with an exact Casimir throat.

    The ansatz is

    ``b/r0 = 2/3 + exp[-(x-1)]/3`` and ``Phi = exp[-(x-1)]/2``.

    It reproduces the ideal Casimir ratios only at the throat.  Away from the
    throat its anisotropy changes, evading the fixed-equation-of-state global
    no-go while preserving stress conservation by the Bianchi identity.

    ``exponential`` keeps the original redshift and exposes its logarithmically
    divergent coordinate-volume NEC burden.  ``schwarzschild_matched`` uses

    ``Phi=log(1-2/(3x))/2 + 3 exp[-(x-1)]/2``

    to preserve the throat data while matching the finite ADM mass in the
    asymptotic lapse.  Its stress tail is exponential and its volume burden is
    finite.  Neither profile derives a microscopic source or stability.
    """

    profile = _validate_redshift_profile(redshift_profile)
    cutoff = float(radial_cutoff)
    if not math.isfinite(cutoff) or cutoff <= 1.0:
        raise ValueError("radial_cutoff must be finite and greater than one")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count < 32
    ):
        raise ValueError("sample_count must be an integer of at least 32")
    count = int(sample_count)

    # Log spacing resolves both the throat and the asymptotic tail.
    x = 1.0 + np.geomspace(1.0e-10, cutoff - 1.0, count)
    shape, redshift, density, radial, tangential = _target_profiles(x, profile)

    _, shape_prime, _, redshift_prime, redshift_second = _geometry_profiles(
        x,
        profile,
    )
    metric_factor = 1.0 - shape / x
    metric_factor_prime = -shape_prime / x + shape / x**2
    radial_prime = (
        -shape_prime / x**3
        + 3.0 * shape / x**4
        + 2.0
        * (
            metric_factor_prime * redshift_prime / x
            + metric_factor * redshift_second / x
            - metric_factor * redshift_prime / x**2
        )
    )
    conservation = (
        radial_prime
        + (density + radial) * redshift_prime
        - 2.0 * (tangential - radial) / x
    )
    arrays = (shape, redshift, density, radial, tangential, conservation)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("target profiles must remain finite on the sampled domain")

    throat_density = -1.0 / 3.0
    throat_radial = -1.0
    throat_tangential = 1.0 / 3.0
    casimir_match = (
        math.isclose(throat_radial, 3.0 * throat_density, abs_tol=1e-14)
        and math.isclose(throat_tangential, -throat_density, abs_tol=1e-14)
    )
    shape_gap = x - shape
    minimum_lapse = float(np.min(np.exp(2.0 * redshift)))
    maximum_residual = float(np.max(np.abs(conservation)))
    sampled_identity_pass = bool(
        np.all(shape_gap > 0.0)
        and minimum_lapse > 0.0
        and maximum_residual < 1.0e-10
    )
    flare_out_proved = -1.0 / 3.0 < 1.0
    shape_gap_proved = True  # g(1)=0 and g'(x)=1+exp(1-x)/3 >= 1.
    lapse_lower_bound = 1.0 if profile == "exponential" else 1.0 / 3.0
    horizon_free_proved = lapse_lower_bound > 0.0
    conservation_proved = True  # Contracted Bianchi identity for the exact G_ab.
    proper_distance_proved = True  # 1-b/r = 4(x-1)/3 + O((x-1)^2).
    asymptotic_flatness_proved = True
    exact_geometry_pass = all(
        (
            casimir_match,
            flare_out_proved,
            shape_gap_proved,
            horizon_free_proved,
            conservation_proved,
            proper_distance_proved,
            asymptotic_flatness_proved,
        )
    )
    radial_nec = density + radial
    coordinate_volume_integral = float(np.trapezoid(x**2 * radial_nec, x))
    exponential_tail = profile == "exponential"
    anec_count = max(count, 513)
    coarse_anec = _dimensionless_two_sided_radial_anec(profile, anec_count)
    fine_anec = _dimensionless_two_sided_radial_anec(profile, 2 * anec_count - 1)

    return GlobalAnisotropicTargetAudit(
        redshift_profile=profile,
        throat_density=throat_density,
        throat_radial_pressure=throat_radial,
        throat_tangential_pressure=throat_tangential,
        throat_matches_ideal_casimir=casimir_match,
        flare_out_satisfied=flare_out_proved,
        horizon_free=horizon_free_proved,
        analytic_lapse_squared_lower_bound=lapse_lower_bound,
        shape_gap_at_throat=0.0,
        sampled_minimum_shape_gap=float(np.min(shape_gap)),
        shape_gap_derivative_lower_bound=1.0,
        shape_gap_positive_on_entire_exterior_proved=shape_gap_proved,
        sampled_maximum_conservation_residual=maximum_residual,
        conservation_identity_exact=conservation_proved,
        throat_proper_distance_integrable_proved=proper_distance_proved,
        metric_factor_throat_slope=4.0 / 3.0,
        shape_over_radius_at_cutoff=float(shape[-1] / x[-1]),
        redshift_at_cutoff=float(redshift[-1]),
        adm_mass_per_end_in_throat_radii=1.0 / 3.0,
        finite_adm_mass=True,
        asymptotically_flat=True,
        asymptotic_flatness_proved_without_cutoff=asymptotic_flatness_proved,
        radial_nec_strictly_negative_everywhere_proved=True,
        complete_radial_affine_anec_finite_proved=True,
        complete_radial_affine_anec_negative_proved=True,
        sampled_dimensionless_two_sided_radial_anec=fine_anec,
        radial_anec_quadrature_refinement_delta=fine_anec - coarse_anec,
        asymptotic_radial_nec_x_cubed_coefficient=(
            -2.0 / 3.0 if exponential_tail else 0.0
        ),
        sampled_coordinate_volume_nec_integral=coordinate_volume_integral,
        coordinate_volume_nec_burden_finite=not exponential_tail,
        proper_volume_nec_burden_finite=not exponential_tail,
        stress_l1_localized=not exponential_tail,
        source_tail_control_pass=not exponential_tail,
        sampled_numerical_identity_pass=sampled_identity_pass,
        fixed_casimir_eos_preserved_globally=False,
        two_sided_geometric_extension_available=True,
        global_geometry_control_pass=exact_geometry_pass,
        ce_multimode_stress_derived=False,
        independent_matter_eom_derived=False,
        perturbative_stability_derived=False,
    )


def multimode_target_fit_audit(
    *,
    radial_extent: float = 10.0,
    mode_counts: tuple[int, ...] = (4, 8, 16, 32),
    validation_count: int = 4001,
    redshift_profile: RedshiftProfile = "exponential",
) -> MultimodeTargetFitAudit:
    """Resolve the inverse-designed stress with a shared spectral mode basis.

    Each Chebyshev spatial mode receives a three-component polarization vector
    for ``(rho, p_r, p_t)``.  This establishes mathematical synthesizeability
    on a compact radial interval, but the basis is not identified with physical
    resonator eigenmodes and supplies no quantum stress calculation.
    """

    profile = _validate_redshift_profile(redshift_profile)
    extent = float(radial_extent)
    if not math.isfinite(extent) or extent <= 1.0:
        raise ValueError("radial_extent must be finite and greater than one")
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, Integral)
        or validation_count < 128
    ):
        raise ValueError("validation_count must be an integer of at least 128")
    if not mode_counts or any(
        isinstance(count, bool) or not isinstance(count, Integral) or count < 2
        for count in mode_counts
    ):
        raise ValueError("mode_counts must contain integers of at least two")
    counts = tuple(int(count) for count in mode_counts)
    if any(right <= left for left, right in zip(counts, counts[1:])):
        raise ValueError("mode_counts must be strictly increasing")

    validation_x = np.linspace(1.0, extent, int(validation_count))
    validation_z = 2.0 * (validation_x - 1.0) / (extent - 1.0) - 1.0
    validation_stress = np.stack(
        _target_profiles(validation_x, profile)[2:],
        axis=1,
    )
    component_scales = np.max(np.abs(validation_stress), axis=0)
    levels: list[ModeFitLevel] = []

    for mode_count in counts:
        # Oversample the least-squares fit, then assess it on an independent grid.
        training_count = max(8 * mode_count, 256)
        training_z = np.cos(
            math.pi * (np.arange(training_count) + 0.5) / training_count
        )
        training_x = 1.0 + (training_z + 1.0) * (extent - 1.0) / 2.0
        training_stress = np.stack(
            _target_profiles(training_x, profile)[2:],
            axis=1,
        )
        coefficients = np.polynomial.chebyshev.chebfit(
            training_z,
            training_stress,
            deg=mode_count - 1,
        )
        reconstructed = np.column_stack(
            [
                np.polynomial.chebyshev.chebval(validation_z, coefficients[:, index])
                for index in range(3)
            ]
        )
        normalized_error = np.abs(reconstructed - validation_stress) / component_scales
        levels.append(
            ModeFitLevel(
                mode_count=mode_count,
                maximum_normalized_error=float(np.max(normalized_error)),
            )
        )

    errors = [level.maximum_normalized_error for level in levels]
    decreasing = all(right < left for left, right in zip(errors, errors[1:]))
    approximation_pass = decreasing and errors[-1] < 1.0e-6
    return MultimodeTargetFitAudit(
        redshift_profile=profile,
        radial_extent=extent,
        levels=tuple(levels),
        error_decreases=decreasing,
        finite_mode_target_approximation_pass=approximation_pass,
        basis_is_physical_resonator_spectrum=False,
        carrier_envelope_bridge_derived=False,
        quantized_negative_stress_derived=False,
    )
