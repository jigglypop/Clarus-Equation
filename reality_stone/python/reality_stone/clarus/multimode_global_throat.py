"""Inverse-designed global throat target for a future multi-mode source.

The construction is a geometry/control result.  It computes the anisotropic
stress tensor required by Einstein's equation; it does not claim that Clarus
fields or a driven resonator can produce that tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GlobalAnisotropicTargetAudit:
    throat_density: float
    throat_radial_pressure: float
    throat_tangential_pressure: float
    throat_matches_ideal_casimir: bool
    flare_out_satisfied: bool
    horizon_free: bool
    minimum_shape_gap: float
    maximum_conservation_residual: float
    shape_over_radius_at_cutoff: float
    redshift_at_cutoff: float
    adm_mass_in_throat_radii: float
    finite_adm_mass: bool
    asymptotically_flat: bool
    fixed_casimir_eos_preserved_globally: bool
    two_sided_geometric_extension_available: bool
    global_geometry_control_pass: bool
    ce_multimode_stress_derived: bool
    perturbative_stability_derived: bool


@dataclass(frozen=True)
class ModeFitLevel:
    mode_count: int
    maximum_normalized_error: float


@dataclass(frozen=True)
class MultimodeTargetFitAudit:
    radial_extent: float
    levels: tuple[ModeFitLevel, ...]
    error_decreases: bool
    finite_mode_target_approximation_pass: bool
    basis_is_physical_resonator_spectrum: bool
    carrier_envelope_bridge_derived: bool
    quantized_negative_stress_derived: bool


def _target_profiles(x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ``y=b/r0, Phi`` and the dimensionless target stresses.

    Stress components are normalized by ``1/(8 pi G r0^2)``.  Derivatives are
    with respect to ``x=r/r0``.
    """

    exponential = np.exp(-(x - 1.0))
    shape = 2.0 / 3.0 + exponential / 3.0
    shape_prime = -exponential / 3.0
    redshift = exponential / 2.0
    redshift_prime = -exponential / 2.0
    redshift_second = exponential / 2.0

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


def global_anisotropic_target_audit(
    *,
    radial_cutoff: float = 1.0e4,
    sample_count: int = 4000,
) -> GlobalAnisotropicTargetAudit:
    """Audit a finite-mass global geometry with an exact Casimir throat.

    The ansatz is

    ``b/r0 = 2/3 + exp[-(x-1)]/3`` and ``Phi = exp[-(x-1)]/2``.

    It reproduces the ideal Casimir ratios only at the throat.  Away from the
    throat its anisotropy changes, evading the fixed-equation-of-state global
    no-go while preserving stress conservation by the Bianchi identity.
    """

    cutoff = float(radial_cutoff)
    if not math.isfinite(cutoff) or cutoff <= 1.0:
        raise ValueError("radial_cutoff must be finite and greater than one")
    if sample_count < 32:
        raise ValueError("sample_count must be at least 32")

    # Log spacing resolves both the throat and the asymptotic tail.
    x = 1.0 + np.geomspace(1.0e-10, cutoff - 1.0, sample_count)
    shape, redshift, density, radial, tangential = _target_profiles(x)

    exponential = np.exp(-(x - 1.0))
    shape_prime = -exponential / 3.0
    redshift_prime = -exponential / 2.0
    redshift_second = exponential / 2.0
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
    asymptotically_flat = bool(
        shape[-1] / x[-1] < 1.0e-3 and abs(redshift[-1]) < 1.0e-10
    )
    geometry_pass = bool(
        casimir_match
        and np.all(shape_gap > 0.0)
        and minimum_lapse > 0.0
        and maximum_residual < 1.0e-10
        and asymptotically_flat
    )

    return GlobalAnisotropicTargetAudit(
        throat_density=throat_density,
        throat_radial_pressure=throat_radial,
        throat_tangential_pressure=throat_tangential,
        throat_matches_ideal_casimir=casimir_match,
        flare_out_satisfied=True,  # b'(r0)=-1/3 < 1
        horizon_free=minimum_lapse > 0.0,
        minimum_shape_gap=float(np.min(shape_gap)),
        maximum_conservation_residual=maximum_residual,
        shape_over_radius_at_cutoff=float(shape[-1] / x[-1]),
        redshift_at_cutoff=float(redshift[-1]),
        adm_mass_in_throat_radii=1.0 / 3.0,  # b(infinity)/(2 r0)
        finite_adm_mass=True,
        asymptotically_flat=asymptotically_flat,
        fixed_casimir_eos_preserved_globally=False,
        two_sided_geometric_extension_available=True,
        global_geometry_control_pass=geometry_pass,
        ce_multimode_stress_derived=False,
        perturbative_stability_derived=False,
    )


def multimode_target_fit_audit(
    *,
    radial_extent: float = 10.0,
    mode_counts: tuple[int, ...] = (4, 8, 16, 32),
    validation_count: int = 4001,
) -> MultimodeTargetFitAudit:
    """Resolve the inverse-designed stress with a shared spectral mode basis.

    Each Chebyshev spatial mode receives a three-component polarization vector
    for ``(rho, p_r, p_t)``.  This establishes mathematical synthesizeability
    on a compact radial interval, but the basis is not identified with physical
    resonator eigenmodes and supplies no quantum stress calculation.
    """

    extent = float(radial_extent)
    if not math.isfinite(extent) or extent <= 1.0:
        raise ValueError("radial_extent must be finite and greater than one")
    if validation_count < 128:
        raise ValueError("validation_count must be at least 128")
    if not mode_counts or any(count < 2 for count in mode_counts):
        raise ValueError("mode_counts must contain integers of at least two")
    if any(right <= left for left, right in zip(mode_counts, mode_counts[1:])):
        raise ValueError("mode_counts must be strictly increasing")

    validation_x = np.linspace(1.0, extent, validation_count)
    validation_z = 2.0 * (validation_x - 1.0) / (extent - 1.0) - 1.0
    validation_stress = np.stack(_target_profiles(validation_x)[2:], axis=1)
    component_scales = np.max(np.abs(validation_stress), axis=0)
    levels: list[ModeFitLevel] = []

    for mode_count in mode_counts:
        # Oversample the least-squares fit, then assess it on an independent grid.
        training_count = max(8 * mode_count, 256)
        training_z = np.cos(
            math.pi * (np.arange(training_count) + 0.5) / training_count
        )
        training_x = 1.0 + (training_z + 1.0) * (extent - 1.0) / 2.0
        training_stress = np.stack(_target_profiles(training_x)[2:], axis=1)
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
        radial_extent=extent,
        levels=tuple(levels),
        error_decreases=decreasing,
        finite_mode_target_approximation_pass=approximation_pass,
        basis_is_physical_resonator_spectrum=False,
        carrier_envelope_bridge_derived=False,
        quantized_negative_stress_derived=False,
    )
