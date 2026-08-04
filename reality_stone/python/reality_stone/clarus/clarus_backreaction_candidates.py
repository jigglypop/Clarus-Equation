"""Backreaction controls for the leading Clarus negative-source candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import NEWTON_G_M3_KG_S2, SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class CasimirTensorMatchAudit:
    shape_derivative_from_radial_match: float
    geometry_rho_over_scale: float
    geometry_radial_pressure_over_scale: float
    geometry_tangential_pressure_over_scale: float
    casimir_rho_over_scale: float
    casimir_radial_pressure_over_scale: float
    casimir_tangential_pressure_over_scale: float
    residual_rho_over_scale: float
    residual_radial_pressure_over_scale: float
    residual_tangential_pressure_over_scale: float
    exact_zero_redshift_tensor_match: bool
    auxiliary_pressure_source_required: bool
    conserved_global_solution_derived: bool


@dataclass(frozen=True)
class VacuumPolarizationScaleAudit:
    throat_radius_m: float
    field_correlation_length_m: float
    large_mass_expansion_parameter: float
    estimated_energy_density_j_m3: float
    required_curvature_density_j_m3: float
    backreaction_ratio: float
    multiplicity_required: float
    large_mass_control_applicable: bool
    order_one_backreaction_reached: bool
    exact_renormalized_stress_derived: bool


@dataclass(frozen=True)
class CasimirThroatSeriesAudit:
    shape_derivative: float
    dimensionless_redshift_slope: float
    dimensionless_plate_separation_slope: float
    geometry_stress_over_scale: tuple[float, float, float]
    casimir_stress_over_scale: tuple[float, float, float]
    radial_pressure_derivative_over_scale_per_radius: float
    conservation_required_derivative_over_scale_per_radius: float
    stress_components_match: bool
    anisotropic_conservation_matches: bool
    flare_out_satisfied: bool
    finite_redshift_slope: bool
    local_throat_series_exists: bool
    global_asymptotically_regular_solution_derived: bool
    physical_boundary_realization_derived: bool
    perturbative_stability_derived: bool


def ideal_casimir_zero_redshift_match_audit() -> CasimirTensorMatchAudit:
    """Compare ideal parallel-plate stress ratios with a throat Einstein tensor.

    Quantities are normalized by ``C=c^4/(8*pi*G*r0^2)``.  At a zero-redshift
    throat, geometry requires ``(rho, p_r, p_t)=(b', -1, (1-b')/2) C``.
    Ideal electromagnetic plates with the plate normal radial have ratios
    ``(-u, -3u, +u)``.  Matching rho and radial pressure fixes ``b'=-1/3``;
    the tangential component then leaves a nonzero residual.
    """

    shape_derivative = -1.0 / 3.0
    geometry = (
        shape_derivative,
        -1.0,
        (1.0 - shape_derivative) / 2.0,
    )
    casimir_magnitude = -shape_derivative
    casimir = (-casimir_magnitude, -3.0 * casimir_magnitude, casimir_magnitude)
    residual = tuple(left - right for left, right in zip(geometry, casimir, strict=True))
    exact = all(abs(value) <= 1e-15 for value in residual)
    return CasimirTensorMatchAudit(
        shape_derivative_from_radial_match=shape_derivative,
        geometry_rho_over_scale=geometry[0],
        geometry_radial_pressure_over_scale=geometry[1],
        geometry_tangential_pressure_over_scale=geometry[2],
        casimir_rho_over_scale=casimir[0],
        casimir_radial_pressure_over_scale=casimir[1],
        casimir_tangential_pressure_over_scale=casimir[2],
        residual_rho_over_scale=residual[0],
        residual_radial_pressure_over_scale=residual[1],
        residual_tangential_pressure_over_scale=residual[2],
        exact_zero_redshift_tensor_match=exact,
        auxiliary_pressure_source_required=not exact,
        conserved_global_solution_derived=False,
    )


def vacuum_polarization_scale_audit(
    *,
    throat_radius_m: float,
    field_correlation_length_m: float,
    dimensionless_coefficient: float = 1.0,
) -> VacuumPolarizationScaleAudit:
    """Compare a massive-field curvature expansion with throat backreaction.

    The leading dimensional control is ``rho_vac ~ C hbar c xi^2 / r^6``.
    It represents the scale of a large-mass DeWitt--Schwinger term, not an exact
    state-dependent tensor.  The throat curvature density scale is
    ``c^4/(8*pi*G*r^2)``.
    """

    radius = float(throat_radius_m)
    correlation = float(field_correlation_length_m)
    coefficient = float(dimensionless_coefficient)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (radius, correlation, coefficient)
    ):
        raise ValueError("vacuum-polarization scale inputs must be finite and positive")

    estimated = coefficient * HBAR_J_S * SPEED_OF_LIGHT_M_S * correlation**2 / radius**6
    required = SPEED_OF_LIGHT_M_S**4 / (
        8.0 * math.pi * NEWTON_G_M3_KG_S2 * radius**2
    )
    ratio = estimated / required
    return VacuumPolarizationScaleAudit(
        throat_radius_m=radius,
        field_correlation_length_m=correlation,
        large_mass_expansion_parameter=radius / correlation,
        estimated_energy_density_j_m3=estimated,
        required_curvature_density_j_m3=required,
        backreaction_ratio=ratio,
        multiplicity_required=1.0 / ratio,
        large_mass_control_applicable=radius / correlation > 10.0,
        order_one_backreaction_reached=ratio >= 1.0,
        exact_renormalized_stress_derived=False,
    )


def ideal_casimir_general_redshift_throat_series() -> CasimirThroatSeriesAudit:
    """Solve the zeroth/first-order throat conditions for ideal Casimir stress.

    With ``u=r0*Phi'(r0)``, the regular throat limit is

    ``(rho,p_r,p_t)/C = (b', -1, (1-b')*(1+u)/2)``.

    Matching ideal radial-plate Casimir stress fixes ``b'=-1/3`` and ``u=-1/2``.
    Stress conservation then fixes ``r0*a'/a=+1/2`` for ``rho ~ -a^-4``.
    This is a local series result only.
    """

    shape_derivative = -1.0 / 3.0
    redshift_slope = -1.0 / 2.0
    plate_slope = 1.0 / 2.0
    geometry = (
        shape_derivative,
        -1.0,
        (1.0 - shape_derivative) * (1.0 + redshift_slope) / 2.0,
    )
    casimir = (-1.0 / 3.0, -1.0, 1.0 / 3.0)

    # rho=-u_C, p_r=-3u_C and u_C proportional to a^-4.  Values below are
    # coefficients of C/r0.  r0*a'/a=1/2 gives r0*u_C'/u_C=-2.
    casimir_magnitude_over_scale = 1.0 / 3.0
    radial_derivative = (
        -3.0 * casimir_magnitude_over_scale * (-4.0 * plate_slope)
    )
    rho, radial, tangential = casimir
    conservation_required = (
        -(rho + radial) * redshift_slope + 2.0 * (tangential - radial)
    )
    component_match = all(
        math.isclose(left, right, abs_tol=1e-15)
        for left, right in zip(geometry, casimir, strict=True)
    )
    conservation_match = math.isclose(
        radial_derivative,
        conservation_required,
        abs_tol=1e-15,
    )
    return CasimirThroatSeriesAudit(
        shape_derivative=shape_derivative,
        dimensionless_redshift_slope=redshift_slope,
        dimensionless_plate_separation_slope=plate_slope,
        geometry_stress_over_scale=geometry,
        casimir_stress_over_scale=casimir,
        radial_pressure_derivative_over_scale_per_radius=radial_derivative,
        conservation_required_derivative_over_scale_per_radius=conservation_required,
        stress_components_match=component_match,
        anisotropic_conservation_matches=conservation_match,
        flare_out_satisfied=shape_derivative < 1.0,
        finite_redshift_slope=math.isfinite(redshift_slope),
        local_throat_series_exists=(
            component_match
            and conservation_match
            and shape_derivative < 1.0
            and math.isfinite(redshift_slope)
        ),
        global_asymptotically_regular_solution_derived=False,
        physical_boundary_realization_derived=False,
        perturbative_stability_derived=False,
    )
