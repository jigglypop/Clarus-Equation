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
