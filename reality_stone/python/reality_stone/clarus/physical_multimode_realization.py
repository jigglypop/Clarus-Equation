"""Physical-scale gates for the inverse-designed multi-mode throat target."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .casimir_carrier_target import exact_casimir_carrier_target
from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import (
    NEWTON_G_M3_KG_S2,
    SPEED_OF_LIGHT_M_S,
)


EARTH_MASS_KG = 5.9722e24
SOLAR_MASS_KG = 1.98847e30
PROTON_CHARGE_RADIUS_M = 0.84e-15


@dataclass(frozen=True)
class PhysicalMultimodeRealizationAudit:
    throat_radius_m: float
    required_density_magnitude_j_m3: float
    required_null_magnitude_j_m3: float
    ideal_casimir_separation_m: float
    fundamental_wavelength_m: float
    fundamental_energy_ev: float
    separation_to_boundary_resolution_ratio: float
    boundary_resolved_at_target_radius: bool
    radius_for_boundary_resolution_m: float
    two_sided_coordinate_density_integral_magnitude_j: float
    two_sided_coordinate_mass_equivalent_kg: float
    two_sided_coordinate_mass_equivalent_earths: float
    proper_volume_integral_dimensionless: float
    proper_volume_tail_bound_dimensionless: float
    proper_volume_quadrature_delta_dimensionless: float
    proper_to_coordinate_energy_ratio: float
    two_sided_proper_matter_energy_magnitude_j: float
    two_sided_proper_mass_equivalent_kg: float
    two_sided_proper_mass_equivalent_earths: float
    resolved_radius_coordinate_mass_equivalent_solar: float
    resolved_radius_proper_mass_equivalent_solar: float
    light_crossing_time_s: float
    flat_scalar_qi_duration_control_s: float
    crossing_to_qi_duration_ratio: float
    multimode_superposition_reduces_geometric_total: bool
    physical_reflector_model_derived: bool
    renormalized_multimode_stress_derived: bool
    current_physical_realization_pass: bool


_PROPER_VOLUME_S_MAX = 8.0


def _proper_volume_integrand(s: float) -> float:
    """Return the regularized proper-volume integrand after ``x=1+s^2``.

    For ``y(x)=2/3+exp(-(x-1))/3``, the original integrand has an
    integrable square-root singularity at the throat.  The substitution
    removes it, and its analytic limit at ``s=0`` is ``1/sqrt(3)``.
    ``expm1`` keeps ``1-exp(-s^2)`` accurate near the throat.
    """

    if s == 0.0:
        return 1.0 / math.sqrt(3.0)
    s_squared = s * s
    exponential = math.exp(-s_squared)
    one_minus_exponential = -math.expm1(-s_squared)
    one_minus_b_over_r = (s_squared + one_minus_exponential / 3.0) / (1.0 + s_squared)
    return 2.0 * s * exponential / 3.0 / math.sqrt(one_minus_b_over_r)


def _composite_simpson_proper_volume(steps: int) -> float:
    spacing = _PROPER_VOLUME_S_MAX / steps
    odd_sum = math.fsum(_proper_volume_integrand(index * spacing) for index in range(1, steps, 2))
    even_sum = math.fsum(_proper_volume_integrand(index * spacing) for index in range(2, steps, 2))
    weighted_sum = (
        _proper_volume_integrand(0.0)
        + _proper_volume_integrand(_PROPER_VOLUME_S_MAX)
        + 4.0 * odd_sum
        + 2.0 * even_sum
    )
    return spacing * weighted_sum / 3.0


def _proper_volume_integral(
    steps: int,
) -> tuple[float, float, float]:
    """Numerically evaluate the dimensionless proper matter-energy integral."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1_000 or steps % 4 != 0:
        raise ValueError(
            "proper-volume integration steps must be an integer multiple of four and at least 1000"
        )

    integral = _composite_simpson_proper_volume(steps)
    coarse_integral = _composite_simpson_proper_volume(steps // 2)

    s_squared = _PROPER_VOLUME_S_MAX**2
    y_at_cutoff = 2.0 / 3.0 + math.exp(-s_squared) / 3.0
    gap_at_cutoff = 1.0 - y_at_cutoff / (1.0 + s_squared)
    tail_bound = math.exp(-s_squared) / (3.0 * math.sqrt(gap_at_cutoff))
    return integral, tail_bound, abs(integral - coarse_integral)


def physical_multimode_realization_audit(
    *,
    throat_radius_m: float = 1.0,
    boundary_resolution_m: float = PROTON_CHARGE_RADIUS_M,
    proper_volume_integration_steps: int = 4_000,
) -> PhysicalMultimodeRealizationAudit:
    """Audit physical scales of the exact variable-anisotropy target.

    At the throat, the target has ``rho=-C/3`` and ``rho+p_r=-4C/3`` with
    ``C=c^4/(8*pi*G*r0^2)``.  Matching the ideal electromagnetic Casimir
    density determines the plate separation.  The areal-coordinate density
    integral follows exactly from ``b(infinity)-b(r0)=-r0/3``.  It is not the
    proper matter energy: the latter includes the spatial-slice factor
    ``1/sqrt(1-b/r)`` and is evaluated with a throat-regularizing substitution.
    Neither quantity can be reduced by splitting the same target among more
    linear modes.

    The duration estimate is only a flat-space massless-scalar quantum-
    inequality control.  It applies to a pulsed/squeezed route, not directly to
    a static Casimir vacuum with boundaries.
    """

    radius = float(throat_radius_m)
    boundary = float(boundary_resolution_m)
    if not all(math.isfinite(value) and value > 0.0 for value in (radius, boundary)):
        raise ValueError("radius and boundary resolution must be finite and positive")

    c = SPEED_OF_LIGHT_M_S
    gravitational_scale = c**4 / (8.0 * math.pi * NEWTON_G_M3_KG_S2 * radius**2)
    density = gravitational_scale / 3.0
    null_magnitude = 4.0 * gravitational_scale / 3.0
    carrier_target = exact_casimir_carrier_target(throat_radius_m=radius)
    separation = carrier_target.separation_m
    wavelength = carrier_target.wavelength_m
    carrier_energy = carrier_target.carrier_energy_ev

    # Since a(r0) is proportional to sqrt(r0), invert the one-point result.
    resolved_radius = radius * (boundary / separation) ** 2

    # Coordinate measure, one side: |integral rho*4*pi*r^2 dr|=c^4*r0/(6G).
    # A symmetric two-sided extension doubles it.  This exact shape-function
    # identity is distinct from integrating rho over the proper spatial volume.
    coordinate_energy = c**4 * radius / (3.0 * NEWTON_G_M3_KG_S2)
    coordinate_mass = coordinate_energy / c**2
    proper_integral, proper_tail_bound, proper_quadrature_delta = _proper_volume_integral(
        proper_volume_integration_steps
    )
    proper_energy = c**4 * radius * proper_integral / NEWTON_G_M3_KG_S2
    proper_mass = proper_energy / c**2
    resolved_coordinate_mass = c**2 * resolved_radius / (3.0 * NEWTON_G_M3_KG_S2)
    resolved_proper_mass = c**2 * resolved_radius * proper_integral / NEWTON_G_M3_KG_S2

    # Fewster--Eveson/Ford--Roman style Lorentzian-sampling scale control:
    # rho >= -3*hbar/(32*pi^2*c^3*tau^4) for a massless scalar in flat space.
    qi_duration = (3.0 * HBAR_J_S / (32.0 * math.pi**2 * c**3 * density)) ** 0.25
    crossing_time = radius / c

    boundary_resolved = separation >= boundary
    return PhysicalMultimodeRealizationAudit(
        throat_radius_m=radius,
        required_density_magnitude_j_m3=density,
        required_null_magnitude_j_m3=null_magnitude,
        ideal_casimir_separation_m=separation,
        fundamental_wavelength_m=wavelength,
        fundamental_energy_ev=carrier_energy,
        separation_to_boundary_resolution_ratio=separation / boundary,
        boundary_resolved_at_target_radius=boundary_resolved,
        radius_for_boundary_resolution_m=resolved_radius,
        two_sided_coordinate_density_integral_magnitude_j=coordinate_energy,
        two_sided_coordinate_mass_equivalent_kg=coordinate_mass,
        two_sided_coordinate_mass_equivalent_earths=(coordinate_mass / EARTH_MASS_KG),
        proper_volume_integral_dimensionless=proper_integral,
        proper_volume_tail_bound_dimensionless=proper_tail_bound,
        proper_volume_quadrature_delta_dimensionless=proper_quadrature_delta,
        proper_to_coordinate_energy_ratio=3.0 * proper_integral,
        two_sided_proper_matter_energy_magnitude_j=proper_energy,
        two_sided_proper_mass_equivalent_kg=proper_mass,
        two_sided_proper_mass_equivalent_earths=proper_mass / EARTH_MASS_KG,
        resolved_radius_coordinate_mass_equivalent_solar=(resolved_coordinate_mass / SOLAR_MASS_KG),
        resolved_radius_proper_mass_equivalent_solar=(resolved_proper_mass / SOLAR_MASS_KG),
        light_crossing_time_s=crossing_time,
        flat_scalar_qi_duration_control_s=qi_duration,
        crossing_to_qi_duration_ratio=crossing_time / qi_duration,
        multimode_superposition_reduces_geometric_total=False,
        physical_reflector_model_derived=False,
        renormalized_multimode_stress_derived=False,
        current_physical_realization_pass=False,
    )
