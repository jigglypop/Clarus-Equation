"""Physical-scale gates for the inverse-designed multi-mode throat target."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import (
    ELECTRON_VOLT_J,
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
    two_sided_negative_energy_magnitude_j: float
    two_sided_mass_equivalent_kg: float
    two_sided_mass_equivalent_earths: float
    resolved_radius_mass_equivalent_solar: float
    light_crossing_time_s: float
    flat_scalar_qi_duration_control_s: float
    crossing_to_qi_duration_ratio: float
    multimode_superposition_reduces_geometric_total: bool
    physical_reflector_model_derived: bool
    renormalized_multimode_stress_derived: bool
    current_physical_realization_pass: bool


def physical_multimode_realization_audit(
    *,
    throat_radius_m: float = 1.0,
    boundary_resolution_m: float = PROTON_CHARGE_RADIUS_M,
) -> PhysicalMultimodeRealizationAudit:
    """Audit physical scales of the exact variable-anisotropy target.

    At the throat, the target has ``rho=-C/3`` and ``rho+p_r=-4C/3`` with
    ``C=c^4/(8*pi*G*r0^2)``.  Matching the ideal electromagnetic Casimir
    density determines the plate separation.  The integrated density follows
    exactly from ``b(infinity)-b(r0)=-r0/3`` and therefore cannot be reduced by
    splitting the same target among more linear modes.

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
    separation = (
        math.pi**2 * HBAR_J_S * c / (720.0 * density)
    ) ** 0.25
    wavelength = 2.0 * separation
    carrier_energy = 2.0 * math.pi * HBAR_J_S * c / wavelength / ELECTRON_VOLT_J

    # Since a(r0) is proportional to sqrt(r0), invert the one-point result.
    resolved_radius = radius * (boundary / separation) ** 2

    # One side: |E|=c^4*r0/(6G).  A symmetric two-sided extension doubles it.
    two_sided_energy = c**4 * radius / (3.0 * NEWTON_G_M3_KG_S2)
    mass_equivalent = two_sided_energy / c**2
    resolved_mass = c**2 * resolved_radius / (3.0 * NEWTON_G_M3_KG_S2)

    # Fewster--Eveson/Ford--Roman style Lorentzian-sampling scale control:
    # rho >= -3*hbar/(32*pi^2*c^3*tau^4) for a massless scalar in flat space.
    qi_duration = (
        3.0 * HBAR_J_S / (32.0 * math.pi**2 * c**3 * density)
    ) ** 0.25
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
        two_sided_negative_energy_magnitude_j=two_sided_energy,
        two_sided_mass_equivalent_kg=mass_equivalent,
        two_sided_mass_equivalent_earths=mass_equivalent / EARTH_MASS_KG,
        resolved_radius_mass_equivalent_solar=resolved_mass / SOLAR_MASS_KG,
        light_crossing_time_s=crossing_time,
        flat_scalar_qi_duration_control_s=qi_duration,
        crossing_to_qi_duration_ratio=crossing_time / qi_duration,
        multimode_superposition_reduces_geometric_total=False,
        physical_reflector_model_derived=False,
        renormalized_multimode_stress_derived=False,
        current_physical_realization_pass=False,
    )
