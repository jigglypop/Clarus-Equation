"""Geometric and stress-energy gates for a spatial-shortcut wormhole.

The module uses the zero-redshift Morris-Thorne throat equations as a control
model.  It establishes the kinematics of a locally subluminal shortcut and the
null-energy-condition (NEC) burden at the throat.  It does not claim that CE
provides the required renormalized stress tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


SPEED_OF_LIGHT_M_S = 299_792_458.0
NEWTON_G_M3_KG_S2 = 6.67430e-11
ELECTRON_VOLT_J = 1.602176634e-19
FM_M = 1e-15


@dataclass(frozen=True)
class SpatialShortcutAudit:
    exterior_distance_m: float
    throat_length_m: float
    local_speed_fraction_c: float
    exterior_light_time_s: float
    throat_traversal_time_s: float
    traveler_proper_time_s: float
    apparent_speed_over_c: float
    locally_subluminal: bool
    positive_proper_time: bool
    beats_exterior_light: bool
    exactly_instantaneous: bool


@dataclass(frozen=True)
class CasimirConversionAudit:
    energy_density_j_m3: float
    mass_equivalent_density_kg_m3: float
    reported_mass_density_kg_m3: float
    reported_to_correct_ratio: float


@dataclass(frozen=True)
class WormholeThroatAudit:
    throat_radius_m: float
    shape_derivative: float
    flare_out_satisfied: bool
    nec_energy_density_j_m3: float
    nec_violated: bool
    candidate_negative_density_j_m3: float
    local_density_gap: float
    candidate_meets_local_density: bool
    ce_correlation_length_m: float
    coherence_q_required: float
    candidate_coherent_over_throat: bool
    ce_stress_tensor_bridge_derived: bool
    physical_wormhole_established: bool


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def spatial_shortcut_audit(
    exterior_distance_m: float,
    throat_length_m: float,
    local_speed_fraction_c: float,
) -> SpatialShortcutAudit:
    """Audit a finite-length shortcut connecting distant exterior mouths."""

    exterior = _positive_finite(exterior_distance_m, "exterior_distance_m")
    throat = _positive_finite(throat_length_m, "throat_length_m")
    beta = float(local_speed_fraction_c)
    if not math.isfinite(beta) or not 0.0 < beta < 1.0:
        raise ValueError("local_speed_fraction_c must lie strictly between zero and one")

    external_light_time = exterior / SPEED_OF_LIGHT_M_S
    traversal_time = throat / (beta * SPEED_OF_LIGHT_M_S)
    proper_time = traversal_time * math.sqrt(1.0 - beta * beta)
    apparent_speed_over_c = exterior / traversal_time / SPEED_OF_LIGHT_M_S

    return SpatialShortcutAudit(
        exterior_distance_m=exterior,
        throat_length_m=throat,
        local_speed_fraction_c=beta,
        exterior_light_time_s=external_light_time,
        throat_traversal_time_s=traversal_time,
        traveler_proper_time_s=proper_time,
        apparent_speed_over_c=apparent_speed_over_c,
        locally_subluminal=beta < 1.0,
        positive_proper_time=proper_time > 0.0,
        beats_exterior_light=traversal_time < external_light_time,
        exactly_instantaneous=traversal_time == 0.0,
    )


def casimir_cell_conversion_audit(
    *,
    energy_kev: float = 0.62,
    volume_fm3: float = 294.0,
    reported_mass_density_kg_m3: float = 3.4e29,
) -> CasimirConversionAudit:
    """Convert the explicit CE ``keV / fm^3`` cell estimate to SI units."""

    energy = _positive_finite(energy_kev, "energy_kev") * 1e3 * ELECTRON_VOLT_J
    volume = _positive_finite(volume_fm3, "volume_fm3") * FM_M**3
    reported = _positive_finite(
        reported_mass_density_kg_m3,
        "reported_mass_density_kg_m3",
    )
    energy_density = energy / volume
    mass_density = energy_density / SPEED_OF_LIGHT_M_S**2

    return CasimirConversionAudit(
        energy_density_j_m3=energy_density,
        mass_equivalent_density_kg_m3=mass_density,
        reported_mass_density_kg_m3=reported,
        reported_to_correct_ratio=reported / mass_density,
    )


def wormhole_throat_audit(
    *,
    throat_radius_m: float,
    shape_derivative: float = -1.0,
    candidate_negative_density_j_m3: float,
    ce_correlation_length_m: float = 6.65e-15,
) -> WormholeThroatAudit:
    """Compare a Morris-Thorne throat NEC requirement with a CE candidate.

    At a throat ``b(r0)=r0`` the radial null projection is

    ``rho + p_r = c^4 (b'(r0)-1) / (8 pi G r0^2)``.

    ``shape_derivative=-1`` is the simple Ellis/Morris-Thorne control profile
    ``b(r)=r0^2/r``.  Density sufficiency is only a necessary local check; a
    conserved renormalized stress tensor and a stable global solution are still
    required.
    """

    radius = _positive_finite(throat_radius_m, "throat_radius_m")
    b_prime = float(shape_derivative)
    if not math.isfinite(b_prime):
        raise ValueError("shape_derivative must be finite")
    candidate = _positive_finite(
        candidate_negative_density_j_m3,
        "candidate_negative_density_j_m3",
    )
    correlation_length = _positive_finite(
        ce_correlation_length_m,
        "ce_correlation_length_m",
    )

    nec = (
        SPEED_OF_LIGHT_M_S**4
        * (b_prime - 1.0)
        / (8.0 * math.pi * NEWTON_G_M3_KG_S2 * radius**2)
    )
    required_magnitude = abs(min(nec, 0.0))
    density_gap = required_magnitude / candidate
    flare_out = b_prime < 1.0

    return WormholeThroatAudit(
        throat_radius_m=radius,
        shape_derivative=b_prime,
        flare_out_satisfied=flare_out,
        nec_energy_density_j_m3=nec,
        nec_violated=nec < 0.0,
        candidate_negative_density_j_m3=candidate,
        local_density_gap=density_gap,
        candidate_meets_local_density=flare_out and candidate >= required_magnitude,
        ce_correlation_length_m=correlation_length,
        coherence_q_required=radius / correlation_length,
        candidate_coherent_over_throat=correlation_length >= radius,
        ce_stress_tensor_bridge_derived=False,
        physical_wormhole_established=False,
    )
