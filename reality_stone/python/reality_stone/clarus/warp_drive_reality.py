"""Energy-condition and scale gates for warp-drive spatial transport."""

from __future__ import annotations

import math
from dataclasses import dataclass


C = 299_792_458.0
G = 6.67430e-11
EARTH_MASS_KG = 5.9722e24


@dataclass(frozen=True)
class AlcubierreWallAudit:
    bubble_radius_m: float
    wall_thickness_m: float
    speed_over_c: float
    radial_gradient_integral_m: float
    total_eulerian_energy_j: float
    negative_mass_earth: float
    thin_wall_energy_j: float
    exact_to_thin_wall_ratio: float
    superluminal_shortcut: bool
    null_energy_condition_violated: bool
    front_back_horizons_expected: bool
    material_source_action_specified: bool
    complete_linear_stability: bool
    realization_pass: bool


@dataclass(frozen=True)
class WarpPathwayAudit:
    name: str
    positive_energy_claim: bool
    superluminal_shortcut: bool
    all_observer_nec_gate: bool
    self_propulsion_free: bool
    explicit_material_source: bool
    verdict: str


def _simpson_integral(values: list[float], step: float) -> float:
    return step * (
        values[0]
        + values[-1]
        + 4.0 * sum(values[1:-1:2])
        + 2.0 * sum(values[2:-1:2])
    ) / 3.0


def audit_alcubierre_tanh_wall(
    bubble_radius_m: float = 10.0,
    wall_thickness_m: float = 1.0,
    speed_over_c: float = 1.0,
    *,
    integration_steps: int = 20_000,
) -> AlcubierreWallAudit:
    """Integrate the Eulerian energy of a smooth spherical tanh wall.

    For unit lapse, flat spatial slices and shift ``beta_x=-v f(r)``,

      rho = -c^4 beta^2 [(d_y f)^2+(d_z f)^2]/(32 pi G).

    Angular integration gives
    ``E=-c^4 beta^2 integral(r^2 f'(r)^2 dr)/(12 G)``.  We use
    ``f=(1-tanh((r-R)/Delta))/2`` and perform the remaining radial integral
    numerically.  The thin-wall limit is ``integral=R^2/(3 Delta)``.
    """

    inputs = (bubble_radius_m, wall_thickness_m, speed_over_c)
    if not all(math.isfinite(value) for value in inputs):
        raise ValueError("warp inputs must be finite")
    if bubble_radius_m <= 0.0:
        raise ValueError("bubble_radius_m must be positive")
    if wall_thickness_m <= 0.0:
        raise ValueError("wall_thickness_m must be positive")
    if speed_over_c < 0.0:
        raise ValueError("speed_over_c must be non-negative")
    if integration_steps < 1000 or integration_steps % 2:
        raise ValueError("integration_steps must be even and at least 1000")

    radial_max = bubble_radius_m + 20.0 * wall_thickness_m
    step = radial_max / integration_steps
    integrand: list[float] = []
    for index in range(integration_steps + 1):
        radius = index * step
        argument = (radius - bubble_radius_m) / wall_thickness_m
        sech_squared = 1.0 / math.cosh(argument) ** 2
        derivative = -0.5 * sech_squared / wall_thickness_m
        integrand.append(radius**2 * derivative**2)

    gradient_integral = _simpson_integral(integrand, step)
    coefficient = -(C**4) * speed_over_c**2 / (12.0 * G)
    total_energy = coefficient * gradient_integral
    thin_wall_integral = bubble_radius_m**2 / (3.0 * wall_thickness_m)
    thin_wall_energy = coefficient * thin_wall_integral
    superluminal = speed_over_c > 1.0

    return AlcubierreWallAudit(
        bubble_radius_m=bubble_radius_m,
        wall_thickness_m=wall_thickness_m,
        speed_over_c=speed_over_c,
        radial_gradient_integral_m=gradient_integral,
        total_eulerian_energy_j=total_energy,
        negative_mass_earth=abs(total_energy) / C**2 / EARTH_MASS_KG,
        thin_wall_energy_j=thin_wall_energy,
        exact_to_thin_wall_ratio=(
            total_energy / thin_wall_energy if thin_wall_energy != 0.0 else 1.0
        ),
        superluminal_shortcut=superluminal,
        null_energy_condition_violated=speed_over_c > 0.0,
        front_back_horizons_expected=superluminal,
        material_source_action_specified=False,
        complete_linear_stability=False,
        realization_pass=False,
    )


def warp_pathway_portfolio() -> tuple[WarpPathwayAudit, ...]:
    """Separate positive-energy subluminal shells from FTL claims."""

    return (
        WarpPathwayAudit(
            name="Alcubierre/Natario superluminal bubble",
            positive_energy_claim=False,
            superluminal_shortcut=True,
            all_observer_nec_gate=False,
            self_propulsion_free=False,
            explicit_material_source=False,
            verdict="GEOMETRY CONTROL / NEC AND SOURCE FAIL",
        ),
        WarpPathwayAudit(
            name="positive-energy spherical subluminal shell",
            positive_energy_claim=True,
            superluminal_shortcut=False,
            all_observer_nec_gate=True,
            self_propulsion_free=False,
            explicit_material_source=False,
            verdict="POSITIVE-ENERGY CONTROL / NOT A SHORTCUT",
        ),
        WarpPathwayAudit(
            name="positive-Eulerian-density superluminal proposal",
            positive_energy_claim=True,
            superluminal_shortcut=True,
            all_observer_nec_gate=False,
            self_propulsion_free=False,
            explicit_material_source=False,
            verdict="REFUTED AS ALL-OBSERVER POSITIVE ENERGY",
        ),
        WarpPathwayAudit(
            name="modified-gravity superluminal warp",
            positive_energy_claim=False,
            superluminal_shortcut=True,
            all_observer_nec_gate=False,
            self_propulsion_free=False,
            explicit_material_source=False,
            verdict="EXTERNAL FRONTIER / NULL-CONVERGENCE VIOLATION",
        ),
    )
