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
    profile_center_value: float
    profile_center_derivative_per_m: float
    profile_outer_cutoff_value: float
    profile_normalized_smooth_localized: bool
    radial_gradient_integral_m: float
    radial_gradient_quadrature_delta_m: float
    total_eulerian_energy_j: float
    negative_mass_earth: float
    minimum_eulerian_energy_density_j_m3: float
    thin_wall_energy_j: float
    exact_to_thin_wall_ratio: float
    superluminal_shortcut: bool
    eulerian_weak_energy_condition_violated: bool
    generic_warp_nec_no_go_applicable: bool
    explicit_null_projection_computed: bool
    axis_horizon_pair_exists: bool
    axis_horizon_radius_m: float | None
    horizon_shape_value_target: float | None
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
        + 4.0 * math.fsum(values[1:-1:2])
        + 2.0 * math.fsum(values[2:-1:2])
    ) / 3.0


_WALL_TAIL_CUTOFF = 20.0


def _log_cosh(value: float) -> float:
    magnitude = abs(value)
    return magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)


def _normalized_shape_and_derivative(
    dimensionless_radius: float,
    dimensionless_bubble_radius: float,
    wall_thickness_m: float,
) -> tuple[float, float]:
    """Evaluate the normalized Alcubierre shape without ``cosh`` overflow."""

    q = dimensionless_radius
    a = dimensionless_bubble_radius
    if q == 0.0:
        return 1.0, 0.0
    log_shape = (
        2.0 * _log_cosh(a)
        - _log_cosh(q + a)
        - _log_cosh(q - a)
    )
    shape = math.exp(min(0.0, log_shape))
    derivative = -shape * (
        math.tanh(q + a) + math.tanh(q - a)
    ) / wall_thickness_m
    return shape, derivative


def _gradient_integral(
    bubble_radius_m: float,
    wall_thickness_m: float,
    steps: int,
) -> tuple[float, float, float]:
    aspect_ratio = bubble_radius_m / wall_thickness_m
    lower_u = max(-aspect_ratio, -_WALL_TAIL_CUTOFF)
    upper_u = _WALL_TAIL_CUTOFF

    def integrate(step_count: int) -> tuple[float, float]:
        spacing = (upper_u - lower_u) / step_count
        values: list[float] = []
        maximum_derivative_squared = 0.0
        for index in range(step_count + 1):
            u = lower_u + index * spacing
            dimensionless_radius = max(0.0, aspect_ratio + u)
            _, derivative = _normalized_shape_and_derivative(
                dimensionless_radius,
                aspect_ratio,
                wall_thickness_m,
            )
            derivative_squared = derivative * derivative
            maximum_derivative_squared = max(
                maximum_derivative_squared,
                derivative_squared,
            )
            radius = wall_thickness_m * dimensionless_radius
            # dr = Delta du after centering the quadrature on the wall.
            values.append(
                radius * radius * derivative_squared * wall_thickness_m
            )
        return _simpson_integral(values, spacing), maximum_derivative_squared

    integral, maximum_derivative_squared = integrate(steps)
    coarse_integral, _ = integrate(steps // 2)
    return integral, abs(integral - coarse_integral), maximum_derivative_squared


def _axis_horizon_radius(
    bubble_radius_m: float,
    wall_thickness_m: float,
    speed_over_c: float,
) -> tuple[float | None, float | None]:
    if speed_over_c <= 1.0:
        return None, None

    target = 1.0 - 1.0 / speed_over_c
    if target >= 1.0:
        return 0.0, target

    aspect_ratio = bubble_radius_m / wall_thickness_m
    lower_radius = 0.0
    upper_radius = bubble_radius_m + _WALL_TAIL_CUTOFF * wall_thickness_m

    for _ in range(100):
        midpoint = (lower_radius + upper_radius) / 2.0
        midpoint_shape, _ = _normalized_shape_and_derivative(
            midpoint / wall_thickness_m,
            aspect_ratio,
            wall_thickness_m,
        )
        if midpoint_shape > target:
            lower_radius = midpoint
        else:
            upper_radius = midpoint
    return (lower_radius + upper_radius) / 2.0, target


def audit_alcubierre_tanh_wall(
    bubble_radius_m: float = 10.0,
    wall_thickness_m: float = 1.0,
    speed_over_c: float = 1.0,
    *,
    integration_steps: int = 20_000,
) -> AlcubierreWallAudit:
    """Integrate the Eulerian energy of a normalized smooth spherical wall.

    For unit lapse, flat spatial slices and shift ``beta_x=-v f(r)``,

      rho = -c^4 beta^2 [(d_y f)^2+(d_z f)^2]/(32 pi G).

    Angular integration gives
    ``E=-c^4 beta^2 integral(r^2 f'(r)^2 dr)/(12 G)``.  The profile is

    ``f=[tanh((r+R)/Delta)-tanh((r-R)/Delta)]/[2*tanh(R/Delta)]``.

    It has ``f(0)=1`` and ``f'(0)=0``, unlike the one-sided half-tanh profile.
    Quadrature uses ``u=(r-R)/Delta`` so thin walls remain resolved.  The
    thin-wall limit is ``integral=R^2/(3 Delta)``.

    Negative Eulerian density directly establishes a WEC violation.  The NEC
    result is kept separate: the audit records applicability of the published
    generic-warp NEC no-go, but does not pretend to compute a local null
    projection.  For ``beta>1``, the axial characteristic equation gives the
    horizon condition ``f=1-1/beta``; its root is solved numerically.
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
    if (
        isinstance(integration_steps, bool)
        or not isinstance(integration_steps, int)
        or integration_steps < 1000
        or integration_steps % 4
    ):
        raise ValueError(
            "integration_steps must be an integer multiple of four "
            "and at least 1000"
        )
    aspect_ratio = bubble_radius_m / wall_thickness_m
    if not math.isfinite(aspect_ratio):
        raise ValueError("bubble-radius/wall-thickness ratio must be finite")

    gradient_integral, quadrature_delta, maximum_derivative_squared = (
        _gradient_integral(
            bubble_radius_m,
            wall_thickness_m,
            integration_steps,
        )
    )
    coefficient = -(C**4) * speed_over_c**2 / (12.0 * G)
    total_energy = coefficient * gradient_integral
    thin_wall_integral = bubble_radius_m**2 / (3.0 * wall_thickness_m)
    thin_wall_energy = coefficient * thin_wall_integral
    superluminal = speed_over_c > 1.0
    horizon_radius, horizon_target = _axis_horizon_radius(
        bubble_radius_m,
        wall_thickness_m,
        speed_over_c,
    )

    center_shape, center_derivative = _normalized_shape_and_derivative(
        0.0,
        aspect_ratio,
        wall_thickness_m,
    )
    outer_shape, _ = _normalized_shape_and_derivative(
        aspect_ratio + _WALL_TAIL_CUTOFF,
        aspect_ratio,
        wall_thickness_m,
    )
    profile_gate = (
        math.isclose(center_shape, 1.0, rel_tol=0.0, abs_tol=1.0e-14)
        and math.isclose(center_derivative, 0.0, rel_tol=0.0, abs_tol=1.0e-14)
        and outer_shape < 1.0e-15
    )
    minimum_density = (
        -(C**4)
        * speed_over_c**2
        * maximum_derivative_squared
        / (32.0 * math.pi * G)
    )
    wec_violated = minimum_density < 0.0

    return AlcubierreWallAudit(
        bubble_radius_m=bubble_radius_m,
        wall_thickness_m=wall_thickness_m,
        speed_over_c=speed_over_c,
        profile_center_value=center_shape,
        profile_center_derivative_per_m=center_derivative,
        profile_outer_cutoff_value=outer_shape,
        profile_normalized_smooth_localized=profile_gate,
        radial_gradient_integral_m=gradient_integral,
        radial_gradient_quadrature_delta_m=quadrature_delta,
        total_eulerian_energy_j=total_energy,
        negative_mass_earth=abs(total_energy) / C**2 / EARTH_MASS_KG,
        minimum_eulerian_energy_density_j_m3=minimum_density,
        thin_wall_energy_j=thin_wall_energy,
        exact_to_thin_wall_ratio=gradient_integral / thin_wall_integral,
        superluminal_shortcut=superluminal,
        eulerian_weak_energy_condition_violated=wec_violated,
        generic_warp_nec_no_go_applicable=(
            speed_over_c > 0.0 and profile_gate
        ),
        explicit_null_projection_computed=False,
        axis_horizon_pair_exists=horizon_radius is not None,
        axis_horizon_radius_m=horizon_radius,
        horizon_shape_value_target=horizon_target,
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
            name="2024 positive-energy spherical subluminal shell",
            positive_energy_claim=True,
            superluminal_shortcut=False,
            all_observer_nec_gate=False,
            self_propulsion_free=False,
            explicit_material_source=False,
            verdict=(
                "INTERIOR CONTROL / 2026 BOUNDARY-TAIL ENERGY-CONDITION "
                "FAIL / NOT A SHORTCUT"
            ),
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
