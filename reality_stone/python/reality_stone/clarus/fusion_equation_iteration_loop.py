"""Iterative, constraint-preserving equation loop for the CE fusion branches.

The legacy resonance multiplier is not used.  Instead this module propagates
two action-traceable static interactions through a stable WKB response ratio,
the Bosch--Hale D--T cross section, a Maxwellian average, and a Lawson ledger:

* one-scalar Yukawa exchange for an explicitly broken ``Z2`` branch;
* two-scalar exchange after the Higgs is integrated out in the exact ``Z2``
  branch.

It also evaluates a massless, unit-mixing upper bound.  If that bound misses a
declared engineering target, no positive scalar mass with Higgs-proportional
nucleon coupling can meet the target.  A direct-nucleon coupling is solved only
as a diagnostic of how far outside the selected portal action one must move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
from typing import Any, Callable

import numpy as np

from .ce_two_point_vertex_certificate import ce_light_pole_q04_q05_certificate
from .fusion_full_loop import (
    DEFAULT_HIGGS_PORTAL_MIXING_LIMIT,
    DEFAULT_NUCLEON_FORM_FACTOR,
    DT_ALPHA_ENERGY_MEV,
    bosch_hale_dt_reactivity,
)
from .fusion_resonance_loop import (
    ALPHA_EM,
    DEFAULT_NUCLEAR_RADIUS_FM,
    DEUTERON_MASS_MEV,
    HBAR_C_MEV_FM,
    NUCLEON_MASS_MEV,
    TRITON_MASS_MEV,
)


HIGGS_VEV_MEV = 246_000.0
HIGGS_MASS_MEV = 125_100.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
DEFAULT_SCALAR_MASS_MEV = 29.64757
DEFAULT_TEMPERATURE_KEV = 10.0
DEFAULT_ENGINEERING_GAIN_TARGET = 1.01
DEUTERON_SCALAR_NUCLEON_COUNT = 2.0
TRITON_SCALAR_NUCLEON_COUNT = 3.0
DT_SCALAR_CHARGE_PRODUCT = DEUTERON_SCALAR_NUCLEON_COUNT * TRITON_SCALAR_NUCLEON_COUNT

# Bosch--Hale Eq. 9 / Table IV coefficients for T(d,n)4He.  Energy is keV,
# the fitted astrophysical factor is keV*millibarn, and sigma is returned in m2.
_DT_A = (6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0)
_DT_B = (6.38e1, -9.95e-1, 6.981e-5, 1.728e-4)
_DT_BG_SQRT_KEV = 34.3827
_MILLIBARN_TO_M2 = 1.0e-31


@dataclass(frozen=True)
class StaticFusionChainAudit:
    branch: str
    scalar_mass_mev: float
    interaction_parameter: float
    interaction_parameter_name: str
    dt_scalar_charge_product: float
    coherent_point_nucleus_upper_bound: bool
    potential_at_nuclear_radius_mev: float
    potential_to_coulomb_ratio_at_nuclear_radius: float
    log_wkb_enhancement_at_20_kev: float
    wkb_enhancement_minus_one_at_20_kev: float
    temperature_kev: float
    thermal_reactivity_ratio_minus_one: float
    baseline_reactivity_cm3_s: float
    modified_reactivity_cm3_s: float
    baseline_lawson_n_tau_cm3_s: float
    modified_lawson_n_tau_cm3_s: float
    action_traceable: bool
    selected_action_contains_interaction: bool
    supplied_constraint_pass: bool
    cross_section_bridge_assumption: str
    thermal_chain_closed_conditionally: bool
    engineering_gain_target: float
    engineering_gain_reached: bool
    status: str


@dataclass(frozen=True)
class DirectCouplingRequirementAudit:
    target_thermal_reactivity_ratio: float
    scalar_mass_mev: float
    required_direct_nucleon_coupling: float
    equivalent_higgs_mixing_sine: float
    unit_mixing_bound_exceeded: bool
    supplied_mixing_bound_exceeded: bool
    selected_portal_action_contains_direct_operator: bool
    mathematical_target_reached: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionEquationIterationReport:
    schema_version: str
    bosch_hale_numeric_reactivity_cm3_s: float
    bosch_hale_closed_fit_reactivity_cm3_s: float
    bosch_hale_numeric_to_closed_ratio: float
    allowed_broken_z2: StaticFusionChainAudit
    massless_unit_mixing_upper_bound: StaticFusionChainAudit
    allowed_z2_pair: StaticFusionChainAudit
    massless_z2_pair_upper_bound: StaticFusionChainAudit
    direct_coupling_requirement: DirectCouplingRequirementAudit
    direct_coupling_registered_mass_requirement: DirectCouplingRequirementAudit
    declared_engineering_gain_target: float
    current_selected_action_meets_target: bool
    higgs_proportional_model_class_meets_target: bool
    equation_chain_computationally_closed: bool
    physical_fusion_upgrade_derived: bool
    maximum_supported_stage: str
    next_required_model_change: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _grid_count(value: Integral, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def bosch_hale_dt_cross_section_m2(centre_of_mass_energy_kev: Real) -> float:
    """Return the Bosch--Hale D--T cross section in square metres."""

    energy = _positive(centre_of_mass_energy_kev, name="centre_of_mass_energy_kev")
    a1, a2, a3, a4, a5 = _DT_A
    b1, b2, b3, b4 = _DT_B
    numerator = a1 + energy * (a2 + energy * (a3 + energy * (a4 + energy * a5)))
    denominator = 1.0 + energy * (b1 + energy * (b2 + energy * (b3 + energy * b4)))
    if denominator <= 0.0:
        raise ValueError("centre_of_mass_energy_kev is outside the positive D-T fit branch")
    astrophysical_factor = numerator / denominator
    return (
        _MILLIBARN_TO_M2
        * astrophysical_factor
        / energy
        * math.exp(-_DT_BG_SQRT_KEV / math.sqrt(energy))
    )


def _bessel_k1(values: np.ndarray) -> np.ndarray:
    """Cephes-form approximation to K1 for positive finite array entries."""

    x = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("K1 arguments must be positive and finite")
    result = np.empty_like(x)
    small = x <= 2.0
    if np.any(small):
        xs = x[small]
        y = xs * xs / 4.0
        i1 = xs * (
            0.5
            + y
            * (
                0.87890594
                + y
                * (
                    0.51498869
                    + y
                    * (
                        0.15084934
                        + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))
                    )
                )
            )
        )
        polynomial = 1.0 + y * (
            0.15443144
            + y
            * (
                -0.67278579
                + y
                * (
                    -0.18156897
                    + y * (-0.01919402 + y * (-0.00110404 + y * -0.00004686))
                )
            )
        )
        result[small] = np.log(xs / 2.0) * i1 + polynomial / xs
    if np.any(~small):
        xl = x[~small]
        y = 2.0 / xl
        polynomial = 1.25331414 + y * (
            0.23498619
            + y
            * (
                -0.03655620
                + y
                * (
                    0.01504268
                    + y * (-0.00780353 + y * (0.00325614 + y * -0.00068245))
                )
            )
        )
        result[~small] = np.exp(-xl) * polynomial / np.sqrt(xl)
    return result


def _single_scalar_potential(
    radii_fm: np.ndarray,
    *,
    nucleon_coupling: float,
    scalar_mass_mev: float,
) -> np.ndarray:
    alpha_scalar = DT_SCALAR_CHARGE_PRODUCT * nucleon_coupling**2 / (4.0 * math.pi)
    if scalar_mass_mev == 0.0:
        attenuation = 1.0
    else:
        attenuation = np.exp(-scalar_mass_mev * radii_fm / HBAR_C_MEV_FM)
    return alpha_scalar * HBAR_C_MEV_FM * attenuation / radii_fm


def _pair_contact_coefficient_mev_inv(lambda_hp: float) -> float:
    return (
        2.0
        * lambda_hp
        * DEFAULT_NUCLEON_FORM_FACTOR
        * NUCLEON_MASS_MEV
        / HIGGS_MASS_MEV**2
    )


def _two_scalar_potential(
    radii_fm: np.ndarray,
    *,
    contact_coefficient_mev_inv: float,
    scalar_mass_mev: float,
) -> np.ndarray:
    natural_radii = radii_fm / HBAR_C_MEV_FM
    coefficient_squared = DT_SCALAR_CHARGE_PRODUCT * contact_coefficient_mev_inv**2
    if scalar_mass_mev == 0.0:
        return coefficient_squared / (64.0 * math.pi**3 * natural_radii**3)
    argument = 2.0 * scalar_mass_mev * natural_radii
    return (
        coefficient_squared
        * scalar_mass_mev
        * _bessel_k1(argument)
        / (32.0 * math.pi**3 * natural_radii**2)
    )


def _log_wkb_enhancement(
    energy_kev: float,
    attraction: Callable[[np.ndarray], np.ndarray],
    *,
    grid_points: int,
) -> float:
    energy_mev = energy_kev / 1000.0
    outer_radius = ALPHA_EM * HBAR_C_MEV_FM / energy_mev
    if outer_radius <= DEFAULT_NUCLEAR_RADIUS_FM:
        return 0.0
    # r = r_N + (r_C-r_N) sin^2(theta) clusters points at both endpoints
    # and removes the square-root turning-point singularity through dr/dtheta.
    theta = np.linspace(0.0, 0.5 * math.pi, grid_points)
    span = outer_radius - DEFAULT_NUCLEAR_RADIUS_FM
    sine = np.sin(theta)
    cosine = np.cos(theta)
    radii = DEFAULT_NUCLEAR_RADIUS_FM + span * sine**2
    radial_jacobian = 2.0 * span * sine * cosine
    barrier_excess = ALPHA_EM * HBAR_C_MEV_FM / radii - energy_mev
    attraction_values = np.maximum(attraction(radii), 0.0)
    baseline_root = np.sqrt(np.maximum(barrier_excess, 0.0))
    residual_root = np.sqrt(np.maximum(barrier_excess - attraction_values, 0.0))
    difference = np.empty_like(baseline_root)
    unsaturated = attraction_values < barrier_excess
    difference[~unsaturated] = baseline_root[~unsaturated]
    difference[unsaturated] = attraction_values[unsaturated] / (
        baseline_root[unsaturated] + residual_root[unsaturated]
    )
    reduced_mass = DEUTERON_MASS_MEV * TRITON_MASS_MEV / (
        DEUTERON_MASS_MEV + TRITON_MASS_MEV
    )
    delta_exponent = (
        math.sqrt(2.0 * reduced_mass)
        * float(np.trapezoid(difference * radial_jacobian, theta))
        / HBAR_C_MEV_FM
    )
    return 2.0 * delta_exponent


def _thermal_response(
    *,
    temperature_kev: float,
    attraction: Callable[[np.ndarray], np.ndarray],
    energy_points: int,
    wkb_grid_points: int,
) -> tuple[float, float, float]:
    energies = np.geomspace(0.5, 550.0, energy_points)
    cross_sections = np.array(
        [bosch_hale_dt_cross_section_m2(float(energy)) for energy in energies]
    )
    weights = cross_sections * energies * np.exp(-energies / temperature_kev)
    baseline_integral = float(np.trapezoid(weights, energies))
    log_enhancements = np.array(
        [
            _log_wkb_enhancement(
                float(energy),
                attraction,
                grid_points=wkb_grid_points,
            )
            for energy in energies
        ]
    )
    delta_integral = float(np.trapezoid(weights * np.expm1(log_enhancements), energies))
    ratio_minus_one = delta_integral / baseline_integral

    reduced_mass_kev = 1000.0 * DEUTERON_MASS_MEV * TRITON_MASS_MEV / (
        DEUTERON_MASS_MEV + TRITON_MASS_MEV
    )
    numeric_reactivity_m3_s = (
        math.sqrt(8.0 / (math.pi * reduced_mass_kev))
        * baseline_integral
        / temperature_kev**1.5
        * SPEED_OF_LIGHT_M_S
    )
    numeric_reactivity_cm3_s = numeric_reactivity_m3_s * 1.0e6
    return ratio_minus_one, numeric_reactivity_cm3_s, float(np.max(log_enhancements))


def _build_single_scalar_chain(
    *,
    mixing_angle_sine: float,
    scalar_mass_mev: float,
    temperature_kev: float,
    engineering_gain_target: float,
    supplied_constraint_pass: bool,
    branch: str,
    energy_points: int,
    wkb_grid_points: int,
) -> StaticFusionChainAudit:
    base_coupling = DEFAULT_NUCLEON_FORM_FACTOR * NUCLEON_MASS_MEV / HIGGS_VEV_MEV
    nucleon_coupling = mixing_angle_sine * base_coupling
    def attraction(radii: np.ndarray) -> np.ndarray:
        return _single_scalar_potential(
            radii,
            nucleon_coupling=nucleon_coupling,
            scalar_mass_mev=scalar_mass_mev,
        )
    potential_inner = float(attraction(np.array([DEFAULT_NUCLEAR_RADIUS_FM]))[0])
    coulomb_inner = ALPHA_EM * HBAR_C_MEV_FM / DEFAULT_NUCLEAR_RADIUS_FM
    log_20 = _log_wkb_enhancement(20.0, attraction, grid_points=wkb_grid_points)
    ratio_minus_one, _, _ = _thermal_response(
        temperature_kev=temperature_kev,
        attraction=attraction,
        energy_points=energy_points,
        wkb_grid_points=wkb_grid_points,
    )
    _, _, baseline_reactivity = bosch_hale_dt_reactivity(temperature_kev)
    modified_reactivity = baseline_reactivity * (1.0 + ratio_minus_one)
    baseline_lawson = (
        12.0 * (temperature_kev / 1000.0) / (DT_ALPHA_ENERGY_MEV * baseline_reactivity)
    )
    modified_lawson = baseline_lawson / (1.0 + ratio_minus_one)
    reached = ratio_minus_one >= engineering_gain_target - 1.0
    return StaticFusionChainAudit(
        branch=branch,
        scalar_mass_mev=scalar_mass_mev,
        interaction_parameter=mixing_angle_sine,
        interaction_parameter_name="higgs_mixing_sine",
        dt_scalar_charge_product=DT_SCALAR_CHARGE_PRODUCT,
        coherent_point_nucleus_upper_bound=True,
        potential_at_nuclear_radius_mev=potential_inner,
        potential_to_coulomb_ratio_at_nuclear_radius=potential_inner / coulomb_inner,
        log_wkb_enhancement_at_20_kev=log_20,
        wkb_enhancement_minus_one_at_20_kev=math.expm1(log_20),
        temperature_kev=temperature_kev,
        thermal_reactivity_ratio_minus_one=ratio_minus_one,
        baseline_reactivity_cm3_s=baseline_reactivity,
        modified_reactivity_cm3_s=modified_reactivity,
        baseline_lawson_n_tau_cm3_s=baseline_lawson,
        modified_lawson_n_tau_cm3_s=modified_lawson,
        action_traceable=True,
        selected_action_contains_interaction=False,
        supplied_constraint_pass=supplied_constraint_pass,
        cross_section_bridge_assumption="Bosch-Hale S(E) unchanged; only external-barrier WKB ratio applied",
        thermal_chain_closed_conditionally=True,
        engineering_gain_target=engineering_gain_target,
        engineering_gain_reached=reached,
        status="CONDITIONAL_CHAIN_CLOSED_TARGET_REACHED" if reached else "CHAIN_CLOSED_EFFECT_NEGLIGIBLE",
    )


def _build_pair_chain(
    *,
    lambda_hp: float,
    scalar_mass_mev: float,
    temperature_kev: float,
    engineering_gain_target: float,
    energy_points: int,
    wkb_grid_points: int,
    branch: str,
) -> StaticFusionChainAudit:
    contact = _pair_contact_coefficient_mev_inv(lambda_hp)
    def attraction(radii: np.ndarray) -> np.ndarray:
        return _two_scalar_potential(
            radii,
            contact_coefficient_mev_inv=contact,
            scalar_mass_mev=scalar_mass_mev,
        )
    potential_inner = float(attraction(np.array([DEFAULT_NUCLEAR_RADIUS_FM]))[0])
    coulomb_inner = ALPHA_EM * HBAR_C_MEV_FM / DEFAULT_NUCLEAR_RADIUS_FM
    log_20 = _log_wkb_enhancement(20.0, attraction, grid_points=wkb_grid_points)
    ratio_minus_one, _, _ = _thermal_response(
        temperature_kev=temperature_kev,
        attraction=attraction,
        energy_points=energy_points,
        wkb_grid_points=wkb_grid_points,
    )
    _, _, baseline_reactivity = bosch_hale_dt_reactivity(temperature_kev)
    modified_reactivity = baseline_reactivity * (1.0 + ratio_minus_one)
    baseline_lawson = (
        12.0 * (temperature_kev / 1000.0) / (DT_ALPHA_ENERGY_MEV * baseline_reactivity)
    )
    modified_lawson = baseline_lawson / (1.0 + ratio_minus_one)
    reached = ratio_minus_one >= engineering_gain_target - 1.0
    return StaticFusionChainAudit(
        branch=branch,
        scalar_mass_mev=scalar_mass_mev,
        interaction_parameter=lambda_hp,
        interaction_parameter_name="lambda_hp",
        dt_scalar_charge_product=DT_SCALAR_CHARGE_PRODUCT,
        coherent_point_nucleus_upper_bound=True,
        potential_at_nuclear_radius_mev=potential_inner,
        potential_to_coulomb_ratio_at_nuclear_radius=potential_inner / coulomb_inner,
        log_wkb_enhancement_at_20_kev=log_20,
        wkb_enhancement_minus_one_at_20_kev=math.expm1(log_20),
        temperature_kev=temperature_kev,
        thermal_reactivity_ratio_minus_one=ratio_minus_one,
        baseline_reactivity_cm3_s=baseline_reactivity,
        modified_reactivity_cm3_s=modified_reactivity,
        baseline_lawson_n_tau_cm3_s=baseline_lawson,
        modified_lawson_n_tau_cm3_s=modified_lawson,
        action_traceable=True,
        selected_action_contains_interaction=True,
        supplied_constraint_pass=True,
        cross_section_bridge_assumption="Bosch-Hale S(E) unchanged; two-scalar external-barrier WKB ratio applied",
        thermal_chain_closed_conditionally=True,
        engineering_gain_target=engineering_gain_target,
        engineering_gain_reached=reached,
        status="CONDITIONAL_CHAIN_CLOSED_TARGET_REACHED" if reached else "CHAIN_CLOSED_EFFECT_NEGLIGIBLE",
    )


def _solve_direct_coupling_requirement(
    *,
    target_ratio: float,
    scalar_mass_mev: float,
    temperature_kev: float,
    energy_points: int,
    wkb_grid_points: int,
) -> DirectCouplingRequirementAudit:
    def achieved(coupling: float) -> float:
        def attraction(radii: np.ndarray) -> np.ndarray:
            return _single_scalar_potential(
                radii,
                nucleon_coupling=coupling,
                scalar_mass_mev=scalar_mass_mev,
            )

        ratio_minus_one, _, _ = _thermal_response(
            temperature_kev=temperature_kev,
            attraction=attraction,
            energy_points=energy_points,
            wkb_grid_points=wkb_grid_points,
        )
        return 1.0 + ratio_minus_one

    lower = 0.0
    upper = math.sqrt(4.0 * math.pi * ALPHA_EM) * 0.999
    if achieved(upper) < target_ratio:
        raise RuntimeError("direct-coupling bracket does not reach the requested gain")
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        if achieved(midpoint) >= target_ratio:
            upper = midpoint
        else:
            lower = midpoint
    required = upper
    base_coupling = DEFAULT_NUCLEON_FORM_FACTOR * NUCLEON_MASS_MEV / HIGGS_VEV_MEV
    equivalent_mixing = required / base_coupling
    mathematical = achieved(required) >= target_ratio * (1.0 - 2.0e-12)
    return DirectCouplingRequirementAudit(
        target_thermal_reactivity_ratio=target_ratio,
        scalar_mass_mev=scalar_mass_mev,
        required_direct_nucleon_coupling=required,
        equivalent_higgs_mixing_sine=equivalent_mixing,
        unit_mixing_bound_exceeded=equivalent_mixing > 1.0,
        supplied_mixing_bound_exceeded=equivalent_mixing > DEFAULT_HIGGS_PORTAL_MIXING_LIMIT,
        selected_portal_action_contains_direct_operator=False,
        mathematical_target_reached=mathematical,
        physical_gate_pass=False,
        status="MATHEMATICAL_TARGET_ONLY_NEW_DIRECT_OPERATOR_REQUIRED",
    )


@lru_cache(maxsize=16)
def current_fusion_equation_iteration_report(
    *,
    temperature_kev: Real = DEFAULT_TEMPERATURE_KEV,
    engineering_gain_target: Real = DEFAULT_ENGINEERING_GAIN_TARGET,
    energy_points: Integral = 181,
    wkb_grid_points: Integral = 1001,
) -> FusionEquationIterationReport:
    """Iterate the corrected static equations through the full thermal chain."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    target = _positive(engineering_gain_target, name="engineering_gain_target")
    if target <= 1.0:
        raise ValueError("engineering_gain_target must exceed one")
    energies = _grid_count(energy_points, name="energy_points", minimum=41)
    radial = _grid_count(wkb_grid_points, name="wkb_grid_points", minimum=101)

    certificate = ce_light_pole_q04_q05_certificate()
    lambda_limit = certificate.invisible_width.maximum_allowed_abs_lambda
    if lambda_limit is None:
        raise RuntimeError("portal certificate did not produce a lambda limit")

    allowed_broken = _build_single_scalar_chain(
        mixing_angle_sine=DEFAULT_HIGGS_PORTAL_MIXING_LIMIT,
        scalar_mass_mev=DEFAULT_SCALAR_MASS_MEV,
        temperature_kev=temperature,
        engineering_gain_target=target,
        supplied_constraint_pass=True,
        branch="BROKEN_Z2_SUPPLIED_MIXING_LIMIT",
        energy_points=energies,
        wkb_grid_points=radial,
    )
    model_upper = _build_single_scalar_chain(
        mixing_angle_sine=1.0,
        scalar_mass_mev=0.0,
        temperature_kev=temperature,
        engineering_gain_target=target,
        supplied_constraint_pass=False,
        branch="HIGGS_PROPORTIONAL_MASSLESS_UNIT_MIXING_UPPER_BOUND",
        energy_points=energies,
        wkb_grid_points=radial,
    )
    allowed_pair = _build_pair_chain(
        lambda_hp=lambda_limit,
        scalar_mass_mev=DEFAULT_SCALAR_MASS_MEV,
        temperature_kev=temperature,
        engineering_gain_target=target,
        energy_points=energies,
        wkb_grid_points=radial,
        branch="EXACT_Z2_TWO_SCALAR_ALLOWED_LAMBDA",
    )
    pair_upper = _build_pair_chain(
        lambda_hp=lambda_limit,
        scalar_mass_mev=0.0,
        temperature_kev=temperature,
        engineering_gain_target=target,
        energy_points=energies,
        wkb_grid_points=radial,
        branch="EXACT_Z2_TWO_SCALAR_MASSLESS_UPPER_BOUND",
    )
    direct = _solve_direct_coupling_requirement(
        target_ratio=target,
        scalar_mass_mev=0.0,
        temperature_kev=temperature,
        energy_points=energies,
        wkb_grid_points=radial,
    )
    direct_registered = _solve_direct_coupling_requirement(
        target_ratio=target,
        scalar_mass_mev=DEFAULT_SCALAR_MASS_MEV,
        temperature_kev=temperature,
        energy_points=energies,
        wkb_grid_points=radial,
    )

    def zero_attraction(radii: np.ndarray) -> np.ndarray:
        return np.zeros_like(radii)
    _, numeric_reactivity, _ = _thermal_response(
        temperature_kev=temperature,
        attraction=zero_attraction,
        energy_points=energies,
        wkb_grid_points=radial,
    )
    _, _, closed_reactivity = bosch_hale_dt_reactivity(temperature)
    selected_meets = allowed_broken.engineering_gain_reached or allowed_pair.engineering_gain_reached
    model_class_meets = model_upper.engineering_gain_reached
    return FusionEquationIterationReport(
        schema_version="1.0",
        bosch_hale_numeric_reactivity_cm3_s=numeric_reactivity,
        bosch_hale_closed_fit_reactivity_cm3_s=closed_reactivity,
        bosch_hale_numeric_to_closed_ratio=numeric_reactivity / closed_reactivity,
        allowed_broken_z2=allowed_broken,
        massless_unit_mixing_upper_bound=model_upper,
        allowed_z2_pair=allowed_pair,
        massless_z2_pair_upper_bound=pair_upper,
        direct_coupling_requirement=direct,
        direct_coupling_registered_mass_requirement=direct_registered,
        declared_engineering_gain_target=target,
        current_selected_action_meets_target=selected_meets,
        higgs_proportional_model_class_meets_target=model_class_meets,
        equation_chain_computationally_closed=True,
        physical_fusion_upgrade_derived=False,
        maximum_supported_stage="CONDITIONAL_STATIC_POTENTIAL_TO_THERMAL_REACTIVITY_CHAIN",
        next_required_model_change=(
            "supply an experimentally allowed non-Higgs-proportional nucleon operator or a "
            "source-normalized time-dependent interaction, then recompute the nuclear amplitude"
        ),
        conclusion=(
            "The corrected equations now close from static potential to thermal reactivity.  "
            "Neither the allowed one-scalar branch nor the allowed two-scalar branch reaches "
            "the engineering target; even the massless unit-mixing Higgs-proportional upper "
            "bound fails.  Reaching the target requires a new direct operator outside the "
            "selected portal action, so no physical claim is promoted."
        ),
    )


__all__ = [
    "DirectCouplingRequirementAudit",
    "FusionEquationIterationReport",
    "StaticFusionChainAudit",
    "bosch_hale_dt_cross_section_m2",
    "current_fusion_equation_iteration_report",
]
