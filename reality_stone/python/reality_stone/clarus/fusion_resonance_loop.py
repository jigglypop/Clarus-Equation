"""Fail-closed loop engineering audit for the CE fusion proposal.

The module deliberately separates three calculations that the legacy fusion
note combined:

* a timelike scalar pole and its vacuum linewidth;
* spacelike scalar exchange between two slowly moving nuclei;
* a counterfactual WKB potential in which the Yukawa coefficient is multiplied
  by a user supplied quality factor.

Only the last item reproduces the numerical toy used by the note.  It is not a
physical bridge: a static exchange has ``q0 = 0`` and therefore cannot sit on a
positive-mass timelike pole.  No WKB result in this file is promoted to a
reactivity, Lawson, capsule, or laser-energy prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


ALPHA_EM = 1.0 / 137.035999084
HBAR_C_MEV_FM = 197.3269804
HBAR_C_MEV_M = 1.973269804e-13
HBAR_MEV_S = 6.582119569e-22
PLANCK_MEV_S = 4.135667696e-21
ELECTRON_MASS_MEV = 0.510998950
NUCLEON_MASS_MEV = 938.2720813
DEUTERON_MASS_MEV = 1875.61294257
TRITON_MASS_MEV = 2808.92113298
HIGGS_VEV_MEV = 246_000.0

DEFAULT_SCALAR_MASS_MEV = 29.648
DEFAULT_LEGACY_MIXING = 0.04344
DEFAULT_NUCLEON_FORM_FACTOR = 0.30
DEFAULT_NUCLEAR_RADIUS_FM = 3.24
DEFAULT_CENTRE_OF_MASS_ENERGY_MEV = 0.020


@dataclass(frozen=True)
class ScalarLineAudit:
    scalar_mass_mev: float
    mixing_angle_sine: float
    electron_width_mev: float
    lifetime_s: float
    vacuum_quality_factor: float
    angular_frequency_rad_s: float
    cyclic_frequency_hz: float
    angular_linewidth_per_s: float
    cyclic_linewidth_hz: float
    nucleon_yukawa_coupling: float
    scalar_fine_structure: float
    compton_length_fm: float
    collision_cross_section_ansatz_m2: float
    collision_rate_ansatz_per_s: float
    collision_width_ansatz_mev: float
    plasma_quality_factor_under_ansatz: float
    electron_g_minus_two_one_loop: float
    collision_model_derived: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaticExchangeAudit:
    energy_transfer_mev: float
    momentum_transfer_mev: float
    invariant_transfer_mev2: float
    pole_invariant_mev2: float
    spacelike_transfer: bool
    timelike_pole_reached: bool
    static_propagator_has_quality_factor_enhancement: bool
    driven_background_equals_pair_potential: bool
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WkbPotentialAudit:
    centre_of_mass_energy_mev: float
    nuclear_radius_fm: float
    outer_coulomb_turning_radius_fm: float
    reduced_mass_mev: float
    scalar_range_fm: float
    scalar_fine_structure: float
    supplied_quality_factor: float
    inner_radius_cancellation_quality_factor: float
    whole_barrier_removal_quality_factor: float
    whole_barrier_fractional_bandwidth: float
    baseline_exponent: float
    modified_exponent: float
    counterfactual_tunnelling_enhancement: float
    whole_barrier_removed: bool
    quality_factor_to_static_potential_bridge_derived: bool
    thermal_reactivity_derived: bool
    ignition_energy_derived: bool
    maximum_supported_stage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusionStageGate:
    name: str
    status: str
    evidence: str


@dataclass(frozen=True)
class FusionResonanceLoopReport:
    schema_version: str
    stages: tuple[FusionStageGate, ...]
    scalar_line: ScalarLineAudit
    static_exchange: StaticExchangeAudit
    wkb_q_1e9: WkbPotentialAudit
    canonical_z2_linear_nucleon_coupling_present: bool
    legacy_counterfactual_wkb_reproduced: bool
    physical_resonant_barrier_reduction_derived: bool
    thermal_reactivity_derived: bool
    nif_capsule_gain_derived: bool
    ignition_energy_derived: bool
    maximum_supported_stage: str
    next_required_gate: str
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


def _grid_count(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("grid_points must be an integer")
    result = int(value)
    if result < 1001:
        raise ValueError("grid_points must be at least 1001")
    return result


def _uniform_trapezoid(values: np.ndarray, left: float, right: float) -> float:
    step = (right - left) / (values.size - 1)
    return float(step * (0.5 * values[0] + np.sum(values[1:-1]) + 0.5 * values[-1]))


def scalar_line_audit(
    *,
    scalar_mass_mev: Real = DEFAULT_SCALAR_MASS_MEV,
    mixing_angle_sine: Real = DEFAULT_LEGACY_MIXING,
    nucleon_form_factor: Real = DEFAULT_NUCLEON_FORM_FACTOR,
    electron_density_m3: Real = 1.0e32,
    electron_speed_m_s: Real = 5.94e7,
    integration_points: Integral = 200_001,
) -> ScalarLineAudit:
    """Recompute the legacy line arithmetic without validating its plasma ansatz."""

    mass = _positive(scalar_mass_mev, name="scalar_mass_mev")
    mixing = _nonnegative(mixing_angle_sine, name="mixing_angle_sine")
    if mixing > 1.0:
        raise ValueError("mixing_angle_sine must not exceed one")
    form_factor = _nonnegative(nucleon_form_factor, name="nucleon_form_factor")
    density = _nonnegative(electron_density_m3, name="electron_density_m3")
    speed = _nonnegative(electron_speed_m_s, name="electron_speed_m_s")
    points = _grid_count(integration_points)
    if mass <= 2.0 * ELECTRON_MASS_MEV:
        raise ValueError("scalar_mass_mev must be above the electron-pair threshold")

    phase_space = (1.0 - 4.0 * ELECTRON_MASS_MEV**2 / mass**2) ** 1.5
    width = (
        mixing**2
        * ELECTRON_MASS_MEV**2
        * mass
        * phase_space
        / (8.0 * math.pi * HIGGS_VEV_MEV**2)
    )
    if width == 0.0:
        lifetime = math.inf
        vacuum_q = math.inf
    else:
        lifetime = HBAR_MEV_S / width
        vacuum_q = mass / width

    nucleon_yukawa = mixing * NUCLEON_MASS_MEV * form_factor / HIGGS_VEV_MEV
    scalar_alpha = nucleon_yukawa**2 / (4.0 * math.pi)
    collision_cross_section = (
        ALPHA_EM * mixing**2 * (HBAR_C_MEV_M / mass) ** 2
    )
    collision_rate = density * collision_cross_section * speed
    collision_width = HBAR_MEV_S * collision_rate
    plasma_q = mass / (width + collision_width) if width + collision_width > 0.0 else math.inf

    x = np.linspace(0.0, 1.0, points, dtype=float)
    mass_ratio_squared = (mass / ELECTRON_MASS_MEV) ** 2
    integrand = (1.0 - x) ** 2 * (1.0 + x) / (
        (1.0 - x) ** 2 + x * mass_ratio_squared
    )
    loop_integral = _uniform_trapezoid(integrand, 0.0, 1.0)
    electron_yukawa = mixing * ELECTRON_MASS_MEV / HIGGS_VEV_MEV
    delta_a_e = electron_yukawa**2 * loop_integral / (8.0 * math.pi**2)

    return ScalarLineAudit(
        scalar_mass_mev=mass,
        mixing_angle_sine=mixing,
        electron_width_mev=width,
        lifetime_s=lifetime,
        vacuum_quality_factor=vacuum_q,
        angular_frequency_rad_s=mass / HBAR_MEV_S,
        cyclic_frequency_hz=mass / PLANCK_MEV_S,
        angular_linewidth_per_s=width / HBAR_MEV_S,
        cyclic_linewidth_hz=width / PLANCK_MEV_S,
        nucleon_yukawa_coupling=nucleon_yukawa,
        scalar_fine_structure=scalar_alpha,
        compton_length_fm=HBAR_C_MEV_FM / mass,
        collision_cross_section_ansatz_m2=collision_cross_section,
        collision_rate_ansatz_per_s=collision_rate,
        collision_width_ansatz_mev=collision_width,
        plasma_quality_factor_under_ansatz=plasma_q,
        electron_g_minus_two_one_loop=delta_a_e,
        collision_model_derived=False,
    )


def static_exchange_audit(
    *,
    scalar_mass_mev: Real = DEFAULT_SCALAR_MASS_MEV,
    energy_transfer_mev: Real = 0.0,
    momentum_transfer_mev: Real | None = None,
) -> StaticExchangeAudit:
    """Classify the exchange invariant before any resonance enhancement is allowed."""

    mass = _positive(scalar_mass_mev, name="scalar_mass_mev")
    energy = _finite_real(energy_transfer_mev, name="energy_transfer_mev")
    if momentum_transfer_mev is None:
        momentum = HBAR_C_MEV_FM / DEFAULT_NUCLEAR_RADIUS_FM
    else:
        momentum = _nonnegative(momentum_transfer_mev, name="momentum_transfer_mev")
    invariant = energy**2 - momentum**2
    pole = mass**2
    tolerance = 1.0e-12 * max(1.0, pole)
    pole_reached = energy > 0.0 and abs(invariant - pole) <= tolerance
    spacelike = invariant < 0.0

    return StaticExchangeAudit(
        energy_transfer_mev=energy,
        momentum_transfer_mev=momentum,
        invariant_transfer_mev2=invariant,
        pole_invariant_mev2=pole,
        spacelike_transfer=spacelike,
        timelike_pole_reached=pole_reached,
        static_propagator_has_quality_factor_enhancement=False,
        driven_background_equals_pair_potential=False,
        conclusion=(
            "Static nuclear exchange is spacelike and cannot use the timelike scalar pole; "
            "a separately solved driven background would be a one-body mass/force field, "
            "not Q times the static two-body Yukawa potential."
        ),
    )


def _whole_barrier_removal_quality_factor(
    *,
    energy_mev: float,
    scalar_range_fm: float,
    scalar_alpha: float,
) -> float:
    if scalar_alpha <= 0.0:
        return math.inf
    stationary_x = ALPHA_EM * HBAR_C_MEV_FM / (scalar_range_fm * energy_mev) - 1.0
    if stationary_x <= 0.0:
        raise ValueError("selected energy has no positive stationary barrier point")
    effective_alpha = ALPHA_EM * math.exp(stationary_x) / (1.0 + stationary_x)
    return effective_alpha / scalar_alpha


def wkb_counterfactual_audit(
    *,
    supplied_quality_factor: Real,
    scalar_line: ScalarLineAudit | None = None,
    centre_of_mass_energy_mev: Real = DEFAULT_CENTRE_OF_MASS_ENERGY_MEV,
    nuclear_radius_fm: Real = DEFAULT_NUCLEAR_RADIUS_FM,
    grid_points: Integral = 200_001,
) -> WkbPotentialAudit:
    """Evaluate the legacy ``Q * Yukawa`` toy while keeping its bridge locked."""

    quality = _nonnegative(supplied_quality_factor, name="supplied_quality_factor")
    energy = _positive(centre_of_mass_energy_mev, name="centre_of_mass_energy_mev")
    nuclear_radius = _positive(nuclear_radius_fm, name="nuclear_radius_fm")
    points = _grid_count(grid_points)
    line = scalar_line if scalar_line is not None else scalar_line_audit()
    scalar_alpha = line.scalar_fine_structure
    scalar_range = line.compton_length_fm
    outer_radius = ALPHA_EM * HBAR_C_MEV_FM / energy
    if nuclear_radius >= outer_radius:
        raise ValueError("nuclear_radius_fm must lie below the Coulomb turning radius")

    reduced_mass = DEUTERON_MASS_MEV * TRITON_MASS_MEV / (
        DEUTERON_MASS_MEV + TRITON_MASS_MEV
    )
    grid = np.linspace(nuclear_radius, outer_radius, points, dtype=float)
    coulomb = ALPHA_EM * HBAR_C_MEV_FM / grid
    yukawa = quality * scalar_alpha * HBAR_C_MEV_FM * np.exp(-grid / scalar_range) / grid
    baseline_integrand = np.sqrt(2.0 * reduced_mass * np.maximum(coulomb - energy, 0.0))
    modified_integrand = np.sqrt(
        2.0 * reduced_mass * np.maximum(coulomb - yukawa - energy, 0.0)
    )
    baseline_exponent = (
        _uniform_trapezoid(baseline_integrand, nuclear_radius, outer_radius)
        / HBAR_C_MEV_FM
    )
    modified_exponent = (
        _uniform_trapezoid(modified_integrand, nuclear_radius, outer_radius)
        / HBAR_C_MEV_FM
    )
    enhancement = math.exp(2.0 * (baseline_exponent - modified_exponent))

    if scalar_alpha == 0.0:
        inner_q = math.inf
    else:
        inner_q = ALPHA_EM * math.exp(nuclear_radius / scalar_range) / scalar_alpha
    whole_q = _whole_barrier_removal_quality_factor(
        energy_mev=energy,
        scalar_range_fm=scalar_range,
        scalar_alpha=scalar_alpha,
    )

    return WkbPotentialAudit(
        centre_of_mass_energy_mev=energy,
        nuclear_radius_fm=nuclear_radius,
        outer_coulomb_turning_radius_fm=outer_radius,
        reduced_mass_mev=reduced_mass,
        scalar_range_fm=scalar_range,
        scalar_fine_structure=scalar_alpha,
        supplied_quality_factor=quality,
        inner_radius_cancellation_quality_factor=inner_q,
        whole_barrier_removal_quality_factor=whole_q,
        whole_barrier_fractional_bandwidth=0.0 if math.isinf(whole_q) else 1.0 / whole_q,
        baseline_exponent=baseline_exponent,
        modified_exponent=modified_exponent,
        counterfactual_tunnelling_enhancement=enhancement,
        whole_barrier_removed=quality >= whole_q,
        quality_factor_to_static_potential_bridge_derived=False,
        thermal_reactivity_derived=False,
        ignition_energy_derived=False,
        maximum_supported_stage="COUNTERFACTUAL_Q_TIMES_YUKAWA_WKB_CONTROL_ONLY",
    )


def current_fusion_resonance_loop_report() -> FusionResonanceLoopReport:
    """Build the current fail-closed stage ledger for the fusion proposal."""

    line = scalar_line_audit()
    exchange = static_exchange_audit(scalar_mass_mev=line.scalar_mass_mev)
    wkb = wkb_counterfactual_audit(supplied_quality_factor=1.0e9, scalar_line=line)
    stages = (
        FusionStageGate(
            "CANONICAL_Z2_LINEAR_NUCLEON_PORTAL",
            "CLOSED_OFF",
            "exact Z2 with zero singlet VEV gives sin(theta)=0 and no single-scalar Yukawa force",
        ),
        FusionStageGate(
            "LEGACY_SCALAR_LINE_ARITHMETIC",
            "CONDITIONAL_PASS",
            "the e+e- width, vacuum Q, frequency conversion, and one-loop a_e are reproducible",
        ),
        FusionStageGate(
            "STATIC_SPACELIKE_RESONANCE",
            "REJECT",
            "q0=0 exchange has q^2<0 and cannot reach the positive timelike scalar pole",
        ),
        FusionStageGate(
            "DRIVEN_BACKGROUND_SOURCE_AND_ENERGY_LEDGER",
            "OPEN",
            "no source geometry, field amplitude, backreaction, or pump work ledger is supplied",
        ),
        FusionStageGate(
            "COUNTERFACTUAL_WKB_POTENTIAL",
            "CONDITIONAL_PASS",
            "the legacy potential can be recomputed but Q times the static potential is not derived",
        ),
        FusionStageGate(
            "MAXWELLIAN_D_T_REACTIVITY",
            "NOT_REACHED",
            "requires a modified scattering amplitude and thermal average, not a lone penetrability ratio",
        ),
        FusionStageGate(
            "ICF_CAPSULE_AND_IGNITION_ENERGY",
            "NOT_REACHED",
            "requires radiation-hydrodynamic coupling, alpha heating, losses, and a calibrated capsule model",
        ),
    )

    return FusionResonanceLoopReport(
        schema_version="1.0",
        stages=stages,
        scalar_line=line,
        static_exchange=exchange,
        wkb_q_1e9=wkb,
        canonical_z2_linear_nucleon_coupling_present=False,
        legacy_counterfactual_wkb_reproduced=True,
        physical_resonant_barrier_reduction_derived=False,
        thermal_reactivity_derived=False,
        nif_capsule_gain_derived=False,
        ignition_energy_derived=False,
        maximum_supported_stage="LEGACY_COUNTERFACTUAL_WKB_CONTROL_ONLY",
        next_required_gate=(
            "derive a nonzero CE interaction vertex and solve a source-normalized spacetime field "
            "whose modified D-T scattering amplitude differs from the Standard Model"
        ),
        conclusion=(
            "The old 7.4 kJ claim is not supported.  The canonical Z2 branch has no linear force, "
            "the legacy static exchange cannot resonate on the scalar pole, and the corrected "
            "Q=1e9 toy gives only the counterfactual WKB enhancement recorded here."
        ),
    )


__all__ = [
    "FusionResonanceLoopReport",
    "FusionStageGate",
    "ScalarLineAudit",
    "StaticExchangeAudit",
    "WkbPotentialAudit",
    "current_fusion_resonance_loop_report",
    "scalar_line_audit",
    "static_exchange_audit",
    "wkb_counterfactual_audit",
]
