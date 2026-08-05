"""Complete fail-closed branch ladder for the CE fusion proposal.

This module continues :mod:`fusion_resonance_loop` through every surviving
branch.  It supplies reproducible controls for the canonical ``Z2`` portal,
an explicitly broken legacy portal, a prescribed coherent scalar background,
the Bosch--Hale D--T baseline, a zero-dimensional Lawson balance, and the NIF
target-gain bookkeeping boundary.  Passing a baseline never supplies the
missing CE scattering amplitude or an ICF radiation-hydrodynamic model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any

from .ce_two_point_vertex_certificate import ce_light_pole_q04_q05_certificate
from .fusion_resonance_loop import (
    DEFAULT_CENTRE_OF_MASS_ENERGY_MEV,
    DEFAULT_LEGACY_MIXING,
    DEFAULT_NUCLEAR_RADIUS_FM,
    DEUTERON_MASS_MEV,
    HBAR_C_MEV_FM,
    HBAR_C_MEV_M,
    NUCLEON_MASS_MEV,
    TRITON_MASS_MEV,
    FusionStageGate,
    current_fusion_resonance_loop_report,
    scalar_line_audit,
)


MEV_TO_JOULE = 1.602176634e-13
HIGGS_VEV_MEV = 246_000.0
DEFAULT_HIGGS_PORTAL_MIXING_LIMIT = 3.8e-3
DEFAULT_NUCLEON_FORM_FACTOR = 0.30
DEFAULT_FRACTIONAL_MASS_MODULATION = 0.01
DEFAULT_DT_TEMPERATURE_KEV = 10.0
DT_ALPHA_ENERGY_MEV = 3.52

# Bosch--Hale coefficients for T(d,n)4He, T in keV and output in cm^3/s.
_BH_C1 = 1.17302e-9
_BH_C2 = 1.51361e-2
_BH_C3 = 7.51886e-2
_BH_C4 = 4.60643e-3
_BH_C5 = 1.35000e-2
_BH_C6 = -1.06750e-4
_BH_C7 = 1.36600e-5
_BH_BG_SQRT_KEV = 34.3827
_BH_REDUCED_MASS_C2_KEV = 1_124_656.0
BOSCH_HALE_DT_TEMPERATURE_MIN_KEV = 0.2
BOSCH_HALE_DT_TEMPERATURE_MAX_KEV = 100.0


@dataclass(frozen=True)
class Z2PairBranchAudit:
    lambda_hp: float
    tree_pair_vertex_present: bool
    single_scalar_source_present: bool
    two_scalar_cut_threshold_mev: float
    two_scalar_asymptotic_range_fm: float
    zero_bare_mass_portal_pole_gev: float
    registered_light_target_predicted: bool
    light_target_portal_dominated: bool
    invisible_branching_fraction: float
    supplied_invisible_limit: float
    maximum_lambda_under_supplied_limit: float | None
    supplied_portal_benchmark_allowed: bool
    renormalized_two_scalar_exchange_amplitude_derived: bool
    dt_scattering_residual_derived: bool
    status: str


@dataclass(frozen=True)
class BrokenZ2BranchAudit:
    legacy_mixing_angle_sine: float
    supplied_mixing_limit: float
    mixing_ratio_to_limit: float
    branching_like_ratio_to_limit: float
    legacy_benchmark_allowed: bool
    legacy_static_force_ratio_at_nuclear_radius: float
    maximum_static_force_ratio_under_supplied_limit: float
    timelike_quality_factor_enhances_static_force: bool
    nonresonant_dt_amplitude_derived: bool
    status: str


@dataclass(frozen=True)
class CoherentBackgroundAudit:
    fractional_nucleon_mass_modulation: float
    scalar_nucleon_coupling: float
    required_field_amplitude_mev: float
    scalar_mass_mev: float
    natural_energy_density_mev4: float
    energy_density_j_m3: float
    quantum_number_density_m3: float
    vacuum_lifetime_s: float
    replenishment_power_density_w_m3: float
    drive_cyclic_frequency_hz: float
    dt_20kev_transit_frequency_hz: float
    drive_to_transit_frequency_ratio: float
    source_current_derived: bool
    coherent_state_preparation_derived: bool
    pump_work_accounted: bool
    backreaction_solved: bool
    floquet_dt_scattering_solved: bool
    status: str


@dataclass(frozen=True)
class ThermalReactivityAudit:
    temperature_kev: float
    bosch_hale_theta_kev: float
    bosch_hale_xi: float
    baseline_reactivity_cm3_s: float
    baseline_ignition_n_tau_cm3_s: float
    candidate_cross_section_supplied: bool
    candidate_reactivity_derived: bool
    counterfactual_wkb_factor_used_as_reactivity: bool
    modified_lawson_value_derived: bool
    status: str


@dataclass(frozen=True)
class IcfIgnitionAudit:
    nif_laser_energy_mj: float
    nif_fusion_yield_mj: float
    published_target_gain: float
    counterfactual_wkb_factor: float
    rejected_linear_rescale_energy_kj: float
    capsule_model_supplied: bool
    laser_to_hotspot_coupling_solved: bool
    implosion_symmetry_solved: bool
    alpha_heating_solved: bool
    radiation_conduction_losses_solved: bool
    hydrodynamic_gain_derived: bool
    ignition_energy_derived: bool
    status: str


@dataclass(frozen=True)
class FullFusionLoopReport:
    schema_version: str
    stages: tuple[FusionStageGate, ...]
    z2_pair: Z2PairBranchAudit
    broken_z2: BrokenZ2BranchAudit
    coherent_background: CoherentBackgroundAudit
    thermal_reactivity: ThermalReactivityAudit
    icf: IcfIgnitionAudit
    all_candidate_branches_exhausted: bool
    physical_dt_amplitude_modified: bool
    modified_thermal_reactivity_derived: bool
    modified_lawson_derived: bool
    nif_ignition_prediction_derived: bool
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


def _fraction(value: Real, *, name: str, allow_zero: bool = False) -> float:
    result = _finite_real(value, name=name)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not lower_ok or result > 1.0:
        qualifier = "[0,1]" if allow_zero else "(0,1]"
        raise ValueError(f"{name} must lie in {qualifier}")
    return result


def audit_z2_pair_branch() -> Z2PairBranchAudit:
    """Bind the fusion branch to the existing Q0.4--Q0.5 portal certificate."""

    certificate = ce_light_pole_q04_q05_certificate()
    width = certificate.invisible_width
    mass_mev = certificate.registered_target_mass_mev
    return Z2PairBranchAudit(
        lambda_hp=width.lambda_hp,
        tree_pair_vertex_present=certificate.conditional_portal_pair_vertex_derived,
        single_scalar_source_present=certificate.vertices.single_phi_source_derived,
        two_scalar_cut_threshold_mev=2.0 * mass_mev,
        two_scalar_asymptotic_range_fm=HBAR_C_MEV_FM / (2.0 * mass_mev),
        zero_bare_mass_portal_pole_gev=(
            certificate.mass_compatibility.zero_bare_mass_portal_pole_gev
        ),
        registered_light_target_predicted=certificate.registered_target_is_predicted_by_portal_action,
        light_target_portal_dominated=(
            certificate.mass_compatibility.same_field_light_pole_and_portal_dominance_compatible
        ),
        invisible_branching_fraction=width.branching_fraction,
        supplied_invisible_limit=width.supplied_branching_fraction_upper_limit,
        maximum_lambda_under_supplied_limit=width.maximum_allowed_abs_lambda,
        supplied_portal_benchmark_allowed=width.supplied_benchmark_allowed,
        renormalized_two_scalar_exchange_amplitude_derived=False,
        dt_scattering_residual_derived=False,
        status="TREE_PAIR_VERTEX_ONLY_PHYSICAL_FUSION_BRANCH_NOT_REACHED",
    )


def audit_broken_z2_branch(
    *,
    legacy_mixing_angle_sine: Real = DEFAULT_LEGACY_MIXING,
    supplied_mixing_limit: Real = DEFAULT_HIGGS_PORTAL_MIXING_LIMIT,
) -> BrokenZ2BranchAudit:
    """Apply a supplied rare-decay mixing limit and retain only static exchange."""

    mixing = _fraction(legacy_mixing_angle_sine, name="legacy_mixing_angle_sine")
    limit = _fraction(supplied_mixing_limit, name="supplied_mixing_limit")
    line = scalar_line_audit(mixing_angle_sine=mixing)
    range_fm = line.compton_length_fm
    legacy_ratio = (
        line.scalar_fine_structure
        * math.exp(-DEFAULT_NUCLEAR_RADIUS_FM / range_fm)
        / (1.0 / 137.035999084)
    )
    constrained_line = scalar_line_audit(mixing_angle_sine=limit)
    constrained_ratio = (
        constrained_line.scalar_fine_structure
        * math.exp(-DEFAULT_NUCLEAR_RADIUS_FM / constrained_line.compton_length_fm)
        / (1.0 / 137.035999084)
    )
    ratio = mixing / limit
    allowed = mixing <= limit
    return BrokenZ2BranchAudit(
        legacy_mixing_angle_sine=mixing,
        supplied_mixing_limit=limit,
        mixing_ratio_to_limit=ratio,
        branching_like_ratio_to_limit=ratio**2,
        legacy_benchmark_allowed=allowed,
        legacy_static_force_ratio_at_nuclear_radius=legacy_ratio,
        maximum_static_force_ratio_under_supplied_limit=constrained_ratio,
        timelike_quality_factor_enhances_static_force=False,
        nonresonant_dt_amplitude_derived=False,
        status=(
            "SUPPLIED_LEGACY_MIXING_ALLOWED_STATIC_ONLY"
            if allowed
            else "LEGACY_MIXING_REJECTED_BY_SUPPLIED_LIMIT"
        ),
    )


def audit_coherent_background(
    *,
    fractional_nucleon_mass_modulation: Real = DEFAULT_FRACTIONAL_MASS_MODULATION,
    mixing_angle_sine: Real = DEFAULT_LEGACY_MIXING,
) -> CoherentBackgroundAudit:
    """Compute the minimum prescribed-field energy scale before source dynamics."""

    fraction = _fraction(
        fractional_nucleon_mass_modulation,
        name="fractional_nucleon_mass_modulation",
    )
    mixing = _fraction(mixing_angle_sine, name="mixing_angle_sine")
    line = scalar_line_audit(mixing_angle_sine=mixing)
    coupling = line.nucleon_yukawa_coupling
    if coupling <= 0.0:
        raise ValueError("mixing_angle_sine must generate a nonzero nucleon coupling")
    amplitude = fraction * NUCLEON_MASS_MEV / coupling
    natural_density = 0.5 * line.scalar_mass_mev**2 * amplitude**2
    mev4_to_j_m3 = MEV_TO_JOULE / HBAR_C_MEV_M**3
    energy_density = natural_density * mev4_to_j_m3
    number_density = (natural_density / line.scalar_mass_mev) / HBAR_C_MEV_M**3
    power_density = energy_density / line.lifetime_s

    reduced_mass_mev = DEUTERON_MASS_MEV * TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
    velocity_fraction = math.sqrt(2.0 * DEFAULT_CENTRE_OF_MASS_ENERGY_MEV / reduced_mass_mev)
    transit_time = DEFAULT_NUCLEAR_RADIUS_FM * 1.0e-15 / (velocity_fraction * 299_792_458.0)
    transit_frequency = 1.0 / transit_time

    return CoherentBackgroundAudit(
        fractional_nucleon_mass_modulation=fraction,
        scalar_nucleon_coupling=coupling,
        required_field_amplitude_mev=amplitude,
        scalar_mass_mev=line.scalar_mass_mev,
        natural_energy_density_mev4=natural_density,
        energy_density_j_m3=energy_density,
        quantum_number_density_m3=number_density,
        vacuum_lifetime_s=line.lifetime_s,
        replenishment_power_density_w_m3=power_density,
        drive_cyclic_frequency_hz=line.cyclic_frequency_hz,
        dt_20kev_transit_frequency_hz=transit_frequency,
        drive_to_transit_frequency_ratio=(line.cyclic_frequency_hz / transit_frequency),
        source_current_derived=False,
        coherent_state_preparation_derived=False,
        pump_work_accounted=False,
        backreaction_solved=False,
        floquet_dt_scattering_solved=False,
        status="PRESCRIBED_BACKGROUND_ENERGY_SCALE_CONTROL_ONLY",
    )


def bosch_hale_dt_reactivity(temperature_kev: Real) -> tuple[float, float, float]:
    """Return ``theta``, ``xi`` and Maxwellian D--T reactivity in ``cm^3/s``."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    if not (BOSCH_HALE_DT_TEMPERATURE_MIN_KEV <= temperature <= BOSCH_HALE_DT_TEMPERATURE_MAX_KEV):
        raise ValueError(
            "temperature_kev lies outside the published 0.2--100 keV Bosch-Hale D-T fit range"
        )
    denominator = 1.0 - temperature * (_BH_C2 + temperature * (_BH_C4 + temperature * _BH_C6)) / (
        1.0 + temperature * (_BH_C3 + temperature * (_BH_C5 + temperature * _BH_C7))
    )
    if denominator <= 0.0:
        raise ValueError("temperature_kev lies outside this Bosch-Hale control domain")
    theta = temperature / denominator
    xi = (_BH_BG_SQRT_KEV**2 / (4.0 * theta)) ** (1.0 / 3.0)
    reactivity = (
        _BH_C1
        * theta
        * math.sqrt(xi / (_BH_REDUCED_MASS_C2_KEV * temperature**3))
        * math.exp(-3.0 * xi)
    )
    return theta, xi, reactivity


def audit_thermal_reactivity(
    *,
    temperature_kev: Real = DEFAULT_DT_TEMPERATURE_KEV,
) -> ThermalReactivityAudit:
    """Close the Standard-Model baseline while locking the CE numerator off."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    theta, xi, reactivity = bosch_hale_dt_reactivity(temperature)
    n_tau = 12.0 * (temperature / 1000.0) / (DT_ALPHA_ENERGY_MEV * reactivity)
    return ThermalReactivityAudit(
        temperature_kev=temperature,
        bosch_hale_theta_kev=theta,
        bosch_hale_xi=xi,
        baseline_reactivity_cm3_s=reactivity,
        baseline_ignition_n_tau_cm3_s=n_tau,
        candidate_cross_section_supplied=False,
        candidate_reactivity_derived=False,
        counterfactual_wkb_factor_used_as_reactivity=False,
        modified_lawson_value_derived=False,
        status="STANDARD_DT_BASELINE_ONLY",
    )


def audit_icf_ignition(
    *,
    counterfactual_wkb_factor: Real,
    nif_laser_energy_mj: Real = 2.05,
    nif_fusion_yield_mj: Real = 3.10,
) -> IcfIgnitionAudit:
    """Record, but reject, the legacy linear laser-energy rescaling."""

    factor = _positive(counterfactual_wkb_factor, name="counterfactual_wkb_factor")
    laser = _positive(nif_laser_energy_mj, name="nif_laser_energy_mj")
    yield_mj = _positive(nif_fusion_yield_mj, name="nif_fusion_yield_mj")
    return IcfIgnitionAudit(
        nif_laser_energy_mj=laser,
        nif_fusion_yield_mj=yield_mj,
        published_target_gain=yield_mj / laser,
        counterfactual_wkb_factor=factor,
        rejected_linear_rescale_energy_kj=1000.0 * laser / factor,
        capsule_model_supplied=False,
        laser_to_hotspot_coupling_solved=False,
        implosion_symmetry_solved=False,
        alpha_heating_solved=False,
        radiation_conduction_losses_solved=False,
        hydrodynamic_gain_derived=False,
        ignition_energy_derived=False,
        status="LINEAR_LASER_ENERGY_RESCALING_REJECTED",
    )


def current_full_fusion_loop_report() -> FullFusionLoopReport:
    """Evaluate every declared fusion branch and compose a fail-closed ledger."""

    resonance = current_fusion_resonance_loop_report()
    z2_pair = audit_z2_pair_branch()
    broken = audit_broken_z2_branch()
    background = audit_coherent_background()
    thermal = audit_thermal_reactivity()
    icf = audit_icf_ignition(
        counterfactual_wkb_factor=(resonance.wkb_q_1e9.counterfactual_tunnelling_enhancement)
    )
    stages = resonance.stages + (
        FusionStageGate(
            "Z2_PAIR_PORTAL_VERTEX",
            "CONDITIONAL_PASS",
            "the supplied tree action has h-phi-phi, but the light pole and rate are not CE-derived",
        ),
        FusionStageGate(
            "Z2_PAIR_FUSION_AMPLITUDE",
            "NOT_REACHED",
            "the two-scalar cut starts at 2m_phi; no renormalized D-T amplitude is supplied",
        ),
        FusionStageGate(
            "BROKEN_Z2_LEGACY_MIXING",
            "REJECT",
            "the legacy mixing exceeds the document-supplied rare-decay limit",
        ),
        FusionStageGate(
            "COHERENT_BACKGROUND_ENERGY_SCALE",
            "NEGATIVE_CONTROL",
            "even a prescribed one-percent nucleon-mass modulation has an enormous field-energy scale",
        ),
        FusionStageGate(
            "TIME_PERIODIC_DT_FLOQUET_SCATTERING",
            "NOT_REACHED",
            "source-normalized field preparation and the time-dependent D-T S-matrix are absent",
        ),
        FusionStageGate(
            "BOSCH_HALE_DT_BASELINE",
            "PASS",
            "the standard 10 keV Maxwellian reactivity and zero-dimensional Lawson baseline reproduce",
        ),
        FusionStageGate(
            "MODIFIED_DT_REACTIVITY_AND_LAWSON",
            "NOT_REACHED",
            "no CE-modified cross section exists to thermally average",
        ),
        FusionStageGate(
            "NIF_RADIATION_HYDRODYNAMIC_GAIN",
            "NOT_REACHED",
            "linear division by a penetrability factor is rejected without a capsule model",
        ),
    )
    return FullFusionLoopReport(
        schema_version="1.0",
        stages=stages,
        z2_pair=z2_pair,
        broken_z2=broken,
        coherent_background=background,
        thermal_reactivity=thermal,
        icf=icf,
        all_candidate_branches_exhausted=True,
        physical_dt_amplitude_modified=False,
        modified_thermal_reactivity_derived=False,
        modified_lawson_derived=False,
        nif_ignition_prediction_derived=False,
        maximum_supported_stage="STANDARD_DT_BASELINE_PLUS_SOURCE_ENERGY_NEGATIVE_CONTROLS",
        next_required_gate=(
            "derive one experimentally allowed renormalized CE vertex, prepare a source-normalized "
            "field, and compute a sign-fixed D-T S-matrix residual before any thermal or ICF upgrade"
        ),
        conclusion=(
            "All declared branches have been looped.  The Z2 branch stops at a supplied tree pair "
            "vertex, the broken branch fails its supplied mixing limit and has no Q enhancement, "
            "the coherent branch stops at its source/energy ledger, and only the Standard-Model "
            "Bosch-Hale and NIF baselines are closed."
        ),
    )


__all__ = [
    "BrokenZ2BranchAudit",
    "CoherentBackgroundAudit",
    "FullFusionLoopReport",
    "IcfIgnitionAudit",
    "ThermalReactivityAudit",
    "Z2PairBranchAudit",
    "audit_broken_z2_branch",
    "audit_coherent_background",
    "audit_icf_ignition",
    "audit_thermal_reactivity",
    "audit_z2_pair_branch",
    "bosch_hale_dt_reactivity",
    "current_full_fusion_loop_report",
]
