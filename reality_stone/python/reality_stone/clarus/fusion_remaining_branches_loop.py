"""Final fail-closed audit of the remaining CE fusion model changes.

This module does not promote a fitted potential into new physics.  It audits
the direct scalar--nucleon operator needed by the static iteration, compares a
gauge-invariant mass-proportional completion scale with the electroweak scale,
evaluates a nuclear-matter mean-field diagnostic, and closes the source-energy
and frequency controls for a time-dependent drive.  The surviving reactivity
bounds are then propagated to Lawson, fusion-power, and ICF bookkeeping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .fusion_equation_iteration_loop import current_fusion_equation_iteration_report
from .fusion_full_loop import audit_coherent_background
from .fusion_resonance_loop import (
    DEFAULT_NUCLEAR_RADIUS_FM,
    DEUTERON_MASS_MEV,
    HBAR_MEV_S,
    NUCLEON_MASS_MEV,
    TRITON_MASS_MEV,
)


ELEMENTARY_CHARGE_C = 1.602176634e-19
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
SPEED_OF_LIGHT_M_S = 299_792_458.0
MEV_C2_TO_KG = 1.7826619216279e-30
FM_TO_M = 1.0e-15
MEV_TO_JOULE = 1.602176634e-13
HBAR_J_S = 1.054571817e-34
HIGGS_VEV_GEV = 246.0
NUCLEAR_SATURATION_DENSITY_FM3 = 0.16
HBAR_C_MEV_FM = 197.3269804
REGISTERED_SCALAR_MASS_MEV = 29.64757
PUBLISHED_CONTROL_TEMPERATURE_KEV = 1.0
PUBLISHED_FIELD_MIN_V_M = 1.0e15
PUBLISHED_FIELD_MAX_V_M = 1.0e16
PUBLISHED_PHOTON_ENERGY_MAX_KEV = 1.0
NIF_LASER_ENERGY_J = 2.05e6


@dataclass(frozen=True)
class DirectOperatorCompletionAudit:
    massless_required_nucleon_coupling: float
    registered_mass_required_nucleon_coupling: float
    massless_equivalent_higgs_mixing_sine: float
    registered_mass_equivalent_higgs_mixing_sine: float
    mass_proportional_completion_scale_massless_gev: float
    mass_proportional_completion_scale_registered_gev: float
    registered_completion_scale_to_higgs_vev: float
    registered_completion_scale_to_scalar_mass: float
    registered_nuclear_matter_mean_field_mev_per_nucleon: float
    perturbative_low_energy_coupling: bool
    electroweak_symmetric_heavy_completion_separated: bool
    selected_portal_action_contains_direct_operator: bool
    nn_scattering_likelihood_supplied: bool
    nuclear_binding_refit_supplied: bool
    rare_decay_likelihood_supplied: bool
    experimental_constraint_gate_pass: bool
    physical_operator_accepted: bool
    status: str


@dataclass(frozen=True)
class TimeDependentDriveAudit:
    published_control_temperature_kev: float
    published_field_min_v_m: float
    published_field_max_v_m: float
    published_photon_energy_max_kev: float
    published_field_min_energy_density_j_m3: float
    published_field_max_energy_density_j_m3: float
    published_field_max_intensity_w_m2: float
    dt_relative_effective_charge_fraction: float
    one_kev_angular_frequency_rad_s: float
    ce_scalar_angular_frequency_rad_s: float
    ce_scalar_quantum_energy_kev: float
    ce_energy_to_published_photon_ceiling_ratio: float
    quiver_amplitude_at_one_kev_and_max_field_fm: float
    quiver_amplitude_at_ce_frequency_and_max_field_fm: float
    field_for_one_nuclear_radius_quiver_at_ce_frequency_v_m: float
    em_energy_density_for_ce_frequency_nuclear_quiver_j_m3: float
    ce_one_percent_mass_modulation_energy_density_j_m3: float
    ce_scalar_to_published_max_em_energy_density_ratio: float
    ce_frequency_inside_published_control_window: bool
    electromagnetic_control_equals_scalar_drive: bool
    source_geometry_supplied: bool
    pump_energy_ledger_closed: bool
    floquet_dt_scattering_solved: bool
    physical_time_dependent_upgrade_derived: bool
    status: str


@dataclass(frozen=True)
class ReactorPropagationAudit:
    allowed_static_reactivity_fractional_gain: float
    allowed_static_lawson_fractional_reduction: float
    higgs_model_class_reactivity_fractional_upper_bound: float
    higgs_model_class_lawson_fractional_reduction_upper_bound: float
    maximum_linear_fusion_power_fractional_gain: float
    rejected_nif_linear_energy_saving_upper_bound_j: float
    direct_operator_target_reactivity_fractional_gain: float
    direct_operator_lawson_fractional_reduction: float
    direct_operator_physical_gate_pass: bool
    radiation_hydrodynamic_capsule_model_supplied: bool
    icf_prediction_derived: bool
    status: str


@dataclass(frozen=True)
class FusionRemainingBranchesReport:
    schema_version: str
    direct_operator: DirectOperatorCompletionAudit
    time_dependent_drive: TimeDependentDriveAudit
    reactor_propagation: ReactorPropagationAudit
    all_declared_remaining_branches_audited: bool
    static_equation_chain_closed: bool
    direct_operator_physical_gate_pass: bool
    time_dependent_physical_gate_pass: bool
    physical_one_percent_reactivity_gain_derived: bool
    physical_reactor_or_icf_upgrade_derived: bool
    maximum_supported_stage: str
    next_required_external_input: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _em_energy_density(field_v_m: float) -> float:
    return 0.5 * VACUUM_PERMITTIVITY_F_M * field_v_m**2


def _dt_relative_effective_charge_fraction() -> float:
    return (TRITON_MASS_MEV - DEUTERON_MASS_MEV) / (
        TRITON_MASS_MEV + DEUTERON_MASS_MEV
    )


def _dt_reduced_mass_kg() -> float:
    reduced_mass_mev = DEUTERON_MASS_MEV * TRITON_MASS_MEV / (
        DEUTERON_MASS_MEV + TRITON_MASS_MEV
    )
    return reduced_mass_mev * MEV_C2_TO_KG


def _relative_quiver_amplitude_m(*, field_v_m: float, angular_frequency_rad_s: float) -> float:
    effective_charge = _dt_relative_effective_charge_fraction() * ELEMENTARY_CHARGE_C
    return effective_charge * field_v_m / (_dt_reduced_mass_kg() * angular_frequency_rad_s**2)


def audit_direct_operator_completion() -> DirectOperatorCompletionAudit:
    """Audit the direct operator required by the one-percent static target."""

    iteration = current_fusion_equation_iteration_report()
    massless = iteration.direct_coupling_requirement
    registered = iteration.direct_coupling_registered_mass_requirement
    f_n = 0.30
    massless_scale_gev = f_n * (NUCLEON_MASS_MEV / 1000.0) / (
        massless.required_direct_nucleon_coupling
    )
    registered_scale_gev = f_n * (NUCLEON_MASS_MEV / 1000.0) / (
        registered.required_direct_nucleon_coupling
    )
    density_mev3 = NUCLEAR_SATURATION_DENSITY_FM3 * HBAR_C_MEV_FM**3
    mean_field_per_nucleon = (
        registered.required_direct_nucleon_coupling**2
        * density_mev3
        / (2.0 * REGISTERED_SCALAR_MASS_MEV**2)
    )
    perturbative = registered.required_direct_nucleon_coupling < math.sqrt(4.0 * math.pi)
    ew_separated = registered_scale_gev > HIGGS_VEV_GEV
    return DirectOperatorCompletionAudit(
        massless_required_nucleon_coupling=massless.required_direct_nucleon_coupling,
        registered_mass_required_nucleon_coupling=(
            registered.required_direct_nucleon_coupling
        ),
        massless_equivalent_higgs_mixing_sine=massless.equivalent_higgs_mixing_sine,
        registered_mass_equivalent_higgs_mixing_sine=(
            registered.equivalent_higgs_mixing_sine
        ),
        mass_proportional_completion_scale_massless_gev=massless_scale_gev,
        mass_proportional_completion_scale_registered_gev=registered_scale_gev,
        registered_completion_scale_to_higgs_vev=registered_scale_gev / HIGGS_VEV_GEV,
        registered_completion_scale_to_scalar_mass=(
            1000.0 * registered_scale_gev / REGISTERED_SCALAR_MASS_MEV
        ),
        registered_nuclear_matter_mean_field_mev_per_nucleon=mean_field_per_nucleon,
        perturbative_low_energy_coupling=perturbative,
        electroweak_symmetric_heavy_completion_separated=ew_separated,
        selected_portal_action_contains_direct_operator=False,
        nn_scattering_likelihood_supplied=False,
        nuclear_binding_refit_supplied=False,
        rare_decay_likelihood_supplied=False,
        experimental_constraint_gate_pass=False,
        physical_operator_accepted=False,
        status="LOW_ENERGY_MATH_SOLUTION_UV_AND_EXPERIMENTAL_GATES_FAIL_CLOSED",
    )


def audit_time_dependent_drive() -> TimeDependentDriveAudit:
    """Close published EM controls and prevent their promotion to a CE scalar drive."""

    one_kev_omega = (1000.0 * ELEMENTARY_CHARGE_C) / HBAR_J_S
    ce_omega = REGISTERED_SCALAR_MASS_MEV / HBAR_MEV_S
    ce_energy_kev = 1000.0 * REGISTERED_SCALAR_MASS_MEV
    quiver_one_kev = _relative_quiver_amplitude_m(
        field_v_m=PUBLISHED_FIELD_MAX_V_M,
        angular_frequency_rad_s=one_kev_omega,
    )
    quiver_ce = _relative_quiver_amplitude_m(
        field_v_m=PUBLISHED_FIELD_MAX_V_M,
        angular_frequency_rad_s=ce_omega,
    )
    effective_charge = _dt_relative_effective_charge_fraction() * ELEMENTARY_CHARGE_C
    field_for_nuclear_quiver = (
        DEFAULT_NUCLEAR_RADIUS_FM
        * FM_TO_M
        * _dt_reduced_mass_kg()
        * ce_omega**2
        / effective_charge
    )
    coherent = audit_coherent_background()
    published_max_density = _em_energy_density(PUBLISHED_FIELD_MAX_V_M)
    return TimeDependentDriveAudit(
        published_control_temperature_kev=PUBLISHED_CONTROL_TEMPERATURE_KEV,
        published_field_min_v_m=PUBLISHED_FIELD_MIN_V_M,
        published_field_max_v_m=PUBLISHED_FIELD_MAX_V_M,
        published_photon_energy_max_kev=PUBLISHED_PHOTON_ENERGY_MAX_KEV,
        published_field_min_energy_density_j_m3=_em_energy_density(PUBLISHED_FIELD_MIN_V_M),
        published_field_max_energy_density_j_m3=published_max_density,
        published_field_max_intensity_w_m2=(published_max_density * SPEED_OF_LIGHT_M_S),
        dt_relative_effective_charge_fraction=_dt_relative_effective_charge_fraction(),
        one_kev_angular_frequency_rad_s=one_kev_omega,
        ce_scalar_angular_frequency_rad_s=ce_omega,
        ce_scalar_quantum_energy_kev=ce_energy_kev,
        ce_energy_to_published_photon_ceiling_ratio=(
            ce_energy_kev / PUBLISHED_PHOTON_ENERGY_MAX_KEV
        ),
        quiver_amplitude_at_one_kev_and_max_field_fm=quiver_one_kev / FM_TO_M,
        quiver_amplitude_at_ce_frequency_and_max_field_fm=quiver_ce / FM_TO_M,
        field_for_one_nuclear_radius_quiver_at_ce_frequency_v_m=field_for_nuclear_quiver,
        em_energy_density_for_ce_frequency_nuclear_quiver_j_m3=(
            _em_energy_density(field_for_nuclear_quiver)
        ),
        ce_one_percent_mass_modulation_energy_density_j_m3=(coherent.energy_density_j_m3),
        ce_scalar_to_published_max_em_energy_density_ratio=(
            coherent.energy_density_j_m3 / published_max_density
        ),
        ce_frequency_inside_published_control_window=(
            ce_energy_kev <= PUBLISHED_PHOTON_ENERGY_MAX_KEV
        ),
        electromagnetic_control_equals_scalar_drive=False,
        source_geometry_supplied=False,
        pump_energy_ledger_closed=False,
        floquet_dt_scattering_solved=False,
        physical_time_dependent_upgrade_derived=False,
        status="PUBLISHED_EM_CONTROL_REPRODUCED_CE_SOURCE_AND_FLOQUET_CHAIN_NOT_REACHED",
    )


def audit_reactor_propagation() -> ReactorPropagationAudit:
    """Propagate only bounded fractional changes, never a capsule prediction."""

    iteration = current_fusion_equation_iteration_report()
    allowed_gain = iteration.allowed_broken_z2.thermal_reactivity_ratio_minus_one
    upper_gain = iteration.massless_unit_mixing_upper_bound.thermal_reactivity_ratio_minus_one
    target_gain = iteration.declared_engineering_gain_target - 1.0
    return ReactorPropagationAudit(
        allowed_static_reactivity_fractional_gain=allowed_gain,
        allowed_static_lawson_fractional_reduction=allowed_gain / (1.0 + allowed_gain),
        higgs_model_class_reactivity_fractional_upper_bound=upper_gain,
        higgs_model_class_lawson_fractional_reduction_upper_bound=(
            upper_gain / (1.0 + upper_gain)
        ),
        maximum_linear_fusion_power_fractional_gain=upper_gain,
        rejected_nif_linear_energy_saving_upper_bound_j=(
            NIF_LASER_ENERGY_J * upper_gain / (1.0 + upper_gain)
        ),
        direct_operator_target_reactivity_fractional_gain=target_gain,
        direct_operator_lawson_fractional_reduction=target_gain / (1.0 + target_gain),
        direct_operator_physical_gate_pass=False,
        radiation_hydrodynamic_capsule_model_supplied=False,
        icf_prediction_derived=False,
        status="FRACTIONAL_REACTIVITY_BOUNDS_ONLY_ICF_LINEAR_RESCALING_REJECTED",
    )


def current_fusion_remaining_branches_report() -> FusionRemainingBranchesReport:
    """Build the final direct-operator, driven-field, and reactor boundary ledger."""

    direct = audit_direct_operator_completion()
    drive = audit_time_dependent_drive()
    reactor = audit_reactor_propagation()
    return FusionRemainingBranchesReport(
        schema_version="1.0",
        direct_operator=direct,
        time_dependent_drive=drive,
        reactor_propagation=reactor,
        all_declared_remaining_branches_audited=True,
        static_equation_chain_closed=True,
        direct_operator_physical_gate_pass=direct.physical_operator_accepted,
        time_dependent_physical_gate_pass=drive.physical_time_dependent_upgrade_derived,
        physical_one_percent_reactivity_gain_derived=False,
        physical_reactor_or_icf_upgrade_derived=False,
        maximum_supported_stage="MODEL_CLASS_NO_GO_PLUS_SOURCE_ENERGY_CONTROLS",
        next_required_external_input=(
            "provide a gauge-invariant UV action and a joint NN-scattering/nuclear-binding/"
            "rare-decay likelihood for the direct operator, or a source-normalized spacetime "
            "drive with pump work and a validated Floquet D-T scattering calculation"
        ),
        conclusion=(
            "Every declared static and time-dependent branch has now been evaluated.  The direct "
            "operator reaches one percent only as a low-energy mathematical fit and lacks UV and "
            "experimental clearance.  The published electromagnetic strong-field control is far "
            "from the CE scalar frequency and is not a scalar-source solution.  No physical reactor "
            "or ICF upgrade is derived."
        ),
    )


__all__ = [
    "DirectOperatorCompletionAudit",
    "FusionRemainingBranchesReport",
    "ReactorPropagationAudit",
    "TimeDependentDriveAudit",
    "audit_direct_operator_completion",
    "audit_reactor_propagation",
    "audit_time_dependent_drive",
    "current_fusion_remaining_branches_report",
]
