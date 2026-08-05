"""Constraint ledger for the closest direct-scalar fusion candidate.

The registered-mass one-percent solution requires a universal per-nucleon
coupling near 0.01743.  A flavor-aligned ``u,d,s`` scalar can reproduce the
same D/T charge product with a gauge-invariant SMEFT-phi operator and a
perturbative vector-like-quark (VLQ) example.  It sits, however, within a few
per mille of an *extrapolated* neutron--nucleus limit and its rare-kaon bound
has an acknowledged order-of-magnitude NLO uncertainty.  This module records
the candidate without converting those central curves into a physical pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

from .fusion_equation_iteration_loop import current_fusion_equation_iteration_report
from .fusion_resonance_loop import HBAR_MEV_S


REGISTERED_SCALAR_MASS_MEV = 29.64757
PROTON_SCALAR_SIGMA_NUMERATOR_GEV = 0.154
NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV = 0.158
BROGGINI_NEUTRON_BOUND_COEFFICIENT = 2.0e-5
BROGGINI_SIGNAL_MASS_CEILING_MEV = 6.05
LEAD_PROTON_NUMBER = 82
LEAD_NEUTRON_NUMBER = 126
LEAD_MASS_NUMBER = LEAD_PROTON_NUMBER + LEAD_NEUTRON_NUMBER
REPRESENTATIVE_PB_MOMENTUM_TRANSFER_MAX_MEV = 13.2
VLQ_BENCHMARK_MASS_GEV = 5_000.0
ELECTROWEAK_VEV_GEV = 246.0
UP_QUARK_MASS_MEV = 2.16
DOWN_QUARK_MASS_MEV = 4.67
STRANGE_QUARK_MASS_MEV = 93.4
DIGITIZED_UDS_INVISIBLE_BOUND_CENTRAL = 182.0
DIGITIZED_BOUND_RELATIVE_LINE_UNCERTAINTY = 0.12
ACKNOWLEDGED_NLO_CORRECTION_FACTOR = 10.0
INVISIBLE_FERMION_MASS_MEV = 5.0
INVISIBLE_YUKAWA = 1.0e-4
SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class FlavorAlignedOperatorAudit:
    scalar_mass_mev: float
    universal_required_nucleon_coupling: float
    required_dt_charge_product: float
    aligned_scale_gev: float
    proton_coupling: float
    neutron_coupling: float
    deuteron_scalar_charge: float
    triton_scalar_charge: float
    reproduced_dt_charge_product: float
    charge_product_relative_residual: float
    gauge_invariant_operator_written: bool
    flavor_alignment_assumed: bool
    vlq_uv_example_supplied: bool
    vlq_mass_gev: float
    required_vlq_kappa: float
    effective_kappa_is_lagrangian_coupling: bool
    required_plot_coordinate_kappa_v_over_m: float
    up_vlq_yukawa: float
    down_vlq_yukawa: float
    strange_vlq_yukawa: float
    maximum_left_mixing_angle: float
    all_displayed_uv_couplings_perturbative: bool
    full_smeft_wet_rg_matching_supplied: bool
    scalar_mass_naturalness_protected: bool
    radiative_mass_stability_gate_pass: bool
    static_one_percent_target_reproduced: bool
    uv_action_gate_pass: bool
    status: str


@dataclass(frozen=True)
class NeutronConstraintAudit:
    extrapolated_equal_coupling_bound: float
    flavor_matched_lead_effective_coupling: float
    central_bound_to_candidate_ratio: float
    central_fractional_margin: float
    neutron_coupling_to_equal_bound_ratio: float
    source_signal_mass_ceiling_mev: float
    candidate_outside_source_signal_mass_range: bool
    representative_max_momentum_transfer_mev: float
    representative_q2_over_m2: float
    contact_limit_correction_scale_exceeds_margin: bool
    original_bound_is_approximate_lesssim: bool
    strong_phase_cancellation_caveat_present: bool
    mass_specific_pb_differential_likelihood_supplied: bool
    nuclear_form_factor_covariance_supplied: bool
    neutron_constraint_gate_pass: bool
    status: str


@dataclass(frozen=True)
class RareDecayConstraintAudit:
    required_plot_coordinate_kappa_v_over_m: float
    digitized_uds_invisible_bound_central: float
    digitized_line_relative_uncertainty: float
    central_bound_to_candidate_ratio: float
    acknowledged_partial_nlo_correction_factor: float
    conservative_nlo_shifted_bound: float
    central_curve_allows_candidate: bool
    conservative_nlo_envelope_allows_candidate: bool
    full_order_p4_weak_chpt_amplitude_supplied: bool
    low_energy_constant_covariance_supplied: bool
    na62_e949_mass_bin_likelihood_supplied: bool
    rare_decay_constraint_gate_pass: bool
    status: str


@dataclass(frozen=True)
class InvisibleCompletionAudit:
    invisible_fermion_mass_mev: float
    invisible_yukawa: float
    partial_width_mev: float
    lifetime_s: float
    decay_length_m: float
    decay_kinematically_open: bool
    invisible_yukawa_perturbative: bool
    prompt_decay_to_invisible_states: bool
    invisible_branching_fraction_computed: bool
    cosmology_supernova_direct_detection_likelihood_supplied: bool
    dark_sector_constraint_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionFlavorAlignedReport:
    schema_version: str
    operator: FlavorAlignedOperatorAudit
    neutron_constraint: NeutronConstraintAudit
    rare_decay_constraint: RareDecayConstraintAudit
    invisible_completion: InvisibleCompletionAudit
    mathematical_one_percent_solution_reproduced: bool
    gauge_invariant_perturbative_uv_candidate_supplied: bool
    all_existing_constraint_gates_pass: bool
    physical_ce_fusion_branch_accepted: bool
    candidate_classification: str
    next_required_input: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_flavor_aligned_operator() -> FlavorAlignedOperatorAudit:
    """Match the one-percent D/T charge product onto aligned quark couplings."""

    iteration = current_fusion_equation_iteration_report()
    universal = (
        iteration.direct_coupling_registered_mass_requirement.required_direct_nucleon_coupling
    )
    required_product = 6.0 * universal**2
    proton_factor = PROTON_SCALAR_SIGMA_NUMERATOR_GEV
    neutron_factor = NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV
    aligned_scale = math.sqrt(
        (proton_factor + neutron_factor) * (proton_factor + 2.0 * neutron_factor) / required_product
    )
    proton_coupling = proton_factor / aligned_scale
    neutron_coupling = neutron_factor / aligned_scale
    deuteron_charge = proton_coupling + neutron_coupling
    triton_charge = proton_coupling + 2.0 * neutron_coupling
    reproduced_product = deuteron_charge * triton_charge

    kappa = VLQ_BENCHMARK_MASS_GEV / aligned_scale
    quark_masses_mev = (UP_QUARK_MASS_MEV, DOWN_QUARK_MASS_MEV, STRANGE_QUARK_MASS_MEV)
    vlq_yukawas = tuple(
        kappa * math.sqrt(2.0) * mass_mev / (1000.0 * ELECTROWEAK_VEV_GEV)
        for mass_mev in quark_masses_mev
    )
    left_mixings = tuple(
        yukawa * ELECTROWEAK_VEV_GEV / (math.sqrt(2.0) * VLQ_BENCHMARK_MASS_GEV)
        for yukawa in vlq_yukawas
    )
    perturbative = max(vlq_yukawas) < math.sqrt(4.0 * math.pi)
    residual = reproduced_product / required_product - 1.0
    # kappa_phi is the low-energy coefficient ratio -lambda*y_F/y_q, not a
    # single Lagrangian coupling.  The displayed lambda=-1 and y_F values can
    # therefore be perturbative even when kappa_phi is large.  That does not,
    # by itself, close RG matching or the 30 MeV scalar's radiative stability.
    uv_pass = False
    return FlavorAlignedOperatorAudit(
        scalar_mass_mev=REGISTERED_SCALAR_MASS_MEV,
        universal_required_nucleon_coupling=universal,
        required_dt_charge_product=required_product,
        aligned_scale_gev=aligned_scale,
        proton_coupling=proton_coupling,
        neutron_coupling=neutron_coupling,
        deuteron_scalar_charge=deuteron_charge,
        triton_scalar_charge=triton_charge,
        reproduced_dt_charge_product=reproduced_product,
        charge_product_relative_residual=residual,
        gauge_invariant_operator_written=True,
        flavor_alignment_assumed=True,
        vlq_uv_example_supplied=True,
        vlq_mass_gev=VLQ_BENCHMARK_MASS_GEV,
        required_vlq_kappa=kappa,
        effective_kappa_is_lagrangian_coupling=False,
        required_plot_coordinate_kappa_v_over_m=(
            kappa * ELECTROWEAK_VEV_GEV / VLQ_BENCHMARK_MASS_GEV
        ),
        up_vlq_yukawa=vlq_yukawas[0],
        down_vlq_yukawa=vlq_yukawas[1],
        strange_vlq_yukawa=vlq_yukawas[2],
        maximum_left_mixing_angle=max(left_mixings),
        all_displayed_uv_couplings_perturbative=perturbative,
        full_smeft_wet_rg_matching_supplied=False,
        scalar_mass_naturalness_protected=False,
        radiative_mass_stability_gate_pass=False,
        static_one_percent_target_reproduced=abs(residual) < 1.0e-12,
        uv_action_gate_pass=uv_pass,
        status="GAUGE_INVARIANT_VLQ_CANDIDATE_DISPLAYED_FULL_UV_CLOSURE_OPEN",
    )


def audit_neutron_constraint(
    operator: FlavorAlignedOperatorAudit,
) -> NeutronConstraintAudit:
    """Apply, but do not over-interpret, the approximate neutron bound."""

    equal_bound = BROGGINI_NEUTRON_BOUND_COEFFICIENT * operator.scalar_mass_mev**2
    lead_charge = (
        LEAD_PROTON_NUMBER * operator.proton_coupling
        + LEAD_NEUTRON_NUMBER * operator.neutron_coupling
    )
    effective = math.sqrt(operator.neutron_coupling * lead_charge / LEAD_MASS_NUMBER)
    ratio = equal_bound / effective
    q2_over_m2 = (REPRESENTATIVE_PB_MOMENTUM_TRANSFER_MAX_MEV / operator.scalar_mass_mev) ** 2
    margin = ratio - 1.0
    correction_exceeds = q2_over_m2 > margin
    return NeutronConstraintAudit(
        extrapolated_equal_coupling_bound=equal_bound,
        flavor_matched_lead_effective_coupling=effective,
        central_bound_to_candidate_ratio=ratio,
        central_fractional_margin=margin,
        neutron_coupling_to_equal_bound_ratio=(operator.neutron_coupling / equal_bound),
        source_signal_mass_ceiling_mev=BROGGINI_SIGNAL_MASS_CEILING_MEV,
        candidate_outside_source_signal_mass_range=(
            operator.scalar_mass_mev > BROGGINI_SIGNAL_MASS_CEILING_MEV
        ),
        representative_max_momentum_transfer_mev=(REPRESENTATIVE_PB_MOMENTUM_TRANSFER_MAX_MEV),
        representative_q2_over_m2=q2_over_m2,
        contact_limit_correction_scale_exceeds_margin=correction_exceeds,
        original_bound_is_approximate_lesssim=True,
        strong_phase_cancellation_caveat_present=True,
        mass_specific_pb_differential_likelihood_supplied=False,
        nuclear_form_factor_covariance_supplied=False,
        neutron_constraint_gate_pass=False,
        status="CENTRAL_EXTRAPOLATION_MARGINAL_MASS_SPECIFIC_LIKELIHOOD_REQUIRED",
    )


def audit_rare_decay_constraint(
    operator: FlavorAlignedOperatorAudit,
) -> RareDecayConstraintAudit:
    """Record the digitized central uds curve and its stated NLO limitation."""

    required = operator.required_plot_coordinate_kappa_v_over_m
    central = DIGITIZED_UDS_INVISIBLE_BOUND_CENTRAL
    conservative = central / ACKNOWLEDGED_NLO_CORRECTION_FACTOR
    return RareDecayConstraintAudit(
        required_plot_coordinate_kappa_v_over_m=required,
        digitized_uds_invisible_bound_central=central,
        digitized_line_relative_uncertainty=DIGITIZED_BOUND_RELATIVE_LINE_UNCERTAINTY,
        central_bound_to_candidate_ratio=central / required,
        acknowledged_partial_nlo_correction_factor=ACKNOWLEDGED_NLO_CORRECTION_FACTOR,
        conservative_nlo_shifted_bound=conservative,
        central_curve_allows_candidate=required < central,
        conservative_nlo_envelope_allows_candidate=required < conservative,
        full_order_p4_weak_chpt_amplitude_supplied=False,
        low_energy_constant_covariance_supplied=False,
        na62_e949_mass_bin_likelihood_supplied=False,
        rare_decay_constraint_gate_pass=False,
        status="CENTRAL_UDS_INVISIBLE_CURVE_ALLOWS_NLO_UNCERTAINTY_FAILS_CLOSED",
    )


def audit_invisible_completion() -> InvisibleCompletionAudit:
    """Supply a perturbative invisible decay example and retain its open constraints."""

    mass = REGISTERED_SCALAR_MASS_MEV
    dark_mass = INVISIBLE_FERMION_MASS_MEV
    yukawa = INVISIBLE_YUKAWA
    kinematic = mass > 2.0 * dark_mass
    phase_space = (1.0 - 4.0 * dark_mass**2 / mass**2) ** 1.5 if kinematic else 0.0
    width = yukawa**2 * mass * phase_space / (8.0 * math.pi)
    lifetime = HBAR_MEV_S / width
    decay_length = SPEED_OF_LIGHT_M_S * lifetime
    return InvisibleCompletionAudit(
        invisible_fermion_mass_mev=dark_mass,
        invisible_yukawa=yukawa,
        partial_width_mev=width,
        lifetime_s=lifetime,
        decay_length_m=decay_length,
        decay_kinematically_open=kinematic,
        invisible_yukawa_perturbative=yukawa < math.sqrt(4.0 * math.pi),
        prompt_decay_to_invisible_states=decay_length < 1.0e-3,
        invisible_branching_fraction_computed=False,
        cosmology_supernova_direct_detection_likelihood_supplied=False,
        dark_sector_constraint_gate_pass=False,
        status="INVISIBLE_DECAY_MECHANISM_EXISTS_JOINT_DARK_CONSTRAINTS_OPEN",
    )


@lru_cache(maxsize=1)
def current_fusion_flavor_aligned_report() -> FusionFlavorAlignedReport:
    """Compose the nearest direct-scalar candidate without rounding up margins."""

    operator = audit_flavor_aligned_operator()
    neutron = audit_neutron_constraint(operator)
    rare = audit_rare_decay_constraint(operator)
    invisible = audit_invisible_completion()
    constraints_pass = (
        neutron.neutron_constraint_gate_pass
        and rare.rare_decay_constraint_gate_pass
        and invisible.dark_sector_constraint_gate_pass
    )
    return FusionFlavorAlignedReport(
        schema_version="1.0",
        operator=operator,
        neutron_constraint=neutron,
        rare_decay_constraint=rare,
        invisible_completion=invisible,
        mathematical_one_percent_solution_reproduced=(
            operator.static_one_percent_target_reproduced
        ),
        gauge_invariant_perturbative_uv_candidate_supplied=(
            operator.gauge_invariant_operator_written
            and operator.vlq_uv_example_supplied
            and operator.all_displayed_uv_couplings_perturbative
        ),
        all_existing_constraint_gates_pass=constraints_pass,
        physical_ce_fusion_branch_accepted=False,
        candidate_classification="CLOSEST_CONDITIONAL_CANDIDATE_NOT_CONSTRAINT_CLEARED",
        next_required_input=(
            "provide a 29.65 MeV Pb differential-scattering likelihood with nuclear form factors "
            "and strong-phase nuisance parameters, a full O(p^4) weak-ChPT uds kaon amplitude "
            "with LEC covariance and NA62/E949 efficiencies, and a joint invisible-sector "
            "cosmology/SN/direct-detection likelihood"
        ),
        conclusion=(
            "The flavor-aligned uds operator exactly reproduces the D/T charge product and has a "
            "gauge-invariant VLQ example with perturbative displayed couplings, but full RG and "
            "radiative mass-stability gates remain open.  Its central invisible-kaon curve is open, "
            "but the neutron-bound extrapolation has only a per-mille margin and the kaon result "
            "admits order-of-magnitude NLO corrections.  It is the closest conditional candidate, "
            "not a constraint-cleared physical CE fusion branch."
        ),
    )


__all__ = [
    "FlavorAlignedOperatorAudit",
    "FusionFlavorAlignedReport",
    "InvisibleCompletionAudit",
    "NeutronConstraintAudit",
    "RareDecayConstraintAudit",
    "audit_flavor_aligned_operator",
    "audit_invisible_completion",
    "audit_neutron_constraint",
    "audit_rare_decay_constraint",
    "current_fusion_flavor_aligned_report",
]
