"""Fail-closed spin/operator audit for the 29.64757 MeV CE fusion mediator.

The scalar fusion loop supplies the D--T Yukawa charge product required for a
one-percent Maxwellian reactivity increase at 10 keV.  This module imports that
product and matches it, without refitting it, onto pseudoscalar, axial-vector,
vector, and massive-spin-2 nonrelativistic potentials.

The numerical matches are operator-level diagnostics.  They are not promoted
to physical fusion predictions: no coupled-channel NCSMC/R-matrix refit of the
5He 3/2+ resonance and no mass-specific pion/kaon/BaBar likelihood are supplied.
Every physical gate therefore remains fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Real
from typing import Any

from .fusion_flavor_aligned_loop import (
    LEAD_NEUTRON_NUMBER,
    LEAD_PROTON_NUMBER,
    REGISTERED_SCALAR_MASS_MEV,
    current_fusion_flavor_aligned_report,
)
from .fusion_resonance_loop import (
    DEUTERON_MASS_MEV,
    HBAR_C_MEV_FM,
    TRITON_MASS_MEV,
)


DEFAULT_TEMPERATURE_KEV = 10.0
DEFAULT_GAMOW_SADDLE_ENERGY_KEV = 30.92
AXIAL_UNIVERSAL_QUARK_BOUND_AT_100_MEV = 3.0e-6
VECTOR_PROMPT_VISIBLE_PROTON_BOUND_17_MEV = 2.42e-4
SPIN_TWO_VISIBLE_BABAR_BOUND_PER_GEV = 3.0e-5
SPIN_TWO_INVISIBLE_BABAR_BOUND_PER_GEV = 2.0e-4


@dataclass(frozen=True)
class SpinAverageAudit:
    deuteron_spin: float
    triton_spin: float
    spin_space_dimension: int
    quartet_degeneracy: int
    doublet_degeneracy: int
    quartet_operator_eigenvalue: float
    doublet_operator_eigenvalue: float
    raw_unpolarized_operator_trace: float
    quartet_projector_formula: str
    quartet_projected_unpolarized_trace: float
    quartet_conditional_operator_expectation: float
    raw_unpolarized_first_order_spin_response_cancels: bool
    dt_s_wave_three_half_resonance_dominates: bool
    exact_ncsmc_or_rmatrix_response_supplied: bool


@dataclass(frozen=True)
class PseudoscalarOperatorAudit:
    action: str
    nr_potential: str
    s_wave_long_range_potential: str
    required_scalar_yukawa_product: float
    required_abs_effective_nuclear_coupling_product: float
    equal_abs_effective_nuclear_coupling: float
    equal_coupling_fine_structure: float
    quartet_attractive_product_sign: int
    same_sign_quartet_force_is_repulsive: bool
    raw_unpolarized_trace_cancels: bool
    quartet_projected_first_order_term_survives: bool
    perturbative_one_boson_exchange: bool
    nuclear_pseudoscalar_form_factors_supplied: bool
    exact_resonance_response_supplied: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class AxialVectorOperatorAudit:
    action: str
    nr_potential: str
    s_wave_long_range_potential: str
    required_scalar_yukawa_product: float
    required_effective_nuclear_coupling_product: float
    equal_effective_nuclear_coupling: float
    quartet_attractive_product_sign: int
    universal_quark_axial_bound_at_mass: float
    naive_nuclear_coupling_to_quark_bound_ratio: float
    universal_bound_scope: str
    nuclear_to_quark_matching_supplied: bool
    nonuniversal_flavor_cancellation_supplied: bool
    mass_specific_kaon_likelihood_supplied: bool
    exact_resonance_response_supplied: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class VectorOperatorAudit:
    action: str
    nr_potential: str
    required_attractive_dt_charge_product: float
    universal_same_sign_vector_is_repulsive: bool
    minimax_gp_over_gn: float
    minimax_proton_coupling: float
    minimax_neutron_coupling: float
    minimax_max_abs_nucleon_coupling: float
    lead_blind_gp_over_gn: float
    lead_blind_proton_coupling: float
    lead_blind_neutron_coupling: float
    lead_zero_momentum_charge: float
    lead_blind_deuteron_charge: float
    lead_blind_triton_charge: float
    lead_blind_dt_charge_product: float
    lead_blind_up_quark_coupling: float
    lead_blind_down_quark_coupling: float
    lead_blind_isovector_quark_coupling: float
    lead_blind_is_attractive_for_dt: bool
    lead_cancellation_is_zero_momentum_only: bool
    finite_momentum_pb_likelihood_supplied: bool
    na48_mass_window_contains_candidate: bool
    prompt_visible_17_mev_proton_bound_proxy: float
    required_to_prompt_visible_proxy_ratio: float
    prompt_visible_proxy_is_mass_specific: bool
    mass_specific_pion_kaon_likelihood_supplied: bool
    anomaly_free_gauge_completion_supplied: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class SpinTwoOperatorAudit:
    action: str
    nr_potential: str
    required_scalar_yukawa_product: float
    required_equal_c_over_lambda_per_gev: float
    required_lambda_over_c_gev: float
    optimistic_dRGT_strong_coupling_scale_gev: float
    visible_babar_bound_per_gev: float
    invisible_babar_bound_per_gev: float
    required_to_visible_bound_ratio: float
    required_to_invisible_bound_ratio: float
    babar_bounds_require_universal_electron_stress_energy_coupling: bool
    mass_specific_babar_likelihood_supplied: bool
    nonuniversal_conserved_uv_completion_supplied: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class DerivativeNodeAudit:
    analytic_exchange_amplitude: str
    on_shell_node_factorization: str
    mediator_pole_invariant_mev2: float
    yukawa_range_fm: float
    gamow_saddle_energy_kev: float
    dt_reduced_mass_mev: float
    incoming_gamow_momentum_mev: float
    on_shell_node_cancels_yukawa_pole_residue: bool
    eom_operator_reduces_to_contact_interaction: bool
    contact_interaction_lowers_long_range_barrier: bool
    broad_spacelike_pb_node_demonstrated: bool
    additional_light_pole_or_mediator_supplied: bool
    mass_specific_differential_likelihood_supplied: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionSpinOperatorReport:
    schema_version: str
    scalar_required_product_source: str
    required_dt_charge_product: float
    spin_average: SpinAverageAudit
    pseudoscalar: PseudoscalarOperatorAudit
    axial_vector: AxialVectorOperatorAudit
    vector: VectorOperatorAudit
    spin_two: SpinTwoOperatorAudit
    derivative_node: DerivativeNodeAudit
    all_declared_operator_math_audited: bool
    exact_ncsmc_or_rmatrix_calculation_supplied: bool
    mass_specific_pion_kaon_babar_likelihoods_supplied: bool
    any_physical_operator_gate_pass: bool
    physical_one_percent_fusion_branch_accepted: bool
    maximum_supported_stage: str
    next_required_input: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scalar_required_dt_charge_product() -> float:
    """Read the registered-mass requirement from the existing scalar audit."""

    return current_fusion_flavor_aligned_report().operator.required_dt_charge_product


def _required_product(value: Real | None) -> float:
    """Return a finite positive scalar target and reject bool/NaN controls."""

    result = scalar_required_dt_charge_product() if value is None else value
    if isinstance(result, bool) or not isinstance(result, Real):
        raise ValueError("required_product must be a real scalar")
    product = float(result)
    if not math.isfinite(product) or product <= 0.0:
        raise ValueError("required_product must be finite and positive")
    return product


def audit_spin_average() -> SpinAverageAudit:
    """Separate the raw six-state trace from the quartet reaction projector."""

    quartet_degeneracy = 4
    doublet_degeneracy = 2
    quartet_eigenvalue = 1.0
    doublet_eigenvalue = -2.0
    dimension = quartet_degeneracy + doublet_degeneracy
    raw_trace = (
        quartet_degeneracy * quartet_eigenvalue + doublet_degeneracy * doublet_eigenvalue
    ) / dimension
    projected_trace = quartet_degeneracy * quartet_eigenvalue / dimension
    return SpinAverageAudit(
        deuteron_spin=1.0,
        triton_spin=0.5,
        spin_space_dimension=dimension,
        quartet_degeneracy=quartet_degeneracy,
        doublet_degeneracy=doublet_degeneracy,
        quartet_operator_eigenvalue=quartet_eigenvalue,
        doublet_operator_eigenvalue=doublet_eigenvalue,
        raw_unpolarized_operator_trace=raw_trace,
        quartet_projector_formula="P_3/2=(Sigma_D.Sigma_T+2)/3",
        quartet_projected_unpolarized_trace=projected_trace,
        quartet_conditional_operator_expectation=quartet_eigenvalue,
        raw_unpolarized_first_order_spin_response_cancels=(raw_trace == 0.0),
        dt_s_wave_three_half_resonance_dominates=True,
        exact_ncsmc_or_rmatrix_response_supplied=False,
    )


def audit_pseudoscalar_operator(
    required_product: Real | None = None,
) -> PseudoscalarOperatorAudit:
    """Match the long-range S-wave pseudoscalar central term to the scalar target."""

    product = _required_product(required_product)
    required_coupling_product = (
        12.0 * DEUTERON_MASS_MEV * TRITON_MASS_MEV * product / REGISTERED_SCALAR_MASS_MEV**2
    )
    equal_coupling = math.sqrt(required_coupling_product)
    return PseudoscalarOperatorAudit(
        action="i*g_Pi*phi*Nbar_i*gamma5*N_i",
        nr_potential=("g_PD*g_PT/(4*m_D*m_T)*(Sigma_D.grad)*(Sigma_T.grad)*Y(r)"),
        s_wave_long_range_potential=("+g_PD*g_PT*M^2*O/(12*m_D*m_T)*Y(r), r>0"),
        required_scalar_yukawa_product=product,
        required_abs_effective_nuclear_coupling_product=required_coupling_product,
        equal_abs_effective_nuclear_coupling=equal_coupling,
        equal_coupling_fine_structure=equal_coupling**2 / (4.0 * math.pi),
        quartet_attractive_product_sign=-1,
        same_sign_quartet_force_is_repulsive=True,
        raw_unpolarized_trace_cancels=True,
        quartet_projected_first_order_term_survives=True,
        perturbative_one_boson_exchange=False,
        nuclear_pseudoscalar_form_factors_supplied=False,
        exact_resonance_response_supplied=False,
        physical_gate_pass=False,
        status="STRONG_EFFECTIVE_COUPLING_AND_EXACT_NUCLEAR_RESPONSE_MISSING",
    )


def audit_axial_vector_operator(
    required_product: Real | None = None,
) -> AxialVectorOperatorAudit:
    """Match the transverse-plus-longitudinal axial S-wave Yukawa term."""

    product = _required_product(required_product)
    required_coupling_product = 1.5 * product
    equal_coupling = math.sqrt(required_coupling_product)
    universal_bound = AXIAL_UNIVERSAL_QUARK_BOUND_AT_100_MEV * REGISTERED_SCALAR_MASS_MEV / 100.0
    return AxialVectorOperatorAudit(
        action="X_mu*Nbar_i*gamma^mu*g_Ai*gamma5*N_i",
        nr_potential=("-g_AD*g_AT*[O+(Sigma_D.q)*(Sigma_T.q)/M^2]/(q^2+M^2)"),
        s_wave_long_range_potential="-(2/3)*g_AD*g_AT*O*Y(r), r>0",
        required_scalar_yukawa_product=product,
        required_effective_nuclear_coupling_product=required_coupling_product,
        equal_effective_nuclear_coupling=equal_coupling,
        quartet_attractive_product_sign=1,
        universal_quark_axial_bound_at_mass=universal_bound,
        naive_nuclear_coupling_to_quark_bound_ratio=equal_coupling / universal_bound,
        universal_bound_scope=(
            "universal diagonal light-quark axial charges; nuclear matching is separate"
        ),
        nuclear_to_quark_matching_supplied=False,
        nonuniversal_flavor_cancellation_supplied=False,
        mass_specific_kaon_likelihood_supplied=False,
        exact_resonance_response_supplied=False,
        physical_gate_pass=False,
        status="PERTURBATIVE_OPERATOR_MATCH_EXISTS_BUT_UV_AND_LIKELIHOOD_GATES_OPEN",
    )


def audit_vector_operator(
    required_product: Real | None = None,
) -> VectorOperatorAudit:
    """Solve the attractive minimax and zero-momentum Pb-blind vector branches."""

    product = _required_product(required_product)

    minimax_ratio = -4.0 / 3.0
    minimax_coefficient = (minimax_ratio + 1.0) * (minimax_ratio + 2.0)
    minimax_gn = -math.sqrt(product / -minimax_coefficient)
    minimax_gp = minimax_ratio * minimax_gn

    lead_ratio = -LEAD_NEUTRON_NUMBER / LEAD_PROTON_NUMBER
    lead_coefficient = (lead_ratio + 1.0) * (lead_ratio + 2.0)
    lead_gn = -math.sqrt(product / -lead_coefficient)
    lead_gp = lead_ratio * lead_gn
    deuteron_charge = lead_gp + lead_gn
    triton_charge = lead_gp + 2.0 * lead_gn
    up_quark = (2.0 * lead_gp - lead_gn) / 3.0
    down_quark = (2.0 * lead_gn - lead_gp) / 3.0

    return VectorOperatorAudit(
        action="X_mu*Nbar_i*gamma^mu*g_Vi*N_i",
        nr_potential="+(g_p+g_n)*(g_p+2*g_n)*Y(r)",
        required_attractive_dt_charge_product=-product,
        universal_same_sign_vector_is_repulsive=True,
        minimax_gp_over_gn=minimax_ratio,
        minimax_proton_coupling=minimax_gp,
        minimax_neutron_coupling=minimax_gn,
        minimax_max_abs_nucleon_coupling=max(abs(minimax_gp), abs(minimax_gn)),
        lead_blind_gp_over_gn=lead_ratio,
        lead_blind_proton_coupling=lead_gp,
        lead_blind_neutron_coupling=lead_gn,
        lead_zero_momentum_charge=(LEAD_PROTON_NUMBER * lead_gp + LEAD_NEUTRON_NUMBER * lead_gn),
        lead_blind_deuteron_charge=deuteron_charge,
        lead_blind_triton_charge=triton_charge,
        lead_blind_dt_charge_product=deuteron_charge * triton_charge,
        lead_blind_up_quark_coupling=up_quark,
        lead_blind_down_quark_coupling=down_quark,
        lead_blind_isovector_quark_coupling=up_quark - down_quark,
        lead_blind_is_attractive_for_dt=(deuteron_charge * triton_charge < 0.0),
        lead_cancellation_is_zero_momentum_only=True,
        finite_momentum_pb_likelihood_supplied=False,
        na48_mass_window_contains_candidate=(9.0 <= REGISTERED_SCALAR_MASS_MEV <= 70.0),
        prompt_visible_17_mev_proton_bound_proxy=(VECTOR_PROMPT_VISIBLE_PROTON_BOUND_17_MEV),
        required_to_prompt_visible_proxy_ratio=(
            abs(lead_gp) / VECTOR_PROMPT_VISIBLE_PROTON_BOUND_17_MEV
        ),
        prompt_visible_proxy_is_mass_specific=False,
        mass_specific_pion_kaon_likelihood_supplied=False,
        anomaly_free_gauge_completion_supplied=False,
        physical_gate_pass=False,
        status="ATTRACTIVE_PB_Q0_BLIND_SOLUTION_EXISTS_BUT_FINITE_Q_AND_UV_GATES_OPEN",
    )


def audit_spin_two_operator(
    required_product: Real | None = None,
) -> SpinTwoOperatorAudit:
    """Match universal massive-spin-2 stress-energy exchange to the target."""

    product = _required_product(required_product)
    required_per_mev = math.sqrt(3.0 * product / (2.0 * DEUTERON_MASS_MEV * TRITON_MASS_MEV))
    required_per_gev = 1000.0 * required_per_mev
    inverse_scale_gev = 1.0 / required_per_gev
    mass_gev = REGISTERED_SCALAR_MASS_MEV / 1000.0
    strong_scale = (mass_gev**2 * inverse_scale_gev) ** (1.0 / 3.0)
    return SpinTwoOperatorAudit(
        action="-(c_i/Lambda)*G_mu_nu*T_i^mu_nu",
        nr_potential="-(2/3)*(c_D*c_T*m_D*m_T/Lambda^2)*Y(r)",
        required_scalar_yukawa_product=product,
        required_equal_c_over_lambda_per_gev=required_per_gev,
        required_lambda_over_c_gev=inverse_scale_gev,
        optimistic_dRGT_strong_coupling_scale_gev=strong_scale,
        visible_babar_bound_per_gev=SPIN_TWO_VISIBLE_BABAR_BOUND_PER_GEV,
        invisible_babar_bound_per_gev=SPIN_TWO_INVISIBLE_BABAR_BOUND_PER_GEV,
        required_to_visible_bound_ratio=(required_per_gev / SPIN_TWO_VISIBLE_BABAR_BOUND_PER_GEV),
        required_to_invisible_bound_ratio=(
            required_per_gev / SPIN_TWO_INVISIBLE_BABAR_BOUND_PER_GEV
        ),
        babar_bounds_require_universal_electron_stress_energy_coupling=True,
        mass_specific_babar_likelihood_supplied=False,
        nonuniversal_conserved_uv_completion_supplied=False,
        physical_gate_pass=False,
        status="UNIVERSAL_PROXY_EXCLUDED_AND_NONUNIVERSAL_CONSERVED_COMPLETION_MISSING",
    )


def audit_derivative_node() -> DerivativeNodeAudit:
    """Record why an analytic on-shell derivative node removes the Yukawa tail."""

    reduced_mass = DEUTERON_MASS_MEV * TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
    saddle_energy_mev = DEFAULT_GAMOW_SADDLE_ENERGY_KEV / 1000.0
    momentum = math.sqrt(2.0 * reduced_mass * saddle_energy_mev)
    return DerivativeNodeAudit(
        analytic_exchange_amplitude="A=F_D(q^2)*F_T(q^2)/(q^2-M^2)",
        on_shell_node_factorization="F(M^2)=0 => F(q^2)=(q^2-M^2)*G(q^2)",
        mediator_pole_invariant_mev2=REGISTERED_SCALAR_MASS_MEV**2,
        yukawa_range_fm=HBAR_C_MEV_FM / REGISTERED_SCALAR_MASS_MEV,
        gamow_saddle_energy_kev=DEFAULT_GAMOW_SADDLE_ENERGY_KEV,
        dt_reduced_mass_mev=reduced_mass,
        incoming_gamow_momentum_mev=momentum,
        on_shell_node_cancels_yukawa_pole_residue=True,
        eom_operator_reduces_to_contact_interaction=True,
        contact_interaction_lowers_long_range_barrier=False,
        broad_spacelike_pb_node_demonstrated=False,
        additional_light_pole_or_mediator_supplied=False,
        mass_specific_differential_likelihood_supplied=False,
        physical_gate_pass=False,
        status="ON_SHELL_NODE_REMOVES_LONG_RANGE_POLE_AND_NO_NEW_MEDIATOR_IS_SUPPLIED",
    )


@lru_cache(maxsize=1)
def current_fusion_spin_operator_report() -> FusionSpinOperatorReport:
    """Compose all non-scalar operator matches without opening a physical gate."""

    product = scalar_required_dt_charge_product()
    spin_average = audit_spin_average()
    pseudoscalar = audit_pseudoscalar_operator(product)
    axial = audit_axial_vector_operator(product)
    vector = audit_vector_operator(product)
    spin_two = audit_spin_two_operator(product)
    derivative = audit_derivative_node()
    any_physical_gate = any(
        (
            pseudoscalar.physical_gate_pass,
            axial.physical_gate_pass,
            vector.physical_gate_pass,
            spin_two.physical_gate_pass,
            derivative.physical_gate_pass,
        )
    )
    return FusionSpinOperatorReport(
        schema_version="1.0",
        scalar_required_product_source=(
            "fusion_flavor_aligned_loop.current_fusion_flavor_aligned_report()."
            "operator.required_dt_charge_product"
        ),
        required_dt_charge_product=product,
        spin_average=spin_average,
        pseudoscalar=pseudoscalar,
        axial_vector=axial,
        vector=vector,
        spin_two=spin_two,
        derivative_node=derivative,
        all_declared_operator_math_audited=True,
        exact_ncsmc_or_rmatrix_calculation_supplied=False,
        mass_specific_pion_kaon_babar_likelihoods_supplied=False,
        any_physical_operator_gate_pass=any_physical_gate,
        physical_one_percent_fusion_branch_accepted=False,
        maximum_supported_stage="OPERATOR_LEVEL_MATCHES_ONLY_FAIL_CLOSED",
        next_required_input=(
            "Coupled-channel NCSMC/R-matrix D-T response plus 29.64757 MeV "
            "pion/kaon/BaBar differential likelihoods and an anomaly-free UV action"
        ),
        conclusion=(
            "The quartet projector preserves first-order spin-dependent terms even though "
            "the raw unpolarized trace vanishes. Pseudoscalar matching is strong, axial "
            "matching is perturbative but not constraint-closed, the vector has an attractive "
            "zero-momentum Pb blind solution only, universal spin-2 exceeds BaBar proxies, "
            "and an analytic on-shell derivative node removes the Yukawa pole itself."
        ),
    )


__all__ = [
    "AxialVectorOperatorAudit",
    "DerivativeNodeAudit",
    "FusionSpinOperatorReport",
    "PseudoscalarOperatorAudit",
    "SpinAverageAudit",
    "SpinTwoOperatorAudit",
    "VectorOperatorAudit",
    "audit_axial_vector_operator",
    "audit_derivative_node",
    "audit_pseudoscalar_operator",
    "audit_spin_average",
    "audit_spin_two_operator",
    "audit_vector_operator",
    "current_fusion_spin_operator_report",
    "scalar_required_dt_charge_product",
]
