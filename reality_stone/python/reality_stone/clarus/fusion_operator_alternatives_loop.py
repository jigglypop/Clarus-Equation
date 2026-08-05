"""Fail-closed audit of direct-scalar operator alternatives for D--T fusion.

The flavor-aligned scalar is the closest direct candidate.  This module checks
whether common changes of operator basis evade its marginal constraints:

* a pure gluon/trace coupling;
* proton- or neutron-phobic isospin choices;
* a Z2-even disformal derivative coupling.

The disformal branch is propagated through the same WKB, Bosch--Hale, and
Maxwellian chain as the static scalar calculation.  Its massless two-scalar
potential is an optimistic upper bound for the registered massive scalar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np

from .fusion_equation_iteration_loop import _thermal_response
from .fusion_flavor_aligned_loop import (
    BROGGINI_NEUTRON_BOUND_COEFFICIENT,
    LEAD_MASS_NUMBER,
    LEAD_NEUTRON_NUMBER,
    LEAD_PROTON_NUMBER,
    REGISTERED_SCALAR_MASS_MEV,
)
from .fusion_resonance_loop import (
    DEUTERON_MASS_MEV,
    HBAR_C_MEV_FM,
    TRITON_MASS_MEV,
)
from .fusion_equation_iteration_loop import current_fusion_equation_iteration_report


PROTON_TRACE_COEFFICIENT_GEV = 0.7844
NEUTRON_TRACE_COEFFICIENT_GEV = 0.7817
DIGITIZED_TRACE_RARE_DECAY_BOUND = 1.38e-3
ELECTROWEAK_VEV_GEV = 246.0
HYDROGEN_SPECTROSCOPY_DISFORMAL_BOUND_MEV = 200.0
STELLAR_BURNING_DISFORMAL_BOUND_MEV = 810.0
ATLAS_DISFORMAL_BOUND_MEV = 1.2e6
NEUTRON_PHOBIC_KAON_COMBINATION_ONE = 198.0
NEUTRON_PHOBIC_KAON_COMBINATION_TWO = 1581.0
DIGITIZED_KAON_COMBINATION_ONE_BOUND = 0.0209
DIGITIZED_KAON_COMBINATION_TWO_BOUND = 0.0574


@dataclass(frozen=True)
class TraceGluonOperatorAudit:
    required_dt_charge_product: float
    required_trace_coefficient_per_gev: float
    required_scale_over_trace_coefficient_gev: float
    required_plot_coordinate_abs_k_theta_v_over_f: float
    digitized_rare_decay_bound: float
    required_to_bound_ratio: float
    gauge_invariant_operator_available: bool
    one_parameter_rare_decay_gate_pass: bool
    tuned_multicoupling_cancellation_supplied: bool
    physical_operator_gate_pass: bool
    status: str


@dataclass(frozen=True)
class IsospinOperatorAudit:
    required_dt_charge_product: float
    universal_minimax_coupling: float
    protophobic_required_neutron_coupling: float
    protophobic_lead_effective_coupling: float
    protophobic_to_neutron_bound_ratio: float
    neutron_phobic_required_proton_coupling: float
    neutron_phobic_kaon_combination_one: float
    neutron_phobic_kaon_combination_one_bound: float
    neutron_phobic_kaon_combination_one_violation: float
    neutron_phobic_kaon_combination_two: float
    neutron_phobic_kaon_combination_two_bound: float
    neutron_phobic_kaon_combination_two_violation: float
    lead_cancellation_gp_over_gn: float
    lead_cancellation_dt_product_coefficient: float
    lead_cancellation_makes_dt_attraction: bool
    universal_minimizes_max_abs_nucleon_coupling: bool
    neutron_only_proxy_favors_neutron_phobic_limit: bool
    protophobic_gate_pass: bool
    neutron_phobic_gate_pass: bool
    lead_blind_spot_gate_pass: bool
    status: str


@dataclass(frozen=True)
class DisformalOperatorAudit:
    potential_coefficient_formula: str
    massless_two_scalar_upper_bound: bool
    required_scale_for_one_percent_mev: float
    coarse_gain_at_required_scale: float
    default_gain_at_required_scale: float
    fine_gain_at_required_scale: float
    maximum_grid_gain_spread: float
    hydrogen_spectroscopy_bound_mev: float
    gain_at_hydrogen_bound: float
    hydrogen_bound_derived_for_massless_scalar: bool
    stellar_burning_bound_mev: float
    gain_at_stellar_bound: float
    stellar_bound_derived_for_massless_scalar: bool
    mass_specific_atomic_or_stellar_bound_supplied: bool
    atlas_bound_mev: float
    gain_at_atlas_bound: float
    atlas_bound_applicable_in_light_mediator_limit: bool
    required_scale_below_applicable_atlas_bound: bool
    required_scale_below_all_supplied_bounds: bool
    nonlinear_screening_completion_supplied: bool
    plasma_unscreening_demonstrated: bool
    experimental_constraint_gate_pass: bool
    physical_operator_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionOperatorAlternativesReport:
    schema_version: str
    trace_gluon: TraceGluonOperatorAudit
    isospin: IsospinOperatorAudit
    disformal: DisformalOperatorAudit
    all_declared_operator_alternatives_audited: bool
    any_alternative_constraint_cleared: bool
    physical_one_percent_ce_branch_derived: bool
    maximum_supported_stage: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_dt_product() -> float:
    coupling = (
        current_fusion_equation_iteration_report()
        .direct_coupling_registered_mass_requirement.required_direct_nucleon_coupling
    )
    return 6.0 * coupling**2


def audit_trace_gluon_operator() -> TraceGluonOperatorAudit:
    """Match a one-parameter trace/gluon coupling and apply its rare-decay bound."""

    product = _required_dt_product()
    charge_factor = (
        (PROTON_TRACE_COEFFICIENT_GEV + NEUTRON_TRACE_COEFFICIENT_GEV)
        * (PROTON_TRACE_COEFFICIENT_GEV + 2.0 * NEUTRON_TRACE_COEFFICIENT_GEV)
    )
    coefficient_per_gev = math.sqrt(product / charge_factor)
    inverse_scale = 1.0 / coefficient_per_gev
    plot_coordinate = ELECTROWEAK_VEV_GEV * coefficient_per_gev
    violation = plot_coordinate / DIGITIZED_TRACE_RARE_DECAY_BOUND
    return TraceGluonOperatorAudit(
        required_dt_charge_product=product,
        required_trace_coefficient_per_gev=coefficient_per_gev,
        required_scale_over_trace_coefficient_gev=inverse_scale,
        required_plot_coordinate_abs_k_theta_v_over_f=plot_coordinate,
        digitized_rare_decay_bound=DIGITIZED_TRACE_RARE_DECAY_BOUND,
        required_to_bound_ratio=violation,
        gauge_invariant_operator_available=True,
        one_parameter_rare_decay_gate_pass=False,
        tuned_multicoupling_cancellation_supplied=False,
        physical_operator_gate_pass=False,
        status="PURE_TRACE_GLUON_REQUIRED_COUPLING_EXCEEDS_RARE_DECAY_BOUND",
    )


def audit_isospin_operators() -> IsospinOperatorAudit:
    """Check phobic endpoints and the Pb blind spot at fixed D/T attraction."""

    product = _required_dt_product()
    universal = math.sqrt(product / 6.0)
    protophobic_gn = math.sqrt(product / 2.0)
    protophobic_pb = protophobic_gn * math.sqrt(
        LEAD_NEUTRON_NUMBER / LEAD_MASS_NUMBER
    )
    neutron_bound = (
        BROGGINI_NEUTRON_BOUND_COEFFICIENT * REGISTERED_SCALAR_MASS_MEV**2
    )
    neutron_phobic_gp = math.sqrt(product)
    cancellation_ratio = -LEAD_NEUTRON_NUMBER / LEAD_PROTON_NUMBER
    cancellation_product_coefficient = (
        cancellation_ratio + 1.0
    ) * (cancellation_ratio + 2.0)
    return IsospinOperatorAudit(
        required_dt_charge_product=product,
        universal_minimax_coupling=universal,
        protophobic_required_neutron_coupling=protophobic_gn,
        protophobic_lead_effective_coupling=protophobic_pb,
        protophobic_to_neutron_bound_ratio=protophobic_pb / neutron_bound,
        neutron_phobic_required_proton_coupling=neutron_phobic_gp,
        neutron_phobic_kaon_combination_one=NEUTRON_PHOBIC_KAON_COMBINATION_ONE,
        neutron_phobic_kaon_combination_one_bound=(
            DIGITIZED_KAON_COMBINATION_ONE_BOUND
        ),
        neutron_phobic_kaon_combination_one_violation=(
            NEUTRON_PHOBIC_KAON_COMBINATION_ONE
            / DIGITIZED_KAON_COMBINATION_ONE_BOUND
        ),
        neutron_phobic_kaon_combination_two=NEUTRON_PHOBIC_KAON_COMBINATION_TWO,
        neutron_phobic_kaon_combination_two_bound=(
            DIGITIZED_KAON_COMBINATION_TWO_BOUND
        ),
        neutron_phobic_kaon_combination_two_violation=(
            NEUTRON_PHOBIC_KAON_COMBINATION_TWO
            / DIGITIZED_KAON_COMBINATION_TWO_BOUND
        ),
        lead_cancellation_gp_over_gn=cancellation_ratio,
        lead_cancellation_dt_product_coefficient=cancellation_product_coefficient,
        lead_cancellation_makes_dt_attraction=(
            cancellation_product_coefficient > 0.0
        ),
        universal_minimizes_max_abs_nucleon_coupling=True,
        neutron_only_proxy_favors_neutron_phobic_limit=True,
        protophobic_gate_pass=False,
        neutron_phobic_gate_pass=False,
        lead_blind_spot_gate_pass=False,
        status="PHOBIC_ENDPOINTS_FAIL_NEUTRON_OR_KAON_AND_PB_BLIND_SPOT_IS_REPULSIVE",
    )


def _disformal_gain(
    scale_mev: float,
    *,
    energy_points: int,
    wkb_grid_points: int,
) -> float:
    def attraction(radii_fm: np.ndarray) -> np.ndarray:
        natural_radii = radii_fm / HBAR_C_MEV_FM
        return (
            3.0
            * DEUTERON_MASS_MEV
            * TRITON_MASS_MEV
            / (32.0 * math.pi**3 * scale_mev**8 * natural_radii**7)
        )

    return _thermal_response(
        temperature_kev=10.0,
        attraction=attraction,
        energy_points=energy_points,
        wkb_grid_points=wkb_grid_points,
    )[0]


def _solve_disformal_scale() -> float:
    lower = 10.0
    upper = 1_000.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        gain = _disformal_gain(
            midpoint,
            energy_points=181,
            wkb_grid_points=1001,
        )
        # The attraction decreases as M^-8.  Keep the largest scale reaching 1%.
        if gain >= 0.01:
            lower = midpoint
        else:
            upper = midpoint
    return lower


@lru_cache(maxsize=1)
def audit_disformal_operator() -> DisformalOperatorAudit:
    """Propagate the optimistic massless disformal potential to reactivity."""

    required = _solve_disformal_scale()
    grid_specs = ((121, 601), (181, 1001), (361, 4001))
    gains = tuple(
        _disformal_gain(
            required,
            energy_points=energy_points,
            wkb_grid_points=wkb_points,
        )
        for energy_points, wkb_points in grid_specs
    )
    hydrogen_gain = _disformal_gain(
        HYDROGEN_SPECTROSCOPY_DISFORMAL_BOUND_MEV,
        energy_points=181,
        wkb_grid_points=1001,
    )
    stellar_gain = _disformal_gain(
        STELLAR_BURNING_DISFORMAL_BOUND_MEV,
        energy_points=181,
        wkb_grid_points=1001,
    )
    atlas_gain = _disformal_gain(
        ATLAS_DISFORMAL_BOUND_MEV,
        energy_points=181,
        wkb_grid_points=1001,
    )
    return DisformalOperatorAudit(
        potential_coefficient_formula="3*m_D*m_T/(32*pi^3*M^8*r^7)",
        massless_two_scalar_upper_bound=True,
        required_scale_for_one_percent_mev=required,
        coarse_gain_at_required_scale=gains[0],
        default_gain_at_required_scale=gains[1],
        fine_gain_at_required_scale=gains[2],
        maximum_grid_gain_spread=max(gains) - min(gains),
        hydrogen_spectroscopy_bound_mev=HYDROGEN_SPECTROSCOPY_DISFORMAL_BOUND_MEV,
        gain_at_hydrogen_bound=hydrogen_gain,
        hydrogen_bound_derived_for_massless_scalar=True,
        stellar_burning_bound_mev=STELLAR_BURNING_DISFORMAL_BOUND_MEV,
        gain_at_stellar_bound=stellar_gain,
        stellar_bound_derived_for_massless_scalar=True,
        mass_specific_atomic_or_stellar_bound_supplied=False,
        atlas_bound_mev=ATLAS_DISFORMAL_BOUND_MEV,
        gain_at_atlas_bound=atlas_gain,
        atlas_bound_applicable_in_light_mediator_limit=True,
        required_scale_below_applicable_atlas_bound=(
            required < ATLAS_DISFORMAL_BOUND_MEV
        ),
        required_scale_below_all_supplied_bounds=(
            required < HYDROGEN_SPECTROSCOPY_DISFORMAL_BOUND_MEV
        ),
        nonlinear_screening_completion_supplied=False,
        plasma_unscreening_demonstrated=False,
        experimental_constraint_gate_pass=False,
        physical_operator_gate_pass=False,
        status=(
            "MASSLESS_ATOMIC_STELLAR_REFERENCES_NOT_MASS_SPECIFIC_"
            "APPLICABLE_COLLIDER_BOUND_EXCLUDES_OPTIMISTIC_UPPER"
        ),
    )


@lru_cache(maxsize=1)
def current_fusion_operator_alternatives_report() -> FusionOperatorAlternativesReport:
    """Compose the trace, isospin, and disformal alternative ledger."""

    trace = audit_trace_gluon_operator()
    isospin = audit_isospin_operators()
    disformal = audit_disformal_operator()
    any_pass = (
        trace.physical_operator_gate_pass
        or isospin.protophobic_gate_pass
        or isospin.neutron_phobic_gate_pass
        or isospin.lead_blind_spot_gate_pass
        or disformal.physical_operator_gate_pass
    )
    return FusionOperatorAlternativesReport(
        schema_version="1.0",
        trace_gluon=trace,
        isospin=isospin,
        disformal=disformal,
        all_declared_operator_alternatives_audited=True,
        any_alternative_constraint_cleared=any_pass,
        physical_one_percent_ce_branch_derived=False,
        maximum_supported_stage="ALTERNATIVE_OPERATOR_MODEL_CLASS_NO_GO",
        conclusion=(
            "The pure trace/gluon direction exceeds its one-parameter rare-decay bound by about "
            "four thousand.  Proton-phobic couplings fail the neutron proxy, neutron-phobic "
            "couplings fail kaon combinations by four to five orders of magnitude, and the Pb "
            "blind spot reverses the D/T force.  The optimistic massless disformal potential "
            "requires M near 181 MeV.  The quoted spectroscopy and stellar numbers are massless "
            "references rather than 29.65 MeV exclusions, while the applicable light-mediator "
            "collider bound alone remains orders of magnitude above the required scale."
        ),
    )


__all__ = [
    "DisformalOperatorAudit",
    "FusionOperatorAlternativesReport",
    "IsospinOperatorAudit",
    "TraceGluonOperatorAudit",
    "audit_disformal_operator",
    "audit_isospin_operators",
    "audit_trace_gluon_operator",
    "current_fusion_operator_alternatives_report",
]
