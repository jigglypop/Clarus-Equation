"""Fail-closed scalar-current audit for the registered D--T fusion candidate.

This module keeps four logically different quantities separate:

* the flavor-aligned one-nucleon scalar charge used by the existing candidate;
* a reproducible Helm-versus-Gaussian one-body nuclear-shape benchmark;
* nucleon scalar-radius and chiral two-body-current diagnostics; and
* a 2026 deuteron/helium-3 sigma-term proxy which is *not* a triton likelihood.

The public literature does not supply a joint, momentum-dependent D/T response
and covariance for the registered 29.64757 MeV mediator.  Numerical agreement
of the central Helm and Gaussian curves therefore cannot open the physical
gate.  Missing triton data, two-body contact calibration, or covariance always
leaves the report fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
from typing import Any

from .fusion_flavor_aligned_loop import (
    NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV,
    PROTON_SCALAR_SIGMA_NUMERATOR_GEV,
    REGISTERED_SCALAR_MASS_MEV,
    current_fusion_flavor_aligned_report,
)
from .fusion_resonance_loop import HBAR_C_MEV_FM


DEUTERON_MASS_NUMBER = 2
TRITON_MASS_NUMBER = 3
DEUTERON_HELM_DIFFUSENESS_FM = 0.47
DEUTERON_HELM_SKIN_FM = 1.09
TRITON_HELM_DIFFUSENESS_FM = 0.38
TRITON_HELM_SKIN_FM = 0.96
DEUTERON_GAUSSIAN_RMS_FM = 1.975
TRITON_GAUSSIAN_RMS_FM = 1.59
SPACELIKE_Q_GRID_MEV = (0.0, 10.0, 20.0, REGISTERED_SCALAR_MASS_MEV, 40.0)
HELM_GAUSSIAN_RELATIVE_TOLERANCE = 1.0e-4

SIGMA_PI_N_MEV = 43.7
SIGMA_PI_N_STD_MEV = 3.6
SIGMA_STRANGE_MEV = 28.6
SIGMA_STRANGE_STD_MEV = 9.3
LIGHT_SCALAR_RADIUS_SQUARED_MIN_FM2 = 1.34
LIGHT_SCALAR_RADIUS_SQUARED_MAX_FM2 = 1.49
STRANGE_SCALAR_SLOPE_GEV_INV = 0.3
STRANGE_SCALAR_SLOPE_STD_GEV_INV = 0.2

DEUTERON_SIGMA_RATIO = 1.975
DEUTERON_SIGMA_RATIO_STAT_STD = 0.006
DEUTERON_SIGMA_RATIO_SYS_STD = 0.041
HELIUM3_SIGMA_RATIO = 2.929
HELIUM3_SIGMA_RATIO_STAT_STD = 0.029
HELIUM3_SIGMA_RATIO_SYS_STD = 0.126

COUPLING_CORRECTION_COMPARISON_BAND = 0.012
LEGACY_REFERENCE_GAUSSIAN_PRODUCT_RATIO = 0.9803244193608747
LEGACY_MORPHOLOGY_PRODUCT_RATIO_MIN = 0.975860710431114
LEGACY_MORPHOLOGY_PRODUCT_RATIO_MAX = 0.9834526532464857


@dataclass(frozen=True)
class SourceProvenance:
    key: str
    title: str
    arxiv_identifier: str
    pinned_version: str
    url: str
    role: str
    primary_source: bool
    machine_readable_covariance_supplied: bool


@dataclass(frozen=True)
class NucleonScalarChargeAudit:
    source_keys: tuple[str, ...]
    candidate_aligned_scale_gev: float
    candidate_proton_sigma_numerator_gev: float
    candidate_neutron_sigma_numerator_gev: float
    candidate_proton_coupling: float
    candidate_neutron_coupling: float
    candidate_deuteron_charge: float
    candidate_triton_charge: float
    candidate_dt_charge_product: float
    broggini_proton_component_fractions: tuple[float, float, float]
    broggini_proton_component_stds: tuple[float, float, float]
    broggini_neutron_component_fractions: tuple[float, float, float]
    broggini_neutron_component_stds: tuple[float, float, float]
    broggini_proton_quadrature_std: float
    broggini_neutron_quadrature_std: float
    modern_sigma_pi_n_mev: float
    modern_sigma_pi_n_std_mev: float
    modern_sigma_strange_mev: float
    modern_sigma_strange_std_mev: float
    modern_sigma_uds_mev: float
    modern_sigma_uds_quadrature_std_mev: float
    modern_light_fraction_of_uds_central: float
    modern_proton_equals_neutron_isoscalar_proxy_assumed: bool
    modern_to_candidate_isoscalar_numerator_ratio: float
    fixed_scale_dt_product_ratio_diagnostic: float
    retuned_aligned_scale_gev_diagnostic: float
    proton_neutron_sigma_covariance_supplied: bool
    modern_sigma_term_covariance_supplied: bool
    normalization_likelihood_supplied: bool
    normalization_certification_pass: bool
    status: str


@dataclass(frozen=True)
class FormFactorPointAudit:
    momentum_transfer_mev: float
    deuteron_helm_form_factor: float
    triton_helm_form_factor: float
    helm_product: float
    gaussian_product: float
    helm_to_gaussian_relative_residual: float


@dataclass(frozen=True)
class OneBodyNuclearShapeAudit:
    source_key: str
    deuteron_diffraction_radius_fm: float
    triton_diffraction_radius_fm: float
    deuteron_helm_rms_radius_fm: float
    triton_helm_rms_radius_fm: float
    deuteron_gaussian_rms_radius_fm: float
    triton_gaussian_rms_radius_fm: float
    spacelike_points: tuple[FormFactorPointAudit, ...]
    maximum_sampled_spacelike_relative_residual: float
    relative_residual_tolerance: float
    central_spacelike_benchmark_pass: bool
    imaginary_momentum_mev: float
    deuteron_helm_imaginary_form_factor: float
    triton_helm_imaginary_form_factor: float
    helm_imaginary_product: float
    gaussian_imaginary_product: float
    imaginary_helm_to_gaussian_relative_residual: float
    exterior_residue_analytic_diagnostic_pass: bool
    analytic_continuation_is_measurement: bool
    analytic_continuation_is_full_folded_barrier_response: bool
    ab_initio_density_covariance_supplied: bool
    one_body_shape_certification_pass: bool
    status: str


@dataclass(frozen=True)
class BarrierSuppressionPoint:
    radius_fm: float
    point_yukawa_exponential: float


@dataclass(frozen=True)
class BarrierWindowAudit:
    mediator_mass_mev: float
    mediator_compton_length_fm: float
    barrier_radius_min_fm: float
    barrier_radius_max_fm: float
    momentum_grid_min_mev: float
    momentum_grid_max_mev: float
    smallest_spatial_scale_resolved_at_qmax_fm: float
    momentum_needed_for_inner_radius_mev: float
    q_grid_resolves_inner_radius: bool
    suppression_points: tuple[BarrierSuppressionPoint, ...]
    dt_real_space_scalar_current_likelihood_supplied: bool
    status: str


@dataclass(frozen=True)
class ScalarRadiusPointAudit:
    momentum_transfer_mev: float
    correction_at_radius_min: float
    correction_at_radius_max: float
    strange_slope_one_sigma: float
    exact_coupling_correction_at_radius_min: float
    exact_coupling_correction_at_radius_max: float


@dataclass(frozen=True)
class IntrinsicScalarRadiusAudit:
    source_keys: tuple[str, ...]
    light_scalar_radius_squared_min_fm2: float
    light_scalar_radius_squared_max_fm2: float
    strange_scalar_slope_gev_inv: float
    strange_scalar_slope_std_gev_inv: float
    light_fraction_of_uds_central: float
    spacelike_points: tuple[ScalarRadiusPointAudit, ...]
    imaginary_momentum_mev: float
    imaginary_correction_at_radius_min: float
    imaginary_correction_at_radius_max: float
    imaginary_strange_slope_one_sigma: float
    imaginary_exact_coupling_correction_at_radius_min: float
    imaginary_exact_coupling_correction_at_radius_max: float
    q40_coupling_correction_exceeds_comparison_band: bool
    scalar_radius_covariance_supplied: bool
    low_q_expansion_promoted_to_full_form_factor: bool
    scalar_radius_certification_pass: bool
    status: str


@dataclass(frozen=True)
class SigmaTermProxyAssumptions:
    helium3_used_as_triton_isospin_proxy: bool
    deuteron_and_helium3_errors_treated_independent: bool
    sigma_pi_and_sigma_strange_central_dilution_only: bool
    sigma_pi_sigma_strange_uncertainty_propagated: bool
    evaluated_at_zero_momentum_only: bool
    first_order_gaussian_error_propagation: bool
    actual_triton_sigma_term_supplied: bool
    dt_covariance_supplied: bool


@dataclass(frozen=True)
class SigmaTermProxyAudit:
    source_keys: tuple[str, ...]
    assumptions: SigmaTermProxyAssumptions
    deuteron_sigma_ratio: float
    deuteron_sigma_ratio_total_std: float
    helium3_sigma_ratio: float
    helium3_sigma_ratio_total_std: float
    deuteron_light_nonadditivity: float
    deuteron_light_nonadditivity_std: float
    triton_proxy_light_nonadditivity: float
    triton_proxy_light_nonadditivity_std: float
    uds_light_dilution_central: float
    deuteron_uds_nonadditivity: float
    deuteron_uds_nonadditivity_std: float
    triton_proxy_uds_nonadditivity: float
    triton_proxy_uds_nonadditivity_std: float
    dt_product_correction: float
    dt_product_correction_std: float
    required_common_coupling_correction: float
    required_common_coupling_correction_std: float
    required_common_coupling_correction_one_sigma_upper: float
    comparison_band_absolute_coupling_correction: float
    central_correction_exceeds_comparison_band: bool
    one_sigma_upper_exceeds_comparison_band: bool
    diagnostic_valid_for_certification: bool
    status: str


@dataclass(frozen=True)
class TwoBodyScalarCurrentAudit:
    source_keys: tuple[str, ...]
    chiral_two_body_operator_recorded: bool
    andreoli_deuteron_q0_two_body_fraction_min: float
    andreoli_deuteron_q0_two_body_fraction_max: float
    korber_higher_order_deuteron_squared_response_central: float
    korber_higher_order_deuteron_squared_response_std: float
    a3_correction_roughly_smaller_than_deuteron_by_factor: float
    a3_n2lo_relative_uncertainty_order_one: bool
    andreoli_cutoff_min_mev: float
    andreoli_cutoff_max_mev: float
    linearized_modern_uds_deuteron_amplitude_correction_min: float
    linearized_modern_uds_deuteron_amplitude_correction_max: float
    exact_modern_uds_deuteron_amplitude_correction_min: float
    exact_modern_uds_deuteron_amplitude_correction_max: float
    filandri_reference_momentum_mev: float
    filandri_av18_nlo_relative_amplitude: float
    filandri_av18_cumulative_relative_amplitude: float
    filandri_n4lo500_nlo_relative_amplitude: float
    filandri_n4lo500_cumulative_relative_amplitude: float
    filandri_momentum_coverage_max_mev: float
    triton_two_body_sign_stable_across_regulators: bool
    unknown_short_range_two_nucleon_contact_resolved: bool
    regulator_consistent_current_and_potential_supplied: bool
    momentum_dependent_dt_joint_likelihood_supplied: bool
    two_body_covariance_supplied: bool
    two_body_certification_pass: bool
    status: str


@dataclass(frozen=True)
class ScalarCurrentCertificationAudit:
    legacy_reference_gaussian_product_ratio: float
    legacy_reference_gaussian_coupling_correction: float
    legacy_morphology_coupling_correction_min: float
    legacy_morphology_coupling_correction_max: float
    comparison_band_absolute_coupling_correction: float
    comparison_band_is_statistical_confidence_interval: bool
    helm_gaussian_central_benchmark_pass: bool
    sigma_proxy_central_within_comparison_band: bool
    sigma_proxy_one_sigma_upper_within_comparison_band: bool
    scalar_radius_q40_within_comparison_band: bool
    actual_triton_q0_sigma_term_supplied: bool
    momentum_dependent_dt_covariance_supplied: bool
    calibrated_two_body_contact_supplied: bool
    full_real_space_barrier_response_supplied: bool
    all_required_scalar_current_inputs_supplied: bool
    nucleon_normalization_leaf_gate_pass: bool
    one_body_shape_leaf_gate_pass: bool
    scalar_radius_leaf_gate_pass: bool
    triton_sigma_response_leaf_gate_pass: bool
    two_body_leaf_gate_pass: bool
    scalar_current_certification_pass: bool
    upstream_uv_action_gate_pass: bool
    upstream_existing_constraints_gate_pass: bool
    physical_ce_fusion_branch_accepted: bool
    status: str


@dataclass(frozen=True)
class FusionScalarCurrentReport:
    schema_version: str
    sources: tuple[SourceProvenance, ...]
    nucleon_scalar_charge: NucleonScalarChargeAudit
    one_body_nuclear_shape: OneBodyNuclearShapeAudit
    barrier_window: BarrierWindowAudit
    intrinsic_scalar_radius: IntrinsicScalarRadiusAudit
    sigma_term_proxy: SigmaTermProxyAudit
    two_body_scalar_current: TwoBodyScalarCurrentAudit
    certification: ScalarCurrentCertificationAudit
    scalar_current_loop_closed: bool
    physical_ce_fusion_branch_accepted: bool
    next_required_input: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_nonnegative(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive(value: Real, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _mass_number(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("mass_number must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("mass_number must be a positive integer")
    return result


def _helm_geometry(
    *, mass_number: Integral, diffuseness_fm: Real, skin_thickness_fm: Real
) -> tuple[float, float]:
    mass = _mass_number(mass_number)
    diffuseness = _positive(diffuseness_fm, name="diffuseness_fm")
    skin = _positive(skin_thickness_fm, name="skin_thickness_fm")
    half_density_radius = 1.23 * mass ** (1.0 / 3.0) - 0.60
    diffraction_squared = (
        half_density_radius**2 + (7.0 / 3.0) * math.pi**2 * diffuseness**2 - 5.0 * skin**2
    )
    if diffraction_squared <= 0.0:
        raise ValueError("Helm parameters produce a nonpositive diffraction radius squared")
    rms_squared = (3.0 / 5.0) * diffraction_squared + 3.0 * skin**2
    return math.sqrt(diffraction_squared), math.sqrt(rms_squared)


def _helm_sphere_core(argument: float, *, imaginary_momentum: bool) -> float:
    squared = argument * argument
    if abs(argument) < 1.0e-3:
        sign = 1.0 if imaginary_momentum else -1.0
        return 1.0 + sign * squared / 10.0 + squared**2 / 280.0 + sign * squared**3 / 15120.0
    if imaginary_momentum:
        return 3.0 * (argument * math.cosh(argument) - math.sinh(argument)) / argument**3
    return 3.0 * (math.sin(argument) - argument * math.cos(argument)) / argument**3


def helm_form_factor(
    momentum_transfer_mev: Real,
    *,
    mass_number: Integral,
    diffuseness_fm: Real,
    skin_thickness_fm: Real,
    imaginary_momentum: bool = False,
) -> float:
    """Evaluate the normalized Helm form factor at ``q`` or at ``q = i |q|``."""

    momentum = _finite_nonnegative(momentum_transfer_mev, name="momentum_transfer_mev")
    if not isinstance(imaginary_momentum, bool):
        raise ValueError("imaginary_momentum must be boolean")
    diffraction_radius, _ = _helm_geometry(
        mass_number=mass_number,
        diffuseness_fm=diffuseness_fm,
        skin_thickness_fm=skin_thickness_fm,
    )
    if momentum == 0.0:
        return 1.0
    inverse_hbar_c = momentum / HBAR_C_MEV_FM
    sphere_argument = inverse_hbar_c * diffraction_radius
    skin_argument = inverse_hbar_c * float(skin_thickness_fm)
    exponent = skin_argument**2 / 2.0
    if not imaginary_momentum:
        exponent = -exponent
    try:
        result = _helm_sphere_core(
            sphere_argument, imaginary_momentum=imaginary_momentum
        ) * math.exp(exponent)
    except OverflowError as exc:
        raise ValueError("form-factor arguments exceed floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError("form-factor result is not finite")
    return result


def gaussian_product_form_factor(
    momentum_transfer_mev: Real,
    *,
    deuteron_rms_fm: Real = DEUTERON_GAUSSIAN_RMS_FM,
    triton_rms_fm: Real = TRITON_GAUSSIAN_RMS_FM,
    imaginary_momentum: bool = False,
) -> float:
    """Return the product of normalized Gaussian D and T one-body factors."""

    momentum = _finite_nonnegative(momentum_transfer_mev, name="momentum_transfer_mev")
    deuteron_radius = _positive(deuteron_rms_fm, name="deuteron_rms_fm")
    triton_radius = _positive(triton_rms_fm, name="triton_rms_fm")
    if not isinstance(imaginary_momentum, bool):
        raise ValueError("imaginary_momentum must be boolean")
    exponent = momentum**2 * (deuteron_radius**2 + triton_radius**2) / (6.0 * HBAR_C_MEV_FM**2)
    if not imaginary_momentum:
        exponent = -exponent
    try:
        result = math.exp(exponent)
    except OverflowError as exc:
        raise ValueError("Gaussian form-factor arguments exceed floating-point range") from exc
    return result


def _source_provenance() -> tuple[SourceProvenance, ...]:
    return (
        SourceProvenance(
            key="broggini_2025_v2",
            title="Probing a Light Scalar Boson with a few-MeV Proton Beam Deep Underground",
            arxiv_identifier="2509.03486",
            pinned_version="v2",
            url="https://arxiv.org/abs/2509.03486v2",
            role="flavor alignment and proton/neutron u,d,s scalar fractions",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="korber_2017_v1",
            title="First-principle calculations of Dark Matter scattering off light nuclei",
            arxiv_identifier="1704.01150",
            pinned_version="v1",
            url="https://arxiv.org/abs/1704.01150v1",
            role="D/T/He3 Helm fits and chiral one- and two-body scalar responses",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="andreoli_2019_v2",
            title="Quantum Monte Carlo calculations of dark matter scattering off light nuclei",
            arxiv_identifier="1811.01843",
            pinned_version="v2",
            url="https://arxiv.org/abs/1811.01843v2",
            role="cutoff dependence of scalar two-body currents in light nuclei",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="devries_2024_v2",
            title="Dark matter scattering off 4He in chiral effective field theory",
            arxiv_identifier="2310.11343",
            pinned_version="v2",
            url="https://arxiv.org/abs/2310.11343v2",
            role="higher-order regulator dependence and missing short-range contact",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="filandri_2024_v2",
            title="Dark matter scattering off 2H and 4He nuclei within chiral effective field theory",
            arxiv_identifier="2403.06599",
            pinned_version="v2",
            url="https://arxiv.org/abs/2403.06599v2",
            role="low-q deuteron scalar reduced-matrix-element order diagnostics",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="chakraborty_2026_v1",
            title=(
                "Quark-Mass Dependence of Light-Nuclei Masses from Lattice QCD and "
                "Trace-Anomaly Contributions to Nuclear Bindings"
            ),
            arxiv_identifier="2603.28872",
            pinned_version="v1",
            url="https://arxiv.org/abs/2603.28872v1",
            role="zero-momentum deuteron and helium-3 light-quark sigma-term ratios",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="agadjanov_2024_v2",
            title="Nucleon Sigma Terms with Nf=2+1 O(a)-improved Wilson fermions",
            arxiv_identifier="2303.08741",
            pinned_version="v2",
            url="https://arxiv.org/abs/2303.08741v2",
            role="modern pion-nucleon and strange sigma-term normalization",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
        SourceProvenance(
            key="alarcon_weiss_2017_v1",
            title=(
                "Nucleon form factors in dispersively improved Chiral Effective Field "
                "Theory I: Scalar form factor"
            ),
            arxiv_identifier="1707.07682",
            pinned_version="v1",
            url="https://arxiv.org/abs/1707.07682v1",
            role="dispersive light-quark nucleon scalar-radius interval",
            primary_source=True,
            machine_readable_covariance_supplied=False,
        ),
    )


def audit_nucleon_scalar_charge() -> NucleonScalarChargeAudit:
    candidate = current_fusion_flavor_aligned_report().operator
    proton_stds = (0.004, 0.005, 0.062)
    neutron_stds = (0.003, 0.008, 0.062)
    modern_total = SIGMA_PI_N_MEV + SIGMA_STRANGE_MEV
    modern_total_std = math.hypot(SIGMA_PI_N_STD_MEV, SIGMA_STRANGE_STD_MEV)
    candidate_isoscalar_mev = (
        PROTON_SCALAR_SIGMA_NUMERATOR_GEV + NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV
    ) * 500.0
    normalization_ratio = modern_total / candidate_isoscalar_mev
    candidate_deuteron_numerator_mev = (
        PROTON_SCALAR_SIGMA_NUMERATOR_GEV + NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV
    ) * 1000.0
    candidate_triton_numerator_mev = (
        PROTON_SCALAR_SIGMA_NUMERATOR_GEV + 2.0 * NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV
    ) * 1000.0
    modern_deuteron_numerator_mev = 2.0 * modern_total
    modern_triton_numerator_mev = 3.0 * modern_total
    fixed_scale_product_ratio = (modern_deuteron_numerator_mev * modern_triton_numerator_mev) / (
        candidate_deuteron_numerator_mev * candidate_triton_numerator_mev
    )
    return NucleonScalarChargeAudit(
        source_keys=("broggini_2025_v2", "agadjanov_2024_v2"),
        candidate_aligned_scale_gev=candidate.aligned_scale_gev,
        candidate_proton_sigma_numerator_gev=PROTON_SCALAR_SIGMA_NUMERATOR_GEV,
        candidate_neutron_sigma_numerator_gev=NEUTRON_SCALAR_SIGMA_NUMERATOR_GEV,
        candidate_proton_coupling=candidate.proton_coupling,
        candidate_neutron_coupling=candidate.neutron_coupling,
        candidate_deuteron_charge=candidate.deuteron_scalar_charge,
        candidate_triton_charge=candidate.triton_scalar_charge,
        candidate_dt_charge_product=candidate.reproduced_dt_charge_product,
        broggini_proton_component_fractions=(0.020, 0.026, 0.118),
        broggini_proton_component_stds=proton_stds,
        broggini_neutron_component_fractions=(0.014, 0.036, 0.118),
        broggini_neutron_component_stds=neutron_stds,
        broggini_proton_quadrature_std=math.sqrt(sum(value**2 for value in proton_stds)),
        broggini_neutron_quadrature_std=math.sqrt(sum(value**2 for value in neutron_stds)),
        modern_sigma_pi_n_mev=SIGMA_PI_N_MEV,
        modern_sigma_pi_n_std_mev=SIGMA_PI_N_STD_MEV,
        modern_sigma_strange_mev=SIGMA_STRANGE_MEV,
        modern_sigma_strange_std_mev=SIGMA_STRANGE_STD_MEV,
        modern_sigma_uds_mev=modern_total,
        modern_sigma_uds_quadrature_std_mev=modern_total_std,
        modern_light_fraction_of_uds_central=SIGMA_PI_N_MEV / modern_total,
        modern_proton_equals_neutron_isoscalar_proxy_assumed=True,
        modern_to_candidate_isoscalar_numerator_ratio=normalization_ratio,
        fixed_scale_dt_product_ratio_diagnostic=fixed_scale_product_ratio,
        retuned_aligned_scale_gev_diagnostic=(
            candidate.aligned_scale_gev * math.sqrt(fixed_scale_product_ratio)
        ),
        proton_neutron_sigma_covariance_supplied=False,
        modern_sigma_term_covariance_supplied=False,
        normalization_likelihood_supplied=False,
        normalization_certification_pass=False,
        status="CENTRAL_NORMALIZATION_SHIFT_LARGE_BUT_COVARIANCE_AND_RETUNED_CONSTRAINTS_OPEN",
    )


def audit_one_body_nuclear_shape() -> OneBodyNuclearShapeAudit:
    deuteron_diffraction, deuteron_rms = _helm_geometry(
        mass_number=DEUTERON_MASS_NUMBER,
        diffuseness_fm=DEUTERON_HELM_DIFFUSENESS_FM,
        skin_thickness_fm=DEUTERON_HELM_SKIN_FM,
    )
    triton_diffraction, triton_rms = _helm_geometry(
        mass_number=TRITON_MASS_NUMBER,
        diffuseness_fm=TRITON_HELM_DIFFUSENESS_FM,
        skin_thickness_fm=TRITON_HELM_SKIN_FM,
    )
    points: list[FormFactorPointAudit] = []
    for momentum in SPACELIKE_Q_GRID_MEV:
        deuteron = helm_form_factor(
            momentum,
            mass_number=DEUTERON_MASS_NUMBER,
            diffuseness_fm=DEUTERON_HELM_DIFFUSENESS_FM,
            skin_thickness_fm=DEUTERON_HELM_SKIN_FM,
        )
        triton = helm_form_factor(
            momentum,
            mass_number=TRITON_MASS_NUMBER,
            diffuseness_fm=TRITON_HELM_DIFFUSENESS_FM,
            skin_thickness_fm=TRITON_HELM_SKIN_FM,
        )
        helm_product = deuteron * triton
        gaussian_product = gaussian_product_form_factor(momentum)
        points.append(
            FormFactorPointAudit(
                momentum_transfer_mev=momentum,
                deuteron_helm_form_factor=deuteron,
                triton_helm_form_factor=triton,
                helm_product=helm_product,
                gaussian_product=gaussian_product,
                helm_to_gaussian_relative_residual=helm_product / gaussian_product - 1.0,
            )
        )
    maximum_residual = max(abs(point.helm_to_gaussian_relative_residual) for point in points)
    mass = REGISTERED_SCALAR_MASS_MEV
    imaginary_deuteron = helm_form_factor(
        mass,
        mass_number=DEUTERON_MASS_NUMBER,
        diffuseness_fm=DEUTERON_HELM_DIFFUSENESS_FM,
        skin_thickness_fm=DEUTERON_HELM_SKIN_FM,
        imaginary_momentum=True,
    )
    imaginary_triton = helm_form_factor(
        mass,
        mass_number=TRITON_MASS_NUMBER,
        diffuseness_fm=TRITON_HELM_DIFFUSENESS_FM,
        skin_thickness_fm=TRITON_HELM_SKIN_FM,
        imaginary_momentum=True,
    )
    imaginary_helm_product = imaginary_deuteron * imaginary_triton
    imaginary_gaussian_product = gaussian_product_form_factor(mass, imaginary_momentum=True)
    imaginary_residual = imaginary_helm_product / imaginary_gaussian_product - 1.0
    return OneBodyNuclearShapeAudit(
        source_key="korber_2017_v1",
        deuteron_diffraction_radius_fm=deuteron_diffraction,
        triton_diffraction_radius_fm=triton_diffraction,
        deuteron_helm_rms_radius_fm=deuteron_rms,
        triton_helm_rms_radius_fm=triton_rms,
        deuteron_gaussian_rms_radius_fm=DEUTERON_GAUSSIAN_RMS_FM,
        triton_gaussian_rms_radius_fm=TRITON_GAUSSIAN_RMS_FM,
        spacelike_points=tuple(points),
        maximum_sampled_spacelike_relative_residual=maximum_residual,
        relative_residual_tolerance=HELM_GAUSSIAN_RELATIVE_TOLERANCE,
        central_spacelike_benchmark_pass=(maximum_residual <= HELM_GAUSSIAN_RELATIVE_TOLERANCE),
        imaginary_momentum_mev=mass,
        deuteron_helm_imaginary_form_factor=imaginary_deuteron,
        triton_helm_imaginary_form_factor=imaginary_triton,
        helm_imaginary_product=imaginary_helm_product,
        gaussian_imaginary_product=imaginary_gaussian_product,
        imaginary_helm_to_gaussian_relative_residual=imaginary_residual,
        exterior_residue_analytic_diagnostic_pass=(
            abs(imaginary_residual) <= HELM_GAUSSIAN_RELATIVE_TOLERANCE
        ),
        analytic_continuation_is_measurement=False,
        analytic_continuation_is_full_folded_barrier_response=False,
        ab_initio_density_covariance_supplied=False,
        one_body_shape_certification_pass=False,
        status="CENTRAL_HELM_GAUSSIAN_MATCH_ONLY_NO_DENSITY_COVARIANCE",
    )


def audit_barrier_window() -> BarrierWindowAudit:
    mass = REGISTERED_SCALAR_MASS_MEV
    radii = (3.24, 5.0, 10.0, 20.0, 50.0)
    return BarrierWindowAudit(
        mediator_mass_mev=mass,
        mediator_compton_length_fm=HBAR_C_MEV_FM / mass,
        barrier_radius_min_fm=radii[0],
        barrier_radius_max_fm=radii[-1],
        momentum_grid_min_mev=SPACELIKE_Q_GRID_MEV[0],
        momentum_grid_max_mev=SPACELIKE_Q_GRID_MEV[-1],
        smallest_spatial_scale_resolved_at_qmax_fm=(HBAR_C_MEV_FM / SPACELIKE_Q_GRID_MEV[-1]),
        momentum_needed_for_inner_radius_mev=HBAR_C_MEV_FM / radii[0],
        q_grid_resolves_inner_radius=False,
        suppression_points=tuple(
            BarrierSuppressionPoint(
                radius_fm=radius,
                point_yukawa_exponential=math.exp(-mass * radius / HBAR_C_MEV_FM),
            )
            for radius in radii
        ),
        dt_real_space_scalar_current_likelihood_supplied=False,
        status="Q_LE_40_MEV_DOES_NOT_RESOLVE_3P24_FM_INNER_EDGE",
    )


def _intrinsic_radius_correction(
    momentum_mev: float, *, radius_squared_fm2: float, imaginary_momentum: bool
) -> float:
    light_weight = SIGMA_PI_N_MEV / (SIGMA_PI_N_MEV + SIGMA_STRANGE_MEV)
    light = momentum_mev**2 * radius_squared_fm2 / (6.0 * HBAR_C_MEV_FM**2)
    strange = (
        STRANGE_SCALAR_SLOPE_GEV_INV * (momentum_mev / 1000.0) ** 2 / (SIGMA_STRANGE_MEV / 1000.0)
    )
    sign = 1.0 if imaginary_momentum else -1.0
    return sign * (light_weight * light + (1.0 - light_weight) * strange)


def _strange_slope_uncertainty(momentum_mev: float) -> float:
    light_weight = SIGMA_PI_N_MEV / (SIGMA_PI_N_MEV + SIGMA_STRANGE_MEV)
    return (
        (1.0 - light_weight)
        * STRANGE_SCALAR_SLOPE_STD_GEV_INV
        * (momentum_mev / 1000.0) ** 2
        / (SIGMA_STRANGE_MEV / 1000.0)
    )


def audit_intrinsic_scalar_radius() -> IntrinsicScalarRadiusAudit:
    points: list[ScalarRadiusPointAudit] = []
    for momentum in SPACELIKE_Q_GRID_MEV[1:]:
        correction_min = _intrinsic_radius_correction(
            momentum,
            radius_squared_fm2=LIGHT_SCALAR_RADIUS_SQUARED_MIN_FM2,
            imaginary_momentum=False,
        )
        correction_max = _intrinsic_radius_correction(
            momentum,
            radius_squared_fm2=LIGHT_SCALAR_RADIUS_SQUARED_MAX_FM2,
            imaginary_momentum=False,
        )
        points.append(
            ScalarRadiusPointAudit(
                momentum_transfer_mev=momentum,
                correction_at_radius_min=correction_min,
                correction_at_radius_max=correction_max,
                strange_slope_one_sigma=_strange_slope_uncertainty(momentum),
                exact_coupling_correction_at_radius_min=1.0 / (1.0 + correction_min) - 1.0,
                exact_coupling_correction_at_radius_max=1.0 / (1.0 + correction_max) - 1.0,
            )
        )
    mass = REGISTERED_SCALAR_MASS_MEV
    imaginary_min = _intrinsic_radius_correction(
        mass,
        radius_squared_fm2=LIGHT_SCALAR_RADIUS_SQUARED_MIN_FM2,
        imaginary_momentum=True,
    )
    imaginary_max = _intrinsic_radius_correction(
        mass,
        radius_squared_fm2=LIGHT_SCALAR_RADIUS_SQUARED_MAX_FM2,
        imaginary_momentum=True,
    )
    q40 = points[-1]
    return IntrinsicScalarRadiusAudit(
        source_keys=(
            "alarcon_weiss_2017_v1",
            "korber_2017_v1",
            "agadjanov_2024_v2",
        ),
        light_scalar_radius_squared_min_fm2=LIGHT_SCALAR_RADIUS_SQUARED_MIN_FM2,
        light_scalar_radius_squared_max_fm2=LIGHT_SCALAR_RADIUS_SQUARED_MAX_FM2,
        strange_scalar_slope_gev_inv=STRANGE_SCALAR_SLOPE_GEV_INV,
        strange_scalar_slope_std_gev_inv=STRANGE_SCALAR_SLOPE_STD_GEV_INV,
        light_fraction_of_uds_central=SIGMA_PI_N_MEV / (SIGMA_PI_N_MEV + SIGMA_STRANGE_MEV),
        spacelike_points=tuple(points),
        imaginary_momentum_mev=mass,
        imaginary_correction_at_radius_min=imaginary_min,
        imaginary_correction_at_radius_max=imaginary_max,
        imaginary_strange_slope_one_sigma=_strange_slope_uncertainty(mass),
        imaginary_exact_coupling_correction_at_radius_min=1.0 / (1.0 + imaginary_min) - 1.0,
        imaginary_exact_coupling_correction_at_radius_max=1.0 / (1.0 + imaginary_max) - 1.0,
        q40_coupling_correction_exceeds_comparison_band=(
            max(
                abs(q40.exact_coupling_correction_at_radius_min),
                abs(q40.exact_coupling_correction_at_radius_max),
            )
            > COUPLING_CORRECTION_COMPARISON_BAND
        ),
        scalar_radius_covariance_supplied=False,
        low_q_expansion_promoted_to_full_form_factor=False,
        scalar_radius_certification_pass=False,
        status="Q40_LOW_Q_RADIUS_DIAGNOSTIC_CROSSES_BAND_BUT_NO_JOINT_FORM_FACTOR_LIKELIHOOD",
    )


def audit_sigma_term_proxy() -> SigmaTermProxyAudit:
    deuteron_total_std = math.hypot(DEUTERON_SIGMA_RATIO_STAT_STD, DEUTERON_SIGMA_RATIO_SYS_STD)
    helium3_total_std = math.hypot(HELIUM3_SIGMA_RATIO_STAT_STD, HELIUM3_SIGMA_RATIO_SYS_STD)
    deuteron_light = DEUTERON_SIGMA_RATIO / 2.0 - 1.0
    triton_proxy_light = HELIUM3_SIGMA_RATIO / 3.0 - 1.0
    deuteron_light_std = deuteron_total_std / 2.0
    triton_proxy_light_std = helium3_total_std / 3.0
    dilution = SIGMA_PI_N_MEV / (SIGMA_PI_N_MEV + SIGMA_STRANGE_MEV)
    deuteron_uds = dilution * deuteron_light
    triton_proxy_uds = dilution * triton_proxy_light
    deuteron_uds_std = dilution * deuteron_light_std
    triton_proxy_uds_std = dilution * triton_proxy_light_std
    product_correction = (1.0 + deuteron_uds) * (1.0 + triton_proxy_uds) - 1.0
    product_std = math.hypot(
        (1.0 + triton_proxy_uds) * deuteron_uds_std,
        (1.0 + deuteron_uds) * triton_proxy_uds_std,
    )
    coupling_correction = (1.0 + product_correction) ** -0.5 - 1.0
    coupling_std = 0.5 * product_std / (1.0 + product_correction) ** 1.5
    assumptions = SigmaTermProxyAssumptions(
        helium3_used_as_triton_isospin_proxy=True,
        deuteron_and_helium3_errors_treated_independent=True,
        sigma_pi_and_sigma_strange_central_dilution_only=True,
        sigma_pi_sigma_strange_uncertainty_propagated=False,
        evaluated_at_zero_momentum_only=True,
        first_order_gaussian_error_propagation=True,
        actual_triton_sigma_term_supplied=False,
        dt_covariance_supplied=False,
    )
    return SigmaTermProxyAudit(
        source_keys=("chakraborty_2026_v1", "agadjanov_2024_v2"),
        assumptions=assumptions,
        deuteron_sigma_ratio=DEUTERON_SIGMA_RATIO,
        deuteron_sigma_ratio_total_std=deuteron_total_std,
        helium3_sigma_ratio=HELIUM3_SIGMA_RATIO,
        helium3_sigma_ratio_total_std=helium3_total_std,
        deuteron_light_nonadditivity=deuteron_light,
        deuteron_light_nonadditivity_std=deuteron_light_std,
        triton_proxy_light_nonadditivity=triton_proxy_light,
        triton_proxy_light_nonadditivity_std=triton_proxy_light_std,
        uds_light_dilution_central=dilution,
        deuteron_uds_nonadditivity=deuteron_uds,
        deuteron_uds_nonadditivity_std=deuteron_uds_std,
        triton_proxy_uds_nonadditivity=triton_proxy_uds,
        triton_proxy_uds_nonadditivity_std=triton_proxy_uds_std,
        dt_product_correction=product_correction,
        dt_product_correction_std=product_std,
        required_common_coupling_correction=coupling_correction,
        required_common_coupling_correction_std=coupling_std,
        required_common_coupling_correction_one_sigma_upper=(coupling_correction + coupling_std),
        comparison_band_absolute_coupling_correction=COUPLING_CORRECTION_COMPARISON_BAND,
        central_correction_exceeds_comparison_band=(
            abs(coupling_correction) > COUPLING_CORRECTION_COMPARISON_BAND
        ),
        one_sigma_upper_exceeds_comparison_band=(
            coupling_correction + coupling_std > COUPLING_CORRECTION_COMPARISON_BAND
        ),
        diagnostic_valid_for_certification=False,
        status="PLUS_1P11_PM_1P48_PERCENT_DIAGNOSTIC_ONLY_NO_TRITON_OR_DT_COVARIANCE",
    )


def audit_two_body_scalar_current() -> TwoBodyScalarCurrentAudit:
    dilution = SIGMA_PI_N_MEV / (SIGMA_PI_N_MEV + SIGMA_STRANGE_MEV)
    av18_nlo = -0.218 / -14.4
    av18_cumulative = (-0.218 + 0.153) / -14.4
    n4lo_nlo = -0.0806 / -14.4
    n4lo_cumulative = (-0.0806 + 0.103) / -14.4
    return TwoBodyScalarCurrentAudit(
        source_keys=(
            "korber_2017_v1",
            "andreoli_2019_v2",
            "devries_2024_v2",
            "filandri_2024_v2",
        ),
        chiral_two_body_operator_recorded=True,
        andreoli_deuteron_q0_two_body_fraction_min=0.007,
        andreoli_deuteron_q0_two_body_fraction_max=0.030,
        korber_higher_order_deuteron_squared_response_central=0.016,
        korber_higher_order_deuteron_squared_response_std=0.008,
        a3_correction_roughly_smaller_than_deuteron_by_factor=5.0,
        a3_n2lo_relative_uncertainty_order_one=True,
        andreoli_cutoff_min_mev=500.0,
        andreoli_cutoff_max_mev=10_000.0,
        linearized_modern_uds_deuteron_amplitude_correction_min=(dilution * 0.007 / 2.0),
        linearized_modern_uds_deuteron_amplitude_correction_max=(dilution * 0.030 / 2.0),
        exact_modern_uds_deuteron_amplitude_correction_min=(
            dilution * ((1.0 - 0.007) ** -0.5 - 1.0)
        ),
        exact_modern_uds_deuteron_amplitude_correction_max=(
            dilution * ((1.0 - 0.030) ** -0.5 - 1.0)
        ),
        filandri_reference_momentum_mev=0.05 * HBAR_C_MEV_FM,
        filandri_av18_nlo_relative_amplitude=av18_nlo,
        filandri_av18_cumulative_relative_amplitude=av18_cumulative,
        filandri_n4lo500_nlo_relative_amplitude=n4lo_nlo,
        filandri_n4lo500_cumulative_relative_amplitude=n4lo_cumulative,
        filandri_momentum_coverage_max_mev=0.20 * HBAR_C_MEV_FM,
        triton_two_body_sign_stable_across_regulators=False,
        unknown_short_range_two_nucleon_contact_resolved=False,
        regulator_consistent_current_and_potential_supplied=False,
        momentum_dependent_dt_joint_likelihood_supplied=False,
        two_body_covariance_supplied=False,
        two_body_certification_pass=False,
        status="PERCENT_SCALE_CENTRALS_EXIST_BUT_REGULATOR_CONTACT_AND_DT_LIKELIHOOD_OPEN",
    )


def _certification(
    *,
    nucleon: NucleonScalarChargeAudit,
    shape: OneBodyNuclearShapeAudit,
    barrier: BarrierWindowAudit,
    radius: IntrinsicScalarRadiusAudit,
    proxy: SigmaTermProxyAudit,
    two_body: TwoBodyScalarCurrentAudit,
    upstream_uv_action_gate_pass: bool,
    upstream_existing_constraints_gate_pass: bool,
) -> ScalarCurrentCertificationAudit:
    legacy_reference_coupling = math.sqrt(LEGACY_REFERENCE_GAUSSIAN_PRODUCT_RATIO) - 1.0
    morphology_min = math.sqrt(LEGACY_MORPHOLOGY_PRODUCT_RATIO_MIN) - 1.0
    morphology_max = math.sqrt(LEGACY_MORPHOLOGY_PRODUCT_RATIO_MAX) - 1.0
    actual_triton_supplied = proxy.assumptions.actual_triton_sigma_term_supplied
    dt_covariance_supplied = (
        proxy.assumptions.dt_covariance_supplied
        and two_body.momentum_dependent_dt_joint_likelihood_supplied
        and two_body.two_body_covariance_supplied
    )
    calibrated_contact_supplied = two_body.unknown_short_range_two_nucleon_contact_resolved
    real_space_response_supplied = barrier.dt_real_space_scalar_current_likelihood_supplied
    nucleon_leaf_pass = all(
        (
            not nucleon.modern_proton_equals_neutron_isoscalar_proxy_assumed,
            nucleon.proton_neutron_sigma_covariance_supplied,
            nucleon.modern_sigma_term_covariance_supplied,
            nucleon.normalization_likelihood_supplied,
            nucleon.normalization_certification_pass,
        )
    )
    shape_leaf_pass = all(
        (
            not shape.analytic_continuation_is_measurement,
            not shape.analytic_continuation_is_full_folded_barrier_response,
            shape.ab_initio_density_covariance_supplied,
            shape.one_body_shape_certification_pass,
        )
    )
    radius_leaf_pass = all(
        (
            radius.scalar_radius_covariance_supplied,
            radius.low_q_expansion_promoted_to_full_form_factor,
            radius.scalar_radius_certification_pass,
        )
    )
    proxy_leaf_pass = all(
        (
            not proxy.assumptions.helium3_used_as_triton_isospin_proxy,
            not proxy.assumptions.sigma_pi_and_sigma_strange_central_dilution_only,
            proxy.assumptions.sigma_pi_sigma_strange_uncertainty_propagated,
            not proxy.assumptions.evaluated_at_zero_momentum_only,
            not proxy.assumptions.first_order_gaussian_error_propagation,
            actual_triton_supplied,
            proxy.assumptions.dt_covariance_supplied,
            proxy.diagnostic_valid_for_certification,
        )
    )
    two_body_leaf_pass = all(
        (
            two_body.chiral_two_body_operator_recorded,
            two_body.triton_two_body_sign_stable_across_regulators,
            calibrated_contact_supplied,
            two_body.regulator_consistent_current_and_potential_supplied,
            two_body.momentum_dependent_dt_joint_likelihood_supplied,
            two_body.two_body_covariance_supplied,
            two_body.two_body_certification_pass,
        )
    )
    required_inputs = all(
        (
            actual_triton_supplied,
            dt_covariance_supplied,
            calibrated_contact_supplied,
            real_space_response_supplied,
        )
    )
    certification_pass = all(
        (
            required_inputs,
            nucleon_leaf_pass,
            shape_leaf_pass,
            radius_leaf_pass,
            proxy_leaf_pass,
            two_body_leaf_pass,
        )
    )
    physical_branch_pass = all(
        (
            certification_pass,
            upstream_uv_action_gate_pass,
            upstream_existing_constraints_gate_pass,
        )
    )
    if physical_branch_pass:
        status = "PHYSICAL_SCALAR_CURRENT_AND_UPSTREAM_GATES_PASS"
    elif certification_pass:
        status = "SCALAR_CURRENT_CERTIFIED_BUT_UPSTREAM_UV_OR_CONSTRAINT_GATE_OPEN"
    else:
        status = "FAIL_CLOSED_MISSING_TRITON_DT_COVARIANCE_TWO_BODY_CONTACT_AND_REAL_SPACE_RESPONSE"
    return ScalarCurrentCertificationAudit(
        legacy_reference_gaussian_product_ratio=LEGACY_REFERENCE_GAUSSIAN_PRODUCT_RATIO,
        legacy_reference_gaussian_coupling_correction=legacy_reference_coupling,
        legacy_morphology_coupling_correction_min=morphology_min,
        legacy_morphology_coupling_correction_max=morphology_max,
        comparison_band_absolute_coupling_correction=COUPLING_CORRECTION_COMPARISON_BAND,
        comparison_band_is_statistical_confidence_interval=False,
        helm_gaussian_central_benchmark_pass=(
            shape.central_spacelike_benchmark_pass
            and shape.exterior_residue_analytic_diagnostic_pass
        ),
        sigma_proxy_central_within_comparison_band=(
            not proxy.central_correction_exceeds_comparison_band
        ),
        sigma_proxy_one_sigma_upper_within_comparison_band=(
            not proxy.one_sigma_upper_exceeds_comparison_band
        ),
        scalar_radius_q40_within_comparison_band=(
            not radius.q40_coupling_correction_exceeds_comparison_band
        ),
        actual_triton_q0_sigma_term_supplied=actual_triton_supplied,
        momentum_dependent_dt_covariance_supplied=dt_covariance_supplied,
        calibrated_two_body_contact_supplied=calibrated_contact_supplied,
        full_real_space_barrier_response_supplied=real_space_response_supplied,
        all_required_scalar_current_inputs_supplied=required_inputs,
        nucleon_normalization_leaf_gate_pass=nucleon_leaf_pass,
        one_body_shape_leaf_gate_pass=shape_leaf_pass,
        scalar_radius_leaf_gate_pass=radius_leaf_pass,
        triton_sigma_response_leaf_gate_pass=proxy_leaf_pass,
        two_body_leaf_gate_pass=two_body_leaf_pass,
        scalar_current_certification_pass=certification_pass,
        upstream_uv_action_gate_pass=upstream_uv_action_gate_pass,
        upstream_existing_constraints_gate_pass=upstream_existing_constraints_gate_pass,
        physical_ce_fusion_branch_accepted=physical_branch_pass,
        status=status,
    )


@lru_cache(maxsize=1)
def current_fusion_scalar_current_report() -> FusionScalarCurrentReport:
    """Build the complete scalar-current ledger and retain fail-closed status."""

    upstream = current_fusion_flavor_aligned_report()
    nucleon = audit_nucleon_scalar_charge()
    shape = audit_one_body_nuclear_shape()
    barrier = audit_barrier_window()
    radius = audit_intrinsic_scalar_radius()
    proxy = audit_sigma_term_proxy()
    two_body = audit_two_body_scalar_current()
    certification = _certification(
        nucleon=nucleon,
        shape=shape,
        barrier=barrier,
        radius=radius,
        proxy=proxy,
        two_body=two_body,
        upstream_uv_action_gate_pass=upstream.operator.uv_action_gate_pass,
        upstream_existing_constraints_gate_pass=upstream.all_existing_constraint_gates_pass,
    )
    return FusionScalarCurrentReport(
        schema_version="fusion-scalar-current-v2",
        sources=_source_provenance(),
        nucleon_scalar_charge=nucleon,
        one_body_nuclear_shape=shape,
        barrier_window=barrier,
        intrinsic_scalar_radius=radius,
        sigma_term_proxy=proxy,
        two_body_scalar_current=two_body,
        certification=certification,
        scalar_current_loop_closed=certification.scalar_current_certification_pass,
        physical_ce_fusion_branch_accepted=(certification.physical_ce_fusion_branch_accepted),
        next_required_input=(
            "provide a registered-mass, q=0--40 MeV ab-initio D and T one-plus-two-body "
            "scalar response with shared regulator, fitted short-range contact, full D/T "
            "covariance, and its r=3.24--50 fm barrier likelihood; replace the He3 proxy "
            "with an actual triton sigma term"
        ),
        conclusion=(
            "The normalization shift is a conditional p=n isoscalar proxy, not a fitted p/n "
            "likelihood. Helm and Gaussian central one-body products agree to 9e-5 on the registered "
            "five-point q grid; the q=i*m comparison is an exterior-residue analytic "
            "diagnostic, not a full folded barrier response. The modern sigma-term proxy is "
            "+1.11% +/- 1.48%, the q=40 MeV scalar-radius diagnostic crosses the 1.2% "
            "comparison band, and no joint D/T scalar-current likelihood exists in the cited "
            "inputs. The physical fusion branch remains rejected."
        ),
    )
