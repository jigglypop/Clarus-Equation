"""Margin-robustness ledger for the flavor-aligned fusion candidate.

This module deliberately separates three different levels of information:

* a reproducible one-body Gaussian folding proxy for the D/T Yukawa potential;
* a finite-propagator and elastic-Pb-form-factor shape proxy for the historical
  neutron--Pb angular-distribution constraint;
* algebraic threshold surfaces for a possible rare-kaon NLO tightening.

None of the proxies is promoted to an experimental likelihood.  In particular,
the historical neutron covariance/strong phase, complete nuclear scalar
currents, and the full weak-ChPT kaon amplitude remain absent.  The physical
gate therefore stays closed even when a central proxy cell lies on the allowed
side of both inequalities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Real
from typing import Any, Callable

import numpy as np

from .fusion_equation_iteration_loop import _thermal_response
from .fusion_flavor_aligned_loop import current_fusion_flavor_aligned_report
from .fusion_resonance_loop import HBAR_C_MEV_FM


TARGET_GAIN_MINUS_ONE = 0.01
PROXY_ENERGY_POINTS = 121
PROXY_WKB_GRID_POINTS = 601
NEUTRON_KINETIC_ENERGY_KEV = 25.0
NEUTRON_MASS_MEV = 939.5654205
PB_MIN_SCATTERING_ANGLE_DEG = 30.0
PB_MAX_SCATTERING_ANGLE_DEG = 150.0
PB_ANGLE_GRID_POINTS = 241
PB_LOW_ENERGY_GRID_POINTS = 181
PB_REFINED_ANGLE_GRID_POINTS = 1001
PB_REFINED_LOW_ENERGY_GRID_POINTS = 1001
PB_PROXY_CONVERGENCE_RELATIVE_TOLERANCE = 1.0e-4
PB_RMS_RADIUS_MIN_FM = 5.4
PB_RMS_RADIUS_MAX_FM = 5.75
LEAD_MASS_NUMBER = 208.0
LINEAR_RESPONSE_PROBE_FRACTION = 1.0e-4
UNIFORM_FOLDING_QUADRATURE_POINTS = 96
LATEST_NA62_BR_IMPROVEMENT_MINIMUM = 1.0
LATEST_NA62_BR_IMPROVEMENT_MAXIMUM = 3.0
LATEST_NA62_CONFIDENCE_LEVEL_PERCENT = 90.0
LATEST_NA62_LOW_MASS_RANGE_MAX_MEV = 110.0
LATEST_NA62_MASS_HYPOTHESIS_SPACING_MEV = 1.4
LATEST_NA62_FIGURE2_PDF_PAGE_INDEX = 4
LATEST_NA62_FIGURE2_AXIS_X_MIN_PT = 105.5185317993164
LATEST_NA62_FIGURE2_AXIS_X_MAX_PT = 292.8013916015625
LATEST_NA62_FIGURE2_AXIS_Y_TOP_PT = 363.77630615234375
LATEST_NA62_FIGURE2_AXIS_Y_BOTTOM_PT = 503.3321838378906
LATEST_NA62_FIGURE2_AXIS_X_MAX_MEV = 260.0
LATEST_NA62_FIGURE2_Y_SCALE = 1.0e-10
LATEST_NA62_FIGURE2_NEW_SEGMENT_PT = (
    (126.69586181640625, 468.6386413574219),
    (127.70443725585938, 469.406005859375),
)
LATEST_NA62_FIGURE2_OLD_SEGMENT_PT = (
    (119.92498779296875, 448.092529296875),
    (127.12799072265625, 457.6601257324219),
)
LATEST_NA62_FIGURE2_RELATIVE_READOUT_UNCERTAINTY = 0.05

# These radii are deliberately a model span, not an empirical confidence
# interval.  The reference values are close to commonly used point-nucleon
# sizes; the compact/broad values expose the sensitivity of the result.
DT_GAUSSIAN_RMS_SCENARIOS_FM = (
    ("COMPACT_ONE_BODY_PROXY", 1.50, 1.30),
    ("REFERENCE_ONE_BODY_PROXY", 1.975, 1.59),
    ("MAXIMUM_FOLDING_PROXY", 2.20, 1.90),
    ("BROAD_ONE_BODY_PROXY", 2.40, 2.10),
)

DT_MORPHOLOGY_RMS_SCENARIOS_FM = (
    ("POINT_STRUCTURE_RADII", 1.97507, 1.5978),
    ("CONSERVATIVE_RADIUS_ENVELOPE", 2.12799, 1.7591),
)


@dataclass(frozen=True)
class DTFoldingAudit:
    scenario: str
    deuteron_rms_radius_fm: float
    triton_rms_radius_fm: float
    gaussian_relative_width_fm: float
    folding_ratio_at_nuclear_radius: float
    asymptotic_imaginary_momentum_form_factor: float
    required_dt_product: float
    required_product_to_point_ratio: float
    required_coupling_to_point_ratio: float
    one_body_gaussian_folding_computed: bool
    two_body_scalar_currents_supplied: bool
    ab_initio_nuclear_density_covariance_supplied: bool
    status: str


@dataclass(frozen=True)
class DTMorphologyAudit:
    radius_scenario: str
    density_morphology: str
    deuteron_rms_radius_fm: float
    triton_rms_radius_fm: float
    folding_ratio_at_nuclear_radius: float
    asymptotic_imaginary_momentum_form_factor: float
    linear_response_enhancement_factor: float
    linearized_required_product_to_point_ratio: float
    linearized_required_coupling_to_point_ratio: float
    full_one_percent_resolve_performed: bool
    two_body_scalar_currents_supplied: bool
    status: str


@dataclass(frozen=True)
class LeadShapeProxyAudit:
    neutron_kinetic_energy_kev: float
    minimum_scattering_angle_deg: float
    maximum_scattering_angle_deg: float
    minimum_momentum_transfer_mev: float
    maximum_momentum_transfer_mev: float
    finite_propagator_response_minimum: float
    finite_propagator_response_maximum: float
    pb_rms_radius_min_fm: float
    pb_rms_radius_max_fm: float
    combined_shape_response_minimum: float
    combined_shape_response_maximum: float
    q4_weighted_shape_response_minimum: float
    q4_weighted_shape_response_maximum: float
    corresponding_bound_multiplier_minimum: float
    corresponding_bound_multiplier_maximum: float
    angular_p_wave_point_propagator_projection_response_minimum: float
    angular_p_wave_projection_response_minimum: float
    angular_p_wave_projection_response_maximum: float
    angular_p_wave_bound_multiplier_minimum: float
    angular_p_wave_bound_multiplier_maximum: float
    low_energy_sigma2_zero_energy_response_minimum: float
    low_energy_sigma2_zero_energy_response_maximum: float
    low_energy_sigma2_finite_window_response_minimum: float
    low_energy_sigma2_finite_window_response_maximum: float
    low_energy_sigma2_bound_multiplier_minimum: float
    low_energy_sigma2_bound_multiplier_maximum: float
    low_energy_sigma2_refined_energy_grid_points: int
    low_energy_sigma2_refined_angle_grid_points: int
    low_energy_sigma2_grid_refinement_max_relative_shift: float
    low_energy_sigma2_numerical_convergence_tolerance: float
    low_energy_sigma2_numerical_convergence_pass: bool
    gaussian_and_uniform_sphere_envelope_used: bool
    exponential_form_factor_in_projection_used: bool
    angular_theta_and_x_weightings_used: bool
    low_energy_linear_and_log_weightings_used: bool
    alternative_recasts_land_on_opposite_sides_of_contact_limit: bool
    experimental_angular_covariance_supplied: bool
    strong_amplitude_phase_profiled: bool
    source_analysis_finite_density_treatment_known: bool
    status: str


@dataclass(frozen=True)
class MarginCellAudit:
    dt_product_to_point_ratio: float
    pb_shape_response_multiplier: float
    kaon_nlo_tightening_factor: float
    kaon_digitized_bound_factor: float
    latest_kaon_br_improvement_factor: float
    latest_kaon_coupling_bound_multiplier: float
    corrected_pb_candidate_coupling: float
    corrected_pb_proxy_bound: float
    pb_response_critical: float
    neutron_proxy_condition_satisfied: bool
    corrected_kaon_candidate_coordinate: float
    corrected_kaon_proxy_bound: float
    kaon_nlo_tightening_critical: float
    kaon_proxy_condition_satisfied: bool
    joint_proxy_conditions_satisfied: bool
    experimental_likelihoods_supplied: bool
    physical_gate_pass: bool
    status: str


@dataclass(frozen=True)
class LatestKaonDataAudit:
    source_arxiv_identifier: str
    dataset_years: str
    confidence_level_percent: float
    low_mass_scan_minimum_mev: float
    low_mass_scan_maximum_mev: float
    mass_hypothesis_spacing_mev: float
    candidate_mass_mev: float
    candidate_inside_low_mass_scan: bool
    branching_ratio_improvement_factor_minimum: float
    branching_ratio_improvement_factor_maximum: float
    coupling_bound_multiplier_minimum: float
    coupling_bound_multiplier_maximum: float
    previous_digitized_coupling_bound: float
    latest_range_coupling_bound_minimum: float
    latest_range_coupling_bound_maximum: float
    point_nlo_tightening_critical_minimum: float
    point_nlo_tightening_critical_maximum: float
    favorable_proxy_nlo_tightening_critical_minimum: float
    favorable_proxy_nlo_tightening_critical_maximum: float
    tree_level_candidate_allowed_across_improvement_range: bool
    figure2_pdf_page_index: int
    figure2_candidate_mass_curve_interpolation_entered: bool
    figure2_relative_readout_uncertainty: float
    figure2_readout_errors_treated_as_independent_box: bool
    figure2_interpolated_2016_2022_observed_br_limit: float
    figure2_interpolated_2016_2018_observed_br_limit: float
    figure2_interpolated_br_improvement_factor: float
    figure2_br_improvement_factor_minimum: float
    figure2_br_improvement_factor_maximum: float
    figure2_interpolated_coupling_bound_multiplier: float
    figure2_coupling_bound_multiplier_minimum: float
    figure2_coupling_bound_multiplier_maximum: float
    figure2_interpolated_latest_uds_bound: float
    figure2_latest_uds_bound_minimum: float
    figure2_latest_uds_bound_maximum: float
    figure2_interpolated_point_nlo_tightening_critical: float
    figure2_point_nlo_tightening_critical_minimum: float
    figure2_point_nlo_tightening_critical_maximum: float
    figure2_interpolated_favorable_nlo_tightening_critical: float
    figure2_favorable_nlo_tightening_critical_minimum: float
    figure2_favorable_nlo_tightening_critical_maximum: float
    exact_candidate_mass_observed_limit_entered: bool
    full_uds_operator_recast_and_nlo_likelihood_supplied: bool
    latest_data_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionFlavorMarginRobustnessReport:
    schema_version: str
    point_required_dt_product: float
    folding_scenarios: tuple[DTFoldingAudit, ...]
    morphology_scenarios: tuple[DTMorphologyAudit, ...]
    minimum_required_product_to_point_ratio: float
    maximum_required_product_to_point_ratio: float
    minimum_linearized_morphology_product_to_point_ratio: float
    maximum_linearized_morphology_product_to_point_ratio: float
    most_favorable_proxy_product_to_point_ratio: float
    lead_shape_proxy: LeadShapeProxyAudit
    latest_kaon_data: LatestKaonDataAudit
    point_pb_response_critical: float
    most_favorable_proxy_pb_response_critical: float
    point_kaon_nlo_tightening_critical: float
    most_favorable_proxy_kaon_nlo_tightening_critical: float
    robust_lower_line_kaon_nlo_tightening_critical: float
    acknowledged_kaon_nlo_factor: float
    acknowledged_nlo_factor_passes_any_proxy_scenario: bool
    local_pb_shape_envelope_crosses_pass_boundary: bool
    q4_weighted_pb_proxy_allows_all_product_scenarios: bool
    angular_projection_proxy_allows_all_product_scenarios: bool
    low_energy_sigma2_proxy_allows_any_product_scenario: bool
    pb_recast_diagnostics_disagree_on_pass_side: bool
    full_dt_scalar_current_calculation_supplied: bool
    mass_specific_pb_likelihood_supplied: bool
    full_kaon_nlo_likelihood_supplied: bool
    margin_robustness_gate_pass: bool
    physical_ce_fusion_branch_accepted: bool
    next_required_input: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _latest_na62_figure2_br_limit(
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Interpolate a Figure 2-a vector segment at the registered CE mass.

    Coordinates are PDF points extracted from the JHEP/arXiv v2 vector figure.
    This preserves the plotted readout reproducibly, but it is not a tabulated
    CLs likelihood or an exact experimental mass-bin value.
    """

    candidate_mass = current_fusion_flavor_aligned_report().operator.scalar_mass_mev
    x_target = LATEST_NA62_FIGURE2_AXIS_X_MIN_PT + (
        candidate_mass
        / LATEST_NA62_FIGURE2_AXIS_X_MAX_MEV
        * (LATEST_NA62_FIGURE2_AXIS_X_MAX_PT - LATEST_NA62_FIGURE2_AXIS_X_MIN_PT)
    )
    (x0, y0), (x1, y1) = segment
    if not min(x0, x1) <= x_target <= max(x0, x1) or x0 == x1:
        raise RuntimeError("NA62 Figure 2 segment does not bracket the registered mass")
    y_target = y0 + (y1 - y0) * (x_target - x0) / (x1 - x0)
    normalized = (LATEST_NA62_FIGURE2_AXIS_Y_BOTTOM_PT - y_target) / (
        LATEST_NA62_FIGURE2_AXIS_Y_BOTTOM_PT - LATEST_NA62_FIGURE2_AXIS_Y_TOP_PT
    )
    return normalized * LATEST_NA62_FIGURE2_Y_SCALE


def _erfc_array(values: np.ndarray) -> np.ndarray:
    """Vectorized erfc approximation with about 1e-7 absolute accuracy."""

    absolute = np.abs(values)
    t = 1.0 / (1.0 + 0.5 * absolute)
    positive = t * np.exp(
        -(absolute**2)
        - 1.26551223
        + t
        * (
            1.00002368
            + t
            * (
                0.37409196
                + t
                * (
                    0.09678418
                    + t
                    * (
                        -0.18628806
                        + t
                        * (
                            0.27886807
                            + t
                            * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))
                        )
                    )
                )
            )
        )
    )
    return np.where(values >= 0.0, positive, 2.0 - positive)


def _point_yukawa_attraction(
    *, product: float, scalar_mass_mev: float
) -> Callable[[np.ndarray], np.ndarray]:
    inverse_range_fm = scalar_mass_mev / HBAR_C_MEV_FM

    def attraction(radii_fm: np.ndarray) -> np.ndarray:
        return (
            product
            * HBAR_C_MEV_FM
            * np.exp(-inverse_range_fm * radii_fm)
            / (4.0 * math.pi * radii_fm)
        )

    return attraction


def _gaussian_folded_attraction(
    *,
    product: float,
    scalar_mass_mev: float,
    deuteron_rms_radius_fm: float,
    triton_rms_radius_fm: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fold a Yukawa Green function over two normalized Gaussian densities."""

    inverse_range_fm = scalar_mass_mev / HBAR_C_MEV_FM
    width = math.sqrt((deuteron_rms_radius_fm**2 + triton_rms_radius_fm**2) / 6.0)
    exponential_prefactor = 0.5 * math.exp((width * inverse_range_fm) ** 2)

    def attraction(radii_fm: np.ndarray) -> np.ndarray:
        lower = width * inverse_range_fm - radii_fm / (2.0 * width)
        upper = width * inverse_range_fm + radii_fm / (2.0 * width)
        folded_green = (
            exponential_prefactor
            / radii_fm
            * (
                np.exp(-inverse_range_fm * radii_fm) * _erfc_array(lower)
                - np.exp(inverse_range_fm * radii_fm) * _erfc_array(upper)
            )
        )
        return product * HBAR_C_MEV_FM * folded_green / (4.0 * math.pi)

    return attraction


def _exponential_folded_attraction(
    *,
    product: float,
    scalar_mass_mev: float,
    deuteron_rms_radius_fm: float,
    triton_rms_radius_fm: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fold over normalized three-dimensional exponential densities."""

    inverse_range_fm = scalar_mass_mev / HBAR_C_MEV_FM
    deuteron_alpha = math.sqrt(12.0) / deuteron_rms_radius_fm
    triton_alpha = math.sqrt(12.0) / triton_rms_radius_fm
    common = deuteron_alpha**4 * triton_alpha**4
    yukawa_coefficient = common / (
        (deuteron_alpha**2 - inverse_range_fm**2) ** 2
        * (triton_alpha**2 - inverse_range_fm**2) ** 2
    )
    density_poles: list[tuple[float, float, float]] = []
    for alpha, other_alpha in (
        (deuteron_alpha, triton_alpha),
        (triton_alpha, deuteron_alpha),
    ):
        double_pole = common / ((inverse_range_fm**2 - alpha**2) * (other_alpha**2 - alpha**2) ** 2)
        simple_pole = double_pole * (
            -1.0 / (inverse_range_fm**2 - alpha**2) - 2.0 / (other_alpha**2 - alpha**2)
        )
        density_poles.append((alpha, simple_pole, double_pole))

    def attraction(radii_fm: np.ndarray) -> np.ndarray:
        folded_green = yukawa_coefficient * np.exp(-inverse_range_fm * radii_fm) / radii_fm
        for alpha, simple_pole, double_pole in density_poles:
            exponential = np.exp(-alpha * radii_fm)
            folded_green += simple_pole * exponential / radii_fm
            folded_green += double_pole * exponential / (2.0 * alpha)
        return product * HBAR_C_MEV_FM * folded_green / (4.0 * math.pi)

    return attraction


def _uniform_sphere_imaginary_form_factor(
    inverse_range_fm: float,
    rms_radius_fm: float,
) -> float:
    sphere_radius = math.sqrt(5.0 / 3.0) * rms_radius_fm
    argument = inverse_range_fm * sphere_radius
    return 3.0 * (argument * math.cosh(argument) - math.sinh(argument)) / argument**3


def _uniform_sphere_folded_attraction(
    *,
    product: float,
    scalar_mass_mev: float,
    deuteron_rms_radius_fm: float,
    triton_rms_radius_fm: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fold over two uniform spheres, including their overlap region."""

    inverse_range_fm = scalar_mass_mev / HBAR_C_MEV_FM
    deuteron_radius = math.sqrt(5.0 / 3.0) * deuteron_rms_radius_fm
    triton_radius = math.sqrt(5.0 / 3.0) * triton_rms_radius_fm
    radius_sum = deuteron_radius + triton_radius
    radius_difference = abs(deuteron_radius - triton_radius)
    deuteron_volume = 4.0 * math.pi * deuteron_radius**3 / 3.0
    triton_volume = 4.0 * math.pi * triton_radius**3 / 3.0
    nodes, weights = np.polynomial.legendre.leggauss(UNIFORM_FOLDING_QUADRATURE_POINTS)
    separations = 0.5 * radius_sum * (nodes + 1.0)
    quadrature_weights = 0.5 * radius_sum * weights
    smaller_radius = min(deuteron_radius, triton_radius)
    full_overlap = 4.0 * math.pi * smaller_radius**3 / 3.0
    partial_overlap = (
        math.pi
        * (radius_sum - separations) ** 2
        * (separations**2 + 2.0 * separations * radius_sum - 3.0 * radius_difference**2)
        / (12.0 * separations)
    )
    overlap_volume = np.where(
        separations <= radius_difference,
        full_overlap,
        partial_overlap,
    )
    convolution_density = overlap_volume / (deuteron_volume * triton_volume)
    radial_weights = 4.0 * math.pi * quadrature_weights * separations**2 * convolution_density
    exterior_factor = _uniform_sphere_imaginary_form_factor(
        inverse_range_fm,
        deuteron_rms_radius_fm,
    ) * _uniform_sphere_imaginary_form_factor(
        inverse_range_fm,
        triton_rms_radius_fm,
    )

    def attraction(radii_fm: np.ndarray) -> np.ndarray:
        folded_green = exterior_factor * np.exp(-inverse_range_fm * radii_fm) / radii_fm
        overlap_mask = radii_fm < radius_sum
        if np.any(overlap_mask):
            radii = radii_fm[overlap_mask, np.newaxis]
            separation_grid = separations[np.newaxis, :]
            minimum = np.minimum(radii, separation_grid)
            maximum = np.maximum(radii, separation_grid)
            angular_kernel = (
                np.exp(-inverse_range_fm * maximum)
                * np.sinh(inverse_range_fm * minimum)
                / (inverse_range_fm * radii * separation_grid)
            )
            folded_green[overlap_mask] = angular_kernel @ radial_weights
        return product * HBAR_C_MEV_FM * folded_green / (4.0 * math.pi)

    return attraction


def _thermal_gain(attraction: Callable[[np.ndarray], np.ndarray]) -> float:
    return _thermal_response(
        temperature_kev=10.0,
        attraction=attraction,
        energy_points=PROXY_ENERGY_POINTS,
        wkb_grid_points=PROXY_WKB_GRID_POINTS,
    )[0]


def _solve_product(
    *,
    point_product: float,
    attraction_factory: Callable[[float], Callable[[np.ndarray], np.ndarray]],
) -> float:
    lower = 0.80 * point_product
    upper = 1.20 * point_product
    if _thermal_gain(attraction_factory(lower)) >= TARGET_GAIN_MINUS_ONE:
        raise RuntimeError("folded-product lower bracket already exceeds target")
    if _thermal_gain(attraction_factory(upper)) < TARGET_GAIN_MINUS_ONE:
        raise RuntimeError("folded-product upper bracket does not reach target")
    for _ in range(36):
        midpoint = 0.5 * (lower + upper)
        if _thermal_gain(attraction_factory(midpoint)) >= TARGET_GAIN_MINUS_ONE:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _folding_ratio_at_radius(
    *,
    scalar_mass_mev: float,
    deuteron_rms_radius_fm: float,
    triton_rms_radius_fm: float,
    radius_fm: float,
) -> float:
    point = _point_yukawa_attraction(product=1.0, scalar_mass_mev=scalar_mass_mev)
    folded = _gaussian_folded_attraction(
        product=1.0,
        scalar_mass_mev=scalar_mass_mev,
        deuteron_rms_radius_fm=deuteron_rms_radius_fm,
        triton_rms_radius_fm=triton_rms_radius_fm,
    )
    radius = np.array([radius_fm], dtype=float)
    return float(folded(radius)[0] / point(radius)[0])


def _build_folding_audits() -> tuple[DTFoldingAudit, ...]:
    flavor = current_fusion_flavor_aligned_report()
    mass = flavor.operator.scalar_mass_mev
    registered_point_product = flavor.operator.required_dt_charge_product

    point_grid_product = _solve_product(
        point_product=registered_point_product,
        attraction_factory=lambda product: _point_yukawa_attraction(
            product=product,
            scalar_mass_mev=mass,
        ),
    )
    audits: list[DTFoldingAudit] = [
        DTFoldingAudit(
            scenario="POINT_NUCLEUS_REFERENCE",
            deuteron_rms_radius_fm=0.0,
            triton_rms_radius_fm=0.0,
            gaussian_relative_width_fm=0.0,
            folding_ratio_at_nuclear_radius=1.0,
            asymptotic_imaginary_momentum_form_factor=1.0,
            required_dt_product=registered_point_product,
            required_product_to_point_ratio=1.0,
            required_coupling_to_point_ratio=1.0,
            one_body_gaussian_folding_computed=False,
            two_body_scalar_currents_supplied=False,
            ab_initio_nuclear_density_covariance_supplied=False,
            status="POINT_NUCLEUS_REFERENCE_ONLY",
        )
    ]
    inverse_range_fm = mass / HBAR_C_MEV_FM
    for scenario, deuteron_radius, triton_radius in DT_GAUSSIAN_RMS_SCENARIOS_FM:
        folded_grid_product = _solve_product(
            point_product=registered_point_product,
            attraction_factory=lambda product, rd=deuteron_radius, rt=triton_radius: (
                _gaussian_folded_attraction(
                    product=product,
                    scalar_mass_mev=mass,
                    deuteron_rms_radius_fm=rd,
                    triton_rms_radius_fm=rt,
                )
            ),
        )
        product_ratio = folded_grid_product / point_grid_product
        width = math.sqrt((deuteron_radius**2 + triton_radius**2) / 6.0)
        audits.append(
            DTFoldingAudit(
                scenario=scenario,
                deuteron_rms_radius_fm=deuteron_radius,
                triton_rms_radius_fm=triton_radius,
                gaussian_relative_width_fm=width,
                folding_ratio_at_nuclear_radius=_folding_ratio_at_radius(
                    scalar_mass_mev=mass,
                    deuteron_rms_radius_fm=deuteron_radius,
                    triton_rms_radius_fm=triton_radius,
                    radius_fm=3.24,
                ),
                asymptotic_imaginary_momentum_form_factor=math.exp((width * inverse_range_fm) ** 2),
                required_dt_product=registered_point_product * product_ratio,
                required_product_to_point_ratio=product_ratio,
                required_coupling_to_point_ratio=math.sqrt(product_ratio),
                one_body_gaussian_folding_computed=True,
                two_body_scalar_currents_supplied=False,
                ab_initio_nuclear_density_covariance_supplied=False,
                status="ONE_BODY_GAUSSIAN_PROXY_ONLY_NO_NUCLEAR_LIKELIHOOD",
            )
        )
    return tuple(audits)


def _build_morphology_audits() -> tuple[DTMorphologyAudit, ...]:
    """Compare one-body density morphologies in the weak-potential response."""

    flavor = current_fusion_flavor_aligned_report()
    mass = flavor.operator.scalar_mass_mev
    point_product = flavor.operator.required_dt_charge_product
    probe_product = LINEAR_RESPONSE_PROBE_FRACTION * point_product
    point_gain = _thermal_gain(
        _point_yukawa_attraction(product=probe_product, scalar_mass_mev=mass)
    )
    inverse_range_fm = mass / HBAR_C_MEV_FM
    audits: list[DTMorphologyAudit] = []
    factories: tuple[
        tuple[
            str,
            Callable[..., Callable[[np.ndarray], np.ndarray]],
        ],
        ...,
    ] = (
        ("GAUSSIAN", _gaussian_folded_attraction),
        ("EXPONENTIAL", _exponential_folded_attraction),
        ("UNIFORM_SPHERE", _uniform_sphere_folded_attraction),
    )
    for radius_scenario, deuteron_radius, triton_radius in DT_MORPHOLOGY_RMS_SCENARIOS_FM:
        gaussian_width = math.sqrt((deuteron_radius**2 + triton_radius**2) / 6.0)
        exponential_scales = (
            deuteron_radius / math.sqrt(12.0),
            triton_radius / math.sqrt(12.0),
        )
        asymptotic_by_morphology = {
            "GAUSSIAN": math.exp((gaussian_width * inverse_range_fm) ** 2),
            "EXPONENTIAL": math.prod(
                (1.0 - (scale * inverse_range_fm) ** 2) ** -2 for scale in exponential_scales
            ),
            "UNIFORM_SPHERE": _uniform_sphere_imaginary_form_factor(
                inverse_range_fm,
                deuteron_radius,
            )
            * _uniform_sphere_imaginary_form_factor(
                inverse_range_fm,
                triton_radius,
            ),
        }
        for morphology, factory in factories:
            attraction = factory(
                product=probe_product,
                scalar_mass_mev=mass,
                deuteron_rms_radius_fm=deuteron_radius,
                triton_rms_radius_fm=triton_radius,
            )
            folded_gain = _thermal_gain(attraction)
            response_factor = folded_gain / point_gain
            product_ratio = 1.0 / response_factor
            radius = np.array([3.24], dtype=float)
            unit_folded = factory(
                product=1.0,
                scalar_mass_mev=mass,
                deuteron_rms_radius_fm=deuteron_radius,
                triton_rms_radius_fm=triton_radius,
            )
            unit_point = _point_yukawa_attraction(
                product=1.0,
                scalar_mass_mev=mass,
            )
            audits.append(
                DTMorphologyAudit(
                    radius_scenario=radius_scenario,
                    density_morphology=morphology,
                    deuteron_rms_radius_fm=deuteron_radius,
                    triton_rms_radius_fm=triton_radius,
                    folding_ratio_at_nuclear_radius=float(
                        unit_folded(radius)[0] / unit_point(radius)[0]
                    ),
                    asymptotic_imaginary_momentum_form_factor=(
                        asymptotic_by_morphology[morphology]
                    ),
                    linear_response_enhancement_factor=response_factor,
                    linearized_required_product_to_point_ratio=product_ratio,
                    linearized_required_coupling_to_point_ratio=math.sqrt(product_ratio),
                    full_one_percent_resolve_performed=False,
                    two_body_scalar_currents_supplied=False,
                    status=("ONE_BODY_LINEAR_RESPONSE_MORPHOLOGY_PROXY_ONLY_NO_TWO_BODY_CURRENT"),
                )
            )
    return tuple(audits)


def _uniform_sphere_form_factor(
    momentum_fm_inv: np.ndarray,
    rms_radius_fm: float,
) -> np.ndarray:
    sphere_radius = math.sqrt(5.0 / 3.0) * rms_radius_fm
    argument = momentum_fm_inv * sphere_radius
    result = np.empty_like(argument)
    small = np.abs(argument) < 1.0e-4
    result[small] = 1.0 - argument[small] ** 2 / 10.0 + argument[small] ** 4 / 280.0
    regular = ~small
    result[regular] = (
        3.0
        * (np.sin(argument[regular]) - argument[regular] * np.cos(argument[regular]))
        / argument[regular] ** 3
    )
    return result


def _lead_form_factors(
    momentum_fm_inv: np.ndarray,
    rms_radius_fm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = momentum_fm_inv * rms_radius_fm
    gaussian = np.exp(-(scaled**2) / 6.0)
    exponential = (1.0 + scaled**2 / 12.0) ** -2
    uniform = _uniform_sphere_form_factor(momentum_fm_inv, rms_radius_fm)
    return gaussian, exponential, uniform


def _regression_slope(
    independent: np.ndarray,
    dependent: np.ndarray,
    integration_coordinate: np.ndarray,
) -> float:
    normalization = float(
        np.trapezoid(np.ones_like(integration_coordinate), integration_coordinate)
    )
    mean = float(np.trapezoid(independent, integration_coordinate) / normalization)
    centered = independent - mean
    covariance = float(np.trapezoid(centered * dependent, integration_coordinate))
    variance = float(np.trapezoid(centered**2, integration_coordinate))
    return covariance / variance


def _angular_p_wave_projection_responses(
    *,
    scalar_mass_mev: float,
    momentum_mev: float,
) -> tuple[float, list[float]]:
    point_responses: list[float] = []
    extended_responses: list[float] = []
    point_heavy_slope = 2.0 * momentum_mev**2 / scalar_mass_mev**4
    angle_radians = np.deg2rad(
        np.linspace(
            PB_MIN_SCATTERING_ANGLE_DEG,
            PB_MAX_SCATTERING_ANGLE_DEG,
            PB_ANGLE_GRID_POINTS,
        )
    )
    cosine_theta = np.cos(angle_radians)
    equal_x = np.linspace(
        math.cos(math.radians(PB_MAX_SCATTERING_ANGLE_DEG)),
        math.cos(math.radians(PB_MIN_SCATTERING_ANGLE_DEG)),
        PB_ANGLE_GRID_POINTS,
    )
    for cosine, coordinate in (
        (cosine_theta, angle_radians),
        (equal_x, equal_x),
    ):
        transfers_mev = np.sqrt(2.0 * momentum_mev**2 * (1.0 - cosine))
        point_shape = 1.0 / (scalar_mass_mev**2 + transfers_mev**2)
        point_responses.append(
            _regression_slope(cosine, point_shape, coordinate) / point_heavy_slope
        )
        transfers_fm_inv = transfers_mev / HBAR_C_MEV_FM
        for radius in (PB_RMS_RADIUS_MIN_FM, PB_RMS_RADIUS_MAX_FM):
            for form_factor in _lead_form_factors(transfers_fm_inv, radius):
                shape = form_factor / (scalar_mass_mev**2 + transfers_mev**2)
                extended_responses.append(
                    _regression_slope(cosine, shape, coordinate) / point_heavy_slope
                )
    return min(point_responses), extended_responses


def _low_energy_sigma2_projection_responses(
    *,
    scalar_mass_mev: float,
    energy_grid_points: int,
    angle_grid_points: int,
) -> list[float]:
    energies_mev = np.geomspace(1.0e-5, 1.0e-2, energy_grid_points)
    momentum_mev = (
        LEAD_MASS_NUMBER / (LEAD_MASS_NUMBER + 1.0) * np.sqrt(2.0 * NEUTRON_MASS_MEV * energies_mev)
    )
    momentum_squared = momentum_mev**2
    cosine_theta = np.linspace(-1.0, 1.0, angle_grid_points)
    point_heavy_slope = -2.0 / scalar_mass_mev**4
    responses: list[float] = []
    for radius in (PB_RMS_RADIUS_MIN_FM, PB_RMS_RADIUS_MAX_FM):
        angle_averages: list[list[float]] = [[], [], []]
        for momentum in momentum_mev:
            transfers_mev = np.sqrt(2.0 * momentum**2 * (1.0 - cosine_theta))
            transfers_fm_inv = transfers_mev / HBAR_C_MEV_FM
            form_factors = _lead_form_factors(transfers_fm_inv, radius)
            for index, form_factor in enumerate(form_factors):
                amplitude = form_factor / (scalar_mass_mev**2 + transfers_mev**2)
                angle_averages[index].append(0.5 * float(np.trapezoid(amplitude, cosine_theta)))
        for averages in angle_averages:
            dependent = np.asarray(averages)
            for coordinate in (energies_mev, np.log(energies_mev)):
                responses.append(
                    _regression_slope(
                        momentum_squared,
                        dependent,
                        coordinate,
                    )
                    / point_heavy_slope
                )
    return responses


def _lead_shape_proxy() -> LeadShapeProxyAudit:
    mass = current_fusion_flavor_aligned_report().operator.scalar_mass_mev
    neutron_energy_mev = NEUTRON_KINETIC_ENERGY_KEV / 1000.0
    momentum = (
        LEAD_MASS_NUMBER
        / (LEAD_MASS_NUMBER + 1.0)
        * math.sqrt(2.0 * NEUTRON_MASS_MEV * neutron_energy_mev)
    )
    angles = np.linspace(
        PB_MIN_SCATTERING_ANGLE_DEG,
        PB_MAX_SCATTERING_ANGLE_DEG,
        PB_ANGLE_GRID_POINTS,
    )
    transfers_mev = 2.0 * momentum * np.sin(0.5 * np.deg2rad(angles))
    transfers_fm_inv = transfers_mev / HBAR_C_MEV_FM
    inverse_range_fm = mass / HBAR_C_MEV_FM
    propagator = mass**2 / (mass**2 + transfers_mev**2)

    combined_responses: list[np.ndarray] = []
    weighted_responses: list[float] = []
    for radius in (PB_RMS_RADIUS_MIN_FM, PB_RMS_RADIUS_MAX_FM):
        for form_factor in _lead_form_factors(transfers_fm_inv, radius):
            response = (
                inverse_range_fm**2
                / (inverse_range_fm**2 + transfers_fm_inv**2)
                * (1.0 + inverse_range_fm**2 * (1.0 - form_factor) / transfers_fm_inv**2)
            )
            combined_responses.append(response)
            weighted_responses.append(
                math.sqrt(
                    float(np.sum(transfers_mev**4 * response**2)) / float(np.sum(transfers_mev**4))
                )
            )

    response_minimum = min(float(np.min(response)) for response in combined_responses)
    response_maximum = max(float(np.max(response)) for response in combined_responses)
    point_angular, angular_responses = _angular_p_wave_projection_responses(
        scalar_mass_mev=mass,
        momentum_mev=momentum,
    )
    sigma2_coarse_responses = _low_energy_sigma2_projection_responses(
        scalar_mass_mev=mass,
        energy_grid_points=PB_LOW_ENERGY_GRID_POINTS,
        angle_grid_points=PB_ANGLE_GRID_POINTS,
    )
    sigma2_responses = _low_energy_sigma2_projection_responses(
        scalar_mass_mev=mass,
        energy_grid_points=PB_REFINED_LOW_ENERGY_GRID_POINTS,
        angle_grid_points=PB_REFINED_ANGLE_GRID_POINTS,
    )
    sigma2_refinement_shift = max(
        abs(refined - coarse) / abs(refined)
        for coarse, refined in zip(
            sigma2_coarse_responses,
            sigma2_responses,
            strict=True,
        )
    )
    sigma2_zero_energy_responses = [
        1.0 + mass**2 * radius**2 / (6.0 * HBAR_C_MEV_FM**2)
        for radius in (PB_RMS_RADIUS_MIN_FM, PB_RMS_RADIUS_MAX_FM)
    ]
    angular_minimum = min(angular_responses)
    angular_maximum = max(angular_responses)
    sigma2_minimum = min(sigma2_responses)
    sigma2_maximum = max(sigma2_responses)
    return LeadShapeProxyAudit(
        neutron_kinetic_energy_kev=NEUTRON_KINETIC_ENERGY_KEV,
        minimum_scattering_angle_deg=PB_MIN_SCATTERING_ANGLE_DEG,
        maximum_scattering_angle_deg=PB_MAX_SCATTERING_ANGLE_DEG,
        minimum_momentum_transfer_mev=float(transfers_mev[0]),
        maximum_momentum_transfer_mev=float(transfers_mev[-1]),
        finite_propagator_response_minimum=float(np.min(propagator)),
        finite_propagator_response_maximum=float(np.max(propagator)),
        pb_rms_radius_min_fm=PB_RMS_RADIUS_MIN_FM,
        pb_rms_radius_max_fm=PB_RMS_RADIUS_MAX_FM,
        combined_shape_response_minimum=response_minimum,
        combined_shape_response_maximum=response_maximum,
        q4_weighted_shape_response_minimum=min(weighted_responses),
        q4_weighted_shape_response_maximum=max(weighted_responses),
        corresponding_bound_multiplier_minimum=1.0 / math.sqrt(response_maximum),
        corresponding_bound_multiplier_maximum=1.0 / math.sqrt(response_minimum),
        angular_p_wave_point_propagator_projection_response_minimum=point_angular,
        angular_p_wave_projection_response_minimum=angular_minimum,
        angular_p_wave_projection_response_maximum=angular_maximum,
        angular_p_wave_bound_multiplier_minimum=1.0 / math.sqrt(angular_maximum),
        angular_p_wave_bound_multiplier_maximum=1.0 / math.sqrt(angular_minimum),
        low_energy_sigma2_zero_energy_response_minimum=min(sigma2_zero_energy_responses),
        low_energy_sigma2_zero_energy_response_maximum=max(sigma2_zero_energy_responses),
        low_energy_sigma2_finite_window_response_minimum=sigma2_minimum,
        low_energy_sigma2_finite_window_response_maximum=sigma2_maximum,
        low_energy_sigma2_bound_multiplier_minimum=(1.0 / math.sqrt(sigma2_maximum)),
        low_energy_sigma2_bound_multiplier_maximum=(1.0 / math.sqrt(sigma2_minimum)),
        low_energy_sigma2_refined_energy_grid_points=(PB_REFINED_LOW_ENERGY_GRID_POINTS),
        low_energy_sigma2_refined_angle_grid_points=PB_REFINED_ANGLE_GRID_POINTS,
        low_energy_sigma2_grid_refinement_max_relative_shift=sigma2_refinement_shift,
        low_energy_sigma2_numerical_convergence_tolerance=(PB_PROXY_CONVERGENCE_RELATIVE_TOLERANCE),
        low_energy_sigma2_numerical_convergence_pass=(
            sigma2_refinement_shift <= PB_PROXY_CONVERGENCE_RELATIVE_TOLERANCE
        ),
        gaussian_and_uniform_sphere_envelope_used=True,
        exponential_form_factor_in_projection_used=True,
        angular_theta_and_x_weightings_used=True,
        low_energy_linear_and_log_weightings_used=True,
        alternative_recasts_land_on_opposite_sides_of_contact_limit=(
            angular_maximum < 1.0 < sigma2_minimum
        ),
        experimental_angular_covariance_supplied=False,
        strong_amplitude_phase_profiled=False,
        source_analysis_finite_density_treatment_known=False,
        status=(
            "FINITE_SHAPE_RECASTS_DISAGREE_SOURCE_COVARIANCE_AND_FINITE_DENSITY_PROVENANCE_ABSENT"
        ),
    )


def evaluate_margin_cell(
    *,
    dt_product_to_point_ratio: Real,
    pb_shape_response_multiplier: Real,
    kaon_nlo_tightening_factor: Real,
    kaon_digitized_bound_factor: Real = 1.0,
    latest_kaon_br_improvement_factor: Real = 1.0,
) -> MarginCellAudit:
    """Evaluate algebraic central-proxy inequalities without opening the gate."""

    product_ratio = _positive(
        dt_product_to_point_ratio,
        name="dt_product_to_point_ratio",
    )
    pb_response = _positive(
        pb_shape_response_multiplier,
        name="pb_shape_response_multiplier",
    )
    nlo_factor = _positive(
        kaon_nlo_tightening_factor,
        name="kaon_nlo_tightening_factor",
    )
    line_factor = _positive(
        kaon_digitized_bound_factor,
        name="kaon_digitized_bound_factor",
    )
    latest_br_improvement = _positive(
        latest_kaon_br_improvement_factor,
        name="latest_kaon_br_improvement_factor",
    )
    latest_coupling_multiplier = 1.0 / math.sqrt(latest_br_improvement)
    flavor = current_fusion_flavor_aligned_report()
    coupling_scale = math.sqrt(product_ratio)
    pb_candidate = flavor.neutron_constraint.flavor_matched_lead_effective_coupling * coupling_scale
    pb_bound = flavor.neutron_constraint.extrapolated_equal_coupling_bound / math.sqrt(pb_response)
    pb_critical = (
        flavor.neutron_constraint.extrapolated_equal_coupling_bound
        / (flavor.neutron_constraint.flavor_matched_lead_effective_coupling * coupling_scale)
    ) ** 2

    kaon_candidate = (
        flavor.rare_decay_constraint.required_plot_coordinate_kappa_v_over_m * coupling_scale
    )
    central_kaon_bound = (
        flavor.rare_decay_constraint.digitized_uds_invisible_bound_central
        * line_factor
        * latest_coupling_multiplier
    )
    kaon_bound = central_kaon_bound / nlo_factor
    kaon_critical = central_kaon_bound / kaon_candidate
    neutron_condition = pb_candidate <= pb_bound
    kaon_condition = kaon_candidate <= kaon_bound
    joint = neutron_condition and kaon_condition
    return MarginCellAudit(
        dt_product_to_point_ratio=product_ratio,
        pb_shape_response_multiplier=pb_response,
        kaon_nlo_tightening_factor=nlo_factor,
        kaon_digitized_bound_factor=line_factor,
        latest_kaon_br_improvement_factor=latest_br_improvement,
        latest_kaon_coupling_bound_multiplier=latest_coupling_multiplier,
        corrected_pb_candidate_coupling=pb_candidate,
        corrected_pb_proxy_bound=pb_bound,
        pb_response_critical=pb_critical,
        neutron_proxy_condition_satisfied=neutron_condition,
        corrected_kaon_candidate_coordinate=kaon_candidate,
        corrected_kaon_proxy_bound=kaon_bound,
        kaon_nlo_tightening_critical=kaon_critical,
        kaon_proxy_condition_satisfied=kaon_condition,
        joint_proxy_conditions_satisfied=joint,
        experimental_likelihoods_supplied=False,
        physical_gate_pass=False,
        status=(
            "CENTRAL_PROXY_CELL_ALGEBRAICALLY_OPEN_LIKELIHOODS_MISSING"
            if joint
            else "CENTRAL_PROXY_CELL_CLOSED"
        ),
    )


def _latest_kaon_data_audit(
    *,
    most_favorable_product_ratio: float,
) -> LatestKaonDataAudit:
    """Propagate the published BR-improvement range without inventing a mass bin."""

    flavor = current_fusion_flavor_aligned_report()
    previous_bound = flavor.rare_decay_constraint.digitized_uds_invisible_bound_central
    figure2_new_limit = _latest_na62_figure2_br_limit(LATEST_NA62_FIGURE2_NEW_SEGMENT_PT)
    figure2_old_limit = _latest_na62_figure2_br_limit(LATEST_NA62_FIGURE2_OLD_SEGMENT_PT)
    figure2_improvement = figure2_old_limit / figure2_new_limit
    figure2_multiplier = 1.0 / math.sqrt(figure2_improvement)
    readout_uncertainty = LATEST_NA62_FIGURE2_RELATIVE_READOUT_UNCERTAINTY
    figure2_improvement_minimum = (figure2_old_limit * (1.0 - readout_uncertainty)) / (
        figure2_new_limit * (1.0 + readout_uncertainty)
    )
    figure2_improvement_maximum = (figure2_old_limit * (1.0 + readout_uncertainty)) / (
        figure2_new_limit * (1.0 - readout_uncertainty)
    )
    figure2_multiplier_minimum = 1.0 / math.sqrt(figure2_improvement_maximum)
    figure2_multiplier_maximum = 1.0 / math.sqrt(figure2_improvement_minimum)
    point_weak = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=LATEST_NA62_BR_IMPROVEMENT_MINIMUM,
    )
    point_strong = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=LATEST_NA62_BR_IMPROVEMENT_MAXIMUM,
    )
    favorable_weak = evaluate_margin_cell(
        dt_product_to_point_ratio=most_favorable_product_ratio,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=LATEST_NA62_BR_IMPROVEMENT_MINIMUM,
    )
    favorable_strong = evaluate_margin_cell(
        dt_product_to_point_ratio=most_favorable_product_ratio,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=LATEST_NA62_BR_IMPROVEMENT_MAXIMUM,
    )
    figure2_point = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=figure2_improvement,
    )
    figure2_favorable = evaluate_margin_cell(
        dt_product_to_point_ratio=most_favorable_product_ratio,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=figure2_improvement,
    )
    figure2_point_tight = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=figure2_improvement_maximum,
    )
    figure2_point_loose = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=figure2_improvement_minimum,
    )
    figure2_favorable_tight = evaluate_margin_cell(
        dt_product_to_point_ratio=most_favorable_product_ratio,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=figure2_improvement_maximum,
    )
    figure2_favorable_loose = evaluate_margin_cell(
        dt_product_to_point_ratio=most_favorable_product_ratio,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=figure2_improvement_minimum,
    )
    minimum_coupling_multiplier = 1.0 / math.sqrt(LATEST_NA62_BR_IMPROVEMENT_MAXIMUM)
    candidate_mass = flavor.operator.scalar_mass_mev
    return LatestKaonDataAudit(
        source_arxiv_identifier="arXiv:2507.17286v2",
        dataset_years="2016-2022",
        confidence_level_percent=LATEST_NA62_CONFIDENCE_LEVEL_PERCENT,
        low_mass_scan_minimum_mev=0.0,
        low_mass_scan_maximum_mev=LATEST_NA62_LOW_MASS_RANGE_MAX_MEV,
        mass_hypothesis_spacing_mev=LATEST_NA62_MASS_HYPOTHESIS_SPACING_MEV,
        candidate_mass_mev=candidate_mass,
        candidate_inside_low_mass_scan=(
            0.0 <= candidate_mass <= LATEST_NA62_LOW_MASS_RANGE_MAX_MEV
        ),
        branching_ratio_improvement_factor_minimum=(LATEST_NA62_BR_IMPROVEMENT_MINIMUM),
        branching_ratio_improvement_factor_maximum=(LATEST_NA62_BR_IMPROVEMENT_MAXIMUM),
        coupling_bound_multiplier_minimum=minimum_coupling_multiplier,
        coupling_bound_multiplier_maximum=1.0,
        previous_digitized_coupling_bound=previous_bound,
        latest_range_coupling_bound_minimum=(previous_bound * minimum_coupling_multiplier),
        latest_range_coupling_bound_maximum=previous_bound,
        point_nlo_tightening_critical_minimum=(point_strong.kaon_nlo_tightening_critical),
        point_nlo_tightening_critical_maximum=(point_weak.kaon_nlo_tightening_critical),
        favorable_proxy_nlo_tightening_critical_minimum=(
            favorable_strong.kaon_nlo_tightening_critical
        ),
        favorable_proxy_nlo_tightening_critical_maximum=(
            favorable_weak.kaon_nlo_tightening_critical
        ),
        tree_level_candidate_allowed_across_improvement_range=(
            point_strong.kaon_proxy_condition_satisfied
        ),
        figure2_pdf_page_index=LATEST_NA62_FIGURE2_PDF_PAGE_INDEX,
        figure2_candidate_mass_curve_interpolation_entered=True,
        figure2_relative_readout_uncertainty=readout_uncertainty,
        figure2_readout_errors_treated_as_independent_box=True,
        figure2_interpolated_2016_2022_observed_br_limit=figure2_new_limit,
        figure2_interpolated_2016_2018_observed_br_limit=figure2_old_limit,
        figure2_interpolated_br_improvement_factor=figure2_improvement,
        figure2_br_improvement_factor_minimum=figure2_improvement_minimum,
        figure2_br_improvement_factor_maximum=figure2_improvement_maximum,
        figure2_interpolated_coupling_bound_multiplier=figure2_multiplier,
        figure2_coupling_bound_multiplier_minimum=figure2_multiplier_minimum,
        figure2_coupling_bound_multiplier_maximum=figure2_multiplier_maximum,
        figure2_interpolated_latest_uds_bound=previous_bound * figure2_multiplier,
        figure2_latest_uds_bound_minimum=previous_bound * figure2_multiplier_minimum,
        figure2_latest_uds_bound_maximum=previous_bound * figure2_multiplier_maximum,
        figure2_interpolated_point_nlo_tightening_critical=(
            figure2_point.kaon_nlo_tightening_critical
        ),
        figure2_point_nlo_tightening_critical_minimum=(
            figure2_point_tight.kaon_nlo_tightening_critical
        ),
        figure2_point_nlo_tightening_critical_maximum=(
            figure2_point_loose.kaon_nlo_tightening_critical
        ),
        figure2_interpolated_favorable_nlo_tightening_critical=(
            figure2_favorable.kaon_nlo_tightening_critical
        ),
        figure2_favorable_nlo_tightening_critical_minimum=(
            figure2_favorable_tight.kaon_nlo_tightening_critical
        ),
        figure2_favorable_nlo_tightening_critical_maximum=(
            figure2_favorable_loose.kaon_nlo_tightening_critical
        ),
        exact_candidate_mass_observed_limit_entered=False,
        full_uds_operator_recast_and_nlo_likelihood_supplied=False,
        latest_data_gate_pass=False,
        status=(
            "LATEST_BR_RANGE_AND_FIGURE2_INTERPOLATION_PROPAGATED_"
            "EXACT_29P65_MEV_UDS_RECAST_NOT_ENTERED"
        ),
    )


@lru_cache(maxsize=1)
def current_fusion_flavor_margin_robustness_report() -> FusionFlavorMarginRobustnessReport:
    """Build the finite-size/constraint-margin report and retain fail-closed status."""

    flavor = current_fusion_flavor_aligned_report()
    folding = _build_folding_audits()
    morphology = _build_morphology_audits()
    lead = _lead_shape_proxy()
    product_ratios = [audit.required_product_to_point_ratio for audit in folding]
    morphology_product_ratios = [
        audit.linearized_required_product_to_point_ratio for audit in morphology
    ]
    minimum_ratio = min(product_ratios)
    maximum_ratio = max(product_ratios)
    minimum_morphology_ratio = min(morphology_product_ratios)
    maximum_morphology_ratio = max(morphology_product_ratios)
    most_favorable_ratio = min(minimum_ratio, minimum_morphology_ratio)
    all_proxy_product_ratios = product_ratios + morphology_product_ratios
    latest_kaon = _latest_kaon_data_audit(most_favorable_product_ratio=most_favorable_ratio)

    point_cell = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
    )
    favorable_cell = evaluate_margin_cell(
        dt_product_to_point_ratio=most_favorable_ratio,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
    )
    lower_line_cell = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        kaon_digitized_bound_factor=(
            1.0 - flavor.rare_decay_constraint.digitized_line_relative_uncertainty
        ),
    )
    acknowledged = flavor.rare_decay_constraint.acknowledged_partial_nlo_correction_factor
    acknowledged_passes = any(
        evaluate_margin_cell(
            dt_product_to_point_ratio=ratio,
            pb_shape_response_multiplier=min(
                lead.combined_shape_response_minimum,
                lead.angular_p_wave_projection_response_minimum,
            ),
            kaon_nlo_tightening_factor=acknowledged,
        ).joint_proxy_conditions_satisfied
        for ratio in all_proxy_product_ratios
    )
    local_crosses = (
        lead.combined_shape_response_minimum
        < point_cell.pb_response_critical
        < lead.combined_shape_response_maximum
    )
    weighted_allows_all = all(
        lead.q4_weighted_shape_response_maximum
        <= evaluate_margin_cell(
            dt_product_to_point_ratio=ratio,
            pb_shape_response_multiplier=1.0,
            kaon_nlo_tightening_factor=1.0,
        ).pb_response_critical
        for ratio in all_proxy_product_ratios
    )
    angular_allows_all = all(
        lead.angular_p_wave_projection_response_maximum
        <= evaluate_margin_cell(
            dt_product_to_point_ratio=ratio,
            pb_shape_response_multiplier=1.0,
            kaon_nlo_tightening_factor=1.0,
        ).pb_response_critical
        for ratio in all_proxy_product_ratios
    )
    sigma2_allows_any = any(
        lead.low_energy_sigma2_finite_window_response_minimum
        <= evaluate_margin_cell(
            dt_product_to_point_ratio=ratio,
            pb_shape_response_multiplier=1.0,
            kaon_nlo_tightening_factor=1.0,
        ).pb_response_critical
        for ratio in all_proxy_product_ratios
    )
    return FusionFlavorMarginRobustnessReport(
        schema_version="1.0",
        point_required_dt_product=flavor.operator.required_dt_charge_product,
        folding_scenarios=folding,
        morphology_scenarios=morphology,
        minimum_required_product_to_point_ratio=minimum_ratio,
        maximum_required_product_to_point_ratio=maximum_ratio,
        minimum_linearized_morphology_product_to_point_ratio=(minimum_morphology_ratio),
        maximum_linearized_morphology_product_to_point_ratio=(maximum_morphology_ratio),
        most_favorable_proxy_product_to_point_ratio=most_favorable_ratio,
        lead_shape_proxy=lead,
        latest_kaon_data=latest_kaon,
        point_pb_response_critical=point_cell.pb_response_critical,
        most_favorable_proxy_pb_response_critical=favorable_cell.pb_response_critical,
        point_kaon_nlo_tightening_critical=point_cell.kaon_nlo_tightening_critical,
        most_favorable_proxy_kaon_nlo_tightening_critical=(
            favorable_cell.kaon_nlo_tightening_critical
        ),
        robust_lower_line_kaon_nlo_tightening_critical=(
            lower_line_cell.kaon_nlo_tightening_critical
        ),
        acknowledged_kaon_nlo_factor=acknowledged,
        acknowledged_nlo_factor_passes_any_proxy_scenario=acknowledged_passes,
        local_pb_shape_envelope_crosses_pass_boundary=local_crosses,
        q4_weighted_pb_proxy_allows_all_product_scenarios=weighted_allows_all,
        angular_projection_proxy_allows_all_product_scenarios=(angular_allows_all),
        low_energy_sigma2_proxy_allows_any_product_scenario=sigma2_allows_any,
        pb_recast_diagnostics_disagree_on_pass_side=(
            angular_allows_all
            and not sigma2_allows_any
            and lead.alternative_recasts_land_on_opposite_sides_of_contact_limit
        ),
        full_dt_scalar_current_calculation_supplied=False,
        mass_specific_pb_likelihood_supplied=False,
        full_kaon_nlo_likelihood_supplied=False,
        margin_robustness_gate_pass=False,
        physical_ce_fusion_branch_accepted=False,
        next_required_input=(
            "replace Gaussian D/T densities by ab-initio one- and two-body scalar currents; "
            "fit the exact finite-propagator Pb amplitude with the historical angular "
            "covariance and strong phase; and provide a full O(p^4) weak-ChPT plus "
            "NA62/E949 likelihood for the uds kaon direction, including a tabulated "
            "29.65 MeV 2016--2022 NA62 CLs point and acceptance"
        ),
        conclusion=(
            "One-body density morphologies lower the required D/T charge product by only about "
            "two percent.  Pb angular and low-energy-total-cross-section recasts move the "
            "per-mille boundary in opposite directions, exposing the missing source covariance "
            "and finite-density provenance.  The kaon curve tolerates a downward NLO tightening "
            "of only about six to seven before the latest-data range, below the acknowledged "
            "factor-ten envelope.  The 2016--2022 NA62 BR improvement tightens a coupling bound "
            "by its square root.  A reproducible Figure 2 vector-curve interpolation at the CE "
            "mass gives a central improvement near 1.33 and a central point-NLO margin near "
            "5.75; independent five-percent readout boxes broaden that margin to about "
            "5.47--6.05.  This is not a tabulated CLs bin or a full uds recast.  No proxy cell is an "
            "experimental likelihood, so the candidate remains fail-closed."
        ),
    )


__all__ = [
    "DTFoldingAudit",
    "DTMorphologyAudit",
    "FusionFlavorMarginRobustnessReport",
    "LeadShapeProxyAudit",
    "LatestKaonDataAudit",
    "MarginCellAudit",
    "current_fusion_flavor_margin_robustness_report",
    "evaluate_margin_cell",
]
