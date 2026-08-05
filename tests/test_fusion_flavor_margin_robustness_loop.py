from __future__ import annotations

import math

import pytest

from reality_stone.clarus.fusion_flavor_margin_robustness_loop import (
    current_fusion_flavor_margin_robustness_report,
    evaluate_margin_cell,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_flavor_margin_robustness_report()


def test_one_body_dt_folding_is_small_and_reproducible(report) -> None:
    point, compact, reference, maximum, broad = report.folding_scenarios

    assert point.scenario == "POINT_NUCLEUS_REFERENCE"
    assert point.required_product_to_point_ratio == 1.0
    assert compact.required_product_to_point_ratio == pytest.approx(
        0.98580,
        rel=2.0e-4,
    )
    assert reference.required_product_to_point_ratio == pytest.approx(
        0.98033,
        rel=2.0e-4,
    )
    assert maximum.required_product_to_point_ratio == pytest.approx(
        0.97902,
        rel=2.0e-4,
    )
    assert broad.required_product_to_point_ratio == pytest.approx(
        0.97976,
        rel=2.0e-4,
    )
    assert report.minimum_required_product_to_point_ratio == pytest.approx(
        0.97902,
        rel=2.0e-4,
    )
    assert report.maximum_required_product_to_point_ratio == 1.0
    assert maximum.required_coupling_to_point_ratio == pytest.approx(
        math.sqrt(maximum.required_product_to_point_ratio),
        rel=1.0e-12,
    )
    assert all(
        not audit.two_body_scalar_currents_supplied
        and not audit.ab_initio_nuclear_density_covariance_supplied
        for audit in report.folding_scenarios
    )


def test_density_morphology_envelope_is_small_and_explicitly_linearized(report) -> None:
    audits = report.morphology_scenarios

    assert len(audits) == 6
    assert {audit.density_morphology for audit in audits} == {
        "GAUSSIAN",
        "EXPONENTIAL",
        "UNIFORM_SPHERE",
    }
    assert report.minimum_linearized_morphology_product_to_point_ratio == pytest.approx(
        0.975861,
        rel=2.0e-5,
    )
    assert report.maximum_linearized_morphology_product_to_point_ratio == pytest.approx(
        0.983453,
        rel=2.0e-5,
    )
    assert report.most_favorable_proxy_product_to_point_ratio == pytest.approx(
        0.975861,
        rel=2.0e-5,
    )
    assert min(audit.folding_ratio_at_nuclear_radius for audit in audits) < 0.951
    assert max(audit.asymptotic_imaginary_momentum_form_factor for audit in audits) > 1.029
    assert all(
        not audit.full_one_percent_resolve_performed and not audit.two_body_scalar_currents_supplied
        for audit in audits
    )


def test_pb_finite_shape_proxy_spans_the_tiny_central_margin(report) -> None:
    audit = report.lead_shape_proxy

    assert audit.minimum_momentum_transfer_mev == pytest.approx(3.53095, rel=2.0e-6)
    assert audit.maximum_momentum_transfer_mev == pytest.approx(13.1777, rel=5.0e-6)
    assert audit.finite_propagator_response_minimum == pytest.approx(
        0.835031,
        rel=2.0e-6,
    )
    assert audit.finite_propagator_response_maximum == pytest.approx(
        0.986014,
        rel=2.0e-6,
    )
    assert audit.combined_shape_response_minimum < report.point_pb_response_critical
    assert audit.combined_shape_response_maximum > report.point_pb_response_critical
    assert audit.q4_weighted_shape_response_maximum < report.point_pb_response_critical
    assert report.local_pb_shape_envelope_crosses_pass_boundary
    assert report.q4_weighted_pb_proxy_allows_all_product_scenarios
    assert not audit.experimental_angular_covariance_supplied
    assert not audit.strong_amplitude_phase_profiled


def test_pb_recast_diagnostics_move_the_bound_in_opposite_directions(report) -> None:
    audit = report.lead_shape_proxy

    assert audit.angular_p_wave_projection_response_minimum == pytest.approx(
        0.909568,
        rel=2.0e-5,
    )
    assert audit.angular_p_wave_projection_response_maximum == pytest.approx(
        0.922537,
        rel=2.0e-5,
    )
    assert audit.low_energy_sigma2_finite_window_response_minimum == pytest.approx(
        1.049888,
        rel=1.0e-5,
    )
    assert audit.low_energy_sigma2_finite_window_response_maximum == pytest.approx(
        1.079729,
        rel=1.0e-5,
    )
    assert audit.low_energy_sigma2_refined_energy_grid_points == 1001
    assert audit.low_energy_sigma2_refined_angle_grid_points == 1001
    assert audit.low_energy_sigma2_grid_refinement_max_relative_shift < 4.0e-5
    assert audit.low_energy_sigma2_numerical_convergence_tolerance == 1.0e-4
    assert audit.low_energy_sigma2_numerical_convergence_pass
    assert audit.low_energy_sigma2_zero_energy_response_minimum > 1.10
    assert audit.angular_p_wave_projection_response_maximum < 1.0
    assert audit.low_energy_sigma2_finite_window_response_minimum > 1.0
    assert audit.alternative_recasts_land_on_opposite_sides_of_contact_limit
    assert report.angular_projection_proxy_allows_all_product_scenarios
    assert not report.low_energy_sigma2_proxy_allows_any_product_scenario
    assert report.pb_recast_diagnostics_disagree_on_pass_side
    assert not audit.source_analysis_finite_density_treatment_known


def test_exact_proxy_thresholds_flip_on_either_side(report) -> None:
    product_ratio = report.most_favorable_proxy_product_to_point_ratio
    critical_pb = report.most_favorable_proxy_pb_response_critical
    critical_kaon = report.most_favorable_proxy_kaon_nlo_tightening_critical

    open_cell = evaluate_margin_cell(
        dt_product_to_point_ratio=product_ratio,
        pb_shape_response_multiplier=critical_pb * (1.0 - 1.0e-8),
        kaon_nlo_tightening_factor=critical_kaon * (1.0 - 1.0e-8),
    )
    pb_closed = evaluate_margin_cell(
        dt_product_to_point_ratio=product_ratio,
        pb_shape_response_multiplier=critical_pb * (1.0 + 1.0e-8),
        kaon_nlo_tightening_factor=critical_kaon * (1.0 - 1.0e-8),
    )
    kaon_closed = evaluate_margin_cell(
        dt_product_to_point_ratio=product_ratio,
        pb_shape_response_multiplier=critical_pb * (1.0 - 1.0e-8),
        kaon_nlo_tightening_factor=critical_kaon * (1.0 + 1.0e-8),
    )

    assert open_cell.joint_proxy_conditions_satisfied
    assert not open_cell.experimental_likelihoods_supplied
    assert not open_cell.physical_gate_pass
    assert not pb_closed.neutron_proxy_condition_satisfied
    assert not pb_closed.joint_proxy_conditions_satisfied
    assert not kaon_closed.kaon_proxy_condition_satisfied
    assert not kaon_closed.joint_proxy_conditions_satisfied


def test_kaon_nlo_axis_has_no_factor_ten_pass(report) -> None:
    assert report.point_kaon_nlo_tightening_critical == pytest.approx(
        6.63708,
        rel=2.0e-5,
    )
    assert report.most_favorable_proxy_kaon_nlo_tightening_critical == pytest.approx(
        6.71866,
        rel=2.0e-5,
    )
    assert report.robust_lower_line_kaon_nlo_tightening_critical == pytest.approx(
        5.84063,
        rel=2.0e-5,
    )
    assert report.acknowledged_kaon_nlo_factor == 10.0
    assert not report.acknowledged_nlo_factor_passes_any_proxy_scenario


def test_latest_na62_br_range_uses_square_root_coupling_scaling(report) -> None:
    audit = report.latest_kaon_data
    point_old = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=1.0,
    )
    point_threefold = evaluate_margin_cell(
        dt_product_to_point_ratio=1.0,
        pb_shape_response_multiplier=1.0,
        kaon_nlo_tightening_factor=1.0,
        latest_kaon_br_improvement_factor=3.0,
    )

    assert audit.source_arxiv_identifier == "arXiv:2507.17286v2"
    assert audit.dataset_years == "2016-2022"
    assert audit.candidate_inside_low_mass_scan
    assert audit.branching_ratio_improvement_factor_minimum == 1.0
    assert audit.branching_ratio_improvement_factor_maximum == 3.0
    assert audit.coupling_bound_multiplier_minimum == pytest.approx(
        1.0 / math.sqrt(3.0),
        rel=1.0e-14,
    )
    assert audit.latest_range_coupling_bound_minimum == pytest.approx(
        182.0 / math.sqrt(3.0),
        rel=1.0e-14,
    )
    assert point_threefold.corrected_kaon_proxy_bound == pytest.approx(
        point_old.corrected_kaon_proxy_bound / math.sqrt(3.0),
        rel=1.0e-14,
    )
    assert point_threefold.kaon_nlo_tightening_critical == pytest.approx(
        point_old.kaon_nlo_tightening_critical / math.sqrt(3.0),
        rel=1.0e-14,
    )
    assert audit.figure2_pdf_page_index == 4
    assert audit.figure2_candidate_mass_curve_interpolation_entered
    assert audit.figure2_relative_readout_uncertainty == 0.05
    assert audit.figure2_readout_errors_treated_as_independent_box
    assert audit.figure2_interpolated_2016_2022_observed_br_limit == pytest.approx(
        2.4762720805654048e-11,
        rel=1.0e-9,
    )
    assert audit.figure2_interpolated_2016_2018_observed_br_limit == pytest.approx(
        3.296824399881053e-11,
        rel=1.0e-9,
    )
    assert audit.figure2_interpolated_br_improvement_factor == pytest.approx(
        audit.figure2_interpolated_2016_2018_observed_br_limit
        / audit.figure2_interpolated_2016_2022_observed_br_limit,
        rel=1.0e-14,
    )
    assert audit.figure2_interpolated_coupling_bound_multiplier == pytest.approx(
        1.0 / math.sqrt(audit.figure2_interpolated_br_improvement_factor),
        rel=1.0e-14,
    )
    assert audit.figure2_interpolated_latest_uds_bound == pytest.approx(
        182.0 * audit.figure2_interpolated_coupling_bound_multiplier,
        rel=1.0e-14,
    )
    assert audit.figure2_interpolated_point_nlo_tightening_critical == pytest.approx(
        point_old.kaon_nlo_tightening_critical
        * audit.figure2_interpolated_coupling_bound_multiplier,
        rel=1.0e-14,
    )
    uncertainty = audit.figure2_relative_readout_uncertainty
    expected_i_minimum = audit.figure2_interpolated_br_improvement_factor * (
        (1.0 - uncertainty) / (1.0 + uncertainty)
    )
    expected_i_maximum = audit.figure2_interpolated_br_improvement_factor * (
        (1.0 + uncertainty) / (1.0 - uncertainty)
    )
    assert audit.figure2_br_improvement_factor_minimum == pytest.approx(
        expected_i_minimum,
        rel=1.0e-14,
    )
    assert audit.figure2_br_improvement_factor_maximum == pytest.approx(
        expected_i_maximum,
        rel=1.0e-14,
    )
    assert audit.figure2_coupling_bound_multiplier_minimum == pytest.approx(
        1.0 / math.sqrt(expected_i_maximum),
        rel=1.0e-14,
    )
    assert audit.figure2_coupling_bound_multiplier_maximum == pytest.approx(
        1.0 / math.sqrt(expected_i_minimum),
        rel=1.0e-14,
    )
    assert audit.figure2_point_nlo_tightening_critical_minimum == pytest.approx(
        point_old.kaon_nlo_tightening_critical / math.sqrt(expected_i_maximum),
        rel=1.0e-14,
    )
    assert audit.figure2_point_nlo_tightening_critical_maximum == pytest.approx(
        point_old.kaon_nlo_tightening_critical / math.sqrt(expected_i_minimum),
        rel=1.0e-14,
    )
    assert audit.figure2_point_nlo_tightening_critical_maximum < 10.0
    assert audit.tree_level_candidate_allowed_across_improvement_range
    assert not audit.exact_candidate_mass_observed_limit_entered
    assert not audit.full_uds_operator_recast_and_nlo_likelihood_supplied
    assert not audit.latest_data_gate_pass


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt_product_to_point_ratio", 0.0),
        ("pb_shape_response_multiplier", float("nan")),
        ("kaon_nlo_tightening_factor", -1.0),
        ("kaon_digitized_bound_factor", True),
        ("latest_kaon_br_improvement_factor", float("inf")),
    ],
)
def test_arbitrary_error_axes_are_rejected(field, value) -> None:
    arguments = {
        "dt_product_to_point_ratio": 1.0,
        "pb_shape_response_multiplier": 1.0,
        "kaon_nlo_tightening_factor": 1.0,
        "kaon_digitized_bound_factor": 1.0,
        "latest_kaon_br_improvement_factor": 1.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="finite positive real"):
        evaluate_margin_cell(**arguments)


def test_margin_robustness_gate_remains_fail_closed(report) -> None:
    assert not report.full_dt_scalar_current_calculation_supplied
    assert not report.mass_specific_pb_likelihood_supplied
    assert not report.full_kaon_nlo_likelihood_supplied
    assert not report.margin_robustness_gate_pass
    assert not report.physical_ce_fusion_branch_accepted
