from __future__ import annotations

from dataclasses import replace

import pytest

from reality_stone.clarus.fusion_scalar_current_loop import (
    _certification,
    audit_scalar_primary_evidence,
    gaussian_product_form_factor,
    helm_form_factor,
    current_fusion_scalar_current_report,
    physical_scalar_primary_evidence_gate_pass,
    scalar_report_gate_results,
)


class _AlwaysEqualProxy:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


@pytest.fixture(scope="module")
def report():
    return current_fusion_scalar_current_report()


def test_primary_scalar_evidence_separates_physical_data_from_proxies(report) -> None:
    audit = report.primary_evidence

    assert audit.registered_mediator_momentum_mev == pytest.approx(29.64757)
    assert audit.physical_triton_partial_q0_two_pion_matrix_element_published
    assert audit.physical_triton_finite_q_curve_figure_only
    assert audit.heavy_pion_triton_mirror_q0_benchmark_available
    assert audit.dt_reaction_rmatrix_method_public
    assert not audit.physical_triton_full_q0_scalar_response_public
    assert not audit.machine_readable_dt_response_arrays_public
    assert not audit.joint_dt_momentum_covariance_public
    assert not audit.physical_point_short_range_contact_fitted
    assert not audit.registered_mass_current_and_potential_share_regulator
    assert not audit.full_form_factor_likelihood_supplied
    assert not audit.full_real_space_scalar_current_likelihood_supplied
    assert not audit.heavy_pion_mirror_accepted_as_physical_triton
    assert not audit.helium3_accepted_as_triton_proxy
    assert not audit.graph_digitization_accepted_for_certification
    assert not audit.low_q_extrapolation_accepted_for_full_response
    assert not audit.dt_reaction_input_and_posterior_public
    assert not audit.reaction_likelihood_is_scalar_current_likelihood
    assert not physical_scalar_primary_evidence_gate_pass(audit)


def test_primary_scalar_evidence_gate_rejects_forged_flags(report) -> None:
    forged = replace(
        report.primary_evidence,
        physical_triton_full_q0_scalar_response_public=True,
        machine_readable_dt_response_arrays_public=True,
        joint_dt_momentum_covariance_public=True,
        physical_point_short_range_contact_fitted=True,
        registered_mass_current_and_potential_share_regulator=True,
        full_form_factor_likelihood_supplied=True,
        full_real_space_scalar_current_likelihood_supplied=True,
        primary_evidence_gate_pass=True,
    )

    fresh_canonical = audit_scalar_primary_evidence()
    assert fresh_canonical == report.primary_evidence
    assert fresh_canonical is not report.primary_evidence
    assert not physical_scalar_primary_evidence_gate_pass(forged)


def test_primary_scalar_evidence_gate_rejects_always_equal_foreign_type(report) -> None:
    assert not physical_scalar_primary_evidence_gate_pass(
        _AlwaysEqualProxy(report.primary_evidence)
    )


@pytest.mark.parametrize(
    "field",
    (
        "heavy_pion_mirror_accepted_as_physical_triton",
        "helium3_accepted_as_triton_proxy",
        "graph_digitization_accepted_for_certification",
        "low_q_extrapolation_accepted_for_full_response",
        "reaction_likelihood_is_scalar_current_likelihood",
    ),
)
def test_primary_scalar_evidence_gate_rejects_every_anti_proxy_leaf(report, field) -> None:
    contradictory = replace(report.primary_evidence, **{field: True})

    assert not physical_scalar_primary_evidence_gate_pass(contradictory)


def test_scalar_primary_source_dois_versions_and_domains_are_canonical(report) -> None:
    sources = {source.key: source for source in report.sources}
    expected = {
        "broggini_2025_v2": (
            "2509.03486",
            "v2",
            "https://arxiv.org/abs/2509.03486v2",
            "https://doi.org/10.1103/ms4p-zymy",
            "scalar-phenomenology",
        ),
        "korber_2017_v1": (
            "1704.01150",
            "v1",
            "https://arxiv.org/abs/1704.01150v1",
            "https://doi.org/10.1103/PhysRevC.96.035805",
            "nuclear-scalar-response",
        ),
        "andreoli_2019_v2": (
            "1811.01843",
            "v2",
            "https://arxiv.org/abs/1811.01843v2",
            "https://doi.org/10.1103/PhysRevC.99.025501",
            "nuclear-scalar-response",
        ),
        "devries_2024_v2": (
            "2310.11343",
            "v2",
            "https://arxiv.org/abs/2310.11343v2",
            "https://doi.org/10.1140/epjc/s10052-024-13477-z",
            "nuclear-scalar-response",
        ),
        "filandri_2024_v2": (
            "2403.06599",
            "v2",
            "https://arxiv.org/abs/2403.06599v2",
            "https://doi.org/10.1103/PhysRevC.110.034002",
            "nuclear-scalar-response",
        ),
        "chakraborty_2026_v1": (
            "2603.28872",
            "v1",
            "https://arxiv.org/abs/2603.28872v1",
            "https://doi.org/10.48550/arXiv.2603.28872",
            "nuclear-scalar-response",
        ),
        "agadjanov_2024_v2": (
            "2303.08741",
            "v2",
            "https://arxiv.org/abs/2303.08741v2",
            "https://doi.org/10.1103/PhysRevLett.131.261902",
            "nucleon-scalar-input",
        ),
        "alarcon_weiss_2017_v1": (
            "1707.07682",
            "v1",
            "https://arxiv.org/abs/1707.07682v1",
            "https://doi.org/10.1103/PhysRevC.96.055206",
            "nucleon-scalar-input",
        ),
        "krebs_2020_published": (
            "2005.07433",
            "published",
            "https://doi.org/10.1140/epja/s10050-020-00249-y",
            "https://doi.org/10.1140/epja/s10050-020-00249-y",
            "nuclear-scalar-response",
        ),
        "chang_nplqcd_2018_v2": (
            "1712.03221",
            "v2",
            "https://arxiv.org/abs/1712.03221v2",
            "https://doi.org/10.1103/PhysRevLett.120.152002",
            "nuclear-scalar-response",
        ),
        "detmold_shanahan_2021_published": (
            "2102.04329",
            "published",
            "https://doi.org/10.1103/PhysRevD.103.074503",
            "https://doi.org/10.1103/PhysRevD.103.074503",
            "nuclear-scalar-response",
        ),
        "richardson_2022_v2": (
            "2110.15959",
            "v2",
            "https://arxiv.org/abs/2110.15959v2",
            "https://doi.org/10.1103/PhysRevC.106.044003",
            "nuclear-scalar-response",
        ),
        "hupin_dt_2019_v3": (
            "1803.11378",
            "v3",
            "https://arxiv.org/abs/1803.11378v3",
            "https://doi.org/10.1038/s41467-018-08052-6",
            "dt-reaction",
        ),
        "desouza_dt_2019_published": (
            "1901.04857",
            "published",
            "https://doi.org/10.1103/PhysRevC.99.014619",
            "https://doi.org/10.1103/PhysRevC.99.014619",
            "dt-reaction",
        ),
        "odell_dt_2022_v2": (
            "2105.06541",
            "v2",
            "https://arxiv.org/abs/2105.06541v2",
            "https://doi.org/10.1103/PhysRevC.105.014625",
            "dt-reaction",
        ),
    }

    assert set(sources) == set(expected)
    for key, canonical in expected.items():
        source = sources[key]
        observed = (
            source.arxiv_identifier,
            source.pinned_version,
            source.url,
            source.doi,
            source.source_domain,
        )
        assert observed == canonical
        assert source.primary_source

    assert {key for key, source in sources.items() if source.source_domain == "dt-reaction"} == {
        "hupin_dt_2019_v3",
        "desouza_dt_2019_published",
        "odell_dt_2022_v2",
    }
    assert sources["hupin_dt_2019_v3"].role.startswith("deterministic NCSMC")


def test_candidate_one_nucleon_charge_is_reproduced_but_not_renormalized_as_a_fit(report) -> None:
    audit = report.nucleon_scalar_charge

    assert audit.candidate_aligned_scale_gev == pytest.approx(8.97099, rel=2.0e-6)
    assert audit.candidate_proton_coupling == pytest.approx(0.01716644089, rel=2.0e-10)
    assert audit.candidate_neutron_coupling == pytest.approx(0.01761232247, rel=2.0e-10)
    assert audit.candidate_dt_charge_product == pytest.approx(0.001822097176, rel=2.0e-10)
    assert audit.candidate_proton_sigma_numerator_gev == pytest.approx(0.154)
    assert audit.candidate_neutron_sigma_numerator_gev == pytest.approx(0.158)
    assert audit.modern_sigma_uds_mev == pytest.approx(72.3)
    assert audit.modern_proton_equals_neutron_isoscalar_proxy_assumed
    assert audit.modern_to_candidate_isoscalar_numerator_ratio == pytest.approx(72.3 / 156.0)
    expected_product_ratio = (144.6 * 216.9) / (312.0 * 470.0)
    assert audit.fixed_scale_dt_product_ratio_diagnostic == pytest.approx(expected_product_ratio)
    assert audit.retuned_aligned_scale_gev_diagnostic == pytest.approx(4.14885404, rel=2.0e-8)
    assert not audit.proton_neutron_sigma_covariance_supplied
    assert not audit.normalization_likelihood_supplied
    assert not audit.normalization_certification_pass


def test_helm_matches_gaussian_central_curve_over_q0_to_q40(report) -> None:
    audit = report.one_body_nuclear_shape
    q0, q10, q20, qmass, q40 = audit.spacelike_points

    assert q0.helm_product == 1.0
    assert q10.helm_product == pytest.approx(0.9972576903056174, rel=2.0e-13)
    assert q20.helm_product == pytest.approx(0.989075799080547, rel=2.0e-13)
    assert qmass.momentum_transfer_mev == pytest.approx(29.64757)
    assert qmass.helm_product == pytest.approx(0.9761515492393972, rel=2.0e-13)
    assert q40.helm_product == pytest.approx(0.957014004953772, rel=2.0e-13)
    assert q40.gaussian_product == pytest.approx(0.956927970009653, rel=2.0e-13)
    assert audit.deuteron_helm_rms_radius_fm == pytest.approx(1.89563583164)
    assert audit.triton_helm_rms_radius_fm == pytest.approx(1.67992806269)
    assert audit.maximum_sampled_spacelike_relative_residual == pytest.approx(
        8.990744e-5, rel=2.0e-7
    )
    assert audit.central_spacelike_benchmark_pass
    assert not audit.ab_initio_density_covariance_supplied
    assert not audit.one_body_shape_certification_pass


def test_imaginary_momentum_central_benchmark_is_an_audit_not_a_measurement(report) -> None:
    audit = report.one_body_nuclear_shape

    assert audit.deuteron_helm_imaginary_form_factor == pytest.approx(1.01361140103)
    assert audit.triton_helm_imaginary_form_factor == pytest.approx(1.01067438303)
    assert audit.helm_imaginary_product == pytest.approx(1.02443107737)
    assert audit.gaussian_imaginary_product == pytest.approx(1.02448169890)
    assert audit.imaginary_helm_to_gaussian_relative_residual == pytest.approx(-4.94118504e-5)
    assert audit.exterior_residue_analytic_diagnostic_pass
    assert not audit.analytic_continuation_is_measurement
    assert not audit.analytic_continuation_is_full_folded_barrier_response


def test_barrier_window_records_range_and_unresolved_inner_edge(report) -> None:
    audit = report.barrier_window

    assert audit.mediator_compton_length_fm == pytest.approx(6.65575561167)
    assert audit.smallest_spatial_scale_resolved_at_qmax_fm == pytest.approx(4.93317451)
    assert audit.momentum_needed_for_inner_radius_mev == pytest.approx(60.9033890)
    assert [point.point_yukawa_exponential for point in audit.suppression_points] == pytest.approx(
        [0.6145919546, 0.4717861321, 0.2225821545, 0.04954281549, 0.000546325798]
    )
    assert not audit.q_grid_resolves_inner_radius
    assert not audit.dt_real_space_scalar_current_likelihood_supplied


def test_scalar_radius_diagnostic_crosses_band_at_q40_but_is_not_certified(report) -> None:
    audit = report.intrinsic_scalar_radius
    q40 = audit.spacelike_points[-1]

    assert q40.momentum_transfer_mev == 40.0
    assert q40.correction_at_radius_min == pytest.approx(-0.01218581977)
    assert q40.correction_at_radius_max == pytest.approx(-0.01280673197)
    assert q40.strange_slope_one_sigma == pytest.approx(0.004426002766)
    assert q40.exact_coupling_correction_at_radius_min == pytest.approx(0.01233614582)
    assert q40.exact_coupling_correction_at_radius_max == pytest.approx(0.01297287206)
    assert audit.imaginary_correction_at_radius_min == pytest.approx(0.006694420281)
    assert audit.imaginary_correction_at_radius_max == pytest.approx(0.007035525539)
    assert audit.imaginary_exact_coupling_correction_at_radius_min == pytest.approx(-0.006649903035)
    assert audit.imaginary_exact_coupling_correction_at_radius_max == pytest.approx(-0.006986372736)
    assert audit.q40_coupling_correction_exceeds_comparison_band
    assert not audit.scalar_radius_covariance_supplied
    assert not audit.full_form_factor_likelihood_supplied
    assert not audit.scalar_radius_certification_pass


def test_latest_sigma_proxy_is_plus_1p11_pm_1p48_percent_with_assumptions_explicit(
    report,
) -> None:
    audit = report.sigma_term_proxy
    assumptions = audit.assumptions

    assert audit.deuteron_sigma_ratio_total_std == pytest.approx(0.04143669871)
    assert audit.helium3_sigma_ratio_total_std == pytest.approx(0.12929423808)
    assert audit.uds_light_dilution_central == pytest.approx(0.604426002766)
    assert audit.dt_product_correction == pytest.approx(-0.02175199674)
    assert audit.dt_product_correction_std == pytest.approx(0.02864839294)
    assert audit.required_common_coupling_correction == pytest.approx(0.01105670805)
    assert audit.required_common_coupling_correction_std == pytest.approx(0.01480460464)
    assert audit.required_common_coupling_correction_one_sigma_upper == pytest.approx(0.02586131269)
    assert assumptions.helium3_used_as_triton_isospin_proxy
    assert assumptions.deuteron_and_helium3_errors_treated_independent
    assert assumptions.sigma_pi_and_sigma_strange_central_dilution_only
    assert not assumptions.sigma_pi_sigma_strange_uncertainty_propagated
    assert assumptions.evaluated_at_zero_momentum_only
    assert assumptions.first_order_gaussian_error_propagation
    assert not assumptions.actual_triton_sigma_term_supplied
    assert not assumptions.dt_covariance_supplied
    assert not audit.central_correction_exceeds_comparison_band
    assert audit.one_sigma_upper_exceeds_comparison_band
    assert not audit.diagnostic_valid_for_certification


def test_chiral_two_body_centrals_exist_but_missing_contact_and_dt_likelihood_fail(report) -> None:
    audit = report.two_body_scalar_current

    assert audit.andreoli_deuteron_q0_two_body_fraction_min == pytest.approx(0.007)
    assert audit.andreoli_deuteron_q0_two_body_fraction_max == pytest.approx(0.030)
    assert audit.korber_higher_order_deuteron_squared_response_central == pytest.approx(0.016)
    assert audit.korber_higher_order_deuteron_squared_response_std == pytest.approx(0.008)
    assert audit.linearized_modern_uds_deuteron_amplitude_correction_min == pytest.approx(
        0.002115491010
    )
    assert audit.linearized_modern_uds_deuteron_amplitude_correction_max == pytest.approx(
        0.009066390041
    )
    assert audit.exact_modern_uds_deuteron_amplitude_correction_min == pytest.approx(
        0.604426002766 * ((1.0 - 0.007) ** -0.5 - 1.0)
    )
    assert audit.exact_modern_uds_deuteron_amplitude_correction_max == pytest.approx(
        0.604426002766 * ((1.0 - 0.030) ** -0.5 - 1.0)
    )
    assert audit.filandri_reference_momentum_mev == pytest.approx(9.86634902)
    assert audit.filandri_momentum_coverage_max_mev == pytest.approx(39.46539608)
    assert not audit.triton_two_body_sign_stable_across_regulators
    assert not audit.unknown_short_range_two_nucleon_contact_resolved
    assert not audit.momentum_dependent_dt_joint_likelihood_supplied
    assert not audit.two_body_covariance_supplied
    assert not audit.two_body_certification_pass


def test_provenance_and_proxy_assumptions_serialize_separately(report) -> None:
    payload = report.to_dict()
    source_keys = {source["key"] for source in payload["sources"]}

    assert "chakraborty_2026_v1" in source_keys
    assert "agadjanov_2024_v2" in source_keys
    assert payload["sigma_term_proxy"]["source_keys"] == (
        "chakraborty_2026_v1",
        "agadjanov_2024_v2",
    )
    assert payload["sigma_term_proxy"]["assumptions"]["helium3_used_as_triton_isospin_proxy"]
    assert not payload["certification"]["comparison_band_is_statistical_confidence_interval"]
    assert "physical_ce_fusion_branch_accepted" not in payload["certification"]
    assert not payload["certification"]["component_and_upstream_gate_pass"]


def test_scalar_current_gate_remains_fail_closed(report) -> None:
    audit = report.certification
    expected_required_inputs = all(
        (
            audit.actual_triton_q0_sigma_term_supplied,
            audit.momentum_dependent_dt_covariance_supplied,
            audit.calibrated_two_body_contact_supplied,
            audit.full_real_space_barrier_response_supplied,
        )
    )
    expected_certification = all(
        (
            expected_required_inputs,
            audit.nucleon_normalization_leaf_gate_pass,
            audit.one_body_shape_leaf_gate_pass,
            audit.scalar_radius_leaf_gate_pass,
            audit.triton_sigma_response_leaf_gate_pass,
            audit.two_body_leaf_gate_pass,
        )
    )
    expected_component_and_upstream = all(
        (
            expected_certification,
            audit.upstream_uv_action_gate_pass,
            audit.upstream_existing_constraints_gate_pass,
        )
    )
    expected_evidence = physical_scalar_primary_evidence_gate_pass(report.primary_evidence)
    expected_report_gates = scalar_report_gate_results(audit)

    assert audit.helm_gaussian_central_benchmark_pass
    assert audit.legacy_reference_gaussian_coupling_correction == pytest.approx(-0.009886663376)
    assert audit.legacy_morphology_coupling_correction_min == pytest.approx(-0.01214337557)
    assert audit.legacy_morphology_coupling_correction_max == pytest.approx(-0.008308186357)
    assert audit.comparison_band_absolute_coupling_correction == 0.012
    assert audit.sigma_proxy_central_within_comparison_band
    assert not audit.sigma_proxy_one_sigma_upper_within_comparison_band
    assert not audit.scalar_radius_q40_within_comparison_band
    assert not audit.actual_triton_q0_sigma_term_supplied
    assert not audit.momentum_dependent_dt_covariance_supplied
    assert not audit.calibrated_two_body_contact_supplied
    assert not audit.full_real_space_barrier_response_supplied
    assert audit.all_required_scalar_current_inputs_supplied is expected_required_inputs
    assert audit.scalar_current_certification_pass is expected_certification
    assert not audit.upstream_uv_action_gate_pass
    assert not audit.upstream_existing_constraints_gate_pass
    assert audit.component_and_upstream_gate_pass is expected_component_and_upstream
    assert expected_report_gates.canonical_primary_evidence_gate_pass is expected_evidence
    assert expected_report_gates.scalar_current_loop_closed is (
        expected_certification and expected_evidence
    )
    assert expected_report_gates.physical_ce_fusion_branch_accepted is (
        expected_component_and_upstream and expected_evidence
    )
    assert report.scalar_current_loop_closed is expected_report_gates.scalar_current_loop_closed
    assert (
        report.physical_ce_fusion_branch_accepted
        is expected_report_gates.physical_ce_fusion_branch_accepted
    )
    assert not expected_certification


def test_physical_scalar_gate_requires_nuclear_uv_and_external_constraint_conjunction(
    report,
) -> None:
    proxy = replace(
        report.sigma_term_proxy,
        assumptions=replace(
            report.sigma_term_proxy.assumptions,
            helium3_used_as_triton_isospin_proxy=False,
            sigma_pi_and_sigma_strange_central_dilution_only=False,
            sigma_pi_sigma_strange_uncertainty_propagated=True,
            evaluated_at_zero_momentum_only=False,
            first_order_gaussian_error_propagation=False,
            actual_triton_sigma_term_supplied=True,
            dt_covariance_supplied=True,
        ),
        diagnostic_valid_for_certification=True,
    )
    complete = dict(
        nucleon=replace(
            report.nucleon_scalar_charge,
            modern_proton_equals_neutron_isoscalar_proxy_assumed=False,
            proton_neutron_sigma_covariance_supplied=True,
            modern_sigma_term_covariance_supplied=True,
            normalization_likelihood_supplied=True,
            normalization_certification_pass=True,
        ),
        shape=replace(
            report.one_body_nuclear_shape,
            ab_initio_density_covariance_supplied=True,
            one_body_shape_certification_pass=True,
        ),
        barrier=replace(
            report.barrier_window,
            dt_real_space_scalar_current_likelihood_supplied=True,
        ),
        radius=replace(
            report.intrinsic_scalar_radius,
            scalar_radius_covariance_supplied=True,
            full_form_factor_likelihood_supplied=True,
            scalar_radius_certification_pass=True,
        ),
        proxy=proxy,
        two_body=replace(
            report.two_body_scalar_current,
            triton_two_body_sign_stable_across_regulators=True,
            unknown_short_range_two_nucleon_contact_resolved=True,
            regulator_consistent_current_and_potential_supplied=True,
            momentum_dependent_dt_joint_likelihood_supplied=True,
            two_body_covariance_supplied=True,
            two_body_certification_pass=True,
        ),
    )

    all_true = _certification(
        **complete,
        upstream_uv_action_gate_pass=True,
        upstream_existing_constraints_gate_pass=True,
    )
    assert all_true.scalar_current_certification_pass
    assert all_true.component_and_upstream_gate_pass
    all_true_report_gates = scalar_report_gate_results(all_true)
    assert not all_true_report_gates.canonical_primary_evidence_gate_pass
    assert not all_true_report_gates.scalar_current_loop_closed
    assert not all_true_report_gates.physical_ce_fusion_branch_accepted

    for upstream_field in (
        "upstream_uv_action_gate_pass",
        "upstream_existing_constraints_gate_pass",
    ):
        upstream = {
            "upstream_uv_action_gate_pass": True,
            "upstream_existing_constraints_gate_pass": True,
        }
        upstream[upstream_field] = False
        audit = _certification(**complete, **upstream)
        assert audit.scalar_current_certification_pass
        assert not audit.component_and_upstream_gate_pass
        assert not scalar_report_gate_results(audit).physical_ce_fusion_branch_accepted

    component_leaf_fields = {
        "nucleon": (
            "proton_neutron_sigma_covariance_supplied",
            "modern_sigma_term_covariance_supplied",
            "normalization_likelihood_supplied",
            "normalization_certification_pass",
        ),
        "shape": (
            "ab_initio_density_covariance_supplied",
            "one_body_shape_certification_pass",
        ),
        "barrier": ("dt_real_space_scalar_current_likelihood_supplied",),
        "radius": (
            "scalar_radius_covariance_supplied",
            "full_form_factor_likelihood_supplied",
            "scalar_radius_certification_pass",
        ),
        "two_body": (
            "chiral_two_body_operator_recorded",
            "triton_two_body_sign_stable_across_regulators",
            "unknown_short_range_two_nucleon_contact_resolved",
            "regulator_consistent_current_and_potential_supplied",
            "momentum_dependent_dt_joint_likelihood_supplied",
            "two_body_covariance_supplied",
            "two_body_certification_pass",
        ),
    }
    for component, fields in component_leaf_fields.items():
        for field in fields:
            failed = dict(complete)
            failed[component] = replace(complete[component], **{field: False})
            audit = _certification(
                **failed,
                upstream_uv_action_gate_pass=True,
                upstream_existing_constraints_gate_pass=True,
            )
            assert not audit.scalar_current_certification_pass
            assert not audit.component_and_upstream_gate_pass
            assert not scalar_report_gate_results(audit).scalar_current_loop_closed

    bad_nucleon = dict(complete)
    bad_nucleon["nucleon"] = replace(
        complete["nucleon"],
        modern_proton_equals_neutron_isoscalar_proxy_assumed=True,
    )
    assert not _certification(
        **bad_nucleon,
        upstream_uv_action_gate_pass=True,
        upstream_existing_constraints_gate_pass=True,
    ).scalar_current_certification_pass

    bad_shape = dict(complete)
    bad_shape["shape"] = replace(complete["shape"], analytic_continuation_is_measurement=True)
    assert not _certification(
        **bad_shape,
        upstream_uv_action_gate_pass=True,
        upstream_existing_constraints_gate_pass=True,
    ).scalar_current_certification_pass
    bad_shape = dict(complete)
    bad_shape["shape"] = replace(
        complete["shape"],
        analytic_continuation_is_full_folded_barrier_response=True,
    )
    assert not _certification(
        **bad_shape,
        upstream_uv_action_gate_pass=True,
        upstream_existing_constraints_gate_pass=True,
    ).scalar_current_certification_pass

    assumption_failures = {
        "helium3_used_as_triton_isospin_proxy": True,
        "sigma_pi_and_sigma_strange_central_dilution_only": True,
        "sigma_pi_sigma_strange_uncertainty_propagated": False,
        "evaluated_at_zero_momentum_only": True,
        "first_order_gaussian_error_propagation": True,
        "actual_triton_sigma_term_supplied": False,
        "dt_covariance_supplied": False,
    }
    for field, value in assumption_failures.items():
        failed = dict(complete)
        failed["proxy"] = replace(
            proxy,
            assumptions=replace(proxy.assumptions, **{field: value}),
        )
        audit = _certification(
            **failed,
            upstream_uv_action_gate_pass=True,
            upstream_existing_constraints_gate_pass=True,
        )
        assert not audit.scalar_current_certification_pass

    failed = dict(complete)
    failed["proxy"] = replace(proxy, diagnostic_valid_for_certification=False)
    assert not _certification(
        **failed,
        upstream_uv_action_gate_pass=True,
        upstream_existing_constraints_gate_pass=True,
    ).scalar_current_certification_pass


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (helm_form_factor, {"mass_number": 2, "diffuseness_fm": 0.47, "skin_thickness_fm": 1.09}),
        (gaussian_product_form_factor, {}),
    ],
)
def test_form_factor_public_inputs_reject_negative_and_boolean_momentum(function, kwargs) -> None:
    with pytest.raises(ValueError):
        function(-1.0, **kwargs)
    with pytest.raises(ValueError):
        function(True, **kwargs)


def test_helm_rejects_invalid_geometry_and_nonboolean_continuation_flag() -> None:
    with pytest.raises(ValueError):
        helm_form_factor(
            10.0,
            mass_number=True,
            diffuseness_fm=0.47,
            skin_thickness_fm=1.09,
        )
    with pytest.raises(ValueError):
        helm_form_factor(
            10.0,
            mass_number=2,
            diffuseness_fm=0.01,
            skin_thickness_fm=2.0,
        )
    with pytest.raises(ValueError):
        gaussian_product_form_factor(10.0, imaginary_momentum=1)
