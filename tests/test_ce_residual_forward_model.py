from __future__ import annotations

import math

from examples.physics.ce_residual_forward_model import (
    BAODataPoint,
    C_KM_S,
    CEForwardParams,
    DEFAULT_N_EFF,
    ForwardCoverage,
    MPC_METERS,
    NEUTRINO_RADIATION_FACTOR,
    OMEGA_GAMMA_H2_REFERENCE,
    TCMB_REFERENCE_K,
    assess_bao_fit,
    bao_chi2,
    bao_chi2_with_covariance,
    bao_observable,
    baryon_photon_sound_speed_km_s,
    chapter1_canonical_params,
    chi_square_survival,
    chi_square_verdict,
    dark_energy_scale,
    early_hubble_rate_s_inverse,
    early_universe_sound_horizon,
    e_of_z,
    eisenstein_hu_drag_redshift,
    f_sigma8_at_z,
    hubble_distance_mpc,
    invert_matrix,
    line_of_sight_comoving_distance_mpc,
    luminosity_distance_mpc,
    named_bao_dataset,
    parameter_provenance,
    parse_bao_data,
    parse_covariance_matrix,
    photon_density_h2,
    radiation_density_h2,
    s8_today,
    solve_growth,
    sound_horizon_selection,
    transverse_comoving_distance_mpc,
    volume_distance_mpc,
)


def test_lcdm_limit_has_constant_dark_energy_density() -> None:
    for a in (0.25, 0.5, 1.0):
        assert math.isclose(dark_energy_scale(a, w0=-1.0, wa=0.0), 1.0, rel_tol=1e-12)


def test_ce_forward_model_uses_ce_density_ratios_for_background() -> None:
    params = CEForwardParams()

    assert math.isclose(params.omega_m0, 0.3110, abs_tol=1e-12)
    assert params.is_flat
    assert math.isclose(e_of_z(0.0, params), 1.0, rel_tol=1e-12)
    assert e_of_z(1.0, params) > 1.0
    assert luminosity_distance_mpc(1.0, params) > 6000.0


def test_legacy_early_mode_closes_residual_density_as_curvature() -> None:
    params = CEForwardParams(rd_mode="early-universe")
    background_sum = (
        params.omega_r0_background
        + params.omega_m0_background
        + params.omega_lambda0_background
        + params.omega_k0_background
    )
    h0_from_early_background = (
        early_hubble_rate_s_inverse(0.0, params) * MPC_METERS / 1000.0
    )

    assert math.isclose(background_sum, 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(e_of_z(0.0, params), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(
        h0_from_early_background,
        params.h0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert params.omega_k0_background < 0.0
    assert not params.is_flat

    z = 2.33
    radial_distance = line_of_sight_comoving_distance_mpc(z, params)
    transverse_distance = transverse_comoving_distance_mpc(z, params)
    hubble_distance_today = C_KM_S / params.h0
    expected_transverse = (
        hubble_distance_today
        * math.sin(
            math.sqrt(-params.omega_k0_background)
            * radial_distance
            / hubble_distance_today
        )
        / math.sqrt(-params.omega_k0_background)
    )
    assert math.isclose(
        transverse_distance,
        5781.325559257,
        rel_tol=1e-11,
    )
    assert math.isclose(transverse_distance, expected_transverse, rel_tol=1e-14)
    assert transverse_distance < radial_distance


def test_open_early_background_uses_hyperbolic_transverse_distance() -> None:
    params = CEForwardParams(
        omega_lambda0=0.68,
        rd_mode="early-universe",
    )
    radial_distance = line_of_sight_comoving_distance_mpc(2.0, params)
    transverse_distance = transverse_comoving_distance_mpc(2.0, params)

    assert params.omega_k0_background > 0.0
    assert not params.is_flat
    assert transverse_distance > radial_distance


def test_ce_s8_today_is_close_to_combined_cmb_s8_baseline() -> None:
    params = CEForwardParams()

    assert abs(s8_today(params) - 0.836) < 0.02


def test_growth_solution_is_normalized_and_monotone() -> None:
    params = CEForwardParams()
    a_grid, d_grid, f_grid = solve_growth(params, n=801)

    assert math.isclose(d_grid[-1], 1.0, rel_tol=1e-12)
    assert 0.0 < d_grid[0] < d_grid[len(d_grid) // 2] < d_grid[-1]
    assert 0.4 < f_grid[-1] < 0.7
    assert f_sigma8_at_z(0.5, params, a_grid, d_grid, f_grid) > 0.3


def test_dynamic_dark_energy_changes_background_observables() -> None:
    lcdm = CEForwardParams(w0=-1.0, wa=0.0)
    dynamic = CEForwardParams(w0=-0.8, wa=-0.7)

    assert not math.isclose(e_of_z(1.0, lcdm), e_of_z(1.0, dynamic), rel_tol=1e-3)
    assert not math.isclose(
        luminosity_distance_mpc(1.0, lcdm),
        luminosity_distance_mpc(1.0, dynamic),
        rel_tol=1e-3,
    )


def test_bao_observable_uses_consistent_distance_definitions() -> None:
    params = CEForwardParams()
    z = 0.8
    bao = bao_observable(z, params)

    dm = transverse_comoving_distance_mpc(z, params)
    dh = hubble_distance_mpc(z, params)
    dv = volume_distance_mpc(z, params)

    assert math.isclose(bao.dm_over_rd, dm / params.rd_mpc, rel_tol=1e-12)
    assert math.isclose(bao.dh_over_rd, dh / params.rd_mpc, rel_tol=1e-12)
    assert math.isclose(bao.dv_over_rd, dv / params.rd_mpc, rel_tol=1e-12)
    assert 15.0 < bao.dm_over_rd < 25.0
    assert 15.0 < bao.dh_over_rd < 25.0


def test_early_universe_radiation_and_sound_speed_relations() -> None:
    omega_gamma_h2 = photon_density_h2(TCMB_REFERENCE_K)
    omega_r_h2 = radiation_density_h2(TCMB_REFERENCE_K, DEFAULT_N_EFF)

    assert math.isclose(omega_gamma_h2, OMEGA_GAMMA_H2_REFERENCE, rel_tol=1e-15)
    assert math.isclose(
        photon_density_h2(2.0 * TCMB_REFERENCE_K),
        16.0 * omega_gamma_h2,
        rel_tol=1e-15,
    )
    assert math.isclose(
        omega_r_h2,
        omega_gamma_h2 * (1.0 + NEUTRINO_RADIATION_FACTOR * DEFAULT_N_EFF),
        rel_tol=1e-15,
    )
    assert math.isclose(
        baryon_photon_sound_speed_km_s(0.0, 0.022, omega_gamma_h2),
        C_KM_S / math.sqrt(3.0),
        rel_tol=1e-15,
    )
    assert (
        baryon_photon_sound_speed_km_s(1.0e-3, 0.022, omega_gamma_h2)
        < C_KM_S / math.sqrt(3.0)
    )


def test_early_universe_sound_horizon_matches_analytic_sanity_gate() -> None:
    params = chapter1_canonical_params(rd_mode="early-universe")
    result = early_universe_sound_horizon(params)
    coarse = early_universe_sound_horizon(params, integration_points=501)

    assert math.isclose(
        result.z_drag,
        eisenstein_hu_drag_redshift(params.omega_m_h2, params.omega_b_h2),
        rel_tol=1e-15,
    )
    assert 1000.0 < result.z_drag < 1050.0
    assert 149.0 < result.rd_mpc < 153.0
    assert math.isclose(coarse.rd_mpc, result.rd_mpc, rel_tol=2e-8)

    baryon_loading_coefficient = (
        3.0 * result.omega_b_h2 / (4.0 * result.omega_gamma_h2)
    )
    photon_loading_scale = 1.0 / baryon_loading_coefficient
    equality_scale = result.omega_r0 / params.omega_m0
    analytic_matter_radiation_rd = (
        2.0
        * C_KM_S
        / (
            params.h0
            * math.sqrt(
                3.0 * baryon_loading_coefficient * params.omega_m0
            )
        )
        * math.log(
            (
                math.sqrt(result.a_drag + photon_loading_scale)
                + math.sqrt(result.a_drag + equality_scale)
            )
            / (
                math.sqrt(photon_loading_scale)
                + math.sqrt(equality_scale)
            )
        )
    )
    assert math.isclose(result.rd_mpc, analytic_matter_radiation_rd, rel_tol=2e-9)
    assert "DESI-independent" in result.status
    assert "not a CE-internal" in result.status
    assert "precision recombination" in result.status
    assert "DR2 is not an untouched holdout" in result.status

    try:
        early_universe_sound_horizon(params, integration_points=500)
    except ValueError as exc:
        assert "odd integer" in str(exc)
    else:
        raise AssertionError("even Simpson grid did not raise")


def test_sound_horizon_modes_and_early_input_provenance_are_separate() -> None:
    external_params = CEForwardParams()
    early_params = chapter1_canonical_params(rd_mode="early-universe")
    external = sound_horizon_selection(external_params)
    early = sound_horizon_selection(early_params)
    provenance = {entry.name: entry for entry in parameter_provenance(early_params)}

    assert external.mode == "external"
    assert external.role == "external_input"
    assert external.early_universe is None
    assert math.isclose(external.rd_mpc, external_params.rd_mpc, rel_tol=1e-15)
    assert early.mode == "early-universe"
    assert early.role == "derived_selection"
    assert early.early_universe is not None
    assert not math.isclose(early.rd_mpc, early_params.rd_mpc, rel_tol=1e-3)
    assert provenance["rd_mpc"].role == "derived_selection"
    assert provenance["omega_b0"].role == "ce_prediction"
    assert provenance["omega_dm0"].role == "derived_selection"
    assert provenance["omega_lambda0"].role == "derived_selection"
    assert "chapter1_canonical_params" in provenance["omega_dm0"].source
    assert provenance["h0"].role == "external_input"
    assert provenance["tcmb_k"].role == "external_input"
    assert provenance["n_eff"].role == "model_assumption"
    assert "external H0/Tcmb + Standard-Model Neff" in provenance["rd_mpc"].source
    assert "precision recombination" in provenance["rd_mpc"].note


def test_bao_observable_is_sensitive_to_dynamic_dark_energy() -> None:
    z = 1.0
    lcdm = bao_observable(z, CEForwardParams(w0=-1.0, wa=0.0))
    dynamic = bao_observable(z, CEForwardParams(w0=-0.8, wa=-0.7))

    assert abs(lcdm.dm_over_rd - dynamic.dm_over_rd) > 0.05
    assert abs(lcdm.dh_over_rd - dynamic.dh_over_rd) > 0.01


def test_bao_diagonal_chi2_is_zero_for_fiducial_generated_data() -> None:
    params = CEForwardParams()
    point = bao_observable(1.0, params)
    data = (
        BAODataPoint(z=1.0, kind="dm", value=point.dm_over_rd, sigma=0.2),
        BAODataPoint(z=1.0, kind="dh", value=point.dh_over_rd, sigma=0.2),
        BAODataPoint(z=1.0, kind="dv", value=point.dv_over_rd, sigma=0.2),
    )

    assert math.isclose(bao_chi2(data, params), 0.0, abs_tol=1e-12)


def test_bao_diagonal_chi2_detects_dynamic_de_shift() -> None:
    fiducial = CEForwardParams(w0=-1.0, wa=0.0)
    dynamic = CEForwardParams(w0=-0.8, wa=-0.7)
    point = bao_observable(1.0, fiducial)
    data = (
        BAODataPoint(z=1.0, kind="dm", value=point.dm_over_rd, sigma=0.05),
        BAODataPoint(z=1.0, kind="dh", value=point.dh_over_rd, sigma=0.05),
    )

    assert bao_chi2(data, dynamic) > 1.0


def test_bao_full_covariance_matches_diagonal_chi2_for_diagonal_covariance() -> None:
    fiducial = CEForwardParams(w0=-1.0, wa=0.0)
    dynamic = CEForwardParams(w0=-0.8, wa=-0.7)
    point = bao_observable(1.0, fiducial)
    data = (
        BAODataPoint(z=1.0, kind="dm", value=point.dm_over_rd, sigma=0.05),
        BAODataPoint(z=1.0, kind="dh", value=point.dh_over_rd, sigma=0.10),
    )
    covariance = ((0.05**2, 0.0), (0.0, 0.10**2))

    assert math.isclose(
        bao_chi2_with_covariance(data, covariance, dynamic),
        bao_chi2(data, dynamic),
        rel_tol=1e-10,
    )


def test_bao_full_covariance_responds_to_correlations() -> None:
    fiducial = CEForwardParams(w0=-1.0, wa=0.0)
    dynamic = CEForwardParams(w0=-0.8, wa=-0.7)
    point = bao_observable(1.0, fiducial)
    data = (
        BAODataPoint(z=1.0, kind="dm", value=point.dm_over_rd, sigma=0.10),
        BAODataPoint(z=1.0, kind="dh", value=point.dh_over_rd, sigma=0.10),
    )
    diagonal = ((0.01, 0.0), (0.0, 0.01))
    correlated = ((0.01, 0.006), (0.006, 0.01))

    diagonal_chi2 = bao_chi2_with_covariance(data, diagonal, dynamic)
    correlated_chi2 = bao_chi2_with_covariance(data, correlated, dynamic)

    assert not math.isclose(diagonal_chi2, correlated_chi2, rel_tol=1e-6)


def test_chi_square_survival_and_preregistered_verdict_boundaries() -> None:
    assert math.isclose(chi_square_survival(0.0, 5), 1.0)
    assert math.isclose(
        chi_square_survival(5.991464547107979, 2),
        0.05,
        rel_tol=1e-12,
    )
    assert chi_square_verdict(0.05) == "PASS"
    assert chi_square_verdict(0.01) == "TENSION"
    assert chi_square_verdict(0.001) == "REJECT"


def test_bao_assessment_decomposes_diagonal_chi2_into_raw_pull_squares() -> None:
    fiducial = CEForwardParams()
    point = bao_observable(1.0, fiducial)
    data = (
        BAODataPoint(z=1.0, kind="dm", value=point.dm_over_rd + 0.1, sigma=0.05),
        BAODataPoint(z=1.0, kind="dh", value=point.dh_over_rd - 0.2, sigma=0.10),
    )

    assessment = assess_bao_fit(data, fiducial)

    assert assessment.covariance_mode == "diagonal"
    assert assessment.n_observations == 2
    assert assessment.fitted_parameter_count == 0
    assert assessment.dof == 2
    assert math.isclose(
        assessment.chi2,
        sum(item.raw_pull**2 for item in assessment.contributions),
        rel_tol=1e-12,
    )
    assert all(
        math.isclose(item.covariance_contribution, item.raw_pull**2, rel_tol=1e-12)
        for item in assessment.contributions
    )


def test_desi_dr2_full_covariance_fixed_model_is_rejected_and_decomposed() -> None:
    dataset = named_bao_dataset("desi-dr2-all")
    assessment = assess_bao_fit(
        dataset.data,
        CEForwardParams(),
        covariance=dataset.covariance,
    )

    assert assessment.n_observations == 13
    assert assessment.fitted_parameter_count == 0
    assert assessment.dof == 13
    assert math.isclose(assessment.chi2, 37.100260857, rel_tol=1e-10)
    assert math.isclose(assessment.reduced_chi2, 2.85386621977, rel_tol=1e-10)
    assert math.isclose(assessment.survival_p_value, 0.000399573259846, rel_tol=1e-10)
    assert assessment.verdict == "REJECT"
    assert math.isclose(assessment.aic, assessment.chi2, rel_tol=1e-12)
    assert math.isclose(assessment.bic, assessment.chi2, rel_tol=1e-12)
    assert len(assessment.contributions) == len(dataset.data)
    assert math.isclose(
        sum(item.covariance_contribution for item in assessment.contributions),
        assessment.chi2,
        rel_tol=1e-12,
    )
    assert all(
        math.isclose(item.residual / item.sigma, item.raw_pull, rel_tol=1e-12)
        for item in assessment.contributions
    )
    assert any(
        not math.isclose(
            item.covariance_contribution,
            item.raw_pull**2,
            rel_tol=1e-3,
            abs_tol=1e-3,
        )
        for item in assessment.contributions
    )
    scale_fit = assessment.scale_fit_diagnostic
    assert scale_fit is not None
    assert math.isclose(scale_fit.scale_factor, 0.98647693346963, rel_tol=1e-12)
    assert math.isclose(scale_fit.chi2, 12.6083468622, rel_tol=1e-10)
    assert math.isclose(scale_fit.chi2_improvement, 24.4919139948, rel_tol=1e-10)
    assert scale_fit.additional_fitted_parameter_count == 1
    assert scale_fit.dof == 12
    assert math.isclose(scale_fit.reduced_chi2, 1.05069557185, rel_tol=1e-10)
    assert math.isclose(scale_fit.survival_p_value, 0.398138192515, rel_tol=1e-10)
    assert scale_fit.verdict == "PASS"
    assert math.isclose(scale_fit.aic, 14.6083468622, rel_tol=1e-10)
    assert math.isclose(scale_fit.bic, 15.1732962197, rel_tol=1e-10)
    assert math.isclose(scale_fit.aic_improvement, 22.4919139948, rel_tol=1e-10)
    assert math.isclose(scale_fit.bic_improvement, 21.9269646373, rel_tol=1e-10)
    assert math.isclose(
        scale_fit.equivalent_rd_mpc_at_fixed_h0,
        149.106375435,
        rel_tol=1e-10,
    )
    assert math.isclose(
        scale_fit.equivalent_h0_at_fixed_rd,
        68.3239493122,
        rel_tol=1e-10,
    )
    assert "not a CE prediction" in scale_fit.note


def test_desi_dr2_early_branch_uses_one_registered_radiation_background() -> None:
    params = chapter1_canonical_params(rd_mode="early-universe")
    precomputed_rd = early_universe_sound_horizon(params)
    dataset = named_bao_dataset("desi-dr2-all")
    early_assessment = assess_bao_fit(
        dataset.data,
        params,
        covariance=dataset.covariance,
    )
    external_assessment = assess_bao_fit(
        dataset.data,
        CEForwardParams(),
        covariance=dataset.covariance,
    )

    assert "DESI-independent" in precomputed_rd.status
    assert "DR2 is not an untouched holdout" in precomputed_rd.status
    assert params.includes_radiation_background
    assert params.omega_r0_background > 0.0
    assert params.is_flat
    assert math.isclose(e_of_z(0.0, params), 1.0, rel_tol=1e-15)
    assert math.isclose(precomputed_rd.rd_mpc, 151.508428775, rel_tol=1e-10)
    assert math.isclose(early_assessment.chi2, 41.9060773313, rel_tol=1e-10)
    assert math.isclose(
        early_assessment.survival_p_value,
        0.0000678476333988,
        rel_tol=1e-10,
    )
    assert early_assessment.verdict == "REJECT"
    assert math.isclose(early_assessment.aic, early_assessment.chi2, rel_tol=1e-15)
    assert math.isclose(early_assessment.bic, early_assessment.chi2, rel_tol=1e-15)
    assert early_assessment.chi2 > external_assessment.chi2
    assert early_assessment.aic > external_assessment.aic
    assert early_assessment.bic > external_assessment.bic
    early_scale_fit = early_assessment.scale_fit_diagnostic
    external_scale_fit = external_assessment.scale_fit_diagnostic
    assert early_scale_fit is not None
    assert external_scale_fit is not None
    assert math.isclose(early_scale_fit.chi2, 12.3135217704, rel_tol=1e-10)
    assert math.isclose(
        early_scale_fit.equivalent_rd_mpc_at_fixed_h0,
        149.225436583,
        rel_tol=1e-10,
    )
    assert not math.isclose(
        early_scale_fit.chi2,
        external_scale_fit.chi2,
        rel_tol=1e-6,
    )


def test_covariance_parser_and_inverter_validate_matrix() -> None:
    matrix = parse_covariance_matrix("0.04,0.01;0.01,0.09")
    inv = invert_matrix(matrix)

    assert len(inv) == 2
    assert inv[0][0] > 0.0
    assert inv[1][1] > 0.0

    try:
        parse_covariance_matrix("0.04,0.01;0.02,0.09")
    except ValueError as exc:
        assert "symmetric" in str(exc)
    else:
        raise AssertionError("asymmetric covariance did not raise")


def test_named_desi_dr2_bao_datasets_are_available() -> None:
    bgs = named_bao_dataset("desi-dr2-bgs")
    all_data = named_bao_dataset("desi-dr2-all")

    assert len(bgs.data) == 1
    assert bgs.data[0].kind == "dv"
    assert math.isclose(bgs.data[0].z, 0.295)
    assert len(all_data.data) == 13
    assert len(all_data.covariance) == 13
    assert all_data.data[1].kind == "dm"
    assert all_data.data[2].kind == "dh"


def test_named_desi_dr2_bao_dataset_chi2_runs_against_ce_model() -> None:
    dataset = named_bao_dataset("desi-dr2-bgs")
    chi2 = bao_chi2_with_covariance(dataset.data, dataset.covariance, CEForwardParams())

    assert chi2 >= 0.0
    assert math.isfinite(chi2)


def test_unknown_named_bao_dataset_raises() -> None:
    try:
        named_bao_dataset("not-a-dataset")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown BAO dataset did not raise")


def test_parse_bao_data_validates_kind_and_sigma() -> None:
    data = parse_bao_data("0.5:dm:13.2:0.1,1.0:dh:16.9:0.2")

    assert data[0] == BAODataPoint(z=0.5, kind="dm", value=13.2, sigma=0.1)
    assert data[1] == BAODataPoint(z=1.0, kind="dh", value=16.9, sigma=0.2)

    try:
        parse_bao_data("0.5:bad:13.2:0.1")
    except ValueError as exc:
        assert "kind" in str(exc)
    else:
        raise AssertionError("invalid BAO kind did not raise")

    try:
        parse_bao_data("0.5:dm:13.2:0")
    except ValueError as exc:
        assert "sigma" in str(exc)
    else:
        raise AssertionError("invalid BAO sigma did not raise")


def test_forward_coverage_keeps_dark_matter_particle_physics_open() -> None:
    coverage = ForwardCoverage()

    assert coverage.has_density_ratios
    assert coverage.has_background_expansion_model
    assert coverage.has_growth_model_for_s8
    assert not coverage.has_particle_dark_matter_model
    assert not coverage.has_detector_likelihood


def test_parameter_provenance_separates_predictions_inputs_and_assumptions() -> None:
    entries = {entry.name: entry for entry in parameter_provenance(CEForwardParams())}

    assert entries["omega_b0"].is_ce_prediction
    assert entries["omega_dm0"].role == "ce_prediction"
    assert entries["omega_lambda0"].role == "ce_prediction"
    assert entries["h0"].is_external_input
    assert entries["rd_mpc"].is_external_input
    assert entries["tcmb_k"].role == "inactive_external_input"
    assert entries["n_eff"].role == "inactive_model_assumption"
    assert entries["sigma8_0"].is_external_input
    assert entries["w0"].role == "model_assumption"
    assert entries["wa"].role == "model_assumption"
    assert entries["gravity_mu_coupling"].role == "model_assumption"
