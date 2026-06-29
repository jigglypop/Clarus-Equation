from __future__ import annotations

import math

from examples.physics.ce_residual_forward_model import (
    BAODataPoint,
    CEForwardParams,
    ForwardCoverage,
    bao_chi2,
    bao_chi2_with_covariance,
    bao_observable,
    dark_energy_scale,
    e_of_z,
    f_sigma8_at_z,
    hubble_distance_mpc,
    luminosity_distance_mpc,
    named_bao_dataset,
    s8_today,
    solve_growth,
    transverse_comoving_distance_mpc,
    volume_distance_mpc,
    invert_matrix,
    parse_bao_data,
    parse_covariance_matrix,
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
