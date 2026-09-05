"""SNKC 급냉 배경 게이트, 자외선 꼬리 상계, 봉인 홀드아웃 평가기의 테스트를 한 파일에 모은다.

홀드아웃 절은 합성 자료만 쓴다. 로더, GLS 프로파일, 블록 결합, 표 -2lnL 보간,
LambdaCDM 주입 검사만 다루며 실제 홀드아웃 값은 여기서 단언하지 않는다. 봉인 평가의
회귀 고정은 계약상 별도의 평가 후 단계다.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from examples.physics.darksector.kinetic_dark_sector_gate import (
    KineticClockConfig,
    OMEGA_K0,
    profile_desi_bao,
    solve_background,
)
from examples.physics.darksector.kinetic_dark_sector_quench import (
    FROZEN_BASE_DESI_CHI2,
    FROZEN_BASE_DESI_SCALE,
    FROZEN_LCDM_CONTROL_CHI2,
    FrozenBoundaryLCDMModel,
    GaussianBlock,
    calibrate_omega_prod,
    calibration_grid,
    cc_covariance,
    e_of_z,
    evaluate_primary_p,
    gls_profile,
    holdout_profiled_chi2,
    lcdm_control_chi2,
    load_cc_data,
    load_elg_dv_table,
    load_gaussian_block,
    load_lya_grid,
    load_mm20_table,
    minimize_scalar,
    neg2_log_like_1d,
    neg2_log_like_2d,
    smooth_quench_created_occupation_tail_upper,
    smooth_quench_present_tail_certificate,
    solve_quench_background,
)
from examples.physics.record.theater_opening import (
    QuantumSeatSpecies,
    smooth_tanh_mode,
)



def test_zero_production_limit_reproduces_frozen_base_gate_regression() -> None:
    quench_fit = profile_desi_bao(solve_quench_background(0.0))
    base_fit = profile_desi_bao(solve_background())

    # frozen ledger values from the base gate run
    assert math.isclose(quench_fit.chi2, FROZEN_BASE_DESI_CHI2, abs_tol=1.0e-5)
    assert math.isclose(quench_fit.scale, FROZEN_BASE_DESI_SCALE, abs_tol=1.0e-7)
    assert quench_fit.dof == 12
    # the quench solver must reduce to the untouched base solver
    assert math.isclose(quench_fit.chi2, base_fit.chi2, rel_tol=1.0e-12)
    assert math.isclose(quench_fit.scale, base_fit.scale, rel_tol=1.0e-12)


def test_lcdm_control_matches_frozen_contract_value() -> None:
    assert math.isclose(lcdm_control_chi2(), FROZEN_LCDM_CONTROL_CHI2, abs_tol=1.0e-5)


def test_flat_budget_is_preserved_with_nonzero_production() -> None:
    config = KineticClockConfig(steps=800)
    solution = solve_quench_background(0.01, config)

    assert math.isclose(solution.nodes[-1].e2, 1.0, rel_tol=2.0e-12)
    assert solution.min_u > 0.0
    assert solution.min_cs2 > 0.0
    assert solution.min_q_s_over_mpl2 > 0.0

    with pytest.raises(ValueError):
        solve_quench_background(-1.0e-9, config)
    with pytest.raises(ValueError):
        solve_quench_background(OMEGA_K0 + 1.0e-9, config)


def test_calibration_grid_is_zero_union_forty_log_uniform_points() -> None:
    grid = calibration_grid()

    assert len(grid) == 41
    assert grid[0] == 0.0
    assert math.isclose(grid[1], 1.0e-6, rel_tol=1.0e-12)
    assert math.isclose(grid[-1], 0.05, rel_tol=1.0e-12)
    positive = grid[1:]
    assert all(b > a for a, b in zip(positive, positive[1:]))
    ratios = [b / a for a, b in zip(positive, positive[1:])]
    assert all(math.isclose(r, ratios[0], rel_tol=1.0e-9) for r in ratios)


def test_calibration_scan_records_curve_and_takes_argmin() -> None:
    config = KineticClockConfig(steps=600)
    calibration = calibrate_omega_prod((0.0, 1.0e-3), config=config)

    assert len(calibration.grid) == 2
    assert all(point.status == "OK" for point in calibration.grid)
    chi2_by_omega = {point.omega_prod0: point.chi2 for point in calibration.grid}
    assert calibration.argmin_chi2 == min(chi2_by_omega.values())
    assert chi2_by_omega[calibration.argmin_omega_prod0] == calibration.argmin_chi2
    assert calibration.dof == 12
    assert math.isclose(
        calibration.delta_chi2_argmin_minus_lcdm_control,
        calibration.argmin_chi2 - calibration.lcdm_control_chi2,
        rel_tol=1.0e-12,
    )
    assert calibration.role == "SEEN_DATA_TARGET_AWARE_CALIBRATION_NOT_HOLDOUT"


def test_holdout_evaluator_is_pure_profiled_and_family_strict() -> None:
    solution = solve_quench_background(0.0, KineticClockConfig(steps=800))

    z_bins = (0.3, 0.6, 1.0)
    identity = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(3)) for i in range(3)
    )

    bao_kinds = ("dm", "dh", "dv")
    from examples.physics.darksector.kinetic_dark_sector_gate import _dimensionless_distance

    shapes = []
    for z, kind in zip(z_bins, bao_kinds):
        dh = 1.0 / e_of_z(solution, z)
        dm = _dimensionless_distance(z, solution)
        dv = (z * dm * dm * dh) ** (1.0 / 3.0)
        shapes.append({"dm": dm, "dh": dh, "dv": dv}[kind])
    observed = tuple(30.0 * shape for shape in shapes)
    bao = holdout_profiled_chi2(solution, z_bins, observed, identity, bao_kinds)
    assert bao.family == "bao_c_over_h0rd"
    assert math.isclose(bao.scale, 30.0, rel_tol=1.0e-12)
    assert bao.chi2 < 1.0e-18
    assert bao.dof == 2

    hz_observed = tuple(70.0 * e_of_z(solution, z) for z in z_bins)
    hz = holdout_profiled_chi2(solution, z_bins, hz_observed, identity, ("hz",) * 3)
    assert hz.family == "hz_h0"
    assert math.isclose(hz.scale, 70.0, rel_tol=1.0e-12)
    assert hz.chi2 < 1.0e-18

    with pytest.raises(ValueError):
        holdout_profiled_chi2(solution, z_bins, observed, identity, ("dm", "hz", "dv"))
    with pytest.raises(ValueError):
        holdout_profiled_chi2(solution, z_bins, observed, identity, ("dm", "bad", "dv"))
    asymmetric = ((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        holdout_profiled_chi2(solution, z_bins, observed, asymmetric, bao_kinds)



def _species(
    *,
    mass_in: float = 1.0,
    mass_out: float = 3.0,
    initial_occupation: float = 0.0,
    degeneracy: int = 2,
) -> QuantumSeatSpecies:
    return QuantumSeatSpecies(
        label="tail-test",
        degeneracy=degeneracy,
        mass_in=mass_in,
        mass_out=mass_out,
        duration=0.2,
        initial_mode_occupation=initial_occupation,
    )


def test_pointwise_bound_dominates_exact_created_occupation() -> None:
    species = _species()
    start = 10.0
    for momentum in (10.0, 12.0, 20.0, 50.0):
        exact = smooth_tanh_mode(species, momentum).created_occupation
        upper = smooth_quench_created_occupation_tail_upper(
            species,
            momentum=momentum,
            momentum_start=start,
        )
        assert exact <= upper


def test_integrated_tail_bound_dominates_direct_finite_tail_segment() -> None:
    species = _species()
    start, stop, intervals = 10.0, 80.0, 4000
    scale_factor = 0.1
    step = (stop - start) / intervals
    number_terms = []
    energy_terms = []
    for index in range(intervals + 1):
        momentum = start + index * step
        weight = (
            1.0
            if index in (0, intervals)
            else (4.0 if index % 2 else 2.0)
        )
        occupation = smooth_tanh_mode(species, momentum).created_occupation
        radial = momentum * momentum * occupation
        number_terms.append(weight * radial)
        energy_terms.append(
            weight
            * radial
            * math.hypot(species.mass_out, scale_factor * momentum)
        )
    prefactor = (
        species.degeneracy
        / (2.0 * math.pi * math.pi)
        * scale_factor**3
        * step
        / 3.0
    )
    number_segment = prefactor * math.fsum(number_terms)
    energy_segment = prefactor * math.fsum(energy_terms)
    certificate = smooth_quench_present_tail_certificate(
        species,
        momentum_start=start,
        scale_factor_at_production=scale_factor,
        critical_density_today=108.0,
    )
    assert number_segment <= certificate.present_number_density_upper
    assert energy_segment <= certificate.present_energy_density_upper
    assert certificate.present_pressure_upper >= 0.0
    assert certificate.omega_produced_upper >= (
        certificate.present_energy_density_upper / 108.0
    )
    assert certificate.numerical_status.endswith("NOT_INTERVAL_CERTIFIED")


def test_tail_bound_tracks_degeneracy_stimulation_and_redshift_volume() -> None:
    common = dict(
        momentum_start=10.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    base = smooth_quench_present_tail_certificate(_species(degeneracy=1), **common)
    doubled = smooth_quench_present_tail_certificate(
        _species(degeneracy=2),
        **common,
    )
    stimulated = smooth_quench_present_tail_certificate(
        _species(degeneracy=1, initial_occupation=2.0),
        **common,
    )
    assert doubled.present_number_density_upper / base.present_number_density_upper == (
        pytest.approx(2.0, rel=3.0e-15)
    )
    assert (
        stimulated.present_number_density_upper
        / base.present_number_density_upper
        == pytest.approx(5.0, rel=3.0e-15)
    )
    half_scale = smooth_quench_present_tail_certificate(
        _species(degeneracy=1),
        momentum_start=10.0,
        scale_factor_at_production=0.05,
        critical_density_today=108.0,
    )
    assert half_scale.present_number_density_upper / base.present_number_density_upper == (
        pytest.approx(1.0 / 8.0, rel=3.0e-15)
    )


def test_later_tail_start_gives_a_stronger_bound() -> None:
    lower = smooth_quench_present_tail_certificate(
        _species(),
        momentum_start=8.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    upper = smooth_quench_present_tail_certificate(
        _species(),
        momentum_start=12.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    assert upper.present_number_density_upper < lower.present_number_density_upper
    assert upper.present_energy_density_upper < lower.present_energy_density_upper


def test_equal_masses_have_an_exact_zero_created_tail() -> None:
    species = _species(
        mass_in=2.0,
        mass_out=2.0,
        initial_occupation=4.0,
    )
    certificate = smooth_quench_present_tail_certificate(
        species,
        momentum_start=10.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    assert certificate.log_occupation_coefficient == -math.inf
    assert certificate.present_number_density_upper == 0.0
    assert certificate.present_energy_density_upper == 0.0
    assert certificate.omega_produced_upper == 0.0
    assert smooth_quench_created_occupation_tail_upper(
        species,
        momentum=10.0,
        momentum_start=10.0,
    ) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(momentum_start=0.0, scale_factor_at_production=0.1, critical_density_today=1.0),
        dict(momentum_start=1.0, scale_factor_at_production=1.1, critical_density_today=1.0),
        dict(momentum_start=1.0, scale_factor_at_production=0.1, critical_density_today=math.nan),
    ],
)
def test_tail_certificate_rejects_invalid_domain(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        smooth_quench_present_tail_certificate(_species(), **kwargs)


def test_pointwise_bound_rejects_momentum_below_tail_start() -> None:
    with pytest.raises(ValueError):
        smooth_quench_created_occupation_tail_upper(
            _species(),
            momentum=9.0,
            momentum_start=10.0,
        )



# --------------------------------------------------------------------------
# GLS profile
# --------------------------------------------------------------------------


def test_gls_profile_recovers_exact_scale_with_zero_chi2() -> None:
    shapes = (1.0, 2.0, 3.0)
    observed = tuple(2.5 * value for value in shapes)
    covariance = ((0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3))
    fit = gls_profile(shapes, observed, covariance)
    assert fit.scale == pytest.approx(2.5, abs=1.0e-12)
    assert fit.chi2 == pytest.approx(0.0, abs=1.0e-10)
    assert fit.dof == 2
    assert all(abs(pull) < 1.0e-8 for pull in fit.pulls)


def test_gls_profile_matches_hand_computed_two_point_case() -> None:
    # cov = diag(2, 1), shapes = (1, 1), observed = (1, 3)
    # gg = 3/2, gd = 7/2, scale = 7/3, chi2 = (4/3)^2/2 + (2/3)^2 = 4/3
    fit = gls_profile((1.0, 1.0), (1.0, 3.0), ((2.0, 0.0), (0.0, 1.0)))
    assert fit.scale == pytest.approx(7.0 / 3.0, rel=1.0e-12)
    assert fit.chi2 == pytest.approx(4.0 / 3.0, rel=1.0e-12)
    assert fit.scale_sigma == pytest.approx(math.sqrt(2.0 / 3.0), rel=1.0e-12)


# --------------------------------------------------------------------------
# loaders and block combination (synthetic files in the upstream formats)
# --------------------------------------------------------------------------


def _write_synthetic_gaussian_files(directory: Path) -> None:
    (directory / "sdss_DR12_LRG_BAO_DMDH.dat").write_text(
        "0.38 10.0 DM_over_rs\n0.38 25.0 DH_over_rs\n"
        "0.51 13.0 DM_over_rs\n0.51 22.0 DH_over_rs\n",
        encoding="utf-8",
    )
    (directory / "sdss_DR12_LRG_BAO_DMDH_covtot.txt").write_text(
        "1.0 0.1 0.0 0.0\n0.1 2.0 0.0 0.0\n0.0 0.0 3.0 0.2\n0.0 0.0 0.2 4.0\n",
        encoding="utf-8",
    )
    (directory / "sdss_DR16_LRG_BAO_DMDH.dat").write_text(
        "0.698 17.8 DM_over_rs\n0.698 19.3 DH_over_rs\n", encoding="utf-8"
    )
    (directory / "sdss_DR16_LRG_BAO_DMDH_covtot.txt").write_text(
        "0.5 -0.05\n-0.05 0.6\n", encoding="utf-8"
    )
    # QSO file mimics upstream: no trailing newline
    (directory / "sdss_DR16_QSO_BAO_DMDH.txt").write_text(
        "1.48 30.6 DM_over_rs\n1.48 13.2 DH_over_rs", encoding="utf-8"
    )
    (directory / "sdss_DR16_QSO_BAO_DMDH_covtot.txt").write_text(
        "0.7 0.15\n0.15 0.3", encoding="utf-8"
    )


def test_load_gaussian_block_assembles_block_diagonal(tmp_path: Path) -> None:
    _write_synthetic_gaussian_files(tmp_path)
    block = load_gaussian_block(tmp_path)
    assert block.z == (0.38, 0.38, 0.51, 0.51, 0.698, 0.698, 1.48, 1.48)
    assert block.kinds == ("dm", "dh", "dm", "dh", "dm", "dh", "dm", "dh")
    assert block.block_sizes == (4, 2, 2)
    assert len(block.covariance) == 8
    # block placement
    assert block.covariance[0][1] == pytest.approx(0.1)
    assert block.covariance[4][5] == pytest.approx(-0.05)
    assert block.covariance[6][7] == pytest.approx(0.15)
    # cross-block entries are exactly zero
    assert block.covariance[0][5] == 0.0
    assert block.covariance[3][6] == 0.0
    assert block.covariance[7][2] == 0.0
    # symmetry
    for i in range(8):
        for j in range(8):
            assert block.covariance[i][j] == block.covariance[j][i]


def test_elg_table_loader_and_interpolation(tmp_path: Path) -> None:
    center, width = 18.0, 0.5
    grid = [center - 2.0 + 0.05 * k for k in range(81)]
    lines = "".join(
        f"{value:.6e} {math.exp(-0.5 * ((value - center) / width) ** 2):.6e}\n"
        for value in grid
    )
    (tmp_path / "sdss_DR16_ELG_BAO_DVtable.txt").write_text(lines, encoding="utf-8")
    table = load_elg_dv_table(tmp_path)
    assert len(table.grid) == 81
    # -2 lnL is minimal at the peak and grows away from it
    at_peak = neg2_log_like_1d(table, center)
    off_peak = neg2_log_like_1d(table, center + width)
    assert at_peak < off_peak
    assert off_peak - at_peak == pytest.approx(1.0, abs=0.05)
    # out-of-grid: boundary value plus a declared large penalty
    outside = neg2_log_like_1d(table, center + 3.0)
    boundary = neg2_log_like_1d(table, table.grid[-1])
    assert outside > boundary + 1.0e3


def test_lya_grid_loader_and_bilinear_interpolation(tmp_path: Path) -> None:
    dm_axis = [30.0 + 0.1 * i for i in range(21)]
    dh_axis = [6.0 + 0.05 * j for j in range(21)]
    lines = ["# D_M D_H likelihood ratio\n"]
    for dm in dm_axis:
        for dh in dh_axis:
            like = math.exp(
                -0.5 * (((dm - 31.0) / 0.4) ** 2 + ((dh - 6.5) / 0.2) ** 2)
            )
            lines.append(f"{dm:.6e} {dh:.6e} {like:.6e}\n")
    (tmp_path / "grid.txt").write_text("".join(lines), encoding="utf-8")
    grid = load_lya_grid("grid.txt", tmp_path)
    assert len(grid.dm_axis) == 21
    assert len(grid.dh_axis) == 21
    at_peak = neg2_log_like_2d(grid, 31.0, 6.5)
    off_peak = neg2_log_like_2d(grid, 31.4, 6.5)
    assert at_peak < off_peak
    assert off_peak - at_peak == pytest.approx(1.0, abs=0.05)
    # out-of-grid penalty on either axis
    assert neg2_log_like_2d(grid, 40.0, 6.5) > at_peak + 1.0e3
    assert neg2_log_like_2d(grid, 31.0, 9.0) > at_peak + 1.0e3


def test_cc_loader_and_moresco_covariance(tmp_path: Path) -> None:
    header = "# z\tHz\terrHz\tstat_contr\tmet_contr\treference\n"
    rows = []
    for k in range(15):
        z = 0.1 + 0.1 * k
        rows.append(f"{z},{70.0 + 30.0 * z},{5.0},{4.0},{3.0},Synthetic (2020)\n")
    (tmp_path / "HzTable_MM_BC03.dat").write_text(header + "".join(rows), encoding="utf-8")
    mm_lines = ["# z   IMF  stlib  mod  mod_ooo\n"]
    for k in range(29):
        z = 0.075 + 0.05 * k
        mm_lines.append(f"{z:.3f} 1.0 2.0 10.0 5.0\n")
    (tmp_path / "data_MM20.dat").write_text("".join(mm_lines), encoding="utf-8")

    data = load_cc_data(tmp_path)
    mm20 = load_mm20_table(tmp_path)
    assert len(data.z) == 15
    assert data.hz[0] == pytest.approx(73.0)
    assert len(mm20["z"]) == 29

    covariance = cc_covariance(data, mm20, "mod")
    # constant percent columns: f_imf=0.01, f_stlib=0.02, f_mod=0.10
    f2 = 0.01**2 + 0.02**2 + 0.10**2
    assert covariance[0][0] == pytest.approx(25.0 + f2 * data.hz[0] ** 2, rel=1.0e-12)
    assert covariance[0][1] == pytest.approx(f2 * data.hz[0] * data.hz[1], rel=1.0e-12)
    # mod_ooo variant swaps only the SPS component
    sensitivity = cc_covariance(data, mm20, "mod_ooo")
    f2_ooo = 0.01**2 + 0.02**2 + 0.05**2
    assert sensitivity[0][1] == pytest.approx(
        f2_ooo * data.hz[0] * data.hz[1], rel=1.0e-12
    )


# --------------------------------------------------------------------------
# numerical profile
# --------------------------------------------------------------------------


def test_minimize_scalar_finds_quadratic_minimum() -> None:
    argmin, value = minimize_scalar(lambda s: 3.0 + (s - 29.5) ** 2, center=30.0)
    assert argmin == pytest.approx(29.5, abs=1.0e-6)
    assert value == pytest.approx(3.0, abs=1.0e-10)


# --------------------------------------------------------------------------
# LambdaCDM-injection sanity: chi2 near dof for the matching model
# --------------------------------------------------------------------------


def test_lcdm_injected_data_gives_chi2_near_dof() -> None:
    model = FrozenBoundaryLCDMModel()
    z_bins = (0.38, 0.38, 0.51, 0.51, 0.698, 0.698, 1.48, 1.48)
    kinds = ("dm", "dh", "dm", "dh", "dm", "dh", "dm", "dh")
    scale_true = 30.0
    rng = random.Random(20260830)
    shapes = []
    for z, kind in zip(z_bins, kinds):
        dh = 1.0 / model.e(z)
        dm = model.dc(z)
        shapes.append({"dm": dm, "dh": dh}[kind])
    truth = [scale_true * shape for shape in shapes]
    sigmas = [0.01 * value for value in truth]
    observed = tuple(
        value + sigma * rng.gauss(0.0, 1.0) for value, sigma in zip(truth, sigmas)
    )
    covariance = tuple(
        tuple(sigmas[i] ** 2 if i == j else 0.0 for j in range(8)) for i in range(8)
    )
    block = GaussianBlock(
        z=z_bins,
        observed=observed,
        kinds=kinds,
        covariance=covariance,
        block_sizes=(8,),
        block_names=("synthetic",),
    )
    fit = evaluate_primary_p(model, block)
    assert fit.dof == 7
    # chi2 should sit near dof for the generating model (loose deterministic band)
    assert 0.5 < fit.chi2 < 18.0
    assert fit.scale == pytest.approx(scale_true, rel=0.01)
