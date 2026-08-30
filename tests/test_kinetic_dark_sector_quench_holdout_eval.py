"""Synthetic-data tests for the SNKC quench sealed holdout evaluator.

Loaders, GLS profiling, block combination, tabulated -2lnL interpolation, and
one LambdaCDM-injection sanity check are exercised with synthetic material
only.  No real holdout value is asserted here: regression pinning of the
sealed evaluation is a separate post-evaluation step by contract.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from examples.physics.kinetic_dark_sector_quench_holdout_eval import (
    FrozenBoundaryLCDMModel,
    GaussianBlock,
    cc_covariance,
    evaluate_primary_p,
    gls_profile,
    load_cc_data,
    load_elg_dv_table,
    load_gaussian_block,
    load_lya_grid,
    load_mm20_table,
    minimize_scalar,
    neg2_log_like_1d,
    neg2_log_like_2d,
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
