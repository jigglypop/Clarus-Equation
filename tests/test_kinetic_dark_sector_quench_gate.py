from __future__ import annotations

import math

import pytest

from examples.physics.kinetic_dark_sector_gate import (
    KineticClockConfig,
    OMEGA_K0,
    profile_desi_bao,
    solve_background,
)
from examples.physics.kinetic_dark_sector_quench_gate import (
    FROZEN_BASE_DESI_CHI2,
    FROZEN_BASE_DESI_SCALE,
    FROZEN_LCDM_CONTROL_CHI2,
    calibrate_omega_prod,
    calibration_grid,
    e_of_z,
    holdout_profiled_chi2,
    lcdm_control_chi2,
    solve_quench_background,
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
    from examples.physics.kinetic_dark_sector_gate import _dimensionless_distance

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
