from __future__ import annotations

import math

from examples.physics.kinetic_dark_sector_gate import (
    KineticClockConfig,
    REFERENCE_RD_MPC,
    SPEED_OF_LIGHT_KM_S,
    compressed_cmb_acoustic_diagnostic,
    solve_bao_cmb_acoustic_closure,
    profile_desi_bao,
    scan_gamma_against_lcdm,
    solve_background,
    zero_current_local_verdict,
)


def test_zero_current_is_an_exact_unhealthy_local_branch() -> None:
    verdict = zero_current_local_verdict()

    assert verdict["initial_current"] == 0.0
    assert verdict["current_derivative_sign"] == -1
    assert verdict["delta_sign_immediately_after"] == -1
    assert verdict["cs2_sign_immediately_after"] == -1
    assert verdict["healthy_branch"] is False


def test_calibrated_gamma10_branch_is_healthy_over_observation_window() -> None:
    solution = solve_background(KineticClockConfig(gamma=10.0, steps=1200))

    assert math.isclose(solution.nodes[0].n, math.log(1.0e-4), abs_tol=1.0e-12)
    assert math.isclose(solution.nodes[-1].n, 0.0, abs_tol=1.0e-15)
    assert solution.min_u > 0.0
    assert solution.min_cs2 > 0.0
    assert solution.min_q_s_over_mpl2 > 0.0
    assert math.isclose(solution.nodes[-1].e2, 1.0, rel_tol=2.0e-12)


def test_compact_desi_profile_is_finite_and_explicitly_posthoc() -> None:
    solution = solve_background(KineticClockConfig(gamma=10.0, steps=1200))
    fit = profile_desi_bao(solution)

    assert fit.scale > 0.0
    assert fit.scale_sigma > 0.0
    assert fit.chi2 >= 0.0
    assert math.isfinite(fit.chi2)
    assert fit.dof == 12
    assert fit.role == "posthoc_boundary_calibrated_shape_test"
    equivalent_h0 = SPEED_OF_LIGHT_KM_S / (fit.scale * REFERENCE_RD_MPC)
    assert 60.0 < equivalent_h0 < 75.0

    cmb = compressed_cmb_acoustic_diagnostic(
        solution, h0_km_s_mpc=equivalent_h0
    )
    assert 0.9 < cmb.predicted_100_theta_star < 1.2
    assert math.isfinite(cmb.raw_observational_pull)
    assert cmb.role.startswith("APPROXIMATE_EARLY_PHYSICS")
    joint = solve_bao_cmb_acoustic_closure(solution, fit)
    assert 50.0 < joint.h0_km_s_mpc < 75.0
    assert 130.0 < joint.rd_mpc < 170.0
    assert joint.sh0es_offset_sigma > 0.0
    assert joint.role.startswith("PLANCK_CALIBRATED")


def test_cmb_acoustic_discrepancy_is_numerically_converged() -> None:
    coarse_solution = solve_background(KineticClockConfig(steps=1200))
    fine_solution = solve_background(KineticClockConfig(steps=2400))
    coarse = compressed_cmb_acoustic_diagnostic(
        coarse_solution, h0_km_s_mpc=68.1283869, distance_intervals=4096
    )
    fine = compressed_cmb_acoustic_diagnostic(
        fine_solution, h0_km_s_mpc=68.1283869, distance_intervals=8192
    )

    assert math.isclose(
        coarse.predicted_100_theta_star,
        fine.predicted_100_theta_star,
        rel_tol=5.0e-5,
    )
    assert fine.raw_observational_pull > 20.0


def test_gamma_scan_penalizes_selection_and_compares_same_scale_profile() -> None:
    comparison = scan_gamma_against_lcdm((5.0, 10.0), steps=600)

    assert comparison.best_gamma in comparison.gamma_values
    assert comparison.best_chi2 == min(comparison.chi2_values)
    assert comparison.best_scale > 0.0
    assert 60.0 < comparison.best_h0_at_reference_rd < 75.0
    assert comparison.lcdm_chi2 >= 0.0
    assert comparison.role == "posthoc_seen_data_model_comparison"
    assert math.isfinite(comparison.delta_aic_vs_lcdm)
    assert math.isfinite(comparison.delta_bic_vs_lcdm)
    assert isinstance(comparison.best_is_upper_boundary, bool)
    assert comparison.tail_delta_chi2 >= 0.0
    assert len(comparison.cmb_raw_pull_values) == len(comparison.gamma_values)
    assert len(comparison.cmb_delta_pull_vs_lcdm_values) == len(comparison.gamma_values)


def test_extended_gamma_scan_identifies_the_saturated_boundary_limit() -> None:
    comparison = scan_gamma_against_lcdm((30.0, 100.0, 300.0), steps=600)

    assert comparison.best_is_upper_boundary
    assert comparison.tail_is_numerically_saturated
    assert comparison.best_chi2 > comparison.lcdm_chi2
    assert comparison.cmb_least_discrepant_gamma == 300.0
    assert comparison.minimum_abs_cmb_pull > 5.0
    assert comparison.minimum_abs_cmb_delta_pull_vs_lcdm < 1.0
