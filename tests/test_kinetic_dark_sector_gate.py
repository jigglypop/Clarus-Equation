from __future__ import annotations

from dataclasses import dataclass
import cmath
import math

import pytest

from examples.physics.darksector.kinetic_dark_sector_gate import (
    FLRWModeSpec,
    KineticClockConfig,
    REFERENCE_RD_MPC,
    SPEED_OF_LIGHT_KM_S,
    SupernovaDataset,
    adiabatic_initial_mode,
    compare_pantheon_binned,
    compare_pantheon_binned_holdout,
    compressed_cmb_acoustic_diagnostic,
    load_pantheon_binned,
    omega_squared_at_n,
    profile_desi_bao,
    profiled_intercept_holdout_fit,
    scan_gamma_against_lcdm,
    solve_background,
    solve_bao_cmb_acoustic_closure,
    solve_flrw_mode,
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



def test_hash_pinned_pantheon_binned_dimensions_and_covariance() -> None:
    dataset = load_pantheon_binned()

    assert len(dataset.redshift) == 40
    assert len(dataset.covariance) == 40
    assert all(len(row) == 40 for row in dataset.covariance)
    assert all(dataset.covariance[i][i] > 0.0 for i in range(40))


def test_pantheon_shape_comparison_profiles_only_the_intercept() -> None:
    solution = solve_background(KineticClockConfig(steps=600))
    result = compare_pantheon_binned(solution)

    assert result.kinetic.dof == 39
    assert result.lcdm.dof == 39
    assert math.isfinite(result.kinetic.chi2)
    assert math.isfinite(result.lcdm.chi2)
    assert result.kinetic.role.endswith("NOT_PANTHEON_PLUS")


def test_correlated_holdout_is_finite_and_disjoint() -> None:
    solution = solve_background(KineticClockConfig(steps=600))
    result = compare_pantheon_binned_holdout(solution)

    assert len(result.kinetic.training_indices) == 30
    assert len(result.kinetic.holdout_indices) == 10
    assert not set(result.kinetic.training_indices) & set(
        result.kinetic.holdout_indices
    )
    assert math.isfinite(result.kinetic.predictive_chi2)
    assert math.isfinite(result.lcdm.predictive_chi2)
    assert math.isfinite(result.delta_predictive_chi2_kinetic_minus_lcdm)
    assert result.kinetic.role.endswith("NOT_PREREGISTERED")


def test_diagonal_covariance_holdout_propagates_intercept_uncertainty() -> None:
    dataset = SupernovaDataset(
        redshift=(0.1, 0.2, 0.3, 0.4),
        apparent_magnitude=(1.0, 1.0, 1.0, 2.0),
        covariance=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        source="synthetic",
    )
    fit = profiled_intercept_holdout_fit(
        (0.0, 0.0, 0.0, 0.0),
        dataset,
        holdout_indices=(3,),
        label="synthetic",
    )

    assert fit.training_intercept == pytest.approx(1.0)
    # Predictive variance is 1 (held-out noise) + 1/3 (intercept posterior).
    assert fit.predictive_chi2 == pytest.approx(0.75)


def test_cross_covariance_enters_conditional_mean_and_variance() -> None:
    dataset = SupernovaDataset(
        redshift=(0.1, 0.2, 0.3),
        apparent_magnitude=(1.0, 1.0, 2.0),
        covariance=(
            (1.0, 0.2, 0.1),
            (0.2, 1.0, 0.1),
            (0.1, 0.1, 1.0),
        ),
        source="correlated synthetic",
    )
    fit = profiled_intercept_holdout_fit(
        (0.0, 0.0, 0.0),
        dataset,
        holdout_indices=(2,),
        label="correlated synthetic",
    )

    # A direct Schur-complement calculation gives predictive variance 7/5
    # and residual -1 after integrating the train-only intercept posterior.
    assert fit.training_intercept == pytest.approx(1.0)
    assert fit.predictive_log_determinant == pytest.approx(math.log(7.0 / 5.0))
    assert fit.predictive_chi2 == pytest.approx(5.0 / 7.0)



@dataclass(frozen=True)
class _DeSitterNode:
    n: float
    e2: float = 1.0


@dataclass(frozen=True)
class _DeSitterBackground:
    nodes: tuple[_DeSitterNode, ...] = (
        _DeSitterNode(-2.0),
        _DeSitterNode(0.0),
    )

    def at_n(self, n: float) -> _DeSitterNode:
        if n < self.nodes[0].n or n > self.nodes[-1].n:
            raise ValueError("outside de Sitter control window")
        return _DeSitterNode(n)


def _massless_conformal_spec(*, steps: int) -> FLRWModeSpec:
    return FLRWModeSpec(
        comoving_wavenumber_over_h0=2.3,
        mass_over_h0=lambda _n: 0.0,
        initial_n=-2.0,
        final_n=0.0,
        steps=steps,
    )


def _de_sitter_endpoint_error(steps: int) -> tuple[float, object]:
    background = _DeSitterBackground()
    spec = _massless_conformal_spec(steps=steps)
    solution = solve_flrw_mode(background, spec)
    q = spec.comoving_wavenumber_over_h0
    delta_x = math.exp(2.0) - 1.0
    exact_u = solution.nodes[0].u * cmath.exp(-1.0j * q * delta_x)
    error = abs(solution.nodes[-1].u - exact_u)
    return error, solution


def test_massless_conformal_de_sitter_mode_matches_exact_phase() -> None:
    error, solution = _de_sitter_endpoint_error(steps=800)
    q = solution.spec.comoving_wavenumber_over_h0
    final = solution.nodes[-1]

    assert len(solution.nodes) == 801
    assert final.x == pytest.approx(math.exp(2.0) - 1.0, rel=2.0e-10)
    assert error < 8.0e-8
    assert final.du_dx == pytest.approx(-1.0j * q * final.u, rel=8.0e-8)
    assert solution.max_wronskian_residual < 8.0e-8
    assert solution.initial_amplitude_residual < 1.0e-15
    assert solution.status == "MODE_ONLY_NO_RENORMALIZED_STRESS_OR_BACKREACTION"
    assert solution.dimensionless_contract == (
        "N=log(a); x=H0*eta; q=k/H0; mu=m/H0; U=sqrt(H0)*u_phys"
    )


def test_de_sitter_mode_has_fourth_order_grid_convergence() -> None:
    coarse_error, _ = _de_sitter_endpoint_error(steps=100)
    fine_error, _ = _de_sitter_endpoint_error(steps=200)

    assert fine_error < coarse_error / 12.0


def test_adiabatic_initializer_is_canonically_normalized() -> None:
    background = _DeSitterBackground()
    spec = _massless_conformal_spec(steps=100)
    initial = adiabatic_initial_mode(background, spec)

    assert initial.omega == pytest.approx(spec.comoving_wavenumber_over_h0)
    assert initial.adiabaticity < 1.0e-12
    assert initial.amplitude_residual < 1.0e-15
    assert initial.wronskian_residual < 1.0e-15


def test_time_dependent_initial_state_is_independent_of_output_grid() -> None:
    background = _DeSitterBackground()
    common = dict(
        comoving_wavenumber_over_h0=1.7,
        mass_over_h0=lambda n: 0.8 + 0.2 * math.exp(n + 2.0),
        initial_n=-2.0,
        final_n=0.0,
        adiabatic_derivative_step_n=2.0e-4,
    )
    coarse_spec = FLRWModeSpec(**common, steps=100)
    fine_spec = FLRWModeSpec(**common, steps=200)
    reference_spec = FLRWModeSpec(**common, steps=800)

    coarse_initial = adiabatic_initial_mode(background, coarse_spec)
    fine_initial = adiabatic_initial_mode(background, fine_spec)
    assert fine_initial.u == coarse_initial.u
    assert fine_initial.du_dx == coarse_initial.du_dx

    coarse = solve_flrw_mode(background, coarse_spec)
    fine = solve_flrw_mode(background, fine_spec)
    reference = solve_flrw_mode(background, reference_spec)
    coarse_error = abs(coarse.nodes[-1].u - reference.nodes[-1].u)
    fine_error = abs(fine.nodes[-1].u - reference.nodes[-1].u)

    assert fine_error < coarse_error / 12.0


def test_rapid_mass_history_fails_declared_adiabaticity_gate() -> None:
    background = _DeSitterBackground()
    spec = FLRWModeSpec(
        comoving_wavenumber_over_h0=1.0,
        mass_over_h0=lambda n: math.exp(20.0 * (n + 2.0)),
        initial_n=-2.0,
        final_n=-1.0,
        steps=100,
        max_initial_adiabaticity=1.0e-3,
    )

    with pytest.raises(ValueError, match="initial adiabaticity"):
        solve_flrw_mode(background, spec)


def test_mode_runs_on_the_solved_kinetic_background_without_stress_claim() -> None:
    background = solve_background(KineticClockConfig(gamma=10.0, steps=600))
    spec = FLRWModeSpec(
        comoving_wavenumber_over_h0=3.0,
        mass_over_h0=lambda _n: 1.5,
        initial_n=-2.0,
        final_n=0.0,
        steps=400,
    )
    solution = solve_flrw_mode(background, spec)

    assert solution.background_window == pytest.approx(
        (background.nodes[0].n, background.nodes[-1].n)
    )
    assert all(math.isfinite(node.omega_squared) for node in solution.nodes)
    assert all(node.omega_squared > 0.0 for node in solution.nodes)
    assert solution.max_wronskian_residual < 1.0e-8
    assert "STRESS" in solution.status
    assert "BACKREACTION" in solution.status


def test_mode_domain_errors_fail_closed() -> None:
    background = _DeSitterBackground()

    with pytest.raises(ValueError, match="comoving_wavenumber"):
        FLRWModeSpec(
            comoving_wavenumber_over_h0=0.0,
            mass_over_h0=lambda _n: 0.0,
        )

    negative_mass = FLRWModeSpec(
        comoving_wavenumber_over_h0=1.0,
        mass_over_h0=lambda _n: -1.0,
    )
    with pytest.raises(ValueError, match="mass_over_h0"):
        omega_squared_at_n(background, negative_mass, -1.0)

    outside = FLRWModeSpec(
        comoving_wavenumber_over_h0=1.0,
        mass_over_h0=lambda _n: 0.0,
        initial_n=-3.0,
        final_n=-1.0,
    )
    with pytest.raises(ValueError, match="outside"):
        solve_flrw_mode(background, outside)

    nonconformal_endpoint = FLRWModeSpec(
        comoving_wavenumber_over_h0=1.0,
        mass_over_h0=lambda _n: 0.0,
        curvature_coupling=0.0,
    )
    with pytest.raises(ValueError, match="curvature derivative stencil"):
        omega_squared_at_n(background, nonconformal_endpoint, -2.0)

    tachyonic = FLRWModeSpec(
        comoving_wavenumber_over_h0=0.1,
        mass_over_h0=lambda _n: 0.0,
        curvature_coupling=0.0,
    )
    with pytest.raises(ValueError, match="omega_squared"):
        omega_squared_at_n(background, tachyonic, -0.1)
