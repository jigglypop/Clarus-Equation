import math

import pytest

from examples.physics.darksector.cosmology import (
    CE_RATIOS,
    Background,
    FlatFLRW,
    compare_all_density_ratios,
    coverage_verdict,
    cpl_density_scale,
    linspace,
    logspace,
    simpson,
    solve_growth,
)


def _quadratic_antiderivative(x: float) -> float:
    return x**3 - x**2 + 4.0 * x


def _warped_log_grid(n: int, a_min: float = 1.0e-2) -> list[float]:
    ln_min = math.log(a_min)
    return [
        math.exp(ln_min + (0.0 - ln_min) * (i / (n - 1)) ** 1.6)
        for i in range(n)
    ]


def _heath_growth_raw(bg: Background, a: float) -> float:
    grid = logspace(1.0e-7, a, 20_001)
    integrand = [1.0 / (x**3 * bg.e_of_a(x) ** 3) for x in grid]
    return bg.e_of_a(a) * simpson(integrand, grid)


def test_nonuniform_quadratic_panels_are_exact() -> None:
    x = [-0.3, -0.05, 0.4, 1.2, 2.0]
    y = [3.0 * xx**2 - 2.0 * xx + 4.0 for xx in x]
    expected = _quadratic_antiderivative(x[-1]) - _quadratic_antiderivative(x[0])

    assert simpson(y, x) == pytest.approx(expected, rel=2.0e-15, abs=2.0e-15)


def test_even_sample_count_retains_final_interval() -> None:
    x = [0.0, 0.07, 0.31, 0.72, 1.1, 2.0]
    y = [2.0 * xx + 1.0 for xx in x]

    assert simpson(y, x) == pytest.approx(6.0, rel=2.0e-15, abs=2.0e-15)


def test_uniform_composite_simpson_has_fourth_order_convergence() -> None:
    exact = math.e - 1.0

    def error(n: int) -> float:
        x = linspace(0.0, 1.0, n)
        return abs(simpson([math.exp(xx) for xx in x], x) - exact)

    coarse = error(17)
    medium = error(33)
    fine = error(65)
    assert coarse / medium > 14.0
    assert medium / fine > 14.0


@pytest.mark.parametrize(
    ("model", "expected_ricci", "expected_e2"),
    [
        (FlatFLRW(omega_m0=0.0, omega_de0=0.0, omega_r0=1.0), 0.0, 16.0),
        (FlatFLRW(omega_m0=1.0, omega_de0=0.0), 3.0, 8.0),
        (FlatFLRW(omega_m0=0.0, omega_de0=1.0), 12.0, 1.0),
    ],
)
def test_exact_radiation_matter_and_de_sitter_limits(
    model: FlatFLRW,
    expected_ricci: float,
    expected_e2: float,
) -> None:
    assert model.e2_of_a(0.5) == pytest.approx(expected_e2)
    assert model.ricci_over_h2(0.5) == pytest.approx(expected_ricci, abs=2.0e-15)


def test_lcdm_ricci_trace_identity_with_radiation() -> None:
    model = FlatFLRW(omega_m0=0.4, omega_de0=0.5, omega_r0=0.1)

    for a in (1.0e-3, 0.1, 0.7, 1.0):
        expected = 12.0 - 9.0 * model.omega_m_of_a(a) - 12.0 * model.omega_r_of_a(a)
        assert model.ricci_over_h2(a) == pytest.approx(expected, abs=3.0e-15)


def test_cpl_kernel_matches_analytic_density_and_log_derivative() -> None:
    model = FlatFLRW(
        omega_m0=0.25,
        omega_de0=0.65,
        omega_r0=0.1,
        w0=-0.8,
        wa=0.3,
    )
    a = 0.63
    expected_e2 = (
        model.omega_r0 * a**-4
        + model.omega_m0 * a**-3
        + model.omega_de0 * cpl_density_scale(a, model.w0, model.wa)
    )
    delta = 1.0e-6
    numerical = (
        math.log(model.e_of_a(a * math.exp(delta)))
        - math.log(model.e_of_a(a * math.exp(-delta)))
    ) / (2.0 * delta)

    assert cpl_density_scale(1.0, model.w0, model.wa) == pytest.approx(1.0)
    assert model.e2_of_a(a) == pytest.approx(expected_e2, rel=2.0e-15)
    assert model.dlnh_dln_a(a) == pytest.approx(numerical, rel=2.0e-10, abs=2.0e-10)


def test_background_default_is_backward_compatible_and_delegates_to_kernel() -> None:
    legacy = Background(omega_m0=0.4, omega_l0=0.6)
    extended = Background(omega_m0=0.4, omega_l0=0.5, omega_r0=0.1)
    kernel = FlatFLRW(omega_m0=0.4, omega_de0=0.5, omega_r0=0.1)

    for a in (0.2, 0.7, 1.0):
        assert legacy.e2_of_a(a) == pytest.approx(0.4 * a**-3 + 0.6)
        assert legacy.omega_r_of_a(a) == 0.0
        assert extended.e2_of_a(a) == kernel.e2_of_a(a)
        assert extended.ricci_over_h2(a) == kernel.ricci_over_h2(a)


def test_growth_uses_interval_local_log_steps_with_fourth_order_convergence() -> None:
    bg = Background(omega_m0=1.0, omega_l0=0.0)

    def midpoint_error(n: int) -> float:
        grid = _warped_log_grid(n)
        growth, _ = solve_growth(bg, grid, [1.0] * n)
        midpoint = n // 2
        return abs(growth[midpoint] - grid[midpoint])

    coarse = midpoint_error(9)
    medium = midpoint_error(17)
    fine = midpoint_error(33)
    assert coarse / medium > 8.0
    assert medium / fine > 8.0

    grid = _warped_log_grid(65)
    growth, rate = solve_growth(bg, grid, [1.0] * len(grid))
    assert max(abs(d - a) for d, a in zip(growth, grid)) < 5.0e-7
    assert max(abs(f - 1.0) for f in rate) < 2.0e-7


def test_nonuniform_growth_matches_heath_lcdm_solution() -> None:
    bg = Background(omega_m0=0.4, omega_l0=0.6)
    grid = _warped_log_grid(401, a_min=1.0e-3)
    growth, _ = solve_growth(bg, grid, [1.0] * len(grid))
    heath_today = _heath_growth_raw(bg, 1.0)

    for index in (80, 200, 320):
        expected = _heath_growth_raw(bg, grid[index]) / heath_today
        assert growth[index] == pytest.approx(expected, rel=2.0e-6, abs=2.0e-8)


def test_ce_density_ratios_are_close_to_recent_cmb_compressed_sets() -> None:
    comparisons = compare_all_density_ratios()

    assert len(comparisons) >= 4
    assert all(comparison.max_abs_relative_error < 0.04 for comparison in comparisons)


def test_ce_baryon_ratio_stays_near_observed_baryon_fraction() -> None:
    comparisons = compare_all_density_ratios()

    assert abs(CE_RATIOS["omega_b"] - 0.0486) < 2.0e-4
    assert max(abs(comparison.omega_b_diff) for comparison in comparisons) < 0.0012


def test_modern_likelihood_physics_is_not_implemented_by_ratio_audit() -> None:
    verdict = coverage_verdict()

    assert verdict.density_ratios_close
    assert not verdict.has_background_expansion_model
    assert not verdict.has_growth_model_for_s8
    assert not verdict.has_particle_dark_matter_model
    assert not verdict.has_detector_likelihood
    assert verdict.summary == "density ratios match; modern likelihood physics not implemented"
