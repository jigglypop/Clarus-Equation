from __future__ import annotations

from dataclasses import dataclass
import cmath
import math

import pytest

from examples.physics.kinetic_dark_sector_flrw_mode import (
    FLRWModeSpec,
    adiabatic_initial_mode,
    omega_squared_at_n,
    solve_flrw_mode,
)
from examples.physics.kinetic_dark_sector_gate import (
    KineticClockConfig,
    solve_background,
)


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
