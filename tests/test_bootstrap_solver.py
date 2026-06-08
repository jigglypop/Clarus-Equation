from __future__ import annotations

import math

from reality_stone.clarus.bootstrap_solver import BootstrapSolver


def test_newton_and_bracketed_solver_agree() -> None:
    solver = BootstrapSolver()
    eps_newton = solver.solve(method="newton")

    bracketed = BootstrapSolver()
    eps_brent = bracketed.solve(method="brent")

    assert math.isclose(eps_newton, 0.0486466333, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(eps_newton, eps_brent, rel_tol=0.0, abs_tol=1e-10)


def test_bootstrap_residual_is_small() -> None:
    solver = BootstrapSolver()
    eps = solver.solve(method="newton")
    report = solver.verify_fixed_point(eps)

    assert report["equation_satisfied"]
    assert report["residual"] < 1e-12


def test_jacobian_matches_central_difference() -> None:
    solver = BootstrapSolver()
    eps = 0.0486466333
    h = 1e-6
    numerical = (
        solver.bootstrap_equation(eps + h) - solver.bootstrap_equation(eps - h)
    ) / (2.0 * h)

    assert math.isclose(solver.jacobian(eps), numerical, rel_tol=1e-9, abs_tol=1e-9)
