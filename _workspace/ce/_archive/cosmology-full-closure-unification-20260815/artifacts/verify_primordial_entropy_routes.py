"""Independent calculations for action-level primordial and entropy routes."""

from __future__ import annotations

import math


ALPHA_S_INPUT = 0.11789
AS_INPUT = 2.099e-9
M_PLANCK_NONREDUCED_EV = 1.220910e28
INV_HBAR_PER_EV_SECOND = 1.519267447e15
MPC_KM = 3.085677581e19


def bisect(function, lower: float, upper: float, iterations: int = 240) -> float:
    f_lower = function(lower)
    f_upper = function(upper)
    if f_lower > 0.0 or f_upper < 0.0:
        raise ValueError("invalid increasing-function bracket")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if function(midpoint) <= 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def starobinsky_slow_roll(efolds: float, scalar_amplitude: float) -> dict[str, float]:
    """Exact potential-slow-roll values for V=V0(1-exp(-sqrt(2/3)phi))^2."""
    y_end = 1.0 + 2.0 / math.sqrt(3.0)

    def efold_residual(y_value: float) -> float:
        return 0.75 * (
            (y_value - y_end) - math.log(y_value / y_end)
        ) - efolds

    y_star = bisect(efold_residual, y_end, 1000.0)
    epsilon = 4.0 / (3.0 * (y_star - 1.0) ** 2)
    eta = 4.0 * (2.0 - y_star) / (3.0 * (y_star - 1.0) ** 2)
    n_s = 1.0 - 6.0 * epsilon + 2.0 * eta
    tensor_ratio = 16.0 * epsilon
    potential_shape = (1.0 - 1.0 / y_star) ** 2
    v0_over_mpl4 = scalar_amplitude * 24.0 * math.pi**2 * epsilon / potential_shape
    scalaron_mass_over_mpl = math.sqrt(4.0 * v0_over_mpl4 / 3.0)
    return {
        "efolds": efolds,
        "y_star": y_star,
        "epsilon": epsilon,
        "eta": eta,
        "n_s": n_s,
        "r": tensor_ratio,
        "v0_over_mpl4": v0_over_mpl4,
        "scalaron_mass_over_mpl": scalaron_mass_over_mpl,
    }


def small_fixed_point(depth: float) -> float:
    value = math.exp(-depth)
    for _ in range(1000):
        next_value = math.exp(-depth * (1.0 - value))
        if abs(next_value - value) <= 1.0e-16:
            return next_value
        value = next_value
    raise RuntimeError("fixed point failed to converge")


def phase_entropy_from_sin2(sin2_theta_w: float, *, boundary_correction: bool) -> dict[str, float]:
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    depth = 3.0 + delta
    q_ext = small_fixed_point(depth)
    n_cells = 0.5 * 3.0 * depth * 12.0
    log_entropy = 0.5 * math.pi**2 * n_cells
    if boundary_correction:
        log_entropy -= math.pi * delta * (1.0 - q_ext)
    hubble_ev = M_PLANCK_NONREDUCED_EV * math.sqrt(math.pi) * math.exp(-0.5 * log_entropy)
    hubble_km_s_mpc = hubble_ev * INV_HBAR_PER_EV_SECOND * MPC_KM
    return {
        "sin2_theta_w": sin2_theta_w,
        "delta": delta,
        "depth": depth,
        "q_ext": q_ext,
        "n_cells": n_cells,
        "log_entropy": log_entropy,
        "hubble_ev": hubble_ev,
        "hubble_km_s_mpc": hubble_km_s_mpc,
    }


def phase_entropy_from_alpha_s(alpha_s: float, *, boundary_correction: bool) -> dict[str, float]:
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    result = phase_entropy_from_sin2(
        sin2_theta_w,
        boundary_correction=boundary_correction,
    )
    result["alpha_s"] = alpha_s
    return result


def finite_log_derivative(function, value: float, step: float = 1.0e-6) -> float:
    return (math.log(function(value + step)) - math.log(function(value - step))) / (2.0 * step)


def main() -> int:
    primordial = [starobinsky_slow_roll(n_value, AS_INPUT) for n_value in (50.0, 55.0, 60.0)]
    for row in primordial:
        assert 0.0 < row["epsilon"] < 1.0
        assert 0.0 < row["r"] < 0.01
        assert 0.95 < row["n_s"] < 0.98
        assert 1.0e-5 < row["scalaron_mass_over_mpl"] < 1.5e-5

    entropy_without_boundary = phase_entropy_from_alpha_s(
        ALPHA_S_INPUT,
        boundary_correction=False,
    )
    entropy_with_boundary = phase_entropy_from_alpha_s(
        ALPHA_S_INPUT,
        boundary_correction=True,
    )
    assert math.isclose(
        entropy_with_boundary["q_ext"],
        math.exp(
            -entropy_with_boundary["depth"] * (1.0 - entropy_with_boundary["q_ext"])
        ),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert entropy_with_boundary["hubble_km_s_mpc"] > entropy_without_boundary[
        "hubble_km_s_mpc"
    ]

    derivative_alpha = finite_log_derivative(
        lambda alpha: phase_entropy_from_alpha_s(
            alpha,
            boundary_correction=True,
        )["hubble_km_s_mpc"],
        ALPHA_S_INPUT,
    )
    one_cell_factor = math.exp(-math.pi**2 / 4.0)

    print("Starobinsky action route (A_s is an external normalization input)")
    for row in primordial:
        print(
            f"N={row['efolds']:.0f} ns={row['n_s']:.12g} r={row['r']:.12g} "
            f"V0/Mpl^4={row['v0_over_mpl4']:.12g} M/Mpl={row['scalaron_mass_over_mpl']:.12g}"
        )
    print()
    print("Phase-entropy identity route (the phase-area law is not derived here)")
    print(
        "without boundary correction "
        f"logS={entropy_without_boundary['log_entropy']:.12g} "
        f"H0={entropy_without_boundary['hubble_km_s_mpc']:.12g} km/s/Mpc"
    )
    print(
        "with boundary correction    "
        f"logS={entropy_with_boundary['log_entropy']:.12g} "
        f"H0={entropy_with_boundary['hubble_km_s_mpc']:.12g} km/s/Mpc"
    )
    print(f"d ln H0 / d alpha_s={derivative_alpha:.12g}")
    print(f"one-unit N_cells shift multiplies H0 by {one_cell_factor:.12g}")
    print("ALL PRIMORDIAL/ENTROPY ROUTE CALCULATIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
