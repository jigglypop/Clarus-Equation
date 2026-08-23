"""Independent U4/U5 verifier for a flat FLRW background and H0 readout.

This file deliberately does not import the CE product cosmology modules.  It
checks the replacement equations against analytic limits, exercises an RK4
growth solver on genuinely nonuniform ln(a) grids, and performs a synthetic
theta_* -> H0 recovery.  The synthetic recovery is a solver test, not a CE
prediction and not an observational fit.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[4]
LEGACY_H0 = ROOT / "examples" / "physics" / "hubble_tension.py"


def close(name: str, got: float, expected: float, tol: float) -> None:
    error = abs(got - expected)
    if not math.isfinite(got) or error > tol:
        raise AssertionError(
            f"{name}: got {got:.17g}, expected {expected:.17g}, "
            f"absolute error {error:.3e} > {tol:.3e}"
        )


def composite_midpoint_simpson(
    function: Callable[[float], float],
    grid: Sequence[float],
) -> float:
    """Apply Simpson on each interval, so the outer grid may be nonuniform."""
    if len(grid) < 2:
        raise ValueError("grid must have at least two points")
    total = 0.0
    previous = float(grid[0])
    if not math.isfinite(previous):
        raise ValueError("grid must be finite")
    for raw_right in grid[1:]:
        right = float(raw_right)
        if not math.isfinite(right) or right <= previous:
            raise ValueError("grid must be finite and strictly increasing")
        middle = 0.5 * (previous + right)
        width = right - previous
        total += width * (
            function(previous) + 4.0 * function(middle) + function(right)
        ) / 6.0
        previous = right
    return total


def warped_grid(left: float, right: float, intervals: int, power: float) -> list[float]:
    if intervals < 1 or power <= 0.0 or right <= left:
        raise ValueError("invalid warped-grid request")
    return [
        left + (right - left) * (index / intervals) ** power
        for index in range(intervals + 1)
    ]


@dataclass(frozen=True)
class FlatLambdaBackground:
    """Flat matter + massless-radiation + Lambda background."""

    omega_m0: float
    omega_r0: float
    omega_lambda0: float

    def __post_init__(self) -> None:
        values = (self.omega_m0, self.omega_r0, self.omega_lambda0)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("density fractions must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2.0e-14):
            raise ValueError("flat density fractions must sum to one")

    def e2(self, a: float) -> float:
        if not math.isfinite(a) or a <= 0.0:
            raise ValueError("scale factor must be finite and positive")
        return (
            self.omega_r0 * a**-4
            + self.omega_m0 * a**-3
            + self.omega_lambda0
        )

    def e(self, a: float) -> float:
        return math.sqrt(self.e2(a))

    def fractions(self, a: float) -> tuple[float, float, float]:
        e2 = self.e2(a)
        return (
            self.omega_m0 * a**-3 / e2,
            self.omega_r0 * a**-4 / e2,
            self.omega_lambda0 / e2,
        )

    def dlnh_dln_a(self, a: float) -> float:
        e2 = self.e2(a)
        derivative = -3.0 * self.omega_m0 * a**-3 - 4.0 * self.omega_r0 * a**-4
        return 0.5 * derivative / e2

    def ricci_over_h2_kinematic(self, a: float) -> float:
        # Sign convention R = +6 (dot(H) + 2 H^2) for flat FLRW.
        return 6.0 * (2.0 + self.dlnh_dln_a(a))

    def ricci_over_h2_trace(self, a: float) -> float:
        omega_m, _omega_r, omega_lambda = self.fractions(a)
        return 3.0 * omega_m + 12.0 * omega_lambda


def comoving_distance_h0_over_c(
    background: FlatLambdaBackground,
    z: float,
    intervals: int = 1200,
) -> float:
    if z < 0.0:
        raise ValueError("redshift must be non-negative")
    grid = warped_grid(0.0, z, intervals, 1.35)
    return composite_midpoint_simpson(
        lambda redshift: 1.0 / background.e(1.0 / (1.0 + redshift)),
        grid,
    )


def sound_horizon_h0_over_c(
    background: FlatLambdaBackground,
    a_stop: float,
    omega_b_h2: float = 0.0,
    omega_gamma_h2: float = 1.0,
    intervals: int = 1200,
) -> float:
    """Return r_s H0/c; integrate with a=a_stop*u^2 to regularize a=0."""
    if not 0.0 < a_stop <= 1.0:
        raise ValueError("a_stop must lie in (0, 1]")
    if omega_b_h2 < 0.0 or omega_gamma_h2 <= 0.0:
        raise ValueError("invalid physical baryon/photon density")

    def transformed_integrand(u: float) -> float:
        if u == 0.0:
            if background.omega_r0 > 0.0:
                return 0.0
            if background.omega_m0 > 0.0:
                return 2.0 * math.sqrt(a_stop) / (
                    math.sqrt(3.0) * math.sqrt(background.omega_m0)
                )
            raise ValueError("sound horizon diverges without an early matter/radiation term")
        a = a_stop * u * u
        baryon_loading = 3.0 * omega_b_h2 * a / (4.0 * omega_gamma_h2)
        sound_speed_over_c = 1.0 / math.sqrt(3.0 * (1.0 + baryon_loading))
        da_du = 2.0 * a_stop * u
        return sound_speed_over_c * da_du / (a * a * background.e(a))

    return composite_midpoint_simpson(
        transformed_integrand,
        warped_grid(0.0, 1.0, intervals, 1.27),
    )


def growth_rhs(
    background: FlatLambdaBackground,
    ln_a: float,
    growth: float,
    growth_prime: float,
    mu: Callable[[float], float],
) -> tuple[float, float]:
    a = math.exp(ln_a)
    omega_m, _omega_r, _omega_lambda = background.fractions(a)
    friction = 2.0 + background.dlnh_dln_a(a)
    return (
        growth_prime,
        -friction * growth_prime + 1.5 * mu(a) * omega_m * growth,
    )


def solve_growth_nonuniform(
    background: FlatLambdaBackground,
    ln_a_grid: Sequence[float],
    growth_initial: float,
    growth_prime_initial: float,
    mu: Callable[[float], float] = lambda _a: 1.0,
) -> tuple[list[float], list[float]]:
    """Classical RK4 with the actual local step of each nonuniform interval."""
    if len(ln_a_grid) < 2:
        raise ValueError("growth grid must have at least two points")
    growth_values = [float(growth_initial)]
    prime_values = [float(growth_prime_initial)]
    for index in range(len(ln_a_grid) - 1):
        left = float(ln_a_grid[index])
        right = float(ln_a_grid[index + 1])
        if not math.isfinite(left) or not math.isfinite(right) or right <= left:
            raise ValueError("ln(a) grid must be finite and strictly increasing")
        step = right - left
        d_value = growth_values[-1]
        p_value = prime_values[-1]
        k1 = growth_rhs(background, left, d_value, p_value, mu)
        k2 = growth_rhs(
            background,
            left + 0.5 * step,
            d_value + 0.5 * step * k1[0],
            p_value + 0.5 * step * k1[1],
            mu,
        )
        k3 = growth_rhs(
            background,
            left + 0.5 * step,
            d_value + 0.5 * step * k2[0],
            p_value + 0.5 * step * k2[1],
            mu,
        )
        k4 = growth_rhs(
            background,
            right,
            d_value + step * k3[0],
            p_value + step * k3[1],
            mu,
        )
        growth_values.append(
            d_value + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        )
        prime_values.append(
            p_value + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        )
    return growth_values, prime_values


def heath_growth_raw(background: FlatLambdaBackground, a: float, intervals: int = 2000) -> float:
    """Exact growing-mode quadrature for matter + Lambda (radiation must be zero)."""
    if background.omega_r0 != 0.0 or background.omega_m0 <= 0.0:
        raise ValueError("Heath quadrature here requires matter + Lambda")

    def transformed(u: float) -> float:
        if u == 0.0:
            return 0.0
        scale = a * u * u
        da_du = 2.0 * a * u
        return da_du / (scale**3 * background.e(scale) ** 3)

    integral = composite_midpoint_simpson(
        transformed,
        warped_grid(0.0, 1.0, intervals, 1.21),
    )
    return 2.5 * background.omega_m0 * background.e(a) * integral


def heath_initial_state(
    background: FlatLambdaBackground,
    a: float,
    intervals: int = 2400,
) -> tuple[float, float]:
    value = heath_growth_raw(background, a, intervals)

    def transformed(u: float) -> float:
        if u == 0.0:
            return 0.0
        scale = a * u * u
        return 2.0 * a * u / (scale**3 * background.e(scale) ** 3)

    integral = composite_midpoint_simpson(
        transformed,
        warped_grid(0.0, 1.0, intervals, 1.21),
    )
    endpoint_integrand = 1.0 / (a**3 * background.e(a) ** 3)
    logarithmic_slope = background.dlnh_dln_a(a) + a * endpoint_integrand / integral
    return value, value * logarithmic_slope


def background_from_physical_densities(
    h: float,
    omega_m_h2: float,
    omega_r_h2: float,
) -> FlatLambdaBackground:
    if h <= 0.0:
        raise ValueError("h must be positive")
    omega_m0 = omega_m_h2 / (h * h)
    omega_r0 = omega_r_h2 / (h * h)
    omega_lambda0 = 1.0 - omega_m0 - omega_r0
    return FlatLambdaBackground(omega_m0, omega_r0, omega_lambda0)


def compressed_theta_star(
    h: float,
    omega_b_h2: float,
    omega_c_h2: float,
    omega_r_h2: float,
    omega_gamma_h2: float,
    z_star: float,
    intervals: int = 1000,
) -> float:
    """Controlled synthetic theta_*; z_* is supplied, not derived here."""
    background = background_from_physical_densities(
        h,
        omega_b_h2 + omega_c_h2,
        omega_r_h2,
    )
    a_star = 1.0 / (1.0 + z_star)
    sound_horizon = sound_horizon_h0_over_c(
        background,
        a_star,
        omega_b_h2=omega_b_h2,
        omega_gamma_h2=omega_gamma_h2,
        intervals=intervals,
    )
    distance = comoving_distance_h0_over_c(background, z_star, intervals=intervals)
    return sound_horizon / distance


def solve_h_from_theta(
    theta_target: float,
    theta_of_h: Callable[[float], float],
    h_low: float,
    h_high: float,
    tolerance: float = 1.0e-11,
) -> float:
    if theta_target <= 0.0 or h_high <= h_low or tolerance <= 0.0:
        raise ValueError("invalid theta root request")

    def residual(h: float) -> float:
        theta = theta_of_h(h)
        if not math.isfinite(theta) or theta <= 0.0:
            raise ArithmeticError("theta model returned a nonphysical value")
        return math.log(theta / theta_target)

    f_low = residual(h_low)
    f_high = residual(h_high)
    if f_low == 0.0:
        return h_low
    if f_high == 0.0:
        return h_high
    if f_low * f_high > 0.0:
        raise ValueError("theta target is not bracketed")
    for _ in range(100):
        middle = 0.5 * (h_low + h_high)
        f_middle = residual(middle)
        if abs(f_middle) <= tolerance or h_high - h_low <= tolerance:
            return middle
        if f_low * f_middle <= 0.0:
            h_high = middle
            f_high = f_middle
        else:
            h_low = middle
            f_low = f_middle
    raise ArithmeticError("theta root failed to converge")


def legacy_unused_inputs() -> tuple[bool, bool]:
    """Report (omega_b_h2 unused, local h2 assignment unused) in the old toy."""
    tree = ast.parse(LEGACY_H0.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    theta_function = functions["lcdm_theta_star_for_h"]
    loaded_theta_names = {
        node.id
        for node in ast.walk(theta_function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    omega_b_unused = "om_b_h2" not in loaded_theta_names
    h2_loads = sum(
        1
        for node in ast.walk(theta_function)
        if isinstance(node, ast.Name)
        and node.id == "h2"
        and isinstance(node.ctx, ast.Load)
    )
    return omega_b_unused, h2_loads == 0


def run_checks() -> None:
    # Arbitrary-grid quadrature must use every interval and remain exact for a cubic.
    irregular = [0.0, 0.01, 0.13, 0.21, 0.62, 0.91, 1.0]
    close(
        "nonuniform Simpson cubic",
        composite_midpoint_simpson(lambda x: x**3, irregular),
        0.25,
        3.0e-15,
    )

    radiation = FlatLambdaBackground(0.0, 1.0, 0.0)
    matter = FlatLambdaBackground(1.0, 0.0, 0.0)
    de_sitter = FlatLambdaBackground(0.0, 0.0, 1.0)
    mixed = FlatLambdaBackground(0.31, 9.0e-5, 1.0 - 0.31 - 9.0e-5)
    close("flat normalization", mixed.e(1.0), 1.0, 2.0e-15)
    close("radiation R/H2", radiation.ricci_over_h2_kinematic(1.0e-5), 0.0, 2.0e-14)
    close("matter R/H2", matter.ricci_over_h2_kinematic(0.2), 3.0, 2.0e-14)
    close("de Sitter R/H2", de_sitter.ricci_over_h2_kinematic(0.2), 12.0, 2.0e-14)
    for a in (1.0e-6, 3.0e-4, 0.2, 1.0):
        close(
            f"Ricci trace/kinematic a={a}",
            mixed.ricci_over_h2_kinematic(a),
            mixed.ricci_over_h2_trace(a),
            3.0e-14,
        )

    a_stop = 8.0e-4
    close(
        "radiation sound horizon",
        sound_horizon_h0_over_c(radiation, a_stop, intervals=500),
        a_stop / math.sqrt(3.0),
        2.0e-14,
    )
    close(
        "matter sound horizon",
        sound_horizon_h0_over_c(matter, a_stop, intervals=500),
        2.0 * math.sqrt(a_stop) / math.sqrt(3.0),
        3.0e-13,
    )
    z_test = 3.0
    close(
        "matter comoving distance",
        comoving_distance_h0_over_c(matter, z_test, intervals=500),
        2.0 * (1.0 - 1.0 / math.sqrt(1.0 + z_test)),
        3.0e-12,
    )
    close(
        "de Sitter comoving distance",
        comoving_distance_h0_over_c(de_sitter, z_test, intervals=500),
        z_test,
        3.0e-13,
    )

    # EdS D=a on a deliberately nonuniform ln(a) grid; verify fourth-order refinement.
    ln_start = math.log(1.0e-3)
    eds_errors: list[float] = []
    for intervals in (24, 48, 96):
        grid = warped_grid(ln_start, 0.0, intervals, 1.61)
        values, _primes = solve_growth_nonuniform(
            matter,
            grid,
            math.exp(ln_start),
            math.exp(ln_start),
        )
        eds_errors.append(abs(values[-1] - 1.0))
    if not (eds_errors[1] < eds_errors[0] / 8.0 and eds_errors[2] < eds_errors[1] / 8.0):
        raise AssertionError(f"nonuniform RK4 did not converge at fourth-order scale: {eds_errors}")

    # Constant growing modes in the source-free radiation and de Sitter limits.
    limit_grid = warped_grid(math.log(1.0e-4), 0.0, 80, 1.43)
    for label, background in (("radiation", radiation), ("de Sitter", de_sitter)):
        values, primes = solve_growth_nonuniform(background, limit_grid, 1.0, 0.0)
        close(f"{label} constant growth", values[-1], 1.0, 2.0e-14)
        close(f"{label} constant growth prime", primes[-1], 0.0, 2.0e-14)

    # Matter+Lambda RK4 must agree with the exact Heath quadrature.
    lambda_background = FlatLambdaBackground(0.31, 0.0, 0.69)
    a_initial = 1.0e-3
    initial, initial_prime = heath_initial_state(lambda_background, a_initial)
    lambda_grid = warped_grid(math.log(a_initial), 0.0, 220, 1.37)
    values, _primes = solve_growth_nonuniform(
        lambda_background,
        lambda_grid,
        initial,
        initial_prime,
    )
    normalization = heath_growth_raw(lambda_background, 1.0)
    sample_indices = (40, 100, 170, 220)
    for index in sample_indices:
        a = math.exp(lambda_grid[index])
        expected = heath_growth_raw(lambda_background, a) / normalization
        got = values[index] / values[-1]
        close(f"Heath/RK4 agreement a={a:.4g}", got, expected, 2.0e-8)

    # Synthetic angular-scale inversion: all early inputs are active and h is recovered.
    omega_b_h2 = 0.0224
    omega_c_h2 = 0.1200
    omega_gamma_h2 = 2.469e-5
    omega_r_h2 = omega_gamma_h2 * (1.0 + 0.22710731766 * 3.044)
    z_star = 1089.0
    h_true = 0.68

    def theta_model(h: float, baryon: float = omega_b_h2) -> float:
        return compressed_theta_star(
            h,
            baryon,
            omega_c_h2,
            omega_r_h2,
            omega_gamma_h2,
            z_star,
            intervals=650,
        )

    theta_target = theta_model(h_true)
    recovered = solve_h_from_theta(theta_target, theta_model, 0.55, 0.85)
    close("synthetic theta -> h", recovered, h_true, 2.0e-8)
    baryon_shift = abs(theta_model(h_true, 0.018) - theta_model(h_true, 0.028))
    if baryon_shift <= 1.0e-5:
        raise AssertionError("omega_b h2 is not active in the controlled theta model")

    legacy_omega_b_unused, legacy_h2_unused = legacy_unused_inputs()
    print("nonuniform_quadrature PASS")
    print("flrw_limits PASS")
    print("ricci_trace_identity PASS")
    print("sound_horizon_limits PASS")
    print("growth_nonuniform_grid_convergence PASS", " ".join(f"{e:.3e}" for e in eds_errors))
    print("growth_heath_crosscheck PASS")
    print("synthetic_theta_h_recovery PASS", f"h={recovered:.10f}")
    print("controlled_omega_b_input_active PASS", f"delta_theta={baryon_shift:.6e}")
    print("legacy_omega_b_h2_unused", legacy_omega_b_unused)
    print("legacy_local_h2_assignment_unused", legacy_h2_unused)
    print("STATUS COMPLETE: numerical verifier only; no observational prediction")


if __name__ == "__main__":
    run_checks()
