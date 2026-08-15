"""Independent cross-checks for the full CE cosmology math lane.

This script is deliberately a research witness, not a theory or observation
gate.  It recomputes the current-run artifacts from their definitions and also
records counterexamples whose scope is limited to the corresponding route.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from types import ModuleType


getcontext().prec = 70
ROOT = Path(__file__).resolve().parents[4]


def load_module(name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decimal_power(base: Decimal, exponent: Decimal) -> Decimal:
    return (exponent * base.ln()).exp()


def small_poisson_root(depth: Decimal) -> Decimal:
    value = (-depth).exp()
    for _ in range(80):
        residual = value.ln() + depth * (Decimal(1) - value)
        derivative = Decimal(1) / value - depth
        candidate = value - residual / derivative
        if abs(candidate - value) < Decimal("1e-65"):
            return candidate
        value = candidate
    raise ArithmeticError("decimal fixed point failed")


def old_simpson_counterexample(cosmology: ModuleType) -> tuple[float, float, float]:
    background = cosmology.Background(
        0.3079184290781293,
        0.6920815709218708,
    )
    grid = cosmology.logspace(1.0e-3, 1.0, 2001)
    old_cumulative = cosmology.compute_s_of_a(background, grid)
    index = min(range(len(grid)), key=lambda i: abs(grid[i] - 0.1))

    def interval_simpson(left: float, right: float, intervals: int) -> float:
        width = (right - left) / intervals
        total = 0.0
        for i in range(intervals):
            x0 = left + i * width
            x1 = x0 + width
            xm = 0.5 * (x0 + x1)
            total += width * (
                background.omega_l_of_a(x0)
                + 4.0 * background.omega_l_of_a(xm)
                + background.omega_l_of_a(x1)
            ) / 6.0
        return total

    denominator = interval_simpson(grid[0], 1.0, 100_000)
    numerator = interval_simpson(grid[0], grid[index], 20_000)
    reference = numerator / denominator
    relative_error = old_cumulative[index] / reference - 1.0
    even_grid_result = cosmology.simpson(
        [0.0, 1.0, 4.0, 9.0],
        [0.0, 1.0, 2.0, 3.0],
    )
    return grid[index], relative_error, even_grid_result


def old_growth_counterexample(cosmology: ModuleType) -> float:
    background = cosmology.Background(1.0, 0.0)
    count = 49
    left = math.log(1.0e-3)
    ln_grid = [left + (0.0 - left) * (i / (count - 1)) ** 1.7 for i in range(count)]
    a_grid = [math.exp(value) for value in ln_grid]
    growth, _rate = cosmology.solve_growth(background, a_grid, [1.0] * count)
    return max(abs(growth[i] - a_grid[i]) / a_grid[i] for i in range(count))


def dark_scalar_rhs(x: float, z: float, lam: float, beta: float) -> tuple[float, float]:
    root_three_halves = math.sqrt(1.5)
    omega_c = 1.0 - x * x - z * z
    common = 1.5 * (1.0 + x * x - z * z)
    return (
        -3.0 * x
        + root_three_halves * lam * z * z
        - root_three_halves * beta * omega_c
        + common * x,
        z * (-root_three_halves * lam * x + common),
    )


def finite_jacobian(
    function, x: float, z: float, step: float = 1.0e-6
) -> tuple[tuple[float, float], tuple[float, float]]:
    xp = function(x + step, z)
    xm = function(x - step, z)
    zp = function(x, z + step)
    zm = function(x, z - step)
    return (
        ((xp[0] - xm[0]) / (2.0 * step), (zp[0] - zm[0]) / (2.0 * step)),
        ((xp[1] - xm[1]) / (2.0 * step), (zp[1] - zm[1]) / (2.0 * step)),
    )


def main() -> int:
    cosmology = load_module("ce_math_cosmology", "examples/physics/cosmology.py")
    hubble = load_module("ce_math_hubble", "examples/physics/hubble_tension.py")
    background = load_module(
        "ce_math_background",
        "_workspace/ce/cosmology-full-closure-unification-20260815/"
        "artifacts/verify_background_h0_forward.py",
    )
    primordial = load_module(
        "ce_math_primordial",
        "_workspace/ce/cosmology-full-closure-unification-20260815/"
        "artifacts/verify_primordial_entropy_routes.py",
    )

    # U1: full-precision and compatibility ledgers are intentionally distinct.
    alpha = Decimal("0.11789")
    sin2 = Decimal(4) * decimal_power(alpha, Decimal(4) / Decimal(3))
    delta = sin2 * (Decimal(1) - sin2)
    depth = Decimal(3) + delta
    q = small_poisson_root(depth)
    q_legacy = small_poisson_root(Decimal("3.17776"))
    survival = Decimal(1) - q
    contraction = depth * q
    fixed_residual = q.ln() + depth * (Decimal(1) - q)

    raw_b = Decimal("0.0487")
    raw_c = Decimal("0.2623")
    raw_de = Decimal("0.6891")
    raw_sum = raw_b + raw_c + raw_de
    normalized_m = (raw_b + raw_c) / raw_sum
    normalized_de = raw_de / raw_sum

    # U2: extinction-conditioned Galton--Watson theorem and non-unique readouts.
    d = float(depth)
    qf = float(q)
    m = d * qf
    probe = 0.37
    conditioned_pgf_error = abs(
        math.exp(d * (qf * probe - 1.0)) / qf
        - math.exp(m * (probe - 1.0))
    )
    expected_total = 1.0 / (1.0 - m)
    expected_descendants = m / (1.0 - m)
    ratio_of_expectations = expected_descendants / expected_total
    readouts = {
        "aggregate_descendant_fraction": m,
        "occupancy": 1.0 - math.exp(-m),
        "exactly_one": m * math.exp(-m),
        "tree_uniform_descendant_fraction": 0.5 * m,
    }
    legacy_matter_baryon_target = float(raw_b / (raw_b + raw_c))

    # The spinodal identity is algebraic.  It does not supply a density map.
    spinodal = 1.0 / d
    spinodal_identity_error = abs(m * spinodal - qf)

    # Historical three-layer R plus the conditioned composition is a new,
    # target-aware algebraic combination, not an action-level derivation.
    coupling_sum = (0.01008 + 0.03353 + float(alpha)) / (1.0 / (2.0 * math.pi))
    r_three = float(alpha) * (3.0 + qf * coupling_sum) + float(alpha) * float(delta) * (
        1.0 + qf * float(delta)
    )
    omega_m_three = r_three / (r_three + 1.0 - m)
    contraction_triplet = (
        m * omega_m_three,
        (1.0 - m) * omega_m_three,
        1.0 - omega_m_three,
    )

    # U3 D1: interacting-vacuum fixed point.
    r_lo = float(alpha) * d
    xi = r_lo / (1.0 + r_lo)
    d1_fixed_residual = 3.0 * (xi - (1.0 - xi) * r_lo)
    d1_ratio_eigenvalue = -3.0 * (1.0 - xi)
    d1_w_eff = -(1.0 - xi)
    conserved_baryon_eigenvalue = 3.0 * d1_w_eff

    # U3 D2: independently finite-difference the dark-only autonomous system.
    omega_phi_target = 1.0 / (1.0 + r_lo)
    u_target = 0.69
    ell = math.sqrt(3.0 / (omega_phi_target - u_target))
    beta = u_target * ell
    lam = (1.0 - u_target) * ell
    x_star = math.sqrt(1.5) / ell
    z_star = math.sqrt(beta / ell + 1.5 / (ell * ell))
    d2_eom = dark_scalar_rhs(x_star, z_star, lam, beta)
    jacobian = finite_jacobian(
        lambda x_value, z_value: dark_scalar_rhs(x_value, z_value, lam, beta),
        x_star,
        z_star,
    )
    trace = jacobian[0][0] + jacobian[1][1]
    determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    discriminant = trace * trace - 4.0 * determinant
    if discriminant < 0.0:
        d2_eigenvalues = (
            complex(0.5 * trace, 0.5 * math.sqrt(-discriminant)),
            complex(0.5 * trace, -0.5 * math.sqrt(-discriminant)),
        )
    else:
        root_disc = math.sqrt(discriminant)
        d2_eigenvalues = (0.5 * (trace + root_disc), 0.5 * (trace - root_disc))

    # U4: reproduce the old arbitrary-grid counterexamples.
    simpson_a, simpson_relative_error, even_simpson_result = old_simpson_counterexample(
        cosmology
    )
    growth_max_relative_error = old_growth_counterexample(cosmology)
    omega_m_radiation_example = 1.0e-12
    omega_r_radiation_example = 1.0 - omega_m_radiation_example
    old_ricci = 12.0 - 9.0 * omega_m_radiation_example
    exact_ricci = 12.0 - 9.0 * omega_m_radiation_example - 12.0 * omega_r_radiation_example

    # U5: same-grid injection recovery is weaker than a grid-doubling check.
    omega_b_h2 = 0.0224
    omega_c_h2 = 0.1200
    omega_gamma_h2 = 2.469e-5
    omega_r_h2 = omega_gamma_h2 * (1.0 + 0.22710731766 * 3.044)
    recombination_z = 1089.0
    h_true = 0.68

    def theta_for(h_value: float, intervals: int) -> float:
        return background.compressed_theta_star(
            h_value,
            omega_b_h2,
            omega_c_h2,
            omega_r_h2,
            omega_gamma_h2,
            recombination_z,
            intervals=intervals,
        )

    theta_reference = theta_for(h_true, 2600)
    theta_grid = {count: theta_for(h_true, count) for count in (200, 400, 650, 1300, 2600)}
    h_grid_recovery: dict[int, float] = {}
    for count in (200, 400, 650, 1300):
        h_grid_recovery[count] = background.solve_h_from_theta(
            theta_reference,
            lambda h_value, count=count: theta_for(h_value, count),
            0.55,
            0.85,
            tolerance=1.0e-13,
        )

    old_theta_low_b = hubble.lcdm_theta_star_for_h(
        67.4, 73.04, 0.001, hubble.EPS_FIX, 1089.0
    )
    old_theta_high_b = hubble.lcdm_theta_star_for_h(
        67.4, 73.04, 0.1, hubble.EPS_FIX, 1089.0
    )

    # U6: action-level slow-roll outputs, running, and the horizon-definition
    # counterexample for the phase entropy route.
    starobinsky = primordial.starobinsky_slow_roll(55.0, primordial.AS_INPUT)
    dn = (
        primordial.starobinsky_slow_roll(55.0001, primordial.AS_INPUT)["n_s"]
        - primordial.starobinsky_slow_roll(54.9999, primordial.AS_INPUT)["n_s"]
    ) / 0.0002
    scalar_running = -dn / (1.0 - starobinsky["epsilon"])
    phase = primordial.phase_entropy_from_alpha_s(
        primordial.ALPHA_S_INPUT,
        boundary_correction=True,
    )
    phase_h = phase["hubble_km_s_mpc"]
    omega_lambda_phase = 0.6891
    h0_if_phase_is_true_de_sitter = phase_h / math.sqrt(omega_lambda_phase)
    phase_boundary_multiplier = math.exp(
        0.5 * math.pi * phase["delta"] * (1.0 - phase["q_ext"])
    )

    checks = {
        "u1": {
            "sin2_theta_w": str(sin2),
            "delta": str(delta),
            "D": str(depth),
            "q_exact": str(q),
            "q_legacy_rounded_D": str(q_legacy),
            "q_ledger_difference": str(q - q_legacy),
            "survival": str(survival),
            "contraction_Dq": str(contraction),
            "fixed_point_residual": str(fixed_residual),
            "raw_runtime_sum": str(raw_sum),
            "normalized_omega_m": str(normalized_m),
            "normalized_omega_de": str(normalized_de),
        },
        "u2": {
            "conditioned_pgf_error": conditioned_pgf_error,
            "expected_total": expected_total,
            "ratio_of_expectations": ratio_of_expectations,
            "alternative_readouts": readouts,
            "legacy_matter_baryon_target": legacy_matter_baryon_target,
            "best_readout_target_difference": m - legacy_matter_baryon_target,
            "spinodal_one_over_D": spinodal,
            "Dq_times_spinodal_minus_q": spinodal_identity_error,
            "three_layer_R": r_three,
            "contraction_three_layer_triplet": contraction_triplet,
            "triplet_sum_error": abs(sum(contraction_triplet) - 1.0),
        },
        "u3": {
            "D1_xi": xi,
            "D1_fixed_residual": d1_fixed_residual,
            "D1_ratio_eigenvalue": d1_ratio_eigenvalue,
            "D1_w_eff": d1_w_eff,
            "conserved_baryon_transverse_eigenvalue": conserved_baryon_eigenvalue,
            "D2_lambda": lam,
            "D2_beta": beta,
            "D2_eom_max_residual": max(abs(d2_eom[0]), abs(d2_eom[1])),
            "D2_jacobian": jacobian,
            "D2_eigenvalues": [str(value) for value in d2_eigenvalues],
        },
        "u4": {
            "old_simpson_a": simpson_a,
            "old_simpson_relative_error": simpson_relative_error,
            "even_grid_x2_old_result": even_simpson_result,
            "even_grid_x2_exact_result": 9.0,
            "old_growth_nonuniform_max_relative_error": growth_max_relative_error,
            "old_radiation_ricci": old_ricci,
            "exact_radiation_ricci": exact_ricci,
            "ricci_route_error": old_ricci - exact_ricci,
        },
        "u5": {
            "theta_grid": theta_grid,
            "h_grid_recovery": h_grid_recovery,
            "h_650_independent_grid_error": h_grid_recovery[650] - h_true,
            "legacy_theta_low_b": old_theta_low_b,
            "legacy_theta_high_b": old_theta_high_b,
            "legacy_baryon_input_difference": old_theta_low_b - old_theta_high_b,
        },
        "u6": {
            "starobinsky_N55": starobinsky,
            "first_order_scalar_running_N55": scalar_running,
            "phase_H_if_apparent_H0": phase_h,
            "phase_H0_if_true_de_sitter_entropy": h0_if_phase_is_true_de_sitter,
            "horizon_interpretation_shift": h0_if_phase_is_true_de_sitter - phase_h,
            "boundary_correction_H_multiplier": phase_boundary_multiplier,
        },
    }

    # Algebraic/numerical claims that are actually closed in this witness.
    assert abs(fixed_residual) < Decimal("1e-60")
    assert abs((q - (-depth * (Decimal(1) - q)).exp())) < Decimal("1e-60")
    assert raw_sum == Decimal("1.0001")
    assert abs(normalized_m + normalized_de - Decimal(1)) < Decimal("1e-60")
    assert conditioned_pgf_error < 2.0e-15
    assert abs(ratio_of_expectations - m) < 2.0e-15
    assert spinodal_identity_error < 2.0e-16
    assert abs(sum(contraction_triplet) - 1.0) < 2.0e-15
    assert abs(contraction_triplet[1] / contraction_triplet[2] - r_three) < 2.0e-15
    assert abs(contraction_triplet[0] / omega_m_three - m) < 2.0e-15
    assert abs(d1_fixed_residual) < 2.0e-15
    assert d1_ratio_eigenvalue < 0.0
    assert conserved_baryon_eigenvalue < 0.0
    assert max(abs(d2_eom[0]), abs(d2_eom[1])) < 2.0e-15
    assert all(complex(value).real < 0.0 for value in d2_eigenvalues)

    # Exact counterexamples and convergence witnesses.
    assert simpson_relative_error > 0.30
    assert abs(even_simpson_result - 9.0) > 6.0
    assert growth_max_relative_error > 2.0
    assert old_ricci - exact_ricci > 11.9
    assert theta_grid[1300] != theta_grid[2600]
    assert abs(h_grid_recovery[1300] - h_true) < abs(h_grid_recovery[650] - h_true)
    assert old_theta_low_b == old_theta_high_b
    assert 0.96 < starobinsky["n_s"] < 0.97
    assert 0.003 < starobinsky["r"] < 0.004
    assert h0_if_phase_is_true_de_sitter - phase_h > 13.0
    assert phase_boundary_multiplier > 1.30

    print(json.dumps(checks, indent=2, sort_keys=True))
    print("ALL FULL-COSMOLOGY MATH CROSS-CHECKS PASSED")
    print("NOTE: passing algebra does not close any missing physical bridge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
