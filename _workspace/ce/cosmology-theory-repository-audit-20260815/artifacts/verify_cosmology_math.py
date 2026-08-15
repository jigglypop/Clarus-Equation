"""Independent numerical checks for the CE cosmology repository audit.

This file intentionally recomputes the mathematical claims from their
definitions.  Repository implementations are imported only after an
independent reference value has been constructed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON_ROOT = ROOT / "reality_stone" / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from examples.physics import cosmology as legacy_cosmology  # noqa: E402
from examples.physics import hubble_tension  # noqa: E402
from examples.physics.ce_residual_forward_model import (  # noqa: E402
    CEForwardParams,
    DESI_DR2_ALL_COVARIANCE,
    dark_energy_scale,
    dlnh_dln_a,
    e2_of_a,
    luminosity_distance_mpc,
    parse_covariance_matrix,
    s8_today,
    solve_growth,
)
from examples.physics.cosmological_constant_holographic_gate import (  # noqa: E402
    ALPHA_S,
    D_SPACE,
    M_PL_EV,
    N_GAUGE,
    OMEGA_LAMBDA,
    derive_entropy,
    rho_lambda_quarter_mev,
)
from examples.physics.cosmology_discrimination_gates import chain  # noqa: E402
from examples.physics.primordial_spectrum_readout_gate import (  # noqa: E402
    D_EFF as PRIMORDIAL_D_EFF,
    N_E,
    OBS_AS_1E9,
    X as PRIMORDIAL_X,
    inferred_geometry_exponent,
    readouts,
)
from reality_stone.clarus.bootstrap_solver import BootstrapSolver  # noqa: E402
from reality_stone.clarus.constants import (  # noqa: E402
    ACTIVE_RATIO,
    BACKGROUND_RATIO,
    STRUCT_RATIO,
)


def bisect_nontrivial_fixed_point(d_eff: float, tol: float = 2.0e-16) -> float:
    """Unique root of q-exp[-D(1-q)] in (0, 1/D), independently bracketed."""

    if d_eff <= 1.0:
        raise ValueError("the nontrivial branch requires D>1")

    def residual(q: float) -> float:
        return q - math.exp(-d_eff * (1.0 - q))

    lo = 0.0
    hi = 1.0 / d_eff
    f_lo = residual(lo)
    f_hi = residual(hi)
    if not f_lo < 0.0 < f_hi:
        raise AssertionError("nontrivial fixed point was not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break
    return 0.5 * (lo + hi)


def adaptive_simpson(function, left: float, right: float, tol: float = 1.0e-13) -> float:
    """Small independent adaptive Simpson integrator."""

    def simp(a: float, b: float) -> float:
        m = 0.5 * (a + b)
        return (b - a) * (function(a) + 4.0 * function(m) + function(b)) / 6.0

    whole = simp(left, right)

    def recurse(a: float, b: float, estimate: float, local_tol: float, depth: int) -> float:
        m = 0.5 * (a + b)
        left_estimate = simp(a, m)
        right_estimate = simp(m, b)
        refined = left_estimate + right_estimate
        if depth <= 0 or abs(refined - estimate) <= 15.0 * local_tol:
            return refined + (refined - estimate) / 15.0
        return recurse(a, m, left_estimate, local_tol / 2.0, depth - 1) + recurse(
            m, b, right_estimate, local_tol / 2.0, depth - 1
        )

    return recurse(left, right, whole, tol, 24)


def cholesky_min_pivot(matrix: tuple[tuple[float, ...], ...]) -> float:
    """Return the smallest Cholesky pivot; fail if the matrix is not SPD."""

    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    pivots: list[float] = []
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0.0:
                    raise ValueError(f"not SPD at pivot {i}: {value}")
                lower[i][j] = math.sqrt(value)
                pivots.append(value)
            else:
                lower[i][j] = value / lower[j][j]
    return min(pivots)


def fixed_point_checks() -> dict[str, float | bool]:
    sin2 = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta_exact = sin2 * (1.0 - sin2)
    d_exact = 3.0 + delta_exact
    d_legacy = BootstrapSolver.D_EFF
    q_exact = bisect_nontrivial_fixed_point(d_exact)
    q_legacy_definition = bisect_nontrivial_fixed_point(d_legacy)
    legacy_solver = BootstrapSolver().solve(method="brent")
    return {
        "alpha_s": ALPHA_S,
        "sin2": sin2,
        "delta_exact": delta_exact,
        "d_exact": d_exact,
        "d_legacy": d_legacy,
        "d_difference": d_legacy - d_exact,
        "q_exact": q_exact,
        "q_exact_residual": abs(q_exact - math.exp(-d_exact * (1.0 - q_exact))),
        "q_legacy_definition": q_legacy_definition,
        "q_legacy_solver": legacy_solver,
        "q_legacy_solver_error": legacy_solver - q_legacy_definition,
        "q_exact_vs_legacy": q_legacy_definition - q_exact,
        "active_ratio": ACTIVE_RATIO,
        "active_ratio_error": ACTIVE_RATIO - q_exact,
        "active_ratio_equation_residual_exact_D": abs(
            ACTIVE_RATIO - math.exp(-d_exact * (1.0 - ACTIVE_RATIO))
        ),
        "trivial_root_residual": abs(1.0 - math.exp(-d_exact * (1.0 - 1.0))),
        "nontrivial_below_1_over_D": q_exact < 1.0 / d_exact,
        "nontrivial_local_multiplier": d_exact * q_exact,
    }


def density_mapping_checks() -> dict[str, float]:
    exact = fixed_point_checks()
    q = float(exact["q_exact"])
    d_eff = float(exact["d_exact"])
    r_lo = ALPHA_S * d_eff
    omega_lambda_lo = (1.0 - q) / (1.0 + r_lo)
    omega_dm_lo = (1.0 - q) * r_lo / (1.0 + r_lo)
    constants_sum = ACTIVE_RATIO + STRUCT_RATIO + BACKGROUND_RATIO
    constants_omega_m = ACTIVE_RATIO + STRUCT_RATIO
    constants_omega_m_normalized = constants_omega_m / constants_sum
    constants_omega_l_normalized = BACKGROUND_RATIO / constants_sum
    three_layer = chain(ALPHA_S)
    return {
        "r_lo": r_lo,
        "omega_b_fixed_point": q,
        "omega_dm_lo": omega_dm_lo,
        "omega_lambda_lo": omega_lambda_lo,
        "omega_m_lo": q + omega_dm_lo,
        "constants_sum": constants_sum,
        "constants_omega_m_raw": constants_omega_m,
        "constants_omega_lambda_raw": BACKGROUND_RATIO,
        "constants_omega_m_normalized": constants_omega_m_normalized,
        "constants_omega_lambda_normalized": constants_omega_l_normalized,
        "three_layer_ratio": three_layer["r_3layer"],
        "three_layer_omega_m": three_layer["om_m"],
        "three_layer_omega_lambda": three_layer["om_l"],
        "lo_vs_constants_omega_m": constants_omega_m_normalized - (q + omega_dm_lo),
        "three_layer_vs_constants_omega_m": constants_omega_m_normalized
        - three_layer["om_m"],
    }


def flrw_cpl_checks() -> dict[str, float | bool]:
    params = CEForwardParams()
    a = 0.37
    finite_step = 1.0e-6
    derivative_numeric = (
        0.5 * math.log(e2_of_a(a * math.exp(finite_step), params))
        - 0.5 * math.log(e2_of_a(a * math.exp(-finite_step), params))
    ) / (2.0 * finite_step)
    derivative_code = dlnh_dln_a(a, params)

    z = 1.0
    reference_integral = adaptive_simpson(
        lambda zz: 1.0
        / math.sqrt(
            params.omega_m0_background * (1.0 + zz) ** 3
            + params.omega_lambda0_background
        ),
        0.0,
        z,
    )
    distance_reference = 299792.458 / params.h0 * (1.0 + z) * reference_integral
    distance_code = luminosity_distance_mpc(z, params, n=2001)

    om = params.omega_m0_background
    ol = params.omega_lambda0_background
    age_analytic = 2.0 / (3.0 * math.sqrt(ol)) * math.asinh(math.sqrt(ol / om))
    age_numeric = adaptive_simpson(
        lambda x: 1.0 / math.sqrt(om * math.exp(-3.0 * x) + ol),
        math.log(1.0e-8),
        0.0,
    )

    a_grid, d_grid, f_grid = solve_growth(params, a_min=1.0e-3, n=4001)
    a_probe = 0.5
    code_growth = legacy_cosmology.interp_linear(a_grid, d_grid, a_probe)
    code_growth_rate = legacy_cosmology.interp_linear(a_grid, f_grid, a_probe)

    def unnormalized_growth(a_value: float) -> float:
        e_value = math.sqrt(om * a_value ** -3 + ol)
        integral = adaptive_simpson(
            lambda aa: 1.0 / (aa**3 * (om * aa**-3 + ol) ** 1.5),
            1.0e-8,
            a_value,
            tol=2.0e-12,
        )
        return 2.5 * om * e_value * integral

    growth_reference = unnormalized_growth(a_probe) / unnormalized_growth(1.0)
    growth_h = 1.0e-5
    growth_plus = unnormalized_growth(a_probe * math.exp(growth_h))
    growth_minus = unnormalized_growth(a_probe * math.exp(-growth_h))
    growth_rate_reference = (
        math.log(growth_plus) - math.log(growth_minus)
    ) / (2.0 * growth_h)

    cpl_a = 0.61
    cpl_w0 = -0.83
    cpl_wa = 0.27
    cpl_scale = dark_energy_scale(cpl_a, cpl_w0, cpl_wa)
    cpl_scale_definition = math.exp(
        -3.0
        * adaptive_simpson(
            lambda ln_aa: 1.0 + cpl_w0 + cpl_wa * (1.0 - math.exp(ln_aa)),
            0.0,
            math.log(cpl_a),
        )
    )

    return {
        "e0": math.sqrt(e2_of_a(1.0, params)),
        "cpl_scale": cpl_scale,
        "cpl_scale_definition": cpl_scale_definition,
        "cpl_scale_error": cpl_scale - cpl_scale_definition,
        "dlnh_code": derivative_code,
        "dlnh_numeric": derivative_numeric,
        "dlnh_error": derivative_code - derivative_numeric,
        "distance_mpc_code_z1": distance_code,
        "distance_mpc_reference_z1": distance_reference,
        "distance_relative_error": distance_code / distance_reference - 1.0,
        "h0_t0_analytic": age_analytic,
        "h0_t0_numeric": age_numeric,
        "h0_t0_error": age_numeric - age_analytic,
        "growth_D_code_a0p5": code_growth,
        "growth_D_reference_a0p5": growth_reference,
        "growth_D_relative_error": code_growth / growth_reference - 1.0,
        "growth_f_code_a0p5": code_growth_rate,
        "growth_f_reference_a0p5": growth_rate_reference,
        "growth_f_error": code_growth_rate - growth_rate_reference,
    }


def cumulative_s_nonuniform_grid_counterexample() -> dict[str, float]:
    bg = legacy_cosmology.Background(omega_m0=0.315, omega_l0=0.685)
    a_grid = legacy_cosmology.logspace(1.0e-3, 1.0, 2001)
    s_code = legacy_cosmology.compute_s_of_a(bg, a_grid)

    def primitive(a_value: float) -> float:
        return adaptive_simpson(bg.omega_l_of_a, 1.0e-3, a_value, tol=2.0e-12)

    denominator = primitive(1.0)
    probes = (0.01, 0.1, 0.5, 0.9)
    result: dict[str, float] = {}
    for probe in probes:
        index = min(range(len(a_grid)), key=lambda i: abs(a_grid[i] - probe))
        expected = primitive(a_grid[index]) / denominator
        result[f"a_{probe}_grid"] = a_grid[index]
        result[f"a_{probe}_code"] = s_code[index]
        result[f"a_{probe}_reference"] = expected
        result[f"a_{probe}_error"] = s_code[index] - expected
    return result


def hubble_counterexamples() -> dict[str, float | bool]:
    eps_today = hubble_tension.OMEGA_M_CANON
    omega_m0, omega_lambda0 = hubble_tension.eps_to_omegas(eps_today)

    def fractions(a: float) -> tuple[float, float, float]:
        matter = omega_m0 * a**-3
        vacuum = omega_lambda0
        radiation = hubble_tension.OMEGA_R0 * a**-4
        total = matter + vacuum + radiation
        return matter / total, vacuum / total, radiation / total

    result: dict[str, float | bool] = {}
    for a in (1.0, 1.0e-3, 1.0e-6):
        omega_m, _omega_l, omega_r = fractions(a)
        ricci_exact = 12.0 - 9.0 * omega_m - 12.0 * omega_r
        ricci_code = 12.0 - 9.0 * omega_m
        result[f"a_{a}_omega_m"] = omega_m
        result[f"a_{a}_omega_r"] = omega_r
        result[f"a_{a}_ricci_exact_over_h2"] = ricci_exact
        result[f"a_{a}_ricci_code_over_h2"] = ricci_code
        result[f"a_{a}_ricci_error"] = ricci_code - ricci_exact

    theta_low_baryon = hubble_tension.lcdm_theta_star_for_h(
        67.4, 73.04, 0.001, eps_today, hubble_tension.Z_STAR, n=1001
    )
    theta_high_baryon = hubble_tension.lcdm_theta_star_for_h(
        67.4, 73.04, 0.1, eps_today, hubble_tension.Z_STAR, n=1001
    )
    result["theta_omega_b_h2_0p001"] = theta_low_baryon
    result["theta_omega_b_h2_0p1"] = theta_high_baryon
    result["theta_baryon_argument_difference"] = theta_high_baryon - theta_low_baryon
    result["omega_b_h2_argument_is_unused"] = theta_high_baryon == theta_low_baryon
    result["E_today_code"] = math.sqrt(
        omega_m0 + omega_lambda0 + hubble_tension.OMEGA_R0
    )
    return result


def holographic_checks() -> dict[str, float]:
    derived = derive_entropy()
    log_s = derived["log_s"]
    hbar_ev_s = 6.582119569e-16
    mpc_km = 3.0856775814913673e19
    h_ev = M_PL_EV * math.sqrt(math.pi) * math.exp(-0.5 * log_s)
    h_s_inverse = h_ev / hbar_ev_s
    h_km_s_mpc = h_s_inverse * mpc_km
    reduced_planck_ev = M_PL_EV / math.sqrt(8.0 * math.pi)
    rho_nonreduced = OMEGA_LAMBDA * (3.0 / 8.0) * M_PL_EV**4 / math.exp(log_s)
    rho_reduced = (
        OMEGA_LAMBDA
        * 24.0
        * math.pi**2
        * reduced_planck_ev**4
        / math.exp(log_s)
    )
    printed_leading_exponent = (math.pi**2 / 2.0) * derived["n_e"] / math.log(10.0)
    actual_hierarchy_magnitude = -math.log10(rho_nonreduced / M_PL_EV**4)
    return {
        "d_space": D_SPACE,
        "n_gauge": N_GAUGE,
        "n_e": derived["n_e"],
        "log_s": log_s,
        "h0_from_entropy_km_s_mpc": h_km_s_mpc,
        "rho_lambda_quarter_mev": rho_lambda_quarter_mev(log_s, OMEGA_LAMBDA),
        "nonreduced_reduced_density_relative_difference": rho_nonreduced / rho_reduced - 1.0,
        "printed_leading_hierarchy_exponent": printed_leading_exponent,
        "actual_hierarchy_magnitude_including_correction_and_prefactor": actual_hierarchy_magnitude,
        "hierarchy_exponent_difference": printed_leading_exponent - actual_hierarchy_magnitude,
    }


def primordial_checks() -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "d_eff": PRIMORDIAL_D_EFF,
        "n_e": N_E,
        "fixed_point": PRIMORDIAL_X,
        "observed_as_1e9": OBS_AS_1E9,
        "inferred_target_exponent": inferred_geometry_exponent(),
        "effective_exponent": PRIMORDIAL_D_EFF / (PRIMORDIAL_D_EFF + 1.0),
        "number_of_readout_candidates": float(len(readouts())),
    }
    for item in readouts():
        key = item.name.replace(" ", "_")
        result[f"{key}_as_1e9"] = item.as_1e9
        result[f"{key}_sigma"] = item.sigma_offset
        result[f"{key}_status"] = item.status
    return result


def covariance_checks() -> dict[str, float | bool | str]:
    min_pivot = cholesky_min_pivot(DESI_DR2_ALL_COVARIANCE)
    indefinite = "1,2;2,1"
    parsed = parse_covariance_matrix(indefinite)
    determinant = parsed[0][0] * parsed[1][1] - parsed[0][1] * parsed[1][0]
    try:
        cholesky_min_pivot(parsed)
        spd_result = "accepted"
    except ValueError as exc:
        spd_result = str(exc)
    # C^-1 for [[1,2],[2,1]] and r=(1,-1) gives r^T C^-1 r=-2.
    inverse_quadratic_form = -2.0
    return {
        "builtin_covariance_min_cholesky_pivot": min_pivot,
        "builtin_covariance_spd": min_pivot > 0.0,
        "parser_accepts_indefinite_positive_diagonal_matrix": parsed == ((1.0, 2.0), (2.0, 1.0)),
        "indefinite_determinant": determinant,
        "independent_spd_check": spd_result,
        "indefinite_example_quadratic_form_r_1_minus1": inverse_quadratic_form,
    }


def s8_normalization_check() -> dict[str, float]:
    params = CEForwardParams()
    raw = s8_today(params)
    normalized = params.sigma8_0 * math.sqrt(params.omega_m0_background / 0.3)
    return {
        "omega_m_raw": params.omega_m0,
        "omega_m_background_normalized": params.omega_m0_background,
        "s8_code_raw_omega_m": raw,
        "s8_background_consistent": normalized,
        "s8_difference": raw - normalized,
    }


def main() -> int:
    report = {
        "fixed_point": fixed_point_checks(),
        "density_mappings": density_mapping_checks(),
        "flrw_cpl": flrw_cpl_checks(),
        "nonuniform_simpson_counterexample": cumulative_s_nonuniform_grid_counterexample(),
        "hubble_tension_counterexamples": hubble_counterexamples(),
        "holographic_identity": holographic_checks(),
        "primordial_readouts": primordial_checks(),
        "covariance": covariance_checks(),
        "s8_normalization": s8_normalization_check(),
    }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
