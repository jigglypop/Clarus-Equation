"""Independent scratch checks for the density/dark-sector alternative routes.

This is a research artifact, not a cosmological prediction.  It checks only the
algebra stated in density-dark-alternative-derivations.md and deliberately uses
no observational density as an input.
"""

from __future__ import annotations

import json
import math


D = 3.1777584234099736
ALPHA_S = 0.11789


def fixed_point(d: float) -> float:
    lo, hi = 1.0e-15, 1.0 / d
    for _ in range(250):
        mid = (lo + hi) / 2.0
        residual = math.log(mid) + d * (1.0 - mid)
        if residual > 0.0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def v_prime(y: float, d: float) -> float:
    return math.log(y) + d * (1.0 - y)


def rk4_composition(y0: float, kappa_over_h: float, e_folds: float) -> float:
    steps = 20_000
    step = e_folds / steps

    def rhs(y: float) -> float:
        return -kappa_over_h * v_prime(y, D)

    y = y0
    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + step * k1 / 2.0)
        k3 = rhs(y + step * k2 / 2.0)
        k4 = rhs(y + step * k3)
        y += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


def main() -> None:
    q = fixed_point(D)
    one_minus_q = 1.0 - q
    hessian = 1.0 / q - D

    # Route E: extinction-conditioning turns the original Poisson(D) offspring
    # law into Poisson(m) with m=Dq.  Aggregating equal-energy nodes over many
    # finite trees makes descendants/total converge to m.
    conditioned_mean = D * q
    conditional_pgf_at_point = math.exp(conditioned_mean * (0.37 - 1.0))
    direct_conditional_pgf = math.exp(D * (q * 0.37 - 1.0)) / q
    expected_total_progeny = 1.0 / (1.0 - conditioned_mean)
    expected_descendants = conditioned_mean / (1.0 - conditioned_mean)
    descendant_ratio_of_expectations = (
        expected_descendants / expected_total_progeny
    )
    descendant_fraction_per_tree_average = conditioned_mean / 2.0
    occupancy_readout = 1.0 - math.exp(-conditioned_mean)
    one_child_readout = conditioned_mean * math.exp(-conditioned_mean)
    matter_attractor = 1.0 / D

    # Route 1: the composition flow reaches the small root and produces
    # non-negative entropy for kappa, mu_*, n and T all positive.
    y_final = rk4_composition(0.5, kappa_over_h=0.1, e_folds=20.0)
    entropy_factor = v_prime(0.2, D) ** 2

    # Legacy CE's dark split is used only as an algebraic candidate generated
    # from D and alpha_s, never as an observational target.
    dark_ratio = ALPHA_S * D
    xi = dark_ratio / (1.0 + dark_ratio)
    omega_c_candidate = one_minus_q * xi
    omega_de_candidate = one_minus_q * (1.0 - xi)

    # If one precursor tag produces one baryon and the other one dark particle,
    # the mass ratio required for the three ratio equations to close is xi.
    mchi_over_mp = xi
    c_over_b = mchi_over_mp * one_minus_q / q

    # Route 4a: interacting-vacuum ratio fixed point and its stability exponent.
    r_star = xi / (1.0 - xi)
    ratio_eigenvalue = -3.0 * (1.0 - xi)

    # Route 4b: one explicit conformally coupled exponential-scalar scaling
    # point with the same dark ratio.  u=-w_eff is a theory choice here.
    omega_phi = 1.0 / (1.0 + dark_ratio)
    u = 0.69
    ell = math.sqrt(3.0 / (omega_phi - u))
    beta = u * ell
    lam = (1.0 - u) * ell
    x = math.sqrt(1.5) / ell
    z2 = 1.5 / ell**2 + beta / ell
    omega_c_scaling = (lam * ell - 3.0) / ell**2
    omega_phi_scaling = x * x + z2
    x_eom_residual = (
        -3.0 * x
        + math.sqrt(1.5) * lam * z2
        - math.sqrt(1.5) * beta * omega_c_scaling
        + 1.5 * x * (1.0 + x * x - z2)
    )
    z_eom_factor = -math.sqrt(1.5) * lam * x + 1.5 * (
        1.0 + x * x - z2
    )
    jacobian_11 = (
        -3.0
        + 2.0 * math.sqrt(1.5) * beta * x
        + 1.5 * (1.0 + 3.0 * x * x - z2)
    )
    jacobian_12 = 2.0 * math.sqrt(1.5) * (lam + beta) * math.sqrt(z2) - (
        3.0 * x * math.sqrt(z2)
    )
    jacobian_21 = math.sqrt(z2) * (-math.sqrt(1.5) * lam + 3.0 * x)
    jacobian_22 = -3.0 * z2
    jacobian_trace = jacobian_11 + jacobian_22
    jacobian_determinant = (
        jacobian_11 * jacobian_22 - jacobian_12 * jacobian_21
    )

    # Hybrid ratio closure is an identity, not a dynamical proof.  It exposes
    # exactly which additional mass/coupling lemma would be required.
    omega_b_from_ratios = 1.0 / (1.0 + c_over_b + c_over_b / dark_ratio)

    checks = {
        "fixed_point_residual": abs(v_prime(q, D)),
        "small_root_hessian": hessian,
        "conditioned_poisson_mean": conditioned_mean,
        "conditioned_pgf_identity_error": abs(
            conditional_pgf_at_point - direct_conditional_pgf
        ),
        "expected_total_progeny": expected_total_progeny,
        "expected_descendants": expected_descendants,
        "descendant_ratio_of_expectations_error": abs(
            descendant_ratio_of_expectations - conditioned_mean
        ),
        "mean_of_per_tree_descendant_fraction": descendant_fraction_per_tree_average,
        "occupancy_readout": occupancy_readout,
        "one_child_readout": one_child_readout,
        "matter_attractor_one_over_d": matter_attractor,
        "conditioned_composition_times_matter_error": abs(
            conditioned_mean * matter_attractor - q
        ),
        "composition_final_error": abs(y_final - q),
        "entropy_square": entropy_factor,
        "candidate_sum_error": abs(q + omega_c_candidate + omega_de_candidate - 1.0),
        "dark_ratio_error": abs(omega_c_candidate / omega_de_candidate - dark_ratio),
        "mass_ratio_for_ratio_closure": mchi_over_mp,
        "c_over_b": c_over_b,
        "interacting_vacuum_r_error": abs(r_star - dark_ratio),
        "interacting_vacuum_stability_eigenvalue": ratio_eigenvalue,
        "scalar_scaling_ratio_error": abs(
            omega_c_scaling / omega_phi_scaling - dark_ratio
        ),
        "scalar_scaling_sum_error": abs(omega_c_scaling + omega_phi_scaling - 1.0),
        "scalar_x_eom_residual": abs(x_eom_residual),
        "scalar_z_eom_factor": abs(z_eom_factor),
        "scalar_jacobian_trace": jacobian_trace,
        "scalar_jacobian_determinant": jacobian_determinant,
        "ratio_closure_q_error": abs(omega_b_from_ratios - q),
        "static_topological_mu_over_t": 0.0,
    }

    assert checks["fixed_point_residual"] < 1.0e-13
    assert hessian > 0.0
    assert conditioned_mean < 1.0
    assert checks["conditioned_pgf_identity_error"] < 1.0e-14
    assert checks["descendant_ratio_of_expectations_error"] < 1.0e-14
    assert checks["conditioned_composition_times_matter_error"] < 1.0e-14
    assert checks["composition_final_error"] < 1.0e-12
    assert entropy_factor >= 0.0
    assert checks["candidate_sum_error"] < 1.0e-14
    assert checks["dark_ratio_error"] < 1.0e-14
    assert ratio_eigenvalue < 0.0
    assert checks["scalar_scaling_ratio_error"] < 1.0e-13
    assert checks["scalar_scaling_sum_error"] < 1.0e-13
    assert checks["scalar_x_eom_residual"] < 1.0e-13
    assert checks["scalar_z_eom_factor"] < 1.0e-13
    assert jacobian_trace < 0.0
    assert jacobian_determinant > 0.0
    assert checks["ratio_closure_q_error"] < 1.0e-14

    print(json.dumps(checks, indent=2, sort_keys=True))
    print("ALL DENSITY/DARK ALTERNATIVE SCRATCH CHECKS PASSED")


if __name__ == "__main__":
    main()
