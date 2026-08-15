#!/usr/bin/env python3
"""Independent arithmetic checks for the CE density-bridge math lane.

This script uses only the Python standard library and does not import the CE
implementation.  It checks the candidate potential, its two stationary
branches, the local de-Sitter attraction exponents, and the conserved-dust
counterexample to a time-independent baryon fraction.
"""

from __future__ import annotations

import math


TOL = 1.0e-12


def stationary_function(x: float, d: float) -> float:
    """v_D'(x) = 0 is the requested fixed-point equation."""

    return math.log(x) + d * (1.0 - x)


def stationary_root(d: float) -> float:
    """Bisect only the nontrivial interval (0, 1/D)."""

    lower = math.nextafter(0.0, 1.0)
    upper = 1.0 / d
    assert stationary_function(lower, d) < 0.0
    assert stationary_function(upper, d) > 0.0
    for _ in range(300):
        midpoint = 0.5 * (lower + upper)
        if stationary_function(midpoint, d) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def potential(x: float, d: float, c: float) -> float:
    return x * math.log(x) - x + d * (x - 0.5 * x * x) + c


def hessian(x: float, d: float) -> float:
    return 1.0 / x - d


def local_de_sitter_modes(mass_over_hubble: float) -> tuple[complex, complex]:
    """Return lambda/H for delta-x proportional to exp(lambda t)."""

    discriminant = 9.0 - 4.0 * mass_over_hubble**2
    root = complex(discriminant, 0.0) ** 0.5
    return ((-3.0 + root) / 2.0, (-3.0 - root) / 2.0)


def lcdm_fractions(
    a: float, omega_r0: float, omega_m0: float, omega_l0: float
) -> tuple[float, float, float, float]:
    e2 = omega_r0 * a**-4 + omega_m0 * a**-3 + omega_l0
    return (
        e2,
        omega_r0 * a**-4 / e2,
        omega_m0 * a**-3 / e2,
        omega_l0 / e2,
    )


def main() -> None:
    d_values = (3.1777584234099736, 3.17776)
    print("stationary branches and local stability")
    for d in d_values:
        q = stationary_root(d)
        fixed_point_residual = q - math.exp(-d * (1.0 - q))
        q_hessian = hessian(q, d)
        one_hessian = hessian(1.0, d)
        c_zero = q - 0.5 * d * q * q

        assert abs(stationary_function(q, d)) < TOL
        assert abs(fixed_point_residual) < TOL
        assert 0.0 < q < 1.0 / d < 1.0
        assert q_hessian > 0.0
        assert one_hessian < 0.0
        assert abs(potential(q, d, c_zero)) < TOL
        assert potential(math.nextafter(0.0, 1.0), d, c_zero) > 0.0
        assert potential(1.0, d, c_zero) > 0.0

        print(
            f"D={d:.16g} q={q:.16g} residual={fixed_point_residual:.3e} "
            f"v''(q)={q_hessian:.12g} v''(1)={one_hessian:.12g} "
            f"C_zero={c_zero:.12g}"
        )

    # Same D and same stationary q, but arbitrary vacuum stress through C.
    d = d_values[-1]
    q = stationary_root(d)
    c_zero = q - 0.5 * d * q * q
    # Use M^4=1 and the same rho_other=1 in both self-consistent Friedmann
    # backgrounds.  A C shift of 1/4 changes only the vacuum stress, giving
    # Omega_x=(1/4)/(1+1/4)=0.2 without moving q.
    rho_other = 1.0
    rho_x_zero = potential(q, d, c_zero)
    rho_x_shifted = potential(q, d, c_zero + 0.25)
    omega_x_zero = rho_x_zero / (rho_other + rho_x_zero)
    omega_x_shifted = rho_x_shifted / (rho_other + rho_x_shifted)
    assert abs(omega_x_zero) < TOL
    assert abs(omega_x_shifted - 0.2) < TOL
    print("\nsame-q stress counterexample")
    print(
        f"q={q:.12g}: Omega_x(C_zero)={omega_x_zero:.12g}, "
        f"Omega_x(C_zero+0.25)={omega_x_shifted:.12g}, "
        "with p_x/rho_x=-1 for the nonzero constant solution"
    )

    # Energy-weighted event fraction.  Let E have probability q and let the
    # conditional mean weight be 2 on E and 1 on its complement.
    mean_weight_event = 2.0
    mean_weight_complement = 1.0
    mean_weight = (
        q * mean_weight_event + (1.0 - q) * mean_weight_complement
    )
    omega_event = q * mean_weight_event / mean_weight
    covariance = q * (1.0 - q) * (
        mean_weight_event - mean_weight_complement
    )
    assert abs((omega_event - q) - covariance / mean_weight) < TOL
    assert abs(omega_event - 0.09277983982535848) < TOL
    equal_mean_omega = q * 3.0 / (q * 3.0 + (1.0 - q) * 3.0)
    assert abs(equal_mean_omega - q) < TOL
    print("\nenergy-weighted event theorem")
    print(
        f"q={q:.12g}, E[W|E]=2, E[W|not E]=1: "
        f"Omega_E={omega_event:.12g}, Omega_E-q={omega_event-q:.12g}"
    )
    print("equal conditional means give Omega_E=q to machine precision")

    # Local attraction in an exactly constant-H background.  The mass ratio is
    # free because m_*^2=(M^4/F^2) v''(q).
    print("\nconstant-H linear attraction")
    for ratio in (0.1, 1.0, 1.5, 2.0, 10.0):
        plus, minus = local_de_sitter_modes(ratio)
        assert plus.real <= 0.0 and minus.real <= 0.0
        if ratio > 1.5:
            envelope_hubble_times = 2.0 / 3.0
        else:
            envelope_hubble_times = math.inf if plus.real == 0.0 else -1.0 / plus.real
        print(
            f"m/H={ratio:4.1f} lambda+/H={plus.real:+.9f}{plus.imag:+.9f}i "
            f"lambda-/H={minus.real:+.9f}{minus.imag:+.9f}i "
            f"tau_amp*H={envelope_hubble_times:.9g}"
        )

    # Conserved pressureless baryons normalized to Omega_b(a=1)=q.  If x=q is
    # held at its attractor, the equality Omega_b=x immediately fails away from
    # a=1 in a radiation+matter+Lambda background.
    omega_r0 = 0.000092
    omega_m0 = 0.310968903
    omega_l0 = 1.0 - omega_r0 - omega_m0
    assert abs(omega_r0 + omega_m0 + omega_l0 - 1.0) < TOL

    print("\nconserved-dust constant-fraction no-go")
    print("a Omega_b(a) w_eff dlnOmega_b/dlna required_Q/(H rho_b)")
    for a in (1.0e-6, 1.0e-3, 0.1, 0.5, 1.0, 2.0):
        e2, frac_r, _frac_m, frac_l = lcdm_fractions(
            a, omega_r0, omega_m0, omega_l0
        )
        omega_b = q * a**-3 / e2
        w_eff = frac_r / 3.0 - frac_l
        conserved_slope = frac_r - 3.0 * frac_l
        required_source = -3.0 * w_eff
        assert abs(conserved_slope + required_source) < 5.0e-15
        print(
            f"{a:.1e} {omega_b:.12g} {w_eff:+.12g} "
            f"{conserved_slope:+.12g} {required_source:+.12g}"
        )

    assert abs(q - q * 1.0**-3) < TOL
    assert abs(q * 0.5**-3 / lcdm_fractions(0.5, omega_r0, omega_m0, omega_l0)[0] - q) > 1.0e-2

    # Dimension ledger in natural units.  Each action-density term must be 4.
    dimensions = {
        "F^2 (partial x)^2": 2 + 2,
        "M^4 v_D": 4 + 0,
        "F^2 Box x": 2 + 2,
        "M^4 v_D'": 4 + 0,
        "m_*^2=(M^4/F^2)v_D''": 4 - 2 + 0,
        "rho_b=m_b n_b": 1 + 3,
        "rho_crit=3 M_Pl^2 H^2": 2 + 2,
    }
    expected = {
        "F^2 (partial x)^2": 4,
        "M^4 v_D": 4,
        "F^2 Box x": 4,
        "M^4 v_D'": 4,
        "m_*^2=(M^4/F^2)v_D''": 2,
        "rho_b=m_b n_b": 4,
        "rho_crit=3 M_Pl^2 H^2": 4,
    }
    assert dimensions == expected
    print("\ndimension ledger")
    for name, dimension in dimensions.items():
        print(f"[{name}]={dimension}")
    print("log(x), D(1-x), m_*/H, and every Omega are dimensionless")

    print("\nALL DENSITY-BRIDGE MATH CHECKS PASSED")


if __name__ == "__main__":
    main()
