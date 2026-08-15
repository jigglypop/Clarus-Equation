"""Independent algebra checks for transient, action-defined cosmology routes.

Only dimensionless toy parameters are used.  No observational central value is
an input to this verifier.  The script checks identities and local stability;
it does not promote any of the actions in the companion note to a prediction.
"""

from __future__ import annotations

import math


RTOL = 2.0e-11
ATOL = 2.0e-12


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def bisect(function, left: float, right: float, iterations: int = 180) -> float:
    f_left = function(left)
    f_right = function(right)
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0.0:
        raise AssertionError("bisection interval does not bracket a root")
    for _ in range(iterations):
        middle = 0.5 * (left + right)
        f_middle = function(middle)
        if f_left * f_middle <= 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return 0.5 * (left + right)


def extinction_root(branching_mean: float) -> float:
    """Return the nontrivial root of log(q)+D(1-q)=0 for D>1."""

    if branching_mean <= 1.0:
        raise ValueError("the nontrivial supercritical branch requires D>1")
    equation = lambda q: math.log(q) + branching_mean * (1.0 - q)
    return bisect(equation, 1.0e-15, 1.0 / branching_mean)


def check_branching_composition() -> None:
    for branching_mean in (1.25, 2.0, 4.0, 7.5):
        q_ext = extinction_root(branching_mean)
        matter_baryon_share = branching_mean * q_ext
        omega_m = 1.0 / branching_mean
        omega_b = matter_baryon_share * omega_m
        omega_c = (1.0 - matter_baryon_share) * omega_m
        omega_other = 1.0 - omega_m

        if not 0.0 < matter_baryon_share < 1.0:
            raise AssertionError("conditioned offspring mean must be subcritical")
        assert_close(omega_b, q_ext, "(D q)(1/D)=q")
        assert_close(omega_b + omega_c + omega_other, 1.0, "flat sum")
        print(
            "branching",
            f"D={branching_mean:.6g}",
            f"q={q_ext:.12g}",
            f"m=Dq={matter_baryon_share:.12g}",
            "PASS",
        )


def v_prime(y: float, branching_mean: float) -> float:
    return math.log(y) + branching_mean * (1.0 - y)


def v_second(y: float, branching_mean: float) -> float:
    return 1.0 / y - branching_mean


def check_spinodal() -> None:
    for branching_mean in (1.25, 2.0, 4.0, 7.5):
        y_spinodal = 1.0 / branching_mean
        h_spinodal = branching_mean - 1.0 - math.log(branching_mean)

        assert_close(v_second(y_spinodal, branching_mean), 0.0, "v''(1/D)")
        assert_close(
            v_prime(y_spinodal, branching_mean) - h_spinodal,
            0.0,
            "tilted free-energy stationarity",
        )
        assert_close(-1.0 / y_spinodal**2, -branching_mean**2, "v'''(1/D)")

        # Immediately below the fold value of h there are a stable lower
        # stationary branch and an unstable upper stationary branch.
        epsilon = min(0.02, 0.2 * h_spinodal)
        h_control = h_spinodal - epsilon
        stationary = lambda y: v_prime(y, branching_mean) - h_control
        lower = bisect(stationary, 1.0e-14, y_spinodal)
        upper = bisect(stationary, y_spinodal, 1.0 - 1.0e-14)
        if not v_second(lower, branching_mean) > 0.0:
            raise AssertionError("lower composition branch should be locally stable")
        if not v_second(upper, branching_mean) < 0.0:
            raise AssertionError("upper composition branch should be locally unstable")
        print(
            "spinodal",
            f"D={branching_mean:.6g}",
            f"y*=1/D={y_spinodal:.12g}",
            f"h*={h_spinodal:.12g}",
            "PASS",
        )


def f_trigger(a: float, branching_mean: float, matter: float, radiation: float, vacuum: float) -> float:
    return (
        (branching_mean - 1.0) * matter * a**-3
        - radiation * a**-4
        - vacuum
    )


def scan_positive_roots(function, lower: float, upper: float, samples: int = 6000) -> list[float]:
    roots: list[float] = []
    log_lower = math.log(lower)
    log_upper = math.log(upper)
    previous_x = lower
    previous_y = function(previous_x)
    for index in range(1, samples + 1):
        x = math.exp(log_lower + (log_upper - log_lower) * index / samples)
        y = function(x)
        if previous_y * y < 0.0:
            roots.append(bisect(function, previous_x, x))
        previous_x = x
        previous_y = y
    return roots


def check_covariant_density_trigger() -> None:
    # Abstract positive integration constants, deliberately not cosmological
    # measurements.  This case demonstrates the generic two-root topology.
    branching_mean = 4.0
    matter = radiation = vacuum = 1.0
    function = lambda a: f_trigger(a, branching_mean, matter, radiation, vacuum)
    roots = scan_positive_roots(function, 1.0e-4, 1.0e4)
    if len(roots) != 2:
        raise AssertionError(f"radiation+vacuum example should have two roots, got {roots}")
    for root in roots:
        rho_m = matter * root**-3
        rho_r = radiation * root**-4
        rho_v = vacuum
        omega_m = rho_m / (rho_m + rho_r + rho_v)
        assert_close(omega_m, 1.0 / branching_mean, "F=0 implies Omega_m=1/D")

    # With radiation removed F is strictly decreasing and has one analytic root.
    unique_root = ((branching_mean - 1.0) * matter / vacuum) ** (1.0 / 3.0)
    no_radiation = lambda a: f_trigger(a, branching_mean, matter, 0.0, vacuum)
    assert_close(no_radiation(unique_root), 0.0, "radiation-free analytic root")
    derivative = -3.0 * (branching_mean - 1.0) * matter * unique_root**-4
    if not derivative < 0.0:
        raise AssertionError("radiation-free trigger must be strictly decreasing")
    print(
        "density trigger",
        f"two roots={roots[0]:.12g},{roots[1]:.12g}",
        f"unique no-radiation root={unique_root:.12g}",
        "PASS",
    )


def check_matter_vacuum_subsystem_trigger() -> None:
    for branching_mean in (1.25, 2.0, 4.0, 7.5):
        q_ext = extinction_root(branching_mean)
        inner_share = branching_mean * q_ext

        # Abstract dust and vacuum normalizations.  The local subsystem ratio is
        # y(a)=rho_m/(rho_m+rho_L), and its crossing is analytic and unique.
        matter = 2.3
        vacuum = 0.7
        crossing = ((branching_mean - 1.0) * matter / vacuum) ** (1.0 / 3.0)
        rho_m = matter * crossing**-3
        y_subsystem = rho_m / (rho_m + vacuum)
        assert_close(y_subsystem, 1.0 / branching_mean, "subsystem y*=1/D")
        derivative_log_a = -3.0 * y_subsystem * (1.0 - y_subsystem)
        if not derivative_log_a < 0.0:
            raise AssertionError("dust+vacuum subsystem ratio must be monotone")

        for omega_r in (0.0, 0.1, 0.35):
            omega_m = (1.0 - omega_r) / branching_mean
            omega_b = q_ext * (1.0 - omega_r)
            omega_c = (1.0 - inner_share) * (1.0 - omega_r) / branching_mean
            omega_v = (1.0 - 1.0 / branching_mean) * (1.0 - omega_r)
            assert_close(omega_b, inner_share * omega_m, "subsystem baryon share")
            assert_close(
                omega_b + omega_c + omega_v + omega_r,
                1.0,
                "radiation-corrected flat sum",
            )
            assert_close(omega_b, q_ext * (1.0 - omega_r), "radiation correction")

        # The spectator zero can be written either as D y-1 or as the scaled
        # Hessian -y v_D''(y); this is an algebraic identity, not a derivation.
        trigger = branching_mean * y_subsystem - 1.0
        scaled_hessian = -y_subsystem * v_second(y_subsystem, branching_mean)
        assert_close(trigger, scaled_hessian, "D y-1 = -y v_D''")
        print(
            "matter-vacuum subsystem",
            f"D={branching_mean:.6g}",
            f"a*={crossing:.12g}",
            "radiation correction PASS",
        )


def coupled_autonomous(x: float, y: float, slope: float, coupling: float) -> tuple[float, float]:
    coefficient = math.sqrt(1.5)
    omega_c = 1.0 - x * x - y * y
    dx = (
        -3.0 * x
        + coefficient * slope * y * y
        - coefficient * coupling * omega_c
        + 1.5 * x * (1.0 + x * x - y * y)
    )
    dy = -coefficient * slope * x * y + 1.5 * y * (1.0 + x * x - y * y)
    return dx, dy


def numerical_jacobian(function, x: float, y: float) -> tuple[float, float, float, float]:
    step = 1.0e-6
    xp = function(x + step, y)
    xm = function(x - step, y)
    yp = function(x, y + step)
    ym = function(x, y - step)
    return (
        (xp[0] - xm[0]) / (2.0 * step),
        (yp[0] - ym[0]) / (2.0 * step),
        (xp[1] - xm[1]) / (2.0 * step),
        (yp[1] - ym[1]) / (2.0 * step),
    )


def check_coupled_scalar_fixed_point() -> None:
    for branching_mean, coupling in ((2.0, 0.1), (4.0, 0.2), (8.0, 0.35)):
        # Solves D[lambda(lambda+beta)-3]=(lambda+beta)^2.
        total_slope = (
            branching_mean * coupling
            + math.sqrt(
                branching_mean**2 * coupling**2
                + 12.0 * branching_mean * (branching_mean - 1.0)
            )
        ) / (2.0 * (branching_mean - 1.0))
        slope = total_slope - coupling
        x = math.sqrt(1.5) / total_slope
        y_squared = (coupling * total_slope + 1.5) / total_slope**2
        y = math.sqrt(y_squared)
        omega_c = (slope * total_slope - 3.0) / total_slope**2
        residual = coupled_autonomous(x, y, slope, coupling)

        assert_close(omega_c, 1.0 / branching_mean, "coupled fixed-point fraction")
        assert_close(residual[0], 0.0, "coupled fixed-point x residual")
        assert_close(residual[1], 0.0, "coupled fixed-point y residual")

        jacobian = numerical_jacobian(
            lambda x_value, y_value: coupled_autonomous(
                x_value, y_value, slope, coupling
            ),
            x,
            y,
        )
        a, b, c, d = jacobian
        trace = a + d
        determinant = a * d - b * c
        if not (trace < 0.0 and determinant > 0.0):
            raise AssertionError("sampled reduced fixed point is not linearly stable")

        w_effective = -coupling / total_slope
        if not 3.0 * w_effective < 0.0:
            raise AssertionError("a separately conserved baryon fraction should decay here")
        print(
            "coupled scalar",
            f"D={branching_mean:.6g}",
            f"beta={coupling:.6g}",
            f"lambda={slope:.12g}",
            f"trJ={trace:.12g}",
            f"detJ={determinant:.12g}",
            "PASS",
        )


def check_dimension_ledger() -> None:
    dimensions = {
        "D,q,m,y,h,C,g,lambda,beta,phi/Mpl,T/M": 0,
        "chi,M,Mpl,phi,mu": 1,
        "number density n": 3,
        "rho,F": 4,
        "F/M^4": 0,
        "g F chi^2/M^2": 4 + 2 - 2,
        "m_eff^2=gF/M^2": 4 - 2,
        "n mu v_D": 3 + 1,
        "Mchi^2 (D y-1) chi^2": 2 + 0 + 2,
    }
    expected = {
        "D,q,m,y,h,C,g,lambda,beta,phi/Mpl,T/M": 0,
        "chi,M,Mpl,phi,mu": 1,
        "number density n": 3,
        "rho,F": 4,
        "F/M^4": 0,
        "g F chi^2/M^2": 4,
        "m_eff^2=gF/M^2": 2,
        "n mu v_D": 4,
        "Mchi^2 (D y-1) chi^2": 4,
    }
    if dimensions != expected:
        raise AssertionError(f"dimension ledger mismatch: {dimensions!r}")
    print("dimension ledger PASS")


def main() -> None:
    check_branching_composition()
    check_spinodal()
    check_covariant_density_trigger()
    check_matter_vacuum_subsystem_trigger()
    check_coupled_scalar_fixed_point()
    check_dimension_ledger()
    print("ALL TRANSIENT-TRANSITION ACTION CHECKS PASSED")


if __name__ == "__main__":
    main()
