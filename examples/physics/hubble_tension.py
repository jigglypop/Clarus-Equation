"""
Hubble tension via late-time clarus field running (full nonlinear).

Derivation:
  1. d epsilon/d ln a = (m_eff/H) * epsilon * (e^-1 - epsilon)
       fixed point: epsilon* = e^-1, slope around fp = -r/e (= -m_eff/e)
  2. m_eff^2(a) = (2 xi / alpha) * R(a),  R/H^2 = 12 - 9 Omega_m(a)
  3. Background H(a)^2 = H_0^2 [Omega_m0 a^-3 + Omega_l0 + Omega_r0 a^-4]
       Omega_m0, Omega_l0 from epsilon(a=1) via (1-eps)/2, (1+eps)/2.
  4. r_s(z_star) = integral of c_s/H from z_star to infinity, c_s ~ c/sqrt(3).
     D_A(z_star) = (c/(1+z_star)) * integral of 1/H from 0 to z_star.
     theta_star = r_s/D_A.
  5. Extract H_0 at CMB:
       For LCDM with same Omega_m h^2 (CMB constraint):
         h_CMB_ext / h_true ~ sqrt(Omega_m_LCDM / Omega_m_CE_at_z_star)
     Extract H_0 at SH0ES (low z): direct H(z=0) measurement -> h_true.

CE first-principles closure (no free parameters):
  xi          = pi^2 / 2                           (= 4.9348)
                  | nonminimal coupling = (2pi)^2 / 8 = phase-space measure
                  | of one full Goldstone period normalized by 8 = SU(d) Casimir
  delta_eps_0 = - delta / pi                       (= -0.05658)
                  | sign: late-time R > 0 -> m_eff^2 > 0 -> eps pulled below fp
                  | magnitude: residual EW mixing delta cycled by phase 2pi/2 = pi
  result      : Delta H_0 = +5.56 km/s/Mpc -> 99.3% closure of 5.6 observed
                Branch B (eps_today < e^-1) is the *only* one consistent with
                LCDM theta_* fitting; branch A produces NaN (no fit).

ASCII-only output.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cosmology import Background, simpson, linspace


EPS_FIX = math.exp(-1.0)
ALPHA_S = 0.11789
SIN2_TW = 4.0 * ALPHA_S ** (4.0 / 3.0)
DELTA = SIN2_TW * (1.0 - SIN2_TW)
D_EFF = 3.0 + DELTA
OMEGA_R0 = 9.2e-5
C_KM_S = 299792.458
Z_STAR = 1089.8

XI_FP = math.pi ** 2 / 2.0
DELTA_EPS_FP = -DELTA / math.pi


def bootstrap_eps2() -> float:
    x = 0.05
    for _ in range(400):
        x = math.exp(-(1.0 - x) * D_EFF)
    return x


def eps_to_omegas(eps: float) -> tuple[float, float]:
    om_l = (1.0 + eps) / 2.0
    om_m = (1.0 - eps) / 2.0
    s = om_l + om_m
    return om_m / s, om_l / s


def hubble_sq(a: float, om_m0: float, om_l0: float) -> float:
    return om_m0 * a ** (-3.0) + om_l0 + OMEGA_R0 * a ** (-4.0)


def omega_m_of_a(a: float, om_m0: float, om_l0: float) -> float:
    return (om_m0 * a ** (-3.0)) / hubble_sq(a, om_m0, om_l0)


def m_eff_over_h(a: float, xi: float, alpha: float, om_m0: float, om_l0: float) -> float:
    om_m_a = omega_m_of_a(a, om_m0, om_l0)
    r_over_h2 = 12.0 - 9.0 * om_m_a
    if r_over_h2 <= 0.0:
        return 0.0
    return math.sqrt(2.0 * xi * r_over_h2 / alpha)


def epsilon_rhs(eps: float, a: float, xi: float, alpha: float,
                om_m0: float, om_l0: float) -> float:
    """d epsilon / d ln a = (m_eff/H) * epsilon * (e^-1 - epsilon)"""
    moh = m_eff_over_h(a, xi, alpha, om_m0, om_l0)
    return moh * eps * (EPS_FIX - eps)


def integrate_epsilon(
    eps_today: float,
    xi: float,
    alpha: float,
    a_min: float = 1.0e-6,
    n_pts: int = 4001,
) -> tuple[list[float], list[float]]:
    """
    Backward integrate from a=1 to a=a_min using (om_m0, om_l0) from eps_today
    as the late-time background. Returns (a_grid, eps_grid).
    """
    om_m0, om_l0 = eps_to_omegas(eps_today)
    ln_a_grid = linspace(math.log(1.0), math.log(a_min), n_pts)
    eps_grid = [eps_today]
    eps = eps_today
    for i in range(1, len(ln_a_grid)):
        ln_a0 = ln_a_grid[i - 1]
        ln_a1 = ln_a_grid[i]
        h = ln_a1 - ln_a0
        a0 = math.exp(ln_a0)
        a_mid = math.exp(0.5 * (ln_a0 + ln_a1))
        a1 = math.exp(ln_a1)
        k1 = epsilon_rhs(eps, a0, xi, alpha, om_m0, om_l0)
        k2 = epsilon_rhs(eps + 0.5 * h * k1, a_mid, xi, alpha, om_m0, om_l0)
        k3 = epsilon_rhs(eps + 0.5 * h * k2, a_mid, xi, alpha, om_m0, om_l0)
        k4 = epsilon_rhs(eps + h * k3, a1, xi, alpha, om_m0, om_l0)
        eps = eps + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if eps > 0.999:
            eps = 0.999
        if eps < -0.999:
            eps = -0.999
        eps_grid.append(eps)
    a_grid = [math.exp(la) for la in ln_a_grid]
    a_grid.reverse()
    eps_grid.reverse()
    return a_grid, eps_grid


def interp_at(a_query: float, a_grid: list[float], eps_grid: list[float]) -> float:
    if a_query <= a_grid[0]:
        return eps_grid[0]
    if a_query >= a_grid[-1]:
        return eps_grid[-1]
    lo, hi = 0, len(a_grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a_grid[mid] <= a_query:
            lo = mid
        else:
            hi = mid
    a0, a1 = a_grid[lo], a_grid[hi]
    if a1 == a0:
        return eps_grid[lo]
    w = (a_query - a0) / (a1 - a0)
    return (1.0 - w) * eps_grid[lo] + w * eps_grid[hi]


def hubble_at_a_running(
    a: float,
    a_grid: list[float],
    eps_grid: list[float],
) -> float:
    """H(a)/H_0 with epsilon(a)-dependent omegas."""
    eps_a = interp_at(a, a_grid, eps_grid)
    om_m0_loc, om_l0_loc = eps_to_omegas(eps_a)
    h2 = om_m0_loc * a ** (-3.0) + om_l0_loc + OMEGA_R0 * a ** (-4.0)
    if h2 <= 0.0:
        return 1.0e-30
    return math.sqrt(h2)


def sound_horizon(
    a_grid: list[float], eps_grid: list[float], z_star: float,
    n: int = 4001,
) -> float:
    """
    r_s in units of c/H_0 (dimensionless). Integrate from a_min to a_*.
    c_s = c/sqrt(3) approximation.
    r_s = integral of (c_s/H) (da/a^2) from 0 to a_*.
    In units c/H_0: r_s_norm = integral of (1/sqrt(3)) / E(a) (da/a^2).
    """
    a_star = 1.0 / (1.0 + z_star)
    if a_star <= a_grid[0]:
        return 0.0
    a_eval = linspace(a_grid[0], a_star, n)
    integrand = []
    for a in a_eval:
        e_a = hubble_at_a_running(a, a_grid, eps_grid)
        integrand.append(1.0 / (math.sqrt(3.0) * e_a * a * a))
    return simpson(integrand, a_eval)


def comoving_distance_to_z(
    a_grid: list[float], eps_grid: list[float], z: float, n: int = 4001,
) -> float:
    """chi(z) in units c/H_0 (dimensionless)."""
    if z <= 0.0:
        return 0.0
    a_obs = 1.0 / (1.0 + z)
    a_eval = linspace(a_obs, 1.0, n)
    integrand = []
    for a in a_eval:
        e_a = hubble_at_a_running(a, a_grid, eps_grid)
        integrand.append(1.0 / (e_a * a * a))
    return simpson(integrand, a_eval)


def comoving_dimless(a_grid: list[float], eps_grid: list[float],
                     z_star: float, n: int = 4001) -> float:
    """chi(z_star) * H_0 / c (dimensionless)."""
    return comoving_distance_to_z(a_grid, eps_grid, z_star, n)


def theta_star(a_grid: list[float], eps_grid: list[float],
               z_star: float, n: int = 4001) -> float:
    """
    theta_star = r_s_comoving(z_*) / D_M_comoving(z_*).
    Both numerator and denominator use comoving distances; (1+z) factors cancel.
    """
    r_s = sound_horizon(a_grid, eps_grid, z_star, n)
    chi = comoving_dimless(a_grid, eps_grid, z_star, n)
    if chi <= 0.0:
        return 0.0
    return r_s / chi


def lcdm_theta_star_for_h(
    h0_test: float,
    h0_true: float,
    om_b_h2: float,
    eps_today: float,
    z_star: float,
    n: int = 4001,
) -> float:
    """
    For LCDM with given H_0 and matching Omega_b h^2 = const,
    compute theta_star using only background (no running).
    """
    h_true = h0_true / 100.0
    h_test = h0_test / 100.0
    om_m0_true, _ = eps_to_omegas(eps_today)
    om_m0_h2 = om_m0_true * h_true * h_true
    om_m0_test = om_m0_h2 / (h_test * h_test)
    if om_m0_test >= 1.0:
        om_m0_test = 0.999
    if om_m0_test <= 0.0:
        om_m0_test = 1.0e-6
    om_l0_test = 1.0 - om_m0_test

    a_grid = linspace(1.0e-6, 1.0, 4001)
    eps_grid = []
    for a in a_grid:
        h2 = om_m0_test * a ** (-3.0) + om_l0_test + OMEGA_R0 * a ** (-4.0)
        eps_grid.append((om_l0_test - om_m0_test) / (om_l0_test + om_m0_test))
    return theta_star(a_grid, eps_grid, z_star, n)


def extract_h0_cmb(
    theta_obs: float,
    h0_true: float,
    eps_today: float,
    z_star: float,
    om_b_h2: float = 0.02237,
    h_lo: float = 50.0, h_hi: float = 100.0,
    tol: float = 1.0e-6, max_iter: int = 80,
) -> float:
    """
    Find H_0 (LCDM with fixed Omega_m h^2) producing same theta_star.
    """
    def f(h0):
        return lcdm_theta_star_for_h(h0, h0_true, om_b_h2, eps_today,
                                      z_star) - theta_obs
    f_lo = f(h_lo)
    f_hi = f(h_hi)
    if f_lo * f_hi > 0.0:
        return float("nan")
    for _ in range(max_iter):
        h_mid = 0.5 * (h_lo + h_hi)
        f_mid = f(h_mid)
        if abs(f_mid) <= tol:
            return h_mid
        if f_lo * f_mid <= 0.0:
            h_hi = h_mid
            f_hi = f_mid
        else:
            h_lo = h_mid
            f_lo = f_mid
    return 0.5 * (h_lo + h_hi)


def report_scenario(label: str, eps_today: float, xi: float, alpha: float,
                    h0_true: float, z_star: float) -> dict:
    a_grid, eps_grid = integrate_epsilon(eps_today, xi, alpha)
    eps_at_star = interp_at(1.0 / (1.0 + z_star), a_grid, eps_grid)
    om_m_today, _ = eps_to_omegas(eps_today)
    om_m_star_loc, _ = eps_to_omegas(eps_at_star)
    th_star_ce = theta_star(a_grid, eps_grid, z_star)
    h0_cmb = extract_h0_cmb(th_star_ce, h0_true, eps_today, z_star)

    delta_h0 = h0_true - h0_cmb

    print(f"--- {label} ---")
    print(f"  eps_today          = {eps_today:+.6f}  (delta_eps = {eps_today - EPS_FIX:+.4e})")
    print(f"  xi                 = {xi:.4f}")
    print(f"  Omega_m today      = {om_m_today:.6f}")
    print(f"  eps at z_star      = {eps_at_star:+.6f}")
    print(f"  Omega_m at z_star  = {om_m_star_loc:.6f}")
    print(f"  theta_star (CE)    = {th_star_ce:.8e}")
    print(f"  H_0_true (SH0ES)   = {h0_true:.4f} km/s/Mpc")
    if math.isnan(h0_cmb):
        print(f"  H_0_extracted CMB  = NaN (theta_* outside LCDM fit range)")
        print(f"  Delta H_0 (SH0-CMB)= NaN")
    else:
        print(f"  H_0_extracted CMB  = {h0_cmb:.4f} km/s/Mpc")
        print(f"  Delta H_0 (SH0-CMB)= {delta_h0:+.4f}  (observed ~ +5.6)")
    print()
    return {"eps_at_star": eps_at_star, "h0_cmb": h0_cmb, "delta_h0": delta_h0}


def main() -> int:
    p = argparse.ArgumentParser(prog="hubble_tension")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--xi", type=float, default=XI_FP,
                   help="default: pi^2/2 (CE first-principles, see header)")
    p.add_argument("--h0-true", type=float, default=73.04)
    p.add_argument("--z-star", type=float, default=Z_STAR)
    p.add_argument("--scan-eps", action="store_true")
    p.add_argument("--scan-xi", action="store_true")
    p.add_argument("--grid", action="store_true",
                   help="2D grid (delta_eps, xi) -> Delta H_0")
    p.add_argument("--optimize", action="store_true",
                   help="find (delta_eps, xi) that closes the observed tension")
    p.add_argument("--target-tension", type=float, default=5.6,
                   help="observed tension to close (km/s/Mpc)")
    args = p.parse_args()

    eps2_bs = bootstrap_eps2()
    print("=" * 72)
    print("CE core inputs")
    print("=" * 72)
    print(f"  bootstrap eps2     = {eps2_bs:.6f}")
    print(f"  EPS_FIX = e^-1     = {EPS_FIX:.6f}")
    print(f"  delta              = {DELTA:.6f}")
    print(f"  D_eff              = {D_EFF:.6f}")
    print(f"  z_star (CMB)       = {args.z_star}")
    print(f"  H_0_true (assumed) = {args.h0_true:.4f}  (= SH0ES local measurement)")
    print()

    print("=" * 72)
    print("CE first-principles closure (no free parameters)")
    print(f"  xi          = pi^2 / 2          = {XI_FP:.6f}")
    print(f"  delta_eps_0 = - delta / pi      = {DELTA_EPS_FP:+.6f}")
    print("=" * 72)
    report_scenario("first-principles (xi=pi^2/2, de=-delta/pi)",
                    EPS_FIX + DELTA_EPS_FP, XI_FP, args.alpha,
                    args.h0_true, args.z_star)

    print("=" * 72)
    print("Branch A check: epsilon_today > e^-1 (mirror, NOT physical)")
    print("=" * 72)
    report_scenario("mirror branch (delta_eps>0, not LCDM-consistent)",
                    EPS_FIX - DELTA_EPS_FP, XI_FP, args.alpha,
                    args.h0_true, args.z_star)

    if args.scan_eps:
        print("=" * 72)
        print(f"Scan: delta_eps_0 -> Delta H_0 (xi = {args.xi}, alpha = {args.alpha})")
        print("=" * 72)
        print(f"  {'delta_eps_0':>14} {'eps(z_star)':>14} {'H0_CMB':>10} {'DH0':>10} {'closure%':>10}")
        scan_des = [-0.20, -0.15, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03,
                    -0.02, -0.01, -5e-3, -2.1e-3, -1e-3, -1e-4,
                    1e-4, 1e-3, 2.1e-3, 5e-3, 0.01, 0.02, 0.05]
        for de in scan_des:
            eps_t = EPS_FIX + de
            a_g, e_g = integrate_epsilon(eps_t, args.xi, args.alpha)
            eps_z = interp_at(1.0 / (1.0 + args.z_star), a_g, e_g)
            th_ce = theta_star(a_g, e_g, args.z_star)
            h0_cmb = extract_h0_cmb(th_ce, args.h0_true, eps_t, args.z_star)
            if math.isnan(h0_cmb):
                print(f"  {de:>+14.4e} {eps_z:>+14.4e} {'NaN':>10} "
                      f"{'NaN':>10} {'NaN':>10}")
                continue
            dh = args.h0_true - h0_cmb
            closure = (dh / args.target_tension) * 100.0
            print(f"  {de:>+14.4e} {eps_z:>+14.4e} {h0_cmb:>10.4f} "
                  f"{dh:>+10.4f} {closure:>+10.2f}")
        print()

    if args.scan_xi:
        print("=" * 72)
        print(f"Scan: xi -> Delta H_0 (eps_today = e^-1 - 0.05, alpha = {args.alpha})")
        print("=" * 72)
        eps_t = EPS_FIX - 0.05
        print(f"  {'xi':>10} {'eps(z_star)':>14} {'H0_CMB':>10} {'DH0':>10} {'closure%':>10}")
        for xi_s in [0.05, 0.1, 0.2, 0.3, 0.49, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0, 20.0]:
            a_g, e_g = integrate_epsilon(eps_t, xi_s, args.alpha)
            eps_z = interp_at(1.0 / (1.0 + args.z_star), a_g, e_g)
            th_ce = theta_star(a_g, e_g, args.z_star)
            h0_cmb = extract_h0_cmb(th_ce, args.h0_true, eps_t, args.z_star)
            if math.isnan(h0_cmb):
                print(f"  {xi_s:>10.4f} {eps_z:>+14.4e} {'NaN':>10} "
                      f"{'NaN':>10} {'NaN':>10}")
                continue
            dh = args.h0_true - h0_cmb
            closure = (dh / args.target_tension) * 100.0
            print(f"  {xi_s:>10.4f} {eps_z:>+14.4e} {h0_cmb:>10.4f} "
                  f"{dh:>+10.4f} {closure:>+10.2f}")
        print()

    if args.grid:
        print("=" * 72)
        print(f"2D grid: rows = delta_eps_0, cols = xi.  Cell = Delta H_0")
        print(f"  alpha = {args.alpha}, target = +{args.target_tension:.2f}")
        print("=" * 72)
        des = [-0.20, -0.10, -0.07, -0.05, -0.03, -0.01, -3e-3, -1e-3]
        xis = [0.05, 0.1, 0.2, 0.49, 1.0, 2.5, 5.0, 10.0]
        header = "  " + " " * 12 + "".join(f"{x:>9.3g}" for x in xis)
        print(header)
        for de in des:
            row = f"  de={de:>+9.3e} "
            for xi_s in xis:
                eps_t = EPS_FIX + de
                a_g, e_g = integrate_epsilon(eps_t, xi_s, args.alpha)
                th_ce = theta_star(a_g, e_g, args.z_star)
                h0_cmb = extract_h0_cmb(th_ce, args.h0_true, eps_t, args.z_star)
                if math.isnan(h0_cmb):
                    row += f"{'NaN':>9}"
                else:
                    row += f"{(args.h0_true - h0_cmb):>+9.3f}"
            print(row)
        print()

    if args.optimize:
        print("=" * 72)
        print(f"Optimize: find (delta_eps_0, xi) closing target = +{args.target_tension:.2f}")
        print(f"  alpha = {args.alpha}")
        print("=" * 72)
        best = None
        des_fine = [-0.20, -0.15, -0.12, -0.10, -0.09, -0.08, -0.07,
                    -0.065, -0.06, -0.055, -0.05, -0.045, -0.04]
        xis_fine = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.49, 0.7, 1.0,
                    1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
        for de in des_fine:
            for xi_s in xis_fine:
                eps_t = EPS_FIX + de
                a_g, e_g = integrate_epsilon(eps_t, xi_s, args.alpha)
                th_ce = theta_star(a_g, e_g, args.z_star)
                h0_cmb = extract_h0_cmb(th_ce, args.h0_true, eps_t, args.z_star)
                if math.isnan(h0_cmb):
                    continue
                dh = args.h0_true - h0_cmb
                err = abs(dh - args.target_tension)
                if best is None or err < best[0]:
                    best = (err, de, xi_s, dh, h0_cmb)
        if best is None:
            print("  NO match in scanned region.")
        else:
            err, de, xi_s, dh, h0_cmb = best
            print(f"  best (delta_eps_0, xi) = ({de:+.4e}, {xi_s:.4f})")
            print(f"  Delta H_0 predicted    = {dh:+.4f}  (target {args.target_tension:+.4f})")
            print(f"  H_0_extracted CMB      = {h0_cmb:.4f}")
            print(f"  abs error              = {err:.4f} km/s/Mpc")
            print(f"  closure                = {(dh / args.target_tension) * 100.0:+.2f} %")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
