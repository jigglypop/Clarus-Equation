"""Pre-registered discrimination gates for two open cosmology choices.

STATUS: both tests below are Open / pre-registered discrimination tests.
Nothing here is a closure claim; this file exists to freeze the decision
rules BEFORE the discriminating data arrive.

Test A -- alpha_s slope discrimination for delta_eps_0
  Two candidates for the eps-flow today-offset delta_eps_0 (hubble_tension.py):
    anchor   : delta_eps_0(alpha_s) = Omega_m^canon(alpha_s) - e^-1
    delta/pi : delta_eps_0(alpha_s) = -delta(alpha_s)/pi
  They coincide to 0.5% at alpha_s ~ 0.118 but have different alpha_s slopes
  (~ +1.6 vs ~ -0.45), so the agreement is an ACCIDENTAL CROSSING, not a
  derivation. This test outputs both curves, their slopes, the crossing
  point, and the sigma-separation achievable for a given alpha_s = X +/- s.

Test B -- DM/DE ratio discrimination for the R selection
  Two candidates for R = Omega_DM / Omega_Lambda (12_전이구간.md sec 7.1):
    3-layer : R = 3-layer baryon-inertia sum (경로적분.md 3.3.1) = 0.38063
              (closed form alpha_s D_eff (1+eps^2 delta) = 0.37789 is only
               a 0.72% approximation and is NOT used here)
    NLO     : R = alpha_s D_eff + (alpha_s D_eff)^2 / (4 pi) = 0.38579
  The repo cannot decide between them internally (both Open). Pre-registered
  kill rule: an observation R_obs +/- sigma kills a candidate at >= 5 sigma.
    if R_obs = 0.380 +/- 0.001 -> NLO dead (+5.2 sigma), 3-layer alive
    if R_obs = 0.386 +/- 0.001 -> 3-layer dead (5.4 sigma), NLO alive

ASCII-only output.
"""

import argparse
import math


EPS_FIX = math.exp(-1.0)
ALPHA_S_CE = 0.11789          # CE internal value (alpha_s_closure_gate)
ALPHA_S_PDG = 0.1180          # PDG world average alpha_s(M_Z)
ALPHA_S_PDG_SIG = 0.0009      # PDG uncertainty
KILL_SIGMA = 5.0              # pre-registered kill threshold


# ---------------------------------------------------------------------------
# Canonical chain: alpha_s -> delta -> D_eff -> eps^2 -> R -> Omega_m
# ---------------------------------------------------------------------------

def chain(alpha_s: float) -> dict:
    s2tw = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = s2tw * (1.0 - s2tw)
    d_eff = 3.0 + delta
    x = 0.05
    for _ in range(400):
        x = math.exp(-(1.0 - x) * d_eff)
    eps2 = x
    r_lo = alpha_s * d_eff
    # 3-layer baryon-inertia sum (경로적분.md 3.3.1):
    #   R = alpha_s * sum_k N_k (1 + eps^2 alpha_k/alpha_tot)
    #       + alpha_s * delta * (1 + eps^2 delta)
    # The documented coupling-ratio sum is 1.015 (doc table); the closed
    # form alpha_s D_eff (1+eps^2 delta) = 0.37789 is only a 0.72% approx.
    ratio_sum = 1.015
    r_3layer = alpha_s * (3.0 + eps2 * ratio_sum) \
        + alpha_s * delta * (1.0 + eps2 * delta)
    r_nlo = r_lo + r_lo * r_lo / (4.0 * math.pi)
    om_l = (1.0 - eps2) / (1.0 + r_3layer)
    om_dm = (1.0 - eps2) * r_3layer / (1.0 + r_3layer)
    om_m = eps2 + om_dm
    return {
        "delta": delta, "d_eff": d_eff, "eps2": eps2,
        "r_3layer": r_3layer, "r_nlo": r_nlo,
        "om_m": om_m, "om_l": om_l,
    }


def de0_anchor(alpha_s: float) -> float:
    """delta_eps_0 = Omega_m^canon(alpha_s) - e^-1 (structural anchor)."""
    return chain(alpha_s)["om_m"] - EPS_FIX


def de0_delta_pi(alpha_s: float) -> float:
    """delta_eps_0 = -delta(alpha_s)/pi (numerological family, no derivation)."""
    return -chain(alpha_s)["delta"] / math.pi


def slope(f, x: float, h: float = 1.0e-5) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def crossing_alpha_s(lo: float = 0.10, hi: float = 0.14,
                     tol: float = 1.0e-12) -> float:
    """Bisect g(a) = anchor(a) - delta_pi(a) = 0."""
    def g(a):
        return de0_anchor(a) - de0_delta_pi(a)
    g_lo, g_hi = g(lo), g(hi)
    if g_lo * g_hi > 0.0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        if abs(g_mid) <= tol:
            return mid
        if g_lo * g_mid <= 0.0:
            hi, g_hi = mid, g_mid
        else:
            lo, g_lo = mid, g_mid
    return 0.5 * (lo + hi)


def test_a_separation(alpha_x: float, sigma: float) -> dict:
    """Given alpha_s = X +/- sigma, how many sigma apart are the two
    delta_eps_0 predictions?

    gap        = |anchor(X) - delta_pi(X)|
    sigma_gap  = |slope_anchor - slope_dpi| * sigma   (alpha_s-induced
                 uncertainty of the gap; predictions are fully correlated
                 through alpha_s, so the DIFFERENCE carries the slope diff)
    n_sigma    = gap / sigma_gap
    """
    a = de0_anchor(alpha_x)
    d = de0_delta_pi(alpha_x)
    sa = slope(de0_anchor, alpha_x)
    sd = slope(de0_delta_pi, alpha_x)
    gap = abs(a - d)
    sigma_gap = abs(sa - sd) * sigma
    n_sigma = gap / sigma_gap if sigma_gap > 0.0 else float("inf")
    # sigma_alpha needed for the intrinsic gap to be a KILL_SIGMA separation
    sigma_needed = gap / (abs(sa - sd) * KILL_SIGMA) if sa != sd else float("nan")
    return {
        "anchor": a, "delta_pi": d, "slope_anchor": sa, "slope_dpi": sd,
        "gap": gap, "sigma_gap": sigma_gap, "n_sigma": n_sigma,
        "sigma_needed_5s": sigma_needed,
    }


def run_test_a(verbose_curve: bool = True) -> None:
    print("=" * 76)
    print("TEST A (pre-registered, Open): alpha_s slope discrimination")
    print("  anchor   de0 = Omega_m^canon(alpha_s) - e^-1")
    print("  delta/pi de0 = -delta(alpha_s)/pi")
    print("=" * 76)
    if verbose_curve:
        print(f"  {'alpha_s':>9} {'de0_anchor':>12} {'de0_-d/pi':>12} {'diff':>12}")
        for i in range(13):
            a = 0.112 + 0.001 * i
            x, y = de0_anchor(a), de0_delta_pi(a)
            print(f"  {a:>9.4f} {x:>+12.6f} {y:>+12.6f} {x - y:>+12.6f}")
        print()
    a_cross = crossing_alpha_s()
    print(f"  crossing point       alpha_s* = {a_cross:.6f}")
    print(f"  slope d(de0)/d(alpha_s) at CE alpha_s={ALPHA_S_CE}:")
    print(f"    anchor   : {slope(de0_anchor, ALPHA_S_CE):+.4f}")
    print(f"    -delta/pi: {slope(de0_delta_pi, ALPHA_S_CE):+.4f}")
    print()
    r = test_a_separation(ALPHA_S_PDG, ALPHA_S_PDG_SIG)
    print(f"  PDG alpha_s = {ALPHA_S_PDG} +/- {ALPHA_S_PDG_SIG}:")
    print(f"    de0_anchor           = {r['anchor']:+.6f}")
    print(f"    de0_-delta/pi        = {r['delta_pi']:+.6f}")
    print(f"    gap                  = {r['gap']:.6f}")
    print(f"    slope-induced sigma  = {r['sigma_gap']:.6f}")
    print(f"    separation           = {r['n_sigma']:.2f} sigma")
    verdict = ("DISCRIMINABLE" if r["n_sigma"] >= KILL_SIGMA
               else "NOT discriminable at current PDG precision")
    print(f"    verdict              = {verdict} (threshold {KILL_SIGMA} sigma)")
    print(f"    sigma(alpha_s) needed for {KILL_SIGMA:.0f}-sigma gap: "
          f"{r['sigma_needed_5s']:.2e}")
    print("  NOTE: alpha_s alone cannot separate them near the crossing;")
    print("        an INDEPENDENT measurement of delta_eps_0 (or alpha_s far")
    print("        from alpha_s*) is required. This is why the 0.5% agreement")
    print("        at alpha_s=0.118 is an accidental crossing, not support.")
    print()


def test_b_verdict(r_obs: float, sigma: float) -> dict:
    c = chain(ALPHA_S_CE)
    r3, rn = c["r_3layer"], c["r_nlo"]
    n3 = abs(r_obs - r3) / sigma
    nn = abs(r_obs - rn) / sigma
    def status(n):
        return "DEAD" if n >= KILL_SIGMA else "alive"
    return {"r3": r3, "rn": rn, "n3": n3, "nn": nn,
            "s3": status(n3), "sn": status(nn)}


def run_test_b() -> None:
    c = chain(ALPHA_S_CE)
    print("=" * 76)
    print("TEST B (pre-registered, Open): DM/DE ratio R selection")
    print(f"  R_3-layer = 3-layer baryon-inertia sum (3.3.1)    = {c['r_3layer']:.5f}")
    print(f"  R_NLO     = alpha_s D_eff + (alpha_s D_eff)^2/4pi = {c['r_nlo']:.5f}")
    print(f"  kill rule: candidate dead if |R_obs - R_pred| >= {KILL_SIGMA:.0f} sigma")
    print("=" * 76)
    scenarios = [
        ("DESI-like low ", 0.380, 0.001),
        ("DESI-like high", 0.386, 0.001),
        ("Planck 2018   ", 0.3854, 0.004),
        # Current-status scenario (not a pre-registered kill rule): DESI DR2
        # + CMB Omega_DM/Omega_Lambda = 0.373 +/- 0.010. Both candidates
        # survive at this precision; recorded here only to track the
        # downward pressure direction as new data arrive.
        ("DESI DR2+CMB  ", 0.373, 0.010),
    ]
    print(f"  {'scenario':<16} {'R_obs':>8} {'sigma':>7} "
          f"{'3-layer':>14} {'NLO':>14}")
    for name, r_obs, sig in scenarios:
        v = test_b_verdict(r_obs, sig)
        print(f"  {name:<16} {r_obs:>8.4f} {sig:>7.4f} "
              f"{v['n3']:>6.2f}s {v['s3']:>6} "
              f"{v['nn']:>6.2f}s {v['sn']:>6}")
    print()
    print("  pre-registered verdicts:")
    v_lo = test_b_verdict(0.380, 0.001)
    v_hi = test_b_verdict(0.386, 0.001)
    print(f"    R_obs = 0.380 +/- 0.001 -> NLO at {v_lo['nn']:.1f} sigma "
          f"({v_lo['sn']}), 3-layer at {v_lo['n3']:.1f} sigma ({v_lo['s3']})")
    print(f"    R_obs = 0.386 +/- 0.001 -> 3-layer at {v_hi['n3']:.1f} sigma "
          f"({v_hi['s3']}), NLO at {v_hi['nn']:.1f} sigma ({v_hi['sn']})")
    print("  NOTE: at current Planck precision (+/-0.004) NEITHER candidate is")
    print("        excluded; the R selection remains Open until sigma ~ 0.001.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(prog="cosmology_discrimination_gates")
    p.add_argument("--alpha-s", type=float, default=ALPHA_S_PDG,
                   help="hypothetical measured alpha_s for test A")
    p.add_argument("--alpha-s-sigma", type=float, default=ALPHA_S_PDG_SIG)
    p.add_argument("--r-obs", type=float, default=None,
                   help="hypothetical observed DM/DE ratio for test B")
    p.add_argument("--r-sigma", type=float, default=0.001)
    args = p.parse_args()

    print("STATUS: both tests are Open / pre-registered. No closure is")
    print("        claimed here; this file freezes decision rules only.")
    print()
    run_test_a()
    if (args.alpha_s, args.alpha_s_sigma) != (ALPHA_S_PDG, ALPHA_S_PDG_SIG):
        r = test_a_separation(args.alpha_s, args.alpha_s_sigma)
        print(f"  custom alpha_s = {args.alpha_s} +/- {args.alpha_s_sigma}: "
              f"gap={r['gap']:.6f}, separation={r['n_sigma']:.2f} sigma")
        print()
    run_test_b()
    if args.r_obs is not None:
        v = test_b_verdict(args.r_obs, args.r_sigma)
        print(f"  custom R_obs = {args.r_obs} +/- {args.r_sigma}: "
              f"3-layer {v['n3']:.2f} sigma ({v['s3']}), "
              f"NLO {v['nn']:.2f} sigma ({v['sn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
