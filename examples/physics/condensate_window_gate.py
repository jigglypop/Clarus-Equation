"""Scale-dependent condensate window gate.

We previously found that high-z halo enhancement needs a small-scale boost,
but a scale-independent boost would damage S8/CMB small-scale constraints.

This gate tests a minimal localized transfer-window ansatz:

    T_CE(k) = T_LCDM(k) [1 + delta W_cond(k)]
    W_cond(k) = exp[-(ln(k/k_c))^2 / (2 s^2)]

The question is not whether this is the final law.  The question is whether
there exists a simple dimensionless window that is negligible at the S8 scale
but can be O(1) at early halo-seed scales.
"""

from __future__ import annotations

import math


DELTA = 0.17775842340997383
OMEGA_M = 0.31092644

# S8 radius in h^-1 Mpc, so k8 is in h Mpc^-1.
R8_HINV_MPC = 8.0
K8 = 1.0 / R8_HINV_MPC

# rho_m = 2.775e11 Omega_m Msun h^2 / Mpc^3.
# With R in h^-1 Mpc and mass in Msun/h, M = 4pi/3 rho_m R^3.
RHO_CRIT_H_UNITS = 2.775e11


def radius_hinv_mpc(mass_msun_h: float, omega_m: float = OMEGA_M) -> float:
    rho_m = RHO_CRIT_H_UNITS * omega_m
    return (3.0 * mass_msun_h / (4.0 * math.pi * rho_m)) ** (1.0 / 3.0)


def k_from_mass(mass_msun_h: float) -> float:
    return 1.0 / radius_hinv_mpc(mass_msun_h)


def log_gaussian_window(k: float, k_c: float, width: float) -> float:
    return math.exp(-0.5 * (math.log(k / k_c) / width) ** 2)


def boost_from_window(w: float) -> float:
    return 1.0 + DELTA * w


def required_w(boost: float) -> float:
    return (boost - 1.0) / DELTA


def tail_ratio(total_amp_ratio: float, nu_lcdm: float) -> float:
    nu_ce = nu_lcdm / total_amp_ratio
    lcdm_tail = math.erfc(nu_lcdm / math.sqrt(2.0))
    ce_tail = math.erfc(nu_ce / math.sqrt(2.0))
    return ce_tail / lcdm_tail


def main() -> int:
    masses = [1.0e8, 1.0e9, 1.0e10, 1.0e11]
    widths = [0.35, 0.50, 0.75, 1.00]
    # Center the window on 1e9 Msun/h, a proxy early-galaxy seed mass.
    k_c = k_from_mass(1.0e9)

    print("# Condensate Transfer Window Gate")
    print()
    print("## Ansatz")
    print()
    print("T_CE(k) = T_LCDM(k) [1 + delta W_cond(k)]")
    print("W_cond(k) = exp[-(ln(k/k_c))^2/(2 s^2)]")
    print(f"delta = {DELTA:.8f}")
    print(f"k8 = 1/8 = {K8:.8f} h/Mpc")
    print(f"k_c = k(M=1e9 Msun/h) = {k_c:.8f} h/Mpc")
    print()

    print("## Mass-to-k proxy")
    print()
    print("| M [Msun/h] | R [h^-1 Mpc] | k=1/R [h/Mpc] |")
    print("|---:|---:|---:|")
    for mass in masses:
        r = radius_hinv_mpc(mass)
        print(f"| {mass:.1e} | {r:.8f} | {1.0 / r:.8f} |")
    print()

    print("## Window leakage and seed boost")
    print()
    print("| width s | W(k8) | large-scale boost | W(k_c) | seed boost |")
    print("|---:|---:|---:|---:|---:|")
    for width in widths:
        w8 = log_gaussian_window(K8, k_c, width)
        wc = log_gaussian_window(k_c, k_c, width)
        print(
            f"| {width:.2f} | {w8:.8e} | {boost_from_window(w8):.8f} | "
            f"{wc:.8f} | {boost_from_window(wc):.8f} |"
        )
    print()

    print("## Candidate tail outcomes at nu=3")
    print()
    ce_static_bg = 0.97125559
    ce_h0_bg = 0.93208112
    print("| branch | width s | total amp at k_c | tail ratio nu=3 |")
    print("|---|---:|---:|---:|")
    for label, bg in [("CE static", ce_static_bg), ("CE H0 branch", ce_h0_bg)]:
        for width in widths:
            wc = log_gaussian_window(k_c, k_c, width)
            total_amp = bg * boost_from_window(wc)
            print(f"| {label} | {width:.2f} | {total_amp:.8f} | {tail_ratio(total_amp, 3.0):.6f} |")
    print()

    print("## Required W for representative targets")
    print()
    print("| branch | target tail at nu=3 | required boost | required W | possible with W<=1? |")
    print("|---|---:|---:|---:|---|")
    rows = [
        ("CE static", 1.0, 1.02959510),
        ("CE static", 2.0, 1.11020527),
        ("CE static", 5.0, 1.25035512),
        ("CE H0 branch", 1.0, 1.07286799),
        ("CE H0 branch", 2.0, 1.15686613),
        ("CE H0 branch", 5.0, 1.30290634),
    ]
    for label, target, boost in rows:
        w_req = required_w(boost)
        possible = "yes" if w_req <= 1.0 else "no"
        print(f"| {label} | {target:.1f} | {boost:.8f} | {w_req:.8f} | {possible} |")
    print()

    print("## Verdict")
    print()
    print("A localized log-Gaussian condensate window can be negligible at k8 and O(1) at seed scales.")
    print("With amplitude delta, it can restore CE to LCDM and produce about 2-3x nu=3 rare-tail boosts.")
    print("It cannot produce arbitrary large enhancements unless W>1, a broader/nonlinear effect, or another source is added.")
    print("This is a viable candidate form, not a closed law.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
