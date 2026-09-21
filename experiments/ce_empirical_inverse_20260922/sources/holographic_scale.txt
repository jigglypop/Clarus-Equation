"""Audit the cosmological constant absolute scale via the holographic channel.

The claim under test (docs/3_상수/7_우주론.md, 따름정리): the dark energy
density scale is not an independent open problem. It is locked to H0 through
the de Sitter entropy identity

    rho_crit = (3 / 8pi) H0^2 M_Pl^2 = (3/8) M_Pl^4 / S_dS,
    S_dS     = pi (M_Pl / H0)^2,

so once the phase-area derivation fixes log S_dS = (pi^2/2) N_e - pi*delta*sigma
with zero free parameters, the absolute scale follows:

    rho_Lambda^{1/4} = (Omega_Lambda * 3/8)^{1/4} M_Pl exp(-[(pi^2/2)N_e - pi*delta*sigma]/4).

The gate recomputes N_e, the recursion defect, and the resulting meV scale, then
compares to the observed 2.24 meV. It also reports the 10^-122 hierarchy exponent
as (pi^2/2) N_e / ln 10.
"""

from __future__ import annotations

import math

ALPHA_S = 0.11789
N_GAUGE = 12
D_SPACE = 3
M_PL_EV = 1.220910e28  # non-reduced Planck mass in eV (S_dS = pi (M_Pl/H0)^2)
OMEGA_LAMBDA = 0.6891   # CE bootstrap value (docs/경로적분.md 3.3.1)
RHO_LAMBDA_OBS_MEV = 2.24


def recursion_fixed_point(d_eff: float, iters: int = 4000) -> float:
    x = 0.05
    for _ in range(iters):
        x = math.exp(-(1.0 - x) * d_eff)
    return x


def derive_entropy() -> dict:
    sin2_tw = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_tw * (1.0 - sin2_tw)
    d_eff = D_SPACE + delta
    n_e = 0.5 * D_SPACE * d_eff * N_GAUGE
    sigma = 1.0 - recursion_fixed_point(d_eff)
    log_s = (math.pi ** 2 / 2.0) * n_e - math.pi * delta * sigma
    return {
        "delta": delta,
        "d_eff": d_eff,
        "n_e": n_e,
        "sigma": sigma,
        "log_s": log_s,
    }


def rho_lambda_quarter_mev(log_s: float, omega_lambda: float) -> float:
    rho_crit = (3.0 / 8.0) * M_PL_EV ** 4 / math.exp(log_s)
    rho_lambda = omega_lambda * rho_crit
    return 1.0e3 * rho_lambda ** 0.25


def rel_error(pred: float, ref: float) -> float:
    return 100.0 * (pred / ref - 1.0)


def main() -> int:
    d = derive_entropy()

    print("# Cosmological Constant Holographic Gate")
    print()
    print("## Phase-area entropy (zero free parameters)")
    print()
    print(f"delta (recursion)      = {d['delta']:.6f}")
    print(f"D_eff                  = {d['d_eff']:.6f}")
    print(f"N_e = (d/2) D_eff N_g  = {d['n_e']:.5f}")
    print(f"sigma (fixed point)    = {d['sigma']:.6f}")
    print(f"log S_dS               = {d['log_s']:.5f}")
    print()

    q_ce = rho_lambda_quarter_mev(d["log_s"], OMEGA_LAMBDA)
    q_crit = rho_lambda_quarter_mev(d["log_s"], 1.0)
    print("## Absolute scale rho_Lambda^{1/4}")
    print()
    print(f"rho_crit^{{1/4}}          = {q_crit:.4f} meV (Omega=1)")
    print(f"rho_Lambda^{{1/4}} (CE)    = {q_ce:.4f} meV (Omega_Lambda={OMEGA_LAMBDA})")
    print(f"observed                = {RHO_LAMBDA_OBS_MEV:.2f} meV")
    print(f"error                   = {rel_error(q_ce, RHO_LAMBDA_OBS_MEV):+.3f}%")
    print()

    exponent_122 = (math.pi ** 2 / 2.0) * d["n_e"] / math.log(10.0)
    print("## Hierarchy exponent")
    print()
    print(f"log10(rho_Lambda/M_Pl^4) magnitude = (pi^2/2) N_e / ln10 = {exponent_122:.2f}")
    print("This is the famous ~122 of the cosmological constant problem.")
    print()

    print("## Verdict")
    print()
    within = abs(rel_error(q_ce, RHO_LAMBDA_OBS_MEV)) < 0.2
    print("rho_Lambda absolute scale reproduced to <0.2%:", within)
    print("Channel: holographic rho_Lambda = Omega_Lambda (3/8) M_Pl^4 / S_dS,")
    print("not additive Coleman-Weinberg vacuum sum.")
    print("Status: Bridge, locked to H0 via the same holographic identity (S_dS = A/4G).")
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())
