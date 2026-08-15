"""Historical holographic-scale reproduction with fail-closed labels.

The old calculation multiplied a phase-area horizon scale by an observational
``Omega_Lambda`` and described the result as a zero-parameter de Sitter
prediction. That interpretation is not valid without first deciding which
horizon the entropy belongs to. For a true asymptotic de Sitter horizon,

    S_dS = pi (M_Pl / H_L)^2,
    rho_Lambda = (3/8) M_Pl^4 / S_dS,

whereas in a mixed matter+Lambda epoch ``H_L = H0*sqrt(Omega_Lambda)``. The
transition scale ``H_*`` is a third quantity. This module keeps the historical
number reproducible, but only behind an explicit opt-in; the unified closure
gate is ``cosmology_closure_gate.py``.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

ALPHA_S = 0.11789
N_GAUGE = 12
D_SPACE = 3
M_PL_EV = 1.220910e28  # non-reduced Planck mass in eV (S_dS = pi (M_Pl/H)^2)
OMEGA_LAMBDA = 0.6891  # historical CE runtime value; not a current observation
RHO_LAMBDA_OBS_MEV = 2.24
HISTORICAL_HOLOGRAPHIC_MODEL_ID = "HISTORICAL_PHASE_AREA_H0_MIX_V1"


def recursion_fixed_point(d_eff: float, iters: int = 4000) -> float:
    x = 0.05
    for _ in range(iters):
        x = math.exp(-(1.0 - x) * d_eff)
    return x


def derive_entropy() -> dict[str, float]:
    """Reproduce the historical dimensionless phase-area entropy candidate."""
    sin2_tw = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_tw * (1.0 - sin2_tw)
    d_eff = D_SPACE + delta
    n_e = 0.5 * D_SPACE * d_eff * N_GAUGE
    sigma = 1.0 - recursion_fixed_point(d_eff)
    log_s = (math.pi**2 / 2.0) * n_e - math.pi * delta * sigma
    return {
        "delta": delta,
        "d_eff": d_eff,
        "n_e": n_e,
        "sigma": sigma,
        "log_s": log_s,
    }


def rho_lambda_quarter_mev(log_s: float, omega_lambda: float) -> float:
    """Return the historical mixed-epoch readout (compatibility API only)."""
    rho_crit = (3.0 / 8.0) * M_PL_EV**4 / math.exp(log_s)
    rho_lambda = omega_lambda * rho_crit
    return 1.0e3 * rho_lambda**0.25


def true_de_sitter_vacuum_quarter_mev(log_s: float) -> float:
    """Return the vacuum scale if ``log_s`` is truly asymptotic dS entropy."""
    rho_lambda = (3.0 / 8.0) * M_PL_EV**4 / math.exp(log_s)
    return 1.0e3 * rho_lambda**0.25


def h_lambda_over_h0(omega_lambda: float) -> float:
    """Return ``H_L/H0`` for flat mixed-epoch LambdaCDM."""
    if not 0.0 < omega_lambda <= 1.0:
        raise ValueError("omega_lambda must be in (0, 1]")
    return math.sqrt(omega_lambda)


def horizon_scale_definitions() -> dict[str, dict[str, object]]:
    """Definitions consumed by the unified fail-closed closure gate."""
    return {
        "H_L": {
            "epoch": "asymptotic_de_sitter",
            "definition": "sqrt(8*pi*G*rho_Lambda/3)",
            "value_status": "phase-law candidate; absolute bridge incomplete",
        },
        "H_*": {
            "epoch": "action_defined_transition_surface",
            "definition": "H evaluated on Sigma_*",
            "value_status": "[미완성]",
        },
        "H0": {
            "epoch": "present_observer",
            "definition": "H(a=1)",
            "value_status": "external/inferred; not derived here",
        },
    }


def rel_error(pred: float, ref: float) -> float:
    return 100.0 * (pred / ref - 1.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cosmological_constant_holographic_gate")
    parser.add_argument(
        "--historical-reproduction",
        action="store_true",
        help="run the target-aware legacy mixed-horizon calculation",
    )
    args = parser.parse_args(argv)
    if not args.historical_reproduction:
        print(
            "Historical mixed-horizon calculation disabled: pass "
            "--historical-reproduction; vacuum-scale closure remains [미완성]."
        )
        return 2

    d = derive_entropy()

    print("# Cosmological Constant Historical Reproduction")
    print()
    print(f"model_id: {HISTORICAL_HOLOGRAPHIC_MODEL_ID}")
    print("physical_closure: false")
    print("blind_prediction: false")
    print()
    print("## Phase-area entropy candidate")
    print()
    print(f"delta (recursion)      = {d['delta']:.6f}")
    print(f"D_eff                  = {d['d_eff']:.6f}")
    print(f"N_e = (d/2) D_eff N_g  = {d['n_e']:.5f}")
    print(f"sigma (fixed point)    = {d['sigma']:.6f}")
    print(f"log S candidate        = {d['log_s']:.5f}")
    print()

    q_legacy = rho_lambda_quarter_mev(d["log_s"], OMEGA_LAMBDA)
    q_true_ds = true_de_sitter_vacuum_quarter_mev(d["log_s"])
    print("## Historical target-aware scale comparison")
    print()
    print(f"true-dS rho_Lambda^{{1/4}} = {q_true_ds:.4f} meV (if S is S_dS)")
    print(f"legacy mixed readout      = {q_legacy:.4f} meV (Omega_Lambda={OMEGA_LAMBDA})")
    print(f"observed                  = {RHO_LAMBDA_OBS_MEV:.2f} meV")
    print(f"legacy comparison error   = {rel_error(q_legacy, RHO_LAMBDA_OBS_MEV):+.3f}%")
    print(f"H_L/H0                    = {h_lambda_over_h0(OMEGA_LAMBDA):.6f}")
    print("H_*                       = undefined until the transition action closes")
    print()

    exponent_122 = (math.pi**2 / 2.0) * d["n_e"] / math.log(10.0)
    print("## Hierarchy exponent")
    print()
    print(f"historical exponent = {exponent_122:.2f}")
    print()

    within = abs(rel_error(q_legacy, RHO_LAMBDA_OBS_MEV)) < 0.2
    print("## Audit verdict")
    print()
    print("legacy numerical proximity <0.2%:", within)
    print("Status: HISTORICAL_REPRODUCTION_ONLY; target-aware; not physical closure.")
    print("H_L, H_*, and H0 are distinct definitions unless a new bridge proves equality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
