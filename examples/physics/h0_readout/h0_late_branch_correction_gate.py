"""Test whether a minimal late-time branch correction can reach high H0 data.

The horizon invariant gives a low-H0 prediction:
    log S = (pi^2 / 2) N_e - pi delta sigma.

This gate asks what additive log-entropy correction is required by each H0
dataset and compares it with small CE defect scales such as delta*sigma.
"""

from __future__ import annotations

import math

from h0_dataset_falsification_gate import (
    ALPHA_S,
    DATASETS,
    D_SPATIAL,
    MPC_KM,
    N_GAUGE,
    T_PLANCK_S,
    bootstrap_x,
    h0_from_log_s,
    log_s_from_h0,
)


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    boundary_integrated = math.pi * delta * sigma
    defect_local = delta * sigma
    log_s_low = phase_area * n_e - boundary_integrated
    h0_low = h0_from_log_s(log_s_low)

    corrections = [
        ("none: integrated horizon defect only", 0.0),
        ("local defect subtract: -delta*sigma", -defect_local),
        ("half local defect subtract: -delta*sigma/2", -0.5 * defect_local),
        ("delta/2 subtract: -delta/2", -0.5 * delta),
        ("baryon endpoint subtract: -x", -x),
        ("pi*x subtract: -pi*x", -math.pi * x),
    ]

    print("# H0 Late-Branch Correction Gate")
    print()
    print("## Core scales")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"delta*sigma = {defect_local:.8f}")
    print(f"pi*delta*sigma = {boundary_integrated:.8f}")
    print(f"H0_low_branch = {h0_low:.6f} km/s/Mpc")
    print()

    print("## Candidate branch corrections")
    print()
    print("| correction | Delta log S | H0 pred | nearest branch |")
    print("|---|---:|---:|---|")
    for name, corr in corrections:
        h0 = h0_from_log_s(log_s_low + corr)
        nearest = min(DATASETS, key=lambda row: abs(float(row["h0"]) - h0))
        print(f"| {name} | {corr:+.8f} | {h0:.6f} | {nearest['name']} |")
    print()

    print("## Required correction by dataset")
    print()
    print("| dataset | H0 obs | required Delta log S | /(-delta*sigma) | residual after -delta*sigma |")
    print("|---|---:|---:|---:|---:|")
    for row in DATASETS:
        h0_obs = float(row["h0"])
        required = log_s_from_h0(h0_obs) - log_s_low
        ratio = required / (-defect_local)
        h0_high = h0_from_log_s(log_s_low - defect_local)
        residual_pull = (h0_high - h0_obs) / float(row["sigma"])
        print(
            f"| {row['name']} | {h0_obs:.3f} | {required:+.8f} | "
            f"{ratio:.4f} | {residual_pull:+.2f} |"
        )
    print()

    h0_high = h0_from_log_s(log_s_low - defect_local)
    print("## Verdict")
    print()
    print(f"A single extra local endpoint defect, -delta*sigma, predicts H0 = {h0_high:.6f}.")
    print("This lands on the high Cepheid/SN branch, but it spoils the low-H0 Planck branch.")
    print("Thus CE now has a possible two-readout structure, not a single universal H0 value:")
    print("global horizon readout -> low H0; local endpoint-defect readout -> high H0.")
    print("The open problem is deriving why a given observational ladder should use one readout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
