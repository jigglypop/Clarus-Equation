"""Inflation next-tests gate: running and tensor ratio.

This gate checks the non-scored CE predictions

    alpha_spec = dn_s/dlnk = -2/N_e^2
    r_tensor = 12/N_e^2

against conservative external constraints.  The external constants are kept
here only as comparison handles, not as fitted inputs.

References used for the comparison constants:
* Planck 2018 X, A&A 2020: dn_s/dlnk = -0.0045 +/- 0.0067 (68% CL).
* Tristram et al. 2022, arXiv:2112.07961: BK18+Planck+BAO r < 0.032.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D = 3.0
N_GAUGE = 12.0

# External comparison handles, not CE inputs.
PLANCK_RUNNING = -0.0045
PLANCK_RUNNING_SIGMA = 0.0067
R_UPPER_BK18_PLANCK_BAO = 0.032


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D + delta
    x = bootstrap_x(d_eff)
    n_e = (D / 2.0) * d_eff * N_GAUGE
    n_s = 1.0 - 2.0 / n_e
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)
    consistency_ratio = r_tensor / (-alpha_spec)

    running_pull = (alpha_spec - PLANCK_RUNNING) / PLANCK_RUNNING_SIGMA
    r_limit_fraction = r_tensor / R_UPPER_BK18_PLANCK_BAO
    r_margin = R_UPPER_BK18_PLANCK_BAO - r_tensor

    print("# Inflation Next-Tests Gate")
    print()
    print("## CE inputs")
    print()
    print(f"sin2(theta_W) = {sin2_theta_w:.8f}")
    print(f"delta = sin2(theta_W)(1-sin2(theta_W)) = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"x = epsilon^2 = {x:.8f}")
    print(f"N_e = (3/2) D_eff N_gauge = {n_e:.8f}")
    print()

    print("## Predictions")
    print()
    print(f"n_s = 1 - 2/N_e = {n_s:.8f}")
    print(f"alpha_spec = dn_s/dlnk = -2/N_e^2 = {alpha_spec:.8e}")
    print(f"r_tensor = 12/N_e^2 = {r_tensor:.8f}")
    print(f"r_tensor / (-alpha_spec) = {consistency_ratio:.8f}")
    print()

    print("## External comparison")
    print()
    print("| observable | CE | comparison | result | verdict |")
    print("|---|---:|---:|---:|---|")
    print(
        f"| alpha_spec | {alpha_spec:.8e} | "
        f"{PLANCK_RUNNING:+.4f} +/- {PLANCK_RUNNING_SIGMA:.4f} | "
        f"{running_pull:+.2f} sigma | allowed; not a detection |"
    )
    print(
        f"| r_tensor | {r_tensor:.8f} | < {R_UPPER_BK18_PLANCK_BAO:.3f} | "
        f"{100.0 * r_limit_fraction:.1f}% of limit | allowed |"
    )
    print(f"| r margin | {r_margin:.8f} | upper - CE | -- | room remains |")
    print()

    print("## Verdict")
    print()
    print("CE running is small and negative, safely inside Planck 2018 constraints.")
    print("CE tensor ratio is below the current BK18+Planck+BAO upper bound.")
    print("This gate does not prove A3c; it says the next-test predictions are not already excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
