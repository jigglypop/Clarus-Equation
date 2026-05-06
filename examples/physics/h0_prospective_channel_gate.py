"""Prospective H0-channel test from topology before comparison.

This gate applies q_topology = L/(L+G) to channels that were not used to build
the initial CMB/BAO/CCHP/SH0ES selector table:

* H0LiCOW/TDCOSMO power-law time-delay lenses
* TDCOSMO+SLACS hierarchical time-delay lenses
* Megamaser Cosmology Project
* GW standard sirens

The topology assignment is made first, then H0(q) is compared to literature
representative values.
"""

from __future__ import annotations

import math

from h0_dataset_falsification_gate import (
    ALPHA_S,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    h0_from_log_s,
)


CHANNELS = [
    {
        "name": "H0LiCOW/TDCOSMO power-law lenses",
        "local": 1.0,
        "global": 0.0,
        "h0": 73.3,
        "sigma": 1.75,
        "reason": "lens time-delay distance with local lens-model closure",
    },
    {
        "name": "TDCOSMO+SLACS hierarchical lenses",
        "local": 1.0,
        "global": 3.0,
        "h0": 67.4,
        "sigma": 3.65,
        "reason": "lens closure plus population/kinematic global prior",
    },
    {
        "name": "Megamaser Cosmology Project",
        "local": 1.0,
        "global": 0.0,
        "h0": 73.9,
        "sigma": 3.0,
        "reason": "one-step geometric local distance in Hubble flow",
    },
    {
        "name": "GW170817 bright standard siren",
        "local": 1.0,
        "global": 1.0,
        "h0": 70.3,
        "sigma": 5.15,
        "reason": "absolute GW distance plus host/environment velocity closure",
    },
    {
        "name": "O4a dark+bright standard sirens",
        "local": 1.0,
        "global": 1.0,
        "h0": 68.0,
        "sigma": 4.1,
        "reason": "GW distance with catalog/statistical redshift closure",
    },
]


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    defect = delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    log_s_global = phase_area * n_e - math.pi * defect

    print("# H0 Prospective Channel Gate")
    print()
    print("## Prediction rule")
    print()
    print("q_topology = L / (L + G)")
    print("log S(q) = (pi^2/2) N_e - pi delta sigma - q delta sigma")
    print(f"delta*sigma = {defect:.8f}")
    print()

    print("## Prospective channel predictions")
    print()
    print("| channel | L:G | q | H0_pred | H0_obs | pull | topology reason |")
    print("|---|---:|---:|---:|---:|---:|---|")
    chi2 = 0.0
    for row in CHANNELS:
        local = float(row["local"])
        global_ = float(row["global"])
        q = local / (local + global_)
        h0_pred = h0_from_log_s(log_s_global - q * defect)
        h0_obs = float(row["h0"])
        sigma_h0 = float(row["sigma"])
        pull = (h0_pred - h0_obs) / sigma_h0
        chi2 += pull * pull
        print(
            f"| {row['name']} | {local:.0f}:{global_:.0f} | {q:.4f} | "
            f"{h0_pred:.3f} | {h0_obs:.3f} +/- {sigma_h0:.3f} | {pull:+.2f} | {row['reason']} |"
        )
    print()

    print("## Verdict")
    print()
    print(f"prospective-channel chi2/dof = {chi2:.3f}/{len(CHANNELS)}")
    print("The topology rule predicts high H0 for local geometric closures,")
    print("low/intermediate H0 for hierarchical/global closures, and middle H0 for standard sirens.")
    print("This is the first nontrivial external-channel check of q_topology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
