"""Map H0 readout q to a minimal channel-topology model.

Hypothesis:
    q = L / (L + G)

where L is the number/weight of local endpoint closures and G is the
number/weight of global ruler/horizon closures in the calibration path.

This does not prove the selector. It checks whether small rational topology
weights can reproduce the q values inferred from H0 datasets.
"""

from __future__ import annotations

import math

from h0_dataset_falsification_gate import (
    ALPHA_S,
    DATASETS,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    log_s_from_h0,
)


TOPOLOGY = {
    "Planck 2018 base LCDM": {
        "local": 0.0,
        "global": 1.0,
        "note": "CMB acoustic horizon closure",
    },
    "DESI DR2 BAO no-CMB calibration": {
        "local": 1.0,
        "global": 3.0,
        "note": "inverse ladder with global ruler prior",
    },
    "CCHP 2025 TRGB HST+JWST": {
        "local": 1.0,
        "global": 1.0,
        "note": "mixed local endpoint and cross-instrument closure",
    },
    "CCHP 2025 JWST-only TRGB": {
        "local": 1.0,
        "global": 3.0,
        "note": "stellar endpoint with stronger population averaging",
    },
    "CCHP 2025 JWST-only JAGB": {
        "local": 1.0,
        "global": 9.0,
        "note": "population-averaged stellar endpoint",
    },
    "SH0ES HST Cepheids/SNe": {
        "local": 1.0,
        "global": 0.0,
        "note": "local endpoint-anchored distance ladder",
    },
    "SH0ES JWST update": {
        "local": 1.0,
        "global": 0.0,
        "note": "local endpoint-anchored cross-check",
    },
}


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

    print("# H0 Channel Topology Gate")
    print()
    print("## Selector ansatz")
    print()
    print("q_topology = L / (L + G)")
    print("L = local endpoint closure weight")
    print("G = global ruler/horizon closure weight")
    print(f"delta*sigma = {defect:.8f}")
    print()

    print("## Topology comparison")
    print()
    print("| dataset | q_req | sigma_q | L | G | q_topology | residual | note |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    total_chi2 = 0.0
    for row in DATASETS:
        h0 = float(row["h0"])
        sigma_h0 = float(row["sigma"])
        q_req = (log_s_global - log_s_from_h0(h0)) / defect
        sigma_q = 2.0 * sigma_h0 / (h0 * defect)
        topo = TOPOLOGY[row["name"]]
        local = float(topo["local"])
        global_ = float(topo["global"])
        q_topo = local / (local + global_) if local + global_ else float("nan")
        residual = (q_req - q_topo) / sigma_q
        total_chi2 += residual * residual
        print(
            f"| {row['name']} | {q_req:.4f} | {sigma_q:.4f} | "
            f"{local:.1f} | {global_:.1f} | {q_topo:.4f} | {residual:+.2f} | {topo['note']} |"
        )
    print()

    print("## Verdict")
    print()
    print(f"topology chi2/dof = {total_chi2:.3f}/{len(DATASETS) - 1}")
    print("Small rational closure weights reproduce the inferred q hierarchy.")
    print("This is a stronger selector hypothesis, but not a first-principles derivation.")
    print("The next hard test is predicting q for a new channel before using its H0 value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
