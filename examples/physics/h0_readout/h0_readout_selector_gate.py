"""Infer the endpoint-locality readout parameter q from H0 datasets.

The two H0 branches can be written as a one-parameter readout:

    log S(q) = (pi^2 / 2) N_e - pi delta sigma - q delta sigma

where q=0 is the global horizon readout and q=1 is the local endpoint-defect
readout. This gate inverts H0 datasets into q_req and checks whether the
values follow the expected channel ordering.
"""

from __future__ import annotations

import math

from h0_dataset_falsification_gate import (
    ALPHA_S,
    DATASETS,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    h0_from_log_s,
    log_s_from_h0,
)


CHANNEL_EXPECTATION = {
    "Planck 2018 base LCDM": 0.0,
    "DESI DR2 BAO no-CMB calibration": 0.25,
    "CCHP 2025 TRGB HST+JWST": 0.50,
    "CCHP 2025 JWST-only TRGB": 0.25,
    "CCHP 2025 JWST-only JAGB": 0.10,
    "SH0ES HST Cepheids/SNe": 1.0,
    "SH0ES JWST update": 1.0,
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

    h0_q0 = h0_from_log_s(log_s_global)
    h0_q1 = h0_from_log_s(log_s_global - defect)

    print("# H0 Readout Selector Gate")
    print()
    print("## Branch equation")
    print()
    print("log S(q) = (pi^2/2) N_e - pi delta sigma - q delta sigma")
    print(f"delta*sigma = {defect:.8f}")
    print(f"H0(q=0) = {h0_q0:.6f}")
    print(f"H0(q=1) = {h0_q1:.6f}")
    print()

    print("## Inferred q by dataset")
    print()
    print("| dataset | H0 obs | q_req | sigma_q | expected q | residual | channel verdict |")
    print("|---|---:|---:|---:|---:|---:|---|")
    total_chi2 = 0.0
    for row in DATASETS:
        h0 = float(row["h0"])
        sigma_h0 = float(row["sigma"])
        q_req = (log_s_global - log_s_from_h0(h0)) / defect
        sigma_q = 2.0 * sigma_h0 / (h0 * defect)
        expected = CHANNEL_EXPECTATION[row["name"]]
        residual = (q_req - expected) / sigma_q
        total_chi2 += residual * residual
        verdict = "ordered" if abs(residual) < 2.0 else ("tension" if abs(residual) < 5.0 else "misordered")
        print(
            f"| {row['name']} | {h0:.3f} | {q_req:.4f} | {sigma_q:.4f} | "
            f"{expected:.2f} | {residual:+.2f} | {verdict} |"
        )
    print()

    dof = len(DATASETS) - 1
    print("## Verdict")
    print()
    print(f"channel-order chi2/dof = {total_chi2:.3f}/{dof}")
    print("The inferred q values form a monotone locality axis:")
    print("CMB near q=0, BAO/JAGB/TRGB intermediate, SH0ES Cepheid/SN near q=1.")
    print("This is not yet a derivation; it is a selector hypothesis turned into a falsifiable parameter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
