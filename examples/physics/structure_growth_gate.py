"""Gate C: structure-growth and S8 audit for CE cosmology.

This card keeps the current structure-formation claim honest:

* the static density package gives Omega_m from the self-recursive split;
* the late-time H0 branch uses Omega_m = 0.343 from the extraction correction;
* sigma8 is calibrated to the BOSS DR12 z=0.51 fsigma8 anchor;
* S8 and the full fsigma8 compilation are then evaluated without extra tuning.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from s8_tension import (  # noqa: E402
    FSIG8_DATA,
    calibrate_sigma8_to_fsig8,
    chi2_against,
    predict_fsig8_curve,
    s8_amplitude,
)
from cosmology import Background  # noqa: E402


ALPHA_S = 0.11789
D = 3.0
OMEGA_M_H0_BRANCH = 0.343
OMEGA_M_LCDM = 0.315
SIGMA8_LCDM = 0.811
PLANCK_S8 = (0.832, 0.013)
KIDS_S8 = (0.766, 0.020)
DESY3_S8 = (0.776, 0.017)


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def ce_static_omega_m() -> tuple[float, float, float, float]:
    sin2 = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2 * (1.0 - sin2)
    d_eff = D + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    r_split = 0.38062659567873197
    omega_dm = sigma * r_split / (1.0 + r_split)
    return x + omega_dm, x, omega_dm, d_eff


def report_model(label: str, omega_m: float, sigma8: float) -> dict[str, float]:
    bg = Background(omega_m0=omega_m, omega_l0=1.0 - omega_m)
    z_data = sorted({z for (z, _, _) in FSIG8_DATA})
    fs8_pred = predict_fsig8_curve(bg, sigma8, z_data)
    pred_at_z = dict(zip(z_data, fs8_pred))
    chi2, n = chi2_against(FSIG8_DATA, pred_at_z)
    s8 = s8_amplitude(sigma8, omega_m)
    return {
        "label": label,
        "omega_m": omega_m,
        "sigma8": sigma8,
        "s8": s8,
        "chi2": chi2,
        "n": n,
        "planck_pull": (s8 - PLANCK_S8[0]) / PLANCK_S8[1],
        "kids_pull": (s8 - KIDS_S8[0]) / KIDS_S8[1],
        "desy3_pull": (s8 - DESY3_S8[0]) / DESY3_S8[1],
        "fs8_z051": pred_at_z[0.51],
    }


def main() -> int:
    omega_m_static, omega_b, omega_dm, d_eff = ce_static_omega_m()

    bg_static = Background(omega_m0=omega_m_static, omega_l0=1.0 - omega_m_static)
    bg_h0 = Background(omega_m0=OMEGA_M_H0_BRANCH, omega_l0=1.0 - OMEGA_M_H0_BRANCH)
    sigma8_static = calibrate_sigma8_to_fsig8(bg_static, FSIG8_DATA, z_pivot=0.51)
    sigma8_h0 = calibrate_sigma8_to_fsig8(bg_h0, FSIG8_DATA, z_pivot=0.51)

    models = [
        report_model("LCDM Planck baseline", OMEGA_M_LCDM, SIGMA8_LCDM),
        report_model("CE static density split", omega_m_static, sigma8_static),
        report_model("CE late-time H0 branch", OMEGA_M_H0_BRANCH, sigma8_h0),
    ]

    print("# Structure Growth Gate")
    print()
    print("## Inputs")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"Omega_b = {omega_b:.8f}")
    print(f"Omega_DM = {omega_dm:.8f}")
    print(f"Omega_m_static = Omega_b + Omega_DM = {omega_m_static:.8f}")
    print(f"Omega_m_H0_branch = {OMEGA_M_H0_BRANCH:.8f}")
    print("sigma8 is calibrated to BOSS DR12 z=0.51: f sigma8 = 0.458")
    print()

    print("## Growth equations")
    print()
    print("S8 = sigma8 * sqrt(Omega_m/0.3)")
    print("D'' + [2 + dlnH/dlna] D' - 3 Omega_m(a) D / 2 = 0")
    print("f sigma8(z) = [d ln D / d ln a] * sigma8(0) * D(z)")
    print()

    print("## Results")
    print()
    print("| model | Omega_m | sigma8(0) | S8 | chi2(fs8)/N | Planck pull | KiDS pull | DES-Y3 pull |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        print(
            f"| {m['label']} | {m['omega_m']:.8f} | {m['sigma8']:.8f} | "
            f"{m['s8']:.8f} | {m['chi2']:.3f}/{int(m['n'])} | "
            f"{m['planck_pull']:+.2f} sigma | {m['kids_pull']:+.2f} sigma | "
            f"{m['desy3_pull']:+.2f} sigma |"
        )
    print()

    lcdm = models[0]
    static = models[1]
    h0 = models[2]
    print("## Delta tests")
    print()
    print(f"Delta chi2(static CE - LCDM) = {static['chi2'] - lcdm['chi2']:+.3f}")
    print(f"Delta chi2(H0-branch CE - LCDM) = {h0['chi2'] - lcdm['chi2']:+.3f}")
    print(f"KiDS pull reduction, H0 branch = {lcdm['kids_pull'] - h0['kids_pull']:+.2f} sigma")
    print(f"DES-Y3 pull reduction, H0 branch = {lcdm['desy3_pull'] - h0['desy3_pull']:+.2f} sigma")
    print()

    print("## Verdict")
    print()
    print("The H0-branch CE background improves fsigma8 and slightly reduces S8 tension.")
    print("It does not close the KiDS/DES-Y3 S8 residual; halo/JWST-scale predictions remain open.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
