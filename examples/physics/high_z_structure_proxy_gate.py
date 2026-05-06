"""High-z structure proxy gate for CE cosmology.

This is not a full halo mass function.  It is a deliberately narrow sanity
check: if CE only changes the background growth and the sigma8 normalization
already used in the S8/fsigma8 gate, does it automatically enhance rare
high-redshift halo abundance?

Answer from this proxy: no.  The high-z linear amplitudes are slightly lower
than the LCDM Planck baseline, so rare-object tails are suppressed unless CE
adds a separate scale-dependent condensate/transfer-function effect.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cosmology import Background, interp_linear, logspace, solve_growth  # noqa: E402
from s8_tension import FSIG8_DATA, calibrate_sigma8_to_fsig8  # noqa: E402
from structure_growth_gate import ce_static_omega_m  # noqa: E402


OMEGA_M_LCDM = 0.315
SIGMA8_LCDM = 0.811
OMEGA_M_H0_BRANCH = 0.343
DELTA_C = 1.686


def growth_curve(omega_m: float, sigma8: float) -> tuple[list[float], list[float], list[float]]:
    bg = Background(omega_m0=omega_m, omega_l0=1.0 - omega_m)
    a_grid = logspace(1.0e-4, 1.0, 3001)
    ln_a_grid = [math.log(a) for a in a_grid]
    mu_grid = [1.0 for _ in a_grid]
    d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)
    sigma8_z = [sigma8 * d for d in d_norm]
    return ln_a_grid, d_norm, sigma8_z


def value_at_z(ln_a_grid: list[float], values: list[float], z: float) -> float:
    return interp_linear(ln_a_grid, values, math.log(1.0 / (1.0 + z)))


def tail_ratio(amplitude_ratio: float, nu_lcdm: float) -> float:
    """Press-Schechter erfc tail ratio at fixed LCDM rarity nu."""
    if amplitude_ratio <= 0.0:
        return 0.0
    nu_ce = nu_lcdm / amplitude_ratio
    lcdm_tail = math.erfc(nu_lcdm / math.sqrt(2.0))
    ce_tail = math.erfc(nu_ce / math.sqrt(2.0))
    if lcdm_tail <= 0.0:
        return float("inf")
    return ce_tail / lcdm_tail


def main() -> int:
    omega_m_static, omega_b, omega_dm, d_eff = ce_static_omega_m()
    bg_static = Background(omega_m0=omega_m_static, omega_l0=1.0 - omega_m_static)
    bg_h0 = Background(omega_m0=OMEGA_M_H0_BRANCH, omega_l0=1.0 - OMEGA_M_H0_BRANCH)
    sigma8_static = calibrate_sigma8_to_fsig8(bg_static, FSIG8_DATA, z_pivot=0.51)
    sigma8_h0 = calibrate_sigma8_to_fsig8(bg_h0, FSIG8_DATA, z_pivot=0.51)

    models = [
        ("LCDM Planck", OMEGA_M_LCDM, SIGMA8_LCDM),
        ("CE static", omega_m_static, sigma8_static),
        ("CE H0 branch", OMEGA_M_H0_BRANCH, sigma8_h0),
    ]
    curves = {
        label: (omega_m, sigma8, *growth_curve(omega_m, sigma8))
        for label, omega_m, sigma8 in models
    }

    redshifts = [8.0, 10.0, 12.0, 15.0]
    rarity_bins = [2.0, 3.0, 4.0]

    print("# High-z Structure Proxy Gate")
    print()
    print("## Inputs")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"Omega_b = {omega_b:.8f}")
    print(f"Omega_DM = {omega_dm:.8f}")
    print(f"Omega_m_static = {omega_m_static:.8f}")
    print(f"sigma8_static = {sigma8_static:.8f}")
    print(f"Omega_m_H0_branch = {OMEGA_M_H0_BRANCH:.8f}")
    print(f"sigma8_H0_branch = {sigma8_h0:.8f}")
    print()

    print("## Linear amplitude at high z")
    print()
    print("sigma8(z) = sigma8(0) D(z), with D(0)=1")
    print()
    print("| z | LCDM sigma8(z) | CE static / LCDM | CE H0 branch / LCDM |")
    print("|---:|---:|---:|---:|")
    for z in redshifts:
        _, _, ln_l, _, sig_l = curves["LCDM Planck"]
        _, _, ln_s, _, sig_s = curves["CE static"]
        _, _, ln_h, _, sig_h = curves["CE H0 branch"]
        lcdm = value_at_z(ln_l, sig_l, z)
        static = value_at_z(ln_s, sig_s, z)
        h0 = value_at_z(ln_h, sig_h, z)
        print(f"| {z:.0f} | {lcdm:.8f} | {static / lcdm:.8f} | {h0 / lcdm:.8f} |")
    print()

    print("## Rare-tail proxy")
    print()
    print("For a fixed mass scale, define a LCDM rarity nu = delta_c/sigma_M(z).")
    print("The CE tail proxy is erfc[(nu/R_amp)/sqrt(2)] / erfc[nu/sqrt(2)].")
    print("Values below 1 mean fewer rare high-z objects than LCDM at the same mass scale.")
    print()
    print("| z | model | nu=2 | nu=3 | nu=4 |")
    print("|---:|---|---:|---:|---:|")
    for z in redshifts:
        _, _, ln_l, _, sig_l = curves["LCDM Planck"]
        lcdm = value_at_z(ln_l, sig_l, z)
        for label in ["CE static", "CE H0 branch"]:
            _, _, ln_c, _, sig_c = curves[label]
            ce = value_at_z(ln_c, sig_c, z)
            amp_ratio = ce / lcdm
            ratios = [tail_ratio(amp_ratio, nu) for nu in rarity_bins]
            print(
                f"| {z:.0f} | {label} | "
                f"{ratios[0]:.6f} | {ratios[1]:.6f} | {ratios[2]:.6f} |"
            )
    print()

    print("## Verdict")
    print()
    print("Background-only CE does not automatically enhance high-z rare halos.")
    print("A JWST/halo closure needs an additional scale-dependent transfer,")
    print("collective-condensate growth term, or revised primordial small-scale power.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
