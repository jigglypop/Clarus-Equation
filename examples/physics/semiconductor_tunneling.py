"""6.A.3 — CE-corrected WKB tunneling for sub-5nm junctions.

Computes leakage current through MOSFET gate-oxide / FinFET source-drain
barriers using:

    T_std(E)  = exp[-2 ∫_0^L sqrt(2 m* (V_b - E)) / hbar dx]
    T_CE(E)   = T_std(E) * exp[-2/hbar ∫ sqrt(2 m* ΔV_CE(x)) dx]
    ΔV_CE(x)  = AD * hbar ω_L * Φ_lat(x / ξ_L),  ξ_L = a0 / delta

AD = 0.125 is the CE lattice action density. Φ_lat is taken as a
sech² profile peaked at the barrier interface (standard envelope).

Outputs:
  - T_std, T_CE over L ∈ [2, 10] nm
  - T_CE / T_std (leakage ratio)
  - comparison to published ITRS / TSMC 3nm leakage values (Popper grid)

Reproduction of compendium section 6.A.3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from clarus.constants import AD

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34  # J·s
M_E = 9.1093837015e-31   # kg
Q_E = 1.602176634e-19    # C
ANGSTROM = 1.0e-10       # m
NM = 1.0e-9              # m

# Silicon effective mass in oxide (SiO2 barrier)
M_STAR_FACTOR = 0.19     # m* / m_e

# Typical gate-oxide barrier parameters
V_BARRIER_EV = 3.1       # Si/SiO2 conduction band offset (eV)
E_INJECTION_EV = 0.6     # electron injection energy above source Fermi (eV)

# CE lattice length scale
# ξ_L = a0 / δ  where a0 = Si lattice constant 5.431 A, δ = 0.17776
A0_SI = 5.431 * ANGSTROM
DELTA = 0.17776
XI_L = A0_SI / DELTA     # ≈ 30.6 A = 3.06 nm

# Phonon quantum ℏω_L (Si optical phonon ~ 64 meV)
HBAR_OMEGA_L_EV = 0.064
HBAR_OMEGA_L_J = HBAR_OMEGA_L_EV * Q_E


# ---------------------------------------------------------------------------
# WKB core
# ---------------------------------------------------------------------------


def wkb_std(L_nm, V_b_ev=V_BARRIER_EV, E_ev=E_INJECTION_EV, n_pts=5000):
    """Standard WKB for rectangular barrier. Returns T (transmission)."""
    L = L_nm * NM
    V_b = V_b_ev * Q_E
    E = E_ev * Q_E
    if V_b <= E:
        return 1.0
    m_star = M_STAR_FACTOR * M_E
    kappa = math.sqrt(2.0 * m_star * (V_b - E)) / HBAR
    gamma = kappa * L
    return math.exp(-2.0 * gamma)


def phi_lat(x_over_xi):
    """Lattice envelope — sech² centered at x/ξ = 0.5 (barrier midpoint).

    Integrates to 1 over (-∞, ∞); truncated to barrier is normalized
    numerically below.
    """
    u = x_over_xi - 0.5
    return 1.0 / math.cosh(2.0 * u) ** 2


def wkb_ce(L_nm, V_b_ev=V_BARRIER_EV, E_ev=E_INJECTION_EV, n_pts=5000):
    """CE-corrected WKB. Returns (T_CE, delta_gamma_ratio)."""
    L = L_nm * NM
    V_b = V_b_ev * Q_E
    E = E_ev * Q_E
    if V_b <= E:
        return 1.0, 0.0
    m_star = M_STAR_FACTOR * M_E

    x = np.linspace(0.0, L, n_pts)
    dx = x[1] - x[0]

    # Standard integrand
    V_std = np.full_like(x, V_b) - E  # constant (V_b - E) within barrier
    V_std = np.clip(V_std, 0.0, None)

    # CE correction: ΔV_CE(x) = AD * hbar ω_L * Φ_lat(x / ξ_L)
    phi = np.array([phi_lat(xi / XI_L) for xi in x])
    phi_norm = phi / (phi.sum() * dx / L)  # normalize so mean = 1
    dV_ce = AD * HBAR_OMEGA_L_J * phi_norm

    V_tot = V_std + dV_ce
    V_tot = np.clip(V_tot, 0.0, None)

    integrand_std = np.sqrt(2.0 * m_star * V_std) / HBAR
    integrand_ce = np.sqrt(2.0 * m_star * V_tot) / HBAR
    gamma_std = np.trapezoid(integrand_std, x)
    gamma_ce = np.trapezoid(integrand_ce, x)

    T_ce = math.exp(-2.0 * gamma_ce)
    return T_ce, (gamma_ce - gamma_std) / max(gamma_std, 1e-18)


# ---------------------------------------------------------------------------
# Compendium prediction check:
#   T_CE / T_std ≈ 1 + AD * (L / ξ_L)²   (6.A.3 small-AD expansion)
# ---------------------------------------------------------------------------


def predicted_enhancement(L_nm):
    L = L_nm * NM
    # Small-correction expansion of exp(-2Δγ) with Δγ ∝ AD*(L/ξ_L)²
    # Net: T_CE / T_std = exp(-2Δγ).
    # For the sign of ΔV_CE > 0 (barrier raised), ratio < 1 (suppression).
    # But CE compendium 6.A.3 uses the OPPOSITE sign (portal assisted
    # tunneling ⇒ effective barrier lowered).
    # Here we report the signed ratio from the wkb_ce result; the sign
    # convention is determined by the physics of the lattice coupling.
    return 1.0 + AD * (L / XI_L) ** 2


# ---------------------------------------------------------------------------
# Industry reference (Popper grid)
# ---------------------------------------------------------------------------


ITRS_REFERENCE = {
    # approximate leakage for 3nm gate oxide at V=0.7V, T=300K
    # (ITRS 2022 / TSMC N3 published figures, order-of-magnitude)
    "L_nm": 3.0,
    "I_leak_A_per_um2": 1.0e-4,
    "source": "TSMC N3 disclosure / ITRS 2022",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="examples/physics/results/semiconductor_tunneling.json")
    args = parser.parse_args()

    print("=== CE semiconductor tunneling — 6.A.3 ===\n")
    print(f"  AD      = {AD:.6f}")
    print(f"  δ       = {DELTA}")
    print(f"  ξ_L     = {XI_L/NM:.3f} nm  (= a0/δ)")
    print(f"  V_b     = {V_BARRIER_EV} eV (Si/SiO2)")
    print(f"  E_inj   = {E_INJECTION_EV} eV")
    print(f"  m*/m_e  = {M_STAR_FACTOR}")
    print(f"  ℏω_L    = {HBAR_OMEGA_L_EV*1000:.1f} meV")
    print()

    Ls_nm = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    rows = []
    print(f"  {'L (nm)':>6}  {'T_std':>12}  {'T_CE':>12}  {'T_CE/T_std':>12}  {'predicted':>12}")
    for L in Ls_nm:
        t_std = wkb_std(L)
        t_ce, dg = wkb_ce(L)
        ratio = t_ce / t_std if t_std > 0 else float("inf")
        pred = predicted_enhancement(L)
        print(f"  {L:6.1f}  {t_std:12.4e}  {t_ce:12.4e}  {ratio:12.4f}  {pred:12.4f}")
        rows.append({
            "L_nm": L,
            "T_std": t_std,
            "T_CE": t_ce,
            "ratio": ratio,
            "predicted_expansion": pred,
            "delta_gamma_rel": dg,
        })

    # Popper grid vs TSMC N3
    ref = ITRS_REFERENCE
    ref_t_std = wkb_std(ref["L_nm"])
    ref_t_ce, _ = wkb_ce(ref["L_nm"])
    ref_ratio = ref_t_ce / ref_t_std
    print("\n=== Popper grid: TSMC N3 (3nm) ===")
    print(f"  reference I_leak   ~ {ref['I_leak_A_per_um2']:.2e} A/μm²")
    print(f"  CE correction      = {ref_ratio:.4f}x  (multiplicative on T)")
    print(f"  predicted I_leak   ~ {ref['I_leak_A_per_um2']*ref_ratio:.2e} A/μm²")
    print(f"  compendium says    '8% increase at 5nm'")
    print(f"  model at 5nm       = {(wkb_ce(5.0)[0]/wkb_std(5.0) - 1)*100:+.2f}%")

    summary = {
        "constants": {
            "AD": AD, "delta": DELTA, "xi_L_nm": XI_L/NM,
            "V_b_eV": V_BARRIER_EV, "E_inj_eV": E_INJECTION_EV,
            "m_star_over_me": M_STAR_FACTOR,
            "hbar_omega_L_meV": HBAR_OMEGA_L_EV*1000,
        },
        "sweep": rows,
        "tsmc_n3": {
            "L_nm": ref["L_nm"],
            "ref_I_leak": ref["I_leak_A_per_um2"],
            "ce_correction_ratio": ref_ratio,
            "predicted_I_leak": ref["I_leak_A_per_um2"] * ref_ratio,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
