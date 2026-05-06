"""Gate B: dark-matter identity audit for CE cosmology.

The purpose of this card is to separate the closed density prediction from the
still-open identity question.  CE fixes the dark-sector split through the
self-recursive fixed point and the QCD feedback ratio R.  Whether that
fluctuation component should be represented by a local particle is a separate
bridge question.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
ALPHA_EM = 1.0 / 129.0
ALPHA_TOTAL = 1.0 / (2.0 * math.pi)
D = 3.0
M_PROTON_MEV = 938.2720813
HBAR_C_MEV_FM = 197.3269804
V_HIGGS_GEV = 246.21965


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def rel_error(pred: float, obs: float) -> float:
    return 100.0 * (pred / obs - 1.0)


def sigma_off(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D + delta
    x = bootstrap_x(d_eff)
    sigma_dark = 1.0 - x

    alpha_w = ALPHA_EM / sin2_theta_w
    alpha_1 = ALPHA_EM / (1.0 - sin2_theta_w)
    fb_u1 = x * alpha_1 / ALPHA_TOTAL
    fb_su2 = x * alpha_w / ALPHA_TOTAL
    fb_su3 = x * ALPHA_S / ALPHA_TOTAL
    fb_delta = x * delta
    r_lo = ALPHA_S * d_eff
    r_closed = ALPHA_S * d_eff * (1.0 + x * delta)
    r_three_layer = ALPHA_S * (
        (1.0 + fb_u1)
        + (1.0 + fb_su2)
        + (1.0 + fb_su3)
        + delta * (1.0 + fb_delta)
    )

    omega_l = sigma_dark / (1.0 + r_three_layer)
    omega_dm = sigma_dark * r_three_layer / (1.0 + r_three_layer)
    omega_m = x + omega_dm

    dm_share_dark = r_three_layer / (1.0 + r_three_layer)
    de_share_dark = 1.0 / (1.0 + r_three_layer)
    dm_to_b = omega_dm / x
    b_to_dm = x / omega_dm
    alpha_s_inverse_three_layer = r_three_layer / (
        (1.0 + fb_u1) + (1.0 + fb_su2) + (1.0 + fb_su3) + delta * (1.0 + fb_delta)
    )
    alpha_s_inverse_closed = r_three_layer / (d_eff * (1.0 + x * delta))

    lambda_hp_candidate = delta * delta
    scalar_bridge_mass = M_PROTON_MEV * lambda_hp_candidate
    scalar_bridge_compton_fm = HBAR_C_MEV_FM / scalar_bridge_mass
    portal_pair_scale_gev = V_HIGGS_GEV * delta

    print("# Dark Matter Identity Gate")
    print()
    print("## Closed density split")
    print()
    print(f"sin2(theta_W) = 4 alpha_s^(4/3) = {sin2_theta_w:.8f}")
    print(f"delta = sin2(theta_W)(1-sin2(theta_W)) = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"x = epsilon^2 = Omega_b = {x:.8f}")
    print(f"sigma_dark = 1 - x = {sigma_dark:.8f}")
    print(f"R_LO = alpha_s D_eff = {r_lo:.8f}")
    print(f"R_closed = alpha_s D_eff(1+x delta) = {r_closed:.8f}")
    print(f"R_3layer = {r_three_layer:.8f}")
    print(f"Omega_Lambda = sigma_dark/(1+R_3layer) = {omega_l:.8f}")
    print(f"Omega_DM = sigma_dark R_3layer/(1+R_3layer) = {omega_dm:.8f}")
    print(f"Omega_m = Omega_b + Omega_DM = {omega_m:.8f}")
    print()

    print("## Ratio tests")
    print()
    print("| quantity | CE | reference | error/pull |")
    print("|---|---:|---:|---:|")
    print(f"| Omega_DM | {omega_dm:.8f} | 0.25890000 | {rel_error(omega_dm, 0.2589):+.3f}%, {sigma_off(omega_dm, 0.2589, 0.0057):+.2f} sigma |")
    print(f"| DM/DE = R | {r_three_layer:.8f} | 0.37814371 | {rel_error(r_three_layer, 0.37814371):+.3f}% |")
    print(f"| DM/dark share | {dm_share_dark:.8f} | non-fitted | -- |")
    print(f"| DE/dark share | {de_share_dark:.8f} | non-fitted | -- |")
    print(f"| DM/baryon | {dm_to_b:.8f} | about 5.25 | structure explanation handle |")
    print(f"| baryon/DM | {b_to_dm:.8f} | 0.19042 | {rel_error(b_to_dm, 0.19042):+.3f}% |")
    print(f"| alpha_s inverse, 3-layer | {alpha_s_inverse_three_layer:.8f} | {ALPHA_S:.8f} | exact by construction |")
    print(f"| alpha_s inverse, closed approx | {alpha_s_inverse_closed:.8f} | {ALPHA_S:.8f} | {rel_error(alpha_s_inverse_closed, ALPHA_S):+.3f}% |")
    print()

    print("## Identity audit")
    print()
    print("| question | CE result | status |")
    print("|---|---|---|")
    print("| local DM particle required? | no; density split is a collective fluctuation component | closed at density level |")
    print("| electromagnetic coupling? | Phi is a gauge singlet | explains darkness |")
    print("| tree-level single-particle recoil? | zero in exact Z2 collective branch | non-detection condition |")
    print(f"| bridge scalar mass | m_phi = m_p delta^2 = {scalar_bridge_mass:.5f} MeV | Bridge, not the DM density itself |")
    print(f"| bridge Compton length | hbar c / m_phi = {scalar_bridge_compton_fm:.5f} fm | nuclear scale, not galaxy-core scale |")
    print(f"| portal pair scale | v delta = {portal_pair_scale_gev:.5f} GeV | pair-coupled scalarization scale |")
    print(f"| lambda_HP candidate | delta^2 = {lambda_hp_candidate:.8f} | Bridge/Open |")
    print()

    print("## Falsifiable handles")
    print()
    print("1. The robust prediction is R = Omega_DM/Omega_Lambda = 0.38062660.")
    print("2. The collective branch predicts no ordinary WIMP-like nuclear recoil.")
    print("3. The 29.65 MeV scalar is a bridge representation, not automatically the cosmic DM particle.")
    print("4. Its Compton length is femtometer scale, so it cannot by itself solve core/cusp as fuzzy DM.")
    print("5. Galaxy-scale effects must come through background/growth or collective condensate dynamics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
