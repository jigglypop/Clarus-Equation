"""암흑에너지 w(z) 분기 W1의 판정(기각). 원장: 43장 §43.25. 예측값을 바꾸지 않는다.

W1(저장소 식): ξ² = α_s^{2/3}, w0 = −1 + 2ξ²/(3Ω_Λ), w_a = −3(1+w0)(1−Ω_Λ), 연속 매개변수 0.
사전 규칙: ν 장부 판본 V(α_s만, BAO 눈금 고정)에서 θ*·BAO 13행·S8을 w(z)로 다시 계산해
(1) 공동 RMSE 감소 (2) 새 3σ 초과 행 없음 (3) CMB–BAO 긴장 감소를 모두 만족하면 채택한다.
DESI+CMB+SN의 w0·w_a는 BAO와 자료가 겹쳐 점수에 넣지 않고 따로 보고한다.

python -B -m examples.physics.rendering.ce_rendering_w_branch
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R

C_KM_S = 299792.458
LAMBDA = (-1.0, 0.0)


def w1(c: dict) -> tuple[float, float]:
    ol = 1 - c["Om"]
    w0 = -1 + 2 * c["a"] ** (2 / 3) / (3 * ol)
    return w0, -3 * (1 + w0) * (1 - ol)


def de_density(a: float, w0: float, wa: float) -> float:
    return a ** (-3 * (1 + w0 + wa)) * math.exp(-3 * wa * (1 - a))


def theta_star_100(wb: float, wc: float, h: float, w: tuple[float, float]) -> float:
    wm = wb + wc
    ol = 1 - (wm + DV.OMEGA_R_H2) / h ** 2
    hub = lambda z: 100 * h * math.sqrt((wm * (1 + z) ** 3 + DV.OMEGA_R_H2 * (1 + z) ** 4) / h ** 2
                                        + ol * de_density(1 / (1 + z), *w))
    zs = DV.z_star(wb, wm)
    cs = lambda z: C_KM_S / math.sqrt(3 * (1 + 3 * wb / (4 * DV.OMEGA_GAMMA_H2) / (1 + z)))
    rs = quad(lambda z: cs(z) / hub(z), zs, np.inf, limit=400)[0]
    dm = quad(lambda z: C_KM_S / hub(z), 0, zs, limit=400)[0]
    return 100 * rs / dm


def bao_vectors(om: float, w: tuple[float, float]) -> np.ndarray:
    z = np.linspace(0.0, 2.5, 25001)
    a = 1 / (1 + z)
    E = np.sqrt(om * (1 + z) ** 3 + (1 - om) * a ** (-3 * (1 + w[0] + w[1])) * np.exp(-3 * w[1] * (1 - a)))
    inv = 1 / E
    dM = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(z))])
    m = np.interp(R.BAO_Z, z, dM)
    hh = 1 / np.interp(R.BAO_Z, z, E)
    return np.array([{"dm": mi, "dh": hi, "dv": (zi * mi * mi * hi) ** (1 / 3)}[k]
                     for mi, hi, zi, k in zip(m, hh, R.BAO_Z, R.BAO_KIND)])


def growth_today(om: float, w: tuple[float, float]) -> float:
    e2 = lambda a: om / a ** 3 + (1 - om) * de_density(a, *w)

    def rhs(lna, y):
        a = math.exp(lna)
        eps = 1e-6
        dln = (math.log(e2(a * (1 + eps))) - math.log(e2(a * (1 - eps)))) / (math.log1p(eps) - math.log1p(-eps))
        return [y[1], -(2 + 0.5 * dln) * y[1] + 1.5 * om / (a ** 3 * e2(a)) * y[0]]
    return float(solve_ivp(rhs, (math.log(1e-3), 0.0), [1e-3, 1e-3], rtol=1e-10, atol=1e-14).y[0, -1])


def branch_rows(c: dict, w: tuple[float, float]) -> dict:
    """R-Pl h 고정에서 w가 바꾸는 행: θ*(σ), 고정·자유 눈금 BAO χ², CMB–BAO 긴장(σ), S8."""
    wb, wc, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b = bao_vectors(c["Om"], w)
    r = C_KM_S / (100 * h * rd) * b - R.BAO_Y
    fisher = float(b @ R.BAO_CINV @ b)
    af = float(b @ R.BAO_CINV @ R.BAO_Y) / fisher
    rf = af * b - R.BAO_Y
    h0b = C_KM_S / (af * rd)
    return {"theta_pull": (theta_star_100(wb, wc, h, w) - DV.theta_star_100(*DV.PLANCK_BEST)) / DV.THETA_SIGMA,
            "bao_chi2_fixed": float(r @ R.BAO_CINV @ r), "bao_chi2_free": float(rf @ R.BAO_CINV @ rf),
            "cmb_bao_tension": (h0b - 100 * h) * math.sqrt(fisher) / h0b * af,
            "S8": NL.s8(c) * growth_today(c["Om"], w) / growth_today(c["Om"], LAMBDA)}


def joint_rmse(c: dict, w: tuple[float, float], rows_from: str = "IV") -> tuple[float, int]:
    br = branch_rows(c, w)
    rows = [dict(o) for o in NL.score(rows_from, "fixed")["rows"]]
    for o in rows:
        if o["key"] == "100 theta*":
            o["pull"] = br["theta_pull"] * DV.THETA_SIGMA / o["sigma"]
        if o["key"] in {n for n, *_ in GR.LENSING_S8}:
            o["pull"] = (br["S8"] - o["obs"]) / o["sigma"]
    n = len(rows) + 13
    return math.sqrt((sum(o["pull"] ** 2 for o in rows) + br["bao_chi2_fixed"]) / n), n


def theta_fitted_h(c: dict, w: tuple[float, float]) -> dict:
    """진단: h를 θ*에 맞추는 판본(연속 적합 1개)."""
    nu, om, q = NL.omega_nu_h2(c), c["Om"], c["q"]
    ref = DV.theta_star_100(*DV.PLANCK_BEST)
    h = brentq(lambda x: theta_star_100(q * x * x, (om - q) * x * x - nu, x, w) - ref, 0.5, 0.9, xtol=1e-10)
    wb, wc = q * h * h, (om - q) * h * h - nu
    rd = BR.r_drag(wb, wc, h)
    r = C_KM_S / (100 * h * rd) * bao_vectors(om, w) - R.BAO_Y
    hd = 100 * h / math.cos(R.TIME_AXIS_TILT)
    return {"h": h, "H0_direct": hd, "SH0ES_pull": (hd - 73.17) / 0.86, "bao_chi2_fixed": float(r @ R.BAO_CINV @ r),
            "omega_b_pull": (wb - DV.OMEGA_B_H2_OBS) / DV.OMEGA_B_H2_ERR,
            "omega_c_pull": (wc - DV.OMEGA_C_H2_OBS) / DV.OMEGA_C_H2_ERR}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    w = w1(c)
    print(f"W1: w0 = {w[0]:.4f}, wa = {w[1]:.4f}, w(z->inf) = {w[0] + w[1]:.4f} (no -1 crossing)")
    for tag, ww in (("Lambda", LAMBDA), ("W1", w)):
        br = branch_rows(c, ww)
        print(tag, {k: round(v, 3) for k, v in br.items()},
              "V39 = %.3f" % joint_rmse(c, ww)[0], "F43 = %.3f" % joint_rmse(c, ww, "full")[0])
        print("   theta-fitted h:", {k: round(v, 3) for k, v in theta_fitted_h(c, ww).items()})


if __name__ == "__main__":
    main()
