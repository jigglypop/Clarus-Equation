"""P41 — CE 판독(G1m)을 넣은 CMB + BAO의 중성미자 질량 합 추론. 원장: 43장 §43.75. 예측값을 바꾸지 않는다.

계산 전에 적은 규칙:
모형. 평탄 ΛCDM + Σm_ν(세 종 같은 질량, FD 정확 처리). 매개변수 (ω_b, ω_c, h, Σ).
CMB(압축). Planck 2018 ω_b = 0.02237 ± 0.00015, ω_c = 0.1200 ± 0.0012, 100θ*: 같은 코드의 Planck 최적값(질량 0,0,60 meV)을
   관측값으로 두어 코드 편차를 없애고 σ = DV.THETA_SIGMA. 상관·렌즈는 넣지 않음(절대 상한은 느슨해짐).
BAO. DESI DR2 벡터와 공분산. 예측 = c/(100 h r_d) × BAO 모양(Ω_m, Λ) × s(z).
   (a) 표준 s = 1. (b) CE 판독 s = G1m 인자(1 − Ω_m(1 − cos ψ/2), 모형의 Ω_m에서).
통계. Σ ≥ 0 평평 사전분포, Σ마다 (ω_b, ω_c, h)를 최적화한 프로파일로 95% 상한 U.
P41 판정(보정된 이동). 52 + (U_CE − U_std) ≥ 59 meV이면 통과, < 55이면 기각, 사이는 경계. 절대 U_CE도 보고.
   U_std와 공개 DESI + Planck 64 meV의 비교는 교정 점검이다.

python -B -m examples.physics.rendering.ce_rendering_nu_fit
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_theta_nu as TN
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

WB, WC = (0.02237, 0.00015), (0.1200, 0.0012)
SUM_GRID_EV = tuple(np.round(np.arange(0.0, 0.3001, 0.02), 4))
PUBLISHED = {"LCDM prior>=0 (CMB-SPA+DESI DR2+DESY5)": 52.0, "DESI DR2 + Planck PR4": 64.2}


def _flat(a: float) -> float:
    return 1.0


@lru_cache(maxsize=1)
def _theta_obs() -> float:
    return TN.planck_reference()


def _chi2(p, s_ev: float, readout: str) -> float:
    wb, wc, h = p
    if not (0.018 < wb < 0.027 and 0.09 < wc < 0.15 and 0.55 < h < 0.80):
        return 1e9
    masses = (s_ev / 3,) * 3
    th = TN.theta_star_100(wb, wc, h, masses)
    om = (wb + wc + s_ev / 93.14) / h ** 2
    shape = VT.bao_vectors(om, _flat)
    if readout == "CE":
        shape = shape * GD.factor("G1m", R.BAO_Z, om)
    pred = GD.C_KM_S / (100 * h * BR.r_drag(wb, wc, h)) * shape
    r = pred - R.BAO_Y
    return (float(r @ R.BAO_CINV @ r) + ((wb - WB[0]) / WB[1]) ** 2 + ((wc - WC[0]) / WC[1]) ** 2
            + ((th - _theta_obs()) / DV.THETA_SIGMA) ** 2)


@lru_cache(maxsize=4)
def profile(readout: str) -> dict:
    out, x0 = {}, np.array([WB[0], WC[0], 0.675])
    for s in SUM_GRID_EV:
        res = minimize(_chi2, x0=x0, args=(float(s), readout), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-6, "maxiter": 1500})
        out[float(s)] = {"chi2": float(res.fun), "wb": res.x[0], "wc": res.x[1], "h": res.x[2]}
        x0 = res.x
    return out


def upper_limit(readout: str, cl: float = 0.95) -> dict:
    prof = profile(readout)
    s = np.array(sorted(prof))
    chi = np.array([prof[k]["chi2"] for k in s])
    fine = np.linspace(s[0], s[-1], 3001)
    post = np.exp(-(np.interp(fine, s, chi) - chi.min()) / 2)
    cdf = np.cumsum(post) / post.sum()
    u = float(np.interp(cl, cdf, fine))
    return {"U_meV": 1000 * u, "chi2_min": float(chi.min()), "best_sum_meV": 1000 * float(s[chi.argmin()]),
            "d_chi2_at_59": float(np.interp(0.059, s, chi) - chi.min())}


def verdict() -> dict:
    std, ce = upper_limit("std"), upper_limit("CE")
    shift = ce["U_meV"] - std["U_meV"]
    equiv = PUBLISHED["LCDM prior>=0 (CMB-SPA+DESI DR2+DESY5)"] + shift
    status = "pass" if equiv >= 59.0 else ("killed" if equiv < 55.0 else "borderline")
    return {"std": std, "CE": ce, "shift_meV": shift, "published_equivalent_meV": equiv, "P41": status,
            "absolute_CE_allows_59": ce["U_meV"] >= 59.0,
            "calibration_std_vs_DESI_Planck": std["U_meV"] - PUBLISHED["DESI DR2 + Planck PR4"]}


def quick_check() -> dict:
    """테스트용: Σ = 0과 59 meV 두 점의 프로파일 Δχ²(표준, CE)."""
    out = {}
    for readout in ("std", "CE"):
        vals = []
        for s in (0.0, 0.059):
            res = minimize(_chi2, x0=np.array([WB[0], WC[0], 0.675]), args=(s, readout), method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-6, "maxiter": 1500})
            vals.append(float(res.fun))
        out[readout] = {"chi2_0": vals[0], "d_chi2_59": vals[1] - vals[0]}
    return out


def main() -> None:
    v = verdict()
    for k, val in v.items():
        print(k, val)


if __name__ == "__main__":
    main()
