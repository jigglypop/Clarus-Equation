"""P41 전체 적합 — CMB 렌즈를 넣은 CE 판독의 중성미자 질량 합. 원장: 43장 §43.86. 예측값을 바꾸지 않는다.

§43.75(압축 CMB + BAO)에 CMB 렌즈를 더한다. 계산 전에 적은 규칙:
모형. 평탄 ΛCDM + Σm_ν(세 종 같은 질량, FD). 매개변수 (ω_b, ω_c, h, ln 10¹⁰A_s, Σ), n_s = 0.9649 고정.
CMB 1차. ω_b = 0.02237 ± 0.00015, ω_c = 0.1200 ± 0.0012, 100θ*(같은 코드 Planck 기준값, σ = DV.THETA_SIGMA),
   ln(10¹⁰A_s) = 3.044 ± 0.016(Planck TT,TE,EE+lowE, 렌즈 없음: 이중 계산 방지).
BAO. DESI DR2. (a) 표준, (b) CE 판독(G1m, 모형의 Ω_m).
렌즈. S8^CMBL = σ8(Ω_m/0.3)^0.25 = 0.813 ± 0.018(ACT DR6 + Planck NPIPE, Qu 외 2023, arXiv:2304.05202).
   CE 규칙(P16): 약한 렌즈 S8은 회전 불변이므로 두 판독 모두 표준값.
σ8. CAMB 2.0.4 격자(ω_c, h, Σ; ω_b = 0.02237, A_s = 2.1e-9, 같은 질량)를 보간하고 σ8 ∝ √A_s로 확장.
   격자는 ce_rendering_nu_lens_sigma8_grid.json(출처 기록)에 있어 CAMB 없이 재현된다.
통계. Σ ≥ 0 평평 사전분포, 프로파일로 95% 상한 U.
판정(§43.75와 같은 규칙). 52 + (U_CE − U_std) ≥ 59 meV이면 통과, < 55이면 기각, 사이는 경계. 절대 U도 보고.
   교정: 렌즈를 넣은 U_std와 공개 DESI DR2 + CMB(렌즈 포함) 64.2 meV 비교.

python -B -m examples.physics.rendering.ce_rendering_nu_lens
"""

from __future__ import annotations

import json
import math
import pathlib
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_theta_nu as TN
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

WB, WC, LNA = (0.02237, 0.00015), (0.1200, 0.0012), (3.044, 0.016)
S8_CMBL = (0.813, 0.018)
SUM_GRID_EV = tuple(np.round(np.arange(0.0, 0.3001, 0.02), 4))
PUBLISHED = {"LCDM prior>=0 (CMB-SPA+DESI DR2+DESY5)": 52.0, "DESI DR2 + CMB (with lensing)": 64.2}
GRID_PATH = pathlib.Path(__file__).with_name("ce_rendering_nu_lens_sigma8_grid.json")


@lru_cache(maxsize=1)
def _grid():
    g = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    interp = RegularGridInterpolator((g["omega_c"], g["h"], g["sum_mnu_eV"]), np.array(g["sigma8"]))
    return g, interp


def sigma8(wc: float, h: float, s_ev: float, ln_as: float) -> float:
    g, interp = _grid()
    a_s = math.exp(ln_as) * 1e-10
    return float(interp([[wc, h, s_ev]])[0]) * math.sqrt(a_s / g["A_s_ref"])


def _flat(a: float) -> float:
    return 1.0


@lru_cache(maxsize=1)
def _theta_obs() -> float:
    return TN.planck_reference()


def _chi2(p, s_ev: float, readout: str) -> float:
    wb, wc, h, ln_as = p
    g, _ = _grid()
    if not (0.018 < wb < 0.027 and g["omega_c"][0] <= wc <= g["omega_c"][-1] and g["h"][0] <= h <= g["h"][-1]):
        return 1e9
    th = TN.theta_star_100(wb, wc, h, (s_ev / 3,) * 3)
    om = (wb + wc + s_ev / 93.14) / h ** 2
    shape = VT.bao_vectors(om, _flat)
    if readout == "CE":
        shape = shape * GD.factor("G1m", R.BAO_Z, om)
    r = GD.C_KM_S / (100 * h * BR.r_drag(wb, wc, h)) * shape - R.BAO_Y
    s8l = sigma8(wc, h, s_ev, ln_as) * (om / 0.3) ** 0.25
    return (float(r @ R.BAO_CINV @ r) + ((wb - WB[0]) / WB[1]) ** 2 + ((wc - WC[0]) / WC[1]) ** 2
            + ((ln_as - LNA[0]) / LNA[1]) ** 2 + ((th - _theta_obs()) / DV.THETA_SIGMA) ** 2
            + ((s8l - S8_CMBL[0]) / S8_CMBL[1]) ** 2)


def _fit(s_ev: float, readout: str, x0=None):
    x0 = np.array([WB[0], WC[0], 0.675, LNA[0]]) if x0 is None else x0
    return minimize(_chi2, x0=x0, args=(s_ev, readout), method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-6, "maxiter": 3000})


@lru_cache(maxsize=4)
def profile(readout: str) -> dict:
    out, x0 = {}, None
    for s in SUM_GRID_EV:
        res = _fit(float(s), readout, x0)
        out[float(s)] = float(res.fun)
        x0 = res.x
    return out


def upper_limit(readout: str, cl: float = 0.95) -> dict:
    prof = profile(readout)
    s = np.array(sorted(prof))
    chi = np.array([prof[k] for k in s])
    fine = np.linspace(s[0], s[-1], 3001)
    post = np.exp(-(np.interp(fine, s, chi) - chi.min()) / 2)
    cdf = np.cumsum(post) / post.sum()
    return {"U_meV": 1000 * float(np.interp(cl, cdf, fine)), "chi2_min": float(chi.min()),
            "best_sum_meV": 1000 * float(s[chi.argmin()]), "d_chi2_at_59": float(np.interp(0.059, s, chi) - chi.min())}


def verdict() -> dict:
    std, ce = upper_limit("std"), upper_limit("CE")
    shift = ce["U_meV"] - std["U_meV"]
    equiv = PUBLISHED["LCDM prior>=0 (CMB-SPA+DESI DR2+DESY5)"] + shift
    status = "pass" if equiv >= 59.0 else ("killed" if equiv < 55.0 else "borderline")
    return {"std": std, "CE": ce, "shift_meV": shift, "published_equivalent_meV": equiv, "P41": status,
            "absolute_CE_allows_59": ce["U_meV"] >= 59.0,
            "calibration_std_vs_published_with_lensing": std["U_meV"] - PUBLISHED["DESI DR2 + CMB (with lensing)"]}


def quick_check() -> dict:
    out = {}
    for readout in ("std", "CE"):
        c0 = _fit(0.0, readout).fun
        c59 = _fit(0.059, readout).fun
        out[readout] = {"chi2_0": float(c0), "d_chi2_59": float(c59 - c0)}
    return out


def main() -> None:
    for k, v in verdict().items():
        print(k, v)


if __name__ == "__main__":
    main()
