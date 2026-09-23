"""W3 성장 보정과 W2–W3 판가름. 원장: 43장 §43.39. 잠긴 모듈은 import만 한다.

모방 작용(§43.28)에서 V는 φ만의 함수이고 φ = 상수인 면이 먼지(λ)의 정지 틀이다. 따라서 진공에서 먼지로 가는
에너지 전달 Q는 먼지 4-속도를 따라 흐르고 그 틀에서 균일하다(δQ = 0). 이것은 선택이 아니라 작용의 귀결이다.
전체 물질(처음 물질 + 생긴 먼지)의 선형 방정식(ln a 변수, Θ = θ/H):
    δ' = −Θ − Γ δ,            Γ = Q/(H ρ_m) = (dρ_d/dln a + 3ρ_d)/ρ_m
    Θ' = −(2 + ½ dln E²/dln a) Θ − (3/2) Ω_m(a) δ
Γ = 0이면 표준 성장이다. 새로 생긴 먼지는 균일하게 태어나 밀도 대비를 희석한다.

계산 전에 고정한 판정 규칙:
- θ*는 두 판본 모두 FD 통일 경로(§43.38)를 쓴다. BAO는 두 판본 모두 G1m 척도 인자를 곱하고 고정 눈금으로 채점한다.
- W3(보정)가 W2보다 43행 고정 눈금 RMSE와 판본 V(39행) RMSE를 모두 낮출 때만 W3를 현재 판정으로 올린다.
  하나라도 나쁘면 W3는 경쟁 판본으로 남고 사전 등록 v12는 그대로다.

python -B -m examples.physics.rendering.ce_rendering_w3_growth
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_mimetic_vacuum as MV
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_theta_nu as TN
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

C_KM_S = 299792.458


def growth_today(m: dict, transfer: bool = True) -> float:
    prim = m["prim"]
    dust = lambda a: float(m["dark"](a) - m["vacuum"](a))
    e2 = lambda a: prim / a ** 3 + float(m["dark"](a))
    rho_m = lambda a: prim / a ** 3 + dust(a)
    eps = 1e-5

    def dlog(f, a):
        return (math.log(f(a * (1 + eps))) - math.log(f(a * (1 - eps)))) / (math.log1p(eps) - math.log1p(-eps))

    def rhs(lna, y):
        a = math.exp(lna)
        d, th = y
        gam = 0.0
        if transfer:
            dd = (dust(a * (1 + eps)) - dust(a * (1 - eps))) / (math.log1p(eps) - math.log1p(-eps))
            gam = (dd + 3 * dust(a)) / rho_m(a)
        return [-th - gam * d, -(2 + 0.5 * dlog(e2, a)) * th - 1.5 * rho_m(a) / e2(a) * d]

    a0 = 1e-3
    return float(solve_ivp(rhs, (math.log(a0), 0.0), [a0, -a0], rtol=1e-10, atol=1e-14).y[0, -1])


def s8_corrected(m: dict) -> float:
    return MV.s8(m) * growth_today(m, True) / growth_today(m, False)


def _bao_chi2_g1m(b: np.ndarray, wb: float, wc: float, h: float, om: float) -> float:
    from examples.physics.rendering import ce_rendering_bao_ruler as BR
    rd = BR.r_drag(wb, wc, h)
    r = C_KM_S / (100 * h * rd) * b * GD.factor("G1m", R.BAO_Z, om) - R.BAO_Y
    return float(r @ R.BAO_CINV @ r)


def _score(rows: list[dict], theta_pull: float, s8v: float | None, chib: float) -> float:
    lens = {n for n, *_ in GR.LENSING_S8}
    chi = chib
    for o in rows:
        p = o["pull"]
        if o["key"] == "100 theta*":
            p = theta_pull
        elif o["key"] in lens and s8v is not None:
            p = (s8v - o["obs"]) / o["sigma"]
        chi += p * p
    return math.sqrt(chi / (len(rows) + 13))


def compare() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    tn = TN.pulls()
    out = {}
    # W2 + G1m
    wb, wc, h = NL.early_densities(c)
    f2 = VT.density(c, VT.ADOPTED_NU)
    chib2 = _bao_chi2_g1m(VT.bao_vectors(c["Om"], f2), wb, wc, h, c["Om"])
    s8_w2 = NL.s8(c) * VT.growth_today(c["Om"], f2) / VT.growth_today(c["Om"], lambda a: 1.0)
    out["W2"] = {"S8": s8_w2, "V39": _score(NL.score("IV", "fixed")["rows"], tn["W2"], None, chib2),
                 "rows43": _score(NL.score("full", "fixed")["rows"], tn["W2"], s8_w2, chib2)}
    # W3 (b) + G1m, with and without the transfer correction
    m = MV.model("b")
    chib3 = _bao_chi2_g1m(MV.bao_vectors(m), m["wb"], m["wc"], m["h"], c["Om"])
    rows_iv = MV.score("b", "fixed")["rows"]
    rows_full = MV.score("b", "fixed", rows_from="full")["rows"]
    for tag, s8v in (("W3_first_order", MV.s8(m)), ("W3_corrected", s8_corrected(m))):
        out[tag] = {"S8": s8v, "V39": _score(rows_iv, tn["W3b"], None, chib3),
                    "rows43": _score(rows_full, tn["W3b"], s8v, chib3)}
    for v in out.values():
        v["S8_DES_pull"] = (v["S8"] - 0.776) / 0.017
        v["S8_KiDS_pull"] = (v["S8"] - 0.815) / (0.016 if v["S8"] >= 0.815 else 0.021)
    w2, w3 = out["W2"], out["W3_corrected"]
    out["verdict"] = "W3 promoted" if (w3["rows43"] < w2["rows43"] and w3["V39"] < w2["V39"]) else "W3 stays competing"
    return out


def main() -> None:
    out = compare()
    for k in ("W2", "W3_first_order", "W3_corrected"):
        v = out[k]
        print(f"{k:15s} S8={v['S8']:.4f} (DES {v['S8_DES_pull']:+.2f}, KiDS {v['S8_KiDS_pull']:+.2f})  V39={v['V39']:.4f}  43rows={v['rows43']:.4f}")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
