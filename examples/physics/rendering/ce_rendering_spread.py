"""무리 안의 미세한 차이 — 추측 A·B·C의 검증. 원장: 43장 §43.49. 예측값을 바꾸지 않는다.

계산 전에 적은 추측과 kill:

A (이른 기록 무리). CMB만(67.2–67.4)보다 BAO를 섞은 H₀(68.2–68.5)가 높은 것은 G1m 때문이다: BAO는 z의 은하 기록이라
   ψ(z)/2의 기울기를 물질 무게만큼 받는다(사상 M). 검증: 같은 코드의 평탄 ΛCDM + BBN 분석이 실제 DESI DR2에서
   DESI 보고값(68.51 ± 0.58)을 재현하는지 먼저 본 뒤, CE가 예측한 BAO(G1m 포함)를 가짜 자료로 같은 분석을 한다.
   kill: 추론 H₀가 CE 나이테 값(67.77)에서 DESI 쪽으로 움직이지 않음. 대조: G1m을 뺀 CE BAO.
B (현재 막대 무리). 교정원이 기록된 시기의 기울기 ψ(t_f)/2로 H₀ = 나이테/cos(ψ(t_f)/2).
   나이 고정: 세페이드 0.1 Gyr, JAGB 1.0 Gyr, TRGB 10 Gyr. kill: CCHP 방법별 값에서 평탄 M(모두 73.36)보다 χ²가 크거나
   순서(세페이드 > JAGB > TRGB)가 틀림.
C (중력 판독). 렌즈·사이렌은 무게 평균 투영 1 − Ω_m(1 − cos π/8)을 읽어 약 69.4. 현재 자료로는 판정만 보고.

python -B -m examples.physics.rendering.ce_rendering_spread
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

BBN_WB = (0.02218, 0.00055)          # DESI DR2 BAO+BBN 분석의 BBN 사전
OMEGA_NU = 0.06 / 93.14              # 표준 분석의 최소 ν 질량
DESI_BBN_H0 = (68.51, 0.58)
CCHP_METHODS = {"Cepheid": (72.05, math.hypot(1.86, 3.10), 0.1), "JAGB": (67.96, math.hypot(1.85, 1.90), 1.0),
                "TRGB": (69.85, math.hypot(1.75, 1.54), 10.0)}       # (H0, σ, 교정원 나이 Gyr)
TDCOSMO = (71.6, 3.9, 3.3)


def lcdm_bbn_fit(y: np.ndarray) -> dict:
    """평탄 ΛCDM + BBN ω_b 사전으로 BAO 벡터 y를 맞춘 (Ω_m, h, ω_b)와 h의 1σ(곡률 근사)."""
    cinv = R.BAO_CINV

    def chi2(p):
        om, h, wb = p
        wc = om * h * h - wb - OMEGA_NU
        if wc <= 0 or not 0.4 < h < 0.9:
            return 1e9
        pred = GD.C_KM_S / (100 * h * BR.r_drag(wb, wc, h)) * VT.bao_vectors(om, VT.density(None, None))
        r = pred - y
        return float(r @ cinv @ r) + ((wb - BBN_WB[0]) / BBN_WB[1]) ** 2

    best = minimize(chi2, x0=[0.30, 0.68, BBN_WB[0]], method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
    om, h, wb = best.x
    # h의 1σ: 나머지를 다시 최적화하는 Δχ² = 1 프로파일
    def prof(hh):
        sub = minimize(lambda q: chi2([q[0], hh, q[1]]), x0=[om, wb], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10})
        return sub.fun - best.fun - 1
    from scipy.optimize import brentq
    hi = brentq(prof, h, h + 0.05)
    lo = brentq(prof, h - 0.05, h)
    return {"Om": om, "H0": 100 * h, "H0_sigma": 50 * (hi - lo), "wb": wb, "chi2": best.fun}


def ce_bao_vector(with_gradient: bool = True) -> np.ndarray:
    c = R.core(R.calibrated_alpha_s()[0])
    rd, h = NL.rd_and_h(c)
    b = VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU))
    if with_gradient:
        b = b * GD.factor("G1m", R.BAO_Z, c["Om"])
    return GD.C_KM_S / (100 * h * rd) * b


def conjecture_a() -> dict:
    real = lcdm_bbn_fit(R.BAO_Y)
    ce_g = lcdm_bbn_fit(ce_bao_vector(True))
    ce_0 = lcdm_bbn_fit(ce_bao_vector(False))
    shift = ce_g["H0"] - ce_0["H0"]
    return {"real_DESI": real, "CE_with_G1m": ce_g, "CE_no_gradient": ce_0, "G1m_shift": shift,
            "calibration_offset": real["H0"] - DESI_BBN_H0[0],
            "CE_pred_vs_DESI_sigma": (ce_g["H0"] - DESI_BBN_H0[0]) / DESI_BBN_H0[1]}


def conjecture_b() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    ring = 100 * NL.early_densities(c)[2]
    om = c["Om"]
    hl_t0 = 2 / 3 * math.asinh(math.sqrt((1 - om) / om))
    t_lambda = 977.79 / (100 * NL.early_densities(c)[2] * math.sqrt(1 - om))
    t0 = hl_t0 * t_lambda
    pred_b = {k: ring / math.cos((t0 - age) / t_lambda / 2) for k, (_, _, age) in CCHP_METHODS.items()}
    flat = ring / math.cos(math.pi / 8)
    chi_b = sum(((pred_b[k] - v) / s) ** 2 for k, (v, s, _) in CCHP_METHODS.items())
    chi_m = sum(((flat - v) / s) ** 2 for _, (v, s, _) in CCHP_METHODS.items())
    order_pred = sorted(pred_b, key=pred_b.get, reverse=True)
    order_obs = sorted(CCHP_METHODS, key=lambda k: CCHP_METHODS[k][0], reverse=True)
    return {"pred_B": pred_b, "flat_M": flat, "chi2_B": chi_b, "chi2_flat_M": chi_m,
            "order_pred": order_pred, "order_obs": order_obs}


def conjecture_c() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    ring = 100 * NL.early_densities(c)[2]
    val = ring / (1 - c["Om"] * (1 - math.cos(math.pi / 8)))
    flat = ring / math.cos(math.pi / 8)
    v, up, dn = TDCOSMO
    pull = lambda p: (p - v) / (up if p > v else dn)
    return {"C_weighted": val, "flat_M": flat, "TDCOSMO_pull_C": pull(val), "TDCOSMO_pull_M": pull(flat)}


def main() -> None:
    a = conjecture_a()
    for k in ("real_DESI", "CE_with_G1m", "CE_no_gradient"):
        print(f"A {k:15s}", {kk: round(vv, 4) for kk, vv in a[k].items()})
    print("A G1m shift:", round(a["G1m_shift"], 3), "| calibration offset vs DESI:", round(a["calibration_offset"], 3),
          "| CE pred vs DESI BAO+BBN:", round(a["CE_pred_vs_DESI_sigma"], 2), "sigma")
    b = conjecture_b()
    print("B:", {k: (round(v, 2) if isinstance(v, float) else ({kk: round(vv, 2) for kk, vv in v.items()} if isinstance(v, dict) else v))
                 for k, v in b.items()})
    print("C:", {k: round(v, 2) for k, v in conjecture_c().items()})


if __name__ == "__main__":
    main()
