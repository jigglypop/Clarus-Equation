"""Higgs 질량과 우주 구성 — M_H/M_Z − 1 = Ω_DM/Ω_Λ. 원장: 43장 §43.59. 예측값을 바꾸지 않는다.

코어에서 M_H/M_Z = F = 1 + α_s D이고 가지치기(BR3)에서 Ω_DM/Ω_Λ = α_s D다. 두 식을 합치면 α_s가 사라진다.
계산 전에 적은 판본과 kill:

관계 H. M_H/M_Z − 1 = Ω_DM/Ω_Λ (Ω_DM은 오늘의 비바리온 물질 전체, 중성미자 포함).
예측. 측정한 r = M_H/M_Z − 1로 Ω_m(참) = (q + r)/(1 + r). q는 코어(α_s)에서, 영향은 작다.
판독(사상 M). CMB 단독 ΛCDM은 탄생 틀 = 참값을 읽는다. BAO+BBN의 ΛCDM 판독은 G1m 때문에 코어에서 확인한
   이동 ΔΩ_m(§43.49: 0.3080 → 0.2981)만큼 낮다.
kill. CMB 단독 Ω_m 또는 BAO+BBN Ω_m이 해당 예측에서 3σ 넘게 벗어남.

python -B -m examples.physics.rendering.ce_rendering_higgs_cosmos
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_spread as SP

MH_MZ = (125.20 / 91.1876, 0.11 / 91.1876)
OBS = {"Planck 2018 (CMB+lensing)": ("cmb", 0.3153, 0.0073), "DESI DR2 BAO+BBN": ("bao", 0.2975, 0.0086),
       "DESI DR2 BAO + CMB (mixed)": ("mixed", 0.3027, 0.0036)}


def identity() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    x = c["a"] * c["D"]
    q = c["q"]
    om_dm = (1 - q) * x / (1 + x)
    om_l = (1 - q) / (1 + x)
    return {"F_minus_1": c["F"] - 1, "OmDM_over_OmL": om_dm / om_l, "alpha_s_D": x}


def from_higgs() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    r, sr = MH_MZ[0] - 1, MH_MZ[1]
    q = c["q"]
    om = (q + r) / (1 + r)
    s_om = sr * (1 - q) / (1 + r) ** 2
    return {"r_obs": r, "r_sigma": sr, "Om_true_from_Higgs": om, "Om_sigma": s_om, "Om_core": c["Om"],
            "F_pull": (c["F"] - MH_MZ[0]) / MH_MZ[1]}


def bao_bias() -> float:
    a = SP.conjecture_a()
    return a["CE_with_G1m"]["Om"] - R.core(R.calibrated_alpha_s()[0])["Om"]


def comparisons() -> dict:
    h = from_higgs()
    bias = bao_bias()
    pred = {"cmb": h["Om_true_from_Higgs"], "bao": h["Om_true_from_Higgs"] + bias,
            "mixed": h["Om_true_from_Higgs"] + bias / 2}
    out = {"bias_bao": bias}
    for name, (kind, v, s) in OBS.items():
        out[name] = {"pred": pred[kind], "obs": v, "pull": (pred[kind] - v) / s}
    return out


def main() -> None:
    print("identity:", {k: round(v, 6) for k, v in identity().items()})
    print("from Higgs:", {k: round(v, 5) for k, v in from_higgs().items()})
    for k, v in comparisons().items():
        print(k, v if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
