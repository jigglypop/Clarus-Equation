"""CCHP JWST 단독 값의 해부. 원장: 43장 §43.69. 예측값을 바꾸지 않는다.

CCHP JWST 단독 TRGB 68.81 ± 1.79 ± 1.32, JAGB 67.80 ± 2.17 ± 1.64(arXiv:2408.06153 v3)는 73.36에서 약 −2σ다.
M 아래 현재 막대 사다리는 물리적으로 0.12 이상 다를 수 없다(P37). 계산 전에 적은 규칙(표를 짜며 대략의 수를 암산으로 봄):

(가) P37: 같은 은하의 JAGB − 세페이드 거리 −0.03 ± 0.02(stat) ± 0.05(sys) mag(Li+ 2025, arXiv:2502.05259)를
    P37의 0(한계 0.0035 mag)과 비교. kill 3σ. H0로는 Δln H0 = −(ln 10/5) Δμ.
(나) 예산(시험 아님): 부족분(73.36 − 값)을 문헌의 분석 효과와 나란히 둔다. 표본 선택 약 2.5(Riess+ 2024,
    arXiv:2408.11770, 오차 미기재 → ±1.0로 둠), 옛 SNooPy +1.74(H0DN 표 7, F25 표본). 이 부분 표본에서 검증된 값이 아니다.
(다) JWST 단독 네 값(CCHP TRGB·JAGB, SH0ES JWST 합본 72.6 ± 2.0, JAGB 2.0 73.3 ± 1.4 ± 2.0)을 독립으로 두고
    가중 평균과 73.36 대비 긴장을 보고한다. 상관은 모형화하지 않았다(같은 팀 두 값은 SN을 공유).
기존 kill(P32): 현재 막대 방법이 σ ≤ 0.8로 ≤ 70.0에 모이면 M 기각.

python -B -m examples.physics.rendering.ce_rendering_jwst_cchp
"""

from __future__ import annotations

import math

M_LATE = 73.36
JWST_ONLY = {"CCHP TRGB JWST-only": (68.81, math.hypot(1.79, 1.32)),
             "CCHP JAGB JWST-only": (67.80, math.hypot(2.17, 1.64)),
             "SH0ES JWST combined (Riess+2024)": (72.6, 2.0),
             "SH0ES JAGB 2.0 (Li+2025)": (73.3, math.hypot(1.4, 2.0))}
JAGB_MINUS_CEPHEID = (-0.03, math.hypot(0.02, 0.05))
P37_BOUND_KMS = 0.12
SUBSAMPLE_SHIFT = (2.5, 1.0)
FITTER_SHIFT = 1.74
NGC4258_JAGB_FIELD = (0.11, 0.02)
DLN_PER_MAG = math.log(10) / 5


def p37_host_distances() -> dict:
    dmu, s = JAGB_MINUS_CEPHEID
    bound_mag = P37_BOUND_KMS / M_LATE / DLN_PER_MAG
    return {"delta_mu": dmu, "sigma": s, "pull_vs_zero": dmu / s, "delta_H0_kms": -DLN_PER_MAG * dmu * M_LATE,
            "sigma_H0_kms": DLN_PER_MAG * s * M_LATE, "P37_bound_mag": bound_mag, "killed": abs(dmu / s) >= 3}


def budget() -> dict:
    out = {}
    for k in ("CCHP TRGB JWST-only", "CCHP JAGB JWST-only"):
        v, s = JWST_ONLY[k]
        shifted = v + SUBSAMPLE_SHIFT[0] + FITTER_SHIFT
        out[k] = {"gap": M_LATE - v, "documented_effects": SUBSAMPLE_SHIFT[0] + FITTER_SHIFT,
                  "residual_pull": (shifted - M_LATE) / math.hypot(s, SUBSAMPLE_SHIFT[1])}
    out["NGC4258 JAGB field-to-field (max H0 shift, kms)"] = DLN_PER_MAG * NGC4258_JAGB_FIELD[0] * M_LATE
    return out


def aggregate() -> dict:
    w = {k: 1 / s ** 2 for k, (v, s) in JWST_ONLY.items()}
    mean = sum(JWST_ONLY[k][0] * w[k] for k in w) / sum(w.values())
    sig = sum(w.values()) ** -0.5
    chi2_m = sum(((v - M_LATE) / s) ** 2 for v, s in JWST_ONLY.values())
    chi2_best = sum(((v - mean) / s) ** 2 for v, s in JWST_ONLY.values())
    pulls = {k: (v - M_LATE) / s for k, (v, s) in JWST_ONLY.items()}
    return {"mean": mean, "sigma": sig, "pull_mean_vs_M": (mean - M_LATE) / sig, "chi2_M": chi2_m,
            "chi2_best": chi2_best, "pulls": pulls, "P32_kill": mean <= 70.0 and sig <= 0.8}


def main() -> None:
    print("(a) P37:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p37_host_distances().items()})
    print("(b) budget:", budget())
    a = aggregate()
    print("(c) aggregate:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k != "pulls"})
    print("    pulls:", {k: round(v, 2) for k, v in a["pulls"].items()})


if __name__ == "__main__":
    main()
