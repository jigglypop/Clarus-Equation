"""SH0ES 대 CCHP — 현재 막대 사다리의 해부. 원장: 43장 §43.61. 예측값을 바꾸지 않는다.

사상 M(§43.48)은 현재 막대·시계로 교정한 H0가 모두 73.36으로 모인다고 본다(P32). §43.48의 kill:
현재 막대로만 교정한 방법이 σ ≲ 0.8로 70 부근에 모이면 M은 기각. 계산 전에 적은 판본과 규칙:

R1(유도). M은 틀 사이 비에 상대 기울기의 1차 투영 cos Δθ를 준다. 열적 시간(§43.53)에서 θ = H_Λ t / 2이므로
   교정(z ≈ 0)과 허블 흐름(z ≤ 0.15) 사이 상대 기울기는 Δθ(z) = H_Λ t_lb(z) / 2. 두 현재 틀 사다리의 물리적 차이의
   상한은 1 − cos Δθ(0.15). 이것이 작으면 사다리 사이의 차이는 모두 분석·표본에서 와야 한다.
R2(가족). H0DN(A&A 2026, arXiv:2510.23823) 표 4의 기준값·직교 경로 O1·O2·Ia 없는 판본, 표 7의 CCHP 사슬 분해,
   표 5의 R22 세페이드 재현, CCHP JWST 단독 JAGB·TRGB(arXiv:2408.06153 v3), TDCOSMO 2025(채점 행).
R3(비교). 서로 독립인 O1 + O2(+ TDCOSMO)에서 세 세계: M 73.36, CCHP 세계 70.39, O1 없음 67.77.
R4(맞춘 분석). 같은 SN 피팅 코드와 표본으로 맞춘 TRGB 사다리가 σ ≤ 1.0으로 70.5 이하에 머물면 M은 곤경.
눈가림. H0DN은 사전 등록(2026-09-23) 전에 나왔지만 개발에 쓰지 않았다: 보류 자료 확인이며 예측 적중이 아니다.

python -B -m examples.physics.rendering.ce_rendering_ladder
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from examples.physics.rendering import ce_rendering_registry as R

M_LATE, M_EARLY, CCHP_WORLD = 73.36, 67.77, 70.39
GYR_PER_INV_KMS_MPC = 977.79
H0DN = {"V00 baseline": (73.499, 0.809), "O1 MW+LMC/SMC+Ceph+SNIa+FP": (73.110, 0.920),
        "O2 N4258+TRGB+SBF+masers": (73.451, 1.777), "V13 no SN Ia": (73.434, 1.795),
        "V08 no Cepheids": (72.509, 1.296), "V21 SN Ia 0.03<z<0.10": (72.667, 0.862),
        "V20 SN Ia z>0.06": (73.190, 0.862)}
CCHP_CHAIN = {"F25 published (SNooPy pre-v2.7, 24)": (70.39, 1.80),
              "F25 reproduced in H0DN": (70.31, 1.80),
              "F25 sample, SNooPy v2.7 (24)": (72.05, 1.85),
              "pre-v2.7, all available (28)": (71.40, 1.72),
              "v2.7, all TRGB calibrators (35)": (72.66, 1.64)}
SH0ES_R22_CEPHEID = (73.17, 0.96)
CCHP_JWST_ONLY = {"TRGB JWST-only": (68.81, math.hypot(1.79, 1.32)), "JAGB JWST-only": (67.80, math.hypot(2.17, 1.64))}
TDCOSMO = (71.6, 3.3, 3.9)
WORLDS = {"M (73.36)": M_LATE, "CCHP world (70.39)": CCHP_WORLD, "no O1 (67.77)": M_EARLY}


def _lookback_gyr(z: float, h0: float, om: float) -> float:
    f = lambda x: 1 / ((1 + x) * math.sqrt(om * (1 + x) ** 3 + 1 - om))
    return quad(f, 0, z)[0] * GYR_PER_INV_KMS_MPC / h0


def physical_spread(z_max: float = 0.15, z_eff: float = 0.05) -> dict:
    """R1: 교정(z≈0)과 허블 흐름 사이 상대 기울기의 투영이 주는 H0의 물리적 차이."""
    om = R.core(R.calibrated_alpha_s()[0])["Om"]
    h_lam = M_EARLY * math.sqrt(1 - om) / GYR_PER_INV_KMS_MPC           # 1/Gyr
    out = {"theta_now": h_lam * _lookback_gyr(math.inf, M_EARLY, om) / 2}   # 탄생부터의 나이로 θ0 ≈ π/8 확인
    for tag, z in (("z_max", z_max), ("z_eff", z_eff)):
        d = h_lam * _lookback_gyr(z, M_EARLY, om) / 2
        out[tag] = {"z": z, "d_theta": d, "rel": 1 - math.cos(d), "kms": M_LATE * (1 - math.cos(d))}
    return out


def _pull(v: float, s: float, pred: float = M_LATE) -> float:
    return (v - pred) / s


def _tdcosmo_pull(pred: float) -> float:
    v, lo, hi = TDCOSMO
    return (v - pred) / (hi if pred > v else lo)


def worlds() -> dict:
    """R3: 독립 경로 O1 + O2(+ TDCOSMO)에서 세 세계의 χ²."""
    o1, o2 = H0DN["O1 MW+LMC/SMC+Ceph+SNIa+FP"], H0DN["O2 N4258+TRGB+SBF+masers"]
    out = {}
    for name, h in WORLDS.items():
        c = _pull(*o1, h) ** 2 + _pull(*o2, h) ** 2
        out[name] = {"chi2_O1_O2": c, "chi2_with_TDCOSMO": c + _tdcosmo_pull(h) ** 2}
    w = 1 / o1[1] ** 2 + 1 / o2[1] ** 2
    out["O1+O2 weighted mean"] = ((o1[0] / o1[1] ** 2 + o2[0] / o2[1] ** 2) / w, w ** -0.5)
    return out


def kill_check(sigma_max: float = 0.85) -> dict:
    """§43.48 kill: σ ≲ 0.8인 현재 막대 판정이 70 부근(±1σ 안에 70.39)에 있는가."""
    precise = {k: v for k, v in {**H0DN, **CCHP_CHAIN}.items() if v[1] <= sigma_max}
    near70 = [k for k, (v, s) in precise.items() if abs(v - CCHP_WORLD) <= s]
    return {"precise": precise, "near_70": near70, "killed": bool(near70)}


def cchp_dissection() -> dict:
    """R4: CCHP 사슬을 맞춘 분석으로 옮길 때의 이동과 M 대비 pull."""
    c = CCHP_CHAIN
    rows = {k: {"H0": v, "sigma": s, "pull_M": _pull(v, s)} for k, (v, s) in c.items()}
    fitter = c["F25 sample, SNooPy v2.7 (24)"][0] - c["F25 reproduced in H0DN"][0]
    sample = c["v2.7, all TRGB calibrators (35)"][0] - c["F25 sample, SNooPy v2.7 (24)"][0]
    matched = c["v2.7, all TRGB calibrators (35)"]
    return {"rows": rows, "shift_fitter": fitter, "shift_sample": sample,
            "gap_published_vs_R22": SH0ES_R22_CEPHEID[0] - c["F25 published (SNooPy pre-v2.7, 24)"][0],
            "gap_matched_vs_R22": SH0ES_R22_CEPHEID[0] - matched[0],
            "R4_trouble": matched[0] <= 70.5 and matched[1] <= 1.0}


def table() -> dict:
    out = {k: _pull(*v) for k, v in H0DN.items()}
    out["SH0ES R22 Cepheid (H0DN emulation)"] = _pull(*SH0ES_R22_CEPHEID)
    out.update({k: _pull(*v) for k, v in CCHP_JWST_ONLY.items()})
    out["TDCOSMO 2025"] = _tdcosmo_pull(M_LATE)
    return out


def main() -> None:
    print("R1 physical spread:", physical_spread())
    print("pulls vs 73.36:", {k: round(v, 2) for k, v in table().items()})
    print("R3 worlds:", worlds())
    print("kill check:", kill_check())
    d = cchp_dissection()
    print("R4:", {k: v for k, v in d.items() if k != "rows"})
    for k, v in d["rows"].items():
        print(f"  {k:40s} {v['H0']:.2f} +- {v['sigma']:.2f}  pull {v['pull_M']:+.2f}")


if __name__ == "__main__":
    main()
