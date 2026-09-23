"""분별의 동전 — E4는 왜 Z에서 공평한 동전 점을 지나는가. 원장: 43장 §43.60. 예측값을 바꾸지 않는다.

사용자 가설 DS: 우주는 분별(나와 타인의 구분)에서 생긴다. 계산 전에 적은 판본과 kill:

D1(대수). 전약 보손이 “나”(전자)에 붙는 방식: 광자 = 왼손·오른손 같은 부호(항등), W = 왼손만(사영, 나 ↔ 타인),
   Z = g_L P_L + g_R P_R, g_L = T3 − Q s², g_R = −Q s². E4 닻 sin θ_W = ½(s² = ¼)에서 Z가 나에게 순수 축(γ5, 선호 없는
   부호 분별)인지, 그리고 순수 축 점 s*² = T3/(2Q)가 닻과 같은 페르미온이 무엇인지 확인한다. kill 없음.
D2(판본). “동전은 운반자의 질량에서 던져진다”의 두 읽기: (가) 운반자의 결합 = MS-bar(무거운 top 떼어냄, 동결된 선택),
   (나) Z 붕괴의 기록 = 유효각. 네 판본 {MS-bar, MS-bar(ND), 유효각, on-shell}(PDG 2024 표 10.2)의 s²를
   P36(α_s 없는 관계)과 E4 α_s(세계 평균)에 넣는다. FP의 ŝ²는 별도 행.
   kill: 가장 좋은 판본보다 공동 Δχ² > 9인 판본은 기각. 동결된 MS-bar가 기각되면 P36·E4의 반례로 기록.
   민감도: 직접 측정 평균 유효각 0.23149 ± 0.00013(PDG 식 10.75).

python -B -m examples.physics.rendering.ce_rendering_distinction
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_e4_anchor as E4A
from examples.physics.rendering import ce_rendering_registry as R

FERMIONS = {"e": (-0.5, -1.0), "nu": (0.5, 0.0), "u": (0.5, 2 / 3), "d": (-0.5, -1 / 3)}  # (T3, Q)
SCHEMES = {"MS-bar": (0.23129, 0.00004), "MS-bar (ND)": (0.23147, 0.00004),
           "effective": (0.23161, 0.00004), "on-shell": (0.22348, 0.00010)}
EFFECTIVE_DIRECT = (0.23149, 0.00013)
AS_WORLD = E4A.AS_WORLD
ANCHOR_S2 = 0.25


def z_couplings(f: str, s2: float) -> dict:
    t3, q = FERMIONS[f]
    gl, gr = t3 - q * s2, -q * s2
    norm = gl * gl + gr * gr
    return {"g_L": gl, "g_R": gr, "g_V": (gl + gr) / 2, "g_A": (gl - gr) / 2, "P_L": gl * gl / norm,
            "width": norm}


def bosons_on_me(s2: float) -> dict:
    """나(전자)의 왼손·오른손 결합(부호 포함, 전체 결합 상수는 뺌)."""
    z = z_couplings("e", s2)
    return {"photon": {"g_L": -1.0, "g_R": -1.0, "P_L": 0.5, "kind": "identity (vector)"},
            "W": {"g_L": 1 / math.sqrt(2), "g_R": 0.0, "P_L": 1.0, "kind": "projection, me <-> other"},
            "Z": {"g_L": z["g_L"], "g_R": z["g_R"], "P_L": z["P_L"],
                  "kind": "pure distinction (axial)" if abs(z["g_V"]) < 1e-12 else "biased distinction"}}


def pure_axial_points() -> dict:
    """g_V = 0이 되는 s*² = T3/(2Q). 중성미자는 없다."""
    return {f: (t3 / (2 * q) if q else None) for f, (t3, q) in FERMIONS.items()}


def anchor_identity() -> dict:
    ze, zn = z_couplings("e", ANCHOR_S2), z_couplings("nu", ANCHOR_S2)
    anchor = E4A.anchor()
    return {"anchor_s2": anchor["s2"], "g_V_e": ze["g_V"], "g_L_e": ze["g_L"], "g_R_e": ze["g_R"], "P_L_e": ze["P_L"],
            "width_ratio_me_over_other": ze["width"] / zn["width"],
            "anchor_owner": [f for f, s in pure_axial_points().items() if s is not None and abs(s - anchor["s2"]) < 1e-12]}


def _p36_pull(s2: float, s2_err: float) -> float:
    """§43.58과 같은 규약: 실험 상대 오차의 제곱합."""
    row = next(r for r in R.rows("SK") if r.key == "m_mu/m_tau")
    d = s2 * (1 - s2)
    pred = s2 / 4 * (1 + d / (2 * math.pi))
    return (pred / row.obs - 1) / math.hypot(row.sig_up / row.obs, s2_err / s2)


def _e4_alpha_pull(s2: float, s2_err: float) -> float:
    a = (s2 / 4) ** 0.75
    return (a - AS_WORLD[0]) / math.hypot(AS_WORLD[1], 0.75 * a * s2_err / s2)


def scheme_test(schemes: dict | None = None) -> dict:
    fp_s2 = E4A.fixed_point(*E4A.ADOPTED)["s2"]
    out = {}
    for name, (s2, err) in (schemes or SCHEMES).items():
        p36, e4 = _p36_pull(s2, err), _e4_alpha_pull(s2, err)
        fp = (fp_s2 - s2) / err
        out[name] = {"s2": s2, "P36": p36, "E4_alpha_s": e4, "FP": fp,
                     "chi2": p36 ** 2 + e4 ** 2, "chi2_with_FP": p36 ** 2 + e4 ** 2 + fp ** 2}
    return out


def verdict(schemes: dict | None = None) -> dict:
    t = scheme_test(schemes)
    out = {}
    for key in ("chi2", "chi2_with_FP"):
        best = min(t, key=lambda k: t[k][key])
        out[key] = {"best": best, **{k: {"d_chi2": v[key] - t[best][key], "killed": v[key] - t[best][key] > 9}
                                     for k, v in t.items()}}
    return out


def sensitivity_direct() -> dict:
    return scheme_test({"MS-bar": SCHEMES["MS-bar"], "effective (direct avg)": EFFECTIVE_DIRECT})


def main() -> None:
    print("bosons on me at the anchor:", bosons_on_me(ANCHOR_S2))
    print("bosons on me at MS-bar:", bosons_on_me(SCHEMES["MS-bar"][0]))
    print("pure axial points:", pure_axial_points())
    print("anchor identity:", anchor_identity())
    for k, v in scheme_test().items():
        print(f"{k:12s}", {kk: round(vv, 3) for kk, vv in v.items()})
    print("verdict:", verdict())
    print("direct effective:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sensitivity_direct().items()})


if __name__ == "__main__":
    main()
