"""E4 공략 — 무차별 진공점의 닻과 α_s의 자기일관 고정점. 원장: 43장 §43.55. 예측값을 바꾸지 않는다.

E4: sin θ_W = A₂ = 2a², α_s = det R = a³. a를 “축 하나가 거짓(미렌더링)으로 남을 확률” r로 읽으면
α_s = P(V₃) = r³, sin θ_W = 2·P(V₂) = 2r². 계산 전에 적은 판본과 kill:

닻(정리). 무차별 r = ½에서 (α_s, sin θ_W) = (1/8, 1/2)이고 이것은 Bool 진공 세계 확률(§43.44)과 같다. 확인만 한다.
고정점 후보(사후 발견, 정직하게 표기). r = ½(1 − ε), ε = L·C. 고려한 가족(계산 전에 명시):
   L ∈ {α_s/2π, α_s/4π, α_s/π, α_s/16π, δ/2π, δ/8}, C ∈ {1, 1 + δ/2π, 1 − δ/2π, 1 + α_s/2π} → 24개.
   α_s = r³, ŝ² = 4r⁴, δ = ŝ²(1 − ŝ²)을 모두 r로 쓴 자기일관 방정식의 고정점. 입력 없음.
   채택 후보는 L = α_s/2π, C = 1 + δ/2π(가족을 적기 전에 찾음).
kill. 채택 후보가 ŝ²(M_Z)에서 3σ 밖이거나 α_s 세계 평균에서 2σ 밖이면 기각. 가족 안 1σ 적중 수를 보고한다.

python -B -m examples.physics.rendering.ce_rendering_e4_anchor
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_registry as R

S2_OBS = (0.23129, 0.00004)
AS_WORLD = (0.1180, 0.0009)
AS_FLAG = (0.1183, 0.0007)

L_TERMS = {"a/2pi": lambda a, d: a / (2 * math.pi), "a/4pi": lambda a, d: a / (4 * math.pi),
           "a/pi": lambda a, d: a / math.pi, "a/16pi": lambda a, d: a / (16 * math.pi),
           "d/2pi": lambda a, d: d / (2 * math.pi), "d/8": lambda a, d: d / 8}
C_TERMS = {"1": lambda a, d: 1.0, "1+d/2pi": lambda a, d: 1 + d / (2 * math.pi),
           "1-d/2pi": lambda a, d: 1 - d / (2 * math.pi), "1+a/2pi": lambda a, d: 1 + a / (2 * math.pi)}
ADOPTED = ("a/2pi", "1+d/2pi")


def anchor() -> dict:
    r = 0.5
    return {"alpha_s": r ** 3, "sin_thetaW": 2 * r ** 2, "s2": 4 * r ** 4, "P_V3": 1 / 8, "two_P_V2": 2 * (1 / 4),
            "e4_holds": abs(4 * (r ** 3) ** (4 / 3) - (2 * r ** 2) ** 2) < 1e-15}


def fixed_point(l_key: str, c_key: str) -> dict:
    def f(r):
        a = r ** 3
        s2 = 4 * r ** 4
        d = s2 * (1 - s2)
        return r - 0.5 * (1 - L_TERMS[l_key](a, d) * C_TERMS[c_key](a, d))
    r = brentq(f, 0.3, 0.5)
    a, s2 = r ** 3, 4 * r ** 4
    return {"r": r, "alpha_s": a, "s2": s2, "eps": 1 - 2 * r, "pull_s2": (s2 - S2_OBS[0]) / S2_OBS[1],
            "pull_as_world": (a - AS_WORLD[0]) / AS_WORLD[1], "pull_as_flag": (a - AS_FLAG[0]) / AS_FLAG[1]}


def family_scan() -> dict:
    rows = {f"{l}|{c}": fixed_point(l, c) for l in L_TERMS for c in C_TERMS}
    hits1 = [k for k, v in rows.items() if abs(v["pull_s2"]) <= 1]
    hits3 = [k for k, v in rows.items() if abs(v["pull_s2"]) <= 3]
    s2_vals = [v["s2"] for v in rows.values()]
    return {"rows": rows, "hits_1sigma": hits1, "hits_3sigma": hits3,
            "s2_range": (min(s2_vals), max(s2_vals)), "n": len(rows)}


def calibrated_eps() -> dict:
    """E4 보정값(ŝ²에서 역산한 α_s)의 ε와 채택 후보의 ε."""
    a_cal = R.calibrated_alpha_s()[0]
    r_cal = a_cal ** (1 / 3)
    fp = fixed_point(*ADOPTED)
    return {"alpha_s_cal": a_cal, "eps_cal": 1 - 2 * r_cal, "eps_adopted": fp["eps"], "alpha_s_fp": fp["alpha_s"],
            "rel_shift_alpha_s": fp["alpha_s"] / a_cal - 1}


def _lam(r: float, inner_exp: bool = False) -> float:
    a, s2 = r ** 3, 4 * r ** 4
    d = s2 * (1 - s2)
    return a / (2 * math.pi) * (math.exp(d / (2 * math.pi)) if inner_exp else 1 + d / (2 * math.pi))


RESUMMATIONS = {  # 사용자 가설 “e와 관련, 무한히 더해 나가는 조합”을 받아 계산 전에 적은 네 판본
    "V0 linear": lambda r: 0.5 * (1 - _lam(r)),
    "V1 exp outer": lambda r: 0.5 * math.exp(-_lam(r)),
    "V2 exp both": lambda r: 0.5 * math.exp(-_lam(r, True)),
    "V3 exp inner": lambda r: 0.5 * (1 - _lam(r, True)),
}


def resummation_scan() -> dict:
    out = {}
    for name, g in RESUMMATIONS.items():
        r = brentq(lambda x: x - g(x), 0.3, 0.5)
        a, s2 = r ** 3, 4 * r ** 4
        out[name] = {"alpha_s": a, "s2": s2, "pull_s2": (s2 - S2_OBS[0]) / S2_OBS[1],
                     "pull_as_world": (a - AS_WORLD[0]) / AS_WORLD[1]}
    return out


def nested_truncations(n_max: int = 8) -> list[dict]:
    """무차별점 r₀ = ½에서 자기 보정을 n번 겹친 값(V0). 무한히 겹친 극한이 고정점이다."""
    r, out = 0.5, []
    for n in range(1, n_max + 1):
        r = RESUMMATIONS["V0 linear"](r)
        s2 = 4 * r ** 4
        out.append({"n": n, "alpha_s": r ** 3, "s2": s2, "pull_s2": (s2 - S2_OBS[0]) / S2_OBS[1]})
    return out


def main() -> None:
    print("anchor:", anchor())
    fp = fixed_point(*ADOPTED)
    print("adopted fixed point:", {k: round(v, 6) for k, v in fp.items()})
    print("vs calibrated:", {k: round(v, 6) for k, v in calibrated_eps().items()})
    scan = family_scan()
    print(f"family n={scan['n']}, s2 range {scan['s2_range'][0]:.4f}..{scan['s2_range'][1]:.4f}")
    for k, v in sorted(scan["rows"].items(), key=lambda kv: abs(kv[1]["pull_s2"]))[:6]:
        print(f"   {k:16s} alpha_s={v['alpha_s']:.5f} s2={v['s2']:.5f} pull_s2={v['pull_s2']:+.1f} pull_as={v['pull_as_world']:+.2f}")
    print("hits within 1 sigma:", scan["hits_1sigma"], "| within 3 sigma:", scan["hits_3sigma"])
    for name, v in resummation_scan().items():
        print(f"resum {name:13s} alpha_s={v['alpha_s']:.6f} s2={v['s2']:.6f} pull_s2={v['pull_s2']:+.2f} pull_as={v['pull_as_world']:+.2f}")
    for t in nested_truncations():
        print(f"nest n={t['n']}: alpha_s={t['alpha_s']:.6f} s2={t['s2']:.6f} pull_s2={t['pull_s2']:+.2f}")


if __name__ == "__main__":
    main()
