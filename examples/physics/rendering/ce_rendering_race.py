"""자기와 붙잡힘의 경주 — BR3와 HB의 유도. 원장: 43장 §43.67. 예측값을 바꾸지 않는다.

BR3(§43.54)는 묶임 대 자유의 비율 α_s D : 1을 공리로 두었다. 계산 전에 적은 판본과 규칙:

RC(경주). 살아남은 계보에 두 사건이 독립 포아송 과정으로 온다: 자기 사건(빠르기 1, 자기는 하나 §43.66)과
   붙잡힘(통로 D개 × 빠르기 α_s = P(V₃), BR1). 기록은 되돌릴 수 없으므로(R1) 먼저 온 사건이 운명을 정한다:
   P(묶임) = α_s D/(1 + α_s D). 몬테카를로로 검산(kill: 3σ 밖).
MR(질량 = 빠르기). 정지 질량은 기록이 새로 쓰이는 빠르기(Mc² = ħω)다: M_Z ∝ 자기 빠르기 1, M_H ∝ 전체 빠르기 1 + α_s D.
경쟁 규칙(같은 틀): 경주(먼저 온 사건) X = m, 창(자기 단위 시간 안에 붙잡힘 하나라도) X = e^m − 1,
   재귀(붙잡힌 것도 다시 경주, 모든 후손) X = m/(1 − m). m = α_s D, X = Ω_DM/Ω_Λ = M_H/M_Z − 1.
kill. 두 독립 관측(Planck Ω_m, M_H/M_Z)의 공동 χ²에서 가장 좋은 규칙보다 Δχ² > 9.
   선형은 두 관측에 맞춰 고른 꼴이므로 경주가 이기는 것은 구성상이다. 새로운 것은 선형의 기제다.

python -B -m examples.physics.rendering.ce_rendering_race
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_registry as R

OM_PLANCK = (0.3153, 0.0073)
MH_MZ = (125.20 / 91.1876, 0.11 / 91.1876)
RULES = {"race (first event)": lambda m: m, "window (any capture in unit time)": lambda m: math.exp(m) - 1,
         "recursive (all descendants)": lambda m: m / (1 - m)}


def _core() -> dict:
    return R.core(R.calibrated_alpha_s()[0])


def race_mc(n: int = 400000, seed: int = 11) -> dict:
    c = _core()
    m = c["a"] * c["D"]
    rng = np.random.default_rng(seed)
    t_self = rng.exponential(1.0, n)
    t_cap = rng.exponential(1.0 / m, n)
    frac = float(np.mean(t_cap < t_self))
    sig = math.sqrt(frac * (1 - frac) / n)
    window = float(np.mean(rng.poisson(m, n) >= 1))
    return {"race_fraction": frac, "race_formula": m / (1 + m), "race_pull": (frac - m / (1 + m)) / sig,
            "window_fraction": window, "window_formula": 1 - math.exp(-m)}


def rules() -> dict:
    c = _core()
    q, m = c["q"], c["a"] * c["D"]
    out = {}
    for name, f in RULES.items():
        x = f(m)
        om = q + (1 - q) * x / (1 + x)
        p_om = (om - OM_PLANCK[0]) / OM_PLANCK[1]
        p_h = (1 + x - MH_MZ[0]) / MH_MZ[1]
        out[name] = {"X": x, "Omega_m": om, "M_H": 91.1876 * (1 + x), "pull_Om": p_om, "pull_MH": p_h,
                     "chi2": p_om ** 2 + p_h ** 2}
    best = min(out, key=lambda k: out[k]["chi2"])
    for v in out.values():
        v["d_chi2"] = v["chi2"] - out[best]["chi2"]
        v["killed"] = v["d_chi2"] > 9
    out["best"] = best
    return out


def self_first() -> dict:
    """MR: M_Z/M_H = 자기가 먼저 움직일 확률 1/(1 + m)."""
    c = _core()
    m = c["a"] * c["D"]
    return {"P_self_first": 1 / (1 + m), "M_Z_over_M_H_core": 1 / c["F"], "M_Z_over_M_H_obs": 1 / MH_MZ[0]}


def main() -> None:
    print("race MC:", {k: round(v, 5) for k, v in race_mc().items()})
    r = rules()
    for k, v in r.items():
        if k != "best":
            print(f"{k:36s}", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("best:", r["best"])
    print("self first:", {k: round(v, 5) for k, v in self_first().items()})


if __name__ == "__main__":
    main()
