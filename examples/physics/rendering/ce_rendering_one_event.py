"""한 동전, 한 사건 — 운반자 원리(E5)의 정리. 원장: 43장 §43.62. 예측값을 바꾸지 않는다.

동전 r은 α_s^{1/3} = (ŝ²/4)^{1/4}인 척도에서만 한 숫자다. 계산 전에 적은 판본과 kill:

OE(정리 후보). 표준 모형의 흐름 f(μ) = ŝ²(μ) − 4 α_s(μ)^{4/3}은 섭동 영역(2 GeV–10^16 GeV)에서 단조 증가해
   E4 곡선을 정확히 한 번 가로지른다. 그러면 모든 렌더링 규칙은 한 사건에서 같은 동전을 읽고, E5에 남는 내용은
   “그 유일한 교차점이 M_Z다”(P20) 하나다.
   ŝ² 흐름은 §43.18 모듈(한 루프, M_W 위 공식)을 쓰고, M_W 아래에서 부호가 틀리는 것은 보수 한계
   |dŝ²/d ln μ| ≤ 0.003(0.2386 → 0.2313이 약 11 e-fold에 걸쳐 일어남)으로 따로 확인한다.
kill. 교차가 둘 이상이면 OE 기각. 교차 척도 μ*가 M_Z에서 3σ 밖이면 P20의 반례.
순환 경고. E4는 M_Z 값으로 찾았으므로 μ* ≈ M_Z는 α_s 세계 평균 행과 같은 사실이다(§43.18).

python -B -m examples.physics.rendering.ce_rendering_one_event
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_event_scale as ES

S2_SLOPE_BOUND_BELOW_MW = 0.003


def f(mu: float, as_mz: float = ES.A_WORLD) -> float:
    return ES.s2_run(mu) - 4 * ES.alpha_s_run(mu, as_mz) ** (4 / 3)


def monotonicity(n: int = 121) -> dict:
    mus = np.logspace(math.log10(2.0), 16, n)
    vals = np.array([f(m) for m in mus])
    t = np.log(mus)
    alpha_slope = np.gradient(np.array([4 * ES.alpha_s_run(m, ES.A_WORLD) ** (4 / 3) for m in mus]), t)
    below = mus < ES.CANDIDATES["M_W"]
    return {"crossings": int(np.sum(np.diff(np.sign(vals)) != 0)), "f_increasing": bool(np.all(np.diff(vals) > 0)),
            "min_alpha_term_slope_below_MW": float(-alpha_slope[below].max()),
            "robust_below_MW": bool(-alpha_slope[below].max() > S2_SLOPE_BOUND_BELOW_MW)}


def crossing(as_mz: float = ES.A_WORLD) -> float:
    return brentq(lambda lm: f(math.exp(lm), as_mz), math.log(20.0), math.log(1000.0))


def event_scale() -> dict:
    mu = math.exp(crossing())
    lo, hi = math.exp(crossing(ES.A_WORLD - ES.A_WORLD_ERR)), math.exp(crossing(ES.A_WORLD + ES.A_WORLD_ERR))
    sig_ln = abs(math.log(hi) - math.log(lo)) / 2
    return {"mu_star": mu, "mu_lo": lo, "mu_hi": hi, "sigma_ln": sig_ln,
            "pull_MZ": math.log(mu / ES.M_Z) / sig_ln,
            "pulls": {k: math.log(mu / v) / sig_ln for k, v in ES.CANDIDATES.items()}}


def main() -> None:
    print("monotonicity:", monotonicity())
    print("event scale:", event_scale())


if __name__ == "__main__":
    main()
