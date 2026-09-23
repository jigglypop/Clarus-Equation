"""양자 중력의 흔적 — 화소 시계의 모듈러 요동 C = S와 간섭계 제약. 원장: 43장 §43.84. 예측값을 바꾸지 않는다.

CE에서 중력은 화소 기록의 상태방정식이다(§43.71–43.72). 흔적은 얽힘이 아니라(P23) 화소 시계의 요동이어야 한다.
모듈러 해밀토니안 K = −ln p의 평균은 엔트로피 S, 분산은 얽힘 용량 C. 계산 전에 적은 판본과 규칙:

(가) 화소 시계 후보: 지수(평균 1), 균등 [0,2], 반정규(평균 1), 감마(2, 평균 1), 가우스(평균 1, σ 1), 공평한 동전(1 bit).
    각각 S와 C = Var(−ln p). 주장: 지수 시계만 C/S = 1.
(나) 지평선 전체(독립 화소 N개): C/S가 N과 무관한가.
(다) Verlinde–Zurek 정규화(Li, Lee, Chen, Zurek 2022, arXiv:2209.07543): α = 1 ⇔ ⟨ΔK²⟩ = ⟨K⟩ = A/4G,
    ⟨δL²⟩ = α ℓ_P L/4π. 3σ 제약: LIGO α ≲ 3(적외선 절단) / 0.1(절단 없음), Holometer α ≲ 0.7 / 0.6.
    GQuEST 도달: α < 0.1(3σ, 2160 h).
판정: CE는 간섭계 요동을 등록한 적이 없으므로 기각이 아니라 위험(§43.72의 국소 지평선 확장)으로 기록한다.
    응답 모형에 기대는 탈출은 구제로 쓰지 않고 열린 문제로만 적는다.

python -B -m examples.physics.rendering.ce_rendering_modular
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats

CONSTRAINTS_3SIGMA = {"LIGO (IR cutoff)": 3.0, "LIGO (no IR cutoff)": 0.1,
                      "Holometer (IR cutoff)": 0.7, "Holometer (no IR cutoff)": 0.6}
GQUEST_REACH = 0.1


def _moments(dist, lo: float, hi: float) -> tuple[float, float]:
    f = lambda x: dist.pdf(x)
    k = lambda x: -dist.logpdf(x)
    s = integrate.quad(lambda x: f(x) * k(x), lo, hi, limit=400)[0]
    k2 = integrate.quad(lambda x: f(x) * k(x) ** 2, lo, hi, limit=400)[0]
    return s, k2 - s * s


def clocks() -> dict:
    cands = {"exponential(mean 1)": (stats.expon(scale=1.0), 0, 60),
             "uniform[0,2]": (stats.uniform(0, 2), 0, 2),
             "half-normal(mean 1)": (stats.halfnorm(scale=math.sqrt(math.pi / 2)), 0, 30),
             "gamma(2, mean 1)": (stats.gamma(2, scale=0.5), 0, 40),
             "gaussian(mean 1, sd 1)": (stats.norm(1, 1), -12, 14)}
    out = {}
    for name, (d, lo, hi) in cands.items():
        s, c = _moments(d, lo, hi)
        out[name] = {"S": s, "C": c, "C_over_S": c / s}
    out["fair coin (1 bit)"] = {"S": math.log(2), "C": 0.0, "C_over_S": 0.0}
    unit = [k for k, v in out.items() if abs(v["C_over_S"] - 1) < 1e-6]
    return {"rows": out, "unit_ratio": unit}


def horizon_ratio(ns=(10, 1000, 100000), seed: int = 2) -> dict:
    """독립 지수 시계 N개의 K = Σt: 평균 N, 분산 N → C/S = 1(몬테카를로 확인)."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        k = rng.gamma(n, 1.0, 200000)        # Σ of n Exp(1)
        out[n] = float(k.var() / k.mean())
    return out


def interferometer() -> dict:
    alpha = clocks()["rows"]["exponential(mean 1)"]["C_over_S"]
    excluded = sorted(k for k, v in CONSTRAINTS_3SIGMA.items() if alpha > v)
    return {"alpha_CE": alpha, "excluded_by_3sigma": excluded, "gquest_decisive": alpha > GQUEST_REACH}


def main() -> None:
    c = clocks()
    for k, v in c["rows"].items():
        print(f"  {k:24s} S={v['S']:.4f} C={v['C']:.4f} C/S={v['C_over_S']:.4f}")
    print("unit ratio:", c["unit_ratio"])
    print("horizon C/S:", {k: round(v, 4) for k, v in horizon_ratio().items()})
    print("interferometer:", interferometer())


if __name__ == "__main__":
    main()
