"""고정점의 읽기 — 기운 동전. 원장: 43장 §43.57. 예측값을 바꾸지 않는다.

FP(§43.55) r = ½(1 − a), a = (α_s/2π)(1 + δ/2π)는 참/거짓 동전이 ½(1 ± a)로 기운 것과 같다. a의 첫 항은 슈윙거
이상 자기모멘트 α/2π와 같은 꼴이다. 계산 전에 적은 판본과 kill:

BC  P(렌더링) = ½(1 + a), P(미렌더링) = ½(1 − a). FP와 같은지(항등) 확인.
G   대안: r = 1/g, g = 2(1 + a) (g 인자의 역수). kill: ŝ² 3σ 밖.
U   보편성: 같은 기울기를 CKM Bool 분할(세대 세계 {V, EXACT₁, MAJ})에도 적용.
    β = π P(V), γ = π P(EXACT₁), α = π P(MAJ), P(참) = ½(1 + a). 측정(PDG 2024): sin 2β = 0.709 ± 0.011,
    γ = 65.7° ± 3.0°, α = 85.2°(+4.8, −4.3). kill: 기운 β가 3σ 밖이면 “기울기는 게이지 축에만”으로 적용 범위 축소.

python -B -m examples.physics.rendering.ce_rendering_biased_coin
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_e4_anchor as EA

SIN2BETA = (0.709, 0.011)
GAMMA_DEG = (65.7, 3.0, 3.0)
ALPHA_DEG = (85.2, 4.8, 4.3)


def anomaly(r: float) -> float:
    a, s2 = r ** 3, 4 * r ** 4
    d = s2 * (1 - s2)
    return a / (2 * math.pi) * (1 + d / (2 * math.pi))


def coin_fixed_point(form: str) -> dict:
    g = {"BC": lambda r: 0.5 * (1 - anomaly(r)), "G": lambda r: 1 / (2 * (1 + anomaly(r)))}[form]
    r = brentq(lambda x: x - g(x), 0.3, 0.5)
    s2 = 4 * r ** 4
    return {"r": r, "a": anomaly(r), "p_true": 1 - r, "alpha_s": r ** 3, "s2": s2,
            "pull_s2": (s2 - EA.S2_OBS[0]) / EA.S2_OBS[1]}


def identity_with_fp() -> float:
    return abs(coin_fixed_point("BC")["r"] - EA.fixed_point(*EA.ADOPTED)["r"])


def ckm_angles(p_true: float) -> dict:
    q = 1 - p_true
    pv, p1 = q ** 3, 3 * p_true * q * q
    pmaj = 3 * p_true ** 2 * q + p_true ** 3
    return {"beta": 180 * pv, "gamma": 180 * p1, "alpha": 180 * pmaj, "sum": 180 * (pv + p1 + pmaj)}


def angle_pulls(ang: dict) -> dict:
    s2b = math.sin(2 * math.radians(ang["beta"]))
    g, gu, gd = GAMMA_DEG
    a, au, ad = ALPHA_DEG
    return {"sin2beta": (s2b - SIN2BETA[0]) / SIN2BETA[1],
            "gamma": (ang["gamma"] - g) / (gu if ang["gamma"] > g else gd),
            "alpha": (ang["alpha"] - a) / (au if ang["alpha"] > a else ad)}


def universality() -> dict:
    fp = coin_fixed_point("BC")
    out = {}
    for tag, p in (("unbiased", 0.5), ("biased", fp["p_true"])):
        ang = ckm_angles(p)
        out[tag] = {**{k: round(v, 3) for k, v in ang.items()}, **{f"pull_{k}": round(v, 2) for k, v in angle_pulls(ang).items()}}
    return out


def main() -> None:
    print("BC == FP (|Δr|):", identity_with_fp())
    for form in ("BC", "G"):
        print(form, {k: round(v, 6) for k, v in coin_fixed_point(form).items()})
    for k, v in universality().items():
        print("U", k, v)


if __name__ == "__main__":
    main()
