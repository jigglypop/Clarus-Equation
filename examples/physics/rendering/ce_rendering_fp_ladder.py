"""고정점의 e 사다리와 자기재귀 — 빠진 보정을 가려낼 수 있는가. 원장: 43장 §43.64. 예측값을 바꾸지 않는다.

FP r = ½[1 − λ], λ = (α_s/2π)(1 + δ/2π)는 렙톤 규칙의 α_s와 −3.75σ다(§43.58). r = ½[1 − λ − ε]로 고칠 때
필요한 ε을 사용자 제안(“부트스트랩 자기재귀, e의 −1, −2, −3 쭉”)으로 늘어놓는다. 계산 전에 적은 가족과 규칙:

F1 e의 급수 자르기: r = ½ Σ_{k≤n} (−λ)^k / k!, n = 1(FP), 2, 3, 4, ∞(= ½ e^{−λ}).
F2 e 사다리: ε = ± λ^k e^{−n}, k ∈ {1, 2}, n = 0 … 8 (36개).
F3 CE 부트스트랩: q = e^{−D(1−q)}(≈ e^{−3}, 인과 계보의 소멸 확률)로 ε = ± λ q^m, m = 1, 2, 3 (6개).
모든 판본은 r로 쓴 자기일관 방정식의 고정점이다(입력 없음).
적중: 렙톤 규칙 α_s와 |pull| ≤ 1. 기준선: 목표 ε이 크기 3σ_ε–10^{-3}에서 로그 균등, 부호 무작위일 때의 기대 적중 수.
   (정정: 처음 적은 하한 10^{-6}은 0 근처 과녁을 섞어 작은 항의 우연 적중을 부풀린다. FP가 3.75σ 어긋난다는 것을
   이미 알므로 과녁은 3σ 이상에서 뽑는다.)
사전 목격(증거로 세지 않음): 암산으로 λe^{−6}, λ²e^{−2}, λq², λ²/8이 맞을 것을 계산 전에 보았다.

python -B -m examples.physics.rendering.ce_rendering_fp_ladder
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_boundary_loop as BLP

S2_OBS = (0.23129, 0.00004)


def _parts(r: float) -> dict:
    a = r ** 3
    s2 = 4 * r ** 4
    d = s2 * (1 - s2)
    lam = a / (2 * math.pi) * (1 + d / (2 * math.pi))
    big_d = 3 + d
    q = brentq(lambda x: x - math.exp(-big_d * (1 - x)), 1e-12, 1 / big_d)
    return {"a": a, "s2": s2, "lam": lam, "q": q}


def _series(n: int | None):
    def g(r):
        lam = _parts(r)["lam"]
        if n is None:
            return 0.5 * math.exp(-lam)
        return 0.5 * sum((-lam) ** k / math.factorial(k) for k in range(n + 1))
    return g


def _eps(term):
    return lambda r: 0.5 * (1 - _parts(r)["lam"] - term(_parts(r)))


def family() -> dict:
    fam = {f"F1 n={n}": _series(n) for n in (1, 2, 3, 4)}
    fam["F1 n=inf"] = _series(None)
    for k in (1, 2):
        for n in range(9):
            for s in (1, -1):
                fam[f"F2 {'+' if s > 0 else '-'}lam^{k} e^-{n}"] = _eps(lambda p, k=k, n=n, s=s: s * p["lam"] ** k * math.exp(-n))
    for m in (1, 2, 3):
        for s in (1, -1):
            fam[f"F3 {'+' if s > 0 else '-'}lam q^{m}"] = _eps(lambda p, m=m, s=s: s * p["lam"] * p["q"] ** m)
    return fam


SEEN_BEFORE = {"lam e^-6": lambda p: -p["lam"] * math.exp(-6), "lam^2 e^-2": lambda p: -p["lam"] ** 2 * math.exp(-2),
               "lam q^2": lambda p: -p["lam"] * p["q"] ** 2, "lam^2/8": lambda p: -p["lam"] ** 2 / 8}


def solve(g) -> dict | None:
    try:
        r = brentq(lambda x: x - g(x), 0.3, 0.6)
    except ValueError:
        return None
    p = _parts(r)
    tgt = BLP.alpha_from_lepton_rule()
    eps = 1 - p["lam"] - 2 * r
    return {"alpha_s": p["a"], "eps": eps, "pull_lepton": (p["a"] - tgt["alpha_s_lepton"]) / tgt["sigma"],
            "pull_s2": (p["s2"] - S2_OBS[0]) / S2_OBS[1]}


def scan() -> dict:
    rows = {k: solve(g) for k, g in family().items()}
    rows = {k: v for k, v in rows.items() if v is not None}
    hits = sorted(k for k, v in rows.items() if abs(v["pull_lepton"]) <= 1)
    groups = {"F1": [k for k in rows if k.startswith("F1")], "F2": [k for k in rows if k.startswith("F2")],
              "F3": [k for k in rows if k.startswith("F3")]}
    return {"rows": rows, "hits": hits, "groups": groups}


def target_eps() -> dict:
    """렙톤 규칙 α_s를 정확히 주는 ε과 그 1σ 폭."""
    tgt = BLP.alpha_from_lepton_rule()

    def eps_for(alpha):
        r = alpha ** (1 / 3)
        return 1 - _parts(r)["lam"] - 2 * r
    e0 = eps_for(tgt["alpha_s_lepton"])
    e1 = eps_for(tgt["alpha_s_lepton"] + tgt["sigma"])
    return {"eps": e0, "sigma": abs(e1 - e0)}


def chance_hits(n_draw: int = 200000, seed: int = 7) -> dict:
    """기준선: 목표 ε이 로그 균등(3σ_ε–10^-3), 부호 무작위일 때 가족별 기대 적중 수와 ‘하나 이상’ 확률."""
    t = target_eps()
    rows = scan()["rows"]
    rng = np.random.default_rng(seed)
    draws = rng.choice([-1, 1], n_draw) * 10 ** rng.uniform(math.log10(3 * t["sigma"]), -3, n_draw)
    out = {}
    for g, keys in scan()["groups"].items():
        member = np.array([rows[k]["eps"] for k in keys])
        within = np.abs(draws[:, None] - member[None, :]) <= t["sigma"]
        out[g] = {"n": len(keys), "expected_hits": float(within.sum(1).mean()),
                  "p_at_least_one": float(within.any(1).mean())}
    return out


def seen_before() -> dict:
    return {k: solve(_eps(f)) for k, f in SEEN_BEFORE.items()}


def main() -> None:
    print("target:", target_eps())
    s = scan()
    for k, v in s["rows"].items():
        mark = " <- hit" if k in s["hits"] else ""
        print(f"{k:22s} alpha_s={v['alpha_s']:.7f} eps={v['eps']:+.3e} pull_lep={v['pull_lepton']:+9.2f} "
              f"pull_s2={v['pull_s2']:+7.2f}{mark}")
    print("hits:", s["hits"])
    print("chance:", chance_hits())
    print("seen before:", {k: {kk: round(vv, 7) for kk, vv in v.items()} for k, v in seen_before().items()})


if __name__ == "__main__":
    main()
