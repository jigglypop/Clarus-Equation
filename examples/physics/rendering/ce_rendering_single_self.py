"""하나의 자기 — 한 걸음의 이유. 원장: 43장 §43.66. 예측값을 바꾸지 않는다.

사용자 가설: “자식이 아니고 자기 자신이 하나라서 우리가 자아라고 인식하는 것.” 계산 전에 적은 판본과 kill:

SS1(유일성). 동전의 자기참조 r = g(r) = ½[1 − λ(r)], λ = (r³/2π)(1 + δ/2π), δ = 4r⁴(1 − 4r⁴)의 해가 물리 범위
   0 < r ≤ 2^{−1/2}(ŝ² ≤ 1)에서 정확히 하나이고 g가 축약(|g′| < 1)인지 본다. kill: 해가 둘 이상이거나 축약이 아님.
SS2(반영). 무차별점 r = ½에서 자기참조를 되풀이할 때 오차 비율(되비침 한 번마다 자기에게 다가가는 비율)을 잰다.
SS3(한 걸음, 읽기). F = 1 + α_s D의 1은 자기, α_s D는 자기에 붙은 결합이다. 붙은 것은 자기가 아니므로 더 붙잡지 않는다.
   끝없는 재귀(모든 결합이 새 자기)와 두 자기 계보(Q2의 1 − q²)는 자기가 여럿인 판본이다. 판정은 §43.65의 수치를 다시 읽을 뿐
   새 증거가 아니다.

python -B -m examples.physics.rendering.ce_rendering_single_self
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_higgs_weight as HW

R_MAX = 2 ** -0.5


def lam(r: float) -> float:
    s2 = 4 * r ** 4
    d = s2 * (1 - s2)
    return r ** 3 / (2 * math.pi) * (1 + d / (2 * math.pi))


def g(r: float) -> float:
    return 0.5 * (1 - lam(r))


def g_prime(r: float, h: float = 1e-7) -> float:
    return (g(r + h) - g(r - h)) / (2 * h)


def uniqueness(n: int = 20001) -> dict:
    rs = np.linspace(1e-6, R_MAX, n)
    f = np.array([r - g(r) for r in rs])
    gp = np.array([g_prime(r) for r in rs[1:-1]])
    root = brentq(lambda r: r - g(r), 0.3, 0.6)
    return {"roots": int(np.sum(np.diff(np.sign(f)) != 0)), "monotone": bool(np.all(np.diff(f) > 0)),
            "max_abs_g_prime": float(np.abs(gp).max()), "root": root, "g_prime_at_root": g_prime(root),
            "alpha_s": root ** 3}


def reflections(n: int = 6) -> dict:
    star = uniqueness()["root"]
    r, seq = 0.5, []
    for _ in range(n):
        seq.append(r - star)
        r = g(r)
    ratios = [seq[i + 1] / seq[i] for i in range(n - 1) if seq[i] != 0]
    return {"errors": seq, "ratios": ratios}


def self_count() -> dict:
    forms = HW.step_forms()
    q2 = HW.q2_universality()
    return {"one self (1 + m)": forms["one step 1+m"]["pull"],
            "every bond a new self 1/(1 - m)": forms["full recursion 1/(1-m)"]["pull"],
            "two self lineages (Q2) v/M_Pl pull": q2["q2"]["pulls"]["v/M_Pl (th+exp)"],
            "Q2 adopted": q2["adopt"]}


def main() -> None:
    print("SS1 uniqueness:", uniqueness())
    rf = reflections()
    print("SS2 errors:", [f"{e:+.2e}" for e in rf["errors"]], "ratios:", [round(x, 4) for x in rf["ratios"]])
    print("SS3:", self_count())


if __name__ == "__main__":
    main()
