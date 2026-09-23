"""섞임각의 닻 — 가둔 기록(색 단일항) 영역의 대각합은 ¼이다. 원장: 43장 §43.80. 예측값을 바꾸지 않는다.

한 세대 Λ(V₃ ⊕ V₂), 색 축 Y = −1/3, 약 축 Y = +1/2, Q = T3 + Y. 대통일 대각합 Tr T3²/Tr Q²는 전체에서 3/8이다.
계산 전에 적은 판본과 kill:

(가) 영역별 대각합: 전체, 색 단일항(k ∈ {0,3}), 색 있음(k ∈ {1,2}), 약 단일항(j ∈ {0,2}), 색 0만, 색 3만.
    주장: 색 단일항 영역만 E4 닻 ŝ² = ¼을 준다. kill: 색 단일항이 아닌 영역도 ¼이면 “가둔 영역” 읽기 기각.
(나) 무게를 준 대각합의 범위: 영역 무게 w_k ≥ 0(색 점유별)로 섞으면 결과는 ¼(단일항)과 9/20(색 있음) 사이에 갇히는가.
    그렇다면 실제 ŝ² = 0.2313 < ¼에는 닿을 수 없고, 닻에서의 벗어남은 색 무게가 아닌 다른 기제(기운 동전)에서 온다.
    E1 무게 w_k = a^k(§43.41의 a^c)도 함께 보고.

python -B -m examples.physics.rendering.ce_rendering_lepton_trace
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

from examples.physics.rendering import ce_rendering_registry as R

COLOR, WEAK = (0, 1, 2), (3, 4)
Y_AXIS = {0: Fraction(-1, 3), 1: Fraction(-1, 3), 2: Fraction(-1, 3), 3: Fraction(1, 2), 4: Fraction(1, 2)}


def states() -> list[dict]:
    out = []
    for n in range(6):
        for s in itertools.combinations(range(5), n):
            k = sum(1 for i in s if i in COLOR)
            weak = [i for i in s if i in WEAK]
            j = len(weak)
            y = sum((Y_AXIS[i] for i in s), Fraction(0))
            if j == 1:  # 약 이중항: 축 3 = T3 +½, 축 4 = T3 −½
                t3 = Fraction(1, 2) if weak[0] == 3 else Fraction(-1, 2)
            else:
                t3 = Fraction(0)
            out.append({"k": k, "j": j, "Y": y, "T3": t3, "Q": t3 + y})
    return out


SECTORS = {"all": lambda s: True, "colour singlet (k=0,3)": lambda s: s["k"] in (0, 3),
           "coloured (k=1,2)": lambda s: s["k"] in (1, 2), "weak singlet (j=0,2)": lambda s: s["j"] in (0, 2),
           "k=0 only": lambda s: s["k"] == 0, "k=3 only": lambda s: s["k"] == 3}


def sector_ratios() -> dict:
    out = {}
    for name, sel in SECTORS.items():
        ss = [s for s in states() if sel(s)]
        t = sum(s["T3"] ** 2 for s in ss)
        q = sum(s["Q"] ** 2 for s in ss)
        out[name] = t / q if q else None
    quarter = [k for k, v in out.items() if v == Fraction(1, 4)]
    return {"ratios": out, "quarter_sectors": quarter,
            "only_colour_singlets_give_quarter": all(k in ("colour singlet (k=0,3)", "k=0 only", "k=3 only")
                                                     for k in quarter)}


def _by_k() -> dict:
    t = {k: Fraction(0) for k in range(4)}
    q = {k: Fraction(0) for k in range(4)}
    for s in states():
        t[s["k"]] += s["T3"] ** 2
        q[s["k"]] += s["Q"] ** 2
    return {"T": t, "Q": q}


def weighted_range(n: int = 20000, seed: int = 3) -> dict:
    """음이 아닌 색 점유 무게로 섞은 대각합의 범위(무작위 무게 표본)와 E1 무게 a^k."""
    b = _by_k()
    tk = np.array([float(b["T"][k]) for k in range(4)])
    qk = np.array([float(b["Q"][k]) for k in range(4)])
    rng = np.random.default_rng(seed)
    w = rng.exponential(1.0, (n, 4)) ** 3
    r = (w @ tk) / (w @ qk)
    a = R.calibrated_alpha_s()[0] ** (1 / 3)
    wa = np.array([a ** k for k in range(4)])
    return {"min": float(r.min()), "max": float(r.max()), "E1_weights_a^k": float((wa @ tk) / (wa @ qk)),
            "s2_obs": R.SZ2, "reachable": float(r.min()) <= R.SZ2}


def main() -> None:
    sr = sector_ratios()
    for k, v in sr["ratios"].items():
        print(f"  {k:24s} Tr T3^2 / Tr Q^2 = {v}")
    print("quarter sectors:", sr["quarter_sectors"], "only colour singlets:", sr["only_colour_singlets_give_quarter"])
    print("weighted range:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in weighted_range().items()})


if __name__ == "__main__":
    main()
