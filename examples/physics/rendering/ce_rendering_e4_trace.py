"""보강 1 — E4(sin θ_W = A₂)를 가중 대각합 비로 유도할 수 있는가. 원장: 43장 §43.41. 예측값을 바꾸지 않는다.

통일 척도의 sin²θ_W = 3/8은 한 세대 Λ^even(C³⊕C²)에 대한 tr T₃²/tr Q²다. 계산 전에 적은 두 부류:

부류 I (항등식). 렌더링 가중치 w_S(a)가 연속이고 w_S(1) = 1(전부 렌더링되면 가중치 없음)인 가중 비 r_w(a).
   kill: a = 1에서 r_w = 3/8이지만 E4는 sin²θ_W = 4a⁴ = 4 > 1을 요구한다.
부류 II (사건). 물리값 a = α_s^{1/3}에서 미리 정한 여섯 가중치 가족을 채점한다.
   색 축당 a^{c}, a^{2c}, a^{−c}, a^{−2c}; 차수당 a^{|S|}, a^{2|S|}. 통과: ŝ²(M_Z)와 1σ 안(σ는 α_s 세계 평균 오차 전파).

python -B -m examples.physics.rendering.ce_rendering_e4_trace
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

S2_MZ = 0.23129                      # MS-bar sin²θ_W(M_Z), PDG 2024
ALPHA_S, ALPHA_S_SIGMA = 0.1180, 0.0009
COLOR, WEAK = (0, 1, 2), (3, 4)


def generation() -> list[dict]:
    """Λ^even(C³⊕C²)의 16상태: 색 통로 수 c, 약 통로 수 w, Y = w/2 − c/3, T₃ = ±1/2(약 통로 하나일 때)."""
    out = []
    for n in (0, 2, 4):
        for s in itertools.combinations(range(5), n):
            c = sum(i in COLOR for i in s)
            w = sum(i in WEAK for i in s)
            y = Fraction(w, 2) - Fraction(c, 3)
            t3 = (Fraction(1, 2) if 3 in s else Fraction(-1, 2)) if w == 1 else Fraction(0)
            out.append({"S": s, "c": c, "w": w, "Y": y, "T3": t3, "Q": t3 + y})
    return out


FAMILIES = {
    "a^c": lambda st, a: a ** st["c"],
    "a^2c": lambda st, a: a ** (2 * st["c"]),
    "a^-c": lambda st, a: a ** (-st["c"]),
    "a^-2c": lambda st, a: a ** (-2 * st["c"]),
    "a^|S|": lambda st, a: a ** len(st["S"]),
    "a^2|S|": lambda st, a: a ** (2 * len(st["S"])),
}


def weighted_ratio(weight, a: float) -> float:
    g = generation()
    num = sum(weight(st, a) * float(st["T3"]) ** 2 for st in g)
    den = sum(weight(st, a) * float(st["Q"]) ** 2 for st in g)
    return num / den


def unweighted_ratio() -> Fraction:
    g = generation()
    return sum(st["T3"] ** 2 for st in g) / sum(st["Q"] ** 2 for st in g)


def class_i_kill() -> dict:
    """a = 1에서 모든 부류 I 가중치는 3/8을 주지만 E4는 4a⁴ = 4다. E4가 물리적(≤ 1)인 영역은 a ≤ 2^{-1/2}."""
    return {"weighted_at_a1": unweighted_ratio(), "e4_at_a1": 4.0, "e4_physical_a_max": 2 ** -0.5,
            "e4_physical_alpha_s_max": 2 ** -1.5}


def class_ii_scan() -> dict:
    a0 = ALPHA_S ** (1 / 3)
    out = {}
    for name, wf in FAMILIES.items():
        r = weighted_ratio(wf, a0)
        da = a0 * ALPHA_S_SIGMA / (3 * ALPHA_S)
        sig = abs(weighted_ratio(wf, a0 + da) - weighted_ratio(wf, a0 - da)) / 2
        out[name] = {"ratio": r, "sigma": sig, "pull": (r - S2_MZ) / math.hypot(sig, 4e-5)}
    e4 = 4 * a0 ** 4
    out["E4 4a^4"] = {"ratio": e4, "sigma": e4 * 4 / 3 * ALPHA_S_SIGMA / ALPHA_S,
                      "pull": (e4 - S2_MZ) / (e4 * 4 / 3 * ALPHA_S_SIGMA / ALPHA_S)}
    return out


def main() -> None:
    g = generation()
    charges = sorted({st["Q"] for st in g})
    print(f"generation: {len(g)} states, unweighted tr T3^2 / tr Q^2 = {unweighted_ratio()}, charges {[str(q) for q in charges]}")
    print("class I kill:", {k: (str(v) if isinstance(v, Fraction) else round(v, 4)) for k, v in class_i_kill().items()})
    for name, r in class_ii_scan().items():
        print(f"class II {name:8s} ratio={r['ratio']:.5f} sigma={r['sigma']:.5f} pull={r['pull']:+.1f}")


if __name__ == "__main__":
    main()
