"""약 통로 하나(Higgs 양자수)의 확정과 전자약 대칭 깨짐. 원장: 43장 §43.22.

§43.20의 렌더링 공간 V3 ⊕ V2에서 Λ^1의 약 통로 (1, 2, +1/2)가 Higgs 양자수와 같다. 그 전하 0 성분에
진공 기댓값을 두고 계산한다.
1. 남는 생성자는 Q = T3 + Y 하나이며 광자는 질량 0, M_W = g v/2, M_Z = √(g²+g'²) v/2, M_W/M_Z = c_W.
2. 16상태의 전하 Q = T3 + Y는 모두 1/3의 정수배다(전하 양자화). 전자 전하는 아래 쿼크의 정확히 −3배.
퍼텐셜의 크기(λ, M_H)는 여기서 유도하지 않는다. 사건 척도는 §43.18의 M_Z다.

python -B -m examples.physics.rendering.ce_rendering_ewsb
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from examples.physics.rendering import ce_rendering_gauge as GA

S2_MZ, AEM_MZ = 0.23129, 1 / 127.951


def weak_t3(s: tuple[int, ...]) -> Fraction:
    w = [i for i in s if i in GA.WEAK]
    if len(w) != 1:
        return Fraction(0)
    return Fraction(1, 2) if w[0] == 3 else Fraction(-1, 2)


def charges_all_states() -> list[Fraction]:
    return [weak_t3(s) + GA.hypercharge(s) for s in GA.subsets()]


def higgs_channel() -> tuple[int, ...]:
    """Λ^1의 약 통로 중 전하 0 성분(T3 = −1/2, Y = +1/2)."""
    return (4,)


def gauge_boson_masses(v: float = 246.21965) -> dict:
    """(W1, W2, W3, B) 질량 행렬을 전하 0 Higgs 성분의 진공 기댓값에서 계산."""
    e2 = 4 * math.pi * AEM_MZ
    g2 = e2 / S2_MZ
    gp2 = e2 / (1 - S2_MZ)
    g, gp = math.sqrt(g2), math.sqrt(gp2)
    t = [np.array([[0, 1], [1, 0]]) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.array([[1, 0], [0, -1]]) / 2]
    y = 0.5 * np.eye(2)
    h = np.array([0.0, v / math.sqrt(2)])          # 아래(T3 = −1/2) 성분
    gens = [g * t[0], g * t[1], g * t[2], gp * y]
    m2 = np.array([[2 * np.real(np.vdot(a @ h, b @ h)) for b in gens] for a in gens])
    vals, vecs = np.linalg.eigh(m2)
    massless = vecs[:, np.argmin(vals)]
    q_dir = np.array([0, 0, 1 / g, 1 / gp]) / math.hypot(1 / g, 1 / gp)   # Q = T3 + Y 방향(결합으로 가중)
    return {"masses_GeV": np.sqrt(np.clip(vals, 0, None)), "M_W": g * v / 2, "M_Z": math.hypot(g, gp) * v / 2,
            "photon_is_Q": abs(abs(massless @ q_dir) - 1.0), "M_W/M_Z": g / math.hypot(g, gp),
            "c_W": math.sqrt(1 - S2_MZ)}


def main() -> None:
    qs = sorted(set(charges_all_states()))
    print("electric charges in Λ(V3⊕V2):", [str(q) for q in qs])
    print("all multiples of 1/3:", all((3 * q).denominator == 1 for q in qs))
    s = higgs_channel()
    print(f"Higgs channel {s}: T3={weak_t3(s)}, Y={GA.hypercharge(s)}, Q={weak_t3(s) + GA.hypercharge(s)}")
    m = gauge_boson_masses()
    print({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(v, 6)) for k, v in m.items()})


if __name__ == "__main__":
    main()
