"""보강 4 — 옥탄트는 나와 바깥의 경계선이다. 원장: 43장 §43.43. 예측값을 바꾸지 않는다.

TBM의 세 열은 (2,−1,−1)/√6 = 나 − 바깥 평균(경계 대비), (1,1,1)/√3 = 전체(분별 이전), (0,−1,1)/√2 = 바깥 내부의 분별이다.
계산 전에 적은 판본과 kill:

B-ii. 순수 순환(순환 행렬, S 대칭)은 전체 열을 보존한다(TM2). 나/바깥 경계를 그으면 경계 대비 열이 질량 고유상태로
   보존된다(TM1). T1(|U_μ1| = |U_τ1|)은 TM1 합규칙이다.
B-iii. TM1 = U_TBM·R23(θ, φ)에서 옥탄트는 cos φ의 부호(경계의 어느 쪽), 크기는 cos φ로 정해진다.
kill: K1 대비 벡터가 TBM 열이 아님. K2 T1의 δ가 TM1 정확 합규칙과 1° 넘게 다름.
   K3 최신 자료가 TM2를 TM1보다 선호. K4 “나”를 τ로 둔 대비가 자료(|U_e1|² ≈ 2/3)와 양립.

python -B -m examples.physics.rendering.ce_rendering_boundary
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_registry as R

TBM = np.array([[2, 1, 0], [-1, 1, -1], [-1, 1, 1]], float) / np.array([math.sqrt(6), math.sqrt(3), math.sqrt(2)])
S12_DATA = {"NuFIT 6.0 SK": (0.308, 0.012, 0.011), "JUNO 2025": (0.3092, 0.0087, 0.0087)}


def contrast_vector(me: int) -> np.ndarray:
    """나(me) − 바깥(나머지 두 축) 평균, 정규화."""
    v = -0.5 * np.ones(3)
    v[me] = 1.0
    return v / np.linalg.norm(v)


def best_tbm_overlap(v: np.ndarray) -> float:
    return float(np.abs(TBM.T @ v).max())


def s12_tm1(s13sq: float) -> float:
    return (1 - 3 * s13sq) / (3 * (1 - s13sq))


def s12_tm2(s13sq: float) -> float:
    return 1 / (3 * (1 - s13sq))


def pull(pred: float, obs: tuple[float, float, float]) -> float:
    v, up, dn = obs
    return (pred - v) / (up if pred > v else dn)


def delta_from_column1(s12sq: float, s23sq: float, s13sq: float, target: float = 1 / 6) -> float:
    """|U_μ1|² = target를 푸는 δ ∈ (π, 2π) (M1 방향)."""
    g = lambda x: abs(R.pmns_matrix(s12sq, s23sq, s13sq, x)[1][0]) ** 2 - target
    return math.degrees(brentq(g, math.pi + 1e-3, 1.5 * math.pi)) if g(math.pi + 1e-3) * g(1.5 * math.pi) < 0 \
        else math.degrees(brentq(g, 1.5 * math.pi, 2 * math.pi - 1e-3))


def tm1_matrix(theta: float, phi: float) -> np.ndarray:
    r = np.array([[1, 0, 0], [0, math.cos(theta), math.sin(theta) * np.exp(-1j * phi)],
                  [0, -math.sin(theta) * np.exp(1j * phi), math.cos(theta)]])
    return TBM @ r


def tm1_angles(u: np.ndarray) -> dict:
    s13sq = abs(u[0, 2]) ** 2
    return {"s13sq": s13sq, "s12sq": abs(u[0, 1]) ** 2 / (1 - s13sq), "s23sq": abs(u[1, 2]) ** 2 / (1 - s13sq),
            "col1": [float(abs(u[i, 0]) ** 2) for i in range(3)]}


def tm1_phase_for_s2(c: dict) -> dict:
    """S2의 s13², s23²를 TM1으로 실현하는 (θ, φ). 옥탄트 = cos φ의 부호."""
    s13sq, s23sq = R.pmns_s2(c, 1), R.pmns_s2(c, 3)
    theta = math.asin(math.sqrt(3 * s13sq))
    phi = brentq(lambda p: tm1_angles(tm1_matrix(theta, p))["s23sq"] - s23sq, 1e-6, math.pi / 2)
    flipped = tm1_angles(tm1_matrix(theta, math.pi - phi))["s23sq"]
    return {"theta": theta, "phi_deg": math.degrees(phi), "cos_phi": math.cos(phi), "half_sqrt_d": math.sqrt(c["d"]) / 2,
            "s23sq_cos_pos": tm1_angles(tm1_matrix(theta, phi))["s23sq"], "s23sq_cos_neg": flipped}


def tm1_delta_for_octant(c: dict, s23sq: float) -> float:
    """TM1(열 1 보존) + M1(J < 0)에서 주어진 s23²의 δ. 옥탄트가 뒤집혀도 경계 구조가 주는 짝."""
    s13 = R.pmns_s2(c, 1)
    return delta_from_column1(s12_tm1(s13), s23sq, s13)


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    for s23 in (0.4556, 0.470, 0.5, 0.5445, 0.561):
        print(f"TM1+M1 pair: s23^2={s23:.4f} -> delta={tm1_delta_for_octant(c, s23):.2f} deg")
    print("K1/K4 contrast vs TBM columns:", {name: round(best_tbm_overlap(contrast_vector(i)), 6)
                                            for i, name in enumerate(("me=e", "me=mu", "me=tau"))})
    s13 = R.pmns_s2(c, 1)
    for name, obs in S12_DATA.items():
        print(f"K3 {name}: TM1 s12^2={s12_tm1(s13):.5f} pull {pull(s12_tm1(s13), obs):+.2f} | "
              f"TM2 {s12_tm2(s13):.5f} pull {pull(s12_tm2(s13), obs):+.2f} | S2 {R.pmns_s2(c, 2):.5f}")
    t1 = R.delta_pmns_tm1(c)
    tm1 = delta_from_column1(s12_tm1(s13), R.pmns_s2(c, 3), s13)
    print(f"K2 T1 delta={t1:.3f} deg, TM1-exact delta={tm1:.3f} deg, diff {t1 - tm1:+.3f}")
    u = R.pmns_matrix(0.308, 0.470, 0.02215, math.radians(212))
    print("data |U_x1|^2 (NuFIT SK best):", [round(abs(u[i][0]) ** 2, 4) for i in range(3)], "TM1: [0.6667, 0.1667, 0.1667]")
    print("B-iii TM1 phase:", {k: round(v, 5) for k, v in tm1_phase_for_s2(c).items()})


if __name__ == "__main__":
    main()
