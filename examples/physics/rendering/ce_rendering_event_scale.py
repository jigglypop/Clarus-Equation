"""2차원 렌더링 사건의 척도. 원장: 43장 §43.18.

잠긴 모듈은 import만 한다.

E5  ŝ²(μ) = 4 α_s(μ)^{4/3}은 척도 항등식이 아니라 2차원 렌더링이 확정되는 한 척도 μ_E의 사건 관계다
    (§43.4 반례 1). 후보 전자약 척도 {M_W, M_Z, M_H, m_t, v} 각각을 μ_E로 두고, 측정된 ŝ²(M_Z)와 α_em(M_Z)을
    한 루프 전자약·두 루프 QCD(n_f=5)로 μ_E까지 옮긴 뒤 사건 관계로 α_s(μ_E)를 정하고 다시 M_Z로 내려
    세계 평균 α_s(M_Z)와 비교한다. 새 상수는 없다.

python -B -m examples.physics.rendering.ce_rendering_event_scale
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

M_Z = 91.1876
AEM_MZ = 1 / 127.951
SZ2_MZ = 0.23129
A_WORLD, A_WORLD_ERR = 0.1180, 0.0009
CANDIDATES = {"M_W": 80.3692, "M_Z": M_Z, "M_H": 125.20, "m_t": 172.57, "v": 246.21965}
B_Y, B_2 = 41 / 6, -19 / 6
B3_0, B3_1 = 11 - 2 * 5 / 3, 102 - 38 * 5 / 3


def alpha_s_run(mu: float, as_mz: float, steps: int = 400) -> float:
    t0, t1 = math.log(M_Z), math.log(mu)
    h = (t1 - t0) / steps
    a = as_mz
    f = lambda x: -(B3_0 * x * x / (2 * math.pi) + B3_1 * x ** 3 / (8 * math.pi ** 2))
    for _ in range(steps):
        k1 = f(a); k2 = f(a + h / 2 * k1); k3 = f(a + h / 2 * k2); k4 = f(a + h * k3)
        a += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return a


def s2_run(mu: float) -> float:
    L = math.log(mu / M_Z)
    a2 = 1 / (SZ2_MZ / AEM_MZ - B_2 / (2 * math.pi) * L)
    ay = 1 / ((1 - SZ2_MZ) / AEM_MZ - B_Y / (2 * math.pi) * L)
    return ay / (a2 + ay)


def predicted_alpha_s_mz(mu_event: float) -> float:
    """사건 척도에서 ŝ² = 4 α_s^{4/3}을 만족시키는 α_s(M_Z)."""
    target = (s2_run(mu_event) / 4.0) ** 0.75
    return brentq(lambda a: alpha_s_run(mu_event, a) - target, 0.08, 0.16)


def event_scale_table() -> dict:
    return {name: predicted_alpha_s_mz(mu) for name, mu in CANDIDATES.items()}


def main() -> None:
    for name, a in event_scale_table().items():
        print(f"event at {name:4s} ({CANDIDATES[name]:7.2f} GeV): alpha_s(M_Z) = {a:.5f}  pull vs world "
              f"{(a - A_WORLD) / A_WORLD_ERR:+.2f}")


if __name__ == "__main__":
    main()
