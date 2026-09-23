"""복소 척도인자 A = a e^{iφ}: 나선 회전의 작용 수준 표현. 원장: 43장 §43.16.

잠긴 모듈(v1–v5)은 import만 한다. 예측값은 바꾸지 않는다.

C5  중력 작용은 |A| = a에만 의존한다(φ 회전은 U(1) 대칭). 그러므로 프리드만 방정식은 표준 그대로이고
    ω → 0에서 FLRW와 정확히 같다. φ는 유클리드 드 시터 시간 원의 각이며 ω = dφ/dt = H_Λ/2는
    원의 주기와 접선–현 정리로 정해진다(역학 변수가 아니다).
    d ln A/dt = H + iω.  나이테(같은 틀의 길이 비)는 |A|만 보므로 Re = H를 읽고,
    오늘의 직접 판독은 크기 |H + iω| = √(H² + ω²)를 읽는다.

python -B -m examples.physics.rendering.ce_rendering_complex_scale
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_ivp

from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_spiral as SP

KMS_MPC_PER_GYR = 1.0 / 977.79


def background(c: dict):
    """평탄 물질+진공 우주를 우주 시간 t(Gyr)로 적분: a(t), H(t), φ(t) = (H_Λ/2) t."""
    h0 = 100.0 * PL.h_rings(c) * KMS_MPC_PER_GYR
    om = c["Om"]
    hub = lambda a: h0 * math.sqrt(om / a ** 3 + 1.0 - om)
    a_start = 1e-6
    t_start = 2.0 / (3.0 * h0 * math.sqrt(om)) * a_start ** 1.5
    sol = solve_ivp(lambda t, y: [y[0] * hub(y[0])], (t_start, 30.0), [a_start], rtol=1e-11, atol=1e-16,
                    dense_output=True, events=lambda t, y: y[0] - 1.0)
    t0 = float(sol.t_events[0][0])
    omega = 0.5 * h0 * math.sqrt(1.0 - om)
    return sol, hub, t0, omega, h0


def complex_rate(c: dict, t: float) -> complex:
    sol, hub, _, omega, _ = background(c)
    a = float(sol.sol(t)[0])
    return complex(hub(a), omega)


def friedmann_residual(c: dict) -> float:
    """|A|가 표준 방정식 (ȧ/a)² = H0²(Ω_m a⁻³ + Ω_Λ)를 만족하는 최대 상대 오차."""
    sol, hub, t0, _, _ = background(c)
    ts = np.linspace(0.05 * t0, t0, 200)
    worst = 0.0
    for t in ts:
        a = float(sol.sol(t)[0])
        adot = float((sol.sol(t + 1e-4)[0] - sol.sol(t - 1e-4)[0]) / 2e-4)
        worst = max(worst, abs(adot / a - hub(a)) / hub(a))
    return worst


def ring_ratio_invariance(c: dict, phase_rate: float) -> float:
    """두 기록 거리 D_M(z)의 비가 위상 회전 속도와 무관함을 확인한다(|A|만 들어감)."""
    om = c["Om"]
    e = lambda z: math.sqrt(om * (1 + z) ** 3 + 1 - om)
    dm = lambda z: quad(lambda x: 1.0 / e(x), 0, z)[0]
    ratio0 = dm(0.5) / dm(2.33)
    # 위상은 |e^{iφ}| = 1이라 적분에 들어오지 않는다: 회전을 넣은 적분도 같은 값이어야 한다.
    dm_rot = lambda z: quad(lambda x: abs(complex(math.cos(phase_rate * x), math.sin(phase_rate * x))) / e(x), 0, z)[0]
    return abs(dm_rot(0.5) / dm_rot(2.33) - ratio0)


def record_phase_gap(c: dict) -> float:
    """CMB(z*≈1090)와 BAO 끌림(z_d≈1020) 기록 사이의 위상 차(rad)."""
    sol, hub, t0, omega, _ = background(c)
    t_of_a = lambda a_target: quad(lambda a: 1.0 / (a * hub(a)), 1e-8, a_target, limit=200)[0]
    return omega * (t_of_a(1 / 1021.0) - t_of_a(1 / 1091.0))


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    sol, hub, t0, omega, h0 = background(c)
    rate = complex_rate(c, t0)
    print(f"t0 = {t0:.3f} Gyr, d ln A/dt today = {rate.real / KMS_MPC_PER_GYR:.2f} + i {rate.imag / KMS_MPC_PER_GYR:.2f} km/s/Mpc")
    print(f"|d ln A/dt| = {abs(rate) / KMS_MPC_PER_GYR:.3f} vs spiral formula {SP.direct_readout(c):.3f}")
    print(f"Friedmann residual of |A| (standard FLRW): {friedmann_residual(c):.2e}")
    print(f"ring ratio change under rotation: {ring_ratio_invariance(c, 0.3):.1e}")
    print(f"phase gap between CMB and BAO drag records: {record_phase_gap(c):.2e} rad (cannot shift the CMB-BAO scale)")


if __name__ == "__main__":
    main()
