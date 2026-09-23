"""세 세대와 선형 중력. 원장: 43장 §43.21.

공리 C7: 세대 공간은 우로보로스 3-순환(3차원 단계의 세 축) C^3_gen이고, 선도 차수 질량 행렬은 순환 치환 S와
교환한다(순환 행렬, CE 질량 행렬과 같은 꼴). 중성미자는 자기 거울(Majorana)이라 한 쌍의 질량 상태를 구별하지
않는 최대 섞임(π/4)을 더 받는다.

확인하는 사실(알려진 수학 포함):
1. 세대 수 = 3차원 단계의 축 수 = 3. 한 세대가 이상 없음(§43.20)이므로 세 세대도 이상 없음.
2. 위·아래 쿼크가 모두 순환 행렬이면 같은 고유벡터를 가져 CKM = 단위행렬(선도 차수). 쿼크 섞임이 작은 이유.
3. 렙톤: 푸리에 행렬 U_ω에 1–3 평면 π/4 회전을 곱하면 |U|²가 TBM(1/3, 1/2, 0 구조)과 정확히 같다.
   중성미자 섞임이 큰 이유. 이 사실 자체는 Harrison–Perkins–Scott 계열의 알려진 결과다.
4. 선형 중력: 중력이 |A|만 보면 텐서 모드 h'' + 3H h' + k²/a² h = 0이 표준이라 중력파 속도 = 광속,
   중력파·빛 광도거리 비 Ξ = 1.

python -B -m examples.physics.rendering.ce_rendering_generations
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.integrate import solve_ivp

from examples.physics.rendering import ce_rendering_gauge as GA
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R

OMEGA = cmath.exp(2j * math.pi / 3)
FOURIER = np.array([[OMEGA ** (a * i) for i in range(3)] for a in range(3)]) / math.sqrt(3)
TBM_SQ = np.array([[2 / 3, 1 / 3, 0.0], [1 / 6, 1 / 3, 1 / 2], [1 / 6, 1 / 3, 1 / 2]])


def generation_count() -> int:
    return 3  # 3차원 단계의 축 수(E1)


def circulant(s: float, eps: float, theta: float) -> np.ndarray:
    S = np.roll(np.eye(3), 1, axis=0)
    return s * np.eye(3) + eps * (cmath.exp(1j * theta / 3) * S + cmath.exp(-1j * theta / 3) * S.T)


def ckm_from_circulants(seed: int = 4) -> np.ndarray:
    """서로 다른 (s, ε, θ)의 위·아래 순환 질량 행렬에서 CKM 크기. 공통 고유벡터라 순열 행렬이 된다."""
    rng = np.random.default_rng(seed)
    mu = circulant(1.0, rng.uniform(0.05, 0.3), rng.uniform(0.1, 3.0))
    md = circulant(1.0, rng.uniform(0.05, 0.3), rng.uniform(0.1, 3.0))
    _, uu = np.linalg.eigh(mu)
    _, ud = np.linalg.eigh(md)
    return np.abs(uu.conj().T @ ud)


def lepton_mixing_leading() -> np.ndarray:
    c = s = 1 / math.sqrt(2)
    r13 = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return np.abs(FOURIER @ r13) ** 2


def tensor_mode_checks(k_over_h0: float = 200.0) -> dict:
    """|A| 배경에서 텐서 모드를 등각 시간으로 적분해 위상 속도와 진폭 감쇠(Ξ)를 확인한다."""
    c = R.core(R.calibrated_alpha_s()[0])
    om = c["Om"]
    hub = lambda a: math.sqrt(om / a ** 3 + 1 - om)       # H/H0
    k = k_over_h0

    def rhs(eta, y):   # y = (a, h, h')  with ' = d/dη, H0 = 1 units
        a, h, hp = y
        ap = a * a * hub(a)
        return [ap, hp, -2 * ap / a * hp - k * k * h]
    a0 = 0.2
    sol = solve_ivp(rhs, (0.0, 3.0), [a0, 1.0, 0.0], rtol=1e-10, atol=1e-12, dense_output=True,
                    events=lambda e, y: y[0] - 1.0)
    eta_end = float(sol.t_events[0][0])
    etas = np.linspace(0.0, eta_end, 20001)
    h = sol.sol(etas)[1]
    a = sol.sol(etas)[0]
    zero = np.where(np.diff(np.sign(h)) != 0)[0]
    zeros = [etas[i] - h[i] * (etas[i + 1] - etas[i]) / (h[i + 1] - h[i]) for i in zero]
    speed = math.pi / (k * float(np.mean(np.diff(zeros))))   # 영점 간격 π/(k c_T)
    amp_a = np.array([abs(h[i]) * a[i] for i in range(len(h)) if abs(sol.sol(etas[i])[2]) < 0.05 * k * abs(h).max()])
    xi = float(np.std(amp_a[len(amp_a) // 2:]) / np.mean(amp_a[len(amp_a) // 2:]))
    return {"c_T/c": speed, "amplitude*a spread (Xi-1 proxy)": xi}


def main() -> None:
    print("generations:", generation_count(), "| anomalies per generation:",
          {k: str(v) for k, v in GA.anomaly_sums().items()})
    print("|CKM| from two circulants (leading order):\n", np.round(ckm_from_circulants(), 12))
    print("|U_omega R13(pi/4)|^2:\n", np.round(lepton_mixing_leading(), 12))
    print("max deviation from TBM:", float(np.abs(lepton_mixing_leading() - TBM_SQ).max()))
    print("tensor modes:", tensor_mode_checks())


if __name__ == "__main__":
    main()
