"""양자중력 = 확률의 무게. 원장: 43장 §43.30, 사전 등록 v10.

QG1(사용자 제시, 후보 공리): 중력의 원천은 확정된 결과가 아니라 확률 분포 전체의 무게다.
무게를 세는 두 방식을 계산 전에 적었다.
 (가) 가지 무게: 측정 가지마다 제 기대값으로 중력을 낸다(순수 상태 슈뢰딩거–뉴턴).
 (나) 확률 무게: 국소 밀도행렬 ρ(확률의 가중합)만 중력을 낸다.
검사:
1. 무신호(L, §43.29): 벨 쌍에서 앨리스의 측정 기저(Z/X)가 밥의 평균 상태를 바꾸는지. 밥은 H = κ⟨σ_z⟩σ_x를 느낀다.
2. BMV: m = 1e-14 kg, Δx = 250 μm, d = 450 μm, t = 2.5 s에서 양자 중력의 얽힘과 확률 무게(평균장)의 얽힘.
3. 우주론: 확률 무게의 합 = 1 → Ω_k = 0; 빛에 보이는 몫 q = Ω_b, 나머지 1 − q는 중력으로만 보인다.

python -B -m examples.physics.rendering.ce_rendering_probability_weight
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from examples.physics.rendering import ce_rendering_registry as R

SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
G, HBAR = 6.67430e-11, 1.054571817e-34


def _evolve(rho: np.ndarray, source_z: float, kappa: float, t: float) -> np.ndarray:
    u = expm(-1j * kappa * source_z * SX * t)
    return u @ rho @ u.conj().T


def bob_average_state(basis: str, rule: str, kappa: float = 1.0, t: float = math.pi / 4) -> np.ndarray:
    """벨 쌍 |Φ+⟩에서 앨리스가 basis로 측정한 뒤 밥의 평균 상태. rule: 'branch'(가) 또는 'weight'(나)."""
    if basis == "Z":
        branches = [(0.5, np.array([1, 0], complex)), (0.5, np.array([0, 1], complex))]
    else:
        branches = [(0.5, np.array([1, 1], complex) / math.sqrt(2)), (0.5, np.array([1, -1], complex) / math.sqrt(2))]
    rho_local = sum(p * np.outer(v, v.conj()) for p, v in branches)
    out = np.zeros((2, 2), complex)
    for p, v in branches:
        rho = np.outer(v, v.conj())
        src = np.real(np.trace(rho @ SZ)) if rule == "branch" else np.real(np.trace(rho_local @ SZ))
        out += p * _evolve(rho, src, kappa, t)
    return out


def signalling(rule: str) -> float:
    """앨리스의 기저 선택이 만드는 밥 평균 상태의 대각합 거리(0이면 무신호)."""
    d = bob_average_state("Z", rule) - bob_average_state("X", rule)
    return float(0.5 * np.abs(np.linalg.eigvalsh(d)).sum())


def bmv(m: float = 1e-14, dx: float = 250e-6, d: float = 450e-6, t: float = 2.5) -> dict:
    """두 질량(각각 L/R 중첩)의 얽힘. 양자 중력은 위치쌍마다 위상, 확률 무게는 상대의 확률 분포로 만든 평균장."""
    xa, xb = (0.0, dx), (dx + d, 2 * dx + d)
    phase = lambda i, j: G * m * m * t / (HBAR * abs(xb[j] - xa[i]))
    psi_q = np.array([math.e ** 0 * np.exp(1j * phase(i, j)) for i in range(2) for j in range(2)]) / 2
    conc_q = 2 * abs(psi_q[0] * psi_q[3] - psi_q[1] * psi_q[2])
    # 평균장: A의 각 위치는 B의 확률 분포(1/2, 1/2)가 만드는 퍼텐셜을 느낀다(대칭).
    pa = [sum(0.5 * phase(i, j) for j in range(2)) for i in range(2)]
    pb = [sum(0.5 * phase(i, j) for i in range(2)) for j in range(2)]
    psi_w = np.array([np.exp(1j * (pa[i] + pb[j])) for i in range(2) for j in range(2)]) / 2
    conc_w = 2 * abs(psi_w[0] * psi_w[3] - psi_w[1] * psi_w[2])
    delta = phase(1, 0) + phase(0, 1) - phase(0, 0) - phase(1, 1)
    return {"concurrence_quantum_gravity": conc_q, "concurrence_probability_weight": conc_w,
            "entangling_phase_rad": delta, "closest_pair_phase_rad": phase(1, 0)}


def cosmic_weights(c: dict) -> dict:
    return {"rendered (light-visible) = Omega_b = q": c["q"], "gravity-only weight = 1 - q": 1 - c["q"],
            "Omega_total": c["Om"] + (1 - c["Om"]), "Omega_k": 0.0,
            "Omega_k pull vs Planck2018+BAO 0.0007+/-0.0019": (0.0 - 0.0007) / 0.0019}


def main() -> None:
    for rule in ("branch", "weight"):
        print(f"signalling [{rule}] = {signalling(rule):.6f}")
    print("BMV:", {k: round(v, 6) for k, v in bmv().items()})
    c = R.core(R.calibrated_alpha_s()[0])
    print("cosmic weights:", {k: round(v, 5) for k, v in cosmic_weights(c).items()})


if __name__ == "__main__":
    main()
