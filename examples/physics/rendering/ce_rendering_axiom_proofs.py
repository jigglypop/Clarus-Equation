"""공리 QG1·C6·L·이등분의 증명 검산. 원장: 43장 §43.35. 예측값을 바꾸지 않는다.

A (QG1). 되먹임이 있는 중력 원천 s가 가지 분해 {p_k, ψ_k}에 의존하면, HJW 정리로 앨리스가 같은 ρ_B의 다른 분해를
   원격 준비할 수 있어 신호가 생긴다. 무신호 ⇒ s는 ρ_B만의 함수. 여기에 확률의 무게(ρ에 아핀) ⇒ s = Tr(ρ T̂)(유일).
   검산: 무작위 분해에서 (i) 가지 원천의 신호, (ii) ρ 원천의 무신호, (iii) 아핀 범함수 = Tr(ρX)의 재구성.
B (C6). 평균 0, ‖f‖₂ 고정, 2π 주기 실함수 중 ∫f'²를 최소로 하는 것은 a cos φ + b sin φ (비르팅거). 우함수 ⇒ cos φ.
   NEC ⇒ 부호 +. 검산: 무작위 우함수 삼각다항식의 비 ∫f'²/∫f² ≥ 1, 등호는 cos만.
C (L). 허수 빠르기 iφ의 속도 매개변수 tanh(iφ) = i tan φ, |tan φ| < 1 ⇔ |φ| < π/4.
D (이등분). argmin_θ max(1 − cos θ, 1 − cos(π/4 − θ)) = π/8(유일).

python -B -m examples.physics.rendering.ce_rendering_axiom_proofs
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)


def _haar_unitary(rng: np.random.Generator, n: int = 2) -> np.ndarray:
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def steered_decomposition(psi_ab: np.ndarray, u_alice: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """앨리스가 기저 u_alice의 열로 측정할 때 밥의 (확률, 조건부 상태)."""
    m = psi_ab.reshape(2, 2)
    out = []
    for k in range(2):
        v = u_alice[:, k].conj() @ m
        p = float(np.vdot(v, v).real)
        out.append((p, v / math.sqrt(p)))
    return out


def bob_after_backreaction(decomp, rule: str, kappa: float = 1.0, t: float = 0.9) -> np.ndarray:
    rho_b = sum(p * np.outer(v, v.conj()) for p, v in decomp)
    out = np.zeros((2, 2), complex)
    for p, v in decomp:
        r = np.outer(v, v.conj())
        s = np.real(np.trace((r if rule == "branch" else rho_b) @ SZ))
        u = expm(-1j * kappa * s * SX * t)
        out += p * u @ r @ u.conj().T
    return out


def qg1_signalling_scan(trials: int = 200, seed: int = 3) -> dict:
    """무작위 얽힌 상태·무작위 앨리스 기저 쌍에서 밥 평균 상태의 최대 차이."""
    rng = np.random.default_rng(seed)
    worst = {"branch": 0.0, "weight": 0.0}
    for _ in range(trials):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        u1, u2 = _haar_unitary(rng), _haar_unitary(rng)
        for rule in worst:
            d = bob_after_backreaction(steered_decomposition(psi, u1), rule) - \
                bob_after_backreaction(steered_decomposition(psi, u2), rule)
            worst[rule] = max(worst[rule], float(0.5 * np.abs(np.linalg.eigvalsh(d)).sum()))
    return worst


def affine_functional_is_trace_form(seed: int = 5) -> float:
    """아핀 범함수 F(ρ) = Tr(ρX)를 파울리 기저의 네 값에서 재구성하고 무작위 ρ에서의 오차."""
    rng = np.random.default_rng(seed)
    x_true = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    x_true = (x_true + x_true.conj().T) / 2
    f = lambda rho: float(np.real(np.trace(rho @ x_true)))
    basis = [np.eye(2) / 2, (np.eye(2) + SX) / 2, (np.eye(2) + SY) / 2, (np.eye(2) + SZ) / 2]
    c0 = 2 * f(basis[0])
    cx, cy, cz = (2 * f(b) - c0 for b in basis[1:])
    x_rec = (c0 * np.eye(2) + cx * SX + cy * SY + cz * SZ) / 2
    err = 0.0
    for _ in range(50):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0, 1)
        rho = (np.eye(2) + v[0] * SX + v[1] * SY + v[2] * SZ) / 2
        err = max(err, abs(f(rho) - float(np.real(np.trace(rho @ x_rec)))))
    return err


def wirtinger_scan(trials: int = 2000, kmax: int = 6, seed: int = 11) -> dict:
    """평균 0 우함수 f = Σ a_k cos kφ에서 ∫f'²/∫f² = Σ k² a_k² / Σ a_k² ≥ 1, 등호는 k=1만."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(trials):
        a = rng.normal(size=kmax)
        k = np.arange(1, kmax + 1)
        ratios.append(float((k ** 2 * a ** 2).sum() / (a ** 2).sum()))
    return {"min_ratio_random": min(ratios), "ratio_cos": 1.0, "ratio_cos2": 4.0}


def imaginary_rapidity_speed(phi: float) -> complex:
    return cmath.tanh(1j * phi)


def minimax_bisector(limit: float = math.pi / 4) -> float:
    res = minimize_scalar(lambda th: max(1 - math.cos(th), 1 - math.cos(limit - th)),
                          bounds=(0.0, limit), method="bounded", options={"xatol": 1e-12})
    return float(res.x)


def main() -> None:
    print("A. signalling (max trace distance over random states/bases):", qg1_signalling_scan())
    print(f"A. affine functional reconstructed as Tr(rho X): max error {affine_functional_is_trace_form():.2e}")
    print("B. Wirtinger:", wirtinger_scan())
    for phi in (0.3, math.pi / 4, 1.0):
        v = imaginary_rapidity_speed(phi)
        print(f"C. phi={phi:.4f}: tanh(i phi) = {v.imag:+.6f} i, |v|<1: {abs(v) < 1 - 1e-12}")
    print(f"D. minimax readout angle = {minimax_bisector():.12f} (pi/8 = {math.pi / 8:.12f})")


if __name__ == "__main__":
    main()
