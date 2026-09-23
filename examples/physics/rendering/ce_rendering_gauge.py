"""렌더링 단계에서 표준모형 게이지군·한 세대·이상 상쇄. 원장: 43장 §43.20.

공리 C6: 렌더링된 공간은 3차원 단계와 2차원 단계의 합 V = V3 ⊕ V2 = C^5이고, 통로는 외대수 Λ(V)의
기저(부분집합)다. 1차원 단계는 두 블록 사이의 상대 위상이다.

아래는 이 공리에서 계산으로 확인하는 사실이다. 대수 구조 자체는 알려진 수학(Georgi–Glashow SU(5),
Baez–Huerta의 GUT 대수)이며, CE의 기여는 렌더링 단계와 통로를 이 구조와 동일시한 것이다.
1. 단계 분해를 보존하고 행렬식 1인 유니터리 군은 S(U(3)×U(2)) = (SU(3)×SU(2)×U(1))/Z6.
2. 그 U(1) 생성자는 초전하 Y = (약 통로 수)/2 − (색 통로 수)/3.
3. Λ^even(V)의 16상태가 표준모형 한 세대(오른손 중성미자 포함)와 (색, 약, Y)까지 정확히 같다.
4. 게이지 이상이 모두 상쇄된다. 통일 척도의 sin²θ_W = tr T3² / tr Q² = 3/8.
5. 렌더링 사상 W(차원 필터)는 등거리이므로 완전 양성·대각합 보존이며 신호를 보내지 않는다.

python -B -m examples.physics.rendering.ce_rendering_gauge
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from fractions import Fraction

import numpy as np

COLOR, WEAK = (0, 1, 2), (3, 4)
SM_GENERATION = Counter({  # (SU3 차원, SU2 차원, Y): 왼손 Weyl 성분 수
    (3, 2, Fraction(1, 6)): 6, (3, 1, Fraction(-2, 3)): 3, (3, 1, Fraction(1, 3)): 3,
    (1, 2, Fraction(-1, 2)): 2, (1, 1, Fraction(1)): 1, (1, 1, Fraction(0)): 1,
})


def subsets(grade_parity: int | None = None) -> list[tuple[int, ...]]:
    out = []
    for k in range(6):
        if grade_parity is not None and k % 2 != grade_parity:
            continue
        out += list(itertools.combinations(range(5), k))
    return out


def hypercharge(s: tuple[int, ...]) -> Fraction:
    w = sum(1 for i in s if i in WEAK)
    c = sum(1 for i in s if i in COLOR)
    return Fraction(w, 2) - Fraction(c, 3)


def multiplet_content(parity: int = 0) -> Counter:
    """Λ^even(또는 odd)를 (색 부분 Λ^c(C3)의 차원, 약 부분 Λ^w(C2)의 차원, Y)로 묶어 센다."""
    content: Counter = Counter()
    for s in subsets(parity):
        c = sum(1 for i in s if i in COLOR)
        w = sum(1 for i in s if i in WEAK)
        content[(math.comb(3, c), math.comb(2, w), hypercharge(s))] += 1
    return content


def matches_one_generation() -> bool:
    return multiplet_content(0) == SM_GENERATION


def anomaly_sums() -> dict:
    ys = []
    su3, su2 = Fraction(0), Fraction(0)
    for (d3, d2, y), n in SM_GENERATION.items():
        ys += [y] * n
        if d3 == 3:
            su3 += Fraction(n, 3) * y      # 색 삼중항마다 T(R)=1/2, 성분 수로 나눠 가중
        if d2 == 2:
            su2 += Fraction(n, 2) * y
    content = multiplet_content(0)
    ys_derived = [y for (d3, d2, y), n in content.items() for _ in range(n)]
    return {"Y": sum(ys_derived), "Y3": sum(y ** 3 for y in ys_derived), "SU3^2 Y": su3, "SU2^2 Y": su2,
            "grav-Y": sum(ys_derived)}


def unification_sin2() -> Fraction:
    """16상태에서 tr T3² / tr Q², Q = T3 + Y."""
    t3sq, qsq = Fraction(0), Fraction(0)
    for s in subsets(0):
        w = [i for i in s if i in WEAK]
        t3 = Fraction(0) if len(w) != 1 else (Fraction(1, 2) if w[0] == 3 else Fraction(-1, 2))
        q = t3 + hypercharge(s)
        t3sq += t3 * t3
        qsq += q * q
    return t3sq / qsq


def fock_operator(x: np.ndarray) -> np.ndarray:
    """C^5 위의 연산 x를 Λ(C^5)(32차원)에 유도: Σ x_ij a_i† a_j (Jordan–Wigner)."""
    n = 5
    I2, Z = np.eye(2), np.diag([1.0, -1.0])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])

    def a(i):
        mats = [Z] * i + [lower] + [I2] * (n - i - 1)
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out
    ops = [a(i) for i in range(n)]
    return sum(x[i, j] * ops[i].conj().T @ ops[j] for i in range(n) for j in range(n))


def gauge_structure_checks(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    y = np.diag([-1 / 3] * 3 + [1 / 2] * 2)
    Y = fock_operator(y)

    def random_su(m):
        h = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        h = h + h.conj().T
        return h - np.trace(h) / m * np.eye(m)
    g3 = np.zeros((5, 5), complex); g3[:3, :3] = random_su(3)
    g2 = np.zeros((5, 5), complex); g2[3:, 3:] = random_su(2)
    G3, G2 = fock_operator(g3), fock_operator(g2)
    mixing = np.zeros((5, 5), complex); mixing[0, 3] = mixing[3, 0] = 1.0
    number = fock_operator(np.eye(5))
    parity = np.diag(np.exp(1j * np.pi * np.diag(number).real))
    comm = lambda A, B: np.abs(A @ B - B @ A).max()
    return {"[Y,su3]": comm(Y, G3), "[Y,su2]": comm(Y, G2), "[su3,su2]": comm(G3, G2),
            "[Y,stage-mixing]": comm(Y, fock_operator(mixing)), "trace y": float(np.trace(y)),
            "even sector invariant": comm(parity, G3 + G2 + Y)}


def rendering_channel_checks(seed: int = 11) -> dict:
    """차원 필터 W = (T1;T2;T3;K_Q): 등거리, Choi 양성, 무신호."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    w = np.array([0.2, 0.3, 0.25])
    blocks = [math.sqrt(w[r - 1]) * q[:, :r].conj().T for r in (1, 2, 3)]
    rest = np.eye(3) - sum(b.conj().T @ b for b in blocks)
    vals, vecs = np.linalg.eigh(rest)
    kq = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ vecs.conj().T
    W = np.vstack(blocks + [kq])                       # (1+2+3+3) x 3
    iso = np.abs(W.conj().T @ W - np.eye(3)).max()
    choi = sum(np.kron(np.outer(e_i, e_j), W @ np.outer(e_i, e_j) @ W.conj().T)
               for e_i in np.eye(3) for e_j in np.eye(3))
    choi_min = float(np.linalg.eigvalsh((choi + choi.conj().T) / 2).min())
    psi = rng.normal(size=9) + 1j * rng.normal(size=9)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())                    # A(3) ⊗ B(3)
    out = np.kron(W, np.eye(3)) @ rho @ np.kron(W, np.eye(3)).conj().T
    rb_before = np.einsum("ijik->jk", rho.reshape(3, 3, 3, 3))
    rb_after = np.einsum("ijik->jk", out.reshape(9, 3, 9, 3))
    return {"isometry error": float(iso), "choi min eigenvalue": choi_min,
            "no-signalling error": float(np.abs(rb_after - rb_before).max())}


def main() -> None:
    print("one SM generation from Λ^even(V3⊕V2):", matches_one_generation())
    for k, v in sorted(multiplet_content(0).items(), key=lambda kv: (-kv[0][0], -kv[0][1], kv[0][2])):
        print(f"   (SU3 {k[0]}, SU2 {k[1]}, Y {str(k[2]):>5s}) x{v}")
    print("anomaly sums:", {k: str(v) for k, v in anomaly_sums().items()})
    print("sin^2 theta_W at unification:", unification_sin2())
    print("gauge structure:", {k: (round(v, 14) if isinstance(v, float) else v) for k, v in gauge_structure_checks().items()})
    print("rendering channel:", rendering_channel_checks())


if __name__ == "__main__":
    main()
