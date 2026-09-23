"""급팽창 길이 N_e = (d/2)·D·N_gauge의 N_gauge를 렌더링 단계의 대칭군에서 계산한다. 원장: 43장 §43.23.

저장소 식(12_전이구간)은 N_gauge = 12를 입력으로 두었다. §43.20에서 단계 분해 V3 ⊕ V2를 보존하는 대칭군이
S(U(3)×U(2))임을 보였으므로, su(5) 안에서 단계 사영과 교환하는 생성자의 수를 직접 세어 12를 얻는다.
N_e = (3/2)·D·12 = 18 D가 레지스트리 값과 같은지 확인한다. 새 수치는 없다.

python -B -m examples.physics.rendering.ce_rendering_inflation
"""

from __future__ import annotations

import itertools

import numpy as np

from examples.physics.rendering import ce_rendering_registry as R


def su_basis(n: int) -> list[np.ndarray]:
    basis = []
    for i, j in itertools.combinations(range(n), 2):
        a = np.zeros((n, n), complex); a[i, j] = a[j, i] = 1
        b = np.zeros((n, n), complex); b[i, j] = -1j; b[j, i] = 1j
        basis += [a, b]
    for k in range(1, n):
        d = np.zeros((n, n), complex)
        d[:k, :k] = np.eye(k)
        d[k, k] = -k
        basis.append(d)
    return basis


def stage_preserving_dimension() -> int:
    """su(5)에서 단계 사영 P3 = diag(1,1,1,0,0)과 교환하는 부분대수의 차원."""
    p3 = np.diag([1, 1, 1, 0, 0]).astype(complex)
    basis = su_basis(5)
    comm = np.array([(p3 @ x - x @ p3).ravel() for x in basis]).T     # 선형 사상 x -> [P3, x]
    rank = np.linalg.matrix_rank(np.hstack([comm.real, comm.imag]), tol=1e-10)
    return len(basis) - rank


def inflation_efolds(c: dict) -> float:
    return 1.5 * c["D"] * stage_preserving_dimension()


def main() -> None:
    n = stage_preserving_dimension()
    c = R.core(R.calibrated_alpha_s()[0])
    print(f"dim su(5) = {len(su_basis(5))}; stage-preserving subalgebra dim = {n} (= 8 + 3 + 1)")
    print(f"N_e = (3/2) D N_gauge = {inflation_efolds(c):.5f}; registry N_e = 18 D = {c['Ne']:.5f}")


if __name__ == "__main__":
    main()
