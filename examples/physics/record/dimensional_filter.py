"""각도에 따른 1·2·3성분 필터. 실행 결과 파일은 생성하지 않는다.

입력은 C^3, 출력은 C^1 ⊕ C^2 ⊕ C^3 ⊕ C^3(비선택)이다.
주어진 unitary frame의 첫 r개 열을 V_r라 하면

    T_r = sqrt(w_r) V_r†,   K_Q = sqrt(I - sum_r T_r† T_r),
    W = (T_1; T_2; T_3; K_Q), W† W = I.

K_Q는 비선택 출력의 진폭 연산자이며, 원고의 사영 Q = I - P와는 구별한다.

회전은 통과하는 성분의 조합을 바꾼다. 통로의 rank 1·2·3은 설계 입력이며,
이 숫자가 자연에서 선택되는 이유나 세 힘의 결합값을 유도한 것은 아니다.
전체 출력 W rho W†는 가지 사이 중첩까지 보존한다. 조건부 상태는 해당
출력을 판독했다고 가정할 때만 사용하며, 임의의 실제 결과를 추첨하지 않는다.

9모드 unitary와 무차원 Hermitian 생성자 G도 제공한다. U = exp(-iG)는
선언한 상호작용 종료점의 필터다. 지속시간 tau를 공급하면 H = hbar G/tau로
실현할 수 있지만, 물리적 시간척도·우주 곡률·불가역 기록은 여기서 정하지 않는다.

python -B -m examples.physics.record.dimensional_filter --angles-deg 20 30 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


_TOL = 1.0e-12
_SLICES = (slice(0, 1), slice(1, 3), slice(3, 6), slice(6, 9))


def frame_from_angles(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """라디안 각도로 Rz(yaw) Ry(pitch) Rx(roll)을 만든다."""
    angles = np.asarray((yaw, pitch, roll), dtype=float)
    if not np.isfinite(angles).all():
        raise ValueError("angles must be finite")
    cy, cp, cr = np.cos(angles)
    sy, sp, sr = np.sin(angles)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _density(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=complex)
    if not np.isfinite(value).all():
        raise ValueError("state must be finite")
    if value.shape == (3,):
        value = np.outer(value, value.conj())
    if value.shape != (3, 3):
        raise ValueError("state must have shape (3,) or (3, 3)")
    if not np.allclose(value, value.conj().T, atol=_TOL, rtol=0):
        raise ValueError("density must be Hermitian")
    if not np.isclose(np.trace(value), 1, atol=_TOL, rtol=0):
        raise ValueError("state must be normalized")
    if np.linalg.eigvalsh(value).min() < -_TOL:
        raise ValueError("density must be positive semidefinite")
    return value


@dataclass(frozen=True)
class FilterOutput:
    """전체 상태와, Born 판독을 가정했을 때의 네 조건부 출력."""

    density: np.ndarray
    probabilities: tuple[float, ...]
    conditional_states: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class DimensionalFilter:
    """배열 순서는 한 성분, 두 성분, 세 성분, 비선택 성분이다."""

    operators: tuple[np.ndarray, ...]
    unitary: np.ndarray
    generator: np.ndarray

    @property
    def isometry(self) -> np.ndarray:
        return np.vstack(self.operators)

    def apply(self, state: np.ndarray) -> FilterOutput:
        rho = _density(state)
        output = self.isometry @ rho @ self.isometry.conj().T
        probabilities, conditional = [], []
        for section in _SLICES:
            block = output[section, section]
            probability = max(0.0, float(np.trace(block).real))
            probabilities.append(probability)
            conditional.append(block / probability if probability > 0 else None)
        return FilterOutput(output, tuple(probabilities), tuple(conditional))

    def evolution(self, fraction: float = 1.0) -> np.ndarray:
        """상호작용 시간/tau에 대한 unitary. fraction=1에서 필터가 된다."""
        if not np.isfinite(fraction):
            raise ValueError("interaction fraction must be finite")
        values, vectors = np.linalg.eigh(self.generator)
        return (vectors * np.exp(-1j * fraction * values)) @ vectors.conj().T


def build_filter(
    frame: np.ndarray | None = None,
    weights: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> DimensionalFilter:
    """세 중첩 부분공간을 동시에 읽는, 비선택 출력을 포함한 필터.

    weights는 분기 결합의 제곱이며 실제 출력 확률은 입력 상태에도 의존한다.
    각 값은 비음이고 합은 1 이하여야 한다. 0인 분기는 꺼진다.
    frame은 복소 unitary도 허용하며 입력 좌표계에서 필터 축을 지정한다.
    """
    basis = np.eye(3, dtype=complex) if frame is None else np.array(frame, dtype=complex)
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise ValueError("frame must be a finite 3 by 3 matrix")
    if not np.allclose(basis.conj().T @ basis, np.eye(3), atol=_TOL, rtol=0):
        raise ValueError("frame must be unitary")
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() > 1:
        raise ValueError("weights must be three nonnegative finite values with sum <= 1")

    remaining = np.array([1 - w.sum(), 1 - w[1:].sum(), 1 - w[2]])
    residual = (basis * np.sqrt(remaining)) @ basis.conj().T
    transmitted = tuple(np.sqrt(w[r - 1]) * basis[:, :r].conj().T for r in (1, 2, 3))

    # 각 입력 모드를, 그 모드가 접근할 수 있는 출력의 조합과 회전시킨다.
    # 출력 0..5와 비선택 6..8의 서로 직교하는 세 평면을 사용한다.
    rotation = np.eye(9, dtype=complex)
    generator = np.zeros((9, 9), dtype=complex)
    starts = (0, 1, 3)
    for j in range(3):
        strength = float(w[j:].sum())
        if strength == 0:
            continue
        sine, cosine = np.sqrt(strength), np.sqrt(remaining[j])
        bright, unseen = np.zeros(9, complex), np.zeros(9, complex)
        for r in range(j, 3):
            bright[starts[r] + j] = np.sqrt(w[r]) / sine
        unseen[6 + j] = 1
        plane = np.outer(bright, bright.conj()) + np.outer(unseen, unseen.conj())
        turn = np.outer(bright, unseen.conj()) - np.outer(unseen, bright.conj())
        rotation += (cosine - 1) * plane + sine * turn
        generator += 1j * np.arctan2(sine, cosine) * turn
    orientation = np.eye(9, dtype=complex)
    orientation[6:, 6:] = basis
    return DimensionalFilter(
        operators=transmitted + (residual,),
        unitary=orientation @ rotation @ orientation.conj().T,
        generator=orientation @ generator @ orientation.conj().T,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="전체 상태를 보존하는 1·2·3성분 필터")
    parser.add_argument("--angles-deg", nargs=3, type=float, default=(0, 0, 0),
                        metavar=("YAW", "PITCH", "ROLL"))
    parser.add_argument("--weights", nargs=3, type=float, default=(0.25, 0.25, 0.25))
    parser.add_argument("--state", nargs=3, type=complex, default=(1, 1j, 1),
                        help="복소 진폭 세 개; CLI에서 길이 1로 정규화")
    args = parser.parse_args()
    try:
        frame = frame_from_angles(*np.deg2rad(args.angles_deg))
        model = build_filter(frame, tuple(args.weights))
        state = np.array(args.state, complex)
        norm = np.linalg.norm(state)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("state must be finite and nonzero")
        result = model.apply(state / norm)
    except ValueError as error:
        parser.error(str(error))
    for label, probability, operator in zip(
        ("한 성분", "두 성분", "세 성분", "비선택"), result.probabilities, model.operators
    ):
        print(f"{label}: 확률 {probability:.9f}, 통로 rank {np.linalg.matrix_rank(operator)}")
    error = np.max(np.abs(model.isometry.conj().T @ model.isometry - np.eye(3)))
    print(f"확률 합: {sum(result.probabilities):.12f}; 상태 보존 잔차: {error:.3e}")
    print("각도·분기 강도는 입력입니다. 세 힘의 값이나 우주 곡률의 예측은 아닙니다.")


if __name__ == "__main__":
    main()
