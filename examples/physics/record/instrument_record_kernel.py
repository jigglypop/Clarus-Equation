"""에너지 분해 양자 기구(quantum instrument) 기록 커널과 유한 보른(Born) 선택 커널을 구성한다.

첫째 부분은 record_dust_bridge의 표식 기록 계약(marked record contract)에 대한
명시적 상류 실현 하나를 제공한다. 기구의 선택은 입력의 일부다. 비선택
채널(nonselective channel)은 유일한 결과 트리를 결정하지 않는다.

정규화된 상태 rho_a와 크라우스 가지(Kraus branch) K_b|a에 대해

    P_b|a = Tr(K_b|a rho_a K_b|a^dagger)

는 무차원 보른 전이 확률이다. 완비 족(complete family)이 조화 에너지
장부(harmonic energy ledger)를 갖는 조건은

    sum_b K_b|a^dagger (H + Delta E_b I) K_b|a = H.

이다. 양자 비파괴(QND) 기록에서는 Delta E_b가 0이고 모든 크라우스 연산자가
H와 교환한다. 측정된 에너지 보존 충돌(energy-conserving collision)에서는
Delta E_b가 새 보조계(ancilla)의 에너지 증가분이다. 이때 계의 에너지는 바뀔 수
있으나 계와 기록을 합한 에너지는 조화적으로 유지된다.

내부 노드의 에너지는 여전히 조건부 기댓값에 불과하다. 말단 기록은 그 지점에서
Var(H)가 소멸할 때에만 정확한 질량껍질(mass-shell) 다리에 공급될 수 있다.
H의 스펙트럼이 물리적 에너지 단위를 제공한다. 무차원 보른 확률은 그 절대
척도를 결정할 수 없다.

수용된 모든 유한 트리에 대해 이 모듈은 표준 추가 전용(append-only) 이력
레지스터(history register)도 구성한다. 그 정규 직교 기저는 완전한 선언 결과
이력으로 색인되며, 지지 흐름(supported flow)에서 생략된 확률 0 또는 임계
이하 이력도 포함한다. 따라서 대각 레지스터 대수는 정확히 가환이고, 연쇄
크라우스 연산자는 수치적 스타인스프링 등거리(Stinespring isometry)와 보른
원통 확률(Born cylinder probability) 증서를 준다. 이는 추상적 기록 대수
구성이며, 견고한 물리적 포인터(pointer), 시공간 매장, 질량껍질 표식을
유도하지 않는다.

H만 주어지면 ``construct_luders_energy_instrument``가 가장 성긴 상이 에너지
스펙트럼 PVM을 정준 실현 하나로 준다. 이것이 물리적 기구를 유일하게 만들지는
않는다. 축퇴 에너지 부문 안의 세분화와 유니터리 운동은 추가 선택으로 남는다.
수치적으로 모호한 근접 축퇴는 조용히 병합되지 않고 거부된다.

둘째 부분은 유한 군집 기구(grouped instrument) ``{I_a}``에 대해 주어진 무차원
씨앗(seed) ``u in [0, 1)``를 보른 길이의 반열린 구간으로 분할한다.

    p_a = Tr I_a(rho),       I_a = [C_{a-1}, C_a).

따라서 균일 씨앗은 보른 분포를 표본추출하고, 정규화된 사후 상태의 씨앗 평균은
비선택 CPTP 채널과 같다. 부동소수점 산술에서는 원시 보른 대각합과 명시적으로
정규화된 구간 가중치를 따로 노출한다. 씨앗 법칙은 명시적 확률 공리다.
유니터리 사전측정도, 이 역 누적분포(inverse-CDF) 표현도 물리적 무작위성이나
존재론적 단일 세계를 유도하지 않는다.

유한 증서는 주어진 에너지 보존 충돌도 재사용한다. 에너지 잔차와 분산은 무차원
허용오차와 비교하기 전에 독립적인 양의 에너지 척도로 나눈다. 국소 벨 쌍(Bell
pair)의 비선택 주변 상태를 검사하고 두 반례를 고정한다. 제어 가능한 씨앗은
원격 조건부 상태를 강제할 수 있고, 결과 스칼라 에너지 영수증은 임의의 측정
에너지 장부를 닫을 수 없다.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.record.record_dust_bridge import (
    CausalRecordFlow,
    CausalRecordNode,
    CausalTransition,
    construct_conserved_record_flow,
)


INSTRUMENT_DEFAULT_TOLERANCE = 1.0e-10  # 무차원 상대 수치 허용오차.


@dataclass(frozen=True)
class KrausBranch:
    """선택된 양자 기구의 명시적으로 이름 붙은 결과 하나."""

    target: str
    label: str
    operator: np.ndarray
    energy_transfer: float = 0.0


@dataclass(frozen=True)
class RecordInstrument:
    """기록 트리의 한 노드에서 선택된 크라우스 분해."""

    node: str
    instrument_id: str
    branches: tuple[KrausBranch, ...]


@dataclass(frozen=True)
class ConditionalEnergyState:
    node: str
    density: np.ndarray
    probability_from_root: float
    system_energy_expectation: float
    cumulative_energy_transfer: float
    energy_expectation: float
    energy_variance: float


@dataclass(frozen=True)
class PersistentRecordHistory:
    """추가 전용 이력 레지스터의 직교 기저 라벨 하나."""

    basis_index: int
    terminal_node: str
    path: tuple[str, ...]
    probability: float
    supported_by_root: bool


@dataclass(frozen=True)
class ClassicalRecordAlgebraCertificate:
    """선언된 기구 트리에 대한 유한 스타인스프링·이력 증서.

    이력 기저와 그 대각 대수는 수학적 구성물이다. 직교성과 추가 전용 라벨링은
    실제 장치가 기록을 견고하게 저장함을 증명하지 않으므로
    physical_pointer_dynamics_derived 플래그는 거짓으로 남는다.
    """

    histories: tuple[PersistentRecordHistory, ...]
    node_probabilities: tuple[tuple[str, float], ...]
    global_isometry_residual: float
    probability_normalization_residual: float
    max_history_probability_residual: float
    max_prefix_probability_residual: float
    orthogonal_history_basis: bool
    commutative_diagonal_algebra: bool
    append_only_history_labels: bool
    physical_pointer_dynamics_derived: bool

    def probability(self, node: str) -> float:
        for candidate, probability in self.node_probabilities:
            if candidate == node:
                return probability
        raise KeyError(node)


@dataclass(frozen=True)
class QuantumKernelCertificate:
    """인증된 보른 커널과 그 하류 키르히호프(Kirchhoff) 흐름."""

    root: str
    nodes: tuple[CausalRecordNode, ...]
    transitions: tuple[CausalTransition, ...]
    conditional_states: tuple[ConditionalEnergyState, ...]
    instrument_ids: tuple[tuple[str, str], ...]
    terminal_nodes: tuple[str, ...]
    terminal_energy_resolved: bool
    qnd_required: bool
    hamiltonian_energy_scale: float
    hamiltonian_energy_min: float
    hamiltonian_energy_max: float
    max_completeness_residual: float
    max_energy_channel_residual: float
    max_relative_energy_channel_residual: float
    max_qnd_commutator_residual: float
    max_relative_qnd_commutator_residual: float
    support_probability_tolerance: float
    record_algebra: ClassicalRecordAlgebraCertificate
    flow: CausalRecordFlow

    def state(self, node: str) -> ConditionalEnergyState:
        for candidate in self.conditional_states:
            if candidate.node == node:
                return candidate
        raise KeyError(node)


@dataclass(frozen=True)
class LudersInstrumentConstruction:
    """가장 성긴 스펙트럼 뤼더스(Lüders) 기구에 대한 수치 증서."""

    instrument: RecordInstrument
    spectral_energies: tuple[float, ...]
    spectral_multiplicities: tuple[int, ...]
    hamiltonian_energy_scale: float
    max_projector_idempotence_residual: float
    max_projector_orthogonality_residual: float
    resolution_of_identity_residual: float
    max_hamiltonian_commutator_residual: float
    max_relative_hamiltonian_commutator_residual: float
    max_spectral_cluster_width: float
    max_relative_spectral_cluster_width: float
    max_eigenprojector_residual: float
    max_relative_eigenprojector_residual: float


@dataclass(frozen=True)
class EnergyConservingCollisionConstruction:
    """측정된 새 보조계 충돌 하나가 유도하는 크라우스 기구."""

    instrument: RecordInstrument
    initial_pointer_index: int
    initial_pointer_energy: float
    pointer_energies: tuple[float, ...]
    branch_energy_transfers: tuple[float, ...]
    pointer_basis_unitarity_residual: float
    collision_unitarity_residual: float
    total_energy_commutator_residual: float
    relative_total_energy_commutator_residual: float
    max_relative_pointer_energy_variance: float
    kraus_completeness_residual: float
    ledger_identity_residual: float
    relative_ledger_identity_residual: float
    physical_pointer_persistence_derived: bool


def _square_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f'{name} must be a non-empty square matrix')
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'{name} must have finite entries')
    return matrix


def _validate_density(
    value: np.ndarray,
    *,
    dimension: int,
    tolerance: float,
    name: str,
) -> np.ndarray:
    density = _square_matrix(value, name)
    if density.shape != (dimension, dimension):
        raise ValueError(f'{name} has the wrong Hilbert-space dimension')
    if np.linalg.norm(density - density.conj().T, ord='fro') > tolerance:
        raise ValueError(f'{name} must be Hermitian')
    eigenvalues = np.linalg.eigvalsh(density)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f'{name} must be positive semidefinite')
    trace = np.trace(density)
    if abs(float(trace.imag)) > tolerance or not math.isclose(
        float(trace.real), 1.0, rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f'{name} must have unit trace')
    return density


def _real_trace(
    value: np.ndarray,
    *,
    tolerance: float,
    name: str,
    scale: float = 1.0,
) -> float:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'{name} trace scale must be finite and positive')
    trace = np.trace(value)
    if abs(float(trace.imag)) > tolerance * scale:
        raise ArithmeticError(f'{name} has a non-real trace')
    result = float(trace.real)
    if not math.isfinite(result):
        raise ArithmeticError(f'{name} trace is not finite')
    return result


def _energy_state(
    node: str,
    density: np.ndarray,
    probability_from_root: float,
    hamiltonian: np.ndarray,
    *,
    cumulative_energy_transfer: float,
    energy_scale: float,
    tolerance: float,
) -> ConditionalEnergyState:
    system_energy = _real_trace(
        hamiltonian @ density,
        tolerance=tolerance,
        name=f'energy at {node}',
        scale=energy_scale,
    )
    second_moment = _real_trace(
        hamiltonian @ hamiltonian @ density,
        tolerance=tolerance,
        name=f'energy second moment at {node}',
        scale=energy_scale**2,
    )
    variance = second_moment - system_energy * system_energy
    if variance < -tolerance * energy_scale**2:
        raise ArithmeticError(f'energy variance at {node} is negative')
    return ConditionalEnergyState(
        node=node,
        density=density,
        probability_from_root=probability_from_root,
        system_energy_expectation=system_energy,
        cumulative_energy_transfer=cumulative_energy_transfer,
        energy_expectation=system_energy + cumulative_energy_transfer,
        energy_variance=max(0.0, variance),
    )


def construct_luders_energy_instrument(
    hamiltonian: np.ndarray,
    *,
    node: str = 'root',
    instrument_id: str = 'spectral-luders-energy',
    target_prefix: str = 'energy-sector',
    tolerance: float = INSTRUMENT_DEFAULT_TOLERANCE,
) -> LudersInstrumentConstruction:
    """유한 H의 가장 성긴 상이 에너지 스펙트럼 PVM을 구성한다.

    스펙트럼 사영자는 주어진 유한 양의 에르미트 해밀토니안의 정준 함수다.
    뤼더스 연산은 상이 에너지마다 사영자 하나를 유일한 크라우스 연산자로
    쓴다. 여기서의 정준성은 스펙트럼 PVM의 정준성뿐이며, 축퇴 부문 안의 더
    미세한 포인터도 그 부문 안의 다른 QND 연산도 배제하지 않는다. 반환된 수치
    스펙트럼에서의 정확한 등식이 축퇴 군집을 정의한다. 선언된 상대 수치
    허용오차보다 가까운 상이 고유값은 비예리 부문으로 병합되지 않고 모호한
    것으로 거부된다.
    """

    if not node or not instrument_id or not target_prefix:
        raise ValueError('node, instrument identifier, and target prefix are required')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    hamiltonian_matrix = _square_matrix(hamiltonian, 'Hamiltonian')
    hamiltonian_matrix_scale = float(np.linalg.norm(hamiltonian_matrix, ord='fro'))
    if hamiltonian_matrix_scale <= 0.0:
        raise ValueError('Hamiltonian must have a non-zero physical energy scale')
    if (
        np.linalg.norm(
            hamiltonian_matrix - hamiltonian_matrix.conj().T,
            ord='fro',
        )
        / hamiltonian_matrix_scale
        > tolerance
    ):
        raise ValueError('Hamiltonian must be Hermitian')
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    energy_scale = float(np.max(np.abs(eigenvalues)))
    if float(np.min(eigenvalues)) <= tolerance * energy_scale:
        raise ValueError('Hamiltonian must be strictly positive on this contract')

    clusters: list[list[int]] = []
    for index, energy in enumerate(eigenvalues):
        if clusters:
            reference = float(eigenvalues[clusters[-1][0]])
            if float(energy) == reference:
                clusters[-1].append(index)
                continue
            if abs(float(energy) - reference) <= tolerance * energy_scale:
                raise ValueError(
                    'Hamiltonian has a numerically ambiguous near-degeneracy; '
                    'lower the tolerance or supply an exactly resolved representation'
                )
        clusters.append([index])

    dimension = hamiltonian_matrix.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    projectors: list[np.ndarray] = []
    energies: list[float] = []
    multiplicities: list[int] = []
    for cluster in clusters:
        basis = eigenvectors[:, cluster]
        projector = basis @ basis.conj().T
        projectors.append(projector)
        energies.append(float(np.mean(eigenvalues[cluster])))
        multiplicities.append(len(cluster))

    max_idempotence = max(
        float(np.linalg.norm(projector @ projector - projector, ord='fro'))
        for projector in projectors
    )
    max_orthogonality = max(
        (
            float(np.linalg.norm(left @ right, ord='fro'))
            for index, left in enumerate(projectors)
            for right in projectors[index + 1 :]
        ),
        default=0.0,
    )
    resolution_residual = float(
        np.linalg.norm(sum(projectors, start=np.zeros_like(identity)) - identity, ord='fro')
    )
    max_commutator = max(
        float(
            np.linalg.norm(
                projector @ hamiltonian_matrix - hamiltonian_matrix @ projector,
                ord='fro',
            )
        )
        for projector in projectors
    )
    max_cluster_width = max(
        float(np.max(eigenvalues[cluster]) - np.min(eigenvalues[cluster]))
        for cluster in clusters
    )
    max_eigenprojector_residual = max(
        float(
            np.linalg.norm(
                hamiltonian_matrix @ projector - energy * projector,
                ord='fro',
            )
        )
        for energy, projector in zip(energies, projectors)
    )
    max_relative_commutator = max_commutator / energy_scale
    max_relative_cluster_width = max_cluster_width / energy_scale
    max_relative_eigenprojector_residual = (
        max_eigenprojector_residual / energy_scale
    )
    numerical_limit = tolerance * max(1.0, float(dimension))
    if max(
        max_idempotence,
        max_orthogonality,
        resolution_residual,
        max_relative_commutator,
        max_relative_cluster_width,
        max_relative_eigenprojector_residual,
    ) > numerical_limit:
        raise ArithmeticError('spectral projector construction exceeded tolerance')

    branches = tuple(
        KrausBranch(
            target=f'{target_prefix}-{index}',
            label=f'Luders energy {energy:.16g}',
            operator=projector,
        )
        for index, (energy, projector) in enumerate(zip(energies, projectors))
    )
    return LudersInstrumentConstruction(
        instrument=RecordInstrument(
            node=node,
            instrument_id=instrument_id,
            branches=branches,
        ),
        spectral_energies=tuple(energies),
        spectral_multiplicities=tuple(multiplicities),
        hamiltonian_energy_scale=energy_scale,
        max_projector_idempotence_residual=max_idempotence,
        max_projector_orthogonality_residual=max_orthogonality,
        resolution_of_identity_residual=resolution_residual,
        max_hamiltonian_commutator_residual=max_commutator,
        max_relative_hamiltonian_commutator_residual=max_relative_commutator,
        max_spectral_cluster_width=max_cluster_width,
        max_relative_spectral_cluster_width=max_relative_cluster_width,
        max_eigenprojector_residual=max_eigenprojector_residual,
        max_relative_eigenprojector_residual=max_relative_eigenprojector_residual,
    )


def construct_energy_conserving_collision_instrument(
    system_hamiltonian: np.ndarray,
    ancilla_hamiltonian: np.ndarray,
    collision_unitary: np.ndarray,
    *,
    pointer_basis: np.ndarray | None = None,
    initial_pointer_index: int = 0,
    outcome_targets: Sequence[str] | None = None,
    outcome_labels: Sequence[str] | None = None,
    node: str = 'root',
    instrument_id: str = 'energy-conserving-collision',
    target_prefix: str = 'ancilla-energy',
    require_ground_input: bool = True,
    tolerance: float = INSTRUMENT_DEFAULT_TOLERANCE,
) -> EnergyConservingCollisionConstruction:
    """국소 충돌 하나에서 전달량 라벨이 붙은 기구를 유도한다.

    텐서 순서는 계 다음 보조계다. 주어진 유니터리는 H_system + H_ancilla와
    교환해야 한다. pointer_basis의 열은 보조계의 정규 직교 에너지 고유기저여야
    하며, 새 보조계는 선언된 기저 벡터 하나에서 출발한다. 나가는 보조계를
    측정하면

        K_r = <r| U |a0>,
        Delta E_r = e_r - e_a0,

    와 정확한 연산자 장부

        sum_r K_r^dagger (H_system + Delta E_r I) K_r = H_system.

    을 얻는다. 바닥 상태 입력이면 모든 전달량은 수치 정밀도 안에서 음이 아니다.
    이는 추상적 결과 기구와 그 에너지 영수증을 구성할 뿐, 지속적인 장치
    포인터나 사건 위치를 유도하지 않는다.
    """

    if not node or not instrument_id or not target_prefix:
        raise ValueError('node, instrument identifier, and target prefix are required')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    system_h = _square_matrix(system_hamiltonian, 'system Hamiltonian')
    ancilla_h = _square_matrix(ancilla_hamiltonian, 'ancilla Hamiltonian')
    system_dimension = system_h.shape[0]
    ancilla_dimension = ancilla_h.shape[0]
    if outcome_targets is None:
        targets = tuple(
            f'{target_prefix}-{index}' for index in range(ancilla_dimension)
        )
    else:
        targets = tuple(outcome_targets)
    if outcome_labels is None:
        labels: tuple[str, ...] | None = None
    else:
        labels = tuple(outcome_labels)
    if len(targets) != ancilla_dimension or (
        labels is not None and len(labels) != ancilla_dimension
    ):
        raise ValueError('collision outcomes must cover the ancilla pointer basis')
    if any(not target for target in targets) or len(set(targets)) != len(targets):
        raise ValueError('collision outcome targets must be unique and non-empty')
    if labels is not None and any(not label for label in labels):
        raise ValueError('collision outcome labels must be non-empty')
    system_scale = float(np.linalg.norm(system_h, ord='fro'))
    ancilla_scale = float(np.linalg.norm(ancilla_h, ord='fro'))
    if system_scale <= 0.0 or ancilla_scale <= 0.0:
        raise ValueError('system and ancilla need non-zero energy scales')
    if (
        np.linalg.norm(system_h - system_h.conj().T, ord='fro') / system_scale
        > tolerance
    ):
        raise ValueError('system Hamiltonian must be Hermitian')
    if (
        np.linalg.norm(ancilla_h - ancilla_h.conj().T, ord='fro') / ancilla_scale
        > tolerance
    ):
        raise ValueError('ancilla Hamiltonian must be Hermitian')
    system_spectrum = np.linalg.eigvalsh(system_h)
    system_spectral_scale = float(np.max(np.abs(system_spectrum)))
    if float(np.min(system_spectrum)) <= tolerance * system_spectral_scale:
        raise ValueError('system Hamiltonian must be strictly positive')
    ancilla_spectrum = np.linalg.eigvalsh(ancilla_h)
    ancilla_spectral_scale = float(np.max(np.abs(ancilla_spectrum)))
    if float(np.min(ancilla_spectrum)) < -tolerance * ancilla_spectral_scale:
        raise ValueError('ancilla Hamiltonian must be positive semidefinite')

    if pointer_basis is None:
        pointer = np.eye(ancilla_dimension, dtype=np.complex128)
    else:
        pointer = _square_matrix(pointer_basis, 'pointer basis')
        if pointer.shape != (ancilla_dimension, ancilla_dimension):
            raise ValueError('pointer basis has the wrong ancilla dimension')
    ancilla_identity = np.eye(ancilla_dimension, dtype=np.complex128)
    pointer_basis_unitarity_residual = float(
        np.linalg.norm(pointer.conj().T @ pointer - ancilla_identity, ord='fro')
    )
    numerical_limit = tolerance * max(
        1.0,
        float(system_dimension),
        float(ancilla_dimension),
    )
    if pointer_basis_unitarity_residual > numerical_limit:
        raise ValueError('pointer basis must be unitary')

    ancilla_h_pointer = pointer.conj().T @ ancilla_h @ pointer
    pointer_energies = tuple(
        float(ancilla_h_pointer[index, index].real)
        for index in range(ancilla_dimension)
    )
    ancilla_h_squared_pointer = (
        pointer.conj().T @ ancilla_h @ ancilla_h @ pointer
    )
    pointer_variances = tuple(
        max(
            0.0,
            float(ancilla_h_squared_pointer[index, index].real)
            - pointer_energies[index] ** 2,
        )
        for index in range(ancilla_dimension)
    )
    max_relative_pointer_energy_variance = (
        max(pointer_variances, default=0.0) / ancilla_spectral_scale**2
    )
    off_diagonal_pointer_h = ancilla_h_pointer - np.diag(
        np.diag(ancilla_h_pointer)
    )
    if (
        np.linalg.norm(off_diagonal_pointer_h, ord='fro') / ancilla_scale
        > numerical_limit
        or max_relative_pointer_energy_variance > numerical_limit
    ):
        raise ValueError('pointer basis must resolve sharp ancilla energies')
    if not 0 <= initial_pointer_index < ancilla_dimension:
        raise ValueError('initial pointer index is outside the ancilla basis')
    initial_pointer_energy = pointer_energies[initial_pointer_index]
    if (
        require_ground_input
        and initial_pointer_energy
        > min(pointer_energies) + tolerance * ancilla_spectral_scale
    ):
        raise ValueError('fresh ancilla must start in a pointer-basis ground state')

    collision = _square_matrix(collision_unitary, 'collision unitary')
    joint_dimension = system_dimension * ancilla_dimension
    if collision.shape != (joint_dimension, joint_dimension):
        raise ValueError('collision unitary has the wrong joint dimension')
    joint_identity = np.eye(joint_dimension, dtype=np.complex128)
    collision_unitarity_residual = float(
        np.linalg.norm(collision.conj().T @ collision - joint_identity, ord='fro')
    )
    if collision_unitarity_residual > numerical_limit:
        raise ValueError('collision operator must be unitary')

    system_identity = np.eye(system_dimension, dtype=np.complex128)
    total_hamiltonian = (
        np.kron(system_h, ancilla_identity)
        + np.kron(system_identity, ancilla_h)
    )
    total_energy_scale = float(np.linalg.norm(total_hamiltonian, ord='fro'))
    total_energy_commutator_residual = float(
        np.linalg.norm(
            collision @ total_hamiltonian - total_hamiltonian @ collision,
            ord='fro',
        )
    )
    relative_total_energy_commutator_residual = (
        total_energy_commutator_residual / total_energy_scale
    )
    if relative_total_energy_commutator_residual > numerical_limit:
        raise ValueError('collision unitary must conserve total energy')

    basis_change = np.kron(system_identity, pointer)
    collision_pointer = basis_change.conj().T @ collision @ basis_change
    collision_tensor = collision_pointer.reshape(
        system_dimension,
        ancilla_dimension,
        system_dimension,
        ancilla_dimension,
    )
    kraus_operators = tuple(
        collision_tensor[:, outcome, :, initial_pointer_index]
        for outcome in range(ancilla_dimension)
    )
    kraus_completeness = sum(
        (operator.conj().T @ operator for operator in kraus_operators),
        start=np.zeros_like(system_identity),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(kraus_completeness - system_identity, ord='fro')
    )
    if kraus_completeness_residual > numerical_limit:
        raise ArithmeticError('collision Kraus family is not complete')

    energy_transfers = []
    for pointer_energy in pointer_energies:
        transfer = pointer_energy - initial_pointer_energy
        if abs(transfer) <= tolerance * ancilla_spectral_scale:
            transfer = 0.0
        if require_ground_input and transfer < 0.0:
            raise ArithmeticError('ground-input collision produced negative transfer')
        energy_transfers.append(transfer)
    ledger_operator = sum(
        (
            operator.conj().T
            @ (system_h + transfer * system_identity)
            @ operator
            for operator, transfer in zip(kraus_operators, energy_transfers)
        ),
        start=np.zeros_like(system_h),
    )
    ledger_identity_residual = float(
        np.linalg.norm(ledger_operator - system_h, ord='fro')
    )
    relative_ledger_identity_residual = ledger_identity_residual / system_scale
    if relative_ledger_identity_residual > numerical_limit:
        raise ArithmeticError('collision energy-transfer ledger identity failed')

    branches = tuple(
        KrausBranch(
            target=targets[index],
            label=(
                labels[index]
                if labels is not None
                else f'ancilla energy {energy:.16g}'
            ),
            operator=operator,
            energy_transfer=transfer,
        )
        for index, (energy, transfer, operator) in enumerate(
            zip(pointer_energies, energy_transfers, kraus_operators)
        )
    )
    return EnergyConservingCollisionConstruction(
        instrument=RecordInstrument(
            node=node,
            instrument_id=instrument_id,
            branches=branches,
        ),
        initial_pointer_index=initial_pointer_index,
        initial_pointer_energy=initial_pointer_energy,
        pointer_energies=pointer_energies,
        branch_energy_transfers=tuple(energy_transfers),
        pointer_basis_unitarity_residual=pointer_basis_unitarity_residual,
        collision_unitarity_residual=collision_unitarity_residual,
        total_energy_commutator_residual=total_energy_commutator_residual,
        relative_total_energy_commutator_residual=(
            relative_total_energy_commutator_residual
        ),
        max_relative_pointer_energy_variance=(
            max_relative_pointer_energy_variance
        ),
        kraus_completeness_residual=kraus_completeness_residual,
        ledger_identity_residual=ledger_identity_residual,
        relative_ledger_identity_residual=relative_ledger_identity_residual,
        physical_pointer_persistence_derived=False,
    )


def apply_nonselective_channel(
    density: np.ndarray,
    branches: Sequence[KrausBranch],
) -> np.ndarray:
    """선언된 크라우스 족을 결과 라벨을 버리고 적용한다."""

    state = np.asarray(density, dtype=np.complex128)
    return sum(
        (
            np.asarray(branch.operator, dtype=np.complex128)
            @ state
            @ np.asarray(branch.operator, dtype=np.complex128).conj().T
            for branch in branches
        ),
        start=np.zeros_like(state),
    )


def _certify_classical_record_algebra(
    *,
    root: str,
    root_density: np.ndarray,
    dimension: int,
    branches_by_node: dict[str, tuple[tuple[KrausBranch, np.ndarray], ...]],
    state_by_node: dict[str, ConditionalEnergyState],
    supported_terminal_nodes: tuple[str, ...],
    tolerance: float,
) -> ClassicalRecordAlgebraCertificate:
    """검증된 트리의 완전한 직교 이력 레지스터를 구성한다."""

    identity = np.eye(dimension, dtype=np.complex128)
    stack: list[tuple[str, tuple[str, ...], np.ndarray]] = [
        (root, (root,), identity)
    ]
    raw_histories: list[tuple[str, tuple[str, ...], np.ndarray, float]] = []
    while stack:
        node, path, chain = stack.pop()
        branches = branches_by_node.get(node)
        if branches is None:
            unnormalized = chain @ root_density @ chain.conj().T
            probability = _real_trace(
                unnormalized,
                tolerance=tolerance,
                name=f'history probability at {node}',
            )
            if probability < -tolerance:
                raise ArithmeticError('history probability is negative')
            raw_histories.append((node, path, chain, max(0.0, probability)))
            continue
        for branch, operator in reversed(branches):
            stack.append(
                (
                    branch.target,
                    path + (branch.target,),
                    operator @ chain,
                )
            )

    if not raw_histories:
        raise ArithmeticError('a finite instrument tree needs a terminal history')

    global_effect = sum(
        (chain.conj().T @ chain for _, _, chain, _ in raw_histories),
        start=np.zeros_like(identity),
    )
    global_isometry_residual = float(
        np.linalg.norm(global_effect - identity, ord='fro')
    )
    probability_normalization_residual = abs(
        sum(probability for _, _, _, probability in raw_histories) - 1.0
    )

    supported = set(supported_terminal_nodes)
    history_probability_residuals: list[float] = []
    prefix_probabilities: dict[str, float] = {}
    histories: list[PersistentRecordHistory] = []
    for index, (terminal, path, _, probability) in enumerate(raw_histories):
        expected = (
            state_by_node[terminal].probability_from_root
            if terminal in supported
            else 0.0
        )
        history_probability_residuals.append(abs(probability - expected))
        for node in path:
            prefix_probabilities[node] = prefix_probabilities.get(node, 0.0) + probability
        histories.append(
            PersistentRecordHistory(
                basis_index=index,
                terminal_node=terminal,
                path=path,
                probability=probability,
                supported_by_root=terminal in supported,
            )
        )

    prefix_probability_residuals = [
        abs(
            probability
            - (
                state_by_node[node].probability_from_root
                if node in state_by_node
                else 0.0
            )
        )
        for node, probability in prefix_probabilities.items()
    ]
    max_history_probability_residual = max(history_probability_residuals, default=0.0)
    max_prefix_probability_residual = max(prefix_probability_residuals, default=0.0)
    numerical_limit = tolerance * max(1.0, float(len(raw_histories)), float(dimension))
    if max(
        global_isometry_residual,
        probability_normalization_residual,
        max_history_probability_residual,
        max_prefix_probability_residual,
    ) > numerical_limit:
        raise ArithmeticError('declared history register exceeded numerical tolerance')

    return ClassicalRecordAlgebraCertificate(
        histories=tuple(histories),
        node_probabilities=tuple(prefix_probabilities.items()),
        global_isometry_residual=global_isometry_residual,
        probability_normalization_residual=probability_normalization_residual,
        max_history_probability_residual=max_history_probability_residual,
        max_prefix_probability_residual=max_prefix_probability_residual,
        orthogonal_history_basis=True,
        commutative_diagonal_algebra=True,
        append_only_history_labels=True,
        physical_pointer_dynamics_derived=False,
    )


def build_energy_resolved_instrument_tree(
    hamiltonian: np.ndarray,
    root_density: np.ndarray,
    instruments: Sequence[RecordInstrument],
    *,
    root: str = 'root',
    root_label: str = 'prepared ensemble',
    initial_weight: float = 1.0,
    require_terminal_energy_sharp: bool = True,
    require_qnd: bool = True,
    tolerance: float = INSTRUMENT_DEFAULT_TOLERANCE,
) -> QuantumKernelCertificate:
    """유한 보른 기록 트리를 유도하고 그 에너지 장부를 인증한다.

    입력 기구는 트리를 이루어야 한다. 뿌리가 아닌 모든 표적은 정확히 하나의
    부모를 갖는다. 모든 크라우스 족은 완비이며 선언된 해밀토니안 더하기
    전달량 장부를 연산자로서 보존한다.

        sum_b K_b^dagger (H + Delta E_b I) K_b = H.

    기본값은 추가로 모든 가지가 QND일 것을 요구하므로 Delta E_b=0인 에너지
    기록을 다룬다. 명시적으로 에너지를 보존하는 계--보조계 충돌은
    require_qnd=False로 두고 측정된 보조계 에너지 증가분을 energy_transfer에
    기록할 수 있다. 확률이 수치 지지 임계를 넘는 결과만 지지 트리에 들어간다.
    임계 이하 결과는 완비성·장부·완전한 추상 이력 레지스터 검사에는 남는다.
    """

    if not root or not root_label:
        raise ValueError('root name and label must be non-empty')
    if not math.isfinite(initial_weight) or initial_weight <= 0.0:
        raise ValueError('initial weight must be finite and positive')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    hamiltonian_matrix = _square_matrix(hamiltonian, 'Hamiltonian')
    hamiltonian_matrix_scale = float(np.linalg.norm(hamiltonian_matrix, ord='fro'))
    if hamiltonian_matrix_scale <= 0.0:
        raise ValueError('Hamiltonian must have a non-zero physical energy scale')
    if (
        np.linalg.norm(
            hamiltonian_matrix - hamiltonian_matrix.conj().T,
            ord='fro',
        )
        / hamiltonian_matrix_scale
        > tolerance
    ):
        raise ValueError('Hamiltonian must be Hermitian')
    energy_spectrum = np.linalg.eigvalsh(hamiltonian_matrix)
    energy_scale = float(np.max(np.abs(energy_spectrum)))
    if float(np.min(energy_spectrum)) <= tolerance * energy_scale:
        raise ValueError('Hamiltonian must be strictly positive on this contract')

    dimension = hamiltonian_matrix.shape[0]
    root_state = _validate_density(
        root_density,
        dimension=dimension,
        tolerance=tolerance,
        name='root density',
    )

    instrument_by_node: dict[str, RecordInstrument] = {}
    for instrument in instruments:
        if not instrument.node or not instrument.instrument_id:
            raise ValueError('instrument node and identifier must be non-empty')
        if instrument.node in instrument_by_node:
            raise ValueError('instrument nodes must be unique')
        if not instrument.branches:
            raise ValueError('a nonterminal instrument needs at least one branch')
        instrument_by_node[instrument.node] = instrument

    identity = np.eye(dimension, dtype=np.complex128)
    state_by_node = {
        root: _energy_state(
            root,
            root_state,
            1.0,
            hamiltonian_matrix,
            cumulative_energy_transfer=0.0,
            energy_scale=energy_scale,
            tolerance=tolerance,
        )
    }
    label_by_node = {root: root_label}
    order = [root]
    queue = [root]
    declared_targets = {root}
    transitions: list[CausalTransition] = []
    used_instruments: list[tuple[str, str]] = []
    validated_branches_by_node: dict[
        str,
        tuple[tuple[KrausBranch, np.ndarray], ...],
    ] = {}
    max_completeness_residual = 0.0
    max_energy_channel_residual = 0.0
    max_relative_energy_channel_residual = 0.0
    max_qnd_commutator_residual = 0.0
    max_relative_qnd_commutator_residual = 0.0

    while queue:
        node = queue.pop(0)
        instrument = instrument_by_node.get(node)
        if instrument is None:
            continue
        used_instruments.append((node, instrument.instrument_id))
        parent = state_by_node[node]
        targets: set[str] = set()
        completeness = np.zeros_like(identity)
        energy_channel = np.zeros_like(hamiltonian_matrix)
        prepared: list[tuple[KrausBranch, np.ndarray]] = []
        for branch in instrument.branches:
            if not branch.target or not branch.label:
                raise ValueError('branch target and label must be non-empty')
            if branch.target in targets:
                raise ValueError('branch targets must be unique at each instrument')
            if branch.target in declared_targets:
                raise ValueError('instrument records must form a tree without merging')
            if not math.isfinite(branch.energy_transfer):
                raise ValueError('branch energy transfer must be finite')
            targets.add(branch.target)
            declared_targets.add(branch.target)
            operator = _square_matrix(
                branch.operator,
                f'Kraus operator {node}->{branch.target}',
            )
            if operator.shape != (dimension, dimension):
                raise ValueError('Kraus operator has the wrong Hilbert-space dimension')
            completeness += operator.conj().T @ operator
            energy_channel += operator.conj().T @ (
                hamiltonian_matrix
                + branch.energy_transfer * identity
            ) @ operator
            qnd_residual = float(
                np.linalg.norm(
                    operator @ hamiltonian_matrix
                    - hamiltonian_matrix @ operator,
                    ord='fro',
                )
            )
            max_qnd_commutator_residual = max(
                max_qnd_commutator_residual,
                qnd_residual,
            )
            max_relative_qnd_commutator_residual = max(
                max_relative_qnd_commutator_residual,
                qnd_residual / hamiltonian_matrix_scale,
            )
            prepared.append((branch, operator))

        completeness_residual = float(
            np.linalg.norm(completeness - identity, ord='fro')
        )
        energy_channel_residual = float(
            np.linalg.norm(
                energy_channel - hamiltonian_matrix,
                ord='fro',
            )
        )
        max_completeness_residual = max(
            max_completeness_residual,
            completeness_residual,
        )
        max_energy_channel_residual = max(
            max_energy_channel_residual,
            energy_channel_residual,
        )
        relative_energy_channel_residual = (
            energy_channel_residual / hamiltonian_matrix_scale
        )
        max_relative_energy_channel_residual = max(
            max_relative_energy_channel_residual,
            relative_energy_channel_residual,
        )
        if completeness_residual > tolerance:
            raise ValueError('Kraus family must be trace-preserving')
        if relative_energy_channel_residual > tolerance:
            raise ValueError(
                'instrument must preserve the declared Hamiltonian-plus-transfer ledger'
            )
        if require_qnd and max(
            float(
                np.linalg.norm(
                    operator @ hamiltonian_matrix
                    - hamiltonian_matrix @ operator,
                    ord='fro',
                )
            )
            / hamiltonian_matrix_scale
            for _, operator in prepared
        ) > tolerance:
            raise ValueError('every Kraus branch must be QND with the Hamiltonian')
        validated_branches_by_node[node] = tuple(prepared)

        child_probabilities = 0.0
        for branch, operator in prepared:
            unnormalized = operator @ parent.density @ operator.conj().T
            probability = _real_trace(
                unnormalized,
                tolerance=tolerance,
                name=f'Born weight {node}->{branch.target}',
            )
            if probability < -tolerance:
                raise ArithmeticError('Born probability is negative')
            if probability <= tolerance:
                continue
            if probability > 1.0 + tolerance:
                raise ArithmeticError('Born probability exceeds one')
            probability = min(1.0, probability)
            child_probabilities += probability
            posterior = _validate_density(
                unnormalized / probability,
                dimension=dimension,
                tolerance=tolerance,
                name=f'posterior density at {branch.target}',
            )
            child = _energy_state(
                branch.target,
                posterior,
                parent.probability_from_root * probability,
                hamiltonian_matrix,
                cumulative_energy_transfer=(
                    parent.cumulative_energy_transfer
                    + branch.energy_transfer
                ),
                energy_scale=energy_scale,
                tolerance=tolerance,
            )
            state_by_node[branch.target] = child
            label_by_node[branch.target] = branch.label
            transitions.append(
                CausalTransition(
                    source=node,
                    target=branch.target,
                    probability=probability,
                )
            )
            order.append(branch.target)
            queue.append(branch.target)
        if not math.isclose(
            child_probabilities,
            1.0,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ArithmeticError('Born branch probabilities do not sum to one')

    unused = set(instrument_by_node) - {node for node, _ in used_instruments}
    if unused:
        raise ValueError('every declared instrument must be reachable from the root')

    terminal_nodes = tuple(
        node for node in order if node not in instrument_by_node
    )
    terminal_energy_resolved = all(
        state_by_node[node].energy_variance
        <= tolerance * energy_scale**2
        for node in terminal_nodes
    )
    if require_terminal_energy_sharp and not terminal_energy_resolved:
        raise ValueError('every terminal record must have sharp Hamiltonian energy')

    nodes = tuple(
        CausalRecordNode(
            name=node,
            label=label_by_node[node],
            energy=state_by_node[node].energy_expectation,
        )
        for node in order
    )
    flow = construct_conserved_record_flow(
        nodes,
        transitions,
        {root: initial_weight},
        tolerance=tolerance,
    )
    record_algebra = _certify_classical_record_algebra(
        root=root,
        root_density=root_state,
        dimension=dimension,
        branches_by_node=validated_branches_by_node,
        state_by_node=state_by_node,
        supported_terminal_nodes=terminal_nodes,
        tolerance=tolerance,
    )
    return QuantumKernelCertificate(
        root=root,
        nodes=nodes,
        transitions=tuple(transitions),
        conditional_states=tuple(state_by_node[node] for node in order),
        instrument_ids=tuple(used_instruments),
        terminal_nodes=terminal_nodes,
        terminal_energy_resolved=terminal_energy_resolved,
        qnd_required=require_qnd,
        hamiltonian_energy_scale=energy_scale,
        hamiltonian_energy_min=float(np.min(energy_spectrum)),
        hamiltonian_energy_max=float(np.max(energy_spectrum)),
        max_completeness_residual=max_completeness_residual,
        max_energy_channel_residual=max_energy_channel_residual,
        max_relative_energy_channel_residual=max_relative_energy_channel_residual,
        max_qnd_commutator_residual=max_qnd_commutator_residual,
        max_relative_qnd_commutator_residual=max_relative_qnd_commutator_residual,
        support_probability_tolerance=tolerance,
        record_algebra=record_algebra,
        flow=flow,
    )


DEFAULT_TOLERANCE = 1.0e-12


def _positive_tolerance(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_density(state: np.ndarray, *, tolerance: float) -> np.ndarray:
    density = np.asarray(state, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1] or density.shape[0] < 2:
        raise ValueError("state must be a square density matrix of dimension at least two")
    if not np.isfinite(density).all():
        raise ValueError("density matrix entries must be finite")
    if np.linalg.norm(density - density.conj().T, ord="fro") > tolerance:
        raise ValueError("density matrix must be Hermitian")
    trace = np.trace(density)
    if abs(float(trace.real) - 1.0) > tolerance or abs(float(trace.imag)) > tolerance:
        raise ValueError("density matrix must have unit trace")
    if float(np.linalg.eigvalsh(density).min()) < -tolerance:
        raise ValueError("density matrix must be positive semidefinite")
    return density


@dataclass(frozen=True)
class CoarseOutcomeOperation:
    """선언된 성긴 결과 하나. 내부 크라우스 항을 여럿 가질 수 있다."""

    label: str
    operators: tuple[np.ndarray, ...]
    energy_transfer: float | None = None


def _validated_outcomes(
    outcomes: Sequence[CoarseOutcomeOperation],
    dimension: int,
) -> tuple[CoarseOutcomeOperation, ...]:
    declared = tuple(outcomes)
    if not declared:
        raise ValueError("instrument outcomes must be non-empty")
    if any(not isinstance(outcome, CoarseOutcomeOperation) for outcome in declared):
        raise TypeError("every outcome must be a CoarseOutcomeOperation")
    if any(not outcome.label for outcome in declared):
        raise ValueError("outcome labels must be non-empty")
    if len({outcome.label for outcome in declared}) != len(declared):
        raise ValueError("coarse outcome labels must be unique")
    for outcome in declared:
        if not outcome.operators:
            raise ValueError("every coarse outcome needs at least one Kraus operator")
        for operator in outcome.operators:
            matrix = np.asarray(operator, dtype=np.complex128)
            if matrix.shape != (dimension, dimension):
                raise ValueError("Kraus operators must match the state dimension")
            if not np.isfinite(matrix).all():
                raise ValueError("Kraus operator entries must be finite")
        if outcome.energy_transfer is not None and not math.isfinite(outcome.energy_transfer):
            raise ValueError("energy transfer must be finite when declared")
    return declared


def _outcome_output(outcome: CoarseOutcomeOperation, state: np.ndarray) -> np.ndarray:
    return sum(
        (
            np.asarray(operator, dtype=np.complex128)
            @ state
            @ np.asarray(operator, dtype=np.complex128).conj().T
            for operator in outcome.operators
        ),
        np.zeros_like(state),
    )


def instrument_completeness_residual(
    outcomes: Sequence[CoarseOutcomeOperation],
    dimension: int,
) -> float:
    """모든 내부 크라우스 항에 대한 ``||sum K^dagger K - I||_2``를 반환한다."""

    declared = _validated_outcomes(outcomes, dimension)
    completeness = sum(
        (
            np.asarray(operator, dtype=np.complex128).conj().T
            @ np.asarray(operator, dtype=np.complex128)
            for outcome in declared
            for operator in outcome.operators
        ),
        np.zeros((dimension, dimension), dtype=np.complex128),
    )
    return float(np.linalg.norm(completeness - np.eye(dimension), ord=2))


def born_probabilities(
    outcomes: Sequence[CoarseOutcomeOperation],
    state: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, ...]:
    """실패 시 닫히는 기구 검사를 거친 뒤 성긴 보른 확률을 반환한다."""

    tol = _positive_tolerance(tolerance)
    density = _validated_density(state, tolerance=tol)
    declared = _validated_outcomes(outcomes, density.shape[0])
    if instrument_completeness_residual(declared, density.shape[0]) > 10.0 * tol:
        raise ValueError("Kraus family must be complete")
    probabilities: list[float] = []
    for outcome in declared:
        probability = float(np.trace(_outcome_output(outcome, density)).real)
        if probability < -tol or not math.isfinite(probability):
            raise ArithmeticError("Born probability must be finite and nonnegative")
        probabilities.append(max(0.0, probability))
    if abs(math.fsum(probabilities) - 1.0) > 10.0 * tol:
        raise ArithmeticError("Born probabilities must sum to one")
    return tuple(probabilities)


@dataclass(frozen=True)
class SeedPartition:
    """단위 씨앗 구간의 수치적으로 정규화된 반열린 분할."""

    input_probabilities: tuple[float, ...]
    cell_probabilities: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    input_normalization_residual: float


def build_seed_partition(
    probabilities: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SeedPartition:
    """확률 0인 칸을 빈 채로 유지하면서 ``[C_{a-1}, C_a)``를 만든다."""

    tol = _positive_tolerance(tolerance)
    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ValueError("probabilities must be non-empty")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and nonnegative")
    total = math.fsum(values)
    residual = abs(total - 1.0)
    if total <= 0.0 or residual > 10.0 * tol:
        raise ValueError("probabilities must sum to one within tolerance")

    # 이 명시적 수치 정규화는 정확히 0인 칸을 보존하면서 u -> 1^- 에서의
    # 부동소수점 틈을 막는다. 입력 잔차는 숨기지 않고 반환한다.
    normalized = tuple(value / total for value in values)
    intervals: list[tuple[float, float]] = []
    cumulative = 0.0
    last_positive_index = max(
        index for index, probability in enumerate(normalized) if probability > 0.0
    )
    for index, probability in enumerate(normalized):
        start = cumulative
        if probability == 0.0:
            end = start
        elif index == last_positive_index:
            end = 1.0
        else:
            end = math.fsum(normalized[: index + 1])
        cumulative = end
        intervals.append((start, end))
    return SeedPartition(
        input_probabilities=values,
        cell_probabilities=normalized,
        intervals=tuple(intervals),
        input_normalization_residual=residual,
    )


def select_partition_cell(partition: SeedPartition, seed: float) -> int:
    """``seed``를 포함하는 유일한 양의 측도 칸을 반환한다."""

    value = float(seed)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("seed must be finite and lie in [0, 1)")
    matches = tuple(
        index
        for index, ((start, end), probability) in enumerate(
            zip(partition.intervals, partition.cell_probabilities)
        )
        if probability > 0.0 and start <= value < end
    )
    if len(matches) != 1:
        raise ArithmeticError("valid seed must belong to exactly one positive interval")
    return matches[0]


@dataclass(frozen=True)
class OutcomeSelection:
    outcome_index: int
    label: str
    seed: float
    raw_born_probability: float
    partition_probability: float
    interval: tuple[float, float]
    subnormalized_state: np.ndarray
    posterior: np.ndarray


def select_outcome(
    outcomes: Sequence[CoarseOutcomeOperation],
    state: np.ndarray,
    seed: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> OutcomeSelection:
    """주어진 씨앗에 대해 성긴 결과 하나를 선택한다. 무작위성은 유도하지 않는다."""

    tol = _positive_tolerance(tolerance)
    density = _validated_density(state, tolerance=tol)
    declared = _validated_outcomes(outcomes, density.shape[0])
    probabilities = born_probabilities(declared, density, tolerance=tol)
    partition = build_seed_partition(probabilities, tolerance=tol)
    index = select_partition_cell(partition, seed)
    operation = _outcome_output(declared[index], density)
    raw_probability = probabilities[index]
    partition_probability = partition.cell_probabilities[index]
    if raw_probability <= 0.0 or partition_probability <= 0.0:
        raise ArithmeticError("zero-probability outcome has no posterior")
    posterior = operation / raw_probability
    if abs(float(np.trace(posterior).real) - 1.0) > 20.0 * tol:
        raise ArithmeticError("selected posterior failed normalization")
    return OutcomeSelection(
        outcome_index=index,
        label=declared[index].label,
        seed=float(seed),
        raw_born_probability=raw_probability,
        partition_probability=partition_probability,
        interval=partition.intervals[index],
        subnormalized_state=operation,
        posterior=posterior,
    )


def apply_nonselective_instrument(
    outcomes: Sequence[CoarseOutcomeOperation],
    state: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """라벨을 버리고 모든 성긴 연산을 정확히 한 번씩 적용한다."""

    tol = _positive_tolerance(tolerance)
    density = _validated_density(state, tolerance=tol)
    declared = _validated_outcomes(outcomes, density.shape[0])
    if instrument_completeness_residual(declared, density.shape[0]) > 10.0 * tol:
        raise ValueError("Kraus family must be complete")
    return sum(
        (_outcome_output(outcome, density) for outcome in declared),
        np.zeros_like(density),
    )


def equal_copy_internal_refinement(
    outcome: CoarseOutcomeOperation,
    multiplicity: int,
) -> CoarseOutcomeOperation:
    """모든 내부 ``K``를 ``k``개의 복사본 ``K/sqrt(k)``로 바꾼다."""

    count = _positive_integer(multiplicity, "multiplicity")
    if not outcome.operators:
        raise ValueError("outcome must contain at least one Kraus operator")
    refined = tuple(
        np.asarray(operator, dtype=np.complex128) / math.sqrt(count)
        for operator in outcome.operators
        for _ in range(count)
    )
    return CoarseOutcomeOperation(
        label=outcome.label,
        operators=refined,
        energy_transfer=outcome.energy_transfer,
    )


def _two_channel_emission_collision(left_probability: float) -> np.ndarray:
    left_amplitude = math.sqrt(left_probability)
    right_amplitude = math.sqrt(1.0 - left_probability)
    collision = np.eye(6, dtype=np.complex128)
    energy_two_sector = (1, 2, 3)
    collision[np.ix_(energy_two_sector, energy_two_sector)] = np.array(
        [
            [right_amplitude, 0.0, left_amplitude],
            [-left_amplitude, 0.0, right_amplitude],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return collision


def _partial_trace_a(joint_state: np.ndarray, dimension_a: int, dimension_b: int) -> np.ndarray:
    tensor = joint_state.reshape(dimension_a, dimension_b, dimension_a, dimension_b)
    return np.trace(tensor, axis1=0, axis2=2)


def _joint_outcome_output(
    outcome: CoarseOutcomeOperation,
    joint_state: np.ndarray,
    remote_dimension: int,
) -> np.ndarray:
    local_identity = np.eye(remote_dimension, dtype=np.complex128)
    return sum(
        (
            np.kron(np.asarray(operator, dtype=np.complex128), local_identity)
            @ joint_state
            @ np.kron(np.asarray(operator, dtype=np.complex128), local_identity).conj().T
            for operator in outcome.operators
        ),
        np.zeros_like(joint_state),
    )


@dataclass(frozen=True)
class LocalBornSelectionCertificate:
    outcome_labels: tuple[str, ...]
    raw_born_probabilities: tuple[float, ...]
    partition_probabilities: tuple[float, ...]
    seed_intervals: tuple[tuple[float, float], ...]
    probe_seeds: tuple[float, ...]
    probe_labels: tuple[str, ...]
    probability_normalization_residual: float
    maximum_interval_probability_residual: float
    maximum_posterior_trace_residual: float
    seed_average_channel_residual: float
    completeness_residual: float
    refinement_operation_residual: float
    refinement_probability_residual: float
    refinement_posterior_residual: float
    refinement_interval_residual: float
    refinement_same_seed_label_mismatches: int
    energy_scale: float
    collision_operator_energy_ledger_residual: float
    maximum_supported_branch_relative_energy_residual: float
    maximum_supported_branch_dimensionless_energy_variance: float
    remote_nonselective_marginal_residual: float
    forced_seed_remote_trace_distance: float
    fixed_seed_born_frequency_error: float
    x_measurement_best_scalar_receipts: tuple[float, float]
    x_measurement_relative_frobenius_receipt_residual: float
    x_measurement_relative_operator_receipt_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *,
    left_probability: float = 0.4,
    energy_scale: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> LocalBornSelectionCertificate:
    """E27 유한 선택·에너지·신호 전달 증서를 만든다."""

    tol = _positive_tolerance(tolerance)
    if not math.isfinite(left_probability) or not 0.0 < left_probability < 1.0:
        raise ValueError("left_probability must be finite and lie in (0, 1)")
    if not math.isfinite(energy_scale) or energy_scale <= 0.0:
        raise ValueError("energy_scale must be finite and positive")

    system_hamiltonian = energy_scale * np.diag([1.0, 2.0]).astype(np.complex128)
    ancilla_hamiltonian = energy_scale * np.diag([0.0, 1.0, 1.0]).astype(
        np.complex128
    )
    collision = construct_energy_conserving_collision_instrument(
        system_hamiltonian,
        ancilla_hamiltonian,
        _two_channel_emission_collision(left_probability),
        outcome_targets=("silent", "left", "right"),
        outcome_labels=("silent", "left", "right"),
        tolerance=tol,
    )
    outcomes = tuple(
        CoarseOutcomeOperation(
            label=branch.target,
            operators=(np.asarray(branch.operator, dtype=np.complex128),),
            energy_transfer=float(branch.energy_transfer),
        )
        for branch in collision.instrument.branches
    )
    initial_state = np.diag([0.0, 1.0]).astype(np.complex128)
    probabilities = born_probabilities(outcomes, initial_state, tolerance=tol)
    partition = build_seed_partition(probabilities, tolerance=tol)
    left_boundary = partition.intervals[1][1]
    probe_seeds = (
        0.0,
        math.nextafter(left_boundary, 0.0),
        left_boundary,
        math.nextafter(1.0, 0.0),
    )
    selections = tuple(
        select_outcome(outcomes, initial_state, seed, tolerance=tol)
        for seed in probe_seeds
    )

    interval_residual = max(
        abs((end - start) - probability)
        for (start, end), probability in zip(
            partition.intervals, partition.cell_probabilities
        )
    )
    posterior_trace_residuals: list[float] = []
    seed_average = np.zeros_like(initial_state)
    for outcome, raw_probability, partition_probability in zip(
        outcomes, probabilities, partition.cell_probabilities
    ):
        if raw_probability <= 0.0:
            continue
        operation = _outcome_output(outcome, initial_state)
        posterior = operation / raw_probability
        posterior_trace_residuals.append(abs(float(np.trace(posterior).real) - 1.0))
        seed_average += partition_probability * posterior
    nonselective = apply_nonselective_instrument(outcomes, initial_state, tolerance=tol)
    seed_average_channel_residual = float(np.linalg.norm(seed_average - nonselective, ord=2))
    completeness_residual = instrument_completeness_residual(outcomes, 2)

    refined_outcomes = list(outcomes)
    refined_outcomes[1] = equal_copy_internal_refinement(refined_outcomes[1], 7)
    refined_tuple = tuple(refined_outcomes)
    refined_probabilities = born_probabilities(refined_tuple, initial_state, tolerance=tol)
    refined_partition = build_seed_partition(refined_probabilities, tolerance=tol)
    base_left_output = _outcome_output(outcomes[1], initial_state)
    refined_left_output = _outcome_output(refined_tuple[1], initial_state)
    refinement_operation_residual = float(
        np.linalg.norm(refined_left_output - base_left_output, ord=2)
    )
    refinement_probability_residual = max(
        abs(left - right)
        for left, right in zip(
            partition.cell_probabilities, refined_partition.cell_probabilities
        )
    )
    refinement_posterior_residual = float(
        np.linalg.norm(
            refined_left_output / refined_probabilities[1]
            - base_left_output / probabilities[1],
            ord=2,
        )
    )
    refinement_interval_residual = max(
        abs(left_endpoint - right_endpoint)
        for left_interval, right_interval in zip(
            partition.intervals, refined_partition.intervals
        )
        for left_endpoint, right_endpoint in zip(left_interval, right_interval)
    )
    refinement_probe_seeds = tuple(
        0.5 * (start + end)
        for (start, end), probability in zip(
            partition.intervals, partition.cell_probabilities
        )
        if probability > 0.0
    )
    refinement_same_seed_label_mismatches = sum(
        select_outcome(outcomes, initial_state, seed, tolerance=tol).label
        != select_outcome(refined_tuple, initial_state, seed, tolerance=tol).label
        for seed in refinement_probe_seeds
    )

    initial_energy = float(np.trace(system_hamiltonian @ initial_state).real)
    supported_branch_relative_energy_residuals: list[float] = []
    supported_branch_dimensionless_variances: list[float] = []
    for outcome, raw_probability in zip(outcomes, probabilities):
        if raw_probability <= 0.0:
            continue
        posterior = _outcome_output(outcome, initial_state) / raw_probability
        system_energy = float(np.trace(system_hamiltonian @ posterior).real)
        system_energy_squared = float(
            np.trace(system_hamiltonian @ system_hamiltonian @ posterior).real
        )
        supported_branch_dimensionless_variances.append(
            max(0.0, system_energy_squared - system_energy * system_energy)
            / (energy_scale * energy_scale)
        )
        if outcome.energy_transfer is None:
            raise ArithmeticError("collision outcome must carry an energy receipt")
        supported_branch_relative_energy_residuals.append(
            abs(system_energy + outcome.energy_transfer - initial_energy) / energy_scale
        )

    projector_zero = np.diag([1.0, 0.0]).astype(np.complex128)
    projector_one = np.diag([0.0, 1.0]).astype(np.complex128)
    local_outcomes = (
        CoarseOutcomeOperation("zero", (projector_zero,)),
        CoarseOutcomeOperation("one", (projector_one,)),
    )
    bell_vector = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    bell_state = np.outer(bell_vector, bell_vector.conj())
    remote_before = _partial_trace_a(bell_state, 2, 2)
    joint_outputs = tuple(
        _joint_outcome_output(outcome, bell_state, 2) for outcome in local_outcomes
    )
    remote_after = _partial_trace_a(sum(joint_outputs, np.zeros_like(bell_state)), 2, 2)
    remote_nonselective_marginal_residual = float(
        np.linalg.norm(remote_after - remote_before, ord=2)
    )
    joint_probabilities = tuple(float(np.trace(output).real) for output in joint_outputs)
    remote_conditionals = tuple(
        _partial_trace_a(output / probability, 2, 2)
        for output, probability in zip(joint_outputs, joint_probabilities)
    )
    forced_seed_remote_trace_distance = 0.5 * float(
        np.linalg.norm(remote_conditionals[0] - remote_conditionals[1], ord="nuc")
    )
    bell_partition = build_seed_partition(
        joint_probabilities,
        tolerance=tol,
    )
    fixed_seed = 0.25
    repeated_labels = tuple(
        select_partition_cell(bell_partition, fixed_seed) for _ in range(32)
    )
    empirical = tuple(repeated_labels.count(index) / len(repeated_labels) for index in range(2))
    fixed_seed_born_frequency_error = max(
        abs(observed - expected)
        for observed, expected in zip(empirical, bell_partition.cell_probabilities)
    )

    energy_unit = energy_scale
    incompatible_hamiltonian = np.diag([0.0, energy_unit]).astype(np.complex128)
    plus = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    minus = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)
    measured_energy = plus @ incompatible_hamiltonian @ plus + minus @ incompatible_hamiltonian @ minus
    target_receipt = incompatible_hamiltonian - measured_energy
    design = np.column_stack((plus.reshape(-1), minus.reshape(-1)))
    best_receipts, _, _, _ = np.linalg.lstsq(design, target_receipt.reshape(-1), rcond=None)
    best_operator = (
        measured_energy + best_receipts[0] * plus + best_receipts[1] * minus
    )
    receipt_residual = best_operator - incompatible_hamiltonian
    relative_frobenius_receipt_residual = float(
        np.linalg.norm(receipt_residual, ord="fro")
        / np.linalg.norm(incompatible_hamiltonian, ord="fro")
    )
    relative_operator_receipt_residual = float(
        np.linalg.norm(receipt_residual, ord=2)
        / np.linalg.norm(incompatible_hamiltonian, ord=2)
    )

    numerical_limit = 20.0 * tol
    inverse_cdf_certified = max(
        partition.input_normalization_residual,
        interval_residual,
        max(posterior_trace_residuals, default=0.0),
        seed_average_channel_residual,
        completeness_residual,
    ) <= numerical_limit
    refinement_certified = max(
        refinement_operation_residual,
        refinement_probability_residual,
        refinement_posterior_residual,
        refinement_interval_residual,
    ) <= numerical_limit and refinement_same_seed_label_mismatches == 0
    collision_energy_certified = max(
        collision.relative_ledger_identity_residual,
        max(supported_branch_relative_energy_residuals, default=0.0),
        max(supported_branch_dimensionless_variances, default=0.0),
    ) <= numerical_limit

    return LocalBornSelectionCertificate(
        outcome_labels=tuple(outcome.label for outcome in outcomes),
        raw_born_probabilities=probabilities,
        partition_probabilities=partition.cell_probabilities,
        seed_intervals=partition.intervals,
        probe_seeds=probe_seeds,
        probe_labels=tuple(selection.label for selection in selections),
        probability_normalization_residual=partition.input_normalization_residual,
        maximum_interval_probability_residual=interval_residual,
        maximum_posterior_trace_residual=max(posterior_trace_residuals, default=0.0),
        seed_average_channel_residual=seed_average_channel_residual,
        completeness_residual=completeness_residual,
        refinement_operation_residual=refinement_operation_residual,
        refinement_probability_residual=refinement_probability_residual,
        refinement_posterior_residual=refinement_posterior_residual,
        refinement_interval_residual=refinement_interval_residual,
        refinement_same_seed_label_mismatches=refinement_same_seed_label_mismatches,
        energy_scale=energy_scale,
        collision_operator_energy_ledger_residual=collision.relative_ledger_identity_residual,
        maximum_supported_branch_relative_energy_residual=max(
            supported_branch_relative_energy_residuals, default=0.0
        ),
        maximum_supported_branch_dimensionless_energy_variance=max(
            supported_branch_dimensionless_variances, default=0.0
        ),
        remote_nonselective_marginal_residual=remote_nonselective_marginal_residual,
        forced_seed_remote_trace_distance=forced_seed_remote_trace_distance,
        fixed_seed_born_frequency_error=fixed_seed_born_frequency_error,
        x_measurement_best_scalar_receipts=tuple(
            float(value.real) for value in best_receipts
        ),
        x_measurement_relative_frobenius_receipt_residual=(
            relative_frobenius_receipt_residual
        ),
        x_measurement_relative_operator_receipt_residual=(
            relative_operator_receipt_residual
        ),
        dimensions={
            "seed_dimensionless": True,
            "born_probabilities_dimensionless": True,
            "cumulative_intervals_dimensionless": True,
            "kraus_and_density_entries_dimensionless": True,
            "hamiltonian_and_receipt_share_energy_dimension": True,
            "branch_energy_residual_divided_by_energy_scale": True,
            "branch_energy_variance_divided_by_energy_scale_squared": True,
            "seed_or_label_does_not_supply_energy_scale": True,
        },
        accounting={
            "probabilities_partition_seed_measure_once": True,
            "weighted_posteriors_equal_nonselective_channel_once": (
                seed_average_channel_residual <= numerical_limit
            ),
            "all_zero_probability_outcomes_not_conditioned": all(
                start == end
                for (start, end), probability in zip(
                    partition.intervals, partition.cell_probabilities
                )
                if probability == 0.0
            ),
            "unselected_probabilities_not_added_as_energy": True,
            "selected_record_energy_receipt_counted_once": collision_energy_certified,
            "seed_carries_energy": False,
        },
        boundaries={
            "uniform_independent_uncontrollable_seed_is_explicit_axiom": True,
            "unitary_or_stinespring_does_not_derive_seed": True,
            "forced_seed_is_prohibited_external_intervention": True,
            "half_open_intervals_use_declared_coarse_outcomes_only": True,
            "internal_kraus_labels_do_not_enter_seed_partition": True,
            "finite_refinement_probe_set_excludes_boundary_neighborhoods": True,
            "same_seed_refinement_claim_limited_to_declared_probe_set": True,
            "outcome_order_is_declared_input": True,
            "physical_seed_independence_from_settings_derived": False,
            "finite_bipartite_witness_is_not_relativistic_qft": True,
            "collision_hamiltonians_and_pointer_basis_are_supplied": True,
            "supported_collision_branches_are_sharp_energy_only": True,
        },
        alternatives={
            "operational_uniform_seed_sampler_route_open": True,
            "microscopic_local_uncontrollable_seed_route_open": True,
            "durable_local_pointer_route_open": True,
            "covariant_selection_and_geometry_route_open": True,
            "deterministic_hidden_variable_route_requires_bell_audit": True,
        },
        status={
            "inverse_cdf_born_partition_certified": inverse_cdf_certified,
            "valid_probe_seed_returns_one_coarse_label": len(selections) == len(probe_seeds),
            "uniform_seed_average_recovers_nonselective_channel": (
                seed_average_channel_residual <= numerical_limit
            ),
            "explicit_collision_instrument_cptp": completeness_residual <= numerical_limit,
            "coarse_selection_internal_refinement_invariant": refinement_certified,
            "supplied_collision_operator_energy_ledger_certified": (
                collision.relative_ledger_identity_residual <= numerical_limit
            ),
            "sharp_supported_branch_energy_receipts_certified": collision_energy_certified,
            "single_local_nonselective_marginal_witness": (
                remote_nonselective_marginal_residual <= numerical_limit
            ),
            "fixed_seed_born_frequency_counterexample": (
                fixed_seed_born_frequency_error > 0.49
            ),
            "controllable_seed_signalling_counterexample": (
                forced_seed_remote_trace_distance > 0.99
            ),
            "general_scalar_energy_receipt_counterexample": (
                relative_frobenius_receipt_residual > 0.7
                and relative_operator_receipt_residual > 0.49
            ),
            "physical_uniform_seed_law_derived": False,
            "objective_single_outcome_selection_derived": False,
            "durable_physical_pointer_derived": False,
            "relativistic_no_signalling_derived": False,
            "general_measurement_energy_conservation_derived": False,
            "spacetime_metric_curvature_or_gravity_derived": False,
            "independent_holdout_complete": False,
            "success_gates_1_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-probability", type=float, default=0.4)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(
                certificate(
                    left_probability=args.left_probability,
                    energy_scale=args.energy_scale,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
