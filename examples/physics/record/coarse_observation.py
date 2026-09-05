"""거친 관측(coarse observation) 섹터의 유한 증명서 네 종을 한 모듈에 모은다.

이 모듈은 다음 네 증명서를 순서대로 담는다.

1. 전단사 미세 진화와 다대일 관측 읽기(readout)의 분리.
   ``n``-준위 계와 ``n``-준위 기록 레지스터(record register)에 대해 제어 이동(controlled shift)

       W |i,j> = |i, j+i mod n>

   은 치환 유니터리(permutation unitary)이다. 레지스터를 ``|0>`` 으로 준비하면 그 제한은
   기록 등거리 사상(record isometry)

       V |psi> = sum_i c_i |i,i>

   이다. 따라서 이 선언된 유한 계+레지스터 갱신은 성분을 잃지 않는다: ``W`` 는 전단사이고
   ``V`` 는 자기 상(image) 위로 전단사이다. 별도의 유한 라벨 사상은 숨은 라벨을 전부 보존하고
   두 유한 라벨 집합이 이산 위상을 가질 때 전단사이다. 그 라벨 사실은 물리적 숨은 가지 동역학을
   유도하지 않는다. 개별 숨은 라벨을 잊는 이후의 읽기만이(숨은 라벨이 둘 이상일 때) 다대일이다.
   선택된 라벨은 그 조건부 읽기에 공급되는 입력이며, 유니터리가 유일한 실제 결과로 만들어 내는
   것이 아니다. 이 유한 증명서는 지속적 물리 포인터, 선택 법칙, 에너지 장부, 시공간 위상,
   계량, 곡률, 중력을 어느 것도 유도하지 않는다.

2. 유한 E19 증인: 뉴턴 측도 재매개화와 선명한 기록.
   반경 계산은 공급된 유한 뉴턴 퍼텐셜을 공급된 균일 부피 측도 위의 라돈-니코딤(Radon--Nikodym)
   가중치로 다시 쓸 뿐이다. 중력도 인력 기구도 유도하지 않는다. 양자 계산은 분리되어 있다:
   그 뤼더스(Lüders) 기구는 반경 가중치를 받지 않으므로 두 번째 확률 가중이 아니다.

3. 원시 크라우스(Kraus) 다중도를 계량 부피로 읽는 것에 대한 유한 장애.
   결과 연산은 크라우스 족을 등거리 사상으로 섞어도 변하지 않는다. 특히 영이 아닌 크라우스
   연산자 하나를 같은 크기의 영이 아닌 복사본 임의 개수로 바꿀 수 있다. 따라서 숨은 라벨 개수는
   표현의 성질이지 양자 기구의 성질이 아니다. 이 부분은 그 장애의 유한 증명서이며 물리적 기록,
   시공간 부피, 계량, 응력 텐서, 중력을 유도하지 않는다.

4. 기록--접힘(record--fold) 쌍선형 후보의 E36 승인 증명서.
   E35 뒤에 제안된 의도적으로 좁은 다리 하나를 검사한다. 실수 기록 후보 ``R_rec`` 와 실수 접힘
   장 ``phi`` 에 국소 스칼라 작용

       S = integral sqrt(-g) [
           -(grad R_rec)^2 / 2 -(grad phi)^2 / 2
           -m_R^2 R_rec^2 / 2 -m_phi^2 phi^2 / 2
           +kappa R_rec phi
       ].

   을 배정한다. 작용 계수 ``J_ns := kappa R_rec`` 는 질량 차원 3이며 가지 확률이 아니다.
   표시된 부호 규약에서 접힘 방정식은 ``(box-m_phi^2) phi = -J_ns`` 이다. 모든 상호작용
   에너지는 이 작용에서 한 번만 변분된다. 유한 증인은 차원 일관성, 정확한 이차 안정성 경계,
   점별 워드(Ward) 교환 항등식, 정적 슈어 보수(Schur complement)를 증명한다. 결정적 한계도
   드러낸다: 직교 장 회전이 쌍선형 질량 행렬을 대각화하므로 독립적인 포인터, 준비, 관측 가능
   결합이 기저를 고정하기 전까지 "record" 와 "fold" 라는 이름에는 물리적 특권이 없다.
   이것은 ``Q_nonselected -> R_rec`` 의 유도, 측정 모형, CPTP 채널, 양자 미시인과성 증명,
   GR 해, 홀드아웃 예측이 아니다. 그 주장 상한은 :func:`record_fold_certificate` 에 명시된다.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable, Sequence

import numpy as np


DEFAULT_TOLERANCE = 1.0e-12


def _record_dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("dimension must be an integer of at least two")
    return value


def _selected_label(value: int, dimension: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("selected label must be an integer")
    if not 0 <= value < dimension:
        raise ValueError("selected label must lie in the record range")
    return value


def _positive_tolerance(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _validated_density(
    state: np.ndarray,
    dimension: int,
    *,
    tolerance: float,
) -> np.ndarray:
    density = np.asarray(state, dtype=np.complex128)
    if density.shape != (dimension, dimension):
        raise ValueError(f"density matrix must have shape ({dimension}, {dimension})")
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


def controlled_record_unitary(dimension: int) -> np.ndarray:
    """치환 유니터리 ``|i,j> -> |i,j+i mod n>`` 을 돌려준다."""

    size = _record_dimension(dimension)
    unitary = np.zeros((size * size, size * size), dtype=np.complex128)
    for system_label in range(size):
        for apparatus_label in range(size):
            source = system_label * size + apparatus_label
            target = system_label * size + (apparatus_label + system_label) % size
            unitary[target, source] = 1.0
    return unitary


def controlled_record_inverse(dimension: int) -> np.ndarray:
    """명시적 역사상 ``|i,j> -> |i,j-i mod n>`` 을 돌려준다."""

    size = _record_dimension(dimension)
    inverse = np.zeros((size * size, size * size), dtype=np.complex128)
    for system_label in range(size):
        for apparatus_label in range(size):
            source = system_label * size + apparatus_label
            target = system_label * size + (apparatus_label - system_label) % size
            inverse[target, source] = 1.0
    return inverse


def apparatus_zero_embedding(dimension: int) -> np.ndarray:
    """``|psi>`` 를 ``|psi> tensor |0>`` 으로 매장한다."""

    size = _record_dimension(dimension)
    embedding = np.zeros((size * size, size), dtype=np.complex128)
    for system_label in range(size):
        embedding[system_label * size, system_label] = 1.0
    return embedding


def record_isometry(dimension: int) -> np.ndarray:
    """``V|i> = |i,i>`` 를 만족하는 ``V = W (I tensor |0>)`` 를 돌려준다."""

    size = _record_dimension(dimension)
    return controlled_record_unitary(size) @ apparatus_zero_embedding(size)


def record_kraus_operators(dimension: int) -> tuple[np.ndarray, ...]:
    """기록 등거리 사상에서 ``K_a = <a|_A V`` 를 추출한다."""

    size = _record_dimension(dimension)
    tensor = record_isometry(size).reshape(size, size, size)
    return tuple(tensor[:, label, :].copy() for label in range(size))


def projective_dephasing(state: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """비선택 채널 ``rho -> sum_i P_i rho P_i`` 를 적용한다."""

    tol = _positive_tolerance(tolerance)
    density = np.asarray(state, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("state must be a square density matrix")
    size = _record_dimension(density.shape[0])
    density = _validated_density(density, size, tolerance=tol)
    return np.diag(np.diag(density)).astype(np.complex128)


def selective_update(
    state: np.ndarray,
    selected: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """``selected`` 에 대한 확률, 부정규화 연산, 사후 상태를 돌려준다."""

    tol = _positive_tolerance(tolerance)
    density = np.asarray(state, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("state must be a square density matrix")
    size = _record_dimension(density.shape[0])
    density = _validated_density(density, size, tolerance=tol)
    label = _selected_label(selected, size)
    projector = np.zeros((size, size), dtype=np.complex128)
    projector[label, label] = 1.0
    operation = projector @ density @ projector
    probability = float(np.trace(operation).real)
    if probability <= tol:
        raise ValueError("selected outcome must have positive probability")
    return probability, operation, operation / probability


def partial_trace_apparatus(
    joint_state: np.ndarray,
    dimension: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """계-우선 결합 밀도 행렬에서 장치(apparatus)를 부분 대각합한다."""

    size = _record_dimension(dimension)
    tol = _positive_tolerance(tolerance)
    joint = _validated_density(joint_state, size * size, tolerance=tol)
    tensor = joint.reshape(size, size, size, size)
    return np.trace(tensor, axis1=1, axis2=3)


def partial_trace_system(
    joint_state: np.ndarray,
    dimension: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """계-우선 결합 밀도 행렬에서 계를 부분 대각합한다."""

    size = _record_dimension(dimension)
    tol = _positive_tolerance(tolerance)
    joint = _validated_density(joint_state, size * size, tolerance=tol)
    tensor = joint.reshape(size, size, size, size)
    return np.trace(tensor, axis1=0, axis2=2)


def fine_visibility_labels(dimension: int, selected: int) -> tuple[tuple[str, int], ...]:
    """라벨을 돌려주는 선언된 미세 라벨 상 위로 전단사로 보낸다."""

    size = _record_dimension(dimension)
    visible = _selected_label(selected, size)
    return tuple(
        ("visible" if label == visible else "hidden", label)
        for label in range(size)
    )


def coarse_visibility_labels(dimension: int, selected: int) -> tuple[str, ...]:
    """개별 숨은 정체를 잊고 visible/hidden 만 남긴다."""

    return tuple(sector for sector, _ in fine_visibility_labels(dimension, selected))


def _apply_kraus_channel(
    operators: tuple[np.ndarray, ...],
    state: np.ndarray,
) -> np.ndarray:
    return sum(
        (operator @ state @ operator.conj().T for operator in operators),
        np.zeros_like(state),
    )


def _choi_matrix(operators: tuple[np.ndarray, ...]) -> np.ndarray:
    size = operators[0].shape[0]
    choi = np.zeros((size * size, size * size), dtype=np.complex128)
    for operator in operators:
        vector = operator.reshape(-1, order="F")
        choi += np.outer(vector, vector.conj())
    return choi


@dataclass(frozen=True)
class FineUnitaryCoarseObservationCertificate:
    dimension: int
    selected_label: int
    branch_probabilities: tuple[float, ...]
    visible_probability: float
    hidden_probability: float
    fine_labels: tuple[tuple[str, int], ...]
    coarse_labels: tuple[str, ...]
    unitary_left_residual: float
    unitary_right_residual: float
    explicit_inverse_residual: float
    record_isometry_residual: float
    record_output_residual: float
    inverse_recovery_residual: float
    reduced_system_residual: float
    reduced_apparatus_residual: float
    kraus_completeness_residual: float
    kraus_channel_residual: float
    choi_minimum_eigenvalue: float
    distinct_input_residual: float
    nonselective_collision_residual: float
    selective_operation_collision_residual: float
    selective_posterior_collision_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *,
    dimension: int = 3,
    selected: int = 1,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FineUnitaryCoarseObservationCertificate:
    """두 사상의 구별에 대한 결정론적 유한 증명서를 만든다."""

    size = _record_dimension(dimension)
    label = _selected_label(selected, size)
    tol = _positive_tolerance(tolerance)
    identity_joint = np.eye(size * size, dtype=np.complex128)
    identity_system = np.eye(size, dtype=np.complex128)

    unitary = controlled_record_unitary(size)
    inverse = controlled_record_inverse(size)
    embedding = apparatus_zero_embedding(size)
    isometry = unitary @ embedding
    unitary_left_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - identity_joint, ord=2)
    )
    unitary_right_residual = float(
        np.linalg.norm(unitary @ unitary.conj().T - identity_joint, ord=2)
    )
    explicit_inverse_residual = max(
        float(np.linalg.norm(inverse @ unitary - identity_joint, ord=2)),
        float(np.linalg.norm(unitary @ inverse - identity_joint, ord=2)),
        float(np.linalg.norm(inverse - unitary.conj().T, ord=2)),
    )
    record_isometry_residual = float(
        np.linalg.norm(isometry.conj().T @ isometry - identity_system, ord=2)
    )

    phases = np.exp(2.0j * np.pi * np.arange(size) / size)
    amplitudes = phases / math.sqrt(size)
    apparatus_zero = np.zeros(size, dtype=np.complex128)
    apparatus_zero[0] = 1.0
    initial_joint_vector = np.kron(amplitudes, apparatus_zero)
    recorded_vector = unitary @ initial_joint_vector
    expected_recorded_vector = np.zeros(size * size, dtype=np.complex128)
    for branch, amplitude in enumerate(amplitudes):
        expected_recorded_vector[branch * size + branch] = amplitude
    record_output_residual = float(
        np.linalg.norm(recorded_vector - expected_recorded_vector)
    )
    inverse_recovery_residual = float(
        np.linalg.norm(inverse @ recorded_vector - initial_joint_vector)
    )

    pure_state = np.outer(amplitudes, amplitudes.conj())
    probabilities = tuple(float(abs(amplitude) ** 2) for amplitude in amplitudes)
    diagonal_state = np.diag(probabilities).astype(np.complex128)
    recorded_density = np.outer(recorded_vector, recorded_vector.conj())
    reduced_system = partial_trace_apparatus(recorded_density, size, tolerance=tol)
    reduced_apparatus = partial_trace_system(recorded_density, size, tolerance=tol)
    dephased = projective_dephasing(pure_state, tolerance=tol)
    reduced_system_residual = float(np.linalg.norm(reduced_system - dephased, ord=2))
    reduced_apparatus_residual = float(np.linalg.norm(reduced_apparatus - dephased, ord=2))

    kraus = record_kraus_operators(size)
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        np.zeros_like(identity_system),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(completeness - identity_system, ord=2)
    )
    kraus_channel_residual = float(
        np.linalg.norm(_apply_kraus_channel(kraus, pure_state) - dephased, ord=2)
    )
    choi_minimum_eigenvalue = float(np.linalg.eigvalsh(_choi_matrix(kraus)).min())

    pure_probability, pure_operation, pure_posterior = selective_update(
        pure_state, label, tolerance=tol
    )
    diagonal_probability, diagonal_operation, diagonal_posterior = selective_update(
        diagonal_state, label, tolerance=tol
    )
    distinct_input_residual = float(np.linalg.norm(pure_state - diagonal_state, ord=2))
    nonselective_collision_residual = float(
        np.linalg.norm(
            projective_dephasing(pure_state, tolerance=tol)
            - projective_dephasing(diagonal_state, tolerance=tol),
            ord=2,
        )
    )
    selective_operation_collision_residual = max(
        abs(pure_probability - diagonal_probability),
        float(np.linalg.norm(pure_operation - diagonal_operation, ord=2)),
    )
    selective_posterior_collision_residual = float(
        np.linalg.norm(pure_posterior - diagonal_posterior, ord=2)
    )

    fine_labels = fine_visibility_labels(size, label)
    coarse_labels = coarse_visibility_labels(size, label)
    # 선언된 공역은 정확히 F_{a,n} = {(visible, a)} union {(hidden, i): i != a} 이며
    # ``set(fine_labels)`` 로 표현된다. 그 원소 수가 정의역과 같으면 그 공역 위로의
    # 전단사가 증명되지만, 더 큰 데카르트 곱 {visible, hidden} x L_n 위로는 아니다.
    fine_bijective_onto_declared_image = len(set(fine_labels)) == size
    coarse_injective = len(set(coarse_labels)) == size
    visible_probability = probabilities[label]
    hidden_probability = sum(
        probability
        for branch, probability in enumerate(probabilities)
        if branch != label
    )
    unitary_certified = max(
        unitary_left_residual,
        unitary_right_residual,
        explicit_inverse_residual,
        record_isometry_residual,
        record_output_residual,
        inverse_recovery_residual,
    ) <= 10.0 * tol
    cptp_certified = (
        kraus_completeness_residual <= 10.0 * tol
        and kraus_channel_residual <= 10.0 * tol
        and choi_minimum_eigenvalue >= -10.0 * tol
    )

    return FineUnitaryCoarseObservationCertificate(
        dimension=size,
        selected_label=label,
        branch_probabilities=probabilities,
        visible_probability=visible_probability,
        hidden_probability=hidden_probability,
        fine_labels=fine_labels,
        coarse_labels=coarse_labels,
        unitary_left_residual=unitary_left_residual,
        unitary_right_residual=unitary_right_residual,
        explicit_inverse_residual=explicit_inverse_residual,
        record_isometry_residual=record_isometry_residual,
        record_output_residual=record_output_residual,
        inverse_recovery_residual=inverse_recovery_residual,
        reduced_system_residual=reduced_system_residual,
        reduced_apparatus_residual=reduced_apparatus_residual,
        kraus_completeness_residual=kraus_completeness_residual,
        kraus_channel_residual=kraus_channel_residual,
        choi_minimum_eigenvalue=choi_minimum_eigenvalue,
        distinct_input_residual=distinct_input_residual,
        nonselective_collision_residual=nonselective_collision_residual,
        selective_operation_collision_residual=selective_operation_collision_residual,
        selective_posterior_collision_residual=selective_posterior_collision_residual,
        dimensions={
            "basis_labels_dimensionless": True,
            "unitary_and_density_entries_dimensionless": True,
            "branch_probabilities_dimensionless": True,
            "energy_requires_independent_hamiltonian_scale": True,
            "physical_duration_requires_independent_time_scale": True,
        },
        accounting={
            "branch_probabilities_sum_to_one": math.isclose(
                sum(probabilities), 1.0, abs_tol=10.0 * tol
            ),
            "visible_plus_hidden_probability_sum_to_one": math.isclose(
                visible_probability + hidden_probability, 1.0, abs_tol=10.0 * tol
            ),
            "hidden_labels_retained_individually_in_fine_label_map": (
                fine_bijective_onto_declared_image
            ),
            "coarse_and_fine_probabilities_added_as_separate_energy": False,
            "energy_or_stress_assigned_without_ledger": False,
        },
        boundaries={
            "selected_label_is_supplied_to_conditional_readout": True,
            "selected_label_is_not_an_input_to_controlled_unitary": True,
            "finite_w_is_declared_model_not_actual_universe_dynamics": True,
            "record_register_is_abstract_not_a_durable_pointer": True,
            "fine_sort_is_label_only_not_physical_branch_dynamics": True,
            "fine_sort_codomain_is_declared_image_not_full_cartesian_product": True,
            "finite_label_topology_declared_discrete": True,
            "coarse_label_map_many_to_one_only_for_dimension_at_least_three": True,
            "finite_dimension_is_hilbert_label_dimension_not_spacetime_dimension": True,
            "hilbert_state_map_uses_norm_topology_only": True,
            "cptp_claim_is_for_the_explicit_projective_instrument": True,
        },
        alternatives={
            "local_decoherence_instrument_route_open": True,
            "actual_selection_law_route_open": True,
            "representation_invariant_geometry_route_open": True,
        },
        status={
            "declared_finite_controlled_unitary_bijective": unitary_certified,
            "record_isometry_bijective_onto_its_image": record_isometry_residual <= 10.0 * tol,
            "record_isometry_surjective_onto_full_joint_space": False,
            "fine_discrete_label_sort_bijective_onto_declared_image": (
                fine_bijective_onto_declared_image
            ),
            "fine_discrete_label_bijection_onto_image_is_homeomorphism": (
                fine_bijective_onto_declared_image
            ),
            "coarse_visibility_readout_injective": coarse_injective,
            "nonselective_dephasing_many_to_one_witness": (
                distinct_input_residual > 10.0 * tol
                and nonselective_collision_residual <= 10.0 * tol
            ),
            "selective_update_many_to_one_witness": (
                distinct_input_residual > 10.0 * tol
                and selective_operation_collision_residual <= 10.0 * tol
                and selective_posterior_collision_residual <= 10.0 * tol
            ),
            "explicit_projective_record_channel_cptp": cptp_certified,
            "premeasurement_components_preserved_by_fine_unitary": unitary_certified,
            "unitary_selects_one_unique_actual_outcome": False,
            "durable_physical_pointer_derived": False,
            "energy_hamiltonian_or_transfer_derived": False,
            "spacetime_homeomorphism_derived": False,
            "spacetime_metric_or_curvature_derived": False,
            "fold_stress_or_gravity_derived": False,
            "relativistic_no_signalling_derived": False,
            "holdout_complete": False,
            "success_gates_5_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--selected", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(certificate(dimension=args.dimension, selected=args.selected)),
            indent=2,
            sort_keys=True,
        )
    )


_TOL = 2.0e-11


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return value


def _legendre_integral(function, left: float, right: float, *, order: int = 192) -> float:
    """공급된 유한 구간 위의 결정론적 가우스-르장드르(Gauss--Legendre) 적분이다."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    radii = 0.5 * (right - left) * nodes + 0.5 * (right + left)
    return float(0.5 * (right - left) * np.dot(weights, function(radii)))


def _partial_trace_a(rho_ab: np.ndarray, a_dimension: int, b_dimension: int) -> np.ndarray:
    return np.trace(rho_ab.reshape(a_dimension, b_dimension, a_dimension, b_dimension), axis1=0, axis2=2)


@dataclass(frozen=True)
class FiniteProbabilityDeformationReadoutCertificate:
    compactness: float
    domain_ratio: float
    holdout_x1: float
    holdout_x2: float
    normalizer: float
    log_normalizer: float
    normalization_residual: float
    constant_shift_invariance_residual: float
    inward_likelihood_ratio: float
    holdout_probability: float
    chi_continuity_residual_at_surface: float
    scaled_radial_laplacian_inside: float
    scaled_radial_laplacian_outside: float
    inside_chi_prime_over_x: float
    outside_x_squared_chi_prime: float
    scaled_acceleration_at_x_half: float
    scaled_acceleration_at_holdout_x1: float
    chi_equals_minus_newtonian_potential_over_c_squared: bool
    finite_sphere_regulates_normalization: bool
    point_source_global_normalization_available: bool
    point_source_uniform_volume_integral_diverges: bool
    epsilon_mass_dimension: int
    scaled_radius_mass_dimension: int
    chi_mass_dimension: int
    normalizer_mass_dimension: int
    probability_mass_dimension: int
    chi_derivative_invariant_mass_dimension: int
    dimensions_pass: bool
    parameter_fit_count: int
    internal_radial_holdout_only: bool
    observational_holdout_gate_closed: bool
    record_probability_rho0: tuple[float, float]
    record_probability_rho1: tuple[float, float]
    record_probability_rho2: tuple[float, float]
    distinct_microstates_same_sharp_record: bool
    kraus_completeness_residual: float
    choi_minimum_eigenvalue: float
    channel_trace_preservation_residual: float
    channel_completely_positive: bool
    channel_trace_preserving: bool
    sharp_projector_repeatability_residual: float
    immediate_sharp_repeatability: bool
    classical_record_dephasing_idempotence_residual: float
    single_witness_remote_marginal_residual: float
    no_probability_double_weighting: bool
    newtonian_reparameterization_only: bool = True
    independent_chi_action_or_dynamics_derived: bool = False
    probability_current_or_attraction_mechanism_derived: bool = False
    causal_retarded_field_or_c_front_derived: bool = False
    scalar_to_gr_or_lensing_derived: bool = False
    gravity_energy_or_backreaction_derived: bool = False
    quantum_matter_dependent_chi_channel_derived: bool = False
    general_observation_repeatability_derived: bool = False
    physical_selection_derived: bool = False
    ideal_point_source_normalization_derived: bool = False
    homology_cohomology_self_duality_derived: bool = False
    actual_data_holdout_or_gates_5_to_8_closed: bool = False
    two_residuals_or_complexity_success: bool = False


def certify_finite_probability_deformation_readout(
    *, compactness: float = 0.01, domain_ratio: float = 10.0,
    holdout_x1: float = 2.0, holdout_x2: float = 3.0,
) -> FiniteProbabilityDeformationReadoutCertificate:
    """유한 영역 라돈-니코딤 증인과 별도의 0차원 선명 기록 증인을 돌려준다.

    ``compactness = GM/(R_s c^2)`` 와 ``x=r/R_s`` 는 무차원이다. 공급된 유한 구는
    ``chi=-Phi/c^2`` 와 기저 측도 ``dmu0=3*x^2/domain_ratio^3 dx`` 를 가진다.
    전역 점원 정규화 상수는 의도적으로 제공하지 않는다: 그 균일 부피 적분은 발산한다.
    """

    epsilon = _unit_interval(compactness, "compactness")
    domain = _positive(domain_ratio, "domain_ratio")
    if domain <= 1.0:
        raise ValueError("domain_ratio must exceed 1")
    x1 = _positive(holdout_x1, "holdout_x1")
    x2 = _positive(holdout_x2, "holdout_x2")
    if not 1.0 < x1 < x2 < domain:
        raise ValueError("holdout radii must satisfy 1 < holdout_x1 < holdout_x2 < domain_ratio")

    def chi(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 1.0, 0.5 * epsilon * (3.0 - x * x), epsilon / x)

    def weighted_volume(x: np.ndarray) -> np.ndarray:
        # 이것은 정확히 ``3*x**2/domain**3`` 이지만, 매우 크되 유한한 공급 영역에서
        # 중간 단계 ``x**2`` 의 오버플로를 피한다.
        return 3.0 * (x / domain) ** 2 * np.exp(chi(x)) / domain

    # x=1 에서 분할한다: 공급된 퍼텐셜은 연속이지만 그 도함수는 다르다.
    normalizer = _legendre_integral(weighted_volume, 0.0, 1.0) + _legendre_integral(weighted_volume, 1.0, domain)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("normalizer must be finite and positive on the supplied domain")
    log_normalizer = math.log(normalizer)
    normalized_integral = (
        _legendre_integral(lambda x: weighted_volume(x) / normalizer, 0.0, 1.0)
        + _legendre_integral(lambda x: weighted_volume(x) / normalizer, 1.0, domain)
    )
    holdout = _legendre_integral(lambda x: weighted_volume(x) / normalizer, x1, x2)
    if not math.isfinite(holdout) or not 0.0 < holdout < 1.0:
        raise ValueError("holdout probability must be finite and lie in (0, 1)")
    shift = 0.731
    shifted_normalizer = (
        _legendre_integral(lambda x: weighted_volume(x) * math.exp(shift), 0.0, 1.0)
        + _legendre_integral(lambda x: weighted_volume(x) * math.exp(shift), 1.0, domain)
    )
    # 같은 부피의 껍질 둘: 안쪽 것이 더 큰 라돈-니코딤 인자를 가진다.
    inward_ratio = math.exp(float(chi(np.array([x1]))[0] - chi(np.array([x2]))[0]))

    p0 = np.diag((1.0, 1.0, 0.0)).astype(complex)
    p1 = np.diag((0.0, 0.0, 1.0)).astype(complex)
    kraus = (p0, p1)
    completeness = sum((operator.conj().T @ operator for operator in kraus), np.zeros((3, 3), complex))
    rho0 = np.diag((1.0, 0.0, 0.0)).astype(complex)
    rho1 = np.diag((0.0, 1.0, 0.0)).astype(complex)
    rho2 = np.diag((0.0, 0.0, 1.0)).astype(complex)
    probabilities0 = tuple(float(np.trace(operator @ rho0 @ operator.conj().T).real) for operator in kraus)
    probabilities1 = tuple(float(np.trace(operator @ rho1 @ operator.conj().T).real) for operator in kraus)
    probabilities2 = tuple(float(np.trace(operator @ rho2 @ operator.conj().T).real) for operator in kraus)
    channel = lambda rho: sum((operator @ rho @ operator.conj().T for operator in kraus), np.zeros_like(rho))
    choi = sum(
        np.kron(np.eye(3)[i : i + 1].T @ np.eye(3)[j : j + 1], channel(np.eye(3, dtype=complex)[i : i + 1].T @ np.eye(3, dtype=complex)[j : j + 1]))
        for i in range(3) for j in range(3)
    )
    # 대표 기록 하나를 위상 소거(dephasing)한 뒤 다시 위상 소거한다.
    record = np.array(((0.5, 0.25j), (-0.25j, 0.5)), dtype=complex)
    record_dephase = lambda matrix: np.diag(np.diag(matrix))
    repeatability_residual = max(
        float(np.linalg.norm(
            second @ first - (first if first_label == second_label else np.zeros((3, 3))),
            ord=2,
        ))
        for first_label, first in enumerate(kraus)
        for second_label, second in enumerate(kraus)
    )

    psi = np.zeros(6, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2.0)  # |0>_A |0>_B
    psi[5] = 1.0 / math.sqrt(2.0)  # |2>_A |1>_B
    rho_ab = np.outer(psi, psi.conj())
    remote_before = _partial_trace_a(rho_ab, 3, 2)
    local_nonselective = sum((np.kron(operator, np.eye(2)) @ rho_ab @ np.kron(operator, np.eye(2)).conj().T for operator in kraus), np.zeros_like(rho_ab))
    remote_after = _partial_trace_a(local_nonselective, 3, 2)

    return FiniteProbabilityDeformationReadoutCertificate(
        compactness=epsilon, domain_ratio=domain, holdout_x1=x1, holdout_x2=x2,
        normalizer=normalizer, log_normalizer=log_normalizer,
        normalization_residual=abs(normalized_integral - 1.0),
        constant_shift_invariance_residual=abs(shifted_normalizer / math.exp(shift) / normalizer - 1.0),
        inward_likelihood_ratio=inward_ratio, holdout_probability=holdout,
        chi_continuity_residual_at_surface=abs(0.5 * epsilon * (3.0 - 1.0) - epsilon),
        scaled_radial_laplacian_inside=-3.0 * epsilon,
        scaled_radial_laplacian_outside=0.0,
        inside_chi_prime_over_x=-epsilon,
        outside_x_squared_chi_prime=-epsilon,
        scaled_acceleration_at_x_half=-0.5 * epsilon,
        scaled_acceleration_at_holdout_x1=-epsilon / x1**2,
        chi_equals_minus_newtonian_potential_over_c_squared=True,
        finite_sphere_regulates_normalization=True,
        point_source_global_normalization_available=False,
        point_source_uniform_volume_integral_diverges=True,
        epsilon_mass_dimension=0, scaled_radius_mass_dimension=0, chi_mass_dimension=0,
        normalizer_mass_dimension=0, probability_mass_dimension=0,
        chi_derivative_invariant_mass_dimension=0, dimensions_pass=True,
        parameter_fit_count=0, internal_radial_holdout_only=True,
        observational_holdout_gate_closed=False,
        record_probability_rho0=probabilities0, record_probability_rho1=probabilities1,
        record_probability_rho2=probabilities2,
        distinct_microstates_same_sharp_record=(not np.allclose(rho0, rho1) and probabilities0 == probabilities1 == (1.0, 0.0)),
        kraus_completeness_residual=float(np.linalg.norm(completeness - np.eye(3), ord=2)),
        choi_minimum_eigenvalue=float(np.linalg.eigvalsh(choi).min()),
        channel_trace_preservation_residual=max(abs(np.trace(channel(np.eye(3) / 3.0)) - 1.0), 0.0),
        channel_completely_positive=bool(np.linalg.eigvalsh(choi).min() >= -_TOL),
        channel_trace_preserving=bool(np.linalg.norm(completeness - np.eye(3), ord=2) <= _TOL),
        sharp_projector_repeatability_residual=repeatability_residual,
        immediate_sharp_repeatability=bool(repeatability_residual <= _TOL),
        classical_record_dephasing_idempotence_residual=float(np.linalg.norm(record_dephase(record_dephase(record)) - record_dephase(record), ord=2)),
        single_witness_remote_marginal_residual=float(np.linalg.norm(remote_after - remote_before, ord=2)),
        no_probability_double_weighting=True,
    )


I2 = np.eye(2, dtype=np.complex128)
P0 = np.diag([1.0, 0.0]).astype(np.complex128)
P1 = np.diag([0.0, 1.0]).astype(np.complex128)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_density_kraus(state: np.ndarray, *, tolerance: float) -> np.ndarray:
    density = np.asarray(state, dtype=np.complex128)
    if density.shape != (2, 2):
        raise ValueError("the finite witness requires a 2 by 2 density matrix")
    if not np.isfinite(density).all():
        raise ValueError("density matrix entries must be finite")
    if np.linalg.norm(density - density.conj().T, ord="fro") > tolerance:
        raise ValueError("density matrix must be Hermitian")
    if abs(float(np.trace(density).real) - 1.0) > tolerance:
        raise ValueError("density matrix must have unit trace")
    if abs(float(np.trace(density).imag)) > tolerance:
        raise ValueError("density matrix trace must be real")
    if float(np.linalg.eigvalsh(density).min()) < -tolerance:
        raise ValueError("density matrix must be positive semidefinite")
    return density


def apply_cp_map(kraus: Sequence[np.ndarray], state: np.ndarray) -> np.ndarray:
    """내부 라벨을 노출하지 않고 유한 크라우스 족을 적용한다."""

    if not kraus:
        raise ValueError("a Kraus family must be non-empty")
    density = np.asarray(state, dtype=np.complex128)
    shape = density.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("state must be a square matrix")
    operators = tuple(np.asarray(operator, dtype=np.complex128) for operator in kraus)
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all Kraus operators must match the state dimension")
    if any(not np.isfinite(operator).all() for operator in operators):
        raise ValueError("Kraus operator entries must be finite")
    return sum(
        (operator @ density @ operator.conj().T for operator in operators),
        np.zeros(shape, dtype=np.complex128),
    )


def choi_matrix(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """열 우선 벡터화로 ``sum |K>><<K|`` 를 돌려준다."""

    if not kraus:
        raise ValueError("a Kraus family must be non-empty")
    operators = tuple(np.asarray(operator, dtype=np.complex128) for operator in kraus)
    shape = operators[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Kraus operators must be square matrices")
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all Kraus operators must have one common shape")
    if any(not np.isfinite(operator).all() for operator in operators):
        raise ValueError("Kraus operator entries must be finite")
    size = shape[0] * shape[1]
    result = np.zeros((size, size), dtype=np.complex128)
    for operator in operators:
        vector = operator.reshape(-1, order="F")
        result += np.outer(vector, vector.conj())
    return result


def isometric_refinement(
    kraus: Sequence[np.ndarray],
    isometry: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, ...]:
    """``u^dagger u = I`` 를 확인한 뒤 크라우스 족을 ``u`` 로 섞는다."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not kraus:
        raise ValueError("a Kraus family must be non-empty")
    operators = tuple(np.asarray(operator, dtype=np.complex128) for operator in kraus)
    shape = operators[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Kraus operators must be square matrices")
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all Kraus operators must have one common shape")
    mixing = np.asarray(isometry, dtype=np.complex128)
    if mixing.ndim != 2 or mixing.shape[1] != len(operators):
        raise ValueError("isometry columns must equal the original Kraus count")
    if mixing.shape[0] < mixing.shape[1]:
        raise ValueError("isometry cannot have fewer rows than columns")
    if not np.isfinite(mixing).all():
        raise ValueError("isometry entries must be finite")
    identity = np.eye(mixing.shape[1], dtype=np.complex128)
    residual = float(np.linalg.norm(mixing.conj().T @ mixing - identity, ord=2))
    if residual > tolerance:
        raise ValueError("mixing matrix must satisfy u^dagger u = I")
    return tuple(
        sum(
            (mixing[row, column] * operators[column] for column in range(len(operators))),
            np.zeros(shape, dtype=np.complex128),
        )
        for row in range(mixing.shape[0])
    )


def duplicate_operation(operator: np.ndarray, multiplicity: int) -> tuple[np.ndarray, ...]:
    """연산자 하나의 영이 아닌 복사본 ``K/sqrt(k)`` 를 ``k`` 개 돌려준다."""

    count = _positive_integer(multiplicity, "multiplicity")
    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("operator entries must be finite")
    return tuple(matrix / math.sqrt(count) for _ in range(count))


def raw_count_conformal_factor(
    raw_count: int,
    *,
    reference_count: int = 1,
    spacetime_dimension: int = 4,
) -> float:
    """의도적으로 소박한 무차원 척도 ``(N/N*)**(1/D)`` 이다."""

    count = _positive_integer(raw_count, "raw_count")
    reference = _positive_integer(reference_count, "reference_count")
    dimension = _positive_integer(spacetime_dimension, "spacetime_dimension")
    if dimension < 2:
        raise ValueError("spacetime_dimension must be at least two")
    return (count / reference) ** (1.0 / dimension)


@dataclass(frozen=True)
class KrausRefinementCertificate:
    outcome_probability: float
    hidden_multiplicities: tuple[int, ...]
    maximum_operation_residual: float
    maximum_full_completeness_residual: float
    maximum_coarse_probability_residual: float
    maximum_total_probability_residual: float
    maximum_posterior_residual: float
    maximum_choi_residual: float
    numerical_choi_ranks: tuple[int, ...]
    sublabel_probability_sums: tuple[float, ...]
    raw_conformal_factors: tuple[float, ...]
    raw_metric_coefficient_ratios: tuple[float, ...]
    general_isometry_shape: tuple[int, int]
    general_isometry_residual: float
    general_channel_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def kraus_certificate(
    *,
    outcome_probability: float = 0.3,
    hidden_multiplicities: tuple[int, ...] = (1, 2, 16, 37),
    spacetime_dimension: int = 4,
    tolerance: float = DEFAULT_TOLERANCE,
) -> KrausRefinementCertificate:
    """결과별 세분화 장애에 대한 결정론적 증명서를 만든다."""

    if not math.isfinite(outcome_probability) or not 0.0 < outcome_probability < 1.0:
        raise ValueError("outcome_probability must be finite and lie in (0, 1)")
    if not hidden_multiplicities:
        raise ValueError("hidden_multiplicities must be non-empty")
    multiplicities = tuple(
        _positive_integer(value, "hidden multiplicity") for value in hidden_multiplicities
    )
    dimension = _positive_integer(spacetime_dimension, "spacetime_dimension")
    if dimension < 2:
        raise ValueError("spacetime_dimension must be at least two")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    p = outcome_probability
    outcome_operator = math.sqrt(p) * I2
    complement_operator = math.sqrt(1.0 - p) * I2
    states = (
        P0,
        P1,
        0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128),
        0.5 * np.array([[1.0, -1.0j], [1.0j, 1.0]], dtype=np.complex128),
        np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=np.complex128),
    )
    states = tuple(_validated_density_kraus(state, tolerance=tolerance) for state in states)
    base_family = (outcome_operator,)
    base_choi = choi_matrix(base_family)

    operation_residuals: list[float] = []
    completeness_residuals: list[float] = []
    probability_residuals: list[float] = []
    total_probability_residuals: list[float] = []
    posterior_residuals: list[float] = []
    choi_residuals: list[float] = []
    numerical_choi_ranks: list[int] = []
    sublabel_sums: list[float] = []

    for count in multiplicities:
        refined = duplicate_operation(outcome_operator, count)
        full_family = refined + (complement_operator,)
        completeness = sum(
            (operator.conj().T @ operator for operator in full_family),
            np.zeros_like(I2),
        )
        completeness_residuals.append(float(np.linalg.norm(completeness - I2, ord=2)))
        refined_choi = choi_matrix(refined)
        choi_residuals.append(float(np.linalg.norm(refined_choi - base_choi, ord=2)))
        choi_singular_values = np.linalg.svd(refined_choi, compute_uv=False)
        choi_scale = float(choi_singular_values[0])
        numerical_choi_ranks.append(
            int(np.count_nonzero(choi_singular_values > tolerance * choi_scale))
        )
        sublabel_sums.append(
            sum(
                float(np.trace(operator @ states[-1] @ operator.conj().T).real)
                for operator in refined
            )
        )
        for state in states:
            base_output = apply_cp_map(base_family, state)
            refined_output = apply_cp_map(refined, state)
            operation_residuals.append(float(np.linalg.norm(refined_output - base_output, ord=2)))
            base_probability = float(np.trace(base_output).real)
            refined_probability = float(np.trace(refined_output).real)
            complement_probability = float(
                np.trace(complement_operator @ state @ complement_operator.conj().T).real
            )
            probability_residuals.append(abs(refined_probability - base_probability))
            total_probability_residuals.append(
                max(
                    abs(complement_probability - (1.0 - p)),
                    abs(refined_probability + complement_probability - 1.0),
                )
            )
            posterior_residuals.append(
                float(
                    np.linalg.norm(
                        refined_output / refined_probability - base_output / base_probability,
                        ord=2,
                    )
                )
            )

    # 자명하지 않은 4 x 2 등거리 사상으로 큐비트 위상 소거 채널 위에서 일반 혼합 공식을
    # 등복사본 구성과 독립적으로 검증한다.
    general_isometry = 0.5 * np.array(
        [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0j], [1.0, -1.0j]],
        dtype=np.complex128,
    )
    isometry_identity = np.eye(2, dtype=np.complex128)
    general_isometry_residual = float(
        np.linalg.norm(general_isometry.conj().T @ general_isometry - isometry_identity, ord=2)
    )
    dephasing = (P0, P1)
    mixed_dephasing = isometric_refinement(dephasing, general_isometry, tolerance=tolerance)
    general_channel_residual = max(
        float(np.linalg.norm(apply_cp_map(mixed_dephasing, state) - apply_cp_map(dephasing, state), ord=2))
        for state in states
    )

    omega = tuple(
        raw_count_conformal_factor(count, spacetime_dimension=dimension)
        for count in multiplicities
    )
    metric_ratios = tuple(value * value for value in omega)
    maximum_operation_residual = max(operation_residuals)
    maximum_completeness_residual = max(completeness_residuals)
    maximum_probability_residual = max(probability_residuals)
    maximum_total_probability_residual = max(total_probability_residuals)
    maximum_posterior_residual = max(posterior_residuals)
    maximum_choi_residual = max(choi_residuals)
    quantum_invariant = max(
        maximum_operation_residual,
        maximum_completeness_residual,
        maximum_probability_residual,
        maximum_total_probability_residual,
        maximum_posterior_residual,
        maximum_choi_residual,
        general_isometry_residual,
        general_channel_residual,
    ) <= 10.0 * tolerance
    raw_metric_changes = len({round(value, 12) for value in metric_ratios}) > 1

    return KrausRefinementCertificate(
        outcome_probability=p,
        hidden_multiplicities=multiplicities,
        maximum_operation_residual=maximum_operation_residual,
        maximum_full_completeness_residual=maximum_completeness_residual,
        maximum_coarse_probability_residual=maximum_probability_residual,
        maximum_total_probability_residual=maximum_total_probability_residual,
        maximum_posterior_residual=maximum_posterior_residual,
        maximum_choi_residual=maximum_choi_residual,
        numerical_choi_ranks=tuple(numerical_choi_ranks),
        sublabel_probability_sums=tuple(sublabel_sums),
        raw_conformal_factors=omega,
        raw_metric_coefficient_ratios=metric_ratios,
        general_isometry_shape=general_isometry.shape,
        general_isometry_residual=general_isometry_residual,
        general_channel_residual=general_channel_residual,
        dimensions={
            "raw_count_dimensionless": True,
            "count_ratio_dimensionless": True,
            "conformal_factor_dimensionless": True,
            "absolute_volume_requires_independent_reference_scale": True,
            "dimension_consistency_does_not_make_count_physical": True,
        },
        accounting={
            "refined_sublabel_probabilities_sum_to_coarse_probability": all(
                math.isclose(value, p, abs_tol=10.0 * tolerance) for value in sublabel_sums
            ),
            "coarse_plus_refined_probability_double_counting_forbidden": True,
            "representation_only_sublabel_adds_energy_or_stress": False,
            "energy_receipt_or_stress_used": False,
        },
        boundaries={
            "sublabel_is_unobserved": True,
            "physical_pointer_record_derived": False,
            "zero_probability_posterior_excluded": True,
            "finite_dimensional_only": True,
        },
        alternatives={
            "physical_recorded_refinement_open": True,
            "choi_invariant_route_open": True,
            "independent_causal_order_and_volume_route_open": True,
        },
        status={
            "outcome_operation_isometry_invariant": quantum_invariant,
            "coarse_probability_invariant": maximum_probability_residual <= 10.0 * tolerance,
            "posterior_invariant": maximum_posterior_residual <= 10.0 * tolerance,
            "cptp_completeness_preserved": maximum_completeness_residual <= 10.0 * tolerance,
            "choi_matrix_invariant": maximum_choi_residual <= 10.0 * tolerance,
            "raw_hidden_count_invariant": False,
            "raw_count_metric_changes_for_same_instrument": raw_metric_changes and quantum_invariant,
            "raw_count_defines_physical_volume_or_metric": False,
            "choi_rank_numerically_invariant": len(set(numerical_choi_ranks)) == 1,
            "minimal_kraus_rank_theorem_proved_by_finite_regression": False,
            "physical_record_algebra_derived": False,
            "local_volume_measure_derived": False,
            "metric_or_curvature_derived": False,
            "fold_stress_derived": False,
            "gr_lensing_backreaction_derived": False,
            "holdout_complete": False,
            "success_gates_5_to_8_complete": False,
        },
    )


def kraus_run() -> dict[str, object]:
    return asdict(kraus_certificate())


def kraus_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-probability", type=float, default=0.3)
    args = parser.parse_args()
    print(json.dumps(asdict(kraus_certificate(outcome_probability=args.outcome_probability)), indent=2, sort_keys=True))


CANONICAL_STABLE_PARAMETERS = (9.0, 4.0, 2.0)
CANONICAL_TACHYON_PARAMETERS = (1.0, 1.0, 2.0)
CANONICAL_BOUNDARY_PARAMETERS = (1.0, 1.0, 1.0)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_fold(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_covector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain four finite components")
    return result


def _tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class DimensionAudit:
    record_field_mass_dimension: int
    fold_field_mass_dimension: int
    derivative_mass_dimension: int
    record_mass_squared_dimension: int
    fold_mass_squared_dimension: int
    mixing_kappa_mass_dimension: int
    source_coefficient_mass_dimension: int
    lagrangian_density_mass_dimension: int
    stress_mass_dimension: int
    ward_current_mass_dimension: int
    spacetime_volume_mass_dimension: int
    action_mass_dimension: int
    dimensions_pass: bool
    probability_used_as_source_coefficient: bool


def dimension_audit() -> DimensionAudit:
    """선언된 작용의 자연 단위 차원 장부를 돌려준다."""

    record_dimension = 1
    fold_dimension = 1
    derivative_dimension = 1
    mass_squared_dimension = 2
    kappa_dimension = 2
    source_dimension = kappa_dimension + record_dimension
    lagrangian_dimension = 4
    stress_dimension = 4
    ward_dimension = 5
    volume_dimension = -4
    action_dimension = lagrangian_dimension + volume_dimension
    dimensions_pass = all(
        (
            2 * (derivative_dimension + record_dimension)
            == lagrangian_dimension,
            2 * (derivative_dimension + fold_dimension)
            == lagrangian_dimension,
            mass_squared_dimension + 2 * record_dimension
            == lagrangian_dimension,
            mass_squared_dimension + 2 * fold_dimension
            == lagrangian_dimension,
            kappa_dimension + record_dimension + fold_dimension
            == lagrangian_dimension,
            source_dimension + fold_dimension == lagrangian_dimension,
            stress_dimension == lagrangian_dimension,
            source_dimension + derivative_dimension + fold_dimension
            == ward_dimension,
            action_dimension == 0,
        )
    )
    return DimensionAudit(
        record_field_mass_dimension=record_dimension,
        fold_field_mass_dimension=fold_dimension,
        derivative_mass_dimension=derivative_dimension,
        record_mass_squared_dimension=mass_squared_dimension,
        fold_mass_squared_dimension=mass_squared_dimension,
        mixing_kappa_mass_dimension=kappa_dimension,
        source_coefficient_mass_dimension=source_dimension,
        lagrangian_density_mass_dimension=lagrangian_dimension,
        stress_mass_dimension=stress_dimension,
        ward_current_mass_dimension=ward_dimension,
        spacetime_volume_mass_dimension=volume_dimension,
        action_mass_dimension=action_dimension,
        dimensions_pass=dimensions_pass,
        probability_used_as_source_coefficient=False,
    )


@dataclass(frozen=True)
class BilinearSpectrumAudit:
    record_mass_squared: float
    fold_mass_squared: float
    mixing_kappa: float
    mass_squared_matrix: tuple[tuple[float, ...], ...]
    trace_mass_squared: float
    determinant_mass_four: float
    eigenmass_squared_high: float
    eigenmass_squared_low: float
    rotation_angle_radians: float
    rotation_angle_degrees: float
    rotation_matrix: tuple[tuple[float, ...], ...]
    rotated_mass_squared_matrix: tuple[tuple[float, ...], ...]
    rotated_off_diagonal_residual: float
    kinetic_rotation_residual: float
    positive_by_principal_minors: bool
    strictly_stable: bool
    tachyonic_mode_present: bool
    boundary_zero_mode_present: bool
    canonical_kinetic_ghost_free: bool


def bilinear_spectrum_audit(
    record_mass_squared: float = 9.0,
    fold_mass_squared: float = 4.0,
    mixing_kappa: float = 2.0,
) -> BilinearSpectrumAudit:
    """``[[m_R^2,-kappa],[-kappa,m_phi^2]]`` 를 대각화한다.

    입력은 모두 질량 차원 2를 가진다. 고정 허용오차는 부동소수점 증인을 분류할 뿐이며
    맞춘 물리 매개변수가 아니다.
    """

    record_mass = _finite(record_mass_squared, "record_mass_squared")
    fold_mass = _finite(fold_mass_squared, "fold_mass_squared")
    kappa = _finite(mixing_kappa, "mixing_kappa")
    matrix = np.asarray(
        ((record_mass, -kappa), (-kappa, fold_mass)),
        dtype=np.float64,
    )
    trace = record_mass + fold_mass
    determinant = record_mass * fold_mass - kappa**2
    discriminant = math.sqrt((record_mass - fold_mass) ** 2 + 4.0 * kappa**2)
    eigen_high = 0.5 * (trace + discriminant)
    eigen_low = 0.5 * (trace - discriminant)

    if kappa == 0.0:
        angle = 0.0
    else:
        angle = 0.5 * math.atan2(-2.0 * kappa, record_mass - fold_mass)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray(
        ((cosine, -sine), (sine, cosine)),
        dtype=np.float64,
    )
    rotated = rotation.T @ matrix @ rotation
    kinetic_residual = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(2, dtype=np.float64)))
    )
    off_diagonal_residual = float(abs(rotated[0, 1]))
    scale_mass_squared = max(
        abs(record_mass),
        abs(fold_mass),
        abs(kappa),
        1.0,
    )
    eigen_tolerance = DEFAULT_TOLERANCE * scale_mass_squared
    positive_by_minors = record_mass > 0.0 and determinant > 0.0
    strictly_stable = positive_by_minors and eigen_low > eigen_tolerance
    tachyonic = eigen_low < -eigen_tolerance
    boundary_zero = abs(eigen_low) <= eigen_tolerance
    return BilinearSpectrumAudit(
        record_mass_squared=record_mass,
        fold_mass_squared=fold_mass,
        mixing_kappa=kappa,
        mass_squared_matrix=_tuple_matrix(matrix),
        trace_mass_squared=trace,
        determinant_mass_four=determinant,
        eigenmass_squared_high=eigen_high,
        eigenmass_squared_low=eigen_low,
        rotation_angle_radians=angle,
        rotation_angle_degrees=math.degrees(angle),
        rotation_matrix=_tuple_matrix(rotation),
        rotated_mass_squared_matrix=_tuple_matrix(rotated),
        rotated_off_diagonal_residual=off_diagonal_residual,
        kinetic_rotation_residual=kinetic_residual,
        positive_by_principal_minors=positive_by_minors,
        strictly_stable=strictly_stable,
        tachyonic_mode_present=tachyonic,
        boundary_zero_mode_present=boundary_zero,
        canonical_kinetic_ghost_free=True,
    )


def require_stable_spectrum(
    record_mass_squared: float,
    fold_mass_squared: float,
    mixing_kappa: float,
) -> BilinearSpectrumAudit:
    """안정 영수증을 돌려주거나 경계 및 그 너머에서 닫힌 채 실패한다."""

    receipt = bilinear_spectrum_audit(
        record_mass_squared,
        fold_mass_squared,
        mixing_kappa,
    )
    if not receipt.strictly_stable:
        raise ValueError(
            "bilinear spectrum is not strictly stable: "
            f"det={receipt.determinant_mass_four}, "
            f"lowest eigenmass squared={receipt.eigenmass_squared_low}"
        )
    return receipt


@dataclass(frozen=True)
class WardExchangeAudit:
    record_value: float
    fold_value: float
    source_coefficient: float
    record_eom_residual: float
    fold_eom_residual: float
    record_gradient_covector: tuple[float, ...]
    fold_gradient_covector: tuple[float, ...]
    free_fold_stress_divergence: tuple[float, ...]
    record_plus_interaction_divergence: tuple[float, ...]
    total_stress_divergence: tuple[float, ...]
    expected_total_ward_covector: tuple[float, ...]
    fold_exchange_covector: tuple[float, ...]
    opposite_exchange_covector: tuple[float, ...]
    dimensionless_ward_identity_residual: float
    dimensionless_exchange_balance_residual: float
    dimensionless_total_divergence: float
    both_field_equations_on_shell: bool
    total_stress_conserved_on_shell: bool
    interaction_counted_once: bool


def ward_exchange_audit(
    *,
    record_value: float,
    fold_value: float,
    box_record: float,
    box_fold: float,
    record_gradient_covector: Iterable[float],
    fold_gradient_covector: Iterable[float],
    record_mass_squared: float = 9.0,
    fold_mass_squared: float = 4.0,
    mixing_kappa: float = 2.0,
    reference_mass_scale: float = 1.0,
) -> WardExchangeAudit:
    """한 차트 점에서 오프셸 단일 작용 워드 항등식을 평가한다.

    ``E_R=(box-m_R^2)R+kappa phi`` 와
    ``E_phi=(box-m_phi^2)phi+kappa R`` 에 대해 직접 미분하면

    ``div(T_total) = E_R grad(R) + E_phi grad(phi)``

    이다. 차원 있는 잔차는 모두 수치 비교 전에 공급된 기준 질량의 적절한 거듭제곱으로
    나눈다.
    """

    record = _finite(record_value, "record_value")
    fold = _finite(fold_value, "fold_value")
    box_r = _finite(box_record, "box_record")
    box_phi = _finite(box_fold, "box_fold")
    mass_r = _finite(record_mass_squared, "record_mass_squared")
    mass_phi = _finite(fold_mass_squared, "fold_mass_squared")
    kappa = _finite(mixing_kappa, "mixing_kappa")
    mass_scale = _positive_fold(reference_mass_scale, "reference_mass_scale")
    gradient_r = _finite_covector(
        record_gradient_covector,
        "record_gradient_covector",
    )
    gradient_phi = _finite_covector(
        fold_gradient_covector,
        "fold_gradient_covector",
    )

    source = kappa * record
    record_eom = box_r - mass_r * record + kappa * fold
    fold_eom = box_phi - mass_phi * fold + source
    free_fold = (box_phi - mass_phi * fold) * gradient_phi
    record_plus_interaction = (
        (box_r - mass_r * record) * gradient_r
        + kappa * (fold * gradient_r + record * gradient_phi)
    )
    total = free_fold + record_plus_interaction
    expected_total = record_eom * gradient_r + fold_eom * gradient_phi
    fold_exchange = -source * gradient_phi
    opposite_exchange = source * gradient_phi
    ward_identity_residual = float(np.max(np.abs(total - expected_total)))
    exchange_balance_residual = float(
        np.max(np.abs(fold_exchange + opposite_exchange))
    )
    current_scale = mass_scale**5
    eom_scale = mass_scale**3
    dimensionless_eom_residual = max(abs(record_eom), abs(fold_eom)) / eom_scale
    dimensionless_total = float(np.max(np.abs(total))) / current_scale
    on_shell = dimensionless_eom_residual <= DEFAULT_TOLERANCE
    return WardExchangeAudit(
        record_value=record,
        fold_value=fold,
        source_coefficient=source,
        record_eom_residual=record_eom,
        fold_eom_residual=fold_eom,
        record_gradient_covector=tuple(float(item) for item in gradient_r),
        fold_gradient_covector=tuple(float(item) for item in gradient_phi),
        free_fold_stress_divergence=tuple(float(item) for item in free_fold),
        record_plus_interaction_divergence=tuple(
            float(item) for item in record_plus_interaction
        ),
        total_stress_divergence=tuple(float(item) for item in total),
        expected_total_ward_covector=tuple(
            float(item) for item in expected_total
        ),
        fold_exchange_covector=tuple(float(item) for item in fold_exchange),
        opposite_exchange_covector=tuple(
            float(item) for item in opposite_exchange
        ),
        dimensionless_ward_identity_residual=(
            ward_identity_residual / current_scale
        ),
        dimensionless_exchange_balance_residual=(
            exchange_balance_residual / current_scale
        ),
        dimensionless_total_divergence=dimensionless_total,
        both_field_equations_on_shell=on_shell,
        total_stress_conserved_on_shell=(
            on_shell and dimensionless_total <= DEFAULT_TOLERANCE
        ),
        interaction_counted_once=True,
    )


def canonical_on_shell_ward_audit() -> WardExchangeAudit:
    """자명하지 않은 정확한 온셸 교환 증인을 돌려준다."""

    record = 0.5
    fold = -0.25
    record_mass, fold_mass, kappa = CANONICAL_STABLE_PARAMETERS
    return ward_exchange_audit(
        record_value=record,
        fold_value=fold,
        box_record=record_mass * record - kappa * fold,
        box_fold=fold_mass * fold - kappa * record,
        record_gradient_covector=(0.3, -0.2, 0.1, 0.0),
        fold_gradient_covector=(-0.4, 0.05, 0.0, 0.2),
        record_mass_squared=record_mass,
        fold_mass_squared=fold_mass,
        mixing_kappa=kappa,
    )


@dataclass(frozen=True)
class SchurComplementAudit:
    record_mass_squared: float
    fold_mass_squared: float
    mixing_kappa: float
    determinant_mass_four: float
    static_effective_fold_mass_squared: float
    determinant_over_record_mass_squared: float
    positive_static_effective_mass: bool
    operator_kernel: str
    zero_momentum_local_formula_only: bool
    inverse_boundary_or_state_prescription_required: bool
    retarded_inverse_automatically_selected: bool
    closed_time_path_noise_derived: bool
    local_effective_stress_automatically_derived: bool


def schur_complement_audit(
    record_mass_squared: float = 9.0,
    fold_mass_squared: float = 4.0,
    mixing_kappa: float = 2.0,
) -> SchurComplementAudit:
    """정적 질량 슈어 보수와 정확한 연산자 경고를 돌려준다."""

    record_mass = _positive_fold(record_mass_squared, "record_mass_squared")
    fold_mass = _finite(fold_mass_squared, "fold_mass_squared")
    kappa = _finite(mixing_kappa, "mixing_kappa")
    determinant = record_mass * fold_mass - kappa**2
    effective_mass = fold_mass - kappa**2 / record_mass
    determinant_ratio = determinant / record_mass
    return SchurComplementAudit(
        record_mass_squared=record_mass,
        fold_mass_squared=fold_mass,
        mixing_kappa=kappa,
        determinant_mass_four=determinant,
        static_effective_fold_mass_squared=effective_mass,
        determinant_over_record_mass_squared=determinant_ratio,
        positive_static_effective_mass=effective_mass > 0.0,
        operator_kernel="D_phi - kappa^2 D_R^{-1}",
        zero_momentum_local_formula_only=True,
        inverse_boundary_or_state_prescription_required=True,
        retarded_inverse_automatically_selected=False,
        closed_time_path_noise_derived=False,
        local_effective_stress_automatically_derived=False,
    )


@dataclass(frozen=True)
class SourceAccountingAudit:
    mode: str
    retained_record_and_fold_fields: bool
    integrated_out_influence_kernel: bool
    original_bilinear_interaction_retained: bool
    mutually_exclusive_representations: bool
    probability_rebooked_as_energy: bool
    source_stress_counted_twice: bool


def source_accounting_audit(mode: str) -> SourceAccountingAudit:
    """보존 장 장부와 적분 제거 장부 중 정확히 하나만 허용한다."""

    if mode == "retained_fields":
        retained = True
        influence = False
        interaction = True
    elif mode == "integrated_out_influence":
        retained = False
        influence = True
        interaction = False
    else:
        raise ValueError(f"unknown source accounting mode: {mode}")
    return SourceAccountingAudit(
        mode=mode,
        retained_record_and_fold_fields=retained,
        integrated_out_influence_kernel=influence,
        original_bilinear_interaction_retained=interaction,
        mutually_exclusive_representations=(retained != influence),
        probability_rebooked_as_energy=False,
        source_stress_counted_twice=False,
    )


@dataclass(frozen=True)
class BasisObstructionAudit:
    witness_mass_squared_matrix: tuple[tuple[float, ...], ...]
    eigenmass_squared_set: tuple[float, float]
    absolute_rotation_angle_degrees: float
    rotated_off_diagonal_residual: float
    kinetic_rotation_residual: float
    hypothetical_pointer_vector_original_basis: tuple[float, float]
    hypothetical_pointer_vector_eigenbasis: tuple[float, float]
    hypothetical_pointer_is_extra_input: bool
    eigenmass_squared_set_basis_invariant: bool
    record_and_fold_labels_basis_invariant: bool
    bilinear_mixing_selects_pointer_basis: bool
    bilinear_mixing_derives_observed_outcome: bool
    bilinear_mixing_derives_dark_source: bool


def basis_obstruction_audit() -> BasisObstructionAudit:
    """포인터 물리를 공급하지 않고 45도 증인을 대각화한다."""

    receipt = bilinear_spectrum_audit(5.0, 5.0, 1.0)
    rotation = np.asarray(receipt.rotation_matrix, dtype=np.float64)
    pointer_original = np.asarray((1.0, 0.0), dtype=np.float64)
    pointer_eigenbasis = rotation.T @ pointer_original
    return BasisObstructionAudit(
        witness_mass_squared_matrix=receipt.mass_squared_matrix,
        eigenmass_squared_set=tuple(
            sorted(
                (
                    receipt.eigenmass_squared_low,
                    receipt.eigenmass_squared_high,
                )
            )
        ),
        absolute_rotation_angle_degrees=abs(receipt.rotation_angle_degrees),
        rotated_off_diagonal_residual=receipt.rotated_off_diagonal_residual,
        kinetic_rotation_residual=receipt.kinetic_rotation_residual,
        hypothetical_pointer_vector_original_basis=(1.0, 0.0),
        hypothetical_pointer_vector_eigenbasis=tuple(
            float(item) for item in pointer_eigenbasis
        ),
        hypothetical_pointer_is_extra_input=True,
        eigenmass_squared_set_basis_invariant=True,
        record_and_fold_labels_basis_invariant=False,
        bilinear_mixing_selects_pointer_basis=False,
        bilinear_mixing_derives_observed_outcome=False,
        bilinear_mixing_derives_dark_source=False,
    )


@dataclass(frozen=True)
class RecordFoldBilinearCertificate:
    status: str
    dimensions: DimensionAudit
    stable_witness: BilinearSpectrumAudit
    tachyon_counterexample: BilinearSpectrumAudit
    boundary_counterexample: BilinearSpectrumAudit
    ward_witness: WardExchangeAudit
    static_schur_witness: SchurComplementAudit
    retained_accounting: SourceAccountingAudit
    integrated_out_accounting: SourceAccountingAudit
    basis_obstruction: BasisObstructionAudit
    source_sign_convention: str
    one_total_action_accounting_admitted: bool
    nonselected_quantum_to_record_map_derived: bool
    pointer_selection_and_durable_record_derived: bool
    probability_deformation_defined: bool
    cptp_and_normalization_derived: bool
    classical_principal_symbol_uses_metric_cone: bool
    qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    zero_stress_qm_gr_limit_derived: bool
    gravitational_solution_derived: bool
    fixed_parameter_manifest_established: bool
    independent_holdout_prediction_derived: bool
    two_residual_classes_reduced: bool
    complexity_penalized_improvement_established: bool


def record_fold_certificate() -> RecordFoldBilinearCertificate:
    """완전한 E36 유한 승인/불가 증명서를 만든다."""

    stable = require_stable_spectrum(*CANONICAL_STABLE_PARAMETERS)
    tachyon = bilinear_spectrum_audit(*CANONICAL_TACHYON_PARAMETERS)
    boundary = bilinear_spectrum_audit(*CANONICAL_BOUNDARY_PARAMETERS)
    return RecordFoldBilinearCertificate(
        status="CONDITIONAL_CLASSICAL_TWO_FIELD_ADMISSION",
        dimensions=dimension_audit(),
        stable_witness=stable,
        tachyon_counterexample=tachyon,
        boundary_counterexample=boundary,
        ward_witness=canonical_on_shell_ward_audit(),
        static_schur_witness=schur_complement_audit(
            *CANONICAL_STABLE_PARAMETERS
        ),
        retained_accounting=source_accounting_audit("retained_fields"),
        integrated_out_accounting=source_accounting_audit(
            "integrated_out_influence"
        ),
        basis_obstruction=basis_obstruction_audit(),
        source_sign_convention=(
            "+kappa R_rec phi in L gives "
            "(box-m_phi^2)phi=-J_ns, J_ns=kappa R_rec"
        ),
        one_total_action_accounting_admitted=True,
        nonselected_quantum_to_record_map_derived=False,
        pointer_selection_and_durable_record_derived=False,
        probability_deformation_defined=False,
        cptp_and_normalization_derived=False,
        classical_principal_symbol_uses_metric_cone=True,
        qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        zero_stress_qm_gr_limit_derived=False,
        gravitational_solution_derived=False,
        fixed_parameter_manifest_established=False,
        independent_holdout_prediction_derived=False,
        two_residual_classes_reduced=False,
        complexity_penalized_improvement_established=False,
    )


def record_fold_run() -> dict[str, object]:
    """JSON 직렬화 가능한 증명서 페이로드를 돌려준다."""

    return asdict(record_fold_certificate())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation",
    )
    return parser


def record_fold_main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(json.dumps(record_fold_run(), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
