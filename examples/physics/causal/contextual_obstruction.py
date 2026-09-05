"""CHSH 상자에 대한 국소 공통 씨앗(seed) 장애와 전역 단면(global section) 장애의 유한 인증서를 모은다.

이 모듈은 두 부분으로 이루어진다.

첫째, 설정 독립 국소 공통 씨앗에 대한 유한 CHSH 장애다. 세 사상을 분리해 둔다.

* 고정 설정 양자 사영 기구(projective instrument)는 단일항(singlet) CHSH 확률 상자를 만들며,
  선언된 설정 쌍마다 CPTP다.
* 공급된 역누적분포(inverse-CDF) 씨앗은 거친 라벨 ``a``에서 미세 좌표 ``(a, r)``로 들어올릴
  수 있다. 이 들어올림은 가중 측도 전단사이지만, 통상의 구간·유한 쌍대곱(coproduct) 위상에서는
  양의 확률 결과가 둘 이상이면 위상동형(homeomorphism)이 아니다.
* 설정 독립 공통 과거 씨앗과 인수분해된 국소 반응은 16개 결정론적 전략의 볼록 혼합이며
  CHSH <= 2를 따른다. 따라서 CHSH = 2 sqrt(2)인 유한 단일항 상자를 재현할 수 없다.

벨(Bell) 장애는 의도적으로 좁다. 전역 또는 맥락적 미세 상태 전단사를 배제하지 않으며, 이
유한 확률 상자의 조작적 무신호(no-signalling)는 상대론적 QFT 미시인과성(microcausality)의
유도가 아니다. 씨앗·결과·확률·CHSH 점수 어디에도 에너지나 시공간 척도를 배정하지 않는다.

둘째, PR 방향 단일항 CHSH 상자에 대한 유한 전역 장부(ledger) 감사다. 집합 수준 장부 문제와
확률 문제를 분리한다.

고정된 측정 맥락 ``(x, y)``마다 결정론적 전역 원자 ``(A0, A1, B0, B1)``는

``((A_x, B_y), (A_{1-x}, B_{1-y}))``

로 전단사적으로 재배열할 수 있다. 첫 쌍은 보이는 판독이고 둘째 쌍은 숨은 장부다. 숨은 쌍을
버리면 4대1 사영이고, 유지하면 전단사(두 유한 공간이 이산 위상을 가지면 위상동형)다.

그 전단사는 16개 원자 위에 설정 독립 *양의 측도*를 주지 않는다. 등방 상자

``P_eta(a,b|x,y) = (1 + a*b*eta*c_xy)/4``

(``c=(-1,-1,-1,+1)``)에 대해 대칭 부호 확장

``q_eta(lambda) = (1 + eta*F_lambda)/16``

은 모든 맥락을 재구성하며, ``F_lambda``는 방향 결정론적 CHSH 점수라서 ``+2`` 또는 ``-2``다.
양자 값 ``eta=1/sqrt(2)``에서 그런 확장은 l1 노름이 최소 ``sqrt(2)``이고 음의 질량이 최소
``(sqrt(2)-1)/2``다. 대칭 확장은 그 하한을 포화한다. 이 대표를 정규화된 절댓값으로 바꾸면
표적 상관의 절반을 가진 다른 양의 국소 상자가 나온다.

이 모듈의 부호 가중치는 선형 표현 계수다. 관측 확률, 음의 빈도, 에너지, 응력, 계량 부피,
물리적 숨은 경로 법칙이 아니다. 결과는 선언된 2x2 이진 시나리오에 한정되며 일반 계량이나
중력 불가 정리를 이루지 않는다.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import product
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.record.instrument_record_kernel import (
    build_seed_partition,
    select_partition_cell,
)


DEFAULT_TOLERANCE = 1.0e-12
OUTCOMES = (-1, 1)
SETTINGS = (0, 1)
CHSH_PATTERN = np.array([[-1.0, -1.0], [-1.0, 1.0]])
QUANTUM_ETA = 1.0 / math.sqrt(2.0)


def _positive_tolerance(value: float) -> float:
    """유한하고 양수인 허용오차만 통과시킨다."""

    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


def _isotropic_parameter(value: float) -> float:
    """[0, 1] 안의 유한한 등방 매개변수만 통과시킨다."""

    parameter = float(value)
    if not math.isfinite(parameter) or not 0.0 <= parameter <= 1.0:
        raise ValueError("eta must be finite and lie in [0, 1]")
    return parameter


def isotropic_chsh_box(eta: float) -> np.ndarray:
    """PR 방향 등방 상자 ``P_eta``를 돌려준다.

    ``P_eta(a,b|x,y) = (1 + a*b*eta*c_xy)/4``, ``c = (-1,-1,-1,+1)``이다. 따라서 절대 CHSH
    값은 ``4*eta``이며, ``eta=1``은 PR 상자이지 양자 단일항 상자가 아니다.
    """

    visibility = _isotropic_parameter(eta)
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for x in SETTINGS:
        for y in SETTINGS:
            for a_index, a in enumerate(OUTCOMES):
                for b_index, b in enumerate(OUTCOMES):
                    probabilities[x, y, a_index, b_index] = 0.25 * (
                        1.0 + a * b * visibility * CHSH_PATTERN[x, y]
                    )
    return probabilities


def singlet_density() -> np.ndarray:
    """계산 기저의 ``(|01>-|10>)(<01|-<10|)/2``를 돌려준다."""

    vector = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.complex128) / math.sqrt(2.0)
    return np.outer(vector, vector.conj())


def chsh_observables() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """``A=(Z,X)``와 ``B=((Z+X)/sqrt(2),(Z-X)/sqrt(2))``를 돌려준다."""

    x_pauli = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_pauli = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    alice = (z_pauli, x_pauli)
    bob = (
        (z_pauli + x_pauli) / math.sqrt(2.0),
        (z_pauli - x_pauli) / math.sqrt(2.0),
    )
    return alice, bob


def _projector(observable: np.ndarray, outcome: int) -> np.ndarray:
    """관측량의 결과 ``outcome`` 고유공간 사영자를 돌려준다."""

    if outcome not in OUTCOMES:
        raise ValueError("outcome must be -1 or +1")
    return 0.5 * (np.eye(2, dtype=np.complex128) + outcome * observable)


@dataclass(frozen=True)
class ProjectiveInstrumentAudit:
    """네 고정 설정 결합 사영 기구의 감사 결과다."""

    probabilities: np.ndarray
    maximum_projector_residual: float
    maximum_completeness_residual: float
    minimum_choi_eigenvalue: float
    maximum_posterior_trace_residual: float


def quantum_projective_instrument_audit() -> ProjectiveInstrumentAudit:
    """네 고정 설정 결합 사영 기구를 모두 감사한다."""

    density = singlet_density()
    alice, bob = chsh_observables()
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    maximum_projector_residual = 0.0
    maximum_completeness_residual = 0.0
    minimum_choi_eigenvalue = math.inf
    maximum_posterior_trace_residual = 0.0
    joint_identity = np.eye(4, dtype=np.complex128)

    for x in SETTINGS:
        for y in SETTINGS:
            operators: list[np.ndarray] = []
            completeness = np.zeros((4, 4), dtype=np.complex128)
            for a_index, a in enumerate(OUTCOMES):
                for b_index, b in enumerate(OUTCOMES):
                    operator = np.kron(
                        _projector(alice[x], a),
                        _projector(bob[y], b),
                    )
                    operators.append(operator)
                    maximum_projector_residual = max(
                        maximum_projector_residual,
                        float(np.linalg.norm(operator @ operator - operator, ord=2)),
                        float(np.linalg.norm(operator - operator.conj().T, ord=2)),
                    )
                    completeness += operator.conj().T @ operator
                    operation = operator @ density @ operator.conj().T
                    probability = float(np.trace(operation).real)
                    probabilities[x, y, a_index, b_index] = probability
                    if probability > 0.0:
                        posterior = operation / probability
                        maximum_posterior_trace_residual = max(
                            maximum_posterior_trace_residual,
                            abs(float(np.trace(posterior).real) - 1.0),
                            abs(float(np.trace(posterior).imag)),
                        )

            maximum_completeness_residual = max(
                maximum_completeness_residual,
                float(np.linalg.norm(completeness - joint_identity, ord=2)),
            )
            choi = np.zeros((16, 16), dtype=np.complex128)
            for operator in operators:
                vector = operator.reshape(-1, order="F")
                choi += np.outer(vector, vector.conj())
            minimum_choi_eigenvalue = min(
                minimum_choi_eigenvalue,
                float(np.linalg.eigvalsh(choi).min()),
            )

    return ProjectiveInstrumentAudit(
        probabilities=probabilities,
        maximum_projector_residual=maximum_projector_residual,
        maximum_completeness_residual=maximum_completeness_residual,
        minimum_choi_eigenvalue=minimum_choi_eigenvalue,
        maximum_posterior_trace_residual=maximum_posterior_trace_residual,
    )


def box_correlations(probabilities: np.ndarray) -> tuple[float, float, float, float]:
    """엄격한 모양 검증 뒤 ``(E00,E01,E10,E11)``을 돌려준다."""

    box = np.asarray(probabilities, dtype=np.float64)
    if box.shape != (2, 2, 2, 2) or not np.isfinite(box).all():
        raise ValueError("probability box must be finite with shape (2, 2, 2, 2)")
    correlations: list[float] = []
    for x in SETTINGS:
        for y in SETTINGS:
            correlations.append(
                math.fsum(
                    a * b * float(box[x, y, a_index, b_index])
                    for a_index, a in enumerate(OUTCOMES)
                    for b_index, b in enumerate(OUTCOMES)
                )
            )
    return tuple(correlations)  # type: ignore[return-value]


def chsh_scores(probabilities: np.ndarray) -> tuple[float, float]:
    """방향 면(facet) 점수와 통상의 절대 CHSH 점수를 돌려준다."""

    correlations = np.asarray(box_correlations(probabilities)).reshape(2, 2)
    facet_score = float(np.sum(CHSH_PATTERN * correlations))
    standard_expression = float(
        correlations[0, 0]
        + correlations[0, 1]
        + correlations[1, 0]
        - correlations[1, 1]
    )
    return facet_score, abs(standard_expression)


@dataclass(frozen=True)
class BoxAudit:
    """확률 상자의 비음성·정규화·무신호 감사 결과다."""

    minimum_probability: float
    maximum_normalization_residual: float
    maximum_no_signalling_residual: float
    maximum_unbiased_marginal_residual: float


def audit_probability_box(probabilities: np.ndarray) -> BoxAudit:
    """비음성, 맥락별 정규화, 무신호를 감사한다."""

    box = np.asarray(probabilities, dtype=np.float64)
    if box.shape != (2, 2, 2, 2) or not np.isfinite(box).all():
        raise ValueError("probability box must be finite with shape (2, 2, 2, 2)")
    normalization_residual = max(
        abs(float(np.sum(box[x, y])) - 1.0) for x in SETTINGS for y in SETTINGS
    )
    no_signalling_residual = 0.0
    unbiased_residual = 0.0
    for x in SETTINGS:
        for a_index in range(2):
            marginals = tuple(float(np.sum(box[x, y, a_index, :])) for y in SETTINGS)
            no_signalling_residual = max(
                no_signalling_residual, abs(marginals[0] - marginals[1])
            )
            unbiased_residual = max(
                unbiased_residual, *(abs(value - 0.5) for value in marginals)
            )
    for y in SETTINGS:
        for b_index in range(2):
            marginals = tuple(float(np.sum(box[x, y, :, b_index])) for x in SETTINGS)
            no_signalling_residual = max(
                no_signalling_residual, abs(marginals[0] - marginals[1])
            )
            unbiased_residual = max(
                unbiased_residual, *(abs(value - 0.5) for value in marginals)
            )
    return BoxAudit(
        minimum_probability=float(box.min()),
        maximum_normalization_residual=normalization_residual,
        maximum_no_signalling_residual=no_signalling_residual,
        maximum_unbiased_marginal_residual=unbiased_residual,
    )


def deterministic_local_strategies() -> tuple[tuple[int, int, int, int], ...]:
    """모든 ``(A0,A1,B0,B1)`` 결정론적 반응 배정을 돌려준다."""

    return tuple(product(OUTCOMES, repeat=4))


def deterministic_facet_score(strategy: Sequence[int]) -> int:
    """국소 전략 하나의 PR 방향 CHSH 면 점수를 돌려준다."""

    values = tuple(strategy)
    if len(values) != 4 or any(value not in OUTCOMES for value in values):
        raise ValueError("strategy must contain four outcomes in {-1, +1}")
    a0, a1, b0, b1 = values
    correlations = np.array(
        [[a0 * b0, a0 * b1], [a1 * b0, a1 * b1]], dtype=np.float64
    )
    return int(np.sum(CHSH_PATTERN * correlations))


def local_boundary_strategies() -> tuple[tuple[int, int, int, int], ...]:
    """선택한 ``S=2`` 면 위의 결정론적 꼭짓점 여덟 개를 돌려준다."""

    return tuple(
        strategy
        for strategy in deterministic_local_strategies()
        if deterministic_facet_score(strategy) == 2
    )


def deterministic_mixture_box(
    strategies: Sequence[Sequence[int]],
) -> np.ndarray:
    """선언된 결정론적 전략들이 만드는 균등 혼합 상자를 돌려준다."""

    declared = tuple(tuple(strategy) for strategy in strategies)
    if not declared:
        raise ValueError("at least one deterministic strategy is required")
    for strategy in declared:
        deterministic_facet_score(strategy)
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    weight = 1.0 / len(declared)
    for a0, a1, b0, b1 in declared:
        alice = (a0, a1)
        bob = (b0, b1)
        for x in SETTINGS:
            for y in SETTINGS:
                a_index = OUTCOMES.index(alice[x])
                b_index = OUTCOMES.index(bob[y])
                probabilities[x, y, a_index, b_index] += weight
    return probabilities


@dataclass(frozen=True)
class FineSeedCoordinate:
    """거친 결과 인덱스와 올(fibre) 안 잔여 좌표로 이루어진 미세 씨앗 좌표다."""

    outcome_index: int
    residual_coordinate: float
    interval: tuple[float, float]
    interval_probability: float


def lift_seed_coordinate(
    probabilities: Sequence[float],
    seed: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FineSeedCoordinate:
    """거친 역누적분포 출력을 ``(outcome, residual coordinate)``로 들어올린다."""

    partition = build_seed_partition(probabilities, tolerance=tolerance)
    outcome_index = select_partition_cell(partition, seed)
    start, end = partition.intervals[outcome_index]
    width = end - start
    if width <= 0.0:
        raise ArithmeticError("selected probability fibre must have positive width")
    residual = (float(seed) - start) / width
    if residual >= 1.0 and residual <= 1.0 + 10.0 * tolerance:
        residual = math.nextafter(1.0, 0.0)
    if not 0.0 <= residual < 1.0:
        raise ArithmeticError("residual coordinate must lie in [0, 1)")
    return FineSeedCoordinate(
        outcome_index=outcome_index,
        residual_coordinate=residual,
        interval=(start, end),
        interval_probability=partition.cell_probabilities[outcome_index],
    )


def invert_seed_coordinate(
    probabilities: Sequence[float],
    outcome_index: int,
    residual_coordinate: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """양의 올 위에서 선언된 미세 씨앗 좌표를 역변환한다."""

    partition = build_seed_partition(probabilities, tolerance=tolerance)
    if (
        isinstance(outcome_index, bool)
        or not isinstance(outcome_index, int)
        or not 0 <= outcome_index < len(partition.intervals)
    ):
        raise ValueError("outcome_index must select a declared probability fibre")
    residual = float(residual_coordinate)
    if not math.isfinite(residual) or not 0.0 <= residual < 1.0:
        raise ValueError("residual_coordinate must be finite and lie in [0, 1)")
    start, end = partition.intervals[outcome_index]
    width = end - start
    if width <= 0.0 or partition.cell_probabilities[outcome_index] <= 0.0:
        raise ValueError("zero-probability fibres are empty")
    seed = start + width * residual
    if seed >= end:
        seed = math.nextafter(end, start)
    if select_partition_cell(partition, seed) != outcome_index:
        raise ArithmeticError("fine coordinate inverse left its declared fibre")
    return seed


def usual_coproduct_seed_lift_is_homeomorphism(
    probabilities: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """통상의 선언된 위상에서 연결성 판정을 돌려준다.

    양의 올이 하나면 아핀 좌표 사상은 위상동형이다. 양의 올이 둘 이상이면 ``[0,1)``은
    연결이지만 올들의 유한 쌍대곱은 비연결이므로 전단사는 위상동형이 아니다. 옮겨 온
    위상(transported topology)을 쓰면 동어반복이 되므로 여기서는 쓰지 않는다.
    """

    partition = build_seed_partition(probabilities, tolerance=tolerance)
    positive_fibres = sum(value > 0.0 for value in partition.cell_probabilities)
    return positive_fibres == 1


@dataclass(frozen=True)
class ChshLocalSeedCertificate:
    """E28 유한 CHSH·미세 씨앗 경계 인증서다."""

    quantum_correlations: tuple[float, float, float, float]
    quantum_oriented_facet_score: float
    quantum_absolute_chsh_score: float
    quantum_formula_residual: float
    quantum_minimum_probability: float
    quantum_normalization_residual: float
    quantum_no_signalling_residual: float
    quantum_unbiased_marginal_residual: float
    maximum_projector_residual: float
    maximum_instrument_completeness_residual: float
    minimum_instrument_choi_eigenvalue: float
    maximum_posterior_trace_residual: float
    deterministic_strategy_count: int
    deterministic_facet_scores: tuple[int, ...]
    maximum_deterministic_absolute_chsh_score: float
    local_boundary_strategy_count: int
    local_boundary_strategies: tuple[tuple[int, int, int, int], ...]
    local_boundary_probability_residual: float
    local_boundary_no_signalling_residual: float
    local_boundary_unbiased_marginal_residual: float
    pr_minimum_probability: float
    pr_normalization_residual: float
    pr_no_signalling_residual: float
    ns_local_fraction: float
    ns_nonlocal_fraction: float
    local_fraction_chsh_upper_bound: float
    local_fraction_upper_bound_residual: float
    local_pr_decomposition_residual: float
    seed_context_probabilities: tuple[float, ...]
    seed_positive_fibre_count: int
    maximum_seed_lift_round_trip_residual: float
    maximum_seed_fibre_measure_residual: float
    usual_coproduct_seed_lift_homeomorphism: bool
    coarse_seed_readout_many_to_one: bool
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def chsh_certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> ChshLocalSeedCertificate:
    """E28 유한 CHSH·미세 씨앗 경계 인증서를 만든다."""

    tol = _positive_tolerance(tolerance)
    instrument = quantum_projective_instrument_audit()
    quantum_box = instrument.probabilities
    expected_quantum_box = isotropic_chsh_box(1.0 / math.sqrt(2.0))
    quantum_formula_residual = float(
        np.max(np.abs(quantum_box - expected_quantum_box))
    )
    quantum_audit = audit_probability_box(quantum_box)
    quantum_correlations = box_correlations(quantum_box)
    quantum_facet_score, quantum_chsh = chsh_scores(quantum_box)

    strategies = deterministic_local_strategies()
    facet_scores = tuple(deterministic_facet_score(strategy) for strategy in strategies)
    maximum_local_chsh = float(max(abs(value) for value in facet_scores))
    boundary_strategies = local_boundary_strategies()
    local_boundary_box = deterministic_mixture_box(boundary_strategies)
    expected_local_boundary = isotropic_chsh_box(0.5)
    local_boundary_probability_residual = float(
        np.max(np.abs(local_boundary_box - expected_local_boundary))
    )
    local_boundary_audit = audit_probability_box(local_boundary_box)

    pr_box = isotropic_chsh_box(1.0)
    pr_audit = audit_probability_box(pr_box)
    local_fraction = 2.0 - math.sqrt(2.0)
    nonlocal_fraction = math.sqrt(2.0) - 1.0
    reconstructed_quantum = (
        local_fraction * local_boundary_box + nonlocal_fraction * pr_box
    )
    decomposition_residual = float(
        np.max(np.abs(reconstructed_quantum - quantum_box))
    )
    local_fraction_upper_bound = (4.0 - quantum_chsh) / 2.0
    local_fraction_upper_bound_residual = abs(
        local_fraction_upper_bound - local_fraction
    )

    seed_probabilities = tuple(float(value) for value in quantum_box[0, 0].reshape(-1))
    seed_partition = build_seed_partition(seed_probabilities, tolerance=tol)
    seed_probes = tuple(
        start + fraction * (end - start)
        for (start, end), probability in zip(
            seed_partition.intervals, seed_partition.cell_probabilities
        )
        if probability > 0.0
        for fraction in (0.25, 0.75)
    )
    seed_coordinates = tuple(
        lift_seed_coordinate(seed_probabilities, seed, tolerance=tol)
        for seed in seed_probes
    )
    seed_round_trip_residual = max(
        abs(
            invert_seed_coordinate(
                seed_probabilities,
                coordinate.outcome_index,
                coordinate.residual_coordinate,
                tolerance=tol,
            )
            - seed
        )
        for seed, coordinate in zip(seed_probes, seed_coordinates)
    )
    seed_fibre_measure_residual = max(
        abs((end - start) - probability)
        for (start, end), probability in zip(
            seed_partition.intervals, seed_partition.cell_probabilities
        )
    )
    positive_fibre_count = sum(
        probability > 0.0 for probability in seed_partition.cell_probabilities
    )
    homeomorphism = usual_coproduct_seed_lift_is_homeomorphism(
        seed_probabilities, tolerance=tol
    )
    coarse_many_to_one = any(
        left.outcome_index == right.outcome_index
        and left.residual_coordinate != right.residual_coordinate
        for left, right in zip(seed_coordinates[::2], seed_coordinates[1::2])
    )

    numerical_limit = 50.0 * tol
    cptp_certified = (
        instrument.maximum_projector_residual <= numerical_limit
        and instrument.maximum_completeness_residual <= numerical_limit
        and instrument.minimum_choi_eigenvalue >= -numerical_limit
        and instrument.maximum_posterior_trace_residual <= numerical_limit
    )
    quantum_box_certified = (
        quantum_formula_residual <= numerical_limit
        and quantum_audit.minimum_probability >= -numerical_limit
        and quantum_audit.maximum_normalization_residual <= numerical_limit
        and quantum_audit.maximum_no_signalling_residual <= numerical_limit
        and quantum_audit.maximum_unbiased_marginal_residual <= numerical_limit
        and abs(quantum_chsh - 2.0 * math.sqrt(2.0)) <= numerical_limit
    )
    local_no_go_certified = (
        len(strategies) == 16
        and set(facet_scores) == {-2, 2}
        and maximum_local_chsh <= 2.0 + numerical_limit
        and quantum_chsh > 2.0 + numerical_limit
    )
    local_fraction_certified = (
        len(boundary_strategies) == 8
        and local_boundary_probability_residual <= numerical_limit
        and local_boundary_audit.maximum_no_signalling_residual <= numerical_limit
        and pr_audit.minimum_probability >= -numerical_limit
        and pr_audit.maximum_no_signalling_residual <= numerical_limit
        and decomposition_residual <= numerical_limit
        and local_fraction_upper_bound_residual <= numerical_limit
    )
    fine_seed_bijection_certified = (
        positive_fibre_count == 4
        and seed_round_trip_residual <= numerical_limit
        and seed_fibre_measure_residual <= numerical_limit
        and coarse_many_to_one
    )

    return ChshLocalSeedCertificate(
        quantum_correlations=quantum_correlations,
        quantum_oriented_facet_score=quantum_facet_score,
        quantum_absolute_chsh_score=quantum_chsh,
        quantum_formula_residual=quantum_formula_residual,
        quantum_minimum_probability=quantum_audit.minimum_probability,
        quantum_normalization_residual=quantum_audit.maximum_normalization_residual,
        quantum_no_signalling_residual=quantum_audit.maximum_no_signalling_residual,
        quantum_unbiased_marginal_residual=(
            quantum_audit.maximum_unbiased_marginal_residual
        ),
        maximum_projector_residual=instrument.maximum_projector_residual,
        maximum_instrument_completeness_residual=(
            instrument.maximum_completeness_residual
        ),
        minimum_instrument_choi_eigenvalue=instrument.minimum_choi_eigenvalue,
        maximum_posterior_trace_residual=(
            instrument.maximum_posterior_trace_residual
        ),
        deterministic_strategy_count=len(strategies),
        deterministic_facet_scores=facet_scores,
        maximum_deterministic_absolute_chsh_score=maximum_local_chsh,
        local_boundary_strategy_count=len(boundary_strategies),
        local_boundary_strategies=boundary_strategies,
        local_boundary_probability_residual=local_boundary_probability_residual,
        local_boundary_no_signalling_residual=(
            local_boundary_audit.maximum_no_signalling_residual
        ),
        local_boundary_unbiased_marginal_residual=(
            local_boundary_audit.maximum_unbiased_marginal_residual
        ),
        pr_minimum_probability=pr_audit.minimum_probability,
        pr_normalization_residual=pr_audit.maximum_normalization_residual,
        pr_no_signalling_residual=pr_audit.maximum_no_signalling_residual,
        ns_local_fraction=local_fraction,
        ns_nonlocal_fraction=nonlocal_fraction,
        local_fraction_chsh_upper_bound=local_fraction_upper_bound,
        local_fraction_upper_bound_residual=local_fraction_upper_bound_residual,
        local_pr_decomposition_residual=decomposition_residual,
        seed_context_probabilities=seed_probabilities,
        seed_positive_fibre_count=positive_fibre_count,
        maximum_seed_lift_round_trip_residual=seed_round_trip_residual,
        maximum_seed_fibre_measure_residual=seed_fibre_measure_residual,
        usual_coproduct_seed_lift_homeomorphism=homeomorphism,
        coarse_seed_readout_many_to_one=coarse_many_to_one,
        dimensions={
            "probabilities_and_marginals_dimensionless": True,
            "eta_and_local_fraction_dimensionless": True,
            "outcomes_settings_correlations_and_chsh_dimensionless": True,
            "seed_and_residual_coordinate_dimensionless": True,
            "no_mass_energy_length_or_time_scale_introduced": True,
        },
        accounting={
            "each_setting_probability_box_normalized_once": True,
            "local_and_pr_mixture_weights_sum_to_one": math.isclose(
                local_fraction + nonlocal_fraction, 1.0, abs_tol=numerical_limit
            ),
            "weighted_fibre_measure_uses_born_probability_once": True,
            "coarse_and_fine_seed_probabilities_not_double_counted": True,
            "unselected_probabilities_not_added_as_energy_or_stress": True,
            "seed_or_hidden_coordinate_carries_energy": False,
        },
        boundaries={
            "isotropic_parameter_one_is_pr_box_not_singlet": True,
            "local_fraction_scenario_is_fixed_two_setting_binary_outcome": True,
            "local_fraction_remainder_class_is_nonsignalling": True,
            "bell_assumes_setting_independent_seed_distribution": True,
            "bell_assumes_factorized_local_response": True,
            "global_or_contextual_fine_bijection_not_excluded": True,
            "conditional_joint_inverse_cdf_is_not_local_factorization": True,
            "fine_seed_residual_is_not_a_derived_physical_hidden_path": True,
            "zero_probability_fibres_are_empty": True,
            "usual_interval_and_finite_coproduct_topologies_declared": True,
            "transported_topology_not_used_as_physical_evidence": True,
            "finite_discrete_observation_label_space_is_zero_dimensional": True,
            "zero_dimensional_readout_is_not_spacetime_dimension": True,
            "measure_bijection_does_not_earn_metric_pullback": True,
            "operational_no_signalling_is_not_qft_microcausality": True,
            "timelike_domino_limited_to_future_cone_pointer_propagation": True,
            "timelike_domino_not_spacelike_bell_correlation_generator": True,
        },
        alternatives={
            "global_contextual_joint_rule_route_open": True,
            "ontic_nonlocal_operational_no_signalling_route_open": True,
            "measurement_dependent_or_retrocausal_route_open": True,
            "timelike_durable_pointer_route_open": True,
            "boundary_glued_representation_invariant_topology_route_open": True,
        },
        status={
            "fixed_setting_quantum_projective_instruments_cptp": cptp_certified,
            "finite_singlet_chsh_box_certified": quantum_box_certified,
            "finite_box_operational_no_signalling_certified": (
                quantum_audit.maximum_no_signalling_residual <= numerical_limit
            ),
            "setting_independent_local_factorization_excluded_for_box": (
                local_no_go_certified
            ),
            "nonsignalling_remainder_local_fraction_certified": (
                local_fraction_certified
            ),
            "fine_seed_weighted_measure_bijection_formula_certified": (
                fine_seed_bijection_certified
            ),
            "usual_topology_homeomorphism_counterexample": (
                fine_seed_bijection_certified and not homeomorphism
            ),
            "usual_topology_homeomorphism_derived": False,
            "physical_seed_law_derived": False,
            "objective_single_outcome_selection_derived": False,
            "durable_physical_pointer_derived": False,
            "relativistic_qft_microcausality_derived": False,
            "full_lightcone_no_controllable_influence_gate_complete": False,
            "spacetime_topology_metric_or_curvature_derived": False,
            "fold_stress_or_gravity_derived": False,
            "mass_dependent_probability_deformation_derived": False,
            "independent_holdout_complete": False,
            "success_gates_1_to_8_complete": False,
        },
    )


def chsh_run() -> dict[str, object]:
    """JSON 직렬화 가능한 E28 인증서를 돌려준다."""

    return asdict(chsh_certificate())


def chsh_main() -> None:
    """E28 인증서를 명령줄에서 출력한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(chsh_run(), indent=2, sort_keys=True))


def _visibility(value: float) -> float:
    """[0, 1] 안의 유한한 가시도(visibility)만 통과시킨다."""

    eta = float(value)
    if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be finite and lie in [0, 1]")
    return eta


def _context_value(value: int, *, name: str) -> int:
    """설정 값이 0 또는 1인지 검사한다."""

    if isinstance(value, bool) or value not in SETTINGS:
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _outcome_pair(values: Sequence[int], *, name: str) -> tuple[int, int]:
    """{-1, +1} 결과 두 개로 이루어진 쌍인지 검사한다."""

    pair = tuple(values)
    if len(pair) != 2 or any(value not in OUTCOMES for value in pair):
        raise ValueError(f"{name} must contain two outcomes in {{-1, +1}}")
    return pair  # type: ignore[return-value]


def _global_assignment(values: Sequence[int]) -> tuple[int, int, int, int]:
    """{-1, +1} 결과 네 개로 이루어진 전역 배정인지 검사한다."""

    assignment = tuple(values)
    if len(assignment) != 4 or any(value not in OUTCOMES for value in assignment):
        raise ValueError("assignment must contain four outcomes in {-1, +1}")
    return assignment  # type: ignore[return-value]


def global_assignments() -> tuple[tuple[int, int, int, int], ...]:
    """반사실적(counterfactual) 원자 ``(A0,A1,B0,B1)`` 16개를 돌려준다."""

    return deterministic_local_strategies()


@dataclass(frozen=True)
class ContextLedgerCoordinate:
    """고정 맥락에 대한 완전한 유한 장부 좌표 하나다."""

    context: tuple[int, int]
    visible_outcomes: tuple[int, int]
    hidden_outcomes: tuple[int, int]


def lift_context_ledger(
    assignment: Sequence[int], x: int, y: int
) -> ContextLedgerCoordinate:
    """전역 원자를 보이는 결과 쌍과 숨은 결과 쌍으로 재배열한다."""

    a0, a1, b0, b1 = _global_assignment(assignment)
    x_value = _context_value(x, name="x")
    y_value = _context_value(y, name="y")
    alice = (a0, a1)
    bob = (b0, b1)
    return ContextLedgerCoordinate(
        context=(x_value, y_value),
        visible_outcomes=(alice[x_value], bob[y_value]),
        hidden_outcomes=(alice[1 - x_value], bob[1 - y_value]),
    )


def invert_context_ledger(
    coordinate: ContextLedgerCoordinate,
) -> tuple[int, int, int, int]:
    """선언된 완전 장부 좌표를 정확히 역변환한다."""

    if not isinstance(coordinate, ContextLedgerCoordinate):
        raise TypeError("coordinate must be a ContextLedgerCoordinate")
    x = _context_value(coordinate.context[0], name="x")
    y = _context_value(coordinate.context[1], name="y")
    visible_a, visible_b = _outcome_pair(
        coordinate.visible_outcomes, name="visible_outcomes"
    )
    hidden_a, hidden_b = _outcome_pair(
        coordinate.hidden_outcomes, name="hidden_outcomes"
    )
    alice = [0, 0]
    bob = [0, 0]
    alice[x] = visible_a
    alice[1 - x] = hidden_a
    bob[y] = visible_b
    bob[1 - y] = hidden_b
    return alice[0], alice[1], bob[0], bob[1]


def deterministic_oriented_scores() -> tuple[int, ...]:
    """모든 원자의 ``F_lambda``를 돌려준다. 값은 모두 ``-2`` 또는 ``+2``다."""

    return tuple(deterministic_facet_score(atom) for atom in global_assignments())


def symmetric_signed_global_extension(eta: float) -> tuple[float, ...]:
    """``q_eta(lambda)=(1+eta*F_lambda)/16``을 돌려준다."""

    visibility = _visibility(eta)
    return tuple(
        (1.0 + visibility * score) / 16.0
        for score in deterministic_oriented_scores()
    )


def _finite_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """유한한 값 16개로 이루어진 가중치인지 검사한다."""

    declared = tuple(float(value) for value in weights)
    if len(declared) != 16 or not all(math.isfinite(value) for value in declared):
        raise ValueError("weights must contain sixteen finite values")
    return declared


def marginalize_global_weights(weights: Sequence[float]) -> np.ndarray:
    """부호 있는 또는 양의 원자 가중치를 네 맥락 모두로 사영한다."""

    declared = _finite_weights(weights)
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for weight, assignment in zip(declared, global_assignments()):
        a0, a1, b0, b1 = assignment
        alice = (a0, a1)
        bob = (b0, b1)
        for x in SETTINGS:
            for y in SETTINGS:
                a_index = OUTCOMES.index(alice[x])
                b_index = OUTCOMES.index(bob[y])
                probabilities[x, y, a_index, b_index] += weight
    return probabilities


def total_variation_norm(weights: Sequence[float]) -> float:
    """무차원 부호 ``l1`` 노름을 돌려준다."""

    return math.fsum(abs(value) for value in _finite_weights(weights))


def negative_mass(weights: Sequence[float]) -> float:
    """부호 있는 정규화 확장의 ``sum(max(-q_lambda,0))``을 돌려준다."""

    return math.fsum(max(-value, 0.0) for value in _finite_weights(weights))


def normalized_absolute_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """``|q|/||q||_1``을 돌려준다. 이는 새로운 양의 모형이다."""

    declared = _finite_weights(weights)
    norm = math.fsum(abs(value) for value in declared)
    if norm <= 0.0:
        raise ValueError("absolute weights must have positive total mass")
    return tuple(abs(value) / norm for value in declared)


def context_cells() -> tuple[tuple[int, int, int, int], ...]:
    """주변 결합(incidence) 행렬의 행 라벨 ``(x,y,a,b)``를 돌려준다."""

    return tuple(
        (x, y, a, b)
        for x in SETTINGS
        for y in SETTINGS
        for a in OUTCOMES
        for b in OUTCOMES
    )


def marginal_incidence_matrix() -> np.ndarray:
    """원자 16개를 맥락 칸 16개로 보내는 정확한 0/1 사상을 돌려준다."""

    matrix = np.zeros((16, 16), dtype=np.int64)
    for row, (x, y, a, b) in enumerate(context_cells()):
        for column, assignment in enumerate(global_assignments()):
            a0, a1, b0, b1 = assignment
            alice = (a0, a1)
            bob = (b0, b1)
            matrix[row, column] = int(alice[x] == a and bob[y] == b)
    return matrix


def exact_rational_rank(matrix: np.ndarray) -> int:
    """작은 행렬의 계수(rank)를 정확한 유리수 행 소거로 계산한다."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if not np.isfinite(values.astype(np.float64)).all():
        raise ValueError("matrix entries must be finite")
    rows = [
        [Fraction(str(values[row, column])) for column in range(values.shape[1])]
        for row in range(values.shape[0])
    ]
    rank = 0
    for column in range(values.shape[1]):
        pivot = next(
            (row for row in range(rank, len(rows)) if rows[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def walsh_kernel_vectors() -> dict[str, tuple[int, ...]]:
    """주변 사상의 관측되지 않는 월시(Walsh) 방향 일곱 개를 돌려준다."""

    vectors: dict[str, list[int]] = {
        "A0A1": [],
        "B0B1": [],
        "A0A1B0": [],
        "A0A1B1": [],
        "A0B0B1": [],
        "A1B0B1": [],
        "A0A1B0B1": [],
    }
    for a0, a1, b0, b1 in global_assignments():
        vectors["A0A1"].append(a0 * a1)
        vectors["B0B1"].append(b0 * b1)
        vectors["A0A1B0"].append(a0 * a1 * b0)
        vectors["A0A1B1"].append(a0 * a1 * b1)
        vectors["A0B0B1"].append(a0 * b0 * b1)
        vectors["A1B0B1"].append(a1 * b0 * b1)
        vectors["A0A1B0B1"].append(a0 * a1 * b0 * b1)
    return {name: tuple(values) for name, values in vectors.items()}


def quantum_kernel_perturbed_extension(delta: float) -> tuple[float, ...]:
    """``q_quantum + delta*A0*A1/16``을 돌려준다.

    모든 맥락 주변 분포는 바뀌지 않는다. 이 확장은 닫힌 구간 ``|delta| <= sqrt(2)-1``에서
    l1 최소화자로 남지만, 양자 CHSH 표적이 국소 다면체 밖에 있으므로 그 구간의 어느 점에서도
    양의 확률분포가 아니다.
    """

    parameter = float(delta)
    if not math.isfinite(parameter):
        raise ValueError("delta must be finite")
    base = symmetric_signed_global_extension(QUANTUM_ETA)
    direction = walsh_kernel_vectors()["A0A1"]
    return tuple(
        weight + parameter * value / 16.0
        for weight, value in zip(base, direction)
    )


def swap_opposite_score_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """주변 결합 자기동형이 아닌 원자 전단사 하나를 적용한다."""

    permuted = list(_finite_weights(weights))
    scores = deterministic_oriented_scores()
    negative_index = scores.index(-2)
    positive_index = scores.index(2)
    permuted[negative_index], permuted[positive_index] = (
        permuted[positive_index],
        permuted[negative_index],
    )
    return tuple(permuted)


@dataclass(frozen=True)
class ContextualGlobalSectionCertificate:
    """E29 유한 전역 단면·부호 측도 인증서다."""

    eta: float
    atom_count: int
    context_count: int
    full_ledger_round_trip_failures: int
    minimum_full_ledger_unique_image_count: int
    minimum_visible_projection_fibre_size: int
    maximum_visible_projection_fibre_size: int
    deterministic_oriented_scores: tuple[int, ...]
    incidence_rank: int
    incidence_nullity: int
    maximum_walsh_kernel_residual: int
    target_correlations: tuple[float, float, float, float]
    target_oriented_score: float
    target_absolute_chsh_score: float
    target_minimum_probability: float
    target_normalization_residual: float
    target_no_signalling_residual: float
    parent_instrument_probability_residual: float
    signed_weights: tuple[float, ...]
    signed_weight_sum: float
    signed_normalization_residual: float
    signed_minimum_weight: float
    signed_maximum_weight: float
    signed_negative_atom_count: int
    signed_positive_atom_count: int
    signed_context_marginal_residual: float
    signed_l1_norm: float
    signed_l1_lower_bound: float
    signed_l1_saturation_residual: float
    signed_negative_mass: float
    signed_negative_mass_lower_bound: float
    positive_global_chsh_gap: float
    delta_minimizer_half_width: float
    delta_witness: float
    delta_context_marginal_residual: float
    delta_l1_residual: float
    endpoint_maximum_context_marginal_residual: float
    endpoint_maximum_l1_residual: float
    endpoint_minimum_absolute_weight: float
    minimum_beyond_interval_l1_excess: float
    raw_absolute_mass: float
    normalized_absolute_mass: float
    normalized_absolute_correlations: tuple[float, float, float, float]
    normalized_absolute_oriented_score: float
    normalized_absolute_target_residual: float
    normalized_absolute_no_signalling_residual: float
    permutation_sum_residual: float
    permutation_l1_residual: float
    permutation_negative_mass_residual: float
    permutation_target_marginal_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> ContextualGlobalSectionCertificate:
    """E29 유한 전역 단면·부호 측도 인증서를 만든다."""

    tol = _positive_tolerance(tolerance)
    assignments = global_assignments()
    contexts = tuple((x, y) for x in SETTINGS for y in SETTINGS)

    round_trip_failures = 0
    unique_image_counts: list[int] = []
    visible_fibre_sizes: list[int] = []
    for x, y in contexts:
        coordinates = tuple(lift_context_ledger(atom, x, y) for atom in assignments)
        round_trip_failures += sum(
            invert_context_ledger(coordinate) != atom
            for atom, coordinate in zip(assignments, coordinates)
        )
        unique_image_counts.append(len(set(coordinates)))
        for visible_a in OUTCOMES:
            for visible_b in OUTCOMES:
                visible_fibre_sizes.append(
                    sum(
                        coordinate.visible_outcomes == (visible_a, visible_b)
                        for coordinate in coordinates
                    )
                )

    incidence = marginal_incidence_matrix()
    incidence_rank = exact_rational_rank(incidence)
    kernel_vectors = walsh_kernel_vectors()
    maximum_kernel_residual = max(
        int(np.max(np.abs(incidence @ np.asarray(vector, dtype=np.int64))))
        for vector in kernel_vectors.values()
    )

    target = isotropic_chsh_box(QUANTUM_ETA)
    target_audit = audit_probability_box(target)
    target_correlations = box_correlations(target)
    target_oriented_score, target_absolute_score = chsh_scores(target)
    instrument = quantum_projective_instrument_audit()
    parent_instrument_residual = float(
        np.max(np.abs(instrument.probabilities - target))
    )

    signed = symmetric_signed_global_extension(QUANTUM_ETA)
    signed_sum = math.fsum(signed)
    signed_box = marginalize_global_weights(signed)
    signed_marginal_residual = float(np.max(np.abs(signed_box - target)))
    signed_l1 = total_variation_norm(signed)
    signed_l1_lower_bound = max(1.0, target_oriented_score / 2.0)
    signed_negativity = negative_mass(signed)
    signed_negativity_lower_bound = 0.5 * (signed_l1_lower_bound - 1.0)

    delta_half_width = math.sqrt(2.0) - 1.0
    delta_witness = 0.5 * delta_half_width
    delta_extension = quantum_kernel_perturbed_extension(delta_witness)
    delta_box = marginalize_global_weights(delta_extension)
    endpoints = tuple(
        quantum_kernel_perturbed_extension(sign * delta_half_width)
        for sign in (-1.0, 1.0)
    )
    beyond_extensions = tuple(
        quantum_kernel_perturbed_extension(sign * 1.1 * delta_half_width)
        for sign in (-1.0, 1.0)
    )

    raw_absolute_mass = signed_l1
    absolute_weights = normalized_absolute_weights(signed)
    absolute_mass = math.fsum(absolute_weights)
    absolute_box = marginalize_global_weights(absolute_weights)
    absolute_correlations = box_correlations(absolute_box)
    absolute_oriented_score, _ = chsh_scores(absolute_box)
    absolute_audit = audit_probability_box(absolute_box)

    permuted = swap_opposite_score_weights(signed)
    permuted_box = marginalize_global_weights(permuted)

    numerical_limit = 50.0 * tol
    full_ledger_bijection = (
        round_trip_failures == 0
        and min(unique_image_counts) == len(assignments)
    )
    visible_projection_many_to_one = (
        min(visible_fibre_sizes) == 4 and max(visible_fibre_sizes) == 4
    )
    signed_extension_certified = (
        abs(signed_sum - 1.0) <= numerical_limit
        and signed_marginal_residual <= numerical_limit
        and sum(value < 0.0 for value in signed) == 8
        and sum(value > 0.0 for value in signed) == 8
    )
    minimum_norm_certified = (
        abs(signed_l1 - signed_l1_lower_bound) <= numerical_limit
        and abs(signed_negativity - signed_negativity_lower_bound)
        <= numerical_limit
    )
    delta_nonunique_minimizer_certified = (
        float(np.max(np.abs(delta_box - target))) <= numerical_limit
        and abs(total_variation_norm(delta_extension) - signed_l1)
        <= numerical_limit
        and max(
            float(np.max(np.abs(marginalize_global_weights(item) - target)))
            for item in endpoints
        )
        <= numerical_limit
        and max(abs(total_variation_norm(item) - signed_l1) for item in endpoints)
        <= numerical_limit
        and min(total_variation_norm(item) for item in beyond_extensions)
        > signed_l1 + numerical_limit
    )
    absolute_replacement_changes_target = (
        float(np.max(np.abs(absolute_box - target))) > numerical_limit
        and np.allclose(
            np.asarray(absolute_correlations),
            0.5 * np.asarray(target_correlations),
            atol=numerical_limit,
            rtol=0.0,
        )
        and abs(absolute_oriented_score - math.sqrt(2.0)) <= numerical_limit
    )
    permutation_invariants_preserved = (
        abs(math.fsum(permuted) - signed_sum) <= numerical_limit
        and abs(total_variation_norm(permuted) - signed_l1) <= numerical_limit
        and abs(negative_mass(permuted) - signed_negativity) <= numerical_limit
    )
    permutation_changes_marginals = (
        float(np.max(np.abs(permuted_box - target))) > numerical_limit
    )

    dimensions = {
        "eta_is_dimensionless": True,
        "born_probabilities_are_dimensionless": True,
        "signed_global_weights_are_dimensionless": True,
        "facet_scores_and_l1_norm_are_dimensionless": True,
        "negativity_is_dimensionless": True,
        "no_energy_length_time_or_mass_scale_introduced": True,
    }
    accounting = {
        "each_context_born_box_normalized_once": (
            target_audit.maximum_normalization_residual <= numerical_limit
        ),
        "full_ledger_relabels_each_atom_once": full_ledger_bijection,
        "visible_projection_does_not_add_hidden_weights": True,
        "signed_extension_is_an_alternative_linear_representation": True,
        "signed_and_absolute_models_are_not_added_together": True,
        "absolute_replacement_is_explicitly_renormalized": (
            abs(absolute_mass - 1.0) <= numerical_limit
        ),
        "signed_weight_not_added_as_energy_or_stress": True,
        "signed_or_hidden_atom_carries_energy": False,
    }
    boundaries = {
        "full_visible_plus_hidden_ledger_is_bijective": full_ledger_bijection,
        "finite_discrete_full_ledger_bijection_is_homeomorphism": (
            full_ledger_bijection
        ),
        "visible_readout_alone_is_many_to_one": visible_projection_many_to_one,
        "set_bijection_does_not_imply_measure_preservation": True,
        "positive_global_measure_failure_is_not_bijection_failure": True,
        "global_state_destruction_not_inferred": True,
        "signed_weight_is_not_observed_probability_or_frequency": True,
        "signed_weight_is_not_negative_energy_or_stress": True,
        "finite_discrete_zero_dimensionality_is_not_spacetime_dimension": True,
        "absolute_value_result_uses_symmetric_delta_zero_representative": True,
        "absolute_value_result_is_not_general_metric_measure_or_gravity_no_go": True,
        "atom_permutation_preserves_marginals_only_if_incidence_is_respected": True,
        "fine_and_global_section_results_are_limited_to_2x2_binary_scenario": True,
        "signed_extension_is_not_a_quantum_channel_or_selection_dynamics": True,
        "operational_no_signalling_is_not_qft_microcausality": True,
    }
    alternatives = {
        "context_dependent_per_setting_instrument_or_ledger": True,
        "measurement_dependent_or_retrocausal_route": True,
        "ontically_nonlocal_but_operationally_no_signalling_route": True,
        "future_lightcone_pointer_domino_only": True,
        "independent_representation_invariant_geometry_and_measure_law": True,
    }
    status = {
        "full_context_ledger_set_bijection_certified": full_ledger_bijection,
        "visible_projection_many_to_one_certified": visible_projection_many_to_one,
        "incidence_rank_nine_nullity_seven_certified": (
            incidence_rank == 9
            and 16 - incidence_rank == 7
            and maximum_kernel_residual == 0
        ),
        "all_context_signed_extension_certified": signed_extension_certified,
        "positive_setting_independent_global_probability_excluded_for_target": (
            target_oriented_score > 2.0 + numerical_limit
            and set(deterministic_oriented_scores()) == {-2, 2}
        ),
        "minimum_signed_l1_and_negativity_certified": minimum_norm_certified,
        "minimum_signed_extension_is_nonunique": (
            delta_nonunique_minimizer_certified
        ),
        "symmetric_absolute_replacement_changes_born_marginals": (
            absolute_replacement_changes_target
        ),
        "arbitrary_atom_bijection_need_not_preserve_physical_marginals": (
            permutation_invariants_preserved and permutation_changes_marginals
        ),
        "fixed_context_parent_instruments_remain_cptp": (
            parent_instrument_residual <= numerical_limit
            and instrument.maximum_completeness_residual <= numerical_limit
            and instrument.minimum_choi_eigenvalue >= -numerical_limit
        ),
        "finite_target_operational_no_signalling_certified": (
            target_audit.maximum_no_signalling_residual <= numerical_limit
        ),
        "physical_hidden_path_or_seed_law_derived": False,
        "objective_single_outcome_selection_derived": False,
        "relativistic_qft_microcausality_derived": False,
        "full_lightcone_no_controllable_influence_gate_complete": False,
        "spacetime_metric_volume_or_gravity_derived": False,
        "mass_dependent_probability_deformation_derived": False,
        "independent_holdout_complete": False,
        "success_gates_1_to_8_complete": False,
    }

    return ContextualGlobalSectionCertificate(
        eta=QUANTUM_ETA,
        atom_count=len(assignments),
        context_count=len(contexts),
        full_ledger_round_trip_failures=round_trip_failures,
        minimum_full_ledger_unique_image_count=min(unique_image_counts),
        minimum_visible_projection_fibre_size=min(visible_fibre_sizes),
        maximum_visible_projection_fibre_size=max(visible_fibre_sizes),
        deterministic_oriented_scores=deterministic_oriented_scores(),
        incidence_rank=incidence_rank,
        incidence_nullity=16 - incidence_rank,
        maximum_walsh_kernel_residual=maximum_kernel_residual,
        target_correlations=target_correlations,
        target_oriented_score=target_oriented_score,
        target_absolute_chsh_score=target_absolute_score,
        target_minimum_probability=target_audit.minimum_probability,
        target_normalization_residual=target_audit.maximum_normalization_residual,
        target_no_signalling_residual=target_audit.maximum_no_signalling_residual,
        parent_instrument_probability_residual=parent_instrument_residual,
        signed_weights=signed,
        signed_weight_sum=signed_sum,
        signed_normalization_residual=abs(signed_sum - 1.0),
        signed_minimum_weight=min(signed),
        signed_maximum_weight=max(signed),
        signed_negative_atom_count=sum(value < 0.0 for value in signed),
        signed_positive_atom_count=sum(value > 0.0 for value in signed),
        signed_context_marginal_residual=signed_marginal_residual,
        signed_l1_norm=signed_l1,
        signed_l1_lower_bound=signed_l1_lower_bound,
        signed_l1_saturation_residual=abs(signed_l1 - signed_l1_lower_bound),
        signed_negative_mass=signed_negativity,
        signed_negative_mass_lower_bound=signed_negativity_lower_bound,
        positive_global_chsh_gap=target_oriented_score - 2.0,
        delta_minimizer_half_width=delta_half_width,
        delta_witness=delta_witness,
        delta_context_marginal_residual=float(np.max(np.abs(delta_box - target))),
        delta_l1_residual=abs(total_variation_norm(delta_extension) - signed_l1),
        endpoint_maximum_context_marginal_residual=max(
            float(np.max(np.abs(marginalize_global_weights(item) - target)))
            for item in endpoints
        ),
        endpoint_maximum_l1_residual=max(
            abs(total_variation_norm(item) - signed_l1) for item in endpoints
        ),
        endpoint_minimum_absolute_weight=min(
            abs(value) for item in endpoints for value in item
        ),
        minimum_beyond_interval_l1_excess=(
            min(total_variation_norm(item) for item in beyond_extensions) - signed_l1
        ),
        raw_absolute_mass=raw_absolute_mass,
        normalized_absolute_mass=absolute_mass,
        normalized_absolute_correlations=absolute_correlations,
        normalized_absolute_oriented_score=absolute_oriented_score,
        normalized_absolute_target_residual=float(
            np.max(np.abs(absolute_box - target))
        ),
        normalized_absolute_no_signalling_residual=(
            absolute_audit.maximum_no_signalling_residual
        ),
        permutation_sum_residual=abs(math.fsum(permuted) - signed_sum),
        permutation_l1_residual=abs(total_variation_norm(permuted) - signed_l1),
        permutation_negative_mass_residual=abs(
            negative_mass(permuted) - signed_negativity
        ),
        permutation_target_marginal_residual=float(
            np.max(np.abs(permuted_box - target))
        ),
        dimensions=dimensions,
        accounting=accounting,
        boundaries=boundaries,
        alternatives=alternatives,
        status=status,
    )


def run() -> dict[str, object]:
    """JSON 직렬화 가능한 E29 인증서를 돌려준다."""

    return asdict(certificate())


def main() -> None:
    """E29 인증서를 명령줄에서 출력한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(), indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
