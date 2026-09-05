"""가시 확률의 피셔 기하, 숨은 원천의 인수분해·영수증 계수(rank), 스칼라 영수증의 원천 허용에 대한 유한 인증서를 모은다.

이 모듈은 세 부분으로 이루어진다.

첫째, 관측 가능한 피셔(Fisher) 기하와 그 시공간 계량 경계다(E30).

E29 2x2 이진 시나리오에서 보이는 엄격히 양의 조건부 확률에서 출발한다. 고정된 맥락 설계
``pi_xy``는 결과 칸마다가 아니라 맥락마다 한 번 세며

``ds^2 = sum_xy pi_xy sum_ab (dP_ab|xy)^2 / P_ab|xy``

를 정의한다. 이 피셔--라오(Fisher--Rao) 형식을 E29 결합 사상 ``P=Mq``로 당겨 오면 부호
들어올림 핵(kernel) 방향 일곱 개가 정확히 소멸한다. 결과 형식은 전역 좌표 위에서 양의
준정부호이며 그 핵과 정규화로 몫을 취한 뒤에만 양의 정부호다. 이는 정보 계량이지 로런츠
시공간 계량이 아니다.

마지막 등각 계산은 명시적으로 조건부다. 로런츠 기준 계량과 독립적인 양의 무차원 부피 비가
공급되면 E24 대수가 영 원뿔을 보존하며 등각 대표를 고정한다. 이는 부피 법칙, 곡률,
아인슈타인 동역학, 중력, 부호 가중치의 물리적 의미를 유도하지 않는다.

둘째, 유한 원천 인수분해와 독립 영수증 계수 경계다(E31).

정준 E29 주변 사상 ``P=Mq``에서 출발한다. 제안된 원천이 ``M``의 모든 올(fibre)에서
상수이면 숨은 들어올림 정보를 담지 않고 가시 상을 통해 인수분해된다. 반대로 추가 선형
영수증 ``E q``는 ``ker(M)``과 ``ker(E)``가 자명하게만 만날 때 모든 숨은 방향을 구별한다.

아래의 월시(Walsh) 행 일곱 개는 무차원 좌표 진단이다. 완전한 유한 부호 좌표 ``q``를
재구성하기 위한 계수 하한을 증명할 뿐이며, 물리 기록, 에너지 영수증, 응력 성분, 시공간
부피, 장, 보손이 아니다. 여기서 계량이나 중력 법칙은 만들지 않는다.

셋째, 공변 응력 후보에 대한 E32 스칼라 영수증 허용 게이트다.

이 부분은 E31 유한 장부가 계수 완전(rank-complete)해진 *뒤*에서 시작한다. 월시 좌표를
물리 장과 동일시하지 않으며, 더 좁은 두 경계를 인증한다.

첫째, 4차원 로런츠 시공간에서 공급된 계량과 무차원 스칼라 영수증 값만으로 만든 점별
대수적 대칭 공변 2-계 텐서는 계량에 비례한다. 이 진술은 미분, 곡률, 벡터나 엽층 자료,
비국소 핵을 배제한다. 유한 인증서는 대칭 텐서의 10차원 공간에 무한소 로런츠 제약을
구성한다. 공간 회전은 2차원 등방 부분공간을 남기고, 부스트를 더하면 ``span(eta)``만 남는다.

둘째, 공급된 국소 스칼라 매장(embedding)이 있어도 작용의 가법 정규화는 선택되지 않는다.
무차원 상수 영수증 ``r0``와 ``phi0 = M_star * r0``에 대해 작용

    S_total^(a) = S_EH + S_visible + epsilon S_h^(a),

    S_h^(a) = -integral sqrt(-g) [
        (grad phi)^2 / 2 + m^2 (phi - phi0)^2 / 2 + a
    ]

은 ``a = 0``과 ``a = M_star**4``에서, 고정된 모든 ``epsilon > 0``에 대해 같은 스칼라
운동방정식, 같은 주 기호(principal symbol), 같은 상수 온셸(on-shell) 장을 가진다. 온셸
응력은 ``-epsilon * M_star**4 * g``만큼 다르다. 따라서 영수증 완전성과 보존은 중력 원천을
고르지 않는다. 극한 ``epsilon -> 0``에서 숨은 응력은 별도로 공급된 GR+가시 방정식에서
사라지며, 계량 해의 수렴에 대해서는 아무 주장도 하지 않는다.

모든 양은 자연 단위다. 결과는 유한 대수적 허용/불가 증인이지 계량 유도, CPTP 구성,
미시인과성 증명, 관측 예측이 아니다.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable, Sequence

import numpy as np

from examples.physics.causal.causal_light_geometry import volume_recovery
from examples.physics.causal.contextual_obstruction import (
    CHSH_PATTERN,
    OUTCOMES,
    QUANTUM_ETA,
    SETTINGS,
    deterministic_oriented_scores,
    exact_rational_rank,
    isotropic_chsh_box,
    marginal_incidence_matrix,
    marginalize_global_weights,
    quantum_kernel_perturbed_extension,
    swap_opposite_score_weights,
    symmetric_signed_global_extension,
    walsh_kernel_vectors,
)


DEFAULT_TOLERANCE = 1.0e-12
UNIFORM_CONTEXT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
ATOM_COUNT = 16
SPACETIME_DIMENSION = 4
SYMMETRIC_COMPONENT_COUNT = 10
COMPONENT_ORDER = (
    "00",
    "01",
    "02",
    "03",
    "11",
    "12",
    "13",
    "22",
    "23",
    "33",
)


def _positive_tolerance(value: float) -> float:
    """유한하고 양수인 허용오차만 통과시킨다."""

    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


def _context_weights(
    values: Sequence[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[float, float, float, float]:
    """합이 1인 유한한 양의 맥락 가중치 네 개인지 검사한다."""

    tol = _positive_tolerance(tolerance)
    weights = tuple(float(value) for value in values)
    if len(weights) != 4 or not all(
        math.isfinite(value) and value > 0.0 for value in weights
    ):
        raise ValueError("context weights must be four finite positive values")
    if abs(math.fsum(weights) - 1.0) > tol:
        raise ValueError("context weights must sum to one")
    return weights  # type: ignore[return-value]


def _strict_probability_box(
    probabilities: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    """모든 칸이 양수이고 맥락마다 정규화된 확률 상자인지 검사한다."""

    tol = _positive_tolerance(tolerance)
    box = np.asarray(probabilities, dtype=np.float64)
    if box.shape != (2, 2, 2, 2) or not np.isfinite(box).all():
        raise ValueError("probability box must be finite with shape (2, 2, 2, 2)")
    if float(box.min()) <= 0.0:
        raise ValueError("Fisher chart requires every probability cell to be positive")
    residual = max(
        abs(float(np.sum(box[x, y])) - 1.0) for x in SETTINGS for y in SETTINGS
    )
    if residual > tol:
        raise ValueError("each context probability box must sum to one")
    return np.array(box, copy=True)


def _conditional_tangent(
    differential: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    """맥락마다 합이 0인 유한한 접벡터인지 검사한다."""

    tol = _positive_tolerance(tolerance)
    tangent = np.asarray(differential, dtype=np.float64)
    if tangent.shape != (2, 2, 2, 2) or not np.isfinite(tangent).all():
        raise ValueError("differential must be finite with shape (2, 2, 2, 2)")
    residual = max(
        abs(float(np.sum(tangent[x, y]))) for x in SETTINGS for y in SETTINGS
    )
    if residual > tol:
        raise ValueError("each context differential must have zero sum")
    return np.array(tangent, copy=True)


def _expanded_context_weights(
    context_weights: Sequence[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    """맥락 가중치를 결과 칸 열여섯 개로 펼친다."""

    weights = _context_weights(context_weights, tolerance=tolerance)
    return np.repeat(np.asarray(weights, dtype=np.float64), 4)


def hellinger_coordinates(
    probabilities: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """E30 규약의 ``Psi_xyab=2*sqrt(pi_xy*P_xyab)``를 돌려준다."""

    box = _strict_probability_box(probabilities, tolerance=tolerance)
    weights = _context_weights(context_weights, tolerance=tolerance)
    expanded = np.asarray(weights, dtype=np.float64).reshape(2, 2, 1, 1)
    return 2.0 * np.sqrt(expanded * box)


def hellinger_tangent(
    probabilities: np.ndarray,
    differential: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """``dPsi=sqrt(pi_xy/P_xyab)*dP_xyab``를 돌려준다."""

    box = _strict_probability_box(probabilities, tolerance=tolerance)
    tangent = _conditional_tangent(differential, tolerance=tolerance)
    weights = _context_weights(context_weights, tolerance=tolerance)
    expanded = np.asarray(weights, dtype=np.float64).reshape(2, 2, 1, 1)
    return np.sqrt(expanded / box) * tangent


def conditional_fisher_quadratic(
    probabilities: np.ndarray,
    differential: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """선언된 조건부 피셔--라오 이차 형식을 계산한다."""

    tangent_coordinates = hellinger_tangent(
        probabilities,
        differential,
        context_weights=context_weights,
        tolerance=tolerance,
    )
    return float(np.sum(tangent_coordinates * tangent_coordinates))


def product_fisher_rao_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """고정 맥락 가중치에 대한 곱 피셔 거리를 돌려준다.

    각 맥락 단체(simplex)의 거리는 ``2 acos(sum_ab sqrt(P_ab Q_ab))``이며, 네 맥락 거리를
    선언된 ``pi_xy``로 결합한다.
    """

    left = _strict_probability_box(first, tolerance=tolerance)
    right = _strict_probability_box(second, tolerance=tolerance)
    weights = _context_weights(context_weights, tolerance=tolerance)
    squared = 0.0
    for context_index, (x, y) in enumerate(
        (pair for pair in ((0, 0), (0, 1), (1, 0), (1, 1)))
    ):
        coefficient = float(np.sum(np.sqrt(left[x, y] * right[x, y])))
        coefficient = min(1.0, max(0.0, coefficient))
        context_distance = 2.0 * math.acos(coefficient)
        squared += weights[context_index] * context_distance * context_distance
    return math.sqrt(max(0.0, squared))


def fisher_weight_matrix(
    probabilities: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """고정 행 순서의 ``diag(pi_xy/P_xyab)``를 돌려준다."""

    box = _strict_probability_box(probabilities, tolerance=tolerance)
    expanded = _expanded_context_weights(context_weights, tolerance=tolerance)
    return np.diag(expanded / box.reshape(-1))


def fisher_pullback_metric(
    probabilities: np.ndarray,
    *,
    incidence: np.ndarray | None = None,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """선언된 선형 사상에 대한 ``M.T @ diag(pi/P) @ M``을 돌려준다.

    모양 검증은 대수를 닫힌 실패로 만들지만, 호출자가 준 행렬이 물리적 주변 결합 사상임을
    인증하지는 않는다. :func:`representation_certificate`의 정확한 E29 계수·핵 주장은 정준
    ``marginal_incidence_matrix()``만 쓴다.
    """

    matrix = (
        marginal_incidence_matrix()
        if incidence is None
        else np.asarray(incidence, dtype=np.float64)
    )
    if matrix.ndim != 2 or matrix.shape[0] != 16 or not np.isfinite(matrix).all():
        raise ValueError("incidence must be finite with sixteen context-cell rows")
    weight = fisher_weight_matrix(
        probabilities, context_weights=context_weights, tolerance=tolerance
    )
    metric = matrix.T @ weight @ matrix
    return 0.5 * (metric + metric.T)


def normalized_atom_tangent_basis() -> np.ndarray:
    """열 합이 0인 16x15 기저를 돌려준다."""

    basis = np.zeros((16, 15), dtype=np.float64)
    for column in range(15):
        basis[column, column] = 1.0
        basis[15, column] = -1.0
    return basis


def matrix_inertia(
    matrix: np.ndarray, *, tolerance: float = 1.0e-10
) -> tuple[int, int, int]:
    """양·음·영 고윳값의 개수를 돌려준다."""

    tol = _positive_tolerance(tolerance)
    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.size == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("matrix must be finite, nonempty, and square")
    if not np.allclose(values, values.T, atol=tol, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values)
    positive = int(np.sum(eigenvalues > tol))
    negative = int(np.sum(eigenvalues < -tol))
    zero = len(eigenvalues) - positive - negative
    return positive, negative, zero


def context_block_permutation(order: Sequence[int]) -> np.ndarray:
    """연속한 맥락 블록 네 개에 대한 행 치환을 돌려준다."""

    declared = tuple(order)
    if (
        len(declared) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in declared)
        or set(declared) != {0, 1, 2, 3}
    ):
        raise ValueError("context order must be a permutation of (0,1,2,3)")
    permutation = np.zeros((16, 16), dtype=np.float64)
    for new_context, old_context in enumerate(declared):
        for cell in range(4):
            permutation[4 * new_context + cell, 4 * old_context + cell] = 1.0
    return permutation


def atom_permutation_matrix(order: Sequence[int]) -> np.ndarray:
    """16x16 원자 좌표 치환 행렬을 돌려준다."""

    declared = tuple(order)
    if (
        len(declared) != 16
        or any(isinstance(value, bool) or not isinstance(value, int) for value in declared)
        or set(declared) != set(range(16))
    ):
        raise ValueError("atom order must be a permutation of range(16)")
    permutation = np.zeros((16, 16), dtype=np.float64)
    for new_index, old_index in enumerate(declared):
        permutation[new_index, old_index] = 1.0
    return permutation


def isotropic_fisher_component(eta: float) -> float:
    """``0 <= eta < 1``에서 ``g_etaeta=1/(1-eta^2)``를 돌려준다."""

    parameter = float(eta)
    if not math.isfinite(parameter) or not 0.0 <= parameter < 1.0:
        raise ValueError("eta must be finite and lie in [0, 1) for the strict chart")
    return 1.0 / (1.0 - parameter * parameter)


def isotropic_fisher_distance(first_eta: float, second_eta: float) -> float:
    """정확한 등방 거리 ``|asin(eta2)-asin(eta1)|``을 돌려준다."""

    first = float(first_eta)
    second = float(second_eta)
    isotropic_fisher_component(first)
    isotropic_fisher_component(second)
    return abs(math.asin(second) - math.asin(first))


def _isotropic_tangent() -> np.ndarray:
    """등방 선을 따라가는 접벡터 ``0.25*a*b*c_xy``를 만든다."""

    tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for x in SETTINGS:
        for y in SETTINGS:
            for a_index, a in enumerate(OUTCOMES):
                for b_index, b in enumerate(OUTCOMES):
                    tangent[x, y, a_index, b_index] = (
                        0.25 * a * b * CHSH_PATTERN[x, y]
                    )
    return tangent


def lorentzian_signature(
    metric: np.ndarray, *, tolerance: float = 1.0e-10
) -> tuple[int, int, int]:
    """대칭 검증 뒤 관성(inertia)을 돌려준다. 규약은 음수 하나다."""

    return matrix_inertia(metric, tolerance=tolerance)


def conditional_conformal_metric(
    reference_metric: np.ndarray,
    volume_ratio: float,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """공급된 E24 등각 부피 대수를 적용한다.

    ``g0``가 음의 고윳값 하나와 양의 고윳값 ``d-1``개를 가지는지 검사한 뒤에만
    ``g = v^(2/d) g0``를 돌려준다. 두 입력 중 어느 것도 유도하지 않는다.
    """

    tol = _positive_tolerance(tolerance)
    reference = np.asarray(reference_metric, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[0] != reference.shape[1]
        or reference.shape[0] < 2
        or not np.isfinite(reference).all()
    ):
        raise ValueError("reference metric must be a finite square matrix of dimension >=2")
    if not np.allclose(reference, reference.T, atol=tol, rtol=0.0):
        raise ValueError("reference metric must be symmetric")
    dimension = reference.shape[0]
    positive, negative, zero = lorentzian_signature(reference, tolerance=tol)
    if (positive, negative, zero) != (dimension - 1, 1, 0):
        raise ValueError("reference metric must have one-negative Lorentzian signature")
    ratio = float(volume_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("volume_ratio must be finite and positive")
    conformal_factor = volume_recovery(ratio, n=dimension)
    return conformal_factor * conformal_factor * reference


def metric_volume_ratio(
    metric: np.ndarray,
    reference_metric: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> float:
    """대칭 비퇴화 계량에 대한 ``sqrt(|det g|/|det g0|)``를 돌려준다."""

    tol = _positive_tolerance(tolerance)
    current = np.asarray(metric, dtype=np.float64)
    reference = np.asarray(reference_metric, dtype=np.float64)
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current.shape != reference.shape
        or not np.isfinite(current).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("metric and reference_metric must be finite square matrices of equal shape")
    if not np.allclose(current, current.T, atol=tol, rtol=0.0) or not np.allclose(
        reference, reference.T, atol=tol, rtol=0.0
    ):
        raise ValueError("metric and reference_metric must be symmetric")
    denominator = abs(float(np.linalg.det(reference)))
    numerator = abs(float(np.linalg.det(current)))
    if denominator <= tol or numerator <= tol:
        raise ValueError("metric determinants must be nonzero")
    return math.sqrt(numerator / denominator)


@dataclass(frozen=True)
class HighFrequencyVolumeWitness:
    """균등 수렴하지만 C2 수렴하지 않는 부피 비 열 ``v_n``의 증인이다."""

    n: int
    minimum_volume_ratio: float
    uniform_value_residual_bound: float
    probe_time: float
    probe_value_residual: float
    probe_first_derivative: float
    probe_second_derivative: float


def high_frequency_volume_witness(n: int) -> HighFrequencyVolumeWitness:
    """C2 수렴 없는 균등 수렴에 대한 ``v_n`` 증인을 돌려준다."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer of at least two")
    frequency = float(n * n)
    amplitude = 1.0 / frequency
    probe_time = math.pi / (2.0 * frequency)
    phase = frequency * probe_time
    return HighFrequencyVolumeWitness(
        n=n,
        minimum_volume_ratio=1.0 - amplitude,
        uniform_value_residual_bound=amplitude,
        probe_time=probe_time,
        probe_value_residual=amplitude * math.sin(phase),
        probe_first_derivative=math.cos(phase),
        probe_second_derivative=-frequency * math.sin(phase),
    )


@dataclass(frozen=True)
class RepresentationInvariantMeasureCertificate:
    """E30 관측 정보 기하·등각 경계 인증서다."""

    context_weights: tuple[float, float, float, float]
    target_minimum_probability: float
    context_normalization_residual: float
    hellinger_coordinate_norm_squared: float
    hellinger_quadratic_residual: float
    incidence_rank: int
    incidence_nullity: int
    pullback_rank: int
    pullback_inertia: tuple[int, int, int]
    normalized_tangent_rank: int
    normalized_tangent_inertia: tuple[int, int, int]
    maximum_incidence_kernel_residual: float
    maximum_pullback_kernel_residual: float
    q_delta_probability_residual: float
    q_delta_pullback_residual: float
    simultaneous_relabel_probability_residual: float
    simultaneous_relabel_congruence_residual: float
    general_relabel_fixed_incidence_residual: float
    fixed_nonuniform_context_swap_residual: float
    co_transformed_context_swap_residual: float
    uniform_context_swap_residual: float
    atom_only_probability_residual: float
    atom_only_fixed_incidence_residual: float
    quantum_isotropic_component: float
    quantum_isotropic_coordinate: float
    analytic_quantum_distance: float
    product_quantum_distance: float
    isotropic_distance_residual: float
    reference_signature: tuple[int, int, int]
    conformal_signature: tuple[int, int, int]
    supplied_volume_ratio: float
    recovered_volume_ratio: float
    conformal_volume_residual: float
    null_vector_reference_residual: float
    null_vector_conformal_residual: float
    unit_volume_reference_residual: float
    high_frequency_uniform_residual_bound: float
    high_frequency_second_derivative_magnitude: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]

    def to_json(self, *, indent: int | None = 2) -> str:
        """물리적 해석을 덧붙이지 않고 인증서를 직렬화한다."""

        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def representation_certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> RepresentationInvariantMeasureCertificate:
    """E30 관측 정보·등각 경계 인증서를 만든다."""

    tol = _positive_tolerance(tolerance)
    weights = _context_weights(UNIFORM_CONTEXT_WEIGHTS, tolerance=tol)
    target = isotropic_chsh_box(QUANTUM_ETA)
    target = _strict_probability_box(target, tolerance=tol)
    normalization_residual = max(
        abs(float(np.sum(target[x, y])) - 1.0)
        for x in SETTINGS
        for y in SETTINGS
    )

    tangent = _isotropic_tangent()
    quadratic = conditional_fisher_quadratic(
        target, tangent, context_weights=weights, tolerance=tol
    )
    dpsi = hellinger_tangent(
        target, tangent, context_weights=weights, tolerance=tol
    )
    psi = hellinger_coordinates(target, context_weights=weights, tolerance=tol)

    incidence = marginal_incidence_matrix().astype(np.float64)
    exact_rank = exact_rational_rank(incidence.astype(np.int64))
    pullback = fisher_pullback_metric(
        target, incidence=incidence, context_weights=weights, tolerance=tol
    )
    inertia = matrix_inertia(pullback)
    pullback_rank = inertia[0] + inertia[1]
    tangent_basis = normalized_atom_tangent_basis()
    normalized_metric = tangent_basis.T @ pullback @ tangent_basis
    normalized_inertia = matrix_inertia(normalized_metric)
    normalized_rank = normalized_inertia[0] + normalized_inertia[1]
    kernel_vectors = walsh_kernel_vectors()
    incidence_kernel_residual = max(
        float(np.max(np.abs(incidence @ np.asarray(vector, dtype=np.float64))))
        for vector in kernel_vectors.values()
    )
    pullback_kernel_residual = max(
        float(np.max(np.abs(pullback @ np.asarray(vector, dtype=np.float64))))
        for vector in kernel_vectors.values()
    )

    base_q = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    delta_q = np.asarray(quantum_kernel_perturbed_extension(0.1))
    base_box = marginalize_global_weights(base_q)
    delta_box = marginalize_global_weights(delta_q)
    base_metric = fisher_pullback_metric(base_box, tolerance=tol)
    delta_metric = fisher_pullback_metric(delta_box, tolerance=tol)

    context_order = (1, 0, 2, 3)
    row_permutation = context_block_permutation(context_order)
    atom_permutation = atom_permutation_matrix(tuple(reversed(range(16))))
    relabelled_q = atom_permutation @ base_q
    relabelled_incidence = row_permutation @ incidence @ atom_permutation.T
    relabelled_vector = relabelled_incidence @ relabelled_q
    expected_relabelled_vector = row_permutation @ base_box.reshape(-1)
    relabelled_box = expected_relabelled_vector.reshape(2, 2, 2, 2)
    relabelled_weights = tuple(weights[index] for index in context_order)
    relabelled_metric = fisher_pullback_metric(
        relabelled_box,
        incidence=relabelled_incidence,
        context_weights=relabelled_weights,
        tolerance=tol,
    )
    expected_congruence = atom_permutation @ base_metric @ atom_permutation.T
    general_relabel_fixed_incidence_residual = float(
        np.max(np.abs(relabelled_incidence - incidence))
    )

    nonuniform_weights = (0.4, 0.3, 0.2, 0.1)
    one_context_tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    one_context_tangent[0, 0] = tangent[0, 0]
    swapped_box = (row_permutation @ target.reshape(-1)).reshape(2, 2, 2, 2)
    swapped_tangent = (
        row_permutation @ one_context_tangent.reshape(-1)
    ).reshape(2, 2, 2, 2)
    original_nonuniform_quadratic = conditional_fisher_quadratic(
        target, one_context_tangent, context_weights=nonuniform_weights, tolerance=tol
    )
    fixed_nonuniform_quadratic = conditional_fisher_quadratic(
        swapped_box,
        swapped_tangent,
        context_weights=nonuniform_weights,
        tolerance=tol,
    )
    co_transformed_weights = tuple(
        nonuniform_weights[index] for index in context_order
    )
    co_transformed_quadratic = conditional_fisher_quadratic(
        swapped_box,
        swapped_tangent,
        context_weights=co_transformed_weights,
        tolerance=tol,
    )
    original_uniform_quadratic = conditional_fisher_quadratic(
        target, one_context_tangent, context_weights=weights, tolerance=tol
    )
    swapped_uniform_quadratic = conditional_fisher_quadratic(
        swapped_box, swapped_tangent, context_weights=weights, tolerance=tol
    )

    scores = deterministic_oriented_scores()
    atom_only_order = list(range(16))
    negative_index = scores.index(-2)
    positive_index = scores.index(2)
    atom_only_order[negative_index], atom_only_order[positive_index] = (
        atom_only_order[positive_index],
        atom_only_order[negative_index],
    )
    atom_only_permutation = atom_permutation_matrix(atom_only_order)
    permuted_q = swap_opposite_score_weights(base_q)
    if not np.array_equal(atom_only_permutation @ base_q, np.asarray(permuted_q)):
        raise AssertionError("declared atom-only permutation disagrees with E29 witness")
    atom_only_box = marginalize_global_weights(permuted_q)
    atom_only_residual = float(np.max(np.abs(atom_only_box - target)))
    atom_only_fixed_incidence_residual = float(
        np.max(np.abs(incidence @ atom_only_permutation.T - incidence))
    )

    zero_box = isotropic_chsh_box(0.0)
    analytic_distance = isotropic_fisher_distance(0.0, QUANTUM_ETA)
    product_distance = product_fisher_rao_distance(
        zero_box, target, context_weights=weights, tolerance=tol
    )

    reference_metric = np.diag((-1.0, 1.0, 1.0, 1.0))
    supplied_ratio = 16.0
    conformal_metric = conditional_conformal_metric(reference_metric, supplied_ratio)
    recovered_ratio = metric_volume_ratio(conformal_metric, reference_metric)
    null_vector = np.array((1.0, 1.0, 0.0, 0.0), dtype=np.float64)
    null_reference = float(null_vector @ reference_metric @ null_vector)
    null_conformal = float(null_vector @ conformal_metric @ null_vector)
    unit_metric = conditional_conformal_metric(reference_metric, 1.0)
    high_frequency = high_frequency_volume_witness(100)

    numerical_limit = 100.0 * tol
    pullback_kernel_certified = (
        exact_rank == 9
        and inertia == (9, 0, 7)
        and incidence_kernel_residual <= numerical_limit
        and pullback_kernel_residual <= numerical_limit
    )
    normalized_quotient_certified = normalized_inertia == (8, 0, 7)
    q_delta_invariant = (
        float(np.max(np.abs(delta_box - base_box))) <= numerical_limit
        and float(np.max(np.abs(delta_metric - base_metric))) <= numerical_limit
    )
    simultaneous_relabel_certified = (
        float(np.max(np.abs(relabelled_vector - expected_relabelled_vector)))
        <= numerical_limit
        and float(np.max(np.abs(relabelled_metric - expected_congruence)))
        <= numerical_limit
    )
    context_weight_boundary_certified = (
        abs(fixed_nonuniform_quadratic - original_nonuniform_quadratic)
        > numerical_limit
        and abs(co_transformed_quadratic - original_nonuniform_quadratic)
        <= numerical_limit
        and abs(swapped_uniform_quadratic - original_uniform_quadratic)
        <= numerical_limit
    )
    isotropic_certified = (
        abs(quadratic - isotropic_fisher_component(QUANTUM_ETA))
        <= numerical_limit
        and abs(analytic_distance - math.pi / 4.0) <= numerical_limit
        and abs(product_distance - analytic_distance) <= numerical_limit
    )
    conformal_control_certified = (
        lorentzian_signature(reference_metric) == (3, 1, 0)
        and lorentzian_signature(conformal_metric) == (3, 1, 0)
        and abs(recovered_ratio - supplied_ratio) <= numerical_limit
        and abs(null_reference) <= numerical_limit
        and abs(null_conformal) <= numerical_limit
        and float(np.max(np.abs(unit_metric - reference_metric))) <= numerical_limit
    )

    dimensions = {
        "probabilities_signed_coordinates_and_context_weights_dimensionless": True,
        "fisher_line_element_and_metric_coefficients_dimensionless_here": True,
        "eta_and_chi_dimensionless": True,
        "volume_ratio_and_conformal_factor_dimensionless": True,
        "reference_metric_supplies_any_physical_length_convention": True,
        "dimensionless_information_distance_is_not_spacetime_length": True,
    }
    accounting = {
        "context_weights_sum_to_one": abs(math.fsum(weights) - 1.0) <= tol,
        "each_context_counted_once_not_once_per_outcome_cell": True,
        "fisher_metric_uses_positive_visible_probabilities_only": True,
        "signed_q_is_not_inserted_as_a_probability": True,
        "signed_q_absolute_q_and_fisher_not_added_as_energy_or_stress": True,
        "supplied_volume_ratio_is_not_derived_from_fisher_or_q": True,
        "probability_energy_or_volume_double_counted": False,
    }
    boundaries = {
        "uniform_context_weights_are_a_symmetry_axiom": True,
        "nonuniform_weights_must_cotransform_with_context_labels": True,
        "zero_probability_cells_are_outside_strict_fisher_chart": True,
        "eta_one_is_a_metric_completion_boundary_not_an_interior_point": True,
        "kernel_rank_eight_is_not_spacetime_dimension": True,
        "same_fisher_metric_does_not_select_a_hidden_signed_lift": True,
        "general_coordinate_relabel_changes_incidence_unless_automorphism": True,
        "fixed_incidence_automorphism_requires_rmc_inverse_equals_m": True,
        "caller_supplied_incidence_shape_is_not_physical_validation": True,
        "fisher_psd_no_go_is_not_a_general_lorentz_geometry_no_go": True,
        "conformal_control_reuses_supplied_e24_inputs": True,
        "pointwise_or_uniform_v_to_one_does_not_control_curvature": True,
        "c2_source_action_and_field_equations_still_required": True,
    }
    alternatives = {
        "operational_fisher_geometry_with_cotransformed_design": True,
        "monotone_or_quantum_fisher_operational_state_metric": True,
        "kraus_refinement_invariant_record_algebra": True,
        "independent_lorentz_metric_volume_and_covariant_action": True,
        "causal_set_or_eps_continuum_bridge": True,
    }
    status = {
        "hellinger_factor_convention_certified": (
            abs(quadratic - float(np.sum(dpsi * dpsi))) <= numerical_limit
        ),
        "pullback_rank_kernel_certified": pullback_kernel_certified,
        "normalized_quotient_rank_eight_certified": normalized_quotient_certified,
        "signed_lift_kernel_fisher_invariance_certified": q_delta_invariant,
        "simultaneous_relabel_congruence_certified": simultaneous_relabel_certified,
        "chosen_general_relabel_is_not_fixed_incidence_automorphism": (
            general_relabel_fixed_incidence_residual > numerical_limit
        ),
        "context_weight_symmetry_boundary_certified": (
            context_weight_boundary_certified
        ),
        "atom_only_fixed_incidence_automorphism_excluded": (
            atom_only_residual > numerical_limit
            and atom_only_fixed_incidence_residual > numerical_limit
        ),
        "isotropic_fisher_chart_certified": isotropic_certified,
        "fisher_form_positive_semidefinite": inertia[1] == 0,
        "fisher_metric_is_spacetime_lorentz_metric_derived": False,
        "lorentzian_signature_or_lightcone_derived_from_fisher": False,
        "supplied_conformal_volume_algebra_certified": conformal_control_certified,
        "physical_volume_law_derived": False,
        "curvature_einstein_dynamics_or_gravity_derived": False,
        "gr_c2_limit_derived": False,
        "parent_fixed_context_cptp_modified": False,
        "relativistic_qft_microcausality_derived": False,
        "full_lightcone_no_controllable_influence_gate_complete": False,
        "independent_holdout_complete": False,
        "success_gates_1_to_8_complete": False,
    }

    return RepresentationInvariantMeasureCertificate(
        context_weights=weights,
        target_minimum_probability=float(target.min()),
        context_normalization_residual=normalization_residual,
        hellinger_coordinate_norm_squared=float(np.sum(psi * psi)),
        hellinger_quadratic_residual=abs(quadratic - float(np.sum(dpsi * dpsi))),
        incidence_rank=exact_rank,
        incidence_nullity=16 - exact_rank,
        pullback_rank=pullback_rank,
        pullback_inertia=inertia,
        normalized_tangent_rank=normalized_rank,
        normalized_tangent_inertia=normalized_inertia,
        maximum_incidence_kernel_residual=incidence_kernel_residual,
        maximum_pullback_kernel_residual=pullback_kernel_residual,
        q_delta_probability_residual=float(np.max(np.abs(delta_box - base_box))),
        q_delta_pullback_residual=float(np.max(np.abs(delta_metric - base_metric))),
        simultaneous_relabel_probability_residual=float(
            np.max(np.abs(relabelled_vector - expected_relabelled_vector))
        ),
        simultaneous_relabel_congruence_residual=float(
            np.max(np.abs(relabelled_metric - expected_congruence))
        ),
        general_relabel_fixed_incidence_residual=(
            general_relabel_fixed_incidence_residual
        ),
        fixed_nonuniform_context_swap_residual=abs(
            fixed_nonuniform_quadratic - original_nonuniform_quadratic
        ),
        co_transformed_context_swap_residual=abs(
            co_transformed_quadratic - original_nonuniform_quadratic
        ),
        uniform_context_swap_residual=abs(
            swapped_uniform_quadratic - original_uniform_quadratic
        ),
        atom_only_probability_residual=atom_only_residual,
        atom_only_fixed_incidence_residual=atom_only_fixed_incidence_residual,
        quantum_isotropic_component=isotropic_fisher_component(QUANTUM_ETA),
        quantum_isotropic_coordinate=math.asin(QUANTUM_ETA),
        analytic_quantum_distance=analytic_distance,
        product_quantum_distance=product_distance,
        isotropic_distance_residual=abs(product_distance - analytic_distance),
        reference_signature=lorentzian_signature(reference_metric),
        conformal_signature=lorentzian_signature(conformal_metric),
        supplied_volume_ratio=supplied_ratio,
        recovered_volume_ratio=recovered_ratio,
        conformal_volume_residual=abs(recovered_ratio - supplied_ratio),
        null_vector_reference_residual=abs(null_reference),
        null_vector_conformal_residual=abs(null_conformal),
        unit_volume_reference_residual=float(
            np.max(np.abs(unit_metric - reference_metric))
        ),
        high_frequency_uniform_residual_bound=(
            high_frequency.uniform_value_residual_bound
        ),
        high_frequency_second_derivative_magnitude=abs(
            high_frequency.probe_second_derivative
        ),
        dimensions=dimensions,
        accounting=accounting,
        boundaries=boundaries,
        alternatives=alternatives,
        status=status,
    )


def representation_run() -> dict[str, object]:
    """JSON 직렬화 가능한 E30 인증서를 돌려준다."""

    return asdict(representation_certificate())


def representation_main() -> None:
    """E30 인증서를 명령줄에서 출력한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(representation_run(), indent=args.indent, sort_keys=True))


def _coordinate_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    """유한한 좌표 열여섯 개로 이루어진 벡터인지 검사한다."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (ATOM_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be sixteen finite coordinates")
    return np.array(vector, copy=True)


def _row_map(values: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    """열이 열여섯 개인 비어 있지 않은 유한 행 사상인지 검사한다."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] != ATOM_COUNT
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite row map with sixteen columns")
    return np.array(matrix, copy=True)


def _numerical_rank(matrix: np.ndarray, *, tolerance: float = 1.0e-10) -> int:
    """특잇값 절단으로 수치 계수를 계산한다."""

    tol = _positive_tolerance(tolerance)
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("matrix must be a nonempty finite two-dimensional array")
    singular_values = np.linalg.svd(values, compute_uv=False)
    return int(np.sum(singular_values > tol))


def walsh_receipt_matrix() -> np.ndarray:
    """정준 무차원 E29 핵 행 일곱 개를 돌려준다."""

    return np.asarray(list(walsh_kernel_vectors().values()), dtype=np.int64)


def linear_source_kernel_residual(source: np.ndarray | Sequence[float]) -> float:
    """선형 원천이 월시 핵 행에 보이는 가장 큰 반응을 돌려준다."""

    source_map = _row_map(source, name="source")
    walsh = walsh_receipt_matrix().astype(np.float64)
    return float(np.max(np.abs(source_map @ walsh.T)))


def linear_source_factorization_residual(
    source: np.ndarray | Sequence[float],
) -> float:
    """최소 노름 주변 인자 ``L=A M``의 잔차를 돌려준다.

    돌려주는 주변 ``A``는 ``im(M)`` 위 유일한 사상의 확장 하나일 뿐이다. 잔차 0이 그 주변
    확장을 유일하게 만들지는 않는다.
    """

    source_map = _row_map(source, name="source")
    incidence = marginal_incidence_matrix().astype(np.float64)
    ambient_factor = source_map @ np.linalg.pinv(incidence)
    return float(np.max(np.abs(source_map - ambient_factor @ incidence)))


def factor_linear_source(
    source: np.ndarray | Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """``M``을 통해 내려오는 원천의 주변 인자 하나를 돌려준다."""

    tol = _positive_tolerance(tolerance)
    source_map = _row_map(source, name="source")
    incidence = marginal_incidence_matrix().astype(np.float64)
    ambient_factor = source_map @ np.linalg.pinv(incidence)
    residual = float(np.max(np.abs(source_map - ambient_factor @ incidence)))
    if residual > tol:
        raise ValueError("source is not constant on the canonical incidence fibres")
    return ambient_factor


def receipt_kernel_rank(
    receipt: np.ndarray | Sequence[float],
    *,
    tolerance: float = 1.0e-10,
) -> int:
    """정준 월시 기저에서 ``rank(E|ker M)``을 돌려준다."""

    receipt_map = _row_map(receipt, name="receipt")
    walsh = walsh_receipt_matrix().astype(np.float64)
    return _numerical_rank(receipt_map @ walsh.T, tolerance=tolerance)


def combined_readout_rank(
    receipt: np.ndarray | Sequence[float],
    *,
    normalized_tangent: bool = False,
) -> int:
    """``[M;E]``의 계수를 돌려주며, 선택적으로 정규화 접공간 위에서 계산한다."""

    receipt_map = _row_map(receipt, name="receipt")
    incidence = marginal_incidence_matrix().astype(np.float64)
    combined = np.vstack((incidence, receipt_map))
    if normalized_tangent:
        combined = combined @ normalized_atom_tangent_basis()
    return exact_rational_rank(combined)


def visible_and_walsh_receipt(
    coordinates: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """가시 벡터와 진단용 월시 좌표 일곱 개를 돌려준다."""

    vector = _coordinate_vector(coordinates, name="coordinates")
    incidence = marginal_incidence_matrix().astype(np.float64)
    walsh = walsh_receipt_matrix().astype(np.float64)
    return incidence @ vector, walsh @ vector


def reconstruct_from_visible_and_walsh(
    visible: Sequence[float],
    receipt: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """정합적인 정준 ``(Mq, Wq)`` 좌표에서 ``q``를 되찾는다.

    무어--펜로즈(Moore--Penrose) 단면은 좌표 규약이지 물리적 선택 법칙이 아니다. 월시
    직교성이 핵 성분을 공급한다.
    """

    tol = _positive_tolerance(tolerance)
    visible_vector = _coordinate_vector(visible, name="visible")
    receipt_vector = np.asarray(receipt, dtype=np.float64)
    if receipt_vector.shape != (7,) or not np.isfinite(receipt_vector).all():
        raise ValueError("receipt must be seven finite Walsh coordinates")
    incidence = marginal_incidence_matrix().astype(np.float64)
    walsh = walsh_receipt_matrix().astype(np.float64)
    coordinates = (
        np.linalg.pinv(incidence) @ visible_vector
        + walsh.T @ receipt_vector / 16.0
    )
    visible_residual = float(
        np.max(np.abs(incidence @ coordinates - visible_vector))
    )
    receipt_residual = float(np.max(np.abs(walsh @ coordinates - receipt_vector)))
    if visible_residual > tol or receipt_residual > tol:
        raise ValueError("visible and receipt coordinates are inconsistent with the canonical maps")
    return coordinates


@dataclass(frozen=True)
class HiddenSourceFactorizationCertificate:
    """E31 유한 인수분해·영수증 계수 인증서다."""

    incidence_rank: int
    incidence_nullity: int
    walsh_rank: int
    maximum_incidence_walsh_residual: float
    maximum_walsh_gram_residual: float
    combined_rank_one_receipt: int
    combined_rank_six_receipts: int
    combined_rank_seven_receipts: int
    normalized_visible_rank: int
    normalized_combined_rank_one_receipt: int
    normalized_combined_rank_six_receipts: int
    normalized_combined_rank_seven_receipts: int
    minimum_receipt_rows_for_full_recovery: int
    duplicate_receipt_kernel_rank: int
    duplicate_receipt_combined_rank: int
    visible_source_kernel_residual: float
    visible_source_factorization_residual: float
    hidden_source_kernel_residual: float
    hidden_source_factorization_residual: float
    ambient_factor_extension_difference: float
    alternative_ambient_factor_residual: float
    q_delta_visible_residual: float
    q_delta_first_walsh_change: float
    q_delta_other_walsh_residual: float
    permutation_norm_residual: float
    same_fibre_norm_square_difference: float
    reconstruction_coordinate_residual: float
    reconstruction_visible_residual: float
    reconstruction_receipt_residual: float
    relabel_visible_residual: float
    relabel_receipt_residual: float
    relabel_combined_rank: int
    relabel_fixed_incidence_residual: float
    relabel_fixed_receipt_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]

    def to_json(self, *, indent: int | None = 2) -> str:
        """인증서를 JSON 문자열로 직렬화한다."""

        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> HiddenSourceFactorizationCertificate:
    """E31 유한 인수분해·영수증 계수 인증서를 만든다."""

    tol = _positive_tolerance(tolerance)
    numerical_limit = 100.0 * tol
    incidence_integer = marginal_incidence_matrix()
    incidence = incidence_integer.astype(np.float64)
    walsh_integer = walsh_receipt_matrix()
    walsh = walsh_integer.astype(np.float64)

    incidence_rank = exact_rational_rank(incidence_integer)
    walsh_rank = exact_rational_rank(walsh_integer)
    incidence_walsh_residual = float(np.max(np.abs(incidence @ walsh.T)))
    walsh_gram_residual = float(
        np.max(np.abs(walsh @ walsh.T - 16.0 * np.eye(7)))
    )
    combined_ranks = {
        count: exact_rational_rank(np.vstack((incidence_integer, walsh_integer[:count])))
        for count in (1, 6, 7)
    }
    tangent = normalized_atom_tangent_basis()
    normalized_visible_rank = exact_rational_rank(incidence_integer @ tangent)
    normalized_combined_ranks = {
        count: exact_rational_rank(
            np.vstack((incidence_integer, walsh_integer[:count])) @ tangent
        )
        for count in (1, 6, 7)
    }

    duplicate_receipt = incidence[:7]
    duplicate_kernel_rank = receipt_kernel_rank(duplicate_receipt)
    duplicate_combined_rank = combined_readout_rank(duplicate_receipt)

    visible_source = incidence[[0, 5]]
    visible_kernel_residual = linear_source_kernel_residual(visible_source)
    visible_factorization_residual = linear_source_factorization_residual(
        visible_source
    )
    minimum_norm_factor = factor_linear_source(visible_source, tolerance=tol)
    _, _, transpose_vh = np.linalg.svd(incidence.T)
    left_null_row = transpose_vh[incidence_rank]
    alternative_factor = np.array(minimum_norm_factor, copy=True)
    alternative_factor[0] += left_null_row
    alternative_factor_residual = float(
        np.max(np.abs(visible_source - alternative_factor @ incidence))
    )
    ambient_factor_difference = float(
        np.max(np.abs(alternative_factor - minimum_norm_factor))
    )

    hidden_source = walsh[[0]]
    hidden_kernel_residual = linear_source_kernel_residual(hidden_source)
    hidden_factorization_residual = linear_source_factorization_residual(hidden_source)

    base_q = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    delta_q = np.asarray(quantum_kernel_perturbed_extension(0.1))
    base_visible, base_receipt = visible_and_walsh_receipt(base_q)
    delta_visible, delta_receipt = visible_and_walsh_receipt(delta_q)
    q_delta_visible_residual = float(np.max(np.abs(delta_visible - base_visible)))
    receipt_change = delta_receipt - base_receipt
    q_delta_first_walsh_change = float(receipt_change[0])
    q_delta_other_walsh_residual = float(np.max(np.abs(receipt_change[1:])))

    atom_permutation = atom_permutation_matrix(tuple(reversed(range(ATOM_COUNT))))
    permutation_norm_residual = abs(
        float(np.dot(atom_permutation @ base_q, atom_permutation @ base_q))
        - float(np.dot(base_q, base_q))
    )
    fibre_norm_difference = abs(
        float(np.dot(delta_q, delta_q)) - float(np.dot(base_q, base_q))
    )

    reconstructed = reconstruct_from_visible_and_walsh(
        delta_visible, delta_receipt, tolerance=tol
    )
    reconstructed_visible, reconstructed_receipt = visible_and_walsh_receipt(
        reconstructed
    )
    reconstruction_coordinate_residual = float(
        np.max(np.abs(reconstructed - delta_q))
    )
    reconstruction_visible_residual = float(
        np.max(np.abs(reconstructed_visible - delta_visible))
    )
    reconstruction_receipt_residual = float(
        np.max(np.abs(reconstructed_receipt - delta_receipt))
    )

    context_order = (1, 0, 2, 3)
    row_permutation = context_block_permutation(context_order)
    receipt_permutation = np.eye(7, dtype=np.float64)[::-1]
    relabelled_q = atom_permutation @ delta_q
    relabelled_incidence = row_permutation @ incidence @ atom_permutation.T
    relabelled_receipt_map = receipt_permutation @ walsh @ atom_permutation.T
    relabel_visible_residual = float(
        np.max(
            np.abs(
                relabelled_incidence @ relabelled_q
                - row_permutation @ delta_visible
            )
        )
    )
    relabel_receipt_residual = float(
        np.max(
            np.abs(
                relabelled_receipt_map @ relabelled_q
                - receipt_permutation @ delta_receipt
            )
        )
    )
    relabel_combined_rank = exact_rational_rank(
        np.vstack((relabelled_incidence, relabelled_receipt_map))
    )
    relabel_fixed_incidence_residual = float(
        np.max(np.abs(relabelled_incidence - incidence))
    )
    relabel_fixed_receipt_residual = float(
        np.max(np.abs(relabelled_receipt_map - walsh))
    )

    dimensions = {
        "q_visible_probabilities_and_walsh_coordinates_dimensionless": True,
        "norms_ranks_nullities_and_residuals_dimensionless": True,
        "dimensionful_receipts_require_fixed_reference_scales_before_rank_comparison": True,
        "finite_coordinate_rank_is_not_spacetime_dimension": True,
    }
    accounting = {
        "duplicate_receipt_factors_through_visible_map": duplicate_combined_rank == 9,
        "duplicate_receipt_not_added_as_new_source": True,
        "signed_q_and_walsh_coordinates_not_booked_as_energy_or_stress": True,
        "receipt_probability_energy_or_volume_double_counted": False,
    }
    boundaries = {
        "fibre_constancy_is_required_for_visible_source_factorization": True,
        "permutation_covariance_does_not_imply_fibre_constancy": True,
        "seven_rows_are_necessary_only_for_full_linear_q_recovery": True,
        "seven_rows_are_not_sufficient_without_kernel_rank_seven": True,
        "seven_is_not_a_gravity_component_field_or_boson_count": True,
        "ambient_factor_extension_is_not_unique": True,
        "walsh_receipts_are_canonical_coordinate_witnesses_only": True,
        "general_relabel_covariance_is_not_fixed_map_automorphism": True,
        "physical_receipt_provenance_and_dynamics_are_not_supplied": True,
        "full_finite_signed_coordinate_is_not_a_physical_ontology": True,
    }
    alternatives = {
        "visible_quotient_source_only": True,
        "independent_operational_receipt_with_provenance": True,
        "hidden_ontology_with_local_covariant_action": True,
        "entanglement_linearized_einstein_dictionary": True,
        "causal_order_plus_independent_volume_dictionary": True,
    }
    status = {
        "canonical_rank_nine_nullity_seven_reused": (
            incidence_rank == 9 and ATOM_COUNT - incidence_rank == 7
        ),
        "walsh_kernel_orthogonality_certified": (
            walsh_rank == 7
            and incidence_walsh_residual <= numerical_limit
            and walsh_gram_residual <= numerical_limit
        ),
        "receipt_rank_lower_bound_witness_certified": (
            combined_ranks == {1: 10, 6: 15, 7: 16}
            and normalized_combined_ranks == {1: 9, 6: 14, 7: 15}
        ),
        "full_walsh_coordinate_reconstruction_certified": (
            reconstruction_coordinate_residual <= numerical_limit
            and reconstruction_visible_residual <= numerical_limit
            and reconstruction_receipt_residual <= numerical_limit
        ),
        "linear_visible_source_factorization_certified": (
            visible_kernel_residual <= numerical_limit
            and visible_factorization_residual <= numerical_limit
        ),
        "ambient_factor_extension_nonuniqueness_certified": (
            alternative_factor_residual <= numerical_limit
            and ambient_factor_difference > numerical_limit
        ),
        "permutation_covariance_not_fibre_invariance_certified": (
            q_delta_visible_residual <= numerical_limit
            and permutation_norm_residual <= numerical_limit
            and fibre_norm_difference > numerical_limit
        ),
        "duplicate_visible_receipt_adds_no_rank_certified": (
            duplicate_kernel_rank == 0 and duplicate_combined_rank == 9
        ),
        "general_relabel_covariance_certified": (
            relabel_visible_residual <= numerical_limit
            and relabel_receipt_residual <= numerical_limit
            and relabel_combined_rank == 16
        ),
        "chosen_general_relabel_is_fixed_map_automorphism": (
            relabel_fixed_incidence_residual <= numerical_limit
            and relabel_fixed_receipt_residual <= numerical_limit
        ),
        "physical_walsh_receipt_derived": False,
        "hidden_signed_coordinate_is_physical_state_derived": False,
        "local_covariant_action_or_stress_derived": False,
        "spacetime_metric_curvature_or_gravity_derived": False,
        "objective_selection_derived": False,
        "relativistic_qft_microcausality_derived": False,
        "full_lightcone_no_controllable_influence_gate_complete": False,
        "independent_holdout_complete": False,
        "success_gates_1_to_8_complete": False,
    }

    return HiddenSourceFactorizationCertificate(
        incidence_rank=incidence_rank,
        incidence_nullity=ATOM_COUNT - incidence_rank,
        walsh_rank=walsh_rank,
        maximum_incidence_walsh_residual=incidence_walsh_residual,
        maximum_walsh_gram_residual=walsh_gram_residual,
        combined_rank_one_receipt=combined_ranks[1],
        combined_rank_six_receipts=combined_ranks[6],
        combined_rank_seven_receipts=combined_ranks[7],
        normalized_visible_rank=normalized_visible_rank,
        normalized_combined_rank_one_receipt=normalized_combined_ranks[1],
        normalized_combined_rank_six_receipts=normalized_combined_ranks[6],
        normalized_combined_rank_seven_receipts=normalized_combined_ranks[7],
        minimum_receipt_rows_for_full_recovery=ATOM_COUNT - incidence_rank,
        duplicate_receipt_kernel_rank=duplicate_kernel_rank,
        duplicate_receipt_combined_rank=duplicate_combined_rank,
        visible_source_kernel_residual=visible_kernel_residual,
        visible_source_factorization_residual=visible_factorization_residual,
        hidden_source_kernel_residual=hidden_kernel_residual,
        hidden_source_factorization_residual=hidden_factorization_residual,
        ambient_factor_extension_difference=ambient_factor_difference,
        alternative_ambient_factor_residual=alternative_factor_residual,
        q_delta_visible_residual=q_delta_visible_residual,
        q_delta_first_walsh_change=q_delta_first_walsh_change,
        q_delta_other_walsh_residual=q_delta_other_walsh_residual,
        permutation_norm_residual=permutation_norm_residual,
        same_fibre_norm_square_difference=fibre_norm_difference,
        reconstruction_coordinate_residual=reconstruction_coordinate_residual,
        reconstruction_visible_residual=reconstruction_visible_residual,
        reconstruction_receipt_residual=reconstruction_receipt_residual,
        relabel_visible_residual=relabel_visible_residual,
        relabel_receipt_residual=relabel_receipt_residual,
        relabel_combined_rank=relabel_combined_rank,
        relabel_fixed_incidence_residual=relabel_fixed_incidence_residual,
        relabel_fixed_receipt_residual=relabel_fixed_receipt_residual,
        dimensions=dimensions,
        accounting=accounting,
        boundaries=boundaries,
        alternatives=alternatives,
        status=status,
    )


def run() -> dict[str, object]:
    """JSON 직렬화 가능한 E31 인증서를 돌려준다."""

    return asdict(certificate())


def main() -> None:
    """E31 인증서를 명령줄에서 출력한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(), indent=args.indent, sort_keys=True))


def _finite(value: float, name: str) -> float:
    """유한한 값만 통과시킨다."""

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    """유한하고 양수인 값만 통과시킨다."""

    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def minkowski_metric() -> np.ndarray:
    """한 정규직교 틀에서 공급된 ``(-,+,+,+)`` 계량을 돌려준다."""

    return np.diag((-1, 1, 1, 1)).astype(np.int64)


def symmetric_tensor_basis() -> tuple[np.ndarray, ...]:
    """``COMPONENT_ORDER`` 순서의 정수 기저를 돌려준다."""

    basis: list[np.ndarray] = []
    for first in range(SPACETIME_DIMENSION):
        for second in range(first, SPACETIME_DIMENSION):
            tensor = np.zeros(
                (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
                dtype=np.int64,
            )
            tensor[first, second] = 1
            tensor[second, first] = 1
            basis.append(tensor)
    return tuple(basis)


def lorentz_generators() -> dict[str, np.ndarray]:
    """회전 셋과 부스트 셋을 정확한 정수 행렬로 돌려준다."""

    generators: dict[str, np.ndarray] = {}
    for first, second in ((1, 2), (1, 3), (2, 3)):
        generator = np.zeros((4, 4), dtype=np.int64)
        generator[first, second] = 1
        generator[second, first] = -1
        generators[f"J{first}{second}"] = generator
    for spatial in (1, 2, 3):
        generator = np.zeros((4, 4), dtype=np.int64)
        generator[0, spatial] = 1
        generator[spatial, 0] = 1
        generators[f"K0{spatial}"] = generator
    return generators


def infinitesimal_invariance_constraint(
    generator_names: Iterable[str],
) -> np.ndarray:
    """``X.T @ T + T @ X = 0``의 정확한 계수를 만든다.

    독립 대칭 성분마다 생성자당 한 행을 기여한다. 인증서가 쓰는 불변량은 맨 행 수가 아니라
    정확한 계수이므로 영 행도 그대로 둔다.
    """

    generators = lorentz_generators()
    basis = symmetric_tensor_basis()
    rows: list[list[int]] = []
    for name in generator_names:
        if name not in generators:
            raise ValueError(f"unknown Lorentz generator: {name}")
        generator = generators[name]
        variations = tuple(
            generator.T @ tensor + tensor @ generator for tensor in basis
        )
        for first in range(SPACETIME_DIMENSION):
            for second in range(first, SPACETIME_DIMENSION):
                rows.append(
                    [
                        int(variation[first, second])
                        for variation in variations
                    ]
                )
    if not rows:
        raise ValueError("at least one Lorentz generator is required")
    return np.asarray(rows, dtype=np.int64)


def tensor_from_components(components: Iterable[float]) -> np.ndarray:
    """정준 성분 열 개로 대칭 텐서를 만든다."""

    values = np.asarray(tuple(components), dtype=np.float64)
    if values.shape != (SYMMETRIC_COMPONENT_COUNT,) or not np.isfinite(values).all():
        raise ValueError("components must contain ten finite values")
    tensor = sum(
        (value * basis for value, basis in zip(values, symmetric_tensor_basis())),
        start=np.zeros((4, 4), dtype=np.float64),
    )
    return tensor


@dataclass(frozen=True)
class LorentzNaturalTensorCertificate:
    """0차 스칼라 전용 로런츠 불변 부분공간의 인증서다."""

    symmetric_tensor_dimension: int
    rotation_constraint_shape: tuple[int, int]
    rotation_constraint_rank: int
    rotation_invariant_nullity: int
    full_lorentz_constraint_shape: tuple[int, int]
    full_lorentz_constraint_rank: int
    full_lorentz_invariant_nullity: int
    metric_generator_residual: int
    rotation_time_basis_residual: int
    rotation_spatial_basis_residual: int
    full_metric_span_unique: bool
    order_zero_scalar_source_form: str


def lorentz_natural_tensor_certificate() -> LorentzNaturalTensorCertificate:
    """0차 스칼라 전용 로런츠 불변 부분공간을 인증한다."""

    rotations = ("J12", "J13", "J23")
    boosts = ("K01", "K02", "K03")
    rotation_constraint = infinitesimal_invariance_constraint(rotations)
    full_constraint = infinitesimal_invariance_constraint(rotations + boosts)
    rotation_rank = exact_rational_rank(rotation_constraint)
    full_rank = exact_rational_rank(full_constraint)
    metric_components = np.asarray(
        (-1, 0, 0, 0, 1, 0, 0, 1, 0, 1),
        dtype=np.int64,
    )
    time_components = np.asarray(
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        dtype=np.int64,
    )
    spatial_components = np.asarray(
        (0, 0, 0, 0, 1, 0, 0, 1, 0, 1),
        dtype=np.int64,
    )
    metric_residual = int(np.max(np.abs(full_constraint @ metric_components)))
    time_residual = int(
        np.max(np.abs(rotation_constraint @ time_components))
    )
    spatial_residual = int(
        np.max(np.abs(rotation_constraint @ spatial_components))
    )
    rotation_nullity = SYMMETRIC_COMPONENT_COUNT - rotation_rank
    full_nullity = SYMMETRIC_COMPONENT_COUNT - full_rank
    return LorentzNaturalTensorCertificate(
        symmetric_tensor_dimension=SYMMETRIC_COMPONENT_COUNT,
        rotation_constraint_shape=tuple(rotation_constraint.shape),
        rotation_constraint_rank=rotation_rank,
        rotation_invariant_nullity=rotation_nullity,
        full_lorentz_constraint_shape=tuple(full_constraint.shape),
        full_lorentz_constraint_rank=full_rank,
        full_lorentz_invariant_nullity=full_nullity,
        metric_generator_residual=metric_residual,
        rotation_time_basis_residual=time_residual,
        rotation_spatial_basis_residual=spatial_residual,
        full_metric_span_unique=(
            full_nullity == 1
            and metric_residual == 0
            and rotation_nullity == 2
            and time_residual == 0
            and spatial_residual == 0
        ),
        order_zero_scalar_source_form="T_mn = C(r) g_mn",
    )


@dataclass(frozen=True)
class VacuumFormReceipt:
    """공급된 틀의 정지 관측자에 대해 ``T=C g``를 평가한 영수증이다."""

    coefficient: float
    stress_covariant: tuple[tuple[float, ...], ...]
    energy_density: float
    isotropic_pressure: float
    equation_of_state: float | None


def vacuum_form_receipt(coefficient: float) -> VacuumFormReceipt:
    """공급된 틀의 정지 관측자에 대해 ``T=C g``를 평가한다."""

    scalar = _finite(coefficient, "coefficient")
    stress = scalar * minkowski_metric().astype(np.float64)
    energy_density = float(stress[0, 0])
    pressure = float(np.trace(stress[1:, 1:]) / 3.0)
    equation_of_state = (
        None if energy_density == 0.0 else pressure / energy_density
    )
    return VacuumFormReceipt(
        coefficient=scalar,
        stress_covariant=tuple(tuple(float(item) for item in row) for row in stress),
        energy_density=energy_density,
        isotropic_pressure=pressure,
        equation_of_state=equation_of_state,
    )


@dataclass(frozen=True)
class AdditiveActionCountermodel:
    """같은 영수증에 다른 응력을 주는 두 공변 작용의 반모형(countermodel)이다."""

    receipt_value: float
    reference_mass_scale: float
    scalar_mass: float
    hidden_action_coefficient: float
    constant_field_value: float
    zero_additive_density: float
    nonzero_additive_density: float
    zero_source_stress_covariant: tuple[tuple[float, ...], ...]
    nonzero_source_stress_covariant: tuple[tuple[float, ...], ...]
    normalized_stress_difference: float
    scalar_eom_difference: float
    principal_symbol_difference: float
    on_shell_eom_residual: float
    on_shell_divergence_residual: float
    zero_coefficient_hidden_stress_residual: float
    zero_coefficient_hidden_eom_coefficient: float
    same_operational_receipt_without_action_normalization: bool
    same_constant_on_shell_field: bool
    same_scalar_eom_for_positive_coefficient: bool
    same_principal_symbol_for_positive_coefficient: bool
    both_stresses_conserved_on_shell: bool
    finite_coefficient_metric_sources_distinct: bool
    additive_source_selected_by_receipt: bool
    zero_coefficient_hidden_metric_source_vanishes: bool
    metric_solution_convergence_derived: bool


def canonical_scalar_potential(
    field_value: float,
    *,
    field_minimum: float,
    scalar_mass: float,
    additive_density: float,
) -> float:
    """선언된 질량 단위 하나로 ``m^2 (phi-phi0)^2 / 2 + a``를 돌려준다."""

    field = _finite(field_value, "field_value")
    minimum = _finite(field_minimum, "field_minimum")
    mass = _positive(scalar_mass, "scalar_mass")
    additive = _finite(additive_density, "additive_density")
    return 0.5 * mass**2 * (field - minimum) ** 2 + additive


def canonical_scalar_potential_derivative(
    field_value: float,
    *,
    field_minimum: float,
    scalar_mass: float,
) -> float:
    """정준 퍼텐셜의 정확한 1차 장 미분을 돌려준다."""

    field = _finite(field_value, "field_value")
    minimum = _finite(field_minimum, "field_minimum")
    mass = _positive(scalar_mass, "scalar_mass")
    return mass**2 * (field - minimum)


def canonical_scalar_eom(
    field_value: float,
    *,
    box_field: float,
    field_minimum: float,
    scalar_mass: float,
    hidden_action_coefficient: float,
) -> float:
    """공급된 작용에 대한 ``epsilon [box(phi)-V'(phi)]``를 돌려준다."""

    epsilon = _finite(
        hidden_action_coefficient,
        "hidden_action_coefficient",
    )
    if epsilon < 0.0:
        raise ValueError("hidden_action_coefficient must be nonnegative")
    return epsilon * (
        _finite(box_field, "box_field")
        - canonical_scalar_potential_derivative(
            field_value,
            field_minimum=field_minimum,
            scalar_mass=scalar_mass,
        )
    )


def canonical_scalar_principal_coefficient(
    hidden_action_coefficient: float,
) -> float:
    """``g^mn d_m d_n``에 곱해지는 스칼라 계수를 돌려준다."""

    epsilon = _finite(
        hidden_action_coefficient,
        "hidden_action_coefficient",
    )
    if epsilon < 0.0:
        raise ValueError("hidden_action_coefficient must be nonnegative")
    return epsilon


def canonical_scalar_stress_at_flat_point(
    field_value: float,
    *,
    gradient_covector: Iterable[float],
    field_minimum: float,
    scalar_mass: float,
    additive_density: float,
    hidden_action_coefficient: float,
) -> np.ndarray:
    """공급된 틀에서 정준 힐베르트 응력(Hilbert stress) 공식을 평가한다."""

    epsilon = canonical_scalar_principal_coefficient(
        hidden_action_coefficient
    )
    gradient = np.asarray(tuple(gradient_covector), dtype=np.float64)
    if gradient.shape != (4,) or not np.isfinite(gradient).all():
        raise ValueError("gradient_covector must contain four finite values")
    metric = minkowski_metric().astype(np.float64)
    inverse_metric = metric
    gradient_square = float(gradient @ inverse_metric @ gradient)
    potential = canonical_scalar_potential(
        field_value,
        field_minimum=field_minimum,
        scalar_mass=scalar_mass,
        additive_density=additive_density,
    )
    return epsilon * (
        np.outer(gradient, gradient)
        - metric * (0.5 * gradient_square + potential)
    )


def canonical_scalar_ward_divergence(
    field_eom: float,
    *,
    gradient_covector: Iterable[float],
) -> np.ndarray:
    """온셸 워드(Ward) 인자 ``E_phi * d_n phi``를 돌려준다."""

    eom = _finite(field_eom, "field_eom")
    gradient = np.asarray(tuple(gradient_covector), dtype=np.float64)
    if gradient.shape != (4,) or not np.isfinite(gradient).all():
        raise ValueError("gradient_covector must contain four finite values")
    return eom * gradient


def additive_action_countermodel(
    *,
    receipt_value: float = 0.375,
    reference_mass_scale: float = 2.0,
    scalar_mass: float = 1.5,
    hidden_action_coefficient: float = 0.25,
) -> AdditiveActionCountermodel:
    """영수증은 하나이지만 응력이 다른 두 공변 작용을 돌려준다.

    ``hidden_action_coefficient``는 전체 무차원 ``epsilon``이다. 비최소 곡률 결합이 아니다.
    비교는 고정된 양의 epsilon에 대해서만 한다. epsilon 0에서는 숨은 방정식 자체가 사라지며,
    공급된 계량 방정식으로부터의 분리(decoupling)만 주장한다.
    """

    receipt = _finite(receipt_value, "receipt_value")
    mass_scale = _positive(reference_mass_scale, "reference_mass_scale")
    mass = _positive(scalar_mass, "scalar_mass")
    epsilon = _positive(
        hidden_action_coefficient,
        "hidden_action_coefficient",
    )
    field_value = mass_scale * receipt
    additive_zero = 0.0
    additive_nonzero = mass_scale**4
    constant_gradient = np.zeros(4, dtype=np.float64)
    stress_zero = canonical_scalar_stress_at_flat_point(
        field_value,
        gradient_covector=constant_gradient,
        field_minimum=field_value,
        scalar_mass=mass,
        additive_density=additive_zero,
        hidden_action_coefficient=epsilon,
    )
    stress_nonzero = canonical_scalar_stress_at_flat_point(
        field_value,
        gradient_covector=constant_gradient,
        field_minimum=field_value,
        scalar_mass=mass,
        additive_density=additive_nonzero,
        hidden_action_coefficient=epsilon,
    )
    stress_scale = epsilon * mass_scale**4
    stress_difference = float(
        np.max(np.abs(stress_nonzero - stress_zero)) / stress_scale
    )

    # 가법 상수의 장 미분은 0이다. 공유 해를 검사하기 전에 두 가지 공식을 공통 오프셸 탐침점에서
    # 평가한다.
    probe_field = field_value + 0.25 * mass_scale
    probe_box = 0.125 * mass_scale**3
    eom_zero_probe = canonical_scalar_eom(
        probe_field,
        box_field=probe_box,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    eom_nonzero_probe = canonical_scalar_eom(
        probe_field,
        box_field=probe_box,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    scalar_eom_difference = abs(eom_zero_probe - eom_nonzero_probe) / max(
        epsilon * mass_scale**3,
        1.0,
    )
    principal_zero = canonical_scalar_principal_coefficient(epsilon)
    principal_nonzero = canonical_scalar_principal_coefficient(epsilon)
    principal_symbol_difference = abs(principal_zero - principal_nonzero)
    eom_zero_on_shell = canonical_scalar_eom(
        field_value,
        box_field=0.0,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    eom_nonzero_on_shell = canonical_scalar_eom(
        field_value,
        box_field=0.0,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    on_shell_eom_residual = max(
        abs(eom_zero_on_shell),
        abs(eom_nonzero_on_shell),
    ) / max(epsilon * mass_scale**3, 1.0)
    divergence_zero = canonical_scalar_ward_divergence(
        eom_zero_on_shell,
        gradient_covector=constant_gradient,
    )
    divergence_nonzero = canonical_scalar_ward_divergence(
        eom_nonzero_on_shell,
        gradient_covector=constant_gradient,
    )
    on_shell_divergence_residual = max(
        float(np.max(np.abs(divergence_zero))),
        float(np.max(np.abs(divergence_nonzero))),
    ) / max(epsilon * mass_scale**5, 1.0)
    zero_epsilon_stress = canonical_scalar_stress_at_flat_point(
        field_value,
        gradient_covector=constant_gradient,
        field_minimum=field_value,
        scalar_mass=mass,
        additive_density=additive_nonzero,
        hidden_action_coefficient=0.0,
    )
    zero_coefficient_hidden_stress_residual = float(
        np.max(np.abs(zero_epsilon_stress))
    )
    zero_coefficient_hidden_eom_coefficient = (
        canonical_scalar_principal_coefficient(0.0)
    )
    receipt_zero_branch = (
        receipt,
        mass_scale,
        field_value,
        tuple(float(item) for item in constant_gradient),
        0.0,
    )
    receipt_nonzero_branch = (
        receipt,
        mass_scale,
        field_value,
        tuple(float(item) for item in constant_gradient),
        0.0,
    )
    same_receipt = receipt_zero_branch == receipt_nonzero_branch
    same_field = on_shell_eom_residual <= DEFAULT_TOLERANCE
    same_eom = scalar_eom_difference <= DEFAULT_TOLERANCE
    same_principal = principal_symbol_difference <= DEFAULT_TOLERANCE
    conserved = on_shell_divergence_residual <= DEFAULT_TOLERANCE
    distinct = stress_difference > DEFAULT_TOLERANCE
    return AdditiveActionCountermodel(
        receipt_value=receipt,
        reference_mass_scale=mass_scale,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
        constant_field_value=field_value,
        zero_additive_density=additive_zero,
        nonzero_additive_density=additive_nonzero,
        zero_source_stress_covariant=tuple(
            tuple(float(item) for item in row) for row in stress_zero
        ),
        nonzero_source_stress_covariant=tuple(
            tuple(float(item) for item in row) for row in stress_nonzero
        ),
        normalized_stress_difference=stress_difference,
        scalar_eom_difference=scalar_eom_difference,
        principal_symbol_difference=principal_symbol_difference,
        on_shell_eom_residual=on_shell_eom_residual,
        on_shell_divergence_residual=on_shell_divergence_residual,
        zero_coefficient_hidden_stress_residual=(
            zero_coefficient_hidden_stress_residual
        ),
        zero_coefficient_hidden_eom_coefficient=(
            zero_coefficient_hidden_eom_coefficient
        ),
        same_operational_receipt_without_action_normalization=same_receipt,
        same_constant_on_shell_field=same_field,
        same_scalar_eom_for_positive_coefficient=same_eom,
        same_principal_symbol_for_positive_coefficient=same_principal,
        both_stresses_conserved_on_shell=conserved,
        finite_coefficient_metric_sources_distinct=distinct,
        additive_source_selected_by_receipt=not (
            same_receipt
            and same_field
            and same_eom
            and same_principal
            and conserved
            and distinct
        ),
        zero_coefficient_hidden_metric_source_vanishes=(
            zero_coefficient_hidden_stress_residual <= DEFAULT_TOLERANCE
        ),
        metric_solution_convergence_derived=False,
    )


@dataclass(frozen=True)
class SourceAccountingReceipt:
    """상호 배타적인 원천 회계(source accounting) 방식 하나의 영수증이다."""

    mode: str
    retained_hidden_stress_added: bool
    integrated_out_influence_response_added: bool
    rn_probability_reweighting_added_as_energy: bool
    rank_or_volume_added_as_energy: bool
    mutually_exclusive_source_accounting: bool
    declared_no_probability_energy_rebooking: bool


def source_accounting_receipt(mode: str) -> SourceAccountingReceipt:
    """배타적 원천 회계 방식 하나를 돌려주거나 요청을 거부한다."""

    if mode not in {
        "retained_hidden_field",
        "integrated_out_influence",
        "receipt_only_no_source",
    }:
        raise ValueError("unknown source accounting mode")
    retained = mode == "retained_hidden_field"
    influence = mode == "integrated_out_influence"
    neither = mode == "receipt_only_no_source"
    exclusive = int(retained) + int(influence) <= 1
    no_rebooking_declared = exclusive and (retained or influence or neither)
    return SourceAccountingReceipt(
        mode=mode,
        retained_hidden_stress_added=retained,
        integrated_out_influence_response_added=influence,
        rn_probability_reweighting_added_as_energy=False,
        rank_or_volume_added_as_energy=False,
        mutually_exclusive_source_accounting=exclusive,
        declared_no_probability_energy_rebooking=no_rebooking_declared,
    )


@dataclass(frozen=True)
class ScalarReceiptSourceAdmissionCertificate:
    """E32 유한 스칼라 영수증 원천 허용 인증서다."""

    lorentz_tensor: LorentzNaturalTensorCertificate
    positive_vacuum_form: VacuumFormReceipt
    action_countermodel: AdditiveActionCountermodel
    source_accounting: SourceAccountingReceipt
    e31_full_receipt_combined_rank: int
    e31_receipt_kernel_rank: int
    e31_rank_complete_receipt: bool
    receipt_mass_dimension: int
    metric_mass_dimension: int
    reference_scale_mass_dimension: int
    scalar_field_mass_dimension: int
    scalar_mass_dimension: int
    derivative_mass_dimension: int
    potential_mass_dimension: int
    stress_mass_dimension: int
    action_density_mass_dimension: int
    volume_element_mass_dimension: int
    action_mass_dimension: int
    hidden_action_coefficient_mass_dimension: int
    dimensions_pass: bool
    rank_complete_receipt_selects_physical_source: bool
    scalar_only_order_zero_source_is_vacuum_form: bool
    dust_source_derived: bool
    current_gradient_or_kinetic_data_required_for_dust: bool
    local_receipt_to_field_map_derived: bool
    supplied_metric_derived_from_receipt: bool
    metric_variation_machine_verified: bool
    conditional_ward_theorem_replaced_by_numerics: bool
    cptp_quantum_dynamics_derived: bool
    qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    finite_coefficient_gr_phenomenology_derived: bool
    independent_holdout_prediction_derived: bool
    two_residual_classes_reduced: bool
    complexity_penalty_success: bool


def scalar_receipt_certificate() -> ScalarReceiptSourceAdmissionCertificate:
    """정준 E32 유한 허용 인증서를 만든다."""

    lorentz = lorentz_natural_tensor_certificate()
    # 양의 진공 밀도 V=M_*^4 는 T=Cg 에서 C=-V 에 대응한다.
    vacuum = vacuum_form_receipt(-16.0)
    countermodel = additive_action_countermodel()
    accounting = source_accounting_receipt("retained_hidden_field")
    walsh = walsh_receipt_matrix()
    combined_rank = combined_readout_rank(walsh)
    hidden_rank = exact_rational_rank(walsh @ walsh.T)
    rank_complete = combined_rank == 16 and hidden_rank == 7
    ambiguity = not countermodel.additive_source_selected_by_receipt

    receipt_dim = 0
    metric_dim = 0
    scale_dim = 1
    field_dim = 1
    mass_dim = 1
    derivative_dim = 1
    potential_dim = 4
    stress_dim = 4
    action_density_dim = 4
    volume_dim = -4
    action_dim = 0
    epsilon_dim = 0
    dimensions_pass = (
        receipt_dim == metric_dim == epsilon_dim == action_dim == 0
        and scale_dim == field_dim == mass_dim == derivative_dim == 1
        and potential_dim == stress_dim == action_density_dim == 4
        and action_density_dim + volume_dim == action_dim
        and 4 * scale_dim == potential_dim
        and 2 * (derivative_dim + field_dim) == action_density_dim
        and 2 * mass_dim + 2 * field_dim == action_density_dim
    )
    return ScalarReceiptSourceAdmissionCertificate(
        lorentz_tensor=lorentz,
        positive_vacuum_form=vacuum,
        action_countermodel=countermodel,
        source_accounting=accounting,
        e31_full_receipt_combined_rank=combined_rank,
        e31_receipt_kernel_rank=hidden_rank,
        e31_rank_complete_receipt=rank_complete,
        receipt_mass_dimension=receipt_dim,
        metric_mass_dimension=metric_dim,
        reference_scale_mass_dimension=scale_dim,
        scalar_field_mass_dimension=field_dim,
        scalar_mass_dimension=mass_dim,
        derivative_mass_dimension=derivative_dim,
        potential_mass_dimension=potential_dim,
        stress_mass_dimension=stress_dim,
        action_density_mass_dimension=action_density_dim,
        volume_element_mass_dimension=volume_dim,
        action_mass_dimension=action_dim,
        hidden_action_coefficient_mass_dimension=epsilon_dim,
        dimensions_pass=dimensions_pass,
        rank_complete_receipt_selects_physical_source=(
            rank_complete and not ambiguity
        ),
        scalar_only_order_zero_source_is_vacuum_form=(
            lorentz.full_metric_span_unique
            and vacuum.equation_of_state == -1.0
        ),
        dust_source_derived=False,
        current_gradient_or_kinetic_data_required_for_dust=True,
        local_receipt_to_field_map_derived=False,
        supplied_metric_derived_from_receipt=False,
        metric_variation_machine_verified=False,
        conditional_ward_theorem_replaced_by_numerics=False,
        cptp_quantum_dynamics_derived=False,
        qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        finite_coefficient_gr_phenomenology_derived=False,
        independent_holdout_prediction_derived=False,
        two_residual_classes_reduced=False,
        complexity_penalty_success=False,
    )


def scalar_receipt_run() -> dict[str, object]:
    """JSON 직렬화 가능한 E32 영수증을 돌려준다."""

    return asdict(scalar_receipt_certificate())


def scalar_receipt_main(argv: list[str] | None = None) -> int:
    """E32 영수증을 명령줄에서 출력하고 종료 코드를 돌려준다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(scalar_receipt_run(), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
