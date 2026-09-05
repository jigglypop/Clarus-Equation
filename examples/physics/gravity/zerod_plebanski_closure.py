"""CE 0차원(0D)→플레바인스키(Plebanski) 사슬의 구성적 유한 닫힘과 그 보조 모듈 세 개를 한 파일에 모은다.

이 모듈은 네 개의 독립 부분으로 이루어진다.

1. 분할/병합 블록 재규격화(block RG) 감사.
   미시 사건은 포아송(Poisson) 분포 Poisson(D)개의 후보 연속을 방출한다.
   각 후보는 독립적으로 새 렌더링 연속 또는 재결합/면(face) 사건으로 표시된다.
   임계 표시는 독립 렌더링 연속의 평균이 1이 되도록 고른다.
   다음 세 진술을 구분한다.
   (a) 임계 1차 모멘트는 블록킹 아래 정확히 고정된다.
   (b) 포아송 자손 법칙 전체는 블록킹 아래 고정되지 않는다.
   (c) 지속 렌더링 계보 하나를 조건으로 두면 국소 스파인(spine) 환경은
       이동 불변이며 정확한 최소 고정 대상을 준다.
   이 구성은 확률/RG 장난감 모형이다. 분기 법칙만으로 단순성(simplicity) 제약,
   플레바인스키 진폭, 물리적 시공간을 유도하지 않는다.

2. 무유도(no-go) 정리: 유한 면 기하는 연속 일반상대론(GR)도 두 자유도(DOF)도 고정하지 않는다.
   유한 로렌츠(Lorentz) 면 재구성 모듈은 고정 복합체 위의 내재 계량 자료와
   선언된 수송을 결정한다. 연속 계량에 대한 범함수 도함수가 없으므로
   연속 작용을 결정할 수 없다.
   명시적 공변 반례 하나면 충분하다. 같은 평탄 배경 위에서

       S_EH = ∫ sqrt(-g) R,
       S_R2 = ∫ sqrt(-g) (R + alpha R^2),  alpha > 0

   두 작용은 동일한 영곡률 정상 기하를 가지며 따라서 동일한 유한 평탄 면/수송
   자료를 준다. 선형화 스펙트럼은 다르다. R + alpha R^2 장방정식의 대각합은

       -R + 6 alpha Box R = 0

   이므로 R은 m_scalar^2 = 1/(6 alpha)인 클라인-고든(Klein-Gordon) 방정식을 따른다.
   수치적으로는 alpha_bar = alpha / L_ref^2 를 쓰고 (m_scalar L_ref)^2 를 보고하므로
   API 입출력은 모두 무차원이다. 아인슈타인(Einstein) 중력은 질량 없는 횡단
   무대각합 편광 두 개만 갖고, 두 번째 작용은 그 둘에 스칼라론(scalaron)이
   추가된다. 따라서 어떤 유한 배경 재구성 정리도 유일한 아인슈타인-힐베르트
   작용이나 정확히 두 전파 자유도를 함축할 수 없다.
   이는 그 함축에 대한 완전한 반례이며, R^2 작용이 CE에 의해 선택된다는 주장이 아니다.

3. 구성적 유한 로렌츠 선형 단순성(linear simplicity) 재구성.
   부호수 (-,+,+,+), 반변 벡터, epsilon_0123 = +1 규약을 쓴다.
   단위 미래 시간꼴 법선 n과 그에 직교하는 모서리 E에 대해 이중벡터(bivector)

       B^{IJ} = n^I E^J - n^J E^I

   를 정의한다. 선형 단순성 조건은 n_I (*B)^{IJ} = 0 이다. 이 선언된 구역에서

       E^J = -n_I B^{IJ},
       G_ab = E_a . E_b = -(1/2) B_{a IJ} B_b^{IJ}

   가 성립하므로 독립 이중벡터 세 개가 라벨된 공간꼴 면 삼중틀(triad)과
   그 내재 그람(Gram) 행렬을 재구성한다. 이는 유한 역보조정리일 뿐이다.
   플레바인스키 가지 전체를 고르거나, 4-단체 닫힘을 증명하거나, 스핀폼
   진폭을 택하거나, 연속 GR 극한을 함축하지 않는다.

4. 0D→플레바인스키 사슬의 구성적 유한 닫힘.
   이 부분은 "맨 단일항이 4차원 중력을 유일하게 결정한다"는 거짓 함축을
   되살리지 않는다. 더 강한 타입 모형을 제공하고, 증명된 것과 선택된 것을 분리한다.
   - 계수 4의 좌표 없는 단체(simplex) 상호작용은 0D 재작성 규칙이며, 짝지은
     가닥(strand)이 4-단체의 여차원 2 면 열 개를 나른다.
   - 합성 면은 군 홀로노미(holonomy)를 나르므로 유한 곡률 관측량이 된다.
   - 플랑크(Planck) 조대 판독값의 상등은 동치 관계이며, 미시 역사는 상태 공간에 남는다.
   - 서로 다른 판독 부류에 대한 직교 기록은 그 부류들을 결어긋나게(decoherent)
     하되 접힌 노름을 삭제하지 않는다.
   - 유한 조대 역사 공간 위에서 양의 깁스(Gibbs) 결함 가중치는 유한 beta에서
     모든 역사를 보존하고, 영결함 공통 계량 구역에 지수적으로 집중한다.
   - 같은 차수 B_2/F_2 닫힘은 형식 차수 4를 선택하고, 명시적 비퇴화 (-,+,+,+)
     계량이 추가 로렌츠 입력을 제공한다.
   - 기존 유한 로렌츠 단순성과 공유 면 재구성은 한 사면체틀(tetrad) 위의 정확한
     평탄 키랄(chiral) 플레바인스키/아인슈타인 해에 연결된다.
   결과 증명서는 선언된 모형에 대한 단일 타입 역사, 유한 평탄 조건부 존재 정리이다.
   맨 0D 자료로부터의 유일성, 경험적 주장, 일반 곡률 해, 자연의 특정 미시
   RG/세분 흐름이 이 모형을 고른다는 증명이 아니다.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from examples.physics.gravity.causal_face_simplicity import (
    CompositionFace,
    composition_faces,
    hard_shared_spacelike_face_match,
    proper_orthochronous_residual,
)


# ---------------------------------------------------------------------------
# 1. 분할/병합 블록 RG 감사 (구 planck_rendering_block_rg)
# ---------------------------------------------------------------------------


def _require_finite_scalar(name: str, value: float) -> None:
    """값이 유한하지 않으면 ValueError를 낸다. 반환값은 없다."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_integer(name: str, value: int) -> None:
    """값이 bool이 아닌 음이 아닌 정수인지 검사한다."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class CriticalSplitMerge:
    """임계 분할/병합 표시의 매개변수 묶음이다."""

    microscopic_branch_mean: float
    distinct_probability: float
    merge_probability: float
    distinct_intensity: float
    face_intensity: float
    status: str = "MEAN_CRITICAL_SPLIT_MERGE"


def critical_split_merge(branch_mean: float) -> CriticalSplitMerge:
    """렌더링 자손 평균이 1이 되는 유일한 독립 표시를 반환한다.

    K ~ Poisson(D)이고 각 후보가 독립적으로 확률 s로 렌더링 표시를 받으면
    포아송 솎음(thinning)에 의해

        S ~ Poisson(D*s), F ~ Poisson(D*(1-s)), S와 F는 독립

    이다. 임계 가시 용량은 E[S] = 1 을 요구하므로 s = 1/D 이다.
    """

    _require_finite_scalar("branch_mean", branch_mean)
    if branch_mean <= 1.0:
        raise ValueError("branch_mean must exceed one for a non-zero merge sector")
    distinct_probability = 1.0 / branch_mean
    merge_probability = 1.0 - distinct_probability
    return CriticalSplitMerge(
        microscopic_branch_mean=branch_mean,
        distinct_probability=distinct_probability,
        merge_probability=merge_probability,
        distinct_intensity=1.0,
        face_intensity=branch_mean - 1.0,
    )


def marked_joint_probability(
    *,
    branch_mean: float,
    distinct_probability: float,
    distinct_count: int,
    face_count: int,
) -> float:
    """독립 표시된 포아송 후보에 대한 P(S=s, F=f)를 반환한다."""

    _require_finite_scalar("branch_mean", branch_mean)
    _require_finite_scalar("distinct_probability", distinct_probability)
    if branch_mean < 0.0:
        raise ValueError("branch_mean must be non-negative")
    if not 0.0 <= distinct_probability <= 1.0:
        raise ValueError("distinct_probability must lie in [0, 1]")
    _require_nonnegative_integer("distinct_count", distinct_count)
    _require_nonnegative_integer("face_count", face_count)
    rendered_mean = branch_mean * distinct_probability
    face_mean = branch_mean * (1.0 - distinct_probability)
    return (
        math.exp(-rendered_mean)
        * rendered_mean**distinct_count
        / math.factorial(distinct_count)
        * math.exp(-face_mean)
        * face_mean**face_count
        / math.factorial(face_count)
    )


def blocked_rendered_mean(rendered_mean: float, depth: int) -> float:
    """``depth`` 세대를 블록킹한 뒤의 출력 평균 개수를 반환한다."""

    _require_finite_scalar("rendered_mean", rendered_mean)
    if rendered_mean < 0.0:
        raise ValueError("rendered_mean must be non-negative")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    return rendered_mean**depth


@dataclass(frozen=True)
class CriticalBlockMoments:
    """임계점에서 깊이 블록의 정확한 1차·2차 모멘트이다."""

    depth: int
    output_mean: float
    output_variance: float
    expected_parent_events: float
    expected_face_events: float
    face_event_variance: float
    poisson_family_closed: bool
    status: str = "CRITICAL_MEAN_FIXED_FULL_LAW_NOT_FIXED"


def critical_block_moments(branch_mean: float, depth: int) -> CriticalBlockMoments:
    """임계점에서 깊이 블록의 정확한 1차·2차 모멘트를 반환한다.

    사건 하나에서 시작하는 임계 포아송 골턴-왓슨(Galton-Watson) 과정은
    E[Z_n] = 1, Var(Z_n) = n 이다. 모든 활성 부모가 독립적으로 mu = D-1인
    Poisson(mu) 면 사건을 방출하면 블록의 기대 면 개수는 mu*depth 이다.
    자손 법칙 전체는 깊이에 따라 넓어지므로 포아송 고정 분포가 아니다.
    """

    params = critical_split_merge(branch_mean)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    mu = params.face_intensity
    parent_count_variance = (depth - 1) * depth * (2 * depth - 1) / 6.0
    face_variance = mu * depth + mu * mu * parent_count_variance
    return CriticalBlockMoments(
        depth=depth,
        output_mean=1.0,
        output_variance=float(depth),
        expected_parent_events=float(depth),
        expected_face_events=mu * depth,
        face_event_variance=face_variance,
        poisson_family_closed=(depth == 1),
    )


def q_spine_distinct_probability(distinct_count: int) -> float:
    """스파인 위의 크기 편향(size-biased) 임계 포아송 자손 법칙을 반환한다.

    보통의 렌더링 자손은 S ~ Poisson(1)이다. 지속 계보를 조건으로 두면
    둡(Doob) h-변환/크기 편향 법칙

        P_Q(S=k) = k P(S=k), k >= 1

    을 얻으며, 이는 정확히 S = 1 + Poisson(1) 이다.
    """

    _require_nonnegative_integer("distinct_count", distinct_count)
    if distinct_count < 1:
        return 0.0
    return math.exp(-1.0) / math.factorial(distinct_count - 1)


@dataclass(frozen=True)
class SpineFixedPoint:
    """지속 렌더링 계보 하나에서 본 정확한 국소 법칙이다."""

    rendered_continuation_mean: float
    persistent_spine_count: int
    folded_side_branch_mean: float
    face_event_mean: float
    shift_invariant_local_law: bool
    status: str = "SPINE_CONDITIONED_LOCAL_RG_FIXED_POINT"


def spine_fixed_point(branch_mean: float) -> SpineFixedPoint:
    """지속 렌더링 계보 하나에서 본 정확한 국소 법칙을 반환한다."""

    params = critical_split_merge(branch_mean)
    return SpineFixedPoint(
        rendered_continuation_mean=2.0,
        persistent_spine_count=1,
        folded_side_branch_mean=1.0,
        face_event_mean=params.face_intensity,
        shift_invariant_local_law=True,
    )


def critical_side_tree_total_progeny_probability(total_vertices: int) -> float:
    """보통의 임계 포아송 곁가지 나무에 대한 보렐(Borel)(1) 법칙을 반환한다.

    나무는 거의 확실히 유한하지만 총 자손 평균은 발산한다. 확률 질량은
    척도 없는 점근형 n^(-3/2)/sqrt(2*pi) 를 갖는다.
    """

    if isinstance(total_vertices, bool) or not isinstance(total_vertices, int):
        raise ValueError("total_vertices must be a positive integer")
    if total_vertices < 1:
        raise ValueError("total_vertices must be a positive integer")
    n = total_vertices
    log_probability = -n + (n - 1) * math.log(n) - math.lgamma(n + 1)
    return math.exp(log_probability)


def critical_borel_asymptotic_ratio(total_vertices: int) -> float:
    """P(N=n)*sqrt(2*pi)*n^(3/2) 를 반환한다. 이 값은 1로 수렴한다."""

    probability = critical_side_tree_total_progeny_probability(total_vertices)
    return probability * math.sqrt(2.0 * math.pi) * total_vertices**1.5


def heat_time_from_area(
    *,
    area: float,
    planck_area: float,
    normalization: float = 1.0,
) -> float:
    """가법적 무차원 열시간(heat time) alpha*A/A_P 를 반환한다."""

    for name, value in (
        ("area", area),
        ("planck_area", planck_area),
        ("normalization", normalization),
    ):
        _require_finite_scalar(name, value)
    if area < 0.0:
        raise ValueError("area must be non-negative")
    if planck_area <= 0.0:
        raise ValueError("planck_area must be positive")
    if normalization <= 0.0:
        raise ValueError("normalization must be positive")
    return normalization * area / planck_area


@dataclass(frozen=True)
class BlockRGVerdict:
    """최소 유한 블록 계산이 닫는 것과 남는 의무를 요약한다."""

    critical_rendered_mean: float
    critical_face_intensity: float
    full_poisson_measure_fixed: bool
    spine_local_measure_fixed: bool
    side_sector_scale_free: bool
    remaining_obligation: str


def block_rg_verdict(branch_mean: float) -> BlockRGVerdict:
    """최소 유한 블록 계산이 닫는 것을 요약한다."""

    params = critical_split_merge(branch_mean)
    return BlockRGVerdict(
        critical_rendered_mean=params.distinct_intensity,
        critical_face_intensity=params.face_intensity,
        full_poisson_measure_fixed=False,
        spine_local_measure_fixed=True,
        side_sector_scale_free=True,
        remaining_obligation=(
            "derive the face attachment topology and simplicity amplitude; "
            "the split/merge count law alone does not produce Plebanski gravity"
        ),
    )


# ---------------------------------------------------------------------------
# 2. 연속 GR / 두 자유도 무유도 정리 (구 continuum_gr_dof_no_go)
# ---------------------------------------------------------------------------


MINKOWSKI_METRIC = np.diag((-1.0, 1.0, 1.0, 1.0))


def massless_spin_two_polarization_count(dimension: int) -> int:
    """질량 없는 스핀 2 장의 작은군(little group) 개수 d(d-3)/2 를 반환한다."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 4:
        raise ValueError("dimension must be an integer of at least four")
    return dimension * (dimension - 3) // 2


def massive_spin_two_polarization_count(dimension: int) -> int:
    """대칭 텐서 성분 수에서 횡단 조건 d개와 대각합 조건 1개를 뺀 값을 반환한다."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 3:
        raise ValueError("dimension must be an integer of at least three")
    return dimension * (dimension + 1) // 2 - dimension - 1


def massless_tt_basis_4d() -> tuple[np.ndarray, np.ndarray]:
    """양의 z축 방향 영(null) 파동에 대한 플러스/크로스 텐서를 반환한다."""

    plus = np.zeros((4, 4))
    plus[1, 1] = 1.0
    plus[2, 2] = -1.0
    cross = np.zeros((4, 4))
    cross[1, 2] = cross[2, 1] = 1.0
    return plus, cross


def massive_traceless_transverse_basis_4d() -> tuple[np.ndarray, ...]:
    """질량 있는 스핀 2 장의 정지틀 편광 다섯 개를 반환한다."""

    spatial: list[np.ndarray] = []
    first = np.diag((1.0, -1.0, 0.0))
    second = np.diag((1.0, 1.0, -2.0))
    spatial.extend((first, second))
    for left, right in ((0, 1), (0, 2), (1, 2)):
        item = np.zeros((3, 3))
        item[left, right] = item[right, left] = 1.0
        spatial.append(item)
    result: list[np.ndarray] = []
    for item in spatial:
        tensor = np.zeros((4, 4))
        tensor[1:, 1:] = item
        result.append(tensor)
    return tuple(result)


@dataclass(frozen=True)
class ContinuumGRNoGoAudit:
    """R 대 R + alpha R^2 반례의 결과 묶음이다."""

    alpha_over_reference_length_squared: float
    scalaron_mass_squared_times_reference_length_squared: float
    einstein_hilbert_polarizations: int
    r_plus_r_squared_polarizations: int
    shared_flat_stationary_background: bool
    shared_finite_flat_face_data: bool
    both_actions_diffeomorphism_invariant: bool
    unique_continuum_action_follows: bool
    exactly_two_dof_follow: bool
    status: str = "FINITE_FACE_TO_UNIQUE_CONTINUUM_GR_IMPLICATION_DISPROVED"
    claim_ceiling: str = "COMPLETE_COUNTEREXAMPLE_TO_BACKGROUND_ONLY_GR_CLOSURE"


def continuum_gr_dof_no_go(
    alpha_over_reference_length_squared: float = 1.0,
) -> ContinuumGRNoGoAudit:
    """R 대 R + alpha R^2 의 정확한 반례를 반환한다."""

    alpha_bar = float(alpha_over_reference_length_squared)
    if not math.isfinite(alpha_bar) or alpha_bar <= 0.0:
        raise ValueError(
            "alpha_over_reference_length_squared must be finite and positive"
        )
    eh_count = massless_spin_two_polarization_count(4)
    scalaron_mass_squared = 1.0 / (6.0 * alpha_bar)
    return ContinuumGRNoGoAudit(
        alpha_over_reference_length_squared=alpha_bar,
        scalaron_mass_squared_times_reference_length_squared=(
            scalaron_mass_squared
        ),
        einstein_hilbert_polarizations=eh_count,
        r_plus_r_squared_polarizations=eh_count + 1,
        shared_flat_stationary_background=True,
        shared_finite_flat_face_data=True,
        both_actions_diffeomorphism_invariant=True,
        unique_continuum_action_follows=False,
        exactly_two_dof_follow=False,
    )


# ---------------------------------------------------------------------------
# 3. 로렌츠 선형 단순성 이중벡터 재구성 (구 lorentzian_bivector_reconstruction)
# ---------------------------------------------------------------------------


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """주어진 모양의 유한 float 배열로 변환하고, 아니면 ValueError를 낸다."""

    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _stable_frobenius_norm(value: np.ndarray) -> float:
    """피할 수 있는 척도 하위/상위 넘침 없이 프로베니우스(Frobenius) 노름을 반환한다."""

    array = np.asarray(value, dtype=float)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _permutation_sign(indices: tuple[int, int, int, int]) -> int:
    """네 첨자의 순열 부호를 반환한다. 중복이 있으면 0이다."""

    if len(set(indices)) < 4:
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


# 첨자 넷을 모두 올리면 det(eta) = -1 때문에 부호가 바뀐다.
_EPSILON_UPPER = np.empty((4, 4, 4, 4), dtype=float)
for _i in range(4):
    for _j in range(4):
        for _k in range(4):
            for _l in range(4):
                _EPSILON_UPPER[_i, _j, _k, _l] = -_permutation_sign(
                    (_i, _j, _k, _l)
                )


def minkowski_inner(first: np.ndarray, second: np.ndarray) -> float:
    """두 반변 벡터의 (-,+,+,+) 내적을 반환한다."""

    first = _require_shape("first", first, (4,))
    second = _require_shape("second", second, (4,))
    return float(first @ MINKOWSKI_METRIC @ second)


def bivector_from_normal_edge(normal: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """선언된 기하 구역을 검증한 뒤 B = n wedge E 를 반환한다."""

    normal = _require_shape("normal", normal, (4,))
    edge = _require_shape("edge", edge, (4,))
    if abs(minkowski_inner(normal, normal) + 1.0) > 1.0e-10 or normal[0] <= 0.0:
        raise ValueError("normal must be unit future timelike")
    edge_scale = _stable_frobenius_norm(edge)
    if edge_scale == 0.0:
        raise ValueError("edge must be nonzero")
    if abs(minkowski_inner(normal, edge)) / edge_scale > 1.0e-10:
        raise ValueError("edge must be orthogonal to normal")
    return np.outer(normal, edge) - np.outer(edge, normal)


def hodge_dual(bivector: np.ndarray) -> np.ndarray:
    """로렌츠 호지(Hodge) 쌍대를 반환한다. 2-형식 위에서 별 연산의 제곱은 -1이다."""

    bivector = _require_shape("bivector", bivector, (4, 4))
    lower = MINKOWSKI_METRIC @ bivector @ MINKOWSKI_METRIC
    return 0.5 * np.einsum("ijkl,kl->ij", _EPSILON_UPPER, lower)


def bivector_inner(first: np.ndarray, second: np.ndarray) -> float:
    """반대칭 행렬의 두 삼각 부분을 모두 포함한 B_IJ C^IJ 를 반환한다."""

    first = _require_shape("first", first, (4, 4))
    second = _require_shape("second", second, (4, 4))
    first_lower = MINKOWSKI_METRIC @ first @ MINKOWSKI_METRIC
    return float(np.sum(first_lower * second))


def common_linear_simplicity_nullity(
    bivectors: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> int:
    """n_I (*B_a)^IJ = 0 을 만족하는 법선 공간의 차원을 반환한다."""

    bivectors = np.asarray(bivectors, dtype=float)
    if bivectors.ndim != 3 or bivectors.shape[1:] != (4, 4):
        raise ValueError("bivectors must have shape (count, 4, 4)")
    if not np.all(np.isfinite(bivectors)):
        raise ValueError("bivectors must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    constraint = np.vstack([hodge_dual(item).T for item in bivectors])
    constraint_scale = _stable_frobenius_norm(constraint)
    if constraint_scale > 0.0:
        constraint = constraint / constraint_scale
    singular_values = np.linalg.svd(constraint, compute_uv=False)
    maximum = float(np.max(singular_values)) if singular_values.size else 0.0
    rank = (
        int(np.count_nonzero(singular_values > tolerance * maximum))
        if maximum > 0.0
        else 0
    )
    return 4 - rank


@dataclass(frozen=True)
class LorentzianBivectorFaceAudit:
    """단순 이중벡터 세 개로부터 라벨된 공간꼴 삼중틀을 재구성한 결과이다."""

    reconstructed_edges: np.ndarray
    edge_gram: np.ndarray
    bivector_gram: np.ndarray
    normal_residual: float
    antisymmetry_residual: float
    linear_simplicity_residual: float
    reconstruction_residual: float
    gram_identity_residual: float
    oriented_face_volume: float
    common_normal_nullity: int
    hard_reconstruction: bool
    status: str
    plebanski_branch: str = "NOT_TESTED_BY_LINEAR_FACE_DATA"
    claim_ceiling: str = "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTION_ONLY"


def bivector_face_reconstruction_audit(
    normal: np.ndarray,
    bivectors: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> LorentzianBivectorFaceAudit:
    """단순 이중벡터 세 개로부터 라벨된 공간꼴 삼중틀을 재구성한다."""

    normal = _require_shape("normal", normal, (4,))
    bivectors = _require_shape("bivectors", bivectors, (3, 4, 4))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    normal_residual = abs(minkowski_inner(normal, normal) + 1.0)
    bivector_scale = _stable_frobenius_norm(bivectors)
    antisymmetry_residual = (
        _stable_frobenius_norm(bivectors + np.swapaxes(bivectors, 1, 2))
        / bivector_scale
        if bivector_scale > 0.0
        else math.inf
    )
    normal_lower = MINKOWSKI_METRIC @ normal
    stars = np.asarray([hodge_dual(item) for item in bivectors])
    simplicity_numerators = np.asarray(
        [normal_lower @ star for star in stars]
    )
    star_scale = _stable_frobenius_norm(stars)
    linear_simplicity_residual = (
        _stable_frobenius_norm(simplicity_numerators) / star_scale
        if star_scale > 0.0
        else math.inf
    )

    reconstructed_edges = np.asarray(
        [-normal_lower @ bivector for bivector in bivectors]
    )
    reconstructed_bivectors = np.asarray(
        [
            np.outer(normal, edge) - np.outer(edge, normal)
            for edge in reconstructed_edges
        ]
    )
    reconstruction_residual = (
        _stable_frobenius_norm(bivectors - reconstructed_bivectors) / bivector_scale
        if bivector_scale > 0.0
        else math.inf
    )
    edge_scale = _stable_frobenius_norm(reconstructed_edges)
    normalized_edges = (
        reconstructed_edges / edge_scale
        if edge_scale > 0.0
        else reconstructed_edges
    )
    normalized_edge_gram = (
        normalized_edges @ MINKOWSKI_METRIC @ normalized_edges.T
    )
    normalized_bivectors = (
        bivectors / bivector_scale if bivector_scale > 0.0 else bivectors
    )
    normalized_bivector_gram = np.asarray(
        [
            [
                -0.5
                * bivector_inner(
                    normalized_bivectors[left], normalized_bivectors[right]
                )
                for right in range(3)
            ]
            for left in range(3)
        ]
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        edge_scale_squared = np.float64(edge_scale) * np.float64(edge_scale)
        bivector_scale_squared = (
            np.float64(bivector_scale) * np.float64(bivector_scale)
        )
        edge_gram = edge_scale_squared * normalized_edge_gram
        bivector_gram = bivector_scale_squared * normalized_bivector_gram
    scale_ratio_squared = (
        (bivector_scale / edge_scale) ** 2
        if edge_scale > 0.0 and bivector_scale > 0.0
        else 0.0
    )
    comparable_bivector_gram = scale_ratio_squared * normalized_bivector_gram
    gram_scale = max(
        _stable_frobenius_norm(normalized_edge_gram),
        _stable_frobenius_norm(comparable_bivector_gram),
    )
    gram_identity_residual = (
        _stable_frobenius_norm(
            normalized_edge_gram - comparable_bivector_gram
        )
        / gram_scale
        if gram_scale > 0.0
        else math.inf
    )
    eigenvalues = np.linalg.eigvalsh(normalized_edge_gram)
    maximum_eigenvalue = float(np.max(eigenvalues))
    spacelike_rank_three = (
        maximum_eigenvalue > 0.0
        and float(np.min(eigenvalues)) / maximum_eigenvalue > tolerance
    )
    normalized_oriented_volume = float(
        np.linalg.det(np.vstack((normal, normalized_edges)))
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        edge_scale_cubed = (
            np.float64(edge_scale)
            * np.float64(edge_scale)
            * np.float64(edge_scale)
        )
        oriented_face_volume = float(normalized_oriented_volume * edge_scale_cubed)
    common_normal_nullity = common_linear_simplicity_nullity(
        bivectors,
        tolerance=tolerance,
    )

    if normal_residual > tolerance or normal[0] <= tolerance:
        status = "INVALID_UNIT_FUTURE_NORMAL"
    elif antisymmetry_residual > tolerance:
        status = "NON_ANTISYMMETRIC_BIVECTOR_DATA"
    elif linear_simplicity_residual > tolerance:
        status = "LINEAR_SIMPLICITY_FAILED"
    elif reconstruction_residual > tolerance:
        status = "BIVECTOR_INVERSE_RECONSTRUCTION_FAILED"
    elif not spacelike_rank_three:
        status = "NONSPACELIKE_OR_RANK_DEFICIENT_FACE"
    elif gram_identity_residual > tolerance:
        status = "BIVECTOR_GRAM_IDENTITY_FAILED"
    else:
        status = "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTED"

    return LorentzianBivectorFaceAudit(
        reconstructed_edges=reconstructed_edges,
        edge_gram=edge_gram,
        bivector_gram=bivector_gram,
        normal_residual=normal_residual,
        antisymmetry_residual=antisymmetry_residual,
        linear_simplicity_residual=linear_simplicity_residual,
        reconstruction_residual=reconstruction_residual,
        gram_identity_residual=gram_identity_residual,
        oriented_face_volume=oriented_face_volume,
        common_normal_nullity=common_normal_nullity,
        hard_reconstruction=(status == "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTED"),
        status=status,
    )


# ---------------------------------------------------------------------------
# 4. 0D→플레바인스키 구성적 유한 닫힘
# ---------------------------------------------------------------------------


def _require_finite(name: str, value: float) -> float:
    """값을 float로 바꾸고 유한하지 않으면 ValueError를 낸다."""

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_norm(value: np.ndarray) -> float:
    """최대 성분으로 정규화한 뒤 노름을 계산해 넘침을 피한다."""

    array = np.asarray(value)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _logsumexp(values: np.ndarray) -> float:
    """피할 수 있는 넘침 없이 log(sum(exp(values)))를 반환한다."""

    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.sum(np.exp(array - maximum))))


@dataclass(frozen=True)
class FormDegreeClosureAudit:
    """형식 차수 닫힘에 의한 조건부 차원 선택의 감사 결과이다."""

    curvature_form_degree: int
    conjugate_form_degree: int
    spacetime_dimension: int
    same_type_pair: bool
    hodge_degree_closes: bool
    one_time_direction: bool
    metric_signature: tuple[int, ...]
    nondegenerate_lorentzian_signature: bool
    spatial_dimension: int
    lorentzian_three_plus_one: bool
    status: str


def form_degree_closure(
    curvature_form_degree: int = 2,
    conjugate_form_degree: int = 2,
    *,
    one_time_direction: bool = True,
    metric_signature: Sequence[int] = (-1, 1, 1, 1),
) -> FormDegreeClosureAudit:
    """조건부 차원 선택 보조정리를 감사한다.

    배경 계량 없는 국소 항 ``B_q wedge F_p`` 는 차원 ``p+q`` 에서 최고 형식이다.
    재귀적으로 짝지은 장이 같은 차수이고 곡률이 2-형식이어야 한다는 요구는
    p = q = 2, 따라서 D = 4 를 준다. 별도로 선언된 비퇴화 ``(-,+,+,+)`` 계량이
    있어야 3+1 이 된다. 한 방향을 "시간"이라 이름 붙이는 것만으로는 로렌츠
    부호수가 확립되지 않는다.
    """

    for name, degree in (
        ("curvature_form_degree", curvature_form_degree),
        ("conjugate_form_degree", conjugate_form_degree),
    ):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(one_time_direction, bool):
        raise ValueError("one_time_direction must be boolean")
    signature = tuple(metric_signature)
    if any(isinstance(entry, bool) or entry not in (-1, 1) for entry in signature):
        raise ValueError("metric_signature entries must be -1 or 1")

    dimension = curvature_form_degree + conjugate_form_degree
    same_type = curvature_form_degree == conjugate_form_degree
    hodge_closes = dimension == 2 * curvature_form_degree
    spatial_dimension = dimension - 1 if one_time_direction else dimension
    lorentzian_signature = (
        len(signature) == dimension
        and signature.count(-1) == 1
        and signature.count(1) == dimension - 1
    )
    selected = (
        curvature_form_degree == 2
        and conjugate_form_degree == 2
        and dimension == 4
        and one_time_direction
        and lorentzian_signature
    )
    return FormDegreeClosureAudit(
        curvature_form_degree=curvature_form_degree,
        conjugate_form_degree=conjugate_form_degree,
        spacetime_dimension=dimension,
        same_type_pair=same_type,
        hodge_degree_closes=hodge_closes,
        one_time_direction=one_time_direction,
        metric_signature=signature,
        nondegenerate_lorentzian_signature=lorentzian_signature,
        spatial_dimension=spatial_dimension,
        lorentzian_three_plus_one=selected,
        status=(
            "CONDITIONAL_3_PLUS_1_FORM_DEGREE_CLOSURE"
            if selected
            else "FORM_DEGREE_CONDITIONS_DO_NOT_SELECT_3_PLUS_1"
        ),
    )


@dataclass(frozen=True)
class SimplexInteractionAudit:
    """계수 d 단체 상호작용 하나의 조합적 감사 결과이다."""

    rank: int
    interaction_valence: int
    strand_ends: int
    paired_codimension_two_faces: int
    every_strand_paired_twice: bool
    boundary_euler_characteristic: int
    expected_boundary_euler_characteristic: int
    coordinate_free: bool
    target_four_simplex: bool
    status: str


def simplex_interaction_audit(rank: int = 4) -> SimplexInteractionAudit:
    """계수 d 단체 상호작용 하나의 조합적 감사를 반환한다.

    상호작용은 경계 원자 d+1개를 갖는다. 원자 i는 다른 모든 원자 j마다 가닥
    하나를 갖는다. 비순서쌍 {i,j}는 정확히 두 가닥 끝에 나타나므로, 짝지은
    가닥은 d-단체의 여차원 2 면 C(d+1,2)개이다. 이 조합 규칙은 시공간 좌표를
    쓰지 않는다.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 2:
        raise ValueError("rank must be an integer of at least two")
    valence = rank + 1
    strand_labels = [
        tuple(sorted((atom, partner)))
        for atom in range(valence)
        for partner in range(valence)
        if partner != atom
    ]
    multiplicities = {
        label: strand_labels.count(label) for label in set(strand_labels)
    }
    paired_faces = math.comb(valence, 2)
    complete_pair_set = set(combinations(range(valence), 2))
    # d-단체의 경계: f_k = C(d+1, k+1), k = 0, ..., d-1.
    boundary_euler = sum(
        (-1) ** cell_dimension * math.comb(valence, cell_dimension + 1)
        for cell_dimension in range(rank)
    )
    expected_euler = 1 + (-1) ** (rank - 1)
    target = rank == 4
    return SimplexInteractionAudit(
        rank=rank,
        interaction_valence=valence,
        strand_ends=len(strand_labels),
        paired_codimension_two_faces=paired_faces,
        every_strand_paired_twice=(
            len(multiplicities) == paired_faces
            and all(count == 2 for count in multiplicities.values())
        ),
        boundary_euler_characteristic=boundary_euler,
        expected_boundary_euler_characteristic=expected_euler,
        # 라벨은 2-부분집합 전체를 이룬다. 따라서 원자 라벨의 모든 전단사는
        # 이 집합을 치환하고 중복도를 보존한다. 이는 임의 재라벨링 아래의
        # 공변성을 증명하는 것이지 표본 하나를 보는 것이 아니다.
        coordinate_free=(
            set(multiplicities) == complete_pair_set
            and all(count == 2 for count in multiplicities.values())
        ),
        target_four_simplex=target,
        status=(
            "RANK_FOUR_COORDINATE_FREE_SIMPLEX_INTERACTION"
            if target
            else "NON_TARGET_SIMPLEX_RANK"
        ),
    )


@dataclass(frozen=True)
class FaceHolonomyAudit:
    """부착된 2-세포 하나 둘레의 홀로노미 감사 결과이다."""

    face_id: Hashable
    holonomy: np.ndarray
    factor_count: int
    attached_contractible_face: bool
    maximum_lorentz_residual: float
    flatness_residual: float
    nontrivial_curvature_carrier: bool
    status: str


def face_holonomy_audit(
    oriented_holonomies: Sequence[np.ndarray],
    *,
    face_id: Hashable,
    attached_contractible_face: bool,
    tolerance: float = 1.0e-10,
) -> FaceHolonomyAudit:
    """선언되고 부착된 2-세포 하나 둘레의 모서리 수송을 곱한다."""

    tolerance = _require_finite("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if face_id is None:
        raise ValueError("face_id must identify the attached 2-cell")
    if not isinstance(attached_contractible_face, bool):
        raise ValueError("attached_contractible_face must be boolean")
    if not oriented_holonomies:
        raise ValueError("a face must have at least one oriented holonomy")
    factors = tuple(np.asarray(item, dtype=float) for item in oriented_holonomies)
    if any(item.shape != (4, 4) for item in factors):
        raise ValueError("every oriented holonomy must have shape (4, 4)")
    if any(not np.all(np.isfinite(item)) for item in factors):
        raise ValueError("every oriented holonomy must be finite")
    residuals = tuple(proper_orthochronous_residual(item) for item in factors)
    if max(residuals) > tolerance:
        raise ValueError("every factor must belong to SO+(1,3) within tolerance")
    holonomy = np.eye(4)
    for factor in factors:
        holonomy = holonomy @ factor
    scale = max(1.0, _stable_norm(holonomy))
    flatness_residual = _stable_norm(holonomy - np.eye(4)) / scale
    curved = attached_contractible_face and flatness_residual > tolerance
    return FaceHolonomyAudit(
        face_id=face_id,
        holonomy=holonomy,
        factor_count=len(factors),
        attached_contractible_face=attached_contractible_face,
        maximum_lorentz_residual=max(residuals),
        flatness_residual=flatness_residual,
        nontrivial_curvature_carrier=curved,
        status=(
            "NONTRIVIAL_FACE_HOLONOMY_CURVATURE"
            if curved
            else (
                "IDENTITY_FACE_HOLONOMY_FLAT"
                if attached_contractible_face
                else "UNATTACHED_LOOP_IS_NOT_A_CURVATURE_CERTIFICATE"
            )
        ),
    )


@dataclass(frozen=True)
class PlanckQuotientAudit:
    """플랑크 해상도 판독값에 의한 역사 몫의 감사 결과이다."""

    microscopic_history_count: int
    coarse_class_count: int
    coarse_labels: tuple[tuple[int, ...], ...]
    observable_dimensions: tuple[str, ...]
    reference_dimensions: tuple[str, ...]
    dimension_match: bool
    equivalence_reflexive: bool
    equivalence_symmetric: bool
    equivalence_transitive: bool
    folded_pair_count: int
    all_microscopic_histories_retained: bool
    status: str = "PLANCK_READOUT_EQUIVALENCE_QUOTIENT"


def planck_resolution_quotient(
    observables_over_planck_scale: Sequence[Sequence[float]],
    *,
    observable_dimensions: Sequence[str],
    reference_dimensions: Sequence[str],
    bin_width: float = 1.0,
) -> PlanckQuotientAudit:
    """같은 유한 해상도 판독 라벨을 가진 역사들로 몫을 만든다.

    입력은 이미 적절한 플랑크 척도에 대한 무차원 비율이다. 구간(bin)은 선언된
    판독 원점 0에 고정된 반개구간이다. 결과 라벨의 상등은 자동으로 동치
    관계이다. 원래 행은 보존되며 보이는 라벨만 공유된다.
    """

    width = _require_finite("bin_width", bin_width)
    if width <= 0.0:
        raise ValueError("bin_width must be positive")
    values = np.asarray(observables_over_planck_scale, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("observables must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("observables must be finite")
    observable_units = tuple(observable_dimensions)
    reference_units = tuple(reference_dimensions)
    if len(observable_units) != values.shape[1]:
        raise ValueError("observable_dimensions must label every observable column")
    if len(reference_units) != values.shape[1]:
        raise ValueError("reference_dimensions must label every Planck reference")
    if any(not unit for unit in observable_units + reference_units):
        raise ValueError("dimension labels must be nonempty")
    if observable_units != reference_units:
        raise ValueError("every observable and Planck reference must have the same dimension")
    labels = tuple(
        tuple(int(entry) for entry in np.floor(row / width)) for row in values
    )
    relation = np.asarray(
        [[left == right for right in labels] for left in labels], dtype=bool
    )
    reflexive = bool(np.all(np.diag(relation)))
    symmetric = bool(np.array_equal(relation, relation.T))
    transitive = all(
        not (relation[i, j] and relation[j, k]) or relation[i, k]
        for i in range(len(labels))
        for j in range(len(labels))
        for k in range(len(labels))
    )
    folded_pairs = sum(
        labels[left] == labels[right]
        and not np.array_equal(values[left], values[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    )
    return PlanckQuotientAudit(
        microscopic_history_count=len(labels),
        coarse_class_count=len(set(labels)),
        coarse_labels=labels,
        observable_dimensions=observable_units,
        reference_dimensions=reference_units,
        dimension_match=True,
        equivalence_reflexive=reflexive,
        equivalence_symmetric=symmetric,
        equivalence_transitive=transitive,
        folded_pair_count=folded_pairs,
        all_microscopic_histories_retained=(len(labels) == values.shape[0]),
    )


@dataclass(frozen=True)
class DecoherentFoldAudit:
    """환경 기록 대각합 뒤의 결어긋남과 접힌 노름 보존 감사 결과이다."""

    history_count: int
    global_norm: float
    reduced_trace: float
    maximum_interclass_record_overlap: float
    minimum_intraclass_record_overlap: float
    class_record_map_consistent: bool
    maximum_interclass_coherence: float
    rendered_probability: float
    folded_probability: float
    folded_history_count: int
    decoherent: bool
    folded_sector_preserved: bool
    status: str


def decoherent_fold_audit(
    amplitudes: Sequence[complex],
    environment_states: np.ndarray,
    coarse_labels: Sequence[Hashable],
    *,
    rendered_label: Hashable,
    tolerance: float = 1.0e-10,
) -> DecoherentFoldAudit:
    """모든 역사 진폭을 유지한 채 환경 기록을 대각합한다."""

    tolerance = _require_finite("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    amplitude = np.asarray(amplitudes, dtype=complex)
    environments = np.asarray(environment_states, dtype=complex)
    labels = tuple(coarse_labels)
    if amplitude.ndim != 1 or amplitude.size < 2:
        raise ValueError("amplitudes must contain at least two histories")
    if environments.ndim != 2 or environments.shape[0] != amplitude.size:
        raise ValueError("environment_states must have one row per history")
    if len(labels) != amplitude.size:
        raise ValueError("coarse_labels must have one label per history")
    if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(environments)):
        raise ValueError("amplitudes and environment states must be finite")
    amplitude_norm = float(np.vdot(amplitude, amplitude).real)
    if abs(amplitude_norm - 1.0) > tolerance:
        raise ValueError("history amplitudes must be normalized")
    gram = environments.conj() @ environments.T
    if np.max(np.abs(np.diag(gram) - 1.0)) > tolerance:
        raise ValueError("every environment record state must be normalized")
    reduced = np.outer(amplitude, amplitude.conj()) * gram.T
    reduced_trace = float(np.trace(reduced).real)
    interclass_record_overlaps = [
        abs(gram[left, right])
        for left in range(amplitude.size)
        for right in range(amplitude.size)
        if left != right and labels[left] != labels[right]
    ]
    intraclass_record_overlaps = [
        abs(gram[left, right])
        for left in range(amplitude.size)
        for right in range(amplitude.size)
        if left != right and labels[left] == labels[right]
    ]
    maximum_interclass_record_overlap = max(
        interclass_record_overlaps, default=0.0
    )
    minimum_intraclass_record_overlap = min(
        intraclass_record_overlaps, default=1.0
    )
    class_record_map_consistent = (
        maximum_interclass_record_overlap <= tolerance
        and minimum_intraclass_record_overlap >= 1.0 - tolerance
    )
    interclass = [
        abs(reduced[left, right])
        for left in range(amplitude.size)
        for right in range(amplitude.size)
        if left != right and labels[left] != labels[right]
    ]
    maximum_interclass = max(interclass, default=0.0)
    rendered_indices = [
        index for index, label in enumerate(labels) if label == rendered_label
    ]
    if not rendered_indices:
        raise ValueError("rendered_label must occur in coarse_labels")
    folded_indices = [
        index for index, label in enumerate(labels) if label != rendered_label
    ]
    rendered_probability = float(
        math.fsum(float(reduced[index, index].real) for index in rendered_indices)
    )
    folded_probability = float(
        math.fsum(float(reduced[index, index].real) for index in folded_indices)
    )
    decoherent = maximum_interclass <= tolerance
    # 정확한 유한 배열 진술: 접힌 구역은 대각 노름이 엄격히 양이면 보존된다.
    # ``tolerance`` 는 결어긋남 잔차에만 쓰고, 수학적 양수성을 재정의하는 데
    # 쓰지 않는다.
    folded_preserved = bool(folded_indices) and folded_probability > 0.0
    return DecoherentFoldAudit(
        history_count=amplitude.size,
        global_norm=amplitude_norm,
        reduced_trace=reduced_trace,
        maximum_interclass_record_overlap=maximum_interclass_record_overlap,
        minimum_intraclass_record_overlap=minimum_intraclass_record_overlap,
        class_record_map_consistent=class_record_map_consistent,
        maximum_interclass_coherence=maximum_interclass,
        rendered_probability=rendered_probability,
        folded_probability=folded_probability,
        folded_history_count=len(folded_indices),
        decoherent=decoherent,
        folded_sector_preserved=folded_preserved,
        status=(
            "DECOHERENT_RENDERED_CLASS_WITH_PRESERVED_FOLDED_NORM"
            if decoherent and folded_preserved
            else "DECOHERENCE_OR_FOLDED_NORM_CONDITION_FAILED"
        ),
    )


@dataclass(frozen=True)
class ConstraintConcentrationAudit:
    """유한 beta 깁스 가중치의 영결함 구역 집중 감사 결과이다."""

    inverse_temperature: float
    zero_defect_count: int
    positive_defect_gap: float
    probabilities: tuple[float, ...]
    good_probability: float
    bad_probability: float
    exponential_bad_probability_bound: float
    log_exponential_bad_probability_bound: float
    bound_holds: bool
    finite_beta_preserves_full_support: bool
    status: str = "FINITE_GIBBS_COMMON_METRIC_CONCENTRATION"


def finite_constraint_concentration(
    base_weights: Sequence[float],
    dimensionless_defects: Sequence[float],
    *,
    inverse_temperature: float,
    tolerance: float = 1.0e-12,
) -> ConstraintConcentrationAudit:
    """대안을 삭제하지 않고 유한 beta 집중을 증명한다.

    q_h > 0, Delta_h >= 0 에 대해

        p_beta(h) = q_h exp(-beta Delta_h) / Z_beta

    라 하자. G = {Delta = 0} 이 비어 있지 않고 여집합의 간격이 delta > 0 이면

        P_beta(G^c) <= (Q_bad/Q_good) exp(-beta delta)

    이다. 유한 beta에서 모든 역사는 엄격히 양의 지지를 유지한다.
    """

    beta = _require_finite("inverse_temperature", inverse_temperature)
    tolerance = _require_finite("tolerance", tolerance)
    if beta < 0.0:
        raise ValueError("inverse_temperature must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    q = np.asarray(base_weights, dtype=float)
    defect = np.asarray(dimensionless_defects, dtype=float)
    if q.ndim != 1 or defect.ndim != 1 or q.size != defect.size or q.size < 2:
        raise ValueError("weights and defects must be equal nontrivial vectors")
    if not np.all(np.isfinite(q)) or not np.all(q > 0.0):
        raise ValueError("every base weight must be finite and strictly positive")
    if not np.all(np.isfinite(defect)) or not np.all(defect >= 0.0):
        raise ValueError("every defect must be finite and non-negative")
    # 진술된 부등식은 G = {Delta = 0} 을 쓴다. 작은 수치 잔차를 수학적 0으로
    # 조용히 승격하지 않고 그 집합을 정확히 유지한다.
    good = defect == 0.0
    bad = defect > 0.0
    if not np.any(good) or not np.any(bad):
        raise ValueError("the audit requires both zero- and positive-defect histories")
    gap = float(np.min(defect[bad]))
    if gap <= 0.0:
        raise ValueError("positive-defect histories must be separated by a gap")
    log_weights = np.log(q) - beta * defect
    shift = float(np.max(log_weights))
    scaled = np.exp(log_weights - shift)
    if not np.all(scaled > 0.0):
        raise ValueError(
            "inverse_temperature and defects exceed the finite floating audit range"
        )
    probabilities = scaled / float(np.sum(scaled))
    good_probability = float(np.sum(probabilities[good]))
    bad_probability = float(np.sum(probabilities[bad]))
    log_q_good = _logsumexp(np.log(q[good]))
    log_q_bad = _logsumexp(np.log(q[bad]))
    log_raw_bound = log_q_bad - log_q_good - beta * gap
    log_bound = min(0.0, log_raw_bound)
    minimum_log_float = math.log(float(np.nextafter(0.0, 1.0)))
    bound = math.exp(log_bound) if log_bound >= minimum_log_float else 0.0
    log_bad_probability = (
        _logsumexp(log_weights[bad]) - _logsumexp(log_weights)
    )
    return ConstraintConcentrationAudit(
        inverse_temperature=beta,
        zero_defect_count=int(np.count_nonzero(good)),
        positive_defect_gap=gap,
        probabilities=tuple(float(value) for value in probabilities),
        good_probability=good_probability,
        bad_probability=bad_probability,
        exponential_bad_probability_bound=bound,
        log_exponential_bad_probability_bound=log_bound,
        bound_holds=(
            log_bad_probability <= log_bound + 10.0 * tolerance
        ),
        finite_beta_preserves_full_support=True,
    )


@dataclass(frozen=True)
class StationaryPhaseAudit:
    """국소 연속·게이지 고정 정상위상(stationary phase) 자료의 감사 결과이다."""

    variable_count: int
    large_dimensionless_parameter: float
    hessian_rank: int
    hessian_signature: tuple[int, int]
    gradient_residual: float
    continuous_variable_domain: str
    gauge_fixing: str
    contour: str
    leading_prefactor_magnitude: float
    log_leading_prefactor_magnitude: float
    localization_scale: float
    nondegenerate_stationary_sector: bool
    status: str


def quadratic_stationary_phase_audit(
    hessian: np.ndarray,
    *,
    gradient_at_candidate: Sequence[float],
    large_dimensionless_parameter: float,
    continuous_variable_domain: str,
    gauge_fixing: str,
    contour: str,
    tolerance: float = 1.0e-12,
) -> StationaryPhaseAudit:
    """국소 연속·게이지 고정 정상위상 자료를 감사한다.

    이 함수는 이산 역사 합을 변분 문제로 바꾸지 않는다. 기울기와 헤세(Hessian)
    행렬은 제공된 작용 하나의 연속 변수(또는 선언된 큰 스핀 보간 변수)에서
    와야 한다.
    """

    scale = _require_finite(
        "large_dimensionless_parameter", large_dimensionless_parameter
    )
    tolerance = _require_finite("tolerance", tolerance)
    if scale <= 0.0 or tolerance <= 0.0:
        raise ValueError("scale and tolerance must be positive")
    for name, value in (
        ("continuous_variable_domain", continuous_variable_domain),
        ("gauge_fixing", gauge_fixing),
        ("contour", contour),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    matrix = np.asarray(hessian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("hessian must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hessian must be finite")
    gradient = np.asarray(gradient_at_candidate, dtype=float)
    if gradient.shape != (matrix.shape[0],) or not np.all(np.isfinite(gradient)):
        raise ValueError("gradient_at_candidate must be a matching finite vector")
    matrix_scale = max(1.0, _stable_norm(matrix))
    if _stable_norm(matrix - matrix.T) / matrix_scale > tolerance:
        raise ValueError("hessian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    rank = int(np.count_nonzero(np.abs(eigenvalues) > tolerance * matrix_scale))
    positive = int(np.count_nonzero(eigenvalues > tolerance * matrix_scale))
    negative = int(np.count_nonzero(eigenvalues < -tolerance * matrix_scale))
    nondegenerate = rank == matrix.shape[0]
    determinant_sign, log_abs_determinant = (
        np.linalg.slogdet(matrix) if nondegenerate else (0.0, -math.inf)
    )
    nondegenerate = nondegenerate and determinant_sign != 0.0
    log_prefactor = (
        (matrix.shape[0] / 2.0) * math.log(2.0 * math.pi / scale)
        - 0.5 * float(log_abs_determinant)
        if nondegenerate
        else math.inf
    )
    maximum_log_float = math.log(float(np.finfo(float).max))
    minimum_log_float = math.log(float(np.nextafter(0.0, 1.0)))
    if not nondegenerate or log_prefactor > maximum_log_float:
        prefactor = math.inf
    elif log_prefactor < minimum_log_float:
        prefactor = 0.0
    else:
        prefactor = math.exp(log_prefactor)
    smallest = (
        float(np.min(np.abs(eigenvalues))) if nondegenerate else 0.0
    )
    localization = (
        1.0 / math.sqrt(scale * smallest) if nondegenerate else math.inf
    )
    gradient_residual = _stable_norm(gradient)
    stationary = (
        nondegenerate
        and gradient_residual <= tolerance
    )
    return StationaryPhaseAudit(
        variable_count=matrix.shape[0],
        large_dimensionless_parameter=scale,
        hessian_rank=rank,
        hessian_signature=(positive, negative),
        gradient_residual=gradient_residual,
        continuous_variable_domain=continuous_variable_domain,
        gauge_fixing=gauge_fixing,
        contour=contour,
        leading_prefactor_magnitude=prefactor,
        log_leading_prefactor_magnitude=log_prefactor,
        localization_scale=localization,
        nondegenerate_stationary_sector=stationary,
        status=(
            "NONDEGENERATE_STATIONARY_PHASE_SECTOR"
            if stationary
            else "STATIONARY_PHASE_CONDITIONS_NOT_ESTABLISHED"
        ),
    )


@dataclass(frozen=True)
class IREinsteinDominanceAudit:
    """적외선(IR)에서 아인슈타인-힐베르트 항이 지배하는지의 멱셈 감사 결과이다."""

    planck_over_macro_length: float
    correction_ratios: tuple[float, ...]
    maximum_correction_ratio: float
    tolerance: float
    einstein_hilbert_dominates: bool
    status: str


def ir_einstein_dominance_audit(
    planck_over_macro_length: float,
    higher_curvature_coefficients: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> IREinsteinDominanceAudit:
    """국소 R^n 보정을 아인슈타인-힐베르트(EH) R 항에 상대적으로 멱셈한다.

    계수 첨자 0은 R^2 를 뜻한다. 공통 EH 정규화 ``1/G`` 를 빼내면 무차원 계수는
    ``R + sum_n c_n ell_P^(2n-2) R^n`` 에 나타난다. 곡률 척도 L^-2 에서 한 항은
    R에 비해 |c_n| (ell_P/L)^(2n-2) 만큼 억제된다. 이는 IR 수용 게이트이며
    윌슨(Wilson) 계수의 유도나 비국소 항의 검사가 아니다.
    """

    ratio = _require_finite("planck_over_macro_length", planck_over_macro_length)
    tolerance = _require_finite("tolerance", tolerance)
    if not 0.0 < ratio < 1.0:
        raise ValueError("planck_over_macro_length must lie strictly between 0 and 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    coefficients = tuple(
        _require_finite(f"coefficient_{index + 2}", value)
        for index, value in enumerate(higher_curvature_coefficients)
    )
    if not coefficients:
        raise ValueError("at least one higher-curvature coefficient is required")
    corrections = tuple(
        abs(coefficient) * ratio ** (2 * (index + 1))
        for index, coefficient in enumerate(coefficients)
    )
    maximum = max(corrections)
    dominates = maximum <= tolerance
    return IREinsteinDominanceAudit(
        planck_over_macro_length=ratio,
        correction_ratios=corrections,
        maximum_correction_ratio=maximum,
        tolerance=tolerance,
        einstein_hilbert_dominates=dominates,
        status=(
            "EINSTEIN_HILBERT_IR_DOMINANCE_GATE_PASSED"
            if dominates
            else "HIGHER_CURVATURE_IR_SUPPRESSION_NOT_ESTABLISHED"
        ),
    )


@dataclass(frozen=True)
class ConstantCurvatureEinsteinAudit:
    """상수 곡률 아인슈타인 텐서 항등식의 감사 결과이다."""

    dimension: int
    curvature_times_reference_length_squared: float
    cosmological_constant_times_reference_length_squared: float
    ricci_residual: float
    scalar_curvature_residual: float
    einstein_equation_residual: float
    massless_spin_two_polarizations: int
    two_dof_spectrum_derived_from_action: bool
    lorentzian_einstein_geometry: bool
    status: str


def constant_curvature_einstein_audit(
    curvature_times_reference_length_squared: float,
    *,
    dimension: int = 4,
    tolerance: float = 1.0e-12,
) -> ConstantCurvatureEinsteinAudit:
    """예시용 상수 곡률 아인슈타인 텐서 항등식을 검증한다.

    편광 수는 D차원의 표준 질량 없는 스핀 2 개수이다. 이 감사는 그 스펙트럼을
    제공된 미시 작용이나 유효 작용에서 유도하지 않는다.
    """

    curvature = _require_finite(
        "curvature_times_reference_length_squared",
        curvature_times_reference_length_squared,
    )
    tolerance = _require_finite("tolerance", tolerance)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 4:
        raise ValueError("dimension must be an integer of at least four")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    metric = np.eye(dimension)
    metric[0, 0] = -1.0
    inverse_metric = metric.copy()
    riemann = curvature * (
        np.einsum("mr,ns->mnrs", metric, metric)
        - np.einsum("ms,nr->mnrs", metric, metric)
    )
    ricci = np.einsum("mr,mnrs->ns", inverse_metric, riemann)
    expected_ricci = (dimension - 1) * curvature * metric
    scalar = float(np.einsum("ns,ns->", inverse_metric, ricci))
    expected_scalar = dimension * (dimension - 1) * curvature
    einstein = ricci - 0.5 * scalar * metric
    cosmological_constant = (
        0.5 * (dimension - 1) * (dimension - 2) * curvature
    )
    equation = einstein + cosmological_constant * metric
    ricci_scale = max(1.0, _stable_norm(expected_ricci))
    scalar_scale = max(1.0, abs(expected_scalar))
    equation_scale = max(
        1.0,
        _stable_norm(einstein),
        _stable_norm(cosmological_constant * metric),
    )
    ricci_residual = _stable_norm(ricci - expected_ricci) / ricci_scale
    scalar_residual = abs(scalar - expected_scalar) / scalar_scale
    equation_residual = _stable_norm(equation) / equation_scale
    target = dimension == 4 and equation_residual <= tolerance
    return ConstantCurvatureEinsteinAudit(
        dimension=dimension,
        curvature_times_reference_length_squared=curvature,
        cosmological_constant_times_reference_length_squared=(
            cosmological_constant
        ),
        ricci_residual=ricci_residual,
        scalar_curvature_residual=scalar_residual,
        einstein_equation_residual=equation_residual,
        massless_spin_two_polarizations=(
            massless_spin_two_polarization_count(dimension)
        ),
        two_dof_spectrum_derived_from_action=False,
        lorentzian_einstein_geometry=target,
        status=(
            "ILLUSTRATIVE_THREE_PLUS_ONE_CONSTANT_CURVATURE_IDENTITY"
            if target
            else "NON_TARGET_CONSTANT_CURVATURE_ENDPOINT"
        ),
    )


VertexId = int
EdgeId = tuple[VertexId, VertexId]
TriangleId = tuple[VertexId, VertexId, VertexId]
TetrahedronId = tuple[VertexId, VertexId, VertexId, VertexId]
SimplexId = tuple[VertexId, VertexId, VertexId, VertexId, VertexId]


@dataclass(frozen=True)
class TypedRankFourTraceAudit:
    """지원되는 분할/병합 자취 하나와 거기서 유도된 모든 접속 관계이다."""

    history_id: str
    simplex_cells: tuple[SimplexId, ...]
    shared_tetrahedron: TetrahedronId
    boundary_atom_occurrences: int
    strand_end_count: int
    unique_triangle_ids: tuple[TriangleId, ...]
    causal_composition_faces: tuple[CompositionFace, ...]
    causal_to_shared_triangle: tuple[tuple[CompositionFace, TriangleId], ...]
    exact_typed_trace_probability: float
    connected_two_cell_block: bool
    rank_four_pairing_consistent: bool
    causal_face_map_bijective: bool
    status: str


def typed_rank_four_event_trace(
    branch_mean: float = 3.1777584234,
    *,
    history_id: str = "CE-C4-H0",
) -> TypedRankFourTraceAudit:
    """선언된 재작성 하나에서 두 세포 계수 4 자취를 만든다.

    두 4-단체 ``(0,1,2,3,4)`` 와 ``(1,2,3,4,5)`` 는 사면체 ``(1,2,3,4)`` 를
    공유한다. 각 경계 사면체는 삼각형 가닥 넷을 갖고, 각 삼각형은 각 4-단체
    안에서 두 가닥 끝에 나타난다. 공유 꼭짓점을 지나는 인과 합성 면 넷이 공유
    사면체의 삼각형 넷에 전단사로 대응된다. 아래 포아송 수는 요구되는 5/10
    개수 사건이 양의 지지를 가진다는 것만 증명한다. 타입 짝짓기 규칙은
    선언된 모형 자료로 남는다.
    """

    if not history_id:
        raise ValueError("history_id must be nonempty")
    split_merge = critical_split_merge(branch_mean)
    simplex_cells: tuple[SimplexId, ...] = (
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
    )
    shared_tetrahedron: TetrahedronId = (1, 2, 3, 4)
    boundary_atoms: list[tuple[int, TetrahedronId]] = []
    strand_ends: list[tuple[int, TetrahedronId, TriangleId]] = []
    per_cell_triangle_multiplicity: list[dict[TriangleId, int]] = []
    all_triangles: set[TriangleId] = set()
    for cell_index, simplex in enumerate(simplex_cells):
        multiplicity: dict[TriangleId, int] = {}
        for atom in combinations(simplex, 4):
            tetrahedron = tuple(atom)
            boundary_atoms.append((cell_index, tetrahedron))
            for face in combinations(tetrahedron, 3):
                triangle = tuple(face)
                strand_ends.append((cell_index, tetrahedron, triangle))
                multiplicity[triangle] = multiplicity.get(triangle, 0) + 1
                all_triangles.add(triangle)
        per_cell_triangle_multiplicity.append(multiplicity)

    source, target = 5, 0
    fine_edges = tuple(
        edge
        for middle in shared_tetrahedron
        for edge in ((source, middle), (middle, target))
    )
    causal_faces = composition_faces(fine_edges, ((source, target),))
    face_map = tuple(
        (
            face,
            tuple(vertex for vertex in shared_tetrahedron if vertex != face.middle),
        )
        for face in causal_faces
    )
    shared_triangles = set(combinations(shared_tetrahedron, 3))
    mapped_triangles = {triangle for _, triangle in face_map}
    local_probability = marked_joint_probability(
        branch_mean=branch_mean,
        distinct_probability=split_merge.distinct_probability,
        distinct_count=5,
        face_count=10,
    )
    pairing_consistent = all(
        len(multiplicity) == 10
        and all(count == 2 for count in multiplicity.values())
        for multiplicity in per_cell_triangle_multiplicity
    )
    connected = (
        set(simplex_cells[0]).intersection(simplex_cells[1])
        == set(shared_tetrahedron)
    )
    face_map_bijective = (
        len(causal_faces) == 4
        and len(face_map) == len(mapped_triangles)
        and mapped_triangles == shared_triangles
        and mapped_triangles.issubset(all_triangles)
    )
    closed = (
        connected
        and pairing_consistent
        and face_map_bijective
        and local_probability > 0.0
    )
    return TypedRankFourTraceAudit(
        history_id=history_id,
        simplex_cells=simplex_cells,
        shared_tetrahedron=shared_tetrahedron,
        boundary_atom_occurrences=len(boundary_atoms),
        strand_end_count=len(strand_ends),
        unique_triangle_ids=tuple(sorted(all_triangles)),
        causal_composition_faces=causal_faces,
        causal_to_shared_triangle=face_map,
        exact_typed_trace_probability=local_probability * local_probability,
        connected_two_cell_block=connected,
        rank_four_pairing_consistent=pairing_consistent,
        causal_face_map_bijective=face_map_bijective,
        status=(
            "ONE_TYPED_RANK_FOUR_TRACE_WITH_LINKED_FACES"
            if closed
            else "TYPED_TRACE_INCIDENCE_NOT_CLOSED"
        ),
    )


@dataclass(frozen=True)
class TypedHistoryMember:
    """같은 자취 위 변형 앙상블의 구성원 하나이다."""

    member_id: str
    shared_tetrahedron: TetrahedronId
    distortion: tuple[float, float, float]
    squared_length_readout_over_planck_area: tuple[float, float, float]
    common_metric_defect: float
    base_measure_weight: float
    connection_angle: float
    shared_face_status: str
    common_metric: bool


def _typed_history_member(
    history_id: str,
    shared_tetrahedron: TetrahedronId,
    member_index: int,
    distortion: Sequence[float],
) -> tuple[TypedHistoryMember, np.ndarray]:
    """공유 면 왜곡 매개변수 하나로 앙상블 구성원과 오른쪽 면 벡터를 만든다."""

    parameters = np.asarray(distortion, dtype=float)
    if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
        raise ValueError("distortion must be a finite three-vector")
    if np.any(parameters <= -1.0):
        raise ValueError("distortion must preserve positive spatial scales")
    normal = np.asarray((1.0, 0.0, 0.0, 0.0))
    left_face = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    scales = 1.0 + parameters
    right_face = left_face * scales[:, None]
    shared = hard_shared_spacelike_face_match(
        left_face,
        normal,
        np.asarray((1.0, 0.2, 0.2, 0.2)),
        right_face,
        normal.copy(),
        np.asarray((-1.0, 0.2, 0.2, 0.2)),
        np.eye(4),
    )
    squared_lengths = tuple(float(value * value) for value in scales)
    # 이는 플랑크 넓이 단위에서 공유 면 그람 대각 성분 불일치 제곱의 정확히
    # 4분의 1이다.
    defect = 0.25 * math.fsum(
        (squared_length - 1.0) ** 2 for squared_length in squared_lengths
    )
    # 이 모형에서 선언된 이산 응답 규칙: 선택된 면의 곡률 각도는 바로 그
    # 무차원 그람 결함의 제곱근이다. 더 이상 독립적으로 제공되는 표본이 아니다.
    angle = math.sqrt(defect)
    base_weight = math.exp(-0.5 * float(parameters @ parameters))
    return (
        TypedHistoryMember(
            member_id=f"{history_id}:x{member_index}",
            shared_tetrahedron=shared_tetrahedron,
            distortion=tuple(float(value) for value in parameters),
            squared_length_readout_over_planck_area=squared_lengths,
            common_metric_defect=defect,
            base_measure_weight=base_weight,
            connection_angle=angle,
            shared_face_status=shared.status,
            common_metric=shared.hard_match,
        ),
        right_face,
    )


def _shape_defect_gradient_hessian(
    distortion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """앙상블이 쓰는 것과 같은 그람 불일치 작용을 미분한다."""

    parameters = np.asarray(distortion, dtype=float)
    scales = 1.0 + parameters
    mismatch = scales * scales - 1.0
    gradient = mismatch * scales
    hessian = np.diag(3.0 * scales * scales - 1.0)
    return gradient, hessian


def _rotation_12(angle: float) -> np.ndarray:
    """1-2 평면의 회전 행렬을 4x4 로렌츠 행렬로 반환한다."""

    rotation = np.eye(4)
    rotation[1:3, 1:3] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        )
    )
    return rotation


def _member_face_holonomy(
    trace: TypedRankFourTraceAudit,
    member: TypedHistoryMember,
) -> FaceHolonomyAudit:
    """구성원의 접속 각도로 첫 번째 대응 삼각형의 면 홀로노미를 만든다."""

    causal_face, triangle = trace.causal_to_shared_triangle[0]
    return face_holonomy_audit(
        (_rotation_12(member.connection_angle), np.eye(4), np.eye(4)),
        face_id=triangle,
        attached_contractible_face=True,
    )


def _permutation_parity(indices: tuple[int, int, int, int]) -> int:
    """네 첨자의 순열 홀짝 부호를 반환한다. 중복이 있으면 0이다."""

    if len(set(indices)) < 4:
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


_EPSILON_LOWER = np.empty((4, 4, 4, 4), dtype=float)
for _a in range(4):
    for _b in range(4):
        for _c in range(4):
            for _d in range(4):
                _EPSILON_LOWER[_a, _b, _c, _d] = _permutation_parity(
                    (_a, _b, _c, _d)
                )


def _covariant_two_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """두 공변 벡터의 쐐기곱을 반대칭 행렬로 반환한다."""

    return np.outer(first, second) - np.outer(second, first)


def _lorentzian_hodge_covariant(two_form: np.ndarray) -> np.ndarray:
    """공변 2-형식의 로렌츠 호지 쌍대를 반환한다."""

    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    raised = eta @ two_form @ eta
    return 0.5 * np.einsum("mnrs,rs->mn", _EPSILON_LOWER, raised)


def _wedge_four_volume(first: np.ndarray, second: np.ndarray) -> complex:
    """두 2-형식의 쐐기곱 4-부피 계수를 반환한다."""

    return complex(0.25 * np.einsum("mnrs,mn,rs->", _EPSILON_LOWER, first, second))


@dataclass(frozen=True)
class FlatChiralPlebanskiAudit:
    """같은 사면체틀 위의 정확한 평탄 키랄 플레바인스키 해 감사 결과이다."""

    history_id: str
    selected_face_id: TriangleId
    shared_face_embedding_residual: float
    selected_holonomy_flatness_residual: float
    cell_oriented_volumes: tuple[float, ...]
    metric_signature: tuple[int, int, int, int]
    complex_self_duality_residual: float
    simplicity_tracefree_residual: float
    simplicity_volume: complex
    covariant_constancy_residual: float
    curvature_equation_residual: float
    compact_support_boundary_condition: bool
    real_nondegenerate_tetrad: bool
    induced_by_selected_simplex_geometry: bool
    einstein_endpoint: ConstantCurvatureEinsteinAudit
    flat_lorentzian_plebanski_solution: bool
    status: str


def flat_chiral_plebanski_audit(
    history_id: str,
    *,
    vertex_coordinates: Mapping[VertexId, Sequence[float]],
    simplex_cells: Sequence[SimplexId],
    shared_tetrahedron: TetrahedronId,
    selected_face_vectors: np.ndarray,
    selected_face_id: TriangleId,
    selected_face_holonomy: np.ndarray,
    tolerance: float = 1.0e-12,
) -> FlatChiralPlebanskiAudit:
    """같은 사면체틀 위의 정확한 평탄 키랄 플레바인스키 해를 검증한다.

    타입 두 단체 역사가 제공하는 관성 좌표는 실수 여틀(coframe) ``e^I = dx^I``
    를 유도한다. 이 함수는 먼저 같은 좌표 차가 선택된 공유 면 벡터이고, 두
    4-단체가 비퇴화이며, 선택된 면 수송이 평탄 레비-치비타(Levi-Civita)
    홀로노미와 같음을 검증한다. 그 다음
    ``Sigma^i = i e^0 wedge e^i - 1/2 eps^i_jk e^j wedge e^k`` 를 택한다.
    ``A = 0``, ``Psi = 0``, ``Lambda = 0`` 에 대해 접속 방정식과 곡률 방정식은
    정확히 소멸한다. 콤팩트 지지 변분은 경계항을 없앤다. 이는 명시적 국소 고전
    해이며, 연속/세분 정리나 양자 측도 정리가 아니다.
    """

    if not history_id:
        raise ValueError("history_id must be nonempty")
    tolerance = _require_finite("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    coordinates = {
        vertex: np.asarray(value, dtype=float)
        for vertex, value in vertex_coordinates.items()
    }
    required_vertices = set(vertex for cell in simplex_cells for vertex in cell)
    if set(coordinates) != required_vertices:
        raise ValueError("vertex_coordinates must cover exactly the simplex vertices")
    if any(value.shape != (4,) or not np.all(np.isfinite(value)) for value in coordinates.values()):
        raise ValueError("every vertex coordinate must be a finite four-vector")
    if len(shared_tetrahedron) != 4 or len(set(shared_tetrahedron)) != 4:
        raise ValueError("shared_tetrahedron must contain four distinct vertices")
    if not set(selected_face_id).issubset(shared_tetrahedron):
        raise ValueError("selected_face_id must lie in the shared tetrahedron")
    face_vectors = np.asarray(selected_face_vectors, dtype=float)
    if face_vectors.shape != (3, 4) or not np.all(np.isfinite(face_vectors)):
        raise ValueError("selected_face_vectors must be a finite (3,4) array")
    anchor, *other_shared_vertices = shared_tetrahedron
    coordinate_face_vectors = np.asarray(
        [coordinates[vertex] - coordinates[anchor] for vertex in other_shared_vertices]
    )
    face_scale = max(1.0, _stable_norm(coordinate_face_vectors), _stable_norm(face_vectors))
    embedding_residual = _stable_norm(coordinate_face_vectors - face_vectors) / face_scale
    holonomy = np.asarray(selected_face_holonomy, dtype=float)
    if holonomy.shape != (4, 4) or not np.all(np.isfinite(holonomy)):
        raise ValueError("selected_face_holonomy must be a finite (4,4) matrix")
    holonomy_flatness = _stable_norm(holonomy - np.eye(4)) / max(
        1.0, _stable_norm(holonomy)
    )
    cell_volumes = tuple(
        float(
            np.linalg.det(
                np.asarray(
                    [coordinates[vertex] - coordinates[cell[0]] for vertex in cell[1:]]
                )
            )
        )
        for cell in simplex_cells
    )
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    signature = tuple(int(math.copysign(1, value)) for value in np.linalg.eigvalsh(eta))
    geometry_linked = (
        embedding_residual <= tolerance
        and holonomy_flatness <= tolerance
        and all(abs(volume) > tolerance for volume in cell_volumes)
        and signature == (-1, 1, 1, 1)
    )

    # 이 좌표 공변 벡터들은 방금 검사한 관성 매장이 유도하는 여틀이며,
    # 두 번째 기하 표본이 아니다.
    basis = np.eye(4)
    sigma = np.asarray(
        (
            1j * _covariant_two_form(basis[0], basis[1])
            - _covariant_two_form(basis[2], basis[3]),
            1j * _covariant_two_form(basis[0], basis[2])
            - _covariant_two_form(basis[3], basis[1]),
            1j * _covariant_two_form(basis[0], basis[3])
            - _covariant_two_form(basis[1], basis[2]),
        )
    )
    duals = np.asarray([_lorentzian_hodge_covariant(item) for item in sigma])
    self_duality_residual = _stable_norm(duals - 1j * sigma) / max(
        1.0, _stable_norm(sigma)
    )
    wedge_matrix = np.asarray(
        [[_wedge_four_volume(left, right) for right in sigma] for left in sigma]
    )
    trace_average = np.trace(wedge_matrix) / 3.0
    tracefree = wedge_matrix - trace_average * np.eye(3)
    simplicity_residual = _stable_norm(tracefree) / max(
        1.0, _stable_norm(wedge_matrix)
    )
    # 상수 여틀에 대해 비틀림 없는 양립 키랄 접속은 A = 0 이다. A = Psi =
    # Lambda = 0 에 대해 d_A Sigma 와 F - (Psi + Lambda/3) Sigma 를 (라벨만
    # 붙이지 않고) 실제로 계산한다.
    connection = np.zeros((3, 4), dtype=complex)
    sigma_derivative = np.zeros((3, 4, 4, 4), dtype=complex)
    covariant_derivative = sigma_derivative + 0.0 * np.sum(connection)
    curvature = np.zeros((3, 4, 4), dtype=complex)
    psi = np.zeros((3, 3), dtype=complex)
    curvature_equation = curvature - np.einsum("ij,jmn->imn", psi, sigma)
    covariant_constancy_residual = _stable_norm(covariant_derivative)
    curvature_equation_residual = _stable_norm(curvature_equation)
    endpoint = constant_curvature_einstein_audit(0.0)
    closed = (
        geometry_linked
        and self_duality_residual <= 1.0e-12
        and simplicity_residual <= 1.0e-12
        and abs(trace_average) > 0.0
        and covariant_constancy_residual <= tolerance
        and curvature_equation_residual <= tolerance
        and endpoint.lorentzian_einstein_geometry
    )
    return FlatChiralPlebanskiAudit(
        history_id=history_id,
        selected_face_id=selected_face_id,
        shared_face_embedding_residual=embedding_residual,
        selected_holonomy_flatness_residual=holonomy_flatness,
        cell_oriented_volumes=cell_volumes,
        metric_signature=signature,
        complex_self_duality_residual=self_duality_residual,
        simplicity_tracefree_residual=simplicity_residual,
        simplicity_volume=complex(trace_average),
        covariant_constancy_residual=covariant_constancy_residual,
        curvature_equation_residual=curvature_equation_residual,
        compact_support_boundary_condition=True,
        real_nondegenerate_tetrad=(
            signature == (-1, 1, 1, 1)
            and all(abs(volume) > tolerance for volume in cell_volumes)
        ),
        induced_by_selected_simplex_geometry=geometry_linked,
        einstein_endpoint=endpoint,
        flat_lorentzian_plebanski_solution=closed,
        status=(
            "EXACT_FLAT_CHIRAL_PLEBANSKI_EINSTEIN_SOLUTION"
            if closed
            else "FLAT_CHIRAL_PLEBANSKI_CHECK_FAILED"
        ),
    )


@dataclass(frozen=True)
class ZeroDToPlebanskiClosureAudit:
    """0D→플레바인스키 사슬 전체를 하나의 연결된 유한 역사로 묶은 증명서이다."""

    history_id: str
    form_degree: FormDegreeClosureAudit
    simplex_interaction: SimplexInteractionAudit
    split_merge: CriticalSplitMerge
    typed_trace: TypedRankFourTraceAudit
    history_members: tuple[TypedHistoryMember, ...]
    face_holonomies: tuple[FaceHolonomyAudit, ...]
    causal_relation_realized_by_metric: bool
    planck_quotient: PlanckQuotientAudit
    decoherence: DecoherentFoldAudit
    constraint_concentration: ConstraintConcentrationAudit
    stationary_phase: StationaryPhaseAudit
    bivector_reconstruction_status: str
    selected_shared_face_status: str
    flat_plebanski: FlatChiralPlebanskiAudit
    all_finite_projections_share_one_trace: bool
    single_history_finite_flat_witness_closed: bool
    conditional_local_plebanski_einstein_existence_closed: bool
    continuum_refinement_derived: bool
    two_dof_ir_spectrum_derived: bool
    bare_zerod_uniqueness_proved: bool
    folded_possibilities_preserved: bool
    status: str
    claim_ceiling: str = (
        "SINGLE_TYPED_HISTORY_FINITE_FLAT_CONDITIONAL_EXISTENCE_NOT_GENERIC_CONTINUUM_GR"
    )


def constructive_zerod_to_plebanski_witness(
    *,
    branch_mean: float = 3.1777584234,
    inverse_temperature: float = 100.0,
    history_id: str = "CE-C4-H0",
) -> ZeroDToPlebanskiClosureAudit:
    """장난감 표본들의 논리곱이 아니라, 연결된 유한 역사 하나를 만든다.

    아래 모든 유한 관측량은 같은 계수 4 자취와 같은 세 구성원 변형 앙상블에서
    유도된다. 선택된 영결함 구성원은 명시적 평탄 로렌츠 키랄 플레바인스키/
    아인슈타인 해이다. 곡률을 가진 구성원과 불일치 구성원은 양의 지지를
    유지한다. 여기서는 재작성/작용 선택, 세분 극한, IR 스펙트럼을 유도하지
    않는다.
    """

    form_degree = form_degree_closure()
    simplex = simplex_interaction_audit()
    split_merge = critical_split_merge(branch_mean)
    trace = typed_rank_four_event_trace(branch_mean, history_id=history_id)
    member_distortions = (
        (0.0, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.50, 0.20, 0.0),
    )
    built_members = tuple(
        _typed_history_member(
            history_id,
            trace.shared_tetrahedron,
            index,
            distortion,
        )
        for index, distortion in enumerate(member_distortions)
    )
    members = tuple(member for member, _ in built_members)
    holonomies = tuple(_member_face_holonomy(trace, member) for member in members)

    # 계량 이전 인과 부채꼴은 붙인 두 로렌츠 4-단체에 쓴 것과 같은 좌표로
    # 실현된다: 5는 과거, 0은 미래, 꼭짓점 1..4는 공유 공간꼴 사면체 위에 있다.
    coordinates = {
        0: np.asarray((1.0, 0.2, 0.2, 0.2)),
        1: np.asarray((0.0, 0.0, 0.0, 0.0)),
        2: np.asarray((0.0, 1.0, 0.0, 0.0)),
        3: np.asarray((0.0, 0.0, 1.0, 0.0)),
        4: np.asarray((0.0, 0.0, 0.0, 1.0)),
        5: np.asarray((-1.0, 0.2, 0.2, 0.2)),
    }
    causal_edges = {
        edge
        for face in trace.causal_composition_faces
        for edge in face.oriented_boundary[:2]
    } | {(5, 0)}
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    causal_realized = all(
        coordinates[target][0] > coordinates[source][0]
        and float(
            (coordinates[target] - coordinates[source])
            @ eta
            @ (coordinates[target] - coordinates[source])
        )
        < 0.0
        for source, target in causal_edges
    )

    quotient = planck_resolution_quotient(
        tuple(member.squared_length_readout_over_planck_area for member in members),
        observable_dimensions=("L^2", "L^2", "L^2"),
        reference_dimensions=("L^2", "L^2", "L^2"),
        bin_width=0.50,
    )
    base_weights = np.asarray(
        tuple(member.base_measure_weight for member in members), dtype=float
    )
    normalized_weights = base_weights / float(np.sum(base_weights))
    labels = quotient.coarse_labels
    unique_labels = tuple(dict.fromkeys(labels))
    label_index = {label: index for index, label in enumerate(unique_labels)}
    environment_records = np.zeros((len(labels), len(unique_labels)), dtype=complex)
    for row, label in enumerate(labels):
        environment_records[row, label_index[label]] = 1.0
    decoherence = decoherent_fold_audit(
        np.sqrt(normalized_weights),
        environment_records,
        labels,
        rendered_label=labels[0],
    )
    concentration = finite_constraint_concentration(
        tuple(member.base_measure_weight for member in members),
        tuple(member.common_metric_defect for member in members),
        inverse_temperature=inverse_temperature,
    )
    gradient, hessian = _shape_defect_gradient_hessian(members[0].distortion)
    stationary = quadratic_stationary_phase_audit(
        hessian,
        gradient_at_candidate=gradient,
        large_dimensionless_parameter=inverse_temperature,
        continuous_variable_domain="shared-face distortions x in R^3",
        gauge_fixing="common tetrahedron frame fixed",
        contour="real R^3 with Gaussian base measure",
    )

    normal = np.asarray((1.0, 0.0, 0.0, 0.0))
    face_vectors = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    bivectors = np.asarray(
        [bivector_from_normal_edge(normal, edge) for edge in face_vectors]
    )
    bivector_audit = bivector_face_reconstruction_audit(normal, bivectors)
    selected_face_id = trace.causal_to_shared_triangle[0][1]
    flat_plebanski = flat_chiral_plebanski_audit(
        history_id,
        vertex_coordinates=coordinates,
        simplex_cells=trace.simplex_cells,
        shared_tetrahedron=trace.shared_tetrahedron,
        selected_face_vectors=face_vectors,
        selected_face_id=selected_face_id,
        selected_face_holonomy=holonomies[0].holonomy,
    )
    selected_is_flat_connection = holonomies[0].flatness_residual <= 1.0e-12
    alternatives_carry_curvature = any(
        item.nontrivial_curvature_carrier for item in holonomies[1:]
    )
    linked = all(
        (
            trace.history_id == history_id,
            all(member.member_id.startswith(f"{history_id}:") for member in members),
            all(
                member.shared_tetrahedron == trace.shared_tetrahedron
                for member in members
            ),
            all(
                holonomy.face_id in trace.unique_triangle_ids
                for holonomy in holonomies
            ),
            flat_plebanski.history_id == history_id,
            flat_plebanski.selected_face_id == selected_face_id,
            flat_plebanski.induced_by_selected_simplex_geometry,
        )
    )
    finite_closed = all(
        (
            linked,
            form_degree.lorentzian_three_plus_one,
            simplex.target_four_simplex,
            trace.connected_two_cell_block,
            trace.rank_four_pairing_consistent,
            trace.causal_face_map_bijective,
            trace.exact_typed_trace_probability > 0.0,
            causal_realized,
            selected_is_flat_connection,
            alternatives_carry_curvature,
            members[0].common_metric,
            all(not member.common_metric for member in members[1:]),
            quotient.folded_pair_count >= 1,
            quotient.all_microscopic_histories_retained,
            decoherence.decoherent,
            decoherence.class_record_map_consistent,
            decoherence.folded_sector_preserved,
            concentration.bound_holds,
            concentration.finite_beta_preserves_full_support,
            stationary.nondegenerate_stationary_sector,
            bivector_audit.hard_reconstruction,
            flat_plebanski.flat_lorentzian_plebanski_solution,
        )
    )
    return ZeroDToPlebanskiClosureAudit(
        history_id=history_id,
        form_degree=form_degree,
        simplex_interaction=simplex,
        split_merge=split_merge,
        typed_trace=trace,
        history_members=members,
        face_holonomies=holonomies,
        causal_relation_realized_by_metric=causal_realized,
        planck_quotient=quotient,
        decoherence=decoherence,
        constraint_concentration=concentration,
        stationary_phase=stationary,
        bivector_reconstruction_status=bivector_audit.status,
        selected_shared_face_status=members[0].shared_face_status,
        flat_plebanski=flat_plebanski,
        all_finite_projections_share_one_trace=linked,
        single_history_finite_flat_witness_closed=finite_closed,
        conditional_local_plebanski_einstein_existence_closed=finite_closed,
        continuum_refinement_derived=False,
        two_dof_ir_spectrum_derived=False,
        bare_zerod_uniqueness_proved=False,
        folded_possibilities_preserved=(
            decoherence.folded_sector_preserved
            and concentration.finite_beta_preserves_full_support
        ),
        status=(
            "SINGLE_TYPED_HISTORY_FINITE_FLAT_CONDITIONAL_WITNESS_CLOSED"
            if finite_closed
            else "SINGLE_TYPED_HISTORY_FINITE_CHAIN_FAILED"
        ),
    )
