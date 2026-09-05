"""플랑크(Planck) 렌더링 브리지를 위한 유한 우르반트케(Urbantke)/모양 맞춤(shape-matching) 감사.

정확한 결과는 의도적으로 좁다. SO(3) 틀 정렬 뒤, 양의 척도 공통 계량 궤도
하나에 속한 삼중항들은 어떤 블록킹 순서에서도 플레바인스키(Plebanski) 단순성을
유지한다. 일반적인 국소 단순 세포는 그렇지 않다. 중심화된 무작위 불일치는
통계적으로 감소할 수 있으나, 결맞은 불일치는 지속한다.

이는 유클리드화된 유한 대수 감사이며 로렌츠(Lorentz) 스핀폼(spin-foam) 모형이 아니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math

import numpy as np

from examples.physics.gravity.causal_face_simplicity import (
    geometric_self_dual_triple,
    plebanski_gram,
    simplicity_residual,
    wedge_scalar,
)

_PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2))


def _levi_civita(rank: int) -> np.ndarray:
    """주어진 계수의 레비-치비타(Levi-Civita) 기호 텐서를 반환한다."""

    out = np.zeros((rank,) * rank, dtype=int)
    for perm in itertools.permutations(range(rank)):
        inversions = sum(
            perm[i] > perm[j]
            for i in range(rank)
            for j in range(i + 1, rank)
        )
        out[perm] = -1 if inversions % 2 else 1
    return out


_EPSILON_3 = _levi_civita(3)
_EPSILON_4 = _levi_civita(4)


def _triple(value: np.ndarray, name: str = "triple") -> np.ndarray:
    """모양 (3, 6)의 유한 float 배열로 변환하고, 아니면 ValueError를 낸다."""

    array = np.asarray(value, dtype=float)
    if array.shape != (3, 6) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape (3, 6)")
    return array


def _matrices(value: np.ndarray) -> np.ndarray:
    """6성분 삼중항을 반대칭 4x4 행렬 셋으로 펼친다."""

    triple = _triple(value)
    out = np.zeros((3, 4, 4), dtype=float)
    for i in range(3):
        for component, (mu, nu) in zip(triple[i], _PAIR_INDEX):
            out[i, mu, nu] = component
            out[i, nu, mu] = -component
    return out


def urbantke_metric_density(value: np.ndarray) -> np.ndarray:
    """대칭 우르반트케 계량 밀도를 반환한다."""

    b = _matrices(value)
    density = np.einsum(
        "ijk,abcd,ima,jbc,kdn->mn",
        _EPSILON_3,
        _EPSILON_4,
        b,
        b,
        b,
        optimize=True,
    ) / 12.0
    return 0.5 * (density + density.T)


def normalized_urbantke_metric(value: np.ndarray) -> np.ndarray:
    """행렬식이 1인 양의 유클리드 등각 계량을 반환한다."""

    density = urbantke_metric_density(value)
    if float(np.min(np.linalg.eigvalsh(density))) <= 1.0e-12:
        raise ValueError("Urbantke metric must be positive and nondegenerate")
    return density / float(np.linalg.det(density)) ** 0.25


def conformal_metric_residual(first: np.ndarray, second: np.ndarray) -> float:
    """두 삼중항의 정규화된 우르반트케 계량 차이의 노름 절반을 반환한다."""

    first_metric = normalized_urbantke_metric(first)
    second_metric = normalized_urbantke_metric(second)
    return float(np.linalg.norm(first_metric - second_metric) / 2.0)


def cross_wedge_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """두 삼중항 사이의 쐐기곱 행렬 W_ij = first^i wedge second^j 를 반환한다."""

    first = _triple(first, "first")
    second = _triple(second, "second")
    return np.array(
        [
            [wedge_scalar(first[i], second[j]) for j in range(3)]
            for i in range(3)
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class InternalAlignmentAudit:
    """SO(3) 내부 정렬의 결과 묶음이다."""

    rotation: np.ndarray
    aligned_candidate: np.ndarray
    relative_scale: float
    orbit_residual: float
    metric_residual: float
    block_residual: float


def optimal_internal_alignment(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> InternalAlignmentAudit:
    """고유 SO(3) 극(polar) 인자로 candidate를 reference에 정렬한다."""

    reference = _triple(reference, "reference")
    candidate = _triple(candidate, "candidate")
    left, _, right_t = np.linalg.svd(cross_wedge_matrix(reference, candidate))
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    aligned = rotation @ candidate
    scale = float(np.sum(reference * aligned) / np.sum(reference * reference))
    orbit = float(np.linalg.norm(aligned - scale * reference) / np.linalg.norm(aligned))
    return InternalAlignmentAudit(
        rotation=rotation,
        aligned_candidate=aligned,
        relative_scale=scale,
        orbit_residual=orbit,
        metric_residual=conformal_metric_residual(reference, candidate),
        block_residual=simplicity_residual(reference + aligned),
    )


@dataclass(frozen=True)
class CommonMetricBlock:
    """양의 유클리드 계량 궤도 하나에 속한 단순 세포들의 블록 합이다."""

    blocked_triple: np.ndarray
    cell_count: int
    maximum_metric_residual: float
    maximum_orbit_residual: float
    blocked_simplicity_residual: float
    common_metric_orbit: bool


def common_metric_block(
    triples: Sequence[np.ndarray],
    *,
    tolerance: float = 1.0e-9,
) -> CommonMetricBlock:
    """양의 유클리드 계량 궤도 하나에 속한 단순 세포들을 정렬하고 합한다."""

    if not triples:
        raise ValueError("at least one triple is required")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    reference = _triple(triples[0], "triples[0]")
    if simplicity_residual(reference) > tolerance:
        raise ValueError("common metric closure requires Plebanski-simple triples")
    # 쌍별 계량 잔차를 계산하지 않는 단일항 경우에도 정리의 양의 비퇴화
    # 유클리드 가지를 검증한다.
    normalized_urbantke_metric(reference)
    if float(np.trace(plebanski_gram(reference))) <= 0.0:
        raise ValueError(
            "common metric closure requires the positive self-dual/orientation branch"
        )
    aligned = [reference]
    metric_residuals = [0.0]
    orbit_residuals = [0.0]
    for index, candidate in enumerate(triples[1:], start=1):
        candidate = _triple(candidate, f"triples[{index}]")
        if simplicity_residual(candidate) > tolerance:
            raise ValueError("common metric closure requires Plebanski-simple triples")
        normalized_urbantke_metric(candidate)
        if float(np.trace(plebanski_gram(candidate))) <= 0.0:
            raise ValueError(
                "common metric closure requires the positive self-dual/orientation branch"
            )
        audit = optimal_internal_alignment(reference, candidate)
        if audit.relative_scale <= 0.0:
            raise ValueError("common metric orbit requires positive relative scales")
        aligned.append(audit.aligned_candidate)
        metric_residuals.append(audit.metric_residual)
        orbit_residuals.append(audit.orbit_residual)
    common = max(metric_residuals) <= tolerance and max(orbit_residuals) <= tolerance
    if not common:
        raise ValueError("triples do not lie in one common conformal-metric orbit")
    blocked = np.sum(np.asarray(aligned), axis=0)
    return CommonMetricBlock(
        blocked_triple=blocked,
        cell_count=len(aligned),
        maximum_metric_residual=max(metric_residuals),
        maximum_orbit_residual=max(orbit_residuals),
        blocked_simplicity_residual=simplicity_residual(blocked),
        common_metric_orbit=True,
    )


def shape_matching_weight(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    metric_width: float,
    orbit_width: float,
    block_width: float,
) -> float:
    """계량·궤도·블록 단순성에 대한 부드러운 사영 가중치를 반환한다."""

    widths = (metric_width, orbit_width, block_width)
    if any(not math.isfinite(width) or width <= 0.0 for width in widths):
        raise ValueError("all widths must be finite and positive")
    audit = optimal_internal_alignment(reference, candidate)
    exponent = (
        (audit.metric_residual / metric_width) ** 2
        + (audit.orbit_residual / orbit_width) ** 2
        + (audit.block_residual / block_width) ** 2
    )
    return math.exp(-0.5 * exponent)


def repeated_coherent_mismatch_residual(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    repeats: int,
) -> float:
    """결맞은 불일치 하나를 반복해도 평균으로 사라지지 않음을 보인다."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    audit = optimal_internal_alignment(reference, candidate)
    return simplicity_residual(repeats * (reference + audit.aligned_candidate))


@dataclass(frozen=True)
class ShapeFluctuationScaling:
    """중심화된 약한 사면체틀 불일치의 유한 블록 억제 측정 결과이다."""

    sample_sizes: tuple[int, ...]
    mean_residuals: tuple[float, ...]
    fitted_power: float
    centered_noise_decreases: bool


@lru_cache(maxsize=16)
def centered_shape_fluctuation_scaling(
    *,
    sample_sizes: tuple[int, ...] = (8, 16, 32, 64),
    trial_count: int = 16,
    perturbation: float = 0.05,
    seed: int = 20_260_828,
) -> ShapeFluctuationScaling:
    """중심화된 약한 사면체틀(tetrad) 불일치의 유한 블록 억제를 측정한다."""

    if not sample_sizes or any(size < 2 for size in sample_sizes):
        raise ValueError("sample_sizes must contain integers of at least two")
    if trial_count < 1 or perturbation <= 0.0:
        raise ValueError("trial_count and perturbation must be positive")
    generator = np.random.default_rng(seed)
    reference = geometric_self_dual_triple(np.eye(4))
    means: list[float] = []
    for size in sample_sizes:
        trial_residuals: list[float] = []
        for _ in range(trial_count):
            noises = generator.normal(size=(size, 4, 4))
            noises -= np.mean(noises, axis=0, keepdims=True)
            blocked = np.zeros_like(reference)
            for noise in noises:
                tetrad = np.eye(4) + perturbation * noise
                candidate = geometric_self_dual_triple(tetrad)
                blocked += optimal_internal_alignment(reference, candidate).aligned_candidate
            trial_residuals.append(simplicity_residual(blocked))
        means.append(float(np.mean(trial_residuals)))
    fitted_power = float(np.polyfit(np.log(sample_sizes), np.log(means), 1)[0])
    return ShapeFluctuationScaling(
        sample_sizes=sample_sizes,
        mean_residuals=tuple(means),
        fitted_power=fitted_power,
        centered_noise_decreases=(means[-1] < means[0]),
    )


@dataclass(frozen=True)
class ShapeMatchingRGVerdict:
    """모양 맞춤 재규격화(RG) 감사의 요약이다."""

    exact_common_orbit_closed: bool
    exact_common_orbit_associative: bool
    coherent_mismatch_suppressed_by_blocking: bool
    centered_mismatch_decreases: bool
    centered_mismatch_fitted_power: float
    remaining_obligation: str


def shape_matching_rg_verdict() -> ShapeMatchingRGVerdict:
    """모양 맞춤 RG 감사를 요약한다."""

    scaling = centered_shape_fluctuation_scaling()
    return ShapeMatchingRGVerdict(
        exact_common_orbit_closed=True,
        exact_common_orbit_associative=True,
        coherent_mismatch_suppressed_by_blocking=False,
        centered_mismatch_decreases=scaling.centered_noise_decreases,
        centered_mismatch_fitted_power=scaling.fitted_power,
        remaining_obligation=(
            "derive a Lorentzian closure/secondary-simplicity/parallel-transport "
            "amplitude that makes common shared-cell geometry an RG attractor"
        ),
    )
