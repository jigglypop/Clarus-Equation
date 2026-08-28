"""Finite Urbantke/shape-matching audit for the Planck-rendering bridge.

The exact result is deliberately narrow: after SO(3) frame alignment, triples
in one positive-scale common metric orbit remain Plebanski-simple under any
blocking order. Generic locally simple cells do not. Centered random mismatch
can decrease statistically, while coherent mismatch persists.

This is a Euclideanized finite algebra audit, not a Lorentzian spin-foam model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math

import numpy as np

from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple,
    simplicity_residual,
    wedge_scalar,
)

_PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2))


def _levi_civita(rank: int) -> np.ndarray:
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
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 6) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape (3, 6)")
    return array


def _matrices(value: np.ndarray) -> np.ndarray:
    triple = _triple(value)
    out = np.zeros((3, 4, 4), dtype=float)
    for i in range(3):
        for component, (mu, nu) in zip(triple[i], _PAIR_INDEX):
            out[i, mu, nu] = component
            out[i, nu, mu] = -component
    return out


def urbantke_metric_density(value: np.ndarray) -> np.ndarray:
    """Return the symmetric Urbantke metric density."""

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
    """Return the positive Euclidean conformal metric with determinant one."""

    density = urbantke_metric_density(value)
    if float(np.min(np.linalg.eigvalsh(density))) <= 1.0e-12:
        raise ValueError("Urbantke metric must be positive and nondegenerate")
    return density / float(np.linalg.det(density)) ** 0.25


def conformal_metric_residual(first: np.ndarray, second: np.ndarray) -> float:
    first_metric = normalized_urbantke_metric(first)
    second_metric = normalized_urbantke_metric(second)
    return float(np.linalg.norm(first_metric - second_metric) / 2.0)


def cross_wedge_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
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
    """Align candidate to reference with the proper SO(3) polar factor."""

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
    """Align and sum cells belonging to one positive-scale metric orbit."""

    if not triples:
        raise ValueError("at least one triple is required")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    reference = _triple(triples[0], "triples[0]")
    aligned = [reference]
    metric_residuals = [0.0]
    orbit_residuals = [0.0]
    for index, candidate in enumerate(triples[1:], start=1):
        audit = optimal_internal_alignment(reference, _triple(candidate, f"triples[{index}]"))
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
    """Return a soft projector for metric, orbit and block simplicity."""

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
    """Show that repeating one coherent mismatch does not average it away."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    audit = optimal_internal_alignment(reference, candidate)
    return simplicity_residual(repeats * (reference + audit.aligned_candidate))


@dataclass(frozen=True)
class ShapeFluctuationScaling:
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
    """Measure finite-block suppression for centered weak tetrad mismatch."""

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
    exact_common_orbit_closed: bool
    exact_common_orbit_associative: bool
    coherent_mismatch_suppressed_by_blocking: bool
    centered_mismatch_decreases: bool
    centered_mismatch_fitted_power: float
    remaining_obligation: str


def shape_matching_rg_verdict() -> ShapeMatchingRGVerdict:
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
