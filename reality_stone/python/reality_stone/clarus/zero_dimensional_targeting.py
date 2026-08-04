"""Finite gates for a zero-coordinate boundary target selector.

``d=0`` is represented by a singleton state with no intrinsic spatial label.
Such a state cannot adaptively distinguish locations without imported geometric
information.  A boundary functional may nevertheless aggregate complete
3-dimensional histories and return a target label.  If that choice changes the
history being read, an additional fixed-point condition is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .pre_eq import normalize_weights


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class AutonomousD0TargetAudit:
    candidate_count: int
    distribution: np.ndarray
    intrinsic_information_bits: float
    target_label_bits: float
    unique_target_from_d0_state: bool
    externally_encoded_choice_required: bool


@dataclass(frozen=True)
class BoundaryTargetAudit:
    location_scores: np.ndarray
    target_distribution: np.ndarray
    minimizing_locations: tuple[int, ...]
    unique_target: bool
    complete_histories_used: bool
    complete_times_used: bool
    selector_has_intrinsic_position: bool
    localized_actuation_derived: bool
    spatial_shortcut_created: bool


@dataclass(frozen=True)
class TargetFixedPointAudit:
    selected_by_choice: tuple[tuple[int, ...], ...]
    fixed_points: tuple[int, ...]
    fixed_point_exists: bool
    unique_fixed_point: bool


def autonomous_d0_targeting(candidate_count: int) -> AutonomousD0TargetAudit:
    """Audit location selection from a singleton state with no readout."""

    count = int(candidate_count)
    if count < 1:
        raise ValueError("candidate_count must be positive")
    distribution = np.full(count, 1.0 / count)
    return AutonomousD0TargetAudit(
        candidate_count=count,
        distribution=distribution,
        intrinsic_information_bits=0.0,
        target_label_bits=math.log2(count),
        unique_target_from_d0_state=count == 1,
        externally_encoded_choice_required=count > 1,
    )


def coordinate_target_bits(
    extent_m: float,
    resolution_m: float,
    *,
    spatial_dimensions: int = 3,
) -> float:
    """Information lower bound for a grid location in a spatial region."""

    extent = float(extent_m)
    resolution = float(resolution_m)
    dimensions = int(spatial_dimensions)
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("extent_m must be finite and positive")
    if not math.isfinite(resolution) or resolution <= 0.0 or resolution > extent:
        raise ValueError("resolution_m must be positive and no larger than extent_m")
    if dimensions < 1:
        raise ValueError("spatial_dimensions must be positive")
    return dimensions * math.log2(extent / resolution)


def boundary_history_targeting(
    history_time_location_cost: ArrayLike,
    history_prior: ArrayLike,
    time_weights: ArrayLike,
    *,
    beta: float,
) -> BoundaryTargetAudit:
    """Aggregate complete 3D-history readouts at a zero-coordinate boundary."""

    cost = np.asarray(history_time_location_cost, dtype=float)
    if cost.ndim != 3 or cost.size == 0:
        raise ValueError(
            "history_time_location_cost must have shape (histories, times, locations)"
        )
    if not np.all(np.isfinite(cost)):
        raise ValueError("history_time_location_cost must be finite")
    histories = normalize_weights(history_prior)
    times = normalize_weights(time_weights)
    if histories.size != cost.shape[0]:
        raise ValueError("history_prior must match the history axis")
    if times.size != cost.shape[1]:
        raise ValueError("time_weights must match the time axis")
    strength = float(beta)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("beta must be finite and non-negative")

    scores = np.einsum("h,t,htx->x", histories, times, cost)
    minimum = float(np.min(scores))
    weights = np.exp(-strength * (scores - minimum))
    distribution = weights / float(weights.sum())
    minimizers = tuple(int(index) for index in np.flatnonzero(scores == minimum))

    return BoundaryTargetAudit(
        location_scores=scores,
        target_distribution=distribution,
        minimizing_locations=minimizers,
        unique_target=len(minimizers) == 1,
        complete_histories_used=bool(np.all(histories > 0.0)),
        complete_times_used=bool(np.all(times > 0.0)),
        selector_has_intrinsic_position=False,
        localized_actuation_derived=False,
        spatial_shortcut_created=False,
    )


def target_fixed_point_audit(cost_by_realized_choice: ArrayLike) -> TargetFixedPointAudit:
    """Audit self-consistency when target choice changes the observed history.

    Row ``a`` contains target costs in the complete history produced if choice
    ``a`` is realized.  Choice ``a`` is a fixed point exactly when it minimizes
    its own row.
    """

    costs = np.asarray(cost_by_realized_choice, dtype=float)
    if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[0] != costs.shape[1]:
        raise ValueError("cost_by_realized_choice must be a non-empty square matrix")
    if not np.all(np.isfinite(costs)):
        raise ValueError("cost_by_realized_choice must be finite")

    selected = tuple(
        tuple(int(index) for index in np.flatnonzero(row == np.min(row)))
        for row in costs
    )
    fixed = tuple(index for index, targets in enumerate(selected) if index in targets)
    return TargetFixedPointAudit(
        selected_by_choice=selected,
        fixed_points=fixed,
        fixed_point_exists=bool(fixed),
        unique_fixed_point=len(fixed) == 1,
    )
