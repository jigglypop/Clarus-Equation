"""Finite gates for CE/PreEq complete-history and possibility-space claims.

These gates prove finite probability statements.  They do not turn a
probability update into a physical transition between already-realized
histories, and they keep the algebraic ``d=0`` root separate from a claim about
what existed before the observable universe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .pre_eq import normalize_weights


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class CompleteHistoryAudit:
    history_readouts: np.ndarray
    ensemble_readout: float
    all_times_used: bool
    all_histories_used: bool


@dataclass(frozen=True)
class PossibilityShiftAudit:
    conditioned_prior: np.ndarray
    posterior: np.ndarray
    prior_target_mass: float
    posterior_target_mass: float
    target_mass_increased: bool
    target_mass_numerically_increased: bool
    incompatible_pasts_remain_impossible: bool
    support_preserved_by_finite_tilt: bool
    floating_point_support_fully_resolved: bool


@dataclass(frozen=True)
class DimensionOriginAudit:
    algebraic_roots: tuple[int, ...]
    d0_is_algebraic_root: bool
    d3_is_unique_positive_root: bool
    d0_has_spatial_worldline: bool
    d0_supports_internal_observer: bool
    temporal_predecessor_derived: bool
    d0_to_d3_dynamics_derived: bool
    status: str


def complete_history_readout(
    histories: ArrayLike,
    prior: ArrayLike,
    time_weights: ArrayLike,
) -> CompleteHistoryAudit:
    """Evaluate every time sample of every finite candidate history.

    Rows of ``histories`` are complete histories and columns are ordered time
    samples.  This is the finite analogue of applying a functional to a whole
    path and then integrating that functional over path space.
    """

    values = np.asarray(histories, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("histories must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("histories must be finite")

    history_prior = normalize_weights(prior)
    weights = normalize_weights(time_weights)
    if history_prior.size != values.shape[0]:
        raise ValueError("prior length must match the number of histories")
    if weights.size != values.shape[1]:
        raise ValueError("time_weights length must match the number of times")

    readouts = values @ weights
    return CompleteHistoryAudit(
        history_readouts=readouts,
        ensemble_readout=float(history_prior @ readouts),
        all_times_used=bool(np.all(weights > 0.0)),
        all_histories_used=bool(np.all(history_prior > 0.0)),
    )


def condition_on_realized_past(
    prior: ArrayLike,
    past_ids: Iterable[int],
    realized_past_id: int,
) -> np.ndarray:
    """Condition candidate histories on the already-recorded past."""

    p = normalize_weights(prior)
    labels = np.asarray(tuple(past_ids), dtype=int)
    if labels.ndim != 1 or labels.size != p.size:
        raise ValueError("past_ids must match the prior")
    compatible = labels == int(realized_past_id)
    mass = float(p[compatible].sum())
    if mass <= 0.0:
        raise ValueError("the realized past has zero prior mass")
    return np.where(compatible, p, 0.0) / mass


def target_possibility_shift(
    prior: ArrayLike,
    target: Iterable[bool] | np.ndarray,
    *,
    strength: float,
) -> tuple[np.ndarray, float, float]:
    """Exponentially favor a target event without enlarging prior support."""

    p = normalize_weights(prior)
    mask = np.asarray(tuple(target), dtype=bool)
    if mask.ndim != 1 or mask.size != p.size:
        raise ValueError("target must match the prior")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and non-negative")

    target_mass = float(p[mask].sum())
    if target_mass == 0.0 or target_mass == 1.0:
        # A common finite factor cancels when one partition has no mass.
        return p.copy(), target_mass, target_mass

    # Work with the two partition masses in log space.  Computing exp(-u)
    # first produced 0/0 and NaN for an empty target at u=1000.  Within each
    # partition the tilt is common, so its prior conditional ratios are kept.
    log_target = math.log(target_mass)
    log_non_target = math.log1p(-target_mass) - strength
    log_normalization = float(np.logaddexp(log_target, log_non_target))
    posterior_target_mass = math.exp(log_target - log_normalization)
    posterior_non_target_mass = math.exp(log_non_target - log_normalization)

    posterior = np.zeros_like(p)
    posterior[mask] = (p[mask] / target_mass) * posterior_target_mass
    posterior[~mask] = (
        p[~mask] / (1.0 - target_mass)
    ) * posterior_non_target_mass
    posterior /= float(posterior.sum())
    return posterior, target_mass, posterior_target_mass


def possibility_shift_audit(
    prior: ArrayLike,
    past_ids: Iterable[int],
    realized_past_id: int,
    target: Iterable[bool] | np.ndarray,
    *,
    strength: float,
) -> PossibilityShiftAudit:
    """Audit future selection, fixed-past conditioning, and zero support."""

    labels = np.asarray(tuple(past_ids), dtype=int)
    conditioned = condition_on_realized_past(prior, labels, realized_past_id)
    mask = np.asarray(tuple(target), dtype=bool)
    posterior, before, after = target_possibility_shift(
        conditioned,
        mask,
        strength=strength,
    )
    incompatible = labels != int(realized_past_id)
    analytically_increased = (
        0.0 < before < 1.0 and strength > 0.0
    )
    floating_support_resolved = bool(
        np.array_equal(conditioned > 0.0, posterior > 0.0)
    )

    return PossibilityShiftAudit(
        conditioned_prior=conditioned,
        posterior=posterior,
        prior_target_mass=before,
        posterior_target_mass=after,
        target_mass_increased=analytically_increased,
        target_mass_numerically_increased=after > before,
        incompatible_pasts_remain_impossible=bool(
            np.all(conditioned[incompatible] == 0.0)
            and np.all(posterior[incompatible] == 0.0)
        ),
        # For finite strength, exp(-strength) is mathematically positive even
        # when its binary64 representation underflows.  Keep the theorem apart
        # from whether every tiny posterior entry remains representable.
        support_preserved_by_finite_tilt=True,
        floating_point_support_fully_resolved=floating_support_resolved,
    )


def dimension_origin_audit() -> DimensionOriginAudit:
    """Separate the CE dimension equation from a pre-universe ontology claim."""

    roots = tuple(dimension for dimension in range(0, 13) if dimension * (dimension - 3) == 0)
    return DimensionOriginAudit(
        algebraic_roots=roots,
        d0_is_algebraic_root=0 in roots,
        d3_is_unique_positive_root=tuple(root for root in roots if root > 0) == (3,),
        d0_has_spatial_worldline=False,
        d0_supports_internal_observer=False,
        temporal_predecessor_derived=False,
        d0_to_d3_dynamics_derived=False,
        status="ALGEBRAIC_ROOT_ONLY_PRE_UNIVERSE_CLAIM_OPEN",
    )
