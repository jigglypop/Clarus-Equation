"""Finite causal gates for an already-realized past.

Future interventions and later evidence are deliberately represented by
different operations:

* an intervention selects a normalized future kernel ``K_u(f | h)``;
* evidence conditions beliefs about an unknown past by Bayes' rule.

The first leaves the past marginal invariant.  The second can change a
posterior belief without changing the past event itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .pre_eq import normalize_weights


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class FutureInterventionAudit:
    past_prior: np.ndarray
    past_marginals: np.ndarray
    future_marginals: np.ndarray
    max_past_invariance_residual: float
    past_invariant: bool


@dataclass(frozen=True)
class RetrodictionAudit:
    past_prior: np.ndarray
    posterior: np.ndarray
    evidence_probability: float
    belief_changed: bool
    support_preserved: bool
    zero_prior_histories_remain_zero: bool


def _future_kernels(value: ArrayLike, past_count: int) -> np.ndarray:
    kernels = np.asarray(value, dtype=float)
    if kernels.ndim != 3 or kernels.shape[1] != past_count or kernels.shape[2] < 1:
        raise ValueError("future_kernels must have shape (controls, pasts, futures)")
    if kernels.shape[0] < 1:
        raise ValueError("future_kernels must contain at least one control")
    if not np.all(np.isfinite(kernels)) or np.any(kernels < 0.0):
        raise ValueError("future_kernels must be finite and non-negative")
    if not np.allclose(kernels.sum(axis=2), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("each future kernel row must sum to one")
    return kernels


def future_intervention_audit(
    past_prior: ArrayLike,
    future_kernels: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> FutureInterventionAudit:
    """Verify that normalized future interventions preserve the past marginal."""

    prior = normalize_weights(past_prior)
    kernels = _future_kernels(future_kernels, prior.size)
    joint = prior[None, :, None] * kernels
    past_marginals = joint.sum(axis=2)
    future_marginals = joint.sum(axis=1)
    residual = float(np.max(np.abs(past_marginals - prior[None, :])))

    return FutureInterventionAudit(
        past_prior=prior,
        past_marginals=past_marginals,
        future_marginals=future_marginals,
        max_past_invariance_residual=residual,
        past_invariant=residual <= float(tolerance),
    )


def retrodiction_audit(
    past_prior: ArrayLike,
    evidence_likelihood: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> RetrodictionAudit:
    """Update beliefs about a past from later evidence without adding support."""

    prior = normalize_weights(past_prior)
    likelihood = np.asarray(evidence_likelihood, dtype=float)
    if likelihood.ndim != 1 or likelihood.size != prior.size:
        raise ValueError("evidence_likelihood must match the past prior")
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
        raise ValueError("evidence_likelihood must be finite and non-negative")

    weighted = prior * likelihood
    evidence_probability = float(weighted.sum())
    if evidence_probability <= 0.0:
        raise ValueError("evidence has zero probability under the past prior")
    posterior = weighted / evidence_probability
    zero_prior = prior == 0.0

    return RetrodictionAudit(
        past_prior=prior,
        posterior=posterior,
        evidence_probability=evidence_probability,
        belief_changed=not bool(np.allclose(posterior, prior, atol=tolerance, rtol=0.0)),
        support_preserved=bool(np.all((posterior > 0.0) <= (prior > 0.0))),
        zero_prior_histories_remain_zero=bool(np.all(posterior[zero_prior] == 0.0)),
    )
