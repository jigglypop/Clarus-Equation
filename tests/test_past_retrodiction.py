from __future__ import annotations

import numpy as np
import pytest

from reality_stone.clarus.past_retrodiction import (
    future_intervention_audit,
    retrodiction_audit,
)


def test_every_normalized_future_intervention_preserves_the_past_marginal() -> None:
    kernels = np.array([
        [[0.9, 0.1], [0.2, 0.8]],
        [[0.1, 0.9], [0.8, 0.2]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])
    audit = future_intervention_audit([0.7, 0.3], kernels)

    assert audit.past_invariant
    assert audit.max_past_invariance_residual < 1e-15
    assert np.allclose(audit.past_marginals, [[0.7, 0.3]] * 3)
    assert not np.allclose(audit.future_marginals[0], audit.future_marginals[1])


def test_later_evidence_can_change_belief_about_an_unknown_past() -> None:
    audit = retrodiction_audit([0.7, 0.3], [0.1, 0.8])

    assert audit.belief_changed
    assert np.allclose(audit.posterior, [0.07 / 0.31, 0.24 / 0.31])
    assert audit.support_preserved


def test_retrodiction_cannot_revive_a_zero_prior_past() -> None:
    audit = retrodiction_audit([0.6, 0.4, 0.0], [0.1, 0.2, 1e100])

    assert audit.posterior[2] == 0.0
    assert audit.zero_prior_histories_remain_zero
    assert audit.support_preserved


def test_uninformative_evidence_does_not_change_past_belief() -> None:
    audit = retrodiction_audit([0.2, 0.8], [0.4, 0.4])

    assert not audit.belief_changed
    assert np.allclose(audit.posterior, audit.past_prior)


def test_future_intervention_rejects_non_normalized_kernel() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        future_intervention_audit([0.5, 0.5], [[[0.2, 0.2], [0.5, 0.5]]])


def test_retrodiction_rejects_impossible_evidence() -> None:
    with pytest.raises(ValueError, match="zero probability"):
        retrodiction_audit([0.5, 0.5], [0.0, 0.0])
