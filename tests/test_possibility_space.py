from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.possibility_space import (
    complete_history_readout,
    condition_on_realized_past,
    dimension_origin_audit,
    possibility_shift_audit,
    target_possibility_shift,
)


def test_complete_history_readout_uses_every_history_and_time() -> None:
    histories = np.array([
        [0.0, 1.0, 2.0],
        [3.0, 2.0, 1.0],
    ])
    audit = complete_history_readout(
        histories,
        prior=[0.25, 0.75],
        time_weights=[0.2, 0.3, 0.5],
    )

    assert np.allclose(audit.history_readouts, [1.3, 1.7])
    assert math.isclose(audit.ensemble_readout, 1.6)
    assert audit.all_times_used
    assert audit.all_histories_used


def test_conditioning_keeps_only_histories_with_the_realized_past() -> None:
    conditioned = condition_on_realized_past(
        [0.1, 0.2, 0.3, 0.4],
        past_ids=[7, 7, 8, 7],
        realized_past_id=7,
    )

    assert np.allclose(conditioned, [1.0 / 7.0, 2.0 / 7.0, 0.0, 4.0 / 7.0])


def test_positive_target_tilt_increases_nontrivial_target_mass() -> None:
    posterior, before, after = target_possibility_shift(
        [0.2, 0.3, 0.5],
        [False, True, False],
        strength=2.0,
    )

    expected = 0.3 / (0.3 + 0.7 * math.exp(-2.0))
    assert math.isclose(before, 0.3)
    assert math.isclose(after, expected)
    assert after > before
    assert math.isclose(float(posterior.sum()), 1.0)


def test_finite_reweighting_cannot_create_a_zero_support_history() -> None:
    posterior, before, after = target_possibility_shift(
        [0.6, 0.4, 0.0],
        [False, False, True],
        strength=100.0,
    )

    assert before == 0.0
    assert after == 0.0
    assert posterior[2] == 0.0


def test_extreme_tilt_with_empty_target_stays_finite_and_preserves_prior() -> None:
    posterior, before, after = target_possibility_shift(
        [0.5, 0.5],
        [False, False],
        strength=1000.0,
    )

    assert np.all(np.isfinite(posterior))
    assert np.allclose(posterior, [0.5, 0.5])
    assert math.isclose(float(posterior.sum()), 1.0)
    assert before == 0.0
    assert after == 0.0


def test_extreme_tilt_toward_zero_prior_target_does_not_create_support() -> None:
    posterior, before, after = target_possibility_shift(
        [0.6, 0.4, 0.0],
        [False, False, True],
        strength=1000.0,
    )

    assert np.all(np.isfinite(posterior))
    assert np.allclose(posterior, [0.6, 0.4, 0.0])
    assert before == 0.0
    assert after == 0.0


def test_extreme_nontrivial_tilt_separates_theorem_from_float_resolution() -> None:
    audit = possibility_shift_audit(
        [0.25, 0.25, 0.5],
        past_ids=[1, 1, 1],
        realized_past_id=1,
        target=[False, True, False],
        strength=1000.0,
    )

    assert np.all(np.isfinite(audit.posterior))
    assert math.isclose(float(audit.posterior.sum()), 1.0)
    assert audit.target_mass_increased
    assert audit.target_mass_numerically_increased
    assert audit.support_preserved_by_finite_tilt
    assert not audit.floating_point_support_fully_resolved


def test_possibility_shift_changes_future_mass_without_rewriting_past() -> None:
    audit = possibility_shift_audit(
        [0.2, 0.2, 0.2, 0.4],
        past_ids=[1, 1, 2, 1],
        realized_past_id=1,
        target=[False, True, True, False],
        strength=3.0,
    )

    assert audit.target_mass_increased
    assert audit.target_mass_numerically_increased
    assert audit.incompatible_pasts_remain_impossible
    assert audit.support_preserved_by_finite_tilt
    assert audit.floating_point_support_fully_resolved
    assert audit.posterior[2] == 0.0


def test_d0_is_an_algebraic_root_not_a_derived_pre_universe() -> None:
    audit = dimension_origin_audit()

    assert audit.algebraic_roots == (0, 3)
    assert audit.d0_is_algebraic_root
    assert audit.d3_is_unique_positive_root
    assert not audit.d0_has_spatial_worldline
    assert not audit.d0_supports_internal_observer
    assert not audit.temporal_predecessor_derived
    assert not audit.d0_to_d3_dynamics_derived


@pytest.mark.parametrize("strength", [-1.0, math.inf, math.nan])
def test_target_shift_rejects_invalid_strength(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        target_possibility_shift([0.5, 0.5], [True, False], strength=strength)
