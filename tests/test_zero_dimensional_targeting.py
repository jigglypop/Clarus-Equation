from __future__ import annotations

import numpy as np
import pytest

from reality_stone.clarus.zero_dimensional_targeting import (
    autonomous_d0_targeting,
    boundary_history_targeting,
    coordinate_target_bits,
    target_fixed_point_audit,
)


def test_autonomous_d0_state_cannot_distinguish_multiple_locations() -> None:
    audit = autonomous_d0_targeting(8)

    assert np.allclose(audit.distribution, np.full(8, 1.0 / 8.0))
    assert audit.intrinsic_information_bits == 0.0
    assert audit.target_label_bits == 3.0
    assert not audit.unique_target_from_d0_state
    assert audit.externally_encoded_choice_required


def test_coordinate_precision_has_nonzero_information_cost() -> None:
    light_year_m = 9.4607304725808e15
    bits = coordinate_target_bits(light_year_m, 1.0)

    assert 159.0 < bits < 160.0


def test_complete_history_boundary_readout_selects_a_location() -> None:
    costs = np.array([
        [[4.0, 3.0, 1.0], [5.0, 2.0, 0.5]],
        [[3.0, 4.0, 1.5], [4.0, 3.0, 0.25]],
    ])
    audit = boundary_history_targeting(
        costs,
        history_prior=[0.4, 0.6],
        time_weights=[0.25, 0.75],
        beta=20.0,
    )

    assert audit.minimizing_locations == (2,)
    assert audit.unique_target
    assert audit.target_distribution[2] > 1.0 - 1e-12
    assert audit.complete_histories_used
    assert audit.complete_times_used
    assert not audit.selector_has_intrinsic_position
    assert not audit.localized_actuation_derived
    assert not audit.spatial_shortcut_created


def test_global_future_targeting_can_have_no_self_consistent_choice() -> None:
    audit = target_fixed_point_audit([[2.0, 1.0], [1.0, 2.0]])

    assert audit.selected_by_choice == ((1,), (0,))
    assert audit.fixed_points == ()
    assert not audit.fixed_point_exists


def test_global_future_targeting_can_have_multiple_fixed_points() -> None:
    audit = target_fixed_point_audit([[0.0, 1.0], [1.0, 0.0]])

    assert audit.fixed_points == (0, 1)
    assert audit.fixed_point_exists
    assert not audit.unique_fixed_point


def test_global_future_targeting_can_have_one_fixed_point() -> None:
    audit = target_fixed_point_audit([[0.0, 1.0], [0.0, 2.0]])

    assert audit.fixed_points == (0,)
    assert audit.unique_fixed_point


@pytest.mark.parametrize("count", [0, -1])
def test_autonomous_targeting_rejects_empty_candidate_space(count: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        autonomous_d0_targeting(count)


def test_coordinate_bits_rejects_resolution_larger_than_region() -> None:
    with pytest.raises(ValueError, match="resolution_m"):
        coordinate_target_bits(1.0, 2.0)
