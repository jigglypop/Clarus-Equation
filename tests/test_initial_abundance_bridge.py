from __future__ import annotations

import math

from examples.physics.initial_abundance_bridge import (
    MatchingChannel,
    matched_density_ratio,
    noninteracting_ratio_evolution,
    required_initial_matter_to_vacuum_ratio,
)


def test_zero_dimensional_composition_is_not_density_without_matching_weights() -> None:
    left = MatchingChannel(0.5, 2.0, 0.25)
    right = MatchingChannel(0.5, 1.0, 1.0)

    assert matched_density_ratio(left, right) == 0.5


def test_matter_to_vacuum_ratio_changes_by_a_cubed() -> None:
    initial = 2.0
    final = noninteracting_ratio_evolution(
        initial, a_initial=1.0e-4, a_final=1.0, w_left=0.0, w_right=-1.0
    )

    assert math.isclose(final, 2.0e-12, rel_tol=1.0e-15)


def test_present_ratio_requires_huge_early_matter_inventory_if_separate() -> None:
    present = 0.26391 / 0.687
    required = required_initial_matter_to_vacuum_ratio(present, a_initial=1.0e-4)

    assert math.isclose(required, present * 1.0e12, rel_tol=1.0e-15)
