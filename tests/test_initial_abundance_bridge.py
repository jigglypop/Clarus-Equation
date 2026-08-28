from __future__ import annotations

import math

from examples.physics.initial_abundance_bridge import (
    MatchingChannel,
    log_density_ratio_identifiability,
    matched_density_ratio,
    matching_degenerate_channel,
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


def test_log_ratio_jacobian_exposes_matching_nonidentifiability() -> None:
    left = MatchingChannel(0.4, 2.0, 0.5)
    right = MatchingChannel(0.6, 1.5, 0.25)

    audit = log_density_ratio_identifiability(left, right)

    assert audit.log_jacobian == (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    assert audit.jacobian_rank == 1
    assert audit.nullity == 5
    assert math.isclose(
        sum(
            derivative * direction
            for derivative, direction in zip(
                audit.log_jacobian,
                audit.energy_efficiency_null_direction,
            )
        ),
        0.0,
        abs_tol=0.0,
    )


def test_energy_efficiency_rescaling_preserves_matched_density() -> None:
    channel = MatchingChannel(0.4, 2.0, 0.5)
    rescaled = matching_degenerate_channel(channel, energy_rescaling=7.0)

    assert math.isclose(
        channel.dimensionless_weight,
        rescaled.dimensionless_weight,
        rel_tol=1.0e-15,
    )
