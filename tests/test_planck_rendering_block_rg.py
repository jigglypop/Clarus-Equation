from __future__ import annotations

import math

import pytest

from examples.physics.planck_rendering_block_rg import (
    block_rg_verdict,
    blocked_rendered_mean,
    critical_block_moments,
    critical_borel_asymptotic_ratio,
    critical_side_tree_total_progeny_probability,
    critical_split_merge,
    heat_time_from_area,
    marked_joint_probability,
    q_spine_distinct_probability,
    spine_fixed_point,
)


D_CE = 3.1777584234


def test_critical_marking_is_unique_and_gives_exact_ce_numbers() -> None:
    params = critical_split_merge(D_CE)

    assert params.distinct_probability == pytest.approx(1.0 / D_CE)
    assert params.merge_probability == pytest.approx(1.0 - 1.0 / D_CE)
    assert params.distinct_intensity == pytest.approx(1.0)
    assert params.face_intensity == pytest.approx(D_CE - 1.0)


def test_marked_poisson_joint_law_is_normalized_on_a_large_window() -> None:
    params = critical_split_merge(D_CE)
    probability = math.fsum(
        marked_joint_probability(
            branch_mean=D_CE,
            distinct_probability=params.distinct_probability,
            distinct_count=rendered,
            face_count=faces,
        )
        for rendered in range(16)
        for faces in range(24)
    )

    assert probability == pytest.approx(1.0, abs=1.0e-12)


def test_blocking_fixes_the_mean_but_not_the_poisson_law() -> None:
    assert blocked_rendered_mean(1.0, depth=8) == pytest.approx(1.0)
    audit = critical_block_moments(D_CE, depth=8)

    assert audit.output_mean == pytest.approx(1.0)
    assert audit.output_variance == pytest.approx(8.0)
    assert not audit.poisson_family_closed
    assert audit.expected_face_events == pytest.approx(8.0 * (D_CE - 1.0))


def test_noncritical_rendered_mean_flows_away_under_blocking() -> None:
    assert blocked_rendered_mean(0.9, depth=4) == pytest.approx(0.9**4)
    assert blocked_rendered_mean(1.1, depth=4) == pytest.approx(1.1**4)
    assert blocked_rendered_mean(0.9, depth=4) < 0.9
    assert blocked_rendered_mean(1.1, depth=4) > 1.1


def test_spine_conditioning_has_one_persistent_line_and_stationary_side_law() -> None:
    audit = spine_fixed_point(D_CE)

    assert audit.persistent_spine_count == 1
    assert audit.rendered_continuation_mean == pytest.approx(2.0)
    assert audit.folded_side_branch_mean == pytest.approx(1.0)
    assert audit.face_event_mean == pytest.approx(D_CE - 1.0)
    assert audit.shift_invariant_local_law
    assert math.fsum(q_spine_distinct_probability(k) for k in range(1, 20)) == pytest.approx(
        1.0,
        abs=1.0e-15,
    )


def test_critical_side_tree_has_borel_three_halves_tail() -> None:
    first_mass = critical_side_tree_total_progeny_probability(1)
    assert first_mass == pytest.approx(math.exp(-1.0))
    assert critical_borel_asymptotic_ratio(10_000) == pytest.approx(1.0, rel=5.0e-5)


def test_heat_time_is_additive_in_area() -> None:
    first = heat_time_from_area(area=2.0, planck_area=0.5)
    second = heat_time_from_area(area=3.0, planck_area=0.5)
    combined = heat_time_from_area(area=5.0, planck_area=0.5)

    assert combined == pytest.approx(first + second)


def test_verdict_keeps_the_remaining_geometry_obligation_explicit() -> None:
    verdict = block_rg_verdict(D_CE)

    assert verdict.critical_rendered_mean == pytest.approx(1.0)
    assert verdict.critical_face_intensity == pytest.approx(D_CE - 1.0)
    assert not verdict.full_poisson_measure_fixed
    assert verdict.spine_local_measure_fixed
    assert verdict.side_sector_scale_free
    assert "simplicity" in verdict.remaining_obligation


@pytest.mark.parametrize("branch_mean", (1.0, 0.5, -1.0, math.inf))
def test_critical_split_merge_rejects_invalid_domain(branch_mean: float) -> None:
    with pytest.raises(ValueError):
        critical_split_merge(branch_mean)
