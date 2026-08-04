from __future__ import annotations

from reality_stone.clarus.realization_pathway_funnel import (
    spatial_folding_realization_funnel,
)


def test_no_current_pathway_passes_every_realization_gate() -> None:
    candidates = spatial_folding_realization_funnel()

    assert len(candidates) == 10
    assert not any(candidate.full_realization_pass for candidate in candidates)
    eligible = [candidate for candidate in candidates if not candidate.fatal_veto]
    vetoed = [candidate for candidate in candidates if candidate.fatal_veto]
    assert candidates == tuple(eligible + vetoed)
    assert all(
        left.physical_gate_count >= right.physical_gate_count
        for left, right in zip(eligible, eligible[1:])
    )


def test_only_two_frontiers_remain_without_a_class_level_veto() -> None:
    active = [
        candidate
        for candidate in spatial_folding_realization_funnel()
        if not candidate.fatal_veto
    ]

    assert [candidate.name for candidate in active] == [
        "beyond-Horndeski wormhole",
        "thin-shell cut-and-paste wormhole",
    ]
    assert active[0].physical_gate_count == 4
    assert active[1].physical_gate_count == 3


def test_known_semiclassical_long_wormhole_sacrifices_shortcut() -> None:
    candidate = next(
        item for item in spatial_folding_realization_funnel() if "charged-fermion" in item.name
    )

    assert candidate.explicit_action
    assert candidate.negative_stress_derived
    assert candidate.self_consistent_backreaction
    assert not candidate.ambient_shortcut


def test_ce_native_routes_have_decisive_action_or_quantum_stress_gate() -> None:
    candidates = [
        item for item in spatial_folding_realization_funnel() if item.name.startswith("CE ")
    ]

    assert len(candidates) == 3
    assert not any(candidate.negative_stress_derived for candidate in candidates)
    assert all(not candidate.full_realization_pass for candidate in candidates)


def test_multimode_dce_is_not_misclassified_as_static_negative_source() -> None:
    candidate = next(
        item for item in spatial_folding_realization_funnel() if item.name.startswith("dynamic")
    )

    assert candidate.engineering_scale_bridge
    assert not candidate.negative_stress_derived
    assert not candidate.self_consistent_backreaction
    assert candidate.fatal_veto
