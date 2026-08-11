from reality_stone.clarus.orbit_budget_benchmark import (
    evaluate_orbit_budget_sidecar,
    generate_interception_episodes,
)


def test_interception_generator_is_balanced_and_margin_locked() -> None:
    episodes = generate_interception_episodes(64, 41064, per_action=3)
    assert [episode.label for episode in episodes].count(0) == 3
    assert [episode.label for episode in episodes].count(1) == 3
    assert [episode.label for episode in episodes].count(2) == 3
    for episode in episodes:
        ordered = sorted(episode.utilities)
        assert ordered[-1] - ordered[-2] >= 0.01


def test_orbit_budget_sidecar_passes_behavior_and_error_gates() -> None:
    result = evaluate_orbit_budget_sidecar()
    assert result["standalone_verdict"] == "GO"
    assert result["gates"]["local_dense_noninferiority"]
    assert result["gates"]["local_beats_quotient"]
    assert result["gates"]["shift_consistency"]
    assert result["gates"]["budget_bound_valid"]
    assert result["gates"]["large_budget_exact"]
