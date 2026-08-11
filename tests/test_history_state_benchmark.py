from reality_stone.clarus.history_state_benchmark import (
    HistoryEpisode,
    HistoryStateConfig,
    evaluate_history_state,
    history_state,
)


def test_controlled_state_requires_action_sign() -> None:
    positive = HistoryEpisode(1, 1, (1,), (0.7,), (1,))
    negative = HistoryEpisode(1, 1, (-1,), (0.7,), (1,))
    assert history_state(positive, 1.0) == 0.7
    assert history_state(negative, 1.0) == -0.7
    assert history_state(positive, 1.0, mode="observation_only") == 0.7
    assert history_state(negative, 1.0, mode="observation_only") == 0.7


def test_small_history_benchmark_is_finite() -> None:
    result = evaluate_history_state(HistoryStateConfig(train_episodes=96, validation_episodes=48))
    assert result["schema"] == "clarus.history-state.validation.v1"
    assert result["future_reads"] == 0
    assert 0.0 <= result["id"]["brier"] <= 1.0
