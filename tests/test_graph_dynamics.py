from __future__ import annotations

import numpy as np

from reality_stone.clarus.cloudcell_dynamics import NeuralRecording
from reality_stone.clarus.graph_dynamics import evaluate_graph_recording


def _directed_ring(
    seed: int = 7,
    *,
    n_units: int = 14,
    n_time: int = 720,
) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    activity = np.zeros((n_units, n_time), dtype=float)
    activity[:, :2] = rng.normal(scale=0.3, size=(n_units, 2))
    for index in range(2, n_time):
        previous = activity[:, index - 1]
        directed_source = np.roll(previous, 1)
        activity[:, index] = (
            0.42 * previous
            + 0.52 * directed_source
            + rng.normal(scale=0.24, size=n_units)
        )
    return NeuralRecording(
        f"directed-ring-{seed}",
        np.arange(n_time, dtype=float),
        activity,
    )


def _independent_ar(
    seed: int = 11,
    *,
    n_units: int = 14,
    n_time: int = 720,
) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    activity = np.zeros((n_units, n_time), dtype=float)
    activity[:, :2] = rng.normal(scale=0.3, size=(n_units, 2))
    for index in range(2, n_time):
        activity[:, index] = (
            0.78 * activity[:, index - 1]
            - 0.08 * activity[:, index - 2]
            + rng.normal(scale=0.35, size=n_units)
        )
    return NeuralRecording(
        f"independent-{seed}",
        np.arange(n_time, dtype=float),
        activity,
    )


def _switching_ring(
    seed: int = 29,
    *,
    n_units: int = 14,
    n_time: int = 900,
) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    activity = np.zeros((n_units, n_time), dtype=float)
    activity[:, :2] = rng.normal(scale=0.3, size=(n_units, 2))
    context_loading = np.linspace(-0.9, 0.9, n_units)
    for index in range(2, n_time):
        regime = 1.0 if (index // 75) % 2 else -1.0
        previous = activity[:, index - 1]
        source = np.roll(previous, 1 if regime > 0.0 else -1)
        activity[:, index] = (
            0.30 * previous
            + 0.54 * source
            + 0.32 * regime * context_loading
            + rng.normal(scale=0.22, size=n_units)
        )
    return NeuralRecording(
        f"switching-ring-{seed}",
        np.arange(n_time, dtype=float),
        activity,
    )


def test_directed_graph_beats_local_diffusion_and_node_permuted_graphs() -> None:
    gate = evaluate_graph_recording(
        _directed_ring(),
        neighbor_count=1,
        n_rewired=19,
        min_graph_delta=0.01,
        min_positive_fraction=0.7,
    )

    assert len(gate.scores) == 14
    assert gate.median("delta_directed_over_local") > 0.10
    assert gate.median("delta_directed_over_diffusion") > 0.01
    assert gate.rewired_p_value <= 0.05
    assert gate.adjacency_spectral_radius <= 1.0 + 1e-8
    assert gate.passed


def test_independent_units_do_not_pass_directed_graph_gate() -> None:
    gate = evaluate_graph_recording(
        _independent_ar(),
        neighbor_count=2,
        n_rewired=19,
        min_graph_delta=0.01,
        min_positive_fraction=0.7,
    )

    assert not gate.passed
    assert gate.median("delta_directed_over_local") < 0.01


def test_sparse_var_keeps_selected_directed_sources_separate() -> None:
    gate = evaluate_graph_recording(
        _directed_ring(seed=13),
        neighbor_count=2,
        graph_feature_mode="sparse_var",
        n_rewired=19,
        min_graph_delta=0.01,
        min_positive_fraction=0.7,
    )

    assert gate.graph_feature_mode == "sparse_var"
    assert gate.median("delta_directed_over_local") > 0.10
    assert gate.rewired_p_value <= 0.05
    assert gate.passed


def test_two_regime_graph_recovers_context_dependent_connectivity() -> None:
    recording = _switching_ring()
    static = evaluate_graph_recording(
        recording,
        neighbor_count=1,
        graph_feature_mode="sparse_var",
        graph_regimes=1,
        n_rewired=19,
        min_graph_delta=0.01,
        min_positive_fraction=0.6,
    )
    dynamic = evaluate_graph_recording(
        recording,
        neighbor_count=1,
        graph_feature_mode="sparse_var",
        graph_regimes=2,
        n_rewired=19,
        min_graph_delta=0.01,
        min_positive_fraction=0.6,
    )

    assert dynamic.graph_regimes == 2
    assert dynamic.median("delta_directed_over_local") > static.median(
        "delta_directed_over_local"
    )
    assert dynamic.rewired_p_value <= 0.05
    assert dynamic.passed


def test_graph_is_learned_at_one_step_and_reused_across_horizons() -> None:
    recording = _directed_ring(seed=17)

    one_step = evaluate_graph_recording(
        recording,
        horizon_steps=1,
        neighbor_count=1,
        n_rewired=3,
    )
    six_step = evaluate_graph_recording(
        recording,
        horizon_steps=6,
        neighbor_count=1,
        n_rewired=3,
    )

    assert one_step.graph_learn_horizon == 1
    assert six_step.graph_learn_horizon == 1
    assert one_step.adjacency_sha256 == six_step.adjacency_sha256


def test_test_block_cannot_change_the_learned_graph() -> None:
    recording = _directed_ring(seed=23)
    changed = recording.activity.copy()
    changed[:, int(0.81 * changed.shape[1]) :] += 100.0
    altered_test = NeuralRecording(recording.recording_id, recording.time, changed)

    original_gate = evaluate_graph_recording(
        recording,
        neighbor_count=1,
        n_rewired=3,
    )
    altered_gate = evaluate_graph_recording(
        altered_test,
        neighbor_count=1,
        n_rewired=3,
    )

    assert original_gate.adjacency_sha256 == altered_gate.adjacency_sha256
