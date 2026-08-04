from __future__ import annotations

import numpy as np

from reality_stone.clarus.cloudcell_dynamics import NeuralRecording
from reality_stone.clarus.local_memory import evaluate_local_memory_recording


def _ar2_recording(seed: int = 7) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    activity = np.zeros((24, 1400), dtype=float)
    activity[:, :2] = rng.normal(size=(24, 2))
    for time_index in range(2, activity.shape[1]):
        activity[:, time_index] = (
            1.15 * activity[:, time_index - 1]
            - 0.48 * activity[:, time_index - 2]
            + rng.normal(scale=0.2, size=activity.shape[0])
        )
    return NeuralRecording("synthetic_ar2", np.arange(activity.shape[1]), activity)


def _ar1_recording(seed: int = 11) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    activity = np.zeros((24, 1800), dtype=float)
    activity[:, 0] = rng.normal(size=activity.shape[0])
    for time_index in range(1, activity.shape[1]):
        activity[:, time_index] = (
            0.75 * activity[:, time_index - 1]
            + rng.normal(scale=0.4, size=activity.shape[0])
        )
    return NeuralRecording("synthetic_ar1", np.arange(activity.shape[1]), activity)


def test_true_second_order_memory_passes_aligned_history_gate() -> None:
    gate = evaluate_local_memory_recording(_ar2_recording())

    assert gate.passed
    assert gate.median_delta_memory > 0.01
    assert gate.positive_fraction >= 0.8
    assert gate.null_p_value == 0.05


def test_first_order_markov_process_does_not_claim_extra_memory() -> None:
    gate = evaluate_local_memory_recording(_ar1_recording())

    assert not gate.passed
    assert gate.median_delta_memory < 0.01


def test_test_block_mutation_cannot_change_fitted_model_hash() -> None:
    original = _ar2_recording()
    changed_activity = original.activity.copy()
    changed_activity[:, int(0.85 * changed_activity.shape[1]) :] += 50.0
    changed = NeuralRecording(original.recording_id, original.time, changed_activity)

    original_gate = evaluate_local_memory_recording(original)
    changed_gate = evaluate_local_memory_recording(changed)

    assert original_gate.model_sha256 == changed_gate.model_sha256
    assert original_gate.median_delta_memory != changed_gate.median_delta_memory


def test_gap_crossing_samples_are_excluded() -> None:
    original = _ar2_recording()
    baseline = evaluate_local_memory_recording(original)
    time = original.time.astype(float)
    time[1200:] += 100.0
    gate = evaluate_local_memory_recording(
        NeuralRecording(original.recording_id, time, original.activity)
    )

    assert gate.scores
    assert all(
        score.n_test < baseline_score.n_test
        for score, baseline_score in zip(gate.scores, baseline.scores, strict=True)
    )
