import copy

import numpy as np
import pytest

from reality_stone.clarus.dual_scc_basal_ganglia import DualSCCBasalGanglia
from reality_stone.clarus.dual_scc_controller import (
    DualSCCController,
    DualSCCControllerConfig,
)


def _begin_and_observe(
    controller: DualSCCController,
    *,
    hold_bias_delta: float,
) -> None:
    controller.begin_trial()
    controller.observe(
        (0.3, -0.2),
        (0.2, -0.1, 0.4),
        hold_bias_delta=hold_bias_delta,
    )


def test_probe_keeps_fast_state_and_charges_exactly_once() -> None:
    controller = DualSCCController()
    _begin_and_observe(controller, hold_bias_delta=20.0)
    before = controller.fast_state
    decision = controller.decide()
    assert decision.action is None
    cost = controller.commit_probe(decision.token)
    assert cost == controller.config.probe_cost
    assert controller.total_probe_cost == controller.config.probe_cost
    assert controller.fast_state == before
    with pytest.raises(RuntimeError, match="no active"):
        controller.commit_probe(decision.token)
    controller.observe((0.2, -0.1), (0.4, -0.2, 0.1), hold_bias_delta=-20.0)


def test_feedback_token_is_delayed_single_use_and_action_bound() -> None:
    controller = DualSCCController()
    _begin_and_observe(controller, hold_bias_delta=-20.0)
    decision = controller.decide()
    assert decision.action is not None
    pending = controller.commit_action(decision.token, feedback_delay=2)
    with pytest.raises(RuntimeError, match="before its causal due tick"):
        controller.commit_feedback(pending.token, 1.0)
    controller.advance_time(2)
    before = controller.slow_state
    controller.commit_feedback(pending.token, 1.0)
    assert controller.slow_state != before
    assert controller.pending_feedback_count == 0
    with pytest.raises(RuntimeError, match="already consumed"):
        controller.commit_feedback(pending.token, 1.0)


def test_unknown_or_mismatched_feedback_token_is_rejected() -> None:
    controller = DualSCCController()
    _begin_and_observe(controller, hold_bias_delta=-20.0)
    decision = controller.decide()
    pending = controller.commit_action(decision.token)
    wrong = copy.copy(pending.token)
    object.__setattr__(wrong, "trial", wrong.trial + 1)
    with pytest.raises(RuntimeError, match="unknown or mismatched"):
        controller.commit_feedback(wrong, 1.0)


def test_snapshot_round_trip_preserves_pending_feedback_and_exact_continuation() -> None:
    original = DualSCCController()
    _begin_and_observe(original, hold_bias_delta=-20.0)
    decision = original.decide()
    pending = original.commit_action(decision.token, feedback_delay=3)
    snapshot = original.state_dict()

    restored = DualSCCController()
    restored.load_state_dict(snapshot)
    assert restored.state_dict() == snapshot
    original.advance_time(3)
    restored.advance_time(3)
    original.commit_feedback(pending.token, -0.5)
    restored.commit_feedback(pending.token, -0.5)
    assert restored.state_dict() == original.state_dict()


def test_jacobi_update_reads_only_previous_iteration_states() -> None:
    core = DualSCCBasalGanglia()
    slow = np.asarray((0.2, -0.4), dtype=np.float64)
    fast = np.asarray((0.5, -0.3, 0.1), dtype=np.float64)
    slow_drive = np.asarray((0.1, -0.2), dtype=np.float64)
    fast_drive = np.asarray((-0.1, 0.2, 0.05), dtype=np.float64)
    next_slow, next_fast = core.update(slow, fast, slow_drive, fast_drive)
    expected_fast = np.tanh(
        fast_drive
        + core.config.fast_recurrence * (core.fast_matrix @ fast)
        + core.config.fast_from_slow * (core.fast_from_slow_matrix @ slow)
    )
    sequential_fast = np.tanh(
        fast_drive
        + core.config.fast_recurrence * (core.fast_matrix @ fast)
        + core.config.fast_from_slow * (core.fast_from_slow_matrix @ next_slow)
    )
    assert np.allclose(next_fast, expected_fast, rtol=0.0, atol=1e-15)
    assert not np.allclose(next_fast, sequential_fast, rtol=0.0, atol=1e-12)


def test_sampled_block_perturbations_respect_analytic_gain_matrix() -> None:
    core = DualSCCBasalGanglia()
    gain = np.asarray(core.certificate.gain_matrix)
    rng = np.random.default_rng(20260811)
    for _ in range(200):
        slow = rng.uniform(-1.0, 1.0, size=core.slow_size)
        fast = rng.uniform(-1.0, 1.0, size=core.fast_size)
        other_slow = rng.uniform(-1.0, 1.0, size=core.slow_size)
        other_fast = rng.uniform(-1.0, 1.0, size=core.fast_size)
        left = core.update(slow, fast, (0.0, 0.0), (0.0, 0.0, 0.0))
        right = core.update(
            other_slow,
            other_fast,
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        input_delta = np.asarray(
            (
                np.max(np.abs(slow - other_slow)),
                np.max(np.abs(fast - other_fast)),
            )
        )
        output_delta = np.asarray(
            (
                np.max(np.abs(left[0] - right[0])),
                np.max(np.abs(left[1] - right[1])),
            )
        )
        assert np.all(output_delta <= gain @ input_delta + 1e-12)


def test_componentwise_residual_certificate_encloses_high_precision_reference() -> None:
    core = DualSCCBasalGanglia()
    result = core.settle((0.7, -0.4), (-0.2, 0.6, 0.3))
    slow = np.asarray(result.slow_state)
    fast = np.asarray(result.fast_state)
    ref_slow, ref_fast = slow.copy(), fast.copy()
    for _ in range(100):
        ref_slow, ref_fast = core.update(
            ref_slow,
            ref_fast,
            (0.7, -0.4),
            (-0.2, 0.6, 0.3),
        )
    errors = (
        float(np.max(np.abs(slow - ref_slow))),
        float(np.max(np.abs(fast - ref_fast))),
    )
    assert errors[0] <= result.error_bound_by_layer[0] + 1e-14
    assert errors[1] <= result.error_bound_by_layer[1] + 1e-14


def test_delayed_feedback_history_changes_the_next_recurrent_decision() -> None:
    positive = DualSCCController()
    negative = DualSCCController()
    for controller, reward in ((positive, 1.0), (negative, -1.0)):
        _begin_and_observe(controller, hold_bias_delta=-20.0)
        decision = controller.decide()
        pending = controller.commit_action(decision.token)
        controller.commit_feedback(pending.token, reward)
        controller.begin_trial()
        controller.observe((0.0, 0.0), (0.0, 0.0, -1.0), hold_bias_delta=-20.0)

    positive_policy = positive.decide().conditional_action_probabilities
    negative_policy = negative.decide().conditional_action_probabilities
    assert positive.slow_state != negative.slow_state
    assert not np.allclose(positive_policy, negative_policy, rtol=0.0, atol=1e-6)


def test_probe_fast_anchor_is_a_frozen_input_not_only_an_initial_guess() -> None:
    persistent = DualSCCController(
        config=DualSCCControllerConfig(fast_memory_gain=0.72)
    )
    reset = DualSCCController(config=DualSCCControllerConfig(fast_memory_gain=0.0))
    for controller in (persistent, reset):
        controller.begin_trial()
        controller.observe((0.0, 0.0), (-0.8, 0.8, 0.8), hold_bias_delta=20.0)
        probe = controller.decide()
        assert probe.action is None
        controller.commit_probe(probe.token)
        controller.observe((0.0, 0.0), (0.0, 0.0, -0.8), hold_bias_delta=-20.0)

    assert not np.allclose(
        persistent.fast_state,
        reset.fast_state,
        rtol=0.0,
        atol=1e-6,
    )
