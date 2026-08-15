from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pytest

import reality_stone.clarus as clarus
from reality_stone.clarus.delayed_linear_credit import (
    DelayedCreditCertificate,
    EligibilityLearner,
    EligibilityState,
    HardLatchLearner,
    HardLatchState,
    HomogeneousCreditState,
    HomogeneousLearner,
    NoTraceControl,
    NoTraceState,
    StrictMetricControl,
    StrictMetricState,
    decode_binary_reward,
)


def _classifier_bytes(state: object) -> bytes:
    values = np.asarray(getattr(state, "classifier"), dtype=np.float64)
    return values.tobytes()


def _episode(
    learner: object,
    state: object,
    cue: np.ndarray,
    label: int,
    *,
    delay: int,
    invert_reward: bool = False,
    lesion: bool = False,
) -> object:
    before = _classifier_bytes(state)
    state = learner.write_cue(state, cue)
    assert _classifier_bytes(state) == before
    for index in range(delay):
        distractor = np.roll(np.ones(cue.size), index % cue.size) / math.sqrt(cue.size)
        state = learner.distract(state, distractor)
        assert _classifier_bytes(state) == before
    if lesion:
        state = learner.trace_lesion(state)
        assert _classifier_bytes(state) == before
    action = learner.action(state)
    assert _classifier_bytes(state) == before
    reward = int(action == label)
    return learner.reward(state, action, reward, invert_reward=invert_reward)


def _train(
    learner: object,
    theta: np.ndarray,
    *,
    invert_reward: bool = False,
    lesion: bool = False,
) -> object:
    state = learner.identity_state()
    for epoch in range(4):
        for coordinate in range(theta.size):
            sign = -1 if (epoch + coordinate) % 2 else +1
            cue = np.zeros(theta.size)
            cue[coordinate] = sign
            label = int(sign * theta[coordinate])
            state = _episode(
                learner,
                state,
                cue,
                label,
                delay=(4, 8, 16)[(epoch + coordinate) % 3],
                invert_reward=invert_reward,
                lesion=lesion,
            )
    return state


@pytest.mark.parametrize("action", [-1, +1])
@pytest.mark.parametrize("label", [-1, +1])
def test_binary_reward_decodes_hidden_label(action: int, label: int) -> None:
    reward = int(action == label)
    assert decode_binary_reward(action, reward) == label


@pytest.mark.parametrize("learner_type", [EligibilityLearner, HardLatchLearner, HomogeneousLearner])
def test_four_coordinate_visits_learn_teacher_with_exact_classifier_timing(
    learner_type: type[object],
) -> None:
    theta = np.array([1.0, -1.0, -1.0, 1.0])
    learner = learner_type(4, 0.25)

    state = _train(learner, theta)

    np.testing.assert_allclose(state.classifier, theta, rtol=0.0, atol=2e-15)
    assert not state.active


@pytest.mark.parametrize("learner_type", [EligibilityLearner, HomogeneousLearner])
def test_registered_trace_and_reward_inversion_lesions_have_exact_predictions(
    learner_type: type[object],
) -> None:
    theta = np.array([1.0, -1.0, -1.0, 1.0])
    learner = learner_type(4, 0.25)

    trace_deleted = _train(learner, theta, lesion=True)
    reward_inverted = _train(learner, theta, invert_reward=True)

    np.testing.assert_array_equal(trace_deleted.classifier, np.zeros(4))
    np.testing.assert_allclose(reward_inverted.classifier, -theta, rtol=0.0, atol=2e-15)


def test_no_trace_control_never_updates_classifier() -> None:
    theta = np.array([1.0, -1.0, -1.0, 1.0])
    control = NoTraceControl(4, 0.25)
    state = control.identity_state()
    for coordinate in range(4):
        cue = np.eye(4)[coordinate]
        state = control.write_cue(state, cue)
        action = control.action(state)
        state = control.reward(state, action, int(action == theta[coordinate]))

    np.testing.assert_array_equal(state.classifier, np.zeros(4))
    assert not state.active


@pytest.mark.parametrize(
    "learner",
    [
        EligibilityLearner(3),
        HardLatchLearner(3),
        HomogeneousLearner(3),
        StrictMetricControl(3),
        NoTraceControl(3),
    ],
)
def test_unmarked_distractor_is_the_exact_immutable_state_noop(learner: object) -> None:
    state = learner.write_cue(learner.identity_state(), [1.0, 0.0, 0.0])

    after = learner.distract(state, [0.0, 1.0, 0.0])

    assert after is state


@pytest.mark.parametrize(
    "learner",
    [
        EligibilityLearner(3),
        HardLatchLearner(3),
        HomogeneousLearner(3),
        StrictMetricControl(3),
        NoTraceControl(3),
    ],
)
def test_batched_distractors_equal_event_loop_and_preserve_exact_state(learner: object) -> None:
    state = learner.write_cue(learner.identity_state(), [1.0, 0.0, 0.0])
    observations = (
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
    )
    loop_state = state
    for observation in observations:
        loop_state = learner.distract(loop_state, observation)

    batch_state = learner.distract_many(state, observations)

    assert loop_state is state
    assert batch_state is state


@pytest.mark.parametrize(
    "learner",
    [
        EligibilityLearner(3),
        HardLatchLearner(3),
        HomogeneousLearner(3),
        StrictMetricControl(3),
        NoTraceControl(3),
    ],
)
def test_batched_distractors_reject_invalid_middle_observation(learner: object) -> None:
    state = learner.write_cue(learner.identity_state(), [1.0, 0.0, 0.0])
    observations = (
        [0.0, 1.0, 0.0],
        [0.0, math.nan, 0.0],
        [0.0, 0.0, 1.0],
    )

    with pytest.raises(ValueError, match=r"observations\[1\].*finite"):
        learner.distract_many(state, observations)


def test_batched_distractor_vectorization_handles_empty_and_rejects_bad_arrays() -> None:
    learner = EligibilityLearner(3)
    state = learner.write_cue(learner.identity_state(), [1.0, 0.0, 0.0])

    assert learner.distract_many(state, []) is state
    assert learner.distract_many(state, np.empty((0, 3))) is state
    with pytest.raises(ValueError, match="real numeric"):
        learner.distract_many(state, [[True, False, False]])
    with pytest.raises(ValueError, match="real numeric"):
        learner.distract_many(state, [[1.0 + 0.0j, 0.0, 0.0]])
    with pytest.raises(ValueError, match="rectangular"):
        learner.distract_many(state, [[1.0, 0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match=r"shape \(K, 3\)"):
        learner.distract_many(state, np.empty((0, 2)))


def test_homogeneous_cross_block_recovers_cue_and_reward_resets_atomically() -> None:
    learner = HomogeneousLearner(4)
    cue = np.array([-1.0, 0.0, 0.0, 0.0])
    active = learner.write_cue(learner.identity_state(), cue)

    np.testing.assert_allclose(learner.eligibility(active), cue, rtol=0.0, atol=3e-16)
    assert tuple(field.name for field in fields(HomogeneousCreditState)) == (
        "classifier",
        "factor",
        "active",
    )
    assert not hasattr(active, "cue")

    action = learner.action(active)
    cleared = learner.reward(active, action, int(action == -1))
    np.testing.assert_array_equal(np.asarray(cleared.factor), np.eye(5))
    assert not cleared.active


def test_homogeneous_lesion_is_identity_inside_spd_space_and_keeps_active_tag() -> None:
    learner = HomogeneousLearner(3)
    active = learner.write_cue(learner.identity_state(), [0.0, 1.0, 0.0])

    lesioned = learner.trace_lesion(active)

    assert lesioned.active
    np.testing.assert_array_equal(np.asarray(lesioned.factor), np.eye(4))
    np.testing.assert_array_equal(learner.eligibility(lesioned), np.zeros(3))


def test_deleting_homogeneous_coordinate_is_sign_even() -> None:
    learner = HomogeneousLearner(3)
    cue = np.array([2.0, -1.0, 2.0]) / 3.0
    positive = learner.write_cue(learner.identity_state(), cue)
    negative = learner.write_cue(learner.identity_state(), -cue)

    np.testing.assert_array_equal(
        learner.spatial_metric(positive),
        learner.spatial_metric(negative),
    )
    np.testing.assert_allclose(learner.eligibility(positive), cue, atol=3e-16)
    np.testing.assert_allclose(learner.eligibility(negative), -cue, atol=3e-16)


@pytest.mark.parametrize("delay", [0, 128])
@pytest.mark.parametrize("learner_type", [EligibilityLearner, HardLatchLearner, HomogeneousLearner])
def test_learned_classifier_generalizes_to_dense_nonzero_margin_compositions(
    learner_type: type[object], delay: int
) -> None:
    theta = np.array([1.0, -1.0, -1.0, 1.0])
    learner = learner_type(4)
    checkpoint = _train(learner, theta)
    queries = (
        np.array([1.0, 1.0, -1.0, 1.0]) / 2.0,
        np.array([-1.0, 1.0, 1.0, 1.0]) / 2.0,
    )
    for query in queries:
        margin = float(theta @ query)
        assert margin != 0.0
        state = learner.from_snapshot(checkpoint)
        state = learner.write_cue(state, query)
        for _ in range(delay):
            state = learner.distract(state, -query)
        assert learner.action(state) == (1 if margin > 0.0 else -1)


def test_strict_metric_is_byte_identical_for_paired_signs_at_every_ensemble_size() -> None:
    control = StrictMetricControl(4)
    query = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
    positive = control.write_cue(control.identity_state(), query)
    negative = control.write_cue(control.identity_state(), -query)
    nuisance = np.array([-1.0, -1.0, 1.0, 1.0]) / 2.0
    positive = control.distract(positive, nuisance)
    negative = control.distract(negative, nuisance)

    assert control.serialize_state(positive) == control.serialize_state(negative)
    assert b"-0x0.0p+0" not in control.serialize_state(positive)
    for size in control.ensemble_sizes:
        positive_ensemble = [positive] * size
        negative_ensemble = [negative] * size
        assert control.serialize_ensemble(positive_ensemble) == control.serialize_ensemble(
            negative_ensemble
        )
        assert control.aggregate_action(positive_ensemble) == +1
        assert control.aggregate_action(negative_ensemble) == +1
        labels = (+1, -1)
        assert sum(control.aggregate_action(positive_ensemble) == label for label in labels) == 1


def test_strict_ensemble_serialization_is_permutation_invariant() -> None:
    control = StrictMetricControl(3)
    first = control.write_cue(control.identity_state(), [1.0, 0.0, 0.0])
    second = control.write_cue(control.identity_state(), [0.0, 1.0, 0.0])

    assert control.serialize_ensemble([first, second]) == control.serialize_ensemble([second, first])


def test_seeded_strict_members_are_distinct_and_sign_even_after_full_training_stream() -> None:
    dimension = 8
    control = StrictMetricControl(dimension)
    checkpoints: list[StrictMetricState] = []
    for seed in (6101, 6102):
        rng = np.random.default_rng(seed)
        dense = rng.normal(size=(dimension, dimension))
        initial_metric = dense @ dense.T + 0.5 * np.eye(dimension)
        state = control.make_state_from_metric(initial_metric)
        for epoch in range(4):
            for coordinate in range(dimension):
                cue = np.zeros(dimension)
                cue[coordinate] = -1.0 if rng.integers(0, 2) == 0 else 1.0
                state = control.write_cue(state, cue)
                for _ in range((4, 8, 16)[(epoch + coordinate) % 3]):
                    state = control.distract(
                        state,
                        rng.choice((-1.0, 1.0), size=dimension) / math.sqrt(dimension),
                    )
                action = control.action(state)
                before_reward = state
                state = control.reward(state, action, int(action == 1))
                assert state is before_reward
        checkpoints.append(state)

    assert control.serialize_state(checkpoints[0]) != control.serialize_state(checkpoints[1])
    query = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]) / math.sqrt(
        dimension
    )
    plus_members: list[StrictMetricState] = []
    minus_members: list[StrictMetricState] = []
    nuisance = np.ones(dimension) / math.sqrt(dimension)
    for checkpoint in checkpoints:
        plus = control.write_cue(control.from_snapshot(checkpoint), query)
        minus = control.write_cue(control.from_snapshot(checkpoint), -query)
        for _ in range(128):
            plus = control.distract(plus, nuisance)
            minus = control.distract(minus, nuisance)
        assert control.serialize_state(plus) == control.serialize_state(minus)
        plus_members.append(plus)
        minus_members.append(minus)

    assert control.serialize_ensemble(plus_members) == control.serialize_ensemble(minus_members)
    assert control.aggregate_action(plus_members) == control.aggregate_action(minus_members) == +1


@pytest.mark.parametrize("cue_sign", [-1.0, 1.0])
@pytest.mark.parametrize("ensemble_size", StrictMetricControl.ensemble_sizes)
def test_strict_bulk_write_is_byte_exact_scalar_equivalent_for_distinct_members(
    cue_sign: float, ensemble_size: int,
) -> None:
    dimension = 8
    control = StrictMetricControl(dimension)
    states: list[StrictMetricState] = []
    for seed in range(ensemble_size):
        rng = np.random.default_rng(9100 + seed)
        dense = rng.normal(size=(dimension, dimension))
        states.append(control.make_state_from_metric(dense @ dense.T + np.eye(dimension)))
    query = cue_sign * np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    query /= np.linalg.norm(query)

    scalar = tuple(control.write_cue(state, query) for state in states)
    bulk = control.write_ensemble(tuple(states), query)
    compatibility = control.write_cue_many(tuple(states), query)

    assert tuple(map(control.serialize_state, bulk)) == tuple(map(control.serialize_state, scalar))
    assert tuple(map(control.serialize_state, compatibility)) == tuple(
        map(control.serialize_state, scalar)
    )
    assert len(set(map(control.serialize_state, bulk))) == len(states)


def test_strict_bulk_distractors_are_scalar_equivalent_and_fail_closed() -> None:
    control = StrictMetricControl(3)
    states = tuple(
        control.make_state_from_metric(np.diag([1.0 + seed, 2.0, 3.0])) for seed in range(2)
    )
    observations = (
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    scalar = tuple(
        control.distract_many(state, observations)
        for state in states
    )

    bulk = control.distract_ensemble(states, observations)

    assert bulk is states
    assert tuple(map(control.serialize_state, bulk)) == tuple(map(control.serialize_state, scalar))
    invalid = (observations[0], [0.0, math.nan, 0.0], observations[2])
    with pytest.raises(ValueError, match=r"observations\[1\].*finite"):
        control.distract_ensemble(states, invalid)
    with pytest.raises(ValueError, match="ensemble size"):
        control.write_ensemble(states * 3, [1.0, 0.0, 0.0])


def test_strict_seeded_metric_constructor_rejects_non_spd_and_nonreal_inputs() -> None:
    control = StrictMetricControl(2)
    with pytest.raises(ValueError, match="symmetric"):
        control.make_state_from_metric([[1.0, 0.2], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        control.make_state_from_metric([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="real numeric"):
        control.make_state_from_metric([[True, False], [False, True]])


def test_malformed_unhashable_state_payloads_follow_explicit_validation() -> None:
    learner = EligibilityLearner(2)
    malformed = EligibilityState([0.0, 0.0], [0.0, 0.0], False)  # type: ignore[arg-type]

    restored = learner.snapshot(malformed)

    assert restored == EligibilityState((0.0, 0.0), (0.0, 0.0), False)


@pytest.mark.parametrize(
    "learner",
    [EligibilityLearner(3), HardLatchLearner(3), HomogeneousLearner(3), NoTraceControl(3)],
)
def test_snapshot_roundtrip_is_exact_and_continuation_matches(learner: object) -> None:
    state = learner.write_cue(learner.identity_state(), [1.0, 0.0, 0.0])
    snapshot = learner.snapshot(state)
    restored = learner.from_snapshot(snapshot)

    assert snapshot == state
    assert restored == state
    action = learner.action(state)
    assert learner.reward(state, action, 1) == learner.reward(restored, action, 1)


def test_certificates_disclose_state_budget_and_keep_broad_claims_false() -> None:
    routes = (
        (EligibilityLearner(8), 8, 8),
        (HardLatchLearner(8), 8, 8),
        (HomogeneousLearner(8), 45, 81),
        (StrictMetricControl(8), 36, 64),
        (NoTraceControl(8), 0, 0),
    )
    for learner, independent, serialized in routes:
        certificate = learner.certificate(learner.identity_state())
        assert isinstance(certificate, DelayedCreditCertificate)
        assert certificate.episodic_real_coordinates == independent
        assert certificate.episodic_serialized_entries == serialized
        assert not certificate.hidden_cue_field_present
        assert certificate.unmarked_distractors_are_exact_noops
        assert certificate.deterministic_tie_is_positive
        assert not certificate.general_delayed_credit_verified
        assert not certificate.learned_event_selection_verified
        assert not certificate.infinite_scc_intelligence_growth_verified
        assert not certificate.biological_fidelity_verified
        assert not certificate.cosmological_identity_verified
        assert not certificate.agi_evidence


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: EligibilityLearner(True), "positive built-in integer"),
        (lambda: EligibilityLearner(3, 0.0), r"\(0, 1\]"),
        (
            lambda: EligibilityLearner(3).write_cue(
                EligibilityLearner(3).identity_state(), [0.0, 0.0, 0.0]
            ),
            "nonzero",
        ),
        (
            lambda: HomogeneousLearner(3).write_cue(
                HomogeneousLearner(3).identity_state(), [1.0, math.nan, 0.0]
            ),
            "finite",
        ),
        (
            lambda: EligibilityLearner(3).write_cue(
                EligibilityLearner(3).identity_state(), [True, False, False]
            ),
            "real numeric",
        ),
        (
            lambda: EligibilityLearner(3).write_cue(
                EligibilityLearner(3).identity_state(), [1.0 + 0.0j, 0.0, 0.0]
            ),
            "real numeric",
        ),
        (lambda: decode_binary_reward(0, 1), r"-1 or \+1"),
        (lambda: decode_binary_reward(+1, 0.5), "0 or 1"),
        (
            lambda: StrictMetricControl(3).serialize_ensemble(
                [StrictMetricControl(3).identity_state()] * 3
            ),
            "ensemble size",
        ),
    ],
)
def test_invalid_types_zero_nonfinite_and_unregistered_ensembles_fail_closed(
    call: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


def test_public_exports_are_production_types() -> None:
    assert clarus.EligibilityState is EligibilityState
    assert clarus.HardLatchState is HardLatchState
    assert clarus.HomogeneousCreditState is HomogeneousCreditState
    assert clarus.StrictMetricState is StrictMetricState
    assert clarus.NoTraceState is NoTraceState
    assert clarus.DelayedCreditCertificate is DelayedCreditCertificate
    assert clarus.EligibilityLearner is EligibilityLearner
    assert clarus.HardLatchLearner is HardLatchLearner
    assert clarus.HomogeneousLearner is HomogeneousLearner
    assert clarus.StrictMetricControl is StrictMetricControl
    assert clarus.NoTraceControl is NoTraceControl
    assert clarus.decode_binary_reward is decode_binary_reward
