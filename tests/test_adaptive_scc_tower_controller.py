from dataclasses import replace
import inspect
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from reality_stone.clarus.adaptive_scc_tower_controller import (
    AdaptiveTowerController,
    CausalEvent,
    CrossScaleCut,
    CutDown,
    CutUp,
    InvalidTowerStateToken,
    LevelReset,
    SignFlip,
    StateShuffle,
    TimeShift,
    TowerCertificateError,
    UpperReset,
    permutation_hash,
)
from reality_stone.clarus.nested_scc_tower import NestedTowerGenerator, TowerSpec


class EqualInt(int):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


class EqualStr(str):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


def _generator(**kwargs) -> NestedTowerGenerator:
    defaults = {
        "shell_width": 3,
        "maximum_depth": 2,
        "observation_scales": (2.0, 4.0, 5.0),
    }
    defaults.update(kwargs)
    return NestedTowerGenerator(TowerSpec(**defaults))


def _warm(controller: AdaptiveTowerController) -> None:
    observations = (
        (1.0, -0.3, 0.2),
        (-0.4, 0.8, 0.1),
        (0.2, 0.1, -0.9),
        (0.7, -0.5, 0.4),
    )
    for tick, observation in enumerate(observations):
        controller.observe(CausalEvent(tick=tick, observation=observation))


def _restored(controller: AdaptiveTowerController) -> AdaptiveTowerController:
    restored = AdaptiveTowerController(controller.generator)
    restored.load_state_dict(controller.state_dict())
    return restored


def test_depth_growth_is_structural_grow_only_and_reports_exhaustion_truthfully() -> None:
    controller = AdaptiveTowerController(_generator())
    token0 = controller.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    assert token0.active_depth == 1
    assert controller.last_depth_decision is not None
    assert controller.last_depth_decision.extended
    assert not controller.last_depth_decision.exact_generic_compatibility

    token1 = controller.observe(CausalEvent(1, (0.0, 0.0, 0.0)))
    assert token1.active_depth == 2
    token2 = controller.observe(CausalEvent(2, (0.0, 0.0, 0.0)))
    assert token2.active_depth == 2
    assert controller.last_depth_decision is not None
    assert controller.last_depth_decision.exhausted
    assert "maximum finite depth" in controller.last_depth_decision.reason

    exact = AdaptiveTowerController(_generator(upward_gain=0.0))
    exact.observe(CausalEvent(0, (1.0, 0.0, 0.0)))
    assert exact.active_depth == 0
    assert exact.last_depth_decision is not None
    assert exact.last_depth_decision.exact_generic_compatibility


@pytest.mark.parametrize("upward_gain", [0.16, 1e-15, 1e-20])
def test_zero_maximum_depth_never_promotes_nonzero_boundary_coupling(upward_gain) -> None:
    controller = AdaptiveTowerController(_generator(maximum_depth=0, upward_gain=upward_gain))
    token = controller.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    decision = controller.last_depth_decision
    assert token.active_depth == 0
    assert decision is not None
    assert not decision.extended
    assert not decision.exact_generic_compatibility
    assert decision.exhausted

    exact = AdaptiveTowerController(_generator(maximum_depth=0, upward_gain=0.0))
    exact.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    assert exact.last_depth_decision is not None
    assert exact.last_depth_decision.exact_generic_compatibility
    assert not exact.last_depth_decision.exhausted


@pytest.mark.parametrize("upward_gain", [1e-15, 1e-16, 1e-20])
def test_tiny_nonzero_boundary_gain_still_forces_controller_growth(upward_gain) -> None:
    controller = AdaptiveTowerController(_generator(upward_gain=upward_gain))
    token = controller.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    assert token.active_depth == 1
    assert controller.last_depth_decision is not None
    assert controller.last_depth_decision.extended
    assert not controller.last_depth_decision.exact_generic_compatibility


def test_same_current_input_retains_history_and_state_swap_transfers_it() -> None:
    generator = _generator()
    history_a = AdaptiveTowerController(generator)
    history_b = AdaptiveTowerController(generator)
    for tick in range(3):
        history_a.observe(CausalEvent(tick, (1.0, -0.2, 0.4)))
        history_b.observe(CausalEvent(tick, (-1.0, 0.2, -0.4)))
    snapshot_a = history_a.state_dict()
    snapshot_b = history_b.state_dict()
    common = CausalEvent(3, (0.0, 0.0, 0.0))

    intact_a = AdaptiveTowerController(generator)
    intact_a.load_state_dict(snapshot_a)
    forecast_a = intact_a.read_forecast(intact_a.observe(common))
    intact_b = AdaptiveTowerController(generator)
    intact_b.load_state_dict(snapshot_b)
    forecast_b = intact_b.read_forecast(intact_b.observe(common))
    assert not np.allclose(forecast_a, forecast_b, rtol=0.0, atol=1e-12)

    donor_b = AdaptiveTowerController(generator)
    donor_b.load_state_dict(snapshot_b)
    swapped_to_b = donor_b.read_forecast(donor_b.observe(common))
    donor_a = AdaptiveTowerController(generator)
    donor_a.load_state_dict(snapshot_a)
    swapped_to_a = donor_a.read_forecast(donor_a.observe(common))
    assert np.array_equal(swapped_to_b, forecast_b)
    assert np.array_equal(swapped_to_a, forecast_a)

    reset_a = AdaptiveTowerController(generator)
    reset_b = AdaptiveTowerController(generator)
    reset_a.observe(CausalEvent(0, common.observation))
    reset_b.observe(CausalEvent(0, common.observation))
    assert reset_a.read_forecast(reset_a.latest_token) == reset_b.read_forecast(
        reset_b.latest_token
    )


def test_token_is_immutable_state_bound_and_stale_or_foreign_tokens_fail_closed() -> None:
    generator = _generator()
    left = AdaptiveTowerController(generator)
    right = AdaptiveTowerController(generator)
    token_left = left.observe(CausalEvent(0, (1.0, 0.0, 0.0)))
    token_right = right.observe(CausalEvent(0, (1.0, 0.0, 0.0)))
    assert len(token_left.state_hash) == 64
    with pytest.raises(InvalidTowerStateToken, match="foreign"):
        left.read_forecast(token_right)
    left.observe(CausalEvent(1, (0.0, 0.0, 0.0)))
    with pytest.raises(InvalidTowerStateToken, match="stale"):
        left.read_forecast(token_left)
    with pytest.raises(InvalidTowerStateToken, match="immutable"):
        left.read_forecast("not-a-token")


@pytest.mark.parametrize("field", ["episode_generation", "tick", "active_depth"])
@pytest.mark.parametrize("poison", [False, 0.0, EqualInt(999)])
def test_token_metadata_requires_nonnegative_exact_integers(field, poison) -> None:
    controller = AdaptiveTowerController(_generator(upward_gain=0.0))
    token = controller.observe(CausalEvent(0, (1.0, 0.0, 0.0)))
    assert token.episode_generation == token.tick == token.active_depth == 0
    with pytest.raises(ValueError, match="exact integer"):
        replace(token, **{field: poison})

    forged = replace(token)
    object.__setattr__(forged, field, poison)
    object.__setattr__(forged, "_validate_schema", lambda: None)
    with pytest.raises(InvalidTowerStateToken, match="schema"):
        controller.read_forecast(forged)


@pytest.mark.parametrize(
    "field,poison",
    [
        ("controller_identity", ""),
        ("controller_identity", EqualStr("forged")),
        ("state_hash", "g" * 64),
        ("state_hash", "0" * 63),
        ("state_hash", EqualStr("0" * 64)),
        ("parameter_hash", "A" * 64),
        ("parameter_hash", EqualStr("0" * 64)),
    ],
)
def test_token_identity_and_digest_schema_fail_closed(field, poison) -> None:
    controller = AdaptiveTowerController(_generator(upward_gain=0.0))
    token = controller.observe(CausalEvent(0, (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="token"):
        replace(token, **{field: poison})


def test_controller_generator_binding_is_read_only_and_constructor_sealed() -> None:
    class DerivedGenerator(NestedTowerGenerator):
        pass

    with pytest.raises(TypeError, match="exact NestedTowerGenerator"):
        AdaptiveTowerController(DerivedGenerator(_generator().spec))

    generator = _generator()
    controller = AdaptiveTowerController(generator)
    replacement = _generator(upward_gain=0.0)
    with pytest.raises(AttributeError):
        controller.generator = replacement
    assert controller.generator is generator

    before_states = controller.state_copy()
    object.__setattr__(controller, "_generator", replacement)
    with pytest.raises(ValueError, match="generator identity seal mismatch"):
        controller.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    assert controller.tick == -1
    assert all(
        np.array_equal(before, after) for before, after in zip(before_states, controller._states)
    )
    with pytest.raises(ValueError, match="generator identity seal mismatch"):
        controller.state_copy()


def test_controller_rejects_a_mutated_generator_before_state_commit() -> None:
    generator = _generator()
    controller = AdaptiveTowerController(generator)
    before_states = controller.state_copy()
    generator._within_base.setflags(write=True)
    generator._within_base[0, 0] = 100.0
    generator._within_base.setflags(write=False)
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        controller.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    assert controller.tick == -1
    assert controller.active_depth == 0
    assert all(
        np.array_equal(before, after) for before, after in zip(before_states, controller._states)
    )
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        controller.state_copy()


@pytest.mark.parametrize("poison", ["generator-swap", "spec", "operator"])
def test_reset_rejection_is_transactional_for_every_generator_seal_failure(poison) -> None:
    generator = _generator()
    controller = AdaptiveTowerController(generator)
    _warm(controller)
    before = controller.state_dict()
    before_trace = controller.last_trace

    if poison == "generator-swap":
        object.__setattr__(controller, "_generator", _generator(upward_gain=0.0))
    elif poison == "spec":
        spec = generator.spec
        object.__setattr__(spec, "input_gain", spec.input_gain + 0.01)
    else:
        generator._within_base.setflags(write=True)
        generator._within_base[0, 0] = 100.0
        generator._within_base.setflags(write=False)

    with pytest.raises(ValueError, match="seal mismatch"):
        controller.reset_episode()
    assert controller._controller_identity == before.controller_identity
    assert controller._episode_generation == before.episode_generation
    assert controller.tick == before.tick
    assert controller.active_depth == before.active_depth
    assert tuple(tuple(float(value) for value in state) for state in controller._states) == (
        before.states
    )
    assert (
        tuple(
            tuple(float(value) for value in message)
            for message in controller._previous_upward_messages
        )
        == before.previous_upward_messages
    )
    assert (
        tuple(
            tuple(float(value) for value in message)
            for message in controller._previous_downward_messages
        )
        == before.previous_downward_messages
    )
    assert controller.latest_token == before.latest_token
    assert controller._pending_intervention == before.pending_intervention
    assert controller.last_depth_decision == before.last_depth_decision
    assert controller.last_trace == before_trace


def test_reset_generation_prevents_an_old_token_from_resurrecting() -> None:
    controller = AdaptiveTowerController(_generator())
    event = CausalEvent(0, (1.0, -0.5, 0.25))
    old_token = controller.observe(event)
    old_output = controller.read_forecast(old_token)
    controller.reset_episode()
    new_token = controller.observe(event)
    assert controller.read_forecast(new_token) == old_output
    assert new_token != old_token
    assert new_token.episode_generation == old_token.episode_generation + 1
    with pytest.raises(InvalidTowerStateToken, match="episode"):
        controller.read_forecast(old_token)


def test_readout_has_no_raw_event_or_parent_output_bypass() -> None:
    generator = _generator()
    controller = AdaptiveTowerController(generator)
    caller_values = [1.0, -2.0, 0.5]
    event = CausalEvent(0, caller_values)
    token = controller.observe(event)
    forecast = controller.read_forecast(token)
    caller_values[:] = [999.0, 999.0, 999.0]
    assert controller.read_forecast(token) == forecast
    parameters = tuple(inspect.signature(controller.read_forecast).parameters)
    assert parameters == ("token",)
    forbidden = ("parent", "target", "posterior", "v5", "v8", "acbsm")
    assert not any(any(term in name.lower() for term in forbidden) for name in vars(controller))


def test_policy_mask_is_fail_closed_and_state_only() -> None:
    controller = AdaptiveTowerController(_generator())
    token = controller.observe(CausalEvent(0, (1.0, -1.0, 0.25)))
    decision = controller.read_policy(token, (True, False, True))
    assert decision.probabilities[1] == 0.0
    assert sum(decision.probabilities) == pytest.approx(1.0)
    assert decision.selected_action in (0, 2)
    with pytest.raises(ValueError, match="boolean"):
        controller.read_policy(token, (1, 0, 1))
    with pytest.raises(ValueError, match="at least one"):
        controller.read_policy(token, (False, False, False))


def test_reset_cut_shift_flip_and_shuffle_mutate_actual_next_update_tensors() -> None:
    base = AdaptiveTowerController(_generator())
    _warm(base)
    assert base.active_depth == 2
    event = CausalEvent(base.tick + 1, (-0.6, 0.2, 0.9))
    snapshot = base.state_dict()

    intact = _restored(base)
    intact.observe(event)
    assert intact.last_trace is not None
    intact_trace = intact.last_trace

    reset = base.with_intervention(LevelReset(1))
    reset.observe(event)
    assert reset.last_trace is not None
    assert reset.last_trace.intervention == "LevelReset"
    assert np.array_equal(reset.last_trace.state_before[1], np.zeros(3))
    assert not np.array_equal(intact_trace.state_before[1], np.zeros(3))
    assert reset.last_trace.state_after != intact_trace.state_after

    cut_up = base.with_intervention(CutUp(0))
    cut_up.observe(event)
    assert cut_up.last_trace is not None
    assert np.array_equal(cut_up.last_trace.consumed_upward_messages[0], np.zeros(3))
    assert not np.array_equal(cut_up.last_trace.raw_upward_messages[0], np.zeros(3))
    assert cut_up.last_trace.state_after != intact_trace.state_after

    cut_down = base.with_intervention(CutDown(0))
    cut_down.observe(event)
    assert cut_down.last_trace is not None
    assert np.array_equal(cut_down.last_trace.consumed_downward_messages[0], np.zeros(3))
    assert not np.array_equal(cut_down.last_trace.raw_downward_messages[0], np.zeros(3))
    assert cut_down.last_trace.state_after != intact_trace.state_after

    shifted = base.with_intervention(TimeShift(0))
    shifted.observe(event)
    assert shifted.last_trace is not None
    assert shifted.last_trace.consumed_upward_messages[0] == snapshot.previous_upward_messages[0]
    assert (
        shifted.last_trace.consumed_downward_messages[0] == snapshot.previous_downward_messages[0]
    )
    assert (
        shifted.last_trace.consumed_upward_messages[0] != shifted.last_trace.raw_upward_messages[0]
    )
    assert shifted.last_trace.state_after != intact_trace.state_after

    flipped = base.with_intervention(SignFlip(0))
    flipped.observe(event)
    assert flipped.last_trace is not None
    assert np.array_equal(
        flipped.last_trace.consumed_upward_messages[0],
        -np.asarray(flipped.last_trace.raw_upward_messages[0]),
    )
    assert np.array_equal(
        flipped.last_trace.consumed_downward_messages[0],
        -np.asarray(flipped.last_trace.raw_downward_messages[0]),
    )
    assert flipped.last_trace.state_after != intact_trace.state_after

    shuffle = StateShuffle(0, (2, 0, 1))
    shuffled = base.with_intervention(shuffle)
    shuffled.observe(event)
    assert shuffled.last_trace is not None
    expected = np.asarray(snapshot.states[0])[[2, 0, 1]]
    assert np.array_equal(shuffled.last_trace.state_before[0], expected)
    assert shuffle.permutation_manifest_hash
    assert shuffled.last_trace.state_after != intact_trace.state_after

    for lesion in (reset, cut_up, cut_down, shifted, flipped, shuffled):
        assert lesion.state_dict().states != snapshot.states
        for lesion_array, base_array in zip(lesion._states, base._states):
            assert not np.shares_memory(lesion_array, base_array)
    assert base.state_dict() == snapshot


def test_upper_reset_and_global_cross_cut_are_real_distinct_lesions() -> None:
    base = AdaptiveTowerController(_generator())
    _warm(base)
    event = CausalEvent(base.tick + 1, (-0.6, 0.2, 0.9))

    intact = _restored(base)
    intact.observe(event)
    reset = base.with_intervention(UpperReset())
    reset.observe(event)
    cut = base.with_intervention(CrossScaleCut())
    cut.observe(event)

    assert reset.last_trace is not None
    assert cut.last_trace is not None
    assert reset.last_trace.intervention == "UpperReset"
    assert cut.last_trace.intervention == "CrossScaleCut"
    assert all(
        np.array_equal(message, np.zeros(3))
        for message in reset.last_trace.consumed_upward_messages
    )
    assert all(
        np.array_equal(message, np.zeros(3))
        for message in cut.last_trace.consumed_downward_messages
    )
    assert reset.last_trace.state_after != intact.last_trace.state_after
    assert cut.last_trace.state_after != intact.last_trace.state_after


@pytest.mark.parametrize(
    "intervention",
    [
        LevelReset(1),
        UpperReset(),
        CrossScaleCut(),
        CutUp(0),
        CutDown(0),
        TimeShift(0),
        SignFlip(0),
        StateShuffle(0, (2, 0, 1)),
    ],
    ids=(
        "reset",
        "upper-reset",
        "cross-scale-cut",
        "cut-up",
        "cut-down",
        "time-shift",
        "sign-flip",
        "shuffle",
    ),
)
def test_pending_intervention_snapshot_roundtrip_is_bitwise_exact(intervention) -> None:
    base = AdaptiveTowerController(_generator())
    _warm(base)
    lesioned = base.with_intervention(intervention)
    snapshot = lesioned.state_dict()
    assert snapshot.latest_token is None
    assert snapshot.pending_intervention == intervention

    restored = AdaptiveTowerController(base.generator)
    restored.load_state_dict(snapshot)
    assert restored.state_dict() == snapshot

    event = CausalEvent(base.tick + 1, (-0.6, 0.2, 0.9))
    lesioned_token = lesioned.observe(event)
    restored_token = restored.observe(event)
    assert restored.last_trace == lesioned.last_trace
    assert restored.state_dict() == lesioned.state_dict()
    assert restored_token == lesioned_token
    assert restored.read_forecast(restored_token) == lesioned.read_forecast(lesioned_token)


def test_pending_intervention_snapshot_validation_is_atomic() -> None:
    source = AdaptiveTowerController(_generator())
    _warm(source)
    target = AdaptiveTowerController(source.generator)
    before = target.state_dict()

    contradictory = _restored(source)
    contradictory._pending_intervention = CutUp(0)
    with pytest.raises(ValueError, match="cannot carry a state token"):
        target.load_state_dict(contradictory.state_dict())
    assert target.state_dict() == before

    unknown = _restored(source)
    unknown._latest_token = None
    unknown._pending_intervention = "not-an-intervention"
    with pytest.raises(ValueError, match="unknown type"):
        target.load_state_dict(unknown.state_dict())
    assert target.state_dict() == before

    disconnected = _restored(source)
    disconnected._latest_token = None
    disconnected._pending_intervention = CutUp(2)
    with pytest.raises(ValueError, match="connect two active"):
        target.load_state_dict(disconnected.state_dict())
    assert target.state_dict() == before

    forged_shuffle = StateShuffle(0, (2, 0, 1))
    object.__setattr__(forged_shuffle, "permutation_manifest_hash", "bad-hash")
    shuffled = _restored(source)
    shuffled._latest_token = None
    shuffled._pending_intervention = forged_shuffle
    with pytest.raises(ValueError, match="permutation hash mismatch"):
        target.load_state_dict(shuffled.state_dict())
    assert target.state_dict() == before


def test_intervention_indices_hashes_and_one_tick_delay_fail_closed() -> None:
    controller = AdaptiveTowerController(_generator())
    _warm(controller)
    with pytest.raises(ValueError, match="active"):
        controller.with_intervention(LevelReset(3))
    with pytest.raises(ValueError, match="connect"):
        controller.with_intervention(CutUp(2))
    with pytest.raises(ValueError, match="one-tick"):
        TimeShift(0, ticks=2)
    with pytest.raises(ValueError, match="integer one"):
        TimeShift(0, ticks=True)
    with pytest.raises(ValueError, match="integer one"):
        TimeShift(0, ticks=1.0)
    with pytest.raises(ValueError, match="does not match"):
        StateShuffle(0, (0, 1, 2), "bad-hash")
    with pytest.raises(ValueError, match="exact SHA-256"):
        StateShuffle(0, (0, 1, 2), False)
    invalid = StateShuffle(0, (0, 0, 1))
    with pytest.raises(ValueError, match="full shell permutation"):
        controller.with_intervention(invalid)
    with pytest.raises(ValueError, match="exact integers"):
        StateShuffle(0, (0, 1, 1.7))
    with pytest.raises(ValueError, match="exact integers"):
        StateShuffle(0, (0, 1, True))


@pytest.mark.parametrize(
    "poison",
    [(False, 1, 2), (0.0, 1, 2), ("0", 1, 2), (EqualInt(0), 1, 2)],
)
def test_permutation_manifest_helper_never_coerces_entries(poison) -> None:
    with pytest.raises(ValueError, match="exact integers"):
        permutation_hash(poison)
    assert permutation_hash([2, 0, 1]) == permutation_hash((2, 0, 1))


@pytest.mark.parametrize(
    "kind,field,poison",
    [
        ("shuffle", "permutation", (False, 1, 2)),
        ("shuffle", "permutation", (0.0, 1, 2)),
        ("shuffle", "permutation", (EqualInt(0), 1, 2)),
        ("shuffle", "permutation", [0, 1, 2]),
        ("shuffle", "permutation_manifest_hash", False),
        ("shuffle", "permutation_manifest_hash", EqualStr("0" * 64)),
        ("reset", "level", EqualInt(1)),
        ("cut-up", "bridge_level", EqualInt(0)),
        ("cut-down", "bridge_level", EqualInt(0)),
        ("time", "ticks", False),
        ("time", "ticks", 1.0),
        ("time", "ticks", EqualInt(999)),
        ("time", "bridge_level", EqualInt(0)),
        ("time", "direction", "sideways"),
        ("time", "direction", EqualStr("both")),
        ("sign", "direction", "sideways"),
        ("sign", "direction", EqualStr("both")),
        ("sign", "bridge_level", 0.0),
    ],
)
def test_frozen_intervention_poison_is_revalidated_without_state_mutation(
    kind, field, poison
) -> None:
    def valid_intervention():
        if kind == "shuffle":
            return StateShuffle(0, (2, 0, 1))
        if kind == "reset":
            return LevelReset(1)
        if kind == "cut-up":
            return CutUp(0)
        if kind == "cut-down":
            return CutDown(0)
        if kind == "time":
            return TimeShift(0)
        return SignFlip(0)

    base = AdaptiveTowerController(_generator())
    _warm(base)
    before = base.state_dict()
    invalid = valid_intervention()
    object.__setattr__(invalid, field, poison)
    with pytest.raises(ValueError):
        base.with_intervention(invalid)
    assert base.state_dict() == before

    pending = valid_intervention()
    arm = base.with_intervention(pending)
    object.__setattr__(pending, field, poison)
    poisoned_before = arm.state_dict()
    event = CausalEvent(base.tick + 1, (0.1, -0.2, 0.3))
    with pytest.raises(ValueError):
        arm.observe(event)
    assert arm.state_dict() == poisoned_before


def test_snapshot_restore_preserves_state_delay_token_and_continuation_exactly() -> None:
    original = AdaptiveTowerController(_generator())
    _warm(original)
    snapshot = original.state_dict()
    restored = AdaptiveTowerController(original.generator)
    restored.load_state_dict(snapshot)
    assert restored.state_dict() == snapshot
    assert restored.latest_token is not None
    assert restored.read_forecast(restored.latest_token) == original.read_forecast(
        original.latest_token
    )
    for restored_array, original_array in zip(restored._states, original._states):
        assert not np.shares_memory(restored_array, original_array)

    event = CausalEvent(original.tick + 1, (0.3, 0.6, -0.2))
    token_original = original.observe(event)
    token_restored = restored.observe(event)
    assert original.read_forecast(token_original) == restored.read_forecast(token_restored)
    assert original.last_trace == restored.last_trace


def test_genuine_snapshot_rejects_parameter_mismatch_and_token_inconsistency() -> None:
    source = AdaptiveTowerController(_generator())
    _warm(source)
    snapshot = source.state_dict()
    mismatched = AdaptiveTowerController(_generator(recurrence_gain=0.23))
    with pytest.raises(ValueError, match="manifest mismatch"):
        mismatched.load_state_dict(snapshot)

    inconsistent = _restored(source)
    inconsistent._states[0] = np.asarray((0.9, 0.9, 0.9), dtype=np.float64)
    signed_inconsistent = inconsistent.state_dict()
    target = AdaptiveTowerController(source.generator)
    before = target.state_dict()
    with pytest.raises(InvalidTowerStateToken, match="inconsistent"):
        target.load_state_dict(signed_inconsistent)
    assert target.state_dict() == before


def test_snapshot_integrity_tag_binds_every_causal_and_diagnostic_field_atomically() -> None:
    source = AdaptiveTowerController(_generator())
    _warm(source)
    snapshot = source.state_dict()
    assert snapshot.last_depth_decision is not None
    target = AdaptiveTowerController(source.generator)
    before = target.state_dict()
    changed_states = list(snapshot.states)
    changed_states[0] = (0.9, 0.9, 0.9)
    tampered = (
        replace(snapshot, parameter_hash="0" * 64),
        replace(snapshot, controller_identity="forged"),
        replace(snapshot, tick=snapshot.tick + 1),
        replace(snapshot, states=tuple(changed_states)),
        replace(snapshot, latest_token=None),
        replace(snapshot, pending_intervention=CutUp(0)),
        replace(
            snapshot,
            last_depth_decision=replace(snapshot.last_depth_decision, reason="forged"),
        ),
        replace(snapshot, integrity_tag="0" * 64),
    )
    for forged in tampered:
        with pytest.raises(ValueError, match="process-local integrity tag mismatch"):
            target.load_state_dict(forged)
        assert target.state_dict() == before


@pytest.mark.parametrize(
    "attribute,poison,error",
    [
        ("_episode_generation", EqualInt(0), "generation"),
        ("_tick", EqualInt(0), "tick or active depth"),
        ("_active_depth", EqualInt(0), "tick or active depth"),
        ("_controller_identity", EqualStr("forged"), "identity"),
    ],
)
def test_genuinely_tagged_snapshot_metadata_still_requires_exact_builtin_types(
    attribute, poison, error
) -> None:
    source = AdaptiveTowerController(_generator())
    object.__setattr__(source, attribute, poison)
    signed = source.state_dict()
    target = AdaptiveTowerController(source.generator)
    before = target.state_dict()
    with pytest.raises(ValueError, match=error):
        target.load_state_dict(signed)
    assert target.state_dict() == before


def test_private_fixture_state_mutation_still_demonstrates_history_mediation() -> None:
    controller = AdaptiveTowerController(_generator())
    _warm(controller)
    intact = _restored(controller)
    mutated = _restored(controller)
    mutated._states[0] = np.asarray((0.9, 0.9, 0.9), dtype=np.float64)
    mutated._latest_token = None

    event = CausalEvent(controller.tick + 1, (0.0, 0.0, 0.0))
    intact_output = intact.read_forecast(intact.observe(event))
    mutated_output = mutated.read_forecast(mutated.observe(event))
    assert not np.array_equal(intact_output, mutated_output)


def test_snapshot_rejects_hidden_inactive_state_and_delay_data() -> None:
    controller = AdaptiveTowerController(_generator())
    controller.observe(CausalEvent(0, (1.0, 0.0, -0.5)))
    assert controller.active_depth == 1

    state_source = _restored(controller)
    state_source._latest_token = None
    state_source._states[2] = np.asarray((0.1, 0.0, 0.0), dtype=np.float64)
    with pytest.raises(ValueError, match="inactive snapshot states"):
        controller.load_state_dict(state_source.state_dict())

    message_source = _restored(controller)
    message_source._latest_token = None
    message_source._previous_upward_messages[1] = np.asarray((0.1, 0.0, 0.0), dtype=np.float64)
    with pytest.raises(ValueError, match="inactive snapshot message"):
        controller.load_state_dict(message_source.state_dict())


@pytest.mark.parametrize(
    "outside",
    [
        1.0 + 5e-13,
        -1.0 - 5e-13,
        np.nextafter(1.0, math.inf),
        np.nextafter(-1.0, -math.inf),
    ],
)
def test_snapshot_state_domain_is_the_exact_closed_interval(outside) -> None:
    controller = AdaptiveTowerController(_generator())
    source = _restored(controller)
    source._latest_token = None
    source._states[0] = np.asarray((outside, 0.0, 0.0), dtype=np.float64)
    with pytest.raises(ValueError, match="normalized state domain"):
        controller.load_state_dict(source.state_dict())


def test_forged_event_types_and_poisoned_fields_are_rejected_atomically() -> None:
    controller = AdaptiveTowerController(_generator())
    before = controller.state_dict()

    missing_fields = object.__new__(CausalEvent)
    bad_tick_bool = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(bad_tick_bool, "tick", False)
    bad_tick_float = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(bad_tick_float, "tick", 0.0)
    bad_tick_subclass = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(bad_tick_subclass, "tick", EqualInt(999))
    object.__setattr__(bad_tick_subclass, "_validate_schema", lambda: None)
    list_observation = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(list_observation, "observation", [0.0, 0.0, 0.0])
    bool_observation = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(bool_observation, "observation", (False, 0.0, 0.0))
    nonfinite_observation = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(nonfinite_observation, "observation", (math.inf, 0.0, 0.0))
    wrong_width = CausalEvent(0, (0.0, 0.0, 0.0))
    object.__setattr__(wrong_width, "observation", (0.0,))

    forged_events = (
        SimpleNamespace(tick=False, observation=(0.0, 0.0, 0.0)),
        missing_fields,
        bad_tick_bool,
        bad_tick_float,
        bad_tick_subclass,
        list_observation,
        bool_observation,
        nonfinite_observation,
        wrong_width,
    )
    for forged in forged_events:
        with pytest.raises(ValueError):
            controller.observe(forged)
        assert controller.state_dict() == before


def test_token_construction_failure_precedes_observe_commit(monkeypatch) -> None:
    controller = AdaptiveTowerController(_generator())
    _warm(controller)
    before = controller.state_dict()
    before_trace = controller.last_trace

    def fail_token(**_kwargs):
        raise RuntimeError("injected token construction failure")

    monkeypatch.setattr(controller, "_build_token", fail_token)
    with pytest.raises(RuntimeError, match="token construction"):
        controller.observe(CausalEvent(controller.tick + 1, (0.1, -0.2, 0.3)))
    assert controller.state_dict() == before
    assert controller.last_trace == before_trace


def test_nonfinite_inputs_wrong_ticks_and_noninteger_fields_fail_closed() -> None:
    controller = AdaptiveTowerController(_generator())
    with pytest.raises(ValueError, match="nonnegative integer"):
        CausalEvent(True, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="nonnegative integer"):
        CausalEvent(0.5, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="nonnegative integer"):
        CausalEvent(EqualInt(999), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        CausalEvent(0, (0.0, math.inf, 0.0))
    with pytest.raises(ValueError, match="real number"):
        CausalEvent(0, (0.0, "1.0", 0.0))
    with pytest.raises(ValueError, match="real number"):
        CausalEvent(0, (0.0, True, 0.0))
    with pytest.raises(ValueError, match="exactly 0"):
        controller.observe(CausalEvent(1, (0.0, 0.0, 0.0)))
    controller.observe(CausalEvent(0, (1e300, -1e300, 0.0)))
    assert all(
        np.all(np.isfinite(state)) and np.max(np.abs(state)) <= 1.0
        for state in controller.state_copy()
    )
    with pytest.raises(ValueError, match="exactly 1"):
        controller.observe(CausalEvent(0, (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="exactly 1"):
        controller.observe(CausalEvent(2, (0.0, 0.0, 0.0)))


@pytest.mark.parametrize(
    "generator,event,error",
    [
        (
            NestedTowerGenerator(
                TowerSpec(
                    shell_width=1,
                    maximum_depth=2,
                    observation_scales=(np.nextafter(0.0, 1.0),),
                )
            ),
            CausalEvent(0, (np.finfo(np.float64).max,)),
            "normalized observation",
        ),
        (
            NestedTowerGenerator(
                TowerSpec(
                    shell_width=1,
                    maximum_depth=2,
                    observation_scales=(1.0,),
                    input_gain=1e308,
                )
            ),
            CausalEvent(0, (np.finfo(np.float64).max,)),
            "recurrent drive",
        ),
    ],
)
def test_rejected_overflow_event_is_warning_free_and_transactionally_atomic(
    generator, event, error
) -> None:
    controller = AdaptiveTowerController(generator)
    before = controller.state_dict()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=error):
            controller.observe(event)
    assert captured == []
    assert controller.state_dict() == before


def test_controller_refuses_topology_only_or_schedule_mismatched_stability() -> None:
    unstable = _generator(
        recurrence_gain=0.8,
        upward_gain=0.2,
        downward_gain=0.1,
    )
    assert unstable.audit_prefix(2).is_strongly_connected
    with pytest.raises(TowerCertificateError, match="strict"):
        AdaptiveTowerController(unstable)
