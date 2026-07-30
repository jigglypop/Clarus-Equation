from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from reality_stone.clarus.cloudcell import CloudCellInput, brain_runtime_kleisli_arrow
from reality_stone.clarus.markov_kleisli import (
    FiniteDistribution,
    ProbabilityAtom,
    deterministic_kleisli_arrow,
    kleisli_compose,
    kleisli_identity,
    state_bind,
    state_pure,
)
from reality_stone.clarus.runtime import (
    BrainRuntime,
    BrainRuntimeConfig,
    BrainRuntimeSnapshot,
    RuntimeMode,
)


def _distribution(*atoms: tuple[object, float]) -> FiniteDistribution[object]:
    return FiniteDistribution(tuple(ProbabilityAtom(value, mass) for value, mass in atoms))


def _assert_distribution_equal(
    left: FiniteDistribution[object],
    right: FiniteDistribution[object],
) -> None:
    assert left.equivalent(right, key=lambda value: value, abs_tol=1e-12)


def _branch(value: int) -> FiniteDistribution[object]:
    return _distribution((value - 1, 0.25), (value + 2, 0.75))


def _second_branch(value: int) -> FiniteDistribution[object]:
    return _distribution((value * 2, 0.4), (-value, 0.6))


def test_finite_distribution_rejects_nonprobability_mass() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _distribution(("bad", -0.1), ("other", 1.1))
    with pytest.raises(ValueError, match="sum to one"):
        _distribution(("bad", 0.2), ("other", 0.2))


def test_probability_monad_left_and_right_identity_on_seeded_distributions() -> None:
    rng = random.Random(731)
    for _ in range(32):
        value = rng.randrange(-20, 21)
        _assert_distribution_equal(FiniteDistribution.pure(value).bind(_branch), _branch(value))

        weights = [rng.random() for _ in range(4)]
        total = sum(weights)
        distribution = FiniteDistribution(
            tuple(
                ProbabilityAtom(index - 2, weight / total)
                for index, weight in enumerate(weights)
            )
        )
        _assert_distribution_equal(
            distribution.bind(FiniteDistribution.pure),
            distribution,
        )


def test_probability_monad_associativity_on_seeded_distributions() -> None:
    rng = random.Random(1907)
    for _ in range(32):
        weight = rng.random()
        distribution = _distribution((-2, weight), (3, 1.0 - weight))
        left = distribution.bind(_branch).bind(_second_branch)
        right = distribution.bind(lambda value: _branch(value).bind(_second_branch))
        _assert_distribution_equal(left, right)


def _state_branch(value: int):
    def run(state: int) -> FiniteDistribution[object]:
        return _distribution(
            ((state + value, value + 1), 0.35),
            ((state - value, value - 1), 0.65),
        )

    return run


def _state_second_branch(value: int):
    def run(state: int) -> FiniteDistribution[object]:
        return _distribution(
            ((state + 2, value * 2), 0.6),
            ((state - 3, -value), 0.4),
        )

    return run


def test_probability_state_monad_laws() -> None:
    for state in range(-4, 5):
        for value in range(-3, 4):
            _assert_distribution_equal(
                state_bind(state_pure(value), _state_branch)(state),
                _state_branch(value)(state),
            )

            computation = _state_branch(value)
            _assert_distribution_equal(
                state_bind(computation, state_pure)(state),
                computation(state),
            )

            left = state_bind(
                state_bind(computation, _state_second_branch),
                _state_branch,
            )(state)
            right = state_bind(
                computation,
                lambda item: state_bind(_state_second_branch(item), _state_branch),
            )(state)
            _assert_distribution_equal(left, right)


def test_kleisli_composition_identity_and_associativity() -> None:
    identity = kleisli_identity
    first = _state_branch
    second = _state_second_branch
    third = _state_branch

    for state in range(-3, 4):
        for value in range(-2, 3):
            _assert_distribution_equal(
                kleisli_compose(identity, first)(value)(state),
                first(value)(state),
            )
            _assert_distribution_equal(
                kleisli_compose(first, identity)(value)(state),
                first(value)(state),
            )
            _assert_distribution_equal(
                kleisli_compose(kleisli_compose(first, second), third)(value)(state),
                kleisli_compose(first, kleisli_compose(second, third))(value)(state),
            )


def test_deterministic_transition_is_a_dirac_kleisli_arrow() -> None:
    arrow = deterministic_kleisli_arrow(
        lambda state, value: (state + value, f"observed:{value}")
    )
    result = arrow(4)(7)
    assert result == FiniteDistribution.pure((11, "observed:4"))


def _make_delayed_runtime(dim: int = 16) -> BrainRuntime:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(dim, dim, generator=generator) * 0.08
    weight.fill_diagonal_(0.0)
    return BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=dim,
            active_ratio=0.5,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=True,
            max_axon_delay=3,
            memory_capacity=8,
        ),
        backend="torch",
        device="cpu",
    )


_SNAPSHOT_TENSORS = (
    "weight",
    "activation",
    "refractory",
    "memory_trace",
    "adaptation",
    "stp_u",
    "stp_x",
    "bitfield",
    "goal",
    "lifecycle",
    "inactive_steps",
    "delay_buffer",
)


def _assert_snapshot_equal(
    left: BrainRuntimeSnapshot,
    right: BrainRuntimeSnapshot,
) -> None:
    assert left.config == right.config
    for name in _SNAPSHOT_TENSORS:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)

    scalar_names = (
        "mode",
        "sleep_pressure",
        "arousal",
        "step",
        "mode_occupancy",
        "active_ratio_ema",
        "stdp_prev_critic_score",
        "stdp_updates",
        "circadian_phase",
        "circadian_value",
        "nrem_cycle_count",
        "delay_idx",
        "brainwave_history",
        "last_stdp_gate",
    )
    for name in scalar_names:
        assert getattr(left, name) == getattr(right, name)

    assert left.hippocampus.keys() == right.hippocampus.keys()
    for name in ("dim", "capacity", "priority"):
        assert left.hippocampus[name] == right.hippocampus[name]
    for name in ("keys", "values"):
        torch.testing.assert_close(
            left.hippocampus[name],
            right.hippocampus[name],
            rtol=0.0,
            atol=0.0,
        )
    assert left.stdp_tracker is None
    assert right.stdp_tracker is None


def test_full_snapshot_is_a_sufficient_state_for_continuation() -> None:
    runtime = _make_delayed_runtime()
    prefix_modes = (
        RuntimeMode.WAKE,
        RuntimeMode.NREM,
        RuntimeMode.NREM,
        RuntimeMode.REM,
        RuntimeMode.WAKE,
    )
    for index, mode in enumerate(prefix_modes):
        external = torch.linspace(0.1, 0.6, runtime.config.dim) + index * 0.02
        runtime.step(external_input=external, force_mode=mode)

    snapshot = runtime.snapshot()
    assert snapshot.delay_buffer is not None
    assert snapshot.delay_idx == len(prefix_modes)
    assert snapshot.nrem_cycle_count == 2
    assert snapshot.circadian_phase == len(prefix_modes)
    assert len(snapshot.brainwave_history) == len(prefix_modes)

    restored = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    _assert_snapshot_equal(runtime.snapshot(), restored.snapshot())

    suffix_modes = (
        RuntimeMode.REM,
        RuntimeMode.WAKE,
        RuntimeMode.NREM,
        RuntimeMode.WAKE,
    )
    for index, mode in enumerate(suffix_modes):
        external = torch.linspace(-0.3, 0.4, runtime.config.dim) - index * 0.01
        original_step = runtime.step(external_input=external, force_mode=mode)
        restored_step = restored.step(external_input=external, force_mode=mode)
        assert original_step == restored_step
        _assert_snapshot_equal(runtime.snapshot(), restored.snapshot())


def test_brain_runtime_is_a_dirac_state_probability_kleisli_arrow() -> None:
    runtime = _make_delayed_runtime()
    for index in range(4):
        runtime.step(
            external_input=torch.linspace(0.0, 0.4, runtime.config.dim) + index * 0.01,
            force_mode=RuntimeMode.WAKE,
        )
    state = runtime.snapshot()
    control = CloudCellInput(
        external_input=torch.linspace(-0.2, 0.3, runtime.config.dim),
        force_mode=RuntimeMode.NREM,
    )
    arrow = brain_runtime_kleisli_arrow(backend="torch", device="cpu")

    first = arrow(control)(state)
    second = arrow(control)(state)

    assert len(first.atoms) == 1
    assert first.atoms[0].probability == 1.0
    first_state, first_output = first.atoms[0].value
    second_state, second_output = second.atoms[0].value
    assert first_output == second_output
    _assert_snapshot_equal(first_state, second_state)
    assert state.step == 4
    assert state.nrem_cycle_count == 0


def test_snapshot_and_restore_do_not_alias_runtime_config_or_tensors() -> None:
    runtime = _make_delayed_runtime()
    runtime.step(
        external_input=torch.linspace(0.0, 0.5, runtime.config.dim),
        force_mode=RuntimeMode.WAKE,
    )
    snapshot = runtime.snapshot()
    original_ratio = snapshot.config.active_ratio
    original_activation = snapshot.activation.clone()

    runtime.config.active_ratio = 0.9
    runtime.activation.add_(0.5)
    assert snapshot.config.active_ratio == original_ratio
    torch.testing.assert_close(snapshot.activation, original_activation, rtol=0.0, atol=0.0)

    restored = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    snapshot.config.active_ratio = 0.1
    snapshot.activation.add_(0.25)
    assert restored.config.active_ratio == original_ratio
    torch.testing.assert_close(restored.activation, original_activation, rtol=0.0, atol=0.0)


def test_restore_rejects_malformed_delay_state() -> None:
    snapshot = _make_delayed_runtime().snapshot()
    missing = replace(snapshot, delay_buffer=None)
    with pytest.raises(ValueError, match="required when axon delay is enabled"):
        BrainRuntime.from_snapshot(missing, backend="torch", device="cpu")

    malformed = replace(
        snapshot,
        delay_buffer=torch.zeros(2, snapshot.config.dim),
    )
    with pytest.raises(ValueError, match="delay buffer shape"):
        BrainRuntime.from_snapshot(malformed, backend="torch", device="cpu")
