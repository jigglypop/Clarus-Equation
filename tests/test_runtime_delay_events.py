"""Focused contracts for event-time axonal delay semantics."""

import pytest
import torch

from reality_stone.clarus.runtime import (
    BrainRuntime,
    BrainRuntimeConfig,
    ModuleLifecycle,
    RuntimeMode,
    _HAS_RUST_KERNEL,
    _LIFECYCLE_TO_CODE,
)


DIM = 2
SOURCE = 0
TARGET = 1


def _config(*, axon_delay: bool = True, max_axon_delay: int = 2) -> BrainRuntimeConfig:
    return BrainRuntimeConfig(
        dim=DIM,
        active_ratio=0.5,
        active_threshold=0.0,
        noise_sigma=0.0,
        dale_law=False,
        axon_delay=axon_delay,
        max_axon_delay=max_axon_delay,
        external_gain=0.0,
        goal_gain=0.0,
        replay_gain=0.0,
        refractory_scale=0.0,
        hippocampal_encoding_enabled=False,
    )


def _runtime(*, backend: str = "torch", max_axon_delay: int = 2) -> BrainRuntime:
    weight = torch.zeros((DIM, DIM), dtype=torch.float32)
    weight[TARGET, SOURCE] = 1.0
    return BrainRuntime(
        weight,
        config=_config(max_axon_delay=max_axon_delay),
        backend=backend,
        device="cpu",
    )


def _set_source(runtime: BrainRuntime, *, activation: float, active: bool) -> None:
    runtime.activation.zero_()
    runtime.activation[SOURCE] = activation
    runtime.lifecycle.fill_(_LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT])
    if active:
        runtime.lifecycle[SOURCE] = _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]


def _private_step(runtime: BrainRuntime) -> torch.Tensor:
    _, recurrent, _ = runtime._step_torch(
        torch.zeros(DIM),
        torch.zeros(DIM),
        RuntimeMode.WAKE,
    )
    return recurrent


def test_emitted_packet_arrives_after_exact_delay_despite_source_deactivation() -> None:
    runtime = _runtime(max_axon_delay=2)

    _set_source(runtime, activation=1.0, active=True)
    recurrent_0 = _private_step(runtime)
    emitted = runtime._delay_buffer[0].clone()
    assert torch.equal(recurrent_0, torch.zeros(DIM))
    assert emitted[SOURCE] > 0.0

    _set_source(runtime, activation=0.0, active=False)
    recurrent_1 = _private_step(runtime)
    assert torch.equal(recurrent_1, torch.zeros(DIM))

    _set_source(runtime, activation=0.0, active=False)
    recurrent_2 = _private_step(runtime)
    assert recurrent_2[TARGET].item() == pytest.approx(emitted[SOURCE].item())


def test_inactive_emission_cannot_be_created_by_activation_at_arrival() -> None:
    runtime = _runtime(max_axon_delay=2)

    _set_source(runtime, activation=1.0, active=False)
    assert torch.equal(_private_step(runtime), torch.zeros(DIM))
    assert torch.equal(runtime._delay_buffer[0], torch.zeros(DIM))

    _set_source(runtime, activation=0.0, active=False)
    assert torch.equal(_private_step(runtime), torch.zeros(DIM))

    _set_source(runtime, activation=1.0, active=True)
    assert torch.equal(_private_step(runtime), torch.zeros(DIM))


def test_delay_snapshot_continuation_preserves_packet_and_cursor() -> None:
    runtime = _runtime(max_axon_delay=2)
    _set_source(runtime, activation=0.75, active=True)
    _private_step(runtime)
    snapshot = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")

    assert snapshot.delay_idx == 1
    assert snapshot.delay_buffer is not None
    assert snapshot.delay_buffer[SOURCE, SOURCE] > 0.0
    torch.testing.assert_close(restored._delay_buffer, runtime._delay_buffer, rtol=0.0, atol=0.0)

    for candidate in (runtime, restored):
        _set_source(candidate, activation=0.0, active=False)
    recurrent_a = _private_step(runtime)
    recurrent_b = _private_step(restored)
    torch.testing.assert_close(recurrent_a, recurrent_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored.activation, runtime.activation, rtol=0.0, atol=0.0)
    assert restored._delay_idx == runtime._delay_idx == 2


def test_delay_backend_is_torch_only_and_rust_fails_closed() -> None:
    auto = _runtime(backend="auto")
    assert not auto._use_rust()

    with pytest.raises(ValueError, match="does not support axon_delay=True"):
        _runtime(backend="rust")

    mutable_auto = BrainRuntime(
        torch.zeros((DIM, DIM)),
        config=_config(axon_delay=False),
        backend="auto",
        device="cpu",
    )
    mutable_auto.config.axon_delay = True
    assert not mutable_auto._use_rust()

    if _HAS_RUST_KERNEL:
        mutable_rust = BrainRuntime(
            torch.zeros((DIM, DIM)),
            config=_config(axon_delay=False),
            backend="rust",
            device="cpu",
        )
        mutable_rust.config.axon_delay = True
        with pytest.raises(ValueError, match="does not support axon_delay=True"):
            mutable_rust._use_rust()
        with pytest.raises(ValueError, match="does not support axon_delay=True"):
            mutable_rust._step_rust(torch.zeros(DIM), torch.zeros(DIM), RuntimeMode.WAKE)
