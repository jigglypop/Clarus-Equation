from __future__ import annotations

from dataclasses import asdict

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


DIM = 3
WEIGHT = torch.tensor(
    [[0.32, -0.18, 0.07], [0.11, 0.27, -0.21], [-0.15, 0.09, 0.24]],
    dtype=torch.float32,
)
EXTERNAL = torch.tensor([0.14, -0.09, 0.21], dtype=torch.float32)


def _config(**overrides) -> BrainRuntimeConfig:
    values = {
        "dim": DIM,
        "active_ratio": 2.0 / 3.0,
        "noise_sigma": 0.0,
        "dale_law": False,
        "axon_delay": False,
        "max_axon_delay": 2,
        "memory_capacity": 1,
        "memory_topk": 1,
        "hippocampal_encoding_enabled": False,
        "f1_self_measure": False,
        "stdp_enabled": False,
    }
    values.update(overrides)
    return BrainRuntimeConfig(**values)


def _runtime(
    config: BrainRuntimeConfig,
    *,
    backend: str = "torch",
    weight: torch.Tensor = WEIGHT,
) -> BrainRuntime:
    runtime = BrainRuntime(weight.clone(), config=config, backend=backend, device="cpu")
    runtime.activation.copy_(torch.tensor([0.22, -0.31, 0.17]))
    runtime.refractory.copy_(torch.tensor([0.06, 0.11, 0.04]))
    runtime.memory_trace.copy_(torch.tensor([-0.08, 0.05, 0.12]))
    runtime.adaptation.copy_(torch.tensor([0.09, 0.13, 0.07]))
    runtime.stp_u.copy_(torch.tensor([0.41, 0.58, 0.36]))
    runtime.stp_x.copy_(torch.tensor([0.83, 0.71, 0.92]))
    runtime.bitfield.copy_(torch.tensor([0, 1, 0], dtype=torch.uint8))
    runtime.lifecycle.copy_(
        torch.tensor(
            [
                _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE],
                _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT],
                _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE],
            ],
            dtype=torch.int64,
        )
    )
    runtime.inactive_steps.zero_()
    runtime.set_goal(torch.tensor([0.08, 0.03, -0.05]))
    return runtime


def _assert_same_runtime_state(
    left: BrainRuntime,
    right: BrainRuntime,
    *,
    atol: float = 0.0,
) -> None:
    for name in (
        "activation",
        "refractory",
        "memory_trace",
        "adaptation",
        "stp_u",
        "stp_x",
        "goal",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        assert torch.allclose(left_value, right_value, atol=atol, rtol=0.0), name
    assert torch.equal(left.bitfield, right.bitfield)
    assert torch.equal(left.lifecycle, right.lifecycle)
    assert torch.equal(left.inactive_steps, right.inactive_steps)


def test_neuronwise_threshold_validation_and_canonicalization():
    config = _config(
        neuronwise_active_threshold=[0.2, 0.3, 0.4],
        neuronwise_bit_lower_threshold=[0.1, 0.12, 0.14],
        neuronwise_bit_upper_threshold=[0.25, 0.27, 0.29],
    )
    assert config.neuronwise_active_threshold == (0.2, 0.3, 0.4)
    assert config.neuronwise_bit_lower_threshold == (0.1, 0.12, 0.14)
    assert config.neuronwise_bit_upper_threshold == (0.25, 0.27, 0.29)

    invalid = (
        {"neuronwise_active_threshold": 0.2},
        {"neuronwise_active_threshold": "0.2,0.3,0.4"},
        {"neuronwise_active_threshold": (0.2, 0.3)},
        {"neuronwise_active_threshold": (0.2, float("nan"), 0.4)},
        {"neuronwise_active_threshold": (0.2, float("inf"), 0.4)},
        {
            "bit_upper_threshold": 0.2,
            "neuronwise_bit_lower_threshold": (0.1, 0.2, 0.1),
        },
        {
            "bit_upper_threshold": 0.2,
            "neuronwise_bit_lower_threshold": (0.1, 0.21, 0.1),
        },
        {
            "bit_lower_threshold": 0.1,
            "neuronwise_bit_upper_threshold": (0.2, 0.1, 0.2),
        },
        {
            "bit_lower_threshold": 0.1,
            "neuronwise_bit_upper_threshold": (0.2, 0.09, 0.2),
        },
        {
            "bit_upper_threshold": float("nan"),
            "neuronwise_bit_lower_threshold": (0.05, 0.06, 0.07),
        },
        {
            "bit_lower_threshold": float("inf"),
            "neuronwise_bit_upper_threshold": (0.2, 0.3, 0.4),
        },
    )
    for fields in invalid:
        with pytest.raises((TypeError, ValueError)):
            _config(**fields)

    legacy_overlap = _config(bit_lower_threshold=0.4, bit_upper_threshold=0.3)
    assert legacy_overlap.bit_lower_threshold == 0.4
    assert legacy_overlap.bit_upper_threshold == 0.3


def test_scalar_threshold_mutation_remains_live():
    runtime = _runtime(_config(), weight=torch.zeros((DIM, DIM)))
    salience = torch.tensor([0.20, 0.23, 0.24])
    assert torch.equal(
        runtime._select_active(salience, 2),
        torch.tensor([False, True, True]),
    )
    runtime.config.active_threshold = 0.235
    assert torch.equal(
        runtime._select_active(salience, 2),
        torch.tensor([False, False, True]),
    )


def test_repeated_vectors_are_exact_scalar_broadcast():
    scalar = _runtime(_config(), backend="torch")
    vector = _runtime(
        _config(
            neuronwise_active_threshold=(0.22, 0.22, 0.22),
            neuronwise_bit_lower_threshold=(0.10, 0.10, 0.10),
            neuronwise_bit_upper_threshold=(0.30, 0.30, 0.30),
        ),
        backend="torch",
    )
    scalar_step = scalar.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    vector_step = vector.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    _assert_same_runtime_state(scalar, vector, atol=0.0)
    assert scalar_step == vector_step
    replay = torch.zeros(DIM)
    scalar_salience = scalar._compute_salience(
        scalar.activation, EXTERNAL, replay, scalar.refractory
    )
    vector_salience = vector._compute_salience(
        vector.activation, EXTERNAL, replay, vector.refractory
    )
    assert torch.equal(scalar_salience, vector_salience)


def test_heterogeneous_bit_and_active_threshold_witness():
    runtime = BrainRuntime(
        torch.zeros((DIM, DIM)),
        config=_config(
            neuronwise_active_threshold=(0.35, 0.35, 0.55),
            neuronwise_bit_lower_threshold=(0.10, 0.17, 0.20),
            neuronwise_bit_upper_threshold=(0.15, 0.22, 0.30),
        ),
        backend="torch",
        device="cpu",
    )
    runtime.activation.fill_(0.2)
    runtime.refractory.zero_()
    runtime.memory_trace.zero_()
    runtime.adaptation.zero_()
    runtime.bitfield.copy_(torch.tensor([0, 1, 1], dtype=torch.uint8))
    runtime.goal.zero_()
    runtime._step_torch(torch.zeros(DIM), torch.zeros(DIM), RuntimeMode.WAKE)
    assert torch.allclose(runtime.activation, torch.full((DIM,), 0.164), atol=1e-7, rtol=0.0)
    assert torch.equal(runtime.bitfield, torch.tensor([1, 0, 0], dtype=torch.uint8))

    mask = runtime._select_active(torch.tensor([0.30, 0.40, 0.50]), 2)
    assert torch.equal(mask, torch.tensor([False, True, False]))


def test_snapshot_preserves_vectors_and_nontrivial_delay_continuation():
    config = _config(
        axon_delay=True,
        neuronwise_active_threshold=(0.20, 0.24, 0.28),
        neuronwise_bit_lower_threshold=(0.08, 0.10, 0.12),
        neuronwise_bit_upper_threshold=(0.26, 0.30, 0.34),
    )
    runtime = _runtime(config, backend="torch")
    assert runtime._delay_buffer is not None
    runtime._delay_buffer.copy_(
        torch.tensor([[0.55, -0.24, 0.33], [-0.12, 0.44, 0.26]])
    )
    runtime._delay_idx = 3
    snapshot = runtime.snapshot()
    restored = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")

    assert restored.config.neuronwise_active_threshold == (0.20, 0.24, 0.28)
    assert restored.config.neuronwise_bit_lower_threshold == (0.08, 0.10, 0.12)
    assert restored.config.neuronwise_bit_upper_threshold == (0.26, 0.30, 0.34)
    schema = asdict(_config())
    assert schema["neuronwise_active_threshold"] is None
    assert schema["neuronwise_bit_lower_threshold"] is None
    assert schema["neuronwise_bit_upper_threshold"] is None

    original_step = runtime.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    restored_step = restored.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    _assert_same_runtime_state(runtime, restored, atol=0.0)
    assert original_step == restored_step
    assert runtime._delay_buffer is not None and restored._delay_buffer is not None
    assert torch.equal(runtime._delay_buffer, restored._delay_buffer)
    assert runtime._delay_idx == restored._delay_idx == 4


def test_vector_bit_backend_is_fail_closed_and_auto_matches_torch():
    auto = _runtime(
        _config(
            neuronwise_bit_lower_threshold=(0.08, 0.10, 0.12),
            neuronwise_bit_upper_threshold=(0.26, 0.30, 0.34),
        ),
        backend="auto",
        weight=torch.zeros((DIM, DIM)),
    )
    forced = _runtime(
        _config(
            neuronwise_bit_lower_threshold=(0.08, 0.10, 0.12),
            neuronwise_bit_upper_threshold=(0.26, 0.30, 0.34),
        ),
        backend="torch",
        weight=torch.zeros((DIM, DIM)),
    )
    assert not auto._use_rust()
    auto_step = auto.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    forced_step = forced.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    _assert_same_runtime_state(auto, forced, atol=0.0)
    assert auto_step == forced_step

    with pytest.raises(ValueError, match="does not support neuronwise bit thresholds"):
        BrainRuntime(
            torch.zeros((DIM, DIM)),
            config=_config(
                neuronwise_bit_lower_threshold=(0.08, 0.10, 0.12),
                neuronwise_bit_upper_threshold=(0.26, 0.30, 0.34),
            ),
            backend="rust",
            device="cpu",
        )

    mutated_auto = _runtime(_config(), backend="auto", weight=torch.zeros((DIM, DIM)))
    mutated_auto.config.neuronwise_bit_upper_threshold = (0.26, 0.30, 0.34)
    assert not mutated_auto._use_rust()

    if _HAS_RUST_KERNEL:
        mutated_rust = _runtime(_config(), backend="rust", weight=torch.zeros((DIM, DIM)))
        mutated_rust.config.neuronwise_bit_upper_threshold = (0.26, 0.30, 0.34)
        with pytest.raises(ValueError, match="does not support neuronwise bit thresholds"):
            mutated_rust._use_rust()
        with pytest.raises(ValueError, match="does not support neuronwise bit thresholds"):
            mutated_rust._step_rust(torch.zeros(DIM), torch.zeros(DIM), RuntimeMode.WAKE)


@pytest.mark.skipif(not _HAS_RUST_KERNEL, reason="Rust runtime kernel unavailable")
def test_active_vector_only_preserves_no_delay_rust_final_selection():
    torch_runtime = _runtime(
        _config(neuronwise_active_threshold=(0.20, 0.25, 0.30)),
        backend="torch",
    )
    rust_runtime = _runtime(
        _config(neuronwise_active_threshold=(0.20, 0.25, 0.30)),
        backend="rust",
    )
    torch_step = torch_runtime.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    rust_step = rust_runtime.step(external_input=EXTERNAL, force_mode=RuntimeMode.WAKE)
    _assert_same_runtime_state(torch_runtime, rust_runtime, atol=1e-5)
    assert torch_step.active_modules == rust_step.active_modules
    assert torch.equal(torch_runtime.active_mask(), rust_runtime.active_mask())
    assert rust_step.energy == pytest.approx(torch_step.energy, abs=1e-5)
