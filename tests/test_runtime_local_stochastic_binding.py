import inspect

import pytest
import torch

from reality_stone.clarus.runtime import (
    BrainRuntime,
    BrainRuntimeConfig,
    ModuleLifecycle,
    RuntimeMode,
    _LIFECYCLE_TO_CODE,
)
from reality_stone.clarus.runtime_local_stochastic_binding import (
    CALIBRATION_SEED,
    LocalStochasticBindingConfig,
    _oja_update,
    run_local_stochastic_binding_seed,
)


def _uniform_packet_runtime(*, sigma: float) -> BrainRuntime:
    weight = torch.zeros(4, 4)
    weight[2:, 0] = 1.0
    return BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=4,
            active_ratio=1.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            hippocampal_encoding_enabled=False,
            competition_indices=(2, 3),
            competition_lateral_gain=1.0,
            competition_homeostasis_gain=0.0,
            competition_homeostasis_rate=0.0,
            competition_jitter_sigma=sigma,
            competition_jitter_seed=719,
        ),
        backend="torch",
        device="cpu",
    )


def test_packet_local_jitter_breaks_only_a_delivered_tie() -> None:
    deterministic = _uniform_packet_runtime(sigma=0.0)
    stochastic = _uniform_packet_runtime(sigma=0.35)
    for runtime in (deterministic, stochastic):
        runtime.activation[0] = 0.8
        runtime.lifecycle[0] = _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]
    deterministic._step_torch(torch.zeros(4), torch.zeros(4), RuntimeMode.WAKE)
    stochastic._step_torch(torch.zeros(4), torch.zeros(4), RuntimeMode.WAKE)
    assert torch.count_nonzero(deterministic.activation[2:].clamp_min(0.0)) == 0
    assert torch.count_nonzero(stochastic.activation[2:].clamp_min(0.0)) == 1

    empty = _uniform_packet_runtime(sigma=0.35)
    empty._step_torch(torch.zeros(4), torch.zeros(4), RuntimeMode.WAKE)
    assert torch.count_nonzero(empty.activation[2:]) == 0


def test_jitter_snapshot_continuation_and_mutation_fail_closed() -> None:
    runtime = _uniform_packet_runtime(sigma=0.35)
    runtime.activation[0] = 0.8
    runtime.lifecycle[0] = _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]
    snapshot = runtime.snapshot()
    left = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    right = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    left._step_torch(torch.zeros(4), torch.zeros(4), RuntimeMode.WAKE)
    right._step_torch(torch.zeros(4), torch.zeros(4), RuntimeMode.WAKE)
    torch.testing.assert_close(left.activation, right.activation, rtol=0.0, atol=0.0)
    left.config.competition_jitter_sigma = 0.0
    with pytest.raises(ValueError, match="jitter are structural"):
        left._step_torch(torch.zeros(4), torch.zeros(4), RuntimeMode.WAKE)


def test_oja_update_is_support_local_and_answer_blind() -> None:
    source = (0, 1, 2, 3)
    hidden = (8, 9, 10, 11)
    config = LocalStochasticBindingConfig()
    weight = torch.zeros(config.dim, config.dim)
    weight[torch.tensor(hidden)[:, None], torch.tensor(source)] = 1.0
    runtime = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=config.dim,
            active_ratio=1.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            hippocampal_encoding_enabled=False,
        ),
        backend="torch",
        device="cpu",
    )
    delivered = torch.zeros(config.dim)
    delivered[1] = 0.4
    runtime.activation[8] = 0.2
    before = runtime.weight.clone()
    receipt = _oja_update(
        runtime,
        delivered,
        source,
        hidden,
        config,
        learning_rate=config.oja_lr,
    )
    changed = runtime.weight - before
    candidate = torch.zeros_like(changed, dtype=torch.bool)
    candidate[torch.tensor(hidden)[:, None], torch.tensor(source)] = True
    assert receipt["outside_candidate_delta_norm"] == 0.0
    assert torch.count_nonzero(changed[~candidate]) == 0
    assert changed[8, 1] > 0.0
    assert changed[8, 0] < 0.0
    forbidden = ("target", "decoder", "reward", "endpoint", "answer")
    source_text = inspect.getsource(_oja_update).lower()
    assert not any(token in source_text for token in forbidden)


def test_calibration_learns_a_durable_noise_off_code() -> None:
    row = run_local_stochastic_binding_seed(CALIBRATION_SEED)
    assert row["status"] == "LOCAL_STOCHASTIC_WEIGHT_CODE_PASS"
    assert not row["endpoint_opened"]
    assert row["learned_evaluation"]["jitter_sigma"] == 0.0
    assert row["learned_evaluation"]["allocation"]["is_bijection"]
    assert row["deterministic_no_jitter"]["evaluation"]["allocation"]["abstention_count"] == 4
    assert row["no_learning"]["evaluation"]["allocation"]["abstention_count"] == 4
    assert row["source_independent_bias"]["status"] == "SOURCE_UNIDENTIFIED"
    assert all(row["gates"].values())

