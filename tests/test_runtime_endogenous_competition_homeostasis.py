import inspect

import pytest
import torch

from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from reality_stone.clarus.experiments.runtime_endogenous_competition_homeostasis import (
    CALIBRATION_SEED,
    EndogenousCompetitionConfig,
    run_endogenous_competition_seed,
)


def _small_runtime(*, backend: str = "torch") -> BrainRuntime:
    weight = torch.zeros(4, 4)
    weight[2, 0] = 1.1
    weight[3, 0] = 0.9
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
            competition_homeostasis_gain=1.0,
            competition_homeostasis_rate=1.0,
            competition_homeostasis_decay=0.0,
            competition_novelty_decay=0.8,
            competition_delay_ticks=1,
        ),
        backend=backend,
        device="cpu",
    )


def test_competition_config_and_rust_fail_closed() -> None:
    with pytest.raises(ValueError, match="distinct"):
        BrainRuntimeConfig(dim=4, competition_indices=(1, 1))
    with pytest.raises(ValueError, match="indices are required"):
        BrainRuntimeConfig(dim=4, competition_homeostasis_gain=1.0)
    with pytest.raises(ValueError, match="does not support local competition"):
        _small_runtime(backend="rust")


def test_runtime_competition_state_is_delayed_snapshotted_and_reset() -> None:
    runtime = _small_runtime()
    runtime.lifecycle.zero_()
    runtime.activation[0] = 0.8
    runtime.step(external_input=torch.zeros(4), force_mode=RuntimeMode.WAKE)
    assert runtime.competition_homeostasis is not None
    assert torch.count_nonzero(runtime.competition_homeostasis) == 0
    snapshot = runtime.snapshot()
    restored_a = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    restored_b = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    restored_a.step(external_input=torch.zeros(4), force_mode=RuntimeMode.WAKE)
    restored_b.step(external_input=torch.zeros(4), force_mode=RuntimeMode.WAKE)
    torch.testing.assert_close(restored_a.activation, restored_b.activation, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored_a.competition_homeostasis,
        restored_b.competition_homeostasis,
        rtol=0.0,
        atol=0.0,
    )
    restored_a.reset_evaluation_state()
    assert torch.count_nonzero(restored_a.competition_homeostasis) == 0
    assert torch.count_nonzero(restored_a._competition_usage_buffer) == 0
    assert float(restored_a.competition_packet_envelope.item()) == 0.0


def test_runtime_competition_transition_has_no_hard_allocator_input() -> None:
    methods = (
        BrainRuntime._advance_local_competition,
        BrainRuntime._apply_local_competition,
        BrainRuntime._commit_local_competition_usage,
    )
    forbidden = ("argmax", "topk", "winner", "binding", "occupied", "decoder", "target")
    for method in methods:
        source = inspect.getsource(method).lower()
        parameters = tuple(inspect.signature(method).parameters)
        assert not any(token in source for token in forbidden)
        assert not any(token in parameter.lower() for token in forbidden for parameter in parameters)
    runtime = _small_runtime()
    assert runtime.competition_homeostasis is not None
    assert runtime.competition_homeostasis.dtype.is_floating_point
    assert runtime._competition_usage_buffer.dtype.is_floating_point


def test_calibration_seed_uses_endogenous_state_and_keeps_endpoint_closed() -> None:
    row = run_endogenous_competition_seed(CALIBRATION_SEED)
    assert row["status"] == "ENDOGENOUS_SOURCE_ALLOCATION_PASS"
    assert not row["endpoint_opened"]
    assert row["persistent"]["allocation"]["is_bijection"]
    assert row["uniform"]["status"] == "ABSTAIN_BOUNDARY_TIE"
    assert row["source_independent_bias"]["status"] == "SOURCE_UNIDENTIFIED"
    assert row["gates"]["hidden_row_permutation_covariant"]
    assert row["gates"]["snapshot_continuation_exact"]
    assert all(episode["hidden_pulse_count"] == 0 for episode in row["persistent"]["episodes"])
