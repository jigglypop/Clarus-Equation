from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from reality_stone.clarus.experiments.runtime_prediction_guided_metacontrol import (
    ACTION_VALUES,
    CONFIRMATION_SEEDS,
    C1MetacontrolConfig,
    DEVELOPMENT_SEEDS,
    EDGE_DERANGEMENT,
    PREDECESSOR_ARTIFACT_SHA256,
    _PREDECESSOR_ARTIFACT,
    _bootstrap_c1,
    _build_fixture,
    _c1_prediction_guided_metacontrol_unchecked,
    _c1_runtime,
    _feature,
    _file_sha256,
    _global_apparatus,
    _snapshot_sha256,
    _validate_stage_results,
    c1_prediction_guided_metacontrol,
    c1_source_hashes,
    run_c1_seed_range,
    run_c1_stage,
    verify_c1_confirmation_manifest,
)


def _small_config(seed: int = 97071) -> C1MetacontrolConfig:
    return C1MetacontrolConfig(
        dim=12,
        fit_states=8,
        audit_states=4,
        policy_states=6,
        warmup_ticks=2,
        bootstrap_samples=100,
        seed=seed,
    )


def test_c1_frozen_banks_schema_and_predecessor_lock() -> None:
    config = C1MetacontrolConfig()
    apparatus = _global_apparatus(config)
    assert tuple(apparatus["goals"].shape) == (16, 4)
    torch.testing.assert_close(
        apparatus["goals"].norm(dim=1),
        torch.ones(16, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert abs(float(apparatus["action_vector"].norm()) - 1.0) <= 1e-12
    assert tuple(apparatus["random_schedule"][:6]) == (-1, 0, 1, -1, 0, 1)
    assert apparatus["random_schedule"].count(-1) == 22
    assert apparatus["random_schedule"].count(0) == 21
    assert apparatus["random_schedule"].count(1) == 21
    assert tuple(apparatus["sign_schedule"][:4]) == (-1, 1, -1, 1)
    assert EDGE_DERANGEMENT == (1, 2, 0)

    runtime = _c1_runtime(config.seed, config)
    feature = _feature(runtime.snapshot(), torch.zeros(config.dim), 1)
    assert feature.shape == (388,)
    assert _file_sha256(_PREDECESSOR_ARTIFACT) == PREDECESSOR_ARTIFACT_SHA256
    assert c1_source_hashes()


def test_c1_snapshot_hash_covers_scalar_and_store_state() -> None:
    config = _small_config()
    snapshot = _c1_runtime(config.seed, config).snapshot()
    original = _snapshot_sha256(snapshot)
    changed_step = replace(snapshot, step=snapshot.step + 1)
    assert _snapshot_sha256(changed_step) != original
    changed_store = replace(
        snapshot,
        hippocampus={**snapshot.hippocampus, "priority": [1.0]},
    )
    assert _snapshot_sha256(changed_store) != original


def test_c1_reduced_fixture_and_route_enforce_one_step_policy() -> None:
    config = _small_config()
    fixture = _build_fixture(config.seed, config)
    assert fixture.audit["total_states"] == 18
    assert fixture.audit["warmup_transition_count"] == 36
    assert fixture.audit["split_ids_disjoint"]
    assert fixture.audit["base_dense_sparse_parity"]
    assert fixture.audit["base_hippocampal_rows"] == 0
    assert fixture.audit["base_temporal_rows"] == 0
    assert fixture.audit["automatic_stdp_updates"] == 0

    result = c1_prediction_guided_metacontrol(config.seed, config)
    assert not result["frozen_protocol"]
    assert not result["integrity"]
    assert result["status"] == "STOP"
    assert result["predictor"]["feature_dim"] == 8 * config.dim + 4
    assert result["predictor"]["fit_rows"] == config.fit_states * 3
    assert result["prediction_audit"]["rows"] == config.audit_states * 3
    policy = result["policy"]
    assert policy["candidate_runtime_steps"] == 0
    assert policy["actual_runtime_steps"] == config.policy_states * 7
    assert policy["all_transition_integrity"]
    assert policy["readout_equivalence"]
    assert policy["readout_action_trace_hash_equal"]
    assert policy["edge_port_identity"]
    assert policy["starting_snapshot_identity"]
    for episode in policy["episodes"]:
        intact = episode["arms"]["intact"]
        readout = episode["arms"]["readout_shuffle"]
        edge = episode["arms"]["edge_shuffle"]
        assert intact["selected_action"] in ACTION_VALUES
        assert readout["selected_action"] == intact["selected_action"]
        assert readout["actual_drive_sha256"] == intact["actual_drive_sha256"]
        assert abs(readout["loss"] - intact["loss"]) <= 1e-12
        assert edge["pre_map_forecasts_sha256"] == intact["pre_map_forecasts_sha256"]
        assert tuple(edge["planner_port_permutation"]) == EDGE_DERANGEMENT
        assert all(arm["candidate_runtime_steps"] == 0 for arm in episode["arms"].values())
        assert all(arm["actual_runtime_steps"] == 1 for arm in episode["arms"].values())
        assert all(
            arm["starting_snapshot_matches_fixture"]
            for arm in episode["arms"].values()
        )


def test_c1_bootstrap_is_paired_deterministic_and_uses_declared_indices() -> None:
    results = [
        {
            "prediction_audit": {"mse_ratio": 0.40 + index * 0.001},
            "policy": {
                "minimum_advantage": 0.20 + index * 0.001,
                "edge_action_change_rate": 0.50 + index * 0.001,
            },
        }
        for index in range(16)
    ]
    first = _bootstrap_c1(results, samples=10_000, seed=97_998)
    second = _bootstrap_c1(results, samples=10_000, seed=97_998)
    assert first == second
    assert first["lower_order_index_zero_based"] == 499
    assert first["upper_order_index_zero_based"] == 9500
    assert first["mean_prediction_ratio_ucb_95"] < 0.90
    assert first["mean_minimum_advantage_lcb_95"] > 0.05
    assert first["mean_edge_change_rate_lcb_95"] > 0.20


def test_c1_stage_validator_rejects_duplicate_or_foreign_seed_units() -> None:
    duplicate = [{"seed": seed} for seed in DEVELOPMENT_SEEDS]
    duplicate[-1]["seed"] = duplicate[-2]["seed"]
    with pytest.raises(ValueError, match="exact ordered unique seed block"):
        _validate_stage_results(duplicate, "development")
    foreign = [{"seed": seed} for seed in DEVELOPMENT_SEEDS]
    foreign[0]["seed"] = 12345
    with pytest.raises(ValueError, match="exact ordered unique seed block"):
        _validate_stage_results(foreign, "development")


def test_c1_confirmation_public_apis_and_forged_manifest_are_sealed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="manifest-verified run_c1_stage"):
        c1_prediction_guided_metacontrol(min(CONFIRMATION_SEEDS))
    with pytest.raises(RuntimeError, match="manifest-verified run_c1_stage"):
        run_c1_seed_range(CONFIRMATION_SEEDS)
    with pytest.raises(RuntimeError, match="verified development manifest"):
        _c1_prediction_guided_metacontrol_unchecked(min(CONFIRMATION_SEEDS))
    with pytest.raises(RuntimeError, match="requires a verified development manifest"):
        run_c1_stage("confirmation")

    artifact = tmp_path / "not-c1.json"
    artifact.write_text(json.dumps({"schema": "not-c1"}), encoding="utf-8")
    manifest = {
        "status": "FROZEN",
        "development_route_verdict": "GO",
        "files": c1_source_hashes(),
        "environment": {
            "python_executable": "forged",
            "python_version": "0",
            "torch_version": "0",
            "torch_device": "cpu",
        },
        "development_artifact": str(artifact),
        "development_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "development_results_sha256": "forged",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="environment mismatch"):
        verify_c1_confirmation_manifest(manifest_path)
