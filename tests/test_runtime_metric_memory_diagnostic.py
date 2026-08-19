import math
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

import reality_stone.clarus.runtime_alternative_memory as frozen_m1
from reality_stone.clarus.runtime_alternative_memory import AlternativeMemoryConfig
from reality_stone.clarus.runtime_metric_memory_diagnostic import (
    ADVERSE_CONDITIONS,
    CONFIRMATION_SEEDS,
    G3DiagnosticConfig,
    M1_PARITY_SMOKE_SEED,
    M1_SOURCE_SHA256,
    RETIRED_DEVELOPMENT_SEEDS,
    _airm_distance,
    _bootstrap_family,
    _continuous_recall,
    _file_sha256,
    _frozen_protocol,
    _install_candidate,
    _tensor_sha256,
    _train_m1_arm,
    _validate_stage_results,
    g3_source_hashes,
    g3_response_recall_diagnostic,
    run_g3_seed_range,
    run_g3_stage,
    verify_g3_confirmation_manifest,
)


def _small_config(seed: int = 97091) -> G3DiagnosticConfig:
    return G3DiagnosticConfig(
        dim=12,
        replay_epochs=1,
        replay_ticks=1,
        rollout_horizon=2,
        lesion_direction_count=2,
        bootstrap_samples=100,
        seed=seed,
    )


def test_g3_source_lock_and_frozen_protocol() -> None:
    config = G3DiagnosticConfig()
    assert _frozen_protocol(config)
    assert _file_sha256(Path(frozen_m1.__file__)) == M1_SOURCE_SHA256
    assert not _frozen_protocol(_small_config())


def test_g3_m1_duplicate_matches_untouched_predecessor(monkeypatch: pytest.MonkeyPatch) -> None:
    config = G3DiagnosticConfig(seed=M1_PARITY_SMOKE_SEED)
    alternative = config.alternative()
    assert _frozen_protocol(config)
    assert asdict(alternative) == asdict(AlternativeMemoryConfig(seed=M1_PARITY_SMOKE_SEED))
    captured = {}
    predecessor_evaluate = frozen_m1._evaluate_sealed

    def capture_weight(runtime, *args, **kwargs):
        result = predecessor_evaluate(runtime, *args, **kwargs)
        captured["weight"] = runtime.weight.detach().clone()
        captured["snapshot"] = runtime.snapshot()
        return result

    monkeypatch.setattr(frozen_m1, "_evaluate_sealed", capture_weight)
    predecessor = frozen_m1._m1_condition(config.seed, alternative, "fixed_clock")
    duplicate = _train_m1_arm(config.seed, config, "fixed_clock")

    torch.testing.assert_close(duplicate.sealed_snapshot.weight, captured["weight"], atol=0.0, rtol=0.0)
    assert duplicate.report["post_weight_sha256"] == _tensor_sha256(captured["weight"])
    for field, value in predecessor.items():
        assert duplicate.report[field] == value

    predecessor_continuous = _continuous_recall(
        captured["snapshot"],
        duplicate.cues,
        duplicate.targets,
        duplicate.indices,
        config,
    )
    duplicate_continuous = _continuous_recall(
        duplicate.sealed_snapshot,
        duplicate.cues,
        duplicate.targets,
        duplicate.indices,
        config,
    )
    assert predecessor_continuous == duplicate_continuous


def test_g3_airm_uses_strict_spd_symmetric_spectrum() -> None:
    first = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))
    second = torch.diag(torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64))
    expected = math.sqrt(math.log(2.0) ** 2 + math.log(0.25) ** 2)
    assert math.isclose(_airm_distance(first, second), expected, rel_tol=0.0, abs_tol=1e-12)
    assert _airm_distance(first, first) <= 1e-12
    with pytest.raises(ValueError, match="strictly SPD"):
        _airm_distance(torch.diag(torch.tensor([1.0, 0.0, 1.0])), second)
    with pytest.raises(ValueError, match="second input must be strictly SPD"):
        _airm_distance(first, torch.diag(torch.tensor([1.0, -1.0, 1.0])))


def test_g3_reduced_route_has_fresh_structural_and_null_lesion_branches() -> None:
    config = _small_config()
    result = g3_response_recall_diagnostic(config.seed, config)

    assert result["m1_source_lock"]
    assert result["common_initial_snapshot"]
    assert result["probe_matrix_orthonormal"]
    assert result["training_integrity"]
    assert result["structural_integrity"]
    assert result["condition_integrity"]
    assert result["pre"]["cutoff"]["temporal_rows_after"] == 0
    assert result["separation_audit"]["fresh_calibration_probe_restores_observed"]
    assert result["separation_audit"]["fresh_recall_probe_restores_observed"]
    assert not result["separation_audit"]["calibration_reads_task_codebook"]
    assert not result["separation_audit"]["recall_reads_calibration_state"]
    structural = result["arms"]["weight_permuted"]["training"]["structural_control"]
    assert structural["provenance"] == "matched_post_learning_coordinate_permutation"
    assert not structural["randomized_learning_contingency"]
    assert structural["no_tensor_storage_alias"]
    assert structural["no_clipping"]
    assert structural["reconstruction_residual"] <= 1e-7
    assert len(structural["permutation_matrix"]) == config.dim
    assert all(len(row) == config.dim for row in structural["permutation_matrix"])

    lesion = result["calibration_null_lesion"]
    assert lesion["integrity"]
    assert not lesion["selection_uses_recall"]
    assert lesion["candidate_count"] == 2 * config.lesion_direction_count
    assert lesion["repeat_hash_matches"]
    matched_hashes = {
        row["install_audit"]["matched_weight_sha256"] for row in lesion["candidates"]
    }
    assert len(matched_hashes) == 1
    assert all(row["install_audit"]["no_clipping"] for row in lesion["candidates"])
    assert all(row["install_audit"]["preinstall_representable"] for row in lesion["candidates"])
    assert all(
        row["install_audit"]["target_reconstruction_residual_float64"]
        <= config.lesion_target_tolerance
        for row in lesion["candidates"]
    )
    assert all(row["calibration_integrity"] for row in lesion["candidates"])

    # Reduced mechanics are intentionally not allowed to masquerade as the frozen route.
    assert not result["frozen_protocol"]
    assert not result["integrity"]
    assert result["status"] == "STOP"
    assert result["mediation_status"] == "BLOCKED_NOT_IDENTIFIED"


def test_g3_family_bootstrap_keeps_same_arm_pairs_and_is_deterministic() -> None:
    results = []
    for index in range(16):
        contrasts = {
            name: {
                "delta_S": 0.10 + 0.01 * index + 0.001 * arm_index,
                "delta_R": 0.20 + 0.02 * index + 0.001 * arm_index,
            }
            for arm_index, name in enumerate(ADVERSE_CONDITIONS)
        }
        results.append({"contrasts": contrasts})
    first = _bootstrap_family(results, samples=200, seed=97898)
    second = _bootstrap_family(results, samples=200, seed=97898)
    assert first == second
    assert first["mean_delta_S_simultaneous_lcb_95"] > 0.0
    assert first["mean_delta_R_simultaneous_lcb_95"] > 0.0
    assert first["same_arm_rho_simultaneous_lcb_95"] > 0.99


def test_g3_stage_validator_rejects_duplicate_or_foreign_seed_units() -> None:
    duplicate = [{"seed": seed} for seed in range(97801, 97817)]
    duplicate[-1]["seed"] = duplicate[-2]["seed"]
    with pytest.raises(ValueError, match="exact ordered unique seed block"):
        _validate_stage_results(duplicate, "development")

    foreign = [{"seed": seed} for seed in range(97801, 97817)]
    foreign[0]["seed"] = 12345
    with pytest.raises(ValueError, match="exact ordered unique seed block"):
        _validate_stage_results(foreign, "development")

    retired = [{"seed": seed} for seed in range(97701, 97717)]
    with pytest.raises(ValueError, match="exact ordered unique seed block"):
        _validate_stage_results(retired, "development")


def test_g3_confirmation_api_rejects_missing_or_forged_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires a verified development manifest"):
        run_g3_stage("confirmation")
    with pytest.raises(RuntimeError, match="manifest-verified run_g3_stage"):
        g3_response_recall_diagnostic(min(CONFIRMATION_SEEDS))
    with pytest.raises(RuntimeError, match="manifest-verified run_g3_stage"):
        run_g3_seed_range(CONFIRMATION_SEEDS)
    with pytest.raises(RuntimeError, match="retired apparatus-invalid"):
        g3_response_recall_diagnostic(min(RETIRED_DEVELOPMENT_SEEDS))
    with pytest.raises(RuntimeError, match="retired apparatus-invalid"):
        run_g3_seed_range(RETIRED_DEVELOPMENT_SEEDS)

    artifact = tmp_path / "not-g3-development.json"
    artifact.write_text(json.dumps({"schema": "not-a-g3-artifact"}), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "status": "FROZEN",
        "development_route_verdict": "DIAGNOSTIC_PASS",
        "files": g3_source_hashes(),
        "development_artifact": str(artifact),
        "development_artifact_sha256": artifact_hash,
        "development_results_sha256": "forged",
    }
    manifest_path = tmp_path / "forged-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a G3-D artifact"):
        verify_g3_confirmation_manifest(manifest_path)


def test_g3_representable_target_headroom_and_preinstall_rejection() -> None:
    config = _small_config(seed=97092)
    matched = _train_m1_arm(config.seed, config, "fixed_clock")
    generator = torch.Generator(device="cpu").manual_seed(1234)
    direction = torch.randn(config.dim, config.dim, generator=generator)
    direction.fill_diagonal_(0.0)

    admitted = direction * (0.2500009 / direction.double().norm())
    snapshot, audit = _install_candidate(matched.sealed_snapshot, admitted, config)
    assert snapshot is not None
    assert audit["preinstall_representable"]
    assert audit["install_performed"]
    assert audit["actual_delta_norm_float64"] <= config.lesion_install_bound
    assert audit["actual_norm_error_from_declared"] <= config.lesion_norm_tolerance
    assert audit["intended_to_actual_residual_float64"] <= config.lesion_quantization_tolerance
    assert audit["no_clipping"]
    assert audit["target_reconstruction_residual_float64"] <= config.lesion_target_tolerance

    rejected = direction * (0.250002 / direction.double().norm())
    rejected_snapshot, rejected_audit = _install_candidate(
        matched.sealed_snapshot, rejected, config,
    )
    assert rejected_snapshot is None
    assert not rejected_audit["preinstall_representable"]
    assert not rejected_audit["install_performed"]
