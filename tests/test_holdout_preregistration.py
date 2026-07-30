from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.preregistration.validate_holdout_manifest import (
    ROOT,
    compute_manifest_sha256,
    load_manifest,
    main,
    validate_manifest,
)


MANIFEST_DIR = ROOT / "experiments" / "preregistration"
COSMOLOGY_PATH = MANIFEST_DIR / "cosmology_future_holdout_v1.json"
QUANTUM_PATH = MANIFEST_DIR / "quantum_future_holdout_v1.json"


def _load(path: Path) -> dict:
    return load_manifest(path)


def _rehash(manifest: dict) -> None:
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)


def _assign_synthetic_future_holdout(manifest: dict) -> None:
    prior_manifest_id = manifest["manifest_id"]
    manifest["manifest_id"] = f"{prior_manifest_id}-assignment-v2"
    manifest["supersedes_manifest_id"] = prior_manifest_id
    manifest["status"] = "frozen_with_assigned_holdout"
    manifest["freeze"]["protocol_revision"] = 2
    manifest["freeze"]["phase"] = "holdout_assigned_pre_access"
    holdout_path = "future/synthetic-holdout.bin"
    holdout_hash = "a" * 64
    manifest["input_artifacts"].append(
        {
            "path": holdout_path,
            "sha256": holdout_hash,
            "kind": "holdout_data",
            "role": "test-only synthetic assigned holdout",
            "provenance": "Synthetic validator fixture; artifact reads are disabled.",
        }
    )
    manifest["data_roles"]["future_holdout"].update(
        {
            "assignment_status": "assigned",
            "dataset_id": "synthetic-future-dataset",
            "release_id": "release-1",
            "release_date": "2026-07-31",
            "source_uri": "https://example.invalid/frozen-before-access",
            "local_artifact_path": holdout_path,
            "sha256": holdout_hash,
            "assigned_at": "2026-07-31",
            "assignment_manifest_id": manifest["manifest_id"],
        }
    )


@pytest.mark.parametrize("path", [COSMOLOGY_PATH, QUANTUM_PATH])
def test_unassigned_manifest_is_structurally_valid_but_not_evaluation_ready(path):
    report = validate_manifest(_load(path))

    assert report.structurally_valid, report.errors
    assert report.holdout_status == "unassigned"
    assert not report.evaluation_ready
    assert not report.errors


@pytest.mark.parametrize("path", [COSMOLOGY_PATH, QUANTUM_PATH])
def test_execution_mode_rejects_unassigned_future_holdout(path):
    report = validate_manifest(
        _load(path),
        require_assigned_holdout=True,
    )

    assert not report.structurally_valid
    assert not report.evaluation_ready
    assert "future holdout is not assigned" in report.errors


@pytest.mark.parametrize("path", [COSMOLOGY_PATH, QUANTUM_PATH])
def test_manifest_self_digest_detects_frozen_content_tampering(path):
    manifest = _load(path)
    manifest["freeze"]["protocol_revision"] += 1

    report = validate_manifest(manifest, verify_artifacts=False)

    assert not report.structurally_valid
    assert any("manifest_sha256 mismatch" in error for error in report.errors)


def test_rehashing_an_edited_v1_does_not_overwrite_the_frozen_trust_anchor():
    manifest = _load(COSMOLOGY_PATH)
    manifest["scope"]["interpretation"] += " edited in place"
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert not report.structurally_valid
    assert any("frozen v1 digest mismatch" in error for error in report.errors)


def test_artifact_digest_is_independent_of_manifest_self_digest():
    manifest = _load(COSMOLOGY_PATH)
    manifest["input_artifacts"][0]["sha256"] = "0" * 64
    _rehash(manifest)

    report = validate_manifest(manifest)

    assert not report.structurally_valid
    assert any("sha256 mismatch for" in error for error in report.errors)


def test_partial_placeholder_assignment_is_not_silently_accepted():
    manifest = _load(COSMOLOGY_PATH)
    future = manifest["data_roles"]["future_holdout"]
    future["dataset_id"] = "only-one-field-was-filled"
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert not report.structurally_valid
    assert any("dataset_id must be null while unassigned" in error for error in report.errors)


@pytest.mark.parametrize("path", [COSMOLOGY_PATH, QUANTUM_PATH])
def test_assignment_requires_a_higher_revision_pre_access_manifest(path):
    manifest = _load(path)
    future = manifest["data_roles"]["future_holdout"]
    future.update(
        {
            "assignment_status": "assigned",
            "dataset_id": "new-future-dataset",
            "release_id": "release-1",
            "release_date": "2026-07-31",
            "source_uri": "https://example.invalid/frozen-before-access",
            "local_artifact_path": "future/new-future-dataset.bin",
            "sha256": "a" * 64,
            "assigned_at": "2026-07-31",
            "assignment_manifest_id": manifest["manifest_id"],
        }
    )
    manifest["status"] = "frozen_with_assigned_holdout"
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert not report.structurally_valid
    assert "an assigned holdout requires protocol_revision >= 2" in report.errors
    assert any("holdout_assigned_pre_access" in error for error in report.errors)
    assert any("local_artifact_path/sha256" in error for error in report.errors)


def test_higher_revision_cosmology_assignment_can_be_evaluation_ready():
    manifest = _load(COSMOLOGY_PATH)
    _assign_synthetic_future_holdout(manifest)
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert report.structurally_valid, report.errors
    assert report.holdout_status == "assigned"
    assert report.evaluation_ready


def test_higher_revision_quantum_assignment_needs_resolved_physics_to_be_ready():
    manifest = _load(QUANTUM_PATH)
    _assign_synthetic_future_holdout(manifest)
    contract = manifest["model_contract"]
    contract["evaluation_readiness"] = "ready"
    contract["unresolved_physical_inputs"] = []
    contract["resolved_physical_inputs"] = {
        "scalar_action_parameters": "frozen-test-value",
        "system_coupling_g": "frozen-test-value",
        "system_operator_A": "frozen-test-value",
        "field_operator_O_phi": "frozen-test-value",
        "bath_state": "frozen-test-value",
        "spectral_density_J": "frozen-test-value",
        "rate_unit_convention_and_spectrum_units": "frozen-test-value",
        "kossakowski_matrix_or_jump_basis": "frozen-test-value",
        "jacobi_operator_and_probe": "frozen-test-value",
    }
    for model in manifest["registered_models"]:
        model["status"] = "ready"
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert report.structurally_valid, report.errors
    assert report.holdout_status == "assigned"
    assert report.evaluation_ready


def test_desi_dr2_is_exploratory_and_cannot_be_relabelled_as_future_holdout():
    manifest = _load(COSMOLOGY_PATH)
    exploratory = manifest["data_roles"]["exploratory_or_calibration"]
    desi = next(
        dataset
        for dataset in exploratory
        if dataset["dataset_id"] == "desi_dr2_all_13_point_compressed_bao"
    )
    future = manifest["data_roles"]["future_holdout"]

    assert desi["role"] == "exploratory_calibration"
    assert desi["observed_before_freeze"] is True
    assert desi["holdout_eligible"] is False
    assert "DESI DR2" in desi["why_not_holdout"]
    assert future["assignment_status"] == "unassigned"
    assert future["dataset_id"] is None
    assert desi["dataset_id"] in future["prohibited_dataset_ids"]

    tampered = copy.deepcopy(manifest)
    tampered["data_roles"]["future_holdout"]["dataset_id"] = desi["dataset_id"]
    _rehash(tampered)
    report = validate_manifest(tampered, verify_artifacts=False)
    assert any("DESI DR2 must never be relabeled" in error for error in report.errors)


def test_cosmology_freezes_zero_fit_full_covariance_gate_and_both_candidates():
    manifest = _load(COSMOLOGY_PATH)
    fit_policy = manifest["fit_policy"]
    acceptance = manifest["acceptance_criteria"]
    model_ids = {model["model_id"] for model in manifest["registered_models"]}
    rules = {rule["rule_id"]: rule for rule in manifest["hard_kill_rules"]}

    assert fit_policy["fitted_parameter_count_on_holdout"] == 0
    assert fit_policy["allow_profile_fit"] is False
    assert fit_policy["allow_post_unblinding_model_selection"] is False
    assert acceptance["covariance_required"] == "full"
    assert acceptance["thresholds"] == {
        "pass_minimum_p_value": 0.05,
        "reject_below_p_value": 0.0027,
    }
    assert model_ids == {
        "ce_density_external_rd_v1",
        "ce_density_eh_hybrid_v1",
    }
    assert rules["COS-SCI-001"]["scope"] == "scientific_candidate"
    assert "not the CE core" in rules["COS-SCI-001"]["target"]


def test_cosmology_required_kill_rule_cannot_be_removed_and_rehashed():
    manifest = _load(COSMOLOGY_PATH)
    manifest["hard_kill_rules"] = [
        rule for rule in manifest["hard_kill_rules"] if rule["rule_id"] != "COS-PROT-003"
    ]
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert not report.structurally_valid
    assert any("cosmology hard kill rules missing" in error for error in report.errors)


def test_quantum_manifest_freezes_branch_a_but_not_an_unresolved_physical_model():
    manifest = _load(QUANTUM_PATH)
    contract = manifest["model_contract"]
    mass = contract["scalar_mass"]
    future = manifest["data_roles"]["future_holdout"]
    historical = manifest["data_roles"]["exploratory_or_calibration"][0]

    assert contract["branch"] == "A_independent_scalar_field"
    assert contract["phi_equals_ricci_scalar"] is False
    assert mass == {
        "value_mev": 29.64757,
        "role": "reference_not_prediction",
        "fit_status": "forbidden_on_holdout",
    }
    assert contract["evaluation_readiness"] == "blocked_open_model"
    assert contract["interaction_template"]["g"] is None
    assert contract["open_system_rate_template"]["spectral_density_J"] is None
    assert historical["dataset_id"] == "historical_arc_94_6_percent_record"
    assert historical["holdout_eligible"] is False
    assert historical["reproducibility_artifact_available"] is False
    assert future["assignment_status"] == "unassigned"
    assert future["sha256"] is None


def test_quantum_physicality_kills_and_tolerances_are_frozen():
    manifest = _load(QUANTUM_PATH)
    tolerance_by_name = {
        tolerance["name"]: tolerance for tolerance in manifest["numeric_tolerances"]
    }
    rules = {rule["rule_id"]: rule for rule in manifest["hard_kill_rules"]}

    assert tolerance_by_name["minimum_choi_eigenvalue"]["value"] == -1e-10
    assert tolerance_by_name["born_probability_sum_absolute_error"]["value"] == 1e-10
    assert tolerance_by_name["minimum_born_probability"]["value"] == -1e-12
    assert tolerance_by_name["no_signalling_max_probability_variation"]["value"] == 1e-8
    assert tolerance_by_name["minimum_spectral_density"]["value"] == -1e-12
    assert tolerance_by_name["minimum_kossakowski_eigenvalue"]["value"] == -1e-10
    assert tolerance_by_name["kms_log_detailed_balance_absolute_error"]["value"] == 1e-8
    assert rules["QUA-SCI-002"]["scope"] == "scientific_candidate"
    assert "complete positivity" in rules["QUA-SCI-002"]["condition"]
    assert "KMS detailed-balance" in rules["QUA-SCI-004"]["condition"]
    assert rules["QUA-PROT-002"]["scope"] == "protocol_validity"
    assert "forbidden" in rules["QUA-PROT-002"]["condition"]


def test_quantum_required_kill_rule_cannot_be_removed_and_rehashed():
    manifest = _load(QUANTUM_PATH)
    manifest["hard_kill_rules"] = [
        rule for rule in manifest["hard_kill_rules"] if rule["rule_id"] != "QUA-SCI-003"
    ]
    _rehash(manifest)

    report = validate_manifest(manifest, verify_artifacts=False)

    assert not report.structurally_valid
    assert any("quantum hard kill rules missing" in error for error in report.errors)


def test_validator_rejects_artifact_paths_outside_repository():
    manifest = _load(QUANTUM_PATH)
    artifact = manifest["input_artifacts"][0]
    artifact["path"] = "../outside.py"
    _rehash(manifest)

    report = validate_manifest(manifest)

    assert not report.structurally_valid
    assert any("must stay within the repository" in error for error in report.errors)


def test_cli_distinguishes_freeze_validation_from_evaluation_readiness(capsys):
    assert main([]) == 0
    valid_output = capsys.readouterr().out
    assert valid_output.count("VALID holdout=unassigned evaluation=NOT_READY") == 2

    assert main(["--require-assigned-holdout"]) == 1
    blocked_output = capsys.readouterr().out
    assert blocked_output.count("future holdout is not assigned") == 2
