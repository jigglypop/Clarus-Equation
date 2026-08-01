"""Validate frozen cosmology and quantum future-holdout manifests.

The manifest digest is SHA-256 over canonical UTF-8 JSON after removing the
top-level ``manifest_sha256`` field.  This avoids a circular digest while
making any other manifest edit detectable.

An unassigned future-data placeholder can be structurally valid without being
ready for evaluation.  ``--require-assigned-holdout`` turns that intentionally
open state into a validation error for runners that are about to evaluate data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATHS = (
    MANIFEST_DIR / "cosmology_future_holdout_v2.json",
    MANIFEST_DIR / "quantum_future_holdout_v2.json",
)

SCHEMA_VERSION = 1
MANIFEST_HASH_POLICY = "sha256-canonical-json-excluding-manifest_sha256"
FROZEN_V1_MANIFEST_SHA256 = {
    "ce-cosmology-future-holdout-v1": (
        "0f79d9fb27abc7326e3bd136768f0a2b560f720b2db42c1079d9d82c3efe7692"
    ),
    "ce-quantum-future-holdout-v1": (
        "4bd3d9777c47465dd419012bbf2622fb0d5c91a312003010dede50cb1c4e853a"
    ),
}
FROZEN_V2_MANIFEST_SHA256 = {
    "ce-cosmology-future-holdout-v2": (
        "787541ccf52c4290c0c809d3f984b9552cdd149b0f9b9533e64d8160327bcf7b"
    ),
    "ce-quantum-future-holdout-v2": (
        "340ceb208d769e3fd2a85cef59b00164812918153abc5db4b2d4f69ca88b0994"
    ),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_DOMAINS = {"cosmology", "quantum"}
ALLOWED_PROVENANCE_ROLES = {
    "ce_prediction_bridge",
    "derived_selection",
    "external_input",
    "model_assumption",
    "protocol_constant",
    "reference_not_prediction",
}
ALLOWED_KILL_SCOPES = {"scientific_candidate", "protocol_validity"}
ALLOWED_COMPARISONS = {"<=", ">=", "absolute<=", "relative<="}


@dataclass(frozen=True)
class ManifestValidationReport:
    """Result of validating one preregistration manifest."""

    manifest_id: str
    domain: str
    holdout_status: str
    structurally_valid: bool
    evaluation_ready: bool
    errors: tuple[str, ...]

    def assert_valid(self) -> None:
        """Raise ``ManifestValidationError`` if any invariant failed."""
        if self.errors:
            raise ManifestValidationError("; ".join(self.errors))


class ManifestValidationError(ValueError):
    """Raised when a preregistration manifest violates its frozen schema."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the canonical byte representation used by manifest hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest after excluding its top-level self-digest field."""
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load a JSON manifest and require an object at the root."""
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManifestValidationError("manifest root must be a JSON object")
    return value


def _mapping(
    value: Any,
    name: str,
    errors: list[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _sequence(
    value: Any,
    name: str,
    errors: list[str],
) -> Sequence[Any]:
    if not isinstance(value, list):
        errors.append(f"{name} must be an array")
        return ()
    return value


def _nonempty_text(value: Any, name: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string")
        return ""
    return value


def _exact_bool(value: Any, expected: bool, name: str, errors: list[str]) -> None:
    if value is not expected:
        errors.append(f"{name} must be {str(expected).lower()}")


def _finite_number(value: Any, name: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name} must be a finite number")
        return None
    result = float(value)
    if not math.isfinite(result):
        errors.append(f"{name} must be a finite number")
        return None
    return result


def _unique_nonempty_strings(value: Any, name: str, errors: list[str]) -> set[str]:
    items = _sequence(value, name, errors)
    result: list[str] = []
    for index, item in enumerate(items):
        text = _nonempty_text(item, f"{name}[{index}]", errors)
        if text:
            result.append(text)
    if len(result) != len(set(result)):
        errors.append(f"{name} must not contain duplicates")
    return set(result)


def _safe_repo_path(repo_root: Path, raw_path: str) -> Path | None:
    relative = Path(raw_path)
    if relative.is_absolute():
        return None
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _validate_freeze(manifest: Mapping[str, Any], errors: list[str]) -> None:
    freeze = _mapping(manifest.get("freeze"), "freeze", errors)
    freeze_date = _nonempty_text(freeze.get("freeze_date"), "freeze.freeze_date", errors)
    if freeze_date:
        try:
            date.fromisoformat(freeze_date)
        except ValueError:
            errors.append("freeze.freeze_date must be an ISO calendar date")
    _nonempty_text(freeze.get("timezone"), "freeze.timezone", errors)
    _nonempty_text(freeze.get("phase"), "freeze.phase", errors)
    _nonempty_text(freeze.get("amendment_policy"), "freeze.amendment_policy", errors)
    _exact_bool(
        freeze.get("gate_scores_are_not_truth_probabilities"),
        True,
        "freeze.gate_scores_are_not_truth_probabilities",
        errors,
    )
    revision = freeze.get("protocol_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append("freeze.protocol_revision must be a positive integer")


def _validate_scope(manifest: Mapping[str, Any], errors: list[str]) -> None:
    scope = _mapping(manifest.get("scope"), "scope", errors)
    _nonempty_text(scope.get("claim_target"), "scope.claim_target", errors)
    excluded = _unique_nonempty_strings(
        scope.get("excluded_claims"),
        "scope.excluded_claims",
        errors,
    )
    if not excluded:
        errors.append("scope.excluded_claims must not be empty")
    _nonempty_text(scope.get("interpretation"), "scope.interpretation", errors)


def _validate_artifacts(
    manifest: Mapping[str, Any],
    repo_root: Path,
    verify_artifacts: bool,
    errors: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    artifacts = _sequence(manifest.get("input_artifacts"), "input_artifacts", errors)
    seen_paths: set[str] = set()
    registered_hashes: dict[str, str] = {}
    registered_kinds: dict[str, str] = {}
    for index, raw_artifact in enumerate(artifacts):
        prefix = f"input_artifacts[{index}]"
        artifact = _mapping(raw_artifact, prefix, errors)
        raw_path = _nonempty_text(artifact.get("path"), f"{prefix}.path", errors)
        expected_hash = _nonempty_text(
            artifact.get("sha256"),
            f"{prefix}.sha256",
            errors,
        )
        kind = _nonempty_text(artifact.get("kind"), f"{prefix}.kind", errors)
        _nonempty_text(artifact.get("role"), f"{prefix}.role", errors)
        _nonempty_text(artifact.get("provenance"), f"{prefix}.provenance", errors)
        if raw_path in seen_paths:
            errors.append(f"{prefix}.path duplicates another artifact")
        seen_paths.add(raw_path)
        if expected_hash and not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"{prefix}.sha256 must be lowercase SHA-256 hex")
        if raw_path and expected_hash:
            registered_hashes[raw_path] = expected_hash
        if raw_path and kind:
            registered_kinds[raw_path] = kind
        if not raw_path or not verify_artifacts:
            continue
        artifact_path = _safe_repo_path(repo_root, raw_path)
        if artifact_path is None:
            errors.append(f"{prefix}.path must stay within the repository")
            continue
        if not artifact_path.is_file():
            errors.append(f"{prefix}.path does not exist: {raw_path}")
            continue
        if expected_hash and SHA256_RE.fullmatch(expected_hash):
            actual_hash = compute_file_sha256(artifact_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"{prefix}.sha256 mismatch for {raw_path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    if not artifacts:
        errors.append("input_artifacts must not be empty")
    return registered_hashes, registered_kinds


def _validate_frozen_inputs(manifest: Mapping[str, Any], errors: list[str]) -> set[str]:
    frozen_inputs = _sequence(manifest.get("frozen_inputs"), "frozen_inputs", errors)
    names: list[str] = []
    for index, raw_input in enumerate(frozen_inputs):
        prefix = f"frozen_inputs[{index}]"
        frozen_input = _mapping(raw_input, prefix, errors)
        name = _nonempty_text(frozen_input.get("name"), f"{prefix}.name", errors)
        role = _nonempty_text(frozen_input.get("role"), f"{prefix}.role", errors)
        _nonempty_text(frozen_input.get("unit"), f"{prefix}.unit", errors)
        _nonempty_text(frozen_input.get("source"), f"{prefix}.source", errors)
        if name:
            names.append(name)
        if role and role not in ALLOWED_PROVENANCE_ROLES:
            errors.append(f"{prefix}.role is not an allowed provenance role")
        value = frozen_input.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            errors.append(f"{prefix}.value must be a number or non-empty string")
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            errors.append(f"{prefix}.value must be finite")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"{prefix}.value must be a number or non-empty string")
    if len(names) != len(set(names)):
        errors.append("frozen_inputs names must be unique")
    if not frozen_inputs:
        errors.append("frozen_inputs must not be empty")
    return set(names)


def _validate_data_roles(
    manifest: Mapping[str, Any],
    registered_hashes: Mapping[str, str],
    registered_kinds: Mapping[str, str],
    errors: list[str],
) -> tuple[Mapping[str, Any], set[str]]:
    data_roles = _mapping(manifest.get("data_roles"), "data_roles", errors)
    exploratory = _sequence(
        data_roles.get("exploratory_or_calibration"),
        "data_roles.exploratory_or_calibration",
        errors,
    )
    exploratory_ids: list[str] = []
    for index, raw_dataset in enumerate(exploratory):
        prefix = f"data_roles.exploratory_or_calibration[{index}]"
        dataset = _mapping(raw_dataset, prefix, errors)
        dataset_id = _nonempty_text(dataset.get("dataset_id"), f"{prefix}.dataset_id", errors)
        if dataset_id:
            exploratory_ids.append(dataset_id)
        _nonempty_text(dataset.get("role"), f"{prefix}.role", errors)
        _nonempty_text(dataset.get("provenance"), f"{prefix}.provenance", errors)
        _nonempty_text(
            dataset.get("why_not_holdout"),
            f"{prefix}.why_not_holdout",
            errors,
        )
        _exact_bool(dataset.get("holdout_eligible"), False, f"{prefix}.holdout_eligible", errors)
        _exact_bool(
            dataset.get("observed_before_freeze"),
            True,
            f"{prefix}.observed_before_freeze",
            errors,
        )
        snapshot_path = dataset.get("local_snapshot_artifact_path")
        snapshot_hash = dataset.get("local_snapshot_sha256")
        if snapshot_path is not None or snapshot_hash is not None:
            path_text = _nonempty_text(
                snapshot_path,
                f"{prefix}.local_snapshot_artifact_path",
                errors,
            )
            hash_text = _nonempty_text(
                snapshot_hash,
                f"{prefix}.local_snapshot_sha256",
                errors,
            )
            if path_text and registered_hashes.get(path_text) != hash_text:
                errors.append(f"{prefix} local snapshot path/hash must match input_artifacts")
    if len(exploratory_ids) != len(set(exploratory_ids)):
        errors.append("exploratory/calibration dataset IDs must be unique")

    future = _mapping(data_roles.get("future_holdout"), "data_roles.future_holdout", errors)
    placeholder_id = _nonempty_text(
        future.get("placeholder_id"),
        "data_roles.future_holdout.placeholder_id",
        errors,
    )
    assignment_status = future.get("assignment_status")
    if assignment_status not in {"unassigned", "assigned"}:
        errors.append(
            "data_roles.future_holdout.assignment_status must be 'unassigned' or 'assigned'"
        )
    _exact_bool(
        future.get("holdout_eligible"),
        True,
        "data_roles.future_holdout.holdout_eligible",
        errors,
    )
    _exact_bool(
        future.get("assignment_requires_new_manifest"),
        True,
        "data_roles.future_holdout.assignment_requires_new_manifest",
        errors,
    )
    _exact_bool(
        future.get("manifest_version_must_increase"),
        True,
        "data_roles.future_holdout.manifest_version_must_increase",
        errors,
    )
    _exact_bool(
        future.get("access_before_assignment_forbidden"),
        True,
        "data_roles.future_holdout.access_before_assignment_forbidden",
        errors,
    )
    _exact_bool(
        future.get("access_log_required"),
        True,
        "data_roles.future_holdout.access_log_required",
        errors,
    )
    selection = _mapping(
        future.get("selection_policy"),
        "data_roles.future_holdout.selection_policy",
        errors,
    )
    _exact_bool(
        selection.get("release_after_freeze_required"),
        True,
        "data_roles.future_holdout.selection_policy.release_after_freeze_required",
        errors,
    )
    _exact_bool(
        selection.get("choose_first_qualifying_release"),
        True,
        "data_roles.future_holdout.selection_policy.choose_first_qualifying_release",
        errors,
    )
    _exact_bool(
        selection.get("assignment_before_access_required"),
        True,
        "data_roles.future_holdout.selection_policy.assignment_before_access_required",
        errors,
    )
    qualification_criteria = _unique_nonempty_strings(
        selection.get("qualification_criteria"),
        "data_roles.future_holdout.selection_policy.qualification_criteria",
        errors,
    )
    if not qualification_criteria:
        errors.append(
            "data_roles.future_holdout.selection_policy.qualification_criteria must not be empty"
        )
    _nonempty_text(
        selection.get("disqualification_rule"),
        "data_roles.future_holdout.selection_policy.disqualification_rule",
        errors,
    )

    assignment_fields = (
        "dataset_id",
        "release_id",
        "release_date",
        "source_uri",
        "local_artifact_path",
        "sha256",
        "assigned_at",
        "assignment_manifest_id",
    )
    if assignment_status == "unassigned":
        for field in assignment_fields:
            if future.get(field) is not None:
                errors.append(f"data_roles.future_holdout.{field} must be null while unassigned")
    elif assignment_status == "assigned":
        for field in assignment_fields:
            value = future.get(field)
            _nonempty_text(value, f"data_roles.future_holdout.{field}", errors)
        assigned_hash = future.get("sha256")
        if isinstance(assigned_hash, str) and not SHA256_RE.fullmatch(assigned_hash):
            errors.append("data_roles.future_holdout.sha256 must be lowercase SHA-256 hex")
        local_artifact_path = future.get("local_artifact_path")
        if (
            isinstance(local_artifact_path, str)
            and isinstance(assigned_hash, str)
            and registered_hashes.get(local_artifact_path) != assigned_hash
        ):
            errors.append("assigned holdout local_artifact_path/sha256 must match input_artifacts")
        if (
            isinstance(local_artifact_path, str)
            and registered_kinds.get(local_artifact_path) != "holdout_data"
        ):
            errors.append("assigned holdout input artifact kind must be holdout_data")
        assigned_id = future.get("dataset_id")
        if assigned_id in exploratory_ids:
            errors.append("future holdout dataset must not also be exploratory/calibration data")
        if assigned_id == placeholder_id:
            errors.append("assigned dataset_id must replace, not reuse, the placeholder_id")
        freeze = _mapping(manifest.get("freeze"), "freeze", errors)
        release_date = future.get("release_date")
        freeze_date = freeze.get("freeze_date")
        if isinstance(release_date, str) and isinstance(freeze_date, str):
            try:
                parsed_release_date = date.fromisoformat(release_date)
                parsed_freeze_date = date.fromisoformat(freeze_date)
            except ValueError:
                errors.append("assigned holdout release_date must be an ISO calendar date")
            else:
                if parsed_release_date <= parsed_freeze_date:
                    errors.append("assigned holdout release_date must be after freeze_date")
        if future.get("assignment_manifest_id") != manifest.get("manifest_id"):
            errors.append("assigned holdout assignment_manifest_id must equal this manifest_id")
        revision = freeze.get("protocol_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 2:
            errors.append("an assigned holdout requires protocol_revision >= 2")
        if freeze.get("phase") != "holdout_assigned_pre_access":
            errors.append("an assigned holdout requires freeze.phase=holdout_assigned_pre_access")
        supersedes = _nonempty_text(
            manifest.get("supersedes_manifest_id"),
            "supersedes_manifest_id",
            errors,
        )
        if supersedes == manifest.get("manifest_id"):
            errors.append("an assignment manifest cannot supersede itself")
    if placeholder_id in exploratory_ids:
        errors.append("future holdout placeholder ID must not be an exploratory dataset ID")
    return future, set(exploratory_ids)


def _validate_fit_policy(manifest: Mapping[str, Any], errors: list[str]) -> set[str]:
    policy = _mapping(manifest.get("fit_policy"), "fit_policy", errors)
    count = policy.get("fitted_parameter_count_on_holdout")
    if isinstance(count, bool) or count != 0:
        errors.append("fit_policy.fitted_parameter_count_on_holdout must be 0")
    for field in (
        "allow_profile_fit",
        "allow_post_unblinding_model_selection",
        "allow_post_unblinding_threshold_changes",
        "allow_bin_or_outlier_removal",
    ):
        _exact_bool(policy.get(field), False, f"fit_policy.{field}", errors)
    forbidden = _unique_nonempty_strings(
        policy.get("forbidden_on_holdout"),
        "fit_policy.forbidden_on_holdout",
        errors,
    )
    if not forbidden:
        errors.append("fit_policy.forbidden_on_holdout must not be empty")
    _nonempty_text(
        policy.get("nuisance_parameter_policy"),
        "fit_policy.nuisance_parameter_policy",
        errors,
    )
    return forbidden


def _validate_tolerances(manifest: Mapping[str, Any], errors: list[str]) -> set[str]:
    tolerances = _sequence(manifest.get("numeric_tolerances"), "numeric_tolerances", errors)
    names: list[str] = []
    for index, raw_tolerance in enumerate(tolerances):
        prefix = f"numeric_tolerances[{index}]"
        tolerance = _mapping(raw_tolerance, prefix, errors)
        name = _nonempty_text(tolerance.get("name"), f"{prefix}.name", errors)
        if name:
            names.append(name)
        comparison = tolerance.get("comparison")
        if comparison not in ALLOWED_COMPARISONS:
            errors.append(f"{prefix}.comparison is not supported")
        numeric_value = _finite_number(tolerance.get("value"), f"{prefix}.value", errors)
        if (
            comparison in {"<=", "absolute<=", "relative<="}
            and numeric_value is not None
            and numeric_value < 0.0
        ):
            errors.append(f"{prefix}.value must be non-negative for {comparison}")
        _nonempty_text(tolerance.get("unit"), f"{prefix}.unit", errors)
        _nonempty_text(tolerance.get("scope"), f"{prefix}.scope", errors)
    if len(names) != len(set(names)):
        errors.append("numeric_tolerances names must be unique")
    if not tolerances:
        errors.append("numeric_tolerances must not be empty")
    return set(names)


def _validate_kill_rules(manifest: Mapping[str, Any], errors: list[str]) -> set[str]:
    rules = _sequence(manifest.get("hard_kill_rules"), "hard_kill_rules", errors)
    rule_ids: list[str] = []
    scopes: set[str] = set()
    for index, raw_rule in enumerate(rules):
        prefix = f"hard_kill_rules[{index}]"
        rule = _mapping(raw_rule, prefix, errors)
        rule_id = _nonempty_text(rule.get("rule_id"), f"{prefix}.rule_id", errors)
        if rule_id:
            rule_ids.append(rule_id)
        scope = _nonempty_text(rule.get("scope"), f"{prefix}.scope", errors)
        if scope:
            scopes.add(scope)
            if scope not in ALLOWED_KILL_SCOPES:
                errors.append(f"{prefix}.scope is not supported")
        _nonempty_text(rule.get("condition"), f"{prefix}.condition", errors)
        _nonempty_text(rule.get("action"), f"{prefix}.action", errors)
        _nonempty_text(rule.get("target"), f"{prefix}.target", errors)
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("hard_kill_rules rule IDs must be unique")
    if not rules:
        errors.append("hard_kill_rules must not be empty")
    if rules and scopes != ALLOWED_KILL_SCOPES:
        errors.append(
            "hard_kill_rules must separate scientific_candidate and protocol_validity scopes"
        )
    return set(rule_ids)


def _validate_cosmology(
    manifest: Mapping[str, Any],
    future: Mapping[str, Any],
    exploratory_ids: set[str],
    frozen_input_names: set[str],
    forbidden: set[str],
    tolerance_names: set[str],
    rule_ids: set[str],
    errors: list[str],
) -> bool:
    if "desi_dr2_all_13_point_compressed_bao" not in exploratory_ids:
        errors.append("cosmology must label DESI DR2 13-point BAO as exploratory/calibration")
    if future.get("dataset_id") == "desi_dr2_all_13_point_compressed_bao":
        errors.append("DESI DR2 must never be relabeled as the future holdout")
    required_inputs = {
        "omega_b0",
        "omega_dm0",
        "omega_lambda0",
        "h0",
        "sigma8_0",
        "w0",
        "wa",
        "gravity_mu_coupling",
    }
    missing_inputs = required_inputs - frozen_input_names
    if missing_inputs:
        errors.append(f"cosmology frozen inputs missing: {sorted(missing_inputs)}")
    required_forbidden = required_inputs | {
        "rd_mpc",
        "rd_mode",
        "tcmb_k",
        "n_eff",
        "covariance",
        "redshift_bins",
    }
    missing_forbidden = required_forbidden - forbidden
    if missing_forbidden:
        errors.append(f"cosmology holdout-fit prohibition missing: {sorted(missing_forbidden)}")
    required_tolerances = {
        "covariance_symmetry_absolute_error",
        "distance_integration_relative_error",
        "chi_square_regression_absolute_error",
    }
    missing_tolerances = required_tolerances - tolerance_names
    if missing_tolerances:
        errors.append(f"cosmology numeric tolerances missing: {sorted(missing_tolerances)}")

    criteria = _mapping(manifest.get("acceptance_criteria"), "acceptance_criteria", errors)
    if criteria.get("primary_endpoint") != "full_covariance_fixed_model_chi_square_p_value":
        errors.append("cosmology primary endpoint must be the fixed-model full-covariance p-value")
    if criteria.get("covariance_required") != "full":
        errors.append("cosmology acceptance requires the full released covariance")
    _exact_bool(
        criteria.get("report_all_registered_candidates"),
        True,
        "acceptance_criteria.report_all_registered_candidates",
        errors,
    )
    thresholds = _mapping(criteria.get("thresholds"), "acceptance_criteria.thresholds", errors)
    if thresholds.get("pass_minimum_p_value") != 0.05:
        errors.append("cosmology pass threshold must remain p >= 0.05")
    if thresholds.get("reject_below_p_value") != 0.0027:
        errors.append("cosmology reject threshold must remain p < 0.0027")

    models = _sequence(manifest.get("registered_models"), "registered_models", errors)
    model_ids: list[str] = []
    for index, raw_model in enumerate(models):
        prefix = f"registered_models[{index}]"
        model = _mapping(raw_model, prefix, errors)
        model_id = _nonempty_text(model.get("model_id"), f"{prefix}.model_id", errors)
        if model_id:
            model_ids.append(model_id)
        _exact_bool(
            model.get("fixed_before_holdout"), True, f"{prefix}.fixed_before_holdout", errors
        )
        _exact_bool(
            model.get("select_after_unblinding"),
            False,
            f"{prefix}.select_after_unblinding",
            errors,
        )
        _mapping(model.get("parameters"), f"{prefix}.parameters", errors)
    if len(model_ids) != len(set(model_ids)):
        errors.append("registered cosmology model IDs must be unique")
    if set(model_ids) != {"ce_density_external_rd_v1", "ce_density_eh_hybrid_v1"}:
        errors.append("cosmology must keep both frozen r_d candidates and report both")
    required_rule_ids = {
        "COS-SCI-001",
        "COS-PROT-001",
        "COS-PROT-002",
        "COS-PROT-003",
        "COS-PROT-004",
        "COS-PROT-005",
    }
    if required_rule_ids - rule_ids:
        errors.append(f"cosmology hard kill rules missing: {sorted(required_rule_ids - rule_ids)}")
    return future.get("assignment_status") == "assigned"


def _validate_quantum(
    manifest: Mapping[str, Any],
    future: Mapping[str, Any],
    exploratory_ids: set[str],
    forbidden: set[str],
    tolerance_names: set[str],
    rule_ids: set[str],
    errors: list[str],
) -> bool:
    if "historical_arc_94_6_percent_record" not in exploratory_ids:
        errors.append("quantum must label the historical ARC 94.6 record as exploratory")
    if future.get("dataset_id") == "historical_arc_94_6_percent_record":
        errors.append("the historical ARC 94.6 record cannot be a future holdout")
    contract = _mapping(manifest.get("model_contract"), "model_contract", errors)
    if contract.get("branch") != "A_independent_scalar_field":
        errors.append("quantum model contract must freeze branch A")
    _exact_bool(
        contract.get("phi_equals_ricci_scalar"),
        False,
        "model_contract.phi_equals_ricci_scalar",
        errors,
    )
    reference_mass = _mapping(
        contract.get("scalar_mass"),
        "model_contract.scalar_mass",
        errors,
    )
    if reference_mass.get("value_mev") != 29.64757:
        errors.append("quantum scalar reference mass must remain 29.64757 MeV")
    if reference_mass.get("role") != "reference_not_prediction":
        errors.append("quantum scalar mass must be labelled reference_not_prediction")
    readiness = contract.get("evaluation_readiness")
    if readiness not in {"blocked_open_model", "ready"}:
        errors.append("quantum evaluation_readiness must be blocked_open_model or ready")
    unresolved = _unique_nonempty_strings(
        contract.get("unresolved_physical_inputs"),
        "model_contract.unresolved_physical_inputs",
        errors,
    )
    required_unresolved = {
        "scalar_action_parameters",
        "system_coupling_g",
        "system_operator_A",
        "field_operator_O_phi",
        "bath_state",
        "spectral_density_J",
        "rate_unit_convention_and_spectrum_units",
        "kossakowski_matrix_or_jump_basis",
        "jacobi_operator_and_probe",
    }
    if readiness == "blocked_open_model":
        if required_unresolved - unresolved:
            errors.append(
                "quantum unresolved physical inputs missing: "
                f"{sorted(required_unresolved - unresolved)}"
            )
    elif readiness == "ready":
        if unresolved:
            errors.append("a ready quantum model must have no unresolved_physical_inputs")
        resolved = _mapping(
            contract.get("resolved_physical_inputs"),
            "model_contract.resolved_physical_inputs",
            errors,
        )
        for name in sorted(required_unresolved):
            if resolved.get(name) is None:
                errors.append(f"ready quantum model missing resolved physical input: {name}")
    required_forbidden = {
        "scalar_mass",
        "curvature_coupling",
        "quartic_coupling",
        "system_coupling_g",
        "system_operator_A",
        "field_operator_O_phi",
        "bath_state",
        "spectral_density_J",
        "rate_unit_convention",
        "spectrum_units",
        "kossakowski_matrix",
        "jump_operator_basis",
        "jacobi_operator",
        "rayleigh_probe",
        "measurement_threshold",
        "model_choice",
    }
    if required_forbidden - forbidden:
        errors.append(
            f"quantum holdout-fit prohibition missing: {sorted(required_forbidden - forbidden)}"
        )
    models = _sequence(manifest.get("registered_models"), "registered_models", errors)
    models_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_model in enumerate(models):
        prefix = f"registered_models[{index}]"
        model = _mapping(raw_model, prefix, errors)
        model_id = _nonempty_text(model.get("model_id"), f"{prefix}.model_id", errors)
        if model_id in models_by_id:
            errors.append("registered quantum model IDs must be unique")
        elif model_id:
            models_by_id[model_id] = model
        _exact_bool(
            model.get("post_unblinding_selection_allowed"),
            False,
            f"{prefix}.post_unblinding_selection_allowed",
            errors,
        )
    required_model_ids = {
        "standard_qm_device_baseline_v1",
        "ce_branch_a_independent_scalar_v1",
    }
    if set(models_by_id) != required_model_ids:
        errors.append("quantum must keep one frozen standard-QM baseline and branch-A candidate")
    if readiness == "blocked_open_model":
        candidate = models_by_id.get("ce_branch_a_independent_scalar_v1", {})
        if candidate.get("status") != "blocked_open_model":
            errors.append("the unresolved branch-A candidate status must be blocked_open_model")
    elif readiness == "ready":
        for model_id in sorted(required_model_ids):
            if models_by_id.get(model_id, {}).get("status") != "ready":
                errors.append(f"ready quantum evaluation requires {model_id} status=ready")
    required_tolerances = {
        "density_trace_absolute_error",
        "density_hermiticity_absolute_error",
        "minimum_choi_eigenvalue",
        "born_probability_sum_absolute_error",
        "minimum_born_probability",
        "no_signalling_max_probability_variation",
        "minimum_spectral_density",
        "minimum_kossakowski_eigenvalue",
        "kms_log_detailed_balance_absolute_error",
    }
    if required_tolerances - tolerance_names:
        errors.append(
            "quantum physicality tolerances missing: "
            f"{sorted(required_tolerances - tolerance_names)}"
        )
    criteria = _mapping(manifest.get("acceptance_criteria"), "acceptance_criteria", errors)
    if criteria.get("primary_endpoint") != "paired_delta_log_predictive_density_per_observation":
        errors.append("quantum primary endpoint must remain paired held-out log predictive density")
    _exact_bool(
        criteria.get("all_physicality_gates_must_pass"),
        True,
        "acceptance_criteria.all_physicality_gates_must_pass",
        errors,
    )
    _exact_bool(
        criteria.get("baseline_and_candidate_use_identical_holdout_records"),
        True,
        "acceptance_criteria.baseline_and_candidate_use_identical_holdout_records",
        errors,
    )
    required_rule_ids = {
        "QUA-SCI-001",
        "QUA-SCI-002",
        "QUA-SCI-003",
        "QUA-SCI-004",
        "QUA-SCI-005",
        "QUA-PROT-001",
        "QUA-PROT-002",
        "QUA-PROT-003",
        "QUA-PROT-004",
        "QUA-PROT-005",
    }
    if required_rule_ids - rule_ids:
        errors.append(f"quantum hard kill rules missing: {sorted(required_rule_ids - rule_ids)}")
    # Even an assigned dataset is not evaluable until a superseding manifest
    # freezes the currently unresolved physical model.
    return future.get("assignment_status") == "assigned" and readiness == "ready"


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path | str = ROOT,
    verify_artifacts: bool = True,
    require_assigned_holdout: bool = False,
) -> ManifestValidationReport:
    """Validate common and domain-specific preregistration invariants."""
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a mapping")
    manifest_id = (
        manifest.get("manifest_id") if isinstance(manifest.get("manifest_id"), str) else ""
    )
    domain = manifest.get("domain") if isinstance(manifest.get("domain"), str) else ""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    _nonempty_text(manifest_id, "manifest_id", errors)
    if domain not in ALLOWED_DOMAINS:
        errors.append(f"domain must be one of {sorted(ALLOWED_DOMAINS)}")
    if manifest.get("manifest_hash_policy") != MANIFEST_HASH_POLICY:
        errors.append(f"manifest_hash_policy must be {MANIFEST_HASH_POLICY}")
    expected_manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(expected_manifest_hash, str) or not SHA256_RE.fullmatch(
        expected_manifest_hash
    ):
        errors.append("manifest_sha256 must be lowercase SHA-256 hex")
    else:
        try:
            actual_manifest_hash = compute_manifest_sha256(manifest)
        except (TypeError, ValueError) as exc:
            errors.append(f"manifest canonicalization failed: {exc}")
        else:
            if expected_manifest_hash != actual_manifest_hash:
                errors.append(
                    "manifest_sha256 mismatch: "
                    f"expected {expected_manifest_hash}, got {actual_manifest_hash}"
                )
    frozen_v1_hash = FROZEN_V1_MANIFEST_SHA256.get(manifest_id)
    if frozen_v1_hash is not None and expected_manifest_hash != frozen_v1_hash:
        errors.append(
            "frozen v1 digest mismatch: preserve v1 and create a higher-revision "
            "manifest with a new manifest_id"
        )
    frozen_v2_hash = FROZEN_V2_MANIFEST_SHA256.get(manifest_id)
    if frozen_v2_hash is not None and expected_manifest_hash != frozen_v2_hash:
        errors.append(
            "frozen v2 digest mismatch: preserve v2 and create a higher-revision "
            "manifest with a new manifest_id"
        )

    _validate_freeze(manifest, errors)
    _validate_scope(manifest, errors)
    registered_hashes, registered_kinds = _validate_artifacts(
        manifest,
        Path(repo_root),
        verify_artifacts,
        errors,
    )
    frozen_input_names = _validate_frozen_inputs(manifest, errors)
    future, exploratory_ids = _validate_data_roles(
        manifest,
        registered_hashes,
        registered_kinds,
        errors,
    )
    forbidden = _validate_fit_policy(manifest, errors)
    tolerance_names = _validate_tolerances(manifest, errors)
    rule_ids = _validate_kill_rules(manifest, errors)

    assignment_status = (
        future.get("assignment_status")
        if isinstance(future.get("assignment_status"), str)
        else "invalid"
    )
    expected_status = (
        "frozen_with_assigned_holdout"
        if assignment_status == "assigned"
        else "frozen_with_unassigned_holdout"
    )
    if manifest.get("status") != expected_status:
        errors.append(f"status must be {expected_status} for this holdout state")
    if require_assigned_holdout and assignment_status != "assigned":
        errors.append("future holdout is not assigned")

    evaluation_ready = False
    if domain == "cosmology":
        evaluation_ready = _validate_cosmology(
            manifest,
            future,
            exploratory_ids,
            frozen_input_names,
            forbidden,
            tolerance_names,
            rule_ids,
            errors,
        )
    elif domain == "quantum":
        evaluation_ready = _validate_quantum(
            manifest,
            future,
            exploratory_ids,
            forbidden,
            tolerance_names,
            rule_ids,
            errors,
        )

    if errors:
        evaluation_ready = False
    return ManifestValidationReport(
        manifest_id=manifest_id,
        domain=domain,
        holdout_status=assignment_status,
        structurally_valid=not errors,
        evaluation_ready=evaluation_ready,
        errors=tuple(errors),
    )


def validate_manifest_path(
    path: Path | str,
    *,
    repo_root: Path | str = ROOT,
    verify_artifacts: bool = True,
    require_assigned_holdout: bool = False,
) -> ManifestValidationReport:
    """Load and validate one manifest path."""
    return validate_manifest(
        load_manifest(path),
        repo_root=repo_root,
        verify_artifacts=verify_artifacts,
        require_assigned_holdout=require_assigned_holdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validate_holdout_manifest")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_MANIFEST_PATHS),
        help="manifest JSON paths (defaults to both frozen manifests)",
    )
    parser.add_argument(
        "--require-assigned-holdout",
        action="store_true",
        help="fail if the future holdout placeholder has not been assigned",
    )
    parser.add_argument(
        "--no-verify-artifacts",
        action="store_true",
        help="validate declared hashes syntactically without reading repository artifacts",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        try:
            report = validate_manifest_path(
                path,
                verify_artifacts=not args.no_verify_artifacts,
                require_assigned_holdout=args.require_assigned_holdout,
            )
        except (OSError, json.JSONDecodeError, ManifestValidationError) as exc:
            print(f"{path}: INVALID: {exc}")
            exit_code = 1
            continue
        state = "VALID" if report.structurally_valid else "INVALID"
        readiness = "READY" if report.evaluation_ready else "NOT_READY"
        print(f"{path}: {state} holdout={report.holdout_status} evaluation={readiness}")
        for error in report.errors:
            print(f"  - {error}")
        if report.errors:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
