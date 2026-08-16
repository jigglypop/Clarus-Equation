from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from reality_stone.clarus import agi_world_memory_integration_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/preregistration/agi_world_memory_integration_v3.json"
BASE_CONFIG = (
    ROOT / "experiments/preregistration/agi_world_memory_integration_v2.json"
)
CONTRACT = (
    ROOT
    / "_workspace/ce/agi-world-memory-integration-v1-20260810/revisions"
    / "00-contract-v2-draft.md"
)
AMENDMENT = (
    ROOT
    / "_workspace/ce/agi-world-memory-integration-v1-20260810/revisions"
    / "31-v3-boundary-amendment.md"
)

EXPECTED_V3_REGISTRATION_SHA256 = (
    "bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b"
)
EXPECTED_V2_REGISTRATION_SHA256 = (
    "b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2"
)
EXPECTED_V2_CONTRACT_SHA256 = (
    "842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70"
)
EXPECTED_V3_AMENDMENT_SHA256 = (
    "9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340"
)
EXPECTED_MERGED_REGISTRATION_SHA256 = (
    "37e7bfb6ee100c47164bec49f2e151234a647964839189ba47bf504552e1644b"
)
EXPECTED_ALLOCATION_LEDGER_SHA256 = (
    "7f5c52b1b4aa01f8141ce821ed1bf4164e3fdf131ae828f08b20a8280f3079b4"
)

V2_REGISTERED_SEEDS = {
    *range(86100, 86140),
    *range(87100, 87140),
    *range(88100, 88160),
}
V3_REGISTERED_SEEDS = {
    *range(92100, 92140),
    *range(93100, 93140),
    *range(94100, 94160),
}
ALL_FORBIDDEN_UNIT_SEEDS = V2_REGISTERED_SEEDS | V3_REGISTERED_SEEDS

EXPECTED_BUDGET = (
    ("N_wake_records", 96),
    ("N_wake_transitions", 1152),
    ("U_core", 0),
    ("P_core", 20),
    ("N_origins", 24),
    ("K", 8),
    ("H", 20),
    ("N_rollout_calls", 192),
    ("N_predicted_transitions", 3840),
    ("N_planner_score_calls", 192),
    ("B_ltm_trace_bytes", 73728),
    ("Q_ltm_call_slots", 72),
    ("N_scoped_distance_rows", 576),
    ("N_schema_key_slots", 72),
    ("N_ordered_pair_enumerations", 288),
    ("N_component_port_checks", 288),
    ("N_same_component_pairs", 72),
    ("N_observed_keys", 48),
    ("N_join_candidates", 24),
    ("N_scalar_endpoint_join_values", 48),
    ("N_dream_output_slots", 24),
    ("U_dream_update_slots", 24),
    ("N_lesion_nonobserved_pairs", 240),
    ("N_lesion_accepted_slots", 24),
    ("N_lesion_capacity_padding", 216),
    ("N_dream_passes", 1),
    ("persistent_numeric_payload_bytes", 393216),
    ("persistent_byte_cap", 524288),
    ("temporary_workspace_byte_cap", 1048576),
)

EXPECTED_PERFORMANCE_CHECKS = {
    "prediction.marginal_ltm_relative_reduction",
    "prediction.marginal_ltm_ci_lower",
    "prediction.marginal_ltm_strict_win_fraction",
    "prediction.dream_M00_to_M01_relative_reduction",
    "prediction.dream_M00_to_M01_ci_lower",
    "prediction.dream_M00_to_M01_strict_win_fraction",
    "prediction.dream_M10_to_M11_relative_reduction",
    "prediction.dream_M10_to_M11_ci_lower",
    "prediction.dream_M10_to_M11_strict_win_fraction",
    "prediction.joint_relative_reduction",
    "prediction.joint_ci_lower",
    "prediction.joint_strict_win_fraction",
    "prediction.M11_E_all_H20",
    "prediction.M01_E_uv_H20",
    "prediction.M11_E_uv_H20",
    "prediction.M11_H20_over_H5",
    "prediction.M11_vs_persistence_relative_reduction",
    "prediction.M11_vs_persistence_ci_lower",
    "prediction.M11_vs_persistence_strict_win_fraction",
    "planning.M11_regret_relative_reduction_vs_M00",
    "planning.regret_ci_lower",
    "planning.success_gain",
    "planning.success_gain_ci_lower",
    "planning.M11_success_mean",
    "planning.M11_invalid_selected_count",
    "recall.M10_coverage",
    "recall.M10_identity_accuracy",
    "recall.M10_wrong_all",
    "recall.M10_wrong_given_accept",
    "recall.M10_false_lure_mean",
    "recall.M10_false_lure_ci_upper",
    "recall.M10_cross_port_accept_count",
    "recall.M11_coverage",
    "recall.M11_identity_accuracy",
    "recall.M11_wrong_all",
    "recall.M11_wrong_given_accept",
    "recall.M11_false_lure_mean",
    "recall.M11_false_lure_ci_upper",
    "recall.M11_cross_port_accept_count",
    "dream.M01_missing_binding_coverage",
    "dream.M01_accepted_invalid_splice_count",
    "dream.M01_observed_overwrite_count",
    "dream.M11_missing_binding_coverage",
    "dream.M11_accepted_invalid_splice_count",
    "dream.M11_observed_overwrite_count",
    "no_antagonism.recall_paired_upper",
    "no_antagonism.dream_paired_upper",
    "attribution.shuffled_vs_M10_ci_lower",
    "attribution.zero_q_vs_M10_ci_lower",
    "attribution.lesion_invalid_splice_vs_M01_ci_lower",
    "attribution.zero_synthetic_vs_M01_E_uv_ci_lower",
    "stability.all_finite",
    "stability.max_abs_prediction",
    "stability.max_seed_invalid_predicted_transition_rate",
    "stability.h5_h20_bit_exact",
}

EXPECTED_RESOURCE_CHECKS = {
    "budget_vector_exact",
    "allocation_bytes_exact",
    "persistent_within_cap",
    "temporary_within_cap",
    "metadata_within_cap",
    "cpu_only",
    "numpy_only",
    "one_process",
    "network_downloads_zero",
    "gpu_disabled",
    "external_trajectory_files_zero",
    "wall_within_target",
}

EXPECTED_HARD_ZERO_CHECKS = {
    "synthetic_with_episode_id",
    "synthetic_tagged_observed",
    "synthetic_tagged_recalled",
    "synthetic_to_ltm_insert_attempts",
    "synthetic_to_ltm_successful_inserts",
    "nonledger_real_record_in_ltm",
    "observed_record_overwrite_or_hash_change",
    "cross_context_or_cross_port_or_cross_component_accepted_splice",
    "accepted_context_component_phase_action_or_key_constraint_violation",
    "accepted_cross_context_recall",
    "heldout_future_reads",
    "evaluator_latent_or_truth_reads",
    "generator_validity_or_outcome_reads",
    "masked_cue_coordinate_reads",
    "test_path_reads_before_unlock",
    "cell_cross_write_or_shared_mutation",
    "nonfinite_outputs",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    assert raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    return json.loads(raw), raw


def _canonical_merged_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reference_merge() -> tuple[dict[str, Any], int, int]:
    amendment, _ = _locked_json(CONFIG)
    base, _ = _locked_json(BASE_CONFIG)
    merged = copy.deepcopy(base)
    deleted = 0
    for segments in amendment["delete_paths"]:
        parent: dict[str, Any] = merged
        for segment in segments[:-1]:
            assert segment in parent and isinstance(parent[segment], dict)
            parent = parent[segment]
        leaf = segments[-1]
        assert leaf in parent
        del parent[leaf]
        deleted += 1

    allowed = {
        tuple(segments)
        for segments in amendment["merge_semantics"][
            "allowed_new_override_paths"
        ]
    }
    added_paths: list[tuple[str, ...]] = []

    def merge_object(
        destination: dict[str, Any],
        source: Mapping[str, Any],
        prefix: tuple[str, ...] = (),
    ) -> None:
        for key, value in source.items():
            path = (*prefix, key)
            if key not in destination:
                assert path in allowed
                added_paths.append(path)
                destination[key] = copy.deepcopy(value)
            elif isinstance(destination[key], dict) and isinstance(value, Mapping):
                merge_object(destination[key], value, path)
            else:
                destination[key] = copy.deepcopy(value)

    merge_object(merged, amendment["overrides"])
    assert set(added_paths) == allowed
    for key, value in amendment.items():
        if key != "overrides":
            merged[key] = copy.deepcopy(value)
    return merged, len(added_paths), deleted


@pytest.fixture(autouse=True)
def _forbid_registered_seed_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before any registered V2/V3 seed can reach a world generator."""

    for name in (
        "_generate_seed_world_v3",
        "generate_seed_world_v3",
        "_generate_seed_world",
    ):
        original = getattr(v3, name, None)
        if original is None:
            continue

        def guarded(*args: Any, __original: Any = original, **kwargs: Any) -> Any:
            seed = kwargs.get("master_seed", args[0] if args else None)
            if isinstance(seed, (int, np.integer)) and int(seed) in (
                ALL_FORBIDDEN_UNIT_SEEDS
            ):
                raise AssertionError("unit tests may not execute a registered seed")
            return __original(*args, **kwargs)

        monkeypatch.setattr(v3, name, guarded)


def test_v3_raw_registration_base_contract_and_amendment_are_byte_locked() -> None:
    registration, raw = _locked_json(CONFIG)

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_V3_REGISTRATION_SHA256
    assert _sha256(BASE_CONFIG) == EXPECTED_V2_REGISTRATION_SHA256
    assert _sha256(CONTRACT) == EXPECTED_V2_CONTRACT_SHA256
    assert _sha256(AMENDMENT) == EXPECTED_V3_AMENDMENT_SHA256
    assert registration["schema_version"] == 3
    assert registration["status"] == "locked_pre_implementation"
    assert registration["experiment"] == "agi_world_memory_integration_v3"
    assert registration["roadmap_stage"] == "G9-CBM/V3"
    assert registration["runner"] == "agi_world_memory_integration_v3"
    assert registration["standalone"] is False
    assert registration["extends"] == "agi_world_memory_integration_v2.json"
    assert registration["predecessor"] == {
        "status": "BLOCKED_PRE_IMPLEMENTATION",
        "git_commit": "407a9714483ffd417660d9fe24db83b96f162301",
        "registration_raw_sha256": EXPECTED_V2_REGISTRATION_SHA256,
        "implementation_lock_created": False,
        "train_execution_count": 0,
        "validation_execution_count": 0,
        "test_execution_count": 0,
        "reason": (
            "Inherited PartialCue field names were mismatched and unstored-lure "
            "cue mask/noise chronology was outcome-determining but unspecified."
        ),
    }
    integrity = registration["amendment_integrity"]
    assert integrity["raw_sha256"] == EXPECTED_V3_AMENDMENT_SHA256
    assert integrity["base_contract_raw_sha256"] == EXPECTED_V2_CONTRACT_SHA256
    assert integrity["base_registration_raw_sha256"] == (
        EXPECTED_V2_REGISTRATION_SHA256
    )


def test_v3_recursive_merge_has_exactly_eighteen_additions_and_five_deletions() -> None:
    merged, additions, deletions = _reference_merge()
    loaded = v3.load_merged_registration_v3(CONFIG)

    assert additions == 18
    assert deletions == 5
    assert loaded == merged
    assert len(_canonical_merged_bytes(merged)) == 62491
    assert hashlib.sha256(_canonical_merged_bytes(merged)).hexdigest() == (
        EXPECTED_MERGED_REGISTRATION_SHA256
    )
    dtypes = merged["candidate_api"]["exact_field_dtypes"]
    assert "PartialCue.raw_codec" not in dtypes
    assert "PartialCue.mask" not in dtypes
    assert dtypes["PartialCue.cue_values"] == "float64[12,8]"
    assert dtypes["PartialCue.cue_mask"] == "bool[12,8]"
    assert "final_status_audit_gate" not in merged["preregistration_integrity"]
    assert "final_mechanical_consistency_gate" not in (
        merged["preregistration_integrity"]
    )
    assert "scientific_post_lock_change_requires_v3_fresh_seeds" not in (
        merged["preregistration_integrity"]
    )
    assert merged["preregistration_integrity"][
        "scientific_post_lock_change_requires_v4_fresh_seeds"
    ] is True


def test_v3_roles_are_fresh_exact_and_never_unit_fixtures() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    expected = {
        "train": tuple(range(92100, 92140)),
        "validation": tuple(range(93100, 93140)),
        "test": tuple(range(94100, 94160)),
    }

    for role, seeds in expected.items():
        item = merged["data_roles"][role]
        assert tuple(item["seeds"]) == seeds
        assert item["count"] == len(seeds)
        assert item["run_exactly_once"] is True
        assert item["fresh_after_v2_block"] is True
    assert set(expected["train"]).isdisjoint(expected["validation"])
    assert set(expected["train"]).isdisjoint(expected["test"])
    assert set(expected["validation"]).isdisjoint(expected["test"])
    assert set().union(*(set(value) for value in expected.values())) == (
        V3_REGISTERED_SEEDS
    )
    assert V2_REGISTERED_SEEDS.isdisjoint(V3_REGISTERED_SEEDS)


def test_partial_cue_and_candidate_api_field_order_is_exact() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    api = merged["candidate_api"]

    assert [field.name for field in fields(v3.PartialCue)] == [
        "context_token",
        "prefix_token",
        "suffix_token",
        "cue_values",
        "cue_mask",
    ]
    for type_name in (
        "CoreModelV2",
        "CostSpecV2",
        "CodecSpecV2",
        "CandidateRequestV2",
        "OriginRecallAuditV2",
        "CandidateResultV2",
        "SeedRecallAuditV2",
    ):
        assert [field.name for field in fields(getattr(v3, type_name))] == api[
            type_name
        ]
    assert api["PartialCue"] == [
        "context_token",
        "prefix_token",
        "suffix_token",
        "cue_values",
        "cue_mask",
    ]


def test_public_callable_boundaries_have_no_evaluator_or_seed_channel() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    assert tuple(inspect.signature(v3.scoped_hard_recall_v3).parameters) == (
        "store",
        "cue",
        "scope_index",
        "enabled",
    )
    assert tuple(inspect.signature(v3.codec_residual_view_v3).parameters) == (
        "codec",
        "codec_spec",
        "standardized",
    )
    assert tuple(
        inspect.signature(v3.constrained_residual_completion_v3).parameters
    ) == ("schema", "join_threshold", "write_enabled", "audit")
    assert tuple(inspect.signature(v3.execute_candidate_v3).parameters) == (
        "request",
    )
    assert tuple(
        inspect.signature(v3.run_agi_world_memory_integration_v3_gate).parameters
    ) == ("config_path", "split", "output_path")
    assert tuple(
        inspect.signature(v3.prepare_implementation_lock_v3).parameters
    ) == ("config_path", "output_path")

    forbidden = {
        "cell_label",
        "world",
        "episode",
        "master_seed",
        "seed",
        "stream_id",
        "split",
        "target_id",
        "evaluator_handle",
        "future_state",
        "true_q",
        "true_schema",
        "generator_validity",
        "outcome",
        "realized_cost",
        "reward",
        "oracle_rank",
    }
    assert not forbidden & set(inspect.signature(v3.execute_candidate_v3).parameters)
    assert merged["candidate_api"]["forbidden_inputs"] == [
        "cell_label",
        "World",
        "Episode",
        "master_seed",
        "stream_id",
        "split",
        "target_id",
        "evaluator_handle",
        "future_state",
        "evaluation_innovation_eta",
        "true_q",
        "true_schema",
        "generator_validity",
        "outcome",
        "realized_cost",
        "reward",
        "oracle_rank",
    ]


def test_v3_cue_mask_noise_and_cross_port_chronology_is_fully_locked() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    cue = merged["evaluation_origins_and_cues"]["cue"]
    lure = merged["evaluation_origins_and_cues"]["lure_cue"]

    assert cue["shape"] == [12, 8]
    assert cue["visible_count"] == 24
    assert cue["visible_by_slots"] == [10, 4, 10]
    assert cue["mask_shared_positive_and_lure"] is True
    assert cue["noise_shared_positive_and_lure"] is False
    assert "stream 11 independently permutes" in cue["mask_sampling"]
    assert "no additional stream-11 draw" in cue["mask_sampling"]
    assert "ascending global C-order flat indices" in cue["noise_chronology"]
    assert "24 iid standard Normals for positive" in cue["noise_chronology"]
    assert "24 fresh iid standard Normals for lure" in cue["noise_chronology"]
    assert lure["mask"] == "exact corresponding positive cue_mask"
    assert lure["hidden_values"] == "poison and never read"
    assert "never inserted into the 96-record ledger" in lure["storage"]
    assert merged["evaluation_origins_and_cues"][
        "cross_port_diagnostic_construction"
    ] == (
        "reuse the complete noisy positive PartialCue byte-for-value and replace "
        "only suffix_token by the same-local suffix token from port (port+1) mod 4"
    )


def test_typed_codebooks_and_invalid_sentinels_are_exact() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    ltm = merged["episodic_ltm"]
    schema = merged["slow_schema_and_dream"]
    api = merged["candidate_api"]

    assert merged["resources"]["boolean_dtype_contract"] == {
        "numpy_dtype_string": "bool",
        "numpy_scalar_type": "numpy.bool_",
        "itemsize_bytes": 1,
        "markdown_semantic_name": "bool8",
        "removed_numpy_alias_bool8_forbidden": True,
    }
    assert np.dtype("bool").itemsize == 1
    assert ltm["scope_codes"] == {
        "disabled_no_queryable": 0,
        "valid_12_row_scope": 1,
        "invalid_context_or_component": 2,
        "invalid_range": [3, 255],
    }
    assert ltm["origin_recall_invariants"]["physical_identity_valid_range"] == [
        0,
        95,
    ]
    assert ltm["origin_recall_invariants"]["rejected_identity"] == -1
    assert schema["key_dtype"] == "int16"
    assert schema["valid_key_range"] == [0, 71]
    assert schema["only_allowed_negative_key"] == -1
    assert schema["source_codes"] == {
        "unresolved": 0,
        "observed_real": 1,
        "synthetic_hypothetical": 2,
        "component_fallback": 3,
        "invalid_range": [4, 255],
    }
    assert schema["pair_reason_codes"] == {
        "unexamined": 0,
        "same_component_candidate": 1,
        "component_port_rejection": 2,
        "observed_key_rejection": 3,
        "join_accepted": 4,
        "left_join_rejection": 5,
        "right_join_rejection": 6,
        "lesion_accepted_valid_missing": 7,
        "lesion_accepted_invalid_cross_port": 8,
        "lesion_capacity_padding": 9,
        "invalid_range": [10, 255],
        "both_join_fail_priority": "left_join_rejection",
    }
    assert api["result_codebook_invariants"] == {
        "resolved_schema_key": "int16 -1 or 0..71 only",
        "schema_source": "uint8 0..3 only; unresolved key -1 iff source 0",
        "selected_index": "int64 0..7",
        "origin_recall_audit": "must satisfy episodic_ltm.origin_recall_invariants",
    }


def test_budget_allocation_and_condition_ledger_are_exact() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    resources = merged["resources"]
    registered = tuple(
        (item["name"], item["value"])
        for item in resources["registered_budget_vector"]
    )

    assert registered == EXPECTED_BUDGET
    assert resources["registered_budget_vector_length"] == 29
    assert resources["allocation_total_bytes"] == 393216
    assert resources["persistent_byte_cap"] == 524288
    assert resources["temporary_workspace_byte_cap"] == 1048576
    assert resources["metadata_utf8_cap"] == 32768
    canonical_ledger = json.dumps(
        resources["allocation_ledger"],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical_ledger).hexdigest() == (
        EXPECTED_ALLOCATION_LEDGER_SHA256
    )
    assert sum(item["bytes"] for item in resources["allocation_ledger"]) == 393216

    condition = v3.ConditionLedgerV3.allocate(merged)
    assert condition.total_bytes == 393216
    assert condition.ledger_sha256 == (
        EXPECTED_ALLOCATION_LEDGER_SHA256
    )
    for item in resources["allocation_ledger"]:
        array = condition.arrays[item["name"]]
        assert array.shape == tuple(item["shape"])
        assert array.dtype == np.dtype(item["dtype"])
        assert array.flags.c_contiguous
        assert array.nbytes == item["bytes"]


def test_all_of_pass_keysets_and_three_way_identity_are_exact() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    mapping = merged["all_of_gate"]["split_pass_mapping"]

    assert set(mapping["checks_exact_keyset"]) == EXPECTED_PERFORMANCE_CHECKS
    assert len(mapping["checks_exact_keyset"]) == len(EXPECTED_PERFORMANCE_CHECKS)
    assert set(mapping["resource_checks_exact_keyset"]) == EXPECTED_RESOURCE_CHECKS
    assert len(mapping["resource_checks_exact_keyset"]) == len(
        EXPECTED_RESOURCE_CHECKS
    )
    assert set(mapping["hard_zero_checks_exact_keyset"]) == (
        EXPECTED_HARD_ZERO_CHECKS
    )
    assert len(mapping["hard_zero_checks_exact_keyset"]) == len(
        EXPECTED_HARD_ZERO_CHECKS
    )
    assert mapping["performance_passed"] == "all values in checks are true"
    assert mapping["integrity_passed"] == (
        "all values in hard_zero_checks are true"
    )
    assert mapping["resource_passed"] == (
        "all values in resource_checks are true"
    )
    assert mapping["passed"] == (
        "performance_passed and integrity_passed and resource_passed"
    )
    assert mapping["validation_and_test_use_identical_mapping"] is True
    assert mapping["recompute_from_primitive_seed_vectors"] is True
    assert mapping["stored_boolean_mismatch_hard_fail"] is True


def test_v3_paths_are_new_and_predecessor_artifacts_remain_unopened() -> None:
    merged = v3.load_merged_registration_v3(CONFIG)
    lock = merged["test_lock"]
    required = {
        "registration_path": "experiments/preregistration/agi_world_memory_integration_v3.json",
        "module_path": "reality_stone/python/reality_stone/clarus/agi_world_memory_integration_v3.py",
        "runner_path": "examples/agi/agi_world_memory_integration_v3_gate.py",
        "unit_test_path": "tests/test_agi_world_memory_integration_v3.py",
        "integrity_test_path": "tests/test_agi_world_memory_integration_integrity_v3.py",
        "implementation_lock_path": "artifacts/agi/agi_world_memory_integration_implementation_lock_v3.json",
        "calibration_path": "artifacts/agi/agi_world_memory_integration_train_calibration_v3.json",
        "validation_path": "artifacts/agi/agi_world_memory_integration_validation_v3.json",
        "test_path": "artifacts/agi/agi_world_memory_integration_test_v3.json",
        "integrity_path": "artifacts/agi/agi_world_memory_integration_integrity_v3.json",
        "unlock_record": "in-memory UnlockRecordV3 serialized inside test artifact",
    }
    for key, value in required.items():
        assert lock[key] == value
    assert all(
        path.endswith("_v3.json")
        for key, path in lock.items()
        if key.endswith("_path") and path.endswith(".json")
    )
    assert merged["artifact_state_machine"]["global_rules"][
        "deleting_an_artifact_never_authorizes_rerun"
    ] is True
    assert merged["artifact_state_machine"]["global_rules"]["no_overwrite"] is True


def _cue_mask() -> np.ndarray:
    mask = np.zeros((12, 8), dtype=bool)
    for row_slice, count in ((slice(0, 5), 10), (slice(5, 7), 4), (slice(7, 12), 10)):
        view = mask[row_slice].reshape(-1)
        view[:count] = True
    return mask


def _codec_spec() -> v3.CodecSpecV2:
    return v3.CodecSpecV2(np.zeros(96), np.ones(96))


def _schema_fixture(*, fingerprint_fill: float = 0.0) -> v3.ResidualSchemaTableV3:
    schema = v3.ResidualSchemaTableV3(_codec_spec())
    schema.context_tokens.extend(("ctx-0", "ctx-1"))
    for context in range(2):
        for component in range(4):
            for local in range(3):
                schema.prefix_tokens[(context, component, local)] = (
                    f"pre-{context}-{component}-{local}"
                )
                schema.suffix_tokens[(context, component, local)] = (
                    f"suf-{context}-{component}-{local}"
                )
            for prefix, suffix in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)):
                key = schema.key_index(context, component, prefix, suffix)
                rows = np.arange(12, dtype=np.float64)[:, None]
                coordinates = np.arange(4, dtype=np.float64)[None, :]
                schema.payload[key, :, :4] = (
                    context * 0.01
                    + component * 0.001
                    + prefix * 0.0001
                    + suffix * 0.00001
                    + rows * 0.000001
                    + coordinates * 0.0000001
                )
                schema.payload[key, :, 4:] = fingerprint_fill
                schema.observed[key] = True
                schema.provenance[key] = 1
    assert int(np.sum(schema.observed)) == 48
    return schema


def _candidate_sequences() -> np.ndarray:
    actions = np.asarray(((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    sequences = np.empty((8, 20, 2), dtype=np.float64)
    sequences[0] = actions[0]
    sequences[1] = actions[1]
    sequences[2] = actions[2]
    for lead in range(20):
        sequences[3, lead] = actions[lead % 3]
        sequences[4, lead] = actions[(lead + 1) % 3]
        sequences[5, lead] = actions[(lead + 2) % 3]
    sequences[6] = sequences[3]
    sequences[7] = sequences[4]
    return sequences


def _action_index() -> v3.WakeActionIndexV2:
    actions = np.asarray(((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    values: dict[str, v3.WakeActionValueV3] = {}
    for context in range(2):
        for component in range(4):
            for local in range(3):
                token = f"act-{context}-{component}-{local}"
                values[token] = v3.WakeActionValueV3(
                    f"ctx-{context}",
                    component,
                    actions[local],
                    f"suf-{context}-{component}-{local}",
                    local,
                )
    return v3.WakeActionIndexV2(values)


def _candidate_request(
    *,
    core: v3.CoreModelV2 | None = None,
    schema: v3.ResidualSchemaTableV3 | None = None,
    facade: v3.ScopedEpisodicFacadeV3 | None = None,
    hidden_fill: float = np.nan,
    public_goal: np.ndarray | None = None,
) -> v3.CandidateRequestV2:
    mask = _cue_mask()
    cue_values = np.full((12, 8), hidden_fill, dtype=np.float64)
    cue_values[mask] = 0.0
    cue = v3.PartialCue(
        "ctx-0",
        "pre-0-0-0",
        "suf-0-0-0",
        cue_values,
        mask,
    )
    sequences = _candidate_sequences()
    tokens = np.empty((8, 20), dtype=object)
    actions = np.asarray(((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    for candidate in range(8):
        for lead in range(20):
            local = next(
                index
                for index, action in enumerate(actions)
                if np.array_equal(action, sequences[candidate, lead])
            )
            tokens[candidate, lead] = f"act-0-0-{local}"
    # Same numeric value, but a token from another port/context.
    tokens[6, 6] = "act-0-1-0"
    tokens[7, 12] = "act-1-0-1"
    model = core or v3.CoreModelV2(
        np.zeros(4), np.zeros(4), np.zeros(4), np.zeros((4, 2))
    )
    return v3.CandidateRequestV2(
        cue,
        np.zeros(4),
        sequences,
        tokens,
        np.zeros((20, 4)) if public_goal is None else public_goal,
        v3.CostSpecV2(np.zeros(4), np.ones(4)),
        _codec_spec(),
        model,
        _action_index(),
        _schema_fixture() if schema is None else schema,
        facade,
    )


def _candidate_bytes(result: v3.CandidateResultV2) -> bytes:
    audit = result.origin_recall_audit
    return b"".join(
        (
            result.predictions.tobytes(order="C"),
            result.inferred_valid.tobytes(order="C"),
            result.resolved_schema_keys.tobytes(order="C"),
            result.schema_sources.tobytes(order="C"),
            result.inferred_costs.tobytes(order="C"),
            np.asarray(result.selected_index, dtype=np.int64).tobytes(),
            np.asarray(audit.accepted, dtype=bool).tobytes(),
            np.asarray(audit.identity, dtype=np.int16).tobytes(),
            np.asarray(audit.confidence, dtype=np.float64).tobytes(),
            np.asarray(audit.scope, dtype=np.uint8).tobytes(),
        )
    )


def _scoped_store_fixture() -> tuple[
    v3.PersistentEpisodicStore,
    v3.ScopedRecallIndexV3,
    v3.PartialCue,
    np.ndarray,
]:
    standardizer = v3.CoordinateStandardizer(np.zeros(96), np.ones(96))
    store = v3.PersistentEpisodicStore(
        standardizer, capacity=96, threshold=-1.0
    )
    mask = _cue_mask()
    visible = np.linspace(0.2, 1.2, 24)
    first_in_scope = np.full((12, 8), 2.0)
    first_in_scope[mask] = visible
    # Put the out-of-component exact tie first in physical storage order.
    for component, hidden in (("outside", -8.0), ("inside", 2.0)):
        for index in range(12):
            trajectory = np.full((12, 8), hidden + index * 0.01)
            trajectory[mask] = visible
            store.insert_real(
                v3.EpisodicRecord(
                    f"{component}-{index}",
                    "ctx",
                    f"pre-{component}",
                    f"suf-{component}",
                    trajectory,
                    v3.REAL_PROVENANCE,
                )
            )
    values = np.full((12, 8), np.nan)
    values[mask] = visible
    cue = v3.PartialCue("ctx", "pre-inside", "suf-inside", values, mask)
    return store, v3.ScopedRecallIndexV3.from_store(store), cue, first_in_scope


def test_scoped_recall_blocks_mixed_component_winner_and_never_reads_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, scope, cue, expected = _scoped_store_fixture()

    unscoped = v3.g7m_v2.hard_cue_anchored_recall(store, cue)
    assert unscoped.episode_id == "outside-0"
    result = v3.scoped_hard_recall_v3(store, cue, scope)
    assert result.accepted is True
    assert result.episode_id == "inside-0"
    assert np.array_equal(result.reconstruction[cue.cue_mask], cue.cue_values[cue.cue_mask])
    assert np.array_equal(result.reconstruction[~cue.cue_mask], expected[~cue.cue_mask])

    finite_hidden = np.array(cue.cue_values, copy=True)
    finite_hidden[~cue.cue_mask] = 9.0e200
    repeated = v3.scoped_hard_recall_v3(
        store,
        v3.PartialCue(
            cue.context_token,
            cue.prefix_token,
            cue.suffix_token,
            finite_hidden,
            cue.cue_mask,
        ),
        scope,
    )
    assert repeated.episode_id == result.episode_id
    assert repeated.confidence == result.confidence
    assert np.array_equal(repeated.reconstruction, result.reconstruction)

    calls = 0

    def forbidden_call(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid scope reached inherited distance routine")

    monkeypatch.setattr(v3.g7m_v2, "hard_cue_anchored_recall", forbidden_call)
    invalid = v3.PartialCue(
        "ctx", "pre-inside", "suf-outside", cue.cue_values, cue.cue_mask
    )
    rejected = v3.scoped_hard_recall_v3(store, invalid, scope)
    assert rejected.accepted is False
    assert rejected.episode_id is None
    assert calls == 0


def test_scoped_wrapper_is_inherited_bit_equivalent_on_already_scoped_bank() -> None:
    mixed, _, cue, _ = _scoped_store_fixture()
    scoped = v3.PersistentEpisodicStore(
        mixed.standardizer, capacity=12, threshold=mixed.threshold
    )
    for record in mixed.records[12:]:
        scoped.insert_real(record)
    index = v3.ScopedRecallIndexV3.from_store(scoped)

    inherited = v3.g7m_v2.hard_cue_anchored_recall(scoped, cue)
    wrapped = v3.scoped_hard_recall_v3(scoped, cue, index)
    assert wrapped.accepted is inherited.accepted
    assert wrapped.episode_id == inherited.episode_id
    assert wrapped.confidence == inherited.confidence
    assert wrapped.iterations == inherited.iterations
    assert wrapped.provenance == inherited.provenance
    assert np.array_equal(wrapped.reconstruction, inherited.reconstruction)


def test_residual_dream_ignores_fingerprint_columns_and_accounts_all_pairs() -> None:
    clean = _schema_fixture(fingerprint_fill=0.0)
    poisoned = _schema_fixture(fingerprint_fill=7.25e199)
    clean_observed = clean.payload[clean.observed].copy()
    poisoned_observed = poisoned.payload[poisoned.observed].copy()
    clean_audit, poison_audit = v3.DreamAuditV3(), v3.DreamAuditV3()

    left = v3.constrained_residual_completion_v3(
        clean, 1.0, write_enabled=True, audit=clean_audit
    )
    right = v3.constrained_residual_completion_v3(
        poisoned, 1.0, write_enabled=True, audit=poison_audit
    )

    assert len(left) == len(right) == 24
    assert np.array_equal(clean_audit.pair_check_flags, np.ones(288, dtype=bool))
    assert np.array_equal(clean_audit.pair_reason_codes, poison_audit.pair_reason_codes)
    assert np.count_nonzero(clean_audit.pair_reason_codes == 2) == 216
    assert np.count_nonzero(clean_audit.pair_reason_codes == 3) == 48
    assert np.count_nonzero(clean_audit.pair_reason_codes == 4) == 24
    assert np.array_equal(clean_audit.output_occupancy, np.ones(24, dtype=bool))
    assert np.array_equal(clean_audit.output_provenance, np.ones(24, dtype=np.uint8))
    assert np.all(np.isfinite(clean_audit.endpoint_join_values))
    assert np.array_equal(
        clean_audit.endpoint_join_values, poison_audit.endpoint_join_values
    )
    assert np.array_equal(
        clean_audit.ordered_pair_indices, poison_audit.ordered_pair_indices
    )
    assert int(np.sum(clean.synthetic)) == int(np.sum(poisoned.synthetic)) == 24
    assert np.array_equal(clean.payload[clean.observed], clean_observed)
    assert np.array_equal(poisoned.payload[poisoned.observed], poisoned_observed)
    for first, second in zip(left, right):
        assert first.key == second.key
        assert first.provenance == second.provenance == v3.SYNTHETIC_PROVENANCE
        assert first.provenance.observed is False
        assert first.provenance.recalled is False
        assert np.array_equal(first.standardized_residual, second.standardized_residual)

    clean_view = v3.codec_residual_view_v3(clean.payload[0], clean.codec_spec, standardized=True)
    poison_view = v3.codec_residual_view_v3(
        poisoned.payload[0], poisoned.codec_spec, standardized=True
    )
    assert clean_view.shape == poison_view.shape == (12, 4)
    assert np.array_equal(clean_view, poison_view)


def test_dream_shadow_pass_has_identical_work_and_discards_every_write() -> None:
    active, shadow = _schema_fixture(), _schema_fixture()
    active_audit, shadow_audit = v3.DreamAuditV3(), v3.DreamAuditV3()

    active_output = v3.constrained_residual_completion_v3(
        active, 1.0, write_enabled=True, audit=active_audit
    )
    shadow_output = v3.constrained_residual_completion_v3(
        shadow, 1.0, write_enabled=False, audit=shadow_audit
    )

    assert len(active_output) == len(shadow_output) == 24
    assert int(np.sum(active.synthetic)) == 24
    assert int(np.sum(shadow.synthetic)) == 0
    for field_name in (
        "pair_check_flags",
        "pair_reason_codes",
        "endpoint_join_values",
        "output_occupancy",
        "output_provenance",
        "ordered_pair_indices",
    ):
        assert np.array_equal(
            getattr(active_audit, field_name), getattr(shadow_audit, field_name)
        )


def test_candidate_invalid_tokens_pad_finite_twenty_steps_with_typed_codebooks() -> None:
    request = _candidate_request()
    result = v3.execute_candidate_v3(request)

    assert result.predictions.shape == (8, 20, 4)
    assert result.predictions.dtype == np.float64
    assert np.all(np.isfinite(result.predictions))
    assert result.inferred_valid.dtype == np.dtype(bool)
    assert result.resolved_schema_keys.dtype == np.dtype(np.int16)
    assert result.schema_sources.dtype == np.dtype(np.uint8)
    expected_invalid = np.zeros((8, 20), dtype=bool)
    expected_invalid[6, 6] = True
    expected_invalid[7, 12] = True
    assert np.array_equal(~result.inferred_valid, expected_invalid)
    assert np.array_equal(result.resolved_schema_keys == -1, expected_invalid)
    assert np.array_equal(result.schema_sources == 0, expected_invalid)
    assert np.all((result.schema_sources[~expected_invalid] >= 1))
    assert result.inferred_costs[6] == 10000.0
    assert result.inferred_costs[7] == 10000.0
    assert result.selected_index == np.int64(0)
    assert request.numeric_actions[7, 12].tobytes() == np.asarray(
        (0.0, 1.0), dtype=np.float64
    ).tobytes()
    wrong_context = request.action_index.resolve(request.action_tokens[7, 12])
    assert wrong_context is not None
    assert wrong_context.context_token == "ctx-1"
    assert wrong_context.numeric_action.tobytes() == request.numeric_actions[
        7, 12
    ].tobytes()
    assert result.origin_recall_audit == v3.OriginRecallAuditV2(
        False, np.int16(-1), np.float64(-2.0), np.uint8(0)
    )


def test_candidate_hidden_poison_and_evaluator_only_poison_are_inert() -> None:
    poison_request = _candidate_request(hidden_fill=np.nan)
    finite_request = _candidate_request(hidden_fill=-8.75e250)
    evaluator_only = {
        "future": np.full((8, 20, 4), np.nan),
        "true_q": np.full(4, np.inf),
        "true_schema": np.full((72, 12, 4), -np.inf),
        "validity": np.zeros((8, 20), dtype=bool),
        "outcomes": np.full(8, np.nan),
        "costs": np.full(8, np.inf),
        "seed_stream": object(),
    }

    first = v3.execute_candidate_v3(poison_request)
    for value in evaluator_only.values():
        if isinstance(value, np.ndarray):
            value[...] = 1.23456789e123
    second = v3.execute_candidate_v3(poison_request)
    third = v3.execute_candidate_v3(finite_request)
    assert _candidate_bytes(first) == _candidate_bytes(second)
    assert _candidate_bytes(first) == _candidate_bytes(third)


def test_candidate_presentation_permutation_maps_back_to_same_unique_choice() -> None:
    action_matrix = np.asarray(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    )
    core = v3.CoreModelV2(
        np.zeros(4), np.zeros(4), np.zeros(4), action_matrix
    )
    goal = np.repeat(np.asarray(((1.0, 0.0, 0.0, 0.0),)), 20, axis=0)
    original = _candidate_request(core=core, public_goal=goal)
    baseline = v3.execute_candidate_v3(original)
    assert int(baseline.selected_index) == 2

    permutation = np.asarray((4, 2, 0, 5, 3, 1, 7, 6))
    permuted = v3.CandidateRequestV2(
        original.cue,
        original.anchor_state,
        original.numeric_actions[permutation],
        original.action_tokens[permutation],
        original.public_goal,
        original.cost_spec,
        original.codec_spec,
        original.core,
        original.action_index,
        original.schema,
        original.episodic_store,
    )
    reordered = v3.execute_candidate_v3(permuted)
    assert int(permutation[int(reordered.selected_index)]) == int(
        baseline.selected_index
    )
    inverse = np.argsort(permutation)
    assert np.array_equal(reordered.predictions[inverse], baseline.predictions)
    assert np.array_equal(reordered.inferred_valid[inverse], baseline.inferred_valid)
    assert np.array_equal(
        reordered.resolved_schema_keys[inverse], baseline.resolved_schema_keys
    )
    assert np.array_equal(reordered.schema_sources[inverse], baseline.schema_sources)


def test_all_twenty_four_cell_orders_are_bit_identical_per_cell() -> None:
    labels = ("M00", "M10", "M01", "M11")

    def run_cell(label: str) -> tuple[bytes, str, str]:
        schema = _schema_fixture()
        observed_before = hashlib.sha256(
            np.ascontiguousarray(schema.payload[schema.observed]).tobytes()
        ).hexdigest()
        dream_enabled = label in {"M01", "M11"}
        v3.constrained_residual_completion_v3(
            schema, 1.0, write_enabled=dream_enabled, audit=v3.DreamAuditV3()
        )
        observed_after = hashlib.sha256(
            np.ascontiguousarray(schema.payload[schema.observed]).tobytes()
        ).hexdigest()
        result = v3.execute_candidate_v3(_candidate_request(schema=schema))
        return _candidate_bytes(result), observed_before, observed_after

    expected = {label: run_cell(label) for label in labels}
    for order in itertools.permutations(labels):
        actual = {label: run_cell(label) for label in order}
        assert actual == expected
    assert all(before == after for _, before, after in expected.values())


class _ScriptedRng:
    def __init__(self, stream: int) -> None:
        self.stream = stream
        self.uniform_calls: list[tuple[int, ...]] = []
        self.permutation_calls: list[np.ndarray] = []
        self.normal_calls: list[np.ndarray] = []

    @staticmethod
    def _shape(size: int | tuple[int, ...] | None) -> tuple[int, ...]:
        if size is None:
            return ()
        if isinstance(size, int):
            return (size,)
        return tuple(size)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        shape = self._shape(size)
        call = len(self.uniform_calls)
        self.uniform_calls.append(shape)
        result = np.zeros(shape, dtype=np.float64)
        if self.stream == 13 and call % 5 in (0, 1):
            assert shape == (4,)
            result[1] = 1.0
        return result

    def permutation(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        self.permutation_calls.append(array.copy())
        shift = len(self.permutation_calls) % len(array)
        return np.roll(array, shift)

    def normal(self, *, size: int | tuple[int, ...]) -> np.ndarray:
        shape = self._shape(size)
        call = len(self.normal_calls)
        result = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1000 * call
        self.normal_calls.append(result.copy())
        return result


def test_paired_mask_fresh_noise_chronology_and_cross_port_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rngs = {stream: _ScriptedRng(stream) for stream in (9, 10, 11, 12, 13)}
    monkeypatch.setattr(v3, "_rng", lambda master_seed, stream: rngs[stream])
    fingerprints = np.zeros((2, 4, 6, 2, 4), dtype=np.float64)
    fingerprints[..., 0] = 1.0
    primitives = v3._PrimitivesV3(
        np.zeros((2, 4, 3, 4)),
        np.zeros((2, 4, 4)),
        np.zeros((2, 4, 3, 4)),
        np.zeros((2, 4, 3, 3, 4)),
        np.zeros((2, 4, 6, 4)),
        fingerprints,
    )
    prefixes = np.empty((2, 4, 3), dtype=object)
    suffixes = np.empty((2, 4, 3), dtype=object)
    actions = np.empty((2, 4, 3), dtype=object)
    for context in range(2):
        for port in range(4):
            for local in range(3):
                prefixes[context, port, local] = f"pre-{context}-{port}-{local}"
                suffixes[context, port, local] = f"suf-{context}-{port}-{local}"
                actions[context, port, local] = f"act-{context}-{port}-{local}"
    world = v3.SeedWorldV3(
        -31337,
        (),
        primitives,
        ("ctx-0", "ctx-1"),
        prefixes,
        suffixes,
        actions,
    )
    core = v3.CoreModelV2(
        np.zeros(4),
        np.diag(v3._D),
        np.asarray((0.05, 0.08, -0.07, 0.06)),
        v3._G,
    )
    origins = v3.build_evaluation_cues_v3(world, core, _codec_spec())

    assert len(origins) == 24
    assert len(rngs[11].permutation_calls) == 24 * 3
    assert len(rngs[12].normal_calls) == 24 * 2
    assert len(rngs[10].uniform_calls) == 24
    positive_raw = np.zeros((12, 8), dtype=np.float64)
    positive_raw[:, 4] = 1.0
    lure_raw = np.zeros((12, 8), dtype=np.float64)
    lure_raw[:, 1] = 0.04
    lure_raw[:, 4] = 0.85
    lure_raw[:, 5] = math.sqrt(1.0 - 0.85**2)
    for index, origin in enumerate(origins):
        positive, lure, cross = origin.cue, origin.lure_cue, origin.cross_port_cue
        assert np.array_equal(positive.cue_mask, lure.cue_mask)
        assert [
            int(np.sum(positive.cue_mask[0:5])),
            int(np.sum(positive.cue_mask[5:7])),
            int(np.sum(positive.cue_mask[7:12])),
        ] == [10, 4, 10]
        visible = np.flatnonzero(positive.cue_mask.reshape(-1))
        recovered_positive = (
            positive.cue_values.reshape(-1)[visible]
            - positive_raw.reshape(-1)[visible]
        ) / 0.01
        recovered_lure = (
            lure.cue_values.reshape(-1)[visible] - lure_raw.reshape(-1)[visible]
        ) / 0.01
        np.testing.assert_allclose(
            recovered_positive, rngs[12].normal_calls[2 * index], rtol=0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            recovered_lure,
            rngs[12].normal_calls[2 * index + 1],
            rtol=0.0,
            atol=1e-10,
        )
        assert np.all(np.isnan(positive.cue_values[~positive.cue_mask]))
        assert np.all(np.isnan(lure.cue_values[~lure.cue_mask]))
        assert cross.context_token == positive.context_token
        assert cross.prefix_token == positive.prefix_token
        assert cross.suffix_token != positive.suffix_token
        assert cross.cue_values.tobytes(order="C") == positive.cue_values.tobytes(
            order="C"
        )
        assert cross.cue_mask.tobytes(order="C") == positive.cue_mask.tobytes(
            order="C"
        )


def test_metric_denominators_planning_cost_and_regret_primitives_are_exact() -> None:
    assert v3.metric_denominator_audit_v3() == {
        "E_all_H20": 11520,
        "E_all_H5": 2880,
        "E_uv_H20": 3840,
        "E_uv_H5": 960,
        "E_recall_hidden": 1728,
        "valid_predicted_transitions": 2880,
    }
    cost_spec = v3.CostSpecV2(np.zeros(4), np.full(4, 0.05))
    actions = np.repeat(np.asarray(((-1.0, 0.0),)), 20, axis=0)
    assert v3.planning_cost_v3(
        np.zeros((20, 4)), actions, np.zeros((20, 4)), cost_spec, valid=True
    ) == pytest.approx(0.01, rel=0.0, abs=1e-15)
    assert v3.planning_cost_v3(
        np.zeros((20, 4)), actions, np.zeros((20, 4)), cost_spec, valid=False
    ) == 10000.0
    assert v3.planning_cost_v3(
        np.full((20, 4), 2.0),
        actions,
        np.full((20, 4), -2.0),
        cost_spec,
        valid=True,
    ) == pytest.approx(6400.01, rel=0.0, abs=1e-10)
    true_costs = np.asarray((10.0, 7.0, 8.0, 10000.0))
    selected_index = 2
    optimal_index = int(np.argmin(true_costs))
    regret = true_costs[selected_index] - true_costs[optimal_index]
    assert optimal_index == 1
    assert regret == 1.0
    assert regret >= -1e-12


def test_factorial_signs_student_t_interval_strict_wins_and_ties() -> None:
    cells = {
        "M00": np.asarray((10.0, 11.0, 12.0, 13.0)),
        "M10": np.asarray((8.0, 9.0, 10.0, 11.0)),
        "M01": np.asarray((7.0, 8.0, 9.0, 10.0)),
        "M11": np.asarray((4.0, 5.0, 6.0, 7.0)),
    }
    lower = v3.factorial_effects_v3(cells, lower_is_better=True)
    assert np.array_equal(lower["ltm"], np.full(4, 2.5))
    assert np.array_equal(lower["dream"], np.full(4, 3.5))
    assert np.array_equal(lower["benefit_interaction"], np.full(4, 1.0))

    higher = v3.factorial_effects_v3(
        {name: -value for name, value in cells.items()}, lower_is_better=False
    )
    assert np.array_equal(higher["ltm"], lower["ltm"])
    assert np.array_equal(higher["dream"], lower["dream"])
    assert np.array_equal(higher["benefit_interaction"], lower["benefit_interaction"])

    vector = np.asarray((1.0, 2.0, 0.0, -1.0))
    critical = 2.022690911734728
    interval = v3.paired_interval_v3(vector, critical)
    mean = float(np.mean(vector))
    half = critical * float(np.std(vector, ddof=1)) / math.sqrt(4)
    assert interval == {
        "mean": mean,
        "sample_sd_ddof1": float(np.std(vector, ddof=1)),
        "ci_lower": mean - half,
        "ci_upper": mean + half,
        "strict_win_count": 2,
        "tie_count": 1,
    }
    with pytest.raises(ValueError, match="at least two finite"):
        v3.paired_interval_v3((1.0,), critical)
    with pytest.raises(ValueError, match="at least two finite"):
        v3.paired_interval_v3((1.0, np.nan), critical)


def _synthetic_pass_report(
    *,
    performance: bool = True,
    integrity: bool = True,
    resource: bool = True,
) -> dict[str, dict[str, bool]]:
    return {
        "checks": {key: performance for key in EXPECTED_PERFORMANCE_CHECKS},
        "hard_zero_checks": {key: integrity for key in EXPECTED_HARD_ZERO_CHECKS},
        "resource_checks": {key: resource for key in EXPECTED_RESOURCE_CHECKS},
    }


def test_exact_55_12_17_pass_aggregation_cannot_hide_one_failed_gate() -> None:
    registration = v3.load_merged_registration_v3(CONFIG)
    assert v3.build_split_pass_mapping_v3(
        registration, _synthetic_pass_report()
    ) == {
        "performance_passed": True,
        "integrity_passed": True,
        "resource_passed": True,
        "passed": True,
    }
    for category, expected_key in (
        ("performance", "performance_passed"),
        ("integrity", "integrity_passed"),
        ("resource", "resource_passed"),
    ):
        arguments = {"performance": True, "integrity": True, "resource": True}
        arguments[category] = False
        result = v3.build_split_pass_mapping_v3(
            registration, _synthetic_pass_report(**arguments)
        )
        assert result[expected_key] is False
        assert result["passed"] is False

    missing = _synthetic_pass_report()
    missing["checks"].pop(next(iter(missing["checks"])))
    with pytest.raises(ValueError, match="performance check keyset mismatch"):
        v3.build_split_pass_mapping_v3(registration, missing)
    extra = _synthetic_pass_report()
    extra["resource_checks"]["undeclared"] = True
    with pytest.raises(ValueError, match="resource check keyset mismatch"):
        v3.build_split_pass_mapping_v3(registration, extra)


def test_implementation_lock_has_six_callable_eight_path_and_exact_field_contract() -> None:
    registration = v3.load_merged_registration_v3(CONFIG)
    manifest = registration["implementation_dependency_manifest"]
    payload = v3.prepare_implementation_lock_v3(CONFIG)
    required = registration["implementation_dependency_manifest"][
        "implementation_lock_required_fields"
    ]

    assert set(payload) == set(required)
    assert payload["experiment"] == "agi_world_memory_integration_v3"
    assert payload["stage"] == "implementation_lock"
    assert payload["registration_raw_sha256"] == EXPECTED_V3_REGISTRATION_SHA256
    assert payload["contract_raw_sha256"] == EXPECTED_V3_AMENDMENT_SHA256
    ordered_paths = payload["ordered_path_raw_sha256"]
    assert [item["path"] for item in ordered_paths] == manifest["ordered_source_paths"]
    assert len(ordered_paths) == 8
    for item in ordered_paths:
        assert item["raw_sha256"] == _sha256(ROOT / item["path"])
    callable_hashes = payload["callable_source_sha256_by_symbol"]
    assert [item["symbol"] for item in callable_hashes] == manifest[
        "callable_boundaries"
    ]
    assert len(callable_hashes) == 6
    assert payload["ordered_allocation_ledger_sha256"] == (
        EXPECTED_ALLOCATION_LEDGER_SHA256
    )
    assert tuple(
        (item["name"], item["value"])
        for item in payload["registered_budget_vector"]
    ) == EXPECTED_BUDGET
    assert payload["registered_seed_execution_count"] == 0
    assert payload["handcrafted_test_results"]["registered_seed_opened"] is False
    source_hash_source = inspect.getsource(v3._source_hash)
    assert ".rstrip(" not in source_hash_source
    assert "exactly one terminal LF" in source_hash_source


def test_canonical_writer_is_lf_only_and_refuses_overwrite(tmp_path: Path) -> None:
    value = {"z": [1, 2], "a": {"finite": True}}
    expected = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert v3.canonical_json_bytes_v3(value) == expected
    path = tmp_path / "locked.json"
    v3.write_json_lf_v3(path, value)
    assert path.read_bytes() == expected
    with pytest.raises(FileExistsError):
        v3.write_json_lf_v3(path, value)
    assert path.read_bytes() == expected
    with pytest.raises(ValueError):
        v3.canonical_json_bytes_v3({"bad": np.nan})


def test_preunlock_denies_missing_failed_incomplete_and_uncommitted_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = v3.load_merged_registration_v3(CONFIG)
    config = tmp_path / "experiments/preregistration/unit-only.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"{}\n")
    validation_path = tmp_path / registration["test_lock"]["validation_path"]
    test_path = tmp_path / registration["test_lock"]["test_path"]
    original_read_bytes = Path.read_bytes

    def deny_test_read(path: Path) -> bytes:
        if path.resolve() == test_path.resolve():
            raise AssertionError("pre-unlock guard read the locked test artifact")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", deny_test_read)
    with pytest.raises(PermissionError, match="validation artifact is missing"):
        v3._assert_test_unlocked_v3(config, registration)

    failed = _synthetic_pass_report(performance=False)
    failed.update(
        {
            "performance_passed": False,
            "integrity_passed": True,
            "resource_passed": True,
            "passed": False,
        }
    )
    v3.write_json_lf_v3(validation_path, failed)
    failed_bytes = original_read_bytes(validation_path)
    with pytest.raises(PermissionError, match="self-consistent all-of PASS"):
        v3._assert_test_unlocked_v3(config, registration)
    assert original_read_bytes(validation_path) == failed_bytes
    assert not test_path.exists()

    validation_path.unlink()
    inconsistent = _synthetic_pass_report()
    inconsistent.update(
        {
            "performance_passed": False,
            "integrity_passed": True,
            "resource_passed": True,
            "passed": True,
        }
    )
    v3.write_json_lf_v3(validation_path, inconsistent)
    with pytest.raises(PermissionError, match="self-consistent all-of PASS"):
        v3._assert_test_unlocked_v3(config, registration)

    validation_path.unlink()
    complete = _synthetic_pass_report()
    complete.update(
        {
            "performance_passed": True,
            "integrity_passed": True,
            "resource_passed": True,
            "passed": True,
            "registration_raw_sha256": "a" * 64,
            "implementation_lock_raw_sha256": "b" * 64,
            "calibration_raw_sha256": "c" * 64,
            "ordered_path_raw_sha256": [],
        }
    )
    v3.write_json_lf_v3(validation_path, complete)
    with pytest.raises(PermissionError, match="committed, clean, and HEAD-identical"):
        v3._assert_test_unlocked_v3(config, registration)
    assert not test_path.exists()


def test_scientific_runner_has_no_placeholder_and_checks_unlock_before_test_world() -> None:
    source = inspect.getsource(v3.run_agi_world_memory_integration_v3_gate)
    assert "NotImplementedError" not in source
    assert source.index("_assert_test_unlocked_v3") < source.index(
        'registration["data_roles"]["test"]["seeds"]'
    )
