from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import episodic_ltm_dream_bridge as v1
from reality_stone.clarus import episodic_ltm_dream_bridge_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments"
    / "preregistration"
    / "episodic_ltm_dream_factorial_v2.json"
)
EXPECTED_REGISTRATION_SHA = (
    "973e90111ee98862a5c9ffc3f86509b46ee4e263b5a977e7e1504e00109092b9"
)
EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA = (
    "b0bc0d0506f6d33894df072d15b744fea6d7881f7c3425a32a4a95efa3cc3ca2"
)
EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA = (
    "90bd7c522ba7cb686102e61d218fcf089f4ab0eb2acbfda59bb42f9336aacf60"
)
V1_DEPENDENCY_SHA256 = {
    "experiments/preregistration/episodic_ltm_dream_factorial_v1.json": (
        "6487156371e4c42877fa0813dd170fb000ce11fe05e51f34bceb74653159fac0"
    ),
    "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py": (
        "fd93755c854384448bf69660a341bbb621da2ade3f7179977f7225579549bc39"
    ),
    "examples/agi/episodic_ltm_dream_bridge_gate.py": (
        "441e295409f65dfaa497a8314c0a91237aebed959f1f154a0e86dccd94839a4a"
    ),
}
V1_HISTORICAL_ARTIFACT_SHA256 = {
    "artifacts/agi/episodic_ltm_dream_factorial_train_calibration_v1.json": (
        "9ab33380e47feaadc608ff147cb012763d1979e114afda7caaf87059f72be8cd"
    ),
    "artifacts/agi/episodic_ltm_dream_factorial_validation_v1.json": (
        "7b3bbf75349ae651fbed1211f15c1b4b26fa102ebdb7053876c12ab63525bf79"
    ),
}
V1_REGISTERED_SEEDS = {
    "train": tuple(range(77100, 77140)),
    "validation": tuple(range(78100, 78140)),
    "test": tuple(range(79100, 79160)),
}
V2_REGISTERED_SEEDS = {
    "train": tuple(range(80100, 80140)),
    "validation": tuple(range(81100, 81140)),
    "test": tuple(range(82100, 82160)),
}
UNIT_SEEDS = (76001, 76002, 76003, 76004)
FACTORIAL_CELLS = {
    "M00": {"persistent_ltm": False, "dream_update": False},
    "M10": {"persistent_ltm": True, "dream_update": False},
    "M01": {"persistent_ltm": False, "dream_update": True},
    "M11": {"persistent_ltm": True, "dream_update": True},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registration() -> tuple[dict, bytes]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), raw


def _assert_unit_seed(seed: int) -> None:
    assert seed in UNIT_SEEDS
    roles = (*V1_REGISTERED_SEEDS.values(), *V2_REGISTERED_SEEDS.values())
    assert all(seed not in values for values in roles)


def _assert_recall_results_exactly_equal(
    left: v1.RecallResult,
    right: v1.RecallResult,
) -> None:
    assert left.accepted is right.accepted
    assert left.episode_id == right.episode_id
    assert left.confidence == right.confidence
    assert left.iterations == right.iterations
    assert left.converged is right.converged
    assert left.extra_step_stable is right.extra_step_stable
    assert left.clamp_max_error == right.clamp_max_error
    assert left.provenance == right.provenance
    assert np.array_equal(left.reconstruction, right.reconstruction)


@pytest.fixture(autouse=True)
def _forbid_every_registered_seed_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = set().union(
        *(set(values) for values in V1_REGISTERED_SEEDS.values()),
        *(set(values) for values in V2_REGISTERED_SEEDS.values()),
    )
    original_v1 = v1._generate_seed_world
    original_v2 = v2._generate_seed_world

    def guarded_v1(master_seed: int) -> object:
        if master_seed in registered:
            raise AssertionError("unit tests may not execute a registered seed")
        return original_v1(master_seed)

    def guarded_v2(master_seed: int) -> object:
        if master_seed in registered:
            raise AssertionError("unit tests may not execute a registered seed")
        return original_v2(master_seed)

    monkeypatch.setattr(v1, "_generate_seed_world", guarded_v1)
    monkeypatch.setattr(v2, "_generate_seed_world", guarded_v2)


def test_v2_registration_raw_lock_fresh_roles_and_factorial_contract() -> None:
    registration, raw = _registration()

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_REGISTRATION_SHA
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert registration["schema_version"] == 2
    assert registration["status"] == "locked_pre_implementation"
    assert registration["experiment"] == (
        "episodic_ltm_dream_factorial_v2_hard_reinstatement"
    )
    assert registration["roadmap_stage"] == "G7-M/V2"
    assert registration["runner"] == "episodic_ltm_dream_factorial_v2"
    assert registration["standalone"] is True
    assert registration["extends"] is None
    assert registration["factorial_design"]["cells"] == FACTORIAL_CELLS

    integrity = registration["preregistration_integrity"]
    assert integrity["registered_v2_pilot_executed"] is False
    assert integrity["development_source_is_opened_v1_validation"] is True
    assert integrity["train_seeds_opened"] is False
    assert integrity["validation_seeds_opened"] is False
    assert integrity["test_seeds_opened"] is False
    assert integrity["implementation_must_be_frozen_before_registered_train_calibration"]

    for role, expected in V2_REGISTERED_SEEDS.items():
        assert tuple(registration["data_roles"][role]["seeds"]) == expected
    assert tuple(registration["data_roles"]["development_unit_seeds"]) == UNIT_SEEDS
    all_roles = [set(values) for values in V2_REGISTERED_SEEDS.values()]
    assert all_roles[0].isdisjoint(all_roles[1])
    assert all_roles[0].isdisjoint(all_roles[2])
    assert all_roles[1].isdisjoint(all_roles[2])
    assert set().union(*all_roles).isdisjoint(
        set().union(*(set(values) for values in V1_REGISTERED_SEEDS.values()))
    )
    assert registration["data_roles"]["unit_test_seed_rule"] == (
        "No registered V1 or V2 train, validation, or test seed may be "
        "executed by unit tests."
    )


def test_v1_dependencies_and_failed_validation_are_immutable_historical_only() -> None:
    registration, _ = _registration()
    boundary = registration["historical_boundary"]["v1_registration"]

    assert boundary["validation_result"] == "FAIL"
    assert boundary["test_opened"] is False
    assert boundary["is_parent"] is False
    assert boundary["is_fresh_evidence"] is False
    assert boundary["is_historical_rationale"] is True
    assert boundary["is_frozen_fresh_seed_comparator_specification"] is True
    assert boundary["registration_sha256"] == V1_DEPENDENCY_SHA256[
        "experiments/preregistration/episodic_ltm_dream_factorial_v1.json"
    ]
    assert boundary["module_sha256"] == V1_DEPENDENCY_SHA256[
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py"
    ]
    assert boundary["runner_sha256"] == V1_DEPENDENCY_SHA256[
        "examples/agi/episodic_ltm_dream_bridge_gate.py"
    ]
    assert boundary["train_calibration_artifact_sha256"] == (
        V1_HISTORICAL_ARTIFACT_SHA256[
            "artifacts/agi/episodic_ltm_dream_factorial_train_calibration_v1.json"
        ]
    )
    assert boundary["validation_artifact_sha256"] == (
        V1_HISTORICAL_ARTIFACT_SHA256[
            "artifacts/agi/episodic_ltm_dream_factorial_validation_v1.json"
        ]
    )
    for relative, expected in {
        **V1_DEPENDENCY_SHA256,
        **V1_HISTORICAL_ARTIFACT_SHA256,
    }.items():
        assert _sha256(ROOT / relative) == expected

    assert not (
        ROOT / "artifacts/agi/episodic_ltm_dream_factorial_test_v1.json"
    ).exists()
    evidence = registration["development_evidence"]
    assert evidence["epistemic_status"] == "DEVELOPMENT_ONLY"
    assert evidence["may_count_as_v2_validation_or_confirmation"] is False
    assert evidence["v1_test_seeds_79100_79159_must_remain_unopened"] is True


def test_hard_reinstatement_and_equivalence_mechanisms_are_fully_locked() -> None:
    registration, _ = _registration()
    hard = registration["long_term_memory"]["hard_cue_anchored_reinstatement"]
    calibration = registration["long_term_memory"]["threshold_calibration"]
    equivalence = registration["implementation_equivalence_before_registered_train"]
    recipe = registration["frozen_v1_soft_comparator"]["equivalence_hash_recipe"]

    assert hard["only_scientific_mechanism_change_from_v1"] is True
    assert hard["name"] == "cue_anchored_single_exemplar_completion"
    assert "numpy.argmax" in hard["winner"]
    assert "first storage index" in hard["winner"]
    assert "strictly greater" in hard["acceptance"]
    assert "coordinates outside M are not read" in hard["standardized_observed_cue"]
    assert hard["not_present"] == [
        "softmax_attention",
        "beta",
        "damping",
        "full_state_rescoring",
        "iterative_trace_mixing",
    ]
    assert calibration["split"] == "train_only"
    assert calibration["separate_bank_sizes"] == [48, 96]
    assert calibration["separate_mechanisms"] == ["V2_hard", "frozen_V1_soft"]
    assert calibration["validation_or_test_recalibration"] is False
    assert "strictly greater" in calibration["acceptance_rule"]
    assert tuple(equivalence["off_range_seeds"]) == UNIT_SEEDS
    assert equivalence["frozen_v1_comparator_must_pass_golden_equivalence"]
    assert equivalence["registered_seed_may_not_be_used_for_equivalence"]
    assert recipe["recipe_identifier"] == "v1_soft_equivalence_v2_recipe_1"
    assert tuple(recipe["off_range_seeds_in_order"]) == UNIT_SEEDS
    assert recipe["threshold_for_equivalence_only"] == -1.0
    assert recipe["result_record_order"] == (
        "seed ascending, stage pre48 then post96, family positive then lure, "
        "query index ascending"
    )
    assert recipe["result_fields"] == [
        "seed",
        "stage",
        "family",
        "query_index",
        "accepted",
        "episode_id_or_null",
        "provenance_source",
        "provenance_epistemic_status",
        "provenance_observed",
        "provenance_recalled",
        "confidence_f64le_hex",
        "iterations",
        "converged",
        "extra_step_stable",
        "clamp_max_error_f64le_hex",
        "returned_reconstruction_f64le_c_order_sha256",
    ]


def test_candidate_and_runner_apis_cannot_receive_evaluator_truth_or_seeds() -> None:
    assert [item.name for item in fields(v2.PartialCue)] == [
        "context_token",
        "prefix_token",
        "suffix_token",
        "cue_values",
        "cue_mask",
    ]
    hard_parameters = set(inspect.signature(v2.hard_cue_anchored_recall).parameters)
    frozen_parameters = set(inspect.signature(v2.frozen_v1_soft_recall).parameters)
    method_parameters = set(
        inspect.signature(v2.PersistentEpisodicStore.hard_cue_anchored_recall).parameters
    )
    assert hard_parameters == {"store", "cue"}
    assert frozen_parameters == {"store", "cue"}
    assert method_parameters == {"self", "cue"}
    forbidden = {
        "target",
        "target_id",
        "episode_target_id",
        "seed",
        "master_seed",
        "stream_id",
        "port",
        "port_label",
        "hidden",
        "truth",
        "validity",
        "outcome",
    }
    assert not forbidden & hard_parameters
    assert not forbidden & frozen_parameters
    assert not forbidden & method_parameters

    runner = inspect.signature(v2.run_episodic_ltm_dream_v2_gate).parameters
    assert tuple(runner) == ("config_path", "split")
    assert runner["split"].default == "validation"


def test_coordinate_standardization_uses_the_registered_one_e_minus_eight_floor() -> None:
    records = tuple(
        v2.EpisodicRecord(
            f"constant-{index}",
            "ctx",
            "pre",
            "suf",
            np.full((12, 8), 3.25, dtype=float),
        )
        for index in range(2)
    )
    standardizer = v2.fit_coordinate_standardizer(records)

    assert np.array_equal(
        standardizer.mean,
        np.full(96, 3.25, dtype=float),
    )
    assert np.array_equal(
        standardizer.scale,
        np.full(96, 1e-8, dtype=float),
    )
    assert np.array_equal(
        standardizer.transform(records[0].trajectory),
        np.zeros((12, 8), dtype=float),
    )


def _tie_store(
    *,
    first_hidden: float = 2.0,
    second_hidden: float = -3.0,
    threshold: float = -math.inf,
) -> tuple[v2.PersistentEpisodicStore, v2.PartialCue, np.ndarray, np.ndarray]:
    standardizer = v2.CoordinateStandardizer(
        np.zeros(96, dtype=float),
        np.ones(96, dtype=float),
    )
    mask = np.zeros((12, 8), dtype=bool)
    mask.reshape(-1)[:24] = True
    first = np.full((12, 8), first_hidden, dtype=float)
    second = np.full((12, 8), second_hidden, dtype=float)
    first[mask] = 1.0
    second[mask] = 1.0
    cue_values = np.full((12, 8), 9.25e15, dtype=float)
    cue_values[mask] = 0.75
    store = v2.PersistentEpisodicStore(standardizer, threshold=threshold)
    store.insert_real(v2.EpisodicRecord("first", "ctx", "pre", "suf", first))
    store.insert_real(v2.EpisodicRecord("second", "ctx", "pre", "suf", second))
    cue = v2.PartialCue("ctx", "pre", "suf", cue_values, mask)
    return store, cue, first, second


def test_hard_recall_exact_hidden_copy_observed_clamp_and_idempotence() -> None:
    store, cue, first, _ = _tie_store()
    result = v2.hard_cue_anchored_recall(store, cue)

    assert result.accepted is True
    assert result.episode_id == "first"
    assert result.confidence == 1.0
    assert result.provenance == v2.RECALLED_PROVENANCE
    assert result.iterations == 1
    assert result.converged is True
    assert result.extra_step_stable is True
    assert result.clamp_max_error == 0.0
    assert np.array_equal(result.reconstruction[cue.cue_mask], cue.cue_values[cue.cue_mask])
    assert np.array_equal(result.reconstruction[~cue.cue_mask], first[~cue.cue_mask])

    repeated = v2.hard_cue_anchored_recall(store, cue)
    _assert_recall_results_exactly_equal(result, repeated)
    completed_cue = v2.PartialCue(
        cue.context_token,
        cue.prefix_token,
        cue.suffix_token,
        result.reconstruction.copy(),
        cue.cue_mask.copy(),
    )
    completed_again = v2.hard_cue_anchored_recall(store, completed_cue)
    _assert_recall_results_exactly_equal(result, completed_again)

    different_hidden = cue.cue_values.copy()
    different_hidden[~cue.cue_mask] = np.nan
    poisoned = v2.PartialCue(
        cue.context_token,
        cue.prefix_token,
        cue.suffix_token,
        different_hidden,
        cue.cue_mask.copy(),
    )
    poison_result = v2.hard_cue_anchored_recall(store, poisoned)
    _assert_recall_results_exactly_equal(result, poison_result)


def test_hard_recall_strict_threshold_and_exact_tie_use_storage_order() -> None:
    accepting_store, cue, _, _ = _tie_store(
        threshold=np.nextafter(1.0, -math.inf)
    )
    accepted = v2.hard_cue_anchored_recall(accepting_store, cue)
    assert accepted.confidence == 1.0
    assert accepted.accepted is True
    assert accepted.episode_id == "first"

    tied_store, tied_cue, _, _ = _tie_store(threshold=1.0)
    tied = v2.hard_cue_anchored_recall(tied_store, tied_cue)
    assert tied.confidence == 1.0
    assert tied.accepted is False
    assert tied.episode_id is None
    assert tied.provenance == v2.FALLBACK_PROVENANCE
    assert np.array_equal(tied.reconstruction, np.zeros((12, 8), dtype=float))

    standardizer = accepting_store.standardizer
    records = tuple(reversed(accepting_store.records))
    reversed_store = v2.PersistentEpisodicStore(standardizer, threshold=-math.inf)
    for record in records:
        reversed_store.insert_real(record)
    reversed_result = v2.hard_cue_anchored_recall(reversed_store, cue)
    assert reversed_result.confidence == 1.0
    assert reversed_result.episode_id == "second"
    assert np.array_equal(
        reversed_result.reconstruction[~cue.cue_mask],
        records[0].trajectory[~cue.cue_mask],
    )


def test_rejected_hard_winner_is_hidden_and_evaluator_scores_schema_fallback() -> None:
    store, cue, first, _ = _tie_store(threshold=1.0)
    rejected = store.hard_cue_anchored_recall(cue)
    assert rejected.accepted is False
    assert rejected.episode_id is None
    assert np.array_equal(rejected.reconstruction, np.zeros((12, 8), dtype=float))

    table = v2.SlowSchemaTable(store.records, store.standardizer)
    query = v1._ScoredQuery(cue=cue, target=first, target_episode_id="first")
    fallback = table.lookup(cue.context_token, cue.prefix_token, cue.suffix_token)
    assert fallback is not None
    assert fallback.provenance == v2.REAL_PROVENANCE
    assert not np.array_equal(
        fallback.standardized_trajectory,
        store.standardizer.transform(rejected.reconstruction),
    )
    metrics = v2._hard_recall_metrics(store, [query], table)
    expected_nrmse = v1._pooled_hidden_nrmse(
        [fallback.standardized_trajectory],
        [store.standardizer.transform(first)],
        [cue.cue_mask],
    )
    assert metrics["identity_accuracy"] == 0.0
    assert metrics["positive_coverage"] == 0.0
    assert metrics["accepted_wrong_rate"] == 0.0
    assert metrics["hidden_nrmse"] == expected_nrmse


def test_frozen_v1_soft_comparator_is_exact_on_off_range_pre_and_post() -> None:
    seed = UNIT_SEEDS[0]
    _assert_unit_seed(seed)
    world = v1._generate_seed_world(seed)
    records = world.records_a + world.records_b
    standardizer = v1.fit_coordinate_standardizer(records)

    for bank in (world.records_a, records):
        store = v2.PersistentEpisodicStore(standardizer, threshold=-1.0)
        for record in bank:
            store.insert_real(record)
        for specification in (world.recall_specs[0], world.lure_specs[0]):
            cue = v1._materialize_query(specification, standardizer).cue
            expected = v1.recurrent_clamped_recall(store, cue)
            actual = v2.frozen_v1_soft_recall(store, cue)
            _assert_recall_results_exactly_equal(expected, actual)


def test_frozen_v1_canonical_equivalence_hash_uses_only_off_range_seeds() -> None:
    registration, _ = _registration()
    digest, payload = v2.frozen_v1_comparator_equivalence(registration)

    assert digest == EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA
    assert digest == v2._canonical_sha256(payload)
    assert payload["schema_version"] == 1
    assert payload["frozen_v1_dependency_hashes"] == V1_DEPENDENCY_SHA256
    assert tuple(payload["off_range_seeds"]) == UNIT_SEEDS
    assert payload["threshold_for_equivalence_only"] == -1.0
    assert len(payload["standardizer_fingerprint"]) == 64
    assert len(payload["result_records"]) == 672
    expected_fields = set(
        registration["frozen_v1_soft_comparator"]["equivalence_hash_recipe"][
            "result_fields"
        ]
    )
    assert all(set(item) == expected_fields for item in payload["result_records"])
    assert payload["result_records"][0]["seed"] == UNIT_SEEDS[0]
    assert payload["result_records"][0]["stage"] == "pre48"
    assert payload["result_records"][0]["family"] == "positive"
    assert payload["result_records"][0]["query_index"] == 0
    assert payload["result_records"][-1]["seed"] == UNIT_SEEDS[-1]
    assert payload["result_records"][-1]["stage"] == "post96"
    assert payload["result_records"][-1]["family"] == "lure"
    assert payload["result_records"][-1]["query_index"] == 47


def test_comprehensive_shared_equivalence_is_prelock_and_off_range_only() -> None:
    registration, _ = _registration()
    digest, report = v2.off_range_shared_equivalence(
        registration,
        comparator_equivalence_sha256=EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA,
    )

    assert digest == EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
    assert digest == v2._canonical_sha256(report)
    assert report["schema_version"] == 1
    assert report["scope"] == "prelock_off_range_comprehensive"
    assert tuple(report["off_range_seeds"]) == UNIT_SEEDS
    assert report["registered_seed_used_for_prelock_equivalence"] == 0
    assert report["standardizer_input_and_values_equal"] is True
    assert report["all_required_equal"] is True
    assert [item["seed"] for item in report["per_seed"]] == list(UNIT_SEEDS)
    equality_fields = {
        "world_and_queries_equal",
        "components_equal",
        "slow_schema_equal",
        "dream_equal",
        "M00_shared_metrics_equal",
        "M01_shared_metrics_equal",
        "M10_nonrecall_metrics_equal",
        "M11_nonrecall_metrics_equal",
    }
    assert all(
        item[field] is True
        for item in report["per_seed"]
        for field in equality_fields
    )


def test_v2_generator_schema_and_dream_are_exact_v1_copies_off_range() -> None:
    seed = UNIT_SEEDS[1]
    _assert_unit_seed(seed)
    left = v1._generate_seed_world(seed)
    right = v2._generate_seed_world(seed)

    for family in ("records_a", "records_b", "canonical_a", "canonical_b"):
        left_items = getattr(left, family)
        right_items = getattr(right, family)
        assert len(left_items) == len(right_items)
        for first, second in zip(left_items, right_items):
            assert first.episode_id == second.episode_id
            assert first.context_token == second.context_token
            assert first.prefix_token == second.prefix_token
            assert first.suffix_token == second.suffix_token
            assert first.provenance == second.provenance
            assert np.array_equal(first.trajectory, second.trajectory)

    for family in ("recall_specs", "novel_specs", "lure_specs", "invalid_specs"):
        left_items = getattr(left, family)
        right_items = getattr(right, family)
        assert len(left_items) == len(right_items)
        for first, second in zip(left_items, right_items):
            assert first.context_token == second.context_token
            assert first.prefix_token == second.prefix_token
            assert first.suffix_token == second.suffix_token
            assert first.target_episode_id == second.target_episode_id
            assert np.array_equal(first.target, second.target)
            assert np.array_equal(first.noise, second.noise)
            assert first.masks.keys() == second.masks.keys()
            for visible in first.masks:
                assert np.array_equal(first.masks[visible], second.masks[visible])

    records = left.records_a + left.records_b
    standardizer = v1.fit_coordinate_standardizer(records)
    v1_table = v1.SlowSchemaTable(records, standardizer)
    v2_table = v2.SlowSchemaTable(records, standardizer)
    assert v1.observed_binding_hash(v1_table) == v2.observed_binding_hash(v2_table)
    v1_dreams = v1.constrained_missing_binding_dream(
        records, standardizer, join_threshold=math.inf
    )
    v2_dreams = v2.constrained_missing_binding_dream(
        records, standardizer, join_threshold=math.inf
    )
    assert len(v1_dreams) == len(v2_dreams) == 24
    for first, second in zip(v1_dreams, v2_dreams):
        assert first.context_token == second.context_token
        assert first.prefix_token == second.prefix_token
        assert first.suffix_token == second.suffix_token
        assert first.left_join_rms == second.left_join_rms
        assert first.right_join_rms == second.right_join_rms
        assert first.provenance == second.provenance == v2.SYNTHETIC_PROVENANCE
        assert first.provenance.epistemic_status == "hypothetical"
        assert first.provenance.recalled is False
        assert np.array_equal(
            first.standardized_trajectory,
            second.standardized_trajectory,
        )


def test_synthetic_dream_content_cannot_enter_real_episodic_memory() -> None:
    seed = UNIT_SEEDS[2]
    _assert_unit_seed(seed)
    world = v2._generate_seed_world(seed)
    records = world.records_a + world.records_b
    standardizer = v2.fit_coordinate_standardizer(records)
    store = v2.PersistentEpisodicStore(standardizer, threshold=-1.0)
    dream = v2.constrained_missing_binding_dream(
        records, standardizer, join_threshold=math.inf
    )[0]
    synthetic = v2.EpisodicRecord(
        "synthetic-attempt",
        dream.context_token,
        dream.prefix_token,
        dream.suffix_token,
        standardizer.inverse(dream.standardized_trajectory),
        provenance=dream.provenance,
    )

    with pytest.raises(ValueError, match="real wake"):
        store.insert_real(synthetic)
    assert not store.records
    assert store.synthetic_insert_attempts == 1
    assert dream.provenance == v2.SYNTHETIC_PROVENANCE
    assert dream.provenance.observed is False
    assert dream.provenance.recalled is False


def test_off_range_factorial_preserves_v1_shared_metrics_and_provenance() -> None:
    seed = UNIT_SEEDS[3]
    _assert_unit_seed(seed)
    registration, _ = _registration()
    world = v2._generate_seed_world(seed)
    calibration = v2.calibrate_train_worlds_v2(
        [world],
        comparator_equivalence_sha256=EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA,
        off_range_shared_equivalence_sha256=(
            EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
        ),
    )
    v2_result = v2.evaluate_factorial_seed_v2(
        seed, calibration, registration
    )
    v1_calibration = v1.TrainCalibration(
        calibration.standardizer,
        calibration.v1_threshold_pre_48,
        calibration.v1_threshold_post_96,
        calibration.join_threshold,
        "off-range-unit-comparison",
    )
    v1_cells = v1.evaluate_factorial_seed(seed, v1_calibration, registration)
    shared = set(v1_cells["M00"]) - v2._SOFT_ONLY_CELL_FIELDS

    for label in ("M00", "M01"):
        for key in shared:
            assert v2_result["cells"][label][key] == v1_cells[label][key]
    assert "implementation_equivalence" not in v2_result
    assert set(v2_result) == {"cells", "v1_soft_comparator"}
    assert {
        label: v2_result["cells"][label]["output_provenance"]
        for label in FACTORIAL_CELLS
    } == {
        "M00": {"schema_fallback": 24},
        "M10": {"schema_fallback": 24},
        "M01": {"synthetic": 24},
        "M11": {"synthetic": 24},
    }
    for label, cell in v2_result["cells"].items():
        assert cell["novel_valid_tagged_recalled_rate"] == 0.0, label
        assert cell["synthetic_to_ltm_insert_count"] == 0.0, label
        assert cell["invalid_query_nonabstain_rate"] == 0.0, label


def _unit_calibration() -> v2.TrainCalibrationV2:
    standardizer = v2.CoordinateStandardizer(
        np.arange(96, dtype=float) / 100.0,
        np.linspace(0.25, 1.25, 96, dtype=float),
    )
    return v2.TrainCalibrationV2(
        standardizer=standardizer,
        v2_threshold_pre_48=0.51,
        v2_threshold_post_96=0.52,
        v1_threshold_pre_48=0.61,
        v1_threshold_post_96=0.62,
        join_threshold=0.17,
        comparator_equivalence_sha256=EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA,
        off_range_shared_equivalence_sha256=(
            EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
        ),
        sha256="unit-construction-only",
    )


def _unit_shared_equivalence_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "prelock_off_range_comprehensive",
        "off_range_seeds": list(UNIT_SEEDS),
        "registered_seed_used_for_prelock_equivalence": 0,
        "standardizer_input_and_values_equal": True,
        "per_seed": [],
        "all_required_equal": True,
    }


def test_calibration_artifact_lf_raw_hash_and_all_locks_round_trip(
    tmp_path: Path,
) -> None:
    registration, _ = _registration()
    calibration = _unit_calibration()
    implementation = {
        "reality_stone/python/reality_stone/clarus/"
        "episodic_ltm_dream_bridge_v2.py": "a" * 64,
        "examples/agi/episodic_ltm_dream_bridge_v2_gate.py": "b" * 64,
    }
    implementation_lock_sha = "c" * 64
    shared_report = _unit_shared_equivalence_report()
    payload = v2._calibration_artifact_payload(
        calibration,
        registration,
        EXPECTED_REGISTRATION_SHA,
        implementation_lock_sha,
        implementation,
        V1_DEPENDENCY_SHA256,
        shared_report,
    )
    assert payload == {
        "schema_version": 2,
        "experiment": registration["experiment"],
        "source_split": "train_only",
        "mu": calibration.standardizer.mean.tolist(),
        "sigma": calibration.standardizer.scale.tolist(),
        "tau_v2_pre": 0.51,
        "tau_v2_post": 0.52,
        "tau_v1_pre": 0.61,
        "tau_v1_post": 0.62,
        "join_threshold": 0.17,
        "registration_sha256": EXPECTED_REGISTRATION_SHA,
        "implementation_lock_artifact_sha256": implementation_lock_sha,
        "implementation_sha256": implementation,
        "immutable_v1_dependency_sha256": V1_DEPENDENCY_SHA256,
        "frozen_v1_comparator_equivalence_sha256": (
            EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA
        ),
        "off_range_shared_equivalence_sha256": (
            EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
        ),
        "off_range_shared_equivalence_report": shared_report,
    }

    path = tmp_path / "calibration.json"
    v2._write_json_lf(path, payload)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r\n" not in raw
    restored = v2._calibration_from_artifact(raw)
    assert restored.sha256 == hashlib.sha256(raw).hexdigest()
    assert restored.comparator_equivalence_sha256 == (
        EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA
    )
    assert restored.off_range_shared_equivalence_sha256 == (
        EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
    )
    assert np.array_equal(restored.standardizer.mean, calibration.standardizer.mean)
    assert np.array_equal(restored.standardizer.scale, calibration.standardizer.scale)
    assert restored.v2_threshold_pre_48 == calibration.v2_threshold_pre_48
    assert restored.v2_threshold_post_96 == calibration.v2_threshold_post_96
    assert restored.v1_threshold_pre_48 == calibration.v1_threshold_pre_48
    assert restored.v1_threshold_post_96 == calibration.v1_threshold_post_96
    assert restored.join_threshold == calibration.join_threshold


def test_test_unlock_rejects_validation_and_calibration_byte_changes(
    tmp_path: Path,
) -> None:
    registration, config_raw = _registration()
    temporary_config = (
        tmp_path
        / "experiments"
        / "preregistration"
        / "episodic_ltm_dream_factorial_v2.json"
    )
    temporary_config.parent.mkdir(parents=True)
    temporary_config.write_bytes(config_raw)
    implementation = {"unit-v2-module": "d" * 64}
    implementation_lock = {
        "implementation_sha256": implementation,
        "immutable_v1_dependency_sha256": V1_DEPENDENCY_SHA256,
        "frozen_v1_comparator_equivalence_sha256": (
            EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA
        ),
        "off_range_shared_equivalence_sha256": (
            EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
        ),
        "off_range_shared_equivalence_report": (
            _unit_shared_equivalence_report()
        ),
    }
    implementation_lock_sha = "e" * 64

    with pytest.raises(PermissionError, match="saved passing validation"):
        v2._assert_test_unlocked(
            temporary_config,
            registration,
            EXPECTED_REGISTRATION_SHA,
            implementation_lock_sha,
            implementation_lock,
        )

    validation_path = tmp_path / registration["test_lock"]["validation_artifact"]
    v2._write_json_lf(
        validation_path,
        {
            "experiment": registration["experiment"],
            "split": "validation",
            "passed": False,
        },
    )
    with pytest.raises(PermissionError, match="all-of gate did not pass"):
        v2._assert_test_unlocked(
            temporary_config,
            registration,
            EXPECTED_REGISTRATION_SHA,
            implementation_lock_sha,
            implementation_lock,
        )

    calibration = _unit_calibration()
    calibration_payload = v2._calibration_artifact_payload(
        calibration,
        registration,
        EXPECTED_REGISTRATION_SHA,
        implementation_lock_sha,
        implementation,
        V1_DEPENDENCY_SHA256,
        _unit_shared_equivalence_report(),
    )
    calibration_path = tmp_path / registration["test_lock"][
        "train_calibration_artifact"
    ]
    v2._write_json_lf(calibration_path, calibration_payload)
    calibration_raw = calibration_path.read_bytes()
    calibration_sha = hashlib.sha256(calibration_raw).hexdigest()
    validation_payload = {
        "experiment": registration["experiment"],
        "split": "validation",
        "passed": True,
        "performance_passed": True,
        "resource_passed": True,
        "checks": {"unit-performance": True},
        "resource_checks": {"unit-resource": True},
        "registration_sha256": EXPECTED_REGISTRATION_SHA,
        "implementation_sha256": implementation,
        "implementation_lock_artifact_sha256": implementation_lock_sha,
        "immutable_v1_dependency_sha256": V1_DEPENDENCY_SHA256,
        "frozen_v1_comparator_equivalence_sha256": (
            EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA
        ),
        "prelock_implementation_equivalence": {
            "off_range_shared_equivalence_sha256": (
                EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
            ),
            "off_range_shared_equivalence_report": (
                _unit_shared_equivalence_report()
            ),
        },
        "train_calibration_sha256": calibration_sha,
    }
    v2._write_json_lf(validation_path, validation_payload)
    expected_validation_sha = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    assert v2._assert_test_unlocked(
        temporary_config,
        registration,
        EXPECTED_REGISTRATION_SHA,
        implementation_lock_sha,
        implementation_lock,
    ) == (calibration_sha, expected_validation_sha)

    inconsistent = dict(validation_payload)
    inconsistent["performance_passed"] = False
    v2._write_json_lf(validation_path, inconsistent)
    with pytest.raises(PermissionError, match="not self-consistent"):
        v2._assert_test_unlocked(
            temporary_config,
            registration,
            EXPECTED_REGISTRATION_SHA,
            implementation_lock_sha,
            implementation_lock,
        )
    v2._write_json_lf(validation_path, validation_payload)

    dependency_tamper = dict(validation_payload)
    dependency_tamper["immutable_v1_dependency_sha256"] = {
        **V1_DEPENDENCY_SHA256,
        "reality_stone/python/reality_stone/clarus/"
        "episodic_ltm_dream_bridge.py": "0" * 64,
    }
    v2._write_json_lf(validation_path, dependency_tamper)
    with pytest.raises(PermissionError, match="dependency lock changed"):
        v2._assert_test_unlocked(
            temporary_config,
            registration,
            EXPECTED_REGISTRATION_SHA,
            implementation_lock_sha,
            implementation_lock,
        )
    v2._write_json_lf(validation_path, validation_payload)

    with pytest.raises(PermissionError, match="implementation lock changed"):
        v2._assert_test_unlocked(
            temporary_config,
            registration,
            EXPECTED_REGISTRATION_SHA,
            "f" * 64,
            implementation_lock,
        )
    calibration_path.write_bytes(calibration_raw + b"\n")
    with pytest.raises(PermissionError, match="train calibration changed"):
        v2._assert_test_unlocked(
            temporary_config,
            registration,
            EXPECTED_REGISTRATION_SHA,
            implementation_lock_sha,
            implementation_lock,
        )


def test_v2_test_lock_declares_all_byte_hash_dependencies() -> None:
    registration, _ = _registration()
    lock = registration["test_lock"]

    assert lock["open_only_after_validation_all_of_pass"] is True
    assert lock["implementation_lock_artifact"] == (
        "artifacts/agi/episodic_ltm_dream_factorial_implementation_lock_v2.json"
    )
    assert lock["validation_artifact"] == (
        "artifacts/agi/episodic_ltm_dream_factorial_validation_v2.json"
    )
    assert lock["test_artifact"] == (
        "artifacts/agi/episodic_ltm_dream_factorial_test_v2.json"
    )
    assert lock["train_calibration_artifact"] == (
        "artifacts/agi/episodic_ltm_dream_factorial_train_calibration_v2.json"
    )
    assert lock["require_identical_raw_registration_sha256"] is True
    assert lock["require_identical_implementation_lock_artifact_sha256"] is True
    assert lock["require_identical_train_calibration_sha256"] is True
    assert lock["require_identical_implementation_sha256"] == [
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge_v2.py",
        "examples/agi/episodic_ltm_dream_bridge_v2_gate.py",
    ]
    assert lock["require_immutable_v1_dependency_sha256"] == V1_DEPENDENCY_SHA256
    assert lock["v1_dependencies_checked_before_any_validation_or_test_seed_generation"]
    assert lock["require_frozen_v1_comparator_equivalence_hash"] is True
    assert lock["early_test_open_is_hard_invalid"] is True
    assert lock["v1_test_79100_79159_remains_forbidden"] is True
    test_path = ROOT / lock["test_artifact"]
    if test_path.exists():
        report = v2._locked_json(test_path.read_bytes(), "G7-M/V2 test artifact")
        validation_path = ROOT / lock["validation_artifact"]
        calibration_path = ROOT / lock["train_calibration_artifact"]
        implementation_lock_path = ROOT / lock["implementation_lock_artifact"]
        assert report["split"] == "test"
        assert report["passed"] is True
        assert report["performance_passed"] is True
        assert report["resource_passed"] is True
        assert all(report["checks"].values())
        assert all(report["resource_checks"].values())
        assert report["test_lock"]["test_opened_after_validation_pass"] is True
        assert report["test_lock"]["validation_artifact_sha256"] == _sha256(
            validation_path
        )
        assert report["train_calibration_sha256"] == _sha256(calibration_path)
        assert report["implementation_lock_artifact_sha256"] == _sha256(
            implementation_lock_path
        )


def test_immutable_v1_dependency_gate_hard_fails_if_v1_test_artifact_exists(
    tmp_path: Path,
) -> None:
    registration, config_raw = _registration()
    temporary_config = (
        tmp_path
        / "experiments"
        / "preregistration"
        / "episodic_ltm_dream_factorial_v2.json"
    )
    temporary_config.parent.mkdir(parents=True)
    temporary_config.write_bytes(config_raw)
    for relative in V1_DEPENDENCY_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())

    assert v2._assert_immutable_v1_dependencies(
        temporary_config, registration
    ) == V1_DEPENDENCY_SHA256
    forbidden = (
        tmp_path
        / "artifacts"
        / "agi"
        / "episodic_ltm_dream_factorial_test_v1.json"
    )
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b'{"opened":true}\n')
    with pytest.raises(PermissionError, match="V1 test artifact"):
        v2._assert_immutable_v1_dependencies(temporary_config, registration)


def test_implementation_lock_persists_comprehensive_prelock_proof(
    tmp_path: Path,
) -> None:
    registration, config_raw = _registration()
    temporary_config = (
        tmp_path
        / "experiments"
        / "preregistration"
        / "episodic_ltm_dream_factorial_v2.json"
    )
    temporary_config.parent.mkdir(parents=True)
    temporary_config.write_bytes(config_raw)
    copied = {
        **V1_DEPENDENCY_SHA256,
        "reality_stone/python/reality_stone/clarus/"
        "episodic_ltm_dream_bridge_v2.py": _sha256(
            ROOT
            / "reality_stone/python/reality_stone/clarus/"
            "episodic_ltm_dream_bridge_v2.py"
        ),
        "examples/agi/episodic_ltm_dream_bridge_v2_gate.py": _sha256(
            ROOT / "examples/agi/episodic_ltm_dream_bridge_v2_gate.py"
        ),
    }
    for relative in copied:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())

    payload, lock_sha = v2._prepare_implementation_lock(
        temporary_config,
        registration,
        EXPECTED_REGISTRATION_SHA,
    )
    lock_path = tmp_path / registration["test_lock"][
        "implementation_lock_artifact"
    ]
    raw = lock_path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r\n" not in raw
    assert lock_sha == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw) == payload
    assert payload["implementation_sha256"] == {
        relative: expected
        for relative, expected in copied.items()
        if relative not in V1_DEPENDENCY_SHA256
    }
    assert payload["immutable_v1_dependency_sha256"] == V1_DEPENDENCY_SHA256
    assert payload["frozen_v1_comparator_equivalence_sha256"] == (
        EXPECTED_OFF_RANGE_V1_EQUIVALENCE_SHA
    )
    assert payload["off_range_shared_equivalence_sha256"] == (
        EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
    )
    report = payload["off_range_shared_equivalence_report"]
    assert v2._canonical_sha256(report) == (
        EXPECTED_OFF_RANGE_SHARED_EQUIVALENCE_SHA
    )
    assert report["scope"] == "prelock_off_range_comprehensive"
    assert report["all_required_equal"] is True
    assert payload["registered_seed_used_for_prelock_equivalence"] == 0


def test_unit_seeds_are_explicitly_outside_every_registered_role() -> None:
    for seed in UNIT_SEEDS:
        _assert_unit_seed(seed)
