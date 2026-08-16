from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import episodic_ltm_dream_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments"
    / "preregistration"
    / "episodic_ltm_dream_factorial_v1.json"
)
EXPECTED_REGISTRATION_SHA = (
    "6487156371e4c42877fa0813dd170fb000ce11fe05e51f34bceb74653159fac0"
)
REGISTERED_SEEDS = {
    "train": tuple(range(77100, 77140)),
    "validation": tuple(range(78100, 78140)),
    "test": tuple(range(79100, 79160)),
}
UNIT_SEEDS = (76001, 76002, 76003, 76004)
FACTORIAL_CELLS = {
    "M00": {"persistent_ltm": False, "dream_update": False},
    "M10": {"persistent_ltm": True, "dream_update": False},
    "M01": {"persistent_ltm": False, "dream_update": True},
    "M11": {"persistent_ltm": True, "dream_update": True},
}


def _registration() -> tuple[dict, bytes]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), raw


def _assert_unit_seed(seed: int) -> None:
    assert seed in UNIT_SEEDS
    assert all(seed not in values for values in REGISTERED_SEEDS.values())


def _unit_world(
    seed: int = UNIT_SEEDS[0],
) -> tuple[object, tuple[bridge.EpisodicRecord, ...], bridge.CoordinateStandardizer]:
    _assert_unit_seed(seed)
    world = bridge._generate_seed_world(seed)
    records = world.records_a + world.records_b
    standardizer = bridge.fit_coordinate_standardizer(records)
    return world, records, standardizer


def _record_fingerprint(records: tuple[bridge.EpisodicRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.episode_id.encode())
        digest.update(record.context_token.encode())
        digest.update(record.prefix_token.encode())
        digest.update(record.suffix_token.encode())
        digest.update(np.ascontiguousarray(record.trajectory, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _trajectory_fingerprints(
    bindings: tuple[bridge.DreamBinding, ...],
) -> list[str]:
    return sorted(
        hashlib.sha256(
            np.ascontiguousarray(
                item.standardized_trajectory, dtype=np.float64
            ).tobytes()
        ).hexdigest()
        for item in bindings
    )


@pytest.fixture(autouse=True)
def _forbid_registered_seed_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bridge._generate_seed_world
    registered = set().union(*map(set, REGISTERED_SEEDS.values()))

    def guarded(master_seed: int) -> object:
        if master_seed in registered:
            raise AssertionError("unit tests may not execute a registered seed")
        return original(master_seed)

    monkeypatch.setattr(bridge, "_generate_seed_world", guarded)


def test_registration_lock_and_exact_two_by_two_contract() -> None:
    registration, raw = _registration()

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_REGISTRATION_SHA
    assert registration["status"] == "locked_pre_implementation"
    assert registration["experiment"] == "episodic_ltm_dream_factorial_v1"
    assert registration["roadmap_stage"] == "G7-M"
    assert registration["runner"] == "episodic_ltm_dream_factorial"
    assert registration["standalone"] is True
    assert registration["extends"] is None
    amendment = registration["preregistration_integrity"][
        "preimplementation_clerical_amendment"
    ]
    assert amendment["previous_registration_sha256"] == (
        "2901ee6151d6d12215e5e9025dd0e9720f8c17297fa891b39b437c1629b6b59f"
    )
    assert amendment["registered_seed_results_seen_before_amendment"] is False
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert registration["transport"] == {
        "registration_eol": (
            "LF via experiments/preregistration/.gitattributes"
        ),
        "artifact_eol": "LF via artifacts/agi/.gitattributes",
        "json_writer": "UTF-8 bytes with one terminal LF",
    }
    assert registration["factorial_design"]["cells"] == FACTORIAL_CELLS
    assert registration["factorial_design"]["paired_unit"] == "master_seed"
    assert registration["factorial_design"]["orthogonality"] == {
        "same_generator_queries_order_and_rng_across_cells": True,
        "same_slow_model_and_wake_updates_across_cells": True,
        "same_offline_workspace_and_budget_in_M01_and_M11": True,
        "no_ltm_cells_have_zero_queryable_episode_records_at_evaluation": True,
        "dream_only_workspace_is_deleted_before_evaluation": True,
        "positive_interaction_required": False,
    }
    for role, expected in REGISTERED_SEEDS.items():
        assert tuple(registration["data_roles"][role]["seeds"]) == expected
    assert not (
        set(REGISTERED_SEEDS["train"])
        & set(REGISTERED_SEEDS["validation"])
    )
    assert not (
        set(REGISTERED_SEEDS["train"]) & set(REGISTERED_SEEDS["test"])
    )
    assert not (
        set(REGISTERED_SEEDS["validation"])
        & set(REGISTERED_SEEDS["test"])
    )

    rng = registration["generator"]["rng"]
    assert rng["constructor"] == "numpy.random.SeedSequence([master_seed, stream_id])"
    assert rng["stream_ids"] == {
        "anchors": 0,
        "primitive_interiors": 1,
        "opaque_id_permutations": 2,
        "instance_residuals": 3,
        "bank_order": 4,
        "cue_masks": 5,
        "cue_noise": 6,
        "lures": 7,
        "invalid_queries": 8,
    }
    assert rng["separate_streams"] == list(rng["stream_ids"])
    assert rng["candidate_may_read_master_seed_or_stream_id"] is False
    assert registration["data_roles"]["unit_test_seed_rule"] == (
        "No registered train, validation, or test seed may be executed by unit tests."
    )


def test_registration_locks_memory_dream_provenance_and_execution_separation() -> None:
    registration, _ = _registration()
    memory = registration["long_term_memory"]
    dream = registration["dream_like_offline_recombination"]
    paths = registration["execution_path_separation"]
    resources = registration["resources"]

    assert memory["capacity"] == 96
    assert memory["allowed_records"] == "real wake observations only"
    assert memory["post_interference_items"] == 96
    assert memory["synthetic_or_hypothetical_insertions_allowed"] is False
    assert memory["threshold_calibration"]["split"] == "train_only"
    assert memory["threshold_calibration"]["separate_bank_sizes"] == [48, 96]
    assert memory["threshold_calibration"]["acceptance_rule"] == (
        "Accept if and only if confidence is strictly greater than the "
        "calibrated threshold."
    )
    assert memory["threshold_calibration"]["validation_or_test_recalibration"] is False
    assert resources["persistent_observed_items_M10_M11"] == 96
    assert resources["persistent_observed_items_M00_M01"] == 0
    assert dream["accepted_per_seed"] == 24
    assert dream["offline_update_passes"] == 1
    assert dream["destination"] == "missing slow schema table binding only"
    assert dream["provenance"] == {
        "source": "synthetic",
        "epistemic_status": "hypothetical",
        "may_be_called_observed_or_recalled": False,
    }
    assert "observed_binding_overwrite" in dream["forbidden"]
    assert "PersistentEpisodicStore.insert_real" in dream["forbidden"]
    assert "any_LTM_insertion_or_mutation" in dream["forbidden"]
    assert paths["runner_function"] == "run_episodic_ltm_dream_gate"
    assert paths["recall_path"] == [
        "PersistentEpisodicStore.insert_real",
        "recurrent_clamped_recall",
    ]
    assert paths["dream_path"] == [
        "constrained_missing_binding_dream",
        "update_missing_slow_binding",
    ]
    assert paths["dream_function_must_be_pure_over_store_input"] is True
    assert paths["recall_path_may_call_dream_path"] is False
    assert paths["dream_path_may_call_recurrent_recall"] is False
    assert paths["synthetic_provenance_may_enter_recall_index"] is False

    metrics = registration["metric_definitions"]
    assert metrics["valid_output_coverage"] == (
        "Fraction of novel valid queries with a binding-specific observed or "
        "synthetic table entry. schema_fallback and abstention both count as "
        "uncovered even though schema_fallback is still scored for reconstruction "
        "error."
    )
    assert metrics["hidden_nrmse_when_uncovered"] == (
        "Always score the returned schema_fallback reconstruction on hidden "
        "coordinates so abstention or fallback cannot improve error by omitting "
        "a query."
    )


def test_relative_magnitude_and_absolute_paired_nrmse_contracts_are_distinct() -> None:
    registration, _ = _registration()
    inference = registration["paired_inference"]

    assert inference["absolute_nrmse_effects_for_inference"] == (
        "For each seed, L_s = 0.5*((E_M00-E_M10)+(E_M01-E_M11)) "
        "and D_s = 0.5*((E_M00-E_M01)+(E_M10-E_M11)). CI lower bounds "
        "are computed from these seed-level absolute standardized-NRMSE "
        "contrasts."
    )
    assert inference["mean_nrmse_reduction_for_magnitude_gate"] == (
        "Let bar_E_cell be the arithmetic mean of seed-level NRMSE. L reduction "
        "= 1-(bar_E_M10+bar_E_M11)/(bar_E_M00+bar_E_M01); D reduction = "
        "1-(bar_E_M01+bar_E_M11)/(bar_E_M00+bar_E_M10). The registered "
        "0.35 and 0.30 minima are dimensionless relative fractions under this "
        "formula, while paired CI gates remain absolute contrasts. A zero "
        "denominator is hard invalid."
    )
    assert inference["strict_seed_win"] == (
        "For the dream NRMSE gate, a strict seed win means D_s > 0; zero is a "
        "tie and not a win."
    )
    assert inference["matched_no_dream_degradation"] == (
        "For every registered upper-degradation gate, M01-M00 and M11-M10 are "
        "tested separately with paired seed-level upper confidence bounds; "
        "averaging the two contrasts is forbidden."
    )


def test_registered_gate_runner_is_only_introspected_never_executed() -> None:
    parameters = inspect.signature(bridge.run_episodic_ltm_dream_gate).parameters
    assert tuple(parameters) == ("config_path", "split")
    assert parameters["split"].default == "validation"


def test_candidate_apis_expose_no_evaluator_truth_seed_port_or_hidden_inputs() -> None:
    assert [item.name for item in fields(bridge.PartialCue)] == [
        "context_token",
        "prefix_token",
        "suffix_token",
        "cue_values",
        "cue_mask",
    ]
    recall_parameters = set(inspect.signature(bridge.recurrent_clamped_recall).parameters)
    method_parameters = set(
        inspect.signature(
            bridge.PersistentEpisodicStore.recurrent_clamped_recall
        ).parameters
    )
    dream_parameters = set(
        inspect.signature(bridge.constrained_missing_binding_dream).parameters
    )
    assert recall_parameters == {"store", "cue"}
    assert method_parameters == {"self", "cue"}
    assert dream_parameters == {"real_records", "standardizer", "join_threshold"}
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
        "outcome",
    }
    assert not forbidden & recall_parameters
    assert not forbidden & method_parameters
    assert not forbidden & dream_parameters


def test_rng_streams_are_reproducible_separate_and_all_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = UNIT_SEEDS[1]
    _assert_unit_seed(seed)
    original_rng = bridge._rng
    consumed: list[int] = []

    def watched_rng(master_seed: int, stream_id: int) -> np.random.Generator:
        assert master_seed == seed
        consumed.append(stream_id)
        return original_rng(master_seed, stream_id)

    monkeypatch.setattr(bridge, "_rng", watched_rng)
    first = bridge._generate_seed_world(seed)
    assert consumed == list(range(9))
    monkeypatch.setattr(bridge, "_rng", original_rng)
    second = bridge._generate_seed_world(seed)

    assert _record_fingerprint(first.records_a + first.records_b) == (
        _record_fingerprint(second.records_a + second.records_b)
    )
    draws = [original_rng(seed, stream).normal(size=32) for stream in range(9)]
    repeated = [original_rng(seed, stream).normal(size=32) for stream in range(9)]
    assert all(np.array_equal(left, right) for left, right in zip(draws, repeated))
    assert all(
        not np.array_equal(draws[left], draws[right])
        for left in range(9)
        for right in range(left + 1, 9)
    )


def test_real_only_store_owns_immutable_copies_and_rejects_synthetic() -> None:
    _, records, standardizer = _unit_world()
    store = bridge.PersistentEpisodicStore(standardizer, capacity=96, threshold=-1.0)
    source = records[0]
    expected = source.trajectory.copy()

    store.insert_real(source)
    source.trajectory[:] = 1e9
    assert np.array_equal(store.records[0].trajectory, expected)
    assert store.records[0].trajectory.flags.writeable is False

    synthetic = bridge.EpisodicRecord(
        episode_id="synthetic-attempt",
        context_token=records[1].context_token,
        prefix_token=records[1].prefix_token,
        suffix_token=records[1].suffix_token,
        trajectory=records[1].trajectory.copy(),
        provenance=bridge.SYNTHETIC_PROVENANCE,
    )
    with pytest.raises(ValueError, match="real wake"):
        store.insert_real(synthetic)
    assert len(store.records) == 1
    assert store.synthetic_insert_attempts == 1


def test_full_ltm_has_96_real_records_and_no_ltm_has_zero() -> None:
    _, records, standardizer = _unit_world(UNIT_SEEDS[2])
    with_ltm = bridge.PersistentEpisodicStore(
        standardizer, capacity=96, threshold=-1.0
    )
    without_ltm = bridge.PersistentEpisodicStore(
        standardizer, capacity=96, threshold=-1.0
    )
    for record in records:
        with_ltm.insert_real(record)

    assert len(records) == len(with_ltm.records) == 96
    assert len(without_ltm.records) == 0
    assert all(item.provenance == bridge.REAL_PROVENANCE for item in with_ltm.records)
    assert with_ltm.trace_bytes == 96 * 12 * 8 * np.dtype(np.float64).itemsize
    assert with_ltm.trace_bytes <= 131072


def test_partial_cue_blanks_hidden_values_and_recall_clamps_and_converges() -> None:
    world, records, standardizer = _unit_world()
    store = bridge.PersistentEpisodicStore(standardizer, threshold=-1.0)
    for record in records:
        store.insert_real(record)
    scored = bridge._materialize_query(world.recall_specs[0], standardizer)
    cue = scored.cue

    assert int(np.count_nonzero(cue.cue_mask)) == 24
    assert np.count_nonzero(cue.cue_values[~cue.cue_mask]) == 0
    result = store.recurrent_clamped_recall(cue)
    assert result.accepted is True
    assert result.episode_id == scored.target_episode_id
    assert result.provenance == bridge.RECALLED_PROVENANCE
    assert result.converged is True
    assert result.extra_step_stable is True
    assert 1 <= result.iterations <= 20
    assert result.clamp_max_error <= 1e-12
    assert np.allclose(
        result.reconstruction[cue.cue_mask],
        cue.cue_values[cue.cue_mask],
        rtol=0.0,
        atol=2e-15,
    )

    poisoned_values = cue.cue_values.copy()
    poisoned_values[~cue.cue_mask] = 1e12
    poisoned = bridge.PartialCue(
        cue.context_token,
        cue.prefix_token,
        cue.suffix_token,
        poisoned_values,
        cue.cue_mask.copy(),
    )
    repeated = bridge.recurrent_clamped_recall(store, poisoned)
    assert repeated.episode_id == result.episode_id
    assert repeated.confidence == result.confidence
    assert np.array_equal(repeated.reconstruction, result.reconstruction)

    tied_threshold_store = bridge.PersistentEpisodicStore(
        standardizer, threshold=result.confidence
    )
    for record in records:
        tied_threshold_store.insert_real(record)
    tied = tied_threshold_store.recurrent_clamped_recall(cue)
    assert tied.confidence == result.confidence
    assert tied.accepted is False
    assert tied.episode_id is None

    invalid_query = bridge._materialize_query(world.invalid_specs[0], standardizer)
    invalid = store.recurrent_clamped_recall(invalid_query.cue)
    assert invalid.accepted is False
    assert invalid.episode_id is None
    assert invalid.provenance == bridge.FALLBACK_PROVENANCE

    cross_context = bridge.PartialCue(
        records[-1].context_token,
        cue.prefix_token,
        cue.suffix_token,
        cue.cue_values.copy(),
        cue.cue_mask.copy(),
    )
    rejected = store.recurrent_clamped_recall(cross_context)
    assert rejected.accepted is False
    assert rejected.episode_id is None


def test_components_missing_combinations_and_opaque_relabeling_equivariance() -> None:
    _, records, standardizer = _unit_world(UNIT_SEEDS[3])
    components = bridge.infer_cooccurrence_components(records)
    dreams = bridge.constrained_missing_binding_dream(
        records, standardizer, join_threshold=np.inf
    )
    observed_keys = {
        (item.context_token, item.prefix_token, item.suffix_token)
        for item in records
    }

    assert len(components.prefix_component) == 24
    assert len(components.suffix_component) == 24
    assert len(set(components.prefix_component.values())) == 8
    assert len(dreams) == 24
    assert all(item.provenance == bridge.SYNTHETIC_PROVENANCE for item in dreams)
    assert all(
        components.same_component(
            item.context_token, item.prefix_token, item.suffix_token
        )
        for item in dreams
    )
    assert not {
        (item.context_token, item.prefix_token, item.suffix_token)
        for item in dreams
    } & observed_keys

    prefix_map = {
        token: f"opaque-p-{index:02d}"
        for index, token in enumerate(
            sorted({item.prefix_token for item in records}, reverse=True)
        )
    }
    suffix_map = {
        token: f"opaque-s-{index:02d}"
        for index, token in enumerate(
            sorted({item.suffix_token for item in records}, reverse=True)
        )
    }
    relabeled = tuple(
        bridge.EpisodicRecord(
            item.episode_id,
            item.context_token,
            prefix_map[item.prefix_token],
            suffix_map[item.suffix_token],
            item.trajectory.copy(),
        )
        for item in records
    )
    relabeled_dreams = bridge.constrained_missing_binding_dream(
        relabeled, standardizer, join_threshold=np.inf
    )
    assert len(relabeled_dreams) == len(dreams)
    assert _trajectory_fingerprints(relabeled_dreams) == _trajectory_fingerprints(dreams)


def test_dream_is_pure_hypothetical_and_cannot_cross_or_overwrite() -> None:
    _, records, standardizer = _unit_world()
    before = _record_fingerprint(records)
    table = bridge.SlowSchemaTable(records, standardizer)
    observed_hash = bridge.observed_binding_hash(table)
    dreams = bridge.constrained_missing_binding_dream(
        records, standardizer, join_threshold=np.inf
    )

    assert _record_fingerprint(records) == before
    assert len(dreams) == 24
    assert all(bridge.update_missing_slow_binding(table, item) for item in dreams)
    assert len(table.synthetic_entries) == 24
    assert len(table.observed_entries) == 48
    assert bridge.observed_binding_hash(table) == observed_hash
    assert table.observed_overwrite_count == 0
    assert all(
        item.provenance == bridge.SYNTHETIC_PROVENANCE
        and item.binding_specific
        for item in table.synthetic_entries.values()
    )

    observed = next(iter(table.observed_entries.values()))
    overwrite = bridge.DreamBinding(
        observed.context_token,
        observed.prefix_token,
        observed.suffix_token,
        np.zeros((12, 8), dtype=float),
        0.0,
        0.0,
    )
    assert bridge.update_missing_slow_binding(table, overwrite) is False
    assert bridge.observed_binding_hash(table) == observed_hash

    context = records[0].context_token
    prefixes = sorted(
        key[1] for key in table.components.prefix_component if key[0] == context
    )
    suffixes = sorted(
        key[1] for key in table.components.suffix_component if key[0] == context
    )
    cross = next(
        (prefix, suffix)
        for prefix in prefixes
        for suffix in suffixes
        if not table.components.same_component(context, prefix, suffix)
    )
    invalid = bridge.DreamBinding(
        context,
        cross[0],
        cross[1],
        np.zeros((12, 8), dtype=float),
        0.0,
        0.0,
    )
    assert bridge.update_missing_slow_binding(table, invalid) is False
    assert (context, cross[0], cross[1]) not in table.synthetic_entries


def _four_cell_unit_summary(seed: int) -> dict[str, dict[str, object]]:
    world, records, standardizer = _unit_world(seed)
    recall_query = bridge._materialize_query(world.recall_specs[0], standardizer)
    novel_query = bridge._materialize_query(world.novel_specs[0], standardizer)
    summary: dict[str, dict[str, object]] = {}
    for cell, factors in FACTORIAL_CELLS.items():
        store = bridge.PersistentEpisodicStore(standardizer, threshold=-1.0)
        if factors["persistent_ltm"]:
            for record in records:
                store.insert_real(record)
        table = bridge.SlowSchemaTable(records, standardizer)
        dream_count = 0
        if factors["dream_update"]:
            proposals = bridge.constrained_missing_binding_dream(
                records, standardizer, join_threshold=np.inf
            )
            dream_count = sum(
                bridge.update_missing_slow_binding(table, item)
                for item in proposals
            )
        recall = store.recurrent_clamped_recall(recall_query.cue)
        novel = table.lookup(
            novel_query.cue.context_token,
            novel_query.cue.prefix_token,
            novel_query.cue.suffix_token,
        )
        assert novel is not None
        summary[cell] = {
            "ltm_records": len(store.records),
            "dream_count": dream_count,
            "recall_accepted": recall.accepted,
            "recall_identity": recall.episode_id,
            "recall_provenance": recall.provenance,
            "novel_binding_specific": novel.binding_specific,
            "novel_provenance": novel.provenance,
            "observed_hash": bridge.observed_binding_hash(table),
            "synthetic_insert_attempts": store.synthetic_insert_attempts,
        }
    return summary


@pytest.mark.parametrize("seed", (UNIT_SEEDS[0], UNIT_SEEDS[3]))
def test_four_cell_off_range_unit_integration_is_deterministic(seed: int) -> None:
    first = _four_cell_unit_summary(seed)
    second = _four_cell_unit_summary(seed)
    assert first == second

    assert first["M00"]["ltm_records"] == first["M01"]["ltm_records"] == 0
    assert first["M10"]["ltm_records"] == first["M11"]["ltm_records"] == 96
    assert first["M00"]["dream_count"] == first["M10"]["dream_count"] == 0
    assert first["M01"]["dream_count"] == first["M11"]["dream_count"] == 24
    assert first["M00"]["recall_accepted"] is False
    assert first["M01"]["recall_accepted"] is False
    assert first["M10"]["recall_accepted"] is True
    assert first["M11"]["recall_accepted"] is True
    assert first["M10"]["recall_identity"] == first["M11"]["recall_identity"]
    assert first["M10"]["recall_provenance"] == bridge.RECALLED_PROVENANCE
    assert first["M11"]["recall_provenance"] == bridge.RECALLED_PROVENANCE
    assert first["M00"]["novel_binding_specific"] is False
    assert first["M10"]["novel_binding_specific"] is False
    assert first["M01"]["novel_binding_specific"] is True
    assert first["M11"]["novel_binding_specific"] is True
    assert first["M00"]["novel_provenance"] == bridge.FALLBACK_PROVENANCE
    assert first["M10"]["novel_provenance"] == bridge.FALLBACK_PROVENANCE
    assert first["M01"]["novel_provenance"] == bridge.SYNTHETIC_PROVENANCE
    assert first["M11"]["novel_provenance"] == bridge.SYNTHETIC_PROVENANCE
    assert len({item["observed_hash"] for item in first.values()}) == 1
    assert all(item["synthetic_insert_attempts"] == 0 for item in first.values())


def test_public_four_cell_evaluator_preserves_factor_separation_off_range() -> None:
    registration, _ = _registration()
    train_worlds = [
        bridge._generate_seed_world(seed) for seed in UNIT_SEEDS[:2]
    ]
    calibration = bridge.calibrate_train_worlds(train_worlds)
    seed = UNIT_SEEDS[3]
    first = bridge.evaluate_factorial_seed(seed, calibration, registration)
    second = bridge.evaluate_factorial_seed(seed, calibration, registration)
    assert first == second
    assert set(first) == set(FACTORIAL_CELLS)

    assert first["M00"]["persistent_observed_items"] == 0
    assert first["M01"]["persistent_observed_items"] == 0
    assert first["M10"]["persistent_observed_items"] == 96
    assert first["M11"]["persistent_observed_items"] == 96
    expected_bytes = 96 * 12 * 8 * np.dtype(np.float64).itemsize
    assert first["M00"]["persistent_trace_bytes"] == 0
    assert first["M01"]["persistent_trace_bytes"] == 0
    assert first["M10"]["persistent_trace_bytes"] == expected_bytes
    assert first["M11"]["persistent_trace_bytes"] == expected_bytes
    assert first["M00"]["accepted_synthetic_bindings"] == 0
    assert first["M10"]["accepted_synthetic_bindings"] == 0
    assert first["M01"]["accepted_synthetic_bindings"] == 24
    assert first["M11"]["accepted_synthetic_bindings"] == 24
    assert first["M00"]["valid_output_coverage"] == 0.0
    assert first["M10"]["valid_output_coverage"] == 0.0
    assert first["M01"]["valid_output_coverage"] == 1.0
    assert first["M11"]["valid_output_coverage"] == 1.0
    assert first["M00"]["output_provenance"] == {"schema_fallback": 24}
    assert first["M10"]["output_provenance"] == {"schema_fallback": 24}
    assert first["M01"]["output_provenance"] == {"synthetic": 24}
    assert first["M11"]["output_provenance"] == {"synthetic": 24}

    zero_integrity_metrics = (
        "observed_binding_overwrite_count",
        "observed_binding_hash_change_count",
        "accepted_dream_port_violation_count",
        "accepted_dream_context_violation_count",
        "accepted_dream_join_violation_count",
        "synthetic_to_ltm_insert_count",
        "heldout_target_read_count",
        "nonfinite_metric_or_prediction_count",
    )
    for cell in FACTORIAL_CELLS:
        assert all(first[cell][metric] == 0.0 for metric in zero_integrity_metrics)


def test_pre_recall_table_sees_only_A_before_B_interference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration, _ = _registration()
    train_worlds = [
        bridge._generate_seed_world(seed) for seed in UNIT_SEEDS[:2]
    ]
    calibration = bridge.calibrate_train_worlds(train_worlds)
    original_table = bridge.SlowSchemaTable
    seen: list[tuple[int, tuple[str, ...]]] = []

    class WatchedTable(original_table):
        def __init__(
            self,
            records: tuple[bridge.EpisodicRecord, ...],
            standardizer: bridge.CoordinateStandardizer,
        ) -> None:
            snapshot = tuple(records)
            seen.append(
                (
                    len(snapshot),
                    tuple(sorted({item.context_token for item in snapshot})),
                )
            )
            super().__init__(snapshot, standardizer)

    monkeypatch.setattr(bridge, "SlowSchemaTable", WatchedTable)
    bridge.evaluate_factorial_seed(UNIT_SEEDS[3], calibration, registration)

    assert seen.count((48, ("context-0",))) == 4
    assert seen.count((96, ("context-0", "context-1"))) == 4
    assert len(seen) == 8


def test_novel_router_never_uses_recalled_provenance_even_if_audit_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration, _ = _registration()
    train_worlds = [
        bridge._generate_seed_world(seed) for seed in UNIT_SEEDS[:2]
    ]
    calibration = bridge.calibrate_train_worlds(train_worlds)
    evaluation_world = bridge._generate_seed_world(UNIT_SEEDS[3])
    novel_keys = {
        (item.context_token, item.prefix_token, item.suffix_token)
        for item in evaluation_world.novel_specs
    }
    recalled_keys: list[tuple[str, str, str]] = []

    def forced_false_memory(
        self: bridge.PersistentEpisodicStore,
        cue: bridge.PartialCue,
    ) -> bridge.RecallResult:
        del self
        recalled_keys.append(
            (cue.context_token, cue.prefix_token, cue.suffix_token)
        )
        return bridge.RecallResult(
            accepted=True,
            episode_id="forced-false-memory",
            reconstruction=np.zeros((12, 8), dtype=float),
            confidence=1.0,
            iterations=1,
            converged=True,
            extra_step_stable=True,
            clamp_max_error=0.0,
            provenance=bridge.RECALLED_PROVENANCE,
        )

    monkeypatch.setattr(
        bridge.PersistentEpisodicStore,
        "recurrent_clamped_recall",
        forced_false_memory,
    )
    result = bridge.evaluate_factorial_seed(
        UNIT_SEEDS[3], calibration, registration
    )

    assert not novel_keys & set(recalled_keys)
    assert result["M10"]["novel_valid_tagged_recalled_rate"] == 0.0
    assert result["M11"]["novel_valid_tagged_recalled_rate"] == 0.0
    assert result["M10"]["output_provenance"] == {"schema_fallback": 24}
    assert result["M11"]["output_provenance"] == {"synthetic": 24}


def test_validation_must_pass_before_the_locked_test_can_open() -> None:
    registration, _ = _registration()
    lock = registration["test_lock"]

    assert lock == {
        "open_only_after_validation_all_of_pass": True,
        "validation_artifact": (
            "artifacts/agi/episodic_ltm_dream_factorial_validation_v1.json"
        ),
        "test_artifact": "artifacts/agi/episodic_ltm_dream_factorial_test_v1.json",
        "train_calibration_artifact": (
            "artifacts/agi/episodic_ltm_dream_factorial_train_calibration_v1.json"
        ),
        "train_calibration_artifact_contract": (
            "UTF-8 JSON with one LF; contains frozen mu, sigma, tau_pre, "
            "tau_post, join_threshold, raw registration SHA-256, and "
            "implementation SHA-256. Validation and test must read "
            "byte-identical calibration bytes."
        ),
        "failed_validation_artifact_must_be_preserved": True,
        "require_identical_raw_registration_sha256": True,
        "require_identical_train_calibration_sha256": True,
        "require_identical_implementation_sha256": [
            "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py",
            "examples/agi/episodic_ltm_dream_bridge_gate.py",
        ],
        "test_may_change_generator_model_threshold_metric_or_gate": False,
        "early_test_open_is_hard_invalid": True,
    }
    assert registration["failure_rules"]["validation_fail"] == (
        "Preserve the validation artifact, keep test unopened, and use a new "
        "version with fresh seeds for any change."
    )


def test_test_unlock_rejects_missing_and_failed_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration, raw = _registration()
    temporary_config = (
        tmp_path
        / "experiments"
        / "preregistration"
        / "episodic_ltm_dream_factorial_v1.json"
    )
    temporary_config.parent.mkdir(parents=True)
    temporary_config.write_bytes(raw)
    config_sha = hashlib.sha256(raw).hexdigest()

    with pytest.raises(PermissionError, match="saved passing validation"):
        bridge._assert_test_unlocked(
            temporary_config, registration, config_sha
        )

    validation = tmp_path / registration["test_lock"]["validation_artifact"]
    bridge._write_json_lf(
        validation,
        {
            "experiment": registration["experiment"],
            "split": "validation",
            "passed": False,
        },
    )
    with pytest.raises(PermissionError, match="all-of gate did not pass"):
        bridge._assert_test_unlocked(
            temporary_config, registration, config_sha
        )

    implementation_sha = {"unit-module": "a" * 64}
    monkeypatch.setattr(
        bridge,
        "_implementation_hashes",
        lambda _config_path: implementation_sha,
    )
    calibration = bridge.TrainCalibration(
        bridge.CoordinateStandardizer(
            np.zeros(96, dtype=float), np.ones(96, dtype=float)
        ),
        threshold_pre_48=0.5,
        threshold_post_96=0.6,
        join_threshold=0.7,
        sha256="development-only",
    )
    calibration_path = tmp_path / registration["test_lock"][
        "train_calibration_artifact"
    ]
    bridge._write_json_lf(
        calibration_path,
        bridge._train_calibration_artifact_payload(
            calibration, config_sha, implementation_sha
        ),
    )
    calibration_raw = calibration_path.read_bytes()
    calibration_sha = hashlib.sha256(calibration_raw).hexdigest()
    bridge._write_json_lf(
        validation,
        {
            "experiment": registration["experiment"],
            "split": "validation",
            "passed": True,
            "registration_sha256": config_sha,
            "implementation_sha256": implementation_sha,
            "train_calibration_sha256": calibration_sha,
        },
    )
    assert (
        bridge._assert_test_unlocked(
            temporary_config, registration, config_sha
        )
        == calibration_sha
    )

    calibration_path.write_bytes(calibration_raw + b"\n")
    with pytest.raises(PermissionError, match="calibration bytes changed"):
        bridge._assert_test_unlocked(
            temporary_config, registration, config_sha
        )


def test_unit_seeds_are_explicitly_outside_every_registered_role() -> None:
    for seed in UNIT_SEEDS:
        _assert_unit_seed(seed)


def test_lure_threshold_selection_is_train_only_strict_and_deterministic() -> None:
    parameters = set(inspect.signature(bridge.calibrate_train_worlds).parameters)
    assert parameters == {"worlds"}
    assert not {
        "validation",
        "test",
        "outcomes",
        "split",
        "target_window",
    } & parameters

    threshold = bridge._select_threshold(
        positive_confidence=[0.9, 0.8, 0.7],
        positive_identity_correct=[True, True, False],
        lure_confidence=[0.85, 0.6, 0.2],
    )
    assert threshold == pytest.approx(0.85, rel=0.0, abs=0.0)
    assert np.mean(np.asarray([0.85, 0.6, 0.2]) > threshold) == 0.0
    assert not 0.85 > threshold

    worlds = [bridge._generate_seed_world(seed) for seed in UNIT_SEEDS[:2]]
    first = bridge.calibrate_train_worlds(worlds)
    second = bridge.calibrate_train_worlds(worlds)
    assert first.sha256 == second.sha256
    assert np.array_equal(first.standardizer.mean, second.standardizer.mean)
    assert np.array_equal(first.standardizer.scale, second.standardizer.scale)
    assert first.threshold_pre_48 == second.threshold_pre_48
    assert first.threshold_post_96 == second.threshold_post_96
    assert first.join_threshold == second.join_threshold
    assert np.all(np.isfinite(first.standardizer.mean))
    assert np.all(np.isfinite(first.standardizer.scale))
    assert np.all(first.standardizer.scale >= 1e-8)
    assert np.isfinite(first.join_threshold)


def test_train_calibration_artifact_is_lf_locked_and_round_trips_bytes(
    tmp_path: Path,
) -> None:
    worlds = [bridge._generate_seed_world(seed) for seed in UNIT_SEEDS[:2]]
    calibration = bridge.calibrate_train_worlds(worlds)
    implementation_sha = {"unit-module": "b" * 64}
    payload = bridge._train_calibration_artifact_payload(
        calibration, EXPECTED_REGISTRATION_SHA, implementation_sha
    )
    assert payload["source_split"] == "train_only"
    assert payload["registration_sha256"] == EXPECTED_REGISTRATION_SHA
    assert payload["implementation_sha256"] == implementation_sha
    assert set(payload) == {
        "schema_version",
        "experiment",
        "source_split",
        "mu",
        "sigma",
        "tau_pre",
        "tau_post",
        "join_threshold",
        "registration_sha256",
        "implementation_sha256",
    }

    path = tmp_path / "calibration.json"
    bridge._write_json_lf(path, payload)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r\n" not in raw
    restored = bridge._calibration_from_artifact(raw)
    assert restored.sha256 == hashlib.sha256(raw).hexdigest()
    assert np.array_equal(restored.standardizer.mean, calibration.standardizer.mean)
    assert np.array_equal(restored.standardizer.scale, calibration.standardizer.scale)
    assert restored.threshold_pre_48 == calibration.threshold_pre_48
    assert restored.threshold_post_96 == calibration.threshold_post_96
    assert restored.join_threshold == calibration.join_threshold
