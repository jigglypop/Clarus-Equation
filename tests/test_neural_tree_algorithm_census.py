from __future__ import annotations

import json
from pathlib import Path

import pytest

from reality_stone.clarus.neural_tree_algorithm_census import (
    CENSUS_LOCKED_STATUS,
    PARTIAL,
    TESTABLE,
    UNAVAILABLE,
    load_neural_tree_algorithm_census,
)


CENSUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "neural_tree_algorithm_census_v1.json"
)


def _payload() -> dict:
    return json.loads(CENSUS_PATH.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "census.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_census_closes_a_finite_behavioral_hypothesis_universe() -> None:
    census = load_neural_tree_algorithm_census(CENSUS_PATH)

    assert census.method_status == CENSUS_LOCKED_STATUS
    assert len(census.families) == 18
    assert len(census.families_with_status(TESTABLE)) == 9
    assert len(census.families_with_status(PARTIAL)) == 4
    assert len(census.families_with_status(UNAVAILABLE)) == 5
    assert {
        item.equivalence_class for item in census.families
    } == {
        "state_space_routing",
        "latent_temporal_segmentation",
        "control_semantics",
        "recursive_memory",
        "relational_tree",
    }


def test_round_one_and_unavailable_branches_are_separated() -> None:
    census = load_neural_tree_algorithm_census(CENSUS_PATH)

    assert census.family("cart_m5_model_tree").status == TESTABLE
    assert census.family("oblique_model_tree").current_round == "ROUND_1"
    assert (
        census.family("tree_structured_rslds").status
        == TESTABLE
    )
    assert census.family("behavior_tree").status == UNAVAILABLE
    assert census.family("pushdown_call_return").status == UNAVAILABLE
    assert all(
        census.family(family_id).status != UNAVAILABLE
        for family_id in census.screening_order
    )


def test_snapshot_and_claim_locks_forbid_false_tree_inference() -> None:
    census = load_neural_tree_algorithm_census(CENSUS_PATH)
    snapshot = census.snapshot_constraints

    assert snapshot.allowed_dimensions_one_based == (1, 3)
    assert not snapshot.simultaneous_403_neuron_population
    assert not snapshot.dimension_rows_are_paired_trials
    assert snapshot.primary_nonoverlap_stride_bins == 10
    assert (
        "tree_shaped_predictor_is_not_a_biological_tree"
        in census.claim_locks
    )
    assert (
        "processed_time_bins_do_not_measure_latency_or_throughput"
        in census.claim_locks
    )
    assert (
        "screening_winner_is_not_confirmed_on_independent_data"
        in census.claim_locks
    )


def test_loader_rejects_removing_a_family(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["families"] = payload["families"][:-1]

    with pytest.raises(ValueError, match="finite family universe changed"):
        load_neural_tree_algorithm_census(
            _write_payload(tmp_path, payload)
        )


def test_loader_rejects_simultaneous_pseudopopulation(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["snapshot_constraints"][
        "simultaneous_403_neuron_population"
    ] = True

    with pytest.raises(ValueError, match="not simultaneous"):
        load_neural_tree_algorithm_census(
            _write_payload(tmp_path, payload)
        )


def test_loader_rejects_paired_dimension_rows(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["snapshot_constraints"][
        "dimension_rows_are_paired_trials"
    ] = True

    with pytest.raises(ValueError, match="must remain unpaired"):
        load_neural_tree_algorithm_census(
            _write_payload(tmp_path, payload)
        )


def test_loader_rejects_unavailable_family_in_screening_order(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["screening_order"].append("behavior_tree")

    with pytest.raises(ValueError, match="unavailable families"):
        load_neural_tree_algorithm_census(
            _write_payload(tmp_path, payload)
        )


def test_loader_rejects_claim_lock_removal(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["claim_locks"].remove(
        "fixed_depth_hierarchy_is_not_recursion"
    )

    with pytest.raises(ValueError, match="claim locks changed"):
        load_neural_tree_algorithm_census(
            _write_payload(tmp_path, payload)
        )
