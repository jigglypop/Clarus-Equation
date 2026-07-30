from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus.neural_code_reverse_engineering import (
    CODE_SKELETON_ONLY_STATUS,
    NO,
    REVERSE_ENGINEERING_SCOPE,
    SCHEMA_VERSION,
    TEST_UNAVAILABLE,
    YES,
    ClassifierSnapshotSpec,
    PublishedDecoderMetric,
    evaluate_neural_code_reverse_engineering,
    load_neural_code_reverse_engineering_manifest,
    summarize_tafazoli_classifier_snapshot,
    verify_report_internal_consistency,
)


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "neural_code_reverse_engineering_v1.json"
)


@pytest.fixture
def manifest():
    return load_neural_code_reverse_engineering_manifest(MANIFEST)


def _object_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.empty(2, dtype=object)
    result[0] = first
    result[1] = second
    return result


def _synthetic_snapshot_report():
    spec = ClassifierSnapshotSpec(
        filename="synthetic.mat",
        md5="0" * 32,
        expected_neuron_count=4,
        expected_timepoint_count=5,
        expected_time_step_seconds=0.01,
        temporal_count_window_seconds=0.1,
        expected_session_group_count=2,
        expected_animals=("A", "B"),
        primary_discovery_dimensions_one_based=(1, 3),
        excluded_discovery_dimensions_one_based=(2,),
    )
    dimensions = np.empty(3, dtype=object)
    rng = np.random.default_rng(7)
    for index in range(3):
        train = rng.poisson(2.0, size=(8, 4, 5)).astype(np.float64)
        test = rng.poisson(2.0, size=(2, 4, 5)).astype(np.float64)
        dimensions[index] = _object_pair(train, test)

    first_signature = _object_pair(
        np.asarray([1, 2, 3]),
        np.asarray([4, 5, 6]),
    )
    second_signature = _object_pair(
        np.asarray([11, 12, 13]),
        np.asarray([14, 15, 16]),
    )
    train_stim_indices = np.empty(4, dtype=object)
    train_stim_indices[0] = first_signature
    train_stim_indices[1] = first_signature.copy()
    train_stim_indices[2] = second_signature
    train_stim_indices[3] = second_signature.copy()
    options = {
        "Dimpredictors": dimensions,
        "TrainStimInds": train_stim_indices,
        "IncludedNeu4Ana_Animal": np.asarray(["A", "A", "B", "B"]),
        "TargetFactors": np.asarray(["ResponseLoc", "Rule"], dtype=object),
        "TargetFactors_2ndD": np.asarray(
            ["ColorCat", "Rule"],
            dtype=object,
        ),
        "TargetFactors_3ndD": np.asarray(
            ["ColorCat", "Rule"],
            dtype=object,
        ),
    }
    return summarize_tafazoli_classifier_snapshot(
        spec,
        options,
        np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02]),
        observed_md5="0" * 32,
    )


def _passing_decoder_metrics() -> tuple[PublishedDecoderMetric, ...]:
    return (
        PublishedDecoderMetric(
            name="cross_color_C1_to_C2",
            peak_accuracy=0.72,
            post_event_mean_accuracy=0.60,
        ),
        PublishedDecoderMetric(
            name="cross_color_C2_to_C1",
            peak_accuracy=0.69,
            post_event_mean_accuracy=0.58,
        ),
        PublishedDecoderMetric(
            name="cross_response_C1_to_S1",
            peak_accuracy=0.96,
            post_event_mean_accuracy=0.80,
        ),
        PublishedDecoderMetric(
            name="cross_response_S1_to_C1",
            peak_accuracy=0.95,
            post_event_mean_accuracy=0.79,
        ),
    )


def test_manifest_declares_unique_missing_s2_cell(manifest) -> None:
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.scope == REVERSE_ENGINEERING_SCOPE
    assert [item.task for item in manifest.task_grid.observed_tasks] == [
        "S1",
        "C1",
        "C2",
    ]
    missing = manifest.task_grid.predicted_missing_task
    assert missing.task == "S2"
    assert missing.program == ("READ_SHAPE", "ROUTE_AXIS_2")
    assert not missing.observed
    assert not manifest.current_capabilities[
        "simultaneous_403_neuron_population"
    ]
    assert not manifest.current_capabilities["unseen_composition_recorded"]
    assert manifest.identification_requirements[
        "multi_area_local_dsl_vs_global_comparison"
    ]
    assert manifest.identification_requirements[
        "minimal_sufficient_region_recruitment_curve"
    ]
    assert manifest.identification_requirements[
        "optimizer_timescale_intervention"
    ]


def test_snapshot_recovery_respects_session_boundaries_and_dim2_lock() -> None:
    report = _synthetic_snapshot_report()

    assert report.checksum_matches
    assert not report.full_pseudopopulation_is_simultaneous
    assert report.session_groups_recoverable
    assert [item.neuron_count for item in report.session_groups] == [2, 2]
    assert [item.animal for item in report.session_groups] == ["A", "B"]
    assert report.adjacent_window_overlap_fraction == pytest.approx(0.9)
    assert report.session_local_operator_pilot_possible
    assert not report.full_neural_language_inverse_problem_possible
    assert report.dimensions[0].primary_discovery_allowed
    assert not report.dimensions[1].primary_discovery_allowed
    assert report.dimensions[1].exclusion_reason is not None
    assert report.dimensions[2].primary_discovery_allowed


def test_claims_are_explicit_yes_no_or_unavailable(manifest) -> None:
    report = evaluate_neural_code_reverse_engineering(
        manifest,
        _synthetic_snapshot_report(),
        _passing_decoder_metrics(),
        processed_artifact_integrity_passed=True,
    )

    assert report.method_status == CODE_SKELETON_ONLY_STATUS
    assert report.published_cross_task_decoder_artifact_pass
    assert report.competing_family_winner == "nonidentifiable"
    assert (
        report.claim(
            "task_design_two_slot_code_skeleton_reconstructed"
        ).answer
        == YES
    )
    assert (
        report.claim(
            "shared_population_transition_primitive_identified"
        ).answer
        == NO
    )
    assert (
        report.claim(
            "shared_interface_frontend_backend_candidate_supported"
        ).answer
        == YES
    )
    assert report.claim("common_callee_assembly_identified").answer == NO
    assert (
        report.claim(
            "hierarchical_inheritance_operator_identified"
        ).answer
        == NO
    )
    assert (
        report.claim("common_callee_or_hierarchy_refuted").answer
        == TEST_UNAVAILABLE
    )
    assert report.claim("fixed_neuron_opcode_identified").answer == NO
    assert report.claim("continuous_dynamics_ruled_out").answer == NO
    assert any(
        "matched stationary VAR" in basis
        for basis in report.claim("continuous_dynamics_ruled_out").basis
    )
    assert (
        report.claim(
            "neural_language_architecture_type_identified"
        ).answer
        == NO
    )
    assert report.claim("optimizer_mechanism_identified").answer == NO
    assert (
        report.claim("monotonic_more_regions_worse_supported").answer
        == NO
    )
    assert (
        report.claim(
            "minimal_sufficient_multi_area_circuit_test_available"
        ).answer
        == TEST_UNAVAILABLE
    )
    assert (
        report.claim("unseen_composition_validated").answer
        == TEST_UNAVAILABLE
    )
    assert (
        report.claim("brain_programming_language_identified").answer == NO
    )
    assert (
        report.claim("brain_programming_language_exists").answer
        == TEST_UNAVAILABLE
    )
    verify_report_internal_consistency(report)


def test_published_decoder_sanity_gate_cannot_unlock_language(manifest) -> None:
    metrics = list(_passing_decoder_metrics())
    metrics[0] = PublishedDecoderMetric(
        name=metrics[0].name,
        peak_accuracy=0.60,
        post_event_mean_accuracy=0.60,
    )

    with pytest.raises(ValueError, match="decoder sanity gate"):
        evaluate_neural_code_reverse_engineering(
            manifest,
            _synthetic_snapshot_report(),
            tuple(metrics),
            processed_artifact_integrity_passed=True,
        )


def test_loader_rejects_turning_403_columns_into_simultaneous_data(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["current_capabilities"][
        "simultaneous_403_neuron_population"
    ] = True
    path = tmp_path / "bad-capability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="capability locks"):
        load_neural_code_reverse_engineering_manifest(path)


def test_loader_rejects_a_nonmissing_s2_prediction(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["task_grid"]["predicted_missing_task"]["route_primitive"] = (
        "ROUTE_AXIS_1"
    )
    path = tmp_path / "bad-grid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique unobserved grid cell"):
        load_neural_code_reverse_engineering_manifest(path)
