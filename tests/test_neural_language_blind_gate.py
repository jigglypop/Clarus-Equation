from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from reality_stone.clarus.neural_language_blind_gate import (
    HIDDEN_ITEMS,
    KNOWN_ITEMS,
    PARTIAL_BLIND_SCOPE,
    PARTIAL_BLIND_SYNTHETIC_AMBIGUOUS,
    PARTIAL_BLIND_SYNTHETIC_PASS,
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    load_neural_language_partial_blind_benchmark,
    neural_language_partial_blind_gate_report,
)


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "neural_language_blind_synthetic_v1.json"
)


@pytest.fixture
def benchmark():
    return load_neural_language_partial_blind_benchmark(BENCHMARK)


def test_strict_benchmark_declares_partial_blind_information_boundary(
    benchmark,
) -> None:
    assert benchmark.schema_version == SCHEMA_VERSION
    assert benchmark.scope == PARTIAL_BLIND_SCOPE
    assert benchmark.hidden_from_inverse == HIDDEN_ITEMS
    assert all(
        getattr(benchmark.known_to_inverse, item)
        for item in KNOWN_ITEMS
    )
    assert not benchmark.real_neural_data_used
    assert not benchmark.full_brain_language_identified
    assert not benchmark.neural_clarus_assembly_validated
    assert not benchmark.causal_instruction_set_validated
    assert not benchmark.fully_blind_inverse_recovery_validated


def test_reference_inverse_passes_only_partial_blind_synthetic_control(
    benchmark,
) -> None:
    report = neural_language_partial_blind_gate_report(benchmark)

    assert report.schema_version == REPORT_SCHEMA_VERSION
    assert report.method_status == PARTIAL_BLIND_SYNTHETIC_PASS
    assert report.partial_blind_synthetic_pass
    assert report.selected_candidate_matches_generator_target
    assert not report.real_neural_data_used
    assert not report.full_brain_language_identified
    assert not report.neural_clarus_assembly_validated
    assert not report.causal_instruction_set_validated
    assert not report.fully_blind_inverse_recovery_validated
    assert "partial-blind synthetic inverse" in report.conclusion


def test_candidate_boundary_is_selected_from_train_context_scores(
    benchmark,
) -> None:
    report = neural_language_partial_blind_gate_report(benchmark)
    scores = {
        item.candidate_group: item
        for item in report.candidate_scores
    }

    assert len(scores) == benchmark.generator.candidate_group_count
    assert not report.selection_abstained
    assert report.selected_candidate_group == max(
        scores,
        key=lambda item: scores[item].mean_train_context_late_accuracy,
    )
    assert report.top_scoring_candidate_group == report.selected_candidate_group
    assert report.top_candidate_train_accuracy >= 0.85
    assert report.top_candidate_margin >= 0.20
    assert report.scoring_only_maximum_distractor_train_accuracy <= 0.65
    assert all(
        len(score.train_context_late_accuracies)
        == benchmark.inference.train_context_count
        for score in scores.values()
    )


def test_heldout_session_uses_early_calibration_and_late_evaluation(
    benchmark,
) -> None:
    report = neural_language_partial_blind_gate_report(benchmark)
    audit = report.heldout_context_audit

    assert audit.heldout_context == benchmark.inference.train_context_count
    assert (
        audit.early_calibration_count + audit.late_evaluation_count
        == benchmark.generator.samples_per_context
    )
    assert audit.clusters_fit_on_early_calibration_only
    assert audit.label_alignment_fit_on_early_calibration_only
    assert audit.late_transition_accuracy_with_alignment >= 0.85
    assert audit.late_latent_state_recovery_accuracy >= 0.90
    assert audit.alignment_over_permutation_null_gain >= 0.20
    assert (
        audit.late_transition_accuracy_with_alignment
        > audit.late_transition_accuracy_permutation_null_mean
    )


def test_ground_truth_is_declared_scoring_only_not_inverse_input(
    benchmark,
) -> None:
    audit = neural_language_partial_blind_gate_report(
        benchmark
    ).information_boundary_audit

    assert audit.known_to_inverse == KNOWN_ITEMS
    assert audit.hidden_from_inverse == HIDDEN_ITEMS
    assert not audit.state_labels_used_for_inference
    assert not audit.generator_target_used_for_selection
    assert audit.ground_truth_used_only_after_inference_for_scoring


def test_every_session_contains_mixing_permutation_dropout_and_noise(
    benchmark,
) -> None:
    audit = neural_language_partial_blind_gate_report(
        benchmark
    ).observation_transformation_audit

    assert audit.context_specific_latent_code_permutation
    assert audit.context_specific_neuron_permutation
    assert audit.context_specific_linear_mixing
    assert audit.context_specific_neuron_dropout
    assert audit.observation_noise_present
    assert audit.neuron_dropout_fraction > 0.0
    assert audit.mixing_strength > 0.0
    assert audit.observation_noise > 0.0


def test_report_is_bitwise_deterministic(benchmark) -> None:
    first = neural_language_partial_blind_gate_report(benchmark).to_dict()
    second = neural_language_partial_blind_gate_report(benchmark).to_dict()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second,
        sort_keys=True,
    )


def test_impossible_threshold_fails_without_unlocking_claims(
    benchmark,
) -> None:
    impossible = replace(
        benchmark.thresholds,
        selected_candidate_train_accuracy_min=1.0,
        heldout_late_transition_accuracy_min=1.0,
        late_state_recovery_accuracy_min=1.0,
        alignment_over_permutation_null_gain_min=1.0,
    )
    report = neural_language_partial_blind_gate_report(
        replace(benchmark, thresholds=impossible)
    )

    assert not report.partial_blind_synthetic_pass
    assert not report.real_neural_data_used
    assert not report.full_brain_language_identified
    assert not report.neural_clarus_assembly_validated
    assert not report.causal_instruction_set_validated
    assert not report.fully_blind_inverse_recovery_validated


@pytest.mark.parametrize(
    "lock_name",
    [
        "real_neural_data_used",
        "full_brain_language_identified",
        "neural_clarus_assembly_validated",
        "causal_instruction_set_validated",
        "fully_blind_inverse_recovery_validated",
    ],
)
def test_loader_rejects_every_attempt_to_unlock_claim(
    tmp_path: Path,
    lock_name: str,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["claim_locks"][lock_name] = True
    path = tmp_path / "unlocked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be false"):
        load_neural_language_partial_blind_benchmark(path)


def test_equally_stable_monolithic_distractor_forces_abstention(
    benchmark,
) -> None:
    ambiguous_generator = replace(
        benchmark.generator,
        stable_monolithic_distractor_count=1,
    )
    report = neural_language_partial_blind_gate_report(
        replace(benchmark, generator=ambiguous_generator)
    )

    assert report.method_status == PARTIAL_BLIND_SYNTHETIC_AMBIGUOUS
    assert not report.partial_blind_synthetic_pass
    assert report.selection_abstained
    assert report.selected_candidate_group is None
    assert report.abstention_reason == "candidate_margin_below_threshold"
    assert report.heldout_context_audit.diagnostic_only_after_abstention
    assert "abstained" in report.conclusion
    assert not report.full_brain_language_identified
    assert not report.neural_clarus_assembly_validated


def test_loader_rejects_unknown_or_missing_top_level_keys(
    tmp_path: Path,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    del payload["description"]
    payload["brain_language_found"] = True
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_neural_language_partial_blind_benchmark(path)


def test_loader_rejects_unknown_nested_key(tmp_path: Path) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["generator"]["oracle_state_labels"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generator has unknown keys"):
        load_neural_language_partial_blind_benchmark(path)


def test_loader_rejects_withheld_item_becoming_known(tmp_path: Path) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["hidden_from_inverse"] = ["generator_target_candidate"]
    path = tmp_path / "leaky.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hidden_from_inverse must equal"):
        load_neural_language_partial_blind_benchmark(path)


def test_loader_rejects_required_known_item_becoming_false(
    tmp_path: Path,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["known_to_inverse"]["latent_cardinality"] = False
    path = tmp_path / "scope-change.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be true"):
        load_neural_language_partial_blind_benchmark(path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("generator", "state_count", 2, "between 3 and 7"),
        ("generator", "token_count", 4, "must equal"),
        ("generator", "candidate_group_count", 2, "at least 3"),
        (
            "generator",
            "stable_monolithic_distractor_count",
            3,
            "below candidate_group_count",
        ),
        ("generator", "samples_per_context", 200, "at least 400"),
        ("generator", "transition_noise", 0.30, "must be in"),
        (
            "generator",
            "neuron_dropout_fraction",
            0.0,
            "must be in",
        ),
        (
            "inference",
            "early_calibration_fraction",
            0.10,
            "must be in",
        ),
        (
            "inference",
            "train_context_count",
            3,
            "leave exactly one",
        ),
    ],
)
def test_loader_rejects_invalid_configuration(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_neural_language_partial_blind_benchmark(path)


def test_loader_rejects_overlapping_candidate_thresholds(
    tmp_path: Path,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["thresholds"]["distractor_train_accuracy_max"] = payload[
        "thresholds"
    ]["selected_candidate_train_accuracy_min"]
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be lower"):
        load_neural_language_partial_blind_benchmark(path)


def test_report_rejects_wrong_container() -> None:
    with pytest.raises(
        TypeError,
        match="NeuralLanguagePartialBlindBenchmark",
    ):
        neural_language_partial_blind_gate_report({})  # type: ignore[arg-type]
