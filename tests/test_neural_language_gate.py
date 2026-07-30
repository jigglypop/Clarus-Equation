from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from reality_stone.clarus.neural_language_gate import (
    BOUNDARY_STATE,
    INPUT_TOKEN,
    MODEL_CLASS,
    PRIMITIVE_OPERATIONS,
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SYNTHETIC_PASS,
    SYNTHETIC_SCOPE,
    load_neural_language_benchmark,
    neural_language_gate_report,
)


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "neural_language_synthetic_v1.json"
)


@pytest.fixture
def benchmark():
    return load_neural_language_benchmark(BENCHMARK)


def test_strict_benchmark_loads_with_biological_claims_locked(benchmark) -> None:
    assert benchmark.schema_version == SCHEMA_VERSION
    assert benchmark.scope == SYNTHETIC_SCOPE
    assert benchmark.transducer.model_class == MODEL_CLASS
    assert benchmark.transducer.boundary_state == BOUNDARY_STATE
    assert benchmark.transducer.input_token == INPUT_TOKEN
    assert benchmark.transducer.primitive_operations == PRIMITIVE_OPERATIONS
    assert not benchmark.full_brain_language_identified
    assert not benchmark.neural_clarus_assembly_validated
    assert not benchmark.causal_instruction_set_validated
    assert "biological_clarus_cell_existence" in benchmark.excluded_inferences


def test_reference_transducer_passes_oracle_labeled_method_control(
    benchmark,
) -> None:
    report = neural_language_gate_report(benchmark)

    assert report.structural_status == SYNTHETIC_PASS
    assert report.schema_version == REPORT_SCHEMA_VERSION
    assert report.synthetic_oracle_labeled_method_control_pass
    assert not report.real_neural_data_used
    assert not report.full_brain_language_identified
    assert not report.neural_clarus_assembly_validated
    assert not report.causal_instruction_set_validated
    assert "forward synthetic pipeline" in report.conclusion


def test_boundary_summary_is_predictively_sufficient_and_closed(
    benchmark,
) -> None:
    audit = neural_language_gate_report(benchmark).boundary_closure_audit

    assert audit.predictive_sufficiency_pass
    assert audit.empirical_closure_pass
    assert audit.mean_heldout_context_accuracy >= 0.85
    assert audit.minimum_heldout_context_accuracy >= 0.85
    assert audit.maximum_context_transition_total_variation <= 0.08
    assert audit.nuisance_accuracy_gain <= 0.02
    assert audit.boundary_variables == (
        BOUNDARY_STATE,
        INPUT_TOKEN,
        "primitive_operation",
    )
    assert "interior_microstate_label" in audit.nuisance_variables


def test_operations_reuse_across_context_and_repetition(benchmark) -> None:
    audit = neural_language_gate_report(benchmark).reuse_audit

    assert audit.same_operation_reused_across_contexts
    assert audit.same_operation_reused_across_repetitions
    assert audit.structural_pass
    assert {item.operation for item in audit.operation_accuracy} == {"A", "B"}
    assert audit.minimum_operation_accuracy >= 0.85


def test_composition_generalizes_on_heldout_tuples_and_controls_fail(
    benchmark,
) -> None:
    audit = neural_language_gate_report(benchmark).composition_audit

    assert audit.sequence == ("A", "B")
    assert audit.heldout_case_count > 0
    assert audit.evaluation_count > 0
    assert audit.seen_evaluation_count > audit.evaluation_count
    assert audit.primitive_composition_accuracy >= 0.75
    assert audit.true_composition_pass
    assert not audit.shuffled_targets_passed_composition_gate
    assert not audit.noncompositional_lookup_passed_composition_gate
    assert audit.shuffled_target_composition_accuracy <= 0.35
    assert audit.noncompositional_lookup_accuracy <= 0.35
    assert audit.lookup_seen_memorization_accuracy >= 0.75
    assert audit.shuffled_target_control_rejected
    assert audit.noncompositional_lookup_control_rejected
    assert audit.negative_controls_rejected
    assert audit.structural_pass


def test_self_and_cross_assembly_feedback_use_open_loop_edge_controls(
    benchmark,
) -> None:
    report = neural_language_gate_report(benchmark)
    self_loop = report.self_feedback_audit
    cross_assembly = report.cross_assembly_feedback_audit

    assert self_loop.topology == "single_assembly_self_feedback"
    assert self_loop.directed_edges == (("assembly_0", "assembly_0"),)
    assert self_loop.self_loop_count == 1
    assert self_loop.cross_assembly_edge_count == 0
    assert self_loop.cycle_closed
    assert self_loop.evaluation_mode == (
        "noiseless_open_loop_depth_extrapolation"
    )
    assert self_loop.step_accuracy >= 0.85
    assert self_loop.edge_ablation_gap >= 0.40
    assert self_loop.edge_dependency_pass
    assert self_loop.structural_pass

    assert cross_assembly.topology == "two_assembly_mutual_feedback"
    assert cross_assembly.directed_edges == (
        ("assembly_0", "assembly_1"),
        ("assembly_1", "assembly_0"),
    )
    assert cross_assembly.self_loop_count == 0
    assert cross_assembly.cross_assembly_edge_count == 2
    assert cross_assembly.cycle_closed
    assert cross_assembly.evaluation_mode == (
        "noiseless_open_loop_depth_extrapolation"
    )
    assert cross_assembly.step_accuracy >= 0.85
    assert cross_assembly.edge_ablation_gap >= 0.40
    assert cross_assembly.edge_dependency_pass
    assert cross_assembly.structural_pass


def test_report_is_bitwise_deterministic_for_fixed_benchmark(benchmark) -> None:
    first = neural_language_gate_report(benchmark).to_dict()
    second = neural_language_gate_report(benchmark).to_dict()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second,
        sort_keys=True,
    )


def test_failed_synthetic_threshold_never_unlocks_biological_claims(
    benchmark,
) -> None:
    impossible = replace(
        benchmark.thresholds,
        boundary_accuracy_min=1.0,
        reuse_accuracy_min=1.0,
        composition_accuracy_min=1.0,
        recursion_step_accuracy_min=1.0,
    )
    report = neural_language_gate_report(
        replace(benchmark, thresholds=impossible)
    )

    assert not report.synthetic_oracle_labeled_method_control_pass
    assert not report.full_brain_language_identified
    assert not report.neural_clarus_assembly_validated
    assert not report.causal_instruction_set_validated


@pytest.mark.parametrize(
    "lock_name",
    [
        "full_brain_language_identified",
        "neural_clarus_assembly_validated",
        "causal_instruction_set_validated",
    ],
)
def test_loader_rejects_attempt_to_unlock_biological_claim(
    tmp_path: Path,
    lock_name: str,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["claim_locks"][lock_name] = True
    path = tmp_path / "unlocked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be false"):
        load_neural_language_benchmark(path)


def test_loader_rejects_missing_or_unknown_top_level_keys(
    tmp_path: Path,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    del payload["description"]
    payload["discovery_claim"] = True
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_neural_language_benchmark(path)


def test_loader_rejects_unknown_nested_generator_key(tmp_path: Path) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["generator"]["biological_confidence"] = 1.0
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generator has unknown keys"):
        load_neural_language_benchmark(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("state_count", 2, "at least 3"),
        ("token_count", 4, "must equal"),
        ("context_count", 2, "at least 3"),
        ("transition_noise", 0.5, "must be in"),
        ("laplace_alpha", 0.0, "must be in"),
        ("composition_holdout_modulus", 1, "at least 2"),
        ("recursion_depth", 1, "at least 2"),
    ],
)
def test_loader_rejects_invalid_generator_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["generator"][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_neural_language_benchmark(path)


def test_report_rejects_wrong_container() -> None:
    with pytest.raises(TypeError, match="NeuralLanguageSyntheticBenchmark"):
        neural_language_gate_report({})  # type: ignore[arg-type]


def test_loader_rejects_overlapping_positive_and_negative_thresholds(
    tmp_path: Path,
) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["thresholds"]["negative_control_accuracy_max"] = payload[
        "thresholds"
    ]["composition_accuracy_min"]
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be lower"):
        load_neural_language_benchmark(path)


def test_feedback_edge_ablation_is_a_required_gate(benchmark) -> None:
    impossible_gap = replace(
        benchmark.thresholds,
        feedback_edge_ablation_gap_min=0.80,
    )
    report = neural_language_gate_report(
        replace(benchmark, thresholds=impossible_gap)
    )

    assert not report.synthetic_oracle_labeled_method_control_pass
    assert (
        not report.self_feedback_audit.edge_dependency_pass
        or not report.cross_assembly_feedback_audit.edge_dependency_pass
    )
    assert not report.neural_clarus_assembly_validated
    assert not report.causal_instruction_set_validated
