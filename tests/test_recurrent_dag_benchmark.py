from reality_stone.clarus.recurrent_dag_benchmark import (
    evaluate_context_boundary,
    evaluate_outcome_posterior,
    evaluate_recurrent_dag,
    evaluate_soft_evidence,
    small_recurrent_dag_config,
)


def test_recurrent_dag_benchmark_integrity_and_schema() -> None:
    result = evaluate_recurrent_dag(small_recurrent_dag_config())
    assert result["schema"] == "clarus.recurrent-bg-dag.validation.v1"
    assert result["future_reads"] == 0
    assert result["environment_clone_calls"] == 0
    assert result["same_tick_feedback_commits"] == 0
    assert result["topology_cycles"] == 0
    assert result["gates"]["finite_valid_topology"]
    assert result["gates"]["bounded_finite_state"]
    assert 0 <= result["promise_score"] <= 100


def test_soft_evidence_benchmark_has_no_unreachable_targets() -> None:
    result = evaluate_soft_evidence(small_recurrent_dag_config())
    assert result["schema"] == "clarus.recurrent-bg-dag-soft-evidence.validation.v1"
    assert result["id"]["unreachable_count"] == 0.0
    assert result["ood"]["unreachable_count"] == 0.0
    assert result["id"]["minimum_true_probability"] > 0.0
    assert result["ood"]["minimum_true_probability"] > 0.0
    assert result["pending_overwrites"] == 0


def test_context_boundary_benchmark_mechanism_identities() -> None:
    result = evaluate_context_boundary(small_recurrent_dag_config())
    assert result["schema"] == "clarus.recurrent-bg-dag-context-boundary.validation.v1"
    assert result["gates"]["exact_surprise_identity"]
    assert result["gates"]["directional_nonexpansive"]
    assert result["future_reads"] == 0


def test_outcome_posterior_benchmark_filter_identities() -> None:
    result = evaluate_outcome_posterior(small_recurrent_dag_config())
    assert result["schema"] == "clarus.recurrent-bg-dag-outcome-posterior.validation.v1"
    assert result["gates"]["simplex_valid"]
    assert result["gates"]["exact_filter_identities"]
    assert result["legacy_decay_updates_in_candidate"] == 0
    assert result["explicit_resets_in_candidate"] == 0
