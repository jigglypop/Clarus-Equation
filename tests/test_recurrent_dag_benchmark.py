from reality_stone.clarus.recurrent_dag_benchmark import (
    evaluate_recurrent_dag,
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
