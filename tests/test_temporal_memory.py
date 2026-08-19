from reality_stone.clarus.temporal_memory_benchmark import (
    TemporalMemoryBenchConfig,
    evaluate_temporal_memory,
)


def test_temporal_memory_locked_small_protocol_is_bounded() -> None:
    result = evaluate_temporal_memory(
        TemporalMemoryBenchConfig(seeds=4, bootstrap_draws=400)
    )
    candidate = result["means"]["candidate"]
    assert result["schema"] == "clarus.temporal-memory.validation.v1"
    assert result["hard_gate"]
    assert candidate["capacity_ok"] == 1.0
    assert candidate["update_audit_count"] == candidate["expected_update_audit_count"]
    assert candidate["delete_audit_count"] == candidate["expected_delete_audit_count"]
    assert result["candidate_storage_ratio_vs_full_context"] <= 0.60
