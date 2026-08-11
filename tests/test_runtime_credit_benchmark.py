from reality_stone.clarus.runtime_credit_benchmark import (
    RuntimeCreditBenchConfig,
    evaluate_runtime_credit,
)


def test_runtime_credit_benchmark_is_deterministic_and_guarded():
    config = RuntimeCreditBenchConfig(dim=12, steps=32, window=8, probes=8)
    first = evaluate_runtime_credit(seeds=(971001, 971002), config=config)
    second = evaluate_runtime_credit(seeds=(971001, 971002), config=config)
    assert first == second
    assert set(first["mean_improvement"]) == {
        "off", "legacy", "signed", "sign_flip", "absolute", "trace_off",
        "reward_shuffle", "homeostasis_only",
    }
    assert first["mean_drift"]["off"] == 0.0
    assert all(first["updates"]["signed"])


def test_matched_manifold_changes_schema_and_preserves_off_weight():
    config = RuntimeCreditBenchConfig(
        dim=12,
        steps=32,
        window=8,
        probes=8,
        matched_initial_manifold=True,
    )
    result = evaluate_runtime_credit(seeds=(972001, 972002), config=config)
    assert result["schema"].endswith("matched-manifold.v2")
    assert result["mean_drift"]["off"] == 0.0
