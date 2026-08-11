from reality_stone.clarus.shared_option_benchmark import (
    SharedOptionBenchConfig,
    evaluate_shared_options,
)


def test_shared_option_benchmark_identifies_sharing_not_dag_specificity() -> None:
    result = evaluate_shared_options(
        SharedOptionBenchConfig(
            train_samples_per_pair=20,
            test_samples_per_pair=48,
            epochs=120,
            seeds=4,
        )
    )
    assert result["schema"] == "clarus.shared-option-topology.validation.v1"
    assert result["gates"]["sharing_identity"]
    assert not result["gates"]["dag_specificity"]
    assert result["verdict"] == "FACTORIZATION_GO_DAG_UNRESOLVED"
    summaries = result["summaries"]
    assert summaries["shared_dag"] == summaries["factorized_flat"]
    assert summaries["shared_dag"]["accuracy"] > summaries["strict_tree"]["accuracy"]
