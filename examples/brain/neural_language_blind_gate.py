"""Run the fixed partial-blind synthetic neural-language inverse control."""

from __future__ import annotations

from pathlib import Path

from reality_stone.clarus.neural_language_blind_gate import (
    load_neural_language_partial_blind_benchmark,
    neural_language_partial_blind_gate_report,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    benchmark_path = (
        repository_root
        / "benchmarks"
        / "neural_language_blind_synthetic_v1.json"
    )
    benchmark = load_neural_language_partial_blind_benchmark(benchmark_path)
    report = neural_language_partial_blind_gate_report(benchmark)
    heldout = report.heldout_context_audit

    print("NEURAL LANGUAGE PARTIAL-BLIND SYNTHETIC INVERSE CONTROL")
    print(f"  benchmark                         {benchmark_path}")
    print(f"  scope                             {report.scope}")
    print(f"  status                            {report.method_status}")
    print(
        "  top / selected / generator target  "
        f"{report.top_scoring_candidate_group} / "
        f"{report.selected_candidate_group} / "
        f"{report.scoring_only_generator_target_group}"
    )
    print(
        "  selection abstained                "
        f"{report.selection_abstained}"
    )
    for score in report.candidate_scores:
        print(
            f"  candidate {score.candidate_group} train accuracy"
            f"              {score.mean_train_context_late_accuracy:.6f}"
        )
    print(
        "  candidate selection margin        "
        f"{report.top_candidate_margin:.6f}"
    )
    print(
        "  maximum distractor accuracy       "
        f"{report.scoring_only_maximum_distractor_train_accuracy:.6f}"
    )
    print(
        "  heldout permutation-null mean      "
        f"{heldout.late_transition_accuracy_permutation_null_mean:.6f}"
    )
    print(
        "  heldout late aligned accuracy      "
        f"{heldout.late_transition_accuracy_with_alignment:.6f}"
    )
    print(
        "  alignment over null gain           "
        f"{heldout.alignment_over_permutation_null_gain:.6f}"
    )
    print(
        "  heldout late state recovery        "
        f"{heldout.late_latent_state_recovery_accuracy:.6f}"
    )
    print(
        "  partial-blind synthetic pass       "
        f"{report.partial_blind_synthetic_pass}"
    )
    print(
        "  real neural data used              "
        f"{report.real_neural_data_used}"
    )
    print(
        "  full brain language identified     "
        f"{report.full_brain_language_identified}"
    )
    print(
        "  fully blind inverse validated      "
        f"{report.fully_blind_inverse_recovery_validated}"
    )
    print(f"  conclusion                         {report.conclusion}")


if __name__ == "__main__":
    main()
