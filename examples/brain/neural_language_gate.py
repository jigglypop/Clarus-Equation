"""Run the fixed oracle-labelled neural-language method control."""

from __future__ import annotations

from pathlib import Path

from reality_stone.clarus.neural_language_gate import (
    load_neural_language_benchmark,
    neural_language_gate_report,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    benchmark_path = (
        repository_root / "benchmarks" / "neural_language_synthetic_v1.json"
    )
    benchmark = load_neural_language_benchmark(benchmark_path)
    report = neural_language_gate_report(benchmark)
    boundary = report.boundary_closure_audit
    reuse = report.reuse_audit
    composition = report.composition_audit
    self_feedback = report.self_feedback_audit
    cross_feedback = report.cross_assembly_feedback_audit

    print("NEURAL LANGUAGE ORACLE-LABELLED SYNTHETIC METHOD CONTROL")
    print(f"  benchmark                         {benchmark_path}")
    print(f"  scope                             {report.scope}")
    print(f"  status                            {report.structural_status}")
    print(
        "  boundary heldout accuracy         "
        f"{boundary.mean_heldout_context_accuracy:.6f}"
    )
    print(
        "  max context transition TV         "
        f"{boundary.maximum_context_transition_total_variation:.6f}"
    )
    print(
        "  nuisance predictive gain          "
        f"{boundary.nuisance_accuracy_gain:.6f}"
    )
    print(
        "  early-to-late reuse accuracy      "
        f"{reuse.early_to_late_repetition_accuracy:.6f}"
    )
    print(
        "  learned A->B composition accuracy "
        f"{composition.primitive_composition_accuracy:.6f}"
    )
    print(
        "  shuffled-target accuracy          "
        f"{composition.shuffled_target_composition_accuracy:.6f}"
    )
    print(
        "  non-compositional lookup accuracy "
        f"{composition.noncompositional_lookup_accuracy:.6f}"
    )
    print(
        "  seen-tuple lookup accuracy        "
        f"{composition.lookup_seen_memorization_accuracy:.6f}"
    )
    print(
        "  self-feedback open-loop accuracy  "
        f"{self_feedback.step_accuracy:.6f}"
    )
    print(
        "  self-feedback severed accuracy    "
        f"{self_feedback.severed_edge_step_accuracy:.6f}"
    )
    print(
        "  cross-assembly open-loop accuracy "
        f"{cross_feedback.step_accuracy:.6f}"
    )
    print(
        "  cross-assembly severed accuracy   "
        f"{cross_feedback.severed_edge_step_accuracy:.6f}"
    )
    print(
        "  oracle-labelled method pass       "
        f"{report.synthetic_oracle_labeled_method_control_pass}"
    )
    print(
        "  real neural data used             "
        f"{report.real_neural_data_used}"
    )
    print(
        "  full brain language identified    "
        f"{report.full_brain_language_identified}"
    )
    print(
        "  neural Clarus assembly validated  "
        f"{report.neural_clarus_assembly_validated}"
    )
    print(
        "  causal instruction set validated  "
        f"{report.causal_instruction_set_validated}"
    )
    print(f"  conclusion                        {report.conclusion}")


if __name__ == "__main__":
    main()
