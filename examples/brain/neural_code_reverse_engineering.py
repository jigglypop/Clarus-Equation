"""Run the claim-by-claim Tafazoli neural-code reverse-engineering gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.neural_code_reverse_engineering import (
    build_neural_code_reverse_engineering_report,
    load_neural_code_reverse_engineering_manifest,
    run_tafazoli_classifier_snapshot_audit,
    verify_report_internal_consistency,
)
from reality_stone.clarus.tafazoli_processed_audit import (
    load_tafazoli_processed_audit_manifest,
    run_tafazoli_processed_audit,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the task-code skeleton, recover pseudopopulation "
            "session boundaries, and print claim-local YES/NO/UNAVAILABLE "
            "verdicts. This does not identify a brain language."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            repository_root
            / "benchmarks"
            / "neural_code_reverse_engineering_v1.json"
        ),
    )
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=(
            repository_root
            / "benchmarks"
            / "tafazoli_processed_audit_v1.json"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            repository_root / "data" / "tafazoli_compositional_v1"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete report as deterministic JSON",
    )
    return parser


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parser(repository_root).parse_args()

    manifest = load_neural_code_reverse_engineering_manifest(args.manifest)
    processed_manifest = load_tafazoli_processed_audit_manifest(
        args.processed_manifest
    )
    processed_report = run_tafazoli_processed_audit(
        processed_manifest,
        args.data_dir,
    )
    snapshot_report = run_tafazoli_classifier_snapshot_audit(
        manifest,
        args.data_dir,
    )
    report = build_neural_code_reverse_engineering_report(
        manifest,
        processed_report,
        snapshot_report,
    )
    verify_report_internal_consistency(report)

    if args.json:
        print(
            json.dumps(
                report.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("NEURAL CODE REVERSE-ENGINEERING VERDICT")
    print(f"  status {report.method_status}")
    print("  reconstructed task programs")
    for task in report.task_programs:
        print(f"    {task.task}: {' -> '.join(task.program)}")
    missing = report.missing_composition_prediction
    print(
        f"    {missing.task}: {' -> '.join(missing.program)} "
        "[PREDICTED, NOT RECORDED]"
    )
    print("  data fitness")
    print(
        "    stitched population simultaneous: "
        f"{snapshot_report.full_pseudopopulation_is_simultaneous}"
    )
    print(
        "    recovered recording sessions: "
        f"{len(snapshot_report.session_groups)}"
    )
    print(
        "    session neuron counts: "
        + ",".join(
            str(group.neuron_count)
            for group in snapshot_report.session_groups
        )
    )
    print(
        "    animal neuron counts: "
        + ", ".join(
            f"{animal}={count}"
            for animal, count in snapshot_report.animal_neuron_counts
        )
    )
    print(
        "    adjacent counting-window overlap: "
        f"{snapshot_report.adjacent_window_overlap_fraction:.0%}"
    )
    for dimension in snapshot_report.dimensions:
        print(
            f"    D{dimension.dimension_one_based} "
            f"{dimension.target_factor:<11} adjacent-corr="
            f"{dimension.mean_adjacent_time_bin_correlation:.4f} "
            "primary="
            f"{dimension.primary_discovery_allowed}"
        )
    print(
        "    session-local operator pilot possible: "
        f"{snapshot_report.session_local_operator_pilot_possible}"
    )
    print("  claim-local verdicts")
    for verdict in report.claim_verdicts:
        print(f"    {verdict.answer:<16} {verdict.key}")
    print(f"  competing family winner: {report.competing_family_winner}")
    print(f"  conclusion: {report.conclusion}")


if __name__ == "__main__":
    main()
