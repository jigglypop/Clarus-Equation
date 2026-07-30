"""Run the label-blind Tafazoli call-graph proxy probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from reality_stone.clarus.tafazoli_call_graph_probe import (
    CallGraphProbeConfig,
    TafazoliCallGraphProbeReport,
    run_tafazoli_call_graph_probe,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run session-local switching, common-successor, state-low-rank, "
            "and frozen D1/D3 transfer proxies. Labels, dimension 2, and the "
            "saved classifier test tensor are not used."
        )
    )
    parser.add_argument(
        "--classifier-file",
        type=Path,
        default=(
            repository_root
            / "data"
            / "tafazoli_compositional_v1"
            / "PFC_ClassifierData.mat"
        ),
        help="official checksum-locked PFC_ClassifierData.mat",
    )
    parser.add_argument(
        "--states",
        type=int,
        nargs="+",
        default=(2, 3),
        help="predeclared gate-state counts (default: 2 3)",
    )
    parser.add_argument(
        "--history-depths",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="past-history depths P, independent of state count S",
    )
    parser.add_argument("--rank-cap", type=int, default=3)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--kmeans-restarts", type=int, default=8)
    parser.add_argument(
        "--no-event-mean-sensitivity",
        action="store_true",
        help="skip the event-time-mean-removed sensitivity; proxy verdicts remain pending",
    )
    parser.add_argument(
        "--no-reverse-control",
        action="store_true",
        help="skip the reverse-time descriptive control",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a compact deterministic JSON report without raw tensors",
    )
    return parser


def _compact_payload(report: TafazoliCallGraphProbeReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "scope": report.scope,
        "method_status": report.method_status,
        "source_file_md5": report.source_file_md5,
        "official_checksum_verified": report.official_checksum_verified,
        "config": asdict(report.config),
        "session_count": len(report.session_specs),
        "fields_used_for_fitting": report.fields_used_for_fitting,
        "blind_fields_used": report.blind_fields_used,
        "saved_test_role": report.saved_test_role,
        "primary_inference_unit": report.primary_inference_unit,
        "codelength_name": report.codelength_name,
        "aggregates": tuple(
            asdict(item) for item in report.aggregates if item.animal == "all"
        ),
        "verdicts": tuple(asdict(item) for item in report.verdicts),
        "claim_locks": asdict(report.claim_locks),
        "limitations": report.limitations,
        "conclusion": report.conclusion,
    }


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _print_report(report: TafazoliCallGraphProbeReport) -> None:
    print("TAFAZOLI CALL-GRAPH OBSERVATIONAL PROXY PROBE")
    print(f"  status              {report.method_status}")
    print(f"  checksum            {report.source_file_md5}")
    print(f"  sessions            {len(report.session_specs)}")
    print(
        "  protocol            "
        f"S={report.config.states} "
        f"P={report.config.history_depths} "
        f"folds={report.config.fold_count} "
        f"lag/stride={report.config.lag_bins}/"
        f"{report.config.primary_stride_bins} bins"
    )
    print(f"  codelength          {report.codelength_name}")
    print("  session x dimension medians (primary counts)")
    print(
        "    S  P  units  switch>VAR  parent>VAR  "
        "hub>time  hub>caller  hub folds complete"
    )
    primary = tuple(
        item
        for item in report.aggregates
        if item.animal == "all" and not item.event_mean_removed
    )
    for item in primary:
        print(
            f"    {item.state_count:<2} {item.history_depth:<2} "
            f"{item.unit_count:<6} "
            f"{_number(item.median_switching_codelength_advantage_bits_per_scalar):<11} "
            f"{_number(item.median_state_parent_rank1_codelength_advantage_bits_per_scalar):<11} "
            f"{_number(item.median_hub_shared_codelength_advantage_over_time_bits_per_scalar):<9} "
            f"{_number(item.median_hub_shared_codelength_advantage_over_caller_bits_per_scalar):<11} "
            f"{item.all_units_have_complete_hub_folds}"
        )
    print("  claim-local verdicts")
    for verdict in report.verdicts:
        print(f"    {verdict.answer:<16} {verdict.key}")
    print(f"  conclusion          {report.conclusion}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parser(repository_root).parse_args()
    config = CallGraphProbeConfig(
        states=tuple(args.states),
        history_depths=tuple(args.history_depths),
        rank_cap=args.rank_cap,
        fold_count=args.folds,
        kmeans_restarts=args.kmeans_restarts,
        run_event_mean_removed_sensitivity=(
            not args.no_event_mean_sensitivity
        ),
        run_reverse_descriptive_control=not args.no_reverse_control,
    )
    report = run_tafazoli_call_graph_probe(
        args.classifier_file,
        config=config,
    )
    if args.json:
        print(
            json.dumps(
                _compact_payload(report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    _print_report(report)


if __name__ == "__main__":
    main()
