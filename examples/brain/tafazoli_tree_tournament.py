"""Run the finite first-round Tafazoli tree-family tournament."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from reality_stone.clarus.tafazoli_tree_tournament import (
    FAMILY_FLAT_SWITCHING,
    FAMILY_MATCHED_VAR,
    TREE_FAMILIES,
    TafazoliTreeTournamentReport,
    TreeTournamentConfig,
    run_tafazoli_tree_tournament,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run five restricted predictor families under outer-train-only nested "
            "selection. A screening survivor is not a biological identification."
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
        "--no-event-mean-sensitivity",
        action="store_true",
        help="skip event-time-mean removal; survivor verdict remains pending",
    )
    parser.add_argument(
        "--no-reverse-control",
        action="store_true",
        help="skip reverse-time control; survivor verdict remains pending",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a compact deterministic JSON report without per-fold records",
    )
    return parser


def _selected_spec_counts(
    report: TafazoliTreeTournamentReport,
) -> dict[str, dict[str, int]]:
    payload: dict[str, dict[str, int]] = {}
    for family in report.config.families:
        counts: Counter[str] = Counter()
        for result in report.results:
            if result.family == family:
                counts.update(result.selected_spec_keys)
        payload[family] = dict(sorted(counts.items()))
    return payload


def _compact_payload(
    report: TafazoliTreeTournamentReport,
) -> dict[str, Any]:
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
        "catalog": tuple(asdict(item) for item in report.catalog),
        "all_session_aggregates": tuple(
            asdict(item) for item in report.aggregates if item.animal == "all"
        ),
        "selected_spec_counts": _selected_spec_counts(report),
        "screening_survivors": report.screening_survivors,
        "model_relative_winner": report.model_relative_winner,
        "verdicts": tuple(asdict(item) for item in report.verdicts),
        "claim_locks": asdict(report.claim_locks),
        "limitations": report.limitations,
        "conclusion": report.conclusion,
    }


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _disposition(
    report: TafazoliTreeTournamentReport,
    family: str,
) -> str:
    if family in (FAMILY_MATCHED_VAR, FAMILY_FLAT_SWITCHING):
        return "INCUMBENT"
    controls_complete = (
        report.config.run_event_mean_removed_sensitivity
        and report.config.run_reverse_descriptive_control
    )
    if not controls_complete:
        return "PENDING"
    if family in report.screening_survivors:
        return "SCREENING_SURVIVOR"
    if family in TREE_FAMILIES:
        return "ELIMINATED_ROUND_1"
    return "PENDING"


def _print_report(report: TafazoliTreeTournamentReport) -> None:
    print("TAFAZOLI FINITE TREE-FAMILY TOURNAMENT")
    print(f"  status              {report.method_status}")
    print(f"  checksum            {report.source_file_md5}")
    print(f"  sessions            {len(report.session_specs)}")
    print(
        "  protocol            "
        f"outer={report.config.outer_fold_count} "
        f"inner={report.config.inner_fold_count} "
        f"lag/stride={report.config.lag_bins}/"
        f"{report.config.primary_stride_bins} bins"
    )
    print("  all-session medians")
    print(
        "    eventmean  family                                  "
        "bits/scalar  >VAR       >flat      >reverse   disposition"
    )
    for aggregate in report.aggregates:
        if aggregate.animal != "all":
            continue
        print(
            f"    {int(aggregate.event_mean_removed):<10} "
            f"{aggregate.family:<39} "
            f"{aggregate.median_model_bits_per_test_scalar:+.4f}      "
            f"{_number(aggregate.median_advantage_over_var_bits_per_scalar):<10} "
            f"{_number(aggregate.median_advantage_over_flat_switching_bits_per_scalar):<10} "
            f"{_number(aggregate.median_forward_advantage_over_reverse_bits_per_scalar):<10} "
            f"{_disposition(report, aggregate.family)}"
        )
    print(f"  survivor set        {report.screening_survivors}")
    print(f"  unique leader       {report.model_relative_winner}")
    print("  claim-local verdicts")
    for verdict in report.verdicts:
        print(f"    {verdict.answer:<16} {verdict.key}")
    print(f"  conclusion          {report.conclusion}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parser(repository_root).parse_args()
    config = TreeTournamentConfig(
        run_event_mean_removed_sensitivity=(
            not args.no_event_mean_sensitivity
        ),
        run_reverse_descriptive_control=not args.no_reverse_control,
    )
    report = run_tafazoli_tree_tournament(
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
