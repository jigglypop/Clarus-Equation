"""Run the label-blind, session-local Tafazoli stationary-operator probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from reality_stone.clarus.tafazoli_session_operator_probe import (
    ProbeConfig,
    TafazoliSessionOperatorProbeReport,
    run_tafazoli_session_operator_probe,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit 100-ms session-local stationary operators to dimensions 1/3. "
            "Labels, AllFactors, dimension 2, and saved classifier test rows "
            "are never used."
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
    )
    parser.add_argument("--rank-cap", type=int, default=5)
    parser.add_argument("--successor-shuffles", type=int, default=100)
    parser.add_argument(
        "--transition-stride-bins",
        type=int,
        default=1,
        help=(
            "start-step for lag-10 transition pairs; 1 reproduces the primary "
            "probe and 10 avoids repeated weighting from adjacent start bins"
        ),
    )
    parser.add_argument(
        "--stride-10-sensitivity",
        action="store_true",
        help=(
            "also run a compact stride-10 sensitivity without event-mean or "
            "multi-rank repeats"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print deterministic JSON (NaN is rejected)",
    )
    return parser


def _all_aggregate(
    report: TafazoliSessionOperatorProbeReport,
    analysis_key: str,
    *,
    event_mean_removed: bool,
):
    matches = tuple(
        item
        for item in report.aggregates
        if item.analysis_key == analysis_key
        and item.animal == "ALL"
        and item.event_mean_removed == event_mean_removed
    )
    if len(matches) != 1:
        raise RuntimeError(f"missing aggregate: {analysis_key}")
    return matches[0]


def _print_report(report: TafazoliSessionOperatorProbeReport) -> None:
    print("TAFAZOLI SESSION-LOCAL STATIONARY OPERATOR PROBE")
    print(f"  status                     {report.method_status}")
    print(f"  official checksum          {report.source_file_md5}")
    print(f"  sessions                   {len(report.session_specs)}")
    print(
        "  animals                    "
        + ", ".join(
            f"{animal}={sum(s.animal == animal for s in report.session_specs)}"
            for animal in sorted({s.animal for s in report.session_specs})
        )
    )
    print(
        "  protocol                   "
        f"folds={report.config.fold_count} "
        f"lag={report.config.lag_bins} bins "
        f"stride={report.config.transition_stride_bins} bins "
        f"rank_cap={report.config.rank_cap} "
        f"shuffles={report.config.successor_shuffle_count}"
    )
    print(
        "  fitting fields             "
        + ", ".join(report.fields_used_for_fitting)
    )
    print("  primary session medians")
    print(
        "    analysis                 source-mean R2  persistence  "
        "time-mean  direction  shuffle  frozen-vs-refit"
    )
    for key in (
        "within_dim1",
        "within_dim3",
        "frozen_dim1_to_dim3",
        "frozen_dim3_to_dim1",
    ):
        item = _all_aggregate(
            report,
            key,
            event_mean_removed=False,
        )
        refit = (
            "n/a"
            if item.median_frozen_vs_target_refit_skill is None
            else f"{item.median_frozen_vs_target_refit_skill:+.4f}"
        )
        print(
            f"    {key:<24} "
            f"{item.median_source_grand_mean_r2:+.4f}       "
            f"{item.median_persistence_skill:+.4f}      "
            f"{item.median_time_locked_mean_skill:+.4f}   "
            f"{item.median_direction_specificity:+.4f}    "
            f"{item.median_successor_shuffle_advantage:+.4f}   "
            f"{refit}"
        )

    if report.event_mean_removed_within_results:
        print("  event-time-mean-removed session medians")
        for key in (
            "within_dim1",
            "within_dim3",
            "frozen_dim1_to_dim3",
            "frozen_dim3_to_dim1",
        ):
            item = _all_aggregate(
                report,
                key,
                event_mean_removed=True,
            )
            print(
                f"    {key:<24} "
                f"source-mean R2={item.median_source_grand_mean_r2:+.4f} "
                f"direction={item.median_direction_specificity:+.4f} "
                "shuffle="
                f"{item.median_successor_shuffle_advantage:+.4f}"
            )

    print("  rank stability")
    for item in report.rank_stability:
        refit = (
            "n/a"
            if item.median_frozen_vs_target_refit_skill is None
            else f"{item.median_frozen_vs_target_refit_skill:+.4f}"
        )
        print(
            f"    r<={item.rank_cap} {item.analysis_key:<24} "
            f"source-mean R2={item.median_source_grand_mean_r2:+.4f} "
            f"direction={item.median_direction_specificity:+.4f} "
            f"frozen-vs-refit={refit}"
        )

    print("  claim-local verdicts")
    for item in report.verdicts:
        print(f"    {item.answer:<16} {item.key}")
    print("  next tests")
    for item in report.next_tests:
        print(f"    {item.status:<16} {item.key}")
    print(f"  conclusion                 {report.conclusion}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parser(repository_root).parse_args()
    config = ProbeConfig(
        rank_cap=args.rank_cap,
        successor_shuffle_count=args.successor_shuffles,
        transition_stride_bins=args.transition_stride_bins,
    )
    primary = run_tafazoli_session_operator_probe(
        args.classifier_file,
        config=config,
    )
    sensitivity = None
    if args.stride_10_sensitivity and config.transition_stride_bins != 10:
        sensitivity = run_tafazoli_session_operator_probe(
            args.classifier_file,
            config=replace(
                config,
                transition_stride_bins=10,
                run_event_mean_removed_sensitivity=False,
                rank_stability_caps=(config.rank_cap,),
            ),
        )

    if args.json:
        payload = {"primary": primary.to_dict()}
        if sensitivity is not None:
            payload["stride_10_sensitivity"] = sensitivity.to_dict()
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    _print_report(primary)
    if sensitivity is not None:
        print()
        print("STRIDE-10 SENSITIVITY")
        _print_report(sensitivity)


if __name__ == "__main__":
    main()
