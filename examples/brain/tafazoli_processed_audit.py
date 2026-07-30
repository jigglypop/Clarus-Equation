"""Audit checksum-matched official processed Tafazoli figure outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.tafazoli_processed_audit import (
    load_tafazoli_processed_audit_manifest,
    run_tafazoli_processed_audit,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract descriptive values from official processed neural "
            "figure outputs. This is not a raw-spike reanalysis."
        )
    )
    parser.add_argument(
        "--manifest",
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
    manifest = load_tafazoli_processed_audit_manifest(args.manifest)
    report = run_tafazoli_processed_audit(manifest, args.data_dir)

    if args.json:
        print(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("TAFAZOLI OFFICIAL PROCESSED-FIGURE AUDIT")
    print(f"  status                  {report.method_status}")
    print(f"  scope                   {report.scope}")
    for source in report.source_files:
        print(
            f"  checksum {source.filename:<31} "
            f"{source.checksum_matches}"
        )

    print("  decoder summaries (250 classifier resamples; not 250 animals)")
    for curve in report.decoder_curves:
        print(
            f"    Fig {curve.figure_panel:<3} {curve.name:<31} "
            f"smoothed_peak={curve.plotted_smoothed_peak_accuracy:.4f} "
            f"at={curve.plotted_smoothed_peak_time_seconds:+.2f}s "
            "smoothed_postmean="
            f"{curve.plotted_smoothed_post_event_mean_accuracy:.4f}"
        )

    print("  dynamic-correlation summaries (author-processed matrices)")
    for item in report.dynamic_correlations:
        print(
            f"    {item.name:<38} "
            f"mean={item.window_mean_correlation:+.5f} "
            f"p<.05 fraction={item.pointwise_p_below_0_05_fraction:.4f} "
            "weighted lag="
            f"{item.positive_diagonal_projection_weighted_lag_seconds:+.3f}s"
        )

    print(
        "  raw trial/spike reanalysis "
        f"{report.raw_trial_or_spike_reanalysis}"
    )
    print(
        "  independent replication   "
        f"{report.independent_replication}"
    )
    print(
        "  unseen composition tested "
        f"{report.unseen_composition_tested}"
    )
    print(
        "  brain language identified "
        f"{report.full_brain_language_identified}"
    )
    print(f"  conclusion               {report.conclusion}")


if __name__ == "__main__":
    main()
