"""Run the checkpointed low-cost Tafazoli drift--diffusion proxy screen."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any

from reality_stone.clarus.tafazoli_diffusion_probe import (
    OU_DIAG,
    OU_FULL,
    OU_ISO,
    QUADRATIC_DRIFT_FULL_Q,
    STATE_SCALE,
    TIME_SCALE,
    DiffusionFoldResult,
    DiffusionProbeConfig,
    DiffusionSessionCheckpoint,
    DiffusionUnitResult,
    DirectionClassification,
    GaussianCodelengthResult,
    MarkovOrderSensitivity,
    SemigroupSensitivity,
    assemble_tafazoli_diffusion_report,
    run_diffusion_session_checkpoint,
    validate_diffusion_session_checkpoint,
)
from reality_stone.clarus.tafazoli_session_operator_probe import (
    SessionSpec,
    load_tafazoli_train_dimensions,
    recovered_session_specs,
    verify_official_classifier_checksum,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a session-local Gaussian drift/noise screen. Checkpoints are "
            "resumable, but no result identifies biological or generative diffusion."
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
        "--checkpoint-dir",
        type=Path,
        default=repository_root / ".tmp" / "tafazoli_diffusion_probe_v1",
        help="directory for one strict JSON checkpoint per physical session",
    )
    parser.add_argument(
        "--no-event-mean-sensitivity",
        action="store_true",
        help="skip the event-mean-removed sensitivity; the survivor verdict stays pending",
    )
    parser.add_argument(
        "--no-markov-sensitivity",
        action="store_true",
        help="skip the observed Markov-order sensitivity",
    )
    parser.add_argument(
        "--no-semigroup-sensitivity",
        action="store_true",
        help="skip the frozen 100-to-200/300 ms semigroup sensitivity",
    )
    parser.add_argument(
        "--no-reverse-classification",
        action="store_true",
        help="skip the descriptive forward/reverse classification",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="recompute and atomically replace matching checkpoints",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a compact deterministic JSON summary",
    )
    return parser


def _decode_score(payload: dict[str, Any]) -> GaussianCodelengthResult:
    return GaussianCodelengthResult(**payload)


def _decode_fold(payload: dict[str, Any]) -> DiffusionFoldResult:
    direction_payload = payload["direction_classification"]
    return DiffusionFoldResult(
        **{
            **payload,
            "scores": tuple(_decode_score(item) for item in payload["scores"]),
            "markov_order_sensitivity": tuple(
                MarkovOrderSensitivity(
                    **{
                        **item,
                        "score": _decode_score(item["score"]),
                    }
                )
                for item in payload["markov_order_sensitivity"]
            ),
            "semigroup_sensitivity": tuple(
                SemigroupSensitivity(**item)
                for item in payload["semigroup_sensitivity"]
            ),
            "direction_classification": (
                None
                if direction_payload is None
                else DirectionClassification(**direction_payload)
            ),
        }
    )


def _decode_unit(payload: dict[str, Any]) -> DiffusionUnitResult:
    return DiffusionUnitResult(
        **{
            **payload,
            "fold_results": tuple(
                _decode_fold(item) for item in payload["fold_results"]
            ),
        }
    )


def _decode_checkpoint(payload: dict[str, Any]) -> DiffusionSessionCheckpoint:
    return DiffusionSessionCheckpoint(
        **{
            **payload,
            "session": SessionSpec(**payload["session"]),
            "results": tuple(_decode_unit(item) for item in payload["results"]),
        }
    )


def _checkpoint_path(directory: Path, session: SessionSpec) -> Path:
    return directory / f"session_{session.index_one_based:02d}.json"


def _load_checkpoint(
    path: Path,
    *,
    session: SessionSpec,
    source_file_md5: str,
    config: DiffusionProbeConfig,
) -> DiffusionSessionCheckpoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = _decode_checkpoint(payload)
    validate_diffusion_session_checkpoint(checkpoint, config=config)
    if checkpoint.session != session:
        raise ValueError(f"{path} belongs to a different physical session")
    if checkpoint.source_file_md5 != source_file_md5:
        raise ValueError(f"{path} belongs to a different source snapshot")
    return checkpoint


def _write_checkpoint(path: Path, checkpoint: DiffusionSessionCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            checkpoint.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unit_family_components_per_scalar(
    unit: DiffusionUnitResult,
    family: str,
) -> dict[str, float]:
    scores = tuple(fold.score(family) for fold in unit.fold_results)
    total_scalars = sum(
        item.test_vector_count * item.latent_rank for item in scores
    )
    return {
        "total": float(
            sum(item.total_codelength_bits for item in scores) / total_scalars
        ),
        "nll": float(
            sum(item.heldout_multivariate_gaussian_nll_bits for item in scores)
            / total_scalars
        ),
        "complexity": float(
            sum(
                item.bic_parameter_bits + item.model_selection_bits
                for item in scores
            )
            / total_scalars
        ),
    }


def _covariance_comparisons(report) -> tuple[dict[str, Any], ...]:
    pairings = (
        ("diag_over_iso", OU_ISO, OU_DIAG),
        ("full_over_iso", OU_ISO, OU_FULL),
        ("full_over_diag", OU_DIAG, OU_FULL),
        ("time_over_full", OU_FULL, TIME_SCALE),
        ("state_over_full", OU_FULL, STATE_SCALE),
        ("state_over_time", TIME_SCALE, STATE_SCALE),
        ("state_over_quadratic", QUADRATIC_DRIFT_FULL_Q, STATE_SCALE),
        ("quadratic_over_full", OU_FULL, QUADRATIC_DRIFT_FULL_Q),
    )
    records = []
    for event_mean_removed in (
        (False, True)
        if report.config.run_event_mean_removed_sensitivity
        else (False,)
    ):
        for animal in ("all", "Chico", "Silas"):
            units = tuple(
                item
                for item in report.results
                if item.event_mean_removed == event_mean_removed
                and (animal == "all" or item.animal == animal)
            )
            for key, incumbent, candidate in pairings:
                component_deltas = {
                    component: tuple(
                        _unit_family_components_per_scalar(item, incumbent)[
                            component
                        ]
                        - _unit_family_components_per_scalar(item, candidate)[
                            component
                        ]
                        for item in units
                    )
                    for component in ("total", "nll", "complexity")
                }
                deltas = component_deltas["total"]
                records.append(
                    {
                        "event_mean_removed": event_mean_removed,
                        "animal": animal,
                        "comparison": key,
                        "unit_count": len(deltas),
                        "median_advantage_bits_per_scalar": float(median(deltas)),
                        "median_heldout_nll_advantage_bits_per_scalar": float(
                            median(component_deltas["nll"])
                        ),
                        "median_complexity_advantage_bits_per_scalar": float(
                            median(component_deltas["complexity"])
                        ),
                        "practical_win_fraction": (
                            sum(
                                value
                                > report.config.minimum_codelength_advantage_bits_per_scalar
                                for value in deltas
                            )
                            / len(deltas)
                        ),
                    }
                )
    return tuple(records)


def _unit_markov_bits(unit: DiffusionUnitResult) -> dict[int, float]:
    bits: dict[int, float] = {}
    scalars: dict[int, int] = {}
    for fold in unit.fold_results:
        for item in fold.markov_order_sensitivity:
            bits[item.order] = bits.get(item.order, 0.0) + item.score.total_codelength_bits
            scalars[item.order] = scalars.get(item.order, 0) + (
                item.score.test_vector_count * item.score.latent_rank
            )
    return {order: bits[order] / scalars[order] for order in bits}


def _markov_margins(report) -> dict[str, Any]:
    margins: dict[str, Any] = {}
    unit_codes = tuple(_unit_markov_bits(item) for item in report.results)
    for order in (2, 3):
        values = tuple(
            item[order] - item[1]
            for item in unit_codes
            if 1 in item and order in item
        )
        if values:
            margins[f"order1_advantage_over_order{order}"] = {
                "unit_count": len(values),
                "median_bits_per_scalar": float(median(values)),
                "positive_fraction": sum(value > 0.0 for value in values)
                / len(values),
            }
    return margins


def _unit_semigroup_excess(
    unit: DiffusionUnitResult,
    horizon: int,
) -> float | None:
    weighted_sum = 0.0
    scalar_count = 0
    for fold in unit.fold_results:
        matches = tuple(
            item
            for item in fold.semigroup_sensitivity
            if item.horizon_steps == horizon
        )
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError("semigroup horizon is not unique within a fold")
        item = matches[0]
        weight = item.test_vector_count * fold.latent_rank
        weighted_sum += item.frozen_excess_over_direct_bits_per_scalar * weight
        scalar_count += weight
    return None if scalar_count == 0 else float(weighted_sum / scalar_count)


def _semigroup_unit_summary(report) -> tuple[dict[str, Any], ...]:
    records = []
    for event_mean_removed in (
        (False, True)
        if report.config.run_event_mean_removed_sensitivity
        else (False,)
    ):
        for animal in ("all", "Chico", "Silas"):
            units = tuple(
                item
                for item in report.results
                if item.event_mean_removed == event_mean_removed
                and (animal == "all" or item.animal == animal)
            )
            for horizon in report.config.semigroup_horizons:
                values = tuple(
                    value
                    for item in units
                    if (value := _unit_semigroup_excess(item, horizon)) is not None
                )
                if values:
                    records.append(
                        {
                            "event_mean_removed": event_mean_removed,
                            "animal": animal,
                            "horizon_steps": horizon,
                            "unit_count": len(values),
                            "median_frozen_excess_bits_per_scalar": float(
                                median(values)
                            ),
                            "within_tolerance_fraction": (
                                sum(
                                    value
                                    <= report.config.semigroup_max_excess_bits_per_scalar
                                    for value in values
                                )
                                / len(values)
                            ),
                        }
                    )
    return tuple(records)


def _compact_payload(report) -> dict[str, Any]:
    markov = Counter(
        "TIE" if item.markov_order_vote is None else f"ORDER_{item.markov_order_vote}"
        for item in report.results
    )
    direction = Counter(
        "SKIPPED" if item.direction_vote is None else item.direction_vote
        for item in report.results
    )
    semigroup_records = tuple(
        sensitivity
        for unit in report.results
        for fold in unit.fold_results
        for sensitivity in fold.semigroup_sensitivity
    )
    semigroup = {
        str(horizon): {
            "comparison_count": sum(
                item.horizon_steps == horizon for item in semigroup_records
            ),
            "within_tolerance_fraction": (
                sum(
                    item.horizon_steps == horizon
                    and item.frozen_semigroup_within_tolerance
                    for item in semigroup_records
                )
                / max(
                    sum(item.horizon_steps == horizon for item in semigroup_records),
                    1,
                )
            ),
        }
        for horizon in report.config.semigroup_horizons
        if any(item.horizon_steps == horizon for item in semigroup_records)
    }
    return {
        "schema_version": report.schema_version,
        "method_status": report.method_status,
        "source_file_md5": report.source_file_md5,
        "official_checksum_verified": report.official_checksum_verified,
        "session_count": len(report.session_specs),
        "unit_count": len(report.results),
        "thresholds": {
            "minimum_codelength_advantage_bits_per_scalar": (
                report.config.minimum_codelength_advantage_bits_per_scalar
            ),
            "minimum_session_unit_win_fraction": (
                report.config.minimum_session_unit_win_fraction
            ),
            "semigroup_max_excess_bits_per_scalar": (
                report.config.semigroup_max_excess_bits_per_scalar
            ),
        },
        "aggregates": tuple(asdict(item) for item in report.aggregates),
        "covariance_comparisons": _covariance_comparisons(report),
        "markov_unit_classification": dict(sorted(markov.items())),
        "markov_unit_margins": _markov_margins(report),
        "direction_unit_classification": dict(sorted(direction.items())),
        "semigroup": semigroup,
        "semigroup_unit_summary": _semigroup_unit_summary(report),
        "verdicts": tuple(asdict(item) for item in report.verdicts),
        "claim_locks": asdict(report.claim_locks),
        "limitations": report.limitations,
        "conclusion": report.conclusion,
    }


def _print_report(payload: dict[str, Any]) -> None:
    print("TAFAZOLI DRIFT--DIFFUSION PROXY SCREEN")
    print(f"  checksum       {payload['source_file_md5']}")
    print(f"  sessions/units {payload['session_count']}/{payload['unit_count']}")
    print("  state-scale advantages (positive favors state-conditioned noise)")
    print("    eventmean  animal  >full       >time       >quadratic  joint wins")
    for item in payload["aggregates"]:
        print(
            f"    {int(item['event_mean_removed']):<10} "
            f"{item['animal']:<7} "
            f"{item['median_state_advantage_over_full_bits_per_scalar']:+.4f}      "
            f"{item['median_state_advantage_over_time_bits_per_scalar']:+.4f}      "
            f"{item['median_state_advantage_over_quadratic_bits_per_scalar']:+.4f}      "
            f"{item['joint_state_survivor_win_fraction']:.3f}"
        )
    print(f"  Markov units   {payload['markov_unit_classification']}")
    print(f"  direction      {payload['direction_unit_classification']}")
    print(f"  semigroup      {payload['semigroup']}")
    print("  verdicts")
    for verdict in payload["verdicts"]:
        print(f"    {verdict['answer']:<16} {verdict['key']}")
    print(f"  conclusion     {payload['conclusion']}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parser(repository_root).parse_args()
    config = DiffusionProbeConfig(
        run_event_mean_removed_sensitivity=(
            not args.no_event_mean_sensitivity
        ),
        run_markov_order_sensitivity=not args.no_markov_sensitivity,
        run_semigroup_sensitivity=not args.no_semigroup_sensitivity,
        run_reverse_classification=not args.no_reverse_classification,
    )
    source_file_md5 = verify_official_classifier_checksum(args.classifier_file)
    session_specs = recovered_session_specs()
    dimension_one = None
    dimension_three = None
    checkpoints = []
    for ordinal, session in enumerate(session_specs, start=1):
        path = _checkpoint_path(args.checkpoint_dir, session)
        if path.exists() and not args.fresh:
            checkpoint = _load_checkpoint(
                path,
                session=session,
                source_file_md5=source_file_md5,
                config=config,
            )
            action = "resumed"
        else:
            if dimension_one is None or dimension_three is None:
                dimension_one, dimension_three = load_tafazoli_train_dimensions(
                    args.classifier_file
                )
            columns = slice(
                session.column_start_zero_based,
                session.column_stop_exclusive,
            )
            checkpoint = run_diffusion_session_checkpoint(
                dimension_one[:, columns, :],
                dimension_three[:, columns, :],
                session=session,
                config=config,
                source_file_md5=source_file_md5,
            )
            _write_checkpoint(path, checkpoint)
            action = "computed"
        checkpoints.append(checkpoint)
        print(
            f"[{ordinal:02d}/{len(session_specs):02d}] "
            f"session {session.index_one_based:02d} {action}",
            file=sys.stderr,
            flush=True,
        )
    report = assemble_tafazoli_diffusion_report(
        checkpoints,
        config=config,
        session_specs=session_specs,
        source_file_md5=source_file_md5,
        official_checksum_verified=True,
    )
    payload = _compact_payload(report)
    summary_path = args.checkpoint_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    if args.json:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    else:
        _print_report(payload)


if __name__ == "__main__":
    main()
