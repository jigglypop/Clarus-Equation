"""Mouse IBL/OpenAlyx channel-region rescue gate.

The cross-session mouse gate left one important caveat: some probes have a
large ``unknown`` bin because ``clusters.brainLocationAcronyms_ccf_2017`` has
fewer rows than the observed spike cluster ids.  IBL also exposes
``clusters.channels`` and ``channels.brainLocationIds_ccf_2017``.  This gate
uses that channel-level atlas registration as a fallback and asks whether the
resulting rescued region-id code keeps the decision/action decoding effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_cross_session_region_generalization_gate import CANDIDATES
from mouse_ibl_multi_probe_region_gate import (
    DEFAULT_EID,
    DEFAULT_SESSION_REF,
    hstack,
    load_common,
    probe_label,
)
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    choice_target,
    evaluate_target,
    first_movement_speed_target,
    wheel_action_direction_target,
    window_bounds,
    window_features,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_channel_region_rescue_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_channel_region_rescue_report.md")
UNKNOWN_LABEL = "unknown"
OTHER_LOW_SPIKE_LABEL = "other_low_spike_labels"
UNKNOWN_REGION_ID = -1


def valid_acronym(acronym: object) -> bool:
    text = str(acronym)
    return text not in {"", "nan", "None", "void", "root"}


def valid_region_id(value: object) -> bool:
    try:
        region_id = int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return region_id > 0


def load_probe(one, eid: str, collection: str) -> dict[str, object]:
    return {
        "collection": collection,
        "label": probe_label(collection),
        "spike_times": np.asarray(
            one.load_dataset(
                eid,
                "spikes.times.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=float,
        ),
        "spike_clusters": np.asarray(
            one.load_dataset(
                eid,
                "spikes.clusters.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=np.int64,
        ),
        "cluster_acronyms": np.asarray(
            one.load_dataset(
                eid,
                "clusters.brainLocationAcronyms_ccf_2017.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=object,
        ),
        "cluster_channels": np.asarray(
            one.load_dataset(
                eid,
                "clusters.channels.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=np.int64,
        ),
        "channel_region_ids": np.asarray(
            one.load_dataset(
                eid,
                "channels.brainLocationIds_ccf_2017.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=np.int64,
        ),
    }


def cluster_strict_labels(cluster_acronyms: np.ndarray, max_cluster_id: int) -> np.ndarray:
    labels = np.full(max_cluster_id + 1, UNKNOWN_LABEL, dtype=object)
    for cluster_id, acronym in enumerate(cluster_acronyms):
        if valid_acronym(acronym):
            labels[cluster_id] = f"acronym:{acronym}"
    return labels


def cluster_channel_region_ids(
    cluster_channels: np.ndarray,
    channel_region_ids: np.ndarray,
    max_cluster_id: int,
) -> np.ndarray:
    region_ids = np.full(max_cluster_id + 1, UNKNOWN_REGION_ID, dtype=np.int64)
    n_clusters = min(len(cluster_channels), max_cluster_id + 1)
    for cluster_id in range(n_clusters):
        channel = int(cluster_channels[cluster_id])
        if 0 <= channel < len(channel_region_ids) and valid_region_id(channel_region_ids[channel]):
            region_ids[cluster_id] = int(channel_region_ids[channel])
    return region_ids


def cluster_hybrid_labels(
    strict_labels: np.ndarray,
    channel_region_ids: np.ndarray,
) -> np.ndarray:
    labels = strict_labels.copy()
    missing = labels == UNKNOWN_LABEL
    rescue = missing & (channel_region_ids != UNKNOWN_REGION_ID)
    labels[rescue] = [f"ccf_id:{region_id}" for region_id in channel_region_ids[rescue]]
    return labels


def label_groups(
    cluster_labels: np.ndarray,
    spike_clusters: np.ndarray,
    min_spikes: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    labels, counts = np.unique(cluster_labels[spike_clusters], return_counts=True)
    spike_counts = {str(label): int(count) for label, count in zip(labels, counts)}
    ranked = sorted(
        (
            (label, count)
            for label, count in spike_counts.items()
            if label != UNKNOWN_LABEL and count >= min_spikes
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    group_names = [label for label, _ in ranked]
    low_spike_known = [
        label
        for label, count in spike_counts.items()
        if label != UNKNOWN_LABEL and count < min_spikes
    ]
    if low_spike_known:
        group_names.append(OTHER_LOW_SPIKE_LABEL)
    group_names.append(UNKNOWN_LABEL)

    group_lookup = {name: idx for idx, name in enumerate(group_names)}
    other_idx = group_lookup.get(OTHER_LOW_SPIKE_LABEL, group_lookup[UNKNOWN_LABEL])
    unknown_idx = group_lookup[UNKNOWN_LABEL]
    cluster_group = np.full(len(cluster_labels), unknown_idx, dtype=np.int16)
    for cluster_id, label in enumerate(cluster_labels):
        if label == UNKNOWN_LABEL:
            cluster_group[cluster_id] = unknown_idx
        elif label in group_lookup:
            cluster_group[cluster_id] = group_lookup[label]
        else:
            cluster_group[cluster_id] = other_idx
    return group_names, cluster_group, spike_counts


def region_id_labels(region_ids: np.ndarray) -> np.ndarray:
    labels = np.full(len(region_ids), UNKNOWN_LABEL, dtype=object)
    known = region_ids != UNKNOWN_REGION_ID
    labels[known] = [f"ccf_id:{region_id}" for region_id in region_ids[known]]
    return labels


def spike_fraction(labels: np.ndarray, spike_clusters: np.ndarray, label: str) -> float:
    return float(np.mean(labels[spike_clusters] == label))


def count_known(labels: np.ndarray, spike_clusters: np.ndarray) -> int:
    return int(np.sum(labels[spike_clusters] != UNKNOWN_LABEL))


def top_counts(counts: dict[str, int], limit: int = 12) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


def probe_feature_block(
    probe: dict[str, object],
    starts: np.ndarray,
    ends: np.ndarray,
    valid_window: np.ndarray,
    min_label_spikes: int,
) -> dict[str, object]:
    spike_times = probe["spike_times"]
    spike_clusters = probe["spike_clusters"]
    max_cluster_id = int(np.max(spike_clusters))
    strict_labels = cluster_strict_labels(probe["cluster_acronyms"], max_cluster_id)
    channel_region_ids = cluster_channel_region_ids(
        probe["cluster_channels"],
        probe["channel_region_ids"],
        max_cluster_id,
    )
    channel_labels = region_id_labels(channel_region_ids)
    hybrid_labels = cluster_hybrid_labels(strict_labels, channel_region_ids)

    strict_names, strict_group, strict_counts = label_groups(
        strict_labels,
        spike_clusters,
        min_label_spikes,
    )
    channel_names, channel_group, channel_counts = label_groups(
        channel_labels,
        spike_clusters,
        min_label_spikes,
    )
    hybrid_names, hybrid_group, hybrid_counts = label_groups(
        hybrid_labels,
        spike_clusters,
        min_label_spikes,
    )
    return {
        "strict_acronym_features": window_features(
            spike_times,
            spike_clusters,
            strict_group,
            starts,
            ends,
            valid_window,
            len(strict_names),
        ),
        "channel_region_id_features": window_features(
            spike_times,
            spike_clusters,
            channel_group,
            starts,
            ends,
            valid_window,
            len(channel_names),
        ),
        "hybrid_acronym_channel_id_features": window_features(
            spike_times,
            spike_clusters,
            hybrid_group,
            starts,
            ends,
            valid_window,
            len(hybrid_names),
        ),
        "strict_names": strict_names,
        "channel_names": channel_names,
        "hybrid_names": hybrid_names,
        "strict_counts": strict_counts,
        "channel_counts": channel_counts,
        "hybrid_counts": hybrid_counts,
        "strict_labels": strict_labels,
        "channel_labels": channel_labels,
        "hybrid_labels": hybrid_labels,
    }


def make_models(blocks: list[dict[str, object]]) -> dict[str, np.ndarray]:
    strict = hstack([block["strict_acronym_features"] for block in blocks])
    channel = hstack([block["channel_region_id_features"] for block in blocks])
    hybrid = hstack([block["hybrid_acronym_channel_id_features"] for block in blocks])
    return {
        "strict_acronym_by_probe": strict,
        "channel_region_id_by_probe": channel,
        "hybrid_acronym_channel_id_by_probe": hybrid,
        "global_rate": np.nansum(hybrid, axis=1, keepdims=True),
    }


def summarize_probe(probe: dict[str, object], stimulus_block: dict[str, object]) -> dict[str, object]:
    spike_clusters = probe["spike_clusters"]
    strict_unknown_fraction = spike_fraction(
        stimulus_block["strict_labels"],
        spike_clusters,
        UNKNOWN_LABEL,
    )
    channel_unknown_fraction = spike_fraction(
        stimulus_block["channel_labels"],
        spike_clusters,
        UNKNOWN_LABEL,
    )
    hybrid_unknown_fraction = spike_fraction(
        stimulus_block["hybrid_labels"],
        spike_clusters,
        UNKNOWN_LABEL,
    )
    strict_unknown_spikes = int(np.sum(stimulus_block["strict_labels"][spike_clusters] == UNKNOWN_LABEL))
    hybrid_unknown_spikes = int(np.sum(stimulus_block["hybrid_labels"][spike_clusters] == UNKNOWN_LABEL))
    rescued_spikes = strict_unknown_spikes - hybrid_unknown_spikes
    return {
        "collection": probe["collection"],
        "label": probe["label"],
        "spike_count": int(len(probe["spike_times"])),
        "observed_cluster_slots": int(np.max(spike_clusters) + 1),
        "strict_cluster_acronym_rows": int(len(probe["cluster_acronyms"])),
        "cluster_channel_rows": int(len(probe["cluster_channels"])),
        "channel_region_id_rows": int(len(probe["channel_region_ids"])),
        "strict_known_spikes": count_known(stimulus_block["strict_labels"], spike_clusters),
        "channel_known_spikes": count_known(stimulus_block["channel_labels"], spike_clusters),
        "hybrid_known_spikes": count_known(stimulus_block["hybrid_labels"], spike_clusters),
        "strict_unknown_spikes": strict_unknown_spikes,
        "hybrid_unknown_spikes": hybrid_unknown_spikes,
        "rescued_unknown_spikes": rescued_spikes,
        "strict_unknown_fraction": strict_unknown_fraction,
        "channel_unknown_fraction": channel_unknown_fraction,
        "hybrid_unknown_fraction": hybrid_unknown_fraction,
        "rescued_unknown_fraction_of_all_spikes": float(rescued_spikes / len(spike_clusters)),
        "rescued_fraction_of_strict_unknown": float(
            rescued_spikes / max(strict_unknown_spikes, 1)
        ),
        "strict_feature_count": int(len(stimulus_block["strict_names"])),
        "channel_feature_count": int(len(stimulus_block["channel_names"])),
        "hybrid_feature_count": int(len(stimulus_block["hybrid_names"])),
        "top_strict_counts": top_counts(stimulus_block["strict_counts"]),
        "top_channel_counts": top_counts(stimulus_block["channel_counts"]),
        "top_hybrid_counts": top_counts(stimulus_block["hybrid_counts"]),
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    wheel_timestamps = np.asarray(common["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(common["wheel_position"], dtype=float)
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]

    stim_start, stim_end, stim_valid = window_bounds(trials, STIMULUS_WINDOW)
    move_start, move_end, move_valid = window_bounds(trials, MOVEMENT_WINDOW)
    stimulus_blocks = [
        probe_feature_block(
            probe,
            stim_start,
            stim_end,
            stim_valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]
    movement_blocks = [
        probe_feature_block(
            probe,
            move_start,
            move_end,
            move_valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]

    choice, choice_valid, choice_meta = choice_target(trials)
    speed, speed_valid, speed_meta = first_movement_speed_target(trials)
    wheel_direction, wheel_valid, wheel_meta = wheel_action_direction_target(
        trials,
        wheel_timestamps,
        wheel_position,
        args.min_abs_wheel_displacement,
    )
    stimulus_models = make_models(stimulus_blocks)
    movement_models = make_models(movement_blocks)
    targets = [
        evaluate_target(
            target_name="choice_sign",
            window_name=STIMULUS_WINDOW.name,
            x_models=stimulus_models,
            y_all=choice,
            valid=choice_valid & stim_valid,
            folds=args.folds,
            ridge=args.ridge,
            permutations=args.permutations,
            seed=args.seed,
        ),
        evaluate_target(
            target_name="first_movement_speed",
            window_name=STIMULUS_WINDOW.name,
            x_models=stimulus_models,
            y_all=speed,
            valid=speed_valid & stim_valid,
            folds=args.folds,
            ridge=args.ridge,
            permutations=args.permutations,
            seed=args.seed,
        ),
        evaluate_target(
            target_name="wheel_action_direction",
            window_name=MOVEMENT_WINDOW.name,
            x_models=movement_models,
            y_all=wheel_direction,
            valid=wheel_valid & move_valid,
            folds=args.folds,
            ridge=args.ridge,
            permutations=args.permutations,
            seed=args.seed,
        ),
    ]
    probe_summaries = [
        summarize_probe(probe, stimulus_block)
        for probe, stimulus_block in zip(probes, stimulus_blocks)
    ]
    strict_unknown = sum(item["strict_unknown_spikes"] for item in probe_summaries)
    hybrid_unknown = sum(item["hybrid_unknown_spikes"] for item in probe_summaries)
    spike_count = sum(item["spike_count"] for item in probe_summaries)
    passed_rows = [
        f"{target['target']}:{row['model']}"
        for target in targets
        for row in target["rows"]
        if row["passed"]
    ]
    return {
        "openalyx_url": OPENALYX_URL,
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "wheel_sample_count": int(len(wheel_timestamps)),
        "probe_summaries": probe_summaries,
        "total_spike_count": int(spike_count),
        "strict_unknown_spikes": int(strict_unknown),
        "hybrid_unknown_spikes": int(hybrid_unknown),
        "rescued_unknown_spikes": int(strict_unknown - hybrid_unknown),
        "strict_unknown_fraction": float(strict_unknown / spike_count),
        "hybrid_unknown_fraction": float(hybrid_unknown / spike_count),
        "rescued_fraction_of_all_spikes": float((strict_unknown - hybrid_unknown) / spike_count),
        "rescued_fraction_of_strict_unknown": float(
            (strict_unknown - hybrid_unknown) / max(strict_unknown, 1)
        ),
        "target_metadata": {
            "choice_sign": choice_meta,
            "first_movement_speed": speed_meta,
            "wheel_action_direction": wheel_meta,
        },
        "folds": int(args.folds),
        "ridge": float(args.ridge),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "targets": targets,
        "passed_rows": passed_rows,
        "passed": bool(passed_rows),
    }


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    if candidate["kind"] == "single":
        collections = [candidate["collection"]]
    else:
        collections = candidate["collections"]
    return SimpleNamespace(
        eid=candidate["eid"],
        session_ref=candidate["session_ref"],
        collections=collections,
        folds=args.folds,
        ridge=args.ridge,
        permutations=args.permutations,
        seed=args.seed,
        min_label_spikes=args.min_label_spikes,
        min_abs_wheel_displacement=args.min_abs_wheel_displacement,
    )


def best_rows_by_target(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        by_model = {row["model"]: row for row in target["rows"]}
        strict = by_model["strict_acronym_by_probe"]
        channel = by_model["channel_region_id_by_probe"]
        hybrid = by_model["hybrid_acronym_channel_id_by_probe"]
        global_row = by_model["global_rate"]
        best = max(
            [strict, channel, hybrid],
            key=lambda row: row["balanced_accuracy"],
        )
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "strict_balanced_accuracy": strict["balanced_accuracy"],
                "channel_balanced_accuracy": channel["balanced_accuracy"],
                "hybrid_balanced_accuracy": hybrid["balanced_accuracy"],
                "hybrid_auc": hybrid["auc"],
                "hybrid_p_balanced_accuracy_ge_observed": hybrid[
                    "p_balanced_accuracy_ge_observed"
                ],
                "hybrid_delta_vs_global_rate": hybrid["balanced_accuracy"]
                - global_row["balanced_accuracy"],
                "hybrid_delta_vs_strict": hybrid["balanced_accuracy"]
                - strict["balanced_accuracy"],
                "global_balanced_accuracy": global_row["balanced_accuracy"],
                "hybrid_passed": bool(hybrid["passed"]),
                "best_model": best["model"],
                "best_balanced_accuracy": best["balanced_accuracy"],
                "n_trials": hybrid["n_trials"],
                "class_counts": hybrid["class_counts"],
            }
        )
    return rows


def candidate_summary(candidate: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    target_rows = best_rows_by_target(result)
    return {
        "name": candidate["name"],
        "kind": candidate["kind"],
        "eid": candidate["eid"],
        "session_ref": candidate["session_ref"],
        "collections": result["collections"],
        "reason": candidate["reason"],
        "trial_count": result["trial_count"],
        "strict_unknown_fraction": result["strict_unknown_fraction"],
        "hybrid_unknown_fraction": result["hybrid_unknown_fraction"],
        "rescued_fraction_of_strict_unknown": result[
            "rescued_fraction_of_strict_unknown"
        ],
        "rescued_unknown_spikes": result["rescued_unknown_spikes"],
        "hybrid_passed_target_count": sum(row["hybrid_passed"] for row in target_rows),
        "hybrid_all_targets_passed": all(row["hybrid_passed"] for row in target_rows),
        "target_rows": target_rows,
    }


def strip_for_panel(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"target_metadata"}
    }


def aggregate(candidates: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    results = []
    summaries = []
    for candidate in candidates:
        result = evaluate_session(candidate_namespace(candidate, args))
        results.append({"candidate": candidate, "result": strip_for_panel(result)})
        summaries.append(candidate_summary(candidate, result))

    total_spikes = sum(item["result"]["total_spike_count"] for item in results)
    strict_unknown = sum(item["result"]["strict_unknown_spikes"] for item in results)
    hybrid_unknown = sum(item["result"]["hybrid_unknown_spikes"] for item in results)
    target_names = sorted({row["target"] for item in summaries for row in item["target_rows"]})
    target_replication = {}
    for target in target_names:
        rows = [
            row for item in summaries for row in item["target_rows"] if row["target"] == target
        ]
        target_replication[target] = {
            "candidate_count": len(rows),
            "hybrid_passed_count": sum(row["hybrid_passed"] for row in rows),
            "mean_strict_balanced_accuracy": sum(
                row["strict_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_channel_balanced_accuracy": sum(
                row["channel_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_hybrid_balanced_accuracy": sum(
                row["hybrid_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_hybrid_delta_vs_strict": sum(
                row["hybrid_delta_vs_strict"] for row in rows
            )
            / len(rows),
            "mean_hybrid_delta_vs_global_rate": sum(
                row["hybrid_delta_vs_global_rate"] for row in rows
            )
            / len(rows),
        }

    hybrid_all_targets = sum(item["hybrid_all_targets_passed"] for item in summaries)
    coverage_passed = hybrid_unknown < strict_unknown and (
        (strict_unknown - hybrid_unknown) / max(strict_unknown, 1)
    ) >= args.min_rescue_fraction
    decoding_passed = (
        hybrid_all_targets >= max(2, len(summaries) // 2)
        and target_replication.get("choice_sign", {}).get("hybrid_passed_count", 0) >= 3
        and target_replication.get("wheel_action_direction", {}).get(
            "hybrid_passed_count",
            0,
        )
        >= 3
    )
    return {
        "candidate_count": len(candidates),
        "folds": args.folds,
        "ridge": args.ridge,
        "permutations": args.permutations,
        "seed": args.seed,
        "min_label_spikes": args.min_label_spikes,
        "min_rescue_fraction": args.min_rescue_fraction,
        "total_spike_count": int(total_spikes),
        "strict_unknown_spikes": int(strict_unknown),
        "hybrid_unknown_spikes": int(hybrid_unknown),
        "rescued_unknown_spikes": int(strict_unknown - hybrid_unknown),
        "strict_unknown_fraction": float(strict_unknown / total_spikes),
        "hybrid_unknown_fraction": float(hybrid_unknown / total_spikes),
        "rescued_fraction_of_all_spikes": float(
            (strict_unknown - hybrid_unknown) / total_spikes
        ),
        "rescued_fraction_of_strict_unknown": float(
            (strict_unknown - hybrid_unknown) / max(strict_unknown, 1)
        ),
        "hybrid_all_targets_count": int(hybrid_all_targets),
        "coverage_passed": bool(coverage_passed),
        "decoding_passed": bool(decoding_passed),
        "rescue_gate_passed": bool(coverage_passed and decoding_passed),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx channel-region rescue gate",
        "",
        "Cross-session gate의 큰 제한이던 strict acronym `unknown` bin을 줄이기 위해 channel-level CCF id fallback을 추가했다.",
        "Strict acronym이 있는 cluster는 그대로 두고, strict acronym이 없는 cluster만 `clusters.channels -> channels.brainLocationIds_ccf_2017`로 복구한다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- min label spikes: {output['min_label_spikes']}",
        f"- coverage passed: `{output['coverage_passed']}`",
        f"- decoding passed: `{output['decoding_passed']}`",
        f"- rescue gate passed: `{output['rescue_gate_passed']}`",
        "",
        "## coverage rescue",
        "",
        "| item | value |",
        "|---|---:|",
        f"| total spikes | {output['total_spike_count']} |",
        f"| strict unknown spikes | {output['strict_unknown_spikes']} |",
        f"| hybrid unknown spikes | {output['hybrid_unknown_spikes']} |",
        f"| rescued unknown spikes | {output['rescued_unknown_spikes']} |",
        f"| strict unknown fraction | {output['strict_unknown_fraction']:.6f} |",
        f"| hybrid unknown fraction | {output['hybrid_unknown_fraction']:.6f} |",
        f"| rescued fraction of strict unknown | {output['rescued_fraction_of_strict_unknown']:.6f} |",
        "",
        "The hybrid cluster map is",
        "",
        "$$",
        "G_{\\mathrm{hybrid}}(c)=\\begin{cases}",
        "A(c),&A(c)\\neq\\varnothing,\\\\",
        "I(\\chi(c)),&A(c)=\\varnothing\\ \\mathrm{and}\\ I(\\chi(c))>0,\\\\",
        "\\varnothing,&\\mathrm{otherwise}.",
        "\\end{cases}",
        "$$",
        "",
        "Here \(A(c)\) is the strict cluster acronym, \\(\\chi(c)\\) is the channel assigned to cluster \\(c\\), and \\(I(\\chi(c))\\) is the channel CCF region id.",
        "The rescued feature keeps the same duration-normalized trial-window form:",
        "",
        "$$",
        "x_{ipg}^{\\mathrm{hybrid}}=\\frac{1}{b_i-a_i}\\sum_k \\mathbf 1[t_{pk}\\in[a_i,b_i]]\\mathbf 1[G_{\\mathrm{hybrid},p}(c_{pk})=g].",
        "$$",
        "",
        "## target replication",
        "",
        "| target | candidates | hybrid passed | mean strict BA | mean channel BA | mean hybrid BA | mean hybrid delta strict | mean hybrid delta global |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["hybrid_passed_count"]),
                    f"{row['mean_strict_balanced_accuracy']:.6f}",
                    f"{row['mean_channel_balanced_accuracy']:.6f}",
                    f"{row['mean_hybrid_balanced_accuracy']:.6f}",
                    f"{row['mean_hybrid_delta_vs_strict']:.6f}",
                    f"{row['mean_hybrid_delta_vs_global_rate']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | strict unknown | hybrid unknown | rescued strict unknown | hybrid passed targets | choice hybrid BA | speed hybrid BA | wheel hybrid BA |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in output["summaries"]:
        by_target = {row["target"]: row for row in summary["target_rows"]}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['name']}`",
                    str(summary["trial_count"]),
                    f"{summary['strict_unknown_fraction']:.6f}",
                    f"{summary['hybrid_unknown_fraction']:.6f}",
                    f"{summary['rescued_fraction_of_strict_unknown']:.6f}",
                    str(summary["hybrid_passed_target_count"]),
                    f"{by_target['choice_sign']['hybrid_balanced_accuracy']:.6f}",
                    f"{by_target['first_movement_speed']['hybrid_balanced_accuracy']:.6f}",
                    f"{by_target['wheel_action_direction']['hybrid_balanced_accuracy']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## per-probe coverage", ""])
    for item in output["results"]:
        candidate = item["candidate"]
        result = item["result"]
        lines.extend(
            [
                f"### {candidate['name']}",
                "",
                f"- eid: `{candidate['eid']}`",
                f"- session: `{candidate['session_ref']}`",
                f"- reason: {candidate['reason']}",
                "",
                "| probe | spikes | strict rows | cluster-channel rows | CCF id rows | strict unknown | hybrid unknown | rescued strict unknown | hybrid features |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for probe in result["probe_summaries"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{probe['collection']}`",
                        str(probe["spike_count"]),
                        str(probe["strict_cluster_acronym_rows"]),
                        str(probe["cluster_channel_rows"]),
                        str(probe["channel_region_id_rows"]),
                        f"{probe['strict_unknown_fraction']:.6f}",
                        f"{probe['hybrid_unknown_fraction']:.6f}",
                        f"{probe['rescued_fraction_of_strict_unknown']:.6f}",
                        str(probe["hybrid_feature_count"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## verdict",
            "",
            f"- coverage passed: `{output['coverage_passed']}`",
            f"- decoding passed: `{output['decoding_passed']}`",
            f"- rescue gate passed: `{output['rescue_gate_passed']}`",
            "",
            "해석:",
            "",
            "- Channel fallback은 strict acronym 밖으로 빠진 spike를 버리지 않고 CCF id bin으로 되살린다.",
            "- 따라서 `unknown`은 더 이상 cluster acronym row 부족만 의미하지 않고, channel CCF id까지 없는 진짜 미등록 잔차가 된다.",
            "- Hybrid code가 여러 target에서 global-rate baseline을 계속 넘으면, mouse 항의 session residual \\(\\epsilon_s\\) 중 `unknown` 성분은 관측 가능 registration error로 한 단계 분해된다.",
        ]
    )
    return "\n".join(lines) + "\n"


def selected_candidates(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.single_session:
        return [
            {
                "name": "nyu30_motor_striatal_multi_probe",
                "kind": "multi",
                "eid": DEFAULT_EID,
                "session_ref": DEFAULT_SESSION_REF,
                "collections": args.collections,
                "reason": "same-session motor cortex plus striatal/septal multi-probe bridge",
            }
        ]
    return CANDIDATES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument("--collections", nargs="+", default=["alf/probe00/pykilosort", "alf/probe01/pykilosort"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--min-label-spikes", type=int, default=100_000)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=1e-3)
    parser.add_argument("--min-rescue-fraction", type=float, default=0.50)
    args = parser.parse_args()

    output = aggregate(selected_candidates(args), args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx channel-region rescue gate")
    print(f"  candidates={output['candidate_count']}")
    print(
        "  unknown "
        + f"strict={output['strict_unknown_fraction']:.6f} "
        + f"hybrid={output['hybrid_unknown_fraction']:.6f} "
        + f"rescued={output['rescued_fraction_of_strict_unknown']:.6f}"
    )
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} hybrid_passed={row['hybrid_passed_count']}/"
            + f"{row['candidate_count']} mean_hybrid_BA="
            + f"{row['mean_hybrid_balanced_accuracy']:.6f}"
        )
    print(f"  rescue_gate_passed={output['rescue_gate_passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
