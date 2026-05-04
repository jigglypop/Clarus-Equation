"""Mouse IBL/OpenAlyx multi-probe motor/striatal region gate.

This follows the motor-striatum audit.  The selected NYU-30 session has one
probe with motor cortex coverage and another with striatal/septal coverage.
The gate concatenates probe-level region features and asks whether this
multi-probe region code predicts decision/action variables better than a
global firing-rate baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mouse_ibl_region_decision_action_gate import (
    FAMILY_NAMES,
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    choice_target,
    cluster_acronym_groups,
    cluster_family_ids,
    evaluate_target,
    first_movement_speed_target,
    wheel_action_direction_target,
    window_bounds,
    window_features,
)


DEFAULT_EID = "5ec72172-3901-4771-8777-6e9490ca51fc"
DEFAULT_SESSION_REF = "angelakilab/Subjects/NYU-30/2020-10-22/001"
DEFAULT_COLLECTIONS = ["alf/probe00/pykilosort", "alf/probe01/pykilosort"]
RESULT_JSON = Path(__file__).with_name("mouse_ibl_multi_probe_region_gate_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_multi_probe_region_gate_report.md")
TARGET_FAMILY_NAMES = [
    "motor_cortex",
    "striatal_complex",
    "septal_subpallium",
    "basal_ganglia_output",
]


def probe_label(collection: str) -> str:
    parts = collection.split("/")
    for part in parts:
        if part.startswith("probe"):
            return part
    return collection.replace("/", "_")


def load_common(one, eid: str) -> dict[str, object]:
    return {
        "trials": one.load_dataset(
            eid, "_ibl_trials.table.pqt", collection="alf", query_type="remote"
        ),
        "wheel_timestamps": one.load_dataset(
            eid, "_ibl_wheel.timestamps.npy", collection="alf", query_type="remote"
        ),
        "wheel_position": one.load_dataset(
            eid, "_ibl_wheel.position.npy", collection="alf", query_type="remote"
        ),
    }


def load_probe(one, eid: str, collection: str) -> dict[str, np.ndarray]:
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
    }


def hstack(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.empty((0, 0), dtype=float)
    return np.column_stack(items)


def probe_feature_block(
    probe: dict[str, object],
    starts: np.ndarray,
    ends: np.ndarray,
    valid_window: np.ndarray,
    min_acronym_spikes: int,
) -> dict[str, object]:
    spike_times = probe["spike_times"]
    spike_clusters = probe["spike_clusters"]
    cluster_acronyms = probe["cluster_acronyms"]
    max_cluster_id = int(np.max(spike_clusters))
    family_ids = cluster_family_ids(cluster_acronyms, max_cluster_id)
    family_features = window_features(
        spike_times,
        spike_clusters,
        family_ids,
        starts,
        ends,
        valid_window,
        len(FAMILY_NAMES),
    )
    acronym_names, acronym_ids, acronym_spike_counts = cluster_acronym_groups(
        cluster_acronyms,
        spike_clusters,
        max_cluster_id,
        min_acronym_spikes,
    )
    acronym_features = window_features(
        spike_times,
        spike_clusters,
        acronym_ids,
        starts,
        ends,
        valid_window,
        len(acronym_names),
    )
    family_spike_counts = np.bincount(
        family_ids[spike_clusters],
        minlength=len(FAMILY_NAMES),
    )
    return {
        "family_features": family_features,
        "acronym_features": acronym_features,
        "family_feature_names": [
            f"{probe['label']}:{family}" for family in FAMILY_NAMES
        ],
        "acronym_feature_names": [
            f"{probe['label']}:{name}" for name in acronym_names
        ],
        "acronym_spike_counts": {
            f"{probe['label']}:{name}": int(count)
            for name, count in acronym_spike_counts.items()
        },
        "family_spike_counts": {
            FAMILY_NAMES[idx]: int(family_spike_counts[idx])
            for idx in range(len(FAMILY_NAMES))
        },
    }


def combine_family_by_probe(blocks: list[dict[str, object]]) -> np.ndarray:
    return hstack([block["family_features"] for block in blocks])


def combine_family_collapsed(blocks: list[dict[str, object]]) -> np.ndarray:
    result = np.zeros_like(blocks[0]["family_features"])
    for block in blocks:
        result = result + block["family_features"]
    return result


def combine_target_families_by_probe(blocks: list[dict[str, object]]) -> np.ndarray:
    indices = [FAMILY_NAMES.index(name) for name in TARGET_FAMILY_NAMES]
    return hstack([block["family_features"][:, indices] for block in blocks])


def combine_acronym_by_probe(blocks: list[dict[str, object]]) -> np.ndarray:
    return hstack([block["acronym_features"] for block in blocks])


def make_models(blocks: list[dict[str, object]]) -> dict[str, np.ndarray]:
    family_by_probe = combine_family_by_probe(blocks)
    family_collapsed = combine_family_collapsed(blocks)
    target_family_by_probe = combine_target_families_by_probe(blocks)
    acronym_by_probe = combine_acronym_by_probe(blocks)
    return {
        "family_collapsed": family_collapsed,
        "family_by_probe": family_by_probe,
        "motor_striatal_family_by_probe": target_family_by_probe,
        "acronym_by_probe": acronym_by_probe,
        "global_rate": np.nansum(family_collapsed, axis=1, keepdims=True),
    }


def summarize_probe(probe: dict[str, object], stimulus_block: dict[str, object]) -> dict[str, object]:
    spike_clusters = probe["spike_clusters"]
    strict_labeled = int(np.sum(spike_clusters < len(probe["cluster_acronyms"])))
    target_spikes = sum(
        stimulus_block["family_spike_counts"].get(family, 0)
        for family in TARGET_FAMILY_NAMES
    )
    top_families = sorted(
        stimulus_block["family_spike_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    top_acronyms = sorted(
        stimulus_block["acronym_spike_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    return {
        "collection": probe["collection"],
        "label": probe["label"],
        "spike_count": int(len(probe["spike_times"])),
        "observed_cluster_slots": int(np.max(spike_clusters) + 1),
        "strict_cluster_acronym_rows": int(len(probe["cluster_acronyms"])),
        "strict_labeled_spike_count": strict_labeled,
        "unknown_spike_count": int(len(spike_clusters) - strict_labeled),
        "unknown_spike_fraction": float((len(spike_clusters) - strict_labeled) / len(spike_clusters)),
        "target_family_spike_count": int(target_spikes),
        "top_family_spike_counts": top_families,
        "top_acronym_spike_counts": top_acronyms,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
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
            args.min_acronym_spikes,
        )
        for probe in probes
    ]
    movement_blocks = [
        probe_feature_block(
            probe,
            move_start,
            move_end,
            move_valid,
            args.min_acronym_spikes,
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
        "probe_summaries": [
            summarize_probe(probe, stimulus_block)
            for probe, stimulus_block in zip(probes, stimulus_blocks)
        ],
        "family_names": FAMILY_NAMES,
        "target_family_names": TARGET_FAMILY_NAMES,
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


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx multi-probe motor-striatal region gate",
        "",
        "Motor-striatum audit에서 선택한 NYU-30 session의 두 probe를 같은 trial table 위에 결합했다.",
        "각 probe에서 region family와 acronym firing-rate feature를 만들고, probe별 feature를 이어 붙여 global-rate baseline과 비교한다.",
        "",
        "## source",
        "",
        f"- OpenAlyx: `{output['openalyx_url']}`",
        f"- eid: `{output['eid']}`",
        f"- session: `{output['session_ref']}`",
        f"- collections: {', '.join(f'`{collection}`' for collection in output['collections'])}",
        "",
        "## loaded arrays",
        "",
        f"- trials: {output['trial_count']}",
        f"- wheel samples: {output['wheel_sample_count']}",
        "",
        "| probe | spikes | cluster slots | acronym rows | unknown fraction | target-family spikes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for probe in output["probe_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{probe['collection']}`",
                    str(probe["spike_count"]),
                    str(probe["observed_cluster_slots"]),
                    str(probe["strict_cluster_acronym_rows"]),
                    f"{probe['unknown_spike_fraction']:.6f}",
                    str(probe["target_family_spike_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## feature construction",
            "",
            "For probe \(p\), trial \(i\), and region group \(g\):",
            "",
            "$$",
            "x_{ipg}=\\frac{1}{b_i-a_i}\\sum_k \\mathbf 1[t_{pk}\\in[a_i,b_i]]\\mathbf 1[G_p(c_{pk})=g].",
            "$$",
            "",
            "The multi-probe region vector is",
            "",
            "$$",
            "R_i^{\\mathrm{multi}}=[x_{i,\\mathrm{probe00},:},x_{i,\\mathrm{probe01},:}].",
            "$$",
            "",
            "The compared models are `family_collapsed`, `family_by_probe`, `motor_striatal_family_by_probe`, `acronym_by_probe`, and `global_rate`.",
            "",
            "## probe region summaries",
            "",
        ]
    )
    for probe in output["probe_summaries"]:
        lines.extend(
            [
                f"### {probe['collection']}",
                "",
                "| family | spikes |",
                "|---|---:|",
            ]
        )
        for family, count in probe["top_family_spike_counts"]:
            lines.append(f"| `{family}` | {count} |")
        lines.extend(["", "| acronym group | spikes |", "|---|---:|"])
        for acronym, count in probe["top_acronym_spike_counts"]:
            lines.append(f"| `{acronym}` | {count} |")
        lines.append("")

    lines.extend(["## decoder results", ""])
    for target in output["targets"]:
        meta = output["target_metadata"][target["target"]]
        lines.extend(
            [
                f"### {target['target']}",
                "",
                f"- window: `{target['window']}`",
                f"- definition: {meta['definition']}",
            ]
        )
        if "median_seconds" in meta:
            lines.append(f"- median seconds: {meta['median_seconds']:.6f}")
        if "min_abs_displacement" in meta:
            lines.append(f"- min absolute wheel displacement: {meta['min_abs_displacement']:.6f}")
        lines.extend(
            [
                "",
                "| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |",
                "|---|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in target["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['model']}`",
                        str(row["n_trials"]),
                        "`" + json.dumps(row["class_counts"], sort_keys=True) + "`",
                        f"{row['balanced_accuracy']:.6f}",
                        f"{row['auc']:.6f}",
                        f"{row['permutation_balanced_accuracy_mean']:.6f}",
                        f"{row['p_balanced_accuracy_ge_observed']:.6f}",
                        f"{row['delta_vs_global_rate']:.6f}",
                        str(row["passed"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## verdict",
            "",
            f"- gate passed: `{output['passed']}`",
            f"- passed non-global rows: {output['passed_rows']}",
            "",
            "이 gate는 첫 thalamic/visual probe 결과를 motor/striatal multi-probe session으로 일반화하는 중간 단계다. 단, 아직 단일 session이므로 최종 mouse 항은 여러 session 반복으로 닫아야 한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--session-ref", default=DEFAULT_SESSION_REF)
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--min-acronym-spikes", type=int, default=100_000)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=1e-3)
    args = parser.parse_args()

    output = evaluate(args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx multi-probe motor-striatal region gate")
    print(f"  eid={output['eid']}")
    print(f"  trials={output['trial_count']}")
    for probe in output["probe_summaries"]:
        print(
            "  "
            + f"{probe['collection']} spikes={probe['spike_count']} "
            + f"target_family_spikes={probe['target_family_spike_count']}"
        )
    for target in output["targets"]:
        best = max(target["rows"], key=lambda row: row["balanced_accuracy"])
        print(
            "  "
            + f"{target['target']} best={best['model']} "
            + f"BA={best['balanced_accuracy']:.6f} "
            + f"p={best['p_balanced_accuracy_ge_observed']:.6f}"
        )
    print(f"  passed={output['passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
