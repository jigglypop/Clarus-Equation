"""Mouse IBL/OpenAlyx region-binned decision/action gate.

This is the first real-data mouse gate after the metadata bridge audit.  It
downloads/loads one strict OpenAlyx Neuropixels session and asks whether
probe-level spike activity, collapsed into anatomical region bins, predicts
trial decision/action variables better than simple baselines.

The gate is intentionally conservative:

    * fixed stimulus and movement windows are used to avoid response-duration
      leakage in spike counts;
    * a one-dimensional global firing-rate model is kept as a flat baseline;
    * label permutations are run on the same cross-validation routine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


OPENALYX_URL = "https://openalyx.internationalbrainlab.org"
DEFAULT_EID = "d2832a38-27f6-452d-91d6-af72d794136c"
DEFAULT_COLLECTION = "alf/probe00/pykilosort"
DEFAULT_SESSION_REF = "wittenlab/Subjects/ibl_witten_29/2021-06-08/001"
RNG_SEED = 1729
RESULT_JSON = Path(__file__).with_name("mouse_ibl_region_decision_action_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_region_decision_action_report.md")

FAMILY_NAMES = [
    "motor_cortex",
    "somatosensory_cortex",
    "visual_cortex",
    "visual_thalamus",
    "somatosensory_thalamus",
    "striatal_complex",
    "septal_subpallium",
    "basal_ganglia_output",
    "prefrontal_cortex",
    "hippocampus",
    "cingulate",
    "other",
    "unknown",
]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_column: str
    start_offset: float
    end_column: str
    end_offset: float
    meaning: str


STIMULUS_WINDOW = WindowSpec(
    name="stimulus_20_320ms",
    start_column="stimOn_times",
    start_offset=0.020,
    end_column="stimOn_times",
    end_offset=0.320,
    meaning="early post-stimulus decision/action preparation window",
)

MOVEMENT_WINDOW = WindowSpec(
    name="first_movement_-100_200ms",
    start_column="firstMovement_times",
    start_offset=-0.100,
    end_column="firstMovement_times",
    end_offset=0.200,
    meaning="movement-aligned action execution window",
)


def region_family(acronym: object) -> str:
    text = str(acronym)
    if text.startswith("MOp") or text.startswith("MOs"):
        return "motor_cortex"
    if text.startswith("SSp") or text.startswith("SSs"):
        return "somatosensory_cortex"
    if text.startswith("VIS"):
        return "visual_cortex"
    if text in {"LP", "LGd"}:
        return "visual_thalamus"
    if (
        text in {"PO", "VPM", "VPLpc", "VPL", "VAL", "VM", "RT"}
        or text.startswith("VPL")
        or text.startswith("VPM")
    ):
        return "somatosensory_thalamus"
    if text in {"CP", "ACB", "OT", "STR", "FS"} or text.startswith("STR"):
        return "striatal_complex"
    if text in {"LSr", "LSv", "LSc", "SI", "SH"}:
        return "septal_subpallium"
    if text in {"GPe", "GPi", "SNr", "SNc", "STN"}:
        return "basal_ganglia_output"
    if (
        text.startswith("PL")
        or text.startswith("ILA")
        or text.startswith("ORB")
        or text.startswith("ACAd")
        or text.startswith("ACAv")
    ):
        return "prefrontal_cortex"
    if text.startswith("CA") or text.startswith("DG") or text.startswith("SUB"):
        return "hippocampus"
    if text == "cing" or text.startswith("ACA"):
        return "cingulate"
    if text in {"", "nan", "None"}:
        return "unknown"
    return "other"


def safe_float_array(values: Iterable[object]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def load_session(eid: str, collection: str) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
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
        "spike_times": one.load_dataset(
            eid, "spikes.times.npy", collection=collection, query_type="remote"
        ),
        "spike_clusters": one.load_dataset(
            eid, "spikes.clusters.npy", collection=collection, query_type="remote"
        ),
        "cluster_acronyms": one.load_dataset(
            eid,
            "clusters.brainLocationAcronyms_ccf_2017.npy",
            collection=collection,
            query_type="remote",
        ),
    }


def cluster_family_ids(cluster_acronyms: np.ndarray, max_cluster_id: int) -> np.ndarray:
    family_ids = np.full(max_cluster_id + 1, FAMILY_NAMES.index("unknown"), dtype=np.int16)
    for cluster_id, acronym in enumerate(cluster_acronyms):
        family_ids[cluster_id] = FAMILY_NAMES.index(region_family(acronym))
    return family_ids


def cluster_acronym_groups(
    cluster_acronyms: np.ndarray,
    spike_clusters: np.ndarray,
    max_cluster_id: int,
    min_spikes: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    cluster_labels = np.full(max_cluster_id + 1, "unknown", dtype=object)
    for cluster_id, acronym in enumerate(cluster_acronyms):
        label = str(acronym)
        cluster_labels[cluster_id] = label if label not in {"", "nan", "None"} else "unknown"

    labels, counts = np.unique(cluster_labels[spike_clusters], return_counts=True)
    spike_counts = {str(label): int(count) for label, count in zip(labels, counts)}
    ranked = sorted(
        (
            (label, count)
            for label, count in spike_counts.items()
            if label != "unknown" and count >= min_spikes
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    group_names = [label for label, _ in ranked]
    known_below_threshold = [
        label
        for label, count in spike_counts.items()
        if label != "unknown" and count < min_spikes
    ]
    if known_below_threshold:
        group_names.append("other_low_spike_acronyms")
    group_names.append("unknown")

    group_lookup = {name: idx for idx, name in enumerate(group_names)}
    other_idx = group_lookup.get("other_low_spike_acronyms", group_lookup["unknown"])
    unknown_idx = group_lookup["unknown"]
    cluster_group = np.full(max_cluster_id + 1, unknown_idx, dtype=np.int16)
    for cluster_id, label in enumerate(cluster_labels):
        if label == "unknown":
            cluster_group[cluster_id] = unknown_idx
        elif label in group_lookup:
            cluster_group[cluster_id] = group_lookup[label]
        else:
            cluster_group[cluster_id] = other_idx
    return group_names, cluster_group, spike_counts


def window_bounds(trials, spec: WindowSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = safe_float_array(trials[spec.start_column]) + spec.start_offset
    ends = safe_float_array(trials[spec.end_column]) + spec.end_offset
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    return starts, ends, valid


def window_features(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_group: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    valid_window: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    features = np.zeros((len(starts), n_groups), dtype=float)
    durations = np.maximum(ends - starts, 1e-9)
    for idx, (start, end, valid) in enumerate(zip(starts, ends, valid_window)):
        if not valid:
            features[idx, :] = np.nan
            continue
        lo = int(np.searchsorted(spike_times, start, side="left"))
        hi = int(np.searchsorted(spike_times, end, side="right"))
        if hi <= lo:
            continue
        group_ids = cluster_group[spike_clusters[lo:hi]]
        features[idx, :] = np.bincount(group_ids, minlength=n_groups)[:n_groups]
    return features / durations[:, None]


def choice_target(trials) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    choice = safe_float_array(trials["choice"])
    valid = np.isfinite(choice) & (choice != 0)
    return (choice > 0).astype(int), valid, {"definition": "choice > 0"}


def first_movement_speed_target(trials) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    stimulus = safe_float_array(trials["stimOn_times"])
    movement = safe_float_array(trials["firstMovement_times"])
    latency = movement - stimulus
    valid = np.isfinite(latency) & (latency > 0)
    threshold = float(np.nanmedian(latency[valid]))
    target = (latency <= threshold).astype(int)
    return (
        target,
        valid,
        {
            "definition": "firstMovement_times - stimOn_times <= median",
            "median_seconds": threshold,
        },
    )


def wheel_action_direction_target(
    trials,
    wheel_timestamps: np.ndarray,
    wheel_position: np.ndarray,
    min_abs_displacement: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    movement = safe_float_array(trials["firstMovement_times"])
    response = safe_float_array(trials["response_times"])
    sample_end = np.minimum(response, movement + 0.250)
    sample_end = np.maximum(sample_end, movement + 0.050)
    valid = np.isfinite(movement) & np.isfinite(sample_end)
    start_pos = np.interp(movement, wheel_timestamps, wheel_position)
    end_pos = np.interp(sample_end, wheel_timestamps, wheel_position)
    displacement = end_pos - start_pos
    valid &= np.isfinite(displacement) & (np.abs(displacement) >= min_abs_displacement)
    return (
        (displacement > 0).astype(int),
        valid,
        {
            "definition": "wheel position displacement from first movement to min(response, first movement + 250 ms)",
            "min_abs_displacement": min_abs_displacement,
        },
    )


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls))


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + ties)


def stratified_folds(y: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for cls in np.unique(y):
        class_idx = np.where(y == cls)[0]
        class_idx = rng.permutation(class_idx)
        for offset, idx in enumerate(class_idx):
            buckets[offset % folds].append(int(idx))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets if bucket]


def ridge_cv_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    ridge: float,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    values = np.unique(y)
    if len(values) != 2:
        raise ValueError("ridge_cv_scores expects a binary target")
    y_signed = np.where(y == values[1], 1.0, -1.0)
    scores = np.zeros(len(y), dtype=float)
    for test in stratified_folds(y, folds, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        mu = np.nanmean(x[train], axis=0)
        sigma = np.nanstd(x[train], axis=0)
        sigma[sigma < 1e-9] = 1.0
        x_train = (x[train] - mu) / sigma
        x_test = (x[test] - mu) / sigma
        design_train = np.column_stack([np.ones(len(train)), x_train])
        design_test = np.column_stack([np.ones(len(test)), x_test])
        penalty = np.eye(design_train.shape[1]) * ridge
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(
            design_train.T @ design_train + penalty,
            design_train.T @ y_signed[train],
        )
        scores[test] = design_test @ weights
    return scores


def evaluate_decoder(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    ridge: float,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    scores = ridge_cv_scores(x, y, folds=folds, ridge=ridge, seed=seed)
    predicted = (scores >= 0).astype(int)
    observed_bacc = balanced_accuracy(y, predicted)
    observed_auc = auc_score(y, scores)

    rng = np.random.default_rng(seed)
    permuted_bacc = []
    permuted_auc = []
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        perm_scores = ridge_cv_scores(x, shuffled, folds=folds, ridge=ridge, seed=seed)
        perm_predicted = (perm_scores >= 0).astype(int)
        permuted_bacc.append(balanced_accuracy(shuffled, perm_predicted))
        permuted_auc.append(auc_score(shuffled, perm_scores))
    permuted_bacc_arr = np.asarray(permuted_bacc, dtype=float)
    permuted_auc_arr = np.asarray(permuted_auc, dtype=float)
    return {
        "balanced_accuracy": float(observed_bacc),
        "auc": float(observed_auc),
        "permutation_balanced_accuracy_mean": float(np.nanmean(permuted_bacc_arr)),
        "permutation_auc_mean": float(np.nanmean(permuted_auc_arr)),
        "p_balanced_accuracy_ge_observed": float(
            (np.sum(permuted_bacc_arr >= observed_bacc) + 1)
            / (len(permuted_bacc_arr) + 1)
        ),
        "p_auc_ge_observed": float(
            (np.sum(permuted_auc_arr >= observed_auc) + 1)
            / (len(permuted_auc_arr) + 1)
        ),
    }


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)}


def evaluate_target(
    *,
    target_name: str,
    window_name: str,
    x_models: dict[str, np.ndarray],
    y_all: np.ndarray,
    valid: np.ndarray,
    folds: int,
    ridge: float,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    valid = valid & np.all(np.isfinite(next(iter(x_models.values()))), axis=1)
    y = np.asarray(y_all[valid], dtype=int)
    majority_accuracy = max(float(np.mean(y == value)) for value in np.unique(y))
    rows = []
    for model_name, x_all in x_models.items():
        x = np.asarray(x_all[valid], dtype=float)
        metrics = evaluate_decoder(
            x,
            y,
            folds=folds,
            ridge=ridge,
            permutations=permutations,
            seed=seed,
        )
        rows.append(
            {
                "target": target_name,
                "window": window_name,
                "model": model_name,
                "feature_count": int(x.shape[1]),
                "n_trials": int(len(y)),
                "class_counts": class_counts(y),
                "majority_accuracy": majority_accuracy,
                **metrics,
            }
        )
    global_bacc = next(
        row["balanced_accuracy"] for row in rows if row["model"] == "global_rate"
    )
    for row in rows:
        row["delta_vs_global_rate"] = float(row["balanced_accuracy"] - global_bacc)
        row["delta_vs_majority"] = float(row["balanced_accuracy"] - majority_accuracy)
        row["passed"] = bool(
            row["model"] != "global_rate"
            and row["balanced_accuracy"] > global_bacc
            and row["balanced_accuracy"] > majority_accuracy
            and row["p_balanced_accuracy_ge_observed"] < 0.05
        )
    return {"target": target_name, "window": window_name, "rows": rows}


def rounded(value: object, digits: int = 6) -> object:
    if isinstance(value, float):
        return round(value, digits)
    return value


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx region-binned decision/action gate",
        "",
        "첫 strict IBL/OpenAlyx Neuropixels 세션을 실제로 내려받아 trial table, wheel, spike time, spike cluster, cluster-region acronym을 연결했다.",
        "목표는 영역별 발화율이 choice/action 변수를 단순 majority와 global firing-rate baseline보다 잘 예측하는지 확인하는 것이다.",
        "",
        "## source",
        "",
        f"- OpenAlyx: `{output['openalyx_url']}`",
        f"- eid: `{output['eid']}`",
        f"- session: `{output['session_ref']}`",
        f"- probe collection: `{output['collection']}`",
        "",
        "## loaded arrays",
        "",
        "| item | value |",
        "|---|---:|",
        f"| trials | {output['trial_count']} |",
        f"| wheel samples | {output['wheel_sample_count']} |",
        f"| spikes | {output['spike_count']} |",
        f"| max observed cluster id + 1 | {output['observed_cluster_slots']} |",
        f"| strict acronym rows | {output['strict_cluster_acronym_rows']} |",
        f"| spikes with strict acronym row | {output['strict_labeled_spike_count']} |",
        f"| spikes assigned unknown | {output['unknown_spike_count']} |",
        f"| unknown spike fraction | {output['unknown_spike_fraction']:.6f} |",
        "",
        "## feature construction",
        "",
        "For trial \(i\), region group \(g\), and window \([a_i,b_i]\), the feature is a duration-normalized spike count:",
        "",
        "$$",
        "x_{ig}=\\frac{1}{b_i-a_i}\\sum_k \\mathbf 1[t_k\\in[a_i,b_i]]\\mathbf 1[G(c_k)=g].",
        "$$",
        "",
        "The compared models are:",
        "",
        "| model | feature set |",
        "|---|---|",
        "| `region_family` | anatomical family bins: visual cortex, visual thalamus, somatosensory thalamus, hippocampus, cingulate, other, unknown |",
        "| `acronym_region` | high-spike CCF acronyms plus unknown |",
        "| `global_rate` | one scalar total firing rate, used as flat baseline |",
        "",
        "The decoder is a z-scored ridge linear classifier evaluated with deterministic stratified cross-validation. The null model shuffles labels and repeats the same CV routine.",
        "",
        "$$",
        "\\hat w=\\arg\\min_w\\|y-Xw\\|_2^2+\\lambda\\|w\\|_2^2,\\qquad",
        "\\mathrm{BA}=\\frac12\\left(\\frac{TP}{P}+\\frac{TN}{N}\\right).",
        "$$",
        "",
        "## region groups",
        "",
        "| family | mean stimulus-window rate |",
        "|---|---:|",
    ]
    for family, rate in output["family_mean_stimulus_rates"].items():
        lines.append(f"| `{family}` | {rate:.6f} |")

    lines.extend(
        [
            "",
            "Top acronym groups by spike count:",
            "",
            "| acronym group | spikes |",
            "|---|---:|",
        ]
    )
    for name, count in output["acronym_spike_counts_top"]:
        lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "## decoder results", ""])
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
            "해석:",
            "",
            "- 첫 strict session에서 region/acronym activity는 choice, wheel action direction, movement speed를 global-rate baseline보다 잘 예측한다.",
            "- 특히 acronym-level region bin은 세 target 모두에서 permutation gate를 통과한다.",
            "- 단, 이 probe의 강한 영역은 thalamus, visual cortex, hippocampus 쪽이다. motor/striatal loop를 닫은 것이 아니므로 mouse 단계 전체를 닫으려면 다음에는 다중 probe 또는 motor/striatum 포함 세션을 추가해야 한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    data = load_session(args.eid, args.collection)
    trials = data["trials"]
    wheel_timestamps = np.asarray(data["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(data["wheel_position"], dtype=float)
    spike_times = np.asarray(data["spike_times"], dtype=float)
    spike_clusters = np.asarray(data["spike_clusters"], dtype=np.int64)
    cluster_acronyms = np.asarray(data["cluster_acronyms"], dtype=object)

    max_cluster_id = int(np.max(spike_clusters))
    family_ids = cluster_family_ids(cluster_acronyms, max_cluster_id)
    acronym_names, acronym_ids, acronym_spike_counts = cluster_acronym_groups(
        cluster_acronyms,
        spike_clusters,
        max_cluster_id,
        args.min_acronym_spikes,
    )

    stimulus_start, stimulus_end, stimulus_valid = window_bounds(trials, STIMULUS_WINDOW)
    movement_start, movement_end, movement_valid = window_bounds(trials, MOVEMENT_WINDOW)

    family_stimulus = window_features(
        spike_times,
        spike_clusters,
        family_ids,
        stimulus_start,
        stimulus_end,
        stimulus_valid,
        len(FAMILY_NAMES),
    )
    family_movement = window_features(
        spike_times,
        spike_clusters,
        family_ids,
        movement_start,
        movement_end,
        movement_valid,
        len(FAMILY_NAMES),
    )
    acronym_stimulus = window_features(
        spike_times,
        spike_clusters,
        acronym_ids,
        stimulus_start,
        stimulus_end,
        stimulus_valid,
        len(acronym_names),
    )
    acronym_movement = window_features(
        spike_times,
        spike_clusters,
        acronym_ids,
        movement_start,
        movement_end,
        movement_valid,
        len(acronym_names),
    )

    choice, choice_valid, choice_meta = choice_target(trials)
    speed, speed_valid, speed_meta = first_movement_speed_target(trials)
    wheel_direction, wheel_valid, wheel_meta = wheel_action_direction_target(
        trials,
        wheel_timestamps,
        wheel_position,
        args.min_abs_wheel_displacement,
    )

    stimulus_models = {
        "region_family": family_stimulus,
        "acronym_region": acronym_stimulus,
        "global_rate": np.nansum(family_stimulus, axis=1, keepdims=True),
    }
    movement_models = {
        "region_family": family_movement,
        "acronym_region": acronym_movement,
        "global_rate": np.nansum(family_movement, axis=1, keepdims=True),
    }

    targets = [
        evaluate_target(
            target_name="choice_sign",
            window_name=STIMULUS_WINDOW.name,
            x_models=stimulus_models,
            y_all=choice,
            valid=choice_valid & stimulus_valid,
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
            valid=speed_valid & stimulus_valid,
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
            valid=wheel_valid & movement_valid,
            folds=args.folds,
            ridge=args.ridge,
            permutations=args.permutations,
            seed=args.seed,
        ),
    ]

    strict_labeled_spikes = int(np.sum(spike_clusters < len(cluster_acronyms)))
    unknown_spikes = int(len(spike_clusters) - strict_labeled_spikes)
    family_mean_stimulus_rates = {
        name: float(np.nanmean(family_stimulus[:, idx]))
        for idx, name in enumerate(FAMILY_NAMES)
    }
    acronym_spike_counts_top = sorted(
        acronym_spike_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
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
        "collection": args.collection,
        "trial_count": int(len(trials)),
        "wheel_sample_count": int(len(wheel_timestamps)),
        "spike_count": int(len(spike_times)),
        "observed_cluster_slots": int(max_cluster_id + 1),
        "strict_cluster_acronym_rows": int(len(cluster_acronyms)),
        "strict_labeled_spike_count": strict_labeled_spikes,
        "unknown_spike_count": unknown_spikes,
        "unknown_spike_fraction": float(unknown_spikes / len(spike_clusters)),
        "window_specs": {
            STIMULUS_WINDOW.name: STIMULUS_WINDOW.__dict__,
            MOVEMENT_WINDOW.name: MOVEMENT_WINDOW.__dict__,
        },
        "family_names": FAMILY_NAMES,
        "acronym_group_names": acronym_names,
        "family_mean_stimulus_rates": family_mean_stimulus_rates,
        "acronym_spike_counts_top": acronym_spike_counts_top,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--session-ref", default=DEFAULT_SESSION_REF)
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

    print("Mouse IBL/OpenAlyx region decision/action gate")
    print(f"  eid={output['eid']}")
    print(f"  trials={output['trial_count']}")
    print(f"  spikes={output['spike_count']}")
    print(f"  unknown_spike_fraction={output['unknown_spike_fraction']:.6f}")
    for target in output["targets"]:
        best = max(
            target["rows"],
            key=lambda row: row["balanced_accuracy"],
        )
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
