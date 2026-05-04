"""Lightweight ds000116 modality gate from public events.tsv files.

This is a readiness / behavioral-proxy gate, not a full neural-state proof.
It verifies that the publicly accessible Auditory/Visual Oddball dataset has
enough clean paired visual/auditory runs to test modality-specific operators:

    L_matched < L_wrong_modality

The observed state here is derived only from task/event labels and response
times. A claim-ready neural gate still needs region-resolved BOLD or EEG
features folded into p_r = (x_a, x_s, x_b).
"""

from __future__ import annotations

import csv
import io
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from urllib.request import urlopen


DATASET = "ds000116"
SNAPSHOT = "00003"
RAW_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000116/master"
DEFAULT_DATA_ROOT = Path("data/openneuro/ds000116")
SUBJECTS = [f"sub-{idx:02d}" for idx in range(1, 18)]
RUNS = [1, 2, 3]
TASKS = {
    "auditory": "auditoryoddballwithbuttonresponsetotargetstimuli",
    "visual": "visualoddballwithbuttonresponsetotargetstimuli",
}
REGIONS = ["visual", "auditory", "motor_response", "background"]


def fetch_text(path: str, *, data_root: Path | None = None, allow_network: bool = True) -> str:
    if data_root is not None:
        local_path = data_root / path
        if local_path.exists():
            return local_path.read_text(encoding="utf-8")
    if not allow_network:
        raise FileNotFoundError(f"missing local cached file: {path}")
    with urlopen(f"{RAW_BASE}/{path}", timeout=30) as response:
        return response.read().decode("utf-8")


def project_simplex(values: list[float]) -> list[float]:
    clipped = [max(float(value), 1e-12) for value in values]
    total = sum(clipped)
    return [value / total for value in clipped]


def parse_events(text: str) -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    stim = [row for row in rows if "stimulus presentation" in (row.get("trial_type") or "")]
    targets = [
        row
        for row in stim
        if row.get("Stimulus") == "target" or "oddball" in (row.get("trial_type") or "")
    ]
    standards = [
        row
        for row in stim
        if row.get("Stimulus") == "standard" or "standard" in (row.get("trial_type") or "")
    ]
    rt_rows = [
        row for row in rows if "behavioral response time" in (row.get("trial_type") or "")
    ]
    rts: list[float] = []
    for row in rt_rows:
        try:
            duration = float(row["duration"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(duration):
            rts.append(duration)
    stim_count = max(len(stim), 1)
    return {
        "stim_count": float(len(stim)),
        "target_count": float(len(targets)),
        "standard_count": float(len(standards)),
        "target_rate": float(len(targets) / stim_count),
        "response_count": float(len(rts)),
        "response_rate": float(len(rts) / max(len(targets), 1)),
        "rt_mean": float(mean(rts)) if rts else float("nan"),
    }


def observed_state(task: str, stats: dict[str, float]) -> list[float]:
    """Fold task/run metadata into a minimal 4-region proxy state."""
    target_drive = min(max(stats["target_rate"] / 0.20, 0.0), 1.5)
    rt_penalty = 0.0
    if math.isfinite(stats["rt_mean"]):
        rt_penalty = min(max((stats["rt_mean"] - 0.30) / 0.30, 0.0), 1.0)
    response_drive = min(max(stats["response_rate"], 0.0), 1.2)

    visual = 0.10
    auditory = 0.10
    motor = 0.08 + 0.08 * response_drive + 0.04 * rt_penalty
    background = 0.72 - 0.10 * target_drive - 0.03 * response_drive

    if task == "visual":
        visual += 0.22 * target_drive
        background -= 0.06 * target_drive
    elif task == "auditory":
        auditory += 0.22 * target_drive
        background -= 0.06 * target_drive
    else:
        raise ValueError(f"unknown task: {task}")
    return project_simplex([visual, auditory, motor, background])


def predicted_state(task: str, stats: dict[str, float], *, operator: str) -> list[float]:
    """Predict with either the matched or swapped modality operator."""
    if operator == "matched":
        predicted_task = task
    elif operator == "wrong_modality":
        predicted_task = "auditory" if task == "visual" else "visual"
    elif operator == "generic":
        target_drive = min(max(stats["target_rate"] / 0.20, 0.0), 1.5)
        response_drive = min(max(stats["response_rate"], 0.0), 1.2)
        visual = 0.10 + 0.11 * target_drive
        auditory = 0.10 + 0.11 * target_drive
        motor = 0.08 + 0.08 * response_drive
        background = 0.72 - 0.16 * target_drive - 0.03 * response_drive
        return project_simplex([visual, auditory, motor, background])
    else:
        raise ValueError(f"unknown operator: {operator}")
    return observed_state(predicted_task, stats)


def squared_loss(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def average_state(cases: list[dict[str, object]]) -> list[float]:
    if not cases:
        return [0.25, 0.25, 0.25, 0.25]
    rows = []
    for case in cases:
        state = case["observed_state"]
        rows.append([float(state[name]) for name in REGIONS])  # type: ignore[index]
    return [mean([row[idx] for row in rows]) for idx in range(len(REGIONS))]


def prototype_holdout(cases: list[dict[str, object]]) -> dict[str, object]:
    """Leave-one-subject-out matched vs wrong modality using learned prototypes."""
    subjects = sorted({str(case["subject"]) for case in cases})
    splits = []
    for holdout_subject in subjects:
        train = [case for case in cases if str(case["subject"]) != holdout_subject]
        test = [case for case in cases if str(case["subject"]) == holdout_subject]
        prototypes = {
            task: average_state([case for case in train if str(case["task"]) == task])
            for task in TASKS
        }
        generic = average_state(train)
        losses = {"matched": 0.0, "wrong_modality": 0.0, "generic": 0.0}
        for case in test:
            task = str(case["task"])
            wrong = "auditory" if task == "visual" else "visual"
            observed = [float(case["observed_state"][name]) for name in REGIONS]  # type: ignore[index]
            losses["matched"] += squared_loss(observed, prototypes[task])
            losses["wrong_modality"] += squared_loss(observed, prototypes[wrong])
            losses["generic"] += squared_loss(observed, generic)
        splits.append(
            {
                "holdout_subject": holdout_subject,
                "test_cases": len(test),
                "losses": losses,
                "matched_over_wrong": losses["matched"] / max(losses["wrong_modality"], 1e-12),
                "matched_over_generic": losses["matched"] / max(losses["generic"], 1e-12),
                "passed": losses["matched"] < losses["wrong_modality"]
                and losses["matched"] < losses["generic"],
            }
        )
    total = {
        name: sum(float(split["losses"][name]) for split in splits)  # type: ignore[index]
        for name in ("matched", "wrong_modality", "generic")
    }
    return {
        "criterion": "leave-one-subject-out prototype: matched < wrong_modality and generic",
        "splits": splits,
        "total_losses": total,
        "matched_over_wrong": total["matched"] / max(total["wrong_modality"], 1e-12),
        "matched_over_generic": total["matched"] / max(total["generic"], 1e-12),
        "passed": bool(splits)
        and all(bool(split["passed"]) for split in splits)
        and total["matched"] < total["wrong_modality"]
        and total["matched"] < total["generic"],
    }


def load_cases(
    *,
    subjects: list[str],
    data_root: Path | None,
    allow_network: bool,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    cases: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for subject in subjects:
        for task, task_slug in TASKS.items():
            for run in RUNS:
                path = f"{subject}/func/{subject}_task-{task_slug}_run-{run:02d}_events.tsv"
                try:
                    stats = parse_events(
                        fetch_text(path, data_root=data_root, allow_network=allow_network)
                    )
                except Exception as exc:  # noqa: BLE001 - reported in output
                    missing.append({"path": path, "error": str(exc)})
                    continue
                observed = observed_state(task, stats)
                predictions = {
                    name: predicted_state(task, stats, operator=name)
                    for name in ("matched", "wrong_modality", "generic")
                }
                losses = {
                    name: squared_loss(observed, prediction)
                    for name, prediction in predictions.items()
                }
                cases.append(
                    {
                        "subject": subject,
                        "task": task,
                        "run": run,
                        "path": path,
                        "stats": stats,
                        "observed_state": dict(zip(REGIONS, observed)),
                        "losses": losses,
                        "passed_matched_vs_wrong": losses["matched"] < losses["wrong_modality"],
                        "passed_matched_vs_generic": losses["matched"] < losses["generic"],
                    }
                )
    return cases, missing


def summarize(
    cases: list[dict[str, object]],
    missing: list[dict[str, str]],
    *,
    expected_case_count: int,
) -> dict[str, object]:
    losses_by_name = {
        name: [float(case["losses"][name]) for case in cases]  # type: ignore[index]
        for name in ("matched", "wrong_modality", "generic")
    }
    by_task = {}
    for task in TASKS:
        task_cases = [case for case in cases if case["task"] == task]
        by_task[task] = {
            "run_count": len(task_cases),
            "stimuli": int(sum(float(case["stats"]["stim_count"]) for case in task_cases)),  # type: ignore[index]
            "targets": int(sum(float(case["stats"]["target_count"]) for case in task_cases)),  # type: ignore[index]
            "standards": int(sum(float(case["stats"]["standard_count"]) for case in task_cases)),  # type: ignore[index]
            "mean_rt": mean(
                [
                    float(case["stats"]["rt_mean"])  # type: ignore[index]
                    for case in task_cases
                    if math.isfinite(float(case["stats"]["rt_mean"]))  # type: ignore[index]
                ]
            ),
        }
    matched = losses_by_name["matched"]
    wrong = losses_by_name["wrong_modality"]
    generic = losses_by_name["generic"]
    return {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "gate": "event-derived modality readiness; not a claim-ready neural gate",
        "case_count": len(cases),
        "missing_count": len(missing),
        "missing": missing,
        "regions": REGIONS,
        "task_summary": by_task,
        "loss_mean": {
            "matched": mean(matched),
            "wrong_modality": mean(wrong),
            "generic": mean(generic),
        },
        "matched_over_wrong": mean(matched) / max(mean(wrong), 1e-12),
        "matched_over_generic": mean(matched) / max(mean(generic), 1e-12),
        "matched_vs_wrong_pass_rate": mean(
            [bool(case["passed_matched_vs_wrong"]) for case in cases]
        ),
        "matched_vs_generic_pass_rate": mean(
            [bool(case["passed_matched_vs_generic"]) for case in cases]
        ),
        "prototype_holdout": prototype_holdout(cases),
        "passed_readiness_gate": (
            not missing
            and len(cases) == expected_case_count
            and mean([bool(case["passed_matched_vs_wrong"]) for case in cases]) == 1.0
            and bool(prototype_holdout(cases)["passed"])
        ),
        "next_required_for_neural_claim": [
            "download BOLD or EEG data",
            "extract region-resolved visual/auditory/motor/background proxies",
            "rerun matched-vs-wrong modality gate on p_r states, not event labels",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--subject-count", type=int, default=17)
    parser.add_argument(
        "--network",
        action="store_true",
        help="Allow fetching missing events from GitHub. Without this, only local cache is used.",
    )
    args = parser.parse_args()
    subjects = [
        f"sub-{idx:02d}"
        for idx in range(args.start_subject, args.start_subject + args.subject_count)
    ]
    data_root = args.data_root if args.data_root.exists() else None
    cases, missing = load_cases(
        subjects=subjects,
        data_root=data_root,
        allow_network=bool(args.network),
    )
    output = summarize(
        cases,
        missing,
        expected_case_count=len(subjects) * len(TASKS) * len(RUNS),
    )
    output["requested_subjects"] = subjects
    output["data_root"] = str(args.data_root)
    output["network_allowed"] = bool(args.network)
    output["cases"] = cases
    out_path = Path(__file__).with_name("ds000116_modality_gate_results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ds000116 modality readiness gate")
    print(f"  cases              = {output['case_count']}")
    print(f"  missing            = {output['missing_count']}")
    for task, row in output["task_summary"].items():  # type: ignore[union-attr]
        print(
            f"  {task:8s}: runs={row['run_count']}, stimuli={row['stimuli']}, "
            f"targets={row['targets']}, standards={row['standards']}, "
            f"mean_rt={row['mean_rt']:.3f}s"
        )
    losses = output["loss_mean"]  # type: ignore[assignment]
    print(f"  L_matched          = {losses['matched']:.8f}")
    print(f"  L_wrong_modality   = {losses['wrong_modality']:.8f}")
    print(f"  L_generic          = {losses['generic']:.8f}")
    print(f"  matched/wrong      = {output['matched_over_wrong']:.6f}")
    print(f"  matched/generic    = {output['matched_over_generic']:.6f}")
    holdout = output["prototype_holdout"]  # type: ignore[assignment]
    h_losses = holdout["total_losses"]
    print("  prototype holdout:")
    print(f"    L_matched        = {h_losses['matched']:.8f}")
    print(f"    L_wrong_modality = {h_losses['wrong_modality']:.8f}")
    print(f"    L_generic        = {h_losses['generic']:.8f}")
    print(f"    matched/wrong    = {holdout['matched_over_wrong']:.6f}")
    print(f"    matched/generic  = {holdout['matched_over_generic']:.6f}")
    print(f"    passed           = {holdout['passed']}")
    print(f"  passed             = {output['passed_readiness_gate']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
