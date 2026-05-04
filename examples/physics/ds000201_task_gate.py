"""Lightweight ds000201 task-domain gate from public events.tsv files.

This is an event-level readiness gate, not a neural-state proof. It checks
whether public task/event files support separable operators for three domains:

    hands  -> somatosensory / pain
    faces  -> face-emotion / social visual processing
    arrows -> cognitive control / regulation

The held-out criterion is:

    L_matched_task < L_wrong_task and L_matched_task < L_generic

Claim-ready validation still requires BOLD-derived region states p_r.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import mean
from urllib.request import urlopen


DATASET = "ds000201"
SNAPSHOT = "1.0.3"
RAW_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000201/master"
DEFAULT_DATA_ROOT = Path("data/openneuro/ds000201")
TASKS = ["hands", "faces", "arrows"]
SESSIONS = ["ses-1", "ses-2"]
REGIONS = [
    "somatosensory_pain",
    "face_emotion",
    "cognitive_control",
    "motor_response",
    "background",
]


def fetch_text(path: str, *, data_root: Path | None, allow_network: bool) -> str:
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


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "n/a", "NA", "nan"}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_events(task: str, text: str) -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    response_rows = [
        row
        for row in rows
        if any(
            (value := numeric(row, key)) is not None and value > 0
            for key in ("response_time", "time_until_response")
        )
    ]
    response_values = [
        value
        for row in rows
        for key in ("response_time", "time_until_response")
        if (value := numeric(row, key)) is not None and value > 0
    ]
    stats: dict[str, float] = {
        "row_count": float(len(rows)),
        "response_count": float(len(response_rows)),
        "mean_response": mean(response_values) if response_values else float("nan"),
    }

    if task == "hands":
        picture_rows = [row for row in rows if row.get("trial_type") == "Pic2"]
        pain = [row for row in picture_rows if row.get("condition") == "Pain"]
        no_pain = [row for row in picture_rows if row.get("condition") == "No_Pain"]
        ratings = [
            value
            for row in rows
            if (value := numeric(row, "unpleasantness_rating")) is not None
        ]
        stats.update(
            {
                "trial_count": float(len(picture_rows)),
                "pain_count": float(len(pain)),
                "no_pain_count": float(len(no_pain)),
                "pain_fraction": len(pain) / max(len(picture_rows), 1),
                "mean_rating": mean(ratings) if ratings else float("nan"),
            }
        )
    elif task == "faces":
        face_rows = [
            row
            for row in rows
            if row.get("trial_type") in {"happy", "neutral", "angry", "fearful"}
        ]
        emotion_rows = [
            row for row in face_rows if row.get("trial_type") in {"happy", "angry", "fearful"}
        ]
        ratings = [value for row in rows if (value := numeric(row, "rating")) is not None]
        stats.update(
            {
                "trial_count": float(len(face_rows)),
                "emotion_count": float(len(emotion_rows)),
                "emotion_fraction": len(emotion_rows) / max(len(face_rows), 1),
                "mean_rating": mean(ratings) if ratings else float("nan"),
            }
        )
    elif task == "arrows":
        cue_rows = [row for row in rows if row.get("cue_to_participant")]
        suppress = [row for row in cue_rows if row.get("cue_to_participant") == "Suppress"]
        enhance = [row for row in cue_rows if row.get("cue_to_participant") == "Enhance"]
        maintain = [row for row in cue_rows if row.get("cue_to_participant") == "Maintain"]
        success = [
            value
            for row in rows
            if (value := numeric(row, "rated_success_of_regulation")) is not None
        ]
        stats.update(
            {
                "trial_count": float(len(cue_rows)),
                "suppress_count": float(len(suppress)),
                "enhance_count": float(len(enhance)),
                "maintain_count": float(len(maintain)),
                "regulation_fraction": (len(suppress) + len(enhance)) / max(len(cue_rows), 1),
                "mean_rating": mean(success) if success else float("nan"),
            }
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return stats


def observed_state(task: str, stats: dict[str, float]) -> list[float]:
    response_drive = min(stats["response_count"] / max(stats["row_count"], 1.0) * 3.0, 1.2)
    rating = stats.get("mean_rating", float("nan"))
    if not math.isfinite(rating):
        rating_drive = 0.0
    elif task == "arrows":
        rating_drive = min(max(rating / 4.0, 0.0), 1.2)
    else:
        rating_drive = min(max(rating / 100.0, 0.0), 1.2)

    pain = 0.08
    face = 0.08
    control = 0.08
    motor = 0.08 + 0.08 * response_drive
    background = 0.68 - 0.03 * response_drive

    if task == "hands":
        pain_drive = min(stats["pain_fraction"] * 1.5 + 0.7 * rating_drive, 1.6)
        pain += 0.24 * pain_drive
        motor += 0.02 * response_drive
        background -= 0.12 * pain_drive
    elif task == "faces":
        emotion_drive = min(stats["emotion_fraction"] + 0.5 * rating_drive, 1.6)
        face += 0.24 * emotion_drive
        motor += 0.02 * response_drive
        background -= 0.12 * emotion_drive
    elif task == "arrows":
        regulation_drive = min(stats["regulation_fraction"] + 0.5 * rating_drive, 1.6)
        control += 0.24 * regulation_drive
        motor += 0.03 * response_drive
        background -= 0.12 * regulation_drive
    else:
        raise ValueError(f"unknown task: {task}")
    return project_simplex([pain, face, control, motor, background])


def squared_loss(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def average_state(cases: list[dict[str, object]]) -> list[float]:
    if not cases:
        return [1.0 / len(REGIONS)] * len(REGIONS)
    rows = [
        [float(case["observed_state"][name]) for name in REGIONS]  # type: ignore[index]
        for case in cases
    ]
    return [mean([row[idx] for row in rows]) for idx in range(len(REGIONS))]


def prototype_holdout(cases: list[dict[str, object]]) -> dict[str, object]:
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
        losses = {"matched": 0.0, "wrong_task": 0.0, "generic": 0.0}
        for case in test:
            task = str(case["task"])
            wrong_candidates = [name for name in TASKS if name != task]
            observed = [float(case["observed_state"][name]) for name in REGIONS]  # type: ignore[index]
            wrong_loss = min(squared_loss(observed, prototypes[name]) for name in wrong_candidates)
            losses["matched"] += squared_loss(observed, prototypes[task])
            losses["wrong_task"] += wrong_loss
            losses["generic"] += squared_loss(observed, generic)
        splits.append(
            {
                "holdout_subject": holdout_subject,
                "test_cases": len(test),
                "losses": losses,
                "matched_over_wrong": losses["matched"] / max(losses["wrong_task"], 1e-12),
                "matched_over_generic": losses["matched"] / max(losses["generic"], 1e-12),
                "passed": losses["matched"] < losses["wrong_task"]
                and losses["matched"] < losses["generic"],
            }
        )
    total = {
        name: sum(float(split["losses"][name]) for split in splits)  # type: ignore[index]
        for name in ("matched", "wrong_task", "generic")
    }
    return {
        "criterion": "leave-one-subject-out prototype: matched < wrong_task and generic",
        "splits": splits,
        "total_losses": total,
        "matched_over_wrong": total["matched"] / max(total["wrong_task"], 1e-12),
        "matched_over_generic": total["matched"] / max(total["generic"], 1e-12),
        "passed": bool(splits)
        and all(bool(split["passed"]) for split in splits)
        and total["matched"] < total["wrong_task"]
        and total["matched"] < total["generic"],
    }


def load_subjects_from_cache(data_root: Path, start: int, count: int) -> list[str]:
    participants = data_root / "participants.tsv"
    if participants.exists():
        rows = csv.DictReader(io.StringIO(participants.read_text(encoding="utf-8")), delimiter="\t")
        subjects = [row["participant_id"] for row in rows if row.get("participant_id")]
        return subjects[max(start - 1, 0) : max(start - 1, 0) + count]
    return [f"sub-{9000 + idx}" for idx in range(start, start + count)]


def load_cases(
    *,
    subjects: list[str],
    data_root: Path | None,
    allow_network: bool,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    cases: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for subject in subjects:
        for session in SESSIONS:
            for task in TASKS:
                path = f"{subject}/{session}/func/{subject}_{session}_task-{task}_events.tsv"
                try:
                    stats = parse_events(
                        task,
                        fetch_text(path, data_root=data_root, allow_network=allow_network),
                    )
                except Exception as exc:  # noqa: BLE001 - reported in output
                    missing.append({"path": path, "error": str(exc)})
                    continue
                state = observed_state(task, stats)
                cases.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "path": path,
                        "stats": stats,
                        "observed_state": dict(zip(REGIONS, state)),
                    }
                )
    return cases, missing


def summarize(cases: list[dict[str, object]], missing: list[dict[str, str]], expected: int) -> dict[str, object]:
    holdout = prototype_holdout(cases)
    by_task = {}
    for task in TASKS:
        task_cases = [case for case in cases if str(case["task"]) == task]
        ratings = [
            float(case["stats"]["mean_rating"])  # type: ignore[index]
            for case in task_cases
            if math.isfinite(float(case["stats"]["mean_rating"]))  # type: ignore[index]
        ]
        by_task[task] = {
            "case_count": len(task_cases),
            "trial_count": int(sum(float(case["stats"]["trial_count"]) for case in task_cases)),  # type: ignore[index]
            "response_count": int(sum(float(case["stats"]["response_count"]) for case in task_cases)),  # type: ignore[index]
            "mean_rating": mean(ratings) if ratings else None,
        }
    return {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "gate": "event-derived task-domain readiness; not a claim-ready neural gate",
        "case_count": len(cases),
        "missing_count": len(missing),
        "missing": missing,
        "regions": REGIONS,
        "task_summary": by_task,
        "prototype_holdout": holdout,
        "passed_readiness_gate": (
            not missing and len(cases) == expected and bool(holdout["passed"])
        ),
        "next_required_for_neural_claim": [
            "download selected BOLD runs for hands/faces/arrows",
            "extract region-resolved pain, face-emotion, control, motor proxies",
            "rerun matched-vs-wrong task gate on p_r states",
            "compare graph operator against flat operator",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--subject-count", type=int, default=2)
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root.exists() else None
    subjects = load_subjects_from_cache(args.data_root, args.start_subject, args.subject_count)
    cases, missing = load_cases(
        subjects=subjects,
        data_root=data_root,
        allow_network=bool(args.network),
    )
    output = summarize(cases, missing, expected=len(subjects) * len(SESSIONS) * len(TASKS))
    output["requested_subjects"] = subjects
    output["data_root"] = str(args.data_root)
    output["network_allowed"] = bool(args.network)
    output["cases"] = cases
    out_path = Path(__file__).with_name("ds000201_task_gate_results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ds000201 task-domain readiness gate")
    print(f"  cases              = {output['case_count']}")
    print(f"  missing            = {output['missing_count']}")
    for task, row in output["task_summary"].items():  # type: ignore[union-attr]
        rating = row["mean_rating"]
        rating_text = "n/a" if rating is None else f"{rating:.3f}"
        print(
            f"  {task:6s}: cases={row['case_count']}, trials={row['trial_count']}, "
            f"responses={row['response_count']}, mean_rating={rating_text}"
        )
    holdout = output["prototype_holdout"]  # type: ignore[assignment]
    losses = holdout["total_losses"]
    print("  prototype holdout:")
    print(f"    L_matched        = {losses['matched']:.8f}")
    print(f"    L_wrong_task     = {losses['wrong_task']:.8f}")
    print(f"    L_generic        = {losses['generic']:.8f}")
    print(f"    matched/wrong    = {holdout['matched_over_wrong']:.6f}")
    print(f"    matched/generic  = {holdout['matched_over_generic']:.6f}")
    print(f"    passed           = {holdout['passed']}")
    print(f"  passed             = {output['passed_readiness_gate']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
