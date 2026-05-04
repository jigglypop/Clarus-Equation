"""Event-level ds000201 cognitive/arousal domain gate.

This checks whether small public event files support separable operators for:

- PVT: vigilance / sustained attention
- workingmemorytest: working-memory load
- sleepiness: subjective arousal / KSS

The state is derived from behavioral/event rows only. It is a readiness gate,
not a neural proof.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import mean


DATASET = "ds000201"
SNAPSHOT = "1.0.3"
DEFAULT_DATA_ROOT = Path("data/openneuro/ds000201")
TASKS = ["PVT", "workingmemorytest", "sleepiness"]
REGIONS = [
    "subjective_arousal",
    "vigilance",
    "working_memory",
    "motor_response",
    "background",
]


def numeric(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    if value in {"", "n/a", "NA", "nan"}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def project_simplex(values: list[float]) -> list[float]:
    clipped = [max(float(value), 1e-12) for value in values]
    total = sum(clipped)
    return [value / total for value in clipped]


def parse_pvt(text: str) -> dict[str, float]:
    rts = []
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            value = numeric(parts[2])
            if value is not None and value > 0:
                rts.append(value)
    if not rts:
        return {"trial_count": 0.0, "mean_rt": float("nan"), "lapse_fraction": float("nan")}
    return {
        "trial_count": float(len(rts)),
        "mean_rt": mean(rts),
        "lapse_fraction": sum(rt >= 0.5 for rt in rts) / len(rts),
        "fast_fraction": sum(rt <= 0.25 for rt in rts) / len(rts),
    }


def parse_sleepiness(text: str) -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    ratings = [
        value
        for row in rows
        if (value := numeric(row.get("KSS_rating"))) is not None
    ]
    rts = [
        value
        for row in rows
        if (value := numeric(row.get("response_time"))) is not None and value > 0
    ]
    return {
        "trial_count": float(len(ratings)),
        "mean_kss": mean(ratings) if ratings else float("nan"),
        "mean_rt": mean(rts) if rts else float("nan"),
    }


def parse_working_memory(text: str) -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    correct = [value for row in rows if (value := numeric(row.get("correct"))) is not None]
    rts = [
        value
        for row in rows
        if (value := numeric(row.get("response_time"))) is not None and value > 0
    ]
    loads = [value for row in rows if (value := numeric(row.get("no_grids"))) is not None]
    return {
        "trial_count": float(len(rows)),
        "accuracy": mean(correct) if correct else float("nan"),
        "mean_rt": mean(rts) if rts else float("nan"),
        "mean_load": mean(loads) if loads else float("nan"),
    }


def observed_state(task: str, stats: dict[str, float]) -> list[float]:
    arousal = 0.08
    vigilance = 0.08
    memory = 0.08
    motor = 0.08
    background = 0.68

    if task == "PVT":
        lapse = stats.get("lapse_fraction", 0.0)
        rt = stats.get("mean_rt", 0.3)
        lapse = 0.0 if not math.isfinite(lapse) else min(max(lapse / 0.20, 0.0), 1.5)
        rt_drive = 0.0 if not math.isfinite(rt) else min(max((rt - 0.22) / 0.35, 0.0), 1.5)
        vigilance += 0.24 * (0.65 * lapse + 0.35 * rt_drive)
        motor += 0.10 * rt_drive
        background -= 0.12 * (0.65 * lapse + 0.35 * rt_drive)
    elif task == "workingmemorytest":
        acc = stats.get("accuracy", 1.0)
        load = stats.get("mean_load", 2.0)
        rt = stats.get("mean_rt", 1.0)
        error_drive = 0.0 if not math.isfinite(acc) else min(max(1.0 - acc, 0.0), 1.0)
        load_drive = 0.0 if not math.isfinite(load) else min(max((load - 2.0) / 6.0, 0.0), 1.0)
        rt_drive = 0.0 if not math.isfinite(rt) else min(max((rt - 1.0) / 5.0, 0.0), 1.0)
        memory += 0.24 * (0.45 * error_drive + 0.40 * load_drive + 0.15 * rt_drive)
        motor += 0.06 * rt_drive
        background -= 0.12 * (0.45 * error_drive + 0.40 * load_drive + 0.15 * rt_drive)
    elif task == "sleepiness":
        kss = stats.get("mean_kss", 1.0)
        rt = stats.get("mean_rt", 1.0)
        kss_drive = 0.0 if not math.isfinite(kss) else min(max((kss - 1.0) / 8.0, 0.0), 1.0)
        rt_drive = 0.0 if not math.isfinite(rt) else min(max((rt - 1.0) / 8.0, 0.0), 1.0)
        arousal += 0.26 * (0.75 * kss_drive + 0.25 * rt_drive)
        motor += 0.04 * rt_drive
        background -= 0.13 * (0.75 * kss_drive + 0.25 * rt_drive)
    else:
        raise ValueError(f"unknown task: {task}")
    return project_simplex([arousal, vigilance, memory, motor, background])


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
            observed = [float(case["observed_state"][name]) for name in REGIONS]  # type: ignore[index]
            wrong = min(
                squared_loss(observed, prototypes[name])
                for name in TASKS
                if name != task
            )
            losses["matched"] += squared_loss(observed, prototypes[task])
            losses["wrong_task"] += wrong
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


def task_from_path(path: Path) -> str:
    name = path.name
    if "task-PVT" in name:
        return "PVT"
    if "task-workingmemorytest" in name:
        return "workingmemorytest"
    if "task-sleepiness" in name:
        return "sleepiness"
    raise ValueError(f"unknown task path: {path}")


def parse_task(task: str, text: str) -> dict[str, float]:
    if task == "PVT":
        return parse_pvt(text)
    if task == "workingmemorytest":
        return parse_working_memory(text)
    if task == "sleepiness":
        return parse_sleepiness(text)
    raise ValueError(f"unknown task: {task}")


def load_cases(data_root: Path, subjects: list[str]) -> list[dict[str, object]]:
    cases = []
    for subject in subjects:
        for path in sorted((data_root / subject).glob("ses-*/*/*_events.tsv")):
            if not any(marker in path.name for marker in ("task-PVT", "task-workingmemorytest", "task-sleepiness")):
                continue
            task = task_from_path(path)
            stats = parse_task(task, path.read_text(encoding="utf-8"))
            state = observed_state(task, stats)
            cases.append(
                {
                    "subject": subject,
                    "session": path.parts[-3],
                    "task": task,
                    "path": str(path.relative_to(data_root)),
                    "stats": stats,
                    "observed_state": dict(zip(REGIONS, state)),
                }
            )
    return cases


def subject_slice(data_root: Path, start: int, count: int) -> list[str]:
    rows = csv.DictReader(
        io.StringIO((data_root / "participants.tsv").read_text(encoding="utf-8")),
        delimiter="\t",
    )
    subjects = [row["participant_id"] for row in rows if row.get("participant_id")]
    return subjects[max(start - 1, 0) : max(start - 1, 0) + count]


def summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    holdout = prototype_holdout(cases)
    by_task = {}
    for task in TASKS:
        task_cases = [case for case in cases if str(case["task"]) == task]
        by_task[task] = {
            "case_count": len(task_cases),
            "trial_count": int(sum(float(case["stats"].get("trial_count", 0.0)) for case in task_cases)),  # type: ignore[union-attr]
        }
    return {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "gate": "event-derived cognitive/arousal readiness; not a claim-ready neural gate",
        "case_count": len(cases),
        "regions": REGIONS,
        "task_summary": by_task,
        "prototype_holdout": holdout,
        "passed_readiness_gate": bool(holdout["passed"]) and all(
            by_task[task]["case_count"] > 0 for task in TASKS
        ),
        "next_required_for_neural_claim": [
            "download selected PVT/working-memory/sleepiness BOLD where available",
            "extract frontoparietal, thalamic/arousal, motor, and default-mode proxies",
            "test matched cognitive operator on p_r instead of event-derived state",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--subject-count", type=int, default=4)
    args = parser.parse_args()

    subjects = subject_slice(args.data_root, args.start_subject, args.subject_count)
    cases = load_cases(args.data_root, subjects)
    output = summarize(cases)
    output["requested_subjects"] = subjects
    output["data_root"] = str(args.data_root)
    output["cases"] = cases
    out_path = Path(__file__).with_name("ds000201_cognitive_gate_results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ds000201 cognitive/arousal readiness gate")
    print(f"  cases              = {output['case_count']}")
    for task, row in output["task_summary"].items():  # type: ignore[union-attr]
        print(f"  {task:18s}: cases={row['case_count']}, trials={row['trial_count']}")
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
