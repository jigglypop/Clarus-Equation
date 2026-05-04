"""Mouse IBL/OpenAlyx cross-session region generalization gate.

The previous mouse gates showed that one thalamic/visual strict session and
one NYU-30 motor/striatal multi-probe session pass region/action decoding
checks.  This script repeats the same decoding protocol across a compact panel
of motor/striatal candidate sessions to test whether the effect is a
single-session accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from mouse_ibl_multi_probe_region_gate import evaluate as evaluate_multi_probe
from mouse_ibl_region_decision_action_gate import evaluate as evaluate_single_probe


RESULT_JSON = Path(__file__).with_name(
    "mouse_ibl_cross_session_region_generalization_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_cross_session_region_generalization_report.md"
)


CANDIDATES = [
    {
        "name": "witten29_thalamic_visual_reference",
        "kind": "single",
        "eid": "d2832a38-27f6-452d-91d6-af72d794136c",
        "session_ref": "wittenlab/Subjects/ibl_witten_29/2021-06-08/001",
        "collection": "alf/probe00/pykilosort",
        "reason": "first strict-session thalamic/visual/hippocampal reference",
    },
    {
        "name": "nyu30_motor_striatal_multi_probe",
        "kind": "multi",
        "eid": "5ec72172-3901-4771-8777-6e9490ca51fc",
        "session_ref": "angelakilab/Subjects/NYU-30/2020-10-22/001",
        "collections": ["alf/probe00/pykilosort", "alf/probe01/pykilosort"],
        "reason": "same-session motor cortex plus striatal/septal multi-probe bridge",
    },
    {
        "name": "dy014_striatal_septal_probe",
        "kind": "single",
        "eid": "4720c98a-a305-4fba-affb-bbfa00a724a4",
        "session_ref": "danlab/Subjects/DY_014/2020-07-14/001",
        "collection": "alf/probe01/pykilosort",
        "reason": "highest target-family spike support in motor-striatum audit",
    },
    {
        "name": "dy011_motor_cortex_probe",
        "kind": "single",
        "eid": "cf43dbb1-6992-40ec-a5f9-e8e838d0f643",
        "session_ref": "danlab/Subjects/DY_011/2020-02-08/001",
        "collection": "alf/probe00/pykilosort",
        "reason": "single-probe motor cortex candidate",
    },
    {
        "name": "dy008_cp_somatosensory_thalamic_probe",
        "kind": "single",
        "eid": "ee13c19e-2790-4418-97ca-48f02e8013bb",
        "session_ref": "danlab/Subjects/DY_008/2020-03-04/001",
        "collection": "alf/probe00/pykilosort",
        "reason": "CP plus somatosensory cortex/thalamus candidate",
    },
]


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    base = {
        "eid": candidate["eid"],
        "session_ref": candidate["session_ref"],
        "folds": args.folds,
        "ridge": args.ridge,
        "permutations": args.permutations,
        "seed": args.seed,
        "min_acronym_spikes": args.min_acronym_spikes,
        "min_abs_wheel_displacement": args.min_abs_wheel_displacement,
    }
    if candidate["kind"] == "single":
        base["collection"] = candidate["collection"]
    else:
        base["collections"] = candidate["collections"]
    return SimpleNamespace(**base)


def strip_for_panel(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "family_names",
            "target_metadata",
            "window_specs",
            "acronym_group_names",
        }
    }


def best_rows_by_target(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        non_global = [
            row for row in target["rows"] if row["model"] != "global_rate"
        ]
        best = max(non_global, key=lambda row: row["balanced_accuracy"])
        global_row = next(row for row in target["rows"] if row["model"] == "global_rate")
        target_passed = any(row["passed"] for row in non_global)
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "best_model": best["model"],
                "best_balanced_accuracy": best["balanced_accuracy"],
                "best_auc": best["auc"],
                "best_p_balanced_accuracy_ge_observed": best[
                    "p_balanced_accuracy_ge_observed"
                ],
                "global_balanced_accuracy": global_row["balanced_accuracy"],
                "delta_vs_global_rate": best["balanced_accuracy"]
                - global_row["balanced_accuracy"],
                "target_passed": bool(target_passed),
                "n_trials": best["n_trials"],
                "class_counts": best["class_counts"],
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
        "collections": candidate.get("collections", [candidate.get("collection")]),
        "reason": candidate["reason"],
        "trial_count": result["trial_count"],
        "target_rows": target_rows,
        "passed_target_count": sum(row["target_passed"] for row in target_rows),
        "all_targets_passed": all(row["target_passed"] for row in target_rows),
    }


def aggregate(candidates: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    results = []
    summaries = []
    for candidate in candidates:
        ns = candidate_namespace(candidate, args)
        if candidate["kind"] == "single":
            result = evaluate_single_probe(ns)
        else:
            result = evaluate_multi_probe(ns)
        results.append(
            {
                "candidate": candidate,
                "result": strip_for_panel(result),
            }
        )
        summaries.append(candidate_summary(candidate, result))

    target_names = sorted({row["target"] for item in summaries for row in item["target_rows"]})
    target_replication = {}
    for target in target_names:
        target_rows = [
            row for item in summaries for row in item["target_rows"] if row["target"] == target
        ]
        target_replication[target] = {
            "candidate_count": len(target_rows),
            "passed_count": sum(row["target_passed"] for row in target_rows),
            "mean_best_balanced_accuracy": sum(
                row["best_balanced_accuracy"] for row in target_rows
            )
            / len(target_rows),
            "mean_delta_vs_global_rate": sum(
                row["delta_vs_global_rate"] for row in target_rows
            )
            / len(target_rows),
        }

    passed_candidates = sum(item["all_targets_passed"] for item in summaries)
    return {
        "candidate_count": len(candidates),
        "passed_all_targets_count": passed_candidates,
        "generalization_passed": bool(
            passed_candidates >= max(2, len(candidates) // 2)
            and target_replication.get("choice_sign", {}).get("passed_count", 0) >= 3
            and target_replication.get("wheel_action_direction", {}).get("passed_count", 0)
            >= 3
        ),
        "folds": args.folds,
        "ridge": args.ridge,
        "permutations": args.permutations,
        "seed": args.seed,
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx cross-session region generalization gate",
        "",
        "목표는 mouse region/action 항이 NYU-30 한 세션의 특이 readout인지, 후보 세션 패널에서 반복되는지 확인하는 것이다.",
        "각 후보는 같은 fixed-window region/acronym decoder와 global-rate baseline, label permutation null을 사용한다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- generalization passed: `{output['generalization_passed']}`",
        "",
        "## target replication",
        "",
        "| target | candidates | passed | mean best BA | mean delta global |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["passed_count"]),
                    f"{row['mean_best_balanced_accuracy']:.6f}",
                    f"{row['mean_delta_vs_global_rate']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | kind | collections | trials | passed targets | choice BA | speed BA | wheel BA |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in output["summaries"]:
        by_target = {row["target"]: row for row in summary["target_rows"]}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['name']}`",
                    summary["kind"],
                    ", ".join(f"`{collection}`" for collection in summary["collections"]),
                    str(summary["trial_count"]),
                    str(summary["passed_target_count"]),
                    f"{by_target['choice_sign']['best_balanced_accuracy']:.6f}",
                    f"{by_target['first_movement_speed']['best_balanced_accuracy']:.6f}",
                    f"{by_target['wheel_action_direction']['best_balanced_accuracy']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## per-candidate details", ""])
    for summary in output["summaries"]:
        lines.extend(
            [
                f"### {summary['name']}",
                "",
                f"- eid: `{summary['eid']}`",
                f"- session: `{summary['session_ref']}`",
                f"- reason: {summary['reason']}",
                "",
                "| target | best model | n | class counts | BA | AUC | p(BA>=obs) | global BA | delta global | pass |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in summary["target_rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['target']}`",
                        f"`{row['best_model']}`",
                        str(row["n_trials"]),
                        "`" + json.dumps(row["class_counts"], sort_keys=True) + "`",
                        f"{row['best_balanced_accuracy']:.6f}",
                        f"{row['best_auc']:.6f}",
                        f"{row['best_p_balanced_accuracy_ge_observed']:.6f}",
                        f"{row['global_balanced_accuracy']:.6f}",
                        f"{row['delta_vs_global_rate']:.6f}",
                        str(row["target_passed"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## verdict",
            "",
            f"- candidates passing all three targets: {output['passed_all_targets_count']} / {output['candidate_count']}",
            f"- generalization passed: `{output['generalization_passed']}`",
            "",
            "이 gate는 mouse 항을 완전히 닫는 최종 다기관 검정은 아니지만, region/probe-indexed action readout이 단일 NYU-30 세션에만 묶인 현상은 아니라는 중간 결론을 준다.",
        ]
    )
    return "\n".join(lines) + "\n"


def selected_candidates(names: list[str] | None) -> list[dict[str, object]]:
    if not names:
        return CANDIDATES
    lookup = {candidate["name"]: candidate for candidate in CANDIDATES}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(f"Unknown candidate names: {', '.join(missing)}")
    return [lookup[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-acronym-spikes", type=int, default=100_000)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=1e-3)
    args = parser.parse_args()

    candidates = selected_candidates(args.candidates)
    output = aggregate(candidates, args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx cross-session region generalization gate")
    print(f"  candidates={output['candidate_count']}")
    print(f"  passed_all_targets={output['passed_all_targets_count']}")
    print(f"  generalization_passed={output['generalization_passed']}")
    for summary in output["summaries"]:
        print(
            "  "
            + f"{summary['name']} passed_targets={summary['passed_target_count']} "
            + f"trials={summary['trial_count']}"
        )
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
