"""Mouse IBL/OpenAlyx latent-axis stability analysis.

The directed-axis gate found target-specific best innovation axes.  This
post-hoc gate asks whether those axes are stable enough to treat as a shared
axis/subspace, or whether the best axis is mostly a per-session selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(__file__).with_name("mouse_ibl_directed_latent_axis_split_results.json")
RESULT_JSON = Path(__file__).with_name("mouse_ibl_latent_axis_stability_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_latent_axis_stability_report.md")

TARGETS = ("choice_sign", "first_movement_speed", "wheel_action_direction")


def normalized_entropy(counts: Counter[int], components: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    probs = np.asarray([count / total for count in counts.values()], dtype=float)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(components))


def monte_carlo_top_count_pvalue(
    observed_top_count: int,
    sessions: int,
    components: int,
    draws: int,
    rng: np.random.Generator,
) -> float:
    exceed = 0
    for _ in range(draws):
        values = rng.integers(1, components + 1, size=sessions)
        top = Counter(int(v) for v in values).most_common(1)[0][1]
        if top >= observed_top_count:
            exceed += 1
    return float(exceed / draws)


def target_by_name(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {target["target"]: target for target in result["targets"]}


def analyze(data: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    components = int(data["components"])
    sessions = int(data["candidate_count"])
    target_summaries = []
    for target in TARGETS:
        axes = []
        increments = []
        positive_axis_counts = []
        for result in data["candidate_results"]:
            row = target_by_name(result)[target]
            axes.append(int(row["best_axis"]))
            increments.append(float(row["best_axis_increment"]))
            positive_axis_counts.append(int(row["positive_axis_count"]))
        counts = Counter(axes)
        most_common = counts.most_common()
        top1_axis, top1_count = most_common[0]
        top3_count = sum(count for _, count in most_common[:3])
        target_summaries.append(
            {
                "target": target,
                "axis_counts": dict(sorted(counts.items())),
                "top1_axis": top1_axis,
                "top1_count": top1_count,
                "top1_share": float(top1_count / sessions),
                "top3_count": top3_count,
                "top3_share": float(top3_count / sessions),
                "normalized_entropy": normalized_entropy(counts, components),
                "top1_uniform_null_p": monte_carlo_top_count_pvalue(
                    top1_count,
                    sessions,
                    components,
                    args.null_draws,
                    rng,
                ),
                "mean_best_increment": float(np.mean(increments)),
                "mean_positive_axis_count": float(np.mean(positive_axis_counts)),
                "axis_identity_stable": bool(
                    top1_count >= args.min_top1_count
                    and normalized_entropy(counts, components) <= args.max_entropy
                ),
                "subspace_concentrated": bool(top3_count >= args.min_top3_count),
            }
        )

    pair_counts = Counter()
    all_same = 0
    low_axis_all = 0
    session_rows = []
    for result in data["candidate_results"]:
        targets = target_by_name(result)
        choice = int(targets["choice_sign"]["best_axis"])
        speed = int(targets["first_movement_speed"]["best_axis"])
        wheel = int(targets["wheel_action_direction"]["best_axis"])
        if choice == speed:
            pair_counts["choice_speed_same"] += 1
        if choice == wheel:
            pair_counts["choice_wheel_same"] += 1
        if speed == wheel:
            pair_counts["speed_wheel_same"] += 1
        if choice == speed == wheel:
            all_same += 1
        if max(choice, speed, wheel) <= args.low_axis_cutoff:
            low_axis_all += 1
        session_rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "choice_axis": choice,
                "speed_axis": speed,
                "wheel_axis": wheel,
                "unique_axis_count": len({choice, speed, wheel}),
                "all_axes_low": bool(max(choice, speed, wheel) <= args.low_axis_cutoff),
            }
        )

    stable_targets = sum(row["axis_identity_stable"] for row in target_summaries)
    concentrated_targets = sum(row["subspace_concentrated"] for row in target_summaries)
    return {
        "source": str(args.input),
        "candidate_count": sessions,
        "components": components,
        "min_top1_count": args.min_top1_count,
        "min_top3_count": args.min_top3_count,
        "max_entropy": args.max_entropy,
        "low_axis_cutoff": args.low_axis_cutoff,
        "null_draws": args.null_draws,
        "seed": args.seed,
        "target_summaries": target_summaries,
        "within_session_summary": {
            "choice_speed_same": int(pair_counts["choice_speed_same"]),
            "choice_wheel_same": int(pair_counts["choice_wheel_same"]),
            "speed_wheel_same": int(pair_counts["speed_wheel_same"]),
            "all_three_same": all_same,
            "all_three_low_axis": low_axis_all,
        },
        "session_rows": session_rows,
        "stability_summary": {
            "axis_identity_stable_targets": int(stable_targets),
            "subspace_concentrated_targets": int(concentrated_targets),
            "axis_identity_gate_passed": bool(stable_targets == len(TARGETS)),
            "subspace_gate_passed": bool(concentrated_targets == len(TARGETS)),
        },
    }


def make_report(output: dict[str, object]) -> str:
    stability = output["stability_summary"]
    within = output["within_session_summary"]
    lines = [
        "# Mouse IBL/OpenAlyx latent-axis stability",
        "",
        "$$",
        "\\epsilon_{t,k_j}\\quad\\mathrm{stable?}",
        "$$",
        "",
        "## setup",
        "",
        f"- source: `{output['source']}`",
        f"- candidates: {output['candidate_count']}",
        f"- components: {output['components']}",
        f"- axis identity gate passed: `{stability['axis_identity_gate_passed']}`",
        f"- subspace gate passed: `{stability['subspace_gate_passed']}`",
        "",
        "## target axis distribution",
        "",
        "| target | top axis | top1 | top3 | entropy | top1 null p | mean best dBA | stable identity | concentrated subspace |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in output["target_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    str(row["top1_axis"]),
                    f"{row['top1_count']}/{output['candidate_count']}",
                    f"{row['top3_count']}/{output['candidate_count']}",
                    f"{row['normalized_entropy']:.6f}",
                    f"{row['top1_uniform_null_p']:.6f}",
                    f"{row['mean_best_increment']:.6f}",
                    f"`{row['axis_identity_stable']}`",
                    f"`{row['subspace_concentrated']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## within-session sharing",
            "",
            "| metric | count |",
            "|---|---:|",
            f"| choice-speed same axis | {within['choice_speed_same']}/{output['candidate_count']} |",
            f"| choice-wheel same axis | {within['choice_wheel_same']}/{output['candidate_count']} |",
            f"| speed-wheel same axis | {within['speed_wheel_same']}/{output['candidate_count']} |",
            f"| all three same axis | {within['all_three_same']}/{output['candidate_count']} |",
            f"| all three axes <= {output['low_axis_cutoff']} | {within['all_three_low_axis']}/{output['candidate_count']} |",
            "",
            "## per-session axes",
            "",
            "| candidate | choice | speed | wheel | unique axes | all low |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in output["session_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    str(row["choice_axis"]),
                    str(row["speed_axis"]),
                    str(row["wheel_axis"]),
                    str(row["unique_axis_count"]),
                    f"`{row['all_axes_low']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Single best axes are behavior-informative, but their identity is not yet stable enough to name one shared axis.",
            "- The next test should use a pre-registered low-dimensional subspace or nested axis selection inside outer folds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--min-top1-count", type=int, default=6)
    parser.add_argument("--min-top3-count", type=int, default=8)
    parser.add_argument("--max-entropy", type=float, default=0.70)
    parser.add_argument("--low-axis-cutoff", type=int, default=6)
    parser.add_argument("--null-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4219)
    args = parser.parse_args()

    output = analyze(json.loads(args.input.read_text()), args)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    args.report_md.write_text(make_report(output))
    stability = output["stability_summary"]
    print("Mouse IBL/OpenAlyx latent-axis stability")
    for row in output["target_summaries"]:
        print(
            f"  {row['target']}: top1={row['top1_count']}/{output['candidate_count']} "
            f"top3={row['top3_count']}/{output['candidate_count']} "
            f"entropy={row['normalized_entropy']:.3f}"
        )
    print(f"  axis_identity_gate_passed={stability['axis_identity_gate_passed']}")
    print(f"  subspace_gate_passed={stability['subspace_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
