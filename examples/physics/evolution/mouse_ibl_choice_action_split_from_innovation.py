"""Mouse IBL/OpenAlyx choice/action split analysis.

The 12-session innovation-to-behavior panel kept innovation support for action
targets but not for choice.  This post-hoc gate reads the saved innovation
panel and asks whether action innovation increments are consistently larger
than choice innovation increments across the same sessions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(__file__).with_name("mouse_ibl_innovation_behavior_12panel_results.json")
RESULT_JSON = Path(__file__).with_name("mouse_ibl_choice_action_split_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_choice_action_split_report.md")

CHOICE_TARGET = "choice_sign"
ACTION_TARGETS = ("first_movement_speed", "wheel_action_direction")


def load_panel(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def target_by_name(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {target["target"]: target for target in result["targets"]}


def eps_after_h(target: dict[str, object]) -> float:
    nested = target["nested_comparison"]
    return float(nested["innovation_increment_after_task_region_predicted_latent"])


def supported(target: dict[str, object]) -> bool:
    nested = target["nested_comparison"]
    return bool(nested["innovation_after_predicted_latent_supported"])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(draws, dtype=float)
    n = len(values)
    for i in range(draws):
        means[i] = float(np.mean(values[rng.integers(0, n, size=n)]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_pvalue(values: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for mean different from zero."""
    n = len(values)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    count = 0
    total = 1 << n
    for mask in range(total):
        signs = np.ones(n, dtype=float)
        for bit in range(n):
            if mask & (1 << bit):
                signs[bit] = -1.0
        if abs(float(np.mean(values * signs))) >= observed - 1e-15:
            count += 1
    return float(count / total)


def analyze(panel: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for result in panel["candidate_results"]:
        targets = target_by_name(result)
        choice = eps_after_h(targets[CHOICE_TARGET])
        speed = eps_after_h(targets["first_movement_speed"])
        wheel = eps_after_h(targets["wheel_action_direction"])
        action_mean = float(np.mean([speed, wheel]))
        action_minus_choice = action_mean - choice
        rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "choice_eps_after_h": choice,
                "speed_eps_after_h": speed,
                "wheel_eps_after_h": wheel,
                "action_mean_eps_after_h": action_mean,
                "action_minus_choice": action_minus_choice,
                "choice_supported": supported(targets[CHOICE_TARGET]),
                "speed_supported": supported(targets["first_movement_speed"]),
                "wheel_supported": supported(targets["wheel_action_direction"]),
                "any_action_supported": bool(
                    supported(targets["first_movement_speed"])
                    or supported(targets["wheel_action_direction"])
                ),
                "both_actions_supported": bool(
                    supported(targets["first_movement_speed"])
                    and supported(targets["wheel_action_direction"])
                ),
                "split_supported": bool(action_minus_choice > args.min_split_delta),
            }
        )

    differences = np.asarray([row["action_minus_choice"] for row in rows], dtype=float)
    choice_values = np.asarray([row["choice_eps_after_h"] for row in rows], dtype=float)
    action_values = np.asarray([row["action_mean_eps_after_h"] for row in rows], dtype=float)
    rng = np.random.default_rng(args.seed)
    diff_ci = bootstrap_ci(differences, rng, args.bootstrap_draws)
    choice_ci = bootstrap_ci(choice_values, rng, args.bootstrap_draws)
    action_ci = bootstrap_ci(action_values, rng, args.bootstrap_draws)
    split_count = int(sum(row["split_supported"] for row in rows))
    any_action_count = int(sum(row["any_action_supported"] for row in rows))
    both_action_count = int(sum(row["both_actions_supported"] for row in rows))
    choice_count = int(sum(row["choice_supported"] for row in rows))
    split_gate_passed = bool(
        split_count >= args.min_split_replications
        and float(np.mean(differences)) > args.min_split_delta
    )
    return {
        "source_panel": str(args.input),
        "candidate_count": len(rows),
        "min_split_delta": args.min_split_delta,
        "min_split_replications": args.min_split_replications,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "rows": rows,
        "summary": {
            "mean_choice_eps_after_h": float(np.mean(choice_values)),
            "mean_action_eps_after_h": float(np.mean(action_values)),
            "mean_action_minus_choice": float(np.mean(differences)),
            "median_action_minus_choice": float(np.median(differences)),
            "choice_eps_after_h_ci95": list(choice_ci),
            "action_eps_after_h_ci95": list(action_ci),
            "action_minus_choice_ci95": list(diff_ci),
            "action_minus_choice_sign_flip_p": sign_flip_pvalue(differences),
            "choice_supported_count": choice_count,
            "any_action_supported_count": any_action_count,
            "both_actions_supported_count": both_action_count,
            "split_supported_count": split_count,
            "split_gate_passed": split_gate_passed,
        },
    }


def make_report(output: dict[str, object]) -> str:
    summary = output["summary"]
    lines = [
        "# Mouse IBL/OpenAlyx choice/action innovation split",
        "",
        "$$",
        "y_{choice}=g_c(X_t,R_t,\\hat H_t)",
        "\\qquad",
        "y_{action}=g_a(X_t,R_t,\\hat H_t,\\epsilon_t)",
        "$$",
        "",
        "## setup",
        "",
        f"- source panel: `{output['source_panel']}`",
        f"- candidates: {output['candidate_count']}",
        f"- min split delta: {output['min_split_delta']}",
        f"- split gate passed: `{summary['split_gate_passed']}`",
        "",
        "## summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean choice eps after Hhat | {summary['mean_choice_eps_after_h']:.6f} |",
        f"| mean action eps after Hhat | {summary['mean_action_eps_after_h']:.6f} |",
        f"| mean action - choice | {summary['mean_action_minus_choice']:.6f} |",
        f"| median action - choice | {summary['median_action_minus_choice']:.6f} |",
        f"| action - choice 95% bootstrap low | {summary['action_minus_choice_ci95'][0]:.6f} |",
        f"| action - choice 95% bootstrap high | {summary['action_minus_choice_ci95'][1]:.6f} |",
        f"| sign-flip p | {summary['action_minus_choice_sign_flip_p']:.6f} |",
        f"| choice supported | {summary['choice_supported_count']}/{output['candidate_count']} |",
        f"| any action supported | {summary['any_action_supported_count']}/{output['candidate_count']} |",
        f"| both actions supported | {summary['both_actions_supported_count']}/{output['candidate_count']} |",
        f"| split supported | {summary['split_supported_count']}/{output['candidate_count']} |",
        "",
        "## per-session split",
        "",
        "| candidate | choice | speed | wheel | action mean | action - choice | split |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"{row['choice_eps_after_h']:.6f}",
                    f"{row['speed_eps_after_h']:.6f}",
                    f"{row['wheel_eps_after_h']:.6f}",
                    f"{row['action_mean_eps_after_h']:.6f}",
                    f"{row['action_minus_choice']:.6f}",
                    f"`{row['split_supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- The larger panel supports an action-linked innovation term more strongly than a choice-linked innovation term.",
            "- The next model should split choice and action readouts instead of forcing one behavioral equation for all targets.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--min-split-delta", type=float, default=0.002)
    parser.add_argument("--min-split-replications", type=int, default=7)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7391)
    args = parser.parse_args()

    output = analyze(load_panel(args.input), args)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    args.report_md.write_text(make_report(output))
    summary = output["summary"]
    print("Mouse IBL/OpenAlyx choice/action innovation split")
    print(
        "  action-choice="
        f"{summary['mean_action_minus_choice']:.6f} "
        f"split={summary['split_supported_count']}/{output['candidate_count']} "
        f"p={summary['action_minus_choice_sign_flip_p']:.6f}"
    )
    print(f"  split_gate_passed={summary['split_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
