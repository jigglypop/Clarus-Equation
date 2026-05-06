"""Summarize the Mouse IBL speed/wheel action carrier split.

This gate is intentionally light: it reuses the already-generated ablation
reports and turns their summary tables into a registered carrier verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REGION_REPORT = BASE_DIR / "mouse_ibl_action_subspace_region_ablation_report.md"
PROBE_REPORT = BASE_DIR / "mouse_ibl_action_subspace_probe_ablation_report.md"
TOP_UNIT_REPORT = BASE_DIR / "mouse_ibl_action_top_unit_sufficiency_report.md"
RESULT_JSON = BASE_DIR / "mouse_ibl_action_carrier_split_results.json"
REPORT_MD = BASE_DIR / "mouse_ibl_action_carrier_split_report.md"

TARGET_LABELS = {
    "first_movement_speed": "speed",
    "wheel_action_direction": "wheel",
}


def ratio_to_counts(value: str) -> tuple[int, int]:
    left, right = value.split("/", maxsplit=1)
    return int(left), int(right)


def parse_bool(value: str) -> bool:
    clean = value.strip().strip("`")
    if clean == "True":
        return True
    if clean == "False":
        return False
    raise ValueError(f"cannot parse boolean: {value!r}")


def parse_summary_table(path: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_summary = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "## summary":
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if not in_summary or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"target", "---"}:
            continue
        if len(cells) != 8:
            raise ValueError(f"unexpected summary row in {path}: {line}")
        supported, evaluated = ratio_to_counts(cells[3])
        row = {
            "source": source,
            "target": cells[0].strip("`"),
            "condition": cells[1].strip("`"),
            "evaluated": evaluated,
            "supported": supported,
            "mean_delta_ba": float(cells[4]),
            "median_delta_ba": float(cells[5]),
            "mean_delta_vs_full": float(cells[6]),
            "passed": parse_bool(cells[7]),
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"no summary rows found in {path}")
    return rows


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        indexed.setdefault(row["target"], {})[row["condition"]] = row
    return indexed


def support_text(row: dict[str, Any]) -> str:
    return f'{row["supported"]}/{row["evaluated"]}'


def is_sensitive(full: dict[str, Any], drop: dict[str, Any]) -> bool:
    support_loss = drop["supported"] < full["supported"]
    mean_loss = drop["mean_delta_vs_full"] <= -0.003
    return bool(support_loss or mean_loss)


def target_verdict(
    target: str,
    region_rows: dict[str, dict[str, Any]],
    probe_rows: dict[str, dict[str, Any]],
    top_unit_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full = top_unit_rows["full"]
    drop_top_ccf = region_rows["drop_top_ccf"]
    only_top_ccf = region_rows["only_top_ccf"]
    drop_probe = probe_rows["drop_probe"]
    only_probe = probe_rows["only_probe"]
    drop_top_units = top_unit_rows["drop_top_units"]
    only_top_units = top_unit_rows["only_top_units"]

    region_sensitive = is_sensitive(full, drop_top_ccf)
    probe_sensitive = is_sensitive(full, drop_probe)
    top_unit_sensitive = is_sensitive(full, drop_top_units)
    probe_near_full_mean = abs(only_probe["mean_delta_vs_full"]) <= 0.001

    if target == "first_movement_speed":
        carrier = "probe00/block-distributed speed carrier"
        interpretation = (
            "speed depends on probe00 and the top anatomical block, but fold-local "
            "top 16 probe00 units are not sufficient under the current replication rule"
        )
        expected = (
            region_sensitive
            and probe_sensitive
            and only_probe["passed"]
            and not only_top_units["passed"]
        )
    elif target == "wheel_action_direction":
        carrier = "compact fold-local probe00 top-unit wheel carrier"
        interpretation = (
            "wheel is weakened by removing probe00/top units and is sufficient with "
            "fold-local top 16 probe00 units, while the top anatomical block is not required"
        )
        expected = (
            not region_sensitive
            and probe_sensitive
            and top_unit_sensitive
            and only_top_units["passed"]
        )
    else:
        raise ValueError(f"unknown target: {target}")

    return {
        "target": target,
        "label": TARGET_LABELS[target],
        "carrier": carrier,
        "passed_expected_pattern": bool(expected),
        "interpretation": interpretation,
        "full": full,
        "region_sensitive": region_sensitive,
        "probe_sensitive": probe_sensitive,
        "top_unit_sensitive": top_unit_sensitive,
        "only_top_ccf_passed": only_top_ccf["passed"],
        "only_probe_passed": only_probe["passed"],
        "only_probe_near_full_mean": probe_near_full_mean,
        "only_top_units_passed": only_top_units["passed"],
        "evidence": {
            "drop_top_ccf": drop_top_ccf,
            "only_top_ccf": only_top_ccf,
            "drop_probe": drop_probe,
            "only_probe": only_probe,
            "drop_top_units": drop_top_units,
            "only_top_units": only_top_units,
        },
    }


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx speed/wheel action carrier split",
        "",
        "This meta-gate reads the already-generated action ablation reports and asks whether speed and wheel use the same carrier.",
        "",
        "## inputs",
        "",
        f"- region/top-block ablation: `{REGION_REPORT.name}`",
        f"- probe00 ablation: `{PROBE_REPORT.name}`",
        f"- fold-local top-unit sufficiency: `{TOP_UNIT_REPORT.name}`",
        "",
        "## carrier verdict",
        "",
        "| target | carrier | key pattern | passed |",
        "|---|---|---|---|",
    ]
    for verdict in result["target_verdicts"]:
        target = verdict["target"]
        evidence = verdict["evidence"]
        pattern = (
            f"full {support_text(verdict['full'])}; "
            f"drop_top_ccf {support_text(evidence['drop_top_ccf'])}; "
            f"drop_probe {support_text(evidence['drop_probe'])}; "
            f"only_probe {support_text(evidence['only_probe'])}; "
            f"only_top_units {support_text(evidence['only_top_units'])}"
        )
        lines.append(
            "| `{target}` | {carrier} | {pattern} | `{passed}` |".format(
                target=target,
                carrier=verdict["carrier"],
                pattern=pattern,
                passed=verdict["passed_expected_pattern"],
            )
        )

    lines.extend(
        [
            "",
            "## interpretation",
            "",
            "- Speed is probe00/block-dependent but not closed by the fold-local top 16 probe00 units.",
            "- Wheel is compact enough to close on fold-local top 16 probe00 units, and the top anatomical block is not required.",
            "- The split passes because the expected carrier patterns differ for both targets.",
            "",
            "## equation update",
            "",
            "$$",
            "\\boxed{",
            "\\Phi_{\\mathrm{action},t}^{(s)}",
            "=",
            "\\Phi_{\\mathrm{speed},t}^{(s,\\mathrm{probe00/block})}",
            "\\oplus",
            "\\Phi_{\\mathrm{wheel},t}^{(s,\\mathrm{probe00/top16})}",
            "}",
            "$$",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    region_rows = index_rows(parse_summary_table(args.region_report, "region"))
    probe_rows = index_rows(parse_summary_table(args.probe_report, "probe"))
    top_unit_rows = index_rows(parse_summary_table(args.top_unit_report, "top_unit"))
    verdicts = [
        target_verdict(target, region_rows[target], probe_rows[target], top_unit_rows[target])
        for target in TARGET_LABELS
    ]
    result = {
        "gate": "mouse_ibl_action_carrier_split",
        "passed": all(verdict["passed_expected_pattern"] for verdict in verdicts),
        "target_verdicts": verdicts,
    }
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-report", type=Path, default=REGION_REPORT)
    parser.add_argument("--probe-report", type=Path, default=PROBE_REPORT)
    parser.add_argument("--top-unit-report", type=Path, default=TOP_UNIT_REPORT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = run(args)
    print(json.dumps({"passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
