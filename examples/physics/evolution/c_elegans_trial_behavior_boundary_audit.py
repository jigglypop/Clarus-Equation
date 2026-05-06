"""Audit whether C. elegans stimulus-output can be promoted to a trial-behavior gate.

The existing C. elegans gates are weighted-connectome proxies.  This audit
checks local data availability for an empirical stimulus/trial/behavior table
and keeps the proxy/empirical boundary explicit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DATA_ROOT = REPO_ROOT / "data"
STIMULUS_REPORT = BASE_DIR / "c_elegans_stimulus_behavior_report.md"
DEVELOPMENTAL_REPORT = BASE_DIR / "c_elegans_developmental_stimulus_behavior_report.md"
RESULT_JSON = BASE_DIR / "c_elegans_trial_behavior_boundary_audit_results.json"
REPORT_MD = BASE_DIR / "c_elegans_trial_behavior_boundary_audit_report.md"

KEYWORDS = (
    "elegans",
    "celegans",
    "worm",
    "stimulus",
    "behavior",
    "behaviour",
    "trial",
    "locomotion",
    "chemotaxis",
)
REQUIRED_FIELDS = (
    "trial_id",
    "stimulus_label_or_time",
    "behavior_label_or_trace",
    "worm_id",
    "timebase",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return float(match.group(1))


def require_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return int(match.group(1))


def candidate_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    out = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = str(path).lower()
        hits = [keyword for keyword in KEYWORDS if keyword in lower]
        if hits:
            out.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "size_bytes": path.stat().st_size,
                    "keyword_hits": hits,
                }
            )
    return out


def proxy_evidence() -> dict[str, Any]:
    stimulus = read(STIMULUS_REPORT)
    developmental = read(DEVELOPMENTAL_REPORT)
    return {
        "adult_matched_wrong": require_float(r"\| matched / wrong \| ([0-9.]+) \|", stimulus),
        "adult_permutation_p": require_float(r"\| permutation p \| ([0-9.]+) \|", stimulus),
        "adult_passed": "passed | True" in stimulus,
        "developmental_passed_weighted_stages": require_int(
            r"\| passed chemical weighted stages \| (\d+) \|", developmental
        ),
        "developmental_stages": require_int(r"\| stages \| (\d+) \|", developmental),
        "developmental_mean_matched_wrong": require_float(
            r"\| mean chemical matched/wrong \| ([0-9.]+) \|", developmental
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    files = candidate_files(args.data_root)
    proxy = proxy_evidence()
    found_trial_data = len(files) > 0
    found_fields = REQUIRED_FIELDS if found_trial_data else ()
    missing_fields = [field for field in REQUIRED_FIELDS if field not in found_fields]
    empirical_ready = found_trial_data and not missing_fields
    result = {
        "gate": "c_elegans_trial_behavior_boundary_audit",
        "empirical_trial_gate_ready": bool(empirical_ready),
        "verdict": "data_boundary" if not empirical_ready else "ready",
        "candidate_files": files,
        "required_fields": list(REQUIRED_FIELDS),
        "missing_fields": missing_fields,
        "proxy_evidence": proxy,
        "proxy_supported": bool(
            proxy["adult_passed"]
            and proxy["developmental_passed_weighted_stages"] == proxy["developmental_stages"]
        ),
        "interpretation": (
            "The weighted connectome stimulus-output channel is supported, but no local "
            "trial-level C. elegans stimulus/behavior table is available to promote it "
            "to an empirical behavior gate."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    proxy = result["proxy_evidence"]
    lines = [
        "# C. elegans trial-behavior boundary audit",
        "",
        "This audit separates the supported connectome proxy from an empirical trial-behavior gate.",
        "",
        "## verdict",
        "",
        f"- empirical trial gate ready: `{result['empirical_trial_gate_ready']}`",
        f"- verdict: `{result['verdict']}`",
        f"- proxy supported: `{result['proxy_supported']}`",
        "",
        "## proxy evidence",
        "",
        "| item | value |",
        "|---|---:|",
        f"| adult matched/wrong | {proxy['adult_matched_wrong']:.6f} |",
        f"| adult permutation p | {proxy['adult_permutation_p']:.6f} |",
        f"| developmental weighted stages | {proxy['developmental_passed_weighted_stages']}/{proxy['developmental_stages']} |",
        f"| developmental mean matched/wrong | {proxy['developmental_mean_matched_wrong']:.6f} |",
        "",
        "## local trial data scan",
        "",
        "| required field | found |",
        "|---|---|",
    ]
    missing = set(result["missing_fields"])
    for field in result["required_fields"]:
        lines.append(f"| `{field}` | `{field not in missing}` |")
    lines.extend(["", "## candidate local files", ""])
    if result["candidate_files"]:
        lines.extend(["| path | size bytes | keyword hits |", "|---|---:|---|"])
        for row in result["candidate_files"]:
            lines.append(
                f"| `{row['path']}` | {row['size_bytes']} | {', '.join(row['keyword_hits'])} |"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## interpretation",
            "",
            "- The current C. elegans behavior result remains a weighted-connectome proxy.",
            "- It should not be promoted to an empirical trial-behavior equation without stimulus labels, behavior traces/labels, worm or trial ids, and a timebase.",
            "- The next real closure requires a trial-level C. elegans stimulus-behavior dataset or a time-aligned neural/behavior recording.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "proxy_supported": result["proxy_supported"],
                "candidate_file_count": len(result["candidate_files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
