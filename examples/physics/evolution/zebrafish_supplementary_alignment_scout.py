"""Scout zebrafish supplementary files for continuous alignment variables.

The remaining zebrafish bottleneck is timestamp-certified continuous movement
decoding. This script scans extracted supplementary folders for files and MAT
variables that might bridge neural e2 frames to behavior traces:

- e2/frame/timestamp/time variables
- stage/head/yolk/tail/speed/turn/heading variables
- behavior/bout/laser metadata

It does not prove alignment. It produces a compact candidate list for the next
gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.io import whosmat


DEFAULT_ROOT = Path("data/evolution/zebrafish/freely_swimming")
RESULT_JSON = Path(__file__).with_name("zebrafish_supplementary_alignment_scout_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_supplementary_alignment_scout_report.md")

KEYWORDS = {
    "neural": ("e2", "neural", "active", "activity", "propa", "corr"),
    "time": ("time", "timestamp", "frame", "fps", "hz", "sync"),
    "movement": ("stage", "head", "yolk", "tail", "speed", "turn", "heading", "angle", "bout"),
    "laser": ("laser", "left", "right", "stim", "onset"),
}


def score_text(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        group: sum(1 for keyword in keywords if keyword in lower)
        for group, keywords in KEYWORDS.items()
    }


def mat_variables(path: Path) -> list[dict[str, object]]:
    try:
        variables = whosmat(path)
    except Exception as exc:  # scipy cannot read HDF5/v7.3 or corrupt files.
        return [{"error": type(exc).__name__, "message": str(exc)[:200]}]
    rows = []
    for name, shape, dtype in variables:
        scores = score_text(name)
        rows.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype,
                "scores": scores,
                "score_total": sum(scores.values()),
            }
        )
    return rows


def scan(root: Path) -> dict[str, object]:
    file_rows = []
    mat_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        suffix = path.suffix.lower()
        file_scores = score_text(relative)
        file_score_total = sum(file_scores.values())
        if suffix in {".mat", ".txt", ".csv", ".m"} or file_score_total:
            file_rows.append(
                {
                    "path": relative,
                    "suffix": suffix,
                    "size": path.stat().st_size,
                    "scores": file_scores,
                    "score_total": file_score_total,
                }
            )
        if suffix == ".mat":
            vars_ = mat_variables(path)
            variable_score = sum(int(row.get("score_total", 0)) for row in vars_)
            mat_rows.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "file_scores": file_scores,
                    "file_score_total": file_score_total,
                    "variable_score_total": variable_score,
                    "variables": vars_,
                }
            )
    candidate_files = sorted(
        file_rows,
        key=lambda row: (
            int(row["scores"]["time"]) + int(row["scores"]["movement"]) + int(row["scores"]["neural"]),
            int(row["score_total"]),
            int(row["size"]),
        ),
        reverse=True,
    )[:80]
    candidate_mats = sorted(
        mat_rows,
        key=lambda row: (
            int(row["variable_score_total"])
            + int(row["file_scores"]["time"])
            + int(row["file_scores"]["movement"])
            + int(row["file_scores"]["neural"]),
            int(row["size"]),
        ),
        reverse=True,
    )[:40]
    bridge_candidates = []
    for row in mat_rows:
        aggregate = {
            "neural": int(row["file_scores"]["neural"]),
            "time": int(row["file_scores"]["time"]),
            "movement": int(row["file_scores"]["movement"]),
            "laser": int(row["file_scores"]["laser"]),
        }
        for var in row["variables"]:
            scores = var.get("scores")
            if isinstance(scores, dict):
                for key, value in scores.items():
                    aggregate[key] += int(value)
        if aggregate["neural"] and (aggregate["time"] or aggregate["movement"]):
            bridge_candidates.append(
                {
                    "path": row["path"],
                    "size": row["size"],
                    "aggregate_scores": aggregate,
                    "top_variables": sorted(
                        [
                            var
                            for var in row["variables"]
                            if int(var.get("score_total", 0)) > 0
                        ],
                        key=lambda var: int(var.get("score_total", 0)),
                        reverse=True,
                    )[:12],
                }
            )
    bridge_candidates.sort(
        key=lambda row: (
            row["aggregate_scores"]["neural"]
            + row["aggregate_scores"]["time"]
            + row["aggregate_scores"]["movement"],
            int(row["size"]),
        ),
        reverse=True,
    )
    return {
        "root": str(root),
        "files_considered": len(file_rows),
        "mat_files": len(mat_rows),
        "candidate_files": candidate_files,
        "candidate_mats": candidate_mats,
        "bridge_candidates": bridge_candidates[:30],
    }


def write_report(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Zebrafish supplementary alignment scout",
        "",
        f"- root: `{payload['root']}`",
        f"- files considered: {payload['files_considered']}",
        f"- mat files: {payload['mat_files']}",
        "",
        "## bridge candidates",
        "",
        "| file | neural | time | movement | laser | size | top variables |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["bridge_candidates"]:  # type: ignore[index]
        scores = row["aggregate_scores"]
        variables = ", ".join(var["name"] for var in row["top_variables"][:6])
        lines.append(
            f"| {row['path']} | {scores['neural']} | {scores['time']} | "
            f"{scores['movement']} | {scores['laser']} | {row['size']} | {variables} |"
        )
    lines.extend(
        [
            "",
            "## top candidate files",
            "",
            "| file | score | size |",
            "|---|---:|---:|",
        ]
    )
    for row in payload["candidate_files"][:30]:  # type: ignore[index]
        lines.append(f"| {row['path']} | {row['score_total']} | {row['size']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    payload = scan(args.root)
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload, REPORT_MD)
    print("Zebrafish supplementary alignment scout")
    print(f"  root={args.root}")
    print(f"  files_considered={payload['files_considered']}")
    print(f"  mat_files={payload['mat_files']}")
    print(f"  bridge_candidates={len(payload['bridge_candidates'])}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()

