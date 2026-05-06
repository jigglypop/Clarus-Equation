"""Audit split chapter-9 files for latest package/boundary terminology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DOC_DIR = REPO_ROOT / "docs" / "6_뇌" / "09_생명에서지능까지"
RESULT_JSON = BASE_DIR / "chapter9_split_docs_consistency_audit_results.json"
REPORT_MD = BASE_DIR / "chapter9_split_docs_consistency_audit_report.md"


CHECKS = {
    "01_개요와공통식.md": {
        "required": (
            "toy positive",
            "empirical gate pending",
            "Evolution ladder closure package",
            "external requirements readiness",
        ),
        "stale": ("아직 gate 없음", "not tested"),
    },
    "02_c_elegans.md": {
        "required": ("trial-behavior boundary audit", "data-boundary"),
        "stale": (),
    },
    "03_drosophila.md": {
        "required": ("trial-dynamics boundary audit", "data-boundary"),
        "stale": (),
    },
    "04_zebrafish.md": {
        "required": ("continuous boundary final audit", "timestamp-certified alignment"),
        "stale": (),
    },
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for filename, checks in CHECKS.items():
        path = args.doc_dir / filename
        text = path.read_text(encoding="utf-8")
        for phrase in checks["required"]:
            rows.append(
                {
                    "file": filename,
                    "kind": "required",
                    "phrase": phrase,
                    "present": phrase in text,
                }
            )
        for phrase in checks["stale"]:
            rows.append(
                {
                    "file": filename,
                    "kind": "stale",
                    "phrase": phrase,
                    "present": phrase in text,
                }
            )
    required_ok = all(row["present"] for row in rows if row["kind"] == "required")
    stale_ok = not any(row["present"] for row in rows if row["kind"] == "stale")
    result = {
        "gate": "chapter9_split_docs_consistency_audit",
        "passed": bool(required_ok and stale_ok),
        "required_ok": required_ok,
        "stale_ok": stale_ok,
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Chapter 9 split-docs consistency audit",
        "",
        f"- passed: `{result['passed']}`",
        f"- required ok: `{result['required_ok']}`",
        f"- stale ok: `{result['stale_ok']}`",
        "",
        "| file | kind | phrase | present |",
        "|---|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['file']}` | `{row['kind']}` | `{row['phrase']}` | `{row['present']}` |"
        )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, default=DOC_DIR)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(json.dumps({"passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
