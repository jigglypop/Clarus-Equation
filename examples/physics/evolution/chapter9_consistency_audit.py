"""Audit whether the chapter-9 entry document reflects the latest closure package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
MAIN_DOC = REPO_ROOT / "docs" / "6_뇌" / "09_생명에서지능까지.md"
CURRENT_DOC = (
    REPO_ROOT
    / "docs"
    / "6_뇌"
    / "09_생명에서지능까지"
    / "06_현재판정과다음병목.md"
)
RESULT_JSON = BASE_DIR / "chapter9_consistency_audit_results.json"
REPORT_MD = BASE_DIR / "chapter9_consistency_audit_report.md"


REQUIRED_MAIN_PHRASES = (
    "life minimum triad",
    "Evolution ladder closure package",
    "external empirical gates ready: `0/5`",
    "speed probe00/block",
    "wheel probe00/top16",
    "promoted-term package",
)

STALE_MAIN_PHRASES = (
    "이제 다음 전진점은 choice를 더 캐는 것이 아니라, action 쪽에서 speed와 wheel이 왜 다른 carrier 조건을 갖는지 식으로 분리하는 일이다.",
)


def phrase_rows(text: str, required: tuple[str, ...], stale: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for phrase in required:
        rows.append({"phrase": phrase, "kind": "required", "present": phrase in text})
    for phrase in stale:
        rows.append({"phrase": phrase, "kind": "stale", "present": phrase in text})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    main = args.main_doc.read_text(encoding="utf-8")
    current = args.current_doc.read_text(encoding="utf-8")
    rows = phrase_rows(main, REQUIRED_MAIN_PHRASES, STALE_MAIN_PHRASES)
    required_ok = all(row["present"] for row in rows if row["kind"] == "required")
    stale_ok = not any(row["present"] for row in rows if row["kind"] == "stale")
    current_has_package = "Evolution ladder closure package" in current
    result = {
        "gate": "chapter9_consistency_audit",
        "passed": bool(required_ok and stale_ok and current_has_package),
        "main_doc": str(args.main_doc),
        "current_doc": str(args.current_doc),
        "required_ok": required_ok,
        "stale_ok": stale_ok,
        "current_has_package": current_has_package,
        "phrase_rows": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Chapter 9 consistency audit",
        "",
        f"- passed: `{result['passed']}`",
        f"- required ok: `{result['required_ok']}`",
        f"- stale ok: `{result['stale_ok']}`",
        f"- current doc has package: `{result['current_has_package']}`",
        "",
        "| kind | phrase | present |",
        "|---|---|---|",
    ]
    for row in result["phrase_rows"]:
        lines.append(f"| `{row['kind']}` | `{row['phrase']}` | `{row['present']}` |")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-doc", type=Path, default=MAIN_DOC)
    parser.add_argument("--current-doc", type=Path, default=CURRENT_DOC)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(json.dumps({"passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
