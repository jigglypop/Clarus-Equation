#!/usr/bin/env python3
"""Run the preregistered scalar-sector CE model-selection audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "reality_stone" / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from reality_stone.clarus.core_model_selection import (  # noqa: E402
    candidate_by_id,
    default_manifest_path,
    evaluate_manifest,
    load_manifest,
)


CE_CANDIDATE_ID = "exponential__linear__ce_delta"


def _print_summary(report) -> None:
    print("# CE Core Model-Selection Gate")
    print()
    print(f"schema_version {report.schema_version}")
    print(f"recursion_scope {report.recursion_scope}")
    print("scope_note scalar equal-row-sum invariant sector only; not the full vector A recursion")
    print(f"manifest_hash {report.manifest_hash}")
    print(f"candidate_count {report.candidate_count}")
    print(f"algebraic_status {report.algebraic_status}")
    print(f"selection_status {report.selection_status}")
    print(f"selection_observations {report.n_selection_observations}")
    print(
        "independent_selection_observations "
        f"{report.n_independent_selection_observations}"
    )
    print()

    algebraic_counts = Counter(item.algebraic_status for item in report.candidates)
    selection_counts = Counter(item.selection_status for item in report.candidates)
    factorization_count = sum(
        item.algebraic.factorization_compatible for item in report.candidates
    )
    print("candidate_algebraic_counts", dict(sorted(algebraic_counts.items())))
    print("candidate_selection_counts", dict(sorted(selection_counts.items())))
    print(f"factorization_compatible_candidates {factorization_count}")
    print()

    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    print(f"ce_candidate {CE_CANDIDATE_ID}")
    print(f"ce_d_eff {ce.algebraic.d_eff:.12f}")
    print(f"ce_algebraic_status {ce.algebraic_status}")
    print(f"ce_selection_status {ce.selection_status}")
    for branch in ce.branches:
        root = branch.root
        print(
            "ce_root "
            f"label={root.branch_label} "
            f"x={root.value:.12f} "
            f"residual={root.residual:.3e} "
            f"rho={root.stability_radius:.9f} "
            f"stable={root.stable} "
            f"eligible={root.eligible_for_selection} "
            f"selection={branch.selection.status}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the preregistered model manifest.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON report instead of the concise summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the complete JSON report.",
    )
    args = parser.parse_args(argv)

    report = evaluate_manifest(load_manifest(args.manifest))
    payload = report.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_summary(report)
    return 0 if report.algebraic_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
