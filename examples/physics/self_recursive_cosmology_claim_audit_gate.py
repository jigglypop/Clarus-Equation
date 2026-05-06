"""Audit self-recursive cosmology claims against scoped evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DOC = REPO_ROOT / "docs" / "3_상수" / "9_우주론_수식_의미와_후보.md"
PACKAGE_REPORT = BASE_DIR / "self_recursive_cosmology_package_report.md"
READINESS_REPORT = BASE_DIR / "h0_readout" / "h0_fisher_real_readiness_report.md"
LEDGER_REPORT = BASE_DIR / "self_recursive_cosmology_prediction_ledger_report.md"
RESULT_JSON = BASE_DIR / "self_recursive_cosmology_claim_audit_results.json"
REPORT_MD = BASE_DIR / "self_recursive_cosmology_claim_audit_report.md"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim: str
    allowed_status: str
    required_markers: tuple[str, ...]
    forbidden_markers: tuple[str, ...]


CLAIMS = (
    Claim(
        "C1",
        "The minimal fixed-point kernel is the current closed core recursion.",
        "claimable",
        ("Closed/minimal", "fixed-point residual", "Kernel deformation is blocked"),
        ("kernel deformation is Exact",),
    ),
    Claim(
        "C2",
        "d=0 is a zero-measure boundary principle, not a physical place.",
        "claimable-with-scope",
        ("Boundary principle", "zero-measure boundary", "physical place가 아니라"),
        ("d=0 physical place", "d=0으로 간다"),
    ),
    Claim(
        "C3",
        "The residual cascade is a Selection candidate, not an Exact theorem.",
        "claimable-with-scope",
        ("Selection candidate", "raw A_s", "GER A_s"),
        ("residual cascade is Exact",),
    ),
    Claim(
        "C4",
        "The H0 q-selector is a channel-corrected Bridge.",
        "claimable-with-scope",
        ("Channel-corrected Bridge", "q-space chi2/dof=0.379/8", "before H0 comparison"),
        ("H0 q-selector is Exact",),
    ),
    Claim(
        "C5",
        "Fisher/covariance IO is ready, but real covariance closure is still a data boundary.",
        "limitation",
        ("Fisher/covariance IO is ready", "real covariance closure is still a data boundary", "real-ready channels | 0"),
        ("real covariance closure is closed",),
    ),
    Claim(
        "C6",
        "The package has a falsifiable future prediction ledger.",
        "claimable-with-future-tests",
        ("real BAO+SN covariance", "primitive spectrum joint likelihood", "falsifier"),
        (),
    ),
    Claim(
        "C7",
        "The whole package is Selection/Bridge, not Exact.",
        "limitation",
        ("Selection/Bridge package, not Exact", "not an Exact theorem"),
        ("self-recursive cosmology is Exact",),
    ),
)


def corpus() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DOC, PACKAGE_REPORT, READINESS_REPORT, LEDGER_REPORT)
        if path.exists()
    )


def audit_claim(claim: Claim, text: str) -> dict[str, object]:
    missing = [marker for marker in claim.required_markers if marker not in text]
    forbidden = [marker for marker in claim.forbidden_markers if marker in text]
    return {
        **asdict(claim),
        "missing_required": missing,
        "forbidden_present": forbidden,
        "passed": not missing and not forbidden,
    }


def main() -> int:
    text = corpus()
    rows = [audit_claim(claim, text) for claim in CLAIMS]
    passed = all(bool(row["passed"]) for row in rows)
    payload = {
        "gate": "self_recursive_cosmology_claim_audit",
        "passed": passed,
        "claim_count": len(rows),
        "rows": rows,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Self-recursive cosmology claim audit",
        "",
        f"- passed: `{passed}`",
        f"- claims: {len(rows)}",
        "",
        "| id | status | claim | passed | missing required | forbidden present |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['claim_id']} | {row['allowed_status']} | {row['claim']} | "
            f"`{row['passed']}` | {', '.join(row['missing_required']) or '-'} | "
            f"{', '.join(row['forbidden_present']) or '-'} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"passed": passed, "claim_count": len(rows)}, indent=2))
    if not passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
