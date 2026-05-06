"""Promotion requirements for real H0 covariance/Fisher ingestion.

The current Fisher/covariance IO bundle is synthetic.  This gate defines the
minimum requirements for promoting a future channel to real covariance status.
It is intentionally strict: a final H0 scalar is not enough; the selector needs
source-labelled covariance nodes before H0 comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULT_JSON = BASE_DIR / "h0_real_covariance_requirements_results.json"
REPORT_MD = BASE_DIR / "h0_real_covariance_requirements_report.md"


@dataclass(frozen=True)
class Requirement:
    requirement: str
    reason: str
    required_for_promotion: bool


@dataclass(frozen=True)
class ChannelPlan:
    channel_class: str
    expected_branch: str
    minimum_artifact: str
    required_labels: str
    failure_mode: str
    priority: int


REQUIREMENTS = [
    Requirement(
        "public source URL or DOI",
        "The covariance/provenance must be independently recoverable.",
        True,
    ),
    Requirement(
        "pinned version, commit, release, or dataset date",
        "Source drift must be detectable before selector comparison.",
        True,
    ),
    Requirement(
        "machine-readable Fisher/covariance or posterior-derived covariance",
        "The q selector needs edges, not a final scalar H0.",
        True,
    ),
    Requirement(
        "node labels including observable, local anchors, and global priors",
        "q_F is defined from source roles; unlabeled matrices are insufficient.",
        True,
    ),
    Requirement(
        "documented role map made before H0 comparison",
        "Prevents fitting q after seeing the H0 branch.",
        True,
    ),
    Requirement(
        "positive definite or validated invertible covariance",
        "The normalized Fisher-edge rule requires stable diagonals and edges.",
        True,
    ),
    Requirement(
        "negative/ablation case",
        "Static all-local/all-global/flipped maps should not explain the channel equally well.",
        False,
    ),
]

CHANNEL_PLANS = [
    ChannelPlan(
        "BAO+SN inverse distance ladder",
        "global/low or low-side bridge",
        "labelled compressed covariance JSON",
        "BAO observable, sound horizon/ruler priors, SN nuisance/population nodes",
        "matrix selects local/high before H0 refit",
        1,
    ),
    ChannelPlan(
        "SH0ES/CCHP local ladder",
        "local/high or semi-local high",
        "calibration covariance graph JSON",
        "Cepheid/TRGB/JAGB calibrators, anchors, SN/Hubble-flow nodes",
        "endpoint-dominated graph selects global/low",
        2,
    ),
    ChannelPlan(
        "GW standard sirens",
        "bridge/intermediate",
        "event/posterior covariance JSON",
        "GW distance node, host/redshift/environment anchor nodes",
        "event covariance collapses to either endpoint without bridge behavior",
        3,
    ),
    ChannelPlan(
        "CMB acoustic-scale covariance",
        "global/low",
        "parameter covariance adapter JSON",
        "acoustic-scale observable, horizon/ruler priors, nuisance nodes",
        "same role map yields bridge/local q_F",
        4,
    ),
]


def main() -> int:
    blocking = [row for row in REQUIREMENTS if row.required_for_promotion]
    payload = {
        "gate": "h0_real_covariance_requirements",
        "passed": True,
        "blocking_requirement_count": len(blocking),
        "requirement_count": len(REQUIREMENTS),
        "channel_plan_count": len(CHANNEL_PLANS),
        "requirements": [asdict(row) for row in REQUIREMENTS],
        "channel_plans": [asdict(row) for row in CHANNEL_PLANS],
        "verdict": (
            "Real covariance promotion requires labelled, version-pinned matrix or posterior covariance "
            "before H0 comparison; scalar H0 rows are not enough."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# H0 real covariance requirements gate",
        "",
        f"- blocking requirements: {len(blocking)}",
        f"- channel plans: {len(CHANNEL_PLANS)}",
        "",
        "## Promotion requirements",
        "",
        "| requirement | reason | required for promotion |",
        "|---|---|---|",
    ]
    for row in REQUIREMENTS:
        lines.append(f"| {row.requirement} | {row.reason} | `{row.required_for_promotion}` |")
    lines.extend(
        [
            "",
            "## Channel plans",
            "",
            "| priority | channel class | expected branch | minimum artifact | required labels | failure mode |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for row in sorted(CHANNEL_PLANS, key=lambda item: item.priority):
        lines.append(
            f"| {row.priority} | {row.channel_class} | {row.expected_branch} | "
            f"{row.minimum_artifact} | {row.required_labels} | {row.failure_mode} |"
        )
    lines.extend(["", "## Verdict", "", payload["verdict"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"passed": True, "blocking_requirement_count": len(blocking)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
