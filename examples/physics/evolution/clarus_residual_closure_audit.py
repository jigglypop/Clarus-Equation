"""Audit whether current brain bottlenecks are Clarus residual closures.

The Clarus-field reading tested here is narrow:

    a blocked term is promoted only if the remaining signal is a measurable
    residual/innovation after stable task, region, and latent-state terms, and
    adding that residual repeatedly reduces held-out prediction error.

Missing alignment data or post-hoc unstable axes remain bottlenecks rather than
Clarus-field closures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


OUT_JSON = Path(__file__).with_name("clarus_residual_closure_audit_results.json")
OUT_REPORT = Path(__file__).with_name("clarus_residual_closure_audit_report.md")


@dataclass(frozen=True)
class AuditRow:
    bottleneck: str
    current_evidence: str
    clarus_fit: str
    verdict: str
    next_gate: str


ROWS = [
    AuditRow(
        bottleneck="zebrafish continuous movement decoding",
        current_evidence=(
            "activity-only and behavior-frame gates pass, but public chunks lack an explicit "
            "e2 timestamp or e2-resampled tail/stage movement trace"
        ),
        clarus_fit=(
            "weak: a residual-reduction field cannot replace the missing behavior-time bridge"
        ),
        verdict="not_solved_data_boundary",
        next_gate="find timestamp-certified neural-to-tail/stage alignment, then test residual readout",
    ),
    AuditRow(
        bottleneck="mouse region-only decision/action closure",
        current_evidence=(
            "region-only, same-window interaction, lagged region coupling, and strict temporal GLM "
            "failed or were partial; low-rank state transition and innovation-to-behavior survived"
        ),
        clarus_fit=(
            "strong: the surviving term is explicitly an innovation residual after X, R, and H_hat"
        ),
        verdict="partly_solved_by_residual_field",
        next_gate="treat Phi_t as residual innovation epsilon_t, not as a pure region loop",
    ),
    AuditRow(
        bottleneck="mouse action readout",
        current_evidence=(
            "epsilon_t after X,R,H_hat survives for speed 9/12 and wheel 7/12; nested action "
            "subspace survives speed 9/12 and wheel 8/12"
        ),
        clarus_fit=(
            "strong: action behavior is where residual-reduction/readout has repeated support"
        ),
        verdict="yes_action_channel",
        next_gate="pre-register Phi_action subspace and test speed/wheel split on a larger panel",
    ),
    AuditRow(
        bottleneck="mouse choice residual",
        current_evidence=(
            "choice innovation reproducibility fails on 24-panel: 8/24 support, mean dBA 0.001288, "
            "top1 axis null p=0.930300"
        ),
        clarus_fit=(
            "limited: choice residual may be session-adaptive Phi, but not a stable universal subspace"
        ),
        verdict="not_yet_choice_closure",
        next_gate="model policy/history and session-adaptive residual separately before promotion",
    ),
    AuditRow(
        bottleneck="stable named latent axis",
        current_evidence=(
            "best single axes can predict targets, but axis identity is unstable; only a broader "
            "top3 concentration survives in some panels"
        ),
        clarus_fit=(
            "medium: Clarus field can be a projection rule over residual subspace, not a named axis"
        ),
        verdict="subspace_not_axis",
        next_gate="use train-selected residual subspace with anatomical/probe ablations",
    ),
    AuditRow(
        bottleneck="d=0 / zero-residual interpretation in brain data",
        current_evidence=(
            "brain gates show residual reduction, not finite-time arrival at a zero-residual state"
        ),
        clarus_fit=(
            "conceptual only: d=0 is a boundary condition/ideal mirror, not an observed brain state"
        ),
        verdict="boundary_principle_only",
        next_gate="measure monotone residual contraction or residual entropy reduction, not d=0 itself",
    ),
]


def verdict_counts(rows: list[AuditRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    return counts


def write_report(rows: list[AuditRow]) -> None:
    lines = [
        "# Clarus residual-closure audit",
        "",
        "Question: can the blocked brain/evolution gates be explained by a Clarus field,",
        "i.e. a self-referential residual direction that reduces the leftover from the 0-space boundary?",
        "",
        "## criterion",
        "",
        "Promotion rule:",
        "",
        r"$$",
        r"\Phi_t \;\widehat{=}\; \epsilon_t"
        r"=H_t-\widehat H_t(X_t,R_t,H_{t-\ell}),",
        r"\qquad",
        r"\Delta_{\Phi}=\mathrm{score}(X,R,\widehat H,\Phi)-\mathrm{score}(X,R,\widehat H)>0.",
        r"$$",
        "",
        "A bottleneck counts as Clarus-closed only when the residual is measured and improves held-out",
        "prediction under train-only selection. Missing timestamps, missing behavior bridges, or unstable",
        "post-hoc axes do not count.",
        "",
        "## verdict table",
        "",
        "| bottleneck | evidence | Clarus fit | verdict | next gate |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.bottleneck} | {row.current_evidence} | {row.clarus_fit} | "
            f"`{row.verdict}` | {row.next_gate} |"
        )
    lines.extend(
        [
            "",
            "## summary",
            "",
            "- Yes for the mouse action channel: the surviving object is already a residual innovation term.",
            "- No for zebrafish continuous decoding: the blocker is a missing timestamp/behavior bridge.",
            "- Not yet for mouse choice: residual signal is weak and not reproducible as a stable subspace.",
            "- The clean mathematical reading is not `brain reaches d=0`; it is `brain dynamics repeatedly reduces residuals relative to a zero-residual boundary condition`.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = list(ROWS)
    payload = {
        "question": "can current brain bottlenecks be closed by Clarus residual reduction?",
        "promotion_rule": "measured residual/innovation must improve held-out prediction after stable terms",
        "verdict_counts": verdict_counts(rows),
        "rows": [asdict(row) for row in rows],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(rows)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_REPORT}")
    print(json.dumps(payload["verdict_counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
