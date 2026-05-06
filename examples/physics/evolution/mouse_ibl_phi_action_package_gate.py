"""Package non-data brain bottlenecks into the next Phi-action gate.

This gate does not download new OpenAlyx data.  It formalizes the current
non-data bottleneck decisions from the already-run mouse reports:

* promote the action residual as a preregistered Clarus-field candidate,
* keep choice as session-adaptive residual rather than a universal field,
* replace named-axis language with train-selected residual subspaces,
* keep d=0 as a zero-residual boundary principle, not a brain state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


RESULT_JSON = Path(__file__).with_name("mouse_ibl_phi_action_package_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_phi_action_package_report.md")


@dataclass(frozen=True)
class MetricRow:
    target: str
    evidence: str
    passed: bool
    interpretation: str


@dataclass(frozen=True)
class DecisionRow:
    bottleneck: str
    decision: str
    status: str
    next_test: str


ACTION_METRICS = [
    MetricRow(
        target="first_movement_speed",
        evidence="nested action subspace: 9/12, mean dBA 0.013697, median dBA 0.011018",
        passed=True,
        interpretation="Phi_action survives train-selected residual-subspace testing",
    ),
    MetricRow(
        target="wheel_action_direction",
        evidence="nested action subspace: 8/12, mean dBA 0.020350, median dBA 0.017363",
        passed=True,
        interpretation="Phi_action survives train-selected residual-subspace testing",
    ),
    MetricRow(
        target="choice_sign",
        evidence="nested choice subspace: 5/12, mean dBA 0.006472, median dBA -0.000203",
        passed=False,
        interpretation="choice is not promoted to a universal Phi residual field",
    ),
]


CARRIER_METRICS = [
    MetricRow(
        target="speed_probe00",
        evidence="only probe00: 7/9, mean dBA 0.008768; drop probe00: 3/6",
        passed=True,
        interpretation="probe00 is a strong speed carrier, though weaker than full field",
    ),
    MetricRow(
        target="wheel_probe00",
        evidence="only probe00: 6/9, mean dBA 0.020289; full mean dBA 0.020350",
        passed=False,
        interpretation="probe00 nearly matches wheel mean but misses strict replication",
    ),
    MetricRow(
        target="wheel_top_probe00_units",
        evidence="only top 16 probe00 units: 7/9, mean dBA 0.023757",
        passed=True,
        interpretation="wheel action has a fold-local top-unit sufficient carrier",
    ),
    MetricRow(
        target="speed_top_probe00_units",
        evidence="only top 16 probe00 units: 6/9, mean dBA 0.008726",
        passed=False,
        interpretation="speed weakens without top units but is not top-unit sufficient",
    ),
]


DECISIONS = [
    DecisionRow(
        bottleneck="mouse action residual",
        decision="Promote Phi_action = train-selected subspace of epsilon_t.",
        status="selection_candidate",
        next_test="Run the same preregistered Phi_action rule on a larger registered panel.",
    ),
    DecisionRow(
        bottleneck="mouse action carrier",
        decision="Treat probe00/top-unit evidence as carrier split, not full localization.",
        status="mechanism_candidate",
        next_test="Separate speed distributed-carrier and wheel top-unit-carrier hypotheses.",
    ),
    DecisionRow(
        bottleneck="mouse choice residual",
        decision="Do not promote choice Phi as a universal subspace.",
        status="theory_refined",
        next_test="Model policy/history and session-adaptive residual as separate terms.",
    ),
    DecisionRow(
        bottleneck="stable named latent axis",
        decision="Replace named-axis claim with train-selected residual-subspace claim.",
        status="theory_refined",
        next_test="Require axis identity only after anatomical/probe-ablation replication.",
    ),
    DecisionRow(
        bottleneck="d=0 brain interpretation",
        decision="Use d=0 only as zero-residual boundary condition.",
        status="boundary_principle",
        next_test="Measure residual contraction or residual entropy reduction, not d=0 arrival.",
    ),
]


def make_report(payload: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL Phi-action package gate",
        "",
        "This packages the non-data bottlenecks after the Clarus residual audit.",
        "Zebrafish continuous movement is excluded because it is a data bridge bottleneck.",
        "",
        "## preregistered object",
        "",
        "$$",
        r"\Phi_{\rm action,t}^{(s)}",
        r"\equiv",
        r"\epsilon_{t,S_{\rm train}}^{(s)}",
        r"\subset",
        r"H_t^{(s)}-\widehat H_t^{(s)}(X_t,R_t,H_{t-\ell}),",
        "$$",
        "",
        "$$",
        r"\Delta_{\Phi}^{(s,y)}",
        r"=",
        r"\mathrm{BA}(X,R,\widehat H,\Phi_{\rm action})",
        r"-",
        r"\mathrm{BA}(X,R,\widehat H).",
        "$$",
        "",
        "Promotion rule: axes/subspaces are selected inside train folds only; held-out test trials only score the preregistered rule.",
        "",
        "## action metrics",
        "",
        "| target | evidence | passed | interpretation |",
        "|---|---|---|---|",
    ]
    for row in payload["action_metrics"]:  # type: ignore[index]
        lines.append(
            f"| `{row['target']}` | {row['evidence']} | `{row['passed']}` | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## carrier metrics",
            "",
            "| target | evidence | passed | interpretation |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["carrier_metrics"]:  # type: ignore[index]
        lines.append(
            f"| `{row['target']}` | {row['evidence']} | `{row['passed']}` | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## decisions",
            "",
            "| bottleneck | decision | status | next test |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["decisions"]:  # type: ignore[index]
        lines.append(
            f"| {row['bottleneck']} | {row['decision']} | `{row['status']}` | {row['next_test']} |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Non-data action bottleneck is advanced to a preregistered Phi-action candidate.",
            "- Choice and d=0 are advanced by narrowing the theory, not by claiming closure.",
            "- The next empirical run should be a larger-panel Phi-action replication, not another post-hoc axis search.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = {
        "gate": "mouse_ibl_phi_action_package",
        "data_bottlenecks_excluded": ["zebrafish continuous movement decoding"],
        "phi_action_promoted": True,
        "choice_phi_promoted": False,
        "d0_as_brain_state": False,
        "action_metrics": [asdict(row) for row in ACTION_METRICS],
        "carrier_metrics": [asdict(row) for row in CARRIER_METRICS],
        "decisions": [asdict(row) for row in DECISIONS],
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(payload), encoding="utf-8")
    print(f"wrote {RESULT_JSON}")
    print(f"wrote {REPORT_MD}")
    print(
        json.dumps(
            {
                "phi_action_promoted": payload["phi_action_promoted"],
                "choice_phi_promoted": payload["choice_phi_promoted"],
                "d0_as_brain_state": payload["d0_as_brain_state"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
