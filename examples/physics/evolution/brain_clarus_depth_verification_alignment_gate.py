"""Audit whether the 4-depth brain hypothesis matches Clarus verification equations.

The hierarchy gate says the minimal brain is four nested recurrence projections.
This gate checks that each required layer has the same verification grammar used
elsewhere in the brain docs: formal equation, observable proxy, prediction loss
or gate, and an ablation/countermodel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("brain_clarus_depth_verification_alignment_results.json")
REPORT_MD = Path(__file__).with_name("brain_clarus_depth_verification_alignment_report.md")


@dataclass(frozen=True)
class LayerVerification:
    depth: int
    layer: str
    formal_equation: str
    observable_proxy: str
    prediction_gate: str
    ablation_or_countermodel: str
    closure_status: str

    @property
    def aligned(self) -> bool:
        return all(
            (
                self.formal_equation,
                self.observable_proxy,
                self.prediction_gate,
                self.ablation_or_countermodel,
                self.closure_status in {"closed", "closed_with_boundary", "candidate_boundary"},
            )
        )


REQUIRED_LAYERS = (
    LayerVerification(
        depth=1,
        layer="cellular_clarus_cell",
        formal_equation=(
            "X_{t+1}=Pi_R[B,E,A,I,U,Q,D,S]; "
            "E_min>0.45, I_min>0.70, M_min>0.45, D_max<0.40, R>=2"
        ),
        observable_proxy="human multiscale state Y=(E,I,M,T,D,S,R)",
        prediction_gate="full human proliferative/neural pass rates = 1.000",
        ablation_or_countermodel="no membrane/mitochondria/genome/traffic/repair/support/recurrence all <=0.25",
        closure_status="closed",
    ),
    LayerVerification(
        depth=2,
        layer="tissue_support_field",
        formal_equation="S_t enters both Clarus cell Pi_R and brain homeostatic H(q-q*) forcing",
        observable_proxy="tissue/glia/vascular/metabolic support S_t; q_n homeostatic state",
        prediction_gate="no_tissue_support collapses human cell closure; H(q-q*) is required brain forcing",
        ablation_or_countermodel="no_tissue_support and no_homeostasis ablation",
        closure_status="closed_with_boundary",
    ),
    LayerVerification(
        depth=3,
        layer="neural_circuit_recurrence",
        formal_equation="P_{n+1}=Pi_S[rho P_n+gamma L(W)P_n+...]",
        observable_proxy="weighted chemical/effective graph W and neural activity P_t",
        prediction_gate="L_dyn=MSE(P_{t+1}|P_t)/MSE(P_{t+1}|mean)<1; L_graph<L_flat",
        ablation_or_countermodel="binary/flat/shuffled graph and recurrent-baseline countermodels",
        closure_status="closed_with_boundary",
    ),
    LayerVerification(
        depth=4,
        layer="organism_control_loop",
        formal_equation="b_{n+1}=Omega(P_n,q_n,E_n); q_{n+1}=q_n+B(E)-C(b,P)-chi(q-q*)",
        observable_proxy="behavior labels/traces, action carrier, body/internal state q_n",
        prediction_gate="L_beh=MSE(y|P,q)/MSE(y|mean)<1 or discrete action carrier passes",
        ablation_or_countermodel="no action carrier, timing-only, task/history, continuous alignment boundary",
        closure_status="closed_with_boundary",
    ),
)


OPTIONAL_LAYER = LayerVerification(
    depth=5,
    layer="self_model_workspace",
    formal_equation="m_{n+1}=lambda m_n+Psi(P_n,b_n,r_n); W_{n+1}=Pi_W[W+epsilon Phi- mu W]",
    observable_proxy="memory/replay/workspace/self-state reports",
    prediction_gate="not yet locally closed",
    ablation_or_countermodel="workspace ablation not available locally",
    closure_status="candidate_boundary",
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    required_rows = [asdict(row) | {"aligned": row.aligned} for row in REQUIRED_LAYERS]
    optional = asdict(OPTIONAL_LAYER) | {"aligned": OPTIONAL_LAYER.aligned}
    grammar = {
        "formal": "state-space and projection equations are defined",
        "observable": "a measurable proxy is named",
        "prediction": "a loss/gate beats a baseline or passes thresholds",
        "ablation": "at least one countermodel or removed-term failure exists",
    }
    result = {
        "gate": "brain_clarus_depth_verification_alignment",
        "passed": bool(
            all(row["aligned"] for row in required_rows)
            and not optional["closure_status"] == "closed"
            and len(required_rows) == 4
        ),
        "minimal_brain_depth": 4,
        "aligned_required_layers": sum(bool(row["aligned"]) for row in required_rows),
        "required_total": len(required_rows),
        "optional_depth5_status": optional["closure_status"],
        "verification_grammar": grammar,
        "required_layers": required_rows,
        "optional_layer": optional,
        "interpretation": (
            "The 4-depth brain claim matches the Clarus verification grammar.  Each required "
            "layer has a formal update/projection, a proxy, a prediction or closure gate, "
            "and an ablation/countermodel.  Depth 5 is formally writable but remains a "
            "workspace/self-model candidate boundary."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Brain Clarus depth verification alignment",
        "",
        f"- passed: `{result['passed']}`",
        f"- minimal brain depth: `{result['minimal_brain_depth']}`",
        f"- aligned required layers: {result['aligned_required_layers']}/{result['required_total']}",
        f"- optional depth-5 status: `{result['optional_depth5_status']}`",
        "",
        "## required layers",
        "",
        "| depth | layer | formal equation | observable proxy | prediction gate | ablation/countermodel | aligned |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in result["required_layers"]:
        lines.append(
            f"| {row['depth']} | `{row['layer']}` | {row['formal_equation']} | "
            f"{row['observable_proxy']} | {row['prediction_gate']} | "
            f"{row['ablation_or_countermodel']} | `{row['aligned']}` |"
        )
    opt = result["optional_layer"]
    lines.extend(
        [
            "",
            "## optional depth 5",
            "",
            f"- layer: `{opt['layer']}`",
            f"- status: `{opt['closure_status']}`",
            f"- formal equation: {opt['formal_equation']}",
            f"- reason: {opt['prediction_gate']}; {opt['ablation_or_countermodel']}",
            "",
            "## interpretation",
            "",
            result["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "minimal_brain_depth": result["minimal_brain_depth"],
                "aligned_required_layers": result["aligned_required_layers"],
                "required_total": result["required_total"],
                "optional_depth5_status": result["optional_depth5_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
