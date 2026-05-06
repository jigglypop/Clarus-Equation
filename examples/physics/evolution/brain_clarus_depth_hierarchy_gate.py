"""Compare 3/4/5-depth Clarus aggregation hypotheses for brain closure.

Depth is counted by nested recurrence projections, not by anatomical size.
The question is whether a brain is merely many Clarus cells, or whether
cell, tissue, circuit, organ-control, and self-model levels must be separated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("brain_clarus_depth_hierarchy_results.json")
REPORT_MD = Path(__file__).with_name("brain_clarus_depth_hierarchy_report.md")


@dataclass(frozen=True)
class DepthLayer:
    index: int
    name: str
    recurrence_projection: str
    carrier: str
    required_for_brain: bool
    local_evidence: str


LAYERS = (
    DepthLayer(
        1,
        "cellular_clarus_cell",
        "X_cell(t) -> same cell-type basin",
        "neuron, glia, vascular, immune, epithelial/stem variants",
        True,
        "human Clarus-cell gates: boundary/metabolism/identity/repair/support/recurrence",
    ),
    DepthLayer(
        2,
        "tissue_support_field",
        "many cells -> stable metabolic/glial/vascular/tissue context",
        "glia, vasculature, ECM, endocrine/immune context",
        True,
        "human postmitotic neural Clarus cell needs tissue/glial support",
    ),
    DepthLayer(
        3,
        "neural_circuit_recurrence",
        "coupled excitable cells -> recurrent activity state",
        "weighted synaptic/electrical/chemical graph",
        True,
        "C. elegans weighted routing and zebrafish recurrent activity closure",
    ),
    DepthLayer(
        4,
        "organism_control_loop",
        "activity state -> behavior/body state -> new sensory and internal input",
        "sensorimotor, autonomic, endocrine, action carrier loops",
        True,
        "cross-species action carrier and mouse speed/wheel action split",
    ),
    DepthLayer(
        5,
        "self_model_workspace",
        "organism-control state -> memory/planning/self-model -> future control policy",
        "human-like workspace, reportability, autobiographical/self-state model",
        False,
        "not closed by current local data; remains higher-cognition candidate",
    ),
)


@dataclass(frozen=True)
class Hypothesis:
    name: str
    included_layers: tuple[int, ...]
    claim: str

    @property
    def depth(self) -> int:
        return len(self.included_layers)


HYPOTHESES = (
    Hypothesis(
        "three_depth_brain",
        (1, 2, 3),
        "brain is cell+tissue+circuit recurrence",
    ),
    Hypothesis(
        "four_depth_brain",
        (1, 2, 3, 4),
        "brain is cell+tissue+circuit plus organism-control recurrence",
    ),
    Hypothesis(
        "five_depth_mind_brain",
        (1, 2, 3, 4, 5),
        "human-like mind/brain adds self-model or workspace recurrence",
    ),
)


REQUIRED_EVIDENCE = {
    "cellular_clarus_cell": True,
    "tissue_support_field": True,
    "neural_circuit_recurrence": True,
    "organism_control_loop": True,
    "self_model_workspace": False,
}


def evaluate_hypothesis(hypothesis: Hypothesis) -> dict[str, Any]:
    layer_by_index = {layer.index: layer for layer in LAYERS}
    included = [layer_by_index[index] for index in hypothesis.included_layers]
    missing_required = [
        layer.name
        for layer in LAYERS
        if layer.required_for_brain and layer.index not in hypothesis.included_layers
    ]
    unsupported_included = [
        layer.name for layer in included if not REQUIRED_EVIDENCE[layer.name]
    ]
    if missing_required:
        verdict = "underfit_brain"
    elif unsupported_included:
        verdict = "overextended_mind_claim"
    else:
        verdict = "minimal_brain_closure"
    return {
        "name": hypothesis.name,
        "depth": hypothesis.depth,
        "claim": hypothesis.claim,
        "layers": [layer.name for layer in included],
        "missing_required": missing_required,
        "unsupported_included": unsupported_included,
        "verdict": verdict,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [evaluate_hypothesis(hypothesis) for hypothesis in HYPOTHESES]
    minimal = next(row for row in rows if row["verdict"] == "minimal_brain_closure")
    result = {
        "gate": "brain_clarus_depth_hierarchy",
        "passed": bool(
            minimal["name"] == "four_depth_brain"
            and any(row["verdict"] == "underfit_brain" for row in rows)
            and any(row["verdict"] == "overextended_mind_claim" for row in rows)
        ),
        "minimal_brain_depth": minimal["depth"],
        "minimal_brain_hypothesis": minimal["name"],
        "mind_candidate_depth": 5,
        "interpretation": (
            "A brain is not just a pile of Clarus cells.  Current closure needs four nested "
            "recurrence projections: cellular self-maintenance, tissue support, neural circuit "
            "activity recurrence, and organism-control recurrence.  A fifth self-model/workspace "
            "layer is a mind/human-cognition candidate, not yet a closed brain requirement."
        ),
        "layers": [asdict(layer) for layer in LAYERS],
        "hypotheses": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Brain Clarus depth hierarchy gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- minimal brain depth: `{result['minimal_brain_depth']}`",
        f"- minimal brain hypothesis: `{result['minimal_brain_hypothesis']}`",
        f"- mind candidate depth: `{result['mind_candidate_depth']}`",
        "",
        "## layers",
        "",
        "| depth | layer | recurrence projection | carrier | required | evidence |",
        "|---:|---|---|---|---|---|",
    ]
    for row in result["layers"]:
        lines.append(
            f"| {row['index']} | `{row['name']}` | {row['recurrence_projection']} | "
            f"{row['carrier']} | `{row['required_for_brain']}` | {row['local_evidence']} |"
        )
    lines.extend(
        [
            "",
            "## hypotheses",
            "",
            "| hypothesis | depth | verdict | missing required | unsupported included |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in result["hypotheses"]:
        lines.append(
            f"| `{row['name']}` | {row['depth']} | `{row['verdict']}` | "
            f"{', '.join(row['missing_required']) or 'none'} | "
            f"{', '.join(row['unsupported_included']) or 'none'} |"
        )
    lines.extend(["", "## interpretation", "", result["interpretation"]])
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
                "minimal_brain_hypothesis": result["minimal_brain_hypothesis"],
                "mind_candidate_depth": result["mind_candidate_depth"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
