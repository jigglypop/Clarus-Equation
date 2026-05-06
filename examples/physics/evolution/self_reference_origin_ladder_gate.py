"""Audit where self-reference recursion first appears on the life-to-brain ladder.

The gate is deliberately conservative.  It does not name a historical first
species.  Instead it asks which primitive stage first satisfies the structural
conditions for self-reference: a bounded system whose own state helps recreate
and select its next own state with heritable distinction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("self_reference_origin_ladder_results.json")
REPORT_MD = Path(__file__).with_name("self_reference_origin_ladder_report.md")


@dataclass(frozen=True)
class Stage:
    name: str
    order: int
    status: str
    self_production: bool
    boundary_retention: bool
    copying_heredity: bool
    own_state_feedback: bool
    sensorimotor_action: bool
    neural_routing: bool
    local_evidence: str
    empirical_boundary: str

    @property
    def minimum_self_reference(self) -> bool:
        return all(
            (
                self.self_production,
                self.boundary_retention,
                self.copying_heredity,
                self.own_state_feedback,
            )
        )

    @property
    def behavioral_self_reference(self) -> bool:
        return self.minimum_self_reference and self.sensorimotor_action

    @property
    def neural_self_reference(self) -> bool:
        return self.behavioral_self_reference and self.neural_routing

    @property
    def closure_score(self) -> int:
        return sum(
            (
                self.self_production,
                self.boundary_retention,
                self.copying_heredity,
                self.own_state_feedback,
                self.sensorimotor_action,
                self.neural_routing,
            )
        )


STAGES = (
    Stage(
        name="prebiotic_open_chemistry",
        order=0,
        status="pre_recursive",
        self_production=False,
        boundary_retention=False,
        copying_heredity=False,
        own_state_feedback=False,
        sensorimotor_action=False,
        neural_routing=False,
        local_evidence="Open reaction chemistry has no local identity term in the current ladder.",
        empirical_boundary="reaction networks can be modeled, but this stage is not an organism",
    ),
    Stage(
        name="autocatalytic_set_without_compartment",
        order=1,
        status="proto_recursive_incomplete",
        self_production=True,
        boundary_retention=False,
        copying_heredity=False,
        own_state_feedback=True,
        sensorimotor_action=False,
        neural_routing=False,
        local_evidence="Autocatalysis gives self-amplification but not protected identity or heredity.",
        empirical_boundary="needs reaction-network growth plus lineage-level retention/copying evidence",
    ),
    Stage(
        name="template_replicator_without_boundary",
        order=2,
        status="proto_recursive_incomplete",
        self_production=True,
        boundary_retention=False,
        copying_heredity=True,
        own_state_feedback=True,
        sensorimotor_action=False,
        neural_routing=False,
        local_evidence="Copying can carry sequence distinction, but the open system leaks identity.",
        empirical_boundary="needs compartment or equivalent retention to become an organism-like unit",
    ),
    Stage(
        name="compartment_without_template_copying",
        order=3,
        status="proto_recursive_incomplete",
        self_production=True,
        boundary_retention=True,
        copying_heredity=False,
        own_state_feedback=True,
        sensorimotor_action=False,
        neural_routing=False,
        local_evidence="Boundary plus metabolism can persist, but no heritable template distinction closes.",
        empirical_boundary="needs copying/template inheritance or equivalent heritable state",
    ),
    Stage(
        name="template_bearing_protocell",
        order=4,
        status="first_minimum_self_reference",
        self_production=True,
        boundary_retention=True,
        copying_heredity=True,
        own_state_feedback=True,
        sensorimotor_action=False,
        neural_routing=False,
        local_evidence=(
            "This is the first stage matching the local life triad: "
            "autocatalysis + boundary + copying."
        ),
        empirical_boundary=(
            "not a named species; empirical promotion needs protocell/ribozyme/LUCA-style data"
        ),
    ),
    Stage(
        name="luca_like_prokaryotic_cell",
        order=5,
        status="first_named_biological_candidate",
        self_production=True,
        boundary_retention=True,
        copying_heredity=True,
        own_state_feedback=True,
        sensorimotor_action=False,
        neural_routing=False,
        local_evidence="A cell-level genotype-metabolism-boundary loop is organismal self-reference.",
        empirical_boundary="historical placement is external to the local toy gate",
    ),
    Stage(
        name="chemotactic_bacterium_or_archaeon",
        order=6,
        status="first_behavioral_self_reference_candidate",
        self_production=True,
        boundary_retention=True,
        copying_heredity=True,
        own_state_feedback=True,
        sensorimotor_action=True,
        neural_routing=False,
        local_evidence=(
            "Action changes the next sensory/input distribution, so the organism loops through environment."
        ),
        empirical_boundary="needs species-specific sensor/action/time-series data for promotion",
    ),
    Stage(
        name="unicellular_eukaryote_or_ciliate_like_cell",
        order=7,
        status="rich_single_cell_recursion",
        self_production=True,
        boundary_retention=True,
        copying_heredity=True,
        own_state_feedback=True,
        sensorimotor_action=True,
        neural_routing=False,
        local_evidence="Internal state and action are richer, but this is not the first recursion threshold.",
        empirical_boundary="not needed to set the earliest boundary",
    ),
    Stage(
        name="c_elegans_primitive_neural_proxy",
        order=8,
        status="first_local_neural_routing_proxy",
        self_production=True,
        boundary_retention=True,
        copying_heredity=True,
        own_state_feedback=True,
        sensorimotor_action=True,
        neural_routing=True,
        local_evidence="Local connectome gates support weighted chemical routing as primitive neural control.",
        empirical_boundary="actual trial behavior remains data-boundary in the local audit",
    ),
)


def first_stage(stages: tuple[Stage, ...], attr: str) -> Stage:
    for stage in stages:
        if bool(getattr(stage, attr)):
            return stage
    raise ValueError(f"no stage satisfies {attr}")


def stage_row(stage: Stage) -> dict[str, Any]:
    row = asdict(stage)
    row["minimum_self_reference"] = stage.minimum_self_reference
    row["behavioral_self_reference"] = stage.behavioral_self_reference
    row["neural_self_reference"] = stage.neural_self_reference
    row["closure_score"] = stage.closure_score
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    stages = tuple(sorted(STAGES, key=lambda stage: stage.order))
    first_minimum = first_stage(stages, "minimum_self_reference")
    first_behavioral = first_stage(stages, "behavioral_self_reference")
    first_neural = first_stage(stages, "neural_self_reference")

    monotonic_scores = all(
        left.closure_score <= right.closure_score for left, right in zip(stages, stages[1:])
    )
    incomplete_proto_stages = [
        stage.name
        for stage in stages
        if stage.status == "proto_recursive_incomplete" and not stage.minimum_self_reference
    ]
    result = {
        "gate": "self_reference_origin_ladder",
        "passed": bool(
            first_minimum.name == "template_bearing_protocell"
            and first_behavioral.name == "chemotactic_bacterium_or_archaeon"
            and first_neural.name == "c_elegans_primitive_neural_proxy"
            and monotonic_scores
            and len(incomplete_proto_stages) == 3
        ),
        "first_minimum_self_reference": first_minimum.name,
        "first_named_biological_candidate": "luca_like_prokaryotic_cell",
        "first_behavioral_self_reference": first_behavioral.name,
        "first_local_neural_self_reference_proxy": first_neural.name,
        "interpretation": (
            "Self-reference recursion first closes structurally at a template-bearing protocell, "
            "not at a named animal.  If a named organismal candidate is required, the conservative "
            "label is LUCA-like prokaryotic cell.  Sensorimotor recursion begins at chemotactic "
            "bacteria/archaea-like cells, and neural recursion is only proxied locally at C. elegans."
        ),
        "monotonic_scores": monotonic_scores,
        "incomplete_proto_stages": incomplete_proto_stages,
        "rows": [stage_row(stage) for stage in stages],
        "empirical_requirements": [
            "reaction network or sequence table",
            "autocatalysis or growth measurement",
            "boundary or compartment retention measurement",
            "copying/template/heritable-state measurement",
            "ablation/control showing the triad is jointly necessary",
        ],
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Self-reference origin ladder gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- first minimum self-reference: `{result['first_minimum_self_reference']}`",
        f"- first named biological candidate: `{result['first_named_biological_candidate']}`",
        f"- first behavioral self-reference: `{result['first_behavioral_self_reference']}`",
        f"- first local neural self-reference proxy: `{result['first_local_neural_self_reference_proxy']}`",
        "",
        "## interpretation",
        "",
        result["interpretation"],
        "",
        "## stage table",
        "",
        "| order | stage | status | score | minimum | behavioral | neural | local evidence |",
        "|---:|---|---|---:|---|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['order']} | `{row['name']}` | `{row['status']}` | "
            f"{row['closure_score']} | `{row['minimum_self_reference']}` | "
            f"`{row['behavioral_self_reference']}` | `{row['neural_self_reference']}` | "
            f"{row['local_evidence']} |"
        )
    lines.extend(
        [
            "",
            "## empirical boundary",
            "",
            "This gate is a structural Clarus-ladder audit, not a historical origin-of-life proof.",
            "To promote the first boundary empirically, the next dataset must contain:",
            "",
        ]
    )
    for requirement in result["empirical_requirements"]:
        lines.append(f"- {requirement}")
    lines.extend(
        [
            "",
            "## falsifiers",
            "",
            "- A heritable, selectable, self-maintaining unit without boundary retention would lower the boundary.",
            "- A bounded autocatalytic unit with no copying or heritable state would weaken the copying criterion.",
            "- A verified sensorimotor loop before organismal cell closure would split behavioral recursion from life recursion.",
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
                "first_minimum_self_reference": result["first_minimum_self_reference"],
                "first_behavioral_self_reference": result["first_behavioral_self_reference"],
                "first_local_neural_self_reference_proxy": result[
                    "first_local_neural_self_reference_proxy"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
