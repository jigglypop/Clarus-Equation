"""Place the Clarus-cell morphology on a cautious origin timeline.

The Clarus cell is not asserted as a historical species.  It is the minimum
cell-like unit required by the local Clarus life gate: a bounded compartment
that couples autocatalytic maintenance, heritable copying, influx/efflux, and
division-level selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("clarus_cell_origin_timeline_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_origin_timeline_report.md")


@dataclass(frozen=True)
class TimeWindow:
    name: str
    earliest_ga: float
    latest_ga: float
    confidence: str
    role: str
    note: str

    @property
    def width_myr(self) -> float:
        return abs(self.earliest_ga - self.latest_ga) * 1000.0


@dataclass(frozen=True)
class MorphologyTerm:
    name: str
    required: bool
    form: str
    equation_role: str
    failure_if_absent: str


WINDOWS = (
    TimeWindow(
        name="habitable_earth_window",
        earliest_ga=4.5,
        latest_ga=3.9,
        confidence="broad_external_review_window",
        role="outer_possible_window",
        note="Earth habitability is the outer bound, not evidence of a Clarus cell.",
    ),
    TimeWindow(
        name="structural_clarus_cell_possible",
        earliest_ga=4.5,
        latest_ga=3.7,
        confidence="theory_window_inside_habitability_to_biosignature",
        role="possible_first_closure",
        note="A template-bearing protocell can only be placed as a structural possibility in this interval.",
    ),
    TimeWindow(
        name="biosignature_boundary",
        earliest_ga=3.8,
        latest_ga=3.4,
        confidence="geological_evidence_window",
        role="life_present_by_then",
        note="Microbial biosphere or biosignature evidence constrains life as present, not the exact first cell.",
    ),
    TimeWindow(
        name="luca_like_cell_candidate",
        earliest_ga=4.3,
        latest_ga=3.7,
        confidence="model_dependent_phylogenetic_window",
        role="first_named_biological_candidate",
        note="LUCA-like placement is model dependent and is not identical to origin of life.",
    ),
)


MORPHOLOGY = (
    MorphologyTerm(
        name="boundary_membrane",
        required=True,
        form="semi-permeable vesicle or compartment wall",
        equation_role="B_boundary - L_leak",
        failure_if_absent="identity diffuses into open chemistry",
    ),
    MorphologyTerm(
        name="autocatalytic_core",
        required=True,
        form="reaction set that increases its own enabling components",
        equation_role="A_auto(X,E)",
        failure_if_absent="growth under dilution fails",
    ),
    MorphologyTerm(
        name="copying_template",
        required=True,
        form="sequence, polymer, or heritable state copied with bias",
        equation_role="C_copy(X)",
        failure_if_absent="mass may persist but lineage distinction collapses",
    ),
    MorphologyTerm(
        name="gradient_ports",
        required=True,
        form="selective influx/efflux through boundary or surface chemistry",
        equation_role="E_in - L_leak",
        failure_if_absent="the unit cannot remain open while preserving itself",
    ),
    MorphologyTerm(
        name="division_threshold",
        required=True,
        form="growth, instability, budding, or fission threshold",
        equation_role="Pi_C lineage projection",
        failure_if_absent="no selectable recurrence across generations",
    ),
    MorphologyTerm(
        name="internal_state_memory",
        required=False,
        form="chemical concentration or template composition that biases the next cycle",
        equation_role="m_n before neural memory",
        failure_if_absent="minimum life still possible, but adaptive recursion is weak",
    ),
    MorphologyTerm(
        name="sensorimotor_surface",
        required=False,
        form="chemotactic or taxis-like input-output coupling",
        equation_role="U_d -> b_d",
        failure_if_absent="minimum self-reference remains, behavioral recursion absent",
    ),
)


def morphology_score(terms: tuple[MorphologyTerm, ...]) -> dict[str, Any]:
    required = [term for term in terms if term.required]
    optional = [term for term in terms if not term.required]
    required_count = len(required)
    optional_count = len(optional)
    return {
        "required_count": required_count,
        "optional_count": optional_count,
        "minimum_clarus_cell_closed": required_count == 5,
        "adaptive_surface_terms": optional_count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    score = morphology_score(MORPHOLOGY)
    possible = next(window for window in WINDOWS if window.name == "structural_clarus_cell_possible")
    evidence = next(window for window in WINDOWS if window.name == "biosignature_boundary")
    luca = next(window for window in WINDOWS if window.name == "luca_like_cell_candidate")
    result = {
        "gate": "clarus_cell_origin_timeline",
        "passed": bool(
            score["minimum_clarus_cell_closed"]
            and possible.earliest_ga >= possible.latest_ga
            and evidence.earliest_ga >= evidence.latest_ga
            and luca.earliest_ga >= luca.latest_ga
        ),
        "answer": {
            "structural_when": "between_habitability_and_biosignature_boundaries",
            "structural_window_ga": [possible.earliest_ga, possible.latest_ga],
            "evidence_by_window_ga": [evidence.earliest_ga, evidence.latest_ga],
            "named_candidate_window_ga": [luca.earliest_ga, luca.latest_ga],
            "do_not_overclaim": (
                "The Clarus cell begins as a template-bearing protocell form, not as a named species."
            ),
        },
        "windows": [asdict(window) | {"width_myr": window.width_myr} for window in WINDOWS],
        "morphology": [asdict(term) for term in MORPHOLOGY],
        "morphology_score": score,
        "minimal_form": (
            "semi-permeable boundary + autocatalytic core + heritable copying template + "
            "gradient ports + division threshold"
        ),
        "next_empirical_gate": [
            "map reaction network to autocatalytic core",
            "measure boundary retention/leakage",
            "measure template or heritable state copying",
            "show division or lineage recurrence",
            "ablate each required term",
        ],
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Clarus cell origin timeline gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- structural window: `{result['answer']['structural_window_ga'][0]}`-`{result['answer']['structural_window_ga'][1]}` Ga",
        f"- evidence-by window: `{result['answer']['evidence_by_window_ga'][0]}`-`{result['answer']['evidence_by_window_ga'][1]}` Ga",
        f"- named candidate window: `{result['answer']['named_candidate_window_ga'][0]}`-`{result['answer']['named_candidate_window_ga'][1]}` Ga",
        f"- minimal form: {result['minimal_form']}",
        "",
        "## timeline",
        "",
        "| window | Ga | confidence | role | note |",
        "|---|---:|---|---|---|",
    ]
    for row in result["windows"]:
        lines.append(
            f"| `{row['name']}` | {row['earliest_ga']:.2f}-{row['latest_ga']:.2f} | "
            f"`{row['confidence']}` | `{row['role']}` | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## morphology",
            "",
            "| term | required | form | equation role | failure if absent |",
            "|---|---|---|---|---|",
        ]
    )
    for row in result["morphology"]:
        lines.append(
            f"| `{row['name']}` | `{row['required']}` | {row['form']} | "
            f"`{row['equation_role']}` | {row['failure_if_absent']} |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- A Clarus cell is the first physical unit that can carry the life triad across cycles.",
            "- It is cell-like before it is neuron-like.",
            "- Its earliest time is a possible interval, not a fossil-dated moment.",
            "- The empirical proof boundary remains the same: reaction, boundary, copying, lineage, ablation.",
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
                "structural_window_ga": result["answer"]["structural_window_ga"],
                "evidence_by_window_ga": result["answer"]["evidence_by_window_ga"],
                "minimal_form": result["minimal_form"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
