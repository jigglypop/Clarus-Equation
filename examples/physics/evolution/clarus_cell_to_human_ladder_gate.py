"""Trace the Clarus-cell mechanism from protocell to human cell forms.

The primitive Clarus cell is a dividing bounded unit.  Human cells preserve the
same closure kernel, but many mature human cells, especially neurons, no longer
use their own division as the recurrence operator.  Their recurrence is carried
by maintenance, repair, signaling, tissue support, and organism-level lineage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("clarus_cell_to_human_ladder_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_to_human_ladder_report.md")


@dataclass(frozen=True)
class CellStage:
    name: str
    order: int
    clade_level: str
    boundary: bool
    metabolism: bool
    heredity: bool
    regulated_ports: bool
    recurrence_operator: str
    energy_specialization: bool
    internal_compartments: bool
    epigenetic_memory: bool
    tissue_signaling: bool
    repair_quality_control: bool
    neural_glial_coupling: bool
    human_specific: bool
    mechanism_note: str

    @property
    def primitive_kernel(self) -> bool:
        return self.boundary and self.metabolism and self.heredity and self.regulated_ports

    @property
    def recurrence_closed(self) -> bool:
        return self.recurrence_operator in {
            "division",
            "asymmetric_division",
            "maintenance_plus_tissue_replacement",
            "postmitotic_maintenance",
        }

    @property
    def human_clarus_cell(self) -> bool:
        return (
            self.human_specific
            and self.primitive_kernel
            and self.recurrence_closed
            and self.energy_specialization
            and self.internal_compartments
            and self.epigenetic_memory
            and self.tissue_signaling
            and self.repair_quality_control
        )

    @property
    def mechanism_score(self) -> int:
        return sum(
            (
                self.boundary,
                self.metabolism,
                self.heredity,
                self.regulated_ports,
                self.recurrence_closed,
                self.energy_specialization,
                self.internal_compartments,
                self.epigenetic_memory,
                self.tissue_signaling,
                self.repair_quality_control,
                self.neural_glial_coupling,
            )
        )


STAGES = (
    CellStage(
        name="template_bearing_protocell",
        order=0,
        clade_level="proto-life",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="division",
        energy_specialization=False,
        internal_compartments=False,
        epigenetic_memory=False,
        tissue_signaling=False,
        repair_quality_control=False,
        neural_glial_coupling=False,
        human_specific=False,
        mechanism_note="minimum Clarus kernel: boundary, resource flow, autocatalysis, template copying, division",
    ),
    CellStage(
        name="luca_like_prokaryotic_cell",
        order=1,
        clade_level="cellular ancestor candidate",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="division",
        energy_specialization=False,
        internal_compartments=False,
        epigenetic_memory=False,
        tissue_signaling=False,
        repair_quality_control=True,
        neural_glial_coupling=False,
        human_specific=False,
        mechanism_note="genome-metabolism-boundary loop gains stronger repair and regulated replication",
    ),
    CellStage(
        name="bacterial_or_archaeal_clarus_cell",
        order=2,
        clade_level="prokaryotic cell",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="division",
        energy_specialization=False,
        internal_compartments=False,
        epigenetic_memory=True,
        tissue_signaling=False,
        repair_quality_control=True,
        neural_glial_coupling=False,
        human_specific=False,
        mechanism_note="sensor surfaces and regulatory memory make behavior-ready single cells",
    ),
    CellStage(
        name="eukaryotic_clarus_cell",
        order=3,
        clade_level="eukaryote",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="division",
        energy_specialization=True,
        internal_compartments=True,
        epigenetic_memory=True,
        tissue_signaling=False,
        repair_quality_control=True,
        neural_glial_coupling=False,
        human_specific=False,
        mechanism_note="organelles split the primitive core into energy, genome, traffic, and degradation subloops",
    ),
    CellStage(
        name="metazoan_stem_or_somatic_clarus_cell",
        order=4,
        clade_level="multicellular animal",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="asymmetric_division",
        energy_specialization=True,
        internal_compartments=True,
        epigenetic_memory=True,
        tissue_signaling=True,
        repair_quality_control=True,
        neural_glial_coupling=False,
        human_specific=False,
        mechanism_note="single-cell recurrence is embedded in tissue signals, differentiation, apoptosis, and stem pools",
    ),
    CellStage(
        name="vertebrate_specialized_clarus_cell",
        order=5,
        clade_level="vertebrate",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="maintenance_plus_tissue_replacement",
        energy_specialization=True,
        internal_compartments=True,
        epigenetic_memory=True,
        tissue_signaling=True,
        repair_quality_control=True,
        neural_glial_coupling=True,
        human_specific=False,
        mechanism_note="cell identity is stabilized by endocrine, immune, neural, and tissue context",
    ),
    CellStage(
        name="human_proliferative_clarus_cell",
        order=6,
        clade_level="human dividing cell",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="asymmetric_division",
        energy_specialization=True,
        internal_compartments=True,
        epigenetic_memory=True,
        tissue_signaling=True,
        repair_quality_control=True,
        neural_glial_coupling=False,
        human_specific=True,
        mechanism_note="stem, epithelial, immune, or repair-capable cells keep the primitive division loop under tissue control",
    ),
    CellStage(
        name="human_postmitotic_neural_clarus_cell",
        order=7,
        clade_level="human postmitotic neural cell",
        boundary=True,
        metabolism=True,
        heredity=True,
        regulated_ports=True,
        recurrence_operator="postmitotic_maintenance",
        energy_specialization=True,
        internal_compartments=True,
        epigenetic_memory=True,
        tissue_signaling=True,
        repair_quality_control=True,
        neural_glial_coupling=True,
        human_specific=True,
        mechanism_note="neuron-like cells replace division recurrence with membrane excitability, synaptic state, glial support, repair, and long-lived maintenance",
    ),
)


HUMAN_OPERATORS = (
    {
        "name": "plasma_membrane_identity",
        "primitive_source": "boundary_retention",
        "human_form": "lipid membrane, channels, transporters, receptors, adhesion",
        "failure_mode": "loss of excitability, osmotic identity, receptor-defined cell state",
    },
    {
        "name": "mitochondrial_energy_closure",
        "primitive_source": "autocatalytic_core",
        "human_form": "ATP/redox/calcium coupling to biosynthesis and maintenance",
        "failure_mode": "maintenance, firing, repair, and division cannot be paid for",
    },
    {
        "name": "genome_epigenome_template",
        "primitive_source": "copying_template",
        "human_form": "DNA sequence plus chromatin and regulatory state",
        "failure_mode": "cell identity and lineage memory drift",
    },
    {
        "name": "vesicle_organelle_traffic",
        "primitive_source": "gradient_ports",
        "human_form": "ER/Golgi/endosome/lysosome/autophagy traffic",
        "failure_mode": "resource flow and waste control decouple from identity",
    },
    {
        "name": "cycle_or_maintenance_recurrence",
        "primitive_source": "division_threshold",
        "human_form": "cell cycle in proliferative cells; repair/autophagy/synaptic turnover in postmitotic cells",
        "failure_mode": "no recurrence operator for keeping the cell as itself over time",
    },
    {
        "name": "tissue_context_closure",
        "primitive_source": "population selection",
        "human_form": "ECM, immune, endocrine, vascular, glial, and neighboring-cell signals",
        "failure_mode": "the human cell cannot be interpreted as an isolated protocell",
    },
)


def stage_row(stage: CellStage) -> dict[str, Any]:
    row = asdict(stage)
    row["primitive_kernel"] = stage.primitive_kernel
    row["recurrence_closed"] = stage.recurrence_closed
    row["human_clarus_cell"] = stage.human_clarus_cell
    row["mechanism_score"] = stage.mechanism_score
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    stages = tuple(sorted(STAGES, key=lambda stage: stage.order))
    scores = [stage.mechanism_score for stage in stages]
    human_stages = [stage for stage in stages if stage.human_specific]
    stem_lineage = [stage for stage in stages if stage.order <= 4] + [
        stage for stage in human_stages if stage.name == "human_proliferative_clarus_cell"
    ]
    neural_lineage = [stage for stage in stages if stage.order <= 5] + [
        stage for stage in human_stages if stage.name == "human_postmitotic_neural_clarus_cell"
    ]
    branch_scores_ok = (
        [stage.mechanism_score for stage in stem_lineage]
        == sorted(stage.mechanism_score for stage in stem_lineage)
        and [stage.mechanism_score for stage in neural_lineage]
        == sorted(stage.mechanism_score for stage in neural_lineage)
    )
    result = {
        "gate": "clarus_cell_to_human_ladder",
        "passed": bool(
            all(stage.primitive_kernel for stage in stages)
            and all(stage.recurrence_closed for stage in stages)
            and branch_scores_ok
            and len(human_stages) == 2
            and all(stage.human_clarus_cell for stage in human_stages)
            and human_stages[0].recurrence_operator != human_stages[1].recurrence_operator
        ),
        "primitive_kernel": "boundary + metabolism + heredity + regulated ports + recurrence",
        "human_clarus_cell_forms": [stage.name for stage in human_stages],
        "branch_scores_ok": branch_scores_ok,
        "human_split": {
            "proliferative": "division/asymmetric-division recurrence under tissue control",
            "postmitotic_neural": "maintenance, repair, synaptic/membrane turnover, and glial support recurrence",
        },
        "operators": HUMAN_OPERATORS,
        "rows": [stage_row(stage) for stage in stages],
        "interpretation": (
            "Human Clarus cells do not abandon the protocell kernel.  They internalize it into "
            "organelles, genome/epigenome regulation, membrane signaling, quality control, and tissue context.  "
            "The key human upgrade is that recurrence can be cell division or long-lived maintenance."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Clarus cell to human ladder gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- primitive kernel: {result['primitive_kernel']}",
        f"- human forms: {', '.join(f'`{name}`' for name in result['human_clarus_cell_forms'])}",
        "",
        "## ladder",
        "",
        "| order | stage | clade | score | recurrence | human Clarus | mechanism note |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['order']} | `{row['name']}` | {row['clade_level']} | "
            f"{row['mechanism_score']} | `{row['recurrence_operator']}` | "
            f"`{row['human_clarus_cell']}` | {row['mechanism_note']} |"
        )
    lines.extend(
        [
            "",
            "## human operators",
            "",
            "| operator | primitive source | human form | failure mode |",
            "|---|---|---|---|",
        ]
    )
    for row in result["operators"]:
        lines.append(
            f"| `{row['name']}` | `{row['primitive_source']}` | {row['human_form']} | {row['failure_mode']} |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            result["interpretation"],
            "",
            "The human Clarus cell therefore has two valid forms:",
            "",
            "1. proliferative Clarus cell: recurrence by division under tissue control",
            "2. postmitotic neural Clarus cell: recurrence by maintenance, repair, membrane/synaptic turnover, and glial/tissue support",
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
                "human_clarus_cell_forms": result["human_clarus_cell_forms"],
                "human_split": result["human_split"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
