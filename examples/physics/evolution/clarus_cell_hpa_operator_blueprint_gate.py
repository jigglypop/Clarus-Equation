"""HPA subcellular blueprint gate for Clarus-cell operators.

The empirical perturbation gates test whether Clarus-cell operators matter for
survival, maintenance, and support.  This gate asks a different question:

    Do the operator proteins sit in the expected cell compartments?

It uses the Human Protein Atlas subcellular-location table.  This is not a
causal perturbation test; it is an architectural blueprint gate for whether
the operator map B,U,E,A,I,D,Q,S,R has physical cellular anchors.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "evolution" / "clarus_cell"
DEFAULT_HPA_ZIP = DATA_DIR / "hpa_subcellular_location.tsv.zip"
RESULT_JSON = Path(__file__).with_name("clarus_cell_hpa_operator_blueprint_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_hpa_operator_blueprint_report.md")

HPA_DOWNLOAD = "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
HPA_SOURCE = "https://www.proteinatlas.org/humanproteome/subcellular/data"
RELIABLE = {"Enhanced", "Supported", "Approved"}


@dataclass(frozen=True)
class OperatorBlueprint:
    key: str
    variables: str
    role: str
    genes: tuple[str, ...]
    expected_locations: tuple[str, ...]
    min_observed: int
    min_expected_fraction: float


BLUEPRINTS = (
    OperatorBlueprint(
        key="B_boundary_membrane",
        variables="B,U,R",
        role="inside/outside boundary and compartment surface identity",
        genes=(
            "ATP1A1",
            "ATP2A2",
            "ATP6V0C",
            "ATP6V0D1",
            "ATP6V1A",
            "ATP6V1B2",
            "CLTC",
            "DNM2",
            "AP2M1",
            "AP3S1",
            "COPA",
            "COPB1",
            "RAB7A",
            "TSG101",
        ),
        expected_locations=(
            "Plasma membrane",
            "Cell Junctions",
            "Vesicles",
            "Endosomes",
            "Lysosomes",
        ),
        min_observed=10,
        min_expected_fraction=0.70,
    ),
    OperatorBlueprint(
        key="U_regulated_ports_traffic",
        variables="U,B,Q,R",
        role="ports, vesicles, ER/Golgi routing, and traffic control",
        genes=(
            "AP1G1",
            "AP2M1",
            "AP3S1",
            "CLTC",
            "COPA",
            "COPB1",
            "SEC61A1",
            "SEC61B",
            "SRP54",
            "RAB5A",
            "RAB7A",
            "RAB11A",
            "VPS11",
            "VPS16",
            "VPS18",
            "VPS29",
            "VPS35",
            "SNX1",
            "TSG101",
            "CHMP4B",
        ),
        expected_locations=(
            "Vesicles",
            "Endosomes",
            "Golgi apparatus",
            "Endoplasmic reticulum",
            "Plasma membrane",
        ),
        min_observed=14,
        min_expected_fraction=0.70,
    ),
    OperatorBlueprint(
        key="E_energy_mitochondria",
        variables="E,A,R",
        role="mitochondrial energy support",
        genes=(
            "ATP5F1A",
            "ATP5F1B",
            "ATP5F1C",
            "ATP5F1D",
            "ATP5F1E",
            "ATP5MC1",
            "ATP5MC2",
            "ATP5MC3",
            "COX4I1",
            "COX5A",
            "COX6B1",
            "NDUFA9",
            "NDUFB4",
            "NDUFB9",
            "NDUFS2",
            "NDUFS8",
            "SDHA",
            "SDHB",
            "SDHC",
            "SDHD",
            "UQCRC1",
            "UQCRC2",
            "VDAC1",
            "TFAM",
        ),
        expected_locations=("Mitochondria",),
        min_observed=12,
        min_expected_fraction=0.85,
    ),
    OperatorBlueprint(
        key="A_metabolic_autocatalytic_core",
        variables="A,E,R",
        role="central metabolism and biosynthetic core",
        genes=(
            "GAPDH",
            "TPI1",
            "ENO1",
            "PKM",
            "LDHA",
            "ACLY",
            "ACACA",
            "FASN",
            "G6PD",
            "SHMT2",
            "MTHFD1",
            "DHFR",
            "IMPDH2",
            "RRM1",
            "RRM2",
            "ATIC",
            "PFKP",
            "PGK1",
        ),
        expected_locations=("Cytosol", "Mitochondria", "Nucleoplasm"),
        min_observed=14,
        min_expected_fraction=0.80,
    ),
    OperatorBlueprint(
        key="I_identity_template",
        variables="I,R",
        role="genome, transcription, replication, and template continuity",
        genes=(
            "POLR2A",
            "POLR2B",
            "POLR2C",
            "RPA1",
            "RPA2",
            "RPA3",
            "PCNA",
            "MCM2",
            "MCM3",
            "MCM4",
            "MCM5",
            "MCM6",
            "MCM7",
            "ORC1",
            "ORC2",
            "ORC3",
            "RFC1",
            "RFC2",
            "RFC3",
            "TOP1",
            "TOP2A",
            "DNMT1",
            "UHRF1",
            "HDAC1",
            "EZH2",
            "SUZ12",
        ),
        expected_locations=(
            "Nucleoplasm",
            "Nucleoli",
            "Nuclear bodies",
            "Nuclear speckles",
            "Mitotic chromosome",
        ),
        min_observed=20,
        min_expected_fraction=0.85,
    ),
    OperatorBlueprint(
        key="D_Q_repair_quality_control",
        variables="D,Q,R",
        role="damage response, repair, autophagy, lysosome, and proteostasis",
        genes=(
            "ATM",
            "ATR",
            "CHEK1",
            "RAD51",
            "BRCA1",
            "BRCA2",
            "BARD1",
            "PARP1",
            "XRCC5",
            "XRCC6",
            "PRKDC",
            "FANCD2",
            "ATG3",
            "ATG5",
            "ATG7",
            "ATG12",
            "ATG13",
            "ATG14",
            "BECN1",
            "SQSTM1",
            "VCP",
            "PSMD1",
            "PSMD2",
            "PSMC1",
            "CTSD",
            "LAMP2",
        ),
        expected_locations=(
            "Lysosomes",
            "Vesicles",
            "Aggresome",
            "Cytosol",
            "Nucleoplasm",
            "Nuclear bodies",
            "Endosomes",
        ),
        min_observed=20,
        min_expected_fraction=0.80,
    ),
    OperatorBlueprint(
        key="S_support_context",
        variables="S,D,Q,U,R",
        role="secreted, membrane, and vesicle/extracellular support signals",
        genes=(
            "APOE",
            "CLU",
            "C3",
            "CCL2",
            "CCL20",
            "CSF1",
            "CXCL8",
            "CXCL10",
            "ICAM1",
            "VCAM1",
            "IL6",
            "SOD2",
            "TREM2",
            "CSF1R",
            "C1QA",
            "C1QB",
            "C1QC",
            "AXL",
            "MERTK",
            "GAS6",
        ),
        expected_locations=(
            "Predicted to be secreted",
            "Secreted",
            "Plasma membrane",
            "Vesicles",
            "Endosomes",
            "Lysosomes",
            "Extracellular",
        ),
        min_observed=10,
        min_expected_fraction=0.70,
    ),
)


def split_locations(value: str) -> list[str]:
    if not value:
        return []
    parts: list[str] = []
    for chunk in value.split(";"):
        clean = chunk.strip()
        if clean:
            parts.append(clean)
    return parts


def row_locations(row: dict[str, str]) -> list[str]:
    locations: list[str] = []
    for field in (
        "Main location",
        "Additional location",
        "Extracellular location",
        "Enhanced",
        "Supported",
        "Approved",
    ):
        locations.extend(split_locations(row.get(field, "")))
    return sorted(set(locations))


def load_hpa(path: Path, include_uncertain: bool) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names:
            return rows
        with archive.open(names[0]) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"), delimiter="\t")
            for row in reader:
                gene = (row.get("Gene name") or "").strip().upper()
                if not gene:
                    continue
                reliability = row.get("Reliability", "")
                if not include_uncertain and reliability not in RELIABLE:
                    continue
                rows[gene] = {
                    "gene": gene,
                    "ensembl": row.get("Gene", ""),
                    "reliability": reliability,
                    "locations": row_locations(row),
                    "go": row.get("GO id", ""),
                }
    return rows


def location_matches(locations: list[str], expected: tuple[str, ...]) -> bool:
    text = ";".join(locations)
    return any(token in text for token in expected)


def summarize_blueprint(
    blueprint: OperatorBlueprint,
    hpa: dict[str, dict[str, Any]],
    example_genes: int,
) -> dict[str, Any]:
    observed = [hpa[gene] for gene in blueprint.genes if gene in hpa]
    matched = [
        row
        for row in observed
        if location_matches(row["locations"], blueprint.expected_locations)
    ]
    unmatched = [row for row in observed if row not in matched]
    location_counts: dict[str, int] = {}
    for row in observed:
        for location in row["locations"]:
            location_counts[location] = location_counts.get(location, 0) + 1
    expected_coverage = {
        location: sum(1 for row in observed if location in ";".join(row["locations"]))
        for location in blueprint.expected_locations
    }
    observed_count = len(observed)
    expected_fraction = len(matched) / observed_count if observed_count else 0.0
    passed = bool(
        observed_count >= blueprint.min_observed
        and expected_fraction >= blueprint.min_expected_fraction
    )
    examples = sorted(
        matched,
        key=lambda row: (row["reliability"] != "Enhanced", row["gene"]),
    )[:example_genes]
    return {
        "key": blueprint.key,
        "variables": blueprint.variables,
        "role": blueprint.role,
        "candidate_genes": len(blueprint.genes),
        "observed_genes": observed_count,
        "matched_expected_location_genes": len(matched),
        "expected_location_fraction": round(expected_fraction, 6),
        "expected_locations": blueprint.expected_locations,
        "expected_coverage": expected_coverage,
        "top_locations": sorted(location_counts.items(), key=lambda item: (-item[1], item[0]))[:8],
        "unmatched_genes": [row["gene"] for row in unmatched],
        "passed": passed,
        "criteria": {
            "observed_ok": observed_count >= blueprint.min_observed,
            "expected_fraction_ok": expected_fraction >= blueprint.min_expected_fraction,
            "min_observed": blueprint.min_observed,
            "min_expected_fraction": blueprint.min_expected_fraction,
        },
        "examples": [
            {
                "gene": row["gene"],
                "reliability": row["reliability"],
                "locations": row["locations"],
            }
            for row in examples
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.hpa_zip.exists():
        return {
            "gate": "clarus_cell_hpa_operator_blueprint",
            "passed": False,
            "reason": "missing_data",
            "expected_path": str(args.hpa_zip),
            "download": HPA_DOWNLOAD,
        }
    hpa = load_hpa(args.hpa_zip, args.include_uncertain)
    summaries = [
        summarize_blueprint(blueprint, hpa, args.example_genes)
        for blueprint in BLUEPRINTS
    ]
    passed_count = sum(1 for summary in summaries if summary["passed"])
    operator_support: set[str] = set()
    for summary in summaries:
        if summary["passed"]:
            operator_support.update(summary["variables"].split(","))
    distinct_expected_locations = sorted(
        {
            location
            for summary in summaries
            if summary["passed"]
            for location, count in summary["expected_coverage"].items()
            if count > 0
        }
    )
    passed = bool(
        passed_count >= args.min_passed_blueprints
        and len(distinct_expected_locations) >= args.min_distinct_location_classes
        and set("BUEAIQS").issubset(operator_support)
    )
    return {
        "gate": "clarus_cell_hpa_operator_blueprint",
        "passed": passed,
        "claim_level": "empirical_subcellular_operator_blueprint"
        if passed
        else "parsed_no_promotion",
        "operators_supported": tuple(operator for operator in "BUEAIQD SR".replace(" ", "") if operator in operator_support),
        "hpa_source": HPA_SOURCE,
        "hpa_download": HPA_DOWNLOAD,
        "hpa_zip": str(args.hpa_zip.resolve()),
        "include_uncertain": args.include_uncertain,
        "hpa_genes_loaded": len(hpa),
        "passed_blueprints": passed_count,
        "min_passed_blueprints": args.min_passed_blueprints,
        "distinct_expected_locations": distinct_expected_locations,
        "operator_summaries": summaries,
        "claim_boundary": (
            "This is a static subcellular localization blueprint. It supports the "
            "physical compartment map of Clarus-cell operators, but it is not a "
            "perturbational recurrence or dynamics proof."
        ),
    }


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("reason") == "missing_data":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell HPA operator blueprint gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local HPA subcellular data",
                    f"- expected path: `{result['expected_path']}`",
                    f"- download: <{result['download']}>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    lines = [
        "# Clarus cell HPA operator blueprint gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- source: [Human Protein Atlas subcellular data]({result['hpa_source']})",
        f"- local data: `{result['hpa_zip']}`",
        f"- HPA genes loaded: `{result['hpa_genes_loaded']}`",
        f"- passed blueprints: `{result['passed_blueprints']}/{len(result['operator_summaries'])}`",
        f"- operators supported: `{','.join(result['operators_supported'])}`",
        f"- distinct expected location classes: `{len(result['distinct_expected_locations'])}`",
        "",
        "## operator summaries",
        "",
        "| operator | vars | observed | matched | fraction | passed |",
        "|---|---|---:|---:|---:|---|",
    ]
    for summary in result["operator_summaries"]:
        lines.append(
            f"| `{summary['key']}` | `{summary['variables']}` | "
            f"{summary['observed_genes']} | {summary['matched_expected_location_genes']} | "
            f"{summary['expected_location_fraction']:.3f} | `{summary['passed']}` |"
        )

    lines.extend(["", "## location coverage", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        lines.append("| expected location | genes |")
        lines.append("|---|---:|")
        for location, count in summary["expected_coverage"].items():
            lines.append(f"| `{location}` | {count} |")
        if summary["unmatched_genes"]:
            lines.append("")
            lines.append(f"- unmatched observed genes: `{','.join(summary['unmatched_genes'])}`")
        lines.append("")

    lines.extend(["## examples", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        for row in summary["examples"]:
            lines.append(
                f"- `{row['gene']}` ({row['reliability']}): "
                f"`{';'.join(row['locations'])}`"
            )
        lines.append("")

    lines.extend(["## claim boundary", "", result["claim_boundary"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpa-zip", type=Path, default=DEFAULT_HPA_ZIP)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--min-passed-blueprints", type=int, default=6)
    parser.add_argument("--min-distinct-location-classes", type=int, default=8)
    parser.add_argument("--example-genes", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
