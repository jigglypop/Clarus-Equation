"""DepMap pilot gate for proliferative Clarus-cell operator dependency.

The postmitotic neuron gate tests recurrence as maintenance.  This gate tests
the other branch: recurrence as cell survival/proliferation.  It uses DepMap
CRISPR Chronos gene-effect scores, where 0 means little fitness effect and -1
approximately matches the median effect of common essential genes.

The gate intentionally keeps only a small local subset of the large DepMap
matrix: Clarus operator genes plus DepMap common-essential and nonessential
control genes.  Run once with --fetch-subset to build the subset from the
official Figshare download URL.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "evolution" / "clarus_cell"
RESULT_JSON = Path(__file__).with_name("clarus_cell_depmap_operator_dependency_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_depmap_operator_dependency_report.md")

DEPMAP_RELEASE = "DepMap 24Q4 Public"
DEPMAP_ARTICLE = "https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248"
DEPMAP_API = "https://api.figshare.com/v2/articles/27993248"
CRISPR_GENE_EFFECT_URL = "https://ndownloader.figshare.com/files/51064667"
COMMON_ESSENTIAL_URL = "https://ndownloader.figshare.com/files/51063560"
NONESSENTIAL_URL = "https://ndownloader.figshare.com/files/51063566"

DEFAULT_SUBSET = DATA_DIR / "depmap_24q4_clarus_operator_dependency_subset.csv"
DEFAULT_COMMON = DATA_DIR / "depmap_24q4_AchillesCommonEssentialControls.csv"
DEFAULT_NONESSENTIAL = DATA_DIR / "depmap_24q4_AchillesNonessentialControls.csv"


@dataclass(frozen=True)
class OperatorClass:
    key: str
    variables: str
    role: str
    genes: tuple[str, ...]
    min_observed: int
    max_median_effect: float
    min_dependent_fraction: float


OPERATOR_CLASSES = (
    OperatorClass(
        key="B_boundary_membrane",
        variables="B,U,R",
        role="membrane potential, vesicle boundary, and compartment maintenance",
        genes=(
            "ATP1A1",
            "ATP2A2",
            "ATP6V0C",
            "ATP6V0D1",
            "ATP6V1A",
            "ATP6V1B2",
            "ATP6V1E1",
            "ATP6V1H",
            "CLTC",
            "DNM2",
            "AP2M1",
            "AP3S1",
            "COPA",
            "COPB1",
            "RAB7A",
            "TSG101",
        ),
        min_observed=10,
        max_median_effect=-0.22,
        min_dependent_fraction=0.15,
    ),
    OperatorClass(
        key="U_regulated_ports_traffic",
        variables="U,B,Q,R",
        role="endosome, ER/Golgi, vesicle routing, and regulated import/export",
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
        min_observed=12,
        max_median_effect=-0.20,
        min_dependent_fraction=0.14,
    ),
    OperatorClass(
        key="E_energy_mitochondria",
        variables="E,A,R",
        role="mitochondrial ATP production and energy-homeostasis capacity",
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
        min_observed=16,
        max_median_effect=-0.35,
        min_dependent_fraction=0.25,
    ),
    OperatorClass(
        key="A_metabolic_autocatalytic_core",
        variables="A,E,R",
        role="central carbon, nucleotide, lipid, and biosynthetic autocatalytic core",
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
        min_observed=14,
        max_median_effect=-0.35,
        min_dependent_fraction=0.25,
    ),
    OperatorClass(
        key="I_identity_template",
        variables="I,R",
        role="DNA replication, transcription, chromatin, and lineage template continuity",
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
        min_observed=18,
        max_median_effect=-0.45,
        min_dependent_fraction=0.35,
    ),
    OperatorClass(
        key="D_Q_repair_quality_control",
        variables="D,Q,R",
        role="DNA repair, proteostasis, autophagy, lysosome, and damage-control reserve",
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
        min_observed=18,
        max_median_effect=-0.24,
        min_dependent_fraction=0.18,
    ),
    OperatorClass(
        key="R_proliferative_recurrence",
        variables="R,I,A",
        role="cell-cycle execution: recurrence by division",
        genes=(
            "CDK1",
            "CDK2",
            "CCNA2",
            "CCNB1",
            "CCNB2",
            "PLK1",
            "AURKA",
            "AURKB",
            "BUB1",
            "BUB1B",
            "CDC20",
            "CDC25A",
            "CDC25C",
            "E2F1",
            "MKI67",
            "PCNA",
            "MCM2",
            "MCM3",
            "MCM4",
            "MCM5",
            "MCM6",
            "MCM7",
        ),
        min_observed=18,
        max_median_effect=-0.45,
        min_dependent_fraction=0.35,
    ),
)


def gene_symbol(header: str) -> str:
    return header.split(" (", 1)[0].strip().upper()


def load_control_genes(path: Path) -> list[str]:
    genes: list[str] = []
    if not path.exists():
        return genes
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gene = (row.get("Gene") or "").strip()
            if gene:
                genes.append(gene)
    return genes


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def build_target_symbols() -> set[str]:
    symbols: set[str] = set()
    for operator in OPERATOR_CLASSES:
        symbols.update(operator.genes)
    return symbols


def fetch_subset(args: argparse.Namespace) -> dict[str, Any]:
    if not args.common_controls.exists():
        download_file(COMMON_ESSENTIAL_URL, args.common_controls)
    if not args.nonessential_controls.exists():
        download_file(NONESSENTIAL_URL, args.nonessential_controls)

    common_controls = set(load_control_genes(args.common_controls))
    nonessential_controls = set(load_control_genes(args.nonessential_controls))
    target_symbols = build_target_symbols()
    selected_exact = common_controls | nonessential_controls

    args.subset_csv.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.source_url, timeout=120) as response:
        text = io.TextIOWrapper(response, encoding="utf-8", newline="")
        reader = csv.reader(text)
        header = next(reader)
        selected_indices = [0]
        selected_headers = [header[0] or "ModelID"]
        selected_symbols: set[str] = set()
        selected_common = 0
        selected_nonessential = 0
        for index, name in enumerate(header[1:], start=1):
            symbol = gene_symbol(name)
            if symbol in target_symbols or name in selected_exact:
                selected_indices.append(index)
                selected_headers.append(name)
                selected_symbols.add(symbol)
                if name in common_controls:
                    selected_common += 1
                if name in nonessential_controls:
                    selected_nonessential += 1

        row_count = 0
        with args.subset_csv.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.writer(out_handle)
            writer.writerow(selected_headers)
            for row in reader:
                if not row:
                    continue
                writer.writerow([row[index] if index < len(row) else "" for index in selected_indices])
                row_count += 1

    return {
        "subset_path": str(args.subset_csv),
        "rows": row_count,
        "columns": len(selected_headers),
        "target_symbols_found": len(selected_symbols & target_symbols),
        "target_symbols_requested": len(target_symbols),
        "common_controls_found": selected_common,
        "nonessential_controls_found": selected_nonessential,
    }


def parse_float(value: str) -> float:
    try:
        score = float(value)
    except ValueError:
        return math.nan
    return score if math.isfinite(score) else math.nan


def median(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return statistics.median(clean)


def quantile(values: Iterable[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    pos = (len(clean) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return clean[int(pos)]
    return clean[lower] * (upper - pos) + clean[upper] * (pos - lower)


def load_subset(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        data = {name: [] for name in header[1:]}
        models: list[str] = []
        for row in reader:
            if not row:
                continue
            models.append(row[0])
            for index, name in enumerate(header[1:], start=1):
                data[name].append(parse_float(row[index] if index < len(row) else ""))
    return models, data


def summarize_gene(header: str, values: list[float], dependent_threshold: float) -> dict[str, Any]:
    clean = [value for value in values if math.isfinite(value)]
    dependent = [value for value in clean if value <= dependent_threshold]
    strong = [value for value in clean if value <= -1.0]
    return {
        "header": header,
        "symbol": gene_symbol(header),
        "n": len(clean),
        "median": median(clean),
        "mean": sum(clean) / len(clean) if clean else math.nan,
        "q10": quantile(clean, 0.10),
        "dependent_fraction": len(dependent) / len(clean) if clean else 0.0,
        "strong_fraction": len(strong) / len(clean) if clean else 0.0,
    }


def summarize_group(gene_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [row["median"] for row in gene_summaries]
    dep_fracs = [row["dependent_fraction"] for row in gene_summaries]
    strong_fracs = [row["strong_fraction"] for row in gene_summaries]
    return {
        "genes": len(gene_summaries),
        "median_gene_median_effect": median(medians),
        "median_dependent_fraction": median(dep_fracs),
        "median_strong_fraction": median(strong_fracs),
    }


def summarize_operator(
    operator: OperatorClass,
    by_symbol: dict[str, dict[str, Any]],
    control_summary: dict[str, Any],
) -> dict[str, Any]:
    selected = [by_symbol[gene] for gene in operator.genes if gene in by_symbol]
    selected.sort(key=lambda row: (row["median"], -row["dependent_fraction"]))
    group = summarize_group(selected)
    criteria = {
        "observed_ok": group["genes"] >= operator.min_observed,
        "median_effect_ok": group["median_gene_median_effect"] <= operator.max_median_effect,
        "dependent_fraction_ok": group["median_dependent_fraction"]
        >= operator.min_dependent_fraction,
        "separated_from_nonessential_ok": (
            group["median_gene_median_effect"]
            <= control_summary["nonessential"]["median_gene_median_effect"] - 0.15
        ),
    }
    return {
        "key": operator.key,
        "variables": operator.variables,
        "role": operator.role,
        "thresholds": {
            "min_observed": operator.min_observed,
            "max_median_effect": operator.max_median_effect,
            "min_dependent_fraction": operator.min_dependent_fraction,
        },
        **group,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "examples": [
            {
                "gene": row["symbol"],
                "median_effect": round(row["median"], 6),
                "dependent_fraction": round(row["dependent_fraction"], 6),
                "strong_fraction": round(row["strong_fraction"], 6),
            }
            for row in selected[:10]
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.fetch_subset or not args.subset_csv.exists() and args.auto_fetch_subset:
        fetch_info = fetch_subset(args)
    else:
        fetch_info = None

    if not args.subset_csv.exists():
        return {
            "gate": "clarus_cell_depmap_operator_dependency",
            "passed": False,
            "reason": "missing_subset",
            "subset_path": str(args.subset_csv),
            "how_to_build": "run with --fetch-subset",
            "source_url": args.source_url,
        }

    models, data = load_subset(args.subset_csv)
    common_controls = set(load_control_genes(args.common_controls))
    nonessential_controls = set(load_control_genes(args.nonessential_controls))

    gene_rows = [
        summarize_gene(header, values, args.dependent_threshold)
        for header, values in data.items()
    ]
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in gene_rows:
        existing = by_symbol.get(row["symbol"])
        if existing is None or row["median"] < existing["median"]:
            by_symbol[row["symbol"]] = row

    common_rows = [row for row in gene_rows if row["header"] in common_controls]
    nonessential_rows = [row for row in gene_rows if row["header"] in nonessential_controls]
    control_summary = {
        "common_essential": summarize_group(common_rows),
        "nonessential": summarize_group(nonessential_rows),
    }
    control_ok = bool(
        control_summary["common_essential"]["median_gene_median_effect"] <= -0.6
        and control_summary["nonessential"]["median_gene_median_effect"] >= -0.2
        and control_summary["common_essential"]["median_gene_median_effect"]
        <= control_summary["nonessential"]["median_gene_median_effect"] - 0.5
    )

    operator_summaries = [
        summarize_operator(operator, by_symbol, control_summary)
        for operator in OPERATOR_CLASSES
    ]
    passed_operators = sum(1 for summary in operator_summaries if summary["passed"])
    required_core = {
        "E_energy_mitochondria",
        "A_metabolic_autocatalytic_core",
        "I_identity_template",
        "R_proliferative_recurrence",
    }
    core_ok = all(
        summary["passed"]
        for summary in operator_summaries
        if summary["key"] in required_core
    )
    gate_passed = bool(
        control_ok
        and core_ok
        and passed_operators >= args.min_passed_operators
        and len(models) >= args.min_models
    )

    return {
        "gate": "clarus_cell_depmap_operator_dependency",
        "passed": gate_passed,
        "claim_level": "empirical_proliferative_recurrence_branch"
        if gate_passed
        else "parsed_no_promotion",
        "release": DEPMAP_RELEASE,
        "depmap_article": DEPMAP_ARTICLE,
        "depmap_api": DEPMAP_API,
        "source_url": args.source_url,
        "subset_path": str(args.subset_csv),
        "fetch_info": fetch_info,
        "models": len(models),
        "genes_in_subset": len(data),
        "dependent_threshold": args.dependent_threshold,
        "control_summary": control_summary,
        "control_ok": control_ok,
        "passed_operators": passed_operators,
        "min_passed_operators": args.min_passed_operators,
        "core_ok": core_ok,
        "operator_summaries": operator_summaries,
        "claim_boundary": (
            "This is a cancer-cell-line proliferative dependency gate. It supports "
            "Clarus recurrence as survival/proliferation for shared human cell "
            "operators, but it is not a normal tissue, developmental, or brain-wide proof."
        ),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("reason") == "missing_subset":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell DepMap operator dependency gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local subset",
                    f"- expected path: `{result['subset_path']}`",
                    f"- source: <{result['source_url']}>",
                    "- build command: `.venv\\Scripts\\python.exe examples\\physics\\evolution\\clarus_cell_depmap_operator_dependency_gate.py --fetch-subset`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    controls = result["control_summary"]
    lines = [
        "# Clarus cell DepMap operator dependency gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- release: [{result['release']}]({result['depmap_article']})",
        f"- local subset: `{result['subset_path']}`",
        f"- models: `{result['models']}`",
        f"- genes in subset: `{result['genes_in_subset']}`",
        f"- dependent threshold: `{result['dependent_threshold']}`",
        f"- control ok: `{result['control_ok']}`",
        f"- passed operators: `{result['passed_operators']}/{len(result['operator_summaries'])}`",
        f"- core ok: `{result['core_ok']}`",
        "",
        "## controls",
        "",
        "| control | genes | median gene median effect | median dependent fraction |",
        "|---|---:|---:|---:|",
    ]
    for key, summary in controls.items():
        lines.append(
            f"| `{key}` | {summary['genes']} | "
            f"{fmt(summary['median_gene_median_effect'])} | "
            f"{fmt(summary['median_dependent_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## operator summaries",
            "",
            "| operator | vars | genes | median effect | dependent frac | passed |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for summary in result["operator_summaries"]:
        lines.append(
            f"| `{summary['key']}` | `{summary['variables']}` | "
            f"{summary['genes']} | {fmt(summary['median_gene_median_effect'])} | "
            f"{fmt(summary['median_dependent_fraction'])} | `{summary['passed']}` |"
        )

    lines.extend(["", "## strongest dependencies by operator", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        lines.append("| gene | median effect | dependent fraction | strong fraction |")
        lines.append("|---|---:|---:|---:|")
        for row in summary["examples"]:
            lines.append(
                f"| `{row['gene']}` | {fmt(row['median_effect'])} | "
                f"{fmt(row['dependent_fraction'])} | {fmt(row['strong_fraction'])} |"
            )
        lines.append("")

    lines.extend(["## claim boundary", "", result["claim_boundary"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-csv", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--common-controls", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--nonessential-controls", type=Path, default=DEFAULT_NONESSENTIAL)
    parser.add_argument("--source-url", default=CRISPR_GENE_EFFECT_URL)
    parser.add_argument("--fetch-subset", action="store_true")
    parser.add_argument("--auto-fetch-subset", action="store_true")
    parser.add_argument("--dependent-threshold", type=float, default=-0.5)
    parser.add_argument("--min-models", type=int, default=900)
    parser.add_argument("--min-passed-operators", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
