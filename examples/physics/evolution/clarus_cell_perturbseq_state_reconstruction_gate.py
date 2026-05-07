"""Replogle Perturb-seq state-reconstruction gate for Clarus-cell operators.

This gate uses the processed pseudo-bulk AnnData files from Replogle et al.
2022:

* K562 essential-scale CRISPRi, normalized pseudo-bulk
* RPE1 essential-scale CRISPRi, normalized pseudo-bulk

The question is not whether every perturbation kills the cell.  The question is
whether perturbing Clarus-cell operator genes reconstructs a detectable
transcriptomic state vector:

    E,A,I,D,Q,R

Two signals are separated on purpose:

* broad state shift: whole-transcriptome RMS above non-targeting controls;
* module-local reconstruction: operator-program RMS above non-targeting
  controls for the same operator's measured genes.

This keeps weak broad E shifts from being overclaimed while still recording
whether the E module itself responds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "evolution" / "clarus_cell"
DEFAULT_K562 = DATA_DIR / "replogle_k562_essential_normalized_bulk_01.h5ad"
DEFAULT_RPE1 = DATA_DIR / "replogle_rpe1_normalized_bulk_01.h5ad"
DEFAULT_SUMMARY_CSV = DATA_DIR / "replogle_perturbseq_clarus_state_summary.csv"
RESULT_JSON = Path(__file__).with_name("clarus_cell_perturbseq_state_reconstruction_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_perturbseq_state_reconstruction_report.md")

PRIMARY_PAPER = "https://doi.org/10.1016/j.cell.2022.05.013"
FIGSHARE_DATASET = (
    "https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-"
    "phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_"
    "processed_Perturb-seq_datasets/20029387"
)
FIGSHARE_API = "https://api.figshare.com/v2/articles/20029387"
K562_DOWNLOAD = "https://ndownloader.figshare.com/files/35780870"
RPE1_DOWNLOAD = "https://ndownloader.figshare.com/files/35775512"

OPERATORS = ("E", "A", "I", "D", "Q", "R")


@dataclass(frozen=True)
class OperatorClass:
    key: str
    variables: str
    role: str
    genes: tuple[str, ...]
    min_observed_rows: int
    min_program_genes: int
    min_module_active_fraction: float
    min_broad_active_fraction: float


OPERATOR_CLASSES = (
    OperatorClass(
        key="E_energy_mitochondria_state",
        variables="E",
        role="mitochondrial energy and homeostasis transcriptional module",
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
        min_observed_rows=8,
        min_program_genes=16,
        min_module_active_fraction=0.60,
        min_broad_active_fraction=0.30,
    ),
    OperatorClass(
        key="A_metabolic_core_state",
        variables="A",
        role="central metabolism and biosynthetic autocatalytic state",
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
        min_observed_rows=10,
        min_program_genes=12,
        min_module_active_fraction=0.60,
        min_broad_active_fraction=0.30,
    ),
    OperatorClass(
        key="I_identity_template_state",
        variables="I",
        role="DNA replication, transcription, and chromatin identity state",
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
        min_observed_rows=16,
        min_program_genes=18,
        min_module_active_fraction=0.60,
        min_broad_active_fraction=0.40,
    ),
    OperatorClass(
        key="D_Q_repair_quality_state",
        variables="D,Q",
        role="repair, proteostasis, autophagy, and damage-control state",
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
        min_observed_rows=10,
        min_program_genes=18,
        min_module_active_fraction=0.60,
        min_broad_active_fraction=0.40,
    ),
    OperatorClass(
        key="R_recurrence_cell_cycle_state",
        variables="R",
        role="cell-cycle recurrence and proliferative state",
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
        min_observed_rows=14,
        min_program_genes=14,
        min_module_active_fraction=0.60,
        min_broad_active_fraction=0.35,
    ),
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: Path
    download_url: str


def optional_dependencies() -> tuple[Any, Any, str | None]:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        return None, None, str(exc)
    return h5py, np, None


def decode_strings(values: Any) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def target_gene(value: str) -> str:
    parts = value.split("_")
    if len(parts) > 1:
        return parts[1].upper()
    return value.upper()


def decode_var_gene_names(handle: Any) -> list[str]:
    categories = decode_strings(handle["var/__categories/gene_name"][:])
    codes = handle["var/gene_name"][:]
    names: list[str] = []
    for code in codes:
        names.append(categories[int(code)].upper() if int(code) >= 0 else "")
    return names


def quantiles(np: Any, values: Any) -> dict[str, float]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return {"median": math.nan, "q75": math.nan, "q90": math.nan, "q95": math.nan}
    return {
        "median": round(float(np.quantile(arr, 0.50)), 6),
        "q75": round(float(np.quantile(arr, 0.75)), 6),
        "q90": round(float(np.quantile(arr, 0.90)), 6),
        "q95": round(float(np.quantile(arr, 0.95)), 6),
    }


def safe_median(np: Any, values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return float(np.median(np.asarray(clean, dtype=float)))


def summarize_dataset(np: Any, h5py: Any, spec: DatasetSpec, args: argparse.Namespace) -> dict[str, Any]:
    with h5py.File(spec.path, "r") as handle:
        matrix = handle["X"][:].astype(float)
        matrix[~np.isfinite(matrix)] = 0.0
        obs_names = decode_strings(handle["obs/gene_transcript"][:])
        obs_genes = [target_gene(value) for value in obs_names]
        var_genes = decode_var_gene_names(handle)
        core_control = handle["obs/core_control"][:].astype(bool)
        ad_counts = handle["obs/anderson_darling_counts"][:].astype(float)
        leverage = handle["obs/mean_leverage_score"][:].astype(float)
        cells = handle["obs/num_cells_filtered"][:].astype(float)

    row_rms = np.sqrt(np.mean(matrix * matrix, axis=1))
    control_indices = np.where(core_control)[0]
    control_global_q95 = float(np.quantile(row_rms[control_indices], args.control_quantile))
    control_ad_q95 = float(np.quantile(ad_counts[control_indices], args.control_quantile))
    control_leverage_q95 = float(np.quantile(leverage[control_indices], args.control_quantile))

    operator_summaries = []
    for operator in OPERATOR_CLASSES:
        target_set = set(operator.genes)
        row_indices = [
            index
            for index, gene in enumerate(obs_genes)
            if gene in target_set and not core_control[index]
        ]
        program_indices = [index for index, gene in enumerate(var_genes) if gene in target_set]
        if program_indices:
            module_rows = matrix[np.ix_(row_indices, program_indices)] if row_indices else np.empty((0, 0))
            control_module = matrix[np.ix_(control_indices, program_indices)]
            module_rms = np.sqrt(np.mean(module_rows * module_rows, axis=1)) if row_indices else np.array([])
            control_module_rms = np.sqrt(np.mean(control_module * control_module, axis=1))
            control_module_q95 = float(np.quantile(control_module_rms, args.control_quantile))
        else:
            module_rms = np.array([])
            control_module_q95 = math.nan

        broad_active = [
            index
            for index in row_indices
            if row_rms[index] > control_global_q95 and ad_counts[index] > control_ad_q95
        ]
        module_active = [
            row_indices[pos]
            for pos, value in enumerate(module_rms)
            if math.isfinite(float(value)) and float(value) > control_module_q95
        ]
        observed = len(row_indices)
        broad_fraction = len(broad_active) / observed if observed else 0.0
        module_fraction = len(module_active) / observed if observed else 0.0
        criteria = {
            "observed_ok": observed >= operator.min_observed_rows,
            "program_genes_ok": len(program_indices) >= operator.min_program_genes,
            "module_active_fraction_ok": module_fraction >= operator.min_module_active_fraction,
            "broad_active_fraction_ok": broad_fraction >= operator.min_broad_active_fraction,
        }
        top_indices = sorted(row_indices, key=lambda index: row_rms[index], reverse=True)[: args.example_genes]
        module_by_row = {
            row_indices[pos]: float(module_rms[pos])
            for pos in range(len(row_indices))
            if math.isfinite(float(module_rms[pos]))
        }
        operator_summaries.append(
            {
                "key": operator.key,
                "dataset": spec.key,
                "variables": operator.variables,
                "role": operator.role,
                "candidate_genes": len(operator.genes),
                "observed_rows": observed,
                "program_genes": len(program_indices),
                "broad_active_rows": len(broad_active),
                "broad_active_fraction": round(broad_fraction, 6),
                "module_active_rows": len(module_active),
                "module_active_fraction": round(module_fraction, 6),
                "median_global_rms": round(safe_median(np, [float(row_rms[index]) for index in row_indices]), 6),
                "median_module_rms": round(safe_median(np, [float(value) for value in module_rms]), 6),
                "control_global_q95": round(control_global_q95, 6),
                "control_module_q95": round(control_module_q95, 6),
                "criteria": criteria,
                "module_passed": all(
                    criteria[key]
                    for key in ("observed_ok", "program_genes_ok", "module_active_fraction_ok")
                ),
                "broad_passed": criteria["observed_ok"] and criteria["broad_active_fraction_ok"],
                "examples": [
                    {
                        "gene": obs_genes[index],
                        "global_rms": round(float(row_rms[index]), 6),
                        "module_rms": round(module_by_row.get(index, math.nan), 6),
                        "anderson_darling_counts": round(float(ad_counts[index]), 6),
                        "mean_leverage_score": round(float(leverage[index]), 6),
                        "num_cells_filtered": round(float(cells[index]), 6),
                    }
                    for index in top_indices
                ],
            }
        )

    return {
        "dataset": spec.key,
        "path": str(spec.path.resolve()),
        "download_url": spec.download_url,
        "rows": matrix.shape[0],
        "genes": matrix.shape[1],
        "core_controls": int(core_control.sum()),
        "control_thresholds": {
            "global_rms_q95": round(control_global_q95, 6),
            "anderson_darling_q95": round(control_ad_q95, 6),
            "mean_leverage_q95": round(control_leverage_q95, 6),
        },
        "global_rms_distribution": quantiles(np, row_rms),
        "control_global_rms_distribution": quantiles(np, row_rms[control_indices]),
        "operator_summaries": operator_summaries,
    }


def aggregate_operator(dataset_results: list[dict[str, Any]], operator: OperatorClass) -> dict[str, Any]:
    summaries = [
        summary
        for dataset in dataset_results
        for summary in dataset["operator_summaries"]
        if summary["key"] == operator.key
    ]
    module_passed = [summary for summary in summaries if summary["module_passed"]]
    broad_passed = [summary for summary in summaries if summary["broad_passed"]]
    replicated_module = len(module_passed) >= 2
    replicated_broad = len(broad_passed) >= 2
    any_broad = bool(broad_passed)
    if operator.key == "E_energy_mitochondria_state":
        passed = replicated_module
    elif operator.key in {"A_metabolic_core_state", "R_recurrence_cell_cycle_state"}:
        passed = bool(replicated_module and any_broad)
    else:
        passed = bool(replicated_module and replicated_broad)
    return {
        "key": operator.key,
        "variables": operator.variables,
        "role": operator.role,
        "datasets": [summary["dataset"] for summary in summaries],
        "observed_rows_total": sum(summary["observed_rows"] for summary in summaries),
        "module_passed_datasets": [
            summary["dataset"] for summary in summaries if summary["module_passed"]
        ],
        "broad_passed_datasets": [
            summary["dataset"] for summary in summaries if summary["broad_passed"]
        ],
        "replicated_module": replicated_module,
        "replicated_broad": replicated_broad,
        "any_broad": any_broad,
        "passed": passed,
        "claim_note": (
            "module-local replicated; broad state in one dataset"
            if passed and any_broad and not replicated_broad
            else "module-local replicated; broad state weak"
            if passed and not replicated_broad
            else "module-local and broad state replicated"
            if passed
            else "not replicated"
        ),
    }


def write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "operator",
                "variables",
                "observed_rows",
                "program_genes",
                "broad_active_rows",
                "broad_active_fraction",
                "module_active_rows",
                "module_active_fraction",
                "median_global_rms",
                "median_module_rms",
                "module_passed",
                "broad_passed",
            ),
        )
        writer.writeheader()
        for dataset in result.get("datasets", []):
            for summary in dataset["operator_summaries"]:
                writer.writerow(
                    {
                        "dataset": dataset["dataset"],
                        "operator": summary["key"],
                        "variables": summary["variables"],
                        "observed_rows": summary["observed_rows"],
                        "program_genes": summary["program_genes"],
                        "broad_active_rows": summary["broad_active_rows"],
                        "broad_active_fraction": summary["broad_active_fraction"],
                        "module_active_rows": summary["module_active_rows"],
                        "module_active_fraction": summary["module_active_fraction"],
                        "median_global_rms": summary["median_global_rms"],
                        "median_module_rms": summary["median_module_rms"],
                        "module_passed": summary["module_passed"],
                        "broad_passed": summary["broad_passed"],
                    }
                )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    h5py, np, missing = optional_dependencies()
    if missing:
        return {
            "gate": "clarus_cell_perturbseq_state_reconstruction",
            "passed": False,
            "reason": "missing_dependency",
            "missing": missing,
            "install_command": ".venv\\Scripts\\python.exe -m pip install h5py",
        }

    specs = (
        DatasetSpec("K562_essential", args.k562_h5ad, K562_DOWNLOAD),
        DatasetSpec("RPE1_essential", args.rpe1_h5ad, RPE1_DOWNLOAD),
    )
    missing_files = [spec for spec in specs if not spec.path.exists()]
    if missing_files:
        return {
            "gate": "clarus_cell_perturbseq_state_reconstruction",
            "passed": False,
            "reason": "missing_data",
            "missing_files": [
                {"dataset": spec.key, "path": str(spec.path), "download_url": spec.download_url}
                for spec in missing_files
            ],
            "figshare_dataset": FIGSHARE_DATASET,
            "figshare_api": FIGSHARE_API,
        }

    dataset_results = [summarize_dataset(np, h5py, spec, args) for spec in specs]
    aggregate = [aggregate_operator(dataset_results, operator) for operator in OPERATOR_CLASSES]
    supported: set[str] = set()
    broad_supported: set[str] = set()
    single_dataset_broad_supported: set[str] = set()
    module_local_only: set[str] = set()
    for summary in aggregate:
        if summary["passed"]:
            for variable in summary["variables"].split(","):
                supported.add(variable)
            if summary["replicated_broad"]:
                for variable in summary["variables"].split(","):
                    broad_supported.add(variable)
            elif summary["any_broad"]:
                for variable in summary["variables"].split(","):
                    single_dataset_broad_supported.add(variable)
            else:
                for variable in summary["variables"].split(","):
                    module_local_only.add(variable)

    broad_core_ok = (
        {"I", "D", "Q"}.issubset(broad_supported)
        and "R" in (broad_supported | single_dataset_broad_supported)
    )
    passed_operators = sum(1 for summary in aggregate if summary["passed"])
    passed = bool(passed_operators >= args.min_passed_operators and broad_core_ok)
    result = {
        "gate": "clarus_cell_perturbseq_state_reconstruction",
        "passed": passed,
        "claim_level": "empirical_perturbseq_operator_state_branch"
        if passed
        else "parsed_no_promotion",
        "primary_paper": PRIMARY_PAPER,
        "figshare_dataset": FIGSHARE_DATASET,
        "figshare_api": FIGSHARE_API,
        "summary_csv": str(args.summary_csv.resolve()),
        "control_quantile": args.control_quantile,
        "source_note": (
            "Processed normalized pseudo-bulk AnnData files. Rows are perturbation "
            "pseudobulk populations; columns are normalized transcript features."
        ),
        "datasets": dataset_results,
        "operator_aggregate": aggregate,
        "passed_operators": passed_operators,
        "min_passed_operators": args.min_passed_operators,
        "operators_supported": tuple(operator for operator in OPERATORS if operator in supported),
        "broad_state_operators": tuple(operator for operator in OPERATORS if operator in broad_supported),
        "single_dataset_broad_state_operators": tuple(
            operator for operator in OPERATORS if operator in single_dataset_broad_supported
        ),
        "module_local_only_operators": tuple(operator for operator in OPERATORS if operator in module_local_only),
        "broad_core_ok": broad_core_ok,
        "claim_boundary": (
            "This is a proliferative K562/RPE1 pseudo-bulk transcriptomic state gate. "
            "It supports operator-level state reconstruction, especially I/D/Q/R broad "
            "state shifts. E is promoted only as a replicated module-local "
            "transcriptomic response here, not as a broad transcriptome or direct "
            "mitochondrial morphology proof."
        ),
    }
    write_summary_csv(args.summary_csv.resolve(), result)
    return result


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("reason") == "missing_dependency":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell Perturb-seq state reconstruction gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing optional dependency",
                    f"- missing: `{result['missing']}`",
                    f"- install command: `{result['install_command']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    if result.get("reason") == "missing_data":
        lines = [
            "# Clarus cell Perturb-seq state reconstruction gate",
            "",
            "- passed: `False`",
            "- reason: missing local Replogle pseudo-bulk h5ad files",
            f"- source: [Figshare processed datasets]({result['figshare_dataset']})",
            "",
        ]
        for item in result["missing_files"]:
            lines.append(
                f"- `{item['dataset']}` expected at `{item['path']}`; "
                f"download: <{item['download_url']}>"
            )
        lines.append("")
        REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    lines = [
        "# Clarus cell Perturb-seq state reconstruction gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- primary paper: [Replogle et al. 2022]({result['primary_paper']})",
        f"- source: [Figshare processed datasets]({result['figshare_dataset']})",
        f"- summary csv: `{result['summary_csv']}`",
        f"- source note: {result['source_note']}",
        f"- operators supported: `{','.join(result['operators_supported'])}`",
        f"- broad state operators: `{','.join(result['broad_state_operators'])}`",
        "- single-dataset broad state operators: "
        f"`{','.join(result['single_dataset_broad_state_operators']) or 'none'}`",
        f"- module-local-only operators: `{','.join(result['module_local_only_operators']) or 'none'}`",
        f"- passed operators: `{result['passed_operators']}/{len(result['operator_aggregate'])}`",
        f"- broad core ok: `{result['broad_core_ok']}`",
        "",
        "## datasets",
        "",
        "| dataset | rows | genes | controls | control global q95 | control AD q95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in result["datasets"]:
        thresholds = dataset["control_thresholds"]
        lines.append(
            f"| `{dataset['dataset']}` | {dataset['rows']} | {dataset['genes']} | "
            f"{dataset['core_controls']} | {fmt(thresholds['global_rms_q95'])} | "
            f"{fmt(thresholds['anderson_darling_q95'])} |"
        )

    lines.extend(
        [
            "",
            "## aggregate operators",
            "",
            "| operator | vars | observed rows | module replicated | broad replicated | passed | note |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for summary in result["operator_aggregate"]:
        lines.append(
            f"| `{summary['key']}` | `{summary['variables']}` | "
            f"{summary['observed_rows_total']} | `{summary['replicated_module']}` | "
            f"`{summary['replicated_broad']}` | `{summary['passed']}` | "
            f"{summary['claim_note']} |"
        )

    lines.extend(["", "## dataset operator summaries", ""])
    for dataset in result["datasets"]:
        lines.append(f"### `{dataset['dataset']}`")
        lines.append("")
        lines.append(
            "| operator | observed | program genes | broad active | module active | "
            "median global RMS | median module RMS | broad pass | module pass |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for summary in dataset["operator_summaries"]:
            lines.append(
                f"| `{summary['key']}` | {summary['observed_rows']} | "
                f"{summary['program_genes']} | {summary['broad_active_rows']} "
                f"({summary['broad_active_fraction']:.3f}) | "
                f"{summary['module_active_rows']} ({summary['module_active_fraction']:.3f}) | "
                f"{fmt(summary['median_global_rms'])} | {fmt(summary['median_module_rms'])} | "
                f"`{summary['broad_passed']}` | `{summary['module_passed']}` |"
            )
        lines.append("")

    lines.extend(["## strongest examples", ""])
    for dataset in result["datasets"]:
        lines.append(f"### `{dataset['dataset']}`")
        lines.append("")
        for summary in dataset["operator_summaries"]:
            lines.append(f"#### `{summary['key']}`")
            lines.append("")
            lines.append("| gene | global RMS | module RMS | AD counts | leverage | cells |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in summary["examples"]:
                lines.append(
                    f"| `{row['gene']}` | {fmt(row['global_rms'])} | "
                    f"{fmt(row['module_rms'])} | {fmt(row['anderson_darling_counts'])} | "
                    f"{fmt(row['mean_leverage_score'])} | {fmt(row['num_cells_filtered'])} |"
                )
            lines.append("")

    lines.extend(["## claim boundary", "", result["claim_boundary"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k562-h5ad", type=Path, default=DEFAULT_K562)
    parser.add_argument("--rpe1-h5ad", type=Path, default=DEFAULT_RPE1)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--control-quantile", type=float, default=0.95)
    parser.add_argument("--min-passed-operators", type=int, default=4)
    parser.add_argument("--example-genes", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
