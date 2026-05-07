"""Empirical pilot gate for the Clarus-cell postmitotic repair branch.

This gate uses a public CRISPRbrain/GEO differential-expression table for
human iPSC-derived neurons comparing WT and PSAP knockout.  It does not prove
the whole Clarus-cell mechanism.  It asks a narrower question:

    Does a lysosomal/repair perturbation expose the D/Q maintenance operators
    expected by the postmitotic Clarus-cell branch?

The Clarus-cell variables touched here are mainly:

    D = damage/stress load
    Q = repair, lysosome, autophagy, proteostasis
    E = energy/mitochondrial maintenance
    I = postmitotic neural identity context
    R = recurrence as survival/maintenance rather than division
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = (
    REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "GSE152988_WT_vs_PSAPKO.csv.gz"
)
RESULT_JSON = Path(__file__).with_name("clarus_cell_psapko_neuron_repair_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_psapko_neuron_repair_report.md")

DATASET_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152988"
DATASET_DOWNLOAD = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE152988"
    "&file=GSE152988_WT_vs_PSAPKO.csv.gz&format=file"
)
PRIMARY_PAPER = "https://www.nature.com/articles/s41593-021-00862-0"


@dataclass(frozen=True)
class GeneRow:
    gene: str
    ensembl_id: str
    base_mean: float
    log2fc: float
    pvalue: float
    padj: float


@dataclass(frozen=True)
class OperatorGeneSet:
    key: str
    variables: str
    role: str
    exact: tuple[str, ...]
    prefixes: tuple[str, ...] = ()


OPERATOR_GENE_SETS = (
    OperatorGeneSet(
        key="Q_repair_lysosome_autophagy",
        variables="Q,D,R",
        role="lysosome/autophagy/proteostasis quality control after PSAP loss",
        exact=(
            "LAMP1",
            "LAMP2",
            "CTSA",
            "CTSB",
            "CTSD",
            "CTSF",
            "CTSH",
            "CTSL",
            "CTSO",
            "CTSZ",
            "GBA",
            "GBA2",
            "NPC1",
            "NPC2",
            "HEXA",
            "HEXB",
            "GLB1",
            "SMPD1",
            "GALC",
            "CLN3",
            "CLN5",
            "CLN6",
            "CLN8",
            "PPT1",
            "TPP1",
            "MFSD8",
            "GNPTAB",
            "GNPTG",
            "MCOLN1",
            "TFEB",
            "TFE3",
            "SQSTM1",
            "MAP1LC3B",
            "GABARAPL1",
            "ATG3",
            "ATG5",
            "ATG7",
            "ATG12",
            "BECN1",
            "WIPI1",
            "WIPI2",
            "ULK1",
            "PINK1",
            "PARK2",
            "VCP",
        ),
    ),
    OperatorGeneSet(
        key="D_damage_stress_response",
        variables="D,Q,R",
        role="ER stress, oxidative stress, and immediate damage response",
        exact=(
            "DDIT3",
            "ATF3",
            "ATF4",
            "ATF5",
            "XBP1",
            "HSPA5",
            "HSP90B1",
            "DNAJB9",
            "HERPUD1",
            "EDEM1",
            "SELENOS",
            "HMOX1",
            "NQO1",
            "TXNIP",
            "SOD1",
            "SOD2",
            "GPX1",
            "GPX4",
            "GCLC",
            "GCLM",
            "JUN",
            "JUNB",
            "FOS",
            "FOSB",
            "EGR1",
            "PPP1R15A",
            "CHAC1",
        ),
    ),
    OperatorGeneSet(
        key="E_mito_energy",
        variables="E,A,R",
        role="mitochondrial energy and organelle maintenance capacity",
        exact=(
            "MFN1",
            "MFN2",
            "OPA1",
            "DNM1L",
            "TFAM",
            "POLG",
            "VDAC1",
            "VDAC2",
            "VDAC3",
        ),
        prefixes=("NDUF", "ATP5", "COX", "UQCR", "SDH", "TOMM", "TIMM", "SLC25", "MT-"),
    ),
    OperatorGeneSet(
        key="U_traffic_boundary",
        variables="B,U,Q",
        role="membrane, endosome, vesicle traffic, and boundary-maintenance proxies",
        exact=(
            "RAB1A",
            "RAB3A",
            "RAB5A",
            "RAB7A",
            "RAB11A",
            "EEA1",
            "AP1G1",
            "AP2M1",
            "COPA",
            "COPB1",
            "CLTC",
            "DNM2",
            "SEC22B",
            "STX6",
            "VAMP7",
            "VPS11",
            "VPS16",
            "VPS18",
            "VPS33A",
            "VPS35",
            "SNX1",
            "SNX2",
            "TSG101",
            "CHMP2A",
            "CHMP4B",
        ),
    ),
    OperatorGeneSet(
        key="I_neural_identity",
        variables="I,R",
        role="postmitotic neuron identity context for interpreting maintenance recurrence",
        exact=(
            "MAP2",
            "RBFOX3",
            "TUBB3",
            "DCX",
            "SYN1",
            "SYP",
            "SNAP25",
            "STXBP1",
            "NEFL",
            "NEFM",
            "NEFH",
            "GRIN1",
            "GRIA1",
            "GAD1",
            "GAD2",
            "SLC17A7",
            "DLG4",
            "NCAM1",
            "CAMK2A",
        ),
    ),
    OperatorGeneSet(
        key="S_glia_support_context",
        variables="S,Q,R",
        role="glial/support context control; expected to be weak in neuron monoculture data",
        exact=(
            "APOE",
            "CLU",
            "GFAP",
            "AQP4",
            "S100B",
            "ALDH1L1",
            "CX3CR1",
            "TMEM119",
            "CSF1R",
            "TREM2",
            "C1QA",
            "C1QB",
            "C1QC",
        ),
    ),
)


def parse_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "" or value.upper() == "NA":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def load_rows(path: Path, min_base_mean: float) -> dict[str, GeneRow]:
    rows: dict[str, GeneRow] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            gene = (raw.get("Gene_symbol") or raw.get("gene") or "").strip().upper()
            if not gene:
                continue
            base_mean = parse_float(raw.get("baseMean"))
            log2fc = parse_float(raw.get("log2FoldChange"))
            pvalue = parse_float(raw.get("pvalue"))
            padj = parse_float(raw.get("padj"))
            if not math.isfinite(base_mean) or not math.isfinite(log2fc):
                continue
            if base_mean < min_base_mean:
                continue
            row = GeneRow(
                gene=gene,
                ensembl_id=(raw.get("") or raw.get("ensembl_id") or "").strip(),
                base_mean=base_mean,
                log2fc=log2fc,
                pvalue=pvalue,
                padj=padj,
            )
            existing = rows.get(gene)
            if existing is None or abs(row.log2fc) > abs(existing.log2fc):
                rows[gene] = row
    return rows


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def matches(row: GeneRow, gene_set: OperatorGeneSet) -> bool:
    if row.gene in gene_set.exact:
        return True
    return any(row.gene.startswith(prefix) for prefix in gene_set.prefixes)


def significant(row: GeneRow, padj_threshold: float, lfc_threshold: float) -> bool:
    return (
        math.isfinite(row.padj)
        and row.padj <= padj_threshold
        and abs(row.log2fc) >= lfc_threshold
    )


def summarize_gene_set(
    gene_set: OperatorGeneSet,
    rows: dict[str, GeneRow],
    top_genes: set[str],
    background_top_fraction: float,
    background_median_abs_lfc: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows.values() if matches(row, gene_set)),
        key=lambda row: (math.inf if not math.isfinite(row.padj) else row.padj, -abs(row.log2fc)),
    )
    abs_values = [abs(row.log2fc) for row in selected]
    sig_rows = [row for row in selected if significant(row, args.padj_threshold, args.lfc_threshold)]
    top_rows = [row for row in selected if row.gene in top_genes]
    ko_induced = [
        row
        for row in selected
        if math.isfinite(row.padj)
        and row.padj <= args.padj_threshold
        and row.log2fc <= -args.lfc_threshold
    ]
    wt_enriched = [
        row
        for row in selected
        if math.isfinite(row.padj)
        and row.padj <= args.padj_threshold
        and row.log2fc >= args.lfc_threshold
    ]

    observed = len(selected)
    top_fraction = len(top_rows) / observed if observed else 0.0
    enrichment = (
        top_fraction / background_top_fraction if background_top_fraction > 0.0 else math.nan
    )
    median_abs = median(abs_values)
    median_abs_ratio = (
        median_abs / background_median_abs_lfc if background_median_abs_lfc > 0.0 else math.nan
    )

    return {
        "key": gene_set.key,
        "variables": gene_set.variables,
        "role": gene_set.role,
        "observed_genes": observed,
        "significant_genes": len(sig_rows),
        "top_genes": len(top_rows),
        "top_fraction": round(top_fraction, 6),
        "top_enrichment": round(enrichment, 6) if math.isfinite(enrichment) else None,
        "median_abs_log2fc": round(median_abs, 6) if math.isfinite(median_abs) else None,
        "median_abs_ratio": round(median_abs_ratio, 6) if math.isfinite(median_abs_ratio) else None,
        "mean_log2fc": round(sum(row.log2fc for row in selected) / observed, 6)
        if observed
        else None,
        "ko_induced_sig_genes": len(ko_induced),
        "wt_enriched_sig_genes": len(wt_enriched),
        "examples": [
            {
                "gene": row.gene,
                "log2fc": round(row.log2fc, 6),
                "padj": row.padj,
                "baseMean": round(row.base_mean, 3),
            }
            for row in selected[: args.example_genes]
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data_path.resolve()
    if not data_path.exists():
        return {
            "gate": "clarus_cell_psapko_neuron_repair",
            "passed": False,
            "reason": "missing_data",
            "data_path": str(data_path),
            "expected_download": DATASET_DOWNLOAD,
        }

    rows = load_rows(data_path, args.min_base_mean)
    all_rows = list(rows.values())
    top_genes = {
        row.gene
        for row in all_rows
        if significant(row, args.padj_threshold, args.lfc_threshold)
    }
    background_top_fraction = len(top_genes) / len(all_rows) if all_rows else 0.0
    background_median_abs_lfc = median(abs(row.log2fc) for row in all_rows)

    target = rows.get("PSAP")
    target_control = {
        "observed": target is not None,
        "log2fc": round(target.log2fc, 6) if target else None,
        "padj": target.padj if target else None,
        "direction_ok_for_wt_vs_psapko": bool(
            target
            and math.isfinite(target.padj)
            and target.padj <= args.target_padj_threshold
            and target.log2fc >= args.target_lfc_threshold
        ),
    }

    summaries = [
        summarize_gene_set(
            gene_set,
            rows,
            top_genes,
            background_top_fraction,
            background_median_abs_lfc,
            args,
        )
        for gene_set in OPERATOR_GENE_SETS
    ]
    by_key = {summary["key"]: summary for summary in summaries}
    repair = by_key["Q_repair_lysosome_autophagy"]
    stress = by_key["D_damage_stress_response"]
    identity = by_key["I_neural_identity"]

    repair_signal = bool(
        repair["observed_genes"] >= args.min_operator_genes
        and repair["top_genes"] >= args.min_top_hits
        and (repair["top_enrichment"] or 0.0) >= args.min_top_enrichment
        and (repair["median_abs_ratio"] or 0.0) >= args.min_abs_ratio
    )
    stress_signal = bool(
        stress["observed_genes"] >= args.min_stress_genes
        and stress["top_genes"] >= args.min_stress_top_hits
        and (stress["median_abs_ratio"] or 0.0) >= args.min_abs_ratio
    )
    neural_context = bool(identity["observed_genes"] >= args.min_identity_genes)
    dq_signal = repair_signal or (repair["top_genes"] >= args.min_top_hits and stress_signal)
    passed = bool(target_control["direction_ok_for_wt_vs_psapko"] and neural_context and dq_signal)

    return {
        "gate": "clarus_cell_psapko_neuron_repair",
        "passed": passed,
        "claim_level": "empirical_pilot_DQ_branch" if passed else "parsed_no_promotion",
        "data_path": str(data_path),
        "dataset_url": DATASET_URL,
        "primary_paper": PRIMARY_PAPER,
        "contrast_note": (
            "File name is WT_vs_PSAPKO. Positive log2FC is treated as WT-enriched/"
            "PSAPKO-reduced; negative log2FC is treated as PSAPKO-induced."
        ),
        "thresholds": {
            "min_base_mean": args.min_base_mean,
            "padj_threshold": args.padj_threshold,
            "lfc_threshold": args.lfc_threshold,
            "min_operator_genes": args.min_operator_genes,
            "min_top_hits": args.min_top_hits,
            "min_top_enrichment": args.min_top_enrichment,
            "min_abs_ratio": args.min_abs_ratio,
        },
        "background": {
            "genes": len(all_rows),
            "top_genes": len(top_genes),
            "top_fraction": round(background_top_fraction, 6),
            "median_abs_log2fc": round(background_median_abs_lfc, 6)
            if math.isfinite(background_median_abs_lfc)
            else None,
        },
        "target_control": target_control,
        "decision": {
            "target_control_ok": bool(target_control["direction_ok_for_wt_vs_psapko"]),
            "neural_identity_context_ok": neural_context,
            "repair_signal_ok": repair_signal,
            "stress_signal_ok": stress_signal,
            "dq_signal_ok": dq_signal,
        },
        "operator_summaries": summaries,
    }


def fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        if value != 0 and abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.3f}"
    return str(value)


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if result.get("reason") == "missing_data":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell PSAP-KO neuron repair gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local data",
                    f"- expected local path: `{result['data_path']}`",
                    f"- download: <{result['expected_download']}>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    bg = result["background"]
    target = result["target_control"]
    decision = result["decision"]
    lines = [
        "# Clarus cell PSAP-KO neuron repair gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- dataset: [GSE152988]({result['dataset_url']})",
        f"- primary paper: [CRISPRbrain human neuron screen]({result['primary_paper']})",
        f"- local data: `{result['data_path']}`",
        f"- contrast note: {result['contrast_note']}",
        "",
        "## background",
        "",
        f"- genes after baseMean filter: `{bg['genes']}`",
        f"- significant/effect genes: `{bg['top_genes']}`",
        f"- background top fraction: `{fmt_float(bg['top_fraction'])}`",
        f"- background median |log2FC|: `{fmt_float(bg['median_abs_log2fc'])}`",
        "",
        "## target control",
        "",
        f"- PSAP observed: `{target['observed']}`",
        f"- PSAP log2FC: `{fmt_float(target['log2fc'])}`",
        f"- PSAP padj: `{fmt_float(target['padj'])}`",
        f"- direction/control ok: `{target['direction_ok_for_wt_vs_psapko']}`",
        "",
        "## decision",
        "",
        f"- target control ok: `{decision['target_control_ok']}`",
        f"- neural identity context ok: `{decision['neural_identity_context_ok']}`",
        f"- repair signal ok: `{decision['repair_signal_ok']}`",
        f"- stress signal ok: `{decision['stress_signal_ok']}`",
        f"- D/Q branch signal ok: `{decision['dq_signal_ok']}`",
        "",
        "## operator summaries",
        "",
        "| operator | vars | observed | top | enrichment | median |log2FC| ratio | KO-induced | WT-enriched |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in result["operator_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['key']}`",
                    f"`{summary['variables']}`",
                    str(summary["observed_genes"]),
                    str(summary["top_genes"]),
                    fmt_float(summary["top_enrichment"]),
                    fmt_float(summary["median_abs_ratio"]),
                    str(summary["ko_induced_sig_genes"]),
                    str(summary["wt_enriched_sig_genes"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## strongest mapped genes", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        if summary["examples"]:
            lines.append("| gene | log2FC | padj | baseMean |")
            lines.append("|---|---:|---:|---:|")
            for row in summary["examples"]:
                lines.append(
                    f"| `{row['gene']}` | {fmt_float(row['log2fc'])} | "
                    f"{fmt_float(row['padj'])} | {fmt_float(row['baseMean'])} |"
                )
        else:
            lines.append("- no mapped genes passed the baseMean filter")
        lines.append("")

    lines.extend(
        [
            "## claim boundary",
            "",
            (
                "This is a narrow empirical pilot for the postmitotic D/Q repair branch. "
                "It does not validate the full Clarus-cell loop, cell origin model, or "
                "human-brain mechanism by itself."
            ),
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--min-base-mean", type=float, default=10.0)
    parser.add_argument("--padj-threshold", type=float, default=0.05)
    parser.add_argument("--lfc-threshold", type=float, default=0.5)
    parser.add_argument("--target-padj-threshold", type=float, default=1e-6)
    parser.add_argument("--target-lfc-threshold", type=float, default=1.0)
    parser.add_argument("--min-operator-genes", type=int, default=10)
    parser.add_argument("--min-stress-genes", type=int, default=8)
    parser.add_argument("--min-identity-genes", type=int, default=6)
    parser.add_argument("--min-top-hits", type=int, default=3)
    parser.add_argument("--min-stress-top-hits", type=int, default=2)
    parser.add_argument("--min-top-enrichment", type=float, default=1.25)
    parser.add_argument("--min-abs-ratio", type=float, default=1.15)
    parser.add_argument("--example-genes", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
