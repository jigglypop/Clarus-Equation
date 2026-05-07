"""Empirical gate for the Clarus-cell glia/tissue support operator S.

Earlier gates tested intrinsic cell recurrence: proliferative survival/division
and postmitotic neuronal maintenance.  This gate tests the next boundary:

    S = support context

For neural Clarus cells, S means that neurons are maintained inside a tissue
field with glia-mediated survival, phagocytosis, inflammatory state control,
and extracellular support signals.  The gate combines two public CRISPRbrain
glial resources:

* Dräger et al. 2022: human iPSC-derived microglia CRISPRi/a screens
* Leng et al. 2022: human iPSC-derived astrocyte CRISPRi screens

The gate uses only public supplementary tables and includes a small XLSX reader
so it does not require openpyxl or pandas.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "evolution" / "clarus_cell"

MICROGLIA_XLSX = DATA_DIR / "41593_2022_1131_microglia_supp_tables.xlsx"
ASTRO_SCREEN_XLSX = DATA_DIR / "41593_2022_1180_astrocyte_screen_table2.xlsx"
ASTRO_CROPSEQ_XLSX = DATA_DIR / "41593_2022_1180_astrocyte_cropseq_table6.xlsx"

RESULT_JSON = Path(__file__).with_name("clarus_cell_glia_support_operator_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_glia_support_operator_report.md")

MICROGLIA_PAPER = "https://www.nature.com/articles/s41593-022-01131-4"
ASTROCYTE_PAPER = "https://www.nature.com/articles/s41593-022-01180-9"
MICROGLIA_TABLES_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-022-01131-4/"
    "MediaObjects/41593_2022_1131_MOESM4_ESM.xlsx"
)
ASTRO_SCREEN_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-022-01180-9/"
    "MediaObjects/41593_2022_1180_MOESM4_ESM.xlsx"
)
ASTRO_CROPSEQ_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-022-01180-9/"
    "MediaObjects/41593_2022_1180_MOESM8_ESM.xlsx"
)

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


MICROGLIA_SUPPORT_GENES = {
    "AIF1",
    "APOE",
    "AXL",
    "C1QA",
    "C1QB",
    "C1QC",
    "CD33",
    "CD38",
    "CD68",
    "CD74",
    "CDK8",
    "CDK12",
    "CSF1R",
    "CSF2RA",
    "CSF2RB",
    "CX3CR1",
    "GAS6",
    "GPR34",
    "INPP5D",
    "ITGAM",
    "LST1",
    "MAPK14",
    "MED1",
    "MERTK",
    "P2RY12",
    "P2RY13",
    "PFN1",
    "PFN2",
    "PICALM",
    "SPP1",
    "SPI1",
    "TGFBR2",
    "TREM2",
    "TYROBP",
}

ASTROCYTE_SUPPORT_REGULATORS = {
    "CEBPB",
    "CEBPD",
    "CHUK",
    "ETS1",
    "FOXC2",
    "IKBKG",
    "IRAK4",
    "IRF1",
    "IRF9",
    "JUN",
    "MAP3K7",
    "MTOR",
    "NFKB1",
    "NFKB2",
    "PDPK1",
    "PIK3R1",
    "PTPRA",
    "RELA",
    "RIPK1",
    "STAT1",
    "STAT2",
    "STAT3",
}

ASTROCYTE_OUTPUT_GENES = {
    "APOE",
    "C3",
    "CCL2",
    "CCL20",
    "CLU",
    "CSF1",
    "CXCL1",
    "CXCL8",
    "CXCL10",
    "ICAM1",
    "IFIT1",
    "IFIT3",
    "IL6",
    "IL32",
    "LCN2",
    "SAA1",
    "SOD2",
    "STAT1",
    "TGFB2",
    "VCAM1",
}


@dataclass(frozen=True)
class XlsxBook:
    path: Path
    archive: zipfile.ZipFile
    shared_strings: list[str]
    sheet_targets: dict[str, str]

    def close(self) -> None:
        self.archive.close()


def column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Za-z]", "", cell_ref)
    index = 0
    for letter in letters.upper():
        index = index * 26 + ord(letter) - 64
    return max(index - 1, 0)


def open_xlsx(path: Path) -> XlsxBook:
    archive = zipfile.ZipFile(path)
    shared_strings: list[str] = []
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in root.findall("a:si", NS):
            shared_strings.append(
                "".join(
                    text.text or ""
                    for text in item.iter(
                        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                    )
                )
            )
    except KeyError:
        pass

    rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet_targets: dict[str, str] = {}
    for sheet in workbook.findall("a:sheets/a:sheet", NS):
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rels[rel_id]
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheet_targets[sheet.attrib["name"]] = target
    return XlsxBook(path, archive, shared_strings, sheet_targets)


def cell_value(cell: ET.Element, book: XlsxBook) -> str:
    if cell.attrib.get("t") == "inlineStr":
        inline = cell.find("a:is", NS)
        if inline is None:
            return ""
        return "".join(
            text.text or ""
            for text in inline.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        )
    value = cell.find("a:v", NS)
    if value is None:
        return ""
    raw = value.text or ""
    if cell.attrib.get("t") == "s" and raw:
        return book.shared_strings[int(raw)]
    return raw


def read_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    book = open_xlsx(path)
    try:
        target = book.sheet_targets[sheet_name]
        root = ET.fromstring(book.archive.read(target))
        rows: list[list[str]] = []
        for row in root.findall("a:sheetData/a:row", NS):
            values: list[str] = []
            for cell in row.findall("a:c", NS):
                index = column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                values[index] = cell_value(cell, book)
            rows.append(values)
        return rows
    finally:
        book.close()


def sheet_names(path: Path) -> list[str]:
    book = open_xlsx(path)
    try:
        return list(book.sheet_targets)
    finally:
        book.close()


def parse_float(value: str | None) -> float:
    if value is None or value == "" or value.upper() == "NA":
        return math.nan
    try:
        result = float(value)
    except ValueError:
        return math.nan
    return result if math.isfinite(result) else math.nan


def phenotype_hit(phenotype: str, pvalue: str, product: str, args: argparse.Namespace) -> bool:
    return (
        parse_float(pvalue) <= args.microglia_pvalue_threshold
        and abs(parse_float(product)) >= args.microglia_product_threshold
        and math.isfinite(parse_float(phenotype))
    )


def analyze_microglia_primary(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_sheet(args.microglia_xlsx, "Supplementary Table 3")
    phenotype_columns = (
        ("survival", 2, 3, 4),
        ("activation", 5, 6, 7),
        ("phagocytosis_crispri", 8, 9, 10),
        ("phagocytosis_crispra", 11, 12, 13),
    )
    genes: dict[str, dict[str, Any]] = {}
    phenotype_hit_counts = {name: 0 for name, _, _, _ in phenotype_columns}
    for row in rows[2:]:
        if len(row) < 14:
            continue
        gene = row[1].strip().upper()
        if gene not in MICROGLIA_SUPPORT_GENES:
            continue
        hits = []
        for name, phenotype_i, pvalue_i, product_i in phenotype_columns:
            if phenotype_hit(row[phenotype_i], row[pvalue_i], row[product_i], args):
                phenotype_hit_counts[name] += 1
                hits.append(
                    {
                        "phenotype": name,
                        "score": round(parse_float(row[phenotype_i]), 6),
                        "pvalue": parse_float(row[pvalue_i]),
                        "gene_score": round(parse_float(row[product_i]), 6),
                    }
                )
        if hits:
            genes[gene] = {"gene": gene, "hits": hits}
    classes_hit = sum(1 for value in phenotype_hit_counts.values() if value > 0)
    return {
        "observed_support_hits": len(genes),
        "phenotype_hit_counts": phenotype_hit_counts,
        "phenotype_classes_hit": classes_hit,
        "passed": len(genes) >= args.min_microglia_primary_genes
        and classes_hit >= args.min_microglia_primary_classes,
        "examples": list(genes.values())[: args.example_genes],
    }


def analyze_microglia_state_shift(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_sheet(args.microglia_xlsx, "Supplementary Table 8")
    header = [cell.strip().upper() for cell in rows[0]]
    shifts = []
    for gene in sorted(MICROGLIA_SUPPORT_GENES):
        if gene not in header:
            continue
        index = header.index(gene)
        values = [parse_float(row[index]) for row in rows[1:] if index < len(row)]
        clean = [value for value in values if math.isfinite(value)]
        if not clean:
            continue
        max_abs = max(abs(value) for value in clean)
        total_abs = sum(abs(value) for value in clean)
        if max_abs >= args.microglia_cluster_shift_threshold:
            shifts.append(
                {
                    "gene": gene,
                    "max_abs_shift": round(max_abs, 6),
                    "total_abs_shift": round(total_abs, 6),
                }
            )
    shifts.sort(key=lambda row: (-row["max_abs_shift"], row["gene"]))
    return {
        "state_shift_genes": len(shifts),
        "passed": len(shifts) >= args.min_microglia_state_shift_genes,
        "examples": shifts[: args.example_genes],
    }


def analyze_astrocyte_screen(args: argparse.Namespace) -> dict[str, Any]:
    sheets = ("H1_combined", "TFs_combined")
    phenotype_groups = (
        ("phagocytosis_vehicle", 2, 3, 5),
        ("phagocytosis_inflammatory", 6, 7, 9),
        ("vcam1_inflammatory", 10, 11, 13),
    )
    genes: dict[str, dict[str, Any]] = {}
    phenotype_hit_counts = {name: 0 for name, _, _, _ in phenotype_groups}
    for sheet in sheets:
        rows = read_sheet(args.astro_screen_xlsx, sheet)
        for row in rows[1:]:
            if len(row) < 14:
                continue
            gene = row[0].strip().upper()
            if gene not in ASTROCYTE_SUPPORT_REGULATORS:
                continue
            entry = genes.setdefault(gene, {"gene": gene, "hits": []})
            for name, score_i, pvalue_i, status_i in phenotype_groups:
                is_hit = row[status_i].strip().lower() == "hit" or (
                    parse_float(row[pvalue_i]) <= args.astro_screen_pvalue_threshold
                    and abs(parse_float(row[score_i])) >= args.astro_screen_score_threshold
                )
                if is_hit:
                    phenotype_hit_counts[name] += 1
                    entry["hits"].append(
                        {
                            "phenotype": name,
                            "score": round(parse_float(row[score_i]), 6),
                            "pvalue": parse_float(row[pvalue_i]),
                            "status": row[status_i],
                        }
                    )
    genes = {gene: entry for gene, entry in genes.items() if entry["hits"]}
    classes_hit = sum(1 for value in phenotype_hit_counts.values() if value > 0)
    examples = sorted(
        genes.values(),
        key=lambda entry: (-len(entry["hits"]), entry["gene"]),
    )[: args.example_genes]
    return {
        "observed_regulator_hits": len(genes),
        "phenotype_hit_counts": phenotype_hit_counts,
        "phenotype_classes_hit": classes_hit,
        "passed": len(genes) >= args.min_astro_screen_regulators
        and classes_hit >= args.min_astro_screen_classes,
        "examples": examples,
    }


def analyze_astrocyte_cropseq(args: argparse.Namespace) -> dict[str, Any]:
    available_sheets = set(sheet_names(args.astro_cropseq_xlsx))
    regulators = sorted(ASTROCYTE_SUPPORT_REGULATORS & available_sheets)
    regulator_hits = []
    output_coverage: dict[str, int] = {gene: 0 for gene in ASTROCYTE_OUTPUT_GENES}
    for regulator in regulators:
        rows = read_sheet(args.astro_cropseq_xlsx, regulator)
        hits = []
        for row in rows[1:]:
            if len(row) < 5:
                continue
            gene = row[0].strip().upper()
            if gene not in ASTROCYTE_OUTPUT_GENES:
                continue
            log_fc = parse_float(row[1])
            padj = parse_float(row[4])
            if padj <= args.astro_cropseq_padj_threshold and abs(log_fc) >= args.astro_cropseq_lfc_threshold:
                hits.append({"gene": gene, "logFC": round(log_fc, 6), "padj": padj})
                output_coverage[gene] += 1
        if len(hits) >= args.min_astro_cropseq_outputs_per_regulator:
            hits.sort(key=lambda row: (row["padj"], -abs(row["logFC"])))
            regulator_hits.append(
                {
                    "regulator": regulator,
                    "support_output_hits": len(hits),
                    "examples": hits[:5],
                }
            )
    regulator_hits.sort(key=lambda row: (-row["support_output_hits"], row["regulator"]))
    covered_outputs = sum(1 for count in output_coverage.values() if count > 0)
    return {
        "regulators_with_support_output_hits": len(regulator_hits),
        "covered_support_outputs": covered_outputs,
        "passed": len(regulator_hits) >= args.min_astro_cropseq_regulators
        and covered_outputs >= args.min_astro_cropseq_outputs_total,
        "output_coverage": output_coverage,
        "examples": regulator_hits[: args.example_genes],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        str(path)
        for path in (args.microglia_xlsx, args.astro_screen_xlsx, args.astro_cropseq_xlsx)
        if not path.exists()
    ]
    if missing:
        return {
            "gate": "clarus_cell_glia_support_operator",
            "passed": False,
            "reason": "missing_data",
            "missing": missing,
            "downloads": {
                "microglia_tables": MICROGLIA_TABLES_URL,
                "astrocyte_screen": ASTRO_SCREEN_URL,
                "astrocyte_cropseq": ASTRO_CROPSEQ_URL,
            },
        }

    microglia_primary = analyze_microglia_primary(args)
    microglia_state_shift = analyze_microglia_state_shift(args)
    astrocyte_screen = analyze_astrocyte_screen(args)
    astrocyte_cropseq = analyze_astrocyte_cropseq(args)
    microglia_ok = microglia_primary["passed"] and microglia_state_shift["passed"]
    astrocyte_ok = astrocyte_screen["passed"] and astrocyte_cropseq["passed"]
    passed = bool(microglia_ok and astrocyte_ok)
    return {
        "gate": "clarus_cell_glia_support_operator",
        "passed": passed,
        "claim_level": "empirical_glia_support_context_branch"
        if passed
        else "parsed_no_promotion",
        "operators_supported": ("S", "D", "Q", "U", "R"),
        "primary_sources": {
            "microglia": MICROGLIA_PAPER,
            "astrocyte": ASTROCYTE_PAPER,
        },
        "local_data": {
            "microglia_xlsx": str(args.microglia_xlsx.resolve()),
            "astro_screen_xlsx": str(args.astro_screen_xlsx.resolve()),
            "astro_cropseq_xlsx": str(args.astro_cropseq_xlsx.resolve()),
        },
        "thresholds": {
            "microglia_pvalue_threshold": args.microglia_pvalue_threshold,
            "microglia_product_threshold": args.microglia_product_threshold,
            "microglia_cluster_shift_threshold": args.microglia_cluster_shift_threshold,
            "astro_screen_pvalue_threshold": args.astro_screen_pvalue_threshold,
            "astro_screen_score_threshold": args.astro_screen_score_threshold,
            "astro_cropseq_padj_threshold": args.astro_cropseq_padj_threshold,
            "astro_cropseq_lfc_threshold": args.astro_cropseq_lfc_threshold,
        },
        "decision": {
            "microglia_primary_ok": microglia_primary["passed"],
            "microglia_state_shift_ok": microglia_state_shift["passed"],
            "astrocyte_screen_ok": astrocyte_screen["passed"],
            "astrocyte_cropseq_ok": astrocyte_cropseq["passed"],
            "microglia_branch_ok": microglia_ok,
            "astrocyte_branch_ok": astrocyte_ok,
        },
        "microglia_primary": microglia_primary,
        "microglia_state_shift": microglia_state_shift,
        "astrocyte_screen": astrocyte_screen,
        "astrocyte_cropseq": astrocyte_cropseq,
        "claim_boundary": (
            "This supports the neural Clarus-cell S operator as glia-mediated "
            "support context. It does not prove in-vivo whole-brain closure or "
            "complete neuron-glia circuit dynamics."
        ),
    }


def fmt(value: Any) -> str:
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
                    "# Clarus cell glia support operator gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local data",
                    f"- missing: `{','.join(result['missing'])}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    lines = [
        "# Clarus cell glia support operator gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- operators supported: `{','.join(result['operators_supported'])}`",
        f"- microglia source: [Drager et al. 2022]({result['primary_sources']['microglia']})",
        f"- astrocyte source: [Leng et al. 2022]({result['primary_sources']['astrocyte']})",
        "",
        "## decision",
        "",
    ]
    for key, value in result["decision"].items():
        lines.append(f"- `{key}`: `{value}`")

    micro_primary = result["microglia_primary"]
    micro_shift = result["microglia_state_shift"]
    astro_screen = result["astrocyte_screen"]
    astro_crop = result["astrocyte_cropseq"]
    lines.extend(
        [
            "",
            "## summary",
            "",
            "| branch | metric | value |",
            "|---|---|---:|",
            f"| microglia primary screens | support genes with hits | {micro_primary['observed_support_hits']} |",
            f"| microglia primary screens | phenotype classes hit | {micro_primary['phenotype_classes_hit']} |",
            f"| microglia CROP-seq | support genes shifting states | {micro_shift['state_shift_genes']} |",
            f"| astrocyte screens | regulators with hits | {astro_screen['observed_regulator_hits']} |",
            f"| astrocyte screens | phenotype classes hit | {astro_screen['phenotype_classes_hit']} |",
            f"| astrocyte CROP-seq | regulators changing support outputs | {astro_crop['regulators_with_support_output_hits']} |",
            f"| astrocyte CROP-seq | covered support output genes | {astro_crop['covered_support_outputs']} |",
            "",
            "## microglia phenotype hits",
            "",
        ]
    )
    for entry in micro_primary["examples"]:
        hits = ", ".join(f"{hit['phenotype']}:{fmt(hit['score'])}" for hit in entry["hits"])
        lines.append(f"- `{entry['gene']}`: {hits}")

    lines.extend(["", "## microglia state-shift examples", ""])
    for entry in micro_shift["examples"]:
        lines.append(
            f"- `{entry['gene']}`: max abs shift `{fmt(entry['max_abs_shift'])}`, "
            f"total abs shift `{fmt(entry['total_abs_shift'])}`"
        )

    lines.extend(["", "## astrocyte screen hits", ""])
    for entry in astro_screen["examples"]:
        hits = ", ".join(f"{hit['phenotype']}:{fmt(hit['score'])}" for hit in entry["hits"])
        lines.append(f"- `{entry['gene']}`: {hits}")

    lines.extend(["", "## astrocyte CROP-seq support-output hits", ""])
    for entry in astro_crop["examples"]:
        examples = ", ".join(
            f"{hit['gene']}:{fmt(hit['logFC'])}" for hit in entry["examples"]
        )
        lines.append(
            f"- `{entry['regulator']}`: `{entry['support_output_hits']}` outputs; {examples}"
        )

    lines.extend(
        [
            "",
            "## claim boundary",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microglia-xlsx", type=Path, default=MICROGLIA_XLSX)
    parser.add_argument("--astro-screen-xlsx", type=Path, default=ASTRO_SCREEN_XLSX)
    parser.add_argument("--astro-cropseq-xlsx", type=Path, default=ASTRO_CROPSEQ_XLSX)
    parser.add_argument("--microglia-pvalue-threshold", type=float, default=0.10)
    parser.add_argument("--microglia-product-threshold", type=float, default=0.05)
    parser.add_argument("--microglia-cluster-shift-threshold", type=float, default=0.50)
    parser.add_argument("--astro-screen-pvalue-threshold", type=float, default=0.01)
    parser.add_argument("--astro-screen-score-threshold", type=float, default=0.25)
    parser.add_argument("--astro-cropseq-padj-threshold", type=float, default=0.05)
    parser.add_argument("--astro-cropseq-lfc-threshold", type=float, default=0.50)
    parser.add_argument("--min-microglia-primary-genes", type=int, default=8)
    parser.add_argument("--min-microglia-primary-classes", type=int, default=3)
    parser.add_argument("--min-microglia-state-shift-genes", type=int, default=5)
    parser.add_argument("--min-astro-screen-regulators", type=int, default=10)
    parser.add_argument("--min-astro-screen-classes", type=int, default=2)
    parser.add_argument("--min-astro-cropseq-regulators", type=int, default=8)
    parser.add_argument("--min-astro-cropseq-outputs-per-regulator", type=int, default=3)
    parser.add_argument("--min-astro-cropseq-outputs-total", type=int, default=8)
    parser.add_argument("--example-genes", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
