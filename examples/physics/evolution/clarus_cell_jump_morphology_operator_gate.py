"""JUMP Cell Painting morphology gate for Clarus-cell operators.

This gate uses the assembled JUMP CRISPR Cell Painting PCA-corrected profile
matrix.  Each well profile is summarized by its RMS distance in the corrected
feature space.  Negative controls define the background morphology radius; gene
perturbations whose median profile radius exceeds that background are treated
as image-based morphology-active.

This is a morphology/operator activity gate, not a recurrence proof.  The PCA
profiles do not preserve direct channel-level interpretability, so a pass here
only says that operator gene perturbations produce measurable morphology state
changes in the public JUMP profile space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "evolution" / "clarus_cell"
DEFAULT_PROFILE = DATA_DIR / "jump_crispr_profiles_pca_corrected.parquet"
DEFAULT_SUBSET_CSV = DATA_DIR / "jump_crispr_clarus_operator_morphology_subset.csv"
RESULT_JSON = Path(__file__).with_name("clarus_cell_jump_morphology_operator_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_jump_morphology_operator_report.md")

JUMP_DATASETS = "https://github.com/jump-cellpainting/datasets"
JUMP_PROFILE_INDEX = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/"
    "manifests/profile_index.json"
)
JUMP_CRISPR_PROFILE_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/"
    "source_all/workspace/profiles_assembled/CRISPR/v1.0a/"
    "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_"
    "PCA_corrected.parquet"
)

OPERATORS = ("B", "U", "E", "A", "I", "D", "Q", "S", "R")


@dataclass(frozen=True)
class OperatorClass:
    key: str
    variables: str
    role: str
    genes: tuple[str, ...]
    min_observed: int
    min_active_genes: int


OPERATOR_CLASSES = (
    OperatorClass(
        key="B_boundary_morphology",
        variables="B",
        role="membrane, vesicle boundary, and compartment-surface morphology",
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
        min_observed=5,
        min_active_genes=3,
    ),
    OperatorClass(
        key="U_traffic_morphology",
        variables="U",
        role="endosome, ER/Golgi, vesicle routing, and traffic morphology",
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
        min_observed=5,
        min_active_genes=2,
    ),
    OperatorClass(
        key="E_energy_mitochondria_morphology",
        variables="E",
        role="mitochondrial energy machinery morphology response",
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
        min_observed=10,
        min_active_genes=2,
    ),
    OperatorClass(
        key="A_metabolic_core_morphology",
        variables="A",
        role="central carbon, nucleotide, lipid, and biosynthetic morphology response",
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
        min_observed=10,
        min_active_genes=3,
    ),
    OperatorClass(
        key="I_identity_template_morphology",
        variables="I",
        role="DNA replication, transcription, chromatin, and template-state morphology",
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
        min_observed=12,
        min_active_genes=6,
    ),
    OperatorClass(
        key="D_Q_repair_quality_morphology",
        variables="D,Q",
        role="DNA repair, proteostasis, autophagy, lysosome, and damage-control morphology",
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
        min_observed=12,
        min_active_genes=4,
    ),
)


def optional_dependencies() -> tuple[Any, Any, Callable[..., Any], str | None]:
    try:
        import numpy as np  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        from broad_babel.query import run_query  # type: ignore
    except ImportError as exc:
        return None, None, None, str(exc)
    return np, pq, run_query, None


def chunks(values: list[str], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def map_jcp_ids(run_query: Callable[..., Any], ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    standard_by_jcp: dict[str, str] = {}
    pert_type_by_jcp: dict[str, str] = {}
    for chunk in chunks(ids, 500):
        rows = run_query(
            chunk,
            input_column="JCP2022",
            output_columns="JCP2022,standard_key,pert_type",
        )
        for jcp_id, standard_key, pert_type in rows:
            if jcp_id is None:
                continue
            standard_by_jcp[str(jcp_id)] = "" if standard_key is None else str(standard_key)
            pert_type_by_jcp[str(jcp_id)] = "" if pert_type is None else str(pert_type)
    return standard_by_jcp, pert_type_by_jcp


def clean_gene(value: str | None) -> str:
    if not value:
        return ""
    gene = value.strip().upper()
    if not gene or gene in {"NO-GUIDE", "NEGCON"}:
        return ""
    return gene


def quantile(np: Any, values: list[float] | Any, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=float), q))


def median(np: Any, values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return float(np.median(np.asarray(clean, dtype=float)))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def distribution(np: Any, values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median": round(quantile(np, values, 0.50), 6),
        "q75": round(quantile(np, values, 0.75), 6),
        "q90": round(quantile(np, values, 0.90), 6),
        "q95": round(quantile(np, values, 0.95), 6),
        "q99": round(quantile(np, values, 0.99), 6),
    }


def load_profile_rms(np: Any, pq: Any, path: Path) -> tuple[list[str], Any, list[str]]:
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    feature_cols = [name for name in names if name.startswith("X_")]
    table = pq.read_table(path, columns=["Metadata_JCP2022", *feature_cols])
    jcp_ids = [str(value) for value in table["Metadata_JCP2022"].to_pylist()]
    rows = len(jcp_ids)
    sum_sq = np.zeros(rows, dtype=float)
    counts = np.zeros(rows, dtype=float)
    for column in feature_cols:
        arr = np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=float)
        mask = np.isfinite(arr)
        sum_sq[mask] += arr[mask] * arr[mask]
        counts[mask] += 1.0
    rms = np.full(rows, np.nan, dtype=float)
    valid = counts > 0
    rms[valid] = np.sqrt(sum_sq[valid] / counts[valid])
    return jcp_ids, rms, feature_cols


def collect_gene_profiles(
    jcp_ids: list[str],
    rms: Any,
    standard_by_jcp: dict[str, str],
    pert_type_by_jcp: dict[str, str],
) -> dict[str, Any]:
    negcon: list[float] = []
    trt: list[float] = []
    other: list[float] = []
    gene_profiles: dict[str, list[float]] = {}
    mapped_rows = 0
    unmapped_rows = 0
    for jcp_id, value in zip(jcp_ids, rms):
        if not math.isfinite(float(value)):
            continue
        pert_type = pert_type_by_jcp.get(jcp_id)
        if pert_type is None:
            unmapped_rows += 1
            continue
        mapped_rows += 1
        score = float(value)
        if pert_type == "negcon":
            negcon.append(score)
        elif pert_type == "trt":
            trt.append(score)
            gene = clean_gene(standard_by_jcp.get(jcp_id))
            if gene:
                gene_profiles.setdefault(gene, []).append(score)
        else:
            other.append(score)
    return {
        "negcon": negcon,
        "trt": trt,
        "other": other,
        "gene_profiles": gene_profiles,
        "mapped_rows": mapped_rows,
        "unmapped_rows": unmapped_rows,
    }


def summarize_gene(
    np: Any,
    gene: str,
    values: list[float],
    active_threshold: float,
    active_fraction_threshold: float,
) -> dict[str, Any]:
    active_values = [value for value in values if value > active_threshold]
    median_rms = median(np, values)
    active_fraction = len(active_values) / len(values) if values else 0.0
    active = bool(median_rms > active_threshold or active_fraction >= active_fraction_threshold)
    return {
        "gene": gene,
        "profiles": len(values),
        "median_rms": round(median_rms, 6),
        "mean_rms": round(float(np.mean(np.asarray(values, dtype=float))), 6) if values else math.nan,
        "q90_rms": round(quantile(np, values, 0.90), 6),
        "active_fraction": round(active_fraction, 6),
        "active": active,
    }


def summarize_operator(
    np: Any,
    operator: OperatorClass,
    gene_profiles: dict[str, list[float]],
    active_threshold: float,
    active_fraction_threshold: float,
    example_genes: int,
) -> dict[str, Any]:
    selected = [
        summarize_gene(np, gene, gene_profiles[gene], active_threshold, active_fraction_threshold)
        for gene in operator.genes
        if gene in gene_profiles
    ]
    selected.sort(key=lambda row: (-row["median_rms"], -row["active_fraction"], row["gene"]))
    active = [row for row in selected if row["active"]]
    median_gene_median = median(np, [row["median_rms"] for row in selected])
    median_active_fraction = median(np, [row["active_fraction"] for row in selected])
    criteria = {
        "observed_ok": len(selected) >= operator.min_observed,
        "active_gene_count_ok": len(active) >= operator.min_active_genes,
    }
    return {
        "key": operator.key,
        "variables": operator.variables,
        "role": operator.role,
        "candidate_genes": len(operator.genes),
        "observed_genes": len(selected),
        "active_genes": len(active),
        "median_gene_median_rms": round(median_gene_median, 6),
        "median_active_fraction": round(median_active_fraction, 6),
        "thresholds": {
            "min_observed": operator.min_observed,
            "min_active_genes": operator.min_active_genes,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "examples": selected[:example_genes],
        "gene_summaries": selected,
    }


def write_subset_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "operator",
                "variables",
                "gene",
                "profiles",
                "median_rms",
                "mean_rms",
                "q90_rms",
                "active_fraction",
                "active",
                "negative_control_q95",
            ),
        )
        writer.writeheader()
        threshold = result["active_threshold"]
        for summary in result.get("operator_summaries", []):
            for row in summary["gene_summaries"]:
                writer.writerow(
                    {
                        "operator": summary["key"],
                        "variables": summary["variables"],
                        "gene": row["gene"],
                        "profiles": row["profiles"],
                        "median_rms": row["median_rms"],
                        "mean_rms": row["mean_rms"],
                        "q90_rms": row["q90_rms"],
                        "active_fraction": row["active_fraction"],
                        "active": row["active"],
                        "negative_control_q95": threshold,
                    }
                )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = args.profile_parquet.resolve()
    if not profile_path.exists():
        return {
            "gate": "clarus_cell_jump_morphology_operator",
            "passed": False,
            "reason": "missing_data",
            "profile_path": str(profile_path),
            "download": JUMP_CRISPR_PROFILE_URL,
            "profile_index": JUMP_PROFILE_INDEX,
        }

    np, pq, run_query, missing = optional_dependencies()
    if missing:
        return {
            "gate": "clarus_cell_jump_morphology_operator",
            "passed": False,
            "reason": "missing_dependency",
            "missing": missing,
            "install_command": ".venv\\Scripts\\python.exe -m pip install pyarrow broad-babel",
            "profile_path": str(profile_path),
        }

    jcp_ids, rms, feature_cols = load_profile_rms(np, pq, profile_path)
    unique_ids = sorted(set(jcp_ids))
    standard_by_jcp, pert_type_by_jcp = map_jcp_ids(run_query, unique_ids)
    collected = collect_gene_profiles(jcp_ids, rms, standard_by_jcp, pert_type_by_jcp)
    negcon = collected["negcon"]
    trt = collected["trt"]
    gene_profiles = collected["gene_profiles"]
    negcon_stats = distribution(np, negcon)
    trt_stats = distribution(np, trt)
    active_threshold = quantile(np, negcon, args.control_quantile)

    morphology_control_ok = bool(
        len(jcp_ids) >= args.min_rows
        and len(feature_cols) >= args.min_features
        and negcon_stats["n"] >= args.min_negcon_profiles
        and trt_stats["n"] >= args.min_trt_profiles
        and trt_stats["median"] > negcon_stats["median"]
        and trt_stats["q90"] > negcon_stats["q90"]
        and trt_stats["q95"] > negcon_stats["q95"]
    )

    operator_summaries = [
        summarize_operator(
            np,
            operator,
            gene_profiles,
            active_threshold,
            args.active_fraction_threshold,
            args.example_genes,
        )
        for operator in OPERATOR_CLASSES
    ]
    passed_operators = sum(1 for summary in operator_summaries if summary["passed"])
    boundary_or_traffic_ok = any(
        summary["passed"]
        for summary in operator_summaries
        if summary["key"] in {"B_boundary_morphology", "U_traffic_morphology"}
    )
    identity_or_quality_ok = any(
        summary["passed"]
        for summary in operator_summaries
        if summary["key"]
        in {"I_identity_template_morphology", "D_Q_repair_quality_morphology"}
    )
    passed = bool(
        morphology_control_ok
        and passed_operators >= args.min_passed_operators
        and boundary_or_traffic_ok
        and identity_or_quality_ok
    )
    supported: set[str] = set()
    for summary in operator_summaries:
        if summary["passed"]:
            supported.update(token.strip() for token in summary["variables"].split(","))

    result = {
        "gate": "clarus_cell_jump_morphology_operator",
        "passed": passed,
        "claim_level": "empirical_morphology_operator_activity_branch"
        if passed
        else "parsed_no_promotion",
        "jump_datasets": JUMP_DATASETS,
        "profile_index": JUMP_PROFILE_INDEX,
        "profile_url": JUMP_CRISPR_PROFILE_URL,
        "profile_path": str(profile_path),
        "subset_csv": str(args.subset_csv.resolve()),
        "source_note": (
            "JUMP CRISPR assembled well-position, cell-count, variance/MAD, outlier, "
            "feature-selected, sphered, Harmony, PCA-corrected profiles."
        ),
        "rows": len(jcp_ids),
        "features": len(feature_cols),
        "unique_jcp_ids": len(unique_ids),
        "mapped_jcp_ids": len(standard_by_jcp),
        "mapped_profile_rows": collected["mapped_rows"],
        "unmapped_profile_rows": collected["unmapped_rows"],
        "unique_treatment_genes": len(gene_profiles),
        "negative_control": negcon_stats,
        "treatment": trt_stats,
        "active_threshold": active_threshold,
        "active_threshold_quantile": args.control_quantile,
        "active_rule": (
            f"gene median RMS > negative-control q{int(args.control_quantile * 100)} "
            f"or active profile fraction >= {args.active_fraction_threshold}"
        ),
        "morphology_control_ok": morphology_control_ok,
        "passed_operators": passed_operators,
        "min_passed_operators": args.min_passed_operators,
        "boundary_or_traffic_ok": boundary_or_traffic_ok,
        "identity_or_quality_ok": identity_or_quality_ok,
        "operators_supported": tuple(operator for operator in OPERATORS if operator in supported),
        "operator_summaries": operator_summaries,
        "claim_boundary": (
            "This gate supports image-based morphology activity of shared human cell "
            "operators.  It does not prove channel-specific organelle causality, "
            "cell recurrence, primitive-cell origin, or the full human brain mechanism. "
            "The weak E result should be read as a limitation of this PCA morphology "
            "branch, not as evidence against DepMap/HPA mitochondrial support."
        ),
    }
    write_subset_csv(args.subset_csv.resolve(), result)
    return result


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("reason") == "missing_data":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell JUMP morphology operator gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local JUMP profile parquet",
                    f"- expected path: `{result['profile_path']}`",
                    f"- profile index: <{result['profile_index']}>",
                    f"- download: <{result['download']}>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    if result.get("reason") == "missing_dependency":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell JUMP morphology operator gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing optional dependency",
                    f"- missing: `{result['missing']}`",
                    f"- install command: `{result['install_command']}`",
                    f"- local profile: `{result['profile_path']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    negcon = result["negative_control"]
    trt = result["treatment"]
    failed = [
        summary["key"] for summary in result["operator_summaries"] if not summary["passed"]
    ]
    lines = [
        "# Clarus cell JUMP morphology operator gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- source: [JUMP Cell Painting datasets]({result['jump_datasets']})",
        f"- profile index: [v0.11.0 manifest]({result['profile_index']})",
        f"- local profile: `{result['profile_path']}`",
        f"- local subset: `{result['subset_csv']}`",
        f"- source note: {result['source_note']}",
        f"- rows/features: `{result['rows']}` / `{result['features']}`",
        f"- unique JCP ids: `{result['unique_jcp_ids']}`",
        f"- mapped JCP ids: `{result['mapped_jcp_ids']}`",
        f"- treatment genes: `{result['unique_treatment_genes']}`",
        f"- active threshold: negative-control q{int(result['active_threshold_quantile'] * 100)} = "
        f"`{fmt(result['active_threshold'])}`",
        f"- active rule: {result['active_rule']}",
        f"- morphology control ok: `{result['morphology_control_ok']}`",
        f"- passed operators: `{result['passed_operators']}/{len(result['operator_summaries'])}`",
        f"- operators supported: `{','.join(result['operators_supported'])}`",
        f"- failed or weak operators: `{','.join(failed) if failed else 'none'}`",
        "",
        "## profile controls",
        "",
        "| group | n | median RMS | q75 | q90 | q95 | q99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `negative_control` | {negcon['n']} | {fmt(negcon['median'])} | "
        f"{fmt(negcon['q75'])} | {fmt(negcon['q90'])} | {fmt(negcon['q95'])} | "
        f"{fmt(negcon['q99'])} |",
        f"| `treatment` | {trt['n']} | {fmt(trt['median'])} | {fmt(trt['q75'])} | "
        f"{fmt(trt['q90'])} | {fmt(trt['q95'])} | {fmt(trt['q99'])} |",
        "",
        "## operator summaries",
        "",
        "| operator | vars | observed | active | median gene RMS | median active frac | passed |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for summary in result["operator_summaries"]:
        lines.append(
            f"| `{summary['key']}` | `{summary['variables']}` | "
            f"{summary['observed_genes']} | {summary['active_genes']} | "
            f"{fmt(summary['median_gene_median_rms'])} | "
            f"{fmt(summary['median_active_fraction'])} | `{summary['passed']}` |"
        )

    lines.extend(["", "## strongest morphology-active genes", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        lines.append("| gene | profiles | median RMS | q90 RMS | active fraction | active |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in summary["examples"]:
            lines.append(
                f"| `{row['gene']}` | {row['profiles']} | {fmt(row['median_rms'])} | "
                f"{fmt(row['q90_rms'])} | {fmt(row['active_fraction'])} | "
                f"`{row['active']}` |"
            )
        lines.append("")

    lines.extend(["## claim boundary", "", result["claim_boundary"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-parquet", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--subset-csv", type=Path, default=DEFAULT_SUBSET_CSV)
    parser.add_argument("--control-quantile", type=float, default=0.95)
    parser.add_argument("--active-fraction-threshold", type=float, default=0.50)
    parser.add_argument("--min-rows", type=int, default=50_000)
    parser.add_argument("--min-features", type=int, default=200)
    parser.add_argument("--min-negcon-profiles", type=int, default=5_000)
    parser.add_argument("--min-trt-profiles", type=int, default=30_000)
    parser.add_argument("--min-passed-operators", type=int, default=4)
    parser.add_argument("--example-genes", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
