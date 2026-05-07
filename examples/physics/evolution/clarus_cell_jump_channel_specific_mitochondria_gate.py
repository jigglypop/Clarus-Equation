"""JUMP direct mitochondrial-channel gate for Clarus-cell E.

The previous JUMP morphology gate used PCA-corrected profiles and found broad
operator activity, but it did not promote the mitochondrial energy operator E.
This gate tests that bottleneck directly with interpretable CellProfiler
features whose names contain the Mito channel.

The decision rule is intentionally conservative:

* build a compact local subset of direct Mito Intensity/Granularity/
  RadialDistribution features from the public JUMP CRISPR interpretable parquet;
* robustly normalize every feature to negative-control median/MAD;
* summarize each well by robust-z RMS across direct Mito features;
* require E mitochondrial genes to exceed the negative-control q95 radius.

If I/DQ positive controls are active but E is not, the result is a useful
non-promotion rather than a failure of data parsing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "evolution" / "clarus_cell"
DEFAULT_SUBSET = DATA_DIR / "jump_crispr_mito_direct_features.parquet"
DEFAULT_GENE_SUMMARY = DATA_DIR / "jump_crispr_mito_direct_gene_summary.csv"
RESULT_JSON = Path(__file__).with_name(
    "clarus_cell_jump_channel_specific_mitochondria_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "clarus_cell_jump_channel_specific_mitochondria_report.md"
)

JUMP_DATASETS = "https://github.com/jump-cellpainting/datasets"
JUMP_PROFILE_INDEX = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/"
    "manifests/profile_index.json"
)
JUMP_CRISPR_INTERPRETABLE_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/"
    "source_all/workspace/profiles_assembled/CRISPR/v1.0a/"
    "profiles_wellpos_cc_var_mad_outlier.parquet"
)
DIRECT_MITO_MODULES = ("Intensity", "Granularity", "RadialDistribution")


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
        key="E_energy_mitochondria_direct_mito",
        variables="E",
        role="mitochondrial ATP/OXPHOS and energy-homeostasis genes",
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
        key="I_identity_template_direct_mito_control",
        variables="I",
        role="identity-template perturbations as positive morphology controls",
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
            "RFC1",
            "RFC2",
            "TOP2A",
        ),
        min_observed=8,
        min_active_genes=3,
    ),
    OperatorClass(
        key="D_Q_repair_quality_direct_mito_control",
        variables="D,Q",
        role="repair/proteostasis perturbations as positive morphology controls",
        genes=(
            "ATM",
            "ATR",
            "CHEK1",
            "RAD51",
            "BRCA1",
            "BARD1",
            "XRCC5",
            "XRCC6",
            "VCP",
            "PSMC1",
            "PSMD1",
            "PSMD2",
            "BECN1",
            "ATG3",
        ),
        min_observed=8,
        min_active_genes=3,
    ),
)


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, timeout: int) -> None:
        self.url = url
        self.timeout = timeout
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.calls = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.size - 1, self.position + size - 1)
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        self.position += len(data)
        self.calls += 1
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


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


def direct_mito_columns(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if "Mito" in name
        and any(f"_{module}_" in name for module in DIRECT_MITO_MODULES)
    ]


def fetch_subset(args: argparse.Namespace, pq: Any) -> dict[str, Any]:
    reader = HTTPRangeReader(args.source_url, args.http_timeout)
    parquet = pq.ParquetFile(reader)
    columns = direct_mito_columns(parquet.schema_arrow.names)
    table = parquet.read(columns=["Metadata_JCP2022", *columns])
    args.subset_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.subset_parquet, compression="zstd")
    return {
        "source_url": args.source_url,
        "source_size_bytes": reader.size,
        "range_calls": reader.calls,
        "range_bytes_read": reader.bytes_read,
        "subset_path": str(args.subset_parquet.resolve()),
        "rows": table.num_rows,
        "columns": table.num_columns,
        "direct_mito_features": len(columns),
    }


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
            standard_by_jcp[str(jcp_id)] = "" if standard_key is None else str(standard_key).upper()
            pert_type_by_jcp[str(jcp_id)] = "" if pert_type is None else str(pert_type)
    return standard_by_jcp, pert_type_by_jcp


def distribution(np: Any, values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": math.nan, "q75": math.nan, "q90": math.nan, "q95": math.nan}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": round(float(np.quantile(arr, 0.50)), 6),
        "q75": round(float(np.quantile(arr, 0.75)), 6),
        "q90": round(float(np.quantile(arr, 0.90)), 6),
        "q95": round(float(np.quantile(arr, 0.95)), 6),
        "q99": round(float(np.quantile(arr, 0.99)), 6),
    }


def median(np: Any, values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return float(np.median(np.asarray(clean, dtype=float)))


def robust_mito_rms(
    np: Any,
    table: Any,
    neg_mask: Any,
    feature_columns: list[str],
) -> Any:
    rows = table.num_rows
    sum_sq = np.zeros(rows, dtype=float)
    counts = np.zeros(rows, dtype=float)
    for column in feature_columns:
        values = np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=float)
        neg_values = values[neg_mask]
        finite_neg = neg_values[np.isfinite(neg_values)]
        if finite_neg.size:
            center = float(np.median(finite_neg))
            mad = float(np.median(np.abs(finite_neg - center)) * 1.4826)
            std = float(np.std(finite_neg))
        else:
            center = 0.0
            mad = 0.0
            std = 0.0
        scale = mad if mad > 1e-6 else std if std > 1e-6 else 1.0
        finite = np.isfinite(values)
        z = np.zeros(rows, dtype=float)
        z[finite] = (values[finite] - center) / scale
        sum_sq[finite] += z[finite] * z[finite]
        counts[finite] += 1.0
    rms = np.full(rows, np.nan, dtype=float)
    valid = counts > 0
    rms[valid] = np.sqrt(sum_sq[valid] / counts[valid])
    return rms


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
        "median_mito_rms": round(median_rms, 6),
        "q90_mito_rms": round(float(np.quantile(np.asarray(values, dtype=float), 0.90)), 6)
        if values
        else math.nan,
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
    selected.sort(key=lambda row: (-row["median_mito_rms"], -row["active_fraction"], row["gene"]))
    active = [row for row in selected if row["active"]]
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
        "median_gene_median_mito_rms": round(median(np, [row["median_mito_rms"] for row in selected]), 6),
        "median_active_fraction": round(median(np, [row["active_fraction"] for row in selected]), 6),
        "criteria": criteria,
        "passed": all(criteria.values()),
        "examples": selected[:example_genes],
        "gene_summaries": selected,
    }


def write_gene_summary(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "operator",
                "variables",
                "gene",
                "profiles",
                "median_mito_rms",
                "q90_mito_rms",
                "active_fraction",
                "active",
                "negative_control_q95",
            ),
        )
        writer.writeheader()
        for summary in result.get("operator_summaries", []):
            for row in summary["gene_summaries"]:
                writer.writerow(
                    {
                        "operator": summary["key"],
                        "variables": summary["variables"],
                        "gene": row["gene"],
                        "profiles": row["profiles"],
                        "median_mito_rms": row["median_mito_rms"],
                        "q90_mito_rms": row["q90_mito_rms"],
                        "active_fraction": row["active_fraction"],
                        "active": row["active"],
                        "negative_control_q95": result["active_threshold"],
                    }
                )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    np, pq, run_query, missing = optional_dependencies()
    if missing:
        return {
            "gate": "clarus_cell_jump_channel_specific_mitochondria",
            "passed": False,
            "reason": "missing_dependency",
            "missing": missing,
            "install_command": ".venv\\Scripts\\python.exe -m pip install pyarrow broad-babel",
        }

    fetch_info = None
    if args.fetch_subset or (args.auto_fetch_subset and not args.subset_parquet.exists()):
        fetch_info = fetch_subset(args, pq)

    if not args.subset_parquet.exists():
        return {
            "gate": "clarus_cell_jump_channel_specific_mitochondria",
            "passed": False,
            "reason": "missing_subset",
            "subset_path": str(args.subset_parquet),
            "source_url": args.source_url,
            "how_to_build": (
                ".venv\\Scripts\\python.exe examples\\physics\\evolution\\"
                "clarus_cell_jump_channel_specific_mitochondria_gate.py --fetch-subset"
            ),
        }

    table = pq.read_table(args.subset_parquet)
    feature_columns = [name for name in table.schema.names if name != "Metadata_JCP2022"]
    jcp_ids = [str(value) for value in table["Metadata_JCP2022"].to_pylist()]
    standard_by_jcp, pert_type_by_jcp = map_jcp_ids(run_query, sorted(set(jcp_ids)))
    neg_mask = np.asarray([pert_type_by_jcp.get(jcp_id) == "negcon" for jcp_id in jcp_ids], dtype=bool)
    trt_mask = np.asarray([pert_type_by_jcp.get(jcp_id) == "trt" for jcp_id in jcp_ids], dtype=bool)
    mito_rms = robust_mito_rms(np, table, neg_mask, feature_columns)

    neg = [float(value) for value in mito_rms[neg_mask] if math.isfinite(float(value))]
    trt = [float(value) for value in mito_rms[trt_mask] if math.isfinite(float(value))]
    neg_stats = distribution(np, neg)
    trt_stats = distribution(np, trt)
    active_threshold = float(np.quantile(np.asarray(neg, dtype=float), args.control_quantile))

    gene_profiles: dict[str, list[float]] = {}
    for jcp_id, value in zip(jcp_ids, mito_rms):
        if not math.isfinite(float(value)) or pert_type_by_jcp.get(jcp_id) != "trt":
            continue
        gene = standard_by_jcp.get(jcp_id, "")
        if gene and gene != "NO-GUIDE":
            gene_profiles.setdefault(gene, []).append(float(value))

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
    by_key = {summary["key"]: summary for summary in operator_summaries}
    e_summary = by_key["E_energy_mitochondria_direct_mito"]
    positive_control_ok = bool(
        by_key["I_identity_template_direct_mito_control"]["passed"]
        and by_key["D_Q_repair_quality_direct_mito_control"]["passed"]
    )
    data_ok = bool(
        table.num_rows >= args.min_rows
        and len(feature_columns) >= args.min_features
        and neg_stats["n"] >= args.min_negcon_profiles
        and trt_stats["n"] >= args.min_trt_profiles
    )
    e_promoted = bool(e_summary["passed"])
    passed = bool(data_ok and positive_control_ok and e_promoted)

    result = {
        "gate": "clarus_cell_jump_channel_specific_mitochondria",
        "passed": passed,
        "claim_level": "empirical_mitochondrial_E_channel_branch"
        if passed
        else "parsed_no_promotion",
        "specific_claim": "mitochondrial_E_promoted" if passed else "mitochondrial_E_not_promoted",
        "jump_datasets": JUMP_DATASETS,
        "profile_index": JUMP_PROFILE_INDEX,
        "source_url": args.source_url,
        "subset_path": str(args.subset_parquet.resolve()),
        "gene_summary_csv": str(args.gene_summary_csv.resolve()),
        "fetch_info": fetch_info,
        "rows": table.num_rows,
        "direct_mito_features": len(feature_columns),
        "direct_mito_modules": DIRECT_MITO_MODULES,
        "unique_jcp_ids": len(set(jcp_ids)),
        "mapped_jcp_ids": len(standard_by_jcp),
        "unique_treatment_genes": len(gene_profiles),
        "negative_control": neg_stats,
        "treatment": trt_stats,
        "active_threshold_quantile": args.control_quantile,
        "active_threshold": round(active_threshold, 6),
        "active_rule": (
            f"gene median direct-Mito robust-z RMS > negative-control q"
            f"{int(args.control_quantile * 100)} or active profile fraction >= "
            f"{args.active_fraction_threshold}"
        ),
        "data_ok": data_ok,
        "positive_control_ok": positive_control_ok,
        "e_promoted": e_promoted,
        "operator_summaries": operator_summaries,
        "claim_boundary": (
            "This gate tests only direct Mito-channel interpretable CellProfiler "
            "features under CRISPR perturbation. A non-promotion does not override "
            "DepMap fitness, HPA localization, or CRISPRbrain stress evidence for E; "
            "it marks direct JUMP Mito-channel morphology as an unresolved E branch."
        ),
    }
    write_gene_summary(args.gene_summary_csv.resolve(), result)
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
                    "# Clarus cell JUMP direct mitochondrial-channel gate",
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
    if result.get("reason") == "missing_subset":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell JUMP direct mitochondrial-channel gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local direct-Mito subset",
                    f"- expected path: `{result['subset_path']}`",
                    f"- source: <{result['source_url']}>",
                    f"- build command: `{result['how_to_build']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    neg = result["negative_control"]
    trt = result["treatment"]
    lines = [
        "# Clarus cell JUMP direct mitochondrial-channel gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- specific claim: `{result['specific_claim']}`",
        f"- source: [JUMP Cell Painting datasets]({result['jump_datasets']})",
        f"- profile index: [v0.11.0 manifest]({result['profile_index']})",
        f"- source parquet: `{result['source_url']}`",
        f"- local subset: `{result['subset_path']}`",
        f"- gene summary: `{result['gene_summary_csv']}`",
        f"- rows/direct Mito features: `{result['rows']}` / `{result['direct_mito_features']}`",
        f"- direct Mito modules: `{','.join(result['direct_mito_modules'])}`",
        f"- unique JCP ids: `{result['unique_jcp_ids']}`",
        f"- treatment genes: `{result['unique_treatment_genes']}`",
        f"- active threshold: negative-control q{int(result['active_threshold_quantile'] * 100)} = "
        f"`{fmt(result['active_threshold'])}`",
        f"- active rule: {result['active_rule']}",
        f"- data ok: `{result['data_ok']}`",
        f"- positive controls ok: `{result['positive_control_ok']}`",
        f"- E promoted: `{result['e_promoted']}`",
        "",
        "## profile controls",
        "",
        "| group | n | median direct-Mito RMS | q75 | q90 | q95 | q99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `negative_control` | {neg['n']} | {fmt(neg['median'])} | {fmt(neg['q75'])} | "
        f"{fmt(neg['q90'])} | {fmt(neg['q95'])} | {fmt(neg['q99'])} |",
        f"| `treatment` | {trt['n']} | {fmt(trt['median'])} | {fmt(trt['q75'])} | "
        f"{fmt(trt['q90'])} | {fmt(trt['q95'])} | {fmt(trt['q99'])} |",
        "",
        "## operator summaries",
        "",
        "| operator | vars | observed | active | median gene direct-Mito RMS | median active frac | passed |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for summary in result["operator_summaries"]:
        lines.append(
            f"| `{summary['key']}` | `{summary['variables']}` | "
            f"{summary['observed_genes']} | {summary['active_genes']} | "
            f"{fmt(summary['median_gene_median_mito_rms'])} | "
            f"{fmt(summary['median_active_fraction'])} | `{summary['passed']}` |"
        )

    lines.extend(["", "## strongest direct-Mito genes", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        lines.append("| gene | profiles | median direct-Mito RMS | q90 | active fraction | active |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in summary["examples"]:
            lines.append(
                f"| `{row['gene']}` | {row['profiles']} | {fmt(row['median_mito_rms'])} | "
                f"{fmt(row['q90_mito_rms'])} | {fmt(row['active_fraction'])} | "
                f"`{row['active']}` |"
            )
        lines.append("")

    lines.extend(["## claim boundary", "", result["claim_boundary"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-parquet", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--gene-summary-csv", type=Path, default=DEFAULT_GENE_SUMMARY)
    parser.add_argument("--source-url", default=JUMP_CRISPR_INTERPRETABLE_URL)
    parser.add_argument("--fetch-subset", action="store_true")
    parser.add_argument("--auto-fetch-subset", action="store_true")
    parser.add_argument("--http-timeout", type=int, default=120)
    parser.add_argument("--control-quantile", type=float, default=0.95)
    parser.add_argument("--active-fraction-threshold", type=float, default=0.50)
    parser.add_argument("--min-rows", type=int, default=50_000)
    parser.add_argument("--min-features", type=int, default=100)
    parser.add_argument("--min-negcon-profiles", type=int, default=5_000)
    parser.add_argument("--min-trt-profiles", type=int, default=30_000)
    parser.add_argument("--example-genes", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
