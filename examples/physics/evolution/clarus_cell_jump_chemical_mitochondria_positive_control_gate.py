"""JUMP compound direct-Mito positive-control gate.

The CRISPR direct-Mito gate did not promote the mitochondrial E operator.  This
gate asks whether the same JUMP direct-Mito feature extraction can detect known
mitochondrial chemical perturbagens.  A pass here does not rescue the genetic E
branch; it validates the image-channel assay and makes the CRISPR non-promotion
more interpretable.

Compound identities are matched by InChIKey through Broad Babel.  The positive
control set is intentionally explicit and small:

    FCCP, CCCP, phenformin, metformin, menadione, niclosamide

Their InChIKeys were resolved through PubChem PUG REST name/property queries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarus_cell_jump_channel_specific_mitochondria_gate import (
    DATA_DIR,
    DIRECT_MITO_MODULES,
    HTTPRangeReader,
    JUMP_DATASETS,
    JUMP_PROFILE_INDEX,
    direct_mito_columns,
    distribution,
    fmt,
    map_jcp_ids,
    optional_dependencies,
    robust_mito_rms,
)


DEFAULT_SUBSET = DATA_DIR / "jump_compound_mito_direct_features.parquet"
DEFAULT_COMPOUND_SUMMARY = DATA_DIR / "jump_compound_mito_positive_control_summary.csv"
RESULT_JSON = Path(__file__).with_name(
    "clarus_cell_jump_chemical_mitochondria_positive_control_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "clarus_cell_jump_chemical_mitochondria_positive_control_report.md"
)

JUMP_COMPOUND_INTERPRETABLE_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/"
    "source_all/workspace/profiles_assembled/COMPOUND/v1.0/"
    "profiles_var_mad_int.parquet"
)
PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest"


@dataclass(frozen=True)
class MitoCompound:
    name: str
    inchikey: str
    mode: str
    pubchem_cid: int


MITO_COMPOUNDS = (
    MitoCompound(
        name="FCCP",
        inchikey="BMZRVOVNUMQTIN-UHFFFAOYSA-N",
        mode="oxidative-phosphorylation uncoupler",
        pubchem_cid=3330,
    ),
    MitoCompound(
        name="CCCP",
        inchikey="UGTJLJZQQFGTJD-UHFFFAOYSA-N",
        mode="oxidative-phosphorylation uncoupler",
        pubchem_cid=2603,
    ),
    MitoCompound(
        name="phenformin",
        inchikey="ICFJFFQQTFMIBG-UHFFFAOYSA-N",
        mode="biguanide mitochondrial complex-I stress",
        pubchem_cid=8249,
    ),
    MitoCompound(
        name="metformin",
        inchikey="XZWYZXLIPXDOLR-UHFFFAOYSA-N",
        mode="biguanide mitochondrial complex-I stress",
        pubchem_cid=4091,
    ),
    MitoCompound(
        name="menadione",
        inchikey="MJVAVZPDRWSRRC-UHFFFAOYSA-N",
        mode="redox/mitochondrial oxidative stress",
        pubchem_cid=4055,
    ),
    MitoCompound(
        name="niclosamide",
        inchikey="RJMUSRYZPJIFPJ-UHFFFAOYSA-N",
        mode="mitochondrial uncoupling/stress",
        pubchem_cid=4477,
    ),
)


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


def median(np: Any, values: list[float]) -> float:
    if not values:
        return math.nan
    return float(np.median(np.asarray(values, dtype=float)))


def summarize_compound(
    np: Any,
    compound: MitoCompound,
    values: list[float],
    jcp_ids: set[str],
    active_threshold: float,
    active_fraction_threshold: float,
) -> dict[str, Any]:
    active_values = [value for value in values if value > active_threshold]
    active_fraction = len(active_values) / len(values) if values else 0.0
    median_rms = median(np, values)
    active = bool(median_rms > active_threshold or active_fraction >= active_fraction_threshold)
    return {
        "name": compound.name,
        "inchikey": compound.inchikey,
        "mode": compound.mode,
        "pubchem_cid": compound.pubchem_cid,
        "jcp_ids": sorted(jcp_ids),
        "profiles": len(values),
        "median_mito_rms": round(median_rms, 6),
        "q90_mito_rms": round(float(np.quantile(np.asarray(values, dtype=float), 0.90)), 6)
        if values
        else math.nan,
        "active_fraction": round(active_fraction, 6),
        "active": active,
    }


def write_compound_summary(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "name",
                "inchikey",
                "mode",
                "pubchem_cid",
                "jcp_ids",
                "profiles",
                "median_mito_rms",
                "q90_mito_rms",
                "active_fraction",
                "active",
                "negative_control_q95",
            ),
        )
        writer.writeheader()
        for row in result.get("compound_summaries", []):
            writer.writerow(
                {
                    "name": row["name"],
                    "inchikey": row["inchikey"],
                    "mode": row["mode"],
                    "pubchem_cid": row["pubchem_cid"],
                    "jcp_ids": ",".join(row["jcp_ids"]),
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
            "gate": "clarus_cell_jump_chemical_mitochondria_positive_control",
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
            "gate": "clarus_cell_jump_chemical_mitochondria_positive_control",
            "passed": False,
            "reason": "missing_subset",
            "subset_path": str(args.subset_parquet),
            "source_url": args.source_url,
            "how_to_build": (
                ".venv\\Scripts\\python.exe examples\\physics\\evolution\\"
                "clarus_cell_jump_chemical_mitochondria_positive_control_gate.py --fetch-subset"
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

    values_by_key: dict[str, list[float]] = {compound.inchikey: [] for compound in MITO_COMPOUNDS}
    ids_by_key: dict[str, set[str]] = {compound.inchikey: set() for compound in MITO_COMPOUNDS}
    target_keys = set(values_by_key)
    for jcp_id, value in zip(jcp_ids, mito_rms):
        if not math.isfinite(float(value)) or pert_type_by_jcp.get(jcp_id) != "trt":
            continue
        standard_key = standard_by_jcp.get(jcp_id, "")
        if standard_key in target_keys:
            values_by_key[standard_key].append(float(value))
            ids_by_key[standard_key].add(jcp_id)

    compound_summaries = [
        summarize_compound(
            np,
            compound,
            values_by_key[compound.inchikey],
            ids_by_key[compound.inchikey],
            active_threshold,
            args.active_fraction_threshold,
        )
        for compound in MITO_COMPOUNDS
    ]
    observed = [row for row in compound_summaries if row["profiles"] > 0]
    active = [row for row in observed if row["active"]]
    uncoupler_active = any(
        row["active"] and "uncoupl" in row["mode"].lower() for row in observed
    )
    data_ok = bool(
        table.num_rows >= args.min_rows
        and len(feature_columns) >= args.min_features
        and neg_stats["n"] >= args.min_negcon_profiles
        and trt_stats["n"] >= args.min_trt_profiles
    )
    positive_control_ok = bool(
        len(observed) >= args.min_observed_compounds
        and len(active) >= args.min_active_compounds
        and uncoupler_active
    )
    partial_sensitivity = bool(len(active) >= 2 and uncoupler_active)
    passed = bool(data_ok and positive_control_ok)

    result = {
        "gate": "clarus_cell_jump_chemical_mitochondria_positive_control",
        "passed": passed,
        "claim_level": "empirical_mito_channel_assay_positive_control"
        if passed
        else "parsed_no_promotion",
        "specific_claim": (
            "direct_mito_channel_assay_sensitive"
            if passed
            else "direct_mito_uncoupler_partial_sensitivity"
            if partial_sensitivity
            else "direct_mito_channel_assay_not_promoted"
        ),
        "jump_datasets": JUMP_DATASETS,
        "profile_index": JUMP_PROFILE_INDEX,
        "source_url": args.source_url,
        "pubchem_pug_rest": PUBCHEM_PUG_REST,
        "subset_path": str(args.subset_parquet.resolve()),
        "compound_summary_csv": str(args.compound_summary_csv.resolve()),
        "fetch_info": fetch_info,
        "rows": table.num_rows,
        "direct_mito_features": len(feature_columns),
        "direct_mito_modules": DIRECT_MITO_MODULES,
        "unique_jcp_ids": len(set(jcp_ids)),
        "mapped_jcp_ids": len(standard_by_jcp),
        "negative_control": neg_stats,
        "treatment": trt_stats,
        "active_threshold_quantile": args.control_quantile,
        "active_threshold": round(active_threshold, 6),
        "active_rule": (
            f"compound median direct-Mito robust-z RMS > negative-control q"
            f"{int(args.control_quantile * 100)} or active profile fraction >= "
            f"{args.active_fraction_threshold}"
        ),
        "data_ok": data_ok,
        "observed_compounds": len(observed),
        "active_compounds": len(active),
        "uncoupler_active": uncoupler_active,
        "partial_sensitivity": partial_sensitivity,
        "positive_control_ok": positive_control_ok,
        "compound_summaries": compound_summaries,
        "claim_boundary": (
            "This is an assay positive-control gate. It shows whether direct "
            "Mito-channel JUMP compound profiles respond to known mitochondrial "
            "perturbagens. It does not by itself validate the genetic Clarus E "
            "operator, cell recurrence, or brain mechanism."
        ),
    }
    write_compound_summary(args.compound_summary_csv.resolve(), result)
    return result


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("reason") == "missing_dependency":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell JUMP chemical mitochondrial positive-control gate",
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
                    "# Clarus cell JUMP chemical mitochondrial positive-control gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local direct-Mito compound subset",
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
        "# Clarus cell JUMP chemical mitochondrial positive-control gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- specific claim: `{result['specific_claim']}`",
        f"- source: [JUMP Cell Painting datasets]({result['jump_datasets']})",
        f"- profile index: [v0.11.0 manifest]({result['profile_index']})",
        f"- source parquet: `{result['source_url']}`",
        f"- compound identity source: [PubChem PUG REST]({result['pubchem_pug_rest']})",
        f"- local subset: `{result['subset_path']}`",
        f"- compound summary: `{result['compound_summary_csv']}`",
        f"- rows/direct Mito features: `{result['rows']}` / `{result['direct_mito_features']}`",
        f"- active threshold: negative-control q{int(result['active_threshold_quantile'] * 100)} = "
        f"`{fmt(result['active_threshold'])}`",
        f"- active rule: {result['active_rule']}",
        f"- data ok: `{result['data_ok']}`",
        f"- observed compounds: `{result['observed_compounds']}/{len(result['compound_summaries'])}`",
        f"- active compounds: `{result['active_compounds']}`",
        f"- uncoupler active: `{result['uncoupler_active']}`",
        f"- partial sensitivity: `{result['partial_sensitivity']}`",
        f"- positive control ok: `{result['positive_control_ok']}`",
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
        "## compound summaries",
        "",
        "| compound | mode | profiles | median direct-Mito RMS | q90 | active frac | active |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["compound_summaries"]:
        lines.append(
            f"| `{row['name']}` | {row['mode']} | {row['profiles']} | "
            f"{fmt(row['median_mito_rms'])} | {fmt(row['q90_mito_rms'])} | "
            f"{fmt(row['active_fraction'])} | `{row['active']}` |"
        )

    lines.extend(
        [
            "",
            "## compound identifiers",
            "",
            "| compound | PubChem CID | InChIKey | JCP ids |",
            "|---|---:|---|---|",
        ]
    )
    for row in result["compound_summaries"]:
        lines.append(
            f"| `{row['name']}` | {row['pubchem_cid']} | `{row['inchikey']}` | "
            f"`{','.join(row['jcp_ids']) or 'not observed'}` |"
        )

    lines.extend(["", "## claim boundary", "", result["claim_boundary"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-parquet", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--compound-summary-csv", type=Path, default=DEFAULT_COMPOUND_SUMMARY)
    parser.add_argument("--source-url", default=JUMP_COMPOUND_INTERPRETABLE_URL)
    parser.add_argument("--fetch-subset", action="store_true")
    parser.add_argument("--auto-fetch-subset", action="store_true")
    parser.add_argument("--http-timeout", type=int, default=120)
    parser.add_argument("--control-quantile", type=float, default=0.95)
    parser.add_argument("--active-fraction-threshold", type=float, default=0.50)
    parser.add_argument("--min-rows", type=int, default=800_000)
    parser.add_argument("--min-features", type=int, default=100)
    parser.add_argument("--min-negcon-profiles", type=int, default=20_000)
    parser.add_argument("--min-trt-profiles", type=int, default=500_000)
    parser.add_argument("--min-observed-compounds", type=int, default=4)
    parser.add_argument("--min-active-compounds", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
