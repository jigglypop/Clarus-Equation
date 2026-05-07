"""CRISPRbrain empirical gate for postmitotic Clarus-cell maintenance.

This gate uses Supplementary Table 2 from Tian et al. 2021, which summarizes
hit classes for human iPSC-derived neuron CRISPRi readouts:

    CellROX        = reactive oxygen species proxy
    Liperfluo      = lipid peroxidation proxy
    LysoTracker    = lysosome state proxy
    FeRhoNox-1     = labile iron proxy

For the Clarus-cell hypothesis, this is a direct empirical test of the
postmitotic maintenance branch:

    D/Q/R = damage load, repair/lysosome/autophagy, and recurrence as survival

The gate checks whether lysosomal/autophagy/repair genes couple across several
damage and maintenance readouts, rather than appearing as isolated one-channel
hits.  It does not validate the full cell-origin model or the human brain.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = (
    REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "41593_2021_862_MOESM4_hit_class.csv"
)
RESULT_JSON = Path(__file__).with_name(
    "clarus_cell_crisprbrain_neuron_maintenance_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "clarus_cell_crisprbrain_neuron_maintenance_report.md"
)

PRIMARY_PAPER = "https://www.nature.com/articles/s41593-021-00862-0"
SUPPLEMENT_TABLE_2 = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-021-00862-0/"
    "MediaObjects/41593_2021_862_MOESM4_ESM.csv"
)

READOUTS = (
    "CellRox_CRISPRi",
    "Liperfluo_CRISPRi",
    "Lysotracker_CRISPRi",
    "FeRhoNox-1_CRISPRi",
)

READOUT_ROLES = {
    "CellRox_CRISPRi": "D: ROS load",
    "Liperfluo_CRISPRi": "D: lipid peroxidation",
    "Lysotracker_CRISPRi": "Q/U: lysosome state",
    "FeRhoNox-1_CRISPRi": "D/E: labile iron",
}


@dataclass(frozen=True)
class HitRow:
    gene: str
    readouts: dict[str, int]

    @property
    def active_channels(self) -> int:
        return sum(1 for value in self.readouts.values() if value != 0)

    @property
    def all_four_active(self) -> bool:
        return self.active_channels == len(READOUTS)

    @property
    def same_sign_active(self) -> bool:
        values = [value for value in self.readouts.values() if value != 0]
        return bool(values) and (all(value > 0 for value in values) or all(value < 0 for value in values))

    @property
    def vector(self) -> str:
        return ",".join(str(self.readouts[readout]) for readout in READOUTS)


@dataclass(frozen=True)
class OperatorClass:
    key: str
    variables: str
    role: str
    genes: tuple[str, ...]


OPERATOR_CLASSES = (
    OperatorClass(
        key="Q_lysosome_autophagy_repair",
        variables="Q,U,D,R",
        role="lysosome, autophagy, endolysosomal repair, and quality control",
        genes=(
            "PSAP",
            "AP3S1",
            "ATG13",
            "ATG14",
            "ATG3",
            "ATG9A",
            "BECN1",
            "CTSD",
            "GM2A",
            "GNPTAB",
            "MON2",
            "PIK3C3",
            "PIK3R3",
            "PQLC2",
            "PTEN",
            "RB1CC1",
            "RPTOR",
            "SLC17A5",
            "TSC1",
            "UVRAG",
            "VPS29",
            "VPS39",
            "VPS41",
            "WIPI2",
        ),
    ),
    OperatorClass(
        key="D_redox_iron_lipid_damage",
        variables="D,Q,E,R",
        role="redox, lipid peroxidation, labile iron, and ferroptosis-adjacent stress",
        genes=(
            "AKR7A2",
            "ATF4",
            "CYB561D2",
            "FBXL5",
            "NDUFA9",
            "NDUFB4",
            "NDUFB9",
            "NDUFS8",
            "NOX5",
            "PSAP",
            "SDHC",
            "SLC17A5",
            "SYVN1",
        ),
    ),
    OperatorClass(
        key="E_mito_energy",
        variables="E,A,D,R",
        role="mitochondrial and metabolic energy support for long-lived neurons",
        genes=(
            "ADSL",
            "AHCY",
            "COASY",
            "FH",
            "NDUFA9",
            "NDUFB4",
            "NDUFB9",
            "NDUFS8",
            "PPCS",
            "SDHC",
            "TK2",
        ),
    ),
    OperatorClass(
        key="U_boundary_traffic",
        variables="B,U,Q,R",
        role="vesicle traffic, membrane traffic, and boundary maintenance",
        genes=(
            "AP3S1",
            "CTAGE5",
            "DYNC1I2",
            "MON2",
            "PQLC2",
            "TMED10",
            "TMED2",
            "TFG",
            "VPS29",
            "VPS39",
            "VPS41",
            "ZW10",
        ),
    ),
    OperatorClass(
        key="A_metabolic_core",
        variables="A,E,R",
        role="basic metabolic and biosynthetic coupling",
        genes=(
            "ADSL",
            "AGPAT6",
            "AHCY",
            "COASY",
            "EBP",
            "FDPS",
            "FH",
            "GFPT1",
            "PPCS",
            "TK2",
        ),
    ),
)


def parse_hit(value: str | None) -> int:
    try:
        hit = int(value or "0")
    except ValueError:
        hit = 0
    if hit > 0:
        return 1
    if hit < 0:
        return -1
    return 0


def load_rows(path: Path) -> dict[str, HitRow]:
    rows: dict[str, HitRow] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            gene = (raw.get("gene") or "").strip().upper()
            if not gene:
                continue
            rows[gene] = HitRow(
                gene=gene,
                readouts={readout: parse_hit(raw.get(readout)) for readout in READOUTS},
            )
    return rows


def summarize_operator(
    operator: OperatorClass,
    rows: dict[str, HitRow],
    min_multi_channels: int,
    example_genes: int,
) -> dict[str, Any]:
    selected = [rows[gene] for gene in operator.genes if gene in rows]
    selected.sort(key=lambda row: (-row.active_channels, row.gene))
    multi = [row for row in selected if row.active_channels >= min_multi_channels]
    all_four = [row for row in selected if row.all_four_active]
    same_sign = [row for row in selected if row.same_sign_active and row.active_channels >= min_multi_channels]
    readout_hits = {
        readout: sum(1 for row in selected if row.readouts[readout] != 0) for readout in READOUTS
    }
    observed = len(selected)
    return {
        "key": operator.key,
        "variables": operator.variables,
        "role": operator.role,
        "candidate_genes": len(operator.genes),
        "observed_genes": observed,
        "multi_channel_genes": len(multi),
        "multi_channel_fraction": round(len(multi) / observed, 6) if observed else 0.0,
        "all_four_genes": len(all_four),
        "same_sign_multi_genes": len(same_sign),
        "mean_active_channels": round(
            sum(row.active_channels for row in selected) / observed, 6
        )
        if observed
        else 0.0,
        "readout_hits": readout_hits,
        "examples": [
            {"gene": row.gene, "channels": row.active_channels, "vector": row.vector}
            for row in selected[:example_genes]
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data_path.resolve()
    if not data_path.exists():
        return {
            "gate": "clarus_cell_crisprbrain_neuron_maintenance",
            "passed": False,
            "reason": "missing_data",
            "data_path": str(data_path),
            "download": SUPPLEMENT_TABLE_2,
        }

    rows = load_rows(data_path)
    all_rows = list(rows.values())
    psap = rows.get("PSAP")
    global_multi = [row for row in all_rows if row.active_channels >= args.min_multi_channels]
    global_all_four = [row for row in all_rows if row.all_four_active]

    summaries = [
        summarize_operator(operator, rows, args.min_multi_channels, args.example_genes)
        for operator in OPERATOR_CLASSES
    ]
    by_key = {summary["key"]: summary for summary in summaries}
    q_summary = by_key["Q_lysosome_autophagy_repair"]
    d_summary = by_key["D_redox_iron_lipid_damage"]
    e_summary = by_key["E_mito_energy"]
    u_summary = by_key["U_boundary_traffic"]

    q_readout_coverage = all(
        hits >= args.min_q_readout_hits for hits in q_summary["readout_hits"].values()
    )
    psap_control = bool(
        psap and psap.all_four_active and psap.same_sign_active and psap.vector == "1,1,1,1"
    )
    q_core = bool(
        q_summary["observed_genes"] >= args.min_q_genes
        and q_summary["multi_channel_genes"] >= args.min_q_multi_genes
        and q_summary["multi_channel_fraction"] >= args.min_q_multi_fraction
    )
    damage_coupling = bool(
        d_summary["observed_genes"] >= args.min_d_genes
        and d_summary["multi_channel_fraction"] >= args.min_d_multi_fraction
    )
    support_coupling = bool(
        e_summary["multi_channel_genes"] >= args.min_e_multi_genes
        and u_summary["multi_channel_genes"] >= args.min_u_multi_genes
    )
    passed = bool(psap_control and q_core and q_readout_coverage and damage_coupling and support_coupling)

    return {
        "gate": "clarus_cell_crisprbrain_neuron_maintenance",
        "passed": passed,
        "claim_level": "empirical_DQ_neuron_maintenance_branch" if passed else "parsed_no_promotion",
        "data_path": str(data_path),
        "primary_paper": PRIMARY_PAPER,
        "supplement_table_2": SUPPLEMENT_TABLE_2,
        "source_note": (
            "Supplementary Table 2 reports hit class values for Fig. 2g. "
            "Values 1, -1, and 0 denote positive phenotype-score hits, negative "
            "phenotype-score hits, and non-hits."
        ),
        "readout_roles": READOUT_ROLES,
        "thresholds": {
            "min_multi_channels": args.min_multi_channels,
            "min_q_genes": args.min_q_genes,
            "min_q_multi_genes": args.min_q_multi_genes,
            "min_q_multi_fraction": args.min_q_multi_fraction,
            "min_q_readout_hits": args.min_q_readout_hits,
            "min_d_genes": args.min_d_genes,
            "min_d_multi_fraction": args.min_d_multi_fraction,
            "min_e_multi_genes": args.min_e_multi_genes,
            "min_u_multi_genes": args.min_u_multi_genes,
        },
        "background": {
            "genes": len(all_rows),
            "multi_channel_genes": len(global_multi),
            "multi_channel_fraction": round(len(global_multi) / len(all_rows), 6)
            if all_rows
            else 0.0,
            "all_four_genes": len(global_all_four),
        },
        "psap_control": {
            "observed": psap is not None,
            "vector": psap.vector if psap else None,
            "all_four_active": bool(psap and psap.all_four_active),
            "same_sign_active": bool(psap and psap.same_sign_active),
            "passed": psap_control,
        },
        "decision": {
            "psap_control_ok": psap_control,
            "q_core_ok": q_core,
            "q_readout_coverage_ok": q_readout_coverage,
            "damage_coupling_ok": damage_coupling,
            "energy_and_traffic_support_ok": support_coupling,
        },
        "operator_summaries": summaries,
    }


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("reason") == "missing_data":
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Clarus cell CRISPRbrain neuron maintenance gate",
                    "",
                    "- passed: `False`",
                    "- reason: missing local hit-class data",
                    f"- expected path: `{result['data_path']}`",
                    f"- download: <{result['download']}>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    bg = result["background"]
    psap = result["psap_control"]
    decision = result["decision"]
    lines = [
        "# Clarus cell CRISPRbrain neuron maintenance gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- primary paper: [Tian et al. 2021]({result['primary_paper']})",
        f"- supplement table: [Supplementary Table 2]({result['supplement_table_2']})",
        f"- local data: `{result['data_path']}`",
        f"- source note: {result['source_note']}",
        "",
        "## readouts",
        "",
    ]
    for readout, role in result["readout_roles"].items():
        lines.append(f"- `{readout}`: {role}")

    lines.extend(
        [
            "",
            "## background",
            "",
            f"- genes in hit-class table: `{bg['genes']}`",
            f"- multi-channel genes: `{bg['multi_channel_genes']}`",
            f"- multi-channel fraction: `{bg['multi_channel_fraction']}`",
            f"- all-four-channel genes: `{bg['all_four_genes']}`",
            "",
            "## PSAP control",
            "",
            f"- observed: `{psap['observed']}`",
            f"- vector CellROX,Liperfluo,LysoTracker,FeRhoNox: `{psap['vector']}`",
            f"- all four active: `{psap['all_four_active']}`",
            f"- same-sign active: `{psap['same_sign_active']}`",
            f"- passed: `{psap['passed']}`",
            "",
            "## decision",
            "",
            f"- PSAP control ok: `{decision['psap_control_ok']}`",
            f"- Q core ok: `{decision['q_core_ok']}`",
            f"- Q readout coverage ok: `{decision['q_readout_coverage_ok']}`",
            f"- D coupling ok: `{decision['damage_coupling_ok']}`",
            f"- E/U support ok: `{decision['energy_and_traffic_support_ok']}`",
            "",
            "## operator summaries",
            "",
            "| operator | vars | observed | multi | multi frac | all four | same-sign multi | mean channels |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in result["operator_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['key']}`",
                    f"`{summary['variables']}`",
                    str(summary["observed_genes"]),
                    str(summary["multi_channel_genes"]),
                    str(summary["multi_channel_fraction"]),
                    str(summary["all_four_genes"]),
                    str(summary["same_sign_multi_genes"]),
                    str(summary["mean_active_channels"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## readout coverage by operator", ""])
    for summary in result["operator_summaries"]:
        hits = summary["readout_hits"]
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        for readout in READOUTS:
            lines.append(f"- `{readout}`: `{hits[readout]}` genes")
        lines.append("")

    lines.extend(["## strongest coupled genes", ""])
    for summary in result["operator_summaries"]:
        lines.append(f"### `{summary['key']}`")
        lines.append("")
        lines.append("| gene | active channels | vector |")
        lines.append("|---|---:|---|")
        for row in summary["examples"]:
            lines.append(f"| `{row['gene']}` | {row['channels']} | `{row['vector']}` |")
        lines.append("")

    lines.extend(
        [
            "## claim boundary",
            "",
            (
                "This promotes only the postmitotic neuron D/Q maintenance branch: "
                "lysosome/autophagy/repair genes are coupled to ROS, lipid peroxidation, "
                "lysosome state, and labile iron readouts.  It does not prove the whole "
                "Clarus-cell mechanism."
            ),
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--min-multi-channels", type=int, default=3)
    parser.add_argument("--min-q-genes", type=int, default=12)
    parser.add_argument("--min-q-multi-genes", type=int, default=10)
    parser.add_argument("--min-q-multi-fraction", type=float, default=0.65)
    parser.add_argument("--min-q-readout-hits", type=int, default=8)
    parser.add_argument("--min-d-genes", type=int, default=8)
    parser.add_argument("--min-d-multi-fraction", type=float, default=0.60)
    parser.add_argument("--min-e-multi-genes", type=int, default=5)
    parser.add_argument("--min-u-multi-genes", type=int, default=5)
    parser.add_argument("--example-genes", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result.get("claim_level")}))


if __name__ == "__main__":
    main()
