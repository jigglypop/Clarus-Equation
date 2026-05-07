"""Scout public empirical data for the Clarus-cell mechanism.

The existing Clarus-cell gates are mechanistic toy/stress-ablation models.
This scout does not claim a new empirical pass.  It ranks public primary
datasets and papers by whether they can test the 9-variable Clarus-cell loop:

    B,U,E,A,I,D,Q,S,R

where B=boundary, U=regulated ports/traffic, E=energy, A=metabolism,
I=identity template, D=damage, Q=repair, S=support context, R=recurrence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("clarus_cell_empirical_scout_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_empirical_scout_report.md")
REPO_ROOT = Path(__file__).resolve().parents[3]

OPERATORS = ("B", "U", "E", "A", "I", "D", "Q", "S", "R")

LOCAL_EMPIRICAL_FILES = {
    "crisprbrain_hit_class": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "41593_2021_862_MOESM4_hit_class.csv",
    "psapko_rnaseq": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "GSE152988_WT_vs_PSAPKO.csv.gz",
    "depmap_operator_subset": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "depmap_24q4_clarus_operator_dependency_subset.csv",
    "microglia_support_tables": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "41593_2022_1131_microglia_supp_tables.xlsx",
    "astrocyte_screen_table": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "41593_2022_1180_astrocyte_screen_table2.xlsx",
    "astrocyte_cropseq_table": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "41593_2022_1180_astrocyte_cropseq_table6.xlsx",
    "hpa_subcellular_location": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "hpa_subcellular_location.tsv.zip",
    "jump_profile_index": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_profile_index_v0.11.0.json",
    "jump_crispr_profiles_pca_corrected": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_crispr_profiles_pca_corrected.parquet",
    "jump_crispr_profiles_interpretable": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_crispr_profiles_interpretable.parquet",
    "jump_operator_morphology_subset": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_crispr_clarus_operator_morphology_subset.csv",
    "jump_mito_direct_feature_subset": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_crispr_mito_direct_features.parquet",
    "jump_mito_direct_gene_summary": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_crispr_mito_direct_gene_summary.csv",
    "jump_compound_mito_direct_feature_subset": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_compound_mito_direct_features.parquet",
    "jump_compound_mito_positive_control_summary": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "jump_compound_mito_positive_control_summary.csv",
    "replogle_k562_essential_normalized_bulk": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "replogle_k562_essential_normalized_bulk_01.h5ad",
    "replogle_rpe1_normalized_bulk": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "replogle_rpe1_normalized_bulk_01.h5ad",
    "replogle_perturbseq_clarus_state_summary": REPO_ROOT
    / "data"
    / "evolution"
    / "clarus_cell"
    / "replogle_perturbseq_clarus_state_summary.csv",
}


@dataclass(frozen=True)
class EvidenceSource:
    name: str
    url: str
    source_kind: str
    biological_form: str
    operator_coverage: tuple[str, ...]
    perturbational: bool
    dynamic_or_conditioned: bool
    direct_readouts: tuple[str, ...]
    why_it_matters: str
    caveat: str
    next_gate: str
    acquisition: str


SOURCES = (
    EvidenceSource(
        name="DepMap Project Achilles CRISPR gene effect",
        url="https://depmap.org/portal/achilles/",
        source_kind="official data portal",
        biological_form="human proliferative/cancer cell lines",
        operator_coverage=("E", "A", "I", "U", "D", "Q", "R"),
        perturbational=True,
        dynamic_or_conditioned=False,
        direct_readouts=("gene knockout fitness", "cell survival/proliferation"),
        why_it_matters=(
            "Tests whether loss of metabolism, genome/epigenome, traffic, repair, and "
            "cell-cycle genes collapses proliferative recurrence across many human cells."
        ),
        caveat="Cancer line fitness is not normal tissue identity; B and S are only indirect.",
        next_gate="clarus_cell_depmap_operator_dependency_gate.py",
        acquisition="download Achilles_gene_effect or current DepMap CRISPR gene effect release",
    ),
    EvidenceSource(
        name="Genome-scale Perturb-seq in K562 and RPE1",
        url="https://pubmed.ncbi.nlm.nih.gov/35688146/",
        source_kind="primary paper plus public single-cell data",
        biological_form="human proliferative K562 and RPE1 cells",
        operator_coverage=("E", "A", "I", "D", "Q", "R"),
        perturbational=True,
        dynamic_or_conditioned=True,
        direct_readouts=("single-cell transcriptomic response", "gene knockdown identity state"),
        why_it_matters=(
            "Tests whether the identity-template and metabolism/repair operators rebuild "
            "cell state after targeted loss-of-function perturbations."
        ),
        caveat="Mostly transcriptomic state, not membrane/organelle morphology.",
        next_gate="clarus_cell_perturbseq_state_reconstruction_gate.py",
        acquisition="download Replogle et al. processed h5ad/matrix files",
    ),
    EvidenceSource(
        name="JUMP Cell Painting genetic and chemical perturbation profiles",
        url="https://github.com/jump-cellpainting/datasets",
        source_kind="official consortium dataset",
        biological_form="human U2OS high-content imaging",
        operator_coverage=("B", "U", "E", "D", "Q", "R"),
        perturbational=True,
        dynamic_or_conditioned=False,
        direct_readouts=("mitochondria channel", "ER/AGP channel", "RNA/DNA morphology", "cell count"),
        why_it_matters=(
            "Provides image-level proxies for boundary, organelle traffic, damage, and "
            "recurrence after thousands of gene and chemical perturbations."
        ),
        caveat="Very large image corpus; gene expression and long-term recurrence are indirect.",
        next_gate="clarus_cell_jump_morphology_operator_gate.py",
        acquisition="start from JUMP metadata/profiles before raw images",
    ),
    EvidenceSource(
        name="OpenCell endogenous tagging map",
        url="https://pubmed.ncbi.nlm.nih.gov/35271311/",
        source_kind="primary paper and official imaging/proteomics resource",
        biological_form="human HEK293T-derived live-cell protein localization",
        operator_coverage=("B", "U", "A", "I", "Q"),
        perturbational=False,
        dynamic_or_conditioned=False,
        direct_readouts=("protein localization", "protein interaction", "cellular compartment map"),
        why_it_matters=(
            "Anchors the architecture side of Clarus cell: which proteins physically sit in "
            "membrane, traffic, nucleus, metabolism, and repair compartments."
        ),
        caveat="Architecture/blueprint source, not an ablation survival gate by itself.",
        next_gate="clarus_cell_opencell_operator_blueprint_gate.py",
        acquisition="download OpenCell localization and interaction tables",
    ),
    EvidenceSource(
        name="Human Protein Atlas subcellular and brain single-cell resources",
        url="https://www.proteinatlas.org/humanproteome/subcellular",
        source_kind="official atlas",
        biological_form="human cells, tissues, and brain cell types",
        operator_coverage=("B", "U", "A", "I", "Q", "S"),
        perturbational=False,
        dynamic_or_conditioned=False,
        direct_readouts=("subcellular protein localization", "brain cell-type expression"),
        why_it_matters=(
            "Gives broad human operator gene sets and tissue/cell-type support context, "
            "especially for neural and glial support terms."
        ),
        caveat="Static atlas; needs pairing with perturbation data for causal tests.",
        next_gate="clarus_cell_hpa_operator_gene_set_gate.py",
        acquisition="download HPA subcellular locations and brain single-cell tables",
    ),
    EvidenceSource(
        name="CRISPRbrain human iPSC-derived neuron screens",
        url="https://www.nature.com/articles/s41593-021-00862-0",
        source_kind="primary paper plus public data commons/GEO",
        biological_form="human postmitotic iPSC-derived neurons",
        operator_coverage=("E", "A", "D", "Q", "R", "S"),
        perturbational=True,
        dynamic_or_conditioned=True,
        direct_readouts=("neuronal survival", "oxidative stress response", "FACS/RNA-seq phenotypes"),
        why_it_matters=(
            "Directly attacks the hardest branch: postmitotic neural Clarus cells where "
            "recurrence means maintenance rather than division."
        ),
        caveat="Neuron monoculture is incomplete tissue support; glia support needs separate screens.",
        next_gate="clarus_cell_crisprbrain_neuron_maintenance_gate.py",
        acquisition="download CRISPRbrain or GEO GSE152988 processed screen tables",
    ),
    EvidenceSource(
        name="CRISPRbrain human microglia and astrocyte screens",
        url="https://www.nature.com/articles/s41593-022-01131-4",
        source_kind="primary paper plus public data commons",
        biological_form="human iPSC-derived glia",
        operator_coverage=("D", "Q", "S", "R"),
        perturbational=True,
        dynamic_or_conditioned=True,
        direct_readouts=("glial disease-state regulators", "inflammatory/reactive state"),
        why_it_matters=(
            "Tests the tissue-support side of neural Clarus cells: whether glial state "
            "and repair/support operators are controllable and cell-type-specific."
        ),
        caveat="Supports the S/Q branch, not the whole cell loop alone.",
        next_gate="clarus_cell_glia_support_operator_gate.py",
        acquisition="download CRISPRbrain glia screen tables",
    ),
    EvidenceSource(
        name="BioGRID ORCS CRISPR screen index",
        url="https://orcs.thebiogrid.org/",
        source_kind="official curated screen database",
        biological_form="many organisms, cell lines, and phenotypes",
        operator_coverage=("B", "U", "E", "A", "I", "D", "Q", "R"),
        perturbational=True,
        dynamic_or_conditioned=True,
        direct_readouts=("standardized CRISPR screen scores", "phenotype-specific hits"),
        why_it_matters=(
            "Lets us search operator-specific screens such as OXPHOS, autophagy, "
            "lysosome, membrane trafficking, and viability in one index."
        ),
        caveat="Aggregator quality varies by underlying publication and phenotype.",
        next_gate="clarus_cell_orcs_screen_triage_gate.py",
        acquisition="query/download ORCS screen tables by operator keyword",
    ),
    EvidenceSource(
        name="Model protocell membrane growth and division experiments",
        url="https://pubs.acs.org/doi/10.1021/ja900919c",
        source_kind="primary experimental paper",
        biological_form="fatty-acid model protocells",
        operator_coverage=("B", "U", "I", "R"),
        perturbational=True,
        dynamic_or_conditioned=True,
        direct_readouts=("vesicle growth", "division", "encapsulated RNA redistribution"),
        why_it_matters=(
            "Empirical primitive-cell anchor for boundary, resource uptake, template "
            "retention, and recurrence before modern organelles."
        ),
        caveat="Not a full autocatalytic metabolism or living cell.",
        next_gate="clarus_cell_protocell_boundary_recurrence_gate.py",
        acquisition="extract figure/supplement measurements or recreate assay table manually",
    ),
    EvidenceSource(
        name="Nonenzymatic RNA synthesis inside fatty-acid vesicles",
        url="https://www.science.org/doi/10.1126/science.1241888",
        source_kind="primary experimental paper",
        biological_form="model RNA protocells",
        operator_coverage=("B", "U", "I", "D"),
        perturbational=True,
        dynamic_or_conditioned=True,
        direct_readouts=("RNA template copying", "vesicle compatibility", "Mg2+ stress mitigation"),
        why_it_matters=(
            "Empirical anchor for copying-template inside a bounded open compartment."
        ),
        caveat="No complete growth-division cycle in the same gate.",
        next_gate="clarus_cell_protocell_template_copying_gate.py",
        acquisition="extract supplementary kinetic/copying measurements",
    ),
)


def score_source(source: EvidenceSource) -> dict[str, Any]:
    coverage = len(set(source.operator_coverage))
    score = coverage / len(OPERATORS)
    if source.perturbational:
        score += 0.20
    if source.dynamic_or_conditioned:
        score += 0.12
    if "human postmitotic" in source.biological_form:
        score += 0.12
    if "protocell" in source.biological_form:
        score += 0.06
    return {
        "operator_count": coverage,
        "operator_fraction": coverage / len(OPERATORS),
        "priority_score": round(score, 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for source in SOURCES:
        row = asdict(source)
        row.update(score_source(source))
        rows.append(row)
    rows.sort(key=lambda row: row["priority_score"], reverse=True)

    recommended = {
        "first_empirical_gate": "clarus_cell_crisprbrain_neuron_maintenance_gate.py",
        "why": (
            "It directly tests the postmitotic neural branch where Clarus recurrence is "
            "maintenance/survival under stress, not division."
        ),
        "parallel_gate": "clarus_cell_depmap_operator_dependency_gate.py",
        "parallel_why": (
            "DepMap gives the broadest perturbational evidence for proliferative "
            "human recurrence across many cell contexts."
        ),
        "architecture_gate": "clarus_cell_jump_morphology_operator_gate.py",
    }
    local_empirical_files = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in LOCAL_EMPIRICAL_FILES.items()
    }
    first_gate_report = Path(__file__).with_name(
        "clarus_cell_crisprbrain_neuron_maintenance_report.md"
    )

    result = {
        "gate": "clarus_cell_empirical_scout",
        "passed": True,
        "local_empirical_data_ready": any(
            item["exists"] for item in local_empirical_files.values()
        ),
        "local_empirical_files": local_empirical_files,
        "first_gate_report_exists": first_gate_report.exists(),
        "operator_symbols": OPERATORS,
        "sources_ranked": rows,
        "recommended_next": recommended,
        "claim_boundary": (
            "This scout only identifies empirical routes.  Promotion decisions belong "
            "to the operator-level gate reports."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Clarus cell empirical scout",
        "",
        f"- passed: `{result['passed']}`",
        f"- local empirical data ready: `{result['local_empirical_data_ready']}`",
        f"- first gate report exists: `{result['first_gate_report_exists']}`",
        f"- first empirical gate: `{result['recommended_next']['first_empirical_gate']}`",
        f"- parallel proliferative gate: `{result['recommended_next']['parallel_gate']}`",
        f"- architecture gate: `{result['recommended_next']['architecture_gate']}`",
        "",
        "## local empirical files",
        "",
    ]
    for name, info in result["local_empirical_files"].items():
        lines.append(f"- `{name}`: `{info['exists']}` at `{info['path']}`")

    lines.extend(
        [
            "",
            "## ranked sources",
        "",
            "| rank | source | form | operators | perturb | conditioned | priority | next gate |",
            "|---:|---|---|---|---|---|---:|---|",
        ]
    )
    for rank, row in enumerate(result["sources_ranked"], start=1):
        lines.append(
            f"| {rank} | [{row['name']}]({row['url']}) | {row['biological_form']} | "
            f"`{','.join(row['operator_coverage'])}` | `{row['perturbational']}` | "
            f"`{row['dynamic_or_conditioned']}` | {row['priority_score']:.3f} | "
            f"`{row['next_gate']}` |"
        )
    lines.extend(
        [
            "",
            "## operator-to-data reading",
            "",
            "- `B` boundary: strongest empirical route is JUMP morphology plus protocell vesicle assays.",
            "- `U` regulated ports/traffic: strongest route is JUMP morphology plus OpenCell/HPA localization.",
            "- `E/A` energy and metabolism: strongest route is DepMap, CRISPRbrain neuron survival, and OXPHOS screens via ORCS.",
            "- `I` identity template: strongest route is Perturb-seq state reconstruction and DepMap essentiality.",
            "- `D/Q` damage and repair: strongest route is CRISPRbrain oxidative-stress/lysosome/autophagy screens.",
            "- `S` support context: strongest route is glia CRISPRbrain plus HPA/brain cell-type atlases.",
            "- `R` recurrence: proliferative branch uses DepMap fitness; postmitotic branch uses neuron survival/maintenance.",
            "",
            "## recommended next gates",
            "",
            "1. `clarus_cell_crisprbrain_neuron_maintenance_gate.py`: test postmitotic neural maintenance.",
            "2. `clarus_cell_depmap_operator_dependency_gate.py`: test proliferative recurrence dependencies.",
            "3. `clarus_cell_jump_morphology_operator_gate.py`: test morphology/operator separation.",
            "4. `clarus_cell_protocell_boundary_recurrence_gate.py`: test primitive boundary/template recurrence.",
            "",
            "## claim boundary",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = run(args)
    print(json.dumps({"passed": result["passed"], "sources": len(result["sources_ranked"])}, indent=2))


if __name__ == "__main__":
    main()
