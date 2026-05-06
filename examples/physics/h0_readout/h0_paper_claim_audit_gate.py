"""Audit paper claims against the reproducible H0 readout gates."""

from __future__ import annotations

from dataclasses import dataclass

from h0_fisher_io_full_suite import COMMANDS


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim: str
    required_gates: tuple[str, ...]
    allowed_status: str


CLAIMS = [
    Claim(
        "C1",
        "TDCOSMO role metadata is not hand-floating; it is reproducible from declared likelihood factors.",
        ("h0_tdcosmo_factor_role_gate.py",),
        "claimable",
    ),
    Claim(
        "C2",
        "The public TDCOSMO notebook sampler composition reproduces the likelihood-factor graph.",
        ("h0_tdcosmo_notebook_factor_extract_gate.py",),
        "claimable",
    ),
    Claim(
        "C3",
        "TDCOSMO changes H0 branch when SLACS population closure is added.",
        ("h0_tdcosmo_role_transition_gate.py",),
        "claimable",
    ),
    Claim(
        "C4",
        "Static TDCOSMO role maps fail relative to the source-aware role transition.",
        ("h0_tdcosmo_role_ablation_gate.py",),
        "claimable",
    ),
    Claim(
        "C5",
        "DESI BAO source labels select the global/low readout branch before an H0 refit.",
        ("h0_bao_mean_cov_role_adapter_gate.py", "h0_bao_global_readout_gate.py"),
        "claimable-with-scope",
    ),
    Claim(
        "C6",
        "Pantheon+SH0ES source labels select the local/high readout branch before an H0 refit.",
        ("h0_pantheon_shoes_role_adapter_gate.py", "h0_pantheon_shoes_local_readout_gate.py"),
        "claimable-with-scope",
    ),
    Claim(
        "C7",
        "Independent channel rows split into local/high and global/low families by source role.",
        ("h0_cross_channel_branch_contrast_gate.py", "h0_paper_figure_table_gate.py"),
        "claimable",
    ),
    Claim(
        "C8",
        "The cross-channel split is not produced by all-local, all-global, or flipped role maps.",
        ("h0_cross_channel_role_ablation_gate.py",),
        "claimable",
    ),
    Claim(
        "C9",
        "The cross-channel split is robust to broad branch-classification thresholds.",
        ("h0_cross_channel_threshold_robustness_gate.py",),
        "claimable",
    ),
    Claim(
        "C10",
        "Planck PR3 CMB covariance selects the global/low branch under the acoustic-scale source-role map.",
        ("h0_cmb_source_scout_gate.py", "h0_cmb_acoustic_global_readout_gate.py", "h0_cmb_planck_covariance_adapter_gate.py"),
        "claimable",
    ),
    Claim(
        "L1",
        "A full joint BAO/SN/TDCOSMO H0 posterior refit has not yet been performed.",
        ("h0_external_channel_roadmap_gate.py",),
        "limitation",
    ),
    Claim(
        "C11",
        "GW170817-like standard sirens select a bridge/intermediate H0 branch rather than either endpoint.",
        ("h0_gw_source_scout_gate.py", "h0_gw_standard_siren_bridge_gate.py"),
        "claimable-with-scope",
    ),
    Claim(
        "C12",
        "The H0 readout table spans global/low, bridge/intermediate, and local/high source-role families.",
        ("h0_three_family_readout_table_gate.py",),
        "claimable-with-scope",
    ),
    Claim(
        "C13",
        "The paper package has reproducible endpoint and three-family figures plus explicit limitations.",
        ("h0_paper_package_gate.py",),
        "claimable",
    ),
    Claim(
        "C14",
        "The paper draft contains the required claim, figure, result, ablation, limitation, and next-test spine.",
        ("h0_paper_draft_gate.py",),
        "claimable",
    ),
    Claim(
        "C15",
        "Every H0 readout row has a paper provenance entry linking source role, public source, and primary gate.",
        ("h0_paper_provenance_table_gate.py",),
        "claimable",
    ),
    Claim(
        "C16",
        "The paper numeric results table reports selectors, readout branches, H0 readouts, and scoped references.",
        ("h0_paper_numeric_results_gate.py",),
        "claimable",
    ),
    Claim(
        "C17",
        "The paper states falsifiable future predictions after source roles are fixed first.",
        ("h0_paper_prediction_ledger_gate.py",),
        "claimable-with-future-tests",
    ),
    Claim(
        "C18",
        "The paper draft explains the physical significance in plain language without replacing the technical claim.",
        ("h0_paper_plain_significance_gate.py",),
        "claimable",
    ),
    Claim(
        "C19",
        "The paper draft contains figure captions that state the branch-selection scope and non-posterior status.",
        ("h0_paper_caption_gate.py",),
        "claimable",
    ),
]


def suite_script_names() -> set[str]:
    return {command[0] for command in COMMANDS}


def main() -> int:
    available = suite_script_names()
    failed = 0

    print("# H0 Paper Claim Audit Gate")
    print()
    print("| id | status | claim scope | required gates | suite coverage |")
    print("|---|---|---|---|---|")

    for item in CLAIMS:
        missing = [gate for gate in item.required_gates if gate not in available]
        coverage = "PASS" if not missing else "MISSING: " + ", ".join(missing)
        if missing:
            failed += 1
        print(
            f"| {item.claim_id} | {item.allowed_status} | {item.claim} | "
            f"{', '.join(item.required_gates)} | {coverage} |"
        )

    print()
    print(f"audited claims = {len(CLAIMS)}")
    print(f"missing gate links = {failed}")

    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: paper claims are linked to reproducible H0 readout gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
