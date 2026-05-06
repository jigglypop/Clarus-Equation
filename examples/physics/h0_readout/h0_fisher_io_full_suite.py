"""Run the full H0 Fisher/covariance IO smoke suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHYSICS = Path(__file__).resolve().parent


COMMANDS = [
    ["h0_fisher_manifest_validate_gate.py"],
    ["h0_fisher_manifest_negative_gate.py"],
    ["h0_fisher_io_validate_gate.py"],
    ["h0_fisher_io_negative_gate.py"],
    ["h0_fisher_io_regression_gate.py"],
    ["h0_fisher_io_batch_gate.py"],
    ["h0_tdcosmo_factor_role_gate.py"],
    ["h0_tdcosmo_notebook_factor_extract_gate.py"],
    ["h0_tdcosmo_role_transition_gate.py"],
    ["h0_tdcosmo_role_ablation_gate.py"],
    ["h0_external_channel_roadmap_gate.py"],
    ["h0_bao_sn_source_scout_gate.py"],
    ["h0_bao_mean_cov_role_adapter_gate.py"],
    ["h0_bao_global_readout_gate.py"],
    ["h0_cmb_source_scout_gate.py"],
    ["h0_cmb_acoustic_global_readout_gate.py"],
    ["h0_cmb_planck_covariance_adapter_gate.py"],
    ["h0_pantheon_shoes_role_adapter_gate.py"],
    ["h0_pantheon_shoes_local_readout_gate.py"],
    ["h0_cross_channel_branch_contrast_gate.py"],
    ["h0_cross_channel_role_ablation_gate.py"],
    ["h0_cross_channel_threshold_robustness_gate.py"],
    ["h0_paper_figure_table_gate.py"],
    ["h0_paper_claim_audit_gate.py"],
    ["h0_gw_source_scout_gate.py"],
    ["h0_gw_standard_siren_bridge_gate.py"],
    ["h0_three_family_readout_table_gate.py"],
    ["h0_paper_package_gate.py"],
    ["h0_paper_provenance_table_gate.py"],
    ["h0_paper_numeric_results_gate.py"],
    ["h0_paper_prediction_ledger_gate.py"],
    ["h0_paper_plain_significance_gate.py"],
    ["h0_paper_caption_gate.py"],
    ["h0_paper_draft_gate.py"],
]


def main() -> int:
    print("# H0 Fisher IO Full Suite")
    print()
    for command in COMMANDS:
        script = PHYSICS / command[0]
        print(f"## {script.name}")
        print()
        completed = subprocess.run(
            [sys.executable, str(script), *command[1:]],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout.strip())
            print()
        if completed.stderr:
            print("stderr:")
            print(completed.stderr.strip())
            print()
        if completed.returncode != 0:
            print(f"FAILED: {script.name} exited with {completed.returncode}")
            return completed.returncode
    print("Verdict: full Fisher/covariance IO suite passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
