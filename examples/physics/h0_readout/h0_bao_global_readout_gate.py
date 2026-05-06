"""Run a branch-only H0 readout check for DESI BAO global closure roles.

This gate turns the DESI BAO mean/covariance role adapter into a minimal
pipeline channel. It does not attach an observed H0 value. The purpose is only
to check the branch direction implied by the source roles: BAO measurements are
distance ratios to a standard ruler, so the channel should have no local
endpoint conductance and should select the global/low-side branch.
"""

from __future__ import annotations

import math

from h0_bao_mean_cov_role_adapter_gate import COV, MEAN, read_matrix, read_mean, validate_covariance
from h0_fisher_matrix_io_gate import channel_from_payload, invert_matrix, run_channel


def branch_payload() -> dict[str, object]:
    data = read_mean(MEAN)
    cov = read_matrix(COV)
    validate_covariance(data, cov)
    fisher = invert_matrix(cov)

    bao_nodes = [datum.node for datum in data]
    nodes = ["bao_aggregate", "sound_horizon_standard_ruler", *bao_nodes]
    n = len(nodes)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0

    diag = [abs(fisher[i][i]) for i in range(len(fisher))]
    total = sum(diag)
    if total <= 0.0:
        raise ValueError("BAO Fisher diagonal has no positive information")

    # Source-derived but scale-safe reliability weights: more precise BAO bins
    # couple more strongly to the aggregate observable, while staying below 1.
    for offset, info in enumerate(diag, start=2):
        weight = 0.5 * math.sqrt(info / max(diag))
        matrix[0][offset] = weight
        matrix[offset][0] = weight

    matrix[0][1] = 0.5
    matrix[1][0] = 0.5

    return {
        "name": "DESI 2024 BAO global standard-ruler branch check",
        "nodes": nodes,
        "observable": "bao_aggregate",
        "local_nodes": [],
        "global_nodes": ["sound_horizon_standard_ruler", *bao_nodes],
        "matrix_type": "fisher",
        "matrix": matrix,
        "conductance_mode": "direct",
        "source": {
            "repo": "https://github.com/CobayaSampler/bao_data",
            "mean_file": MEAN.name,
            "cov_file": COV.name,
            "role_basis": "BAO distance ratios over sound horizon",
        },
    }


def main() -> int:
    payload = branch_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)

    print("# H0 BAO Global Readout Gate")
    print()
    print(f"channel = {channel.name}")
    print(f"nodes = {len(channel.nodes)}")
    print(f"local_nodes = {len(channel.local_nodes)}")
    print(f"global_nodes = {len(channel.global_nodes)}")
    print(f"C_local = {result['c_local']:.8f}")
    print(f"C_global = {result['c_global']:.8f}")
    print(f"q_F = {result['q_f']:.8f}")
    print(f"H0_branch_pred = {result['h0_pred']:.6f} km/s/Mpc")
    print()

    if result["c_global"] <= 0.0:
        raise SystemExit("BAO global conductance should be positive")
    if result["c_local"] != 0.0:
        raise SystemExit("BAO branch-only check should not have local conductance")
    if result["q_f"] != 0.0:
        raise SystemExit("BAO branch-only check should select the global endpoint")

    print("Verdict: DESI BAO source roles select the global/low-side readout branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
