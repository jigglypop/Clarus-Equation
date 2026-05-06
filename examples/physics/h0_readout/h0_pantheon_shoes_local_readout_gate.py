"""Run a branch-only H0 readout check for Pantheon+SH0ES local ladder roles."""

from __future__ import annotations

import math

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel
from h0_pantheon_shoes_role_adapter_gate import counts, role_summary


def branch_payload() -> dict[str, object]:
    stats = counts()
    roles = role_summary(stats)
    nodes = [*roles["observable_nodes"], *roles["local_nodes"]]
    n = len(nodes)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0

    total = stats.calibrators + stats.hubble_flow + stats.surveys
    weights = [
        stats.calibrators / total,
        stats.hubble_flow / total,
        stats.surveys / total,
    ]
    for index, weight in enumerate(weights, start=1):
        reliability = 0.5 * math.sqrt(weight / max(weights))
        matrix[0][index] = reliability
        matrix[index][0] = reliability

    return {
        "name": "Pantheon+SH0ES local distance-ladder branch check",
        "nodes": nodes,
        "observable": roles["observable_nodes"][0],
        "local_nodes": roles["local_nodes"],
        "global_nodes": roles["global_nodes"],
        "matrix_type": "fisher",
        "matrix": matrix,
        "conductance_mode": "direct",
        "source": {
            "repo": "https://github.com/PantheonPlusSH0ES/DataRelease",
            "distance_file": "Pantheon+SH0ES.dat",
            "covariance_file": "Pantheon+SH0ES_STAT+SYS.cov",
            "role_basis": "IS_CALIBRATOR and USED_IN_SH0ES_HF source labels",
        },
    }


def main() -> int:
    payload = branch_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)

    print("# H0 Pantheon+SH0ES Local Readout Gate")
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

    if result["c_local"] <= 0.0:
        raise SystemExit("Pantheon+SH0ES local conductance should be positive")
    if result["c_global"] != 0.0:
        raise SystemExit("Pantheon+SH0ES branch-only check should not have global conductance")
    if abs(result["q_f"] - 1.0) > 1e-12:
        raise SystemExit("Pantheon+SH0ES branch-only check should select the local endpoint")

    print("Verdict: Pantheon+SH0ES source roles select the local/high-side readout branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
