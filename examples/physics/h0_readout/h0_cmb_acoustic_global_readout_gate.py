"""Run a branch-only H0 readout check for CMB acoustic-scale closure roles.

This gate is deliberately source-role only. It does not ingest a Planck
likelihood or covariance yet. The point is to record the next cosmology
prediction: an acoustic-scale CMB channel is an early global horizon closure,
so it should select the same global/low branch as BAO.
"""

from __future__ import annotations

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel


def branch_payload() -> dict[str, object]:
    nodes = [
        "theta_star_observable",
        "sound_horizon_at_drag_epoch",
        "last_scattering_surface",
        "early_density_closure",
        "angular_diameter_distance_to_recombination",
        "recombination_history",
    ]
    n = len(nodes)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0

    # Minimal acoustic-scale factor graph: the observed angle couples to the
    # sound horizon and the global distance-to-recombination closure.
    for j, weight in [(1, 0.50), (2, 0.35), (3, 0.45), (4, 0.40), (5, 0.25)]:
        matrix[0][j] = weight
        matrix[j][0] = weight

    return {
        "name": "CMB acoustic-scale global horizon branch check",
        "nodes": nodes,
        "observable": "theta_star_observable",
        "local_nodes": [],
        "global_nodes": nodes[1:],
        "matrix_type": "fisher",
        "matrix": matrix,
        "conductance_mode": "direct",
        "source": {
            "role_basis": "CMB acoustic angle as an early global horizon and distance-to-recombination closure",
            "data_status": "branch-only; public likelihood/covariance ingestion remains future work",
        },
    }


def main() -> int:
    payload = branch_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)

    print("# H0 CMB Acoustic Global Readout Gate")
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
        raise SystemExit("CMB global conductance should be positive")
    if result["c_local"] != 0.0:
        raise SystemExit("CMB acoustic branch-only check should not have local conductance")
    if result["q_f"] != 0.0:
        raise SystemExit("CMB acoustic branch-only check should select the global endpoint")

    print("Verdict: CMB acoustic source roles select the global/low-side readout branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
