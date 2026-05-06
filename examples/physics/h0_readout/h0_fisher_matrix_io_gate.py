"""Compute the H0 readout selector from a Fisher/covariance JSON file.

This is the first data-facing interface for the H0 readout law. It accepts a
JSON payload with nodes, a Fisher matrix or covariance matrix, and the labels
for observable/local/global nodes. If no file is supplied, it runs a built-in
smoke test equivalent to a GW-like mixed readout.

Expected JSON shape:

{
  "name": "example channel",
  "nodes": ["obs", "local_anchor", "global_prior"],
  "observable": "obs",
  "local_nodes": ["local_anchor"],
  "global_nodes": ["global_prior"],
  "matrix_type": "fisher",
  "matrix": [[1.0, 0.2, 0.2], [0.2, 1.0, 0.0], [0.2, 0.0, 1.0]],
  "h0_obs": 70.3,
  "h0_sigma": 5.15
}

For covariance input, set "matrix_type": "covariance"; the script inverts the
matrix with a small Gauss-Jordan routine to avoid extra dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from h0_dataset_falsification_gate import (
    ALPHA_S,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    h0_from_log_s,
)


@dataclass(frozen=True)
class MatrixChannel:
    name: str
    nodes: list[str]
    observable: str
    local_nodes: set[str]
    global_nodes: set[str]
    fisher: list[list[float]]
    h0_obs: float | None = None
    h0_sigma: float | None = None
    conductance_mode: str = "path"


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [
        [float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-15:
            raise ValueError("matrix is singular or numerically singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def validate_square_matrix(matrix: Any, n: int) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != n:
        raise ValueError("matrix must be a square list with size len(nodes)")
    out: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("matrix must be square")
        out.append([float(value) for value in row])
    return out


def channel_from_payload(payload: dict[str, Any]) -> MatrixChannel:
    nodes = [str(node) for node in payload["nodes"]]
    matrix = validate_square_matrix(payload["matrix"], len(nodes))
    matrix_type = str(payload.get("matrix_type", "fisher")).lower()
    if matrix_type == "covariance":
        fisher = invert_matrix(matrix)
    elif matrix_type == "fisher":
        fisher = matrix
    else:
        raise ValueError("matrix_type must be 'fisher' or 'covariance'")

    return MatrixChannel(
        name=str(payload.get("name", "unnamed channel")),
        nodes=nodes,
        observable=str(payload["observable"]),
        local_nodes={str(node) for node in payload.get("local_nodes", [])},
        global_nodes={str(node) for node in payload.get("global_nodes", [])},
        fisher=fisher,
        h0_obs=float(payload["h0_obs"]) if "h0_obs" in payload else None,
        h0_sigma=float(payload["h0_sigma"]) if "h0_sigma" in payload else None,
        conductance_mode=str(payload.get("conductance_mode", "path")).lower(),
    )


def normalized_edge_graph(channel: MatrixChannel) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = {node: [] for node in channel.nodes}
    for i, a in enumerate(channel.nodes):
        if channel.fisher[i][i] <= 0:
            raise ValueError(f"Fisher diagonal must be positive for node {a}")
        for j in range(i + 1, len(channel.nodes)):
            raw = channel.fisher[i][j]
            if raw == 0.0:
                continue
            if channel.fisher[j][j] <= 0:
                raise ValueError(f"Fisher diagonal must be positive for node {channel.nodes[j]}")
            reliability = abs(raw) / math.sqrt(channel.fisher[i][i] * channel.fisher[j][j])
            graph[a].append((channel.nodes[j], reliability))
            graph[channel.nodes[j]].append((a, reliability))
    return graph


def conductance(
    graph: dict[str, list[tuple[str, float]]],
    observable: str,
    targets: set[str],
) -> float:
    total = 0.0
    stack = [(observable, 1.0, 0, frozenset({observable}))]
    while stack:
        node, product, depth, seen = stack.pop()
        if depth > 0 and node in targets:
            total += product / depth
            continue
        for nxt, reliability in graph.get(node, []):
            if nxt in seen:
                continue
            stack.append((nxt, product * reliability, depth + 1, seen | {nxt}))
    return total


def direct_conductance(
    graph: dict[str, list[tuple[str, float]]],
    observable: str,
    targets: set[str],
) -> float:
    return sum(reliability for node, reliability in graph.get(observable, []) if node in targets)


def ce_scales() -> tuple[float, float]:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    defect = delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    log_s_global = phase_area * n_e - math.pi * defect
    return defect, log_s_global


def run_channel(channel: MatrixChannel) -> dict[str, float]:
    defect, log_s_global = ce_scales()
    graph = normalized_edge_graph(channel)
    if channel.conductance_mode == "path":
        c_local = conductance(graph, channel.observable, channel.local_nodes)
        c_global = conductance(graph, channel.observable, channel.global_nodes)
    elif channel.conductance_mode == "direct":
        c_local = direct_conductance(graph, channel.observable, channel.local_nodes)
        c_global = direct_conductance(graph, channel.observable, channel.global_nodes)
    else:
        raise ValueError("conductance_mode must be 'path' or 'direct'")
    q_f = c_local / (c_local + c_global) if c_local + c_global else 0.0
    h0_pred = h0_from_log_s(log_s_global - q_f * defect)
    return {
        "c_local": c_local,
        "c_global": c_global,
        "q_f": q_f,
        "h0_pred": h0_pred,
    }


def default_payload() -> dict[str, Any]:
    return {
        "name": "built-in GW-like Fisher smoke test",
        "nodes": ["gw", "gw_distance", "host_redshift"],
        "observable": "gw",
        "local_nodes": ["gw_distance"],
        "global_nodes": ["host_redshift"],
        "matrix_type": "fisher",
        "matrix": [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.0],
            [0.2, 0.0, 1.0],
        ],
        "h0_obs": 70.3,
        "h0_sigma": 5.15,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", nargs="?", help="optional Fisher/covariance channel JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.json_file).read_text(encoding="utf-8")) if args.json_file else default_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)

    print("# H0 Fisher-Matrix IO Gate")
    print()
    print(f"channel = {channel.name}")
    print(f"C_local = {result['c_local']:.8f}")
    print(f"C_global = {result['c_global']:.8f}")
    print(f"q_F = {result['q_f']:.8f}")
    print(f"H0_pred = {result['h0_pred']:.6f} km/s/Mpc")
    if channel.h0_obs is not None and channel.h0_sigma is not None:
        pull = (result["h0_pred"] - channel.h0_obs) / channel.h0_sigma
        print(f"H0_obs = {channel.h0_obs:.6f} +/- {channel.h0_sigma:.6f}")
        print(f"pull = {pull:+.3f}")
    print()
    print("Verdict: JSON Fisher/covariance ingestion path is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
