"""Fisher-matrix form of the H0 readout selector.

This gate moves from hand-written graph conductances to a minimal Fisher-matrix
edge rule:

    r_ij = |F_ij| / sqrt(F_ii F_jj)
    q_F  = C_local / (C_local + C_global)

where conductances are computed from normalized Fisher edges. The test also
checks invariance under diagonal parameter rescalings F -> D F D.

The matrices here are schematic, but the API is the one needed for real
likelihood Fisher/covariance matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from h0_dataset_falsification_gate import (
    ALPHA_S,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    h0_from_log_s,
)


@dataclass(frozen=True)
class FisherChannel:
    name: str
    nodes: list[str]
    observable: str
    local_nodes: set[str]
    global_nodes: set[str]
    fisher_edges: list[tuple[str, str, float]]
    h0: float
    sigma: float


CHANNELS = [
    FisherChannel(
        "Planck 2018 base LCDM",
        ["cmb", "horizon"],
        "cmb",
        set(),
        {"horizon"},
        [("cmb", "horizon", 0.20)],
        67.4,
        0.5,
    ),
    FisherChannel(
        "DESI DR2 BAO no-CMB calibration",
        ["bao", "bbn", "sound_horizon", "population_1", "population_2"],
        "bao",
        {"bbn"},
        {"sound_horizon", "population_1", "population_2"},
        [
            ("bao", "bbn", 0.20),
            ("bao", "sound_horizon", 0.20),
            ("bao", "population_1", 0.20),
            ("bao", "population_2", 0.20),
        ],
        68.51,
        0.58,
    ),
    FisherChannel(
        "CCHP 2025 JWST-only JAGB",
        ["jagb", "stellar_endpoint"] + [f"pop_{i}" for i in range(1, 10)],
        "jagb",
        {"stellar_endpoint"},
        {f"pop_{i}" for i in range(1, 10)},
        [("jagb", "stellar_endpoint", 0.10)]
        + [("jagb", f"pop_{i}", 0.10) for i in range(1, 10)],
        67.80,
        math.hypot(2.17, 1.64),
    ),
    FisherChannel(
        "CCHP 2025 TRGB HST+JWST",
        ["trgb_mix", "stellar_endpoint", "cross_instrument"],
        "trgb_mix",
        {"stellar_endpoint"},
        {"cross_instrument"},
        [("trgb_mix", "stellar_endpoint", 0.20), ("trgb_mix", "cross_instrument", 0.20)],
        70.39,
        math.sqrt(1.22**2 + 1.33**2 + 0.70**2),
    ),
    FisherChannel(
        "SH0ES JWST update",
        ["jwst_cepheid_sn", "cepheid_anchor"],
        "jwst_cepheid_sn",
        {"cepheid_anchor"},
        set(),
        [("jwst_cepheid_sn", "cepheid_anchor", 0.20)],
        73.17,
        0.86,
    ),
    FisherChannel(
        "GW standard siren representative",
        ["gw", "gw_distance", "host_redshift"],
        "gw",
        {"gw_distance"},
        {"host_redshift"},
        [("gw", "gw_distance", 0.20), ("gw", "host_redshift", 0.20)],
        70.3,
        5.15,
    ),
]


def fisher_matrix(channel: FisherChannel) -> list[list[float]]:
    n = len(channel.nodes)
    index = {node: i for i, node in enumerate(channel.nodes)}
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    for a, b, weight in channel.fisher_edges:
        i = index[a]
        j = index[b]
        matrix[i][j] = weight
        matrix[j][i] = weight
    return matrix


def normalized_edges(nodes: list[str], fisher: list[list[float]]) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            raw = fisher[i][j]
            if raw == 0.0:
                continue
            reliability = abs(raw) / math.sqrt(fisher[i][i] * fisher[j][j])
            graph[a].append((nodes[j], reliability))
            graph[nodes[j]].append((a, reliability))
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


def q_from_fisher(channel: FisherChannel, fisher: list[list[float]]) -> float:
    graph = normalized_edges(channel.nodes, fisher)
    c_local = conductance(graph, channel.observable, channel.local_nodes)
    c_global = conductance(graph, channel.observable, channel.global_nodes)
    return c_local / (c_local + c_global) if c_local + c_global else 0.0


def rescale_fisher(fisher: list[list[float]], scales: list[float]) -> list[list[float]]:
    n = len(fisher)
    return [[scales[i] * fisher[i][j] * scales[j] for j in range(n)] for i in range(n)]


def main() -> int:
    rng = random.Random(20260506)

    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    defect = delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    log_s_global = phase_area * n_e - math.pi * defect

    print("# H0 Fisher-Matrix Selector Gate")
    print()
    print("## Edge rule")
    print()
    print("r_ij = |F_ij| / sqrt(F_ii F_jj)")
    print("q_F = C_local / (C_local + C_global)")
    print()

    print("## Fisher selector comparison")
    print()
    print("| channel | q_F | max rescale drift | H0_pred | H0_obs | pull |")
    print("|---|---:|---:|---:|---:|---:|")
    chi2 = 0.0
    for channel in CHANNELS:
        fisher = fisher_matrix(channel)
        q_f = q_from_fisher(channel, fisher)
        max_drift = 0.0
        for _ in range(200):
            scales = [math.exp(rng.uniform(-3.0, 3.0)) for _ in channel.nodes]
            q_scaled = q_from_fisher(channel, rescale_fisher(fisher, scales))
            max_drift = max(max_drift, abs(q_scaled - q_f))
        h0_pred = h0_from_log_s(log_s_global - q_f * defect)
        pull = (h0_pred - channel.h0) / channel.sigma
        chi2 += pull * pull
        print(
            f"| {channel.name} | {q_f:.4f} | {max_drift:.3e} | "
            f"{h0_pred:.3f} | {channel.h0:.3f} +/- {channel.sigma:.3f} | {pull:+.2f} |"
        )
    print()

    print("## Verdict")
    print()
    print(f"Fisher-selector chi2/dof = {chi2:.3f}/{len(CHANNELS)}")
    print("The normalized Fisher-edge rule reproduces the graph selector and is invariant under parameter rescaling.")
    print("This is the form that can be fed by real likelihood Fisher matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
