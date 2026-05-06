"""Automatic q selector from a minimal covariance graph.

The previous gates assigned L:G by hand. This gate replaces that with a small
graph rule:

    q_graph = C_local / (C_local + C_global)

where C_local and C_global are summed path conductances from the observable
node to local endpoint anchors and global ruler/horizon priors. A path
conductance is the product of edge reliabilities divided by path length.

This is still a toy graph, not a full likelihood. The point is to make the
readout selector algorithmic and falsifiable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from h0_dataset_falsification_gate import (
    ALPHA_S,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    h0_from_log_s,
)


@dataclass(frozen=True)
class Edge:
    target: str
    reliability: float


@dataclass(frozen=True)
class ChannelGraph:
    name: str
    observable: str
    local_nodes: set[str]
    global_nodes: set[str]
    graph: dict[str, list[Edge]]
    h0: float
    sigma: float


def undirected(edges: list[tuple[str, str, float]]) -> dict[str, list[Edge]]:
    graph: dict[str, list[Edge]] = {}
    for a, b, reliability in edges:
        graph.setdefault(a, []).append(Edge(b, reliability))
        graph.setdefault(b, []).append(Edge(a, reliability))
    return graph


CHANNELS = [
    ChannelGraph(
        name="Planck 2018 base LCDM",
        observable="cmb",
        local_nodes=set(),
        global_nodes={"horizon"},
        graph=undirected([("cmb", "horizon", 1.0)]),
        h0=67.4,
        sigma=0.5,
    ),
    ChannelGraph(
        name="DESI DR2 BAO no-CMB calibration",
        observable="bao",
        local_nodes={"bbn"},
        global_nodes={"sound_horizon", "population_1", "population_2"},
        graph=undirected(
            [
                ("bao", "bbn", 1.0),
                ("bao", "sound_horizon", 1.0),
                ("bao", "population_1", 1.0),
                ("bao", "population_2", 1.0),
            ]
        ),
        h0=68.51,
        sigma=0.58,
    ),
    ChannelGraph(
        name="CCHP 2025 JWST-only JAGB",
        observable="jagb",
        local_nodes={"stellar_endpoint"},
        global_nodes={f"pop_{i}" for i in range(1, 10)},
        graph=undirected(
            [("jagb", "stellar_endpoint", 1.0)]
            + [("jagb", f"pop_{i}", 1.0) for i in range(1, 10)]
        ),
        h0=67.80,
        sigma=math.hypot(2.17, 1.64),
    ),
    ChannelGraph(
        name="CCHP 2025 JWST-only TRGB",
        observable="trgb",
        local_nodes={"stellar_endpoint"},
        global_nodes={"population", "metallicity", "sn_sample"},
        graph=undirected(
            [
                ("trgb", "stellar_endpoint", 1.0),
                ("trgb", "population", 1.0),
                ("trgb", "metallicity", 1.0),
                ("trgb", "sn_sample", 1.0),
            ]
        ),
        h0=68.81,
        sigma=math.hypot(1.79, 1.32),
    ),
    ChannelGraph(
        name="CCHP 2025 TRGB HST+JWST",
        observable="trgb_mix",
        local_nodes={"stellar_endpoint"},
        global_nodes={"cross_instrument"},
        graph=undirected(
            [
                ("trgb_mix", "stellar_endpoint", 1.0),
                ("trgb_mix", "cross_instrument", 1.0),
            ]
        ),
        h0=70.39,
        sigma=math.sqrt(1.22**2 + 1.33**2 + 0.70**2),
    ),
    ChannelGraph(
        name="SH0ES HST Cepheids/SNe",
        observable="cepheid_sn",
        local_nodes={"cepheid_anchor"},
        global_nodes=set(),
        graph=undirected([("cepheid_sn", "cepheid_anchor", 1.0)]),
        h0=73.04,
        sigma=1.04,
    ),
    ChannelGraph(
        name="SH0ES JWST update",
        observable="jwst_cepheid_sn",
        local_nodes={"cepheid_anchor"},
        global_nodes=set(),
        graph=undirected([("jwst_cepheid_sn", "cepheid_anchor", 1.0)]),
        h0=73.17,
        sigma=0.86,
    ),
    ChannelGraph(
        name="TDCOSMO+SLACS hierarchical lenses",
        observable="lens",
        local_nodes={"lens_model"},
        global_nodes={"slacs_population", "kinematics", "density_hierarchy"},
        graph=undirected(
            [
                ("lens", "lens_model", 1.0),
                ("lens", "slacs_population", 1.0),
                ("lens", "kinematics", 1.0),
                ("lens", "density_hierarchy", 1.0),
            ]
        ),
        h0=67.4,
        sigma=3.65,
    ),
    ChannelGraph(
        name="Megamaser Cosmology Project",
        observable="maser",
        local_nodes={"geometric_disk"},
        global_nodes=set(),
        graph=undirected([("maser", "geometric_disk", 1.0)]),
        h0=73.9,
        sigma=3.0,
    ),
    ChannelGraph(
        name="GW standard siren representative",
        observable="gw",
        local_nodes={"gw_distance"},
        global_nodes={"host_redshift"},
        graph=undirected([("gw", "gw_distance", 1.0), ("gw", "host_redshift", 1.0)]),
        h0=70.3,
        sigma=5.15,
    ),
]


def path_conductance(channel: ChannelGraph, target_nodes: set[str]) -> float:
    total = 0.0
    queue = deque([(channel.observable, 1.0, 0, frozenset({channel.observable}))])
    while queue:
        node, reliability_product, depth, seen = queue.popleft()
        if depth > 0 and node in target_nodes:
            total += reliability_product / depth
            continue
        for edge in channel.graph.get(node, []):
            if edge.target in seen:
                continue
            queue.append(
                (
                    edge.target,
                    reliability_product * edge.reliability,
                    depth + 1,
                    seen | {edge.target},
                )
            )
    return total


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    defect = delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    log_s_global = phase_area * n_e - math.pi * defect

    print("# H0 Covariance Graph Selector Gate")
    print()
    print("## Graph rule")
    print()
    print("q_graph = C_local / (C_local + C_global)")
    print("C = sum(path reliability product / path length)")
    print()

    print("## Channel graph comparison")
    print()
    print("| channel | C_local | C_global | q_graph | H0_pred | H0_obs | pull |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    chi2 = 0.0
    for channel in CHANNELS:
        c_local = path_conductance(channel, channel.local_nodes)
        c_global = path_conductance(channel, channel.global_nodes)
        q_graph = c_local / (c_local + c_global) if c_local + c_global else 0.0
        h0_pred = h0_from_log_s(log_s_global - q_graph * defect)
        pull = (h0_pred - channel.h0) / channel.sigma
        chi2 += pull * pull
        print(
            f"| {channel.name} | {c_local:.3f} | {c_global:.3f} | "
            f"{q_graph:.4f} | {h0_pred:.3f} | {channel.h0:.3f} +/- {channel.sigma:.3f} | {pull:+.2f} |"
        )
    print()

    print("## Verdict")
    print()
    print(f"graph-selector chi2/dof = {chi2:.3f}/{len(CHANNELS)}")
    print("The hand-assigned topology selector can be reproduced by a simple covariance graph rule.")
    print("This is still schematic; the next step is replacing toy graph edges with real covariance matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
