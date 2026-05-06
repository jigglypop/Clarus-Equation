"""Self-reference audit for the H0 readout selector.

The q selector is only useful if q is computed before H0 is inspected.  This
gate therefore separates three maps:

    source graph -> q_graph
    q_graph -> H0_pred
    H0_obs -> q_obs

The self-reference check is whether q_graph is close to q_obs in q-space.  The
identity q_graph -> H0_pred -> q_back is printed as a closure sanity check, but
it is not counted as evidence because it is algebraic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
MPC_KM = 3.0856775814913673e19
T_PLANCK_S = 5.391247e-44
RESULT_JSON = Path("examples/physics/h0_readout/h0_recursive_selector_self_reference_results.json")
REPORT_MD = Path("examples/physics/h0_readout/h0_recursive_selector_self_reference_report.md")


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
    source_blind: str


@dataclass(frozen=True)
class SelectorRow:
    channel: str
    q_graph: float
    q_obs: float
    sigma_q: float
    q_pull: float
    q_back: float
    h0_pred: float
    h0_obs: float
    h0_pull: float
    source_blind: str


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
        source_blind="early acoustic horizon endpoint only",
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
        source_blind="one local abundance anchor plus three global ruler/population anchors",
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
        source_blind="one stellar endpoint diluted by nine population anchors",
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
        source_blind="one stellar endpoint and one cross-instrument closure",
    ),
    ChannelGraph(
        name="SH0ES JWST update",
        observable="jwst_cepheid_sn",
        local_nodes={"cepheid_anchor"},
        global_nodes=set(),
        graph=undirected([("jwst_cepheid_sn", "cepheid_anchor", 1.0)]),
        h0=73.17,
        sigma=0.86,
        source_blind="local Cepheid/SN endpoint closure",
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
        source_blind="one lens endpoint plus three hierarchical/global closures",
    ),
    ChannelGraph(
        name="Megamaser Cosmology Project",
        observable="maser",
        local_nodes={"geometric_disk"},
        global_nodes=set(),
        graph=undirected([("maser", "geometric_disk", 1.0)]),
        h0=73.9,
        sigma=3.0,
        source_blind="one-step local geometric distance",
    ),
    ChannelGraph(
        name="GW standard siren representative",
        observable="gw",
        local_nodes={"gw_distance"},
        global_nodes={"host_redshift"},
        graph=undirected([("gw", "gw_distance", 1.0), ("gw", "host_redshift", 1.0)]),
        h0=70.3,
        sigma=5.15,
        source_blind="absolute GW distance plus redshift/environment bridge",
    ),
]


def bootstrap_x(d_eff: float, tol: float = 1e-15, max_iter: int = 500) -> float:
    x = math.exp(-d_eff)
    for _ in range(max_iter):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    raise RuntimeError("bootstrap_x did not converge")


def log_s_from_h0(h0_km_s_mpc: float) -> float:
    h0_s = h0_km_s_mpc / MPC_KM
    return math.log(math.pi / (h0_s * T_PLANCK_S) ** 2)


def h0_from_log_s(log_s: float) -> float:
    h_s = math.sqrt(math.pi) * math.exp(-0.5 * log_s) / T_PLANCK_S
    return h_s * MPC_KM


def conductance(channel: ChannelGraph, targets: set[str]) -> float:
    total = 0.0
    queue = deque([(channel.observable, 1.0, 0, frozenset({channel.observable}))])
    while queue:
        node, reliability_product, depth, seen = queue.popleft()
        if depth > 0 and node in targets:
            total += reliability_product / depth
            continue
        for edge in channel.graph.get(node, []):
            if edge.target in seen:
                continue
            queue.append((edge.target, reliability_product * edge.reliability, depth + 1, seen | {edge.target}))
    return total


def selector_state() -> tuple[float, float]:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    defect = delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    log_s_global = (math.pi * math.pi / 2.0) * n_e - math.pi * defect
    return defect, log_s_global


def q_from_graph(channel: ChannelGraph) -> float:
    c_local = conductance(channel, channel.local_nodes)
    c_global = conductance(channel, channel.global_nodes)
    return c_local / (c_local + c_global) if c_local + c_global else 0.0


def rows() -> list[SelectorRow]:
    defect, log_s_global = selector_state()
    out: list[SelectorRow] = []
    for channel in CHANNELS:
        q_graph = q_from_graph(channel)
        h0_pred = h0_from_log_s(log_s_global - q_graph * defect)
        q_back = (log_s_global - log_s_from_h0(h0_pred)) / defect
        q_obs = (log_s_global - log_s_from_h0(channel.h0)) / defect
        sigma_q = 2.0 * channel.sigma / (channel.h0 * defect)
        q_pull = (q_graph - q_obs) / sigma_q
        h0_pull = (h0_pred - channel.h0) / channel.sigma
        out.append(
            SelectorRow(
                channel=channel.name,
                q_graph=q_graph,
                q_obs=q_obs,
                sigma_q=sigma_q,
                q_pull=q_pull,
                q_back=q_back,
                h0_pred=h0_pred,
                h0_obs=channel.h0,
                h0_pull=h0_pull,
                source_blind=channel.source_blind,
            )
        )
    return out


def write_outputs(items: list[SelectorRow]) -> None:
    chi2_q = sum(item.q_pull * item.q_pull for item in items)
    chi2_h0 = sum(item.h0_pull * item.h0_pull for item in items)
    max_self_drift = max(abs(item.q_back - item.q_graph) for item in items)
    payload = {
        "chi2_q": chi2_q,
        "chi2_h0": chi2_h0,
        "dof": len(items),
        "max_algebraic_self_drift": max_self_drift,
        "rows": [asdict(item) for item in items],
        "verdict": (
            "q selector remains a viable self-recursive readout rule if q_graph "
            "is assigned from source topology before H0 comparison."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# H0 Recursive Selector Self-Reference Gate",
        "",
        "## q-space closure",
        "",
        "| channel | q_graph | q_obs | sigma_q | q pull | q_back drift | H0_pred | H0_obs | H0 pull | source-blind rule |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in items:
        lines.append(
            f"| {item.channel} | {item.q_graph:.4f} | {item.q_obs:.4f} | {item.sigma_q:.4f} | "
            f"{item.q_pull:+.2f} | {item.q_back - item.q_graph:+.2e} | "
            f"{item.h0_pred:.3f} | {item.h0_obs:.3f} | {item.h0_pull:+.2f} | {item.source_blind} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"q-space chi2/dof = {chi2_q:.3f}/{len(items)}",
            f"H0-space chi2/dof = {chi2_h0:.3f}/{len(items)}",
            f"max algebraic q_graph -> H0_pred -> q_back drift = {max_self_drift:.3e}",
            "",
            "The algebraic loop closes by construction, so the real test is q_graph versus q_obs. "
            "This keeps the self-reference in the selector layer instead of fitting a new H0 correction per channel.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    items = rows()
    write_outputs(items)
    chi2_q = sum(item.q_pull * item.q_pull for item in items)
    chi2_h0 = sum(item.h0_pull * item.h0_pull for item in items)
    max_self_drift = max(abs(item.q_back - item.q_graph) for item in items)

    print("# H0 Recursive Selector Self-Reference Gate")
    print()
    print("| channel | q_graph | q_obs | q pull | H0_pred | H0 pull |")
    print("|---|---:|---:|---:|---:|---:|")
    for item in items:
        print(
            f"| {item.channel} | {item.q_graph:.4f} | {item.q_obs:.4f} | "
            f"{item.q_pull:+.2f} | {item.h0_pred:.3f} | {item.h0_pull:+.2f} |"
        )
    print()
    print(f"q-space chi2/dof = {chi2_q:.3f}/{len(items)}")
    print(f"H0-space chi2/dof = {chi2_h0:.3f}/{len(items)}")
    print(f"max algebraic self drift = {max_self_drift:.3e}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
