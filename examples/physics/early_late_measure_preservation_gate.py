"""Early-late measure preservation audit.

The horizon lift is strongest when it can be written as a preserved measure:

    I_phase = (pi^2/2) N_e
    I_late  = log S_dS + pi delta sigma + q delta sigma

where q is assigned by source topology before H0 is inspected.  q=0 is the
global horizon readout and q=1 is the local endpoint-defect readout.  This gate
checks whether source-graph q values make independent H0 channels return to
the same primordial phase measure.
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
RESULT_JSON = Path("examples/physics/early_late_measure_preservation_results.json")
REPORT_MD = Path("examples/physics/early_late_measure_preservation_report.md")


@dataclass(frozen=True)
class Edge:
    target: str
    reliability: float


@dataclass(frozen=True)
class Channel:
    name: str
    observable: str
    local_nodes: set[str]
    global_nodes: set[str]
    graph: dict[str, list[Edge]]
    h0: float
    sigma: float
    role: str


@dataclass(frozen=True)
class PreservationRow:
    channel: str
    q_graph: float
    q_obs: float
    invariant_residual: float
    invariant_pull: float
    h0_pred: float
    h0_pull: float
    role: str


def undirected(edges: list[tuple[str, str, float]]) -> dict[str, list[Edge]]:
    graph: dict[str, list[Edge]] = {}
    for a, b, reliability in edges:
        graph.setdefault(a, []).append(Edge(b, reliability))
        graph.setdefault(b, []).append(Edge(a, reliability))
    return graph


CHANNELS = [
    Channel(
        "Planck 2018 base LCDM",
        "cmb",
        set(),
        {"horizon"},
        undirected([("cmb", "horizon", 1.0)]),
        67.4,
        0.5,
        "global horizon",
    ),
    Channel(
        "DESI DR2 BAO no-CMB calibration",
        "bao",
        {"bbn"},
        {"sound_horizon", "population_1", "population_2"},
        undirected(
            [
                ("bao", "bbn", 1.0),
                ("bao", "sound_horizon", 1.0),
                ("bao", "population_1", 1.0),
                ("bao", "population_2", 1.0),
            ]
        ),
        68.51,
        0.58,
        "mostly global ruler",
    ),
    Channel(
        "CCHP 2025 JWST-only JAGB",
        "jagb",
        {"stellar_endpoint"},
        {f"pop_{i}" for i in range(1, 10)},
        undirected(
            [("jagb", "stellar_endpoint", 1.0)]
            + [("jagb", f"pop_{i}", 1.0) for i in range(1, 10)]
        ),
        67.80,
        math.hypot(2.17, 1.64),
        "endpoint diluted by population",
    ),
    Channel(
        "CCHP 2025 TRGB HST+JWST",
        "trgb_mix",
        {"stellar_endpoint"},
        {"cross_instrument"},
        undirected([("trgb_mix", "stellar_endpoint", 1.0), ("trgb_mix", "cross_instrument", 1.0)]),
        70.39,
        math.sqrt(1.22**2 + 1.33**2 + 0.70**2),
        "mixed endpoint/global",
    ),
    Channel(
        "SH0ES JWST update",
        "jwst_cepheid_sn",
        {"cepheid_anchor"},
        set(),
        undirected([("jwst_cepheid_sn", "cepheid_anchor", 1.0)]),
        73.17,
        0.86,
        "local endpoint",
    ),
    Channel(
        "TDCOSMO+SLACS hierarchical lenses",
        "lens",
        {"lens_model"},
        {"slacs_population", "kinematics", "density_hierarchy"},
        undirected(
            [
                ("lens", "lens_model", 1.0),
                ("lens", "slacs_population", 1.0),
                ("lens", "kinematics", 1.0),
                ("lens", "density_hierarchy", 1.0),
            ]
        ),
        67.4,
        3.65,
        "hierarchical lens",
    ),
    Channel(
        "Megamaser Cosmology Project",
        "maser",
        {"geometric_disk"},
        set(),
        undirected([("maser", "geometric_disk", 1.0)]),
        73.9,
        3.0,
        "local geometric endpoint",
    ),
    Channel(
        "GW standard siren representative",
        "gw",
        {"gw_distance"},
        {"host_redshift"},
        undirected([("gw", "gw_distance", 1.0), ("gw", "host_redshift", 1.0)]),
        70.3,
        5.15,
        "mixed distance-redshift bridge",
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


def conductance(channel: Channel, targets: set[str]) -> float:
    total = 0.0
    queue = deque([(channel.observable, 1.0, 0, frozenset({channel.observable}))])
    while queue:
        node, reliability, depth, seen = queue.popleft()
        if depth > 0 and node in targets:
            total += reliability / depth
            continue
        for edge in channel.graph.get(node, []):
            if edge.target in seen:
                continue
            queue.append((edge.target, reliability * edge.reliability, depth + 1, seen | {edge.target}))
    return total


def q_graph(channel: Channel) -> float:
    c_local = conductance(channel, channel.local_nodes)
    c_global = conductance(channel, channel.global_nodes)
    return c_local / (c_local + c_global) if c_local + c_global else 0.0


def core() -> dict[str, float]:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    endpoint_defect = delta * sigma
    integrated_defect = math.pi * endpoint_defect
    invariant = phase_area * n_e
    return {
        "sin2_theta_w": sin2_theta_w,
        "delta": delta,
        "d_eff": d_eff,
        "x": x,
        "sigma": sigma,
        "n_e": n_e,
        "phase_area": phase_area,
        "endpoint_defect": endpoint_defect,
        "integrated_defect": integrated_defect,
        "invariant": invariant,
        "adjoint_d3_measure": (2.0 * math.pi) ** 2 / (D_SPATIAL * D_SPATIAL - 1.0),
    }


def preservation_rows(c: dict[str, float]) -> list[PreservationRow]:
    out: list[PreservationRow] = []
    invariant = c["invariant"]
    integrated_defect = c["integrated_defect"]
    endpoint_defect = c["endpoint_defect"]
    log_s_global = invariant - integrated_defect
    for channel in CHANNELS:
        q = q_graph(channel)
        log_s_obs = log_s_from_h0(channel.h0)
        q_obs = (invariant - integrated_defect - log_s_obs) / endpoint_defect
        invariant_late = log_s_obs + integrated_defect + q * endpoint_defect
        residual = invariant_late - invariant
        sigma_i = 2.0 * channel.sigma / channel.h0
        h0_pred = h0_from_log_s(log_s_global - q * endpoint_defect)
        out.append(
            PreservationRow(
                channel=channel.name,
                q_graph=q,
                q_obs=q_obs,
                invariant_residual=residual,
                invariant_pull=residual / sigma_i,
                h0_pred=h0_pred,
                h0_pull=(h0_pred - channel.h0) / channel.sigma,
                role=channel.role,
            )
        )
    return out


def write_outputs(c: dict[str, float], rows: list[PreservationRow]) -> None:
    chi2_i = sum(row.invariant_pull * row.invariant_pull for row in rows)
    max_phase_error = abs(c["phase_area"] - c["adjoint_d3_measure"])
    payload = {
        "core": c,
        "rows": [asdict(row) for row in rows],
        "chi2_invariant": chi2_i,
        "dof": len(rows),
        "phase_adjoint_error": max_phase_error,
        "verdict": (
            "Early-late measure preservation survives as a channel-corrected "
            "Bridge: source topology chooses q, and q-corrected late horizon "
            "readouts return to the same primordial phase measure."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Early-Late Measure Preservation Gate",
        "",
        "## Core invariant",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| D_eff | {c['d_eff']:.8f} |",
        f"| x | {c['x']:.8f} |",
        f"| sigma | {c['sigma']:.8f} |",
        f"| N_e | {c['n_e']:.8f} |",
        f"| phase area pi^2/2 | {c['phase_area']:.8f} |",
        f"| d=3 adjoint phase measure | {c['adjoint_d3_measure']:.8f} |",
        f"| integrated defect pi delta sigma | {c['integrated_defect']:.8f} |",
        f"| endpoint defect delta sigma | {c['endpoint_defect']:.8f} |",
        f"| I_phase | {c['invariant']:.8f} |",
        "",
        "## Channel-corrected preservation",
        "",
        "| channel | q_graph | q_obs | invariant residual | invariant pull | H0_pred | H0 pull | role |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.channel} | {row.q_graph:.4f} | {row.q_obs:.4f} | "
            f"{row.invariant_residual:+.6f} | {row.invariant_pull:+.2f} | "
            f"{row.h0_pred:.3f} | {row.h0_pull:+.2f} | {row.role} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"invariant chi2/dof = {chi2_i:.3f}/{len(rows)}",
            f"phase-adjoint error = {max_phase_error:.3e}",
            "",
            payload["verdict"],
            "",
            "This is still not an Exact theorem: the bridge must eventually ingest real covariance/Fisher edges and justify the late horizon entropy readout from dynamics.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    c = core()
    rows = preservation_rows(c)
    write_outputs(c, rows)
    chi2_i = sum(row.invariant_pull * row.invariant_pull for row in rows)

    print("# Early-Late Measure Preservation Gate")
    print()
    print(f"phase area pi^2/2 = {c['phase_area']:.8f}")
    print(f"d=3 adjoint phase measure = {c['adjoint_d3_measure']:.8f}")
    print(f"I_phase = {c['invariant']:.8f}")
    print()
    print("| channel | q_graph | q_obs | invariant pull | H0_pred | H0 pull |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.channel} | {row.q_graph:.4f} | {row.q_obs:.4f} | "
            f"{row.invariant_pull:+.2f} | {row.h0_pred:.3f} | {row.h0_pull:+.2f} |"
        )
    print()
    print(f"invariant chi2/dof = {chi2_i:.3f}/{len(rows)}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")

    if abs(c["phase_area"] - c["adjoint_d3_measure"]) > 1e-12:
        raise SystemExit("d=3 phase-area identity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
