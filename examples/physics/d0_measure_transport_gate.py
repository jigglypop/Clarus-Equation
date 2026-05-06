"""Measure-transport audit for the d=0 boundary interpretation.

d=0 is not treated as a physical location or reachable state.  The only
allowed use here is as a zero-recursive-entropy boundary whose d=3 trace is
measured by dimensionless transports:

    identity boundary:      x0 = 1, sigma0 = 0, S_R0 = 0
    contracted branch:      x = exp[-D_eff(1-x)]
    entropy transport:      S_R = -log x = D_eff(1-x)
    source measure:         Q_source = x(1-x)
    half-cycle projection:  (2/pi) Q_source
    spatial +1 projection:  sigma^(D/(D+1))

The gate separates closed dimensionless transports from open scale lifts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
RESULT_JSON = Path("examples/physics/d0_measure_transport_results.json")
REPORT_MD = Path("examples/physics/d0_measure_transport_report.md")


@dataclass(frozen=True)
class TransportState:
    x0: float
    sigma0: float
    s0: float
    sin2_theta_w: float
    delta: float
    d_eff: float
    x: float
    sigma: float
    s_recursive: float
    entropy_identity_error: float
    contraction: float
    reverse_amplification: float
    n_e: float
    entropy_per_efold: float
    q_source: float
    q_phase: float
    q_ger: float
    phase_projection: float
    spatial_projection: float
    p_ger: float
    residual_after_10: float
    residual_after_20: float
    residual_iterations_1e60: float
    curvature_efold: float
    curvature_recursive: float


@dataclass(frozen=True)
class TransportRow:
    name: str
    equation: str
    value: float
    status: str
    guardrail: str


def bootstrap_x(d_eff: float, tol: float = 1e-15, max_iter: int = 500) -> float:
    x = math.exp(-d_eff)
    for _ in range(max_iter):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    raise RuntimeError("bootstrap_x did not converge")


def iterations_for_decay(k: float, target: float) -> float:
    if not (0.0 < k < 1.0):
        return float("nan")
    return math.log(target) / math.log(k)


def state() -> TransportState:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    s_recursive = -math.log(x)
    entropy_identity_error = s_recursive - d_eff * sigma
    contraction = d_eff * x
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    q_source = x * sigma
    phase_projection = 2.0 / math.pi
    spatial_projection = d_eff / (d_eff + 1.0)
    p_ger = phase_projection * sigma**spatial_projection
    q_phase = phase_projection * q_source
    q_ger = p_ger * q_source
    return TransportState(
        x0=1.0,
        sigma0=0.0,
        s0=0.0,
        sin2_theta_w=sin2_theta_w,
        delta=delta,
        d_eff=d_eff,
        x=x,
        sigma=sigma,
        s_recursive=s_recursive,
        entropy_identity_error=entropy_identity_error,
        contraction=contraction,
        reverse_amplification=1.0 / contraction,
        n_e=n_e,
        entropy_per_efold=s_recursive / n_e,
        q_source=q_source,
        q_phase=q_phase,
        q_ger=q_ger,
        phase_projection=phase_projection,
        spatial_projection=spatial_projection,
        p_ger=p_ger,
        residual_after_10=contraction**10,
        residual_after_20=contraction**20,
        residual_iterations_1e60=iterations_for_decay(contraction, 1.0e-60),
        curvature_efold=math.exp(-2.0 * n_e),
        curvature_recursive=contraction**n_e,
    )


def rows(s: TransportState) -> list[TransportRow]:
    return [
        TransportRow(
            name="identity boundary",
            equation="x0=1, sigma0=0, S_R0=0",
            value=s.s0,
            status="closed",
            guardrail="boundary identity, not a physical location",
        ),
        TransportRow(
            name="recursive entropy",
            equation="S_R=-log(x)=D_eff(1-x)",
            value=s.s_recursive,
            status="closed dimensionless transport",
            guardrail="not absolute thermodynamic entropy",
        ),
        TransportRow(
            name="branch contraction",
            equation="k=D_eff*x",
            value=s.contraction,
            status="stable forward d=3 branch",
            guardrail="reverse map amplifies residuals; no finite-time arrival at d=0",
        ),
        TransportRow(
            name="source measure",
            equation="Q_source=x(1-x)",
            value=s.q_source,
            status="closed residual source",
            guardrail="must pass through readout taxonomy before becoming observable",
        ),
        TransportRow(
            name="half-cycle transport",
            equation="Q_phase=(2/pi)Q_source",
            value=s.q_phase,
            status="projection candidate",
            guardrail="not sufficient alone for A_s",
        ),
        TransportRow(
            name="+1 spatial transport",
            equation="Q_GER=(2/pi)sigma^(D/(D+1))Q_source",
            value=s.q_ger,
            status="selection candidate",
            guardrail="not exact until shared likelihood tests survive",
        ),
        TransportRow(
            name="curvature dilution",
            equation="exp(-2N_e)",
            value=s.curvature_efold,
            status="closed dimensionless flatness direction",
            guardrail="not an Omega_k measurement without FLRW scale map",
        ),
        TransportRow(
            name="recursive residual erasure",
            equation="k^N_e",
            value=s.curvature_recursive,
            status="closed dimensionless residual suppression",
            guardrail="not reheating or horizon entropy",
        ),
    ]


def write_outputs(s: TransportState, items: list[TransportRow]) -> None:
    payload = {
        "state": asdict(s),
        "rows": [asdict(item) for item in items],
        "passed": abs(s.entropy_identity_error) < 1e-12 and 0.0 < s.contraction < 1.0,
        "verdict": (
            "d=0 remains usable as a zero-measure boundary condition.  The "
            "allowed transport is dimensionless entropy/residual projection, "
            "not motion to a physical d=0 state."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# d=0 Measure Transport Gate",
        "",
        "## Transport state",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| x0 | {s.x0:.8f} |",
        f"| sigma0 | {s.sigma0:.8f} |",
        f"| S_R0 | {s.s0:.8f} |",
        f"| D_eff | {s.d_eff:.8f} |",
        f"| x | {s.x:.8f} |",
        f"| sigma | {s.sigma:.8f} |",
        f"| S_R | {s.s_recursive:.8f} |",
        f"| S_R-D_eff(1-x) | {s.entropy_identity_error:+.3e} |",
        f"| contraction k | {s.contraction:.8f} |",
        f"| reverse amplification 1/k | {s.reverse_amplification:.8f} |",
        f"| N_e | {s.n_e:.8f} |",
        f"| S_R/N_e | {s.entropy_per_efold:.8f} |",
        f"| residual after 10 steps k^10 | {s.residual_after_10:.8e} |",
        f"| residual after 20 steps k^20 | {s.residual_after_20:.8e} |",
        f"| steps for 1e-60 residual | {s.residual_iterations_1e60:.2f} |",
        "",
        "## Transport rows",
        "",
        "| name | equation | value | status | guardrail |",
        "|---|---|---:|---|---|",
    ]
    for item in items:
        lines.append(f"| {item.name} | {item.equation} | {item.value:.8e} | {item.status} | {item.guardrail} |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
            "The next unresolved step is a true scale lift: derive how this dimensionless transport maps to FLRW curvature, reheating, or late horizon entropy without importing the answer.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    s = state()
    items = rows(s)
    write_outputs(s, items)

    print("# d=0 Measure Transport Gate")
    print()
    print(f"S_R-D_eff(1-x) = {s.entropy_identity_error:+.3e}")
    print(f"contraction k = {s.contraction:.8f}")
    print(f"reverse amplification 1/k = {s.reverse_amplification:.8f}")
    print(f"steps for 1e-60 residual = {s.residual_iterations_1e60:.2f}")
    print()
    print("| transport | value | status |")
    print("|---|---:|---|")
    for item in items:
        print(f"| {item.name} | {item.value:.8e} | {item.status} |")
    print()
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")

    if abs(s.entropy_identity_error) > 1e-12:
        raise SystemExit("recursive entropy identity failed")
    if not (0.0 < s.contraction < 1.0):
        raise SystemExit("d=3 branch should be contractive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
