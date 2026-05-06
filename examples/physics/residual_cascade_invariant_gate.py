"""Residual-cascade invariant audit for self-recursive cosmology.

The previous audits identified a second open lever after the H0 q-selector:
raw residuals should not be repaired with a different recursion per observable.
This gate checks whether the reopened scalar-amplitude sector has one reusable
cascade:

    total sensitivity -> source residual -> half-cycle projection -> GER

where

    Q_total  = x(1-x)/(1-D_eff x)
    Q_source = x(1-x)
    Q_phase  = (2/pi) Q_source
    Q_GER    = (2/pi) sigma^(D_eff/(D_eff+1)) Q_source

The gate records which parts are algebraic, which parts are observationally
compatible, and which parts remain open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9
A_S_SIGMA = 0.03e-9
RESULT_JSON = Path("examples/physics/residual_cascade_invariant_results.json")
REPORT_MD = Path("examples/physics/residual_cascade_invariant_report.md")


@dataclass(frozen=True)
class CascadeState:
    sin2_theta_w: float
    delta: float
    d_eff: float
    x: float
    sigma: float
    contraction: float
    gamma_eff: float
    n_e: float
    q_total: float
    q_source: float
    q_phase: float
    q_ger: float
    p_phase: float
    p_ger: float
    total_to_source_gain: float
    raw_as: float
    source_as: float
    phase_as: float
    ger_as: float
    raw_pull: float
    ger_pull: float
    n_s: float
    alpha_spec: float
    r_tensor: float
    tensor_running_ratio: float
    quadrupole_handle: float
    hemispherical_handle: float
    large_angle_fractional: float


@dataclass(frozen=True)
class InvariantRow:
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


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def cascade_state() -> CascadeState:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    contraction = d_eff * x
    gamma_eff = d_eff / (d_eff + 1.0)
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    q_total = x * sigma / (1.0 - contraction)
    q_source = x * sigma
    p_phase = 2.0 / math.pi
    p_ger = p_phase * sigma**gamma_eff
    q_phase = p_phase * q_source
    q_ger = p_ger * q_source
    total_to_source_gain = q_total / q_source

    raw_as = compute_a_s(x, sigma, n_e, q_total)
    source_as = compute_a_s(x, sigma, n_e, q_source)
    phase_as = compute_a_s(x, sigma, n_e, q_phase)
    ger_as = compute_a_s(x, sigma, n_e, q_ger)

    n_s = 1.0 - 2.0 / n_e
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)
    tensor_running_ratio = r_tensor / (-alpha_spec)
    quadrupole_handle = p_ger * p_ger
    hemispherical_handle = 2.0 * q_ger / sigma
    large_angle_fractional = q_ger / sigma

    return CascadeState(
        sin2_theta_w=sin2_theta_w,
        delta=delta,
        d_eff=d_eff,
        x=x,
        sigma=sigma,
        contraction=contraction,
        gamma_eff=gamma_eff,
        n_e=n_e,
        q_total=q_total,
        q_source=q_source,
        q_phase=q_phase,
        q_ger=q_ger,
        p_phase=p_phase,
        p_ger=p_ger,
        total_to_source_gain=total_to_source_gain,
        raw_as=raw_as,
        source_as=source_as,
        phase_as=phase_as,
        ger_as=ger_as,
        raw_pull=pull(raw_as, A_S_REF, A_S_SIGMA),
        ger_pull=pull(ger_as, A_S_REF, A_S_SIGMA),
        n_s=n_s,
        alpha_spec=alpha_spec,
        r_tensor=r_tensor,
        tensor_running_ratio=tensor_running_ratio,
        quadrupole_handle=quadrupole_handle,
        hemispherical_handle=hemispherical_handle,
        large_angle_fractional=large_angle_fractional,
    )


def invariant_rows(state: CascadeState) -> list[InvariantRow]:
    return [
        InvariantRow(
            name="raw gain",
            equation="Q_total/Q_source = 1/(1-D_eff x)",
            value=state.total_to_source_gain,
            status="algebraic source of raw A_s overshoot",
            guardrail="do not use total susceptibility as scalar amplitude readout",
        ),
        InvariantRow(
            name="half-cycle projection",
            equation="P_phase = 2/pi",
            value=state.p_phase,
            status="shared d0->d3 projection candidate",
            guardrail="projection alone undershoots A_s; it is not the final scalar readout",
        ),
        InvariantRow(
            name="GER projection",
            equation="P_GER = (2/pi) sigma^(D_eff/(D_eff+1))",
            value=state.p_ger,
            status="single projection reused by A_s and large-angle handles",
            guardrail="selection candidate, not exact theorem",
        ),
        InvariantRow(
            name="A_s GER pull",
            equation="pull(A_s[Q_GER])",
            value=state.ger_pull,
            status="inside broad scalar amplitude gate",
            guardrail="must survive running/tensor/common-readout tests",
        ),
        InvariantRow(
            name="tensor-running lock",
            equation="r_tensor/(-alpha_spec)",
            value=state.tensor_running_ratio,
            status="exact N_e-family ratio",
            guardrail="open until joint primitive spectrum likelihood",
        ),
        InvariantRow(
            name="hemispherical identity",
            equation="2 Q_GER/sigma = 2 P_GER x",
            value=state.hemispherical_handle,
            status="large-angle amplitude handle",
            guardrail="does not select a preferred axis by itself",
        ),
    ]


def write_outputs(state: CascadeState, rows: list[InvariantRow]) -> None:
    payload = {
        "state": asdict(state),
        "rows": [asdict(row) for row in rows],
        "passed": abs(state.ger_pull) < 3.0 and state.raw_pull > 10.0 and abs(state.tensor_running_ratio - 6.0) < 1e-12,
        "verdict": (
            "The scalar residual cascade is coherent: raw total sensitivity is rejected, "
            "while a single GER projection reuses the same source residual across A_s "
            "and large-angle amplitude handles."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Residual Cascade Invariant Gate",
        "",
        "## Cascade state",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| D_eff | {state.d_eff:.8f} |",
        f"| x | {state.x:.8f} |",
        f"| sigma | {state.sigma:.8f} |",
        f"| contraction D_eff*x | {state.contraction:.8f} |",
        f"| gamma_eff | {state.gamma_eff:.8f} |",
        f"| N_e | {state.n_e:.8f} |",
        f"| Q_total | {state.q_total:.8f} |",
        f"| Q_source | {state.q_source:.8f} |",
        f"| Q_phase | {state.q_phase:.8f} |",
        f"| Q_GER | {state.q_ger:.8f} |",
        "",
        "## A_s cascade",
        "",
        "| readout | A_s | pull | status |",
        "|---|---:|---:|---|",
        f"| total susceptibility | {state.raw_as:.8e} | {state.raw_pull:+.2f} | rejected |",
        f"| source residual | {state.source_as:.8e} | {pull(state.source_as, A_S_REF, A_S_SIGMA):+.2f} | source only |",
        f"| half-cycle source | {state.phase_as:.8e} | {pull(state.phase_as, A_S_REF, A_S_SIGMA):+.2f} | undershoot |",
        f"| GER source | {state.ger_as:.8e} | {state.ger_pull:+.2f} | selection candidate |",
        "",
        "## Invariants",
        "",
        "| name | equation | value | status | guardrail |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row.name} | {row.equation} | {row.value:.8f} | {row.status} | {row.guardrail} |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
            "The next hard test is data-facing: running, tensor, and CMB large-angle likelihoods must reuse this same cascade without adding observable-specific recursion.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    state = cascade_state()
    rows = invariant_rows(state)
    write_outputs(state, rows)

    print("# Residual Cascade Invariant Gate")
    print()
    print(f"raw A_s pull = {state.raw_pull:+.2f}")
    print(f"GER A_s pull = {state.ger_pull:+.2f}")
    print(f"Q_total/Q_source = {state.total_to_source_gain:.8f}")
    print(f"P_GER = {state.p_ger:.8f}")
    print(f"r_tensor/(-alpha_spec) = {state.tensor_running_ratio:.8f}")
    print()
    print("| invariant | value | status |")
    print("|---|---:|---|")
    for row in rows:
        print(f"| {row.name} | {row.value:.8f} | {row.status} |")
    print()
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")

    if abs(state.ger_pull) > 3.0:
        raise SystemExit("GER scalar readout drifted outside the broad A_s gate")
    if state.raw_pull < 10.0:
        raise SystemExit("raw total susceptibility should remain rejected")
    if abs(state.tensor_running_ratio - 6.0) > 1e-12:
        raise SystemExit("tensor-running lock changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
