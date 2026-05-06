"""Audit remaining self-recursive leverage in the cosmology package.

The current cosmology scripts already use recursion in several different
roles: fixed-point kernel, boundary condition, residual readout, and channel
selector.  This audit keeps those layers separate and records where additional
self-reference is still useful without promoting bridge assumptions to exact
theorems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
MPC_KM = 3.0856775814913673e19
T_PLANCK_S = 5.391247e-44
RESULT_JSON = Path("examples/physics/recursive_cosmology_research_audit_results.json")
REPORT_MD = Path("examples/physics/recursive_cosmology_research_audit_report.md")


@dataclass(frozen=True)
class RecursiveState:
    sin2_theta_w: float
    delta: float
    d_eff: float
    x: float
    sigma: float
    residual: float
    contraction: float
    n_e: float
    phase_area: float
    integrated_boundary_defect: float
    endpoint_defect: float
    h0_global: float
    h0_endpoint: float
    h0_branch_gap: float


@dataclass(frozen=True)
class AuditRow:
    name: str
    layer: str
    current_status: str
    safe_extension: str
    overreach_guard: str
    next_gate: str
    priority: int


def bootstrap_x(d_eff: float, tol: float = 1e-15, max_iter: int = 500) -> float:
    x = math.exp(-d_eff)
    for _ in range(max_iter):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    raise RuntimeError("bootstrap_x did not converge")


def h0_from_log_s(log_s: float) -> float:
    h_s = math.sqrt(math.pi) * math.exp(-0.5 * log_s) / T_PLANCK_S
    return h_s * MPC_KM


def core_state() -> RecursiveState:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    residual = x - math.exp(-(1.0 - x) * d_eff)
    contraction = d_eff * x
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = math.pi * math.pi / 2.0
    endpoint_defect = delta * sigma
    integrated_boundary_defect = math.pi * endpoint_defect
    log_s_global = phase_area * n_e - integrated_boundary_defect
    log_s_endpoint = log_s_global - endpoint_defect
    h0_global = h0_from_log_s(log_s_global)
    h0_endpoint = h0_from_log_s(log_s_endpoint)
    return RecursiveState(
        sin2_theta_w=sin2_theta_w,
        delta=delta,
        d_eff=d_eff,
        x=x,
        sigma=sigma,
        residual=residual,
        contraction=contraction,
        n_e=n_e,
        phase_area=phase_area,
        integrated_boundary_defect=integrated_boundary_defect,
        endpoint_defect=endpoint_defect,
        h0_global=h0_global,
        h0_endpoint=h0_endpoint,
        h0_branch_gap=h0_endpoint - h0_global,
    )


def audit_rows() -> list[AuditRow]:
    return [
        AuditRow(
            name="fixed-point kernel",
            layer="kernel",
            current_status="closed mathematical fixed point",
            safe_extension="test constrained deformations of K(x) before adding observables",
            overreach_guard="do not change the kernel per observable",
            next_gate="kernel_deformation_no_free_parameter_gate",
            priority=2,
        ),
        AuditRow(
            name="d0 zero-residual boundary",
            layer="boundary",
            current_status="candidate boundary condition",
            safe_extension="derive measure map from d0 identity to d3 contracted branch",
            overreach_guard="do not call d0 a physical place or reachable state",
            next_gate="d0_measure_transport_gate",
            priority=3,
        ),
        AuditRow(
            name="A_s / A3c residual readout",
            layer="readout",
            current_status="selection candidate; raw total sensitivity rejected",
            safe_extension="reuse the same projected residual in spectrum and anomaly handles",
            overreach_guard="do not mark A3c exact before likelihood or n_i derivation",
            next_gate="primitive_spectrum_common_readout_gate",
            priority=2,
        ),
        AuditRow(
            name="horizon phase-area lift",
            layer="bridge",
            current_status="conditional bridge",
            safe_extension="derive why late horizon entropy reads primordial phase area",
            overreach_guard="do not treat pi^2/2 lift as local slow-roll entropy growth",
            next_gate="early_late_measure_preservation_gate",
            priority=3,
        ),
        AuditRow(
            name="H0 q-selector",
            layer="selector",
            current_status="strongest open recursive lever",
            safe_extension="predict q from source/covariance graph before reading H0",
            overreach_guard="do not patch high H0 branch without selector derivation",
            next_gate="prospective_covariance_graph_selector_gate",
            priority=1,
        ),
        AuditRow(
            name="residual contraction cascade",
            layer="cross-observable residual",
            current_status="open",
            safe_extension="ask whether failed raw terms share a contraction/projection rule",
            overreach_guard="do not tune a separate recursion for each residual",
            next_gate="residual_cascade_invariant_gate",
            priority=1,
        ),
    ]


def verdict(rows: list[AuditRow]) -> str:
    top = [row.name for row in rows if row.priority == 1]
    return (
        "More self-recursion remains usable, but mainly as selector/readout "
        f"recursion rather than a new core fixed point. Top targets: {', '.join(top)}."
    )


def write_report(state: RecursiveState, rows: list[AuditRow]) -> None:
    payload = {
        "core_state": asdict(state),
        "rows": [asdict(row) for row in rows],
        "verdict": verdict(rows),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Recursive Cosmology Research Audit",
        "",
        "## Core recursive state",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| sin2_theta_w | {state.sin2_theta_w:.8f} |",
        f"| delta | {state.delta:.8f} |",
        f"| D_eff | {state.d_eff:.8f} |",
        f"| x | {state.x:.8f} |",
        f"| sigma | {state.sigma:.8f} |",
        f"| fixed-point residual | {state.residual:+.3e} |",
        f"| contraction D_eff*x | {state.contraction:.8f} |",
        f"| N_e | {state.n_e:.8f} |",
        f"| phase area pi^2/2 | {state.phase_area:.8f} |",
        f"| integrated boundary defect pi*delta*sigma | {state.integrated_boundary_defect:.8f} |",
        f"| endpoint defect delta*sigma | {state.endpoint_defect:.8f} |",
        f"| H0 global q=0 | {state.h0_global:.6f} |",
        f"| H0 endpoint q=1 | {state.h0_endpoint:.6f} |",
        f"| branch gap | {state.h0_branch_gap:.6f} |",
        "",
        "## Remaining recursive leverage",
        "",
        "| priority | name | layer | current status | safe extension | guardrail | next gate |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (item.priority, item.name)):
        lines.append(
            f"| {row.priority} | {row.name} | {row.layer} | {row.current_status} | "
            f"{row.safe_extension} | {row.overreach_guard} | {row.next_gate} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict(rows),
            "",
            "The next useful calculation is not another unconstrained correction term. "
            "It is a prospective selector/residual audit: predict the readout layer first, "
            "then compare the observable.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    state = core_state()
    rows = audit_rows()
    write_report(state, rows)

    print("# Recursive Cosmology Research Audit")
    print()
    print(f"fixed-point residual = {state.residual:+.3e}")
    print(f"contraction D_eff*x = {state.contraction:.8f}")
    print(f"H0 global q=0 = {state.h0_global:.6f} km/s/Mpc")
    print(f"H0 endpoint q=1 = {state.h0_endpoint:.6f} km/s/Mpc")
    print(f"branch gap = {state.h0_branch_gap:.6f} km/s/Mpc")
    print()
    print("| priority | recursive lever | next gate |")
    print("|---:|---|---|")
    for row in sorted(rows, key=lambda item: (item.priority, item.name)):
        print(f"| {row.priority} | {row.name} | {row.next_gate} |")
    print()
    print("Verdict:", verdict(rows))
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
