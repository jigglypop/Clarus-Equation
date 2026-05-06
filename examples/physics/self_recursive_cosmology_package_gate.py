"""Package gate for the self-recursive cosmology research thread.

This gate consolidates the recent self-reference audits into one promotion
table.  It deliberately does not promote every successful numerical relation
to Exact.  The package status is:

* core kernel: closed/minimal, but deformation is blocked without derivation
* d=0: boundary/measure transport, not a physical state
* residual cascade: Selection candidate
* H0 q selector: channel-corrected Bridge
* early-late horizon lift: channel-corrected Bridge
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
OMEGA_B_OBS = 0.0493
OMEGA_B_SIGMA = 0.0004
N_S_OBS = 0.9649
N_S_SIGMA = 0.0042
MPC_KM = 3.0856775814913673e19
T_PLANCK_S = 5.391247e-44
RESULT_JSON = Path("examples/physics/self_recursive_cosmology_package_results.json")
REPORT_MD = Path("examples/physics/self_recursive_cosmology_package_report.md")


@dataclass(frozen=True)
class CoreState:
    d_eff: float
    x: float
    sigma: float
    residual: float
    contraction: float
    n_e: float
    phase_area: float
    endpoint_defect: float
    integrated_defect: float
    h0_global: float
    h0_endpoint: float
    branch_gap: float
    q_ger: float
    a_s_raw_pull: float
    a_s_ger_pull: float
    tensor_running_ratio: float
    d0_entropy_identity_error: float
    kernel_minimal_aic: float
    kernel_tuned_aic: float
    q_selector_chi2: float
    q_selector_dof: int
    early_late_chi2: float
    early_late_dof: int


@dataclass(frozen=True)
class PackageRow:
    lever: str
    layer: str
    status: str
    evidence: str
    guardrail: str
    next_gate: str


def bootstrap_x(d_eff: float, c_kernel: float = 1.0, kappa: float = 0.0, tol: float = 1e-15) -> float:
    x = math.exp(-c_kernel * d_eff - kappa)
    for _ in range(500):
        nxt = math.exp(-c_kernel * d_eff * (1.0 - x) - kappa * (1.0 - x) ** 2)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    raise RuntimeError("bootstrap_x did not converge")


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def h0_from_log_s(log_s: float) -> float:
    h_s = math.sqrt(math.pi) * math.exp(-0.5 * log_s) / T_PLANCK_S
    return h_s * MPC_KM


def d_eff() -> float:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    return D_SPATIAL + sin2_theta_w * (1.0 - sin2_theta_w)


def model_aic(c_kernel: float, kappa: float, fitted_parameters: int) -> float:
    d = d_eff()
    x = bootstrap_x(d, c_kernel, kappa)
    sigma = 1.0 - x
    n_e = (D_SPATIAL / 2.0) * d * N_GAUGE
    n_s = 1.0 - 2.0 / n_e
    gamma_eff = d / (d + 1.0)
    q_ger = (2.0 / math.pi) * sigma**gamma_eff * x * sigma
    a_s_ger = compute_a_s(x, sigma, n_e, q_ger)
    chi2 = (
        pull(x, OMEGA_B_OBS, OMEGA_B_SIGMA) ** 2
        + pull(n_s, N_S_OBS, N_S_SIGMA) ** 2
        + pull(a_s_ger, A_S_REF, A_S_SIGMA) ** 2
    )
    return chi2 + 2.0 * fitted_parameters


def tuned_c_to_omega_b() -> float:
    d = d_eff()
    return -math.log(OMEGA_B_OBS) / (d * (1.0 - OMEGA_B_OBS))


def q_selector_chi2() -> tuple[float, int]:
    d = d_eff()
    x = bootstrap_x(d)
    sigma = 1.0 - x
    delta = d - D_SPATIAL
    endpoint_defect = delta * sigma
    phase_area = math.pi * math.pi / 2.0
    n_e = (D_SPATIAL / 2.0) * d * N_GAUGE
    log_s_global = phase_area * n_e - math.pi * endpoint_defect

    def log_s_from_h0(h0_km_s_mpc: float) -> float:
        h0_s = h0_km_s_mpc / MPC_KM
        return math.log(math.pi / (h0_s * T_PLANCK_S) ** 2)

    # From the self-reference and early-late preservation gates.
    # Rows are q_graph, h0_obs, sigma_h0.
    rows = [
        (0.0, 67.4, 0.5),
        (0.25, 68.51, 0.58),
        (0.10, 67.80, math.hypot(2.17, 1.64)),
        (0.50, 70.39, math.sqrt(1.22**2 + 1.33**2 + 0.70**2)),
        (1.00, 73.17, 0.86),
        (0.25, 67.4, 3.65),
        (1.00, 73.9, 3.0),
        (0.50, 70.3, 5.15),
    ]
    chi2 = 0.0
    for q_graph, h0_obs, sigma_h0 in rows:
        q_obs = (log_s_global - log_s_from_h0(h0_obs)) / endpoint_defect
        sigma_q = 2.0 * sigma_h0 / (h0_obs * endpoint_defect)
        chi2 += ((q_graph - q_obs) / sigma_q) ** 2
    return chi2, len(rows)


def core_state() -> CoreState:
    d = d_eff()
    x = bootstrap_x(d)
    sigma = 1.0 - x
    residual = x - math.exp(-(1.0 - x) * d)
    contraction = d * x
    n_e = (D_SPATIAL / 2.0) * d * N_GAUGE
    phase_area = math.pi * math.pi / 2.0
    delta = d - D_SPATIAL
    endpoint_defect = delta * sigma
    integrated_defect = math.pi * endpoint_defect
    log_s_global = phase_area * n_e - integrated_defect
    h0_global = h0_from_log_s(log_s_global)
    h0_endpoint = h0_from_log_s(log_s_global - endpoint_defect)

    q_total = x * sigma / (1.0 - contraction)
    gamma_eff = d / (d + 1.0)
    q_ger = (2.0 / math.pi) * sigma**gamma_eff * x * sigma
    a_s_raw = compute_a_s(x, sigma, n_e, q_total)
    a_s_ger = compute_a_s(x, sigma, n_e, q_ger)
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)
    s_recursive = -math.log(x)
    minimal_aic = model_aic(1.0, 0.0, 0)
    tuned_aic = model_aic(tuned_c_to_omega_b(), 0.0, 1)
    selector_chi2, selector_dof = q_selector_chi2()
    return CoreState(
        d_eff=d,
        x=x,
        sigma=sigma,
        residual=residual,
        contraction=contraction,
        n_e=n_e,
        phase_area=phase_area,
        endpoint_defect=endpoint_defect,
        integrated_defect=integrated_defect,
        h0_global=h0_global,
        h0_endpoint=h0_endpoint,
        branch_gap=h0_endpoint - h0_global,
        q_ger=q_ger,
        a_s_raw_pull=pull(a_s_raw, A_S_REF, A_S_SIGMA),
        a_s_ger_pull=pull(a_s_ger, A_S_REF, A_S_SIGMA),
        tensor_running_ratio=r_tensor / (-alpha_spec),
        d0_entropy_identity_error=s_recursive - d * sigma,
        kernel_minimal_aic=minimal_aic,
        kernel_tuned_aic=tuned_aic,
        q_selector_chi2=selector_chi2,
        q_selector_dof=selector_dof,
        early_late_chi2=selector_chi2,
        early_late_dof=selector_dof,
    )


def package_rows(s: CoreState) -> list[PackageRow]:
    return [
        PackageRow(
            lever="minimal fixed-point kernel",
            layer="core kernel",
            status="Closed/minimal",
            evidence=f"residual={s.residual:+.1e}, tuned AIC {s.kernel_tuned_aic:.3f} > minimal AIC {s.kernel_minimal_aic:.3f}",
            guardrail="no c/kappa deformation without independent derivation",
            next_gate="kernel derivation only, not observable tuning",
        ),
        PackageRow(
            lever="d0 measure transport",
            layer="boundary",
            status="Boundary principle",
            evidence=f"S_R identity error={s.d0_entropy_identity_error:+.1e}, contraction={s.contraction:.8f}",
            guardrail="do not call d=0 a physical location or reachable state",
            next_gate="FLRW/reheating/horizon scale lift",
        ),
        PackageRow(
            lever="residual cascade",
            layer="readout",
            status="Selection candidate",
            evidence=f"raw A_s pull={s.a_s_raw_pull:+.2f}, GER A_s pull={s.a_s_ger_pull:+.2f}",
            guardrail="no observable-specific recursion; same cascade must serve running/tensor/CMB handles",
            next_gate="joint primitive-spectrum and CMB large-angle likelihood",
        ),
        PackageRow(
            lever="H0 q-selector",
            layer="selector",
            status="Channel-corrected Bridge",
            evidence=f"q-space chi2/dof={s.q_selector_chi2:.3f}/{s.q_selector_dof}",
            guardrail="q must be assigned from source/covariance graph before H0 comparison",
            next_gate="real covariance/Fisher edge ingest",
        ),
        PackageRow(
            lever="early-late phase measure",
            layer="horizon bridge",
            status="Channel-corrected Bridge",
            evidence=f"phase_area={s.phase_area:.8f}, early-late chi2/dof={s.early_late_chi2:.3f}/{s.early_late_dof}",
            guardrail="not local slow-roll entropy growth and not Exact",
            next_gate="dynamical derivation of late horizon readout",
        ),
    ]


def write_outputs(s: CoreState, rows: list[PackageRow]) -> None:
    payload = {
        "core": asdict(s),
        "rows": [asdict(row) for row in rows],
        "package_status": "Selection/Bridge package, not Exact",
        "passed": (
            abs(s.residual) < 1e-12
            and abs(s.d0_entropy_identity_error) < 1e-12
            and s.a_s_raw_pull > 10.0
            and abs(s.a_s_ger_pull) < 3.0
            and s.kernel_tuned_aic > s.kernel_minimal_aic
            and s.q_selector_chi2 / s.q_selector_dof < 1.0
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Self-Recursive Cosmology Package Gate",
        "",
        "## Core numbers",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| D_eff | {s.d_eff:.8f} |",
        f"| x | {s.x:.8f} |",
        f"| sigma | {s.sigma:.8f} |",
        f"| fixed-point residual | {s.residual:+.3e} |",
        f"| contraction | {s.contraction:.8f} |",
        f"| N_e | {s.n_e:.8f} |",
        f"| H0 global q=0 | {s.h0_global:.6f} |",
        f"| H0 endpoint q=1 | {s.h0_endpoint:.6f} |",
        f"| branch gap | {s.branch_gap:.6f} |",
        f"| Q_GER | {s.q_ger:.8f} |",
        "",
        "## Promotion table",
        "",
        "| lever | layer | status | evidence | guardrail | next gate |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.lever} | {row.layer} | {row.status} | {row.evidence} | "
            f"{row.guardrail} | {row.next_gate} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "The package is a Selection/Bridge package, not Exact.  The safe use of self-reference is now concentrated in readout/selector/measure-preservation layers.  Kernel deformation is blocked unless derived before data contact.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    s = core_state()
    rows = package_rows(s)
    write_outputs(s, rows)

    print("# Self-Recursive Cosmology Package Gate")
    print()
    print(f"fixed-point residual = {s.residual:+.3e}")
    print(f"d0 entropy identity error = {s.d0_entropy_identity_error:+.3e}")
    print(f"raw/GER A_s pulls = {s.a_s_raw_pull:+.2f}, {s.a_s_ger_pull:+.2f}")
    print(f"kernel AIC minimal/tuned = {s.kernel_minimal_aic:.3f}, {s.kernel_tuned_aic:.3f}")
    print(f"q selector chi2/dof = {s.q_selector_chi2:.3f}/{s.q_selector_dof}")
    print()
    print("| lever | status | next gate |")
    print("|---|---|---|")
    for row in rows:
        print(f"| {row.lever} | {row.status} | {row.next_gate} |")
    print()
    print("Verdict: Selection/Bridge package, not Exact.")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")

    if s.kernel_tuned_aic <= s.kernel_minimal_aic:
        raise SystemExit("kernel deformation unexpectedly beat minimal kernel")
    if abs(s.a_s_ger_pull) > 3.0:
        raise SystemExit("GER residual cascade drifted outside broad gate")
    if s.q_selector_chi2 / s.q_selector_dof > 1.0:
        raise SystemExit("q selector package degraded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
