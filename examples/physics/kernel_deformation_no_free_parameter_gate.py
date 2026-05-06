"""No-free-parameter audit for fixed-point kernel deformations.

Self-recursion can be pushed into the bootstrap kernel only if the deformation
is constrained before observables are fitted.  This gate compares the minimal
CE kernel

    x = exp[-D_eff(1-x)]

against simple one-parameter deformations

    x = exp[-c D_eff(1-x)]
    x = exp[-D_eff(1-x) - kappa(1-x)^2]

The aim is not to find a better fit.  It is to show what becomes an empirical
fit parameter and therefore cannot be promoted as a self-recursive theorem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
OMEGA_B_OBS = 0.0493
OMEGA_B_SIGMA = 0.0004
N_S_OBS = 0.9649
N_S_SIGMA = 0.0042
A_S_OBS = 2.10e-9
A_S_SIGMA = 0.03e-9
RESULT_JSON = Path("examples/physics/kernel_deformation_no_free_parameter_results.json")
REPORT_MD = Path("examples/physics/kernel_deformation_no_free_parameter_report.md")


@dataclass(frozen=True)
class ModelRow:
    name: str
    c_kernel: float
    kappa: float
    fitted_parameters: int
    x: float
    n_s: float
    a_s_ger: float
    omega_b_pull: float
    n_s_pull: float
    a_s_ger_pull: float
    chi2: float
    aic: float
    guardrail: str


def bootstrap_x(d_eff: float, c_kernel: float = 1.0, kappa: float = 0.0, tol: float = 1e-15) -> float:
    x = math.exp(-c_kernel * d_eff - kappa)
    for _ in range(1000):
        nxt = math.exp(-c_kernel * d_eff * (1.0 - x) - kappa * (1.0 - x) ** 2)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    raise RuntimeError("bootstrap_x did not converge")


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def d_eff() -> float:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    return D_SPATIAL + delta


def row(
    name: str,
    c_kernel: float,
    kappa: float,
    fitted_parameters: int,
    guardrail: str,
) -> ModelRow:
    d = d_eff()
    x = bootstrap_x(d, c_kernel, kappa)
    sigma = 1.0 - x
    n_e = (D_SPATIAL / 2.0) * d * N_GAUGE
    n_s = 1.0 - 2.0 / n_e
    gamma_eff = d / (d + 1.0)
    p_ger = (2.0 / math.pi) * sigma**gamma_eff
    q_ger = p_ger * x * sigma
    a_s_ger = compute_a_s(x, sigma, n_e, q_ger)
    pulls = [
        pull(x, OMEGA_B_OBS, OMEGA_B_SIGMA),
        pull(n_s, N_S_OBS, N_S_SIGMA),
        pull(a_s_ger, A_S_OBS, A_S_SIGMA),
    ]
    chi2 = sum(item * item for item in pulls)
    aic = chi2 + 2.0 * fitted_parameters
    return ModelRow(
        name=name,
        c_kernel=c_kernel,
        kappa=kappa,
        fitted_parameters=fitted_parameters,
        x=x,
        n_s=n_s,
        a_s_ger=a_s_ger,
        omega_b_pull=pulls[0],
        n_s_pull=pulls[1],
        a_s_ger_pull=pulls[2],
        chi2=chi2,
        aic=aic,
        guardrail=guardrail,
    )


def tune_c_to_omega_b() -> float:
    d = d_eff()
    return -math.log(OMEGA_B_OBS) / (d * (1.0 - OMEGA_B_OBS))


def tune_kappa_to_omega_b() -> float:
    d = d_eff()
    c_kernel = 1.0
    return (-math.log(OMEGA_B_OBS) - c_kernel * d * (1.0 - OMEGA_B_OBS)) / ((1.0 - OMEGA_B_OBS) ** 2)


def scan_rows() -> list[ModelRow]:
    tuned_c = tune_c_to_omega_b()
    tuned_kappa = tune_kappa_to_omega_b()
    return [
        row(
            "CE minimal kernel",
            c_kernel=1.0,
            kappa=0.0,
            fitted_parameters=0,
            guardrail="allowed: kernel fixed before observables",
        ),
        row(
            "c tuned to Omega_b",
            c_kernel=tuned_c,
            kappa=0.0,
            fitted_parameters=1,
            guardrail="fit parameter: cannot be promoted without derivation",
        ),
        row(
            "kappa tuned to Omega_b",
            c_kernel=1.0,
            kappa=tuned_kappa,
            fitted_parameters=1,
            guardrail="fit parameter: interaction term must be independently derived",
        ),
    ]


def write_outputs(rows: list[ModelRow]) -> None:
    minimal = rows[0]
    best_aic = min(row.aic for row in rows)
    payload = {
        "rows": [asdict(item) for item in rows],
        "best_aic": best_aic,
        "minimal_delta_aic": minimal.aic - best_aic,
        "verdict": (
            "Kernel deformation is not the next safe self-recursive lever.  "
            "One-parameter variants can tune Omega_b, but the gain is a fit "
            "parameter unless c or kappa is fixed by an independent theorem."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kernel Deformation No-Free-Parameter Gate",
        "",
        "## Model comparison",
        "",
        "| model | c | kappa | fitted params | x | Omega_b pull | n_s pull | A_s GER pull | chi2 | AIC | guardrail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item.name} | {item.c_kernel:.8f} | {item.kappa:.8f} | {item.fitted_parameters} | "
            f"{item.x:.8f} | {item.omega_b_pull:+.2f} | {item.n_s_pull:+.2f} | "
            f"{item.a_s_ger_pull:+.2f} | {item.chi2:.3f} | {item.aic:.3f} | {item.guardrail} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
            "This keeps the core recursion conservative: use kernel deformation only after a no-free-parameter derivation, not as another observable readout correction.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = scan_rows()
    write_outputs(rows)
    minimal = rows[0]
    best = min(rows, key=lambda item: item.aic)

    print("# Kernel Deformation No-Free-Parameter Gate")
    print()
    print("| model | c | kappa | params | Omega_b pull | n_s pull | A_s GER pull | chi2 | AIC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in rows:
        print(
            f"| {item.name} | {item.c_kernel:.8f} | {item.kappa:.8f} | {item.fitted_parameters} | "
            f"{item.omega_b_pull:+.2f} | {item.n_s_pull:+.2f} | {item.a_s_ger_pull:+.2f} | "
            f"{item.chi2:.3f} | {item.aic:.3f} |"
        )
    print()
    print(f"best AIC model = {best.name}")
    print(f"minimal kernel delta AIC = {minimal.aic - best.aic:+.3f}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")

    if best.fitted_parameters > 0 and minimal.aic - best.aic < 2.0:
        print("Verdict: tuned deformation is not worth promoting over the no-free-parameter kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
