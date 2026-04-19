"""Quantum phase evolution (12_Equation.md 1.5 / 4.1).

psi_{k+1} = exp(-i E dt) psi_k  (quantum form)
Wick rotation t -> -i*tau yields real relaxation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

ALPHA_B_DEFAULT = math.exp(1.0 / 3.0) * (math.pi ** (1.0 / 3.0))


def quantum_phase_step(
    psi: torch.Tensor,
    energy: float,
    dt: float = 0.01,
) -> torch.Tensor:
    """psi_{k+1} = exp(-i*E*dt) * psi_k. Complex state evolution."""
    if not psi.is_complex():
        psi = torch.complex(psi, torch.zeros_like(psi))
    phase = torch.tensor(-energy * dt)
    rotation = torch.complex(torch.cos(phase), torch.sin(phase))
    return psi * rotation


def wick_rotate(
    psi: torch.Tensor,
    energy: float,
    dt: float = 0.01,
) -> torch.Tensor:
    """Euclidean rotation t -> -i*tau: exp(-E*dt) * psi (real damping)."""
    if psi.is_complex():
        return psi * torch.exp(torch.tensor(-energy * dt))
    return psi * torch.exp(torch.tensor(-energy * dt))


def quantum_to_real(psi: torch.Tensor) -> torch.Tensor:
    """Project complex state to real for classical relaxation."""
    if psi.is_complex():
        return psi.real
    return psi


def check_norm_conservation(psi_before: torch.Tensor, psi_after: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify unitary evolution preserves norm."""
    norm_before = psi_before.abs().norm()
    norm_after = psi_after.abs().norm()
    return bool(torch.abs(norm_before - norm_after).item() < tol)


def convergence_inequality(
    grad_norm: float,
    c_k: float,
    phi_norm: float,
    alpha_b: float = ALPHA_B_DEFAULT,
) -> bool:
    """Pointwise sufficient condition: ||grad E|| > C_k * ||phi|| / alpha_b.

    Reference: docs/7_AGI/12_Equation.md 4.7 (gate F2). Scope: pointwise
    monotone-energy region; not a global convergence guarantee.
    """
    return grad_norm > c_k * phi_norm / alpha_b


def time_curvature(m_history: Sequence[torch.Tensor]) -> float:
    """C_k = ||m_k - 2 m_{k-1} + m_{k-2}|| (docs/7_AGI/12_Equation.md 1.5).

    Returns 0.0 when fewer than 3 samples are available.
    """
    if len(m_history) < 3:
        return 0.0
    m_k, m_km1, m_km2 = m_history[-1], m_history[-2], m_history[-3]
    return float((m_k - 2.0 * m_km1 + m_km2).norm().item())


def estimate_mu(
    residuals: Sequence[float],
    *,
    dt_over_tau: float,
    skip: int = 0,
) -> float:
    """Estimate local Hessian floor mu from observed residual contraction.

    Near an attractor, the gradient flow gives
        ||m_{k+1} - m*|| approx (1 - mu * dt/tau) * ||m_k - m*||,
    so a least-squares fit on log residuals returns
        mu approx (1 - mean(r_{k+1}/r_k)) * tau / dt.

    Args:
        residuals: sequence of ||m_k - m*|| values (must be > 0).
        dt_over_tau: integration step ratio dt/tau used by `relax`.
        skip: optional warm-up steps to drop from the head of the sequence.

    Returns:
        mu estimate. Returns 0.0 if the sequence is too short or expanding.
    """
    if dt_over_tau <= 0.0:
        raise ValueError("dt_over_tau must be positive")
    seq = [float(r) for r in residuals[skip:] if r > 0.0]
    if len(seq) < 3:
        return 0.0
    log_ratios = []
    for prev, curr in zip(seq[:-1], seq[1:]):
        ratio = curr / prev
        if ratio <= 0.0 or ratio >= 1.0:
            continue
        log_ratios.append(math.log(ratio))
    if not log_ratios:
        return 0.0
    contraction = math.exp(sum(log_ratios) / len(log_ratios))
    mu = (1.0 - contraction) / dt_over_tau
    return max(mu, 0.0)


def iss_ball_radius(
    *,
    c_k_max: float,
    phi_inf_norm: float,
    mu: float,
    alpha_b: float = ALPHA_B_DEFAULT,
) -> float:
    """ISS ball radius for the bypass-driven memory dynamics (gate F2).

    Reference: docs/7_AGI/12_Equation.md appendix A.1. With dm/dt =
    -nabla_m E / tau + d(t) and ||d||_inf <= C_k_max * ||phi||_inf / (tau * alpha_b),
    the Hessian floor mu yields the closed-form bound

        limsup ||m - m*|| <= C_k_max * ||phi||_inf / (mu * alpha_b).

    Args:
        c_k_max: empirical maximum of trajectory time-curvature C_k.
        phi_inf_norm: empirical sup-norm of the residue field phi.
        mu: local Hessian floor (use `estimate_mu` from contraction history).
        alpha_b: bypass denominator e^{1/3} pi^{1/3}.

    Returns:
        ISS ball radius. Returns +inf if mu == 0.
    """
    if mu <= 0.0:
        return float("inf")
    return c_k_max * phi_inf_norm / (mu * alpha_b)


def pci_regression(
    stability: Sequence[float],
    pci: Sequence[float],
) -> dict:
    """OLS regression PCI ~ alpha * stability + beta (gate F4, A.4).

    Returns alpha, beta, R^2, sample count and Pearson r. With less than 3
    samples or zero stability variance, returns NaNs and zero R^2 so callers
    can detect insufficient data without exception handling.

    Stability is the metacognitive scalar exp(-c_d * d_tau); see
    clarus/agent.py::ConsciousnessMonitor and 12_Equation.md A.4.
    """
    xs = [float(v) for v in stability]
    ys = [float(v) for v in pci]
    n = min(len(xs), len(ys))
    if n < 3:
        return {"n": n, "alpha": float("nan"), "beta": float("nan"), "r2": 0.0, "pearson_r": float("nan")}
    xs = xs[:n]
    ys = ys[:n]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sx2 = sum((x - mean_x) ** 2 for x in xs)
    sy2 = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sx2 <= 0.0:
        return {"n": n, "alpha": float("nan"), "beta": float("nan"), "r2": 0.0, "pearson_r": float("nan")}
    alpha = sxy / sx2
    beta = mean_y - alpha * mean_x
    r2 = (sxy * sxy) / (sx2 * sy2) if sy2 > 0.0 else 0.0
    pearson_r = sxy / math.sqrt(sx2 * sy2) if sy2 > 0.0 else float("nan")
    return {"n": n, "alpha": alpha, "beta": beta, "r2": r2, "pearson_r": pearson_r}


def iss_report(
    m_history: Sequence[torch.Tensor],
    phi: torch.Tensor,
    *,
    dt_over_tau: float,
    m_star: torch.Tensor | None = None,
    alpha_b: float = ALPHA_B_DEFAULT,
) -> dict:
    """End-to-end gate F2 measurement from a relaxation trajectory.

    Args:
        m_history: list of m_k tensors recorded during `relax`.
        phi: residue field tensor at end of relaxation (or any reference point).
        dt_over_tau: integrator step ratio.
        m_star: optional fixed point estimate. Defaults to mean of last
            quarter of the trajectory.
        alpha_b: bypass denominator.

    Returns:
        Dict with `c_k_max`, `phi_inf_norm`, `mu`, `iss_ball_radius`,
        `residuals`, `samples`. Suitable for direct logging.
    """
    if not m_history:
        return {
            "samples": 0,
            "c_k_max": 0.0,
            "phi_inf_norm": 0.0,
            "mu": 0.0,
            "iss_ball_radius": float("inf"),
        }

    if m_star is None:
        tail = m_history[max(1, 3 * len(m_history) // 4):]
        if not tail:
            tail = m_history[-1:]
        m_star = torch.stack(list(tail)).mean(dim=0)

    residuals = [float((m_k - m_star).norm().item()) for m_k in m_history]
    c_k_max = 0.0
    for k in range(2, len(m_history)):
        c_k = float(
            (m_history[k] - 2.0 * m_history[k - 1] + m_history[k - 2]).norm().item()
        )
        if c_k > c_k_max:
            c_k_max = c_k
    phi_inf_norm = float(phi.detach().abs().max().item())
    mu = estimate_mu(residuals, dt_over_tau=dt_over_tau, skip=0)
    radius = iss_ball_radius(
        c_k_max=c_k_max, phi_inf_norm=phi_inf_norm, mu=mu, alpha_b=alpha_b
    )
    return {
        "samples": len(m_history),
        "c_k_max": c_k_max,
        "phi_inf_norm": phi_inf_norm,
        "mu": mu,
        "iss_ball_radius": radius,
        "residuals": residuals,
    }
