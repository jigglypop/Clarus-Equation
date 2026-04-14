"""Quantum phase evolution (12_Equation.md 1.5 / 4.1).

psi_{k+1} = exp(-i E dt) psi_k  (quantum form)
Wick rotation t -> -i*tau yields real relaxation.
"""

from __future__ import annotations

import torch


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
    alpha_b: float = 2.044,
) -> bool:
    """Check convergence sufficient condition: ||grad E|| > C_k * ||phi|| / alpha_b (4.7)."""
    return grad_norm > c_k * phi_norm / alpha_b
