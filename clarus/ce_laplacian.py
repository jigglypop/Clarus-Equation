"""Dual-graph attention via row-stochastic Laplacian kernels.

Restructured from the earlier residual-Laplacian form (which had
asymmetry bugs and a dead-gradient alpha=0 path). The block is now an
attention head whose kernel is built from explicit graph adjacency:

    A_lang_ij  = cosine(P_lang h_i, P_lang h_j)_+    (symmetric)
    A_grav_ij  = exp(-||P_grav h_i - P_grav h_j||^2 / 2sigma^2)

Both graphs are SYMMETRIC (no mask-induced asymmetry in A), so the
normalized Laplacian has eigenvalues in [0, 2] as expected. Causal
constraints are enforced ONLY on the row-normalized transition
matrices P_lang_rw, P_grav_rw (D^-1 A with upper-tri zeroed and rows
renormalized) — this is mathematically equivalent to restricting the
random walk to past neighbors.

Output per head:
    y_i = sum_j [omega_lang * P_lang_rw + omega_grav * P_grav_rw]_ij
            * V(h)_j

which is a convex mixture of two row-stochastic kernels acting on V,
i.e. a concrete instantiation of the compendium 6.B.1 attention
kernel family with interpretable graph-theoretic weights.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import T_WAKE


def _cosine_adjacency(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """A_ij = max(0, cos(z_i, z_j)), diagonal zeroed. Symmetric."""
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    zn = z / norm
    A = torch.matmul(zn, zn.transpose(-1, -2)).clamp_min(0.0)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A * (1.0 - eye)


def _rbf_adjacency(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """A_ij = exp(-||z_i - z_j||^2 / 2 sigma^2), diagonal zeroed. Symmetric."""
    sq = (z * z).sum(dim=-1, keepdim=True)
    d2 = (sq + sq.transpose(-1, -2) - 2.0 * torch.matmul(z, z.transpose(-1, -2))).clamp_min(0.0)
    A = torch.exp(-d2 / (2.0 * sigma * sigma))
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A * (1.0 - eye)


def _row_stochastic_causal(A: torch.Tensor, causal_mask: Optional[torch.Tensor],
                           eps: float = 1e-8) -> torch.Tensor:
    """Convert symmetric adjacency into a causal row-stochastic transition.

    Order of ops matters:
      1. A is symmetric here (both i->j and j->i).
      2. Apply causal mask: (i,j) with j > i dropped.
      3. Re-normalize rows so each row sums to 1 (random-walk kernel).
    """
    if causal_mask is not None:
        A = A * causal_mask.to(A.dtype)
    deg = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / deg


def _sym_normalized_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L = I - D^{-1/2} A D^{-1/2} for SYMMETRIC A. Eigenvalues in [0, 2]."""
    deg = A.sum(dim=-1).clamp_min(eps)
    inv_sqrt = deg.pow(-0.5)
    A_norm = A * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return eye - A_norm


class DualLaplacianBlock(nn.Module):
    """Dual-graph attention head.

    Produces attention output as a convex mix of two row-stochastic
    causal random-walk kernels (cosine on P_lang h, RBF on P_grav h).
    """

    def __init__(
        self,
        d_model: int,
        d_lang: Optional[int] = None,
        d_grav: Optional[int] = None,
        sigma_grav: float = 1.0,
        mode: str = "wake",
    ) -> None:
        super().__init__()
        d_lang = d_lang or d_model
        d_grav = d_grav or d_model
        self.P_lang = nn.Linear(d_model, d_lang, bias=False)
        self.P_grav = nn.Linear(d_model, d_grav, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.O = nn.Linear(d_model, d_model, bias=False)
        self.sigma_grav = sigma_grav
        self.mode = mode

    def gate_weights(self) -> tuple[float, float]:
        if self.mode == "wake":
            return (1.0 - T_WAKE, T_WAKE)
        if self.mode == "nrem":
            return (T_WAKE, 1.0 - T_WAKE)
        return (0.5, 0.5)

    def forward(self, h: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_l = self.P_lang(h)
        z_g = self.P_grav(h)
        v = self.V(h)

        A_l = _cosine_adjacency(z_l)
        A_g = _rbf_adjacency(z_g, sigma=self.sigma_grav)

        K_l = _row_stochastic_causal(A_l, causal_mask)
        K_g = _row_stochastic_causal(A_g, causal_mask)

        w_l, w_g = self.gate_weights()
        K = w_l * K_l + w_g * K_g  # still row-stochastic (convex comb of stochastic)
        return self.O(torch.matmul(K, v))


def graph_spectrum(
    adjacency_fn,
    h: torch.Tensor,
    *,
    symmetric: bool = True,
) -> torch.Tensor:
    """Compute eigenvalues of the symmetric normalized Laplacian of a
    graph whose adjacency is ``adjacency_fn(h)``. Returns sorted real
    eigenvalues (only valid when adjacency is symmetric)."""
    A = adjacency_fn(h)
    if not symmetric:
        A = 0.5 * (A + A.transpose(-1, -2))
    L = _sym_normalized_laplacian(A)
    L = 0.5 * (L + L.transpose(-1, -2))  # numeric cleanup
    return torch.linalg.eigvalsh(L).sort().values


__all__ = [
    "DualLaplacianBlock",
    "graph_spectrum",
    "_cosine_adjacency",
    "_rbf_adjacency",
    "_sym_normalized_laplacian",
    "_row_stochastic_causal",
]
