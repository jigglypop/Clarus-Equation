"""Dual-Laplacian diffusion layer — user's 5.2-5.6 proposal.

Instead of mixing two attention distributions, maintain two separate
projections of the common latent h_i:

    z_i^{lang} = P_lang(h_i)
    z_i^{grav} = P_grav(h_i)

Each projection lives in its own metric. Similarity adjacencies
A_lang, A_grav are built from cosine / RBF kernels in the respective
spaces. The Laplacian diffusion update

    h^{t+1} = h^t - eta_lang L_lang h^t - eta_grav L_grav h^t

acts as a RESIDUAL (GNN message passing) rather than replacing
attention. Because the two operators add in h-space (not in the
softmax-normalized distribution space), sharpness is preserved.

Mode gating enters through eta_lang / eta_grav, which scale with
Borbely T_WAKE: wake favors lang diffusion, nrem favors grav.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import T_WAKE


def _rbf_adjacency(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """Dense RBF kernel adjacency A_ij = exp(-||z_i - z_j||^2 / 2sigma^2).

    Returns (..., n, n), diagonal zeroed. Uses the BLAS expansion.
    """
    sq = (z * z).sum(dim=-1, keepdim=True)
    d2 = sq + sq.transpose(-1, -2) - 2.0 * torch.matmul(z, z.transpose(-1, -2))
    d2 = d2.clamp_min(0.0)
    A = torch.exp(-d2 / (2.0 * sigma * sigma))
    # zero the diagonal in a shape-agnostic way
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A = A * (1.0 - eye)
    return A


def _cosine_adjacency(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity adjacency, diagonal zeroed, thresholded at 0."""
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    zn = z / norm
    A = torch.matmul(zn, zn.transpose(-1, -2))
    A = A.clamp_min(0.0)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A = A * (1.0 - eye)
    return A


def _normalized_laplacian(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

    eps-clamped degree avoids NaN from isolated nodes (e.g., position 0
    under a causal mask).
    """
    deg = A.sum(dim=-1).clamp_min(eps)
    inv_sqrt = deg.pow(-0.5)
    D_is = inv_sqrt.unsqueeze(-1)
    D_isT = inv_sqrt.unsqueeze(-2)
    A_norm = A * D_is * D_isT
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return eye - A_norm


class DualLaplacianBlock(nn.Module):
    """Residual dual-Laplacian diffusion block.

    Args:
        d_model:    feature dimension of h
        d_lang:     dim of language projection
        d_grav:     dim of gravity projection
        sigma_grav: RBF bandwidth for gravity graph (cosine for lang)
        mode:       "wake" | "nrem" | "rem" for default eta scaling
        max_steps:  diffusion steps per forward pass (1 is usual)
    """

    def __init__(
        self,
        d_model: int,
        d_lang: Optional[int] = None,
        d_grav: Optional[int] = None,
        sigma_grav: float = 1.0,
        mode: str = "wake",
        max_steps: int = 1,
    ) -> None:
        super().__init__()
        d_lang = d_lang or d_model
        d_grav = d_grav or d_model
        self.P_lang = nn.Linear(d_model, d_lang, bias=False)
        self.P_grav = nn.Linear(d_model, d_grav, bias=False)
        # per-channel residual projection back to h-space
        self.O_lang = nn.Linear(d_lang, d_model, bias=False)
        self.O_grav = nn.Linear(d_grav, d_model, bias=False)
        self.sigma_grav = sigma_grav
        self.mode = mode
        self.max_steps = max_steps

    def eta_pair(self) -> tuple[float, float]:
        # wake: lang-heavy diffusion; nrem: grav-heavy
        if self.mode == "wake":
            return (1.0 - T_WAKE, T_WAKE)
        if self.mode == "nrem":
            return (T_WAKE, 1.0 - T_WAKE)
        return (0.5, 0.5)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (b, n, d_model)
        z_l = self.P_lang(h)
        z_g = self.P_grav(h)

        A_l = _cosine_adjacency(z_l)
        A_g = _rbf_adjacency(z_g, sigma=self.sigma_grav)
        if mask is not None:
            # mask is (b, n, n) bool
            A_l = A_l * mask.to(A_l.dtype)
            A_g = A_g * mask.to(A_g.dtype)

        L_l = _normalized_laplacian(A_l)
        L_g = _normalized_laplacian(A_g)

        eta_l, eta_g = self.eta_pair()
        # Diffusion in the projected spaces, then mix back via O
        dz_l = torch.matmul(L_l, z_l)
        dz_g = torch.matmul(L_g, z_g)

        for _ in range(self.max_steps - 1):
            dz_l = torch.matmul(L_l, dz_l)
            dz_g = torch.matmul(L_g, dz_g)

        update = eta_l * self.O_lang(dz_l) + eta_g * self.O_grav(dz_g)
        return h - update


__all__ = ["DualLaplacianBlock"]
