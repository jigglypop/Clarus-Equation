"""CE Metric-Family Attention (MFA) — applied equation 6.B.1.

Standard transformer attention collapses every relational mode (syntax,
semantics, event causality, replay) into a single inner-product kernel
``q_i^T k_j / sqrt(d)``. CE predicts this kernel is insufficient: the
same latent ``h_i`` must be projected through multiple metrics

    d_G^{(m)}(z_i, z_j)^2 = (z_i - z_j)^T G^{(m)}(z) (z_i - z_j)

and the attention weights combined with mode-dependent gates ``omega_m``.
During WAKE the linguistic metric dominates, during NREM the
event/gravity metric dominates (Borbely 2-process switch).

This module provides a drop-in PyTorch layer that can run standalone or
replace ``torch.nn.functional.scaled_dot_product_attention`` in toy
benchmarks. It is inference-oriented; gradients flow through the gates
if ``requires_grad`` is set on ``omega``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import T_WAKE, BYPASS, PORTAL


# ---------------------------------------------------------------------------
# Mode gating (WAKE / NREM / REM) — Borbely-driven omega schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeGate:
    """Convex weights for the metric family in a given brain mode.

    CE prediction: WAKE pushes weight to the linguistic metric
    (omega_lang ~ 1 - T_WAKE = 0.685), NREM inverts this so the
    event/gravity metric becomes primary (omega_grav ~ 1 - T_WAKE).
    """

    omega_lang: float
    omega_grav: float

    def as_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        t = torch.tensor([self.omega_lang, self.omega_grav], dtype=torch.float32)
        return t if device is None else t.to(device)


def mode_gate(mode: str) -> ModeGate:
    mode = mode.lower()
    if mode == "wake":
        return ModeGate(omega_lang=1.0 - T_WAKE, omega_grav=T_WAKE)
    if mode == "nrem":
        return ModeGate(omega_lang=T_WAKE, omega_grav=1.0 - T_WAKE)
    if mode == "rem":
        return ModeGate(omega_lang=0.5, omega_grav=0.5)
    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Individual metric kernels
# ---------------------------------------------------------------------------


def lang_attention(q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Standard scaled-dot-product kernel (linguistic metric).

    Shapes: q, k in (..., n, d). Returns (..., n, n).
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def grav_attention(
    z: torch.Tensor,
    sigma: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Distance-kernel attention (event/gravity metric).

    Uses squared Euclidean distance in the latent space. A learned or
    fixed metric tensor ``G`` can be provided via whitening ``z``
    before calling this function. Equation 6.B.1 with G = I.
    """
    diff = z.unsqueeze(-2) - z.unsqueeze(-3)  # (..., n, n, d)
    d2 = (diff * diff).sum(dim=-1)  # (..., n, n)
    scores = -d2 / (2.0 * sigma * sigma)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def metric_family_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z_grav: Optional[torch.Tensor] = None,
    gate: ModeGate = None,
    sigma_grav: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full MFA — equation 6.B.1 with two metrics (lang, grav).

    A_total = omega_lang * A_lang + omega_grav * A_grav
    output  = A_total @ v

    When ``z_grav`` is None, the projection defaults to k (no separate
    event embedding). Returns (..., n, d_v).
    """
    if gate is None:
        gate = mode_gate("wake")
    if z_grav is None:
        z_grav = k

    a_lang = lang_attention(q, k, mask=mask)
    a_grav = grav_attention(z_grav, sigma=sigma_grav, mask=mask)

    a_total = gate.omega_lang * a_lang + gate.omega_grav * a_grav
    return torch.matmul(a_total, v)


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------


class CESoftmaxAttention(nn.Module):
    """Drop-in multi-head attention with CE metric family.

    For a fair comparison against torch's MHA, we keep the Q/K/V/O
    projection identical and only change the attention kernel.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        sigma_grav: float = 1.0,
        mode: str = "wake",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sigma_grav = sigma_grav
        self.mode = mode

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_mode(self, mode: str) -> None:
        _ = mode_gate(mode)  # validate
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.w_q(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)

        gate = mode_gate(self.mode)
        out = metric_family_attention(q, k, v, gate=gate, sigma_grav=self.sigma_grav, mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.dropout(self.w_o(out))


__all__ = [
    "ModeGate",
    "mode_gate",
    "lang_attention",
    "grav_attention",
    "metric_family_attention",
    "CESoftmaxAttention",
]
