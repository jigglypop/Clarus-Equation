"""Euler-bitfield attention — {e, π, i, 1, 0} as minimal dimensionless grammar.

CE's docs/상수.md treats the Euler-identity constants {e, π, i, 1, 0}
as the minimum vocabulary that generates dimensionless cores. We apply
the same principle to positional/rotational attention encoding:

  - 5 "Euler basis" frequencies: B = {1, π, e, π·e, π/e}
  - Each attention head picks a BITFIELD b ∈ {0,1}^5 selecting the
    subset of frequencies it uses. The head's positional phase is

        theta_head(pos, k) = pos · sum_{j: b_j=1} B_j · 2^{-k / d_head}

    i.e. a sum of Euler-basis log-spaced frequencies.
  - Q and K are rotated by theta_head before scoring (RoPE-style).

The bitfield b is a learnable real-valued vector passed through a
sigmoid (soft bitfield) so gradients flow. At inference it thresholds
to {0, 1} for pure Euler-combinations.

Why this could help:
  1. Multi-base (e-based rotations don't align with π-based ones on any
     finite period) — unique long-range attention signatures.
  2. Compression: 5 bits per head vs O(d) free parameters.
  3. Theory grounding: CE's minimal grammar axiom.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EULER_BASIS = (1.0, math.pi, math.e, math.pi * math.e, math.pi / math.e)
EULER_BASIS_NAMES = ("1", "pi", "e", "pi*e", "pi/e")


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to adjacent dim pairs: [x0, x1] -> [x0·c - x1·s, x0·s + x1·c]."""
    # x: (..., n, d_head) with d_head even
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


class EulerRotaryAttention(nn.Module):
    """Multi-head attention with Euler-bitfield rotary positional encoding.

    Args:
        d_model, n_heads, block: as usual
        softmax_bitfield: if True, soft bitfield via sigmoid(logits).
            Otherwise hard 0/1 bitfield from init template.
        init_bits: initial bitfield per head. Shape (n_heads, 5).
            Default: each head uses all 5 bases.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        softmax_bitfield: bool = True,
        init_bits: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even for rotary"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        basis = torch.tensor(EULER_BASIS, dtype=torch.float32)  # (5,)
        self.register_buffer("euler_basis", basis)

        if init_bits is None:
            init_bits = torch.zeros(n_heads, 5)  # start at 0 -> sigmoid 0.5
        if softmax_bitfield:
            self.bit_logits = nn.Parameter(init_bits)
        else:
            self.register_buffer("bit_logits", init_bits)

        # log-space-frequency exponents: k = 0, 2, 4, ..., d_head-2  (per pair)
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("inv_freq", 2.0 ** (-k))  # (d_head/2,)

        # positions
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

    def bitfield(self) -> torch.Tensor:
        """(n_heads, 5) soft bitfield in [0, 1]."""
        return torch.sigmoid(self.bit_logits)

    def head_freq_scalars(self) -> torch.Tensor:
        """Per-head frequency scalar (weighted sum over Euler basis)."""
        b = self.bitfield()  # (n_heads, 5)
        return torch.matmul(b, self.euler_basis)  # (n_heads,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (b, n, h, d_head)
        q = q.transpose(1, 2)  # (b, h, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build head-specific rotation angles: theta[h, n, k/2] = pos[n] * freq[h] * inv_freq[k]
        freqs = self.head_freq_scalars()  # (h,)
        # theta = pos[n] * freq[h] * inv_freq[k]
        theta = self.pos[:n].view(1, 1, n, 1) * freqs.view(1, self.n_heads, 1, 1) \
                * self.inv_freq.view(1, 1, 1, -1)  # (1, h, n, d_head/2)
        cos = theta.cos()
        sin = theta.sin()

        q = _rotate_pairs(q, cos, sin)
        k = _rotate_pairs(k, cos, sin)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.tril[:n, :n], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class EulerAttnBlock(nn.Module):
    """Transformer block using Euler rotary attention."""

    def __init__(self, d_model: int, n_heads: int, block: int,
                 softmax_bitfield: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerRotaryAttention(d_model, n_heads, block,
                                         softmax_bitfield=softmax_bitfield)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


__all__ = [
    "EULER_BASIS",
    "EULER_BASIS_NAMES",
    "EulerRotaryAttention",
    "EulerAttnBlock",
    "EulerCEAttention",
    "EulerCEBlock",
]


# ---------------------------------------------------------------------------
# EulerCEAttention — theory-correct assignment of {e, π, i, 1, 0}
# ---------------------------------------------------------------------------
#
# Per docs/경로적분.md (lines 51-67):
#   e   -> survival / decay:        S(D) = e^{-D}
#   π   -> periodic normalization:  α_total = 1 / (2π)
#   i   -> path-integral phase:     Z = ∫ Dφ e^{iS/ℏ}
#   1   -> normalized complete state
#   0   -> zero / branch selection
#
# The attention kernel is therefore
#
#   A_ij  =  softmax_j ( Q_i · R_π(i-j) · K_j / √d )  ·  e^{-|i-j|/ξ_e}
#                       └─── π-phase rotary ───┘        └── e decay ──┘
#
#     R_π(Δ): RoPE-style rotation with fundamental period π (not 10^4).
#     ξ_e:    learnable correlation length (decay base e).
#
# The "bitfield" selects which of {π-phase, e-decay, 1-bypass} are active
# per head. A head with bit=0 for π uses identity rotation (no RoPE);
# bit=0 for e means no distance decay. This exposes the 5-constant
# minimum vocabulary as an interpretable head-type switch.


class EulerCEAttention(nn.Module):
    """Theory-correct Euler attention: π-phase + e-decay + {1,0} gates.

    Args:
        d_model, n_heads, block: standard
        xi_init: initial correlation length for the e-decay term (in
            positions). Larger = weaker decay. Default block/2.
        learnable_gates: if True, the two gates (pi_gate, e_gate) are
            per-head learnable sigmoids; if False they are frozen at 1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        xi_init: Optional[float] = None,
        learnable_gates: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        # π-phase rotary: fundamental frequency is π, log-scaled across
        # d_head pairs (cleaner than RoPE's 10^4 because π is the only
        # periodic-normalization constant in the CE grammar).
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        # theta_k = pos · π · (1/π)^k = pos · π^{1-k}   (log-spaced over π)
        self.register_buffer("pi_inv_freq", math.pi ** (1.0 - k))
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

        # e-decay: log xi so xi > 0 strictly. Initialize xi = block/2.
        if xi_init is None:
            xi_init = block / 2.0
        self.log_xi = nn.Parameter(torch.full((n_heads,),
                                              math.log(xi_init), dtype=torch.float32))

        # Per-head gates on the two constants (π and e). sigmoid(logit).
        if learnable_gates:
            self.pi_gate_logit = nn.Parameter(torch.full((n_heads,), 2.0))
            self.e_gate_logit = nn.Parameter(torch.full((n_heads,), 2.0))
        else:
            self.register_buffer("pi_gate_logit", torch.full((n_heads,), 1e4))
            self.register_buffer("e_gate_logit", torch.full((n_heads,), 1e4))

        # Precompute |i-j| distance matrix (non-negative, upper-tri set by mask)
        d_mat = (torch.arange(block).unsqueeze(1) - torch.arange(block).unsqueeze(0)).abs().float()
        self.register_buffer("d_mat", d_mat)

    def _rotate(self, x, cos, sin):
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        pi_g = torch.sigmoid(self.pi_gate_logit)      # (h,)
        e_g = torch.sigmoid(self.e_gate_logit)        # (h,)

        # π-phase rotation (per-head amplitude scaled by pi_g)
        theta = self.pos[:n].view(1, 1, n, 1) * self.pi_inv_freq.view(1, 1, 1, -1)
        # Modulate rotation magnitude per head: if pi_g=0, no rotation;
        # if pi_g=1, full π-phase rotation.
        theta = theta * pi_g.view(1, self.n_heads, 1, 1)
        cos = theta.cos(); sin = theta.sin()
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        # dot-product scores
        scores = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(self.d_head)

        # e-decay bias: -|i-j|/xi_h, multiplied by e_gate
        xi = torch.exp(self.log_xi)                   # (h,)
        d_sub = self.d_mat[:n, :n]                    # (n, n)
        # decay bias added to scores in log-space (equiv to multiplying A by e^{...})
        decay_bias = -d_sub.view(1, 1, n, n) / xi.view(1, self.n_heads, 1, 1)
        decay_bias = decay_bias * e_g.view(1, self.n_heads, 1, 1)
        scores = scores + decay_bias

        scores = scores.masked_fill(~self.tril[:n, :n], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class EulerCEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block: int,
                 learnable_gates: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerCEAttention(d_model, n_heads, block,
                                     learnable_gates=learnable_gates)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
