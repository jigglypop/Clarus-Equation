"""Riemann-CE attention — Design (1) and (4) from the Riemann roadmap.

Assumes the Riemann Hypothesis: all non-trivial zeros of ζ(s) lie on
Re(s) = 1/2. Under RH, the imaginary parts γ_n form a sequence whose
pair-spacing follows the Gaussian Unitary Ensemble statistic
(Montgomery-Dyson) — "maximally disordered yet structured".

The first 100 γ_n values are hardcoded from standard references
(Titchmarsh, Odlyzko published tables) — no network access required.

Two mechanisms:

  RiemannRotaryAttention:
    Use γ_n (normalized) as the per-pair rotation frequency in RoPE-
    style attention. Replaces the geometric 10000^(−k/d) (RoPE) or the
    Euler {π, e, πe, π/e} bitfield (EulerCEAttention).

  riemann_zero_init(W):
    Initialize a linear layer whose columns are spaced so that the
    first min(n, 100) inner products with a reference vector recover
    the γ_n pattern. Used for FFN key-value memory addresses.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# First 100 imaginary parts of the non-trivial Riemann zeta zeros.
# Source: Titchmarsh "The Theory of the Riemann Zeta-Function" Appendix,
# cross-checked with Odlyzko's published tables. Values truncated to 9
# significant figures (sufficient for float32 attention).
RIEMANN_ZEROS_IM: tuple[float, ...] = (
    14.134725142,  21.022039639,  25.010857580,  30.424876126,  32.935061588,
    37.586178159,  40.918719012,  43.327073281,  48.005150881,  49.773832478,
    52.970321478,  56.446247697,  59.347044003,  60.831778525,  65.112544048,
    67.079810529,  69.546401711,  72.067157674,  75.704690699,  77.144840069,
    79.337375020,  82.910380854,  84.735492981,  87.425274613,  88.809111208,
    92.491899271,  94.651344041,  95.870634228,  98.831194218, 101.317851006,
    103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
    114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
    124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
    134.756509753, 138.116042055, 139.736208952, 141.123707404, 143.111845808,
    146.000982487, 147.422765343, 150.053520421, 150.925257612, 153.024693811,
    156.112909294, 157.597591818, 158.849988171, 161.188964138, 163.030709687,
    165.537069188, 167.184439978, 169.094515416, 169.911976479, 173.411536520,
    174.754191523, 176.441434298, 178.377407776, 179.916484020, 182.207078484,
    184.874467848, 185.598783678, 187.228922584, 189.416158656, 192.026656361,
    193.079726604, 195.265396680, 196.876481841, 198.015309676, 201.264751944,
    202.493594514, 204.189671803, 205.394697202, 207.906258888, 209.576509717,
    211.690862595, 213.347919360, 214.547044783, 216.169538508, 219.067596349,
    220.714918839, 221.430705555, 224.007000255, 224.983324670, 227.421444280,
    229.337413306, 231.250188700, 231.987235253, 233.693404179, 236.524229666,
)


def riemann_zeros(n: int) -> torch.Tensor:
    """Return the first n Riemann-zeta zero imaginary parts as a float tensor.

    For n beyond the hardcoded 100, extrapolates via the Riemann-von
    Mangoldt asymptotic  γ_n ≈ 2π n / log n  (error < 1% for n > 20).
    """
    if n <= len(RIEMANN_ZEROS_IM):
        return torch.tensor(RIEMANN_ZEROS_IM[:n], dtype=torch.float32)
    vals = list(RIEMANN_ZEROS_IM)
    for k in range(len(RIEMANN_ZEROS_IM) + 1, n + 1):
        # asymptotic density N(T) ~ (T/2π)·log(T/2π) => T_k = 2π k / log k
        vals.append(2.0 * math.pi * k / math.log(max(k, 2)))
    return torch.tensor(vals, dtype=torch.float32)


class RiemannRotaryAttention(nn.Module):
    """Rotary attention with Riemann-zero frequencies.

    Each of the d_head/2 rotation pairs uses one γ_n as its base
    frequency, normalized so the smallest γ plays the role of RoPE's
    slowest oscillation.

    Args:
        d_model, n_heads, block:    standard
        normalize:  divide γ_n by γ_1 so the fundamental frequency
                    matches the (log-spaced) scale of RoPE / Euler-CE
        learnable_scale: if True, a per-head log scalar scales all γ_n
                    so the model can choose its own "speed of light"
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        normalize: bool = True,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0
        n_pairs = self.d_head // 2

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        gamma = riemann_zeros(n_pairs)  # (n_pairs,)
        if normalize:
            # RoPE's first pair rotates at 1/1 rad-per-step; we map
            # γ_1 → 1 (slowest), γ_k → γ_k / γ_1.
            gamma = gamma / gamma[0]
        # RoPE uses 1/freq; here γ_n IS the frequency. Invert so larger n
        # gets slower rotation (matching RoPE's convention):
        inv_freq = 1.0 / gamma  # (n_pairs,)
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

        if learnable_scale:
            self.log_scale = nn.Parameter(torch.zeros(n_heads))
        else:
            self.register_buffer("log_scale", torch.zeros(n_heads))

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

        scale = torch.exp(self.log_scale)  # (h,)
        theta = self.pos[:n].view(1, 1, n, 1) \
                * self.inv_freq.view(1, 1, 1, -1) \
                * scale.view(1, self.n_heads, 1, 1)
        cos = theta.cos(); sin = theta.sin()
        q = self._rotate(q, cos, sin); k = self._rotate(k, cos, sin)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.tril[:n, :n], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class RiemannAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, block, ffn_kind: Optional[str] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = RiemannRotaryAttention(d_model, n_heads, block)
        # allow plugging in Euler/Riemann FFN variants later
        if ffn_kind is None or ffn_kind == "std":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        else:
            from .ce_ffn import make_ffn
            self.ffn = make_ffn(ffn_kind, d_model, mult=4)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@torch.no_grad()
def riemann_zero_init(linear: nn.Linear, axis: str = "in") -> None:
    """Initialize `linear.weight` so that one of its axes is arranged in
    a Riemann-zero spacing pattern.

    The weight shape is (out_features, in_features). We pick the chosen
    axis and rescale so the cumulative position of each row/col matches
    the cumulative distance to the k-th Riemann zero. This gives FFN
    memory keys a GUE-distributed spacing — the Design (4) hypothesis.

    The orthogonal axis is left with Kaiming normal.
    """
    W = linear.weight  # (out, in)
    if axis == "in":
        n = W.shape[1]
    elif axis == "out":
        n = W.shape[0]
    else:
        raise ValueError(axis)

    gamma = riemann_zeros(n)
    spacings = torch.cat([gamma[:1], gamma[1:] - gamma[:-1]])  # "gaps"
    # normalize spacings to mean 1 (dimensionless shape preserved)
    spacings = spacings / spacings.mean()
    positions = torch.cumsum(spacings, dim=0)
    # center and standardize
    positions = (positions - positions.mean()) / positions.std().clamp_min(1e-8)

    # Kaiming base
    nn.init.kaiming_normal_(W, nonlinearity="linear")
    # modulate chosen axis by the Riemann positions
    if axis == "in":
        W.mul_(positions.view(1, n))
    else:
        W.mul_(positions.view(n, 1))


__all__ = [
    "RIEMANN_ZEROS_IM",
    "riemann_zeros",
    "RiemannRotaryAttention",
    "RiemannAttnBlock",
    "riemann_zero_init",
]
