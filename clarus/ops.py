"""Tensor-friendly facade over the Rust ``clarus._rust`` neural ops.

ClarusLM (`examples/ai/clarus_lm.py`) consumes these wrappers via
``from clarus.ops import topk_silu, lbo_fused_fwd, power_iter_step,
gauge_lattice_fwd, ops_backend``. The wrappers convert between
PyTorch tensors and the contiguous f32 ``numpy`` arrays expected by the
Rust pyfunctions.

CUDA dispatch lives in :mod:`clarus.kernels`. CPU dispatch (the path
this module covers) routes through Rust when ``clarus._rust`` is built;
otherwise we fall back to a pure-PyTorch implementation that matches the
Rust kernel's mathematical contract.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import _rust as _r
    _HAS_RUST = True
except ImportError:
    _r = None
    _HAS_RUST = False

from .ce_ops import _as_cpu_numpy_flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_flat_f32(t: torch.Tensor) -> np.ndarray:
    """Float32 view used by ``nn_*`` Rust pyfunctions. Wraps the dtype-preserving
    helper from :mod:`clarus.ce_ops` and adds the f32 cast Rust expects."""
    arr = _as_cpu_numpy_flat(t)
    return arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)


def _from_flat(arr: np.ndarray, shape: Tuple[int, ...], device, dtype) -> torch.Tensor:
    out = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if shape:
        out = out.view(*shape)
    return out.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# TopK SiLU
# ---------------------------------------------------------------------------
def topk_silu(x: torch.Tensor, k: int, ratio: float, threshold: float = 0.0) -> torch.Tensor:
    """Fused SiLU + per-row Top-K masking.

    The Rust kernel ignores ``threshold`` and recomputes the per-row threshold
    from ``ratio``. ``threshold`` is accepted only for ClarusLM API parity.
    """
    dim = int(x.shape[-1])
    if k >= dim or ratio >= 1.0:
        return F.silu(x)

    if _HAS_RUST and not x.is_cuda:
        flat = _as_flat_f32(x)
        out_flat, _mask = _r.nn_topk_silu_fwd(flat, dim, float(ratio))
        return _from_flat(out_flat, x.shape, x.device, x.dtype)

    h = F.silu(x)
    abs_h = h.abs()
    thr = abs_h.kthvalue(dim - k + 1, dim=-1, keepdim=True).values
    return h.masked_fill(abs_h < thr, 0.0)


# ---------------------------------------------------------------------------
# LBO fused forward
# ---------------------------------------------------------------------------
def lbo_fused_fwd(
    x_normed: torch.Tensor,
    V: torch.Tensor,
    h: float,
    scale: torch.Tensor,
    bias: torch.Tensor,
    alpha_conf: float,
    dim: int,
    rank: int,
    *,
    need_curvature: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Fused (post-LayerNorm) Laplace-Beltrami diffusion.

    out = ((1-h)*x + h*x @ V_eff^T @ V_eff) * scale + bias
    where V_eff = V * exp(-alpha_conf * mean(x^2)).
    """
    if _HAS_RUST and not x_normed.is_cuda:
        flat_x = _as_flat_f32(x_normed)
        flat_v = _as_flat_f32(V)
        out_flat, curv = _r.nn_lbo_fused_fwd(
            flat_x,
            flat_v,
            float(h),
            _as_flat_f32(scale),
            _as_flat_f32(bias),
            float(alpha_conf),
            int(dim),
            int(rank),
        )
        out = _from_flat(out_flat, x_normed.shape, x_normed.device, x_normed.dtype)
        return out, float(curv) if need_curvature else 0.0

    phi_sq = x_normed.detach().pow(2).mean()
    conformal = torch.exp(-abs(alpha_conf) * phi_sq)
    v_eff = V * conformal
    proj = x_normed @ v_eff.t()
    xw = proj @ v_eff
    pre = torch.lerp(x_normed, xw, h)
    out = torch.addcmul(bias, pre, scale)
    curvature = 0.0
    if need_curvature:
        lx = x_normed - xw
        curvature = float((lx.detach().pow(2).sum() / lx.numel()).item())
    return out, curvature


# ---------------------------------------------------------------------------
# Power iteration step
# ---------------------------------------------------------------------------
def power_iter_step(
    V: torch.Tensor,
    spectral_v: torch.Tensor,
    dim: int,
    rank: int,
) -> Tuple[torch.Tensor, float]:
    """One step of power iteration to estimate sigma_max(V)."""
    if _HAS_RUST and not V.is_cuda:
        new_v_flat, sigma = _r.nn_power_iter(
            _as_flat_f32(V),
            _as_flat_f32(spectral_v),
            int(dim),
            int(rank),
        )
        new_v = _from_flat(new_v_flat, (dim,), V.device, V.dtype)
        return new_v, float(sigma)

    u = F.normalize(V @ spectral_v, dim=0)
    new_v = F.normalize(V.t() @ u, dim=0)
    sigma = float((V @ new_v).norm().item())
    return new_v, sigma


# ---------------------------------------------------------------------------
# Gauge lattice forward
# ---------------------------------------------------------------------------
def gauge_lattice_fwd(
    x: torch.Tensor,
    su3_up: torch.Tensor, su3_down: torch.Tensor,
    su2_up: torch.Tensor, su2_down: torch.Tensor,
    u1_up: torch.Tensor, u1_down: torch.Tensor,
    mix_down: torch.Tensor | None,
    mix_up: torch.Tensor | None,
    *,
    d3: int, d2: int, d1: int,
    h3: int, h2: int, h1: int,
    mix_rank: int,
    ratio: float,
    dim: int,
) -> torch.Tensor:
    """3x3+1 gauge lattice forward with optional cross-channel mixing."""
    if _HAS_RUST and not x.is_cuda:
        empty = np.zeros(0, dtype=np.float32)
        out_flat = _r.nn_gauge_lattice_fwd(
            _as_flat_f32(x),
            _as_flat_f32(su3_up), _as_flat_f32(su3_down),
            _as_flat_f32(su2_up), _as_flat_f32(su2_down),
            _as_flat_f32(u1_up), _as_flat_f32(u1_down),
            empty if mix_down is None else _as_flat_f32(mix_down),
            empty if mix_up is None else _as_flat_f32(mix_up),
            int(d3), int(d2), int(d1),
            int(h3), int(h2), int(h1),
            int(mix_rank),
            float(ratio),
            int(dim),
        )
        return _from_flat(out_flat, x.shape, x.device, x.dtype)

    s3 = d3
    s32 = d3 + d2
    k3 = max(1, math.ceil(ratio * h3))
    k2 = max(1, math.ceil(ratio * h2))
    k1 = max(1, math.ceil(ratio * h1))

    def channel(x_part, up, down, k, hid):
        h = F.silu(F.linear(x_part, up))
        if k < hid:
            abs_h = h.abs()
            thr = abs_h.kthvalue(hid - k + 1, dim=-1, keepdim=True).values
            h = h.masked_fill(abs_h < thr, 0.0)
        return F.linear(h, down)

    y3 = channel(x[..., :s3], su3_up, su3_down, k3, h3)
    y2 = channel(x[..., s3:s32], su2_up, su2_down, k2, h2)
    y1 = channel(x[..., s32:], u1_up, u1_down, k1, h1)
    y = torch.cat([y3, y2, y1], dim=-1)
    if mix_rank > 0 and mix_down is not None and mix_up is not None:
        y = y + F.linear(F.linear(y, mix_down), mix_up)
    return y


# ---------------------------------------------------------------------------
# Backend introspection
# ---------------------------------------------------------------------------
def ops_backend() -> str:
    """Identifier of the active CPU dispatch backend ('rust' or 'torch')."""
    return "rust" if _HAS_RUST else "torch"


__all__ = [
    "topk_silu",
    "lbo_fused_fwd",
    "power_iter_step",
    "gauge_lattice_fwd",
    "ops_backend",
]
