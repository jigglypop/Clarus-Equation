"""Fused ops for ClarusLM: auto-dispatches to CUDA / Rust / PyTorch.

Priority: CUDA (GPU tensor) > Rust (CPU tensor) > PyTorch fallback.

All public functions accept standard PyTorch tensors and return tensors.
Backward is supported via torch.autograd.Function.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

# ---- backend discovery ------------------------------------------------------

_RUST = None
_CUDA = None

try:
    from clarus._rust import (
        nn_topk_silu_fwd as _rust_topk_silu_fwd,
        nn_topk_silu_bwd as _rust_topk_silu_bwd,
        nn_lbo_fused_fwd as _rust_lbo_fused_fwd,
        nn_power_iter as _rust_power_iter,
        nn_gauge_lattice_fwd as _rust_gauge_lattice_fwd,
    )
    _RUST = True
except ImportError:
    _RUST = False

try:
    from clarus.kernels import get_cuda_ops
    _cuda_mod = get_cuda_ops()
    _CUDA = _cuda_mod is not None
except ImportError:
    _CUDA = False
    _cuda_mod = None


def has_rust() -> bool:
    return bool(_RUST)


def has_cuda() -> bool:
    return bool(_CUDA)


def ops_backend(device: torch.device) -> str:
    if device.type == "cuda" and _CUDA:
        return "cuda"
    if device.type == "cpu" and _RUST:
        return "rust"
    return "torch"


# =============================================================================
# TopK SiLU
# =============================================================================

class _TopKSiLUCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        out, mask = _cuda_mod.topk_silu_fwd(x.contiguous(), threshold)
        ctx.save_for_backward(x, mask)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, mask = ctx.saved_tensors
        return _cuda_mod.topk_silu_bwd(grad_out.contiguous(), x, mask), None


class _TopKSiLURust(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim_last, ratio):
        shape = x.shape
        flat = x.detach().contiguous().view(-1).numpy()
        out_np, mask_np = _rust_topk_silu_fwd(flat, dim_last, ratio)
        out_t = torch.from_numpy(out_np).reshape(shape)
        mask_t = torch.from_numpy(mask_np).reshape(shape)
        ctx.save_for_backward(x, mask_t)
        ctx.dim_last = dim_last
        return out_t

    @staticmethod
    def backward(ctx, grad_out):
        x, mask_t = ctx.saved_tensors
        flat_g = grad_out.detach().contiguous().view(-1).numpy()
        flat_x = x.detach().contiguous().view(-1).numpy()
        flat_m = mask_t.contiguous().view(-1).numpy()
        gi_np = _rust_topk_silu_bwd(flat_g, flat_x, flat_m, ctx.dim_last)
        return torch.from_numpy(gi_np).reshape(grad_out.shape), None, None


def _topk_silu_torch(x, k, cal_state=None):
    """Pure PyTorch fallback (running-threshold)."""
    h = F.silu(x)
    if k >= x.size(-1):
        return h
    abs_h = h.abs()
    if x.requires_grad:
        thr = abs_h.kthvalue(x.size(-1) - k + 1, dim=-1, keepdim=True).values
    else:
        thr = abs_h.kthvalue(x.size(-1) - k + 1, dim=-1, keepdim=True).values
    return h.masked_fill(abs_h < thr, 0.0)


def topk_silu(x: torch.Tensor, k: int, ratio: float,
              threshold: float = 0.0) -> torch.Tensor:
    """Fused SiLU + TopK sparse masking.

    Dispatches to CUDA / Rust / PyTorch based on device and backend.
    For CUDA: uses pre-computed threshold (element-wise kernel).
    For Rust:  uses exact quickselect per row (rayon parallel).
    """
    if k >= x.size(-1) or ratio >= 1.0:
        return F.silu(x)

    if x.is_cuda and _CUDA:
        return _TopKSiLUCUDA.apply(x, threshold)

    if not x.is_cuda and _RUST and not x.requires_grad:
        # Rust forward-only (inference path)
        shape = x.shape
        flat = x.detach().contiguous().view(-1).numpy()
        out_np, _ = _rust_topk_silu_fwd(flat, x.size(-1), ratio)
        return torch.from_numpy(out_np).reshape(shape)

    if not x.is_cuda and _RUST:
        return _TopKSiLURust.apply(x, x.size(-1), ratio)

    return _topk_silu_torch(x, k)


# =============================================================================
# LBO Norm (fused post-LayerNorm)
# =============================================================================

class _LBOFusedCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normed, v_eff, h, scale, bias, dim, rank):
        n2d = normed.contiguous().view(-1, dim)
        out, row_curv = _cuda_mod.lbo_norm_fwd(n2d, v_eff, h, scale, bias, dim, rank)
        curvature = row_curv.mean().item()
        ctx.save_for_backward(normed, v_eff, scale)
        ctx.h = h
        return out.view(normed.shape), curvature

    @staticmethod
    def backward(ctx, grad_out, _grad_curv):
        normed, v_eff, scale = ctx.saved_tensors
        # backward through: out = (normed - h*(normed - xW)) * scale + bias
        #                       = ((1-h)*normed + h*xW) * scale + bias
        # d_out/d_normed = ((1-h)*I + h*V_eff^T V_eff) * diag(scale)
        grad_scaled = grad_out * scale
        h = ctx.h
        grad_normed = (1 - h) * grad_scaled + h * (grad_scaled @ v_eff.t() @ v_eff)
        return grad_normed, None, None, None, None, None, None


class _LBOFusedRust(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normed, v, h, scale, bias, alpha_conf, dim, rank):
        shape = normed.shape
        flat_n = normed.detach().contiguous().view(-1).numpy()
        flat_v = v.detach().contiguous().view(-1).numpy()
        flat_s = scale.detach().contiguous().numpy()
        flat_b = bias.detach().contiguous().numpy()

        out_np, curvature = _rust_lbo_fused_fwd(
            flat_n, flat_v, float(h), flat_s, flat_b,
            float(alpha_conf), dim, rank,
        )
        out_t = torch.from_numpy(out_np).reshape(shape)
        ctx.save_for_backward(normed, v, scale)
        ctx.h = h
        ctx.alpha_conf = alpha_conf
        ctx.dim = dim
        ctx.rank = rank
        return out_t, curvature

    @staticmethod
    def backward(ctx, grad_out, _grad_curv):
        normed, v, scale = ctx.saved_tensors
        h = ctx.h
        # simplified backward: ignoring conformal gradient (it's a scalar)
        phi_sq = normed.detach().pow(2).mean()
        conformal = torch.exp(-abs(ctx.alpha_conf) * phi_sq)
        v_eff = v * conformal
        grad_scaled = grad_out * scale
        grad_normed = (1 - h) * grad_scaled + h * (grad_scaled @ v_eff.t() @ v_eff)
        return grad_normed, None, None, None, None, None, None, None


def lbo_fused_fwd(normed: torch.Tensor, v: torch.Tensor, h: float,
                  scale: torch.Tensor, bias: torch.Tensor,
                  alpha_conf: float, dim: int, rank: int,
                  need_curvature: bool = True):
    """Fused LBO normalization forward.

    Returns (output, curvature).
    For CUDA: single fused kernel.
    For Rust:  fused conformal + projection + Laplacian.
    """
    if normed.is_cuda and _CUDA:
        phi_sq = normed.detach().pow(2).mean()
        conformal = torch.exp(-abs(alpha_conf) * phi_sq)
        v_eff = v * conformal
        return _LBOFusedCUDA.apply(normed, v_eff, h, scale, bias, dim, rank)

    if not normed.is_cuda and _RUST and not normed.requires_grad:
        shape = normed.shape
        flat_n = normed.detach().contiguous().view(-1).numpy()
        flat_v = v.detach().contiguous().view(-1).numpy()
        flat_s = scale.detach().contiguous().numpy()
        flat_b = bias.detach().contiguous().numpy()
        out_np, curv = _rust_lbo_fused_fwd(
            flat_n, flat_v, h, flat_s, flat_b,
            alpha_conf, dim, rank,
        )
        return torch.from_numpy(out_np).reshape(shape), curv

    if not normed.is_cuda and _RUST:
        return _LBOFusedRust.apply(
            normed, v, h, scale, bias, alpha_conf, dim, rank,
        )

    # PyTorch fallback
    phi_sq = normed.detach().pow(2).mean()
    conformal = torch.exp(-torch.abs(torch.tensor(alpha_conf)) * phi_sq)
    v_eff = v * conformal
    proj = normed @ v_eff.t()
    xW = proj @ v_eff
    Lx = normed - xW
    curvature = Lx.detach().pow(2).mean().item() if need_curvature else 0.0
    out = (normed - h * Lx) * scale + bias
    return out, curvature


# =============================================================================
# Power iteration
# =============================================================================

def power_iter_step(v_mat: torch.Tensor, spectral_v: torch.Tensor,
                    dim: int, rank: int):
    """1-step power iteration -> (new_spectral_v, sigma_max).

    Uses Rust on CPU, PyTorch on CUDA.
    """
    if not v_mat.is_cuda and _RUST:
        flat_v = v_mat.detach().contiguous().view(-1).numpy()
        flat_sv = spectral_v.detach().contiguous().numpy()
        new_v_np, sigma = _rust_power_iter(flat_v, flat_sv, dim, rank)
        return torch.from_numpy(new_v_np), sigma

    with torch.no_grad():
        u = F.normalize(v_mat @ spectral_v, dim=0)
        new_v = F.normalize(v_mat.t() @ u, dim=0)
        sigma = (v_mat @ new_v).norm().item()
    return new_v, sigma


# =============================================================================
# Gauge lattice (fused 3-channel forward, Rust CPU only)
# =============================================================================

def gauge_lattice_fwd(x: torch.Tensor,
                      su3_up_w, su3_down_w,
                      su2_up_w, su2_down_w,
                      u1_up_w, u1_down_w,
                      mix_down_w, mix_up_w,
                      d3, d2, d1, h3, h2, h1,
                      mix_rank, ratio, dim):
    """Fused gauge lattice forward (Rust CPU only, inference).

    For training, returns None (caller should use PyTorch path).
    """
    if x.is_cuda or x.requires_grad or not _RUST:
        return None

    shape = x.shape
    flat_x = x.detach().contiguous().view(-1).numpy()
    flat_s3u = su3_up_w.detach().contiguous().view(-1).numpy()
    flat_s3d = su3_down_w.detach().contiguous().view(-1).numpy()
    flat_s2u = su2_up_w.detach().contiguous().view(-1).numpy()
    flat_s2d = su2_down_w.detach().contiguous().view(-1).numpy()
    flat_u1u = u1_up_w.detach().contiguous().view(-1).numpy()
    flat_u1d = u1_down_w.detach().contiguous().view(-1).numpy()

    import numpy as np
    if mix_down_w is not None and mix_up_w is not None and mix_rank > 0:
        flat_md = mix_down_w.detach().contiguous().view(-1).numpy()
        flat_mu = mix_up_w.detach().contiguous().view(-1).numpy()
    else:
        flat_md = np.empty(0, dtype=np.float32)
        flat_mu = np.empty(0, dtype=np.float32)

    out_np = _rust_gauge_lattice_fwd(
        flat_x,
        flat_s3u, flat_s3d,
        flat_s2u, flat_s2d,
        flat_u1u, flat_u1d,
        flat_md, flat_mu,
        d3, d2, d1, h3, h2, h1,
        mix_rank, ratio, dim,
    )
    return torch.from_numpy(out_np).reshape(shape)
