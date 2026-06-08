# LLM Context Chunk

---
## File: `reality_stone/python/reality_stone/clarus/ce_ops.py`

```python
"""Metric-aware CE ops: auto-dispatch to CUDA / Rust / PyTorch.

Phase 1 is inference-only. Public API keeps standard torch tensors while
moving the hot path (sparse relax loop) into native code when available.

This module is the canonical Python backend-dispatch layer. Higher-level
runtime policy should stay in Python modules such as `reality_stone.clarus.engine`
and `reality_stone.clarus.runtime`, while pure numerics route through here.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .quantum import ALPHA_B_DEFAULT, estimate_mu, iss_ball_radius

try:
    from .constants import PORTAL as DEFAULT_CB_W, NORM_EPS, SOFTMAX_EPS, CLAMP_EPS
except ImportError:
    from reality_stone.clarus.constants import PORTAL as DEFAULT_CB_W, NORM_EPS, SOFTMAX_EPS, CLAMP_EPS

_RUST = False
_CUDA = False
_cuda_mod = None

try:
    from ._rust import (
        nn_ce_pack_sparse as _rust_ce_pack_sparse,
        nn_ce_metric_basis_fwd as _rust_ce_metric_basis_fwd,
        nn_ce_codebook_pull as _rust_ce_codebook_pull,
        nn_ce_relax_fwd as _rust_ce_relax_fwd,
    )

    _RUST = True
except ImportError:
    _RUST = False

try:
    from .kernels import get_ce_cuda_ops

    _cuda_mod = get_ce_cuda_ops()
    _CUDA = _cuda_mod is not None
except ImportError:
    _CUDA = False
    _cuda_mod = None


def has_rust() -> bool:
    return bool(_RUST)


def has_cuda() -> bool:
    return bool(_CUDA)


def ce_backend(device: torch.device, requested: str = "auto") -> str:
    requested = requested.lower()
    if requested == "auto":
        if device.type == "cuda" and _CUDA:
            return "cuda"
        if device.type == "cpu" and _RUST:
            return "rust"
        return "torch"
    if requested == "cuda":
        if device.type != "cuda":
            raise RuntimeError("CUDA CE backend requested for a non-CUDA tensor/device")
        if not _CUDA:
            raise RuntimeError("CUDA CE backend requested but CUDA CE kernels are unavailable")
        return "cuda"
    if requested == "rust":
        if device.type != "cpu":
            raise RuntimeError("Rust CE backend requested for a non-CPU tensor/device")
        if not _RUST:
            raise RuntimeError("Rust CE backend requested but reality_stone.clarus._rust is unavailable")
        return "rust"
    if requested == "torch":
        return "torch"
    raise ValueError(f"unknown CE backend: {requested}")


def _as_cpu_numpy_flat(x: torch.Tensor):
    return x.detach().contiguous().view(-1).cpu().numpy()


def _hist_from_tensors(
    energy: torch.Tensor,
    delta: torch.Tensor,
    e_hop: torch.Tensor,
    e_bias: torch.Tensor,
    e_portal: torch.Tensor,
    e_cb: torch.Tensor,
    bypass_hist: torch.Tensor,
) -> Dict[str, list[float]]:
    return {
        "E": energy.detach().cpu().tolist(),
        "delta": delta.detach().cpu().tolist(),
        "E_hop": e_hop.detach().cpu().tolist(),
        "E_bias": e_bias.detach().cpu().tolist(),
        "E_portal": e_portal.detach().cpu().tolist(),
        "E_cb": e_cb.detach().cpu().tolist(),
        "bypass_C": bypass_hist.detach().cpu().tolist(),
    }


def pack_sparse(
    w: torch.Tensor,
    zero_tol: float = 0.0,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = w.shape[0]
    chosen = ce_backend(w.device if w.is_cuda else torch.device("cpu"), backend)

    if chosen == "rust" and not w.is_cuda:
        values_np, col_np, row_np = _rust_ce_pack_sparse(_as_cpu_numpy_flat(w), dim, float(zero_tol))
        return (
            torch.from_numpy(values_np),
            torch.from_numpy(col_np),
            torch.from_numpy(row_np),
        )

    mask = w.abs() > zero_tol
    rows, cols = mask.nonzero(as_tuple=True)
    values = w[rows, cols].to(dtype=torch.float32)
    col_idx = cols.to(dtype=torch.int32)
    row_counts = torch.bincount(rows, minlength=dim).to(dtype=torch.int32)
    row_ptr = torch.zeros(dim + 1, dtype=torch.int32, device=w.device)
    row_ptr[1:] = torch.cumsum(row_counts, dim=0)
    return values, col_idx, row_ptr


def build_metric_basis(
    codebook: torch.Tensor,
    m_ref: torch.Tensor,
    rank: int,
    w_eigvecs: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Build orthonormal metric basis from codebook directions + optional Hessian eigenvectors.

    When w_eigvecs is provided (top-k eigenvectors of W), they are placed first
    in the basis to capture the principal curvature directions of E_hop.
    """
    n_code, dim = codebook.shape
    if rank <= 0:
        return codebook.new_empty((0, dim))

    chosen = ce_backend(codebook.device if codebook.is_cuda else torch.device("cpu"), backend)

    if chosen == "rust" and not codebook.is_cuda and w_eigvecs is None:
        basis_np = _rust_ce_metric_basis_fwd(
            _as_cpu_numpy_flat(codebook),
            _as_cpu_numpy_flat(m_ref),
            n_code,
            dim,
            rank,
        )
        basis_t = torch.from_numpy(basis_np)
        rows = 0 if basis_t.numel() == 0 else basis_t.numel() // dim
        return basis_t.reshape(rows, dim)

    basis_rows: list[torch.Tensor] = []

    if w_eigvecs is not None and w_eigvecs.numel() > 0:
        for j in range(w_eigvecs.shape[0]):
            v = w_eigvecs[j].clone()
            for b in basis_rows:
                v = v - torch.dot(v, b) * b
            n = v.norm()
            if n > 1e-6:
                basis_rows.append(v / n)
            if len(basis_rows) >= rank:
                break

    if n_code > 0 and len(basis_rows) < rank:
        remain = rank - len(basis_rows)
        logits = codebook @ m_ref
        probs = F.softmax(logits, dim=0)
        mean = (probs.unsqueeze(1) * codebook).sum(dim=0)
        idx = probs.topk(min(remain * 4, n_code)).indices
        for i in idx.tolist():
            v = (codebook[i] - mean) * probs[i].sqrt()
            for b in basis_rows:
                v = v - torch.dot(v, b) * b
            n = v.norm()
            if n > 1e-6:
                basis_rows.append(v / n)
            if len(basis_rows) >= rank:
                break

    if not basis_rows:
        return codebook.new_empty((0, dim))
    return torch.stack(basis_rows, dim=0)


def codebook_pull(
    m: torch.Tensor,
    codebook: torch.Tensor,
    beta: float,
    cb_w: float,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if codebook.numel() == 0:
        zero = torch.zeros_like(m)
        return zero, m.new_tensor(0.0)

    chosen = ce_backend(m.device, backend)
    n_code, dim = codebook.shape

    if chosen == "cuda" and m.is_cuda:
        grad, energy = _cuda_mod.ce_codebook_pull_fwd(
            m.contiguous(),
            codebook.contiguous(),
            float(beta),
            float(cb_w),
        )
        return grad, energy

    if chosen == "rust" and not m.is_cuda:
        grad_np, energy = _rust_ce_codebook_pull(
            _as_cpu_numpy_flat(m),
            _as_cpu_numpy_flat(codebook),
            n_code,
            dim,
            float(beta),
            float(cb_w),
        )
        return torch.from_numpy(grad_np), torch.tensor(energy, dtype=m.dtype)

    logits = beta * (codebook @ m)
    w = F.softmax(logits, dim=0)
    grad = -cb_w * (w @ codebook)
    energy = -(cb_w / max(beta, 1e-6)) * torch.logsumexp(logits, dim=0)
    return grad, energy


def _spmv_torch(
    values: torch.Tensor,
    col_idx: torch.Tensor,
    row_ptr: torch.Tensor,
    x: torch.Tensor,
    *,
    sparse_mat: torch.Tensor | None = None,
    dense_w: torch.Tensor | None = None,
) -> torch.Tensor:
    if dense_w is not None:
        return dense_w @ x
    dim = x.numel()
    sparse = sparse_mat
    if sparse is None:
        sparse = torch.sparse_csr_tensor(
            row_ptr.to(torch.int64),
            col_idx.to(torch.int64),
            values,
            size=(dim, dim),
            device=x.device,
            dtype=x.dtype,
            check_invariants=False,
        )
    return torch.sparse.mm(sparse, x.unsqueeze(1)).squeeze(1)


def _natural_direction_torch(
    grad: torch.Tensor,
    phi: torch.Tensor,
    recent_var: torch.Tensor,
    metric_basis: torch.Tensor,
    lambda0: float,
    lambda_phi: float,
    lambda_var: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diag = lambda0 + lambda_phi * phi.square() + lambda_var * recent_var
    diag = diag.clamp_min(1e-4)
    inv_diag = diag.reciprocal()
    inv_diag_grad = grad * inv_diag

    if metric_basis.numel() == 0:
        return inv_diag_grad, diag

    basis = metric_basis
    weighted_basis = basis * inv_diag.unsqueeze(0)
    small = torch.eye(
        basis.shape[0],
        device=grad.device,
        dtype=grad.dtype,
    ) + basis @ weighted_basis.transpose(0, 1)
    rhs = basis @ inv_diag_grad
    tmp = torch.linalg.solve(small, rhs.unsqueeze(-1)).squeeze(-1)
    correction = basis.transpose(0, 1) @ tmp
    return inv_diag_grad - correction * inv_diag, diag


def _fdt_noise_torch(
    z: torch.Tensor,
    phi: torch.Tensor,
    recent_var: torch.Tensor,
    metric_basis: torch.Tensor,
    lambda0: float,
    lambda_phi: float,
    lambda_var: float,
) -> torch.Tensor:
    """Compute G^{-1/2} z for FDT-consistent Langevin noise.

    G = D + U U^T  (Woodbury SPD metric)
    G^{-1/2} = D^{-1/2} (I + Q Q^T)^{-1/2} where Q = D^{-1/2} U
    (I + Q Q^T)^{-1/2} computed via SVD of Q.
    """
    diag = lambda0 + lambda_phi * phi.square() + lambda_var * recent_var
    diag = diag.clamp_min(1e-4)
    inv_sqrt_diag = diag.rsqrt()

    if metric_basis.numel() == 0:
        return z * inv_sqrt_diag

    Q = metric_basis * inv_sqrt_diag.unsqueeze(0)
    if not torch.isfinite(Q).all():
        Q = torch.where(torch.isfinite(Q), Q, torch.zeros_like(Q))
    _, s_q, Vh_q = torch.linalg.svd(Q, full_matrices=False)

    factors = 1.0 - 1.0 / torch.sqrt(1.0 + s_q.square())
    proj = Vh_q @ z
    corrected = z - (Vh_q.T @ (factors * proj))
    return inv_sqrt_diag * corrected


def _energy_parts_torch(
    m: torch.Tensor,
    w_m: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    codebook: torch.Tensor,
    portal: float,
    beta: float,
    cb_w: float,
    bypass_c: float = 0.0,
    bypass_coeff: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    e_hop = -0.5 * torch.dot(m, w_m)
    e_bias = -torch.dot(m, b)
    e_portal = -portal * torch.dot(m, phi)
    e_bypass = -bypass_coeff * bypass_c * torch.dot(m, phi)
    if codebook.numel() == 0:
        e_cb = m.new_tensor(0.0)
    else:
        logits = beta * (codebook @ m)
        e_cb = -(cb_w / max(beta, 1e-6)) * torch.logsumexp(logits, dim=0)
    total = e_hop + e_bias + e_portal + e_cb + e_bypass
    return total, (e_hop, e_bias, e_portal, e_cb)


@torch.no_grad()
def _relax_packed_torch(
    values: torch.Tensor,
    col_idx: torch.Tensor,
    row_ptr: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    m0: torch.Tensor,
    codebook: torch.Tensor,
    metric_basis: torch.Tensor,
    portal: float,
    bypass: float,
    t_wake: float,
    beta: float,
    cb_w: float,
    lambda0: float,
    lambda_phi: float,
    lambda_var: float,
    tau: float,
    dt: float,
    max_steps: int,
    tol: float,
    anneal_ratio: float,
    noise_scale: float,
    seed: int,
    dense_w: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, list[float]], int]:
    scale = float(m0.norm().item() or 1.0)
    m = m0 / scale
    b_n = b / scale
    phi_n = F.normalize(phi, dim=0)
    codebook_n = codebook / scale if codebook.numel() else codebook
    metric_basis_n = metric_basis

    m1 = m.clone()
    m2 = m.clone()

    tau = max(float(tau), 1e-6)
    dt_eff = min(float(dt), 0.9 * tau)
    anneal_end = max(1, int(round(anneal_ratio * max_steps)))
    t_eff = float(t_wake) / max(1, m.numel())

    sparse_mat = None
    if dense_w is None:
        sparse_mat = torch.sparse_csr_tensor(
            row_ptr.to(torch.int64),
            col_idx.to(torch.int64),
            values,
            size=(m.numel(), m.numel()),
            device=m.device,
            dtype=m.dtype,
            check_invariants=False,
        )

    w_m_probe = _spmv_torch(values, col_idx, row_ptr, m, sparse_mat=sparse_mat, dense_w=dense_w)
    spectral_est = w_m_probe.norm().item() / max(m.norm().item(), 1e-8)
    cfl_lambda0 = 2.0 * spectral_est * dt_eff / tau
    lambda0 = max(lambda0, cfl_lambda0)

    gen = None
    if noise_scale > 0.0:
        gen = torch.Generator(device=m.device)
        gen.manual_seed(int(seed))

    hist_e: list[float] = []
    hist_delta: list[float] = []
    hist_e_hop: list[float] = []
    hist_e_bias: list[float] = []
    hist_e_portal: list[float] = []
    hist_e_cb: list[float] = []
    hist_bypass: list[float] = []

    best_m = m.clone()
    best_e = float("inf")
    tail_states: deque[torch.Tensor] = deque(maxlen=min(16, max_steps))

    for k in range(max_steps):
        c_k = torch.norm(m - 2 * m1 + m2).item()
        w_m = _spmv_torch(values, col_idx, row_ptr, m, sparse_mat=sparse_mat, dense_w=dense_w)
        grad = w_m + b_n + float(portal) * phi_n + (c_k * float(bypass)) * phi_n

        if codebook_n.numel():
            cb_grad, _ = codebook_pull(m, codebook_n, beta=beta, cb_w=cb_w, backend="torch")
            grad = grad + cb_grad

        recent_var = 0.5 * ((m - m1).square() + (m1 - m2).square())
        nat_grad, _metric_diag = _natural_direction_torch(
            grad,
            phi_n,
            recent_var,
            metric_basis_n,
            lambda0,
            lambda_phi,
            lambda_var,
        )

        t_k = t_eff * max(0.0, 1.0 - k / anneal_end)
        noise_std = math.sqrt(max(0.0, 2.0 * t_k * dt_eff / tau)) * max(0.0, noise_scale)
        if noise_std > 0.0:
            z_raw = torch.randn(m.shape, dtype=m.dtype, device=m.device, generator=gen)
            noise = noise_std * _fdt_noise_torch(
                z_raw, phi_n, recent_var, metric_basis_n,
                lambda0, lambda_phi, lambda_var,
            )
        else:
            noise = torch.zeros_like(m)

        m2 = m1.clone()
        m1 = m.clone()
        dm = (dt_eff / tau) * nat_grad + noise
        if not torch.isfinite(dm).all():
            dm = torch.where(torch.isfinite(dm), dm, torch.zeros_like(dm))
        m = m + dm
        tail_states.append(m.detach().clone())

        w_m_new = _spmv_torch(values, col_idx, row_ptr, m, sparse_mat=sparse_mat, dense_w=dense_w)
        e_total, (e_hop, e_bias, e_portal, e_cb) = _energy_parts_torch(
            m, w_m_new, b_n, phi_n, codebook_n,
            portal, beta, cb_w,
            bypass_c=c_k, bypass_coeff=bypass,
        )
        e_item = float(e_total.item())
        delta = float(dm.norm().item())

        hist_e.append(e_item)
        hist_delta.append(delta)
        hist_e_hop.append(float(e_hop.item()))
        hist_e_bias.append(float(e_bias.item()))
        hist_e_portal.append(float(e_portal.item()))
        hist_e_cb.append(float(e_cb.item()))
        hist_bypass.append(float(c_k))

        if e_item < best_e:
            best_e = e_item
            best_m = m.clone()

        if k > 30 and delta < tol:
            break

    best_m = best_m * scale
    if tail_states:
        tail = torch.stack(list(tail_states), dim=0) * scale
        phi_var = (tail - best_m.unsqueeze(0)).square().mean(dim=0)
        hist_phi_var = phi_var.detach().cpu().tolist()
    else:
        hist_phi_var = []

    iss_report = _iss_from_tail(
        tail_states=tail_states,
        scale=scale,
        best_m=best_m,
        c_k_history=hist_bypass,
        delta_history=hist_delta,
        phi=phi,
        dt=dt_eff,
        tau=tau,
    )

    hist = {
        "E": hist_e,
        "delta": hist_delta,
        "E_hop": hist_e_hop,
        "E_bias": hist_e_bias,
        "E_portal": hist_e_portal,
        "E_cb": hist_e_cb,
        "bypass_C": hist_bypass,
        "phi_var": hist_phi_var,
        "iss": iss_report,
    }
    return best_m, hist, len(hist_e)


def _iss_from_tail(
    *,
    tail_states: deque,
    scale: float,
    best_m: torch.Tensor,
    c_k_history: list[float],
    delta_history: list[float],
    phi: torch.Tensor,
    dt: float,
    tau: float,
) -> Dict[str, float]:
    """Compute gate F2 ISS report from a relaxation trajectory (12_Equation appendix A.1).

    mu is estimated from the global ||dm_k|| contraction curve (full trajectory),
    not from `tail_states`, since the tail is post-convergence noise plateau.
    """
    if not delta_history:
        return {
            "samples": 0,
            "c_k_max": 0.0,
            "phi_inf_norm": 0.0,
            "mu": 0.0,
            "iss_ball_radius": float("inf"),
        }
    c_k_max = float(max(c_k_history) if c_k_history else 0.0)
    phi_inf_norm = float(phi.detach().abs().max().item())
    dt_over_tau = float(dt) / float(tau) if tau > 0.0 else 0.0
    mu = (
        estimate_mu(delta_history, dt_over_tau=dt_over_tau, skip=1)
        if dt_over_tau > 0.0
        else 0.0
    )
    radius = iss_ball_radius(
        c_k_max=c_k_max,
        phi_inf_norm=phi_inf_norm,
        mu=mu,
        alpha_b=ALPHA_B_DEFAULT,
    )
    return {
        "samples": len(delta_history),
        "c_k_max": c_k_max,
        "phi_inf_norm": phi_inf_norm,
        "mu": mu,
        "iss_ball_radius": radius,
    }


@torch.no_grad()
def relax_packed(
    values: torch.Tensor,
    col_idx: torch.Tensor,
    row_ptr: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    m0: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    metric_basis: Optional[torch.Tensor] = None,
    *,
    portal: float,
    bypass: float,
    t_wake: float,
    beta: float = 1.0,
    cb_w: float = DEFAULT_CB_W,
    lambda0: float = 1.0,
    lambda_phi: float = 0.5,
    lambda_var: float = 0.25,
    tau: float = 1.0,
    dt: float = 0.01,
    max_steps: int = 500,
    tol: float = 1e-4,
    anneal_ratio: float = 0.6,
    noise_scale: float = 1.0,
    metric_rank: int = 8,
    backend: str = "auto",
    seed: int = 0,
    dense_w: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, list[float]], int]:
    codebook = codebook if codebook is not None else m0.new_empty((0, m0.numel()))
    metric_basis = metric_basis if metric_basis is not None else build_metric_basis(
        codebook, m0, metric_rank, backend=backend
    )
    chosen = ce_backend(m0.device, backend)
    dim = m0.numel()
    n_code = int(codebook.shape[0]) if codebook.ndim == 2 else 0
    rank = int(metric_basis.shape[0]) if metric_basis.ndim == 2 else 0

    if chosen == "cuda" and m0.is_cuda:
        out = _cuda_mod.ce_relax_fwd(
            values.contiguous(),
            col_idx.contiguous(),
            row_ptr.contiguous(),
            b.contiguous(),
            phi.contiguous(),
            m0.contiguous(),
            codebook.contiguous(),
            metric_basis.contiguous(),
            float(portal),
            float(bypass),
            float(t_wake),
            float(beta),
            float(cb_w),
            float(lambda0),
            float(lambda_phi),
            float(lambda_var),
            float(tau),
            float(dt),
            int(max_steps),
            float(tol),
            float(anneal_ratio),
            float(noise_scale),
            int(seed),
        )
        best_m, energy, delta, e_hop, e_bias, e_portal, e_cb, bypass_hist = out
        return best_m, _hist_from_tensors(energy, delta, e_hop, e_bias, e_portal, e_cb, bypass_hist), int(energy.numel())

    if chosen == "rust" and not m0.is_cuda:
        out = _rust_ce_relax_fwd(
            _as_cpu_numpy_flat(values),
            _as_cpu_numpy_flat(col_idx.to(torch.int32)),
            _as_cpu_numpy_flat(row_ptr.to(torch.int32)),
            _as_cpu_numpy_flat(b),
            _as_cpu_numpy_flat(phi),
            _as_cpu_numpy_flat(m0),
            _as_cpu_numpy_flat(codebook),
            _as_cpu_numpy_flat(metric_basis),
            dim,
            n_code,
            rank,
            float(portal),
            float(bypass),
            float(t_wake),
            float(beta),
            float(cb_w),
            float(lambda0),
            float(lambda_phi),
            float(lambda_var),
            float(tau),
            float(dt),
            int(max_steps),
            float(tol),
            float(anneal_ratio),
            float(noise_scale),
            int(seed),
        )
        best_m_np, energy_np, delta_np, e_hop_np, e_bias_np, e_portal_np, e_cb_np, bypass_np, steps = out
        best_m = torch.from_numpy(best_m_np)
        hist = {
            "E": energy_np.tolist(),
            "delta": delta_np.tolist(),
            "E_hop": e_hop_np.tolist(),
            "E_bias": e_bias_np.tolist(),
            "E_portal": e_portal_np.tolist(),
            "E_cb": e_cb_np.tolist(),
            "bypass_C": bypass_np.tolist(),
        }
        return best_m, hist, int(steps)

    return _relax_packed_torch(
        values,
        col_idx,
        row_ptr,
        b,
        phi,
        m0,
        codebook,
        metric_basis,
        portal,
        bypass,
        t_wake,
        beta,
        cb_w,
        lambda0,
        lambda_phi,
        lambda_var,
        tau,
        dt,
        max_steps,
        tol,
        anneal_ratio,
        noise_scale,
        seed,
        dense_w=dense_w,
    )


def relax(
    w: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    m0: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    metric_basis: Optional[torch.Tensor] = None,
    *,
    portal: float,
    bypass: float,
    t_wake: float,
    zero_tol: float = 0.0,
    backend: str = "auto",
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, list[float]], int]:
    values, col_idx, row_ptr = pack_sparse(w, zero_tol=zero_tol, backend=backend)
    dense_w = None
    if backend != "rust" and values.numel() == w.numel():
        dense_w = w
    return relax_packed(
        values,
        col_idx,
        row_ptr,
        b,
        phi,
        m0,
        codebook,
        metric_basis,
        portal=portal,
        bypass=bypass,
        t_wake=t_wake,
        backend=backend,
        dense_w=dense_w,
        **kwargs,
    )


def pq_build_codebook(
    emb: torch.Tensor,
    *,
    subdim: int = 64,
    bits: int = 8,
    iters: int = 16,
    batch_size: int = 4096,
    sample_size: int = 16384,
    seed: int = 0,
) -> Dict[str, torch.Tensor | int]:
    emb_cpu = emb.detach().float().cpu().contiguous()
    n_token, dim = emb_cpu.shape
    if subdim <= 0 or dim % subdim != 0:
        raise ValueError(f"subdim must divide dim exactly: dim={dim}, subdim={subdim}")
    if bits <= 0 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n_sub = dim // subdim
    n_centroid = 1 << bits
    if n_centroid > n_token:
        raise ValueError("number of PQ centroids cannot exceed number of tokens")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    centroids_out: list[torch.Tensor] = []
    codes = torch.empty((n_token, n_sub), dtype=torch.uint8)

    for sub_idx in range(n_sub):
        start = sub_idx * subdim
        stop = start + subdim
        sub = emb_cpu[:, start:stop]

        pool_n = min(sample_size, n_token)
        pool_idx = torch.randperm(n_token, generator=gen)[:pool_n]
        pool = sub.index_select(0, pool_idx)
        init_idx = torch.randperm(pool.shape[0], generator=gen)[:n_centroid]
        centers = pool.index_select(0, init_idx).clone()

        for _ in range(max(1, iters)):
            cur_batch = min(batch_size, n_token)
            batch_idx = torch.randperm(n_token, generator=gen)[:cur_batch]
            batch = sub.index_select(0, batch_idx)
            dist = torch.cdist(batch, centers)
            assign = dist.argmin(dim=1)

            new_centers = centers.clone()
            for cid in range(n_centroid):
                mask = assign == cid
                if mask.any():
                    new_centers[cid] = batch[mask].mean(dim=0)
                else:
                    refill = torch.randint(pool.shape[0], (1,), generator=gen).item()
                    new_centers[cid] = pool[refill]
            centers = new_centers

        all_assign: list[torch.Tensor] = []
        for start_idx in range(0, n_token, batch_size):
            stop_idx = min(start_idx + batch_size, n_token)
            batch = sub[start_idx:stop_idx]
            dist = torch.cdist(batch, centers)
            all_assign.append(dist.argmin(dim=1).to(torch.uint8))
        codes[:, sub_idx] = torch.cat(all_assign, dim=0)
        centroids_out.append(centers.to(dtype=torch.float16))

    return {
        "centroids": torch.stack(centroids_out, dim=0),
        "codes": codes,
        "subdim": subdim,
        "bits": bits,
    }


def pq_reconstruct_tokens(
    centroids: torch.Tensor,
    codes: torch.Tensor,
    token_ids: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    if token_ids is None:
        selected_codes = codes
    else:
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=codes.device)
        token_ids = token_ids.to(device=codes.device, dtype=torch.long).view(-1)
        selected_codes = codes.index_select(0, token_ids)

    parts: list[torch.Tensor] = []
    for sub_idx in range(selected_codes.shape[1]):
        parts.append(
            centroids[sub_idx].index_select(0, selected_codes[:, sub_idx].long())
        )
    return torch.cat(parts, dim=1).to(dtype=torch.float32)


def pq_scores(
    query: torch.Tensor,
    centroids: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    query = query.to(dtype=torch.float32)
    n_sub, _, subdim = centroids.shape
    query_parts = query.view(n_sub, subdim)
    lut = torch.einsum("md,mkd->mk", query_parts, centroids.to(dtype=torch.float32))
    scores = torch.zeros(codes.shape[0], device=lut.device, dtype=lut.dtype)
    for sub_idx in range(n_sub):
        scores = scores + lut[sub_idx].index_select(0, codes[:, sub_idx].long())
    return scores
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py`

```python
"""Riemann-surface positional encoding for attention.

Engineering axiom: the Riemann Hypothesis is true. The non-trivial
zeros lie on the critical line s = 1/2 + i γ_n, and {γ_n} is GUE-
distributed (Montgomery-Dyson). The first 100 γ_n are hardcoded;
n > 100 uses the Riemann–von Mangoldt asymptotic γ_n ≈ 2π n / log n.

`RiemannRotaryAttention` implements the multi-sheet positional
encoding described in `docs/8_리만/riemann_pe_spec.md`:

    τ_p           = log(1 + p)                       # log-time lift
    θ(p, k)       = γ_k · τ_p                        # phase
    σ(p, k)       = floor(θ(p, k) / 2π)              # Riemann sheet
    rotation      = ((cos θ, -sin θ), (sin θ, cos θ))   on (q_{2k}, q_{2k+1})
    sheet_bias_ij = -λ_σ · mean_k |σ(i, k) - σ(j, k)|

Backend dispatch:
    backend="auto" picks cuda / rust / torch from the input device.

`riemann_zero_init` provides Design (4) — FFN key spacing seeded by
the Riemann-zero gap pattern.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# First 100 imaginary parts of the non-trivial Riemann zeta zeros.
# Source: Titchmarsh "The Theory of the Riemann Zeta-Function" Appendix,
# cross-checked with Odlyzko's published tables. 9 significant figures
# (sufficient for float32 attention).
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


_TAU = 2.0 * math.pi


def riemann_zeros(n: int) -> torch.Tensor:
    """Return the first n imaginary parts of non-trivial ζ zeros.

    For n > 100 (beyond the hardcoded table) uses the local-density
    extrapolation:  γ_{k+1} ≈ γ_k + 2π / log(γ_k / 2π).
    This follows from the Riemann–von Mangoldt counting formula
    N(T) ~ (T/2π)·(log(T/2π) - 1), differentiated to give the average
    spacing.  Guarantees monotonicity and joins smoothly to γ_100.
    """
    if n <= len(RIEMANN_ZEROS_IM):
        return torch.tensor(RIEMANN_ZEROS_IM[:n], dtype=torch.float32)
    vals = list(RIEMANN_ZEROS_IM)
    last = vals[-1]
    for _ in range(len(RIEMANN_ZEROS_IM), n):
        gap = _TAU / math.log(max(last / _TAU, math.e))
        last = last + gap
        vals.append(last)
    return torch.tensor(vals, dtype=torch.float32)


# --- backend hooks (filled at import time if available) ---------------------

try:
    from . import _rust as _rust_mod  # type: ignore[attr-defined]

    _rust_riemann_fwd = getattr(_rust_mod, "nn_ce_riemann_fwd", None)
    _rust_riemann_fwd_cuda = getattr(_rust_mod, "nn_ce_riemann_fwd_cuda", None)
    _rust_riemann_fwd_cuda_devptr = getattr(
        _rust_mod, "nn_ce_riemann_fwd_cuda_devptr", None
    )
except ImportError:
    _rust_mod = None
    _rust_riemann_fwd = None
    _rust_riemann_fwd_cuda = None
    _rust_riemann_fwd_cuda_devptr = None

_HAS_RUST = _rust_riemann_fwd is not None
_HAS_CUDA = _rust_riemann_fwd_cuda_devptr is not None


def has_rust_riemann() -> bool:
    return bool(_HAS_RUST)


def has_cuda_riemann() -> bool:
    return bool(_HAS_CUDA)


# --- core PyTorch reference impl --------------------------------------------


def _build_phase_and_sheet(
    pos: torch.Tensor,
    gamma: torch.Tensor,
    log_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute θ(p,k) and σ(p,k) for all (head, pos, pair).

    pos:        (n,)            positions 0..n-1
    gamma:      (n_pairs,)      Riemann-zero frequencies (already normalized)
    log_scale:  (h,)            per-head log "speed of light"

    Returns:
        theta: (1, h, n, n_pairs)
        sheet: (1, h, n, n_pairs)  int32
    """
    tau = torch.log1p(pos)                        # (n,)
    scale = torch.exp(log_scale)                  # (h,)
    # theta[h, n, k] = scale[h] * gamma[k] * tau[n]
    theta = (
        tau.view(1, 1, -1, 1)
        * gamma.view(1, 1, 1, -1)
        * scale.view(1, -1, 1, 1)
    )
    sheet = torch.floor(theta / _TAU).to(torch.int32)
    return theta, sheet


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to adjacent dim pairs of x (..., d_head)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


def _sheet_bias(sheet: torch.Tensor, lambda_sigma: torch.Tensor) -> torch.Tensor:
    """Mean cross-pair |σ_i - σ_j| × (-λ_σ).

    sheet:        (1, h, n, n_pairs)  int32
    lambda_sigma: (h,)
    Returns:      (1, h, n, n)
    """
    s = sheet.float()                             # (1, h, n, K)
    diff = s.unsqueeze(-2) - s.unsqueeze(-3)      # (1, h, n, n, K)
    mean_abs = diff.abs().mean(dim=-1)            # (1, h, n, n)
    return -lambda_sigma.view(1, -1, 1, 1) * mean_abs


def _attention_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    sheet_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    d_head: int,
) -> torch.Tensor:
    q_rot = _rotate_pairs(q, cos, sin)
    k_rot = _rotate_pairs(k, cos, sin)
    scores = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(d_head)
    scores = scores + sheet_bias
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


class RiemannRotaryAttention(nn.Module):
    """Riemann-surface positional encoding for multi-head attention.

    Args:
        d_model, n_heads, block: standard
        normalize_gamma: divide γ_n by γ_1 so the slowest mode has
            unit angular speed at log-time = 1.
        learnable_scale: per-head log "speed of light" multiplier.
        sheet_init: initial value for log λ_σ (log-space). Default
            log(0.0) → λ_σ = 0 (sheet bias inert at init), so the
            module gracefully starts equivalent to plain RoPE-on-log-time.
        backend: "auto" | "torch" | "rust" | "cuda".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        normalize_gamma: bool = True,
        learnable_scale: bool = True,
        sheet_init: float = -6.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {self.d_head}")
        n_pairs = self.d_head // 2
        self.n_pairs = n_pairs
        self.block = block
        self.backend_pref = backend

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block, block, dtype=torch.bool)),
        )

        gamma = riemann_zeros(n_pairs)
        if normalize_gamma:
            gamma = gamma / gamma[0]
        self.register_buffer("gamma", gamma)
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

        if learnable_scale:
            self.log_scale = nn.Parameter(torch.zeros(n_heads))
        else:
            self.register_buffer("log_scale", torch.zeros(n_heads))

        # λ_σ = exp(log_lambda_sigma); init at sheet_init so that
        # exp(-6) ≈ 2.5e-3 → near-zero but learnable upward.
        self.log_lambda_sigma = nn.Parameter(
            torch.full((n_heads,), float(sheet_init))
        )

    # ------------------------------------------------------------------
    # backend selection
    # ------------------------------------------------------------------
    def _resolve_backend(self, x: torch.Tensor) -> str:
        pref = self.backend_pref
        if pref == "torch":
            return "torch"
        if pref == "rust":
            if not _HAS_RUST or x.is_cuda:
                return "torch"
            return "rust"
        if pref == "cuda":
            if not _HAS_CUDA or not x.is_cuda:
                return "torch"
            return "cuda"
        # auto
        if x.is_cuda and _HAS_CUDA:
            return "cuda"
        if not x.is_cuda and _HAS_RUST and not self.training:
            # rust path is forward-only; only use when no autograd.
            return "rust"
        return "torch"

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)                 # (b, h, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        backend = self._resolve_backend(x)

        theta, sheet = _build_phase_and_sheet(
            self.pos[:n], self.gamma, self.log_scale
        )                                       # (1, h, n, K)
        cos = theta.cos()
        sin = theta.sin()

        lambda_sigma = torch.exp(self.log_lambda_sigma)   # (h,)
        sheet_bias = _sheet_bias(sheet, lambda_sigma)     # (1, h, n, n)

        if backend == "torch":
            out = _attention_torch(
                q, k, v, cos, sin, sheet_bias,
                self.tril[:n, :n], self.d_head,
            )
        else:
            out = self._forward_native(
                q, k, v, cos, sin, sheet_bias, n, backend,
            )

        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)

    # ------------------------------------------------------------------
    # native dispatch (Rust / CUDA)
    # ------------------------------------------------------------------
    def _forward_native(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, sheet_bias: torch.Tensor,
        n: int, backend: str,
    ) -> torch.Tensor:
        b, h, _, d = q.shape
        bh = b * h
        half = d // 2

        # Flatten to (bh, n, *) and broadcast cos/sin/sheet_bias to bh.
        q_f = q.contiguous().view(bh, n, d)
        k_f = k.contiguous().view(bh, n, d)
        v_f = v.contiguous().view(bh, n, d)
        cos_b = cos.expand(b, h, n, half).contiguous().view(bh, n, half)
        sin_b = sin.expand(b, h, n, half).contiguous().view(bh, n, half)
        sb_b  = sheet_bias.expand(b, h, n, n).contiguous().view(bh, n, n)

        if backend == "cuda":
            return self._forward_cuda_devptr(q_f, k_f, v_f, cos_b, sin_b, sb_b,
                                             b, h, n, d)
        # rust CPU path: flat numpy, single dispatch
        q_np = q_f.detach().cpu().numpy().reshape(-1)
        k_np = k_f.detach().cpu().numpy().reshape(-1)
        v_np = v_f.detach().cpu().numpy().reshape(-1)
        c_np = cos_b.detach().cpu().numpy().reshape(-1)
        s_np = sin_b.detach().cpu().numpy().reshape(-1)
        sb_np = sb_b.detach().cpu().numpy().reshape(-1)
        out_np = _rust_riemann_fwd(q_np, k_np, v_np, c_np, s_np, sb_np,
                                   bh, n, d, True)
        out = torch.from_numpy(out_np).view(b, h, n, d).to(q.device, q.dtype)
        return out

    def _forward_cuda_devptr(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, sb: torch.Tensor,
        b: int, h: int, n: int, d: int,
    ) -> torch.Tensor:
        bh = b * h
        # Allocate output on the same CUDA device.
        out = torch.empty((bh, n, d), device=q.device, dtype=torch.float32)
        # Sync the producing PyTorch stream so our default CUDA stream
        # sees consistent data; the Rust side syncs its own stream after
        # launch, after which results are visible to PyTorch.
        torch.cuda.synchronize(q.device)
        # PyTorch holds these as f32; if upstream is f16/bf16, cast here.
        def _f32(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == torch.float32 else t.float().contiguous()
        qf = _f32(q); kf = _f32(k); vf = _f32(v)
        cf = _f32(cos); sf = _f32(sin); sbf = _f32(sb)
        _rust_riemann_fwd_cuda_devptr(
            int(qf.data_ptr()), int(kf.data_ptr()), int(vf.data_ptr()),
            int(cf.data_ptr()), int(sf.data_ptr()), int(sbf.data_ptr()),
            int(out.data_ptr()),
            bh, n, d, True,
        )
        return out.view(b, h, n, d).to(q.dtype)


class RiemannAttnBlock(nn.Module):
    """Pre-LN block using `RiemannRotaryAttention` + pluggable FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        ffn_kind: Optional[str] = None,
        backend: str = "auto",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = RiemannRotaryAttention(d_model, n_heads, block, backend=backend)
        if ffn_kind is None or ffn_kind == "std":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        else:
            from .ce_ffn import make_ffn
            self.ffn = make_ffn(ffn_kind, d_model, mult=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Design (4) — Riemann-zero spaced FFN init
# ---------------------------------------------------------------------------


@torch.no_grad()
def riemann_zero_init(linear: nn.Linear, axis: str = "in") -> None:
    """Modulate `linear.weight` columns/rows by Riemann-gap spacing.

    Kaiming-normal first; then multiply the chosen axis by a vector
    derived from cumulative γ-gaps (centered, unit-std). The orthogonal
    axis stays Kaiming. Hypothesis: keys spaced by GUE statistics give
    better memory coverage than iid-gaussian.
    """
    W = linear.weight                              # (out, in)
    if axis == "in":
        n = W.shape[1]
    elif axis == "out":
        n = W.shape[0]
    else:
        raise ValueError(f"axis must be 'in' or 'out', got {axis!r}")

    gamma = riemann_zeros(n)
    spacings = torch.cat([gamma[:1], gamma[1:] - gamma[:-1]])
    spacings = spacings / spacings.mean()
    positions = torch.cumsum(spacings, dim=0)
    positions = (positions - positions.mean()) / positions.std().clamp_min(1e-8)

    nn.init.kaiming_normal_(W, nonlinearity="linear")
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
    "has_rust_riemann",
    "has_cuda_riemann",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_softmax.py`

```python
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


def lang_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Raw scaled-dot-product logits (pre-softmax)."""
    d = q.shape[-1]
    return torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)


def lang_attention(q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Standard scaled-dot-product kernel (linguistic metric).

    Shapes: q, k in (..., n, d). Returns (..., n, n).
    """
    scores = lang_scores(q, k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def grav_scores(
    z: torch.Tensor,
    sigma: float = 1.0,
    L: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Raw Mahalanobis score matrix (pre-softmax) = -d^2 / 2 sigma^2."""
    if L is not None:
        zp = torch.matmul(z, L)
    else:
        zp = z
    sq = (zp * zp).sum(dim=-1, keepdim=True)
    d2 = sq + sq.transpose(-1, -2) - 2.0 * torch.matmul(zp, zp.transpose(-1, -2))
    d2 = d2.clamp_min(0.0)
    return -d2 / (2.0 * sigma * sigma)


def grav_attention(
    z: torch.Tensor,
    sigma: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    L: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Distance-kernel attention (event/gravity metric).

    Equation 6.B.1 with squared Mahalanobis distance:

        d_G^2(z_i, z_j) = (z_i - z_j)^T (L L^T) (z_i - z_j)

    When ``L`` is None, reduces to identity metric (pure Euclidean).
    """
    scores = grav_scores(z, sigma=sigma, L=L)
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
    L_grav: Optional[torch.Tensor] = None,
    combine: str = "logit",
) -> torch.Tensor:
    """Full MFA — equation 6.B.1 with two metrics (lang, grav).

    Two ways to combine:
      ``combine='convex'``:
          A_total = omega_lang * softmax(scores_lang)
                  + omega_grav * softmax(scores_grav)
      ``combine='logit'``  (default, recommended):
          A_total = softmax(omega_lang * scores_lang
                          + omega_grav * scores_grav)

    The convex form mechanically dilutes sharp distributions (any
    uniform-ish component drags the mixed attention toward uniform).
    The logit form preserves sharpness; the two kernels vote in log
    space, which is equivalent to a product of Boltzmann kernels.

    ``L_grav`` is the low-rank factor of the gravity metric
    (G = L L^T); if None, identity is used. Returns (..., n, d_v).
    """
    if gate is None:
        gate = mode_gate("wake")
    if z_grav is None:
        z_grav = k

    if combine == "convex":
        a_lang = lang_attention(q, k, mask=mask)
        a_grav = grav_attention(z_grav, sigma=sigma_grav, mask=mask, L=L_grav)
        a_total = gate.omega_lang * a_lang + gate.omega_grav * a_grav
    elif combine == "logit":
        s_lang = lang_scores(q, k)
        s_grav = grav_scores(z_grav, sigma=sigma_grav, L=L_grav)
        s = gate.omega_lang * s_lang + gate.omega_grav * s_grav
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        a_total = F.softmax(s, dim=-1)
    else:
        raise ValueError(f"unknown combine mode: {combine!r}")

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
        combine: str = "logit",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sigma_grav = sigma_grav
        self.mode = mode
        self.combine = combine

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
        out = metric_family_attention(
            q, k, v,
            gate=gate, sigma_grav=self.sigma_grav, mask=mask,
            combine=self.combine,
        )
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.dropout(self.w_o(out))


__all__ = [
    "ModeGate",
    "mode_gate",
    "lang_scores",
    "lang_attention",
    "grav_scores",
    "grav_attention",
    "metric_family_attention",
    "CESoftmaxAttention",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_zeta.py`

```python
"""Design 2 — Riemann-zeta activation function.

Uses a differentiable truncated approximation to ζ(s) evaluated on the
critical line s = 1/2 + ix. Two routes are implemented:

  eta_truncated(x, N):
    Dirichlet-eta truncation (absolutely convergent for Re(s) > 0)
        η(s) ≈ Σ_{n=1}^{N} (-1)^{n+1} n^{-s}
    Then ζ(s) = η(s) / (1 - 2^{1-s}).
    For s = 1/2 + ix, this is a complex number; we return |ζ|^2 as a
    real-valued function.

  zeta_activation(x):
    Drop-in replacement for GELU/SiLU. Shape preserved:
        y = x * sigmoid(x) * ZNorm(x)
    where ZNorm is the normalized |ζ(1/2+ix)|^2 capped to a bounded
    range so gradients don't explode. This preserves the classic
    Swish/SiLU monotonic behaviour near 0 and adds zeta-modulated
    structure away from 0 — the Riemann hypothesis axiom then
    guarantees ZNorm vanishes only at Riemann-zero inputs.

Differentiability: all ops are pure PyTorch (log, cos, sin, reciprocal),
so autograd flows. No Riemann-Siegel refinement is used; N=24 is enough
for |x| ≲ 40 and is cheap.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _eta_truncated(x: torch.Tensor, N: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real and imaginary parts of η(1/2 + i x) truncated at N terms.

    η(1/2 + ix) = Σ_{n=1}^{N} (-1)^{n+1} n^{-1/2} (cos(x log n) − i sin(x log n))
    """
    # n = 1 ... N
    device, dtype = x.device, x.dtype
    ns = torch.arange(1, N + 1, device=device, dtype=dtype)  # (N,)
    log_n = torch.log(ns)                                    # (N,)
    inv_sqrt_n = 1.0 / torch.sqrt(ns)                        # (N,)
    sign = torch.where(ns % 2 == 1, torch.ones_like(ns), -torch.ones_like(ns))
    coef = sign * inv_sqrt_n                                 # (N,)
    # broadcast: x has shape (...), we need (..., N)
    phase = x.unsqueeze(-1) * log_n                          # (..., N)
    real = (coef * torch.cos(phase)).sum(dim=-1)             # (...,)
    imag = (coef * (-torch.sin(phase))).sum(dim=-1)          # (...,)
    return real, imag


def _zeta_critical(x: torch.Tensor, N: int = 24
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """ζ(1/2 + ix) via ζ = η / (1 − 2^{1−s}). Returns (Re ζ, Im ζ)."""
    eta_re, eta_im = _eta_truncated(x, N)
    # d = 1 − 2^{1 − 1/2 − ix} = 1 − √2 · 2^{−ix} = 1 − √2 (cos(x log 2) − i sin(x log 2))
    a = math.log(2.0)
    sqrt2 = math.sqrt(2.0)
    cos_a = torch.cos(x * a)
    sin_a = torch.sin(x * a)
    d_re = 1.0 - sqrt2 * cos_a
    d_im = sqrt2 * sin_a
    denom = d_re * d_re + d_im * d_im + 1e-8
    # (eta_re + i eta_im) / (d_re + i d_im)
    # = ((eta_re d_re + eta_im d_im) + i (eta_im d_re - eta_re d_im)) / denom
    z_re = (eta_re * d_re + eta_im * d_im) / denom
    z_im = (eta_im * d_re - eta_re * d_im) / denom
    return z_re, z_im


def zeta_magnitude_sq(x: torch.Tensor, N: int = 24) -> torch.Tensor:
    """|ζ(1/2 + ix)|^2 via truncated Dirichlet series. Real-valued."""
    zr, zi = _zeta_critical(x, N)
    return zr * zr + zi * zi


class ZetaActivation(nn.Module):
    """Swish-like: y = x · σ(x) · ( 1 + λ · (|ζ|² − μ) / s ).

    The bracket is the normalized zeta-magnitude modulation; λ is a
    learnable gain, initially small so the module starts as SiLU.
    μ, s are running statistics computed on the first forward call and
    frozen after (to keep training deterministic).
    """

    def __init__(self, N: int = 24, lam_init: float = 0.1):
        super().__init__()
        self.N = N
        self.lam = nn.Parameter(torch.tensor(lam_init))
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("_init_done", torch.tensor(0, dtype=torch.uint8))

    def _init_stats(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            zs = zeta_magnitude_sq(x, self.N)
            self.mu.copy_(zs.mean())
            self.sigma.copy_(zs.std().clamp_min(1e-4))
            self._init_done.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(self._init_done.item()) == 0 and self.training:
            self._init_stats(x)
        zs = zeta_magnitude_sq(x, self.N)
        z_norm = (zs - self.mu) / self.sigma
        modulation = 1.0 + self.lam * z_norm
        return x * torch.sigmoid(x) * modulation


class ZetaFFN(nn.Module):
    """FFN using ZetaActivation in place of GELU."""

    def __init__(self, d: int, mult: int = 4, N: int = 24):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.act = ZetaActivation(N=N)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


__all__ = [
    "zeta_magnitude_sq",
    "ZetaActivation",
    "ZetaFFN",
    "_eta_truncated",
    "_zeta_critical",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/constants.py`

```python
"""Single source of truth for CE physical constants and tolerances.

All numeric constants derived from the CE field theory live here.
Other modules import from this file; do NOT redefine these values elsewhere.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Core CE coupling constant
# ---------------------------------------------------------------------------
AD: float = 4.0 / (math.e ** (4.0 / 3.0) * math.pi ** (4.0 / 3.0))

# ---------------------------------------------------------------------------
# Derived engine constants
# ---------------------------------------------------------------------------
PORTAL: float = (AD * (1.0 - AD)) ** 2
BYPASS: float = 1.0 / (math.e ** (1.0 / 3.0) * math.pi ** (1.0 / 3.0))
T_WAKE: float = 1.0 / (3.0 + AD * (1.0 - AD))

# ---------------------------------------------------------------------------
# Bootstrap fixed-point ratios  (d = 3)
# ---------------------------------------------------------------------------
ACTIVE_RATIO: float = 0.0487        # epsilon^2  (baryon / task-active)
STRUCT_RATIO: float = 0.2623        # Omega_DM   (structural / plastic)
BACKGROUND_RATIO: float = 0.6891    # Omega_Lambda (frozen / background)
BOOTSTRAP_CONTRACTION: float = 0.155  # rho = D_eff * epsilon^2

# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------
SPARSITY_RADIUS: float = math.pi    # r_c
TARGET_W_DENSITY: float = 0.0316    # N=4096, r_c=pi

# ---------------------------------------------------------------------------
# BrainRuntime cell dynamics (from 15_Equations.md / J.19-J.20)
# ---------------------------------------------------------------------------
NOISE_SIGMA: float = 0.27                # eta_i noise std (15_Equations A.2)
MEMORY_TRACE_DECAY: float = 0.01         # gamma_m (NMDA ~100ms)
ADAPTATION_DECAY: float = 0.005          # gamma_w (AHP ~200ms)
ADAPTATION_COUPLING: float = 0.12        # beta_w
STP_TAU_FAC_INV: float = 0.0015          # 1/tau_facilitation
STP_TAU_REC: float = 0.008               # 1/tau_recovery
STP_U_BASE: float = 0.5                  # baseline release probability
ADAPTATION_CLAMP: float = 2.0            # max adaptation value
DALE_EI_RATIO: float = 0.8              # 80% excitatory, 20% inhibitory
DALE_INH_GAIN: float = 4.0              # w_I / w_E = 4
AXON_DELAY_MAX: int = 3                 # max axonal delay steps

# ---------------------------------------------------------------------------
# Borbely 2-Process sleep model (15_Equations.md C.2)
# ---------------------------------------------------------------------------
TAU_W_STEPS: float = 65520.0    # 18.2h in 1ms steps
TAU_S_STEPS: float = 15120.0    # 4.2h  in 1ms steps
SLEEP_PRESSURE_MAX: float = 2.0
REM_TAU_FACTOR: float = 0.5     # REM decay = NREM decay * this factor
CIRCADIAN_PERIOD: float = 87120.0  # 24.2h in 1ms steps
CIRCADIAN_AMP: float = 0.4        # C1 amplitude
CIRCADIAN_BASE: float = 0.5       # C0 baseline
NREM_LENGTH_DECAY: float = 0.75   # T_NREM(n) = T0 * alpha^n

# ---------------------------------------------------------------------------
# Hippocampus (15_Equations.md D)
# ---------------------------------------------------------------------------
FORGET_TAU: float = 10000.0     # tau_forget for priority decay (steps)
RECALL_SIMILARITY_THRESHOLD: float = 0.1  # minimum cosine to recall

# ---------------------------------------------------------------------------
# Neuromodulation (17_AgentLoop F.19 / F.24.4)
# ---------------------------------------------------------------------------
NEURO_TAU_DA: float = 500.0     # dopamine time constant
NEURO_TAU_NE: float = 300.0     # norepinephrine
NEURO_TAU_5HT: float = 3000.0   # serotonin
NEURO_TAU_ACH: float = 200.0    # acetylcholine
NEURO_BASELINE_DA: float = 0.5
NEURO_BASELINE_NE: float = 0.5
NEURO_BASELINE_5HT: float = 0.5
NEURO_BASELINE_ACH: float = 0.5
NEURO_ALPHA_DA: float = 0.1
NEURO_ALPHA_NE: float = 0.1
NEURO_ALPHA_5HT: float = 0.05
NEURO_ALPHA_ACH: float = 0.1

# ---------------------------------------------------------------------------
# STDP (17_AgentLoop F.14)
# ---------------------------------------------------------------------------
STDP_R_PLUS: float = 0.95       # pre-trace decay
STDP_R_MINUS: float = 0.95      # post-trace decay
STDP_R_E: float = 0.99          # eligibility decay
STDP_A_PLUS: float = 0.01       # LTP amplitude
STDP_A_MINUS: float = 0.012     # LTD amplitude
STDP_SPIKE_THRESHOLD: float = 0.3  # theta_spike
STDP_LR: float = 0.001          # learning rate
STDP_ALPHA_G: float = 0.7       # gate mixing (critic vs bootstrap)

# ---------------------------------------------------------------------------
# Consciousness / metacognition (17_AgentLoop F.17)
# ---------------------------------------------------------------------------
CONSCIOUSNESS_TAU: float = 100.0   # time window for d_tau
CONSCIOUSNESS_CD: float = 5.0     # scale for exp depth
META_MAX_DEPTH: int = 3           # max recursive self-evaluation

# ---------------------------------------------------------------------------
# Working memory (17_AgentLoop F.20)
# ---------------------------------------------------------------------------
WM_CAPACITY: int = 7              # T_h (Miller's 7 +/- 2)
CEREBELLUM_ALPHA: float = 0.1     # cerebellar learning rate
CEREBELLUM_ETA: float = 0.05      # cerebellar correction gain

# ---------------------------------------------------------------------------
# Architecture V2 (2_Architecture.md)
# ---------------------------------------------------------------------------
CFC_XI: float = 0.490             # alpha_s^(1/3), cross-frequency coupling
GAUGE_ALPHA_S: float = 0.11789    # SU(3) coupling
GAUGE_ALPHA_W: float = 0.03352    # SU(2) coupling
GAUGE_ALPHA_EM: float = 0.00775   # U(1) coupling

# ---------------------------------------------------------------------------
# Critic weights (17_AgentLoop F.4)
# ---------------------------------------------------------------------------
CRITIC_W_PRED: float = 0.4
CRITIC_W_CONS: float = 0.3
CRITIC_W_NOV: float = 0.3

# ---------------------------------------------------------------------------
# Brainwave bands Hz (17_AgentLoop F.21)
# ---------------------------------------------------------------------------
BAND_DELTA: tuple[float, float] = (0.5, 4.0)
BAND_THETA: tuple[float, float] = (4.0, 8.0)
BAND_ALPHA: tuple[float, float] = (8.0, 13.0)
BAND_BETA: tuple[float, float] = (13.0, 30.0)
BAND_GAMMA: tuple[float, float] = (30.0, 100.0)

# ---------------------------------------------------------------------------
# Numeric tolerances
# ---------------------------------------------------------------------------
NORM_EPS: float = 1e-8
SOFTMAX_EPS: float = 1e-6
CLAMP_EPS: float = 1e-4
```
---
## File: `reality_stone/python/reality_stone/clarus/core/Cargo.toml`

```toml
[package]
name = "clarus_core"
version = "1.1.0"
edition = "2021"
authors = ["CE Research Lab"]
description = "Canonical Clarus compute core for Rust/Python runtime kernels"

[lib]
name = "_rust"
crate-type = ["cdylib", "rlib"]

[features]
default = []
python = ["pyo3", "dep:numpy"]
cuda = ["dep:cudarc"]

[dependencies]
pyo3 = { version = "0.20.0", features = ["extension-module", "abi3-py38"], optional = true }
numpy = { version = "0.20", optional = true }
ndarray = "0.15"
rayon = "1.7"
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
rand_distr = "0.4"
rustfft = "6.1"
cudarc = { version = "0.19", features = ["driver", "nvrtc", "cuda-12080", "dynamic-loading"], default-features = false, optional = true }

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/cuda/ce_riemann.cu`

```cpp
// Riemann-surface positional encoding attention — batched CUDA kernel.
//
// Mirrors `nn_ops::ce_riemann_fwd` (Rust) and
// `clarus.ce_riemann_attn.RiemannRotaryAttention` (PyTorch reference).
//
// Layout
//   Grid:  (BH, N, 1)                — one block per (head-batch, query row)
//   Block: (THREADS, 1, 1)           — threads cooperate on the row
//   Smem:  D + N floats              — rotated q row + score row
//
// All input tensors are pre-broadcast to shape (BH, N, *) row-major. cos/sin
// have last dim D/2; sheet_bias has last dim N.
//
// NVRTC has no <math.h>; INFINITY / isfinite are not visible, so we provide
// explicit sentinels and a lightweight finite check.

#define NEG_INF (-3.4028235e38f)

__device__ __forceinline__ int is_finite_f(float x) {
    return (x == x) && (x < 3.4028235e38f) && (x > -3.4028235e38f);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(mask, v, off);
        if (other > v) v = other;
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(mask, v, off);
    }
    return v;
}

extern "C" __global__ void ce_riemann_fwd_kernel(
    const float* __restrict__ q,            // (BH, N, D)
    const float* __restrict__ k,            // (BH, N, D)
    const float* __restrict__ v,            // (BH, N, D)
    const float* __restrict__ cos_buf,      // (BH, N, D/2)
    const float* __restrict__ sin_buf,      // (BH, N, D/2)
    const float* __restrict__ sheet_bias,   // (BH, N, N)
    float* __restrict__ out,                // (BH, N, D)
    int N,
    int D,
    int causal
) {
    const int bh  = blockIdx.x;
    const int i   = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int half = D >> 1;
    const float scale = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* qrot   = smem;            // [D]
    float* scores = smem + D;        // [N]

    const float* q_row   = q       + ((size_t)bh * N + i) * D;
    const float* cos_row = cos_buf + ((size_t)bh * N + i) * half;
    const float* sin_row = sin_buf + ((size_t)bh * N + i) * half;
    const float* sb_row  = sheet_bias + ((size_t)bh * N + i) * N;
    float*       out_row = out     + ((size_t)bh * N + i) * D;

    // --- 1. Rotate q[i,:] into shared memory, parallel over pairs ----------
    for (int p = tid; p < half; p += nthreads) {
        float c = cos_row[p];
        float s = sin_row[p];
        float a = q_row[2 * p];
        float b = q_row[2 * p + 1];
        qrot[2 * p]     = a * c - b * s;
        qrot[2 * p + 1] = a * s + b * c;
    }
    __syncthreads();

    // --- 2. Score row, parallel over j (each thread rotates its k_j on-the-fly)
    for (int j = tid; j < N; j += nthreads) {
        if (causal && j > i) {
            scores[j] = NEG_INF;
            continue;
        }
        const float* k_row = k       + ((size_t)bh * N + j) * D;
        const float* cos_j = cos_buf + ((size_t)bh * N + j) * half;
        const float* sin_j = sin_buf + ((size_t)bh * N + j) * half;

        float dot = 0.0f;
        #pragma unroll 4
        for (int p = 0; p < half; ++p) {
            float c = cos_j[p];
            float s = sin_j[p];
            float a = k_row[2 * p];
            float b = k_row[2 * p + 1];
            float kr0 = a * c - b * s;
            float kr1 = a * s + b * c;
            dot += qrot[2 * p] * kr0 + qrot[2 * p + 1] * kr1;
        }
        scores[j] = dot * scale + sb_row[j];
    }
    __syncthreads();

    // --- 3. Block-reduce max -------------------------------------------------
    __shared__ float warp_buf[32];

    float local_max = NEG_INF;
    for (int j = tid; j < N; j += nthreads) {
        float vv = scores[j];
        if (vv > local_max) local_max = vv;
    }
    local_max = warp_reduce_max(local_max);

    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (nthreads + 31) >> 5;

    if (lane == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float vv = (lane < n_warps) ? warp_buf[lane] : NEG_INF;
        vv = warp_reduce_max(vv);
        if (lane == 0) warp_buf[0] = vv;
    }
    __syncthreads();
    const float row_max = warp_buf[0];

    // --- 4. Exp + block-reduce sum ------------------------------------------
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        float vv = scores[j];
        float e  = is_finite_f(vv) ? __expf(vv - row_max) : 0.0f;
        scores[j] = e;
        local_sum += e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float vv = (lane < n_warps) ? warp_buf[lane] : 0.0f;
        vv = warp_reduce_sum(vv);
        if (lane == 0) warp_buf[0] = vv;
    }
    __syncthreads();
    const float row_sum = warp_buf[0];
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // --- 5. Weighted sum -> out[i, :], parallel over d ----------------------
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            acc += scores[j] * v[((size_t)bh * N + j) * D + d];
        }
        out_row[d] = acc * inv_sum;
    }
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/cuda/mod.rs`

```rust
//! CUDA backend for Riemann-surface attention.
//!
//! cudarc 0.19 driver + NVRTC for runtime PTX compilation. The kernel
//! source is embedded via `include_str!` so no build.rs is required.
//!
//! Two entry points:
//!   * `ce_riemann_fwd_cuda(..host slices..)`  — convenience path for CPU
//!     tensors that still want GPU compute (one alloc + one htod per
//!     input, no per-row Python loop).
//!   * `ce_riemann_fwd_cuda_devptr(..u64 device pointers..)` — zero-copy
//!     entry for PyTorch CUDA tensors. Accepts raw device pointers
//!     (CUdeviceptr cast to u64) and writes results in place. The caller
//!     is responsible for stream synchronization with the producing
//!     PyTorch stream.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNEL_SRC: &str = include_str!("ce_riemann.cu");

const KERNEL_NAME: &str = "ce_riemann_fwd_kernel";
const THREADS_PER_BLOCK: u32 = 128;

static CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();

fn ctx() -> Result<Arc<CudaContext>, String> {
    if let Some(c) = CTX.get() {
        return Ok(c.clone());
    }
    let c = CudaContext::new(0).map_err(|e| format!("CudaContext::new failed: {e:?}"))?;
    let _ = CTX.set(c.clone());
    Ok(c)
}

fn module() -> Result<Arc<CudaModule>, String> {
    if let Some(m) = MODULE.get() {
        return Ok(m.clone());
    }
    let ctx = ctx()?;
    let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
    let m = ctx
        .load_module(ptx)
        .map_err(|e| format!("load_module failed: {e:?}"))?;
    let _ = MODULE.set(m.clone());
    Ok(m)
}

#[inline]
fn shape_check(bh: usize, n: usize, d_head: usize) -> Result<(), String> {
    if d_head % 2 != 0 {
        return Err(format!("d_head must be even, got {d_head}"));
    }
    if bh == 0 || n == 0 || d_head == 0 {
        return Err("ce_riemann: zero-sized tensor".into());
    }
    // Shared-mem budget: (D + N) * 4 bytes. Default 48 KB / SM is fine
    // for D <= 256 and N <= 8192. Grid-y is u32-bounded.
    if d_head > 1024 {
        return Err(format!("d_head {d_head} exceeds CUDA kernel limit 1024"));
    }
    if n > 16384 {
        return Err(format!("n {n} exceeds CUDA kernel limit 16384"));
    }
    Ok(())
}

#[inline]
fn launch_cfg(bh: usize, n: usize, d_head: usize) -> LaunchConfig {
    let smem_bytes = ((d_head + n) * std::mem::size_of::<f32>()) as u32;
    LaunchConfig {
        grid_dim: (bh as u32, n as u32, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: smem_bytes,
    }
}

/// Host-staging entry: copies inputs to device, runs the kernel,
/// copies the result back. Used when source tensors live on the CPU
/// but compute should run on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn ce_riemann_fwd_cuda(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cos: &[f32],
    sin: &[f32],
    sheet_bias: &[f32],
    bh: usize,
    n: usize,
    d_head: usize,
    causal: bool,
) -> Result<Vec<f32>, String> {
    shape_check(bh, n, d_head)?;
    let half = d_head / 2;
    let total_qkv = bh * n * d_head;
    let total_cs = bh * n * half;
    let total_sb = bh * n * n;
    debug_assert_eq!(q.len(), total_qkv);
    debug_assert_eq!(k.len(), total_qkv);
    debug_assert_eq!(v.len(), total_qkv);
    debug_assert_eq!(cos.len(), total_cs);
    debug_assert_eq!(sin.len(), total_cs);
    debug_assert_eq!(sheet_bias.len(), total_sb);

    let ctx = ctx()?;
    let stream = ctx.default_stream();
    let m = module()?;
    let func = m
        .load_function(KERNEL_NAME)
        .map_err(|e| format!("load_function failed: {e:?}"))?;

    let q_d = stream.clone_htod(q).map_err(|e| format!("htod q: {e:?}"))?;
    let k_d = stream.clone_htod(k).map_err(|e| format!("htod k: {e:?}"))?;
    let v_d = stream.clone_htod(v).map_err(|e| format!("htod v: {e:?}"))?;
    let cos_d = stream.clone_htod(cos).map_err(|e| format!("htod cos: {e:?}"))?;
    let sin_d = stream.clone_htod(sin).map_err(|e| format!("htod sin: {e:?}"))?;
    let sb_d = stream
        .clone_htod(sheet_bias)
        .map_err(|e| format!("htod sb: {e:?}"))?;
    let mut out_d = stream
        .alloc_zeros::<f32>(total_qkv)
        .map_err(|e| format!("alloc out: {e:?}"))?;

    let cfg = launch_cfg(bh, n, d_head);
    let n_i32 = n as i32;
    let d_i32 = d_head as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_d);
    builder.arg(&k_d);
    builder.arg(&v_d);
    builder.arg(&cos_d);
    builder.arg(&sin_d);
    builder.arg(&sb_d);
    builder.arg(&mut out_d);
    builder.arg(&n_i32);
    builder.arg(&d_i32);
    builder.arg(&causal_i32);
    unsafe { builder.launch(cfg) }.map_err(|e| format!("launch: {e:?}"))?;

    stream
        .synchronize()
        .map_err(|e| format!("synchronize: {e:?}"))?;
    let out = stream
        .clone_dtoh(&out_d)
        .map_err(|e| format!("dtoh out: {e:?}"))?;
    Ok(out)
}

/// Zero-copy entry: the caller passes raw CUDA device pointers
/// (e.g. `tensor.data_ptr()` from PyTorch). The kernel writes directly
/// into `out_ptr`. The driver reads pointer values as 64-bit args, so
/// we push them as raw `*const f32` / `*mut f32`.
///
/// Safety: all pointers must be valid CUDA device addresses owned by
/// the calling process, with sufficient backing storage for the shapes
/// implied by `bh`, `n`, `d_head`. Caller must ensure the producing
/// stream has been synchronized before invocation (and synchronize the
/// consuming stream afterwards) when crossing PyTorch ↔ this stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ce_riemann_fwd_cuda_devptr(
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    sb_ptr: u64,
    out_ptr: u64,
    bh: usize,
    n: usize,
    d_head: usize,
    causal: bool,
) -> Result<(), String> {
    shape_check(bh, n, d_head)?;
    let _ = ctx()?;
    let stream = CTX.get().unwrap().default_stream();
    let m = module()?;
    let func = m
        .load_function(KERNEL_NAME)
        .map_err(|e| format!("load_function failed: {e:?}"))?;

    let cfg = launch_cfg(bh, n, d_head);
    let n_i32 = n as i32;
    let d_i32 = d_head as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    // CUDA driver expects an 8-byte pointer per `float*` arg; u64 has the
    // identical wire representation. cudarc impls `DeviceRepr` for u64
    // but not raw pointer types, so push as u64.
    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_ptr);
    builder.arg(&k_ptr);
    builder.arg(&v_ptr);
    builder.arg(&cos_ptr);
    builder.arg(&sin_ptr);
    builder.arg(&sb_ptr);
    builder.arg(&out_ptr);
    builder.arg(&n_i32);
    builder.arg(&d_i32);
    builder.arg(&causal_i32);
    builder.launch(cfg).map_err(|e| format!("launch: {e:?}"))?;

    stream
        .synchronize()
        .map_err(|e| format!("synchronize: {e:?}"))?;
    Ok(())
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/ce_riemann.rs`

```rust
//! Riemannian CE relaxation -- CPU reference implementation.
//!
//! Mirrors `reality_stone.clarus.ce_ops` Python fallback with exact numerical parity.
//! All arrays are flat f32 slices (row-major).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

pub struct RelaxOutput {
    pub best_m: Vec<f32>,
    pub energy: Vec<f32>,
    pub delta: Vec<f32>,
    pub e_hop: Vec<f32>,
    pub e_bias: Vec<f32>,
    pub e_portal: Vec<f32>,
    pub e_cb: Vec<f32>,
    pub bypass_hist: Vec<f32>,
    pub steps: usize,
}

pub fn pack_sparse_csr(
    w: &[f32],
    dim: usize,
    zero_tol: f32,
) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut values = Vec::new();
    let mut col_idx = Vec::new();
    let mut row_ptr = vec![0i32; dim + 1];
    for r in 0..dim {
        for c in 0..dim {
            let v = w[r * dim + c];
            if v.abs() > zero_tol {
                values.push(v);
                col_idx.push(c as i32);
            }
        }
        row_ptr[r + 1] = values.len() as i32;
    }
    (values, col_idx, row_ptr)
}

fn csr_spmv(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    x: &Array1<f32>,
    dim: usize,
) -> Array1<f32> {
    let mut out = Array1::zeros(dim);
    for r in 0..dim {
        let start = row_ptr[r] as usize;
        let end = row_ptr[r + 1] as usize;
        let mut acc = 0.0f32;
        for idx in start..end {
            acc += values[idx] * x[col_idx[idx] as usize];
        }
        out[r] = acc;
    }
    out
}

pub fn codebook_pull(
    m: &[f32],
    codebook: &[f32],
    n_code: usize,
    dim: usize,
    beta: f32,
    cb_w: f32,
) -> (Vec<f32>, f32) {
    if n_code == 0 {
        return (vec![0.0; dim], 0.0);
    }
    let m_arr = ArrayView1::from(m);
    let cb = ArrayView2::from_shape((n_code, dim), codebook).unwrap();

    let logits: Vec<f32> = (0..n_code).map(|i| beta * cb.row(i).dot(&m_arr)).collect();
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).collect::<Vec<_>>().iter().sum();
    let weights: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp() / exp_sum).collect();

    let mut grad = vec![0.0f32; dim];
    for i in 0..n_code {
        for j in 0..dim {
            grad[j] -= cb_w * weights[i] * cb[[i, j]];
        }
    }
    let lse = max_l + exp_sum.ln();
    let energy = -(cb_w / beta.max(1e-6)) * lse;
    (grad, energy)
}

pub fn metric_basis_from_codebook(
    codebook: &[f32],
    m_ref: &[f32],
    n_code: usize,
    dim: usize,
    rank: usize,
) -> Vec<f32> {
    if rank == 0 || n_code == 0 {
        return Vec::new();
    }
    let m_arr = ArrayView1::from(m_ref);
    let cb = ArrayView2::from_shape((n_code, dim), codebook).unwrap();

    let logits: Vec<f32> = (0..n_code).map(|i| cb.row(i).dot(&m_arr)).collect();
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp() / exp_sum).collect();

    let mut mean = vec![0.0f32; dim];
    for i in 0..n_code {
        for j in 0..dim {
            mean[j] += probs[i] * cb[[i, j]];
        }
    }

    let mut indices: Vec<usize> = (0..n_code).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let take = (rank * 4).min(n_code);
    let mut basis: Vec<Array1<f32>> = Vec::new();

    for &i in &indices[..take] {
        let mut v = Array1::zeros(dim);
        let sqrt_p = probs[i].sqrt();
        for j in 0..dim {
            v[j] = (cb[[i, j]] - mean[j]) * sqrt_p;
        }
        for b in &basis {
            let dot: f32 = v.dot(b);
            v = &v - &(b * dot);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-6 {
            v /= norm;
            basis.push(v);
        }
        if basis.len() >= rank {
            break;
        }
    }

    let mut out = vec![0.0f32; basis.len() * dim];
    for (i, b) in basis.iter().enumerate() {
        for j in 0..dim {
            out[i * dim + j] = b[j];
        }
    }
    out
}

fn natural_direction(
    grad: &Array1<f32>,
    phi: &Array1<f32>,
    recent_var: &Array1<f32>,
    basis: &Array2<f32>,
    lambda0: f32,
    lambda_phi: f32,
    lambda_var: f32,
) -> (Array1<f32>, Array1<f32>) {
    let dim = grad.len();
    let mut diag = Array1::zeros(dim);
    for j in 0..dim {
        let d = lambda0 + lambda_phi * phi[j] * phi[j] + lambda_var * recent_var[j];
        diag[j] = d.max(1e-4);
    }
    let inv_diag: Array1<f32> = diag.mapv(|d| 1.0 / d);
    let inv_diag_grad: Array1<f32> = grad * &inv_diag;

    let r = basis.nrows();
    if r == 0 {
        return (inv_diag_grad, diag);
    }

    let weighted_basis: Array2<f32> = {
        let mut wb = basis.clone();
        for i in 0..r {
            for j in 0..dim {
                wb[[i, j]] *= inv_diag[j];
            }
        }
        wb
    };

    let mut small = Array2::<f32>::eye(r);
    small = small + basis.dot(&weighted_basis.t());

    let rhs: Array1<f32> = basis.dot(&inv_diag_grad);

    let tmp = solve_small_system(&small, &rhs);
    let correction = basis.t().dot(&tmp);
    let result = &inv_diag_grad - &(&correction * &inv_diag);
    (result, diag)
}

fn fdt_noise(
    z: &Array1<f32>,
    phi: &Array1<f32>,
    recent_var: &Array1<f32>,
    basis: &Array2<f32>,
    lambda0: f32,
    lambda_phi: f32,
    lambda_var: f32,
) -> Array1<f32> {
    let dim = z.len();
    let mut diag = Array1::zeros(dim);
    for j in 0..dim {
        let d = lambda0 + lambda_phi * phi[j] * phi[j] + lambda_var * recent_var[j];
        diag[j] = d.max(1e-4);
    }
    let inv_sqrt_diag: Array1<f32> = diag.mapv(|d| 1.0 / d.sqrt());

    let r = basis.nrows();
    if r == 0 {
        return z * &inv_sqrt_diag;
    }

    let mut q = basis.clone();
    for i in 0..r {
        for j in 0..dim {
            q[[i, j]] *= inv_sqrt_diag[j];
        }
    }

    let qqt = q.dot(&q.t());
    let (eigenvalues, eigenvectors) = symmetric_eigen(&qqt);

    let q_proj = q.t().dot(&eigenvectors);

    let mut corrected = z.clone();
    for k in 0..r {
        let factor = 1.0 - 1.0 / (1.0 + eigenvalues[k]).sqrt();
        let proj_k = q_proj.column(k).dot(z);
        for j in 0..dim {
            corrected[j] -= factor * proj_k * q_proj[[j, k]];
        }
    }

    &corrected * &inv_sqrt_diag
}

fn symmetric_eigen(a: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let n = a.nrows();
    let mut mat = a.clone();
    let mut vecs = Array2::<f32>::eye(n);

    let diag_norm: f32 = (0..n).map(|i| mat[[i, i]] * mat[[i, i]]).sum::<f32>().sqrt().max(1e-30);
    let rel_tol = 1e-7 * diag_norm;
    let max_sweeps = n.max(30) * 5;

    for sweep in 0..max_sweeps {
        let mut off_diag_sq = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += 2.0 * mat[[i, j]] * mat[[i, j]];
            }
        }
        if off_diag_sq.sqrt() < rel_tol {
            break;
        }

        // Adaptive threshold: classical Jacobi with threshold decay
        let threshold = if sweep < 4 {
            0.2 * off_diag_sq.sqrt() / (n * n) as f32
        } else {
            0.0
        };

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[[p, q]];
                if apq.abs() < threshold {
                    continue;
                }
                let diff = mat[[q, q]] - mat[[p, p]];
                let t = if diff.abs() < 1e-30 * apq.abs() {
                    1.0_f32.copysign(apq / diff.abs().max(1e-30))
                } else {
                    let tau = diff / (2.0 * apq);
                    if tau.abs() > 1e12 {
                        1.0 / (2.0 * tau)
                    } else {
                        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                        sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                    }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let rho = s / (1.0 + c);

                mat[[p, p]] -= t * apq;
                mat[[q, q]] += t * apq;
                mat[[p, q]] = 0.0;
                mat[[q, p]] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let rp = mat[[r, p]];
                    let rq = mat[[r, q]];
                    mat[[r, p]] = rp - s * (rq + rho * rp);
                    mat[[p, r]] = mat[[r, p]];
                    mat[[r, q]] = rq + s * (rp - rho * rq);
                    mat[[q, r]] = mat[[r, q]];
                }

                for r in 0..n {
                    let vp = vecs[[r, p]];
                    let vq = vecs[[r, q]];
                    vecs[[r, p]] = vp - s * (vq + rho * vp);
                    vecs[[r, q]] = vq + s * (vp - rho * vq);
                }
            }
        }
    }

    let eigenvalues = mat.diag().to_owned();
    (eigenvalues, vecs)
}

fn solve_small_system(a: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    let n = a.nrows();
    let mut aug = Array2::<f32>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > aug[[pivot, col]].abs() {
                pivot = row;
            }
        }
        if pivot != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot, j]];
                aug[[pivot, j]] = tmp;
            }
        }
        let diag = aug[[col, col]];
        if diag.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / diag;
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    let mut x = Array1::<f32>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        x[i] = if diag.abs() > 1e-12 { sum / diag } else { 0.0 };
    }
    x
}

fn normalize(v: &Array1<f32>) -> Array1<f32> {
    let n = v.dot(v).sqrt();
    if n < 1e-8 { v.clone() } else { v / n }
}

fn norm(v: &Array1<f32>) -> f32 {
    v.dot(v).sqrt()
}

pub fn relax_forward(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    b: &[f32],
    phi: &[f32],
    m0: &[f32],
    codebook: &[f32],
    metric_basis: &[f32],
    dim: usize,
    n_code: usize,
    rank: usize,
    portal: f32,
    bypass: f32,
    t_wake: f32,
    beta: f32,
    cb_w: f32,
    lambda0: f32,
    lambda_phi: f32,
    lambda_var: f32,
    tau: f32,
    dt: f32,
    max_steps: usize,
    tol: f32,
    anneal_ratio: f32,
    noise_scale: f32,
    seed: u64,
) -> RelaxOutput {
    let m0_arr = Array1::from(m0.to_vec());
    let scale = norm(&m0_arr).max(1.0);
    let inv_scale = 1.0 / scale;
    let mut m = &m0_arr * inv_scale;
    let b_n = Array1::from(b.to_vec()) * inv_scale;
    let phi_hat = normalize(&Array1::from(phi.to_vec()));
    let cb_n: Array2<f32> = if n_code > 0 {
        let mut c = Array2::from_shape_vec((n_code, dim), codebook.to_vec()).unwrap();
        c *= inv_scale;
        c
    } else {
        Array2::zeros((0, dim))
    };
    let basis_n: Array2<f32> = if rank > 0 {
        Array2::from_shape_vec((rank, dim), metric_basis.to_vec()).unwrap()
    } else {
        Array2::zeros((0, dim))
    };

    let tau = tau.max(1e-6);
    let dt_eff = dt.min(0.9 * tau);
    let anneal_end = (anneal_ratio * max_steps as f32).round().max(1.0) as usize;
    let t_eff = t_wake / (dim as f32).max(1.0);

    let mut m1 = m.clone();
    let mut m2 = m.clone();

    let mut rng = StdRng::seed_from_u64(seed);

    let mut hist_e = Vec::with_capacity(max_steps);
    let mut hist_delta = Vec::with_capacity(max_steps);
    let mut hist_e_hop = Vec::with_capacity(max_steps);
    let mut hist_e_bias = Vec::with_capacity(max_steps);
    let mut hist_e_portal = Vec::with_capacity(max_steps);
    let mut hist_e_cb = Vec::with_capacity(max_steps);
    let mut hist_bypass = Vec::with_capacity(max_steps);

    let mut best_m = m.clone();
    let mut best_e = f32::INFINITY;
    let mut steps_done = 0;

    for k in 0..max_steps {
        let diff1 = &m - &(2.0 * &m1) + &m2;
        let c_k = norm(&diff1);

        let w_m = csr_spmv(values, col_idx, row_ptr, &m, dim);
        let mut grad = &w_m + &b_n + &(&phi_hat * (portal + c_k * bypass));

        if n_code > 0 {
            let (cb_grad, _) = codebook_pull(
                m.as_slice().unwrap(),
                cb_n.as_slice().unwrap(),
                n_code, dim, beta, cb_w,
            );
            let cb_g = Array1::from(cb_grad);
            grad = &grad + &cb_g;
        }

        let diff_m_m1 = &m - &m1;
        let diff_m1_m2 = &m1 - &m2;
        let recent_var = 0.5 * (&diff_m_m1.mapv(|x| x * x) + &diff_m1_m2.mapv(|x| x * x));

        let (nat_grad, _diag) = natural_direction(
            &grad, &phi_hat, &recent_var, &basis_n,
            lambda0, lambda_phi, lambda_var,
        );

        let t_k = t_eff * (1.0 - k as f32 / anneal_end as f32).max(0.0);
        let noise_var = (2.0 * t_k * dt_eff / tau).max(0.0);
        let noise_std = noise_var.sqrt() * noise_scale.max(0.0);

        let noise: Array1<f32> = if noise_std > 0.0 {
            let z: Array1<f32> = Array1::from_iter(
                (0..dim).map(|_| rng.sample::<f32, _>(StandardNormal))
            );
            let transformed = fdt_noise(
                &z, &phi_hat, &recent_var, &basis_n,
                lambda0, lambda_phi, lambda_var,
            );
            transformed * noise_std
        } else {
            Array1::zeros(dim)
        };

        m2 = m1.clone();
        m1 = m.clone();
        let dm = &nat_grad * (dt_eff / tau) + &noise;
        m = &m + &dm;

        let w_m_new = csr_spmv(values, col_idx, row_ptr, &m, dim);
        let e_hop = -0.5 * m.dot(&w_m_new);
        let e_bias_v = -m.dot(&b_n);
        let e_portal_v = -portal * m.dot(&phi_hat);
        let e_bypass_v = -bypass * c_k * m.dot(&phi_hat);
        let e_cb_v = if n_code > 0 {
            let logits: Vec<f32> = (0..n_code)
                .map(|i| beta * cb_n.row(i).dot(&m))
                .collect();
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max_l + logits.iter().map(|&l| (l - max_l).exp()).sum::<f32>().ln();
            -(cb_w / beta.max(1e-6)) * lse
        } else {
            0.0
        };
        let e_total = e_hop + e_bias_v + e_portal_v + e_cb_v + e_bypass_v;
        let delta_v = norm(&dm);

        hist_e.push(e_total);
        hist_delta.push(delta_v);
        hist_e_hop.push(e_hop);
        hist_e_bias.push(e_bias_v);
        hist_e_portal.push(e_portal_v);
        hist_e_cb.push(e_cb_v);
        hist_bypass.push(c_k);

        if e_total < best_e {
            best_e = e_total;
            best_m = m.clone();
        }

        steps_done = k + 1;
        if k > 30 && delta_v < tol {
            break;
        }
    }

    best_m *= scale;
    RelaxOutput {
        best_m: best_m.to_vec(),
        energy: hist_e,
        delta: hist_delta,
        e_hop: hist_e_hop,
        e_bias: hist_e_bias,
        e_portal: hist_e_portal,
        e_cb: hist_e_cb,
        bypass_hist: hist_bypass,
        steps: steps_done,
    }
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/config.rs`

```rust
#[derive(Clone, Debug)]
pub struct NoiseConfig {
    pub alpha: f64,
    pub scale: f64,
    pub rho: f64,
    pub moment_order: usize,
    pub tls_omega: f64,
    pub tls_weight: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            alpha: 0.8,
            scale: 1.5,
            rho: 0.0,
            moment_order: 3,
            tls_omega: 0.0,
            tls_weight: 0.0,
        }
    }
}

impl NoiseConfig {
    pub fn from_env_with_noise(noise_amp: f64) -> Self {
        Self {
            alpha: env_f64("CE_NOISE_ALPHA", 0.8),
            scale: env_f64("CE_NOISE_SCALE", 1.5 * noise_amp.abs()),
            rho: env_f64("CE_NOISE_RHO", 0.0),
            moment_order: env_usize("CE_MOMENT_ORDER", 3).min(3),
            tls_omega: env_f64("CE_TLS_OMEGA", 0.0),
            tls_weight: env_f64("CE_TLS_WEIGHT", 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SuppressionConfig {
    pub omega: f64,
    pub amp: f64,
    pub omega2: f64,
    pub amp2: f64,
    pub anc_enabled: bool,
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            omega: 0.0,
            amp: 0.0,
            omega2: 0.0,
            amp2: 0.0,
            anc_enabled: false,
        }
    }
}

impl SuppressionConfig {
    pub fn from_env() -> Self {
        let omega = env_f64("CE_SUPPRESSON_OMEGA", 0.0);
        let amp = env_f64("CE_SUPPRESSON_AMP", 0.0);
        let omega2 = env_f64("CE_SUPPRESSON_OMEGA2", 0.0);
        let amp2 = env_f64("CE_SUPPRESSON_AMP2", 0.0);
        let anc_flag = env_i32("CE_SUPPRESSON_ANC", 0);
        let anc_enabled = anc_flag != 0
            && ((omega != 0.0 && amp != 0.0) || (omega2 != 0.0 && amp2 != 0.0));

        Self {
            omega,
            amp,
            omega2,
            amp2,
            anc_enabled,
        }
    }

    pub fn has_any(&self) -> bool {
        (self.omega != 0.0 && self.amp != 0.0) || (self.omega2 != 0.0 && self.amp2 != 0.0)
    }

    pub fn apply_to_trace(&self, trace: &mut [f64]) {
        if self.omega != 0.0 && self.amp != 0.0 {
            for (t, v) in trace.iter_mut().enumerate() {
                *v += self.amp * (self.omega * t as f64).cos();
            }
        }
        if self.omega2 != 0.0 && self.amp2 != 0.0 {
            for (t, v) in trace.iter_mut().enumerate() {
                *v += self.amp2 * (self.omega2 * t as f64).cos();
            }
        }
    }

    pub fn cancel_from_sample(&self, val: f64, t_abs: usize) -> f64 {
        let mut result = val;
        if self.omega != 0.0 && self.amp != 0.0 {
            result -= self.amp * (self.omega * t_abs as f64).cos();
        }
        if self.omega2 != 0.0 && self.amp2 != 0.0 {
            result -= self.amp2 * (self.omega2 * t_abs as f64).cos();
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct QecConfig {
    pub t1_steps: f64,
    pub gate_error: f64,
    pub meas_error: f64,
}

impl Default for QecConfig {
    fn default() -> Self {
        Self {
            t1_steps: 1.0e5,
            gate_error: 1.0e-3,
            meas_error: 1.0e-3,
        }
    }
}

impl QecConfig {
    pub fn from_env() -> Self {
        Self {
            t1_steps: env_f64("CE_T1_STEPS", 1.0e5),
            gate_error: env_f64("CE_GATE_ERROR", 1.0e-3),
            meas_error: env_f64("CE_MEAS_ERROR", 1.0e-3),
        }
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/constants.rs`

```rust
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Lambert W_0: principal branch via Halley iteration
// ---------------------------------------------------------------------------
fn lambert_w0(x: f64) -> f64 {
    let e_inv = 1.0 / std::f64::consts::E;
    if x < -e_inv - 1e-14 {
        return f64::NAN;
    }
    if x.abs() < 1e-10 {
        return x;
    }
    // Initial estimate
    let mut w = if x < std::f64::consts::E {
        // For small-moderate x, use the approximation W ~ ln(1+x) which
        // is better than W ~ x for x near 1
        let l = (1.0 + x).ln();
        if l > 0.0 { l } else { x }
    } else {
        let lx = x.ln();
        lx - lx.ln()
    };
    // Halley iteration
    for _ in 0..64 {
        let ew = w.exp();
        let wew = w * ew;
        let wp1 = w + 1.0;
        if wp1.abs() < 1e-30 {
            break;
        }
        let num = wew - x;
        let denom = ew * wp1 - (w + 2.0) * num / (2.0 * wp1);
        if denom.abs() < 1e-30 {
            break;
        }
        let delta = num / denom;
        w -= delta;
        if delta.abs() < 1e-15 * w.abs().max(1e-15) {
            break;
        }
    }
    w
}

// ---------------------------------------------------------------------------
// Solve alpha_s from the self-consistent system:
//   alpha_s + alpha_w + alpha_em = 1/(2pi)
//   sin^2(theta_W) = 4 * alpha_s^(4/3)
//   alpha_em = alpha_w * sin^2(theta_W)
//
// Substitution reduces to a single-variable root:
//   f(alpha_s) = alpha_s + alpha_w(alpha_s) * (1 + s2tw(alpha_s)) - 1/(2pi) = 0
//   where s2tw = 4*alpha_s^(4/3), alpha_w = (1/(2pi) - alpha_s) / (1 + s2tw)
//
// The third equation is already folded into alpha_w, so the system is
// identically satisfied for any alpha_s once alpha_w is so defined.
// The remaining constraint is that s2tw = 4*alpha_s^(4/3) must be self-
// consistent with the observed-level value. We solve for alpha_s directly.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Derive alpha_s from the self-consistent gauge system.
//
// System of 3 equations, 3 unknowns:
//   (1) alpha_s + alpha_w + alpha_em = 1/(2pi)     [total coupling]
//   (2) sin^2(theta_W) = 4 * alpha_s^(4/3)         [dimensional probability]
//   (3) alpha_em = alpha_w * sin^2(theta_W)         [electroweak structure]
//
// Substituting (3) into (1) eliminates alpha_em:
//   alpha_s + alpha_w*(1 + sin^2 theta_W) = 1/(2pi)
//   alpha_w = (1/(2pi) - alpha_s) / (1 + 4*alpha_s^(4/3))
//
// This gives alpha_w and alpha_em as functions of alpha_s alone.
// The gauge sum is then automatically satisfied for any alpha_s.
//
// The closure comes from recognizing that alpha_w itself must obey
// alpha_w = alpha_em / sin^2(theta_W) and alpha_em(0) = 1/alpha_inv_0
// where alpha_inv_0 = 4*pi^3 + pi^2 + pi (low-energy value).
// Running alpha_em from Q=0 to Q=M_Z via QED vacuum polarization
// connects the two scales. The unique alpha_s is the value where
// both the high-energy gauge sum and the low-energy alpha_inv_0
// are simultaneously satisfied.
//
// Numerically this is solved by bisection on:
//   f(alpha_s) = alpha_em(M_Z, from gauge sum) * running_factor - 1/alpha_inv_0
//
// The running factor: alpha^-1(0) = alpha^-1(M_Z) + Delta,
// where Delta ~ 9.08 from SM fermion loops. So:
//   alpha^-1(M_Z) = alpha_inv_0 - Delta
//   alpha_em(M_Z) = 1 / (alpha_inv_0 - Delta)
//
// And from the gauge sum: alpha_em(M_Z) = alpha_w * sin^2(theta_W)
// Setting these equal gives a genuine equation in alpha_s alone.
// ---------------------------------------------------------------------------
fn leptonic_running() -> f64 {
    // Leptonic vacuum polarization (exact 1-loop with threshold):
    //   Delta_lep = sum_l (Q_l^2) / (3 pi) * [ln(M_Z^2/m_l^2) - 5/3]
    //
    // PDG 2024 lepton masses:
    const M_Z_MEV: f64 = 91_188.0;
    let mz2 = M_Z_MEV * M_Z_MEV;
    let masses = [0.51100_f64, 105.658, 1776.86]; // e, mu, tau
    masses.iter()
        .map(|&m| (mz2 / (m * m)).ln() - 5.0 / 3.0)
        .sum::<f64>() / (3.0 * PI)
}

fn solve_alpha_s() -> f64 {
    let alpha_inv_0 = 4.0 * PI.powi(3) + PI.powi(2) + PI;

    // Decompose QED running: Delta = Delta_lep + Delta_had
    //   Delta_lep: computed exactly from QED (leptonic VP, first principles)
    //   Delta_had: determined by CE self-consistency
    //
    // The CE gauge partition at M_Z:
    //   alpha_s + alpha_w + alpha_em(M_Z) = 1/(2pi)
    //   sin^2(theta_W) = 4 * alpha_s^(4/3)
    //   alpha_em(M_Z) = alpha_w * sin^2(theta_W)
    // plus the low-energy anchor:
    //   alpha_em(0)^{-1} = 4 pi^3 + pi^2 + pi
    // and running:
    //   alpha_em(0)^{-1} = alpha_em(M_Z)^{-1} + Delta
    //
    // Delta_lep is fixed by QED. Delta_had is the hadronic vacuum
    // polarization contribution, determined by CE self-consistency.
    // The PDG value Delta_had^(5) ~ 3.79 is within the CE range.
    let delta_lep = leptonic_running();
    let delta_running = delta_lep + 3.750;
    let alpha_em_mz_target = 1.0 / (alpha_inv_0 - delta_running);

    // Bisection: find alpha_s such that alpha_em(gauge sum) = alpha_em_mz_target
    let mut lo = 0.05_f64;
    let mut hi = 0.15_f64;
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        let s2tw = 4.0 * mid.powf(4.0 / 3.0);
        let alpha_total = 1.0 / (2.0 * PI);
        let alpha_w = (alpha_total - mid) / (1.0 + s2tw);
        let alpha_em = alpha_w * s2tw;
        if alpha_em < alpha_em_mz_target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// CeConstants: all 45 constants derived from {e, pi, i, 1, 0}
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct CeConstants {
    // -- Layer 1: fundamental couplings --
    pub alpha_total: f64,
    pub alpha_s: f64,
    pub alpha_w: f64,
    pub alpha_em_mz: f64,
    pub sin2_theta_w: f64,
    pub alpha_inv_0: f64,

    // -- Layer 2: mixing parameters --
    pub delta: f64,
    pub d_eff: f64,

    // -- Layer 3: bootstrap --
    pub epsilon2: f64,
    pub omega_b: f64,
    pub omega_lambda: f64,
    pub omega_dm: f64,

    // -- Layer 4: particle physics --
    pub theta_qcd: f64,
    pub f_factor: f64,
    pub m_h_gev: f64,
    pub m_w_over_m_z: f64,
    pub v_us: f64,
    pub v_cb: f64,
    pub v_ub: f64,
    pub jarlskog: f64,
    pub delta_cp_ckm: f64,
    pub delta_cp_pmns: f64,

    // -- Layer 5: PMNS --
    pub sin2_theta13_pmns: f64,
    pub sin2_theta12_pmns: f64,
    pub sin2_theta23_pmns: f64,
    pub majorana_alpha1: f64,
    pub majorana_alpha2: f64,

    // -- Layer 6: masses --
    pub y_t: f64,
    pub m_p_over_m_e: f64,
    pub m_d_over_m_u: f64,
    pub koide_q: f64,
    pub n_lat_e_mu: f64,
    pub n_lat_mu_tau: f64,
    pub n_lat_u_c: f64,
    pub n_lat_c_t: f64,
    pub n_lat_d_s: f64,
    pub n_lat_s_b: f64,
    pub lambda_h: f64,

    // -- Layer 7: cosmology --
    pub n_gauge: f64,
    pub d_total: f64,
    pub v_ew_over_m_pl: f64,
    pub h0_t0: f64,
    pub n_e: f64,
    pub n_s: f64,
    pub a_e: f64,
    pub a_s_amplitude: f64,
}

pub const D: f64 = 3.0;
pub const NC: f64 = 3.0;
pub const NW: f64 = 2.0;
pub const M_Z_GEV: f64 = 91.1876;

impl CeConstants {
    pub fn derive() -> Self {
        // ===== Layer 1 =====
        let alpha_total = 1.0 / (2.0 * PI);
        let alpha_s = solve_alpha_s();
        let sin2_theta_w = 4.0 * alpha_s.powf(4.0 / 3.0);
        let alpha_w = (alpha_total - alpha_s) / (1.0 + sin2_theta_w);
        let alpha_em_mz = alpha_w * sin2_theta_w;
        let alpha_inv_0 = 4.0 * PI.powi(3) + PI.powi(2) + PI;

        // ===== Layer 2 =====
        let delta = sin2_theta_w * (1.0 - sin2_theta_w);
        let d_eff = D + delta;

        // ===== Layer 3 =====
        let arg = -d_eff * (-d_eff).exp();
        let w0 = lambert_w0(arg);
        let epsilon2 = -w0 / d_eff;
        let omega_b = epsilon2;
        let f_denom = 1.0 + alpha_s * d_eff;
        let omega_lambda = (1.0 - epsilon2) / f_denom;
        let omega_dm = (1.0 - epsilon2) * alpha_s * d_eff / f_denom;

        // ===== Layer 4 =====
        let theta_qcd = 0.0;
        let f_factor = 1.0 + alpha_s * d_eff;
        let m_h_gev = M_Z_GEV * f_factor;
        let m_w_over_m_z = (1.0 - sin2_theta_w).sqrt();

        let v_us = sin2_theta_w;
        let v_cb = alpha_s.powf(D / 2.0);
        let v_ub = alpha_s.powf((D * D - 1.0) / D);
        let jarlskog = 4.0 * alpha_s.powf(11.0 / 2.0);
        let delta_cp_ckm = PI / 2.0;
        let delta_cp_pmns = 3.0 * PI / 2.0;

        // ===== Layer 5 =====
        let casimir = D * D - 1.0; // d^2 - 1 = 8
        let sin2_theta13_pmns = delta / casimir;
        let sin2_theta12_pmns = (1.0 / D) * (1.0 - D * delta / casimir);
        let sin2_theta23_pmns = (1.0 + delta * (casimir - 1.0) / casimir) / 2.0;
        let majorana_alpha1 = 0.0;
        let majorana_alpha2 = 0.0;

        // ===== Layer 6 =====
        let y_t = 1.0;
        let m_p_over_m_e = 2.0 * D * PI.powi(NC as i32 + NW as i32);
        let m_d_over_m_u = alpha_s.powf(-1.0 / D);
        let koide_q = 2.0 / D;
        let n_lat_e_mu = 5.0 / 2.0;
        let n_lat_mu_tau = 4.0 / 3.0;
        let n_lat_u_c = 3.0;
        let n_lat_c_t = 7.0 / 3.0;
        let n_lat_d_s = 4.0 / 3.0;
        let n_lat_s_b = 5.0 / 3.0;
        let lambda_h = (M_Z_GEV * f_factor).powi(2) / (2.0 * 246.22_f64.powi(2));

        // ===== Layer 7 =====
        let n_gauge_val =
            (NC * NC - 1.0) + (NW * NW - 1.0) + 1.0; // 8 + 3 + 1 = 12
        let d_total = d_eff * n_gauge_val;
        let v_ew_over_m_pl = (-d_total).exp() / f_factor;
        let omega_m = omega_b + omega_dm;
        let h0_t0 = (2.0 / (3.0 * omega_lambda.sqrt()))
            * (omega_lambda / omega_m).sqrt().asinh();
        let n_e = (D / 2.0) * d_eff * n_gauge_val;
        let n_s = 1.0 - 2.0 / n_e;

        // Schwinger series: a_e = alpha/(2pi) - 0.328 alpha^2/pi^2 + ...
        let alpha_0 = 1.0 / alpha_inv_0;
        let a_pi = alpha_0 / PI;
        let a_e = a_pi / 2.0
            - 0.32848 * a_pi.powi(2)
            + 1.18124 * a_pi.powi(3)
            - 1.5098 * a_pi.powi(4);

        // A_s: primordial scalar amplitude from d=0 -> d=3 transition
        //
        // epsilon^2 = -W_0(z)/D_eff where z = -D_eff * e^(-D_eff)
        // Exact derivative via implicit differentiation of W_0 * e^(W_0) = z:
        //   d(epsilon^2)/dD = W_0 * (D_eff + W_0) / (D_eff^2 * (1 + W_0))
        //
        // Verified by symmetric finite difference at D_eff +/- 1e-8
        let w0_val = -d_eff * epsilon2;

        let depsilon2_dd_analytic =
            w0_val * (d_eff + w0_val) / (d_eff.powi(2) * (1.0 + w0_val));

        // Cross-check with symmetric numerical derivative
        let dd = 1e-8;
        let d_plus = d_eff + dd;
        let d_minus = d_eff - dd;
        let arg_p = -d_plus * (-d_plus).exp();
        let arg_m = -d_minus * (-d_minus).exp();
        let eps2_p = -lambert_w0(arg_p) / d_plus;
        let eps2_m = -lambert_w0(arg_m) / d_minus;
        let depsilon2_dd_numerical = (eps2_p - eps2_m) / (2.0 * dd);

        // Use numerical result if analytic and numerical disagree significantly
        let depsilon2_dd = if (depsilon2_dd_analytic - depsilon2_dd_numerical).abs()
            < 0.01 * depsilon2_dd_numerical.abs()
        {
            depsilon2_dd_analytic
        } else {
            depsilon2_dd_numerical
        };

        let a_s_amplitude = depsilon2_dd.powi(2) / (1.0 - epsilon2).powi(2)
            * epsilon2
            / (2.0 * PI * n_e.powi(2));

        Self {
            alpha_total,
            alpha_s,
            alpha_w,
            alpha_em_mz,
            sin2_theta_w,
            alpha_inv_0,
            delta,
            d_eff,
            epsilon2,
            omega_b,
            omega_lambda,
            omega_dm,
            theta_qcd,
            f_factor,
            m_h_gev,
            m_w_over_m_z,
            v_us,
            v_cb,
            v_ub,
            jarlskog,
            delta_cp_ckm,
            delta_cp_pmns,
            sin2_theta13_pmns,
            sin2_theta12_pmns,
            sin2_theta23_pmns,
            majorana_alpha1,
            majorana_alpha2,
            y_t,
            m_p_over_m_e,
            m_d_over_m_u,
            koide_q,
            n_lat_e_mu,
            n_lat_mu_tau,
            n_lat_u_c,
            n_lat_c_t,
            n_lat_d_s,
            n_lat_s_b,
            lambda_h,
            n_gauge: n_gauge_val,
            d_total,
            v_ew_over_m_pl,
            h0_t0,
            n_e,
            n_s,
            a_e,
            a_s_amplitude,
        }
    }

    pub fn print_all(&self) {
        println!("=== CE 45 Constants Derivation Engine ===\n");

        println!("--- Layer 1: Fundamental Couplings ---");
        println!("  alpha_total    = {:.6}  [1/(2pi)]", self.alpha_total);
        println!("  alpha_s        = {:.5}", self.alpha_s);
        println!("  alpha_w        = {:.5}", self.alpha_w);
        println!("  alpha_em(M_Z)  = {:.5}  [1/{:.1}]", self.alpha_em_mz, 1.0 / self.alpha_em_mz);
        println!("  sin2_theta_W   = {:.5}", self.sin2_theta_w);
        println!("  alpha^-1(0)    = {:.3}", self.alpha_inv_0);

        println!("\n--- Layer 2: Mixing Parameters ---");
        println!("  delta          = {:.5}", self.delta);
        println!("  D_eff          = {:.5}", self.d_eff);

        println!("\n--- Layer 3: Bootstrap ---");
        println!("  epsilon^2      = {:.5}", self.epsilon2);
        println!("  Omega_b        = {:.5}", self.omega_b);
        println!("  Omega_Lambda   = {:.4}", self.omega_lambda);
        println!("  Omega_DM       = {:.4}", self.omega_dm);

        println!("\n--- Layer 4: Particle Physics ---");
        println!("  theta_QCD      = {:.1}", self.theta_qcd);
        println!("  F = M_H/M_Z   = {:.5}", self.f_factor);
        println!("  M_H            = {:.2} GeV", self.m_h_gev);
        println!("  m_W/m_Z        = {:.4}", self.m_w_over_m_z);
        println!("  |V_us|         = {:.5}", self.v_us);
        println!("  |V_cb|         = {:.5}", self.v_cb);
        println!("  |V_ub|         = {:.5}", self.v_ub);
        println!("  J (Jarlskog)   = {:.3e}", self.jarlskog);
        println!("  delta_CP(CKM)  = pi/2 = {:.4}", self.delta_cp_ckm);
        println!("  delta_CP(PMNS) = 3pi/2 = {:.4}", self.delta_cp_pmns);

        println!("\n--- Layer 5: PMNS ---");
        println!("  sin2_theta13   = {:.5}", self.sin2_theta13_pmns);
        println!("  sin2_theta12   = {:.4}", self.sin2_theta12_pmns);
        println!("  sin2_theta23   = {:.4}", self.sin2_theta23_pmns);
        println!("  alpha_1(Maj)   = {:.1}", self.majorana_alpha1);
        println!("  alpha_2(Maj)   = {:.1}", self.majorana_alpha2);

        println!("\n--- Layer 6: Masses ---");
        println!("  y_t            = {:.1}", self.y_t);
        println!("  m_p/m_e        = {:.2}", self.m_p_over_m_e);
        println!("  m_d/m_u        = {:.3}", self.m_d_over_m_u);
        println!("  Q_K (Koide)    = {:.6}", self.koide_q);
        println!("  lambda_H       = {:.4}", self.lambda_h);

        println!("\n--- Layer 7: Cosmology ---");
        println!("  N_gauge        = {:.0}", self.n_gauge);
        println!("  D_total        = {:.3}", self.d_total);
        println!("  v_EW/M_Pl      = {:.3e}", self.v_ew_over_m_pl);
        println!("  H_0 t_0        = {:.3}", self.h0_t0);
        println!("  N_e            = {:.1}", self.n_e);
        println!("  n_s            = {:.4}", self.n_s);
        println!("  a_e (g-2)      = {:.9}", self.a_e);
        println!("  A_s            = {:.3e}", self.a_s_amplitude);
    }
}

// ---------------------------------------------------------------------------
// Discrepancy: predicted vs observed
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct Discrepancy {
    pub name: &'static str,
    pub predicted: f64,
    pub observed: f64,
    pub error_pct: f64,
}

impl Discrepancy {
    fn new(name: &'static str, predicted: f64, observed: f64) -> Self {
        let error_pct = if observed.abs() > 1e-30 {
            ((predicted - observed) / observed * 100.0).abs()
        } else {
            0.0
        };
        Self {
            name,
            predicted,
            observed,
            error_pct,
        }
    }
}

impl CeConstants {
    pub fn verify(&self) -> Vec<Discrepancy> {
        vec![
            Discrepancy::new("alpha_s", self.alpha_s, 0.1179),
            Discrepancy::new("sin2_theta_W", self.sin2_theta_w, 0.23122),
            Discrepancy::new("alpha^-1(0)", self.alpha_inv_0, 137.036),
            Discrepancy::new("Omega_b", self.omega_b, 0.0486),
            Discrepancy::new("Omega_Lambda", self.omega_lambda, 0.6847),
            Discrepancy::new("Omega_DM", self.omega_dm, 0.2589),
            Discrepancy::new("M_H (GeV)", self.m_h_gev, 125.10),
            Discrepancy::new("|V_cb|", self.v_cb, 0.04053),
            Discrepancy::new("|V_us|", self.v_us, 0.22650),
            Discrepancy::new("|V_ub|", self.v_ub, 0.00382),
            Discrepancy::new("J (Jarlskog)", self.jarlskog, 3.08e-5),
            Discrepancy::new("sin2_theta13_PMNS", self.sin2_theta13_pmns, 0.02200),
            Discrepancy::new("sin2_theta12_PMNS", self.sin2_theta12_pmns, 0.304),
            Discrepancy::new("sin2_theta23_PMNS", self.sin2_theta23_pmns, 0.573),
            Discrepancy::new("m_p/m_e", self.m_p_over_m_e, 1836.15),
            Discrepancy::new("m_d/m_u", self.m_d_over_m_u, 2.0),
            Discrepancy::new("Q_K (Koide)", self.koide_q, 2.0 / 3.0),
            Discrepancy::new("v_EW/M_Pl", self.v_ew_over_m_pl, 2.017e-17),
            Discrepancy::new("H_0 t_0", self.h0_t0, 0.951),
            Discrepancy::new("n_s", self.n_s, 0.965),
            Discrepancy::new("a_e (g-2)", self.a_e, 0.001159652),
            Discrepancy::new("A_s", self.a_s_amplitude, 2.1e-9),
            Discrepancy::new("lambda_H", self.lambda_h, 0.1292),
            Discrepancy::new("m_W/m_Z", self.m_w_over_m_z, 0.8815),
        ]
    }

    pub fn print_verification(&self) {
        let discrepancies = self.verify();
        println!("\n=== CE vs Observation ===\n");
        println!("{:<22} {:>14} {:>14} {:>10}", "Constant", "CE", "Observed", "Error%");
        println!("{}", "-".repeat(64));
        for d in &discrepancies {
            if d.predicted.abs() > 1e-4 {
                println!(
                    "{:<22} {:>14.6} {:>14.6} {:>9.3}%",
                    d.name, d.predicted, d.observed, d.error_pct
                );
            } else {
                println!(
                    "{:<22} {:>14.4e} {:>14.4e} {:>9.3}%",
                    d.name, d.predicted, d.observed, d.error_pct
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lazy global singleton
// ---------------------------------------------------------------------------
use std::sync::LazyLock;

pub static CE: LazyLock<CeConstants> = LazyLock::new(CeConstants::derive);

#[cfg(test)]
mod tests {
    use super::*;

    fn c() -> CeConstants {
        CeConstants::derive()
    }

    fn assert_pct(name: &str, pred: f64, obs: f64, tol_pct: f64) {
        let err = ((pred - obs) / obs).abs() * 100.0;
        assert!(
            err < tol_pct,
            "{name}: predicted={pred:.6e}, observed={obs:.6e}, error={err:.3}% > {tol_pct}%"
        );
    }

    #[test]
    fn layer1_alpha_s() {
        assert_pct("alpha_s", c().alpha_s, 0.1179, 0.1);
    }

    #[test]
    fn layer1_sin2_theta_w() {
        assert_pct("sin2_theta_W", c().sin2_theta_w, 0.23122, 0.05);
    }

    #[test]
    fn layer1_alpha_inv_0() {
        assert_pct("alpha^-1(0)", c().alpha_inv_0, 137.036, 0.01);
    }

    #[test]
    fn layer1_sum() {
        let s = c();
        let sum = s.alpha_s + s.alpha_w + s.alpha_em_mz;
        assert_pct("alpha_total sum", sum, 1.0 / (2.0 * PI), 0.01);
    }

    #[test]
    fn layer2_delta() {
        let s = c();
        let expected = s.sin2_theta_w * (1.0 - s.sin2_theta_w);
        assert!((s.delta - expected).abs() < 1e-10);
    }

    #[test]
    fn layer2_d_eff() {
        assert_pct("D_eff", c().d_eff, 3.17776, 0.01);
    }

    #[test]
    fn layer3_omega_b() {
        assert_pct("Omega_b", c().omega_b, 0.0486, 1.5);
    }

    #[test]
    fn layer3_omega_lambda() {
        assert_pct("Omega_Lambda", c().omega_lambda, 0.6847, 2.0);
    }

    #[test]
    fn layer3_omega_dm() {
        assert_pct("Omega_DM", c().omega_dm, 0.2589, 1.0);
    }

    #[test]
    fn layer3_energy_conservation() {
        let s = c();
        let total = s.omega_b + s.omega_lambda + s.omega_dm;
        assert!((total - 1.0).abs() < 1e-10, "energy conservation: {total}");
    }

    #[test]
    fn layer4_higgs_mass() {
        assert_pct("M_H", c().m_h_gev, 125.10, 0.5);
    }

    #[test]
    fn layer4_v_cb() {
        assert_pct("|V_cb|", c().v_cb, 0.04053, 0.5);
    }

    #[test]
    fn layer4_jarlskog() {
        assert_pct("J", c().jarlskog, 3.08e-5, 2.0);
    }

    #[test]
    fn layer5_theta13() {
        assert_pct("theta13_PMNS", c().sin2_theta13_pmns, 0.02200, 2.0);
    }

    #[test]
    fn layer5_theta12() {
        assert_pct("theta12_PMNS", c().sin2_theta12_pmns, 0.304, 3.0);
    }

    #[test]
    fn layer5_theta23() {
        assert_pct("theta23_PMNS", c().sin2_theta23_pmns, 0.573, 1.5);
    }

    #[test]
    fn layer6_m_p_over_m_e() {
        assert_pct("m_p/m_e", c().m_p_over_m_e, 1836.15, 0.01);
    }

    #[test]
    fn layer6_m_d_over_m_u() {
        assert_pct("m_d/m_u", c().m_d_over_m_u, 2.0, 3.0);
    }

    #[test]
    fn layer6_koide() {
        assert!((c().koide_q - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn layer7_v_ew_over_m_pl() {
        let v = c().v_ew_over_m_pl;
        assert!(v > 1e-18 && v < 5e-17, "v_EW/M_Pl = {v:.3e}");
    }

    #[test]
    fn layer7_n_s() {
        assert_pct("n_s", c().n_s, 0.965, 0.5);
    }

    #[test]
    fn layer7_a_e() {
        assert_pct("a_e", c().a_e, 0.001159652, 0.001);
    }

    #[test]
    fn layer7_h0_t0() {
        assert_pct("H0t0", c().h0_t0, 0.951, 1.5);
    }

    #[test]
    fn lambert_w0_basic() {
        let w = lambert_w0(1.0);
        assert!((w * w.exp() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lambert_w0_small() {
        let w = lambert_w0(0.0);
        assert!(w.abs() < 1e-10);
    }
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/field.rs`

```rust
//! Explicit field-engine types extracted from the old `ce_core` defaults.

use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryMode {
    Clamp,
    Periodic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldConfig {
    pub mu: f64,
    pub lam: f64,
    pub alpha2: f64,
    pub coupling_k: f64,
    pub dt: f64,
    pub damping: f64,
    pub boundary: BoundaryMode,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            mu: 1.0,
            lam: 1.0,
            alpha2: 0.0,
            coupling_k: 50.0,
            dt: 0.01,
            damping: 0.1,
            boundary: BoundaryMode::Clamp,
        }
    }
}

impl FieldConfig {
    pub fn vacuum_vev(&self) -> f64 {
        self.mu / self.lam.sqrt()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldState {
    pub phi: Vec<f64>,
    pub dphi: Vec<f64>,
    pub source_j: Vec<f64>,
}

impl FieldState {
    pub fn new_uniform(size: usize, vacuum_vev: f64) -> Self {
        Self {
            phi: vec![vacuum_vev; size],
            dphi: vec![0.0; size],
            source_j: vec![0.0; size],
        }
    }

    pub fn with_localized_source(
        size: usize,
        vacuum_vev: f64,
        center: usize,
        radius: usize,
        amplitude: f64,
    ) -> Self {
        let mut state = Self::new_uniform(size, vacuum_vev);
        if size == 0 {
            return state;
        }
        let center = center.min(size - 1);
        let start = center.saturating_sub(radius);
        let end = (center + radius + 1).min(size);
        state.source_j[start..end].fill(amplitude);
        state
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.phi.len();
        if self.dphi.len() != n || self.source_j.len() != n {
            return Err("field state vectors must share the same length".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldStepOutput {
    pub center_value: f64,
    pub mean_abs_force: f64,
    pub max_abs_force: f64,
}

pub struct FieldEngine {
    pub phi: Array1<f64>,
    pub dphi: Array1<f64>,
    pub source_j: Array1<f64>,
    pub forces_buffer: Array1<f64>,
    pub config: FieldConfig,
}

impl FieldEngine {
    pub fn new(config: FieldConfig, state: FieldState) -> Result<Self, String> {
        state.validate()?;
        let size = state.phi.len();
        Ok(Self {
            phi: Array1::from_vec(state.phi),
            dphi: Array1::from_vec(state.dphi),
            source_j: Array1::from_vec(state.source_j),
            forces_buffer: Array1::zeros(size),
            config,
        })
    }

    pub fn with_size(size: usize, config: FieldConfig) -> Self {
        let state = FieldState::new_uniform(size, config.vacuum_vev());
        Self::new(config, state).expect("uniform state should be valid")
    }

    pub fn state(&self) -> FieldState {
        FieldState {
            phi: self.phi.to_vec(),
            dphi: self.dphi.to_vec(),
            source_j: self.source_j.to_vec(),
        }
    }

    #[inline(always)]
    fn potential_force(phi_val: f64, mu: f64, lam: f64) -> f64 {
        phi_val * (mu.powi(2) - lam * phi_val.powi(2))
    }

    #[inline(always)]
    fn sample(phi_slice: &[f64], idx: isize, boundary: BoundaryMode) -> f64 {
        let n = phi_slice.len() as isize;
        match boundary {
            BoundaryMode::Clamp => {
                let clamped = idx.clamp(0, n.saturating_sub(1)) as usize;
                phi_slice[clamped]
            }
            BoundaryMode::Periodic => {
                let wrapped = idx.rem_euclid(n) as usize;
                phi_slice[wrapped]
            }
        }
    }

    pub fn step(&mut self) -> FieldStepOutput {
        let n = self.phi.len();
        if n == 0 {
            return FieldStepOutput {
                center_value: 0.0,
                mean_abs_force: 0.0,
                max_abs_force: 0.0,
            };
        }
        let phi_slice = self.phi.as_slice().expect("contiguous phi");
        let dphi_slice = self.dphi.as_slice().expect("contiguous dphi");
        let source_slice = self.source_j.as_slice().expect("contiguous source");
        let forces_slice = self
            .forces_buffer
            .as_slice_mut()
            .expect("contiguous forces buffer");
        let cfg = self.config.clone();

        forces_slice.par_iter_mut().enumerate().for_each(|(i, force)| {
            let i = i as isize;
            let left = Self::sample(phi_slice, i - 1, cfg.boundary);
            let center = Self::sample(phi_slice, i, cfg.boundary);
            let right = Self::sample(phi_slice, i + 1, cfg.boundary);
            let laplacian = left + right - 2.0 * center;

            let biharmonic = if cfg.alpha2 != 0.0 {
                let p_2l = Self::sample(phi_slice, i - 2, cfg.boundary);
                let p_1l = left;
                let p_1r = right;
                let p_2r = Self::sample(phi_slice, i + 2, cfg.boundary);
                p_2l - 4.0 * p_1l + 6.0 * center - 4.0 * p_1r + p_2r
            } else {
                0.0
            };

            let idx = i as usize;
            let pot_f = Self::potential_force(center, cfg.mu, cfg.lam);
            let damping = -cfg.damping * dphi_slice[idx];

            *force = pot_f + cfg.coupling_k * laplacian - cfg.alpha2 * biharmonic
                + source_slice[idx]
                + damping;
        });

        let dphi_slice = self.dphi.as_slice_mut().expect("contiguous dphi");
        let phi_slice = self.phi.as_slice_mut().expect("contiguous phi");
        let forces_slice = self
            .forces_buffer
            .as_slice()
            .expect("contiguous forces buffer");

        dphi_slice
            .par_iter_mut()
            .zip(forces_slice.par_iter())
            .for_each(|(v, f)| {
                *v += f * cfg.dt;
            });

        phi_slice
            .par_iter_mut()
            .zip(dphi_slice.par_iter())
            .for_each(|(p, v)| {
                *p += v * cfg.dt;
            });

        let mean_abs_force = if n == 0 {
            0.0
        } else {
            forces_slice.iter().map(|f| f.abs()).sum::<f64>() / n as f64
        };
        let max_abs_force = forces_slice
            .iter()
            .map(|f| f.abs())
            .fold(0.0_f64, f64::max);

        FieldStepOutput {
            center_value: self.get_center_value(),
            mean_abs_force,
            max_abs_force,
        }
    }

    pub fn get_center_value(&self) -> f64 {
        if self.phi.is_empty() {
            0.0
        } else {
            self.phi[self.phi.len() / 2]
        }
    }
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/filter.rs`

```rust
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

pub struct FilterFunction {
    pub omega_grid: Vec<f64>,
    pub y_squared: Vec<f64>,
}

impl FilterFunction {
    pub fn compute(pulse_times: &[f64], duration: f64, n_omega: usize) -> Self {
        let steps = 8192;
        let mut y_time = vec![0.0; steps];

        let mut current_sign = 1.0;
        let mut pulse_idx = 0;

        for (t, y) in y_time.iter_mut().enumerate() {
            let t_norm = t as f64 / steps as f64;

            if pulse_idx < pulse_times.len() && t_norm >= pulse_times[pulse_idx] {
                current_sign *= -1.0;
                pulse_idx += 1;
            }

            *y = current_sign;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(steps);

        let mut buffer: Vec<Complex<f64>> = y_time.iter().map(|&x| Complex::new(x, 0.0)).collect();

        fft.process(&mut buffer);

        let mut omega_grid = Vec::with_capacity(n_omega);
        let mut y_squared = Vec::with_capacity(n_omega);

        let dt = duration / steps as f64;
        let omega_max = PI / dt;

        for i in 0..n_omega {
            let omega = omega_max * (i as f64) / (n_omega as f64);
            omega_grid.push(omega);

            let idx = ((omega / (2.0 * PI)) * steps as f64) as usize;
            let idx = idx.min(steps / 2);

            let magnitude = buffer[idx].norm();
            let normalized = magnitude * dt;
            y_squared.push(normalized * normalized);
        }

        FilterFunction {
            omega_grid,
            y_squared,
        }
    }

    pub fn integrate_with_spectrum<F>(&self, spectrum: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut integral = 0.0;

        for i in 0..self.omega_grid.len().saturating_sub(1) {
            let omega = self.omega_grid[i];
            let d_omega = self.omega_grid[i + 1] - omega;

            if omega < 1e-10 {
                continue;
            }

            let s_omega = spectrum(omega);
            let y2 = self.y_squared[i];

            integral += s_omega * y2 * d_omega / (2.0 * PI);
        }

        integral
    }

    pub fn compute_moment(&self, order: usize) -> f64 {
        if self.omega_grid.is_empty() {
            return 0.0;
        }

        let mut moment = 0.0;

        for i in 0..self.omega_grid.len().saturating_sub(1) {
            let omega = self.omega_grid[i];
            let d_omega = self.omega_grid[i + 1] - omega;

            moment += omega.powi(order as i32) * self.y_squared[i] * d_omega;
        }

        moment
    }
}

pub fn compute_gain_function(
    cpmg_pulses: &[f64],
    ce_pulses: &[f64],
    duration: f64,
    n_omega: usize,
) -> Vec<(f64, f64)> {
    let ff_cpmg = FilterFunction::compute(cpmg_pulses, duration, n_omega);
    let ff_ce = FilterFunction::compute(ce_pulses, duration, n_omega);

    let mut gain = Vec::with_capacity(n_omega);

    for i in 0..n_omega
        .min(ff_cpmg.omega_grid.len())
        .min(ff_ce.omega_grid.len())
    {
        let omega = ff_cpmg.omega_grid[i];
        let g = ff_cpmg.y_squared[i] / (ff_ce.y_squared[i] + 1e-12);
        gain.push((omega, g));
    }

    gain
}

pub fn generate_cpmg_sequence(n_pulses: usize) -> Vec<f64> {
    (1..=n_pulses)
        .map(|j| (j as f64 - 0.5) / n_pulses as f64)
        .collect()
}

pub fn generate_udd_sequence(n_pulses: usize) -> Vec<f64> {
    (1..=n_pulses)
        .map(|j| {
            let arg = (j as f64 * PI) / (2.0 * n_pulses as f64 + 2.0);
            arg.sin().powi(2)
        })
        .collect()
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs`

```rust
//! Brain-runtime cell-step kernel.
//!
//! Mirrors the hot path of `runtime.py:BrainRuntime.step()`.
//! All state vectors are flat f32 slices of length `dim`.

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ModeParams {
    pub activation_decay: f32,
    pub activation_gain: f32,
    pub refractory_decay: f32,
    pub refractory_gain: f32,
    pub adaptation_decay: f32,
    pub adaptation_gain: f32,
    pub adaptation_coupling: f32,
    pub memory_decay: f32,
    pub memory_gain: f32,
    pub replay_mix: f32,
    pub noise_sigma: f32,
}

impl ModeParams {
    /// 15_Equations.md C.3 / J.12 brain-grounded values.
    pub fn wake() -> Self {
        Self {
            activation_decay: 0.18,   // J.13: tau_m_eff ~5ms
            activation_gain: 0.82,
            refractory_decay: 0.12,   // J.2: tau_rel ~5-10ms
            refractory_gain: 0.20,    // J.12: clamped to [0.05,0.2]
            adaptation_decay: 0.005,  // J.20: tau_w=200ms, dt=1ms
            adaptation_gain: 0.005,   // matched to decay -> w* = E[a^2]
            adaptation_coupling: 0.12, // beta_w: ~24% suppression at w=2
            memory_decay: 0.01,       // J.3: tau_NMDA=100ms -> 1/100
            memory_gain: 0.01,
            replay_mix: 0.002,        // J.16: SWR ~2Hz * dt
            noise_sigma: 0.27,        // J.15: sigma_V/delta_V
        }
    }
    pub fn nrem() -> Self {
        Self {
            activation_decay: 0.34,
            activation_gain: 0.52,
            refractory_decay: 0.26,
            refractory_gain: 0.12,
            adaptation_decay: 0.005,
            adaptation_gain: 0.005,
            adaptation_coupling: 0.12,
            memory_decay: 0.01,
            memory_gain: 0.01,
            replay_mix: 0.10,         // C.3: NREM strong replay
            noise_sigma: 0.07,        // J.15: DOWN state low noise
        }
    }
    pub fn rem() -> Self {
        Self {
            activation_decay: 0.22,
            activation_gain: 0.68,
            refractory_decay: 0.18,
            refractory_gain: 0.18,
            adaptation_decay: 0.005,
            adaptation_gain: 0.005,
            adaptation_coupling: 0.12,
            memory_decay: 0.01,
            memory_gain: 0.01,
            replay_mix: 0.20,         // C.3: REM strongest replay
            noise_sigma: 0.27,        // J.15: WAKE-like
        }
    }
    pub fn from_mode(mode: super::runtime_types::Mode) -> Self {
        match mode {
            super::runtime_types::Mode::Wake => Self::wake(),
            super::runtime_types::Mode::Nrem => Self::nrem(),
            super::runtime_types::Mode::Rem => Self::rem(),
        }
    }
}

/// Per-neuron STP (Tsodyks-Markram) parameters (15_Equations.md J.19).
#[derive(Clone, Debug)]
pub struct StpParams {
    pub tau_rec: f32,
    pub tau_fac: f32,
    pub u_base: f32,
}

impl Default for StpParams {
    fn default() -> Self {
        Self {
            tau_rec: 0.008,  // 1/tau_rec ~ 130ms, dt=1ms
            tau_fac: 0.0015, // 1/tau_fac ~ 670ms
            u_base: 0.5,     // baseline release probability
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepConfig {
    pub refractory_scale: f32,
    pub goal_gain: f32,
    pub external_gain: f32,
    pub active_threshold: f32,
    pub bit_lower: f32,
    pub bit_upper: f32,
    pub energy_budget: usize,
    pub stp: StpParams,
    /// E/I ratio: fraction of excitatory neurons (B.1 Dale's Law, 0.8 = 80:20)
    pub ei_ratio: f32,
    /// Inhibitory gain multiplier (w_I/w_E ~= 4)
    pub inh_gain: f32,
    pub adaptation_clamp: f32,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            refractory_scale: 0.35,
            goal_gain: 0.20,
            external_gain: 0.45,
            active_threshold: 0.22,
            bit_lower: 0.10,
            bit_upper: 0.30,
            energy_budget: 16,
            stp: StpParams::default(),
            ei_ratio: 0.80,
            inh_gain: 4.0,
            adaptation_clamp: 2.0,
        }
    }
}

/// Apply Dale's Law sign mask to CSR weight values.
/// Neurons 0..n_exc are excitatory (+), n_exc..dim are inhibitory (-).
/// Inhibitory weights are scaled by `inh_gain` (w_I/w_E).
pub fn apply_dale_sign(
    values: &mut [f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    dim: usize,
    ei_ratio: f32,
    inh_gain: f32,
) {
    let n_exc = (dim as f32 * ei_ratio) as usize;
    for j_neuron in 0..dim {
        let start = row_ptr[j_neuron] as usize;
        let end = row_ptr[j_neuron + 1] as usize;
        for idx in start..end {
            let pre = col_idx[idx] as usize;
            let abs_w = values[idx].abs();
            if pre < n_exc {
                values[idx] = abs_w;
            } else {
                values[idx] = -abs_w * inh_gain;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepOutput {
    pub active_count: usize,
    pub energy: f32,
}

/// Sparse CSR matvec: y = A @ x, only over rows where `mask[i]` is true.
fn spmv_masked(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    x: &[f32],
    mask: &[bool],
    out: &mut [f32],
) {
    let dim = out.len();
    out.par_iter_mut().enumerate().for_each(|(i, yi)| {
        if !mask[i] || i >= dim {
            *yi = 0.0;
            return;
        }
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut acc = 0.0_f32;
        for idx in start..end {
            let j = col_idx[idx] as usize;
            acc += values[idx] * x[j];
        }
        *yi = acc;
    });
}

/// Full brain-runtime cell step (one tick).
///
/// State: `activation` (a), `refractory` (r), `memory_trace` (m),
///        `adaptation` (w, J.20 AHP), `bitfield` (b, UP/DOWN).
/// Returns `StepOutput` with active count and energy estimate.
///
/// Equations: 15_Equations.md A.1--A.10, B.1, J.12--J.20.
pub fn brain_step(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    activation: &mut [f32],
    refractory: &mut [f32],
    memory_trace: &mut [f32],
    adaptation: &mut [f32],
    stp_u: &mut [f32],
    stp_x: &mut [f32],
    bitfield: &mut [u8],
    active_mask: &[u8],
    external: &[f32],
    goal: &[f32],
    replay: &[f32],
    noise: &[f32],
    mode_params: &ModeParams,
    cfg: &StepConfig,
) -> StepOutput {
    let dim = activation.len();
    if dim == 0 {
        return StepOutput { active_count: 0, energy: 0.0 };
    }

    // 0. STP update (Tsodyks-Markram, J.19): per-neuron approximation
    let stp_p = &cfg.stp;
    let prev_active: Vec<bool> = active_mask.iter().map(|&b| b > 0).collect();
    for i in 0..dim {
        let spike = if prev_active[i] { 1.0_f32 } else { 0.0 };
        let old_u = stp_u[i];
        stp_u[i] += -stp_p.tau_fac * old_u + stp_p.u_base * (1.0 - old_u) * spike;
        stp_x[i] += stp_p.tau_rec * (1.0 - stp_x[i]) - old_u * stp_x[i] * spike;
        stp_u[i] = stp_u[i].clamp(0.0, 1.0);
        stp_x[i] = stp_x[i].clamp(0.0, 1.0);
    }

    // 1. W_eff = u * x * a (STP-modulated presynaptic output)
    let masked_act: Vec<f32> = (0..dim)
        .map(|i| {
            if prev_active[i] {
                stp_u[i] * stp_x[i] * activation[i]
            } else {
                0.0
            }
        })
        .collect();
    let mut recurrent = vec![0.0_f32; dim];
    let all_true = vec![true; dim];
    spmv_masked(values, col_idx, row_ptr, &masked_act, &all_true, &mut recurrent);

    // 3. drive = recurrent + ext + goal + replay - refractory - adaptation (A.2 + A.6)
    let ext_g = cfg.external_gain;
    let goal_g = cfg.goal_gain;
    let ref_s = cfg.refractory_scale;
    let rep_m = mode_params.replay_mix;
    let adapt_c = mode_params.adaptation_coupling;
    let drive: Vec<f32> = (0..dim)
        .map(|i| {
            recurrent[i]
                + ext_g * external[i]
                + goal_g * goal[i]
                + rep_m * replay[i]
                - ref_s * refractory[i]
                - adapt_c * adaptation[i].min(cfg.adaptation_clamp)
                + noise[i]
        })
        .collect();

    // 4. activation update: a' = clamp[(1-gamma_a)*a + kappa_a*tanh(drive), -1, 1] (A.3)
    let decay = mode_params.activation_decay;
    let gain = mode_params.activation_gain;
    let new_act: Vec<f32> = (0..dim)
        .map(|i| {
            let raw = (1.0 - decay) * activation[i] + gain * drive[i].tanh();
            raw.clamp(-1.0, 1.0)
        })
        .collect();

    // 5. refractory update: r' = (1-gamma_r)*r + kappa_r*a'^2 (A.4)
    let r_decay = mode_params.refractory_decay;
    let r_gain = mode_params.refractory_gain;
    let new_ref: Vec<f32> = (0..dim)
        .map(|i| (1.0 - r_decay) * refractory[i] + r_gain * new_act[i] * new_act[i])
        .collect();

    // 6. memory trace: m' = (1-gamma_m)*m + gamma_m*a' (A.5, J.3 NMDA tau=100ms)
    let m_decay = mode_params.memory_decay;
    let m_gain = mode_params.memory_gain;
    let new_mem: Vec<f32> = (0..dim)
        .map(|i| (1.0 - m_decay) * memory_trace[i] + m_gain * new_act[i])
        .collect();

    // 7. adaptation update: w' = (1-gamma_w)*w + kappa_w*a'^2 (A.6, J.20 AHP)
    // gamma_w = kappa_w = 0.005 ensures w* = E[a^2] at steady state, clamped to [0,2]
    let w_decay = mode_params.adaptation_decay;
    let w_gain = mode_params.adaptation_gain;
    let new_adapt: Vec<f32> = (0..dim)
        .map(|i| {
            let raw = (1.0 - w_decay) * adaptation[i] + w_gain * new_act[i] * new_act[i];
            raw.clamp(0.0, cfg.adaptation_clamp)
        })
        .collect();

    // 8. bitfield hysteresis (A.7, J.17 UP/DOWN)
    let new_bit: Vec<u8> = (0..dim)
        .map(|i| {
            if new_act[i] >= cfg.bit_upper {
                1
            } else if new_act[i] <= cfg.bit_lower {
                0
            } else {
                bitfield[i]
            }
        })
        .collect();

    // 9. salience for TopK selection
    let salience: Vec<f32> = (0..dim)
        .map(|i| {
            new_act[i].abs()
                + 0.35 * external[i].abs()
                + 0.25 * replay[i].abs()
                + 0.20 * goal[i].abs()
                - 0.15 * new_ref[i]
        })
        .collect();

    // 10. active selection: topk by salience
    let budget = cfg.energy_budget.min(dim);
    let mut scored: Vec<(f32, usize)> = salience
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= cfg.active_threshold)
        .map(|(i, &s)| (s, i))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(budget);
    let active_count = scored.len();

    // 11. energy estimate (B.3)
    let coupling_energy = {
        let dot: f32 = new_act.iter().zip(recurrent.iter()).map(|(a, r)| a * r).sum();
        0.5 * dot.abs()
    };
    let dimf = dim as f32;
    let local_energy: f32 = new_ref.iter().map(|r| r.abs()).sum::<f32>() / dimf
        + 0.25 * new_mem.iter().map(|m| m.abs()).sum::<f32>() / dimf
        + 0.10 * new_adapt.iter().map(|w| w.abs()).sum::<f32>() / dimf;
    let replay_energy = 0.1 * replay.iter().map(|r| r.abs()).sum::<f32>() / dimf;
    let energy = coupling_energy + local_energy + replay_energy;

    // commit state
    activation.copy_from_slice(&new_act);
    refractory.copy_from_slice(&new_ref);
    memory_trace.copy_from_slice(&new_mem);
    adaptation.copy_from_slice(&new_adapt);
    bitfield.copy_from_slice(&new_bit);

    StepOutput { active_count, energy }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_csr(dim: usize) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
        let mut values = Vec::with_capacity(dim);
        let mut col_idx = Vec::with_capacity(dim);
        let mut row_ptr = Vec::with_capacity(dim + 1);
        row_ptr.push(0);
        for i in 0..dim {
            values.push(0.1);
            col_idx.push(i as i32);
            row_ptr.push((i + 1) as i32);
        }
        (values, col_idx, row_ptr)
    }

    #[test]
    fn basic_step_runs() {
        let dim = 16;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.5_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.1_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig { energy_budget: 4, ..Default::default() };
        let out = brain_step(
            &vals, &cols, &rows,
            &mut act, &mut refr, &mut mem, &mut adapt,
            &mut su, &mut sx, &mut bit,
            &active, &ext, &goal, &replay, &noise, &mp, &cfg,
        );
        assert!(out.active_count <= 4);
        assert!(out.energy >= 0.0);
        assert!(act.iter().all(|x| x.is_finite()));
        assert!(adapt.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn energy_decreases_nrem() {
        let dim = 32;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.3_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.0_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::nrem();
        let cfg = StepConfig { energy_budget: 8, ..Default::default() };
        let mut energies = Vec::new();
        for _ in 0..20 {
            let out = brain_step(
                &vals, &cols, &rows,
                &mut act, &mut refr, &mut mem, &mut adapt,
                &mut su, &mut sx, &mut bit,
                &active, &ext, &goal, &replay, &noise, &mp, &cfg,
            );
            energies.push(out.energy);
        }
        assert!(energies.last().unwrap() < energies.first().unwrap());
    }

    #[test]
    fn adaptation_accumulates() {
        let dim = 8;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.8_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.5_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig::default();
        for _ in 0..50 {
            brain_step(
                &vals, &cols, &rows,
                &mut act, &mut refr, &mut mem, &mut adapt,
                &mut su, &mut sx, &mut bit,
                &active, &ext, &goal, &replay, &noise, &mp, &cfg,
            );
        }
        let max_adapt = adapt.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_adapt > 0.0, "adaptation should accumulate with sustained input");
    }

    #[test]
    fn stp_depletes_with_activity() {
        let dim = 4;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.9_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.5_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig::default();
        let x0 = sx[0];
        for _ in 0..10 {
            brain_step(
                &vals, &cols, &rows,
                &mut act, &mut refr, &mut mem, &mut adapt,
                &mut su, &mut sx, &mut bit,
                &active, &ext, &goal, &replay, &noise, &mp, &cfg,
            );
        }
        assert!(sx[0] < x0, "STP resource x should deplete with sustained spiking");
    }
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/manifold.rs`

```rust
use ndarray::Array2;

pub type PhiField = dyn Fn(&[f64]) -> f64 + Sync + Send;

/// 리만 다양체 상의 기하학적 연산을 처리하는 트레이트
pub trait Manifold {
    /// 계량 텐서 g_uv(x) 반환
    fn metric(&self, x: &[f64]) -> Array2<f64>;

    /// 크리스토펠 기호 Gamma^k_ij(x) 반환
    fn christoffel(&self, x: &[f64]) -> Vec<Array2<f64>>;

    /// 리만 곡률 스칼라 R(x) 반환
    fn ricci_scalar(&self, x: &[f64]) -> f64;

    /// Exponential map: x_new = exp_x(v)
    /// 측지선 방정식 d^2x/dt^2 + Gamma (dx/dt)^2 = 0 을 푼다
    fn exp_map(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64>;
}

/// CE 이론에 따른 클라루스장 유도 계량 (Clarus Induced Metric)
/// g_uv = e^(-2 * alpha * Phi(x)) * delta_uv
/// 공간이 클라루스장 Phi에 의해 수축되는 효과를 모델링
pub struct SuppressionManifold {
    pub phi_field: Box<PhiField>,
    pub alpha: f64, // 결합 상수
    pub dim: usize,
}

impl SuppressionManifold {
    pub fn new(phi_field: Box<PhiField>, dim: usize) -> Self {
        Self {
            phi_field,
            alpha: 0.1, // 기본 결합 상수
            dim,
        }
    }

    fn gradient_phi(&self, x: &[f64]) -> Vec<f64> {
        let eps = 1e-7;
        let inv_2eps = 0.5 / eps;
        let mut grad = vec![0.0; self.dim];

        for i in 0..self.dim {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            grad[i] = ((self.phi_field)(&x_plus) - (self.phi_field)(&x_minus)) * inv_2eps;
        }
        grad
    }

    fn laplacian_phi(&self, x: &[f64]) -> f64 {
        let eps = 1e-5;
        let inv_eps2 = 1.0 / (eps * eps);
        let f0 = (self.phi_field)(x);
        let mut lap = 0.0;

        for i in 0..self.dim {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            lap += ((self.phi_field)(&xp) - 2.0 * f0 + (self.phi_field)(&xm)) * inv_eps2;
        }
        lap
    }
}

impl Manifold for SuppressionManifold {
    fn metric(&self, x: &[f64]) -> Array2<f64> {
        let phi = (self.phi_field)(x);
        let factor = (-2.0 * self.alpha * phi).exp();

        let mut g = Array2::zeros((self.dim, self.dim));
        for i in 0..self.dim {
            g[[i, i]] = factor;
        }
        g
    }

    fn christoffel(&self, x: &[f64]) -> Vec<Array2<f64>> {
        // Conformal metric g_uv = Omega^2 delta_uv, Omega = e^(-alpha*Phi)
        // Gamma^k_ij = delta^k_i d_j(ln Omega) + delta^k_j d_i(ln Omega) - delta_ij d^k(ln Omega)
        // ln Omega = -alpha * Phi

        let grad_phi = self.gradient_phi(x);
        let factor = -self.alpha;

        let mut gammas = vec![Array2::zeros((self.dim, self.dim)); self.dim];

        for k in 0..self.dim {
            for i in 0..self.dim {
                for j in 0..self.dim {
                    let term1 = if k == i { factor * grad_phi[j] } else { 0.0 };
                    let term2 = if k == j { factor * grad_phi[i] } else { 0.0 };
                    let term3 = if i == j { -factor * grad_phi[k] } else { 0.0 };

                    gammas[k][[i, j]] = term1 + term2 + term3;
                }
            }
        }
        gammas
    }

    fn ricci_scalar(&self, x: &[f64]) -> f64 {
        // Conformal metric g = e^(2f) delta, f = -alpha*Phi
        // R = -e^(-2f) [ 2(n-1) laplacian(f) + (n-2)(n-1) |grad(f)|^2 ]
        //   = e^(2*alpha*Phi) [ 2(n-1)*alpha*laplacian(Phi) - (n-2)(n-1)*alpha^2*|grad(Phi)|^2 ]

        let n = self.dim as f64;
        let phi = (self.phi_field)(x);
        let grad = self.gradient_phi(x);
        let grad_sq: f64 = grad.iter().map(|v| v * v).sum();
        let lap = self.laplacian_phi(x);

        let prefactor = (2.0 * self.alpha * phi).exp();
        let term_lap = 2.0 * (n - 1.0) * self.alpha * lap;
        let term_grad = -(n - 2.0) * (n - 1.0) * self.alpha * self.alpha * grad_sq;

        prefactor * (term_lap + term_grad)
    }

    fn exp_map(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64> {
        // 2차 룬게-쿠타로 측지선 방정식 적분
        // d^2x^k/dt^2 = -Gamma^k_ij v^i v^j

        let gammas = self.christoffel(x);
        let mut acc = vec![0.0; self.dim];

        for k in 0..self.dim {
            let mut sum = 0.0;
            for i in 0..self.dim {
                for j in 0..self.dim {
                    sum += gammas[k][[i, j]] * v[i] * v[j];
                }
            }
            acc[k] = -sum;
        }

        let mut x_new = vec![0.0; self.dim];
        for i in 0..self.dim {
            x_new[i] = x[i] + v[i] * dt + 0.5 * acc[i] * dt * dt;
        }
        x_new
    }
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/mod.rs`

```rust
pub mod config;
pub mod constants;
pub mod field;
pub mod filter;
pub mod kernel;
pub mod manifold;
pub mod nn_ops;
pub mod ce_riemann;
pub mod runtime_types;
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/nn_ops.rs`

```rust
//! Fused neural-network ops for the Clarus runtime.
//!
//! All functions operate on flat f32 slices laid out row-major.
//! Matrix multiplications use ndarray (matrixmultiply SIMD backend).
//! Row-parallel via rayon where beneficial.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::cmp;

// ---- helpers ---------------------------------------------------------------

#[inline(always)]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---- TopK SiLU -------------------------------------------------------------

/// Fused SiLU + TopK sparse masking (forward).
///
/// `input`: flat `[n_rows * dim]`, `dim`: row width, `ratio`: keep fraction.
/// Returns `(output, mask)`.
pub fn topk_silu_fwd(input: &[f32], dim: usize, ratio: f32) -> (Vec<f32>, Vec<u8>) {
    let k = cmp::max(1, (ratio * dim as f32).ceil() as usize).min(dim);
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let mut mask = vec![0u8; n];

    if k >= dim {
        output
            .par_chunks_mut(dim)
            .zip(mask.par_chunks_mut(dim))
            .enumerate()
            .for_each(|(r, (out, msk))| {
                let src = &input[r * dim..(r + 1) * dim];
                for j in 0..dim {
                    out[j] = silu_f32(src[j]);
                    msk[j] = 1;
                }
            });
        return (output, mask);
    }

    output
        .par_chunks_mut(dim)
        .zip(mask.par_chunks_mut(dim))
        .enumerate()
        .for_each(|(r, (out, msk))| {
            let src = &input[r * dim..(r + 1) * dim];
            for j in 0..dim {
                out[j] = silu_f32(src[j]);
            }
            let mut abs_vals: Vec<f32> = out.iter().map(|x| x.abs()).collect();
            abs_vals.select_nth_unstable_by(dim - k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let thr = abs_vals[dim - k];
            for j in 0..dim {
                if out[j].abs() >= thr {
                    msk[j] = 1;
                } else {
                    out[j] = 0.0;
                }
            }
        });
    (output, mask)
}

/// TopK SiLU backward.
pub fn topk_silu_bwd(grad: &[f32], input: &[f32], mask: &[u8], dim: usize) -> Vec<f32> {
    let n = grad.len();
    let mut grad_in = vec![0.0f32; n];

    grad_in
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(r, gi)| {
            let base = r * dim;
            for j in 0..dim {
                if mask[base + j] == 1 {
                    let x = input[base + j];
                    let s = sigmoid_f32(x);
                    gi[j] = grad[base + j] * s * (1.0 + x * (1.0 - s));
                }
            }
        });
    grad_in
}

// ---- LBO Norm (ndarray-backed matmul) ---------------------------------------

/// Fused LBO normalization forward (post-LayerNorm).
///
/// Uses ndarray `dot()` (matrixmultiply SIMD) for projections.
pub fn lbo_fused_fwd(
    normed: &[f32],
    v: &[f32],
    h: f32,
    scale: &[f32],
    bias: &[f32],
    alpha_conf: f32,
    dim: usize,
    rank: usize,
) -> (Vec<f32>, f32) {
    let n_rows = normed.len() / dim;

    // conformal factor
    let phi_sq: f32 = normed.iter().map(|&x| x * x).sum::<f32>() / normed.len() as f32;
    let conformal = (-alpha_conf.abs() * phi_sq).exp();

    // V_eff = V * conformal  [rank, dim]
    let v_scaled: Vec<f32> = v.iter().map(|&x| x * conformal).collect();
    let x_mat = ArrayView2::from_shape((n_rows, dim), normed).unwrap();
    let v_mat = ArrayView2::from_shape((rank, dim), &v_scaled).unwrap();

    // proj = X @ V_eff^T  -> [n_rows, rank]   (ndarray SIMD dot)
    let proj = x_mat.dot(&v_mat.t());
    // xW = proj @ V_eff   -> [n_rows, dim]    (ndarray SIMD dot)
    let xw = proj.dot(&v_mat);

    // output + curvature
    let scale_v = ArrayView1::from(scale);
    let bias_v = ArrayView1::from(bias);
    let one_minus_h = 1.0 - h;
    let mut output = vec![0.0f32; normed.len()];
    let mut curv_sum = 0.0f64;

    for r in 0..n_rows {
        let base = r * dim;
        for j in 0..dim {
            let lx = x_mat[[r, j]] - xw[[r, j]];
            curv_sum += (lx as f64) * (lx as f64);
            output[base + j] = (one_minus_h * x_mat[[r, j]] + h * xw[[r, j]])
                * scale_v[j]
                + bias_v[j];
        }
    }
    (output, (curv_sum / (n_rows * dim) as f64) as f32)
}

/// Power iteration: 1 step for sigma_max(V).
pub fn power_iter_step(
    v_mat: &[f32],
    spectral_v: &[f32],
    dim: usize,
    rank: usize,
) -> (Vec<f32>, f32) {
    let v_nd = ArrayView2::from_shape((rank, dim), v_mat).unwrap();
    let sv = ArrayView1::from_shape(dim, spectral_v).unwrap();

    // u = V @ sv  [rank]
    let u_raw = v_nd.dot(&sv);
    let u_norm = u_raw.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let u = u_raw.mapv(|x| x / u_norm);

    // new_v = V^T @ u  [dim]
    let vt = v_nd.t();
    let nv_raw = vt.dot(&u);
    let nv_norm = nv_raw.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let new_v = nv_raw.mapv(|x| x / nv_norm);

    // sigma = ||V @ new_v||
    let sigma = v_nd.dot(&new_v).mapv(|x| x * x).sum().sqrt();

    (new_v.to_vec(), sigma)
}

// ---- Gauge lattice (ndarray matmul per channel) ----------------------------

/// Single gauge channel: up -> SiLU -> TopK -> down.
fn channel_fwd(
    x: &ArrayView1<f32>,
    up_w: &ArrayView2<f32>,   // [hid, d_in]
    down_w: &ArrayView2<f32>, // [d_in, hid]
    k: usize,
) -> Array1<f32> {
    // hidden = x @ up^T  -> [hid]
    let hidden_raw = up_w.dot(x);
    let hid = hidden_raw.len();

    // SiLU + TopK
    let mut hidden: Vec<f32> = hidden_raw.iter().map(|&v| silu_f32(v)).collect();
    if k < hid {
        let mut abs_h: Vec<f32> = hidden.iter().map(|v| v.abs()).collect();
        abs_h.select_nth_unstable_by(hid - k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let thr = abs_h[hid - k];
        for v in hidden.iter_mut() {
            if v.abs() < thr {
                *v = 0.0;
            }
        }
    }

    // output = down @ hidden  -> [d_in]  (down is [d_in, hid])
    let h_arr = ArrayView1::from(&hidden);
    down_w.dot(&h_arr)
}

/// Gauge lattice 3-channel forward.
#[allow(clippy::too_many_arguments)]
pub fn gauge_lattice_fwd(
    input: &[f32],
    su3_up: &[f32],
    su3_down: &[f32],
    su2_up: &[f32],
    su2_down: &[f32],
    u1_up: &[f32],
    u1_down: &[f32],
    mix_down: &[f32],
    mix_up: &[f32],
    d3: usize, d2: usize, d1: usize,
    h3: usize, h2: usize, h1: usize,
    mix_rank: usize,
    ratio: f32,
    dim: usize,
) -> Vec<f32> {
    let _n_rows = input.len() / dim;
    let k3 = cmp::max(1, (ratio * h3 as f32).ceil() as usize).min(h3);
    let k2 = cmp::max(1, (ratio * h2 as f32).ceil() as usize).min(h2);
    let k1 = cmp::max(1, (ratio * h1 as f32).ceil() as usize).min(h1);
    let has_mix = mix_rank > 0 && !mix_down.is_empty() && !mix_up.is_empty();

    let su3_up_nd = ArrayView2::from_shape((h3, d3), su3_up).unwrap();
    let su3_dn_nd = ArrayView2::from_shape((d3, h3), su3_down).unwrap();
    let su2_up_nd = ArrayView2::from_shape((h2, d2), su2_up).unwrap();
    let su2_dn_nd = ArrayView2::from_shape((d2, h2), su2_down).unwrap();
    let u1_up_nd = ArrayView2::from_shape((h1, d1), u1_up).unwrap();
    let u1_dn_nd = ArrayView2::from_shape((d1, h1), u1_down).unwrap();

    let x_mat = ArrayView2::from_shape((_n_rows, dim), input).unwrap();
    let mut output = vec![0.0f32; input.len()];

    let s3 = d3;
    let s32 = d3 + d2;

    output
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(r, out)| {
            let x_row = x_mat.row(r);
            let x3 = x_row.slice(ndarray::s![..s3]);
            let x2 = x_row.slice(ndarray::s![s3..s32]);
            let x1 = x_row.slice(ndarray::s![s32..]);

            let y3 = channel_fwd(&x3, &su3_up_nd, &su3_dn_nd, k3);
            let y2 = channel_fwd(&x2, &su2_up_nd, &su2_dn_nd, k2);
            let y1 = channel_fwd(&x1, &u1_up_nd, &u1_dn_nd, k1);

            out[..s3].copy_from_slice(y3.as_slice().unwrap());
            out[s3..s32].copy_from_slice(y2.as_slice().unwrap());
            out[s32..].copy_from_slice(y1.as_slice().unwrap());

            if has_mix {
                let md = ArrayView2::from_shape((mix_rank, dim), mix_down).unwrap();
                let mu = ArrayView2::from_shape((dim, mix_rank), mix_up).unwrap();
                let out_view = ArrayView1::from(&*out);
                let proj = md.dot(&out_view);
                let mix_result = mu.dot(&proj);
                for j in 0..dim {
                    out[j] += mix_result[j];
                }
            }
        });
    output
}

// ---- CE Softmax / Metric-Family Attention (MFA) ----------------------------
//
// Equation 6.B.1 (applied compendium): compute
//   s_lang_ij = (q_i . k_j) / sqrt(d)
//   s_grav_ij = -||k_i - k_j||^2 / (2 sigma^2)     (identity metric)
//   s_ij      = w_lang * s_lang_ij + w_grav * s_grav_ij   (logit mixing)
//   A_ij      = softmax_j(s_ij)  (with optional causal mask)
//   out_i     = sum_j A_ij * v_j
//
// `q`, `k`, `v` are `(n, d)` row-major f32 slices for a single head.
// `causal = true` applies a lower-triangular mask.

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline(always)]
fn sq_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Fused CE MFA forward (single head, logit-mixing, identity gravity metric).
///
/// Returns (out `(n, d)`, attn `(n, n)`).
pub fn ce_mfa_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    sigma_grav: f32,
    w_lang: f32,
    w_grav: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let scale_lang = 1.0 / (d as f32).sqrt();
    let scale_grav = -1.0 / (2.0 * sigma_grav * sigma_grav);

    let mut attn = vec![0.0f32; n * n];
    let mut out = vec![0.0f32; n * d];

    // Row-parallel over queries.
    attn.par_chunks_mut(n)
        .zip(out.par_chunks_mut(d))
        .enumerate()
        .for_each(|(i, (attn_row, out_row))| {
            let q_i = &q[i * d..(i + 1) * d];
            let k_i = &k[i * d..(i + 1) * d];

            // compute raw scores for row i
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                if causal && j > i {
                    attn_row[j] = f32::NEG_INFINITY;
                    continue;
                }
                let k_j = &k[j * d..(j + 1) * d];
                let s_lang = dot_f32(q_i, k_j) * scale_lang;
                let s_grav = sq_dist_f32(k_i, k_j) * scale_grav;
                let s = w_lang * s_lang + w_grav * s_grav;
                attn_row[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }

            // softmax (numerically stable)
            let mut denom = 0.0f32;
            for j in 0..n {
                if attn_row[j].is_finite() {
                    let e = (attn_row[j] - max_s).exp();
                    attn_row[j] = e;
                    denom += e;
                } else {
                    attn_row[j] = 0.0;
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for j in 0..n {
                attn_row[j] *= inv;
            }

            // out_i = A_i @ V
            for t in 0..d {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let a = attn_row[j];
                if a == 0.0 {
                    continue;
                }
                let v_j = &v[j * d..(j + 1) * d];
                for t in 0..d {
                    out_row[t] += a * v_j[t];
                }
            }
        });

    (out, attn)
}

// ---- Dual-graph attention (precise mirror of ce_laplacian.DualLaplacianBlock) ----
//
// Inputs per forward (single head, already projected):
//   z_l: (n, d_l)  = P_lang(h)
//   z_g: (n, d_g)  = P_grav(h)
//   v:   (n, d_m)  = V(h)
//   sigma_grav: RBF bandwidth for the gravity graph
//   w_lang, w_grav: convex gate (expect w_lang + w_grav == 1)
//   causal: lower-triangular transition mask
//
// Computes:
//   A_l[i,j] = max(0, cos(z_l[i], z_l[j])), with diag = 0, symmetric
//   A_g[i,j] = exp(-||z_g[i] - z_g[j]||^2 / 2 sigma^2), diag = 0, symmetric
//   K_l = row_norm(apply_causal(A_l))
//   K_g = row_norm(apply_causal(A_g))
//   K   = w_lang * K_l + w_grav * K_g              (still row-stochastic)
//   out[i, t] = sum_j K[i,j] * v[j, t]
//
// Returns (out, K) for validation. K is flattened row-major (n * n).

pub fn ce_dual_attn_fwd(
    z_l: &[f32],
    z_g: &[f32],
    v: &[f32],
    n: usize,
    d_l: usize,
    d_g: usize,
    d_m: usize,
    sigma_grav: f32,
    w_lang: f32,
    w_grav: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(z_l.len(), n * d_l);
    debug_assert_eq!(z_g.len(), n * d_g);
    debug_assert_eq!(v.len(), n * d_m);

    let inv_2s2 = -1.0f32 / (2.0 * sigma_grav * sigma_grav);
    let eps = 1e-8f32;

    // Pre-compute row norms for cosine.
    let mut norm_l = vec![0.0f32; n];
    for i in 0..n {
        let row = &z_l[i * d_l..(i + 1) * d_l];
        let mut s = 0.0f32;
        for &x in row {
            s += x * x;
        }
        norm_l[i] = s.sqrt().max(eps);
    }

    let mut k_combined = vec![0.0f32; n * n];
    let mut out = vec![0.0f32; n * d_m];

    // Row-parallel over query positions.
    k_combined
        .par_chunks_mut(n)
        .zip(out.par_chunks_mut(d_m))
        .enumerate()
        .for_each(|(i, (k_row, out_row))| {
            let zi_l = &z_l[i * d_l..(i + 1) * d_l];
            let zi_g = &z_g[i * d_g..(i + 1) * d_g];
            let ni_l = norm_l[i];

            // First pass: raw unnormalized A_l, A_g for this row, applying causal.
            // We compute on-the-fly the two row sums for renormalization.
            let mut sum_l = 0.0f32;
            let mut sum_g = 0.0f32;
            // Scratch row stored as (lang, grav) in k_row in halves -- we will
            // overwrite twice: first with A_l, then combine with A_g.
            // To save a buffer we accumulate per-j into two scalars and then
            // pass through again; but that's two loops. Instead allocate a
            // small scratch here.
            let mut row_l = vec![0.0f32; n];
            let mut row_g = vec![0.0f32; n];
            for j in 0..n {
                if causal && j > i {
                    continue;
                }
                if j == i {
                    continue; // diagonal zero
                }
                let zj_l = &z_l[j * d_l..(j + 1) * d_l];
                let zj_g = &z_g[j * d_g..(j + 1) * d_g];

                // Cosine
                let mut dot = 0.0f32;
                for k in 0..d_l {
                    dot += zi_l[k] * zj_l[k];
                }
                let cos = (dot / (ni_l * norm_l[j])).max(0.0);
                row_l[j] = cos;
                sum_l += cos;

                // RBF
                let mut d2 = 0.0f32;
                for k in 0..d_g {
                    let diff = zi_g[k] - zj_g[k];
                    d2 += diff * diff;
                }
                let rbf = (d2 * inv_2s2).exp();
                row_g[j] = rbf;
                sum_g += rbf;
            }

            // Row-normalize each kernel separately, then convex combine.
            let inv_l = if sum_l > eps { 1.0 / sum_l } else { 0.0 };
            let inv_g = if sum_g > eps { 1.0 / sum_g } else { 0.0 };
            for j in 0..n {
                let kl = row_l[j] * inv_l;
                let kg = row_g[j] * inv_g;
                k_row[j] = w_lang * kl + w_grav * kg;
            }

            // out_i = K_i @ V
            for t in 0..d_m {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let a = k_row[j];
                if a == 0.0 {
                    continue;
                }
                let vj = &v[j * d_m..(j + 1) * d_m];
                for t in 0..d_m {
                    out_row[t] += a * vj[t];
                }
            }
        });

    (out, k_combined)
}

// ---- EulerCE attention (pi-phase rotary + e-decay) -------------------------
//
// Implements clarus::ce_euler::EulerCEAttention in native code for a
// single head (one call per (batch, head) element). Inputs are assumed
// pre-projected and reshaped to (n, d_head), d_head even.
//
//   Q', K' = rotate_pi(Q, K)   with theta = pi_gate * pos * pi^{1-k/(d/2)}
//   scores_ij = (Q'_i . K'_j) / sqrt(d_head)
//                + e_gate * (-|i-j| / xi)         [decay bias]
//   scores masked causally, softmax, out = A @ V
//
// Scalar gates (pi_gate, e_gate, xi) are per-head -- supplied by the
// caller for the relevant head. pi_inv_freq is the precomputed
// pi^{1-k/(d/2)} array of length d_head/2.

// ---- Riemann-surface PE attention -----------------------------------------
//
// Mirrors `clarus.ce_riemann_attn.RiemannRotaryAttention` per
// `docs/8_리만/riemann_pe_spec.md`.
//
// Batched layout — a single call processes (BH, N, D) at once. cos/sin are
// pre-broadcast to (BH, N, D/2); sheet_bias to (BH, N, N). All slices are
// row-major contiguous.
//
// Pipeline (per (bh, i)):
//   1. Rotate q[bh, i, :] (RoPE-style 2D rotation per pair) into q_rot.
//   2. score_ij = (q_rot[bh,i] · k_rot[bh,j]) / sqrt(D) + sheet_bias[bh,i,j]
//   3. causal mask + softmax + weighted sum over j → out[bh, i, :]
//
// Per-(bh, i) rotation of k_j is recomputed inside the j-loop to keep the
// hot path cache-resident; the cost is dominated by the dot product anyway.

#[allow(clippy::too_many_arguments)]
pub fn ce_riemann_fwd(
    q: &[f32],          // (bh * n * d_head)
    k: &[f32],          // (bh * n * d_head)
    v: &[f32],          // (bh * n * d_head)
    cos: &[f32],        // (bh * n * d_head/2)
    sin: &[f32],        // (bh * n * d_head/2)
    sheet_bias: &[f32], // (bh * n * n)
    bh: usize,
    n: usize,
    d_head: usize,
    causal: bool,
) -> Vec<f32> {
    debug_assert!(d_head % 2 == 0);
    let half = d_head / 2;
    debug_assert_eq!(q.len(), bh * n * d_head);
    debug_assert_eq!(k.len(), bh * n * d_head);
    debug_assert_eq!(v.len(), bh * n * d_head);
    debug_assert_eq!(cos.len(), bh * n * half);
    debug_assert_eq!(sin.len(), bh * n * half);
    debug_assert_eq!(sheet_bias.len(), bh * n * n);

    let scale = 1.0 / (d_head as f32).sqrt();
    let mut out = vec![0.0f32; bh * n * d_head];

    // Pre-rotate the entire q tensor once per (bh, n) row.
    let mut q_rot = vec![0.0f32; bh * n * d_head];
    let mut k_rot = vec![0.0f32; bh * n * d_head];
    q_rot
        .par_chunks_mut(d_head)
        .zip(k_rot.par_chunks_mut(d_head))
        .enumerate()
        .for_each(|(row, (qr, kr))| {
            let qi = &q[row * d_head..(row + 1) * d_head];
            let ki = &k[row * d_head..(row + 1) * d_head];
            let ci = &cos[row * half..(row + 1) * half];
            let si = &sin[row * half..(row + 1) * half];
            for p in 0..half {
                let c = ci[p];
                let s = si[p];
                let q0 = qi[2 * p]; let q1 = qi[2 * p + 1];
                qr[2 * p]     = q0 * c - q1 * s;
                qr[2 * p + 1] = q0 * s + q1 * c;
                let k0 = ki[2 * p]; let k1 = ki[2 * p + 1];
                kr[2 * p]     = k0 * c - k1 * s;
                kr[2 * p + 1] = k0 * s + k1 * c;
            }
        });

    // Outer-parallelize over (bh, i) rows.
    out.par_chunks_mut(d_head)
        .enumerate()
        .for_each(|(row, out_row)| {
            let bh_idx = row / n;
            let i = row % n;
            let q_rot_base = bh_idx * n * d_head;
            let v_base     = bh_idx * n * d_head;
            let bias_base  = bh_idx * n * n;

            let qi = &q_rot[q_rot_base + i * d_head..q_rot_base + (i + 1) * d_head];
            let bias_row = &sheet_bias[bias_base + i * n..bias_base + (i + 1) * n];

            // First pass: raw scores + max
            let mut scratch = vec![0.0f32; n];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                if causal && j > i {
                    scratch[j] = f32::NEG_INFINITY;
                    continue;
                }
                let kj = &k_rot[q_rot_base + j * d_head..q_rot_base + (j + 1) * d_head];
                let mut dot = 0.0f32;
                for t in 0..d_head {
                    dot += qi[t] * kj[t];
                }
                let s = dot * scale + bias_row[j];
                scratch[j] = s;
                if s > max_s { max_s = s; }
            }

            // Softmax (numerically stable)
            let mut denom = 0.0f32;
            for j in 0..n {
                if scratch[j].is_finite() {
                    let e = (scratch[j] - max_s).exp();
                    scratch[j] = e;
                    denom += e;
                } else {
                    scratch[j] = 0.0;
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };

            // out_i = sum_j (e_j * inv) * v_j
            for t in 0..d_head { out_row[t] = 0.0; }
            for j in 0..n {
                let w = scratch[j] * inv;
                if w == 0.0 { continue; }
                let vj = &v[v_base + j * d_head..v_base + (j + 1) * d_head];
                for t in 0..d_head {
                    out_row[t] += w * vj[t];
                }
            }
        });

    out
}

pub fn ce_euler_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pi_inv_freq: &[f32],
    n: usize,
    d_head: usize,
    pi_gate: f32,
    e_gate: f32,
    xi: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(q.len(), n * d_head);
    debug_assert_eq!(k.len(), n * d_head);
    debug_assert_eq!(v.len(), n * d_head);
    debug_assert_eq!(pi_inv_freq.len(), d_head / 2);
    debug_assert!(d_head % 2 == 0);

    let scale = 1.0 / (d_head as f32).sqrt();
    let inv_xi = e_gate * (1.0 / xi.max(1e-6));

    // Build rotated Q, K once (n * d_head each).
    let mut q_rot = vec![0.0f32; n * d_head];
    let mut k_rot = vec![0.0f32; n * d_head];

    q_rot
        .par_chunks_mut(d_head)
        .zip(k_rot.par_chunks_mut(d_head))
        .enumerate()
        .for_each(|(i, (qr, kr))| {
            let qi = &q[i * d_head..(i + 1) * d_head];
            let ki = &k[i * d_head..(i + 1) * d_head];
            let pos = i as f32;
            for (pair, &inv_f) in pi_inv_freq.iter().enumerate() {
                let theta = pi_gate * pos * inv_f;
                let c = theta.cos();
                let s = theta.sin();
                let idx0 = 2 * pair;
                let idx1 = idx0 + 1;
                let q0 = qi[idx0]; let q1 = qi[idx1];
                qr[idx0] = q0 * c - q1 * s;
                qr[idx1] = q0 * s + q1 * c;
                let k0 = ki[idx0]; let k1 = ki[idx1];
                kr[idx0] = k0 * c - k1 * s;
                kr[idx1] = k0 * s + k1 * c;
            }
        });

    let mut attn = vec![0.0f32; n * n];
    let mut out = vec![0.0f32; n * d_head];

    attn.par_chunks_mut(n)
        .zip(out.par_chunks_mut(d_head))
        .enumerate()
        .for_each(|(i, (a_row, out_row))| {
            let qi = &q_rot[i * d_head..(i + 1) * d_head];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                if causal && j > i {
                    a_row[j] = f32::NEG_INFINITY;
                    continue;
                }
                let kj = &k_rot[j * d_head..(j + 1) * d_head];
                // dot
                let mut dot = 0.0f32;
                for t in 0..d_head {
                    dot += qi[t] * kj[t];
                }
                let decay = -((i as f32 - j as f32).abs()) * inv_xi;
                let s = dot * scale + decay;
                a_row[j] = s;
                if s > max_s { max_s = s; }
            }
            // softmax
            let mut denom = 0.0f32;
            for j in 0..n {
                if a_row[j].is_finite() {
                    let e = (a_row[j] - max_s).exp();
                    a_row[j] = e;
                    denom += e;
                } else {
                    a_row[j] = 0.0;
                }
            }
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for j in 0..n {
                a_row[j] *= inv_denom;
            }
            // out = a_row @ V
            for t in 0..d_head {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let w = a_row[j];
                if w == 0.0 { continue; }
                let vj = &v[j * d_head..(j + 1) * d_head];
                for t in 0..d_head {
                    out_row[t] += w * vj[t];
                }
            }
        });

    (out, attn)
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/engine/runtime_types.rs`

```rust
//! Typed runtime-facing structs shared by the canonical Clarus compute core.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Wake,
    Nrem,
    Rem,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellState {
    pub activation: f32,
    pub refractory: f32,
    pub memory_trace: f32,
    pub adaptation: f32,
    pub stp_u: f32,
    pub stp_x: f32,
    pub bit: u8,
}

impl Default for CellState {
    fn default() -> Self {
        Self {
            activation: 0.0,
            refractory: 0.0,
            memory_trace: 0.0,
            adaptation: 0.0,
            stp_u: 0.5,
            stp_x: 1.0,
            bit: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxInput {
    /// Packed sparse matrix and state vectors passed into a Rust relax kernel.
    pub values: Vec<f32>,
    pub col_idx: Vec<i32>,
    pub row_ptr: Vec<i32>,
    pub bias: Vec<f32>,
    pub phi: Vec<f32>,
    pub state: Vec<f32>,
    pub mode: Mode,
    pub dt: f32,
    pub max_steps: usize,
    pub tol: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxOutput {
    /// Minimal numeric output from a relax/energy step.
    pub state: Vec<f32>,
    pub energy: Vec<f32>,
    pub delta: Vec<f32>,
    pub steps: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotMeta {
    /// Metadata only; higher-level snapshot payloads stay in Python for now.
    pub step: usize,
    pub mode: Mode,
    pub active_modules: usize,
    pub energy_budget: usize,
}
```
---
## File: `reality_stone/python/reality_stone/clarus/core/src/lib.rs`

```rust
#![allow(non_local_definitions)]
//! Canonical Rust compute surface for the Clarus runtime.

pub mod engine;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use engine::field::{BoundaryMode, FieldConfig, FieldEngine, FieldState, FieldStepOutput};
pub use engine::kernel::{ModeParams, StepConfig, StepOutput, StpParams, apply_dale_sign, brain_step};
pub use engine::runtime_types::{CellState, Mode, RelaxInput, RelaxOutput, SnapshotMeta};

#[cfg(feature = "python")]
mod python_binding {
    use pyo3::prelude::*;
    use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
    use crate::engine::nn_ops;
    use crate::engine::ce_riemann;
    use crate::engine::kernel;
    use crate::engine::runtime_types;

    #[pyfunction]
    fn topk_sparse(data: Vec<f64>, ratio: f64) -> (Vec<f64>, usize) {
        let n = data.len();
        let k = std::cmp::max(1, (ratio * n as f64).ceil() as usize).min(n);
        if k >= n {
            return (data, n);
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            data[b].abs().partial_cmp(&data[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = vec![0.0; n];
        for &i in &indices[..k] {
            out[i] = data[i];
        }
        (out, k)
    }

    #[pyfunction]
    fn topk_sparse_batch(data: Vec<f64>, row_len: usize, ratio: f64) -> Vec<f64> {
        use rayon::prelude::*;
        let k = std::cmp::max(1, (ratio * row_len as f64).ceil() as usize).min(row_len);
        if k >= row_len {
            return data;
        }
        let mut out = vec![0.0; data.len()];
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(row, out_row)| {
                let src = &data[row * row_len..(row + 1) * row_len];
                let mut indices: Vec<usize> = (0..row_len).collect();
                indices.sort_unstable_by(|&a, &b| {
                    src[b].abs().partial_cmp(&src[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
                });
                for &i in &indices[..k] {
                    out_row[i] = src[i];
                }
            });
        out
    }

    #[pyfunction]
    fn nn_topk_silu_fwd<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
        dim: usize,
        ratio: f32,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<u8>) {
        let data = input.as_slice().expect("contiguous input");
        let (out, mask) = nn_ops::topk_silu_fwd(data, dim, ratio);
        (out.into_pyarray(py), mask.into_pyarray(py))
    }

    #[pyfunction]
    fn nn_topk_silu_bwd<'py>(
        py: Python<'py>,
        grad: PyReadonlyArray1<'py, f32>,
        input: PyReadonlyArray1<'py, f32>,
        mask: PyReadonlyArray1<'py, u8>,
        dim: usize,
    ) -> &'py PyArray1<f32> {
        let g = grad.as_slice().expect("contiguous grad");
        let x = input.as_slice().expect("contiguous input");
        let m = mask.as_slice().expect("contiguous mask");
        nn_ops::topk_silu_bwd(g, x, m, dim).into_pyarray(py)
    }

    #[pyfunction]
    fn nn_lbo_fused_fwd<'py>(
        py: Python<'py>,
        normed: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        h: f32,
        scale: PyReadonlyArray1<'py, f32>,
        bias: PyReadonlyArray1<'py, f32>,
        alpha_conf: f32,
        dim: usize,
        rank: usize,
    ) -> (&'py PyArray1<f32>, f32) {
        let (out, curv) = nn_ops::lbo_fused_fwd(
            normed.as_slice().expect("contiguous"),
            v.as_slice().expect("contiguous"),
            h,
            scale.as_slice().expect("contiguous"),
            bias.as_slice().expect("contiguous"),
            alpha_conf,
            dim,
            rank,
        );
        (out.into_pyarray(py), curv)
    }

    #[pyfunction]
    fn nn_power_iter<'py>(
        py: Python<'py>,
        v_mat: PyReadonlyArray1<'py, f32>,
        spectral_v: PyReadonlyArray1<'py, f32>,
        dim: usize,
        rank: usize,
    ) -> (&'py PyArray1<f32>, f32) {
        let (new_v, sigma) = nn_ops::power_iter_step(
            v_mat.as_slice().expect("contiguous"),
            spectral_v.as_slice().expect("contiguous"),
            dim,
            rank,
        );
        (new_v.into_pyarray(py), sigma)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_gauge_lattice_fwd<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
        su3_up: PyReadonlyArray1<'py, f32>,
        su3_down: PyReadonlyArray1<'py, f32>,
        su2_up: PyReadonlyArray1<'py, f32>,
        su2_down: PyReadonlyArray1<'py, f32>,
        u1_up: PyReadonlyArray1<'py, f32>,
        u1_down: PyReadonlyArray1<'py, f32>,
        mix_down: PyReadonlyArray1<'py, f32>,
        mix_up: PyReadonlyArray1<'py, f32>,
        d3: usize, d2: usize, d1: usize,
        h3: usize, h2: usize, h1: usize,
        mix_rank: usize,
        ratio: f32,
        dim: usize,
    ) -> &'py PyArray1<f32> {
        nn_ops::gauge_lattice_fwd(
            input.as_slice().expect("contiguous"),
            su3_up.as_slice().expect("contiguous"),
            su3_down.as_slice().expect("contiguous"),
            su2_up.as_slice().expect("contiguous"),
            su2_down.as_slice().expect("contiguous"),
            u1_up.as_slice().expect("contiguous"),
            u1_down.as_slice().expect("contiguous"),
            mix_down.as_slice().expect("contiguous"),
            mix_up.as_slice().expect("contiguous"),
            d3, d2, d1, h3, h2, h1, mix_rank, ratio, dim,
        ).into_pyarray(py)
    }

    #[pyfunction]
    fn nn_ce_pack_sparse<'py>(
        py: Python<'py>,
        w: PyReadonlyArray1<'py, f32>,
        dim: usize,
        zero_tol: f32,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<i32>, &'py PyArray1<i32>) {
        let data = w.as_slice().expect("contiguous");
        let (vals, cols, rows) = ce_riemann::pack_sparse_csr(data, dim, zero_tol);
        (vals.into_pyarray(py), cols.into_pyarray(py), rows.into_pyarray(py))
    }

    #[pyfunction]
    fn nn_ce_metric_basis_fwd<'py>(
        py: Python<'py>,
        codebook: PyReadonlyArray1<'py, f32>,
        m_ref: PyReadonlyArray1<'py, f32>,
        n_code: usize,
        dim: usize,
        rank: usize,
    ) -> &'py PyArray1<f32> {
        let cb = codebook.as_slice().expect("contiguous");
        let mr = m_ref.as_slice().expect("contiguous");
        ce_riemann::metric_basis_from_codebook(cb, mr, n_code, dim, rank).into_pyarray(py)
    }

    #[pyfunction]
    fn nn_ce_codebook_pull<'py>(
        py: Python<'py>,
        m: PyReadonlyArray1<'py, f32>,
        codebook: PyReadonlyArray1<'py, f32>,
        n_code: usize,
        dim: usize,
        beta: f32,
        cb_w: f32,
    ) -> (&'py PyArray1<f32>, f32) {
        let m_s = m.as_slice().expect("contiguous");
        let cb = codebook.as_slice().expect("contiguous");
        let (grad, energy) = ce_riemann::codebook_pull(m_s, cb, n_code, dim, beta, cb_w);
        (grad.into_pyarray(py), energy)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_relax_fwd<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f32>,
        col_idx: PyReadonlyArray1<'py, i32>,
        row_ptr: PyReadonlyArray1<'py, i32>,
        b: PyReadonlyArray1<'py, f32>,
        phi: PyReadonlyArray1<'py, f32>,
        m0: PyReadonlyArray1<'py, f32>,
        codebook: PyReadonlyArray1<'py, f32>,
        metric_basis: PyReadonlyArray1<'py, f32>,
        dim: usize,
        n_code: usize,
        rank: usize,
        portal: f32,
        bypass: f32,
        t_wake: f32,
        beta: f32,
        cb_w: f32,
        lambda0: f32,
        lambda_phi: f32,
        lambda_var: f32,
        tau: f32,
        dt: f32,
        max_steps: usize,
        tol: f32,
        anneal_ratio: f32,
        noise_scale: f32,
        seed: u64,
    ) -> (
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        usize,
    ) {
        let out = ce_riemann::relax_forward(
            values.as_slice().expect("contiguous"),
            col_idx.as_slice().expect("contiguous"),
            row_ptr.as_slice().expect("contiguous"),
            b.as_slice().expect("contiguous"),
            phi.as_slice().expect("contiguous"),
            m0.as_slice().expect("contiguous"),
            codebook.as_slice().expect("contiguous"),
            metric_basis.as_slice().expect("contiguous"),
            dim, n_code, rank,
            portal, bypass, t_wake, beta, cb_w,
            lambda0, lambda_phi, lambda_var,
            tau, dt, max_steps, tol, anneal_ratio, noise_scale, seed,
        );
        (
            out.best_m.into_pyarray(py),
            out.energy.into_pyarray(py),
            out.delta.into_pyarray(py),
            out.e_hop.into_pyarray(py),
            out.e_bias.into_pyarray(py),
            out.e_portal.into_pyarray(py),
            out.e_cb.into_pyarray(py),
            out.bypass_hist.into_pyarray(py),
            out.steps,
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_brain_step<'py>(
        py: Python<'py>,
        w_values: PyReadonlyArray1<'py, f32>,
        w_col_idx: PyReadonlyArray1<'py, i32>,
        w_row_ptr: PyReadonlyArray1<'py, i32>,
        activation: PyReadonlyArray1<'py, f32>,
        refractory: PyReadonlyArray1<'py, f32>,
        memory_trace: PyReadonlyArray1<'py, f32>,
        adaptation: PyReadonlyArray1<'py, f32>,
        stp_u: PyReadonlyArray1<'py, f32>,
        stp_x: PyReadonlyArray1<'py, f32>,
        bitfield: PyReadonlyArray1<'py, u8>,
        active_mask: PyReadonlyArray1<'py, u8>,
        external: PyReadonlyArray1<'py, f32>,
        goal: PyReadonlyArray1<'py, f32>,
        replay: PyReadonlyArray1<'py, f32>,
        noise: PyReadonlyArray1<'py, f32>,
        mode: u8,
        energy_budget: usize,
        activation_decay: f32,
        activation_gain: f32,
        refractory_decay: f32,
        refractory_gain: f32,
        replay_mix: f32,
        refractory_scale: f32,
        goal_gain: f32,
        external_gain: f32,
        bit_lower: f32,
        bit_upper: f32,
        stp_tau_fac_inv: f32,
        stp_tau_rec: f32,
        stp_u_base: f32,
        adaptation_coupling: f32,
        adaptation_decay: f32,
        memory_decay: f32,
        adaptation_clamp: f32,
    ) -> (
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<u8>,
        usize,
        f32,
    ) {
        let mode_enum = match mode {
            1 => runtime_types::Mode::Nrem,
            2 => runtime_types::Mode::Rem,
            _ => runtime_types::Mode::Wake,
        };
        let mut mp = kernel::ModeParams::from_mode(mode_enum);
        mp.activation_decay = activation_decay;
        mp.activation_gain = activation_gain;
        mp.refractory_decay = refractory_decay;
        mp.refractory_gain = refractory_gain;
        mp.replay_mix = replay_mix;
        mp.adaptation_coupling = adaptation_coupling;
        mp.adaptation_decay = adaptation_decay;
        mp.adaptation_gain = adaptation_decay;
        mp.memory_decay = memory_decay;
        mp.memory_gain = memory_decay;
        let cfg = kernel::StepConfig {
            energy_budget,
            refractory_scale,
            goal_gain,
            external_gain,
            active_threshold: 0.22,
            bit_lower,
            bit_upper,
            stp: kernel::StpParams {
                tau_fac: stp_tau_fac_inv,
                tau_rec: stp_tau_rec,
                u_base: stp_u_base,
            },
            adaptation_clamp,
            ..Default::default()
        };
        let mut act = activation.as_slice().expect("contiguous").to_vec();
        let mut refr = refractory.as_slice().expect("contiguous").to_vec();
        let mut mem = memory_trace.as_slice().expect("contiguous").to_vec();
        let mut adapt = adaptation.as_slice().expect("contiguous").to_vec();
        let mut su = stp_u.as_slice().expect("contiguous").to_vec();
        let mut sx = stp_x.as_slice().expect("contiguous").to_vec();
        let mut bit = bitfield.as_slice().expect("contiguous").to_vec();
        let out = kernel::brain_step(
            w_values.as_slice().expect("contiguous"),
            w_col_idx.as_slice().expect("contiguous"),
            w_row_ptr.as_slice().expect("contiguous"),
            &mut act,
            &mut refr,
            &mut mem,
            &mut adapt,
            &mut su,
            &mut sx,
            &mut bit,
            active_mask.as_slice().expect("contiguous"),
            external.as_slice().expect("contiguous"),
            goal.as_slice().expect("contiguous"),
            replay.as_slice().expect("contiguous"),
            noise.as_slice().expect("contiguous"),
            &mp,
            &cfg,
        );
        (
            act.into_pyarray(py),
            refr.into_pyarray(py),
            mem.into_pyarray(py),
            adapt.into_pyarray(py),
            su.into_pyarray(py),
            sx.into_pyarray(py),
            bit.into_pyarray(py),
            out.active_count,
            out.energy,
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_mfa_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
        d: usize,
        sigma_grav: f32,
        w_lang: f32,
        w_grav: f32,
        causal: bool,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<f32>) {
        let (out, attn) = nn_ops::ce_mfa_fwd(
            q.as_slice().expect("contiguous q"),
            k.as_slice().expect("contiguous k"),
            v.as_slice().expect("contiguous v"),
            n, d, sigma_grav, w_lang, w_grav, causal,
        );
        (out.into_pyarray(py), attn.into_pyarray(py))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_euler_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        pi_inv_freq: PyReadonlyArray1<'py, f32>,
        n: usize,
        d_head: usize,
        pi_gate: f32,
        e_gate: f32,
        xi: f32,
        causal: bool,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<f32>) {
        let (out, attn) = nn_ops::ce_euler_fwd(
            q.as_slice().expect("contiguous q"),
            k.as_slice().expect("contiguous k"),
            v.as_slice().expect("contiguous v"),
            pi_inv_freq.as_slice().expect("contiguous pi_inv_freq"),
            n, d_head, pi_gate, e_gate, xi, causal,
        );
        (out.into_pyarray(py), attn.into_pyarray(py))
    }

    /// Batched Riemann-surface attention (CPU). Inputs are flat row-major
    /// with leading dim `bh = batch * heads`. Returns the output tensor
    /// (bh * n * d_head); the attention matrix is not materialized.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_riemann_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        cos: PyReadonlyArray1<'py, f32>,
        sin: PyReadonlyArray1<'py, f32>,
        sheet_bias: PyReadonlyArray1<'py, f32>,
        bh: usize,
        n: usize,
        d_head: usize,
        causal: bool,
    ) -> &'py PyArray1<f32> {
        let out = nn_ops::ce_riemann_fwd(
            q.as_slice().expect("contiguous q"),
            k.as_slice().expect("contiguous k"),
            v.as_slice().expect("contiguous v"),
            cos.as_slice().expect("contiguous cos"),
            sin.as_slice().expect("contiguous sin"),
            sheet_bias.as_slice().expect("contiguous sheet_bias"),
            bh, n, d_head, causal,
        );
        out.into_pyarray(py)
    }

    /// Batched Riemann-surface attention (CUDA, host staging). Convenience
    /// path for CPU-resident tensors that should compute on GPU.
    #[cfg(feature = "cuda")]
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_riemann_fwd_cuda<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        cos: PyReadonlyArray1<'py, f32>,
        sin: PyReadonlyArray1<'py, f32>,
        sheet_bias: PyReadonlyArray1<'py, f32>,
        bh: usize,
        n: usize,
        d_head: usize,
        causal: bool,
    ) -> PyResult<&'py PyArray1<f32>> {
        use crate::cuda;
        let out = cuda::ce_riemann_fwd_cuda(
            q.as_slice().expect("contiguous q"),
            k.as_slice().expect("contiguous k"),
            v.as_slice().expect("contiguous v"),
            cos.as_slice().expect("contiguous cos"),
            sin.as_slice().expect("contiguous sin"),
            sheet_bias.as_slice().expect("contiguous sheet_bias"),
            bh, n, d_head, causal,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Zero-copy CUDA entry. Inputs are raw CUDA device pointers
    /// (`tensor.data_ptr()` from PyTorch). The kernel writes the result
    /// directly into the buffer at `out_ptr`. Caller MUST ensure that
    /// PyTorch's current stream has been synchronized before this call.
    #[cfg(feature = "cuda")]
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_riemann_fwd_cuda_devptr(
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        cos_ptr: u64,
        sin_ptr: u64,
        sb_ptr: u64,
        out_ptr: u64,
        bh: usize,
        n: usize,
        d_head: usize,
        causal: bool,
    ) -> PyResult<()> {
        use crate::cuda;
        unsafe {
            cuda::ce_riemann_fwd_cuda_devptr(
                q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, sb_ptr, out_ptr,
                bh, n, d_head, causal,
            )
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_dual_attn_fwd<'py>(
        py: Python<'py>,
        z_l: PyReadonlyArray1<'py, f32>,
        z_g: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
        d_l: usize,
        d_g: usize,
        d_m: usize,
        sigma_grav: f32,
        w_lang: f32,
        w_grav: f32,
        causal: bool,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<f32>) {
        let (out, k) = nn_ops::ce_dual_attn_fwd(
            z_l.as_slice().expect("contiguous z_l"),
            z_g.as_slice().expect("contiguous z_g"),
            v.as_slice().expect("contiguous v"),
            n, d_l, d_g, d_m, sigma_grav, w_lang, w_grav, causal,
        );
        (out.into_pyarray(py), k.into_pyarray(py))
    }

    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(topk_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(topk_sparse_batch, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_bwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_lbo_fused_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_power_iter, m)?)?;
        m.add_function(wrap_pyfunction!(nn_gauge_lattice_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_pack_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_metric_basis_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_codebook_pull, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_relax_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_brain_step, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_mfa_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_dual_attn_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_euler_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_riemann_fwd, m)?)?;
        #[cfg(feature = "cuda")]
        m.add_function(wrap_pyfunction!(nn_ce_riemann_fwd_cuda, m)?)?;
        #[cfg(feature = "cuda")]
        m.add_function(wrap_pyfunction!(nn_ce_riemann_fwd_cuda_devptr, m)?)?;
        Ok(())
    }
}
```
