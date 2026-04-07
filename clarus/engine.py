"""Rust-backed CE engine with pure-Python fallback.

Usage:
    from clarus.engine import get_constants, get_brain_engine, topk

If Rust backend (maturin build) is available, uses native code.
Otherwise falls back to pure Python implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from clarus._rust import (
        BrainEngine as _RustBrain,
        CeConstants as _RustConst,
        topk_sparse as _rust_topk,
        topk_sparse_batch as _rust_topk_batch,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# -------------------------------------------------------------------
# Constants fallback
# -------------------------------------------------------------------
@dataclass
class _PyConstants:
    """Pure-Python CE constants (subset, matching Rust CeConstants)."""
    alpha_s: float = 0.11789
    alpha_w: float = 0.03352
    alpha_em_mz: float = 0.00775
    sin2_theta_w: float = 0.23122
    alpha_inv_0: float = 137.036
    delta: float = 0.17776
    d_eff: float = 3.17776
    epsilon2: float = 0.0487
    omega_b: float = 0.0487
    omega_lambda: float = 0.6847
    omega_dm: float = 0.2589
    f_factor: float = 1.3748
    m_h_gev: float = 125.35


def get_constants():
    """Return CeConstants (Rust-derived if available)."""
    if _HAS_RUST:
        return _RustConst()
    return _PyConstants()


# -------------------------------------------------------------------
# BrainEngine
# -------------------------------------------------------------------
def get_brain_engine(field_size: int = 128):
    """Return BrainEngine (Rust-native if available, else raises)."""
    if _HAS_RUST:
        return _RustBrain(field_size)
    raise ImportError(
        "Rust backend not built. Run: uv run maturin develop --release"
    )


# -------------------------------------------------------------------
# TopK sparse
# -------------------------------------------------------------------
def topk(data: list[float], ratio: float) -> tuple[list[float], int]:
    """TopK sparse activation. Rust kernel if available."""
    if _HAS_RUST:
        return _rust_topk(data, ratio)
    n = len(data)
    k = max(1, math.ceil(ratio * n))
    if k >= n:
        return list(data), n
    indexed = sorted(range(n), key=lambda i: abs(data[i]), reverse=True)
    out = [0.0] * n
    for i in indexed[:k]:
        out[i] = data[i]
    return out, k


def topk_batch(data: list[float], row_len: int, ratio: float) -> list[float]:
    """Batch TopK over flattened rows. Rust kernel if available."""
    if _HAS_RUST:
        return _rust_topk_batch(data, row_len, ratio)
    n_rows = len(data) // row_len
    out = [0.0] * len(data)
    k = max(1, math.ceil(ratio * row_len))
    for r in range(n_rows):
        base = r * row_len
        row = data[base:base + row_len]
        indexed = sorted(range(row_len), key=lambda i: abs(row[i]), reverse=True)
        for i in indexed[:k]:
            out[base + i] = row[i]
    return out


def backend_info() -> str:
    """Return current backend status."""
    parts = [f"Rust={'yes' if _HAS_RUST else 'no'}"]
    try:
        from clarus.ops import has_cuda
        parts.append(f"CUDA_kernels={'yes' if has_cuda() else 'no'}")
    except ImportError:
        parts.append("CUDA_kernels=no")
    return ", ".join(parts)
