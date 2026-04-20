"""Numerical parity + throughput: PyTorch DualLaplacianBlock vs Rust kernel.

Verifies the Rust implementation (clarus._rust.nn_ce_dual_attn_fwd)
produces outputs identical to the PyTorch reference
DualLaplacianBlock.forward() to float32 precision on multiple sizes.

Both paths receive the SAME z_lang, z_grav, v tensors — we inspect the
forward numerics only (projection layers are applied in Python).
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

from clarus.ce_laplacian import (
    DualLaplacianBlock,
    _cosine_adjacency,
    _rbf_adjacency,
    _row_stochastic_causal,
)
from clarus.constants import T_WAKE

try:
    import _rust
    RUST_OK = True
except ImportError:
    RUST_OK = False


def pytorch_dual(z_l, z_g, v, sigma_grav, w_lang, w_grav, causal):
    n = z_l.shape[0]
    A_l = _cosine_adjacency(z_l)
    A_g = _rbf_adjacency(z_g, sigma=sigma_grav)
    if causal:
        tril = torch.tril(torch.ones(n, n, dtype=torch.bool))
    else:
        tril = None
    K_l = _row_stochastic_causal(A_l, tril)
    K_g = _row_stochastic_causal(A_g, tril)
    K = w_lang * K_l + w_grav * K_g
    return torch.matmul(K, v), K


def rust_dual(z_l, z_g, v, sigma_grav, w_lang, w_grav, causal):
    n, d_l = z_l.shape
    _, d_g = z_g.shape
    _, d_m = v.shape
    out_np, k_np = _rust.nn_ce_dual_attn_fwd(
        z_l.detach().cpu().numpy().astype(np.float32).ravel(),
        z_g.detach().cpu().numpy().astype(np.float32).ravel(),
        v.detach().cpu().numpy().astype(np.float32).ravel(),
        n, d_l, d_g, d_m, float(sigma_grav), float(w_lang), float(w_grav), causal,
    )
    return (
        torch.from_numpy(out_np).reshape(n, d_m),
        torch.from_numpy(k_np).reshape(n, n),
    )


def bench_call(fn, warmup=5, n_iters=40):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1000.0


def main():
    if not RUST_OK:
        print("Rust module not importable")
        return

    torch.manual_seed(0)
    w_lang = 1.0 - T_WAKE
    w_grav = T_WAKE

    print(f"Rust ok. gate = (lang={w_lang:.4f}, grav={w_grav:.4f})")
    print(f"\n{'n':>4} {'d_l':>4} {'d_g':>4} {'d_m':>4}  "
          f"{'py (ms)':>9} {'rust (ms)':>10}  {'out_err':>10}  {'K_err':>10}  {'rowsum_K':>10}")

    for (n, d_l, d_g, d_m) in [
        (32, 32, 32, 64),
        (64, 48, 48, 96),
        (128, 48, 48, 96),
        (256, 64, 64, 128),
    ]:
        z_l = torch.randn(n, d_l)
        z_g = torch.randn(n, d_g)
        v = torch.randn(n, d_m)
        sigma_grav = math.sqrt(d_g)

        out_py, K_py = pytorch_dual(z_l, z_g, v, sigma_grav, w_lang, w_grav, True)
        out_rs, K_rs = rust_dual(z_l, z_g, v, sigma_grav, w_lang, w_grav, True)

        out_err = (out_py - out_rs).abs().max().item()
        K_err = (K_py - K_rs).abs().max().item()

        # Row-sum check: convex combination of row-stochastic should give row-stochastic
        # (each row of K_rs should sum to 1 where there is at least one allowed predecessor).
        row_sums = K_rs.sum(dim=-1)
        mean_rowsum = row_sums[1:].mean().item()  # skip row 0 (degenerate)

        t_py = bench_call(lambda: pytorch_dual(z_l, z_g, v, sigma_grav, w_lang, w_grav, True))
        t_rs = bench_call(lambda: rust_dual(z_l, z_g, v, sigma_grav, w_lang, w_grav, True))

        print(f"{n:4d} {d_l:4d} {d_g:4d} {d_m:4d}  "
              f"{t_py:9.3f} {t_rs:10.3f}  {out_err:10.2e}  {K_err:10.2e}  {mean_rowsum:10.6f}")


if __name__ == "__main__":
    main()
