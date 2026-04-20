"""Parity + throughput: PyTorch EulerCEAttention vs Rust nn_ce_euler_fwd.

Verifies the Rust fused kernel produces outputs identical (float32
precision) to the PyTorch reference EulerCEAttention forward, across
sizes, for a single head with hardcoded gate/xi scalars.
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

from clarus.ce_euler import EulerCEAttention

try:
    import _rust
    RUST_OK = True
except ImportError:
    RUST_OK = False


def pytorch_euler_single_head(q, k, v, pi_inv_freq, pi_gate, e_gate, xi):
    """Reproduces EulerCEAttention.forward for one head, returning (out, attn)."""
    n, d_head = q.shape
    pos = torch.arange(n, dtype=torch.float32)
    theta = pos.view(n, 1) * pi_inv_freq.view(1, -1) * pi_gate
    cos = theta.cos(); sin = theta.sin()

    def rotate(x):
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    qr = rotate(q); kr = rotate(k)
    scores = (qr @ kr.transpose(-1, -2)) / math.sqrt(d_head)

    d_mat = (torch.arange(n).unsqueeze(1) - torch.arange(n).unsqueeze(0)).abs().float()
    scores = scores - e_gate * d_mat / xi

    # causal
    tril = torch.tril(torch.ones(n, n, dtype=torch.bool))
    scores = scores.masked_fill(~tril, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn


def rust_euler(q, k, v, pi_inv_freq, pi_gate, e_gate, xi):
    n, d_head = q.shape
    out_np, attn_np = _rust.nn_ce_euler_fwd(
        q.detach().cpu().numpy().astype(np.float32).ravel(),
        k.detach().cpu().numpy().astype(np.float32).ravel(),
        v.detach().cpu().numpy().astype(np.float32).ravel(),
        pi_inv_freq.detach().cpu().numpy().astype(np.float32).ravel(),
        n, d_head, float(pi_gate), float(e_gate), float(xi), True,
    )
    return (
        torch.from_numpy(out_np).reshape(n, d_head),
        torch.from_numpy(attn_np).reshape(n, n),
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

    # Use same scheme as EulerCEAttention: theta_k = pos * pi^{1-k/(d/2)} * pi_gate
    # pi_inv_freq[k] = pi^{1 - k/(d/2)}
    configs = [
        (32, 32),
        (64, 64),
        (128, 64),
        (256, 64),
        (512, 64),
    ]

    pi_gate = 0.88  # sigmoid(2.0)
    e_gate = 0.88
    xi = 16.0

    print(f"Rust ok. pi_gate={pi_gate}, e_gate={e_gate}, xi={xi}")
    print(f"\n{'n':>4} {'d':>4}  {'py (ms)':>9} {'rust (ms)':>10}  "
          f"{'out_err':>11}  {'attn_err':>11}  {'rowsum':>8}  {'speedup':>7}")

    for n, d_head in configs:
        q = torch.randn(n, d_head)
        k = torch.randn(n, d_head)
        v = torch.randn(n, d_head)
        kk = torch.arange(0, d_head, 2, dtype=torch.float32) / d_head
        pi_inv_freq = torch.tensor([math.pi ** (1.0 - k_val.item()) for k_val in kk])

        out_py, attn_py = pytorch_euler_single_head(q, k, v, pi_inv_freq,
                                                    pi_gate, e_gate, xi)
        out_rs, attn_rs = rust_euler(q, k, v, pi_inv_freq, pi_gate, e_gate, xi)

        out_err = (out_py - out_rs).abs().max().item()
        attn_err = (attn_py - attn_rs).abs().max().item()
        rowsum = float(attn_rs.sum(-1).mean().item())

        t_py = bench_call(
            lambda: pytorch_euler_single_head(q, k, v, pi_inv_freq,
                                              pi_gate, e_gate, xi)
        )
        t_rs = bench_call(
            lambda: rust_euler(q, k, v, pi_inv_freq, pi_gate, e_gate, xi)
        )
        speedup = t_py / t_rs if t_rs > 0 else float('inf')

        print(f"{n:4d} {d_head:4d}  {t_py:9.3f} {t_rs:10.3f}  "
              f"{out_err:11.2e}  {attn_err:11.2e}  {rowsum:8.4f}  {speedup:7.2f}x")


if __name__ == "__main__":
    main()
