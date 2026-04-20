"""Rust vs PyTorch throughput for CE MFA logit-space kernel.

Confirms that the pure-Python MFA overhead (measured earlier as
~14x vs std scaled-dot-product on CPU) collapses when routed
through the fused Rust kernel.
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn.functional as F

from clarus.ce_softmax import grav_scores, lang_scores, mode_gate

try:
    import _rust  # installed at top-level by maturin build
except ImportError:
    _rust = None


def torch_std_attn(q, k, v):
    d = q.shape[-1]
    s = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    return F.softmax(s, dim=-1) @ v


def torch_mfa_logit(q, k, v, w_lang, w_grav, sigma):
    s_lang = lang_scores(q.unsqueeze(0), k.unsqueeze(0)).squeeze(0)
    s_grav = grav_scores(k.unsqueeze(0), sigma=sigma).squeeze(0)
    s = w_lang * s_lang + w_grav * s_grav
    return F.softmax(s, dim=-1) @ v


def rust_mfa(q, k, v, n, d, sigma, w_lang, w_grav):
    out_np, _ = _rust.nn_ce_mfa_fwd(
        q.numpy().astype(np.float32).ravel(),
        k.numpy().astype(np.float32).ravel(),
        v.numpy().astype(np.float32).ravel(),
        n, d, float(sigma), float(w_lang), float(w_grav), False,
    )
    return torch.from_numpy(out_np).reshape(n, d)


def bench(fn, n_iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1000.0  # ms


def main():
    torch.manual_seed(0)

    configs = [
        (64, 64),
        (128, 64),
        (256, 64),
        (256, 128),
        (512, 128),
    ]

    gate = mode_gate("wake")
    w_lang, w_grav = gate.omega_lang, gate.omega_grav

    print(f"Rust available: {_rust is not None}")
    print(f"\n{'n':>4} {'d':>4}  {'std (ms)':>10}  {'py-mfa':>10}  {'rust-mfa':>10}  "
          f"{'py/std':>8}  {'rust/std':>8}")

    for n, d in configs:
        q = torch.randn(n, d)
        k = torch.randn(n, d)
        v = torch.randn(n, d)
        sigma = math.sqrt(d)

        t_std = bench(lambda: torch_std_attn(q, k, v))
        t_py = bench(lambda: torch_mfa_logit(q, k, v, w_lang, w_grav, sigma))
        if _rust is not None:
            t_rust = bench(lambda: rust_mfa(q, k, v, n, d, sigma, w_lang, w_grav))
        else:
            t_rust = float("nan")

        # correctness check: rust output vs pytorch mfa
        if _rust is not None:
            out_rust = rust_mfa(q, k, v, n, d, sigma, w_lang, w_grav)
            out_py = torch_mfa_logit(q, k, v, w_lang, w_grav, sigma)
            err = (out_rust - out_py).abs().max().item()
        else:
            err = float("nan")

        print(f"{n:4d} {d:4d}  {t_std:10.3f}  {t_py:10.3f}  {t_rust:10.3f}  "
              f"{t_py/t_std:8.2f}  {t_rust/t_std:8.2f}   max_err={err:.2e}")


if __name__ == "__main__":
    main()
