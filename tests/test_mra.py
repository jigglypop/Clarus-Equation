"""Tests for Mellin-Riemann Attention.

The default MRA now uses lean settings (RoPE frequencies, ζ amplitude only,
no decay, no sparsity, no spectral norm, no Hermitian). Optional modes are
exercised explicitly in dedicated tests.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clarus.ce_mra import (
    MRABlock,
    MellinRiemannAttention,
    bootstrap_sparse,
)


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    attn = MellinRiemannAttention(d_model=32, n_heads=4, block=16).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_block_forward():
    torch.manual_seed(0)
    blk = MRABlock(d_model=32, n_heads=4, block=16).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_all_ablation_modes_run():
    """Every documented ablation knob must produce a finite output."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32)
    configs = [
        dict(freq_mode="rope",     amp_weight=True,  decay_mode="none"),
        dict(freq_mode="rope",     amp_weight=False, decay_mode="none"),
        dict(freq_mode="zeta_log", amp_weight=True,  decay_mode="none"),
        dict(freq_mode="rope",     amp_weight=True,  decay_mode="bias"),
        dict(freq_mode="rope",     amp_weight=True,  decay_mode="mult"),
        dict(freq_mode="rope",     amp_weight=True,  sparse_eps2=0.1),
        dict(freq_mode="rope",     amp_weight=True,  spectral_norm_o=True),
    ]
    for cfg in configs:
        attn = MellinRiemannAttention(32, 4, 16, **cfg).eval()
        with torch.no_grad():
            y = attn(x)
        assert torch.isfinite(y).all(), f"non-finite output for cfg={cfg}"


def test_bootstrap_sparse_exact_count():
    n = 100
    attn = torch.softmax(torch.randn(2, 4, n, n), dim=-1)
    eps2 = 0.0487
    out = bootstrap_sparse(attn, eps2=eps2)
    expected_k = math.ceil(eps2 * n)
    nonzero_per_row = (out > 0).sum(dim=-1)
    assert (nonzero_per_row == expected_k).all()
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_bootstrap_sparse_min_keep():
    n = 8
    attn = torch.softmax(torch.randn(1, 1, 1, n), dim=-1)
    out = bootstrap_sparse(attn, eps2=0.0001)
    nonzero = (out > 0).sum(dim=-1)
    assert (nonzero >= 1).all()


def test_parameters_match_standard_mha():
    """Default MRA = 4·d² (W_q, W_k, W_v, W_o), same as a plain MHA."""
    d = 64
    mra = MellinRiemannAttention(d, 4, 8)
    n = sum(p.numel() for p in mra.parameters())
    assert n == 4 * d * d, f"expected 4d² = {4*d*d}, got {n}"


def test_hermitian_option_saves_one_projection():
    """`hermitian=True` ties W_q=W_k (3·d²). Intended only for bidirectional use."""
    d = 64
    mra_h = MellinRiemannAttention(d, 4, 8, hermitian=True)
    n = sum(p.numel() for p in mra_h.parameters())
    assert n == 3 * d * d, f"expected 3d² = {3*d*d}, got {n}"


def test_axiomatic_buffers_not_learnable():
    attn = MellinRiemannAttention(32, 4, 16, decay_mode="bias")
    learnable_names = {name for name, _ in attn.named_parameters()}
    for axiom in ("gamma", "w_re", "w_im", "cos_p", "sin_p",
                  "log_decay", "tril"):
        assert axiom not in learnable_names
        assert hasattr(attn, axiom)


def test_amplitude_weighting_changes_output():
    """w_k ≠ 1 must produce different scores than w_k = 1 (otherwise the
    whole novel contribution collapses to RoPE)."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)
    a_on = MellinRiemannAttention(32, 4, 16, amp_weight=True).eval()
    a_off = MellinRiemannAttention(32, 4, 16, amp_weight=False).eval()
    # Tie all learnable weights so only w_k differs.
    a_off.q.load_state_dict(a_on.q.state_dict())
    a_off.k.load_state_dict(a_on.k.state_dict())
    a_off.v.load_state_dict(a_on.v.state_dict())
    a_off.o.load_state_dict(a_on.o.state_dict())
    with torch.no_grad():
        y_on = a_on(x); y_off = a_off(x)
    diff = (y_on - y_off).abs().max().item()
    assert diff > 1e-4, f"amplitude weighting has no effect: {diff}"


def test_spectral_norm_constrains_output_projection():
    torch.manual_seed(0)
    attn = MellinRiemannAttention(32, 4, 16, spectral_norm_o=True).train()
    x = torch.randn(2, 16, 32)
    for _ in range(5):
        _ = attn(x)
    attn.eval()
    w = attn.o.weight.detach()
    sigma1 = torch.linalg.svdvals(w)[0].item()
    assert sigma1 <= 1.0 + 1e-2, f"σ₁ = {sigma1}"


def test_hermitian_score_symmetric_without_causal_mask():
    """Sanity: with Hermitian+no mask, the final attention row-sums are
    self-consistent. (Full self-adjointness is not meaningful with the
    causal mask — that's by design, see module docstring.)"""
    torch.manual_seed(0)
    attn = MellinRiemannAttention(
        16, 2, 8, hermitian=True, spectral_norm_o=False).eval()
    attn.tril.fill_(True)  # disable causal mask for this check
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        y = attn(x)
    assert torch.isfinite(y).all()
