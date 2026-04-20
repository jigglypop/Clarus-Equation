"""Numerical-consistency tests for Riemann surface positional encoding.

Verifies the three backends (PyTorch reference, Rust CPU, CUDA) produce
matching outputs for `RiemannRotaryAttention`, and that the FFN
Riemann-zero init gives a finite, GUE-spaced weight tensor.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from clarus.ce_riemann_attn import (
    RiemannRotaryAttention,
    has_cuda_riemann,
    has_rust_riemann,
    riemann_zero_init,
    riemann_zeros,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_attn(D=32, H=4, N=16, backend="torch"):
    return RiemannRotaryAttention(D, H, N, backend=backend).eval()


def test_riemann_zeros_first_value():
    g = riemann_zeros(5)
    assert g.shape == (5,)
    assert math.isclose(float(g[0]), 14.134725142, rel_tol=1e-7)


def test_riemann_zeros_extrapolation():
    g = riemann_zeros(150)
    assert g.shape == (150,)
    assert torch.all(g[1:] > g[:-1])  # monotone


def test_torch_forward_shapes_and_finite():
    attn = _make_attn()
    x = torch.randn(2, 16, 32)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not has_rust_riemann(), reason="Rust backend unavailable")
def test_torch_vs_rust_consistency():
    attn = _make_attn(backend="torch")
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_torch = attn(x)
    attn.backend_pref = "rust"
    with torch.no_grad():
        y_rust = attn(x)
    diff = (y_torch - y_rust).abs().max().item()
    assert diff < 1e-4, f"torch vs rust diverged: {diff}"


@pytest.mark.skipif(not has_cuda_riemann(), reason="CUDA backend unavailable")
def test_torch_vs_cuda_consistency():
    attn = _make_attn(backend="torch")
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_torch = attn(x)
    attn.backend_pref = "cuda"
    with torch.no_grad():
        y_cuda = attn(x)
    diff = (y_torch - y_cuda).abs().max().item()
    assert diff < 1e-3, f"torch vs cuda diverged: {diff}"


def test_sheet_bias_inactive_at_init_gives_pure_rope_on_log_time():
    """At init log_lambda_sigma=-6 → λ_σ ≈ 2.5e-3 → sheet bias near zero.
    Output should be very close to torch backend with sheet bias forced to 0."""
    attn = _make_attn()
    x = torch.randn(1, 12, 32)
    with torch.no_grad():
        y_default = attn(x)
        attn.log_lambda_sigma.data.fill_(-1e6)  # force λ_σ ≈ 0
        y_no_sheet = attn(x)
    diff = (y_default - y_no_sheet).abs().max().item()
    # init sheet contribution should be small
    assert diff < 0.05


def test_sheet_bias_changes_output_when_active():
    """Sheet bias only fires when phase exceeds 2π — needs long-enough
    sequence (or large frequency scale). Use N=128 to ensure σ varies."""
    attn = _make_attn(D=32, H=4, N=128)
    x = torch.randn(1, 128, 32)
    with torch.no_grad():
        attn.log_lambda_sigma.data.fill_(-1e6)
        y_off = attn(x)
        attn.log_lambda_sigma.data.fill_(2.0)
        y_on = attn(x)
    diff = (y_on - y_off).abs().max().item()
    assert diff > 1e-2, f"sheet bias should perturb output, got {diff}"


def test_riemann_zero_init_finite_and_modulated():
    lin = nn.Linear(32, 64, bias=False)
    riemann_zero_init(lin, axis="in")
    assert torch.isfinite(lin.weight).all()
    # column norms shouldn't all be equal — Riemann modulation breaks isotropy
    col_norms = lin.weight.norm(dim=0)
    cv = col_norms.std() / col_norms.mean().clamp_min(1e-8)
    assert cv > 0.05
