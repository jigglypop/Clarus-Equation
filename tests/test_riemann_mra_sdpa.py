"""Equivalence tests for the SDPA refactor of RiemannRotaryAttention
and MellinRiemannAttention.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from clarus.ce_riemann_attn import RiemannRotaryAttention, RiemannAttnBlock, \
    _build_phase_and_sheet, _rotate_pairs, _sheet_bias
from clarus.ce_mra import MellinRiemannAttention, MRABlock, bootstrap_sparse


def _riemann_reference(attn: RiemannRotaryAttention, x: torch.Tensor) -> torch.Tensor:
    b, n, _ = x.shape
    qkv = attn.qkv(x).view(b, n, 3, attn.n_heads, attn.d_head)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    theta, sheet = _build_phase_and_sheet(attn.pos[:n], attn.gamma, attn.log_scale)
    cos = theta.cos(); sin = theta.sin()
    lambda_sigma = torch.exp(attn.log_lambda_sigma)
    sb = _sheet_bias(sheet, lambda_sigma)
    q_rot = _rotate_pairs(q, cos, sin)
    k_rot = _rotate_pairs(k, cos, sin)
    scores = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(attn.d_head)
    scores = scores + sb
    scores = scores.masked_fill(~attn.tril[:n, :n], float("-inf"))
    a = torch.softmax(scores, dim=-1)
    out = (a @ v).transpose(1, 2).contiguous().view(b, n, attn.d_model)
    return attn.o(out)


def test_riemann_rotary_sdpa_matches_manual_short():
    torch.manual_seed(0)
    for block in (16, 64, 256, 512):
        attn = RiemannRotaryAttention(64, 4, block, backend="torch").eval()
        # Lift sheet_bias out of inert-at-init to exercise it.
        with torch.no_grad():
            attn.log_lambda_sigma.fill_(-1.0)
        x = torch.randn(2, block, 64)
        with torch.no_grad():
            actual = attn(x)
            expected = _riemann_reference(attn, x)
        d = (actual - expected).abs().max().item()
        assert d < 1e-4, f"block={block}: diff {d}"


def test_riemann_rotary_sdpa_tiled_long():
    """Above Q_CHUNK_THRESHOLD the tiled sheet-bias path kicks in."""
    torch.manual_seed(0)
    attn = RiemannRotaryAttention(64, 4, 2048, backend="torch").eval()
    with torch.no_grad():
        attn.log_lambda_sigma.fill_(-1.0)
    x = torch.randn(1, 2048, 64)
    with torch.no_grad():
        actual = attn(x)
        expected = _riemann_reference(attn, x)
    d = (actual - expected).abs().max().item()
    assert d < 1e-4, f"tiled diff {d}"


def test_riemann_autograd():
    torch.manual_seed(0)
    attn = RiemannRotaryAttention(32, 4, 32, backend="torch")
    x = torch.randn(1, 32, 32, requires_grad=True)
    attn(x).sum().backward()
    for p in (attn.log_scale, attn.log_lambda_sigma, attn.qkv.weight, attn.o.weight):
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_riemann_block_forward():
    torch.manual_seed(0)
    blk = RiemannAttnBlock(32, 4, 16).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# --- MellinRiemannAttention -------------------------------------------------


def _mra_reference(attn: MellinRiemannAttention, x: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    H, dh, K = attn.n_heads, attn.d_head, attn.K
    q = attn.q(x).view(B, N, H, dh)
    k = attn.k(x).view(B, N, H, dh)
    v = attn.v(x).view(B, N, H, dh)
    q_re = q[..., 0::2]; q_im = q[..., 1::2]
    k_re = k[..., 0::2]; k_im = k[..., 1::2]
    cos_b = attn.cos_p[:N].view(1, N, 1, K); sin_b = attn.sin_p[:N].view(1, N, 1, K)
    qt_re = cos_b * q_re + sin_b * q_im
    qt_im = cos_b * q_im - sin_b * q_re
    kt_re = cos_b * k_re + sin_b * k_im
    kt_im = cos_b * k_im - sin_b * k_re
    w_re = attn.w_re.view(1, 1, 1, K); w_im = attn.w_im.view(1, 1, 1, K)
    qhat_re = w_re * qt_re - w_im * qt_im
    qhat_im = w_re * qt_im + w_im * qt_re
    qhat_re = qhat_re.transpose(1, 2); qhat_im = qhat_im.transpose(1, 2)
    kt_re_t = kt_re.transpose(1, 2); kt_im_t = kt_im.transpose(1, 2)
    scores = qhat_re @ kt_re_t.transpose(-1, -2) + qhat_im @ kt_im_t.transpose(-1, -2)
    scores = scores / math.sqrt(dh)
    if attn.decay_mode == "bias":
        scores = scores + attn.log_decay[:N, :N].view(1, 1, N, N)
    elif attn.decay_mode == "mult":
        scores = scores * torch.exp(attn.log_decay[:N, :N]).view(1, 1, N, N)
    if attn.hermitian:
        scores = 0.5 * (scores + scores.transpose(-1, -2))
    scores = scores.masked_fill(~attn.tril[:N, :N], float("-inf"))
    a = torch.softmax(scores, dim=-1)
    if attn.sparse_eps2 > 0.0:
        a = bootstrap_sparse(a, attn.sparse_eps2)
    v_t = v.transpose(1, 2)
    out = (a @ v_t).transpose(1, 2).contiguous().view(B, N, attn.d_model)
    return attn.o(out)


def test_mra_fast_path_none_decay():
    torch.manual_seed(0)
    for N in (16, 64, 256):
        attn = MellinRiemannAttention(
            64, 4, N, decay_mode="none", hermitian=False, sparse_eps2=0.0
        ).eval()
        x = torch.randn(2, N, 64)
        with torch.no_grad():
            actual = attn(x)
            expected = _mra_reference(attn, x)
        d = (actual - expected).abs().max().item()
        assert d < 1e-4, f"none N={N}: diff {d}"


def test_mra_fast_path_bias_decay_short():
    torch.manual_seed(0)
    for N in (64, 256, 512):
        attn = MellinRiemannAttention(
            64, 4, N, decay_mode="bias", hermitian=False, sparse_eps2=0.0
        ).eval()
        x = torch.randn(2, N, 64)
        with torch.no_grad():
            actual = attn(x)
            expected = _mra_reference(attn, x)
        d = (actual - expected).abs().max().item()
        assert d < 1e-4, f"bias N={N}: diff {d}"


def test_mra_fast_path_bias_decay_tiled():
    """Long context → Q-chunked bias mask."""
    torch.manual_seed(0)
    attn = MellinRiemannAttention(
        64, 4, 2048, decay_mode="bias", hermitian=False, sparse_eps2=0.0
    ).eval()
    x = torch.randn(1, 2048, 64)
    with torch.no_grad():
        actual = attn(x)
        expected = _mra_reference(attn, x)
    d = (actual - expected).abs().max().item()
    assert d < 1e-4, f"tiled diff {d}"


def test_mra_legacy_path_mult_decay_still_works():
    """Mult decay falls through to the legacy manual path."""
    torch.manual_seed(0)
    attn = MellinRiemannAttention(
        32, 4, 16, decay_mode="mult", hermitian=False, sparse_eps2=0.0
    ).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        actual = attn(x)
        expected = _mra_reference(attn, x)
    d = (actual - expected).abs().max().item()
    assert d < 1e-5, f"mult diff {d}"


def test_mra_legacy_path_hermitian_still_works():
    torch.manual_seed(0)
    attn = MellinRiemannAttention(
        32, 4, 16, decay_mode="none", hermitian=True, sparse_eps2=0.0
    ).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        actual = attn(x)
        expected = _mra_reference(attn, x)
    d = (actual - expected).abs().max().item()
    assert d < 1e-5, f"hermitian diff {d}"


def test_mra_legacy_path_sparse_still_works():
    torch.manual_seed(0)
    attn = MellinRiemannAttention(
        32, 4, 16, decay_mode="none", hermitian=False, sparse_eps2=0.05
    ).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        actual = attn(x)
        expected = _mra_reference(attn, x)
    d = (actual - expected).abs().max().item()
    assert d < 1e-5, f"sparse diff {d}"


def test_mra_autograd():
    torch.manual_seed(0)
    attn = MellinRiemannAttention(32, 4, 32, decay_mode="bias")
    x = torch.randn(1, 32, 32, requires_grad=True)
    attn(x).sum().backward()
    for p in (attn.q.weight, attn.k.weight, attn.v.weight, attn.o.weight):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_mra_block_forward():
    torch.manual_seed(0)
    blk = MRABlock(32, 4, 16, decay_mode="bias").eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
