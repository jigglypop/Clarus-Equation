"""Equivalence tests for the SDPA refactor of EulerRotaryAttention
and EulerCEAttention.

Each forward pass is compared against a manual softmax reference that
mirrors the pre-refactor code path. Max abs diff must be within float32
rounding (<1e-4 is ample).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from clarus.ce_euler import (
    EulerRotaryAttention,
    EulerAttnBlock,
    EulerCEAttention,
    EulerCEBlock,
    RecursiveEulerCEBlock,
    _rotate_pairs,
)


def _rotary_reference(attn: EulerRotaryAttention, x: torch.Tensor) -> torch.Tensor:
    b, n, _ = x.shape
    qkv = attn.qkv(x).view(b, n, 3, attn.n_heads, attn.d_head)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    freqs = attn.head_freq_scalars()
    theta = attn.pos[:n].view(1, 1, n, 1) * freqs.view(1, attn.n_heads, 1, 1) \
            * attn.inv_freq.view(1, 1, 1, -1)
    cos = theta.cos(); sin = theta.sin()
    q = _rotate_pairs(q, cos, sin)
    k = _rotate_pairs(k, cos, sin)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(attn.d_head)
    s = s.masked_fill(~attn.tril[:n, :n], float("-inf"))
    a = torch.softmax(s, dim=-1)
    out = (a @ v).transpose(1, 2).contiguous().view(b, n, attn.d_model)
    return attn.o(out)


def _ce_reference(attn: EulerCEAttention, x: torch.Tensor) -> torch.Tensor:
    b, n, _ = x.shape
    H = attn.n_heads
    qkv = attn.qkv(x).view(b, n, 3, H, attn.d_head)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    pi_g = torch.sigmoid(attn.pi_gate_logit)
    e_g = torch.sigmoid(attn.e_gate_logit)
    theta = attn.pos[:n].view(1, 1, n, 1) * attn.pi_inv_freq.view(1, 1, 1, -1)
    theta = theta * pi_g.view(1, H, 1, 1)
    cos = theta.cos(); sin = theta.sin()
    q_rot = attn._rotate(q, cos, sin)
    k_rot = attn._rotate(k, cos, sin)
    s = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(attn.d_head)
    xi = torch.exp(attn.log_xi)
    d_sub = attn.d_mat[:n, :n]
    decay = -d_sub.view(1, 1, n, n) / xi.view(1, H, 1, 1)
    decay = decay * e_g.view(1, H, 1, 1)
    s = s + decay
    s = s.masked_fill(~attn.tril[:n, :n], float("-inf"))
    a = torch.softmax(s, dim=-1)
    out = (a @ v).transpose(1, 2).contiguous().view(b, n, attn.d_model)
    return attn.o(out)


# --- EulerRotaryAttention ---------------------------------------------------


def test_rotary_sdpa_matches_manual_softmax():
    torch.manual_seed(0)
    for block in (16, 64, 256):
        attn = EulerRotaryAttention(64, 4, block).eval()
        x = torch.randn(2, block, 64)
        with torch.no_grad():
            actual = attn(x)
            expected = _rotary_reference(attn, x)
        d = (actual - expected).abs().max().item()
        assert d < 1e-4, f"block={block}: diff {d}"


def test_rotary_autograd_flows_through_bit_logits():
    torch.manual_seed(0)
    attn = EulerRotaryAttention(32, 4, 16, softmax_bitfield=True)
    x = torch.randn(1, 16, 32, requires_grad=True)
    attn(x).sum().backward()
    assert attn.bit_logits.grad is not None
    assert torch.isfinite(attn.bit_logits.grad).all()


def test_rotary_block_forward():
    torch.manual_seed(0)
    blk = EulerAttnBlock(32, 4, 16).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# --- EulerCEAttention -------------------------------------------------------


def test_cea_sdpa_matches_manual_softmax_short():
    """Under Q_CHUNK_THRESHOLD=1024 → helper takes the single-mask path."""
    torch.manual_seed(0)
    for block in (16, 64, 256, 512):
        attn = EulerCEAttention(64, 4, block).eval()
        x = torch.randn(2, block, 64)
        with torch.no_grad():
            actual = attn(x)
            expected = _ce_reference(attn, x)
        d = (actual - expected).abs().max().item()
        assert d < 1e-4, f"block={block}: diff {d}"


def test_cea_sdpa_matches_manual_softmax_long_chunked():
    """Above threshold → helper takes the Q-chunked path. Mathematically
    identical (each row's softmax is independent of other rows)."""
    torch.manual_seed(0)
    attn = EulerCEAttention(64, 4, 2048).eval()
    x = torch.randn(1, 2048, 64)
    with torch.no_grad():
        actual = attn(x)
        expected = _ce_reference(attn, x)
    d = (actual - expected).abs().max().item()
    assert d < 1e-4, f"chunked: diff {d}"


def test_cea_autograd_flows_through_gates_and_xi():
    torch.manual_seed(0)
    attn = EulerCEAttention(32, 4, 32, learnable_gates=True)
    x = torch.randn(1, 32, 32, requires_grad=True)
    attn(x).sum().backward()
    assert attn.pi_gate_logit.grad is not None
    assert attn.e_gate_logit.grad is not None
    assert attn.log_xi.grad is not None
    for g in (attn.pi_gate_logit.grad, attn.e_gate_logit.grad, attn.log_xi.grad):
        assert torch.isfinite(g).all()


def test_cea_block_forward():
    torch.manual_seed(0)
    blk = EulerCEBlock(32, 4, 32).eval()
    x = torch.randn(2, 32, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_recursive_ce_block_forward():
    torch.manual_seed(0)
    blk = RecursiveEulerCEBlock(32, 4, 32, max_iters=3).eval()
    x = torch.randn(2, 32, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
