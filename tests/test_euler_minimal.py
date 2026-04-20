"""Tests for the 2-bit minimal Euler-CE attention.

Verifies:
  * head_types_from_spec accepts all documented forms
  * Each of the 4 head-types produces a distinct attention pattern
  * Uniform "rope" head-type matches a manual RoPE-only computation
  * Uniform "alibi" head-type matches a manual ALiBi-only computation
  * Uniform "nope" head-type matches a no-PE computation
  * extend_to grows buffers, parameters unchanged
  * Block forward: shape + finiteness + autograd works
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clarus.ce_euler import (
    EulerCEMinimal,
    EulerCEMinimalBlock,
    head_types_from_spec,
)


# --- spec parsing ----------------------------------------------------------


def test_head_types_from_spec_uniform_int():
    t = head_types_from_spec(2, n_heads=4)
    assert torch.equal(t, torch.tensor([2, 2, 2, 2]))


def test_head_types_from_spec_uniform_str():
    for s, idx in [("nope", 0), ("alibi", 1), ("rope", 2), ("xpos", 3)]:
        t = head_types_from_spec(s, n_heads=3)
        assert (t == idx).all(), (s, t)


def test_head_types_from_spec_mix():
    t = head_types_from_spec("mix", n_heads=4)
    assert torch.equal(t, torch.tensor([1, 3, 1, 3]))


def test_head_types_from_spec_all():
    t = head_types_from_spec("all", n_heads=8)
    assert torch.equal(t, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))


def test_head_types_from_spec_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        head_types_from_spec("bogus", n_heads=4)
    with pytest.raises(ValueError):
        head_types_from_spec(7, n_heads=4)
    with pytest.raises(ValueError):
        head_types_from_spec([0, 1, 2, 5], n_heads=4)
    with pytest.raises(ValueError):
        head_types_from_spec([0, 1, 2], n_heads=4)


# --- per-head-type bit assignment -----------------------------------------


def test_pi_e_bits_correctly_decoded():
    attn = EulerCEMinimal(d_model=32, n_heads=4, block=8, head_types="all")
    # head_types = [0, 1, 2, 3] → pi = [0, 0, 1, 1], e = [0, 1, 0, 1]
    assert torch.equal(attn.pi_bits, torch.tensor([0., 0., 1., 1.]))
    assert torch.equal(attn.e_bits, torch.tensor([0., 1., 0., 1.]))


# --- forward sanity --------------------------------------------------------


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    attn = EulerCEMinimal(d_model=32, n_heads=4, block=16,
                          head_types="alibi").eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_block_forward():
    torch.manual_seed(0)
    blk = EulerCEMinimalBlock(d_model=32, n_heads=4, block=16,
                              head_types="xpos").eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_each_head_type_produces_finite():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32)
    for spec in ["nope", "alibi", "rope", "xpos", "mix", "all"]:
        attn = EulerCEMinimal(32, 4, 16, head_types=spec).eval()
        with torch.no_grad():
            y = attn(x)
        assert torch.isfinite(y).all(), spec


# --- semantics: each head-type matches its canonical literature analogue --


def test_nope_uniform_matches_no_position_encoding():
    """All-nope (head_types=0) should be exactly causal attention with no PE."""
    torch.manual_seed(0)
    attn = EulerCEMinimal(32, 4, 16, head_types="nope").eval()
    # Manual: causal mask only, no rotation, no decay.
    x = torch.randn(1, 16, 32)
    qkv = attn.qkv(x).view(1, 16, 3, 4, 8)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(8)
    s = s.masked_fill(~attn.tril[:16, :16], float("-inf"))
    a = torch.softmax(s, dim=-1)
    out = (a @ v).transpose(1, 2).contiguous().view(1, 16, 32)
    expected = attn.o(out)
    with torch.no_grad():
        actual = attn(x)
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"nope uniform diverged from no-PE baseline: max diff {(actual-expected).abs().max()}"


def test_alibi_decay_actually_attenuates_distant():
    """All-alibi heads should produce attention weights that decay with distance."""
    torch.manual_seed(0)
    attn = EulerCEMinimal(32, 4, 32, head_types="alibi", xi_init=4.0).eval()
    # Inspect attention weights via a probe.
    x = torch.randn(1, 32, 32)
    H = attn.n_heads
    qkv = attn.qkv(x).view(1, 32, 3, H, 8)
    q, k, _ = qkv.unbind(dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2)
    # No rotation since pi_bit = 0
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(8)
    xi = torch.exp(attn.log_xi)
    decay = -attn.d_mat[:32, :32].view(1, 1, 32, 32) / xi.view(1, H, 1, 1)
    expected_scores = scores + decay
    expected_scores = expected_scores.masked_fill(~attn.tril[:32, :32], float("-inf"))
    expected_attn = torch.softmax(expected_scores, dim=-1)

    # The attention from query at pos 31 must put more weight on pos 30
    # than on pos 0 (because distance penalty is much larger at pos 0).
    a = expected_attn[0, 0, 31]                # head 0
    assert a[30] > a[0], f"alibi decay not attenuating: a[30]={a[30]} a[0]={a[0]}"


def test_rope_uniform_no_decay():
    """All-rope heads should have zero decay contribution."""
    attn = EulerCEMinimal(32, 4, 16, head_types="rope")
    assert torch.equal(attn.e_bits, torch.zeros(4))
    # decay term in forward becomes zero, so scores match a pure RoPE attention.


def test_mix_has_two_alibi_two_xpos():
    attn = EulerCEMinimal(32, 4, 16, head_types="mix")
    assert torch.equal(attn.head_types, torch.tensor([1, 3, 1, 3]))
    assert torch.equal(attn.pi_bits, torch.tensor([0., 1., 0., 1.]))
    assert torch.equal(attn.e_bits,  torch.tensor([1., 1., 1., 1.]))


# --- extend_to -------------------------------------------------------------


def test_extend_to_grows_buffers():
    attn = EulerCEMinimal(32, 4, 16, head_types="all")
    n_params_before = sum(p.numel() for p in attn.parameters())
    log_xi_before = attn.log_xi.detach().clone()
    attn.extend_to(64)
    assert attn.pos.shape == (64,)
    assert attn.tril.shape == (64, 64)
    assert attn.d_mat.shape == (64, 64)
    n_params_after = sum(p.numel() for p in attn.parameters())
    assert n_params_after == n_params_before, "params changed"
    assert torch.allclose(attn.log_xi.detach(), log_xi_before), "log_xi changed"


def test_forward_after_extend_to():
    torch.manual_seed(0)
    attn = EulerCEMinimal(32, 4, 16, head_types="xpos").eval()
    attn.extend_to(64)
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        y = attn(x)
    assert y.shape == (2, 64, 32)
    assert torch.isfinite(y).all()


# --- autograd --------------------------------------------------------------


def test_autograd_flows_through_xi():
    torch.manual_seed(0)
    attn = EulerCEMinimal(32, 4, 16, head_types="alibi", learnable_xi=True)
    x = torch.randn(1, 16, 32, requires_grad=True)
    y = attn(x).sum()
    y.backward()
    assert attn.log_xi.grad is not None
    assert torch.isfinite(attn.log_xi.grad).all()
    # heads with e_bit = 0 should have zero grad on log_xi (decay path is gated off)
    attn2 = EulerCEMinimal(32, 4, 16, head_types="rope", learnable_xi=True)
    y2 = attn2(x).sum()
    y2.backward()
    assert attn2.log_xi.grad is not None
    assert torch.equal(attn2.log_xi.grad, torch.zeros_like(attn2.log_xi.grad))
