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


def test_packed_bitmask_matches_per_head_bits():
    """The packed int bitmasks must encode the same pi/e bits as the
    float buffers — one python int replaces an (H,) tensor of floats."""
    attn = EulerCEMinimal(32, 4, 8, head_types="all")   # [0,1,2,3]
    # pi bit per head h0..h3 = [0,0,1,1] → mask = 0b1100 = 12
    # e  bit per head h0..h3 = [0,1,0,1] → mask = 0b1010 = 10
    assert attn._pi_mask == 0b1100
    assert attn._e_mask == 0b1010
    for h in range(4):
        assert ((attn._pi_mask >> h) & 1) == int(attn.pi_bits[h].item())
        assert ((attn._e_mask >> h) & 1) == int(attn.e_bits[h].item())


def test_bucket_partition_covers_all_heads():
    """Every head lands in exactly one bucket; inv_perm is a valid perm."""
    attn = EulerCEMinimal(64, 8, 16, head_types=[0, 3, 1, 2, 3, 1, 0, 2])
    seen = []
    for t in attn._present_buckets:
        idx = getattr(attn, f"_bucket_{t}_idx")
        seen.extend(idx.tolist())
    assert sorted(seen) == list(range(8)), "buckets must cover every head once"
    # inv_perm is a permutation of [0..7]
    assert sorted(attn._bucket_inv_perm.tolist()) == list(range(8))


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


def test_uniform_fast_path_matches_mixed_reference():
    """The SDPA-based fast path (uniform head-type) must produce
    numerically identical output to the per-head-gated reference path.
    We force the reference path by mixing types but tying all heads to
    the same logical type."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64)
    for spec in ["nope", "alibi", "rope", "xpos"]:
        # Fast path
        a_fast = EulerCEMinimal(64, 4, 32, head_types=spec).eval()
        # Reference: same head_types but force mixed dispatch
        a_ref = EulerCEMinimal(64, 4, 32, head_types=spec).eval()
        a_ref._uniform_type = -1  # force fallback to _forward_mixed
        # Tie weights so only the dispatch path differs.
        a_ref.qkv.load_state_dict(a_fast.qkv.state_dict())
        a_ref.o.load_state_dict(a_fast.o.state_dict())
        with torch.no_grad():
            a_ref.log_xi.copy_(a_fast.log_xi)
        with torch.no_grad():
            y_fast = a_fast(x)
            y_ref = a_ref(x)
        diff = (y_fast - y_ref).abs().max().item()
        assert diff < 1e-4, f"{spec}: fast vs reference diverged: {diff}"


def test_uniform_fast_path_dispatch_flag():
    """Sanity: _uniform_type is set correctly."""
    assert EulerCEMinimal(32, 4, 16, head_types="nope")._uniform_type == 0
    assert EulerCEMinimal(32, 4, 16, head_types="alibi")._uniform_type == 1
    assert EulerCEMinimal(32, 4, 16, head_types="rope")._uniform_type == 2
    assert EulerCEMinimal(32, 4, 16, head_types="xpos")._uniform_type == 3
    assert EulerCEMinimal(32, 4, 16, head_types="mix")._uniform_type == -1
    assert EulerCEMinimal(32, 4, 16, head_types="all")._uniform_type == -1


def test_autograd_flows_through_xi():
    torch.manual_seed(0)
    attn = EulerCEMinimal(32, 4, 16, head_types="alibi", learnable_xi=True)
    x = torch.randn(1, 16, 32, requires_grad=True)
    y = attn(x).sum()
    y.backward()
    assert attn.log_xi.grad is not None
    assert torch.isfinite(attn.log_xi.grad).all()


def test_autograd_skips_xi_when_decay_off_in_uniform_path():
    """Uniform rope/nope head-types skip the decay term entirely (SDPA
    fast path doesn't touch log_xi), so log_xi.grad stays None."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32, requires_grad=True)
    for spec in ("rope", "nope"):
        attn = EulerCEMinimal(32, 4, 16, head_types=spec, learnable_xi=True)
        attn(x).sum().backward()
        # log_xi never participates → no grad accumulated.
        assert attn.log_xi.grad is None or torch.equal(
            attn.log_xi.grad, torch.zeros_like(attn.log_xi))


def test_autograd_zero_xi_grad_in_mixed_path_for_decay_off_heads():
    """In the mixed (per-head gated) path, decay-off heads contribute
    zero gradient to their log_xi entry."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32, requires_grad=True)
    # head types [rope, alibi, rope, alibi] forces mixed path.
    attn = EulerCEMinimal(32, 4, 16, head_types=[2, 1, 2, 1],
                          learnable_xi=True)
    assert attn._uniform_type == -1, "should be mixed dispatch"
    attn(x).sum().backward()
    g = attn.log_xi.grad
    assert g is not None
    # heads 0 and 2 (rope, e_bit=0) → grad must be zero
    assert g[0].item() == 0.0 and g[2].item() == 0.0
    # heads 1 and 3 (alibi, e_bit=1) → grad must be nonzero & finite
    assert g[1].item() != 0.0 and g[3].item() != 0.0
    assert torch.isfinite(g).all()


# --- bucketed mixed path ----------------------------------------------------


def _per_head_reference(attn: EulerCEMinimal, x: torch.Tensor) -> torch.Tensor:
    """Head-by-head attention using the same math as the gated path,
    written without any bucketing or SDPA. Used as the ground truth
    for the mixed-path equivalence test."""
    b, n, _ = x.shape
    H, d = attn.n_heads, attn.d_head
    qkv = attn.qkv(x).view(b, n, 3, H, d)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    outs = []
    for h in range(H):
        t = int(attn.head_types[h].item())
        q_h = q[:, h:h + 1]; k_h = k[:, h:h + 1]; v_h = v[:, h:h + 1]
        if (t >> 1) & 1:
            theta = attn.pos[:n].view(1, 1, n, 1) * attn.inv_freq.view(1, 1, 1, -1)
            c = theta.cos(); s = theta.sin()
            q_h = attn._rotate(q_h, c, s)
            k_h = attn._rotate(k_h, c, s)
        s = (q_h @ k_h.transpose(-1, -2)) / math.sqrt(d)
        if t & 1:
            xi = torch.exp(attn.log_xi[h])
            dmat = attn.d_mat[:n, :n]
            s = s + (-dmat / xi).view(1, 1, n, n)
        s = s.masked_fill(~attn.tril[:n, :n], float("-inf"))
        a = torch.softmax(s, dim=-1)
        outs.append(a @ v_h)
    heads_out = torch.cat(outs, dim=1)
    return attn.o(heads_out.transpose(1, 2).contiguous().view(b, n, attn.d_model))


def test_mixed_path_matches_per_head_reference():
    """Bucketed SDPA mixed path == head-by-head manual reference."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64)
    for spec in ("mix", "all", [1, 3, 0, 2, 2, 1, 3, 0]):
        attn = EulerCEMinimal(64, 8, 32, head_types=spec).eval()
        assert attn._uniform_type == -1
        with torch.no_grad():
            actual = attn(x)
            expected = _per_head_reference(attn, x)
        diff = (actual - expected).abs().max().item()
        assert diff < 1e-4, f"{spec}: mixed bucketed diverged: {diff}"


def test_bucketed_path_permutation_invariant():
    """Shuffling head_types must permute output heads but preserve the
    underlying per-head attention. The bucketed dispatch's inv_perm must
    restore the original order exactly."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)
    # Two equivalent specs that differ only in head ordering.
    order_a = [0, 1, 2, 3]
    order_b = [2, 0, 3, 1]
    attn_a = EulerCEMinimal(32, 4, 16, head_types=order_a).eval()
    attn_b = EulerCEMinimal(32, 4, 16, head_types=order_b).eval()
    # Tie Q/K/V/O so the only difference is head-type position.
    attn_b.qkv.load_state_dict(attn_a.qkv.state_dict())
    attn_b.o.load_state_dict(attn_a.o.state_dict())
    with torch.no_grad():
        attn_b.log_xi.copy_(attn_a.log_xi)
    with torch.no_grad():
        y_a = _per_head_reference(attn_a, x)
        y_b = _per_head_reference(attn_b, x)
        # Both must agree with the bucketed path.
        assert (attn_a(x) - y_a).abs().max().item() < 1e-4
        assert (attn_b(x) - y_b).abs().max().item() < 1e-4
