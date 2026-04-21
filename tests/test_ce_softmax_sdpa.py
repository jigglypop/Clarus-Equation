"""Equivalence test for CESoftmaxAttention SDPA refactor."""
import math
import torch
import torch.nn.functional as F
from clarus.ce_softmax import (
    CESoftmaxAttention, metric_family_attention,
    lang_scores, grav_scores, mode_gate,
)


def _logit_reference(q, k, v, z, gate, sigma, mask):
    s_l = lang_scores(q, k)
    s_g = grav_scores(z, sigma=sigma)
    s = gate.omega_lang * s_l + gate.omega_grav * s_g
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    a = F.softmax(s, dim=-1)
    return torch.matmul(a, v)


def test_logit_matches_manual():
    torch.manual_seed(0)
    for mode in ("wake", "nrem", "rem"):
        gate = mode_gate(mode)
        q = torch.randn(2, 4, 16, 8)
        k = torch.randn(2, 4, 16, 8)
        v = torch.randn(2, 4, 16, 8)
        mask = torch.tril(torch.ones(16, 16, dtype=torch.bool))
        actual = metric_family_attention(q, k, v, z_grav=k, gate=gate,
                                         sigma_grav=1.0, mask=mask, combine="logit")
        expected = _logit_reference(q, k, v, k, gate, 1.0, mask)
        d = (actual - expected).abs().max().item()
        assert d < 1e-4, f"{mode}: diff {d}"


def test_logit_without_mask():
    torch.manual_seed(0)
    gate = mode_gate("wake")
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    actual = metric_family_attention(q, k, v, z_grav=k, gate=gate,
                                     sigma_grav=1.0, mask=None, combine="logit")
    expected = _logit_reference(q, k, v, k, gate, 1.0, None)
    d = (actual - expected).abs().max().item()
    assert d < 1e-4, f"no-mask diff {d}"


def test_module_forward():
    torch.manual_seed(0)
    attn = CESoftmaxAttention(32, 4, combine="logit").eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_convex_unchanged():
    torch.manual_seed(0)
    gate = mode_gate("wake")
    q = torch.randn(1, 2, 8, 4); k = torch.randn(1, 2, 8, 4); v = torch.randn(1, 2, 8, 4)
    y = metric_family_attention(q, k, v, gate=gate, combine="convex")
    assert torch.isfinite(y).all()
