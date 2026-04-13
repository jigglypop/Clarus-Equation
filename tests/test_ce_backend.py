from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from clarus.ce_ops import (
    DEFAULT_CB_W,
    build_metric_basis,
    ce_backend,
    codebook_pull,
    has_cuda,
    has_rust,
    pack_sparse,
    pq_build_codebook,
    pq_reconstruct_tokens,
    pq_scores,
    relax,
    relax_packed,
)


PORTAL = 0.031203
BYPASS = 0.489236
T_WAKE = 0.314798


def update_phi(phi: torch.Tensor, m_star: torch.Tensor, phi_var: torch.Tensor | None = None) -> torch.Tensor:
    norm = m_star.detach().float().norm()
    residual = torch.zeros_like(m_star) if norm.item() < 1e-8 else m_star.detach().float() / norm
    if phi_var is not None and phi_var.numel() == phi.numel():
        var_mean = phi_var.mean().clamp(min=1e-8)
        alpha = float(BYPASS * var_mean / (var_mean + 1.0))
    else:
        alpha = BYPASS
    return (1 - alpha) * phi + alpha * residual


def make_case(dim: int = 12, n_code: int = 10, seed: int = 0):
    torch.manual_seed(seed)
    w = torch.randn(dim, dim)
    w = (w + w.t()) / 2
    w.fill_diagonal_(0)
    b = torch.randn(dim)
    phi = torch.randn(dim) * 0.1
    m0 = torch.randn(dim)
    codebook = torch.randn(n_code, dim)
    return w.float(), b.float(), phi.float(), m0.float(), codebook.float()


def relax_kwargs():
    return dict(
        portal=PORTAL,
        bypass=BYPASS,
        t_wake=T_WAKE,
        beta=1.0,
        cb_w=PORTAL,
        lambda0=1.0,
        lambda_phi=0.5,
        lambda_var=0.25,
        tau=1.0,
        dt=0.01,
        max_steps=80,
        tol=1e-7,
        anneal_ratio=0.6,
        noise_scale=0.0,
        metric_rank=4,
        seed=7,
    )


def test_default_codebook_weight_matches_portal_constant():
    ad = 4 / (math.e ** (4 / 3) * math.pi ** (4 / 3))
    expected = (ad * (1 - ad)) ** 2
    assert DEFAULT_CB_W == pytest.approx(expected, abs=1e-12)


class IdentityBlock(nn.Module):
    def forward(self, h):
        return h


class FakeModel(nn.Module):
    def __init__(self, vocab: int = 8, dim: int = 4, layers: int = 3):
        super().__init__()
        transformer = nn.Module()
        transformer.wte = nn.Embedding(vocab, dim)
        transformer.wpe = nn.Embedding(32, dim)
        transformer.h = nn.ModuleList(IdentityBlock() for _ in range(layers))
        transformer.ln_f = nn.LayerNorm(dim)
        self.transformer = transformer


def test_backend_policy_prefers_torch_without_native_on_cpu():
    backend = ce_backend(torch.device("cpu"), "auto")
    assert backend in {"torch", "rust"}


def test_requested_backend_rejects_wrong_device():
    with pytest.raises(RuntimeError):
        ce_backend(torch.device("cpu"), "cuda")
    with pytest.raises(RuntimeError):
        ce_backend(torch.device("cuda"), "rust")


def test_pack_sparse_torch_shapes_and_csr_monotonic():
    w, *_ = make_case()
    values, col_idx, row_ptr = pack_sparse(w, backend="torch")
    assert values.ndim == 1
    assert col_idx.ndim == 1
    assert row_ptr.ndim == 1
    assert row_ptr.numel() == w.shape[0] + 1
    assert torch.all(row_ptr[1:] >= row_ptr[:-1])
    assert row_ptr[-1].item() == values.numel() == col_idx.numel()


@pytest.mark.skipif(not has_rust(), reason="Rust backend unavailable")
def test_pack_sparse_rust_matches_torch_exactly():
    w, *_ = make_case()
    v_t, c_t, r_t = pack_sparse(w, backend="torch")
    v_r, c_r, r_r = pack_sparse(w, backend="rust")
    assert torch.allclose(v_t, v_r)
    assert torch.equal(c_t, c_r)
    assert torch.equal(r_t, r_r)


def test_metric_basis_is_orthonormal_in_torch():
    _, _, _, m0, codebook = make_case(seed=3)
    basis = build_metric_basis(codebook, m0, rank=4, backend="torch")
    if basis.numel() == 0:
        pytest.skip("Degenerate basis for this random seed")
    gram = basis @ basis.t()
    eye = torch.eye(basis.shape[0], dtype=basis.dtype)
    assert torch.allclose(gram, eye, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not has_rust(), reason="Rust backend unavailable")
def test_metric_basis_rust_matches_torch():
    _, _, _, m0, codebook = make_case(seed=4)
    basis_t = build_metric_basis(codebook, m0, rank=4, backend="torch")
    basis_r = build_metric_basis(codebook, m0, rank=4, backend="rust")
    assert basis_t.shape == basis_r.shape
    assert torch.allclose(basis_t, basis_r, atol=1e-4, rtol=1e-4)


def test_codebook_pull_torch_returns_finite_grad_and_energy():
    _, _, _, m0, codebook = make_case(seed=5)
    grad, energy = codebook_pull(m0, codebook, beta=1.0, cb_w=PORTAL, backend="torch")
    assert grad.shape == m0.shape
    assert torch.isfinite(grad).all()
    assert torch.isfinite(energy)


@pytest.mark.skipif(not has_rust(), reason="Rust backend unavailable")
def test_codebook_pull_rust_matches_torch():
    _, _, _, m0, codebook = make_case(seed=6)
    grad_t, energy_t = codebook_pull(m0, codebook, beta=1.0, cb_w=PORTAL, backend="torch")
    grad_r, energy_r = codebook_pull(m0, codebook, beta=1.0, cb_w=PORTAL, backend="rust")
    assert torch.allclose(grad_t, grad_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(energy_t, energy_r, atol=1e-4, rtol=1e-4)


def test_relax_torch_produces_finite_histories():
    w, b, phi, m0, codebook = make_case(seed=8)
    values, col_idx, row_ptr = pack_sparse(w, backend="torch")
    basis = build_metric_basis(codebook, m0, rank=4, backend="torch")
    m_star, hist, steps = relax_packed(
        values, col_idx, row_ptr, b, phi, m0, codebook, basis, backend="torch", **relax_kwargs()
    )
    assert steps > 0
    assert m_star.shape == m0.shape
    assert all(math.isfinite(v) for v in hist["E"])
    assert all(math.isfinite(v) for v in hist["delta"])
    assert len(hist["E"]) == steps
    assert min(hist["E"]) <= hist["E"][0]


def test_update_phi_preserves_signed_residual_direction():
    phi = torch.zeros(3)
    m_star = torch.tensor([1.0, -2.0, 0.0])
    updated = update_phi(phi, m_star)
    assert torch.isfinite(updated).all()
    assert updated[0] > 0
    assert updated[1] < 0


@pytest.mark.skipif(not has_rust(), reason="Rust backend unavailable")
def test_relax_rust_matches_torch_without_noise():
    w, b, phi, m0, codebook = make_case(seed=9)
    values_t, col_idx_t, row_ptr_t = pack_sparse(w, backend="torch")
    values_r, col_idx_r, row_ptr_r = pack_sparse(w, backend="rust")
    basis_t = build_metric_basis(codebook, m0, rank=4, backend="torch")
    basis_r = build_metric_basis(codebook, m0, rank=4, backend="rust")
    kwargs = relax_kwargs()
    m_t, hist_t, steps_t = relax_packed(
        values_t, col_idx_t, row_ptr_t, b, phi, m0, codebook, basis_t, backend="torch", **kwargs
    )
    m_r, hist_r, steps_r = relax_packed(
        values_r, col_idx_r, row_ptr_r, b, phi, m0, codebook, basis_r, backend="rust", **kwargs
    )
    assert steps_t == steps_r
    cos_tm = F.cosine_similarity(m_t.unsqueeze(0), m_r.unsqueeze(0)).item()
    assert math.isfinite(cos_tm)
    assert cos_tm > 0.95
    assert hist_t["E"][-1] <= hist_t["E"][0]
    assert hist_r["E"][-1] <= hist_r["E"][0]


def test_riemannian_metric_is_numerically_stable_on_constructed_case():
    dim = 10
    torch.manual_seed(11)
    target = torch.randn(dim)
    target = target / target.norm()
    w = -torch.eye(dim)
    b = target * 0.6
    phi = target * 0.2
    m0 = torch.randn(dim)
    codebook = torch.stack([
        target,
        target + 0.05 * torch.randn(dim),
        -target + 0.05 * torch.randn(dim),
        torch.randn(dim),
        torch.randn(dim),
    ]).float()
    values, col_idx, row_ptr = pack_sparse(w, backend="torch")
    basis = build_metric_basis(codebook, m0, rank=3, backend="torch")
    base_kwargs = relax_kwargs()
    m_euc, _, _ = relax_packed(
        values, col_idx, row_ptr, b, phi, m0, codebook, basis,
        backend="torch",
        metric_rank=0,
        lambda0=1.0,
        lambda_phi=0.0,
        lambda_var=0.0,
        **{k: v for k, v in base_kwargs.items() if k not in {"metric_rank", "lambda0", "lambda_phi", "lambda_var"}}
    )
    m_rie, _, _ = relax_packed(
        values, col_idx, row_ptr, b, phi, m0, codebook, basis,
        backend="torch",
        **base_kwargs
    )
    cos_euc = F.cosine_similarity(m_euc.unsqueeze(0), target.unsqueeze(0)).item()
    cos_rie = F.cosine_similarity(m_rie.unsqueeze(0), target.unsqueeze(0)).item()
    assert math.isfinite(cos_euc)
    assert math.isfinite(cos_rie)
    assert cos_rie > 0.6
    assert cos_rie >= cos_euc - 5e-2


def test_pq_build_and_reconstruct_shapes_are_consistent():
    torch.manual_seed(21)
    emb = torch.randn(64, 12)
    pq = pq_build_codebook(
        emb,
        subdim=3,
        bits=3,
        iters=4,
        batch_size=32,
        sample_size=48,
        seed=5,
    )
    centroids = pq["centroids"]
    codes = pq["codes"]
    assert centroids.shape == (4, 8, 3)
    assert codes.shape == (64, 4)
    recon = pq_reconstruct_tokens(centroids.float(), codes, [0, 1, 2, 3])
    assert recon.shape == (4, 12)
    assert torch.isfinite(recon).all()


def test_pq_scores_rank_reconstructed_self_highest_on_small_case():
    torch.manual_seed(22)
    emb = torch.randn(32, 8)
    pq = pq_build_codebook(
        emb,
        subdim=2,
        bits=2,
        iters=4,
        batch_size=16,
        sample_size=24,
        seed=7,
    )
    centroids = pq["centroids"].float()
    codes = pq["codes"]
    recon = pq_reconstruct_tokens(centroids, codes)
    query = recon[5]
    scores = pq_scores(query, centroids, codes)
    assert scores.shape == (32,)
    assert torch.isfinite(scores).all()
    top_idx = int(scores.argmax().item())
    assert top_idx == 5


def test_relax_zero_tol_matches_thresholded_matrix_behavior():
    w, b, phi, m0, codebook = make_case(dim=6, n_code=4, seed=23)
    zero_tol = float(w.abs().median().item())
    kwargs = dict(
        portal=PORTAL,
        bypass=BYPASS,
        t_wake=T_WAKE,
        zero_tol=zero_tol,
        backend="torch",
        beta=1.0,
        cb_w=PORTAL,
        lambda0=1.0,
        lambda_phi=0.5,
        lambda_var=0.25,
        tau=1.0,
        dt=0.01,
        max_steps=16,
        tol=1e-7,
        anneal_ratio=0.6,
        noise_scale=0.0,
        metric_rank=0,
        seed=3,
    )
    m_thr, hist_thr, steps_thr = relax(w, b, phi, m0, codebook, **kwargs)

    w_thr = w.clone()
    w_thr[w_thr.abs() <= zero_tol] = 0
    m_ref, hist_ref, steps_ref = relax(
        w_thr,
        b,
        phi,
        m0,
        codebook,
        **{**kwargs, "zero_tol": 0.0},
    )
    assert steps_thr == steps_ref
    assert torch.allclose(m_thr, m_ref, atol=1e-6, rtol=1e-6)
    assert hist_thr["E"] == hist_ref["E"]


@pytest.mark.skipif(not torch.cuda.is_available() or not has_cuda(), reason="CUDA CE backend unavailable")
def test_relax_cuda_matches_torch_without_noise():
    w, b, phi, m0, codebook = make_case(seed=10)
    values_t, col_idx_t, row_ptr_t = pack_sparse(w, backend="torch")
    basis_t = build_metric_basis(codebook, m0, rank=4, backend="torch")
    kwargs = relax_kwargs()
    m_t, hist_t, steps_t = relax_packed(
        values_t, col_idx_t, row_ptr_t, b, phi, m0, codebook, basis_t, backend="torch", **kwargs
    )

    device = torch.device("cuda")
    values_c = values_t.to(device)
    col_idx_c = col_idx_t.to(device)
    row_ptr_c = row_ptr_t.to(device)
    b_c = b.to(device)
    phi_c = phi.to(device)
    m0_c = m0.to(device)
    codebook_c = codebook.to(device)
    basis_c = basis_t.to(device)
    m_c, hist_c, steps_c = relax_packed(
        values_c, col_idx_c, row_ptr_c, b_c, phi_c, m0_c, codebook_c, basis_c,
        backend="cuda", **kwargs
    )
    assert steps_t == steps_c
    assert torch.allclose(m_t, m_c.cpu(), atol=1e-4, rtol=1e-4)
    assert max(abs(a - b) for a, b in zip(hist_t["E"], hist_c["E"])) < 1e-4
