"""Layer B: Field coupling, energy, brainwave observable."""

import torch
import pytest
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


def make_runtime(dim=64):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(dim, dim, generator=g)
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0)
    cfg = BrainRuntimeConfig(dim=dim, dale_law=False, axon_delay=False, noise_sigma=0.0)
    return BrainRuntime(w, config=cfg, backend="torch")


class TestEnergy:
    def test_energy_full_finite(self):
        rt = make_runtime()
        for _ in range(10):
            rt.step(external_input=torch.randn(64) * 0.5)
        e = rt.energy_full()
        assert isinstance(e, float)
        assert abs(e) < 1e6

    def test_energy_changes_with_activity(self):
        rt = make_runtime()
        e0 = rt.energy_full()
        for _ in range(20):
            rt.step(external_input=torch.randn(64))
        e1 = rt.energy_full()
        assert e0 != e1


class TestBrainwave:
    def test_observable_returns_psi(self):
        rt = make_runtime()
        obs = rt.brainwave_observable()
        assert "psi_global" in obs
        assert isinstance(obs["psi_global"], float)

    def test_band_powers_after_history(self):
        rt = make_runtime()
        for _ in range(20):
            rt.step(external_input=torch.randn(64) * 0.3)
        obs = rt.brainwave_observable()
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            assert band in obs
            assert obs[band] >= 0.0

    def test_brainwave_history_bounded(self):
        rt = make_runtime()
        for _ in range(2000):
            rt.step(external_input=torch.randn(64) * 0.1)
        assert len(rt._brainwave_history) <= rt._brainwave_max_len


class TestRiemannianWeight:
    def test_build_riemannian_weight(self):
        dim = 32
        coords = torch.randn(dim, 3)
        sigma = 1.0
        dist = torch.cdist(coords, coords)
        w = torch.exp(-dist ** 2 / sigma ** 2)
        w.fill_diagonal_(0)
        assert w.shape == (dim, dim)
        assert torch.allclose(w, w.T)
        assert (w >= 0).all()

    def test_sparse_mask(self):
        dim = 32
        coords = torch.randn(dim, 3)
        dist = torch.cdist(coords, coords)
        w = torch.exp(-dist ** 2 / 1.0)
        w.fill_diagonal_(0)
        mask = dist < 2.0
        w_sparse = w * mask.float()
        density = (w_sparse > 0).float().mean().item()
        assert 0.0 < density < 1.0
