"""Layer E: Global summary, Self state, snapshots."""

import torch
import pytest
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


def make_runtime(dim=32):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(dim, dim, generator=g)
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0)
    cfg = BrainRuntimeConfig(dim=dim, dale_law=False, axon_delay=False, noise_sigma=0.0)
    return BrainRuntime(w, config=cfg, backend="torch")


class TestSelfState:
    def test_compute_self_state(self):
        rt = make_runtime()
        for _ in range(5):
            rt.step(external_input=torch.randn(32) * 0.3)
        ss = rt.compute_self_state()
        assert "active_fraction" in ss
        assert "bootstrap_deviation" in ss
        assert "sleep_pressure" in ss
        assert "energy" in ss
        assert "mode" in ss
        assert 0.0 <= ss["active_fraction"] <= 1.0
        assert ss["bootstrap_deviation"] >= 0.0

    def test_self_state_changes(self):
        rt = make_runtime()
        ss1 = rt.compute_self_state()
        for _ in range(20):
            rt.step(external_input=torch.randn(32))
        ss2 = rt.compute_self_state()
        assert ss1["energy"] != ss2["energy"]


class TestSnapshot:
    def test_snapshot_restore_continuity(self):
        rt = make_runtime()
        for _ in range(10):
            rt.step(external_input=torch.randn(32) * 0.3)
        snap = rt.snapshot()
        a_before = rt.activation.clone()
        rt2 = BrainRuntime.from_snapshot(snap, backend="torch")
        assert torch.allclose(a_before, rt2.activation)
        assert rt2.step_index == rt.step_index
        assert rt2.mode == rt.mode

    def test_snapshot_preserves_hippocampus(self):
        rt = make_runtime()
        rt.hippocampus.encode(torch.randn(32), priority=1.0)
        snap = rt.snapshot()
        rt2 = BrainRuntime.from_snapshot(snap, backend="torch")
        assert len(rt2.hippocampus) == len(rt.hippocampus)


class TestRuntimeStep:
    def test_step_returns_all_fields(self):
        rt = make_runtime()
        s = rt.step(external_input=torch.randn(32))
        assert hasattr(s, "step")
        assert hasattr(s, "mode")
        assert hasattr(s, "energy")
        assert hasattr(s, "active_modules")
        assert hasattr(s, "replay_norm")
        assert hasattr(s, "sleep_pressure")
        assert hasattr(s, "arousal")
        assert hasattr(s, "lifecycle_counts")

    def test_lifecycle_counts_sum(self):
        rt = make_runtime()
        rt.step(external_input=torch.randn(32))
        lc = rt.lifecycle_counts()
        total = sum(lc.values())
        assert total == 32
