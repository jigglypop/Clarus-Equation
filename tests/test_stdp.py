"""STDP learning tests (F.14)."""

import torch
import pytest
from clarus.stdp import (
    STDPConfig, EligibilityTracker, compute_learning_gate,
    structural_projection, apply_stdp_update,
)
from clarus.constants import STDP_R_E, ACTIVE_RATIO


class TestEligibilityTracker:
    def test_trace_decay(self):
        cfg = STDPConfig(dim=16)
        tracker = EligibilityTracker(cfg)
        act = torch.zeros(16)
        act[0] = 0.5
        tracker.update(act)
        e0 = tracker.eligibility.clone()
        tracker.update(torch.zeros(16))
        e1 = tracker.eligibility.clone()
        assert e1.abs().max() < e0.abs().max() or e0.abs().max() == 0

    def test_eligibility_shape(self):
        cfg = STDPConfig(dim=32)
        tracker = EligibilityTracker(cfg)
        assert tracker.eligibility.shape == (32, 32)

    def test_spike_creates_eligibility(self):
        cfg = STDPConfig(dim=8, spike_threshold=0.1)
        tracker = EligibilityTracker(cfg)
        act = torch.ones(8) * 0.5
        tracker.update(act)
        assert tracker.eligibility.abs().sum().item() > 0

    def test_reset_clears(self):
        cfg = STDPConfig(dim=8, spike_threshold=0.1)
        tracker = EligibilityTracker(cfg)
        tracker.update(torch.ones(8))
        tracker.reset()
        assert tracker.eligibility.abs().sum().item() == 0
        assert tracker.pre_trace.abs().sum().item() == 0


class TestLearningGate:
    def test_gate_positive_on_improvement(self):
        g = compute_learning_gate(critic_score=0.5, prev_critic_score=0.3, active_ratio=0.05)
        assert g > 0

    def test_gate_includes_bootstrap(self):
        g = compute_learning_gate(
            critic_score=0.0, prev_critic_score=0.0,
            active_ratio=0.5, alpha_g=0.0,
        )
        assert g > 0


class TestProjection:
    def test_projection_density(self):
        w = torch.randn(16, 16)
        proj = structural_projection(w, density=0.2)
        density = (proj != 0).float().mean().item()
        assert density < 0.5

    def test_projection_preserves_shape(self):
        w = torch.randn(8, 8)
        proj = structural_projection(w)
        assert proj.shape == w.shape


class TestSTDPUpdate:
    def test_weight_changes(self):
        cfg = STDPConfig(dim=8, spike_threshold=0.1)
        tracker = EligibilityTracker(cfg)
        w = torch.randn(8, 8)
        for _ in range(5):
            tracker.update(torch.randn(8) * 0.5)
        w_new = apply_stdp_update(w, tracker, gate=1.0)
        assert not torch.allclose(w, w_new)

    def test_zero_gate_no_change(self):
        cfg = STDPConfig(dim=8, spike_threshold=0.1)
        tracker = EligibilityTracker(cfg)
        w = torch.randn(8, 8)
        tracker.update(torch.randn(8))
        w_new = apply_stdp_update(w, tracker, gate=0.0, density=1.0)
        proj_w = structural_projection(w, density=1.0)
        assert torch.allclose(w_new, proj_w, atol=1e-5)
