"""STDP learning tests (F.14)."""

import torch
import pytest
from reality_stone.clarus.stdp import (
    STDPConfig, EligibilityTracker, compute_learning_gate,
    structural_projection, apply_stdp_update,
)
from reality_stone.clarus.constants import STDP_R_E, ACTIVE_RATIO


class TestEligibilityTracker:
    def test_causal_orientation_is_asymmetric_and_legacy_default_is_preserved(self):
        legacy = EligibilityTracker(STDPConfig(dim=3, spike_threshold=0.1))
        causal = EligibilityTracker(STDPConfig(dim=3, spike_threshold=0.1, orientation="causal"))
        pre, post = torch.zeros(3), torch.zeros(3)
        pre[0], post[1] = 1.0, 1.0
        legacy.update(pre); legacy.update(post)
        causal.update(pre); causal.update(post)
        # W[row=post, col=pre] is the explicit causal convention.
        assert causal.eligibility[1, 0] > 0
        assert causal.eligibility[0, 1] != causal.eligibility[1, 0]
        assert legacy.eligibility[0, 1] > legacy.eligibility[1, 0]
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


# ---------------------------------------------------------------------------
# F.14.2 closed-loop wiring: the Layer-F critic must drive the STDP gate.
# ---------------------------------------------------------------------------
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from reality_stone.clarus.agent import RuntimeAgent


def _plastic_runtime(dim: int = 16) -> BrainRuntime:
    cfg = BrainRuntimeConfig(
        dim=dim,
        stdp_enabled=True,
        stdp_interval=1,
        stdp_apply_interval=1,
        stdp_gate_threshold=0.0,
        stdp_spike_threshold=0.05,
        noise_sigma=0.0,
        dale_law=False,
        axon_delay=False,
        f1_self_measure=False,
    )
    torch.manual_seed(0)
    weight = torch.randn(dim, dim) * 0.2
    weight.fill_diagonal_(0.0)
    return BrainRuntime(weight, config=cfg)


class TestRuntimeCriticGate:
    def test_runtime_causal_pre_to_post_potentiates_applied_weight(self):
        """The live runtime matrix, not just eligibility, uses post/pre rows."""
        dim, pre, post = 4, 0, 1
        cfg = BrainRuntimeConfig(
            dim=dim,
            active_ratio=1.0,
            stdp_enabled=True,
            stdp_interval=1,
            stdp_apply_interval=2,
            stdp_lr=4.0,  # keeps the causal entry above structural projection.
            stdp_density=1.0,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.05,
            stdp_gate_mode="external_signed",
            stdp_orientation="causal",
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            hippocampal_encoding_enabled=False,
        )
        runtime = BrainRuntime(torch.zeros(dim, dim), config=cfg)
        before = runtime.weight.clone()
        pre_drive, post_drive = torch.zeros(dim), torch.zeros(dim)
        pre_drive[pre], post_drive[post] = 10.0, 10.0
        runtime.step(external_input=pre_drive, force_mode=RuntimeMode.WAKE, learning_signal=0.5)
        applied = runtime.step(external_input=post_drive, force_mode=RuntimeMode.WAKE, learning_signal=0.5)
        delta = runtime.weight - before
        assert applied.stdp_gate == pytest.approx(1.0)
        assert torch.isfinite(runtime.weight).all()
        assert delta[post, pre] > 0.0
        # The reverse diagnostic is LTD-dominated for this ordered pair.
        assert delta[pre, post] <= 0.0
        assert delta[post, pre] > delta[pre, post]
        assert BrainRuntimeConfig(dim=dim).stdp_orientation == "legacy"

    def test_critic_score_drives_gate(self):
        """Two identical runtimes diverge in STDP gate when fed different critics."""
        ext = torch.ones(16) * 0.5
        rt_hi = _plastic_runtime()
        rt_lo = _plastic_runtime()
        step_hi = rt_hi.step(external_input=ext, force_mode=RuntimeMode.WAKE, critic_score=5.0)
        step_lo = rt_lo.step(external_input=ext, force_mode=RuntimeMode.WAKE, critic_score=0.0)
        # Same dynamics, different critic -> different learning gate (F.14.2).
        assert step_hi.stdp_gate != step_lo.stdp_gate

    def test_default_falls_back_to_energy(self):
        """No critic_score -> gate still computed (energy proxy), behavior preserved."""
        rt = _plastic_runtime()
        step = rt.step(external_input=torch.ones(16) * 0.5, force_mode=RuntimeMode.WAKE)
        assert isinstance(step.stdp_gate, float)

    def test_agent_feeds_critic_into_plasticity(self):
        """RuntimeAgent wires its critic into the runtime STDP gate over an episode."""
        rt = _plastic_runtime()
        agent = RuntimeAgent(rt)
        ext = torch.ones(16) * 0.5
        gates = []
        for _ in range(6):
            out = agent.step(external_input=ext, observation=ext, force_mode=RuntimeMode.WAKE)
            gates.append(out.runtime_step.stdp_gate)
        # Critic becomes nonzero after the first tick and drives later gates.
        assert agent._last_critic_score >= 0.0
        assert any(g != 0.0 for g in gates)
        assert rt._stdp_updates > 0

    def test_external_signed_gate_preserves_signal_sign(self):
        cfg = BrainRuntimeConfig(
            dim=16,
            stdp_enabled=True,
            stdp_interval=1,
            stdp_apply_interval=1,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.05,
            stdp_gate_mode="external_signed",
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
        )
        torch.manual_seed(2)
        weight = torch.randn(16, 16) * 0.1
        weight.fill_diagonal_(0.0)
        positive = BrainRuntime(weight.clone(), config=cfg)
        negative = BrainRuntime(weight.clone(), config=cfg)
        ext = torch.ones(16) * 0.5
        out_positive = positive.step(
            external_input=ext,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.25,
        )
        out_negative = negative.step(
            external_input=ext,
            force_mode=RuntimeMode.WAKE,
            learning_signal=-0.25,
        )
        assert out_positive.stdp_gate == pytest.approx(0.25)
        assert out_negative.stdp_gate == pytest.approx(-0.25)

    def test_external_signed_gate_fails_closed_without_signal(self):
        cfg = BrainRuntimeConfig(
            dim=8,
            stdp_enabled=True,
            stdp_apply_interval=1,
            stdp_gate_threshold=0.0,
            stdp_gate_mode="external_signed",
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
        )
        runtime = BrainRuntime(torch.zeros(8, 8), config=cfg)
        before = runtime.weight.clone()
        out = runtime.step(external_input=torch.ones(8), force_mode=RuntimeMode.WAKE)
        assert out.stdp_gate == 0.0
        assert out.stdp_updates == 0
        assert torch.equal(runtime.weight, before)

    def test_external_signal_survives_until_apply_interval_and_snapshot(self):
        cfg = BrainRuntimeConfig(
            dim=8,
            stdp_enabled=True,
            stdp_interval=1,
            stdp_apply_interval=3,
            stdp_gate_threshold=0.0,
            stdp_gate_mode="external_signed",
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
        )
        runtime = BrainRuntime(torch.randn(8, 8) * 0.02, config=cfg)
        external = torch.ones(8) * 0.2
        first = runtime.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.2,
        )
        assert first.stdp_gate == 0.0
        restored = BrainRuntime.from_snapshot(runtime.snapshot(), backend="torch", device="cpu")
        restored.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=-0.05,
        )
        applied = restored.step(
            external_input=external,
            force_mode=RuntimeMode.WAKE,
            learning_signal=0.1,
        )
        assert applied.stdp_gate == pytest.approx(0.25)
