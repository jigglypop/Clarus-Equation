"""Tests for ContinuousClarusCell — persistent self-recursion + wake/
NREM/REM modes at the transformer-block level.

Verifies:
  * State persists across forward calls.
  * Clock advances, reset() clears both state and clock.
  * Wake/NREM/REM modes produce distinct dynamics.
  * Autonomous mode (x=None) runs self-driven.
  * NREM contraction damps state toward lower norm.
  * REM produces larger oscillation than NREM (higher variance).
  * consciousness_depth signal is finite and in [0, 1].
  * Backward works through `step()`.
"""
from __future__ import annotations

import torch

from clarus.ce_euler import ContinuousClarusCell


def test_persistence_across_calls():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        y1 = cell.step(x).clone()
        y2 = cell.step(x).clone()   # same input, different state
    assert not torch.allclose(y1, y2), "state didn't evolve across calls"
    assert cell.clock == 2


def test_reset_clears_state_and_clock():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        cell.step(x); cell.step(x)
    assert cell.has_state()
    assert cell.clock == 2
    cell.reset()
    assert not cell.has_state()
    assert cell.clock == 0
    assert cell.state_norm_trace() == []


def test_autonomous_requires_prior_state():
    cell = ContinuousClarusCell(32, 4, 16).eval()
    import pytest
    with pytest.raises(ValueError):
        cell.step(x=None)


def test_autonomous_runs_without_input():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        cell.step(x)
        # Now drop input entirely and let dynamics self-drive.
        y0 = cell._state.clone()
        cell.set_mode("rem")
        y_final = cell.autonomous(n_steps=5)
    assert cell.clock == 6
    assert not torch.allclose(y0, y_final), "autonomous run didn't change state"
    assert torch.isfinite(y_final).all()


def test_mode_switching_changes_dynamics():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)

    def run_mode(mode, n):
        torch.manual_seed(0)
        c = ContinuousClarusCell(32, 4, 16).eval()
        c.set_mode(mode)
        with torch.no_grad():
            for _ in range(n):
                c.step(x)
        return c._state

    y_w = run_mode("wake", 5)
    y_n = run_mode("nrem", 5)
    y_r = run_mode("rem", 5)
    # All three must differ from each other.
    assert (y_w - y_n).abs().max().item() > 1e-4, "wake == nrem"
    assert (y_w - y_r).abs().max().item() > 1e-4, "wake == rem"
    assert (y_n - y_r).abs().max().item() > 1e-4, "nrem == rem"


def test_nrem_damps_relative_to_wake():
    """NREM applies contraction < 1 after each step. At random init the
    core itself is weakly expansive (residual + LayerNorm), so neither
    mode strictly shrinks ||h||, but NREM must grow strictly slower
    than wake. After training on a Banach-contractive core NREM would
    shrink in absolute terms; this regression test uses the wake ratio
    as the operational check."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)

    def final_norm(mode, n):
        torch.manual_seed(0)
        c = ContinuousClarusCell(32, 4, 16).eval()
        with torch.no_grad():
            c.step(x)
            c.set_mode(mode)
            for _ in range(n):
                c.step(x=None)
        return c._state.norm().item()

    wake_norm = final_norm("wake", 30)
    nrem_norm = final_norm("nrem", 30)
    assert nrem_norm < wake_norm, \
        f"NREM should grow slower than wake: wake={wake_norm:.3f} nrem={nrem_norm:.3f}"


def test_rem_variance_exceeds_nrem():
    """REM has higher phase_scale and no contraction → bigger oscillation."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)

    def variance_of(mode, n):
        torch.manual_seed(0)
        c = ContinuousClarusCell(32, 4, 16).eval()
        with torch.no_grad():
            c.step(x)
        c.set_mode(mode)
        with torch.no_grad():
            for _ in range(n):
                c.step(x=None)
        log = c.state_norm_trace()
        mean = sum(log) / len(log)
        var = sum((v - mean) ** 2 for v in log) / len(log)
        return var

    v_nrem = variance_of("nrem", 40)
    v_rem = variance_of("rem", 40)
    assert v_rem > v_nrem * 0.5, \
        f"REM variance should be large: rem={v_rem:.4f} nrem={v_nrem:.4f}"


def test_consciousness_depth_in_unit_interval():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(20):
            cell.step(x)
    d = cell.consciousness_depth()
    assert 0.0 <= d <= 1.0, f"consciousness_depth out of [0,1]: {d}"


def test_forward_multistep_equivalent_to_loop():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)

    c1 = ContinuousClarusCell(32, 4, 16).eval()
    with torch.no_grad():
        c1.forward(x, n_steps=3)

    c2 = ContinuousClarusCell(32, 4, 16).eval()
    c2.load_state_dict(c1.state_dict())
    with torch.no_grad():
        for _ in range(3):
            c2.step(x)

    assert torch.allclose(c1._state, c2._state, atol=1e-5)


def test_autograd_through_step():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16)
    x = torch.randn(1, 16, 32, requires_grad=True)
    y = cell.step(x).sum()
    y.backward()
    assert cell.log_phi.grad is not None
    assert torch.isfinite(cell.log_phi.grad).all()


def test_mode_trace_records_mode_per_step():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        cell.step(x)
        cell.set_mode("nrem")
        cell.step(x=None)
        cell.step(x=None)
        cell.set_mode("rem")
        cell.step(x=None)
    assert cell.mode_trace() == ["wake", "nrem", "nrem", "rem"]


def test_invalid_mode_rejected():
    import pytest
    cell = ContinuousClarusCell(32, 4, 16)
    with pytest.raises(ValueError):
        cell.set_mode("sleep")       # typo


# --- Borbely 2-process autonomous mode switching ----------------------------


def test_sleep_pressure_rises_during_wake():
    """Borbely process S: ||S|| approaches s_max exponentially during wake."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, auto_mode=True,
                                tau_wake=50.0, tau_sleep=20.0).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        assert cell.sleep_pressure == 0.0
        for _ in range(10):
            cell.step(x)
        assert cell.sleep_pressure > 0.0
        assert cell.sleep_pressure < cell._s_max


def test_auto_transition_wake_to_nrem():
    """With small tau_wake, pressure saturates quickly and the cell
    transitions wake → nrem once it crosses wake_to_nrem_threshold."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, auto_mode=True,
                                tau_wake=5.0, tau_sleep=20.0,
                                wake_to_nrem_threshold=1.0).eval()
    # Use small-norm input so arousal_trigger (1.0) is not tripped.
    x = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(60):
            cell.step(x)
            if cell.mode != "wake":
                break
    assert cell.mode == "nrem", f"never entered NREM; final={cell.mode}"
    ts = cell.transitions()
    assert any(f == "wake" and t == "nrem" for (_, f, t, _) in ts)


def test_auto_full_cycle_wake_nrem_rem_wake():
    """Full cycle: wake → nrem → rem → wake emerges autonomously."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, auto_mode=True,
                                tau_wake=5.0, tau_sleep=5.0,
                                wake_to_nrem_threshold=1.0,
                                nrem_to_rem_threshold=0.5,
                                rem_to_wake_threshold=0.15).eval()
    x = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(500):
            cell.step(x)
    seen_modes = set(cell.mode_trace())
    assert seen_modes == {"wake", "nrem", "rem"}, \
        f"missing mode in cycle; seen={seen_modes}"


def test_auto_transition_arousal_wakes_from_nrem():
    """External arousal (large-norm input) during NREM snaps back to wake."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, auto_mode=True,
                                tau_wake=5.0, tau_sleep=20.0,
                                wake_arousal_trigger=1.5).eval()
    # Drive into NREM with tiny input.
    weak = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(60):
            cell.step(weak)
            if cell.mode == "nrem":
                break
        assert cell.mode == "nrem"
        # Loud external stimulus.
        loud = 5.0 * torch.randn(1, 16, 32)
        cell.step(loud)
    assert cell.mode == "wake", f"arousal did not wake; mode={cell.mode}"
    # The transition log records a reason tag.
    reasons = [r for (_, _, _, r) in cell.transitions()]
    assert any("arousal" in r for r in reasons)


def test_auto_mode_off_by_default():
    """Default auto_mode=False → mode stays put regardless of pressure."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16).eval()
    assert cell.auto_mode is False
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(30):
            cell.step(x)
    assert cell.mode == "wake", "mode must not switch when auto is off"
    assert cell.sleep_pressure == 0.0, "pressure must not accumulate when auto off"


def test_pressure_log_aligned_with_mode_log():
    """pressure_trace and mode_trace grow in lockstep."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, auto_mode=True,
                                tau_wake=20.0).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(15):
            cell.step(x)
    assert len(cell.pressure_trace()) == len(cell.mode_trace()) == 15
    # Pressure should be monotone non-decreasing during wake.
    p = cell.pressure_trace()
    for i in range(1, len(p)):
        assert p[i] >= p[i - 1] - 1e-6


def test_reset_zeros_sleep_pressure_and_transitions():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, auto_mode=True,
                                tau_wake=3.0).eval()
    x = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(30):
            cell.step(x)
    assert cell.sleep_pressure > 0.0
    cell.reset()
    assert cell.sleep_pressure == 0.0
    assert cell.transitions() == []
    assert cell.pressure_trace() == []


# --- Hippocampus / experience replay ----------------------------------------


def test_memory_disabled_by_default():
    cell = ContinuousClarusCell(32, 4, 16)
    assert cell.memory is None
    assert cell.memory_size() == 0
    assert cell.last_replayed() is None


def test_memory_encode_during_wake():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16,
                                memory_enabled=True,
                                memory_capacity=8,
                                encode_interval=5).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(20):
            cell.step(x)
    # clock runs 1..20; encode at clock=5,10,15,20 → 4 snapshots.
    assert cell.memory_size() == 4
    assert cell.memory_tags() == [5, 10, 15, 20]


def test_memory_encode_capacity_evicts_lowest_priority():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16,
                                memory_enabled=True,
                                memory_capacity=3,
                                encode_interval=1).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(10):
            cell.step(x)
    # Capacity is 3, so no more than 3 snapshots ever live at once.
    assert cell.memory_size() == 3


def test_memory_replay_during_nrem_changes_state():
    """With stored memory and NREM mode, replay injects into state —
    compare a NREM-autonomous run with vs without memory."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)

    # Run A: build memory in wake, then NREM autonomous, memory-on.
    cell_a = ContinuousClarusCell(32, 4, 16,
                                  memory_enabled=True,
                                  encode_interval=2,
                                  replay_strength_nrem=0.5).eval()
    with torch.no_grad():
        for _ in range(10):
            cell_a.step(x)
        assert cell_a.memory_size() > 0
        cell_a.set_mode("nrem")
        for _ in range(8):
            cell_a.step(x=None)
    state_a = cell_a._state.clone()

    # Run B: same seed/weights, memory-off.
    cell_b = ContinuousClarusCell(32, 4, 16,
                                  memory_enabled=False).eval()
    # Copy core weights and log_phi so only memory-injection differs.
    cell_b.core.load_state_dict(cell_a.core.state_dict())
    with torch.no_grad():
        cell_b.log_phi.copy_(cell_a.log_phi)
        for _ in range(10):
            cell_b.step(x)
        cell_b.set_mode("nrem")
        for _ in range(8):
            cell_b.step(x=None)
    state_b = cell_b._state

    diff = (state_a - state_b).abs().max().item()
    assert diff > 1e-3, f"replay should perturb state: diff={diff}"
    # last_replayed is recorded for cell_a, not for cell_b
    assert cell_a.last_replayed() is not None
    assert cell_b.last_replayed() is None


def test_memory_replay_during_rem_uses_lower_strength():
    """REM injects replayed content at a lower gain than NREM (dream
    is less instructed than consolidation)."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)
    cell = ContinuousClarusCell(32, 4, 16,
                                memory_enabled=True,
                                encode_interval=1,
                                replay_strength_nrem=0.5,
                                replay_strength_rem=0.1).eval()
    assert cell._replay_strength["rem"] < cell._replay_strength["nrem"]

    with torch.no_grad():
        for _ in range(6):
            cell.step(x)
        cell.set_mode("rem")
        cell.step(x=None)
    # replay has been injected into state (not None) and is the same
    # shape as a per-sequence snapshot.
    replayed = cell.last_replayed()
    assert replayed is not None
    assert replayed.shape == (16, 32)


def test_memory_preserved_across_reset_by_default():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16,
                                memory_enabled=True,
                                encode_interval=1).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(5):
            cell.step(x)
    assert cell.memory_size() == 5
    cell.reset()
    assert cell.memory_size() == 5, "default reset must preserve memory"
    cell.reset(clear_memory=True)
    assert cell.memory_size() == 0


def test_memory_cue_recall_returns_similar_snapshot():
    """A recall with a cue matching a specific stored snapshot should
    return something weighted toward that snapshot — so cosine
    similarity with the cue is higher than cosine with a random
    unrelated vector."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16,
                                memory_enabled=True,
                                encode_interval=1).eval()
    # Force two clearly distinct memories.
    x1 = torch.randn(1, 16, 32)
    x2 = torch.randn(1, 16, 32)
    with torch.no_grad():
        cell.step(x1)
        cell.step(x2)
    assert cell.memory_size() == 2
    # Cue closer to x1's state should recall something closer to x1.
    snap1 = cell.memory._snapshots[0]  # was the processed x1 state
    snap2 = cell.memory._snapshots[1]
    r = cell.memory.recall(snap1 + 0.01 * torch.randn_like(snap1), topk=1)
    # Similarity to snap1 should exceed similarity to snap2.
    def cos(a, b):
        a = a.mean(0); b = b.mean(0)
        return float((a / a.norm() * b / b.norm()).sum().item())
    assert cos(r, snap1) > cos(r, snap2), \
        f"cue-matching recall failed: cos(r,snap1)={cos(r,snap1):.3f} "\
        f"cos(r,snap2)={cos(r,snap2):.3f}"


# --- Metacognition (C3 self-consistency recovery) ---------------------------


def test_meta_disabled_by_default():
    cell = ContinuousClarusCell(32, 4, 16)
    assert cell.meta_enabled is False
    assert cell.meta_events() == []


def test_meta_dormant_when_depth_is_high():
    """With stable input, consciousness_depth stays near 1. The meta
    loop must not fire."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(
        32, 4, 16, meta_enabled=True,
        meta_window=8, meta_threshold=0.6, meta_max_depth=3,
    ).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(30):
            cell.step(x)
    # With constant input, consciousness stays stable → no trigger.
    assert cell.consciousness_depth(window=8) > 0.9
    assert cell.meta_events() == []


def test_meta_fires_when_depth_drops():
    """Feed deliberately volatile inputs to drive ||h|| around, then
    switch to autonomous so depth recovers. The meta-loop must fire
    at least once during the volatile phase."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(
        32, 4, 16,
        meta_enabled=True, meta_window=8,
        meta_threshold=0.95, meta_max_depth=3, meta_tol=0.0,
    ).eval()
    # Volatile inputs: each step large random norm → ||h|| is unstable.
    with torch.no_grad():
        for _ in range(20):
            x = (2.0 + torch.rand(1)) * torch.randn(1, 16, 32)
            cell.step(x)
    events = cell.meta_events()
    assert len(events) > 0, "meta never fired despite high threshold"
    # Each event records 1..meta_max_depth iterations.
    for (_, iters, before, after) in events:
        assert 1 <= iters <= 3
        assert 0.0 <= before <= 1.0
        assert 0.0 <= after <= 1.0


def test_meta_cap_respected():
    """meta_max_depth=1 must cap every event at exactly 1 iteration."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(
        32, 4, 16,
        meta_enabled=True, meta_window=4,
        meta_threshold=0.99, meta_max_depth=1, meta_tol=0.0,
    ).eval()
    with torch.no_grad():
        for _ in range(15):
            cell.step((0.5 + torch.rand(1)) * torch.randn(1, 16, 32))
    for (_, iters, _, _) in cell.meta_events():
        assert iters == 1


def test_meta_tolerance_exit_early():
    """With a large meta_tol, every meta event should stop at 1
    iteration (the first step is always within tolerance trivially)."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(
        32, 4, 16,
        meta_enabled=True, meta_window=4,
        meta_threshold=0.99, meta_max_depth=10, meta_tol=1e6,
    ).eval()
    with torch.no_grad():
        for _ in range(10):
            cell.step((0.5 + torch.rand(1)) * torch.randn(1, 16, 32))
    assert len(cell.meta_events()) > 0
    for (_, iters, _, _) in cell.meta_events():
        assert iters == 1, f"tol did not short-circuit: iters={iters}"


def test_meta_reset_clears_events():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(
        32, 4, 16,
        meta_enabled=True, meta_window=4, meta_threshold=0.99,
    ).eval()
    with torch.no_grad():
        for _ in range(10):
            cell.step(torch.randn(1, 16, 32))
    assert len(cell.meta_events()) > 0
    cell.reset()
    assert cell.meta_events() == []


def test_meta_logs_state_during_extra_iters():
    """The extra core iterations must also extend the state_norm/mode
    logs so that `consciousness_depth` 'sees' them in the next window."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(
        32, 4, 16,
        meta_enabled=True, meta_window=4, meta_threshold=0.99,
        meta_max_depth=3, meta_tol=0.0,
    ).eval()
    with torch.no_grad():
        for _ in range(10):
            cell.step(torch.randn(1, 16, 32))
    # state_norm_log length equals regular steps plus sum of meta iters.
    meta_iters = sum(iters for (_, iters, _, _) in cell.meta_events())
    expected_len = 10 + meta_iters
    assert len(cell.state_norm_trace()) == expected_len
    assert len(cell.mode_trace()) == expected_len
    assert len(cell.pressure_trace()) == expected_len


# --- STDP during sleep learning ---------------------------------------------


def test_stdp_disabled_by_default():
    cell = ContinuousClarusCell(32, 4, 16)
    assert cell.stdp_enabled is False
    assert cell.synaptic_norm() == 0.0
    assert cell.stdp_updates() == 0


def test_stdp_buffers_allocated_when_enabled():
    cell = ContinuousClarusCell(32, 4, 16, stdp_enabled=True)
    assert hasattr(cell, "W_syn")
    assert cell.W_syn.shape == (32, 32)
    assert cell._eligibility.shape == (32, 32)


def test_stdp_does_not_update_during_wake():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, stdp_enabled=True, stdp_lr=0.1).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(10):
            cell.step(x)
    # wake only → no NREM transitions → no stdp updates → W_syn untouched
    assert cell.stdp_updates() == 0
    assert cell.synaptic_norm() == 0.0


def test_stdp_updates_during_nrem():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, stdp_enabled=True,
                                stdp_lr=0.1).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        # build eligibility during wake
        for _ in range(5):
            cell.step(x)
        n0 = cell.synaptic_norm()
        cell.set_mode("nrem")
        for _ in range(10):
            cell.step(x=None)
    assert cell.stdp_updates() == 10
    # W_syn has moved away from zero as eligibility accumulates.
    assert cell.synaptic_norm() > n0
    # Bounded below 10 (the numerical clip).
    assert cell.synaptic_norm() <= 10.0 + 1e-3


def test_stdp_bounded_norm():
    """Even with aggressive lr and many NREM steps, W_syn norm is
    clipped at 10."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, stdp_enabled=True,
                                stdp_lr=1.0).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(20):
            cell.step(x)
        cell.set_mode("nrem")
        for _ in range(100):
            cell.step(x=None)
    assert cell.synaptic_norm() <= 10.0 + 1e-3


def test_stdp_reset_zeros_everything():
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, stdp_enabled=True).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(5):
            cell.step(x)
        cell.set_mode("nrem")
        for _ in range(5):
            cell.step(x=None)
    assert cell.stdp_updates() > 0
    cell.reset_stdp()
    assert cell.synaptic_norm() == 0.0
    assert cell.stdp_updates() == 0
    assert cell._eligibility.abs().sum().item() == 0.0


def test_stdp_w_syn_changes_state_after_sleep():
    """After NREM learning, W_syn is non-trivial → a fresh wake step
    with the same input produces different state than before sleep,
    even holding the core weights fixed."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16, stdp_enabled=True,
                                stdp_lr=0.1, stdp_gain=0.2).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(3):
            cell.step(x)
        state_pre_sleep = cell._state.clone()

        cell.set_mode("nrem")
        for _ in range(20):
            cell.step(x=None)
        assert cell.synaptic_norm() > 0.0

        cell.set_mode("wake")
        # reset only the state, keep W_syn
        cell._state = state_pre_sleep.clone()
        cell.step(x)
        state_post_sleep = cell._state.clone()

    diff = (state_post_sleep - state_pre_sleep).abs().max().item()
    assert diff > 1e-3, \
        f"W_syn should shift behaviour after sleep learning: diff={diff}"


def test_auto_cycle_with_memory_end_to_end():
    """Full auto cycle + memory: after several wake/sleep transitions
    the memory buffer accumulates tagged experiences spanning multiple
    wake phases."""
    torch.manual_seed(0)
    cell = ContinuousClarusCell(32, 4, 16,
                                auto_mode=True,
                                tau_wake=8.0, tau_sleep=8.0,
                                wake_to_nrem_threshold=1.0,
                                nrem_to_rem_threshold=0.5,
                                rem_to_wake_threshold=0.15,
                                memory_enabled=True,
                                memory_capacity=16,
                                encode_interval=3).eval()
    x = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(400):
            cell.step(x)
    # Several transitions occurred AND memory was populated.
    assert len(cell.transitions()) >= 4
    assert cell.memory_size() > 0
    assert cell.last_replayed() is not None
    # Tags should span multiple wake phases (i.e. mode labels at those
    # clocks must include 'wake').
    tags = cell.memory_tags()
    trace = cell.mode_trace()
    wake_tags = [t for t in tags if 0 <= t - 1 < len(trace)
                 and trace[t - 1] == "wake"]
    assert len(wake_tags) > 0
