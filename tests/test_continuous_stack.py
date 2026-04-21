"""Tests for ContinuousClarusStack — multi-cell synchronized sleep."""
from __future__ import annotations

import torch

from clarus.ce_euler import ContinuousClarusCell, ContinuousClarusStack


def test_stack_basic_forward():
    torch.manual_seed(0)
    stack = ContinuousClarusStack(n_layers=3, d_model=32, n_heads=4, block=16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        y = stack.step(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert stack.clock == 1
    # Every cell advanced once.
    for c in stack.cells:
        assert c.clock == 1


def test_stack_mode_shared_across_cells():
    stack = ContinuousClarusStack(n_layers=3, d_model=32, n_heads=4, block=16)
    stack.set_mode("nrem")
    for c in stack.cells:
        assert c.mode == "nrem"
    stack.set_mode("rem")
    for c in stack.cells:
        assert c.mode == "rem"


def test_stack_rejects_invalid_mode():
    import pytest
    stack = ContinuousClarusStack(2, 32, 4, 16)
    with pytest.raises(ValueError):
        stack.set_mode("bogus")


def test_stack_clock_synchronized_across_cells():
    torch.manual_seed(0)
    stack = ContinuousClarusStack(n_layers=3, d_model=32, n_heads=4, block=16).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(5):
            stack.step(x)
    # All cells share the same clock value each step (stack forces
    # cell._clock = stack._clock BEFORE every cell.step()).
    assert stack.clock == 5
    for c in stack.cells:
        assert c.clock == 5


def test_stack_auto_cycle():
    """Stack-level Borbely: pressure saturates → wake→nrem; relaxes
    → nrem→rem; dissipates → rem→wake. All cells follow."""
    torch.manual_seed(0)
    stack = ContinuousClarusStack(
        n_layers=2, d_model=32, n_heads=4, block=16,
        auto_mode=True,
        tau_wake=5.0, tau_sleep=5.0,
        wake_to_nrem_threshold=1.0,
        nrem_to_rem_threshold=0.5,
        rem_to_wake_threshold=0.15,
    ).eval()
    x = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(400):
            stack.step(x)
    seen = {m for (_, _, m, _) in stack.transitions()}
    assert "nrem" in seen or "rem" in seen
    # Both cells saw the same mode sequence.
    trace_a = stack.cells[0].mode_trace()
    trace_b = stack.cells[1].mode_trace()
    assert trace_a == trace_b


def test_stack_arousal_wakes_all_cells():
    torch.manual_seed(0)
    stack = ContinuousClarusStack(
        n_layers=2, d_model=32, n_heads=4, block=16,
        auto_mode=True, tau_wake=3.0, tau_sleep=30.0,
        wake_arousal_trigger=1.5,
    ).eval()
    weak = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        # Drive into NREM first.
        for _ in range(60):
            stack.step(weak)
            if stack.mode == "nrem":
                break
        assert stack.mode == "nrem"
        stack.step(5.0 * torch.randn(1, 16, 32))   # loud stimulus
    assert stack.mode == "wake"
    for c in stack.cells:
        assert c.mode == "wake"


def test_stack_layer_coherence_is_finite():
    """With a shared clock all layers rotate phase in sync, but at
    random init the resulting state-norm trajectories are only
    structurally (not behaviourally) phase-aligned. Assert only that
    the metric lies in [-1, 1] and can be computed. A strong positive
    coherence is an emergent training outcome tested elsewhere."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32)
    stack = ContinuousClarusStack(
        n_layers=3, d_model=32, n_heads=4, block=16,
    ).eval()
    with torch.no_grad():
        for _ in range(40):
            stack.step(x)
    coh = stack.layer_coherence(window=32)
    assert -1.0 <= coh <= 1.0


def test_stack_clock_sync_applies_equal_rotation_every_step():
    """Shared clock means every cell, on every step, uses the same
    iteration index `t` for its depth-phase rotation. Probing cell
    `._clock` across the stack must therefore produce a single value
    equal to the stack clock."""
    torch.manual_seed(0)
    stack = ContinuousClarusStack(
        n_layers=4, d_model=32, n_heads=4, block=16,
    ).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(10):
            stack.step(x)
            clocks = [c.clock for c in stack.cells]
            assert len(set(clocks)) == 1
            assert clocks[0] == stack.clock


def test_stack_reset():
    torch.manual_seed(0)
    stack = ContinuousClarusStack(
        n_layers=2, d_model=32, n_heads=4, block=16,
        auto_mode=True, tau_wake=3.0,
    ).eval()
    x = 0.01 * torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(20):
            stack.step(x)
    assert stack.clock > 0
    stack.reset()
    assert stack.clock == 0
    assert stack.sleep_pressure == 0.0
    assert stack.mode == "wake"
    assert stack.transitions() == []
    for c in stack.cells:
        assert c.clock == 0
        assert not c.has_state()


def test_stack_memory_independent_per_cell():
    """Memory buffers are NOT shared — each cell encodes its own
    post-transform activations."""
    torch.manual_seed(0)
    stack = ContinuousClarusStack(
        n_layers=2, d_model=32, n_heads=4, block=16,
        memory_enabled=True, encode_interval=1,
    ).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        for _ in range(5):
            stack.step(x)
    # Each cell wrote 5 snapshots.
    for c in stack.cells:
        assert c.memory_size() == 5
    # But the two cells' memories differ because they saw different
    # inputs (layer 1's input = layer 0's output).
    snap_a = stack.cells[0].memory._snapshots[-1]
    snap_b = stack.cells[1].memory._snapshots[-1]
    assert (snap_a - snap_b).abs().max().item() > 1e-4
