from __future__ import annotations

import torch

from reality_stone.clarus.constants import ACTIVE_RATIO
from reality_stone.clarus.research import (
    build_runtime,
    evaluate_transition_probe,
    hopfield_from_hidden,
    normalized_drive,
    periodic_stimulus,
    phase_alignment,
    phase_coherence,
    phase_coupling_step,
    phase_lock_probe,
    phase_lock_step,
    phase_network_probe,
    pca_projection,
    train_runtime_stdp,
)


def test_pca_projection_reduces_hidden_dimension():
    hidden = torch.randn(12, 8)

    projected, basis = pca_projection(hidden, max_dim=3)

    assert projected.shape == (12, 3)
    assert basis.shape == (8, 3)


def test_hopfield_from_hidden_returns_stable_square_weight():
    hidden = torch.randn(20, 6)

    weight = hopfield_from_hidden(hidden)

    assert weight.shape == (6, 6)
    assert torch.allclose(weight, weight.T, atol=1e-6)
    assert torch.linalg.eigvalsh(weight)[-1] < 0.0


def test_runtime_stdp_probe_runs_and_reports_metrics():
    hidden = torch.randn(18, 10)
    projected, _ = pca_projection(hidden, max_dim=5)
    train_hidden = projected[:10]
    eval_hidden = projected[10:]
    weight = hopfield_from_hidden(train_hidden)

    runtime = build_runtime(
        weight,
        active_ratio=ACTIVE_RATIO,
        stdp_enabled=True,
        stdp_lr=0.02,
        stdp_apply_interval=2,
        stdp_density=1.0,
    )
    train_runtime_stdp(runtime, normalized_drive(train_hidden), steps=8)

    probe = evaluate_transition_probe(
        build_runtime(
            runtime.weight,
            active_ratio=ACTIVE_RATIO,
            stdp_enabled=False,
            stdp_lr=0.02,
            stdp_apply_interval=2,
            stdp_density=1.0,
        ),
        normalized_drive(eval_hidden),
    )

    assert probe.mse >= 0.0
    assert -1.0 <= probe.cosine <= 1.0
    assert probe.active_ratio >= 0.0


def test_phase_lock_step_improves_external_alignment():
    reference = periodic_stimulus(1, 8, frequency=0.05)[0]
    state = reference * torch.polar(torch.ones(8), torch.full((8,), 1.2))
    before = phase_alignment(state, reference)

    for _ in range(8):
        state = phase_lock_step(state, reference, coupling=0.35)

    assert phase_alignment(state, reference) > before
    assert phase_alignment(state, reference) > 0.95


def test_internal_coherence_does_not_prove_grounding():
    state = torch.ones(8, dtype=torch.complex64)
    reference = torch.polar(torch.ones(8), torch.arange(8, dtype=torch.float32) % 2 * torch.pi)

    assert phase_coherence(state) == 1.0
    assert abs(phase_alignment(state, reference)) < 0.1


def test_phase_lock_probe_tracks_periodic_stimulus():
    stimulus = periodic_stimulus(24, 12, frequency=0.02)
    initial_state = stimulus[0] * torch.polar(torch.ones(12), torch.full((12,), 1.4))

    probe = phase_lock_probe(stimulus, initial_state, coupling=0.3)

    assert probe.steps == 24
    assert probe.final_alignment > probe.initial_alignment
    assert probe.final_alignment > 0.8


def test_phase_coupling_step_raises_neighbor_coherence():
    phase = torch.linspace(0.0, torch.pi, 10)
    state = torch.polar(torch.ones(10), phase)
    adjacency = torch.ones(10, 10) - torch.eye(10)
    before = phase_coherence(state)

    for _ in range(16):
        state = phase_coupling_step(state, adjacency, neighbor_coupling=0.4)

    assert phase_coherence(state) > before


def test_phase_network_probe_tracks_collective_sync():
    phase = torch.linspace(0.0, torch.pi, 12)
    initial_state = torch.polar(torch.ones(12), phase)
    adjacency = torch.ones(12, 12) - torch.eye(12)

    probe = phase_network_probe(
        initial_state,
        adjacency,
        steps=24,
        neighbor_coupling=0.35,
    )

    assert probe.steps == 24
    assert probe.final_coherence > probe.initial_coherence
    assert probe.final_alignment is None
