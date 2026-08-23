from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus import clarus_field as clarus_field_module
from reality_stone.clarus.clarus_field import (
    ClarusField,
    ClarusFieldConfig,
    bounded_hrr_bind,
    normalized_graph_laplacian,
    prediction_error_gate_scores,
    project_rows_to_unit_ball,
)


def _ring(size: int) -> np.ndarray:
    adjacency = np.zeros((size, size), dtype=np.float64)
    for index in range(size):
        adjacency[index, (index - 1) % size] = 1.0
        adjacency[index, (index + 1) % size] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _field(size: int = 4, width: int = 3, **overrides: float) -> ClarusField:
    config = ClarusFieldConfig(width=width, **overrides)
    return ClarusField(_ring(size), config)


def test_normalized_laplacian_is_symmetric_psd_with_one_zero_mode() -> None:
    laplacian = normalized_graph_laplacian(_ring(5))
    eigenvalues = np.linalg.eigvalsh(laplacian)
    assert np.array_equal(laplacian, laplacian.T)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1.0e-12)
    assert eigenvalues[-1] <= 2.0 + 1.0e-12
    assert np.count_nonzero(abs(eigenvalues) < 1.0e-12) == 1


@pytest.mark.parametrize(
    "adjacency, message",
    [
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "symmetric"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "nonnegative"),
        (np.zeros((2, 2)), "connected"),
        (np.eye(2), "zero diagonal"),
    ],
)
def test_invalid_field_graphs_are_rejected(adjacency: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalized_graph_laplacian(adjacency)


def test_closed_gate_preserves_memory_bit_for_bit_for_long_horizon() -> None:
    runtime = _field(gate_threshold=0.6)
    memory = np.array(
        [[-0.0, 0.25, -0.75], [1.5, -2.0, 0.5], [0.1, 0.2, 0.3], [-1.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    state = runtime.make_state(memory, np.zeros(4))
    drive = runtime.make_drive(np.full(4, 0.6), np.full((4, 3), 1.0e100))
    original_bits = np.asarray(state.memory, dtype=np.float64).view(np.uint64).copy()
    for _ in range(256):
        state = runtime.step(state, drive).state
    final_bits = np.asarray(state.memory, dtype=np.float64).view(np.uint64)
    assert np.array_equal(final_bits, original_bits)


def test_open_writes_are_projected_and_memory_respects_the_global_bound() -> None:
    runtime = _field()
    initial = np.full((4, 3), 2.0)
    state = runtime.make_state(initial, np.zeros(4))
    drive = runtime.make_drive(np.ones(4), np.full((4, 3), 100.0))
    result = runtime.step(state, drive)
    norms = np.linalg.norm(np.asarray(result.state.memory), axis=1)
    assert np.all(norms <= 1.0 + 1.0e-14)
    assert result.memory_bound == pytest.approx(math.sqrt(12.0))


def test_projection_and_hrr_readout_are_bounded_without_entering_transition() -> None:
    projected = project_rows_to_unit_ball([[3.0, 4.0], [0.1, 0.2]])
    assert np.linalg.norm(projected[0]) == pytest.approx(1.0)
    assert np.array_equal(projected[1], np.array([0.1, 0.2]))
    bound = bounded_hrr_bind([1.0, 2.0, 3.0], [-4.0, 5.0, 6.0])
    assert np.linalg.norm(bound) <= 1.0 + 1.0e-14


def test_prediction_error_gate_is_sign_invariant_and_dimensionlessly_scaled() -> None:
    positive = prediction_error_gate_scores([[2.0, 0.0]], [[0.0, 0.0]], reference_scale=2.0)
    negative = prediction_error_gate_scores([[-2.0, 0.0]], [[0.0, 0.0]], reference_scale=2.0)
    rescaled = prediction_error_gate_scores([[4.0, 0.0]], [[0.0, 0.0]], reference_scale=4.0)
    assert np.array_equal(positive, negative)
    assert np.array_equal(positive, rescaled)
    assert positive[0] > 0.95
    with pytest.raises(ValueError, match="reference_scale"):
        prediction_error_gate_scores([[1.0]], [[0.0]], reference_scale=0.0)


def test_one_node_field_step_matches_the_exact_damped_solution() -> None:
    config = ClarusFieldConfig(width=1, field_decay=0.4, tick_duration=0.25)
    runtime = ClarusField([[0.0]], config)
    state = runtime.make_state([[0.75]], [0.2])
    drive = runtime.make_drive([0.0], [[0.0]])
    result = runtime.step(state, drive)
    decay = math.exp(-0.4 * 0.25)
    expected = decay * 0.2 + (1.0 - decay) * 0.75 / 0.4
    assert result.state.field[0] == pytest.approx(expected, rel=1.0e-13)


def test_field_stays_nonnegative_and_below_the_cf1_two_norm_bound() -> None:
    runtime = _field(field_decay=0.35, diffusion_strength=0.8, source_cap=0.7)
    state = runtime.make_state(np.full((4, 3), 10.0), np.zeros(4))
    drive = runtime.make_drive(np.zeros(4), np.zeros((4, 3)))
    for _ in range(500):
        result = runtime.step(state, drive)
        state = result.state
        assert min(state.field) >= 0.0
        assert np.linalg.norm(state.field) <= result.field_bound + 1.0e-12
    assert runtime.certificate.stationary_field_bound == pytest.approx(4.0)


def test_common_exogenous_writes_couple_initial_conditions_at_event_rate() -> None:
    runtime = _field(size=3, width=2, gate_threshold=0.5)
    left = runtime.make_state(np.full((3, 2), -5.0), np.zeros(3))
    right = runtime.make_state(np.full((3, 2), 5.0), np.zeros(3))
    closed = runtime.make_drive(np.zeros(3), np.zeros((3, 2)))
    opened = runtime.make_drive(np.full(3, 0.75), np.full((3, 2), 0.25))
    previous = np.linalg.norm(np.asarray(left.memory) - np.asarray(right.memory))
    for tick in range(40):
        drive = opened if tick % 4 == 0 else closed
        left = runtime.step(left, drive).state
        right = runtime.step(right, drive).state
    difference = np.linalg.norm(np.asarray(left.memory) - np.asarray(right.memory))
    assert difference == pytest.approx(previous * (1.0 - 0.75) ** 10, rel=1.0e-12)


def test_phase_occupancy_uses_dimensionless_field_and_sums_to_one() -> None:
    runtime = _field(
        size=4,
        width=1,
        field_decay=0.5,
        source_cap=1.0,
        gate_threshold=0.5,
        structure_threshold=0.2,
    )
    state = runtime.make_state([[1.0], [1.0], [0.0], [0.0]], [1.0, 1.0, 0.0, 0.0])
    drive = runtime.make_drive([0.9, 0.0, 0.0, 0.0], np.zeros((4, 1)))
    result = runtime.step(state, drive)
    assert result.phase_labels[0] == "active"
    assert set(result.phase_labels) <= {"active", "structural", "frozen"}
    assert sum(result.occupancy.as_tuple()) == pytest.approx(1.0)


def test_occupancy_tracks_external_signal_rate_instead_of_injecting_p_star() -> None:
    runtime = _field(size=20, width=1, gate_threshold=0.5)
    state = runtime.zero_state()
    candidates = np.ones((20, 1))
    observed = []
    for active_count in (2, 6, 14):
        gates = np.zeros(20)
        gates[:active_count] = 1.0
        result = runtime.step(state, runtime.make_drive(gates, candidates))
        observed.append(result.occupancy.active)
        state = result.state
    assert observed == pytest.approx([0.1, 0.3, 0.7])
    assert runtime.certificate.p_star_self_convergence is False


def test_snapshot_roundtrip_is_deterministic_and_detached_from_inputs() -> None:
    runtime = _field(size=3, width=2)
    memory = np.arange(6, dtype=np.float64).reshape(3, 2)
    field = np.array([0.1, 0.2, 0.3])
    state = runtime.make_state(memory, field, tick=7)
    snapshot = runtime.snapshot(state)
    memory[:] = -100.0
    field[:] = -100.0
    assert runtime.from_snapshot(snapshot) == state
    drive = runtime.make_drive([0.0, 1.0, 0.75], np.ones((3, 2)))
    assert runtime.step(state, drive) == runtime.step(runtime.from_snapshot(snapshot), drive)


def test_certificate_keeps_conditional_and_failed_claims_explicit() -> None:
    certificate = _field().certificate
    assert certificate.cf1_bounded_positive_field
    assert certificate.cf2_exact_closed_gate
    assert certificate.cf3_scope.startswith("conditional:")
    assert not certificate.p_star_self_convergence
    assert not certificate.v14_route_l_inherited


def test_clarus_field_module_exports_expected_names() -> None:
    assert clarus_field_module.ClarusField is ClarusField
    assert clarus_field_module.ClarusFieldConfig is ClarusFieldConfig
    assert clarus_field_module.bounded_hrr_bind is bounded_hrr_bind
    assert clarus_field_module.prediction_error_gate_scores is prediction_error_gate_scores
