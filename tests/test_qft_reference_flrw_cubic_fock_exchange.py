from itertools import product

import numpy as np

from examples.physics.qft_reference_flrw_cubic_dynamics import (
    FrozenCubicVertex,
)
from examples.physics.qft_reference_flrw_cubic_fock_exchange import (
    evaluate_all_signed_fock_exchange_gate,
    expected_one_insertion_states,
    explicit_normal_ordered_cubic_matrix,
    normal_ordered_cubic_transitions,
)
from examples.physics.qft_reference_flrw_quartic_contact_gate import (
    reference_state,
)


def _synthetic_vertices(
    overrides: dict[tuple[int, int, int, int, int, int], complex],
    *,
    default: complex = 0.0j,
) -> tuple[FrozenCubicVertex, ...]:
    vertices = []
    for first, second, third in product(range(2), repeat=3):
        for first_sign, second_sign, third_sign in product((-1, 1), repeat=3):
            key = (
                first,
                second,
                third,
                first_sign,
                second_sign,
                third_sign,
            )
            vertices.append(
                FrozenCubicVertex(
                    first_mode=first,
                    second_mode=second,
                    third_mode=third,
                    first_frequency_sign=first_sign,
                    second_frequency_sign=second_sign,
                    third_frequency_sign=third_sign,
                    value=complex(overrides.get(key, default)),
                )
            )
    return tuple(vertices)


def test_all_signed_normal_ordered_reachable_closure_and_factors() -> None:
    source = (2, 0, 0, 0)
    all_nonzero = _synthetic_vertices({}, default=1.0 + 0.0j)
    assert {
        item.target
        for item in normal_ordered_cubic_transitions(all_nonzero, source)
    } == set(expected_one_insertion_states(0))
    assert len(expected_one_insertion_states(0)) == 12
    assert len(expected_one_insertion_states(1)) == 12
    assert len(
        set(expected_one_insertion_states(0))
        | set(expected_one_insertion_states(1))
    ) == 18

    overrides = {
        (0, 0, 0, 1, 1, -1): 1.0,
        (0, 0, 0, -1, -1, -1): 1.0,
        (0, 0, 0, 1, -1, -1): 1.0,
        (0, 0, 0, -1, 1, -1): 1.0,
        (0, 1, 0, 1, -1, -1): 1.0,
        (1, 0, 0, -1, 1, -1): 1.0,
        (1, 1, 0, -1, -1, -1): 1.0,
    }
    vertices = _synthetic_vertices(overrides)
    transitions = {
        item.target: item.matrix_element
        for item in normal_ordered_cubic_transitions(vertices, source)
    }
    assert abs(transitions[(0, 0, 1, 0)] - 1.0 / np.sqrt(2.0)) < 1.0e-14
    assert abs(transitions[(4, 0, 1, 0)] - np.sqrt(3.0)) < 1.0e-14
    assert abs(transitions[(2, 0, 1, 0)] - 2.0) < 1.0e-14
    assert abs(transitions[(1, 1, 1, 0)] - np.sqrt(2.0)) < 1.0e-14
    assert abs(transitions[(2, 2, 1, 0)] - 1.0 / np.sqrt(2.0)) < 1.0e-14

    explicit = explicit_normal_ordered_cubic_matrix(vertices)
    dimensions = (5, 5, 2, 2)
    source_index = int(np.ravel_multi_index(source, dimensions))
    for target, matrix_element in transitions.items():
        target_index = int(np.ravel_multi_index(target, dimensions))
        assert abs(explicit[target_index, source_index] - matrix_element) < 1.0e-14


def test_preregistered_all_signed_diagonal_fock_exchange_gate() -> None:
    state, parameters = reference_state()
    receipt = evaluate_all_signed_fock_exchange_gate(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        phase_points=256,
        grid_phase_points=512,
        simpson_subintervals=(256, 512),
    )

    assert receipt.signed_assignment_count == 64
    assert all(
        item.candidate_intermediate_count == 12
        and item.active_intermediate_count == 4
        for item in receipt.branches
    )
    assert receipt.maximum_signed_conjugation_residual < 1.0e-10
    assert receipt.maximum_first_leg_exchange_residual < 1.0e-10
    assert receipt.maximum_vertex_step_residual < 2.0e-4
    assert receipt.maximum_vertex_grid_residual < 1.0e-8
    assert receipt.maximum_vertex_gauge_residual < 1.0e-6
    assert receipt.maximum_explicit_fock_matrix_residual < 1.0e-12
    assert receipt.maximum_projected_h3_hermiticity_residual < 1.0e-10
    assert receipt.maximum_intermediate_parity_residual < 1.0e-12
    assert receipt.maximum_rotating_subset_reproduction_residual < 1.0e-12
    assert receipt.maximum_kernel_quadrature_residual < 1.0e-10
    assert receipt.maximum_kernel_grid_refinement < 1.0e-10
    assert receipt.minimum_nonrotating_energy_gap_magnitude > 1.0e-8
    assert receipt.maximum_finest_exact_normalized_error < 1.0e-4
    assert receipt.maximum_lambda_quarter_scaling_residual < 0.1
    assert receipt.maximum_zero_coupling_residual < 1.0e-12
    assert receipt.minimum_negative_control_to_numerical_error_ratio > 10.0
    assert receipt.quartic_contact_gate_passed
    assert receipt.cubic_resonance_classification_gate_passed
    assert receipt.declared_all_signed_diagonal_exchange_gate_passed
