from __future__ import annotations

import math

import numpy as np

from examples.physics.causal_record_dust_bridge import (
    ExitPhaseMark,
    flat_flrw_cauchy_witness,
    match_exit_antichain,
    monokinetic_dust_data,
)
from examples.physics.lattice_scalar_transition_bridge import (
    LayeredCausalEdge,
    LayeredRecordEvent,
    certify_layered_cauchy_embedding,
)
from examples.physics.quantum_instrument_record_kernel import (
    build_energy_resolved_instrument_tree,
    construct_energy_conserving_collision_instrument,
)


def _two_channel_emission_collision(
    left_probability: float,
) -> np.ndarray:
    left_amplitude = math.sqrt(left_probability)
    right_amplitude = math.sqrt(1.0 - left_probability)
    collision = np.eye(6, dtype=np.complex128)
    energy_two_sector = (1, 2, 3)
    block = np.array(
        [
            [right_amplitude, 0.0, left_amplitude],
            [-left_amplitude, 0.0, right_amplitude],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    collision[np.ix_(energy_two_sector, energy_two_sector)] = block
    return collision


def test_local_collision_record_has_a_nontrivial_conditional_dust_witness() -> None:
    system_hamiltonian = np.diag([1.0, 2.0])
    ancilla_hamiltonian = np.diag([0.0, 1.0, 1.0])
    collision = construct_energy_conserving_collision_instrument(
        system_hamiltonian,
        ancilla_hamiltonian,
        _two_channel_emission_collision(0.4),
        outcome_targets=('silent', 'left', 'right'),
        outcome_labels=(
            'no emitted excitation',
            'left outgoing excitation',
            'right outgoing excitation',
        ),
    )

    assert collision.branch_energy_transfers == (0.0, 1.0, 1.0)
    assert collision.relative_total_energy_commutator_residual < 1.0e-12
    assert collision.relative_ledger_identity_residual < 1.0e-12

    quantum = build_energy_resolved_instrument_tree(
        system_hamiltonian,
        np.diag([0.0, 1.0]),
        (collision.instrument,),
        initial_weight=5.0,
        require_qnd=False,
    )

    assert quantum.terminal_nodes == ('left', 'right')
    assert np.allclose(
        [edge.probability for edge in quantum.transitions],
        [0.4, 0.6],
    )
    assert [history.terminal_node for history in quantum.record_algebra.histories] == [
        'silent',
        'left',
        'right',
    ]
    assert [
        history.supported_by_root
        for history in quantum.record_algebra.histories
    ] == [False, True, True]
    assert math.isclose(quantum.flow.initial_energy, 10.0)
    assert math.isclose(quantum.flow.terminal_energy, 10.0)

    # The collision theorem supplies the orthogonal energy pointer modes.
    # Their lattice locations and mass-shell interpretation remain declared
    # physical inputs, and are checked rather than inferred here.
    embedding = certify_layered_cauchy_embedding(
        (
            LayeredRecordEvent('root', 0, (0, 0, 0)),
            LayeredRecordEvent('left', 1, (0, 0, 0)),
            LayeredRecordEvent('right', 1, (1, 0, 0)),
        ),
        (
            LayeredCausalEdge('root', 'left'),
            LayeredCausalEdge('root', 'right'),
        ),
        exit_nodes=quantum.terminal_nodes,
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        spatial_shape=(2, 1, 1),
    )
    marks = tuple(
        ExitPhaseMark(
            node=node,
            position=embedding.coordinate(node)[1:],
            mass=2.0,
            spatial_momentum=(0.0, 0.0, 0.0),
            residual_efficiency=0.4,
        )
        for node in quantum.terminal_nodes
    )
    kinetic = match_exit_antichain(
        quantum.flow,
        quantum.nodes,
        marks,
        cell_volume=embedding.spatial_volume,
    )
    dust = monokinetic_dust_data(kinetic)
    gravity = flat_flrw_cauchy_witness(dust, newton_constant=1.0)

    assert math.isclose(kinetic.residual_energy_density, 2.0)
    assert math.isclose(kinetic.complement_energy_density, 3.0)
    assert math.isclose(dust.rest_number_density, 1.0)
    assert math.isclose(dust.rest_energy_density, 2.0)
    assert math.isclose(gravity.hamiltonian_residual, 0.0, abs_tol=1.0e-12)
