from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal_record_dust_bridge import (
    ExitPhaseMark,
    match_exit_antichain,
    monokinetic_dust_data,
)
from examples.physics.quantum_instrument_record_kernel import (
    KrausBranch,
    RecordInstrument,
    apply_nonselective_channel,
    build_energy_resolved_instrument_tree,
    construct_energy_conserving_collision_instrument,
    construct_luders_energy_instrument,
)


def _energy_projectors() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.diag([1.0, 0.0]).astype(np.complex128),
        np.diag([0.0, 1.0]).astype(np.complex128),
    )


def test_qnd_instrument_derives_born_kernel_and_harmonic_energy_tree() -> None:
    hamiltonian = np.diag([3.0, 5.0])
    root_density = np.diag([0.4, 0.6])
    project_three, project_five = _energy_projectors()
    weak_a = np.diag([math.sqrt(0.75), math.sqrt(0.25)])
    weak_b = np.diag([math.sqrt(0.25), math.sqrt(0.75)])
    instruments = (
        RecordInstrument(
            node='root',
            instrument_id='declared-qnd-weak-readout',
            branches=(
                KrausBranch('A', 'coarse A', weak_a),
                KrausBranch('B', 'coarse B', weak_b),
            ),
        ),
        RecordInstrument(
            node='A',
            instrument_id='energy-refinement-after-A',
            branches=(
                KrausBranch('A3', 'A and energy 3', project_three),
                KrausBranch('A5', 'A and energy 5', project_five),
            ),
        ),
        RecordInstrument(
            node='B',
            instrument_id='energy-refinement-after-B',
            branches=(
                KrausBranch('B3', 'B and energy 3', project_three),
                KrausBranch('B5', 'B and energy 5', project_five),
            ),
        ),
    )

    certificate = build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        instruments,
    )

    assert math.isclose(certificate.state('root').energy_expectation, 4.2)
    assert math.isclose(certificate.state('A').probability_from_root, 0.45)
    assert math.isclose(certificate.state('A').energy_expectation, 11.0 / 3.0)
    assert math.isclose(certificate.state('B').probability_from_root, 0.55)
    assert math.isclose(certificate.state('B').energy_expectation, 51.0 / 11.0)
    assert certificate.terminal_energy_resolved
    assert all(
        math.isclose(certificate.state(node).energy_variance, 0.0, abs_tol=1.0e-12)
        for node in certificate.terminal_nodes
    )
    assert math.isclose(certificate.flow.initial_energy, 4.2)
    assert math.isclose(certificate.flow.terminal_energy, 4.2)
    assert all(
        math.isclose(balance.energy_residual, 0.0, abs_tol=1.0e-12)
        for balance in certificate.flow.balances
    )
    record = certificate.record_algebra
    assert record.orthogonal_history_basis
    assert record.commutative_diagonal_algebra
    assert record.append_only_history_labels
    assert not record.physical_pointer_dynamics_derived
    assert [history.path for history in record.histories] == [
        ('root', 'A', 'A3'),
        ('root', 'A', 'A5'),
        ('root', 'B', 'B3'),
        ('root', 'B', 'B5'),
    ]
    assert np.allclose(
        [history.probability for history in record.histories],
        [0.3, 0.15, 0.1, 0.45],
    )
    assert math.isclose(record.probability('A'), 0.45)
    assert math.isclose(record.probability('B'), 0.55)
    assert record.global_isometry_residual < 1.0e-12
    assert record.probability_normalization_residual < 1.0e-12
    assert record.max_history_probability_residual < 1.0e-12
    assert record.max_prefix_probability_residual < 1.0e-12


def test_luders_constructor_derives_coarsest_energy_pvm_and_sharp_record() -> None:
    hamiltonian = np.diag([2.0, 2.0, 5.0])
    root_density = np.array(
        [
            [0.2, 0.1, 0.0],
            [0.1, 0.3, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.complex128,
    )
    construction = construct_luders_energy_instrument(hamiltonian)

    assert construction.spectral_energies == (2.0, 5.0)
    assert construction.spectral_multiplicities == (2, 1)
    assert construction.max_spectral_cluster_width == 0.0
    assert len(construction.instrument.branches) == 2
    assert np.allclose(
        construction.instrument.branches[0].operator,
        np.diag([1.0, 1.0, 0.0]),
    )
    assert np.allclose(
        apply_nonselective_channel(root_density, construction.instrument.branches),
        root_density,
    )
    rescaled = construct_luders_energy_instrument(7.0 * hamiltonian)
    assert rescaled.spectral_energies == (14.0, 35.0)
    assert all(
        np.allclose(original.operator, scaled.operator)
        for original, scaled in zip(
            construction.instrument.branches,
            rescaled.instrument.branches,
        )
    )

    certificate = build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (construction.instrument,),
    )

    assert certificate.terminal_energy_resolved
    assert np.allclose(
        [edge.probability for edge in certificate.transitions],
        [0.5, 0.5],
    )
    assert all(
        math.isclose(certificate.state(node).energy_variance, 0.0, abs_tol=1.0e-12)
        for node in certificate.terminal_nodes
    )
    assert math.isclose(certificate.flow.initial_energy, 3.5)
    assert math.isclose(certificate.flow.terminal_energy, 3.5)


def test_luders_rejects_ambiguous_near_degeneracy_until_resolved() -> None:
    hamiltonian = np.diag([2.0, 2.0 + 1.0e-10])

    with pytest.raises(ValueError, match='numerically ambiguous near-degeneracy'):
        construct_luders_energy_instrument(hamiltonian)

    resolved = construct_luders_energy_instrument(
        hamiltonian,
        tolerance=1.0e-12,
    )
    assert len(resolved.instrument.branches) == 2
    assert resolved.max_spectral_cluster_width == 0.0


def test_energy_certificate_is_invariant_under_energy_unit_rescaling() -> None:
    root_density = np.diag([0.4, 0.6])
    receipts = []
    for unit_scale in (1.0e-18, 1.0, 1.0e18):
        hamiltonian = unit_scale * np.diag([2.0, 5.0])
        construction = construct_luders_energy_instrument(hamiltonian)
        certificate = build_energy_resolved_instrument_tree(
            hamiltonian,
            root_density,
            (construction.instrument,),
        )
        receipts.append(
            (
                tuple(edge.probability for edge in certificate.transitions),
                certificate.terminal_energy_resolved,
                certificate.record_algebra.commutative_diagonal_algebra,
            )
        )
        assert math.isclose(
            construction.hamiltonian_energy_scale,
            5.0 * unit_scale,
        )
        assert construction.max_relative_hamiltonian_commutator_residual < 1.0e-12
        assert construction.max_relative_eigenprojector_residual < 1.0e-12
        assert certificate.max_relative_energy_channel_residual < 1.0e-12
        assert certificate.max_relative_qnd_commutator_residual < 1.0e-12

    assert receipts == [receipts[0], receipts[0], receipts[0]]


def test_luders_tree_skips_subthreshold_records_without_dropping_completeness() -> None:
    hamiltonian = np.diag([2.0, 5.0])
    root_density = np.diag([1.0 - 1.0e-12, 1.0e-12])
    construction = construct_luders_energy_instrument(hamiltonian)

    certificate = build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (construction.instrument,),
    )

    assert len(construction.instrument.branches) == 2
    assert len(certificate.transitions) == 1
    assert certificate.terminal_nodes == ('energy-sector-0',)
    assert math.isclose(certificate.max_completeness_residual, 0.0)
    assert certificate.support_probability_tolerance == 1.0e-10
    assert certificate.terminal_energy_resolved
    assert len(certificate.record_algebra.histories) == 2
    assert [
        history.supported_by_root
        for history in certificate.record_algebra.histories
    ] == [True, False]
    assert math.isclose(
        certificate.record_algebra.histories[1].probability,
        1.0e-12,
        rel_tol=1.0e-5,
    )
    assert certificate.record_algebra.global_isometry_residual < 1.0e-12


def test_luders_pvm_does_not_make_degenerate_qnd_instrument_unique() -> None:
    hamiltonian = 2.0 * np.eye(2, dtype=np.complex128)
    root_density = np.diag([1.0, 0.0]).astype(np.complex128)
    construction = construct_luders_energy_instrument(hamiltonian)
    swap = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    swapped = RecordInstrument(
        node='root',
        instrument_id='degenerate-sector-swap',
        branches=(KrausBranch('swapped', 'swapped inside energy sector', swap),),
    )

    luders_channel = apply_nonselective_channel(
        root_density,
        construction.instrument.branches,
    )
    swapped_channel = apply_nonselective_channel(root_density, swapped.branches)
    assert np.allclose(luders_channel, root_density)
    assert not np.allclose(swapped_channel, root_density)
    assert build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (construction.instrument,),
    ).terminal_energy_resolved
    assert build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (swapped,),
    ).terminal_energy_resolved


def test_same_channel_can_have_different_declared_record_trees() -> None:
    hamiltonian = np.diag([3.0, 5.0])
    root_density = np.array(
        [
            [0.4, 0.2],
            [0.2, 0.6],
        ],
        dtype=np.complex128,
    )
    project_three, project_five = _energy_projectors()
    identity = np.eye(2, dtype=np.complex128)
    phase = np.diag([1.0, -1.0]).astype(np.complex128)
    energy_branches = (
        KrausBranch('energy-3', 'resolved energy 3', project_three),
        KrausBranch('energy-5', 'resolved energy 5', project_five),
    )
    phase_branches = (
        KrausBranch('phase-plus', 'phase unravelling plus', identity / math.sqrt(2.0)),
        KrausBranch('phase-minus', 'phase unravelling minus', phase / math.sqrt(2.0)),
    )

    assert np.allclose(
        apply_nonselective_channel(root_density, energy_branches),
        apply_nonselective_channel(root_density, phase_branches),
    )

    energy_tree = build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (
            RecordInstrument(
                'root',
                'energy-resolved-unravelling',
                energy_branches,
            ),
        ),
    )
    phase_tree = build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (
            RecordInstrument(
                'root',
                'phase-labelled-unravelling',
                phase_branches,
            ),
        ),
        require_terminal_energy_sharp=False,
    )

    assert energy_tree.terminal_energy_resolved
    assert not phase_tree.terminal_energy_resolved
    assert [edge.probability for edge in energy_tree.transitions] == [0.4, 0.6]
    assert np.allclose(
        [edge.probability for edge in phase_tree.transitions],
        [0.5, 0.5],
    )
    with pytest.raises(ValueError, match='sharp Hamiltonian energy'):
        build_energy_resolved_instrument_tree(
            hamiltonian,
            root_density,
            (
                RecordInstrument(
                    'root',
                    'phase-labelled-unravelling',
                    phase_branches,
                ),
            ),
        )


def test_energy_changing_instrument_cannot_receive_qnd_certificate() -> None:
    hamiltonian = np.diag([1.0, 2.0])
    root_density = np.diag([0.25, 0.75])
    decay_probability = 0.2
    no_jump = np.diag([1.0, math.sqrt(1.0 - decay_probability)])
    jump = np.array(
        [
            [0.0, math.sqrt(decay_probability)],
            [0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    with pytest.raises(
        ValueError,
        match='preserve the declared Hamiltonian-plus-transfer ledger',
    ):
        build_energy_resolved_instrument_tree(
            hamiltonian,
            root_density,
            (
                RecordInstrument(
                    'root',
                    'amplitude-damping-readout',
                    (
                        KrausBranch('no-jump', 'no jump', no_jump),
                        KrausBranch('jump', 'energy-losing jump', jump),
                    ),
                ),
            ),
        )


def _excitation_swap_collision() -> np.ndarray:
    collision = np.eye(4, dtype=np.complex128)
    collision[1, 1] = 0.0
    collision[2, 2] = 0.0
    collision[1, 2] = 1.0
    collision[2, 1] = 1.0
    return collision


def test_energy_conserving_collision_closes_system_plus_record_ledger() -> None:
    system_hamiltonian = np.diag([1.0, 2.0])
    ancilla_hamiltonian = np.diag([0.0, 1.0])
    construction = construct_energy_conserving_collision_instrument(
        system_hamiltonian,
        ancilla_hamiltonian,
        _excitation_swap_collision(),
    )

    assert construction.branch_energy_transfers == (0.0, 1.0)
    assert construction.relative_total_energy_commutator_residual < 1.0e-12
    assert construction.relative_ledger_identity_residual < 1.0e-12
    assert construction.kraus_completeness_residual < 1.0e-12
    assert not construction.physical_pointer_persistence_derived

    certificate = build_energy_resolved_instrument_tree(
        system_hamiltonian,
        np.diag([0.25, 0.75]),
        (construction.instrument,),
        require_qnd=False,
    )

    assert not certificate.qnd_required
    assert certificate.max_relative_qnd_commutator_residual > 0.0
    assert certificate.terminal_energy_resolved
    assert math.isclose(certificate.state('root').energy_expectation, 1.75)
    assert math.isclose(
        certificate.state('ancilla-energy-0').system_energy_expectation,
        1.0,
    )
    assert math.isclose(
        certificate.state('ancilla-energy-0').cumulative_energy_transfer,
        0.0,
    )
    assert math.isclose(
        certificate.state('ancilla-energy-0').energy_expectation,
        1.0,
    )
    assert math.isclose(
        certificate.state('ancilla-energy-1').system_energy_expectation,
        1.0,
    )
    assert math.isclose(
        certificate.state('ancilla-energy-1').cumulative_energy_transfer,
        1.0,
    )
    assert math.isclose(
        certificate.state('ancilla-energy-1').energy_expectation,
        2.0,
    )
    assert math.isclose(certificate.flow.initial_energy, 1.75)
    assert math.isclose(certificate.flow.terminal_energy, 1.75)
    assert np.allclose(
        [history.probability for history in certificate.record_algebra.histories],
        [0.25, 0.75],
    )


def test_collision_transfer_receipt_does_not_make_energy_exchange_qnd() -> None:
    construction = construct_energy_conserving_collision_instrument(
        np.diag([1.0, 2.0]),
        np.diag([0.0, 1.0]),
        _excitation_swap_collision(),
    )

    with pytest.raises(ValueError, match='QND'):
        build_energy_resolved_instrument_tree(
            np.diag([1.0, 2.0]),
            np.diag([0.25, 0.75]),
            (construction.instrument,),
        )


def test_non_energy_conserving_collision_cannot_receive_transfer_receipt() -> None:
    flip_ancilla = np.kron(
        np.eye(2, dtype=np.complex128),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    )

    with pytest.raises(ValueError, match='conserve total energy'):
        construct_energy_conserving_collision_instrument(
            np.diag([1.0, 2.0]),
            np.diag([0.0, 1.0]),
            flip_ancilla,
        )


def _degenerate_position_certificate(
    *,
    length_scale: float,
):
    energy = 5.0 / length_scale
    hamiltonian = energy * np.eye(2, dtype=np.complex128)
    root_density = np.diag([0.4, 0.6])
    left, right = _energy_projectors()
    certificate = build_energy_resolved_instrument_tree(
        hamiltonian,
        root_density,
        (
            RecordInstrument(
                'root',
                'energy-degenerate-position-pointer',
                (
                    KrausBranch('left', 'left phase cell', left),
                    KrausBranch('right', 'right phase cell', right),
                ),
            ),
        ),
        initial_weight=5.0,
    )
    marks = (
        ExitPhaseMark(
            'left',
            (0.25 * length_scale, 0.0, 0.0),
            energy,
            (0.0, 0.0, 0.0),
        ),
        ExitPhaseMark(
            'right',
            (0.75 * length_scale, 0.0, 0.0),
            energy,
            (0.0, 0.0, 0.0),
        ),
    )
    matching = match_exit_antichain(
        certificate.flow,
        certificate.nodes,
        marks,
        cell_volume=length_scale**3,
    )
    return certificate, matching


def test_energy_resolved_pointer_feeds_the_existing_mass_shell_dust_bridge() -> None:
    certificate, matching = _degenerate_position_certificate(length_scale=1.0)

    assert certificate.terminal_energy_resolved
    assert certificate.flow.terminal_composition == (
        ('left phase cell', 0.4),
        ('right phase cell', 0.6),
    )
    assert matching.current == (5.0, 0.0, 0.0, 0.0)
    assert matching.stress[0][0] == 25.0
    dust = monokinetic_dust_data(matching)
    assert dust.rest_number_density == 5.0
    assert dust.rest_energy_density == 25.0


def test_hamiltonian_and_embedding_inputs_propagate_their_declared_dimensions() -> None:
    _, unit_matching = _degenerate_position_certificate(length_scale=1.0)
    _, doubled_matching = _degenerate_position_certificate(length_scale=2.0)

    assert math.isclose(
        doubled_matching.current[0],
        unit_matching.current[0] / 2.0**3,
    )
    assert math.isclose(
        doubled_matching.stress[0][0],
        unit_matching.stress[0][0] / 2.0**4,
    )
