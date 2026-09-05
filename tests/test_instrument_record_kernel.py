from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.record.record_dust_bridge import (
    ExitPhaseMark,
    match_exit_antichain,
    monokinetic_dust_data,
)
from examples.physics.record.instrument_record_kernel import (
    CoarseOutcomeOperation,
    KrausBranch,
    RecordInstrument,
    apply_nonselective_channel,
    apply_nonselective_instrument,
    born_probabilities,
    build_energy_resolved_instrument_tree,
    build_seed_partition,
    certificate,
    construct_energy_conserving_collision_instrument,
    construct_luders_energy_instrument,
    equal_copy_internal_refinement,
    run,
    select_outcome,
    select_partition_cell,
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


P0 = np.diag([1.0, 0.0]).astype(np.complex128)
P1 = np.diag([0.0, 1.0]).astype(np.complex128)
PROJECTIVE_OUTCOMES = (
    CoarseOutcomeOperation("zero", (P0,)),
    CoarseOutcomeOperation("one", (P1,)),
)


def test_half_open_seed_partition_handles_zero_cells_and_exact_boundaries() -> None:
    partition = build_seed_partition((0.0, 0.4, 0.6))
    assert partition.input_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert partition.cell_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert partition.intervals[0] == (0.0, 0.0)
    assert partition.intervals[1] == pytest.approx((0.0, 0.4))
    assert partition.intervals[2] == pytest.approx((0.4, 1.0))
    assert select_partition_cell(partition, 0.0) == 1
    assert select_partition_cell(partition, math.nextafter(0.4, 0.0)) == 1
    assert select_partition_cell(partition, 0.4) == 2
    assert select_partition_cell(partition, math.nextafter(1.0, 0.0)) == 2

    nonleading_zeros = build_seed_partition(
        (0.1, 0.0, 0.2, 0.3000000000002, 0.4000000000003, 0.0)
    )
    first_cell_end = nonleading_zeros.intervals[0][1]
    assert nonleading_zeros.intervals[1] == pytest.approx(
        (first_cell_end, first_cell_end)
    )
    assert nonleading_zeros.intervals[-1] == pytest.approx((1.0, 1.0))


def test_supplied_seed_returns_one_born_outcome_and_normalized_posterior() -> None:
    plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    state = np.outer(plus, plus.conj())
    probabilities = born_probabilities(PROJECTIVE_OUTCOMES, state)
    assert probabilities == pytest.approx((0.5, 0.5))

    selected_zero = select_outcome(PROJECTIVE_OUTCOMES, state, 0.25)
    selected_one = select_outcome(PROJECTIVE_OUTCOMES, state, 0.75)
    assert selected_zero.label == "zero"
    assert selected_one.label == "one"
    assert selected_zero.raw_born_probability == pytest.approx(0.5)
    assert selected_zero.partition_probability == pytest.approx(0.5)
    assert np.allclose(selected_zero.posterior, P0)
    assert np.allclose(selected_one.posterior, P1)
    assert np.allclose(
        0.5 * selected_zero.posterior + 0.5 * selected_one.posterior,
        apply_nonselective_instrument(PROJECTIVE_OUTCOMES, state),
    )


def test_internal_equal_copy_refinement_preserves_coarse_selection() -> None:
    plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    state = np.outer(plus, plus.conj())
    refined = (
        equal_copy_internal_refinement(PROJECTIVE_OUTCOMES[0], 11),
        PROJECTIVE_OUTCOMES[1],
    )
    assert born_probabilities(refined, state) == pytest.approx((0.5, 0.5))
    # 정확한 누적 경계는 측도 0이며 수치적으로 동등한 크라우스 세분화 아래서
    # 반올림으로 움직일 수 있으므로 내부 점만 비교한다.
    for seed in (0.0, 0.25, 0.5001, math.nextafter(1.0, 0.0)):
        base = select_outcome(PROJECTIVE_OUTCOMES, state, seed)
        changed = select_outcome(refined, state, seed)
        assert changed.label == base.label
        assert changed.raw_born_probability == pytest.approx(base.raw_born_probability)
        assert changed.partition_probability == pytest.approx(
            base.partition_probability
        )
        assert np.allclose(changed.subnormalized_state, base.subnormalized_state)
        assert np.allclose(changed.posterior, base.posterior)


def test_raw_born_and_numerically_normalized_partition_probabilities_are_explicit() -> None:
    delta = 4.0e-13
    near_complete_outcomes = (
        CoarseOutcomeOperation("zero", (math.sqrt(1.0 + delta) * P0,)),
        CoarseOutcomeOperation("one", (P1,)),
    )
    plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    state = np.outer(plus, plus.conj())

    selected = select_outcome(near_complete_outcomes, state, 0.25)
    interval_length = selected.interval[1] - selected.interval[0]
    assert abs(selected.raw_born_probability - selected.partition_probability) > 1.0e-14
    assert interval_length == pytest.approx(selected.partition_probability, abs=1.0e-15)
    assert np.trace(selected.posterior).real == pytest.approx(1.0, abs=1.0e-14)


def test_collision_selection_keeps_zero_branch_out_and_closes_energy_receipts() -> None:
    result = certificate()
    assert result.outcome_labels == ("silent", "left", "right")
    assert result.raw_born_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert result.partition_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert result.seed_intervals[0] == (0.0, 0.0)
    assert result.probe_labels == ("left", "left", "right", "right")
    assert result.collision_operator_energy_ledger_residual < 1.0e-12
    assert result.maximum_supported_branch_relative_energy_residual < 1.0e-12
    assert result.maximum_supported_branch_dimensionless_energy_variance < 1.0e-12
    assert result.status["supplied_collision_operator_energy_ledger_certified"]
    assert result.status["sharp_supported_branch_energy_receipts_certified"]
    assert result.accounting["all_zero_probability_outcomes_not_conditioned"]

    scaled = certificate(energy_scale=7.0)
    assert scaled.energy_scale == pytest.approx(7.0)
    assert scaled.maximum_supported_branch_relative_energy_residual < 1.0e-12
    assert scaled.maximum_supported_branch_dimensionless_energy_variance < 1.0e-12
    assert scaled.status["sharp_supported_branch_energy_receipts_certified"]


def test_uniform_seed_average_is_cptp_and_coarse_refinement_invariant() -> None:
    result = certificate()
    assert result.probability_normalization_residual < 1.0e-12
    assert result.maximum_interval_probability_residual < 1.0e-12
    assert result.maximum_posterior_trace_residual < 1.0e-12
    assert result.seed_average_channel_residual < 1.0e-12
    assert result.completeness_residual < 1.0e-12
    assert result.refinement_operation_residual < 1.0e-12
    assert result.refinement_probability_residual < 1.0e-12
    assert result.refinement_posterior_residual < 1.0e-12
    assert result.refinement_interval_residual < 1.0e-12
    assert result.refinement_same_seed_label_mismatches == 0
    assert result.status["inverse_cdf_born_partition_certified"]
    assert result.status["explicit_collision_instrument_cptp"]
    assert result.status["coarse_selection_internal_refinement_invariant"]


def test_bell_witness_separates_nonselective_locality_from_forced_seed_signal() -> None:
    result = certificate()
    assert result.remote_nonselective_marginal_residual < 1.0e-12
    assert result.forced_seed_remote_trace_distance == pytest.approx(1.0)
    assert result.status["single_local_nonselective_marginal_witness"]
    assert result.status["controllable_seed_signalling_counterexample"]
    assert result.boundaries["forced_seed_is_prohibited_external_intervention"]
    assert not result.status["relativistic_no_signalling_derived"]


def test_fixed_seed_and_scalar_energy_receipt_counterexamples_are_locked() -> None:
    result = certificate()
    assert result.fixed_seed_born_frequency_error == pytest.approx(0.5)
    assert result.status["fixed_seed_born_frequency_counterexample"]
    assert result.x_measurement_best_scalar_receipts == pytest.approx((0.0, 0.0), abs=1.0e-12)
    assert result.x_measurement_relative_frobenius_receipt_residual == pytest.approx(
        1.0 / math.sqrt(2.0)
    )
    assert result.x_measurement_relative_operator_receipt_residual == pytest.approx(0.5)
    assert result.status["general_scalar_energy_receipt_counterexample"]
    assert not result.status["general_measurement_energy_conservation_derived"]


def test_dimension_accounting_axiom_and_claim_ceiling_are_explicit() -> None:
    result = certificate()
    assert all(result.dimensions.values())
    assert result.accounting["probabilities_partition_seed_measure_once"]
    assert result.accounting["weighted_posteriors_equal_nonselective_channel_once"]
    assert result.accounting["unselected_probabilities_not_added_as_energy"]
    assert result.accounting["selected_record_energy_receipt_counted_once"]
    assert not result.accounting["seed_carries_energy"]
    assert result.boundaries[
        "uniform_independent_uncontrollable_seed_is_explicit_axiom"
    ]
    assert result.boundaries["unitary_or_stinespring_does_not_derive_seed"]
    assert result.boundaries["internal_kraus_labels_do_not_enter_seed_partition"]
    assert result.boundaries[
        "finite_refinement_probe_set_excludes_boundary_neighborhoods"
    ]
    assert result.boundaries["same_seed_refinement_claim_limited_to_declared_probe_set"]
    assert all(result.alternatives.values())
    assert not result.status["physical_uniform_seed_law_derived"]
    assert not result.status["objective_single_outcome_selection_derived"]
    assert not result.status["durable_physical_pointer_derived"]
    assert not result.status["spacetime_metric_curvature_or_gravity_derived"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_public_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        build_seed_partition((0.2, 0.2))
    with pytest.raises(ValueError, match="nonnegative"):
        build_seed_partition((-0.1, 1.1))
    partition = build_seed_partition((0.5, 0.5))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        select_partition_cell(partition, 1.0)
    with pytest.raises(ValueError, match="positive integer"):
        equal_copy_internal_refinement(PROJECTIVE_OUTCOMES[0], True)
    with pytest.raises(ValueError, match="complete"):
        born_probabilities((CoarseOutcomeOperation("only", (0.5 * P0,)),), P0)
    with pytest.raises(ValueError, match="left_probability"):
        certificate(left_probability=1.0)
    with pytest.raises(ValueError, match="energy_scale"):
        certificate(energy_scale=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)


def test_run_payload_is_json_serializable_without_promoting_physical_selection() -> None:
    payload = run()
    json.dumps(payload)
    assert payload["status"]["inverse_cdf_born_partition_certified"]
    assert not payload["status"]["objective_single_outcome_selection_derived"]
