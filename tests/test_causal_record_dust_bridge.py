from __future__ import annotations

import math

import pytest

from examples.physics.causal_record_dust_bridge import (
    CausalRecordNode,
    CausalTransition,
    ExitPhaseMark,
    construct_conserved_record_flow,
    flat_flrw_cauchy_witness,
    free_stream_exit_marks,
    match_exit_antichain,
    monokinetic_dust_data,
)


def _rest_record_witness(
    *,
    length_scale: float = 1.0,
):
    energy = 2.0 / length_scale
    nodes = (
        CausalRecordNode("root", "seed", energy),
        CausalRecordNode("left", "folded-left", energy),
        CausalRecordNode("right", "folded-right", energy),
    )
    transitions = (
        CausalTransition("root", "left", 2.0 / 5.0),
        CausalTransition("root", "right", 3.0 / 5.0),
    )
    flow = construct_conserved_record_flow(nodes, transitions, {"root": 5.0})
    marks = (
        ExitPhaseMark("left", (0.1, 0.2, 0.3), energy, (0.0, 0.0, 0.0)),
        ExitPhaseMark("right", (0.7, 0.8, 0.9), energy, (0.0, 0.0, 0.0)),
    )
    matching = match_exit_antichain(
        flow,
        nodes,
        marks,
        cell_volume=length_scale**3,
    )
    return nodes, transitions, flow, marks, matching


def test_marked_causal_kernel_constructs_unique_local_balances() -> None:
    _, _, flow, _, _ = _rest_record_witness()

    assert flow.terminal_composition == (
        ("folded-left", 2.0 / 5.0),
        ("folded-right", 3.0 / 5.0),
    )
    assert flow.initial_number == flow.terminal_number == 5.0
    assert flow.initial_energy == flow.terminal_energy == 10.0
    assert all(balance.number_residual == 0.0 for balance in flow.balances)
    assert all(balance.energy_residual == 0.0 for balance in flow.balances)


def test_exit_pushforward_gives_positive_dust_stress_without_double_counting() -> None:
    _, _, _, _, matching = _rest_record_witness()

    assert matching.current == (5.0, 0.0, 0.0, 0.0)
    assert matching.stress == (
        (10.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    assert matching.residual_energy_density == 10.0
    assert matching.complement_energy_density == 0.0
    assert matching.no_double_counting_residual == 0.0

    dust = monokinetic_dust_data(matching)
    assert dust.gamma == 1.0
    assert dust.surface_number_density == dust.rest_number_density == 5.0
    assert dust.rest_energy_density == 10.0
    assert dust.stress == matching.stress


def test_moving_monokinetic_measure_has_the_exact_dust_normalization() -> None:
    mass = 2.0
    energy = math.sqrt(5.0)
    nodes = (
        CausalRecordNode('root', 'seed', energy),
        CausalRecordNode('left', 'left', energy),
        CausalRecordNode('right', 'right', energy),
    )
    flow = construct_conserved_record_flow(
        nodes,
        (
            CausalTransition('root', 'left', 0.4),
            CausalTransition('root', 'right', 0.6),
        ),
        {'root': 5.0},
    )
    matching = match_exit_antichain(
        flow,
        nodes,
        (
            ExitPhaseMark('left', (0.0, 0.0, 0.0), mass, (1.0, 0.0, 0.0)),
            ExitPhaseMark('right', (0.5, 0.0, 0.0), mass, (1.0, 0.0, 0.0)),
        ),
        cell_volume=1.0,
    )

    dust = monokinetic_dust_data(matching)

    assert math.isclose(
        -dust.four_velocity[0] ** 2 + dust.four_velocity[1] ** 2,
        -1.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(dust.rest_number_density, 10.0 / math.sqrt(5.0))
    assert math.isclose(dust.rest_energy_density, 4.0 * math.sqrt(5.0))
    assert math.isclose(matching.stress[0][0], 5.0 * math.sqrt(5.0))
    assert math.isclose(matching.stress[0][1], 5.0)
    assert math.isclose(matching.stress[1][1], math.sqrt(5.0))


def test_efficiency_is_an_explicit_partition_not_a_second_energy_copy() -> None:
    nodes, _, flow, _, _ = _rest_record_witness()
    marks = (
        ExitPhaseMark(
            "left", (0.1, 0.2, 0.3), 2.0, (0.0, 0.0, 0.0), 0.25
        ),
        ExitPhaseMark(
            "right", (0.7, 0.8, 0.9), 2.0, (0.0, 0.0, 0.0), 0.5
        ),
    )

    matching = match_exit_antichain(flow, nodes, marks, cell_volume=1.0)

    assert math.isclose(matching.residual_energy_density, 4.0)
    assert math.isclose(matching.complement_energy_density, 6.0)
    assert math.isclose(matching.total_energy_density, 10.0)
    assert matching.no_double_counting_residual == 0.0


def test_multistream_data_keep_kinetic_stress_but_do_not_fake_single_dust() -> None:
    nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("slow", "slow", 1.0),
        CausalRecordNode("fast", "fast", 3.0),
    )
    transitions = (
        CausalTransition("root", "slow", 0.5),
        CausalTransition("root", "fast", 0.5),
    )
    flow = construct_conserved_record_flow(nodes, transitions, {"root": 4.0})
    marks = (
        ExitPhaseMark("slow", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0)),
        ExitPhaseMark("fast", (0.5, 0.0, 0.0), 3.0, (0.0, 0.0, 0.0)),
    )
    matching = match_exit_antichain(flow, nodes, marks, cell_volume=1.0)

    assert matching.residual_energy_density == 8.0
    assert matching.current[0] == 4.0
    with pytest.raises(ValueError, match="multi-stream or multi-mass"):
        monokinetic_dust_data(matching)


def test_flat_flrw_witness_satisfies_both_initial_constraints() -> None:
    _, _, _, _, matching = _rest_record_witness()
    witness = flat_flrw_cauchy_witness(
        monokinetic_dust_data(matching),
        newton_constant=1.0,
    )

    assert math.isclose(
        witness.hubble_rate**2,
        8.0 * math.pi * witness.energy_density / 3.0,
        rel_tol=1.0e-15,
    )
    assert math.isclose(witness.hamiltonian_residual, 0.0, abs_tol=1.0e-12)
    assert witness.momentum_residual == (0.0, 0.0, 0.0)


def test_reference_length_scaling_has_the_declared_dimensions() -> None:
    _, _, _, _, unit_matching = _rest_record_witness(length_scale=1.0)
    _, _, _, _, doubled_matching = _rest_record_witness(length_scale=2.0)

    # [J] = L^-3 and [T] = [rho] = L^-4.
    assert math.isclose(
        doubled_matching.current[0], unit_matching.current[0] / 2.0**3
    )
    assert math.isclose(
        doubled_matching.stress[0][0], unit_matching.stress[0][0] / 2.0**4
    )

    unit_witness = flat_flrw_cauchy_witness(
        monokinetic_dust_data(unit_matching),
        newton_constant=1.0,
    )
    doubled_witness = flat_flrw_cauchy_witness(
        monokinetic_dust_data(doubled_matching),
        newton_constant=2.0**2,
    )
    assert math.isclose(
        doubled_witness.hubble_rate,
        unit_witness.hubble_rate / 2.0,
    )


def test_free_liouville_pushforward_commutes_with_spatial_translation() -> None:
    mark = ExitPhaseMark("exit", (0.1, 0.2, 0.3), 2.0, (1.0, 0.0, 0.0))
    shift = (0.17, 0.23, 0.31)
    translated = ExitPhaseMark(
        "exit",
        tuple((x + dx) % 1.0 for x, dx in zip(mark.position, shift)),
        mark.mass,
        mark.spatial_momentum,
    )

    streamed = free_stream_exit_marks((mark,), coordinate_time=0.4, box_length=1.0)[0]
    translated_streamed = free_stream_exit_marks(
        (translated,), coordinate_time=0.4, box_length=1.0
    )[0]

    assert all(
        math.isclose((x + dx) % 1.0, shifted_x, abs_tol=1.0e-15)
        for x, dx, shifted_x in zip(
            streamed.position,
            shift,
            translated_streamed.position,
        )
    )
    assert streamed.four_momentum == translated_streamed.four_momentum


def test_bridge_rejects_unnormalized_nonconserving_or_mistyped_inputs() -> None:
    nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("left", "left", 2.0),
        CausalRecordNode("right", "right", 2.0),
    )
    with pytest.raises(ValueError, match="sum to one"):
        construct_conserved_record_flow(
            nodes,
            (
                CausalTransition("root", "left", 0.2),
                CausalTransition("root", "right", 0.3),
            ),
            {"root": 1.0},
        )

    energy_mismatch_nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("left", "left", 1.0),
        CausalRecordNode("right", "right", 2.0),
    )
    with pytest.raises(ValueError, match="harmonic"):
        construct_conserved_record_flow(
            energy_mismatch_nodes,
            (
                CausalTransition("root", "left", 0.5),
                CausalTransition("root", "right", 0.5),
            ),
            {"root": 1.0},
        )

    _, _, flow, _, _ = _rest_record_witness()
    with pytest.raises(ValueError, match="mass-shell energy"):
        match_exit_antichain(
            flow,
            nodes,
            (
                ExitPhaseMark("left", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0)),
                ExitPhaseMark("right", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0)),
            ),
            cell_volume=1.0,
        )
