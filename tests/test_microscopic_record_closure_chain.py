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
    DarkReadoutChoice,
    LayeredCausalEdge,
    LayeredRecordEvent,
    ScalarLatticeState,
    certify_layered_cauchy_embedding,
    flat_flrw_scalar_cauchy_witness,
    match_constant_vacuum_on_cauchy_slice,
    rescale_scalar_to_total_energy,
    select_single_dark_readout,
    split_identical_scalar_species,
)
from examples.physics.quantum_instrument_record_kernel import (
    KrausBranch,
    RecordInstrument,
    build_energy_resolved_instrument_tree,
    construct_luders_energy_instrument,
)


def test_qnd_record_to_cauchy_and_three_alternative_dark_readouts() -> None:
    hamiltonian = 5.0 * np.eye(2, dtype=np.complex128)
    energy_only = construct_luders_energy_instrument(hamiltonian)
    assert energy_only.spectral_energies == (5.0,)
    assert energy_only.spectral_multiplicities == (2,)

    # The coarsest energy PVM is sharp but cannot select a position inside its
    # degenerate sector, so the downstream genealogy remains a supplied input.
    left_projector = np.diag([1.0, 0.0]).astype(np.complex128)
    right_projector = np.diag([0.0, 1.0]).astype(np.complex128)
    kernel = build_energy_resolved_instrument_tree(
        hamiltonian,
        np.diag([0.4, 0.6]),
        (
            RecordInstrument(
                'root',
                'declared-degenerate-position-pointer',
                (
                    KrausBranch('left', 'left exit', left_projector),
                    KrausBranch('right', 'right exit', right_projector),
                ),
            ),
        ),
        initial_weight=5.0,
    )
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
        exit_nodes=kernel.terminal_nodes,
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        spatial_shape=(2, 1, 1),
    )
    marks = tuple(
        ExitPhaseMark(
            node,
            embedding.coordinate(node)[1:],
            5.0,
            (0.0, 0.0, 0.0),
            0.4,
        )
        for node in kernel.terminal_nodes
    )
    kinetic = match_exit_antichain(
        kernel.flow,
        kernel.nodes,
        marks,
        cell_volume=embedding.spatial_volume,
    )
    dust = monokinetic_dust_data(kinetic)
    dust_gr = flat_flrw_cauchy_witness(
        dust,
        newton_constant=1.0,
    )

    assert math.isclose(kernel.flow.terminal_energy, 25.0)
    assert math.isclose(kinetic.residual_energy_density, 5.0)
    assert math.isclose(kinetic.complement_energy_density, 7.5)
    assert math.isclose(kinetic.total_energy_density, 12.5)
    assert math.isclose(dust.rest_energy_density, 5.0)
    assert math.isclose(dust_gr.hamiltonian_residual, 0.0, abs_tol=1.0e-12)

    scalar_template = ScalarLatticeState(
        field=np.ones((2, 1, 1)),
        momentum=np.zeros((2, 1, 1)),
        spacing=1.0,
        mass=1.0,
    )
    scalar_source = rescale_scalar_to_total_energy(
        scalar_template,
        target_energy=kernel.flow.terminal_energy,
    )
    scalar_split = split_identical_scalar_species(
        scalar_source,
        residual_efficiency=0.4,
    )
    scalar_gr = flat_flrw_scalar_cauchy_witness(
        scalar_split.residual,
        newton_constant=1.0,
    )
    assert math.isclose(scalar_split.residual_energy_after, 10.0)
    assert math.isclose(scalar_split.complement_energy_after, 15.0)
    assert math.isclose(scalar_gr.energy_density, dust.rest_energy_density)
    assert math.isclose(
        scalar_gr.hubble_rate,
        dust_gr.hubble_rate,
        rel_tol=1.0e-15,
    )

    vacuum = match_constant_vacuum_on_cauchy_slice(
        battery_energy=kernel.flow.terminal_energy,
        spatial_volume=embedding.spatial_volume,
        residual_efficiency=0.4,
    )
    assert math.isclose(vacuum.transferred_energy, 10.0)
    assert math.isclose(vacuum.complement_energy_after, 15.0)
    assert math.isclose(vacuum.vacuum_density, dust.rest_energy_density)
    assert vacuum.no_double_counting_residual == 0.0

    selected = select_single_dark_readout(
        DarkReadoutChoice.SCALAR,
        dust_stress=np.asarray(kinetic.stress),
        scalar_stress=scalar_split.residual_stress_after,
        vacuum_stress=vacuum.vacuum_stress,
    )
    simultaneous_sum = (
        np.asarray(kinetic.stress)
        + scalar_split.residual_stress_after
        + vacuum.vacuum_stress
    )
    assert selected.choice is DarkReadoutChoice.SCALAR
    assert math.isclose(selected.energy_density, 5.0)
    assert not np.allclose(selected.stress, simultaneous_sum)
