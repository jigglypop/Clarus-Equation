from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.lattice_scalar_transition_bridge import (
    LayeredCausalEdge,
    LayeredRecordEvent,
    ScalarLatticeState,
    cell_averaged_scalar_stress,
    certify_layered_cauchy_embedding,
    continuum_mode_frequency,
    flat_flrw_scalar_cauchy_witness,
    lattice_mode_frequency,
    match_constant_vacuum_on_cauchy_slice,
    rescale_scalar_to_total_energy,
    scalar_lattice_energy,
    split_identical_scalar_species,
)


def test_layered_lattice_certifies_a_causal_exit_antichain() -> None:
    events = (
        LayeredRecordEvent('root', 0, (0, 0, 0)),
        LayeredRecordEvent('left', 1, (0, 0, 0)),
        LayeredRecordEvent('right', 1, (1, 0, 0)),
    )
    certificate = certify_layered_cauchy_embedding(
        events,
        (
            LayeredCausalEdge('root', 'left'),
            LayeredCausalEdge('root', 'right'),
        ),
        exit_nodes=('left', 'right'),
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        spatial_shape=(4, 4, 4),
    )

    assert certificate.exit_time == 1.0
    assert certificate.coordinate('left') == (1.0, 0.0, 0.0, 0.0)
    assert certificate.coordinate('right') == (1.0, 1.0, 0.0, 0.0)
    assert certificate.minimum_causal_margin == 0.0
    assert certificate.spatial_volume == 64.0


def test_layered_lattice_rejects_a_spacelike_declared_edge() -> None:
    with pytest.raises(ValueError, match='spacelike'):
        certify_layered_cauchy_embedding(
            (
                LayeredRecordEvent('root', 0, (0, 0, 0)),
                LayeredRecordEvent('exit', 1, (2, 0, 0)),
            ),
            (LayeredCausalEdge('root', 'exit'),),
            exit_nodes=('exit',),
            lattice_spacing=1.0,
            clock_step=1.0,
            causal_speed=1.0,
            spatial_shape=(6, 2, 2),
        )


def _inhomogeneous_scalar_state(
    *,
    length_scale: float = 1.0,
) -> ScalarLatticeState:
    field = np.arange(24, dtype=np.float64).reshape(3, 2, 4) / 17.0
    momentum = np.cos(np.arange(24, dtype=np.float64)).reshape(3, 2, 4) / 5.0
    return ScalarLatticeState(
        field=field / length_scale,
        momentum=momentum / length_scale**2,
        spacing=0.4 * length_scale,
        mass=0.7 / length_scale,
    )


def test_species_rotation_exactly_matches_total_stress_and_eta_partition() -> None:
    source = _inhomogeneous_scalar_state()
    matching = split_identical_scalar_species(
        source,
        residual_efficiency=0.37,
    )

    assert np.allclose(matching.total_stress_residual, 0.0, atol=1.0e-12)
    assert np.allclose(
        matching.residual_stress_after,
        0.37 * matching.source_stress_before,
        atol=1.0e-12,
    )
    assert np.allclose(
        matching.complement_stress_after,
        0.63 * matching.source_stress_before,
        atol=1.0e-12,
    )
    assert math.isclose(matching.total_energy_residual, 0.0, abs_tol=1.0e-12)
    assert math.isclose(
        matching.residual_energy_after,
        0.37 * matching.source_energy_before,
        rel_tol=1.0e-12,
    )


def test_record_energy_fixes_scalar_amplitude_before_the_stress_split() -> None:
    template = _inhomogeneous_scalar_state()
    source = rescale_scalar_to_total_energy(template, target_energy=25.0)
    matching = split_identical_scalar_species(
        source,
        residual_efficiency=0.4,
    )

    assert math.isclose(scalar_lattice_energy(source), 25.0, rel_tol=1.0e-12)
    assert math.isclose(matching.residual_energy_after, 10.0, rel_tol=1.0e-12)
    assert math.isclose(matching.complement_energy_after, 15.0, rel_tol=1.0e-12)
    assert math.isclose(matching.total_energy_residual, 0.0, abs_tol=1.0e-12)


def test_nearest_neighbor_scalar_dispersion_has_second_order_kg_limit() -> None:
    wave_vector = (0.7, -0.4, 0.2)
    mass = 0.9
    continuum = continuum_mode_frequency(wave_vector, mass=mass)
    coarse_error = abs(
        lattice_mode_frequency(wave_vector, spacing=0.2, mass=mass)
        - continuum
    )
    fine_error = abs(
        lattice_mode_frequency(wave_vector, spacing=0.1, mass=mass)
        - continuum
    )

    assert fine_error < coarse_error / 3.9
    assert math.isclose(
        lattice_mode_frequency((0.0, 0.0, 0.0), spacing=0.1, mass=mass),
        mass,
    )


def test_homogeneous_scalar_residual_satisfies_flat_flrw_constraints() -> None:
    state = ScalarLatticeState(
        field=np.full((2, 2, 2), 1.5),
        momentum=np.full((2, 2, 2), 0.7),
        spacing=0.4,
        mass=0.3,
    )
    witness = flat_flrw_scalar_cauchy_witness(
        state,
        newton_constant=1.0,
    )

    assert math.isclose(witness.energy_density, 0.34625)
    assert all(math.isclose(value, 0.14375) for value in witness.pressure_diagonal)
    assert math.isclose(
        witness.hubble_rate**2,
        8.0 * math.pi * witness.energy_density / 3.0,
        rel_tol=1.0e-15,
    )
    assert math.isclose(witness.hamiltonian_residual, 0.0, abs_tol=1.0e-12)
    assert witness.momentum_residual == (0.0, 0.0, 0.0)


def test_scalar_lattice_scaling_has_energy_and_stress_dimensions() -> None:
    unit = _inhomogeneous_scalar_state(length_scale=1.0)
    doubled = _inhomogeneous_scalar_state(length_scale=2.0)

    assert np.allclose(
        cell_averaged_scalar_stress(doubled),
        cell_averaged_scalar_stress(unit) / 2.0**4,
    )
    assert math.isclose(
        scalar_lattice_energy(doubled),
        scalar_lattice_energy(unit) / 2.0,
    )


def test_constant_vacuum_channel_has_exact_equal_gap_and_covariant_stress() -> None:
    matching = match_constant_vacuum_on_cauchy_slice(
        battery_energy=25.0,
        spatial_volume=10.0,
        residual_efficiency=0.4,
    )

    assert matching.transferred_energy == 10.0
    assert matching.complement_energy_after == 15.0
    assert matching.vacuum_density == 1.0
    assert np.array_equal(matching.vacuum_stress, np.diag([1.0, -1.0, -1.0, -1.0]))
    assert matching.no_double_counting_residual == 0.0
    assert matching.energy_preserving_rotation_commutator_residual == 0.0
    assert math.isclose(matching.battery_energy_after, 15.0, abs_tol=1.0e-12)
    assert math.isclose(
        matching.vacuum_register_energy_after,
        10.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        matching.unitary_total_energy_residual,
        0.0,
        abs_tol=1.0e-12,
    )


def test_flat_flrw_scalar_witness_rejects_inhomogeneous_data() -> None:
    with pytest.raises(ValueError, match='homogeneous'):
        flat_flrw_scalar_cauchy_witness(
            _inhomogeneous_scalar_state(),
            newton_constant=1.0,
        )
