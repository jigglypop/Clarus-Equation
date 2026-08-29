from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.causal_record_dust_bridge import (
    CausalRecordNode,
    CausalTransition,
    ExitPhaseMark,
    construct_conserved_record_flow,
    flat_flrw_cauchy_witness,
    match_exit_antichain,
    monokinetic_dust_data,
)
from examples.physics.homogeneous_dust_flrw_propagation import (
    propagate_homogeneous_flat_flrw_dust,
)


def _record_dust_and_initial_witness():
    nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("left", "folded-left", 2.0),
        CausalRecordNode("right", "folded-right", 2.0),
    )
    flow = construct_conserved_record_flow(
        nodes,
        (
            CausalTransition("root", "left", 0.4),
            CausalTransition("root", "right", 0.6),
        ),
        {"root": 5.0},
    )
    matching = match_exit_antichain(
        flow,
        nodes,
        (
            ExitPhaseMark("left", (0.0, 0.0, 0.0), 2.0, (0.0, 0.0, 0.0)),
            ExitPhaseMark("right", (0.5, 0.0, 0.0), 2.0, (0.0, 0.0, 0.0)),
        ),
        cell_volume=1.0,
    )
    dust = monokinetic_dust_data(matching)
    witness = flat_flrw_cauchy_witness(dust, newton_constant=1.0)
    return dust, witness


def test_record_dust_propagates_from_one_cauchy_slice_to_exact_flrw_dust() -> None:
    dust, witness = _record_dust_and_initial_witness()
    result = propagate_homogeneous_flat_flrw_dust(
        dust, witness, evaluation_scale_factor=4.0
    )

    assert result.energy_density == pytest.approx(dust.rest_energy_density / 64.0)
    assert result.rest_number_density == pytest.approx(dust.rest_number_density / 64.0)
    assert result.hubble_rate == pytest.approx(witness.hubble_rate / 8.0)
    assert result.physical_volume == pytest.approx(64.0)
    assert result.conserved_comoving_particle_number == pytest.approx(
        dust.rest_number_density
    )
    assert result.comoving_orthonormal_stress[0][0] == pytest.approx(
        result.energy_density
    )
    assert result.homogeneous_source_free_dust_propagation_closed
    assert result.background_covariant_conservation_closed
    assert result.status == (
        "CONDITIONAL_HOMOGENEOUS_RECORD_DUST_FLRW_PROPAGATION_CLOSED"
    )


@pytest.mark.parametrize("scale_factor", (0.25, 0.5, 1.0, 2.0, 10.0))
def test_exact_dust_scaling_continuity_and_friedmann_constraints(
    scale_factor: float,
) -> None:
    dust, witness = _record_dust_and_initial_witness()
    result = propagate_homogeneous_flat_flrw_dust(
        dust,
        witness,
        evaluation_scale_factor=scale_factor,
        reference_comoving_coordinate_volume=3.0,
    )

    assert result.dimensionless_scale_factor_ratio == scale_factor
    assert result.energy_density * scale_factor**3 == pytest.approx(
        dust.rest_energy_density
    )
    assert result.rest_number_density * scale_factor**3 == pytest.approx(
        dust.rest_number_density
    )
    assert result.comoving_particle_number_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.comoving_rest_energy_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.continuity_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.friedmann_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.raychaudhuri_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.scale_factor_solution_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.pressure == 0.0
    assert result.equation_of_state_parameter == 0.0


def test_elapsed_time_reconstructs_the_expanding_dust_solution() -> None:
    dust, witness = _record_dust_and_initial_witness()
    result = propagate_homogeneous_flat_flrw_dust(
        dust, witness, evaluation_scale_factor=3.0
    )

    dimensionless_time = witness.hubble_rate * result.elapsed_cosmic_time
    reconstructed = (1.0 + 1.5 * dimensionless_time) ** (2.0 / 3.0)

    assert reconstructed == pytest.approx(3.0)
    assert result.acceleration_over_scale_factor == pytest.approx(
        -4.0 * math.pi * witness.newton_constant * result.energy_density / 3.0
    )


def test_nonzero_energy_source_is_outside_the_source_free_theorem() -> None:
    dust, witness = _record_dust_and_initial_witness()

    with pytest.raises(ValueError, match="requires Q=0"):
        propagate_homogeneous_flat_flrw_dust(
            dust,
            witness,
            evaluation_scale_factor=2.0,
            source_energy_transfer_rate_density=0.1,
        )


def test_noncomoving_or_pressured_data_are_rejected() -> None:
    dust, witness = _record_dust_and_initial_witness()
    noncomoving = replace(
        dust,
        energy=math.sqrt(dust.mass**2 + 1.0),
        gamma=math.sqrt(dust.mass**2 + 1.0) / dust.mass,
        four_velocity=(math.sqrt(1.25), 0.5, 0.0, 0.0),
    )
    pressured_stress = tuple(
        tuple(
            0.1 if mu == nu == 1 else dust.stress[mu][nu]
            for nu in range(4)
        )
        for mu in range(4)
    )
    pressured = replace(dust, stress=pressured_stress)

    with pytest.raises(ValueError, match="rest-frame dust|comoving dust"):
        propagate_homogeneous_flat_flrw_dust(
            noncomoving, witness, evaluation_scale_factor=2.0
        )
    with pytest.raises(ValueError, match="pressureless"):
        propagate_homogeneous_flat_flrw_dust(
            pressured, witness, evaluation_scale_factor=2.0
        )


def test_mismatched_initial_constraint_witness_is_rejected() -> None:
    dust, witness = _record_dust_and_initial_witness()
    mismatched = replace(witness, energy_density=0.5 * witness.energy_density)

    with pytest.raises(ValueError, match="must match"):
        propagate_homogeneous_flat_flrw_dust(
            dust, mismatched, evaluation_scale_factor=2.0
        )


def test_background_propagation_does_not_promote_dark_sector_claims() -> None:
    dust, witness = _record_dust_and_initial_witness()
    result = propagate_homogeneous_flat_flrw_dust(
        dust, witness, evaluation_scale_factor=2.0
    )

    assert not result.record_to_renormalized_quantum_stress_derived
    assert not result.dust_readout_selected_by_ce_dynamics
    assert not result.observed_dark_matter_abundance_predicted
    assert not result.vacuum_dark_energy_sector_derived
    assert not result.perturbations_and_structure_growth_derived
    assert not result.ce_specific_independent_observational_prediction_derived
    assert result.claim_ceiling.endswith(
        "NOT_DARK_SECTOR_SELECTION_OR_ABUNDANCE"
    )


@pytest.mark.parametrize(
    "keyword,value,message",
    (
        ("evaluation_scale_factor", 0.0, "scale factor"),
        ("evaluation_scale_factor", math.nan, "scale factor"),
        ("reference_comoving_coordinate_volume", -1.0, "comoving"),
        ("source_energy_transfer_rate_density", math.inf, "must be finite"),
        ("tolerance", 0.0, "tolerance"),
    ),
)
def test_invalid_propagation_inputs_are_rejected(
    keyword: str, value: float, message: str
) -> None:
    dust, witness = _record_dust_and_initial_witness()
    arguments = {"evaluation_scale_factor": 2.0, keyword: value}

    with pytest.raises(ValueError, match=message):
        propagate_homogeneous_flat_flrw_dust(
            dust, witness, **arguments
        )
