from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.quantum_selection_action_routes import (
    MINKOWSKI,
    action_first_hybrid_witness,
    certificate,
    controlled_flux_storage_witness,
    dephasing_trajectory_counterexample,
    feshbach_phase_counterexample,
    gram_minors,
    stinespring_zero_energy_counterexample,
    time_varying_vacuum_nonconservation,
)


def test_feshbach_same_q_can_have_different_energy_expectation() -> None:
    result = feshbach_phase_counterexample(phase_a=0.0, phase_b=math.pi / 2.0)
    assert result["same_q"] and result["same_hamiltonian"]
    assert result["phase_is_dimensionless"]
    assert result["different_energy_expectation"]


def test_complete_stinespring_record_can_have_no_energy() -> None:
    result = stinespring_zero_energy_counterexample(q=0.41)
    assert result["orthogonal_records"] and result["hamiltonian_zero"]
    assert result["isometry_norm"] == pytest.approx(1.0)
    assert result["energy_expectation"] == pytest.approx(0.0)


def test_same_dephasing_channel_has_nonunique_zero_energy_trajectories() -> None:
    result = dephasing_trajectory_counterexample()
    assert result["same_cptp_channel"]
    assert result["different_trajectory_labels"]
    assert result["energy_expectation_each_trajectory"] == 0.0


def test_locked_born_weight_is_stored_by_energy_conserving_controlled_swap() -> None:
    result = controlled_flux_storage_witness(
        interaction_angle=math.pi / 3.0, energy_star=7.0
    )
    assert result["p_mobile"] == pytest.approx(0.75)
    assert result["q_locked"] == pytest.approx(0.25)
    assert result["flux_occupation"] == pytest.approx(0.25)
    assert result["flux_energy_expectation"] == pytest.approx(1.75)
    assert result["initial_total_energy"] == pytest.approx(7.0)
    assert result["final_total_energy"] == pytest.approx(7.0)
    assert result["one_receipt_no_double_count"]
    assert result["finite_branch_to_storage_closed"]
    assert result["receipt_interpretation"]["ensemble_expectation_partition"]
    assert not result["receipt_interpretation"]["per_run_two_simultaneous_energy_receipts"]
    assert not result["fail_closed"]["universal_quantum_storage_law_proved"]


def test_flux_storage_distinguishes_unconditional_and_branch_local_readout() -> None:
    result = controlled_flux_storage_witness(interaction_angle=math.pi / 4.0)
    boundary = result["conditioning_boundary"]
    assert boundary["unconditional_flux_occupation"] == pytest.approx(0.5)
    assert boundary["mobile_branch_conditional_flux_occupation"] == 0.0
    assert not boundary["branch_local_gravity_contains_nonselected_flux"]
    assert not boundary["unconditioned_semiclassical_gravity_rule_derived"]
    assert not result["fail_closed"]["finite_flux_bit_is_covariant_four_form"]


def test_time_varying_vacuum_weight_is_not_conserved() -> None:
    result = time_varying_vacuum_nonconservation(q_dot=0.2, energy_density_scale=3.0)
    assert not result["conserved"]
    assert result["nabla_mu_T_mu_nu"][0] == pytest.approx(0.6)
    assert np.allclose(result["T_covariant"], -result["rho"] * MINKOWSKI)


def test_relation_rods_have_three_positive_gram_minors() -> None:
    minors = gram_minors(((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 5.0)))
    assert minors["D1"] == pytest.approx(4.0)
    assert minors["D2"] == pytest.approx(36.0)
    assert minors["D3"] == pytest.approx(900.0)


def test_action_first_hybrid_has_one_receipt_dust_and_four_form_vacuum() -> None:
    result = action_first_hybrid_witness(q_mobile=0.3, energy_star=10.0, cell_volume=2.0)
    assert result["three_direction_witness"] and result["no_double_count"]
    assert result["receipt_interpretation"]["ensemble_expectation_partition"]
    assert not result["receipt_interpretation"]["per_run_two_simultaneous_energy_receipts"]
    assert result["mobile_receipt_energy"] == pytest.approx(3.0)
    assert result["locked_receipt_energy"] == pytest.approx(7.0)
    assert result["mobile_massive_receipt"]["T_covariant"][0][0] == pytest.approx(1.5)
    assert result["mobile_massive_receipt"]["component_frame"] == "local comoving orthonormal frame"
    assert not result["mobile_massive_receipt"]["flrw_continuity_law_derived_by_this_finite_witness"]
    assert np.allclose(result["locked_four_form"]["T_covariant"], -3.5 * MINKOWSKI)
    assert result["locked_four_form"]["constant_flux_conserved"]
    assert "matching reference volume" in result["locked_four_form"]["cell_volume_role"]
    assert result["fail_closed"]["born_q_does_not_dynamically_derive_covariant_four_form"]


def test_hybrid_rejects_degenerate_relation_rods() -> None:
    with pytest.raises(ValueError, match="independent rods"):
        action_first_hybrid_witness(rods=((1, 0, 0), (2, 0, 0), (3, 0, 0)))


def test_certificate_is_fail_closed_about_the_open_physical_bridge() -> None:
    result = certificate()
    assert result["status"] == "MAJOR_ALTERNATIVE_ROUTES_AUDITED_FINITE_FLUX_STORAGE_CLOSED_COVARIANT_BRIDGE_OPEN"
    assert all(result["counterexamples_closed"].values())
    assert result["finite_branch_to_flux_storage_closed"]
    assert result["finite_storage_to_action_partition_consistent"]
    assert result["joint_probability_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["joint_energy_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["conditional_action_first_hybrid_closed"]
    assert not result["unconditional_quantum_to_dark_sector_proved"]
