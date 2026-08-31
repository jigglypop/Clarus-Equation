from __future__ import annotations

import pytest

from examples.physics.causal_probability_deformation_lattice import (
    certify_causal_probability_deformation_lattice,
)


def test_cfl_one_support_and_first_detector_arrival_are_causal() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.support_violation == 0.0
    assert certificate.first_nonzero_detector_sample == certificate.expected_first_detector_sample == 4
    assert certificate.source_detector_chi[:4] == (0.0, 0.0, 0.0, 0.0)
    assert certificate.source_detector_chi[4] == pytest.approx(0.8)
    assert certificate.finite_lattice_causal_front_witness
    assert certificate.front_speed == pytest.approx(certificate.light_speed)


def test_probability_response_waits_for_arrival_then_differs() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.prearrival_probability_difference == 0.0
    assert certificate.source_probabilities[:4] == certificate.control_probabilities[:4]
    assert certificate.postarrival_probability_difference > 1.0e-6


def test_source_and_coupling_negative_controls_are_inert() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.source_off_probability_difference == 0.0
    assert certificate.coupling_off_probability_difference == 0.0
    source_off = certify_causal_probability_deformation_lattice(source_amplitude=0.0)
    coupling_off = certify_causal_probability_deformation_lattice(coupling_energy=0.0)
    assert source_off.source_probabilities == source_off.control_probabilities
    assert coupling_off.source_probabilities == coupling_off.control_probabilities


def test_local_channel_is_unitary_cptp_and_trace_preserving() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.local_unitary_residual < 1.0e-14
    assert certificate.local_trace_residual < 1.0e-14
    assert certificate.local_choi_minimum_eigenvalue >= -1.0e-13


def test_rn_separation_dimensions_and_false_claim_ceiling_are_explicit() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert not certificate.rn_reweighting_used
    assert certificate.chi_dimensionless
    assert certificate.source_q_dimensionless
    assert certificate.continuum_source_s_length_power == -2
    assert certificate.coupling_g_is_energy
    assert certificate.dt_hamiltonian_over_hbar_dimensionless
    assert certificate.dimensions_pass
    assert not any((
        certificate.mass_to_q_derived,
        certificate.energy_current_or_backreaction_derived,
        certificate.probability_deformation_equals_attraction_derived,
        certificate.continuous_qft_microcausality_derived,
        certificate.gr_or_lensing_derived,
        certificate.repeated_measurement_or_physical_selection_derived,
        certificate.observational_holdout_derived,
        certificate.gates_5_to_8_closed,
        certificate.two_residuals_reduced,
        certificate.complexity_success,
    ))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lattice_spacing": 0.0}, "lattice_spacing"),
        ({"light_speed": 0.0}, "light_speed"),
        ({"omega": 0.0}, "omega"),
        ({"hbar": 0.0}, "hbar"),
        ({"detector_distance_cells": -1}, "detector_distance_cells"),
        ({"time_steps": 3}, "time_steps"),
        ({"grid_radius_cells": 5}, "grid_radius_cells"),
    ],
)
def test_horizon_and_numeric_contract_fails_closed(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        certify_causal_probability_deformation_lattice(**kwargs)
