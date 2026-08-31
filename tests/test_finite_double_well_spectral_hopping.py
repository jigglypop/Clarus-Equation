from __future__ import annotations

import math

import pytest

from examples.physics.finite_double_well_spectral_hopping import (
    certify_finite_double_well_spectral_hopping,
)


def _certificate(**overrides: object):
    inputs: dict[str, object] = {
        "nonrelativistic_mass": 1.0,
        "barrier_height": 30.0,
        "total_barrier_width": 1.0,
        "well_width": 1.0,
        "scattering_energy": 2.0,
    }
    inputs.update(overrides)
    return certify_finite_double_well_spectral_hopping(**inputs)


def test_contract_fails_closed_for_inputs_scattering_and_safe_root_domain() -> None:
    with pytest.raises(ValueError, match="well_width"):
        _certificate(well_width=0.0)
    with pytest.raises(ValueError, match="scattering_energy"):
        _certificate(scattering_energy=30.0)
    with pytest.raises(ValueError, match="nu > pi\\^2"):
        _certificate(barrier_height=1.0, scattering_energy=0.5)


def test_parity_roots_have_certified_signs_order_and_energy_intervals() -> None:
    certificate = _certificate()
    assert certificate.even_endpoint_values[0] > 0.0 > certificate.even_endpoint_values[1]
    assert certificate.odd_endpoint_values[0] > 0.0 > certificate.odd_endpoint_values[1]
    assert certificate.even_z_bracket[1] < certificate.odd_z_bracket[0]
    assert certificate.spectral_order_holds
    assert certificate.even_z_bracket[0] < certificate.even_z < certificate.even_z_bracket[1]
    assert certificate.odd_z_bracket[0] < certificate.odd_z < certificate.odd_z_bracket[1]
    assert abs(certificate.even_root_residual) < 1.0e-10
    assert abs(certificate.odd_root_residual) < 1.0e-10
    assert certificate.even_energy_interval[1] < certificate.odd_energy_interval[0]
    assert certificate.hopping_interval[0] > 0.0
    assert certificate.hopping_interval[0] <= certificate.hopping <= certificate.hopping_interval[1]
    assert certificate.ground_energy == pytest.approx(3.8496174065135653, abs=2.0e-12)
    assert certificate.first_excited_energy == pytest.approx(3.8519766342946045, abs=2.0e-12)
    assert certificate.hopping == pytest.approx(0.0011796138905195708, abs=2.0e-12)


def test_modes_are_numerical_witnesses_with_bias_not_exact_localization() -> None:
    certificate = _certificate()
    assert certificate.right_mode_norm_witness == pytest.approx(1.0, abs=2.0e-8)
    assert certificate.left_mode_norm_witness == pytest.approx(1.0, abs=2.0e-8)
    assert certificate.right_left_overlap_witness == pytest.approx(0.0, abs=2.0e-8)
    assert 0.5 < certificate.right_mode_right_probability_witness < 1.0
    assert 0.5 < certificate.left_mode_left_probability_witness < 1.0
    assert certificate.maximum_join_residual < 1.0e-9
    assert not certificate.exact_spatial_localization_derived


def test_spectral_hamiltonian_and_swap_are_exact_only_inside_prepared_pair() -> None:
    certificate = _certificate()
    assert certificate.spectral_hamiltonian[0, 0] == pytest.approx(certificate.mean_energy)
    assert certificate.spectral_hamiltonian[0, 1] == pytest.approx(-certificate.hopping)
    assert certificate.spectral_hamiltonian == pytest.approx(certificate.spectral_hamiltonian.T)
    assert certificate.ideal_swap_time == pytest.approx(math.pi / (2.0 * certificate.hopping))
    assert certificate.spectral_swap_phase_residual < 1.0e-11
    assert certificate.finite_double_well_spectrum_to_J_derived
    assert certificate.prepared_exact_spectral_pair_invariant_by_construction
    assert not certificate.arbitrary_continuum_preparation_projects_to_subspace


def test_same_open_scattering_t_but_width_dependent_spectrum_forbids_t_to_j() -> None:
    certificates = [_certificate(well_width=width) for width in (0.8, 1.0, 1.2)]
    transmissions = [item.auxiliary_scattering_transmission for item in certificates]
    assert transmissions == pytest.approx([transmissions[0]] * 3, rel=0.0, abs=0.0)
    intervals = [item.hopping_interval for item in certificates]
    assert intervals[0][1] < intervals[1][0] or intervals[1][1] < intervals[0][0]
    assert intervals[1][1] < intervals[2][0] or intervals[2][1] < intervals[1][0]
    assert len({round(item.hopping, 10) for item in certificates}) == 3
    assert not any(item.transmission_to_hopping_derived or item.wkb_to_hopping_derived for item in certificates)


def test_dimensions_and_all_unclosed_bridges_remain_false() -> None:
    certificate = _certificate()
    assert certificate.dimensions_pass
    assert certificate.wavefunction_mass_dimension == 0.5
    assert certificate.hopping_mass_dimension == 1
    assert certificate.time_mass_dimension == -1
    false_flags = (
        certificate.transmission_to_hopping_derived,
        certificate.wkb_to_hopping_derived,
        certificate.exact_spatial_localization_derived,
        certificate.e15_material_lattice_embedding_derived,
        certificate.periodic_or_n_chain_derived,
        certificate.arbitrary_continuum_preparation_projects_to_subspace,
        certificate.scattering_instrument_or_energy_receipt_derived,
        certificate.cptp_or_fresh_ancilla_derived,
        certificate.causal_c_front_derived,
        certificate.qft_microcausality_or_gr_derived,
        certificate.selection_or_residual_explanation_derived,
        certificate.gates_5_to_8_closed,
    )
    assert not any(false_flags)
