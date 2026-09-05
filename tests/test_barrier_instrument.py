"""유한 장벽 계측기 묶음(E16 누설·E17 스펙트럼 도약·단일 에너지 두 포트 계측기) 테스트."""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.record.barrier_instrument import (
    certify_finite_barrier_mode_leakage,
    certify_finite_double_well_spectral_hopping,
    certify_single_energy_barrier_instrument,
    exact_rectangular_barrier_transmission_probability,
    gaussian_overlap_amplitude,
)


def _certificate(**overrides: object):
    inputs: dict[str, object] = {
        "mode_count": 3,
        "sigma": 1.0,
        "center_spacing": 5.0,
        "delta_basis": 0.01,
        "nonrelativistic_mass": 1.0,
        "barrier_height": 8.0,
        "incident_energy": 2.0,
        "barrier_width": 2.0,
        "delta_leak": 0.01,
        "ideal_projected_hopping": 2.0,
        "projected_hamiltonian_norm_error": 0.001,
        "delta_dyn": 0.01,
    }
    inputs.update(overrides)
    return certify_finite_barrier_mode_leakage(**inputs)


def test_gaussian_convention_overlap_and_required_spacing_are_amplitudes() -> None:
    assert gaussian_overlap_amplitude(sigma=2.0, center_spacing=4.0) == pytest.approx(math.exp(-1.0))
    certificate = _certificate()
    assert certificate.gaussian_overlap_amplitude == pytest.approx(math.exp(-25.0 / 4.0))
    assert certificate.basis_amplitude_budget == pytest.approx(3.0 * math.exp(-25.0 / 4.0))
    assert certificate.required_center_spacing == pytest.approx(2.0 * math.sqrt(math.log(300.0)))
    assert certificate.basis_amplitude_budget_holds
    boundary = _certificate(center_spacing=certificate.required_center_spacing)
    assert boundary.basis_amplitude_budget == pytest.approx(
        boundary.basis_amplitude_target, rel=1.0e-12
    )
    deep_separation = _certificate(center_spacing=100.0)
    assert math.isfinite(deep_separation.gaussian_log_overlap_amplitude)
    assert math.isfinite(deep_separation.basis_log_amplitude_budget)
    assert deep_separation.gaussian_overlap_amplitude == 0.0
    assert deep_separation.gaussian_overlap_numerically_underflowed
    assert deep_separation.basis_amplitude_budget_holds


def test_exact_barrier_probability_and_exponential_certificate_are_separate() -> None:
    certificate = _certificate()
    kappa = math.sqrt(12.0)
    expected = 1.0 / (1.0 + 64.0 / 48.0 * math.sinh(2.0 * kappa) ** 2)
    assert exact_rectangular_barrier_transmission_probability(
        nonrelativistic_mass=1.0, barrier_height=8.0, incident_energy=2.0, barrier_width=2.0
    ) == pytest.approx(expected)
    assert certificate.barrier_transmission_probability == pytest.approx(expected)
    assert certificate.exponential_regime_holds
    assert certificate.exponential_probability_upper is not None
    assert certificate.barrier_transmission_probability <= certificate.exponential_probability_upper
    assert certificate.exact_required_barrier_width > 0.0
    assert certificate.exponential_required_barrier_width > 0.0
    exact_boundary = _certificate(barrier_width=certificate.exact_required_barrier_width)
    assert exact_boundary.barrier_probability_budget == pytest.approx(
        exact_boundary.barrier_probability_target, rel=1.0e-12
    )
    exponential_boundary = _certificate(
        barrier_width=certificate.exponential_required_barrier_width
    )
    assert exponential_boundary.exponential_regime_holds
    assert exponential_boundary.exponential_probability_upper is not None
    assert (
        3.0 * exponential_boundary.exponential_probability_upper
        <= exponential_boundary.barrier_probability_target * (1.0 + 1.0e-12)
    )


def test_below_regime_and_deep_barrier_stay_finite_in_the_log_domain() -> None:
    below_regime = _certificate(barrier_width=0.01)
    assert not below_regime.exponential_regime_holds
    assert below_regime.exponential_probability_upper is None
    assert 0.0 < below_regime.barrier_transmission_probability < 1.0
    deep = _certificate(
        barrier_width=1.0e6,
        delta_leak=1.0e-300,
    )
    assert math.isfinite(deep.barrier_log_transmission_probability)
    assert math.isfinite(deep.exact_required_barrier_width)
    assert math.isfinite(deep.exponential_required_barrier_width)
    assert deep.barrier_transmission_probability >= 0.0
    assert deep.barrier_probability_numerically_underflowed


def test_budgets_have_distinct_types_and_are_never_aggregated() -> None:
    certificate = _certificate()
    assert certificate.error_type_tuple == (
        "basis_amplitude", "barrier_probability", "projected_operator_norm"
    )
    assert certificate.basis_amplitude_budget_holds
    assert certificate.barrier_probability_budget_holds
    assert certificate.projected_operator_norm_budget_holds
    assert not hasattr(certificate, "aggregate_error")


def test_ideal_swap_and_finite_duhamel_telescoping_witness() -> None:
    exact = _certificate(projected_hamiltonian_norm_error=0.0)
    assert exact.ideal_swap_time == pytest.approx(math.pi / 4.0)
    assert exact.ideal_swap_probability == pytest.approx(1.0)
    assert exact.ideal_unitarity_residual < 1.0e-12
    assert exact.ideal_swap_phase_residual < 1.0e-12
    assert exact.single_step_operator_difference < 1.0e-12
    assert exact.repeated_step_operator_difference < 1.0e-12
    perturbed = _certificate(projected_hamiltonian_norm_error=0.01)
    assert perturbed.single_step_operator_difference <= perturbed.single_step_duhamel_bound_raw + 1.0e-11
    assert perturbed.repeated_step_operator_difference <= perturbed.repeated_step_telescoping_bound_raw + 1.0e-11
    assert perturbed.single_step_duhamel_bound_clipped <= 2.0
    assert perturbed.repeated_step_telescoping_bound_clipped <= 2.0


def test_dimensions_and_scope_flags_are_explicitly_limited() -> None:
    certificate = _certificate()
    assert certificate.dimensions_pass
    assert certificate.sigma_mass_dimension == -1
    assert certificate.energy_mass_dimension == 1
    assert certificate.kappa_mass_dimension == 1
    assert certificate.barrier_width_mass_dimension == -1
    assert certificate.transmission_probability_mass_dimension == 0
    assert certificate.identities_and_finite_witness_only
    false_physical_flags = (
        certificate.e15_modes_derived,
        certificate.kg_to_schrodinger_projection_derived,
        certificate.rectangular_barrier_represents_periodic_lattice,
        certificate.barrier_or_wkb_to_hopping_derived,
        certificate.finite_barrier_exact_localization,
        certificate.autonomous_dwell_time_derived,
        certificate.scattering_instrument_or_energy_receipt_derived,
        certificate.repeated_cptp_or_fresh_ancilla_derived,
        certificate.causal_or_strict_front_derived,
        certificate.qft_microcausality_or_no_signalling_derived,
        certificate.gr_source_derived,
        certificate.selection_derived,
        certificate.gates_5_to_8_closed,
    )
    assert not any(false_physical_flags)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode_count", 0, "mode_count"),
        ("sigma", 0.0, "sigma"),
        ("delta_basis", 1.0, "delta_basis"),
        ("incident_energy", 8.0, "incident_energy"),
        ("projected_hamiltonian_norm_error", -0.1, "projected_hamiltonian_norm_error"),
        ("delta_dyn", 0.0, "delta_dyn"),
    ],
)
def test_invalid_contract_fails_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _certificate(**{field: value})


def _certificate__finite_double_well_spectral_hopping(**overrides: object):
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
        _certificate__finite_double_well_spectral_hopping(well_width=0.0)
    with pytest.raises(ValueError, match="scattering_energy"):
        _certificate__finite_double_well_spectral_hopping(scattering_energy=30.0)
    with pytest.raises(ValueError, match="nu > pi\\^2"):
        _certificate__finite_double_well_spectral_hopping(barrier_height=1.0, scattering_energy=0.5)


def test_parity_roots_have_certified_signs_order_and_energy_intervals() -> None:
    certificate = _certificate__finite_double_well_spectral_hopping()
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
    certificate = _certificate__finite_double_well_spectral_hopping()
    assert certificate.right_mode_norm_witness == pytest.approx(1.0, abs=2.0e-8)
    assert certificate.left_mode_norm_witness == pytest.approx(1.0, abs=2.0e-8)
    assert certificate.right_left_overlap_witness == pytest.approx(0.0, abs=2.0e-8)
    assert 0.5 < certificate.right_mode_right_probability_witness < 1.0
    assert 0.5 < certificate.left_mode_left_probability_witness < 1.0
    assert certificate.maximum_join_residual < 1.0e-9
    assert not certificate.exact_spatial_localization_derived


def test_spectral_hamiltonian_and_swap_are_exact_only_inside_prepared_pair() -> None:
    certificate = _certificate__finite_double_well_spectral_hopping()
    assert certificate.spectral_hamiltonian[0, 0] == pytest.approx(certificate.mean_energy)
    assert certificate.spectral_hamiltonian[0, 1] == pytest.approx(-certificate.hopping)
    assert certificate.spectral_hamiltonian == pytest.approx(certificate.spectral_hamiltonian.T)
    assert certificate.ideal_swap_time == pytest.approx(math.pi / (2.0 * certificate.hopping))
    assert certificate.spectral_swap_phase_residual < 1.0e-11
    assert certificate.finite_double_well_spectrum_to_J_derived
    assert certificate.prepared_exact_spectral_pair_invariant_by_construction
    assert not certificate.arbitrary_continuum_preparation_projects_to_subspace


def test_same_open_scattering_t_but_width_dependent_spectrum_forbids_t_to_j() -> None:
    certificates = [_certificate__finite_double_well_spectral_hopping(well_width=width) for width in (0.8, 1.0, 1.2)]
    transmissions = [item.auxiliary_scattering_transmission for item in certificates]
    assert transmissions == pytest.approx([transmissions[0]] * 3, rel=0.0, abs=0.0)
    intervals = [item.hopping_interval for item in certificates]
    assert intervals[0][1] < intervals[1][0] or intervals[1][1] < intervals[0][0]
    assert intervals[1][1] < intervals[2][0] or intervals[2][1] < intervals[1][0]
    assert len({round(item.hopping, 10) for item in certificates}) == 3
    assert not any(item.transmission_to_hopping_derived or item.wkb_to_hopping_derived for item in certificates)


def test_dimensions_and_all_unclosed_bridges_remain_false() -> None:
    certificate = _certificate__finite_double_well_spectral_hopping()
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


def _certificate__single_energy_barrier_instrument(**overrides: object):
    arguments: dict[str, object] = dict(nonrelativistic_mass=1.0, barrier_height=8.0, incident_energy=2.0, barrier_width=2.0)
    arguments.update(overrides)
    return certify_single_energy_barrier_instrument(**arguments)


def test_spot_amplitudes_and_e16_transmission_agree() -> None:
    certificate = _certificate__single_energy_barrier_instrument()
    assert certificate.k == pytest.approx(2.0)
    assert certificate.kappa == pytest.approx(math.sqrt(12.0))
    assert certificate.transmission_probability == pytest.approx(
        exact_rectangular_barrier_transmission_probability(nonrelativistic_mass=1.0, barrier_height=8.0, incident_energy=2.0, barrier_width=2.0)
    )
    assert certificate.transmission_e16_residual < 1.0e-12
    assert abs(certificate.transmission_amplitude) ** 2 == pytest.approx(certificate.transmission_probability)
    assert certificate.reflection_probability_residual < 1.0e-12
    assert certificate.transmission_amplitude == pytest.approx(0.0014696395355568164 - 0.0008484951524735881j)
    assert certificate.reflection_amplitude == pytest.approx(-0.4999985601078058 - 0.8660245724606968j)
    assert certificate.transmission_probability == pytest.approx(2.879784388242834e-6)
    assert certificate.reflection_amplitude / certificate.transmission_amplitude == pytest.approx(-589.2768610995332j)


def test_scattering_identities_and_port_instrument_for_arbitrary_density_matrix() -> None:
    rho = np.array(((0.4, 0.3 + 0.1j), (0.3 - 0.1j, 0.6)))
    certificate = _certificate__single_energy_barrier_instrument(rho=rho)
    assert certificate.coefficient_identity_residual < 1.0e-12
    assert certificate.cross_amplitude_residual < 1.0e-12
    assert certificate.scattering_unitarity_residual < 1.0e-12
    assert certificate.kraus_completeness_residual < 1.0e-12
    assert certificate.record_isometry.shape == (4, 2)
    assert certificate.record_isometry_residual < 1.0e-12
    assert certificate.choi_minimum_eigenvalue >= -1.0e-12
    assert certificate.output_trace_residual < 1.0e-12
    assert certificate.output_minimum_eigenvalue >= -1.0e-12
    assert sum(certificate.output_port_probabilities) == pytest.approx(1.0)
    assert certificate.nonselective_energy_residual == pytest.approx(0.0)
    assert certificate.energy_intertwining_residual < 1.0e-12
    assert certificate.nonselective_shell_energy_residual < 1.0e-12
    assert certificate.isometric_shell_energy_residual < 1.0e-12
    assert certificate.input_shell_energy_expectation == pytest.approx(certificate.nonselective_output_shell_energy_expectation)
    assert certificate.input_shell_energy_expectation == pytest.approx(certificate.isometric_total_output_energy_expectation)
    assert certificate.final_shell_hamiltonian.shape == (4, 4)


def test_left_only_input_has_only_the_stated_reflection_transmission_labels() -> None:
    certificate = _certificate__single_energy_barrier_instrument()
    assert certificate.left_input_port0_reflection_probability == pytest.approx(certificate.reflection_probability)
    assert certificate.left_input_port1_transmission_probability == pytest.approx(certificate.transmission_probability)
    assert abs(certificate.conventional_coordinate_transmission_amplitude) == pytest.approx(abs(certificate.transmission_amplitude))


def test_deep_finite_barrier_preserves_log_semantics_under_display_underflow() -> None:
    certificate = _certificate__single_energy_barrier_instrument(barrier_width=1.0e6)
    assert math.isfinite(certificate.log_transmission_probability)
    assert certificate.transmission_probability == 0.0
    assert certificate.transmission_probability_numerically_underflowed
    assert certificate.reflection_probability == pytest.approx(1.0)


def test_subnormal_transmission_is_not_mislabelled_as_underflow() -> None:
    # 이 장벽에서 log(T) 는 -720 근처다. 비정규(subnormal) 값이지만 0은 아니다.
    certificate = _certificate__single_energy_barrier_instrument(barrier_width=104.0)
    assert certificate.log_transmission_probability == pytest.approx(-720.0, abs=5.0)
    assert certificate.transmission_probability > 0.0
    assert not certificate.transmission_probability_numerically_underflowed


def test_dimensions_scope_and_no_hopping_relation() -> None:
    certificate = _certificate__single_energy_barrier_instrument()
    assert certificate.dimensions_pass
    assert certificate.width_mass_dimension == -1
    assert certificate.wavenumber_mass_dimension == 1
    assert certificate.kappa_mass_dimension == 1
    assert certificate.dimensionless_barrier_width_mass_dimension == 0
    assert not hasattr(certificate, "ideal_projected_hopping")
    assert all((
        certificate.conditional_single_energy_scattering_unitarity,
        certificate.output_port_cptp_instrument,
        certificate.prepared_record_isometry,
        certificate.elastic_degenerate_energy_bookkeeping,
        certificate.one_sided_port_label_statement,
    ))
    assert not any((
        certificate.physical_observation_or_selection_derived,
        certificate.general_reflection_transmission_labels_derived,
        certificate.wavepacket_or_energy_spread_derived,
        certificate.autonomous_detector_derived,
        certificate.durable_record_reset_or_battery_derived,
        certificate.physical_non_degenerate_record_energy_receipt_derived,
        certificate.repeated_fresh_ancilla_cptp_derived,
        certificate.causal_front_derived, certificate.qft_or_gr_derived,
        certificate.e17_j_transmission_relation_derived,
        certificate.residual_prediction_derived, certificate.gates_3_to_8_closed,
    ))


@pytest.mark.parametrize("field,value", [
    ("nonrelativistic_mass", 0.0), ("barrier_height", 0.0),
    ("incident_energy", 8.0), ("barrier_width", 0.0),
    ("rho", np.eye(3)), ("rho", np.array(((1.0, 1.0), (0.0, 0.0)))),
    ("rho", np.array(((1.2, 0.0), (0.0, -0.2)))),
])
def test_invalid_domain_or_density_matrix_fails_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _certificate__single_energy_barrier_instrument(**{field: value})
