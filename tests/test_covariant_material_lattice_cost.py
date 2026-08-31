import math

import pytest

from examples.physics.covariant_material_lattice_cost import (
    certify_covariant_material_lattice_cost,
)


def _certificate(**overrides):
    arguments = dict(
        cells_per_axis=4,
        proper_cell_spacing=2.0,
        rod_scale=1.5,
        battery_energy_per_cell=0.75,
        carrier_mass=2.0,
        carrier_momentum=1.25,
        onsite_exchange_coupling=-0.4,
        quartic_coupling=0.4,
        guide_well_mass_squared=0.2,
        cell_well_mass_squared=0.0,
    )
    arguments.update(overrides)
    return certify_covariant_material_lattice_cost(**arguments)


def test_dimensions_geometry_and_supplied_winding_are_closed():
    certificate = _certificate()
    q = math.pi

    assert certificate.action_terms_have_mass_dimension_four
    assert certificate.dimensionless_core_arguments == (
        ("q a = 2 pi", "compact phase winding"),
        ("q L / (2 pi) = N", "supplied integer winding"),
        ("v_g = |k| / sqrt(|k|^2 + m_H^2)", "free-particle sample"),
    )
    assert certificate.compact_phase_period_is_two_pi
    assert certificate.clock_field_used is False
    assert certificate.wave_number == pytest.approx(q)
    assert certificate.material_gram_diagonal == pytest.approx((q**2,) * 3)
    assert certificate.material_gram_determinant == pytest.approx(q**6)
    assert certificate.normalized_gram_determinant == pytest.approx(1.0)
    assert certificate.proper_cell_volume == pytest.approx(2.0**3)
    assert certificate.winding_per_axis == pytest.approx(4.0)
    assert not certificate.spacing_action_winding_derived


def test_free_rod_stress_and_finite_energy_receipt_are_separate():
    certificate = _certificate()
    expected_rho = 1.5 * 1.5**2 * math.pi**2

    assert certificate.rod_energy_density == pytest.approx(expected_rho)
    assert certificate.rod_pressure == pytest.approx(-expected_rho / 3.0)
    assert certificate.rod_equation_of_state == pytest.approx(-1.0 / 3.0)
    assert certificate.finite_rod_energy == pytest.approx(expected_rho * 8.0**3)
    assert certificate.finite_rod_receipt
    assert certificate.supplied_finite_free_rod_background_bookkeeping


def test_guide_and_volume_battery_capacities_are_not_the_rod_ledger():
    certificate = _certificate()

    assert certificate.guide_all_success_battery_count == 4
    assert certificate.guide_battery_capacity == pytest.approx(4 * 0.75)
    assert certificate.full_volume_cell_count == 4**3
    assert certificate.full_volume_battery_capacity == pytest.approx(4**3 * 0.75)
    assert certificate.rod_and_battery_ledgers_kept_separate
    assert certificate.finite_rod_energy == pytest.approx(
        1.5 * 1.5**2 * math.pi**2 * 8.0**3
    )


def test_exact_onsite_quartic_bound_and_extremal_witness():
    certificate = _certificate()

    assert certificate.quartic_lower_bound_coefficient == pytest.approx(0.0)
    assert certificate.extremal_quartic_potential == pytest.approx(0.0)
    assert certificate.quartic_saturation_residual < 1.0e-14
    strict = _certificate(quartic_coupling=0.9)
    assert strict.quartic_lower_bound_coefficient == pytest.approx((0.9 - 0.4) / 4.0)
    assert strict.extremal_quartic_potential == pytest.approx(
        strict.quartic_lower_bound_coefficient
    )
    assert strict.quartic_saturation_residual < 1.0e-14


def test_below_quartic_bound_fails_closed():
    with pytest.raises(ValueError, match="quartic stability"):
        _certificate(quartic_coupling=0.399)


def test_free_group_speed_has_a_ceiling_but_is_not_a_front_proof():
    certificate = _certificate()

    assert 0.0 <= certificate.carrier_group_velocity < 1.0
    assert certificate.canonical_fixed_background_classical_principal_symbol
    assert certificate.fixed_background_classical_domain_of_dependence
    assert not certificate.band_or_front_speed_derived


def test_unproved_claims_remain_false_and_invalid_inputs_fail():
    certificate = _certificate()
    assert certificate.diffeomorphism_covariant_scalar_candidate_by_construction
    assert certificate.static_common_coupling_without_coordinate_time_schedule
    assert not any(
        (
            certificate.interacting_backreacted_theta_solution_derived,
            certificate.background_stability_or_caustic_freedom_derived,
            certificate.periodic_well_localized_modes_derived,
            certificate.action_to_projected_rates_or_resonance_derived,
            certificate.scattering_energy_transfer_receipt_derived,
            certificate.durable_record_or_selection_derived,
            certificate.repeated_cptp_fresh_ancilla_derived,
            certificate.qft_microcausality_or_no_signalling_derived,
            certificate.coupled_gr_source_derived,
            certificate.infinite_isolated_lattice_finite_total_energy_derived,
            certificate.gates_five_to_eight_derived,
        )
    )

    for overrides in (
        {"cells_per_axis": 0},
        {"cells_per_axis": 1.5},
        {"proper_cell_spacing": 0.0},
        {"onsite_exchange_coupling": 0.0},
        {"guide_well_mass_squared": -0.1},
    ):
        with pytest.raises(ValueError):
            _certificate(**overrides)
