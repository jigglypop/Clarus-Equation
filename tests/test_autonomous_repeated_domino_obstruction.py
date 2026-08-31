import math

import pytest

from examples.physics.autonomous_repeated_domino_obstruction import (
    certify_autonomous_repeated_domino_obstruction,
)


def _certificate(**overrides):
    arguments = dict(
        n_links=3,
        couplings=(0.8, -1.1, 0.6),
        field_mass=1.5,
        clock_scale=2.0,
        prep_mass_squared=0.7,
        battery_energy_per_cell=1.25,
        exchange_coupling=0.3,
        quartic_coupling=0.6,
    )
    arguments.update(overrides)
    return certify_autonomous_repeated_domino_obstruction(**arguments)


def test_open_chain_bound_is_tied_to_the_declared_normalization():
    certificate = _certificate()

    assert certificate.stability_bound_pass
    assert certificate.quartic_lower_bound_coefficient == pytest.approx(0.0)
    assert certificate.quartic_coupling == pytest.approx(
        2.0 * abs(certificate.exchange_coupling)
    )
    assert certificate.dimensions_closed
    assert not certificate.explicit_coordinate_time_switching_present
    assert certificate.local_coordinate_species_field_candidate_by_construction
    assert certificate.carrier_prep_stability_pass
    assert certificate.carrier_quadratic_minimum_mass_squared == pytest.approx(
        1.5**2 - 0.7
    )

    above_bound = _certificate(quartic_coupling=0.8)
    assert above_bound.quartic_lower_bound_coefficient > 0.0

    dimensionless_arguments = dict(certificate.dimensionless_core_arguments)
    assert "J_j Delta_tau" in dimensionless_arguments
    assert "mass dimension" in dimensionless_arguments["J_j Delta_tau"]
    assert "g Delta_tau" not in dimensionless_arguments


def test_below_bound_is_rejected_fail_closed():
    with pytest.raises(ValueError, match="lambda >= 2 \\|g\\|"):
        _certificate(quartic_coupling=0.599)
    with pytest.raises(ValueError, match="carrier preparation stability"):
        _certificate(prep_mass_squared=1.5**2)
    with pytest.raises(ValueError, match="carrier preparation stability"):
        _certificate(prep_mass_squared=1.5**2 + 0.1)


def test_endpoint_powers_and_taylor_coefficient_are_exact_for_the_path():
    certificate = _certificate()
    product = math.prod(certificate.couplings)

    assert certificate.finite_hamiltonian_hermitian
    assert certificate.hamiltonian_hermiticity_residual < 1.0e-14
    assert certificate.lower_order_endpoint_power_residual < 1.0e-14
    assert certificate.endpoint_order_n_value == pytest.approx(product, abs=1.0e-14)
    assert certificate.endpoint_taylor_coefficient == pytest.approx(
        (-1j) ** certificate.n_links * product / math.factorial(certificate.n_links),
        abs=1.0e-14,
    )
    assert certificate.analytic_coefficient_conditions_pass
    assert certificate.small_time_remainder_magnitude > 0.0
    # At t=1e-3 and N=3, the next allowed path contribution is O(t^5).
    # This is an asymptotic numerical witness, not an exact propagator claim.
    assert abs(
        certificate.small_time_endpoint_amplitude - certificate.small_time_leading_term
    ) < 1.0e-12


def test_all_success_receipt_scales_as_capacity_not_expected_energy():
    certificate = _certificate(n_links=4, couplings=(0.2, 0.3, 0.4, 0.5))

    assert certificate.finite_all_success_resource_receipt
    assert certificate.all_success_initially_clean_battery_count == 4
    assert certificate.all_success_initially_clean_record_count == 4
    assert certificate.all_success_battery_energy == pytest.approx(4 * 1.25)


def test_claim_ceilings_stay_false():
    certificate = _certificate()

    assert not certificate.species_index_is_physical_spatial_distance
    assert not certificate.physical_lattice_or_worldtube_completion
    assert not certificate.coupled_clock_global_monotonicity_one_pass
    assert not certificate.exact_delayed_front_derived
    assert not certificate.projected_link_rates_from_action_derived
    assert not certificate.iterated_fresh_ancilla_cptp_instrument_derived
    assert not certificate.continuum_qft_microcausality_derived
    assert not certificate.operational_no_signalling_derived
    assert not certificate.gr_source_stress_matching_derived
    assert not certificate.unbounded_front_from_finite_resources_derived
    assert not certificate.durable_records_derived
    assert not certificate.cross_dataset_parameter_fixing_derived
    assert not certificate.independent_holdout_prediction_derived


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_links": 0}, "n_links"),
        ({"couplings": (0.2, 0.0, 0.4)}, "nonzero"),
        ({"field_mass": 0.0}, "field_mass"),
        ({"clock_scale": -1.0}, "clock_scale"),
        ({"prep_mass_squared": 0.0}, "prep_mass_squared"),
        ({"battery_energy_per_cell": 0.0}, "battery_energy_per_cell"),
        ({"couplings": (0.2, 0.3)}, "exactly n_links"),
    ],
)
def test_invalid_parameters_fail_closed(override, message):
    with pytest.raises(ValueError, match=message):
        _certificate(**override)
