from __future__ import annotations

import math

import pytest

from examples.physics.chapter2_open_bridge_audit import (
    BRIDGE_LEDGER,
    allocation_exponent,
    bao_distance_ratio,
    baryon_to_photon_ratio,
    build_audit,
    casas_ibarra_rank2,
    coherent_additivity_defect,
    constant_vacuum_shift,
    covariance_rank_one_certificate,
    dark_split,
    euclidean_energy,
    fixed_point_map,
    gradient_potential_rate,
    growth_driving_term,
    higher_operator_ratio,
    inverse_coupling_run,
    koide_quadratic_selector,
    linear_transport,
    linearized_covariance,
    newton_coupling_proxy,
    omega_b_from_energy_fraction,
    oriented_domain_average,
    oscillator_potential_rate,
    path_energy_readout,
    photon_temperature,
    poisson_extinction,
    poisson_vector_map,
    portal_proxy,
    pq_anomaly_coefficient,
    projector_overlap_realization,
    registered_coupling_family,
    reheating_expansion,
    required_baryon_transfer,
    threshold_mass_shift,
    vacuum_variance,
    validate,
    weak_mixing_angle,
    wormhole_nec_at_throat,
)


def test_every_registered_bridge_has_a_proof_or_counterexample_certificate() -> None:
    audit = build_audit()
    validate(audit)

    assert tuple(BRIDGE_LEDGER) == tuple(f"B{index}" for index in range(36))
    assert all(
        certificate
        in {
            "counterexample",
            "iff_condition",
            "iff_and_counterexample",
            "construction_nonunique",
            "linearized_rank_bound",
        }
        for _, certificate in BRIDGE_LEDGER.values()
    )
    assert not audit.unconditional_implications_valid
    assert not audit.physical_realizations_validated


def test_interference_is_the_exact_obstruction_to_path_additivity() -> None:
    assert coherent_additivity_defect(1.0 + 0j, -1.0 + 0j) == -2.0
    assert coherent_additivity_defect(1.0 + 0j, 1.0j) == pytest.approx(0.0)
    assert coherent_additivity_defect(1.0 + 0j, 1.0 + 0j) == 2.0


def test_extinction_solver_selects_the_nontrivial_minimal_root_near_criticality() -> None:
    mean = 1.0001
    extinction = poisson_extinction(mean)
    assert 0.0 < extinction < 1.0
    assert extinction == pytest.approx(
        0.9998000266635562,
        rel=0.0,
        abs=2e-15,
    )
    assert extinction == pytest.approx(
        math.exp(-mean * (1.0 - extinction)),
        rel=0.0,
        abs=1e-14,
    )
    assert poisson_extinction(1.0) == 1.0

    machine_adjacent_mean = math.nextafter(1.0, 2.0)
    adjacent_extinction = poisson_extinction(machine_adjacent_mean)
    assert 0.0 < adjacent_extinction <= 1.0


def test_path_and_energy_fractions_agree_exactly_at_zero_covariance() -> None:
    path, energy, covariance = path_energy_readout(
        (0.25, 0.75),
        (1.0, 0.0),
        (3.0, 3.0),
    )
    assert covariance == 0.0
    assert energy == path

    path, energy, covariance = path_energy_readout(
        (0.5, 0.5),
        (1.0, 0.0),
        (1.0, 9.0),
    )
    assert math.isclose(path, 0.5)
    assert math.isclose(energy, 0.1)
    assert math.isclose(energy - path, covariance / 5.0)


@pytest.mark.parametrize(
    ("probabilities", "survivors", "energies"),
    [
        ((), (), ()),
        ((0.2, 0.2), (1.0, 0.0), (1.0, 1.0)),
        ((0.5, 0.5), (1.1, 0.0), (1.0, 1.0)),
        ((0.5, 0.5), (1.0, 0.0), (1.0, 0.0)),
    ],
)
def test_energy_readout_rejects_ill_typed_inputs(
    probabilities: tuple[float, ...],
    survivors: tuple[float, ...],
    energies: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        path_energy_readout(probabilities, survivors, energies)


def test_depth_and_rate_are_not_separately_identifiable() -> None:
    baseline = fixed_point_map(3.2, 1.0, 0.2)
    assert fixed_point_map(32.0, 0.1, 0.2) == baseline


def test_distinct_multitype_matrices_share_the_uniform_scalar_map() -> None:
    state = (0.4, 0.4)
    diagonal = poisson_vector_map(((2.0, 0.0), (0.0, 2.0)), state)
    coupled = poisson_vector_map(((1.0, 1.0), (1.0, 1.0)), state)
    assert diagonal == coupled


def test_any_scalar_overlap_has_a_projector_realization_but_not_a_unique_one() -> None:
    for overlap in (0.0, 0.2, 0.5, 1.0):
        realized, norm_residual = projector_overlap_realization(overlap)
        assert realized == pytest.approx(overlap)
        assert abs(norm_residual) < 1e-15

    with pytest.raises(ValueError):
        projector_overlap_realization(1.1)


def test_qcd_input_does_not_fix_an_independent_weak_angle() -> None:
    assert weak_mixing_angle(1.0, 0.5) == 0.2
    assert weak_mixing_angle(1.0, 2.0) == 0.8


def test_same_total_density_admits_distinct_dark_splits() -> None:
    first = dark_split(0.95, 0.25)
    second = dark_split(0.95, 4.0)

    assert math.isclose(sum(first), sum(second))
    assert math.isclose(first[0] / first[1], 0.25)
    assert math.isclose(second[0] / second[1], 4.0)


def test_canonical_normalization_changes_vacuum_amplitude() -> None:
    assert vacuum_variance(1.0, 1.0) / vacuum_variance(100.0, 1.0) == 100.0


def test_one_point_registration_does_not_fix_a_coupling_function() -> None:
    alpha_star = 0.118
    assert registered_coupling_family(
        alpha_star, alpha_star, 0.0
    ) == registered_coupling_family(alpha_star, alpha_star, 7.0)
    assert registered_coupling_family(
        0.12, alpha_star, 0.0
    ) != registered_coupling_family(0.12, alpha_star, 7.0)


def test_environment_weight_and_reheating_history_remain_independent() -> None:
    assert allocation_exponent(2.0, 1.0) != allocation_exponent(2.0, 2.0)
    assert reheating_expansion(1.0, 1e-12, 0.0) > reheating_expansion(
        1.0, 1e-12, 1.0 / 3.0
    )


def test_a_pole_location_does_not_fix_its_observable_residue() -> None:
    assert portal_proxy(0.0, 0.025) == 0.0
    assert portal_proxy(1e-4, 0.025) > 0.0


def test_casas_ibarra_family_has_same_mass_and_different_yukawa() -> None:
    first, first_residual = casas_ibarra_rank2(0.01, 0.05, 0j)
    second, second_residual = casas_ibarra_rank2(0.01, 0.05, 0.4 + 0.2j)

    assert first_residual < 1e-15
    assert second_residual < 1e-15
    assert first != second


def test_energy_fraction_needs_curvature_and_asymmetry_readouts() -> None:
    assert omega_b_from_energy_fraction(0.05, 1.2) == pytest.approx(0.06)
    assert baryon_to_photon_ratio(0.05, 1.0, 1.0, 2.0, 0.0) == 0.0
    assert baryon_to_photon_ratio(0.05, 1.0, 1.0, 2.0, 1.0) > 0.0
    assert required_baryon_transfer(1.0, 0.05, 0.0, 1.0, 0.0, -1.0) == (
        pytest.approx(0.15)
    )


def test_closed_oscillator_and_gradient_flow_allow_opposite_potential_rates() -> None:
    assert oscillator_potential_rate(1.0, 1.0, 1.0) > 0.0
    assert gradient_potential_rate(1.0) < 0.0


def test_sector_no_go_witnesses_are_numerically_nontrivial() -> None:
    audit = build_audit()
    assert audit.flat_time_circle_interval_squared < 0.0
    assert higher_operator_ratio(1.0, 1.3434991214e-10, 11.0974588093) > 1e12
    assert audit.axion_quality_shift != 0.0
    assert linear_transport(0.0, 2.0) == 0.0
    assert linear_transport(2.0, 2.0) / linear_transport(1.0, 2.0) == 2.0
    assert koide_quadratic_selector(0.0, 1.0) == 1.0
    assert koide_quadratic_selector(1.0, 0.0) == 0.0
    assert audit.same_spatial_dimension_gauge_algebra_dimensions == (1, 3, 8)


def test_dimension_flavour_wall_and_euclidean_witnesses_are_distinct() -> None:
    audit = build_audit()
    assert abs(audit.dimension_depth_same_tau_residual) < 1e-15
    assert audit.same_mass_flavour_angle_gap > 0.0
    assert oriented_domain_average(1.0, 0.5) == 0.0
    assert oriented_domain_average(1.0, 1.0) == 1.0
    assert euclidean_energy(2.0, 1.0) / euclidean_energy(2.0, 2.0) == 2.0


def test_vacuum_shift_changes_energy_without_changing_force() -> None:
    first_energy, first_force = constant_vacuum_shift(0.0, 3.0, 0.0)
    second_energy, second_force = constant_vacuum_shift(0.0, 3.0, 1.0)
    assert second_energy - first_energy == 1.0
    assert second_force - first_force == 0.0


def test_background_rg_hierarchy_and_gravity_witnesses() -> None:
    standard = growth_driving_term(0.3, 1.0, 1.0)
    modified = growth_driving_term(0.3, 1.2, 1.0)
    assert modified != standard

    first_run = inverse_coupling_run(25.0, 1.0, 1.0)
    second_run = inverse_coupling_run(25.0, -2.0, 1.0)
    assert first_run != second_run

    assert threshold_mass_shift(1.0, 1.0) / 1e-12 > 1e9
    assert newton_coupling_proxy(4.0) / newton_coupling_proxy(1.0) == 0.25
    assert wormhole_nec_at_throat(0.5, 1.0) < 0.0


def test_absolute_scale_pq_and_linearized_covariance_witnesses() -> None:
    baseline_bao = bao_distance_ratio(14.0, 0.147)
    rescaled_bao = bao_distance_ratio(42.0, 0.441)
    assert rescaled_bao == pytest.approx(baseline_bao)
    assert photon_temperature(2.0, 1.0) / photon_temperature(1.0, 1.0) == 2.0

    first_anomaly = pq_anomaly_coefficient((1.0,), (0.5,))
    second_anomaly = pq_anomaly_coefficient((1.0, 1.0), (0.5, 0.5))
    assert second_anomaly - first_anomaly == 1.0

    covariance = linearized_covariance((1.0, 2.0, 3.0), 0.04)
    assert covariance_rank_one_certificate(covariance) == 1
