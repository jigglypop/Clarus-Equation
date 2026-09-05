"""인과 기록 → 먼지 브리지와 비회전 먼지 작용 허용 판정의 테스트."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.record.record_dust_bridge import (
    CausalRecordNode,
    CausalTransition,
    ExitPhaseMark,
    admit_irrotational_dust_action,
    canonical_dust_data,
    certificate,
    clock_constraint_residual,
    clock_gradient_from_receipt,
    construct_conserved_record_flow,
    epsilon_reparameterization,
    flat_flrw_cauchy_witness,
    free_stream_exit_marks,
    lower_minkowski,
    match_exit_antichain,
    monokinetic_dust_data,
    multiplier_dust_stress_covariant,
    raise_rank_two_minkowski,
    velocity_vorticity_residual,
    vortical_monokinetic_counterexample,
)


def _rest_record_witness(
    *,
    length_scale: float = 1.0,
):
    energy = 2.0 / length_scale
    nodes = (
        CausalRecordNode("root", "seed", energy),
        CausalRecordNode("left", "folded-left", energy),
        CausalRecordNode("right", "folded-right", energy),
    )
    transitions = (
        CausalTransition("root", "left", 2.0 / 5.0),
        CausalTransition("root", "right", 3.0 / 5.0),
    )
    flow = construct_conserved_record_flow(nodes, transitions, {"root": 5.0})
    marks = (
        ExitPhaseMark("left", (0.1, 0.2, 0.3), energy, (0.0, 0.0, 0.0)),
        ExitPhaseMark("right", (0.7, 0.8, 0.9), energy, (0.0, 0.0, 0.0)),
    )
    matching = match_exit_antichain(
        flow,
        nodes,
        marks,
        cell_volume=length_scale**3,
    )
    return nodes, transitions, flow, marks, matching


def test_marked_causal_kernel_constructs_unique_local_balances() -> None:
    _, _, flow, _, _ = _rest_record_witness()

    assert flow.terminal_composition == (
        ("folded-left", 2.0 / 5.0),
        ("folded-right", 3.0 / 5.0),
    )
    assert flow.initial_number == flow.terminal_number == 5.0
    assert flow.initial_energy == flow.terminal_energy == 10.0
    assert all(balance.number_residual == 0.0 for balance in flow.balances)
    assert all(balance.energy_residual == 0.0 for balance in flow.balances)


def test_exit_pushforward_gives_positive_dust_stress_without_double_counting() -> None:
    _, _, _, _, matching = _rest_record_witness()

    assert matching.current == (5.0, 0.0, 0.0, 0.0)
    assert matching.stress == (
        (10.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    assert matching.residual_energy_density == 10.0
    assert matching.complement_energy_density == 0.0
    assert matching.no_double_counting_residual == 0.0

    dust = monokinetic_dust_data(matching)
    assert dust.gamma == 1.0
    assert dust.surface_number_density == dust.rest_number_density == 5.0
    assert dust.rest_energy_density == 10.0
    assert dust.stress == matching.stress


def test_moving_monokinetic_measure_has_the_exact_dust_normalization() -> None:
    mass = 2.0
    energy = math.sqrt(5.0)
    nodes = (
        CausalRecordNode('root', 'seed', energy),
        CausalRecordNode('left', 'left', energy),
        CausalRecordNode('right', 'right', energy),
    )
    flow = construct_conserved_record_flow(
        nodes,
        (
            CausalTransition('root', 'left', 0.4),
            CausalTransition('root', 'right', 0.6),
        ),
        {'root': 5.0},
    )
    matching = match_exit_antichain(
        flow,
        nodes,
        (
            ExitPhaseMark('left', (0.0, 0.0, 0.0), mass, (1.0, 0.0, 0.0)),
            ExitPhaseMark('right', (0.5, 0.0, 0.0), mass, (1.0, 0.0, 0.0)),
        ),
        cell_volume=1.0,
    )

    dust = monokinetic_dust_data(matching)

    assert math.isclose(
        -dust.four_velocity[0] ** 2 + dust.four_velocity[1] ** 2,
        -1.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(dust.rest_number_density, 10.0 / math.sqrt(5.0))
    assert math.isclose(dust.rest_energy_density, 4.0 * math.sqrt(5.0))
    assert math.isclose(matching.stress[0][0], 5.0 * math.sqrt(5.0))
    assert math.isclose(matching.stress[0][1], 5.0)
    assert math.isclose(matching.stress[1][1], math.sqrt(5.0))


def test_efficiency_is_an_explicit_partition_not_a_second_energy_copy() -> None:
    nodes, _, flow, _, _ = _rest_record_witness()
    marks = (
        ExitPhaseMark(
            "left", (0.1, 0.2, 0.3), 2.0, (0.0, 0.0, 0.0), 0.25
        ),
        ExitPhaseMark(
            "right", (0.7, 0.8, 0.9), 2.0, (0.0, 0.0, 0.0), 0.5
        ),
    )

    matching = match_exit_antichain(flow, nodes, marks, cell_volume=1.0)

    assert math.isclose(matching.residual_energy_density, 4.0)
    assert math.isclose(matching.complement_energy_density, 6.0)
    assert math.isclose(matching.total_energy_density, 10.0)
    assert matching.no_double_counting_residual == 0.0


def test_multistream_data_keep_kinetic_stress_but_do_not_fake_single_dust() -> None:
    nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("slow", "slow", 1.0),
        CausalRecordNode("fast", "fast", 3.0),
    )
    transitions = (
        CausalTransition("root", "slow", 0.5),
        CausalTransition("root", "fast", 0.5),
    )
    flow = construct_conserved_record_flow(nodes, transitions, {"root": 4.0})
    marks = (
        ExitPhaseMark("slow", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0)),
        ExitPhaseMark("fast", (0.5, 0.0, 0.0), 3.0, (0.0, 0.0, 0.0)),
    )
    matching = match_exit_antichain(flow, nodes, marks, cell_volume=1.0)

    assert matching.residual_energy_density == 8.0
    assert matching.current[0] == 4.0
    with pytest.raises(ValueError, match="multi-stream or multi-mass"):
        monokinetic_dust_data(matching)


def test_flat_flrw_witness_satisfies_both_initial_constraints() -> None:
    _, _, _, _, matching = _rest_record_witness()
    witness = flat_flrw_cauchy_witness(
        monokinetic_dust_data(matching),
        newton_constant=1.0,
    )

    assert math.isclose(
        witness.hubble_rate**2,
        8.0 * math.pi * witness.energy_density / 3.0,
        rel_tol=1.0e-15,
    )
    assert math.isclose(witness.hamiltonian_residual, 0.0, abs_tol=1.0e-12)
    assert witness.momentum_residual == (0.0, 0.0, 0.0)


def test_reference_length_scaling_has_the_declared_dimensions() -> None:
    _, _, _, _, unit_matching = _rest_record_witness(length_scale=1.0)
    _, _, _, _, doubled_matching = _rest_record_witness(length_scale=2.0)

    # [J] = L^-3, [T] = [rho] = L^-4 이다.
    assert math.isclose(
        doubled_matching.current[0], unit_matching.current[0] / 2.0**3
    )
    assert math.isclose(
        doubled_matching.stress[0][0], unit_matching.stress[0][0] / 2.0**4
    )

    unit_witness = flat_flrw_cauchy_witness(
        monokinetic_dust_data(unit_matching),
        newton_constant=1.0,
    )
    doubled_witness = flat_flrw_cauchy_witness(
        monokinetic_dust_data(doubled_matching),
        newton_constant=2.0**2,
    )
    assert math.isclose(
        doubled_witness.hubble_rate,
        unit_witness.hubble_rate / 2.0,
    )


def test_free_liouville_pushforward_commutes_with_spatial_translation() -> None:
    mark = ExitPhaseMark("exit", (0.1, 0.2, 0.3), 2.0, (1.0, 0.0, 0.0))
    shift = (0.17, 0.23, 0.31)
    translated = ExitPhaseMark(
        "exit",
        tuple((x + dx) % 1.0 for x, dx in zip(mark.position, shift)),
        mark.mass,
        mark.spatial_momentum,
    )

    streamed = free_stream_exit_marks((mark,), coordinate_time=0.4, box_length=1.0)[0]
    translated_streamed = free_stream_exit_marks(
        (translated,), coordinate_time=0.4, box_length=1.0
    )[0]

    assert all(
        math.isclose((x + dx) % 1.0, shifted_x, abs_tol=1.0e-15)
        for x, dx, shifted_x in zip(
            streamed.position,
            shift,
            translated_streamed.position,
        )
    )
    assert streamed.four_momentum == translated_streamed.four_momentum


def test_bridge_rejects_unnormalized_nonconserving_or_mistyped_inputs() -> None:
    nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("left", "left", 2.0),
        CausalRecordNode("right", "right", 2.0),
    )
    with pytest.raises(ValueError, match="sum to one"):
        construct_conserved_record_flow(
            nodes,
            (
                CausalTransition("root", "left", 0.2),
                CausalTransition("root", "right", 0.3),
            ),
            {"root": 1.0},
        )

    energy_mismatch_nodes = (
        CausalRecordNode("root", "seed", 2.0),
        CausalRecordNode("left", "left", 1.0),
        CausalRecordNode("right", "right", 2.0),
    )
    with pytest.raises(ValueError, match="harmonic"):
        construct_conserved_record_flow(
            energy_mismatch_nodes,
            (
                CausalTransition("root", "left", 0.5),
                CausalTransition("root", "right", 0.5),
            ),
            {"root": 1.0},
        )

    _, _, flow, _, _ = _rest_record_witness()
    with pytest.raises(ValueError, match="mass-shell energy"):
        match_exit_antichain(
            flow,
            nodes,
            (
                ExitPhaseMark("left", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0)),
                ExitPhaseMark("right", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0)),
            ),
            cell_volume=1.0,
        )


def _canonical_arguments() -> tuple[object, tuple[float, ...], float]:
    dust = canonical_dust_data()
    scale = 2.0
    velocity_covector = lower_minkowski(dust.four_velocity)
    receipt_gradient = tuple(-scale * value for value in velocity_covector)
    return dust, receipt_gradient, scale


def test_multiplier_action_matches_existing_monokinetic_dust_exactly() -> None:
    receipt = certificate()

    assert receipt.clock_constraint_residual == pytest.approx(0.0)
    assert receipt.continuity_residual == pytest.approx(0.0)
    assert receipt.vorticity_residual == pytest.approx(0.0)
    assert receipt.geodesic_residual == pytest.approx(0.0)
    assert receipt.ward_residual == pytest.approx(0.0)
    assert receipt.stress_match_residual == pytest.approx(0.0)
    assert receipt.mass_current_match_residual == pytest.approx(0.0)
    assert receipt.action_stress_contravariant == receipt.kinetic_stress_contravariant
    assert receipt.action_energy_current == receipt.kinetic_mass_current


def test_metric_variation_has_pressureless_on_shell_form() -> None:
    receipt = certificate()
    covariant = multiplier_dust_stress_covariant(
        receipt.lambda_density, receipt.clock_gradient_covector
    )

    assert covariant == receipt.action_stress_covariant
    assert raise_rank_two_minkowski(covariant) == receipt.kinetic_stress_contravariant
    assert receipt.rest_energy_density == pytest.approx(6.0)
    assert receipt.isotropic_pressure == pytest.approx(0.0)
    assert receipt.equation_of_state == 0.0


def test_receipt_clock_requires_a_nonconstant_unit_timelike_gradient() -> None:
    gradient = clock_gradient_from_receipt(
        (2.5, -1.5, 0.0, 0.0), reference_mass_scale=2.0
    )

    assert gradient == pytest.approx((1.25, -0.75, 0.0, 0.0))
    assert clock_constraint_residual(gradient) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="constant receipt"):
        clock_gradient_from_receipt(
            (0.0, 0.0, 0.0, 0.0), reference_mass_scale=2.0
        )


def test_dimension_ledger_is_explicit_and_closes_the_action() -> None:
    receipt = certificate()

    assert receipt.receipt_mass_dimension == 0
    assert receipt.reference_scale_mass_dimension == 1
    assert receipt.clock_mass_dimension == -1
    assert receipt.multiplier_mass_dimension == 4
    assert receipt.stress_mass_dimension == 4
    assert receipt.action_density_mass_dimension == 4
    assert receipt.volume_element_mass_dimension == -4
    assert receipt.action_mass_dimension == 0
    assert receipt.dimensions_pass


def test_overall_epsilon_is_an_exact_multiplier_reparameterization() -> None:
    receipt = epsilon_reparameterization(epsilon=0.25, multiplier_density=20.0)

    assert receipt.physical_density == pytest.approx(5.0)
    assert receipt.absorbed_multiplier_density == pytest.approx(5.0)
    assert receipt.exact_stress_reparameterization
    assert not receipt.epsilon_is_independent_dust_coupling
    assert not receipt.finite_epsilon_gr_limit_derived


def test_vortical_geodesic_dust_is_a_complete_limit_counterexample() -> None:
    witness = vortical_monokinetic_counterexample(kappa=0.2, y=1.0)

    assert witness.norm_residual == pytest.approx(0.0)
    assert witness.continuity_residual == pytest.approx(0.0)
    assert witness.geodesic_residual == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert witness.vorticity_residual == pytest.approx(
        0.2 / (1.0 - 0.2**2) ** 1.5
    )
    assert witness.kinetic_dust_admissible_at_point
    assert not witness.single_clock_admissible


def test_admission_rejects_vorticity_caustics_and_multistream_data() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()
    vortical_jacobian = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.2, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )

    assert velocity_vorticity_residual(vortical_jacobian) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="irrotational"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            velocity_covector_jacobian=vortical_jacobian,
        )
    with pytest.raises(ValueError, match="caustic|shell crossing"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            lagrangian_flow_jacobian=0.0,
        )
    with pytest.raises(ValueError, match="multistream"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            single_stream=False,
        )


def test_action_and_kinetic_sources_cannot_be_summed() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()

    with pytest.raises(ValueError, match="matched, not summed"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            accounting_mode="kinetic_plus_action",
        )


def test_lambda_positivity_is_an_admission_not_an_action_theorem() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()
    negative = replace(dust, rest_energy_density=-1.0)
    zero_vector = (0.0, 0.0, 0.0, 0.0)
    zero_tensor = (zero_vector, zero_vector, zero_vector, zero_vector)
    zero = replace(
        dust,
        surface_number_density=0.0,
        rest_number_density=0.0,
        rest_energy_density=0.0,
        current=zero_vector,
        stress=zero_tensor,
    )

    with pytest.raises(ValueError, match="non-negative by admission"):
        admit_irrotational_dust_action(
            negative,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
        )
    assert certificate().lambda_nonnegative_admission
    zero_receipt = admit_irrotational_dust_action(
        zero,
        receipt_gradient_covector=receipt_gradient,
        reference_mass_scale=scale,
    )
    assert zero_receipt.rest_energy_density == 0.0
    assert zero_receipt.isotropic_pressure == 0.0
    assert zero_receipt.equation_of_state is None
    tiny_density = 1.0e-15
    tiny_number_density = tiny_density / dust.mass
    tiny_current = tuple(
        tiny_number_density * component for component in dust.four_velocity
    )
    tiny_stress = tuple(
        tuple(
            tiny_density * dust.four_velocity[mu] * dust.four_velocity[nu]
            for nu in range(4)
        )
        for mu in range(4)
    )
    tiny = replace(
        dust,
        surface_number_density=tiny_number_density * dust.gamma,
        rest_number_density=tiny_number_density,
        rest_energy_density=tiny_density,
        current=tiny_current,
        stress=tiny_stress,
    )
    tiny_receipt = admit_irrotational_dust_action(
        tiny,
        receipt_gradient_covector=receipt_gradient,
        reference_mass_scale=scale,
    )
    assert tiny_receipt.equation_of_state == 0.0


def test_timelike_dust_worldline_is_subluminal_but_not_no_signalling_proof() -> None:
    receipt = certificate()

    assert receipt.coordinate_speed == pytest.approx(0.6)
    assert receipt.coordinate_speed < 1.0
    assert receipt.worldline_speed_below_c
    assert receipt.proper_time_interval_squared == pytest.approx(-1.0)
    assert not receipt.qft_microcausality_derived
    assert not receipt.operational_no_signalling_derived


def test_certificate_keeps_the_physical_claim_ceiling_false() -> None:
    receipt = certificate()

    assert receipt.smooth_single_stream_irrotational_precaustic
    assert receipt.matched_not_summed
    assert receipt.action_is_variational_reexpression_not_new_energy
    assert not receipt.local_receipt_field_map_derived
    assert not receipt.reference_mass_scale_derived
    assert not receipt.multiplier_initial_law_derived
    assert not receipt.supplied_metric_derived
    assert not receipt.gravitational_boson_derived
    assert not receipt.cptp_quantum_dynamics_derived
    assert not receipt.finite_coefficient_gr_phenomenology_derived
    assert not receipt.independent_holdout_prediction_derived
    assert not receipt.two_residual_classes_reduced
    assert not receipt.complexity_penalty_success
    assert not receipt.curved_metric_continuum_verified
    assert not receipt.flow_jacobian_computed_from_dynamics


def test_invalid_kinematics_and_transport_fail_closed() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()

    with pytest.raises(ValueError, match="u_mu=-partial_mu tau"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=(2.0, 0.0, 0.0, 0.0),
            reference_mass_scale=scale,
        )
    with pytest.raises(ValueError, match="continuity"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            density_gradient_covector=(1.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="future unit timelike"):
        admit_irrotational_dust_action(
            replace(dust, four_velocity=(1.0, 1.0, 0.0, 0.0)),
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
        )
    with pytest.raises(ValueError, match="positive"):
        epsilon_reparameterization(epsilon=0.0, multiplier_density=1.0)
    assert math.isfinite(certificate().lambda_density)
