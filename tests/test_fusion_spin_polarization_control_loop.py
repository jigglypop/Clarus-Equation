from __future__ import annotations

from dataclasses import replace
import math

import pytest

from reality_stone.clarus.fusion_spin_polarization_control_loop import (
    _equilibrium_deuteron_polarization,
    audit_polarization_retention,
    audit_polarized_fuel_pump_ledger,
    audit_spin_reaction,
    audit_thermal_equilibrium_polarization,
    current_fusion_spin_polarization_control_report,
    physical_spin_branch_gate_pass,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_spin_polarization_control_report()


def test_unpolarized_limit_recovers_bosch_hale(report) -> None:
    audit = audit_spin_reaction(
        deuteron_polarization=0.0,
        triton_polarization=0.0,
    )

    assert audit.quartet_probability == pytest.approx(2.0 / 3.0)
    assert audit.cross_section_multiplier == 1.0
    assert audit.maxwellian_reactivity_ratio == 1.0
    assert audit.fractional_reactivity_gain == 0.0
    assert not audit.ideal_projector_arithmetic_target_reached


def test_one_percent_requires_polarization_product_point_zero_two(report) -> None:
    audit = report.target_reaction

    assert audit.required_polarization_product == pytest.approx(0.02)
    assert audit.polarization_product == pytest.approx(0.02)
    assert audit.cross_section_multiplier == pytest.approx(1.01)
    assert audit.maxwellian_reactivity_ratio == pytest.approx(1.01)
    assert audit.fractional_reactivity_gain == pytest.approx(0.01)
    assert audit.ideal_projector_arithmetic_target_reached
    assert audit.constant_ideal_projector_multiplier_commuted_with_maxwellian_average
    assert not audit.energy_dependent_polarized_cross_section_integrated
    assert audit.conditional_ideal_projector_target_reached
    assert not audit.spin_correlation_czz_directly_measured
    assert not audit.polarized_dt_rate_directly_validated
    assert not audit.energy_dependent_polarized_reaction_operator_provenance_pass


def test_source_design_has_retention_margin_but_not_an_engineering_pass(report) -> None:
    reaction = report.source_design_reaction
    retention = report.retention

    assert reaction.polarization_product == pytest.approx(0.05)
    assert reaction.fractional_reactivity_gain == pytest.approx(0.025)
    assert retention.minimum_product_retention_fraction == pytest.approx(0.4)
    assert retention.minimum_common_species_retention_fraction == pytest.approx(math.sqrt(0.4))
    assert retention.high_frequency_wave_depolarization_small_in_simulation
    assert retention.partial_helium3_plasma_retention_evidence_available
    assert not retention.burn_weighted_product_measured
    assert retention.burn_weighted_polarization_product is None
    assert not retention.burn_weighted_product_meets_required_threshold
    assert not retention.burn_weighted_product_not_above_source_product
    assert retention.source_polarization_product_meets_required_threshold
    assert not retention.tritium_polarization_source_demonstrated_at_reactor_throughput
    assert not retention.dt_in_plasma_retention_measured
    assert not retention.injection_and_retention_gate_pass


def test_ten_kev_equilibrium_magnetization_is_an_energy_no_go(report) -> None:
    audit = report.thermal_equilibrium

    assert audit.required_uniform_magnetic_field_t == pytest.approx(
        3.501356618e10,
        rel=3.0e-10,
    )
    assert audit.equilibrium_deuteron_polarization == pytest.approx(
        0.0630011879,
        rel=3.0e-9,
    )
    assert audit.equilibrium_triton_polarization == pytest.approx(
        0.317454332,
        rel=3.0e-9,
    )
    assert audit.magnetic_field_energy_density_j_m3 == pytest.approx(
        4.877899333e26,
        rel=3.0e-9,
    )
    assert audit.field_to_plasma_thermal_energy_density_ratio > 1.0e20
    assert audit.required_to_qed_critical_field_ratio > 7.0
    assert audit.polarization_product_residual < 1.0e-13
    assert audit.adaptive_field_bracket_pass
    assert not audit.equilibrium_magnetization_route_pass


def test_high_target_uses_an_adaptive_bracket_and_checks_the_residual() -> None:
    audit = audit_thermal_equilibrium_polarization(target_fractional_gain=0.49)

    assert audit.required_uniform_magnetic_field_t > 1.0e12
    assert (
        audit.equilibrium_deuteron_polarization * audit.equilibrium_triton_polarization
    ) == pytest.approx(0.98, abs=1.0e-12)
    assert audit.polarization_product_residual < 1.0e-12
    assert audit.adaptive_field_bracket_pass


def test_pump_ledger_uses_incremental_not_total_fusion_energy(report) -> None:
    audit = report.pump_ledger

    assert audit.expected_baseline_fusion_energy_per_injected_pair_kev == pytest.approx(176.0)
    assert audit.fixed_profile_linear_rate_response_assumed
    assert audit.linearized_incremental_fusion_energy_budget_per_injected_pair_kev == pytest.approx(
        1.76
    )
    assert audit.linearized_raw_incremental_fusion_power_w == pytest.approx(5.0e6)
    assert audit.fixed_exposure_burn_probability_model_evaluated
    assert audit.fixed_exposure_incremental_burn_fraction == pytest.approx(9.949332516e-5)
    assert audit.conservative_fixed_exposure_budget_used
    assert audit.incremental_fusion_energy_budget_per_injected_pair_kev == pytest.approx(
        1.751082523
    )
    assert audit.declared_reference_fusion_power_w == pytest.approx(5.0e8)
    assert audit.required_fusion_reaction_rate_s == pytest.approx(1.773156e20, rel=3.0e-6)
    assert audit.declared_dt_pair_injection_rate_s == pytest.approx(1.773156e22, rel=3.0e-6)
    assert audit.declared_dt_mass_throughput_kg_s == pytest.approx(1.480752e-4, rel=3.0e-6)
    assert audit.raw_incremental_fusion_power_w == pytest.approx(4.974666258e6)
    assert audit.electrical_break_even_energy_per_injected_pair_ev == pytest.approx(700.4330091)
    assert audit.maximum_wall_plug_pump_power_for_net_gain_w == pytest.approx(1.989866503e6)
    assert audit.engineering_margin_energy_per_injected_pair_ev == pytest.approx(70.04330091)
    assert audit.engineering_margin_wall_plug_power_w == pytest.approx(1.989866503e5)
    assert audit.reference_ideal_polarized_deuterium_beam_target_rate_s == pytest.approx(1.0e18)
    assert audit.reactor_per_species_to_reference_deuterium_beam_target_ratio > 1.7e4
    assert not audit.measured_wall_plug_energy_per_polarized_dt_pair_available
    assert audit.measured_wall_plug_energy_per_polarized_dt_pair_ev is None
    assert not audit.measured_pair_energy_below_break_even
    assert not audit.net_incremental_energy_positive_demonstrated
    assert not audit.source_and_pump_gate_pass


def test_final_report_does_not_promote_the_control_to_ce(report) -> None:
    assert report.standard_model_non_ce_mechanism_identified
    assert report.conditional_ideal_projector_one_percent_control_derived
    assert not report.full_source_and_pump_ledger_closed
    assert not report.physical_spin_polarized_branch_pass
    assert not report.ce_new_physics_used
    assert not report.physical_ce_one_percent_branch_pass
    assert report.maximum_supported_stage == (
        "STANDARD_MODEL_IDEAL_QUARTET_ONE_PERCENT_CONTROL_"
        "SOURCE_THROUGHPUT_AND_RETENTION_FAIL_CLOSED"
    )


def test_physical_gate_requires_every_leaf_blocker(report) -> None:
    target = replace(
        report.target_reaction,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    retention = replace(
        report.retention,
        burn_weighted_product_measured=True,
        burn_weighted_polarization_product=report.retention.burn_weighted_product_required,
        burn_weighted_product_meets_required_threshold=True,
        burn_weighted_product_not_above_source_product=True,
        source_polarization_product_meets_required_threshold=True,
        deuteron_pellet_source_demonstrated_at_reactor_throughput=True,
        tritium_polarization_source_demonstrated_at_reactor_throughput=True,
        dt_in_plasma_retention_measured=True,
        injection_and_retention_gate_pass=True,
    )
    pump = replace(
        report.pump_ledger,
        measured_wall_plug_energy_per_polarized_dt_pair_ev=(
            report.pump_ledger.engineering_margin_energy_per_injected_pair_ev
        ),
        measured_wall_plug_energy_per_polarized_dt_pair_available=True,
        measured_pair_energy_below_break_even=True,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_depolarization_and_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
        source_and_pump_gate_pass=True,
    )
    kwargs = {
        "target_reaction": target,
        "source_design_reaction": report.source_design_reaction,
        "retention": retention,
        "pump_ledger": pump,
    }

    assert physical_spin_branch_gate_pass(**kwargs)
    for field in (
        "ideal_projector_arithmetic_target_reached",
        "conditional_ideal_projector_target_reached",
        "standard_model_spin_projector_derived",
        "j32_s_wave_dominance_assumed",
        "ab_initio_channel_support_available",
        "constant_ideal_projector_multiplier_commuted_with_maxwellian_average",
        "energy_dependent_polarized_cross_section_integrated",
        "energy_dependent_polarized_reaction_operator_provenance_pass",
        "spin_correlation_czz_directly_measured",
        "polarized_dt_rate_directly_validated",
    ):
        failed = dict(kwargs)
        failed["target_reaction"] = replace(target, **{field: False})
        assert not physical_spin_branch_gate_pass(**failed)
    for field in (
        "ideal_projector_arithmetic_target_reached",
        "conditional_ideal_projector_target_reached",
    ):
        failed = dict(kwargs)
        failed["source_design_reaction"] = replace(report.source_design_reaction, **{field: False})
        assert not physical_spin_branch_gate_pass(**failed)
    for field in (
        "burn_weighted_product_measured",
        "burn_weighted_product_meets_required_threshold",
        "burn_weighted_product_not_above_source_product",
        "source_polarization_product_meets_required_threshold",
        "deuteron_pellet_source_demonstrated_at_reactor_throughput",
        "tritium_polarization_source_demonstrated_at_reactor_throughput",
        "dt_in_plasma_retention_measured",
        "injection_and_retention_gate_pass",
    ):
        failed = dict(kwargs)
        failed["retention"] = replace(retention, **{field: False})
        assert not physical_spin_branch_gate_pass(**failed)
    assert not physical_spin_branch_gate_pass(
        **{
            **kwargs,
            "retention": replace(retention, burn_weighted_polarization_product=None),
        }
    )
    assert not physical_spin_branch_gate_pass(
        **{
            **kwargs,
            "retention": replace(
                retention,
                burn_weighted_polarization_product=(
                    retention.burn_weighted_product_required * 0.999
                ),
            ),
        }
    )
    for field in (
        "source_throughput_declared",
        "fixed_exposure_burn_probability_model_evaluated",
        "conservative_fixed_exposure_budget_used",
        "measured_wall_plug_energy_per_polarized_dt_pair_available",
        "measured_pair_energy_below_break_even",
        "cryogenic_and_microwave_power_accounted",
        "tritium_handling_power_accounted",
        "recycle_depolarization_and_repolarization_accounted",
        "net_incremental_energy_positive_demonstrated",
        "source_and_pump_gate_pass",
    ):
        failed = dict(kwargs)
        failed["pump_ledger"] = replace(pump, **{field: False})
        assert not physical_spin_branch_gate_pass(**failed)
    assert not physical_spin_branch_gate_pass(
        **{
            **kwargs,
            "pump_ledger": replace(
                pump,
                measured_wall_plug_energy_per_polarized_dt_pair_ev=None,
            ),
        }
    )
    assert not physical_spin_branch_gate_pass(
        **{
            **kwargs,
            "pump_ledger": replace(
                pump,
                measured_wall_plug_energy_per_polarized_dt_pair_ev=(
                    pump.electrical_break_even_energy_per_injected_pair_ev * 1.001
                ),
            ),
        }
    )


def test_aggregate_gates_equal_their_required_leaf_conjunctions(report) -> None:
    retention = report.retention
    pump = report.pump_ledger

    assert retention.injection_and_retention_gate_pass == all(
        (
            retention.burn_weighted_product_measured,
            retention.burn_weighted_product_meets_required_threshold,
            retention.burn_weighted_product_not_above_source_product,
            retention.source_polarization_product_meets_required_threshold,
            retention.deuteron_pellet_source_demonstrated_at_reactor_throughput,
            retention.tritium_polarization_source_demonstrated_at_reactor_throughput,
            retention.dt_in_plasma_retention_measured,
        )
    )
    assert pump.source_and_pump_gate_pass == all(
        (
            pump.measured_wall_plug_energy_per_polarized_dt_pair_available,
            pump.measured_pair_energy_below_break_even,
            pump.cryogenic_and_microwave_power_accounted,
            pump.tritium_handling_power_accounted,
            pump.recycle_depolarization_and_repolarization_accounted,
            pump.net_incremental_energy_positive_demonstrated,
        )
    )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (audit_spin_reaction, {"temperature_kev": 0.0}),
        (audit_spin_reaction, {"temperature_kev": 0.199999}),
        (audit_spin_reaction, {"temperature_kev": 100.000001}),
        (audit_spin_reaction, {"deuteron_polarization": 1.1}),
        (audit_spin_reaction, {"triton_polarization": True}),
        (audit_thermal_equilibrium_polarization, {"target_fractional_gain": 0.5}),
        (audit_thermal_equilibrium_polarization, {"temperature_kev": 0.199999}),
        (audit_thermal_equilibrium_polarization, {"temperature_kev": 100.000001}),
        (audit_polarization_retention, {"burn_weighted_polarization_product": True}),
        (audit_polarized_fuel_pump_ledger, {"single_pass_burn_fraction": 0.0}),
        (audit_polarized_fuel_pump_ledger, {"single_pass_burn_fraction": 1.0}),
        (audit_polarized_fuel_pump_ledger, {"electric_conversion_efficiency": 0.0}),
        (audit_polarized_fuel_pump_ledger, {"engineering_margin_factor": 0.9}),
        (audit_polarized_fuel_pump_ledger, {"target_fractional_gain": 0.500001}),
        (
            audit_polarized_fuel_pump_ledger,
            {"measured_wall_plug_energy_per_polarized_dt_pair_ev": -1.0},
        ),
        (audit_polarized_fuel_pump_ledger, {"cryogenic_and_microwave_power_accounted": 1}),
    ],
)
def test_invalid_inputs_fail_closed(function, kwargs) -> None:
    with pytest.raises(ValueError):
        function(**kwargs)


def test_tiny_target_cannot_pass_via_absolute_floating_point_tolerance() -> None:
    audit = audit_spin_reaction(
        deuteron_polarization=1.0e-10,
        triton_polarization=1.0e-10,
        target_fractional_gain=1.0e-16,
    )

    assert audit.polarization_product == pytest.approx(1.0e-20, rel=1.0e-15, abs=0.0)
    assert audit.fractional_reactivity_gain == pytest.approx(5.0e-21, rel=1.0e-15, abs=0.0)
    assert not audit.ideal_projector_arithmetic_target_reached


def test_product_just_below_target_never_passes_via_relative_tolerance() -> None:
    audit = audit_spin_reaction(
        deuteron_polarization=0.2,
        triton_polarization=0.1,
        target_fractional_gain=0.01000000000000005,
    )

    assert audit.polarization_product < audit.required_polarization_product
    assert not audit.ideal_projector_arithmetic_target_reached


def test_tiny_fixed_exposure_gain_uses_cancellation_safe_increment() -> None:
    audit = audit_polarized_fuel_pump_ledger(target_fractional_gain=1.0e-16)

    assert audit.fixed_exposure_incremental_burn_fraction == pytest.approx(
        9.949832494966426e-19,
        rel=2.0e-15,
        abs=0.0,
    )
    assert audit.incremental_fusion_energy_budget_per_injected_pair_kev > 0.0


def test_extreme_zeeman_argument_is_numerically_stable() -> None:
    polarization = _equilibrium_deuteron_polarization(1.0e14, 1.0e-12)

    assert math.isfinite(polarization)
    assert polarization == pytest.approx(1.0)


def test_direct_physical_gate_rejects_nonfinite_out_of_domain_and_zero_net_values(report) -> None:
    target = replace(
        report.target_reaction,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    retention = replace(
        report.retention,
        burn_weighted_polarization_product=report.retention.burn_weighted_product_required,
        burn_weighted_product_measured=True,
        burn_weighted_product_meets_required_threshold=True,
        burn_weighted_product_not_above_source_product=True,
        source_polarization_product_meets_required_threshold=True,
        deuteron_pellet_source_demonstrated_at_reactor_throughput=True,
        tritium_polarization_source_demonstrated_at_reactor_throughput=True,
        dt_in_plasma_retention_measured=True,
        injection_and_retention_gate_pass=True,
    )
    pump = replace(
        report.pump_ledger,
        measured_wall_plug_energy_per_polarized_dt_pair_ev=(
            report.pump_ledger.engineering_margin_energy_per_injected_pair_ev
        ),
        measured_wall_plug_energy_per_polarized_dt_pair_available=True,
        measured_pair_energy_below_break_even=True,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_depolarization_and_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
        source_and_pump_gate_pass=True,
    )
    kwargs = {
        "target_reaction": target,
        "source_design_reaction": report.source_design_reaction,
        "retention": retention,
        "pump_ledger": pump,
    }
    assert physical_spin_branch_gate_pass(**kwargs)

    for value in (-1.0, 1.0001, math.inf, math.nan):
        failed = dict(kwargs)
        failed["retention"] = replace(retention, burn_weighted_polarization_product=value)
        assert not physical_spin_branch_gate_pass(**failed)

    for value in (
        -1.0,
        math.inf,
        math.nan,
        pump.electrical_break_even_energy_per_injected_pair_ev,
    ):
        failed = dict(kwargs)
        failed["pump_ledger"] = replace(
            pump,
            measured_wall_plug_energy_per_polarized_dt_pair_ev=value,
        )
        assert not physical_spin_branch_gate_pass(**failed)

    target_mutations = (
        {"temperature_kev": math.nan},
        {"temperature_kev": 0.1},
        {"deuteron_vector_polarization": 2.0},
    )
    for mutation in target_mutations:
        failed = dict(kwargs)
        failed["target_reaction"] = replace(target, **mutation)
        assert not physical_spin_branch_gate_pass(**failed)

    below_target = replace(
        audit_spin_reaction(
            deuteron_polarization=0.1,
            triton_polarization=0.1,
            target_fractional_gain=0.01,
        ),
        ideal_projector_arithmetic_target_reached=True,
        conditional_ideal_projector_target_reached=True,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    failed = dict(kwargs)
    failed["target_reaction"] = below_target
    assert not physical_spin_branch_gate_pass(**failed)

    failed = dict(kwargs)
    failed["source_design_reaction"] = replace(
        report.source_design_reaction,
        deuteron_vector_polarization=2.0,
    )
    assert not physical_spin_branch_gate_pass(**failed)
    failed = dict(kwargs)
    failed["source_design_reaction"] = audit_spin_reaction(
        temperature_kev=0.2,
        deuteron_polarization=0.5,
        triton_polarization=0.1,
    )
    assert not physical_spin_branch_gate_pass(**failed)

    for field in ("declared_single_pass_burn_fraction", "declared_electric_conversion_efficiency"):
        failed = dict(kwargs)
        failed["pump_ledger"] = replace(pump, **{field: math.nan})
        assert not physical_spin_branch_gate_pass(**failed)

    forged_ceiling = pump.electrical_break_even_energy_per_injected_pair_ev * 2.0
    failed = dict(kwargs)
    failed["pump_ledger"] = replace(
        pump,
        measured_wall_plug_energy_per_polarized_dt_pair_ev=(
            pump.electrical_break_even_energy_per_injected_pair_ev * 1.5
        ),
        electrical_break_even_energy_per_injected_pair_ev=forged_ceiling,
        measured_pair_energy_below_break_even=True,
        net_incremental_energy_positive_demonstrated=True,
        source_and_pump_gate_pass=True,
    )
    assert not physical_spin_branch_gate_pass(**failed)

    for component, field, value in (
        ("target_reaction", "energy_dependent_polarized_cross_section_integrated", "yes"),
        ("retention", "injection_and_retention_gate_pass", 1),
        ("pump_ledger", "source_and_pump_gate_pass", "yes"),
    ):
        failed = dict(kwargs)
        failed[component] = replace(kwargs[component], **{field: value})
        assert not physical_spin_branch_gate_pass(**failed)


def test_physical_gate_requires_one_percent_and_shared_component_target(report) -> None:
    tiny_target = 1.0e-3
    tiny_reaction = replace(
        audit_spin_reaction(
            deuteron_polarization=math.sqrt(2.0 * tiny_target),
            triton_polarization=math.sqrt(2.0 * tiny_target),
            target_fractional_gain=tiny_target,
        ),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    tiny_source = audit_spin_reaction(
        deuteron_polarization=0.5,
        triton_polarization=0.1,
        target_fractional_gain=tiny_target,
    )
    tiny_retention = audit_polarization_retention(
        target_fractional_gain=tiny_target,
        burn_weighted_polarization_product=2.0 * tiny_target,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    tiny_pump = audit_polarized_fuel_pump_ledger(
        target_fractional_gain=tiny_target,
        measured_wall_plug_energy_per_polarized_dt_pair_ev=1.0,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
    )
    assert not physical_spin_branch_gate_pass(
        target_reaction=tiny_reaction,
        source_design_reaction=tiny_source,
        retention=tiny_retention,
        pump_ledger=tiny_pump,
    )

    one_percent_target = replace(
        report.target_reaction,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    complete_retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.02,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    mismatched_pump = audit_polarized_fuel_pump_ledger(
        target_fractional_gain=0.5,
        measured_wall_plug_energy_per_polarized_dt_pair_ev=1000.0,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
    )
    assert mismatched_pump.source_and_pump_gate_pass
    assert not physical_spin_branch_gate_pass(
        target_reaction=one_percent_target,
        source_design_reaction=report.source_design_reaction,
        retention=complete_retention,
        pump_ledger=mismatched_pump,
    )


def test_component_pass_statuses_and_strict_break_even_boundary(report) -> None:
    retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.02,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    pump = audit_polarized_fuel_pump_ledger(
        measured_wall_plug_energy_per_polarized_dt_pair_ev=70.0,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
    )
    equality = audit_polarized_fuel_pump_ledger(
        measured_wall_plug_energy_per_polarized_dt_pair_ev=(
            report.pump_ledger.electrical_break_even_energy_per_injected_pair_ev
        ),
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
    )
    insufficient_retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.019999,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )

    assert retention.injection_and_retention_gate_pass
    assert retention.status == "SOURCE_AND_BURN_WEIGHTED_RETENTION_GATE_PASS"
    assert pump.source_and_pump_gate_pass
    assert pump.status == "MEASURED_SOURCE_AND_PUMP_LEDGER_PASS"
    assert not equality.measured_pair_energy_below_break_even
    assert not equality.net_incremental_energy_positive_demonstrated
    assert not equality.source_and_pump_gate_pass
    assert equality.status == "MEASURED_PUMP_LEDGER_PRESENT_BUT_NET_OR_ACCOUNTING_GATE_FAIL"
    assert not insufficient_retention.injection_and_retention_gate_pass
    assert (
        insufficient_retention.status
        == "MEASURED_OR_DECLARED_RETENTION_INPUT_PRESENT_BUT_GATE_FAIL"
    )


def test_canonical_recompute_accepts_a_valid_shared_two_percent_control() -> None:
    target = replace(
        audit_spin_reaction(
            deuteron_polarization=0.2,
            triton_polarization=0.2,
            target_fractional_gain=0.02,
        ),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    source = audit_spin_reaction(
        deuteron_polarization=0.5,
        triton_polarization=0.1,
        target_fractional_gain=0.02,
    )
    retention = audit_polarization_retention(
        target_fractional_gain=0.02,
        burn_weighted_polarization_product=0.04,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    pump = audit_polarized_fuel_pump_ledger(
        target_fractional_gain=0.02,
        measured_wall_plug_energy_per_polarized_dt_pair_ev=1.0,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
    )

    assert retention.injection_and_retention_gate_pass
    assert pump.source_and_pump_gate_pass
    assert physical_spin_branch_gate_pass(
        target_reaction=target,
        source_design_reaction=source,
        retention=retention,
        pump_ledger=pump,
    )
