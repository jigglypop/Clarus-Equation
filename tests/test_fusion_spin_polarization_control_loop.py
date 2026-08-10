from __future__ import annotations

from dataclasses import replace
import math

import pytest

from reality_stone.clarus.fusion_spin_polarization_control_loop import (
    _equilibrium_deuteron_polarization,
    _spin_evidence_join_pass,
    _spin_component_gate_pass,
    audit_polarization_retention,
    audit_polarized_fuel_pump_ledger,
    audit_spin_reaction,
    audit_thermal_equilibrium_polarization,
    current_fusion_spin_polarization_control_report,
    physical_spin_branch_gate_pass,
)


OPERATOR_PROVENANCE_DIGEST = "sha256:" + "a" * 64
SOURCE_D_PROVENANCE_DIGEST = "sha256:" + "b" * 64
SOURCE_T_PROVENANCE_DIGEST = "sha256:" + "c" * 64
BURN_PROVENANCE_DIGEST = "sha256:" + "d" * 64
WALL_PLUG_PROVENANCE_DIGEST = "sha256:" + "e" * 64


class _AlwaysEqualProxy:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


@pytest.fixture(scope="module")
def report():
    return current_fusion_spin_polarization_control_report()


def _complete_pump(
    *,
    target_fractional_gain: float = 0.01,
    measured_energy_ev: float = 1.0,
    measured_energy_std_ev: float = 0.1,
    measured_energy_upper_ev: float | None = None,
    reference_fusion_power_w: float = 5.0e8,
    single_pass_burn_fraction: float = 0.01,
):
    seed = audit_polarized_fuel_pump_ledger(
        target_fractional_gain=target_fractional_gain,
        reference_fusion_power_w=reference_fusion_power_w,
        single_pass_burn_fraction=single_pass_burn_fraction,
    )
    return audit_polarized_fuel_pump_ledger(
        target_fractional_gain=target_fractional_gain,
        reference_fusion_power_w=reference_fusion_power_w,
        single_pass_burn_fraction=single_pass_burn_fraction,
        measured_wall_plug_energy_per_injected_dt_pair_ev=measured_energy_ev,
        measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev=(
            measured_energy_std_ev
        ),
        measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev=(
            measured_energy_ev + 3.0 * measured_energy_std_ev
            if measured_energy_upper_ev is None
            else measured_energy_upper_ev
        ),
        measured_wall_plug_energy_3sigma_bound_model=(
            "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
        ),
        wall_plug_measurement_duration_s=1.0,
        wall_plug_measurement_injected_dt_pair_count=(seed.declared_dt_pair_injection_rate_s),
        wall_plug_measurement_provenance=WALL_PLUG_PROVENANCE_DIGEST,
        cryogenic_and_microwave_power_accounted=True,
        tritium_handling_power_accounted=True,
        recycle_repolarization_accounted=True,
        net_incremental_energy_positive_demonstrated=True,
    )


def test_unpolarized_limit_recovers_bosch_hale(report) -> None:
    audit = audit_spin_reaction(
        deuteron_polarization=0.0,
        triton_polarization=0.0,
    )

    assert audit.quartet_probability == pytest.approx(2.0 / 3.0)
    assert audit.deuteron_tensor_polarization == 0.0
    assert (
        audit.deuteron_mplus_population,
        audit.deuteron_mzero_population,
        audit.deuteron_mminus_population,
    ) == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    assert audit.deuteron_spin_density_matrix_physical
    assert audit.cross_section_multiplier == 1.0
    assert audit.maxwellian_reactivity_ratio == 1.0
    assert audit.fractional_reactivity_gain == 0.0
    assert not audit.ideal_projector_arithmetic_target_reached


def test_one_percent_requires_polarization_product_point_zero_two(report) -> None:
    audit = report.target_reaction

    assert audit.required_polarization_product == pytest.approx(0.02)
    assert audit.deuteron_tensor_polarization == 0.0
    assert sum(
        (
            audit.deuteron_mplus_population,
            audit.deuteron_mzero_population,
            audit.deuteron_mminus_population,
        )
    ) == pytest.approx(1.0)
    assert (
        min(
            audit.deuteron_mplus_population,
            audit.deuteron_mzero_population,
            audit.deuteron_mminus_population,
        )
        >= 0.0
    )
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
    assert audit.energy_dependent_polarized_reaction_operator_provenance_digest is None


def test_source_design_has_retention_margin_but_not_an_engineering_pass(report) -> None:
    reaction = report.source_design_reaction
    retention = report.retention

    assert reaction.polarization_product == pytest.approx(0.05)
    assert reaction.fractional_reactivity_gain == pytest.approx(0.025)
    assert retention.minimum_product_retention_fraction == pytest.approx(0.4)
    assert retention.minimum_common_species_retention_fraction == pytest.approx(math.sqrt(0.4))
    assert retention.source_deuteron_tensor_polarization == 0.0
    assert retention.required_reaction_point_deuteron_tensor_polarization == 0.0
    assert retention.high_frequency_wave_depolarization_small_in_simulation
    assert retention.partial_helium3_plasma_retention_evidence_available
    assert not retention.burn_weighted_product_measured
    assert retention.burn_weighted_polarization_product is None
    assert retention.burn_weighted_deuteron_tensor_polarization is None
    assert retention.burn_weighted_product_tensor_physical_upper_bound is None
    assert retention.burn_weighted_measurement_provenance is None
    assert not retention.burn_weighted_product_meets_required_threshold
    assert not retention.burn_weighted_deuteron_tensor_polarization_measured
    assert not retention.burn_weighted_deuteron_tensor_matches_required_state
    assert not retention.burn_weighted_product_with_tensor_physical_bound_pass
    assert not retention.burn_weighted_measurement_provenance_available
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
    assert audit.electrical_break_even_energy_per_baseline_reacted_pair_ev == pytest.approx(
        70043.30091
    )
    assert audit.electrical_break_even_energy_per_incremental_fusion_reaction_ev == pytest.approx(
        7.039999999e6
    )
    assert audit.maximum_wall_plug_pump_power_for_net_gain_w == pytest.approx(1.989866503e6)
    assert audit.engineering_margin_energy_per_injected_pair_ev == pytest.approx(70.04330091)
    assert audit.engineering_margin_wall_plug_power_w == pytest.approx(1.989866503e5)
    assert audit.reference_ideal_polarized_deuterium_beam_target_rate_s == pytest.approx(1.0e18)
    assert audit.reactor_per_species_to_reference_deuterium_beam_target_ratio > 1.7e4
    assert audit.energy_ledger_denominator == "injected D-T pair"
    assert audit.reference_throughput_calculation_declared
    assert not audit.measured_wall_plug_energy_per_injected_dt_pair_available
    assert audit.measured_wall_plug_energy_per_injected_dt_pair_ev is None
    assert not audit.wall_plug_measurement_uncertainty_available
    assert not audit.measured_injected_pair_energy_below_break_even
    assert not audit.measured_injected_pair_energy_below_engineering_margin
    assert not audit.uncertainty_upper_below_engineering_margin
    assert not audit.wall_plug_measurement_pair_flow_meets_declared_throughput
    assert not audit.wall_plug_measurement_provenance_available
    assert not audit.net_incremental_energy_positive_demonstrated
    assert not audit.source_and_pump_gate_pass


def test_final_report_does_not_promote_the_control_to_ce(report) -> None:
    assert report.schema_version == "fusion-spin-polarization-control-v10"
    assert report.standard_model_non_ce_mechanism_identified
    assert report.conditional_ideal_projector_one_percent_control_derived
    assert not report.full_source_and_pump_ledger_closed
    assert not report.physical_spin_polarized_branch_pass
    assert not report.ce_new_physics_used
    assert not report.physical_ce_one_percent_branch_pass
    assert report.published_evidence.energy_dependent_figure_control_reproduced
    assert not report.published_evidence.physical_reaction_evidence_gate_pass
    assert not report.published_evidence.physical_source_evidence_gate_pass
    assert not (
        report.published_evidence.energy_resolved_burn_operator_sufficient_statistics_available
    )
    assert report.maximum_supported_stage == (
        "STANDARD_MODEL_ENERGY_DEPENDENT_FIGURE_DIGITIZED_CONTROL_"
        "SOURCE_THROUGHPUT_AND_RETENTION_FAIL_CLOSED"
    )


def test_physical_spin_gate_rejects_foreign_proxy_types(report) -> None:
    kwargs = {
        "target_reaction": report.target_reaction,
        "source_design_reaction": report.source_design_reaction,
        "retention": report.retention,
        "pump_ledger": report.pump_ledger,
        "published_evidence": report.published_evidence,
    }
    for field in tuple(kwargs):
        proxied = dict(kwargs)
        proxied[field] = _AlwaysEqualProxy(kwargs[field])
        assert not physical_spin_branch_gate_pass(**proxied)


def test_physical_gate_requires_every_leaf_blocker(report) -> None:
    target = replace(
        report.target_reaction,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    retention = replace(
        report.retention,
        burn_weighted_product_measured=True,
        burn_weighted_polarization_product=report.retention.burn_weighted_product_required,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_product_tensor_physical_upper_bound=2.0 / 3.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        burn_weighted_deuteron_tensor_polarization_measured=True,
        burn_weighted_deuteron_tensor_matches_required_state=True,
        burn_weighted_product_with_tensor_physical_bound_pass=True,
        burn_weighted_measurement_provenance_available=True,
        burn_weighted_product_meets_required_threshold=True,
        burn_weighted_product_not_above_source_product=True,
        source_polarization_product_meets_required_threshold=True,
        deuteron_pellet_source_demonstrated_at_reactor_throughput=True,
        tritium_polarization_source_demonstrated_at_reactor_throughput=True,
        dt_in_plasma_retention_measured=True,
        injection_and_retention_gate_pass=True,
    )
    pump = _complete_pump()
    kwargs = {
        "target_reaction": target,
        "source_design_reaction": report.source_design_reaction,
        "retention": retention,
        "pump_ledger": pump,
    }

    assert _spin_component_gate_pass(**kwargs)
    assert not physical_spin_branch_gate_pass(
        **kwargs,
        published_evidence=report.published_evidence,
    )
    for field in (
        "ideal_projector_arithmetic_target_reached",
        "deuteron_spin_density_matrix_physical",
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
        assert not _spin_component_gate_pass(**failed)
    failed = dict(kwargs)
    failed["target_reaction"] = replace(
        target,
        energy_dependent_polarized_reaction_operator_provenance_digest="not-a-digest",
    )
    assert not _spin_component_gate_pass(**failed)
    for field in (
        "ideal_projector_arithmetic_target_reached",
        "deuteron_spin_density_matrix_physical",
        "conditional_ideal_projector_target_reached",
    ):
        failed = dict(kwargs)
        failed["source_design_reaction"] = replace(report.source_design_reaction, **{field: False})
        assert not _spin_component_gate_pass(**failed)
    for field in (
        "burn_weighted_product_measured",
        "burn_weighted_product_meets_required_threshold",
        "burn_weighted_product_not_above_source_product",
        "burn_weighted_deuteron_tensor_polarization_measured",
        "burn_weighted_deuteron_tensor_matches_required_state",
        "burn_weighted_product_with_tensor_physical_bound_pass",
        "burn_weighted_measurement_provenance_available",
        "source_polarization_product_meets_required_threshold",
        "deuteron_pellet_source_demonstrated_at_reactor_throughput",
        "tritium_polarization_source_demonstrated_at_reactor_throughput",
        "dt_in_plasma_retention_measured",
        "injection_and_retention_gate_pass",
    ):
        failed = dict(kwargs)
        failed["retention"] = replace(retention, **{field: False})
        assert not _spin_component_gate_pass(**failed)
    assert not _spin_component_gate_pass(
        **{
            **kwargs,
            "retention": replace(retention, burn_weighted_polarization_product=None),
        }
    )
    assert not _spin_component_gate_pass(
        **{
            **kwargs,
            "retention": replace(
                retention,
                burn_weighted_deuteron_tensor_polarization=None,
            ),
        }
    )
    assert not _spin_component_gate_pass(
        **{
            **kwargs,
            "retention": replace(
                retention,
                burn_weighted_deuteron_tensor_polarization=0.1,
            ),
        }
    )
    assert not _spin_component_gate_pass(
        **{
            **kwargs,
            "retention": replace(
                retention,
                burn_weighted_measurement_provenance="",
            ),
        }
    )
    assert not _spin_component_gate_pass(
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
        "reference_throughput_calculation_declared",
        "fixed_exposure_burn_probability_model_evaluated",
        "conservative_fixed_exposure_budget_used",
        "measured_wall_plug_energy_per_injected_dt_pair_available",
        "wall_plug_measurement_uncertainty_available",
        "measured_injected_pair_energy_below_break_even",
        "measured_injected_pair_energy_below_engineering_margin",
        "uncertainty_upper_below_engineering_margin",
        "wall_plug_measurement_pair_flow_meets_declared_throughput",
        "wall_plug_measurement_provenance_available",
        "cryogenic_and_microwave_power_accounted",
        "tritium_handling_power_accounted",
        "recycle_depolarization_and_repolarization_accounted",
        "net_incremental_energy_positive_demonstrated",
        "source_and_pump_gate_pass",
    ):
        failed = dict(kwargs)
        failed["pump_ledger"] = replace(pump, **{field: False})
        assert not _spin_component_gate_pass(**failed)
    assert not _spin_component_gate_pass(
        **{
            **kwargs,
            "pump_ledger": replace(
                pump,
                measured_wall_plug_energy_per_injected_dt_pair_ev=None,
            ),
        }
    )
    assert not _spin_component_gate_pass(
        **{
            **kwargs,
            "pump_ledger": replace(
                pump,
                measured_wall_plug_energy_per_injected_dt_pair_ev=(
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
            retention.burn_weighted_deuteron_tensor_polarization_measured,
            retention.burn_weighted_deuteron_tensor_matches_required_state,
            retention.burn_weighted_product_with_tensor_physical_bound_pass,
            retention.burn_weighted_measurement_provenance_available,
            retention.source_polarization_product_meets_required_threshold,
            retention.deuteron_pellet_source_demonstrated_at_reactor_throughput,
            retention.tritium_polarization_source_demonstrated_at_reactor_throughput,
            retention.dt_in_plasma_retention_measured,
        )
    )
    assert pump.source_and_pump_gate_pass == all(
        (
            pump.measured_wall_plug_energy_per_injected_dt_pair_available,
            pump.measured_injected_pair_energy_below_break_even,
            pump.measured_injected_pair_energy_below_engineering_margin,
            pump.wall_plug_measurement_uncertainty_available,
            pump.uncertainty_upper_below_engineering_margin,
            pump.wall_plug_measurement_pair_flow_meets_declared_throughput,
            pump.wall_plug_measurement_provenance_available,
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
        (audit_spin_reaction, {"deuteron_tensor_polarization": -2.1}),
        (
            audit_spin_reaction,
            {"deuteron_polarization": 1.0, "deuteron_tensor_polarization": 0.0},
        ),
        (audit_spin_reaction, {"triton_polarization": True}),
        (audit_thermal_equilibrium_polarization, {"target_fractional_gain": 0.5}),
        (audit_thermal_equilibrium_polarization, {"temperature_kev": 0.199999}),
        (audit_thermal_equilibrium_polarization, {"temperature_kev": 100.000001}),
        (audit_polarization_retention, {"burn_weighted_polarization_product": True}),
        (
            audit_polarization_retention,
            {"source_deuteron_tensor_polarization": 1.1},
        ),
        (
            audit_polarization_retention,
            {
                "source_deuteron_polarization": 0.5,
                "source_deuteron_tensor_polarization": -2.0,
            },
        ),
        (
            audit_polarization_retention,
            {"burn_weighted_deuteron_tensor_polarization": -2.1},
        ),
        (
            audit_polarization_retention,
            {"burn_weighted_measurement_provenance": 123},
        ),
        (audit_polarized_fuel_pump_ledger, {"single_pass_burn_fraction": 0.0}),
        (audit_polarized_fuel_pump_ledger, {"single_pass_burn_fraction": 1.0}),
        (audit_polarized_fuel_pump_ledger, {"electric_conversion_efficiency": 0.0}),
        (audit_polarized_fuel_pump_ledger, {"engineering_margin_factor": 0.9}),
        (audit_polarized_fuel_pump_ledger, {"target_fractional_gain": 0.500001}),
        (
            audit_polarized_fuel_pump_ledger,
            {"measured_wall_plug_energy_per_injected_dt_pair_ev": -1.0},
        ),
        (
            audit_polarized_fuel_pump_ledger,
            {
                "measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev": -1.0
            },
        ),
        (audit_polarized_fuel_pump_ledger, {"wall_plug_measurement_duration_s": 0.0}),
        (
            audit_polarized_fuel_pump_ledger,
            {"wall_plug_measurement_injected_dt_pair_count": 0.0},
        ),
        (
            audit_polarized_fuel_pump_ledger,
            {"wall_plug_measurement_provenance": 123},
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
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    retention = replace(
        report.retention,
        burn_weighted_polarization_product=report.retention.burn_weighted_product_required,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_product_tensor_physical_upper_bound=2.0 / 3.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        burn_weighted_product_measured=True,
        burn_weighted_deuteron_tensor_polarization_measured=True,
        burn_weighted_deuteron_tensor_matches_required_state=True,
        burn_weighted_product_with_tensor_physical_bound_pass=True,
        burn_weighted_measurement_provenance_available=True,
        burn_weighted_product_meets_required_threshold=True,
        burn_weighted_product_not_above_source_product=True,
        source_polarization_product_meets_required_threshold=True,
        deuteron_pellet_source_demonstrated_at_reactor_throughput=True,
        tritium_polarization_source_demonstrated_at_reactor_throughput=True,
        dt_in_plasma_retention_measured=True,
        injection_and_retention_gate_pass=True,
    )
    pump = _complete_pump()
    kwargs = {
        "target_reaction": target,
        "source_design_reaction": report.source_design_reaction,
        "retention": retention,
        "pump_ledger": pump,
    }
    assert _spin_component_gate_pass(**kwargs)

    for value in (-1.0, 1.0001, math.inf, math.nan):
        failed = dict(kwargs)
        failed["retention"] = replace(retention, burn_weighted_polarization_product=value)
        assert not _spin_component_gate_pass(**failed)

    for value in (-2.1, 1.1, math.inf, math.nan):
        failed = dict(kwargs)
        failed["retention"] = replace(
            retention,
            burn_weighted_deuteron_tensor_polarization=value,
        )
        assert not _spin_component_gate_pass(**failed)

    for value in (
        -1.0,
        math.inf,
        math.nan,
        pump.electrical_break_even_energy_per_injected_pair_ev,
    ):
        failed = dict(kwargs)
        failed["pump_ledger"] = replace(
            pump,
            measured_wall_plug_energy_per_injected_dt_pair_ev=value,
        )
        assert not _spin_component_gate_pass(**failed)

    target_mutations = (
        {"temperature_kev": math.nan},
        {"temperature_kev": 0.1},
        {"deuteron_vector_polarization": 2.0},
        {"deuteron_tensor_polarization": 2.0},
        {"deuteron_mplus_population": 0.99},
    )
    for mutation in target_mutations:
        failed = dict(kwargs)
        failed["target_reaction"] = replace(target, **mutation)
        assert not _spin_component_gate_pass(**failed)

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
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    failed = dict(kwargs)
    failed["target_reaction"] = below_target
    assert not _spin_component_gate_pass(**failed)

    failed = dict(kwargs)
    failed["source_design_reaction"] = replace(
        report.source_design_reaction,
        deuteron_vector_polarization=2.0,
    )
    assert not _spin_component_gate_pass(**failed)
    failed = dict(kwargs)
    failed["source_design_reaction"] = audit_spin_reaction(
        temperature_kev=0.2,
        deuteron_polarization=0.5,
        triton_polarization=0.1,
    )
    assert not _spin_component_gate_pass(**failed)

    for field in ("declared_single_pass_burn_fraction", "declared_electric_conversion_efficiency"):
        failed = dict(kwargs)
        failed["pump_ledger"] = replace(pump, **{field: math.nan})
        assert not _spin_component_gate_pass(**failed)

    forged_ceiling = pump.electrical_break_even_energy_per_injected_pair_ev * 2.0
    failed = dict(kwargs)
    failed["pump_ledger"] = replace(
        pump,
        measured_wall_plug_energy_per_injected_dt_pair_ev=(
            pump.electrical_break_even_energy_per_injected_pair_ev * 1.5
        ),
        electrical_break_even_energy_per_injected_pair_ev=forged_ceiling,
        measured_injected_pair_energy_below_break_even=True,
        net_incremental_energy_positive_demonstrated=True,
        source_and_pump_gate_pass=True,
    )
    assert not _spin_component_gate_pass(**failed)

    for component, field, value in (
        ("target_reaction", "energy_dependent_polarized_cross_section_integrated", "yes"),
        ("retention", "injection_and_retention_gate_pass", 1),
        ("pump_ledger", "source_and_pump_gate_pass", "yes"),
    ):
        failed = dict(kwargs)
        failed[component] = replace(kwargs[component], **{field: value})
        assert not _spin_component_gate_pass(**failed)


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
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
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
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    tiny_pump = _complete_pump(
        target_fractional_gain=tiny_target,
    )
    assert not _spin_component_gate_pass(
        target_reaction=tiny_reaction,
        source_design_reaction=tiny_source,
        retention=tiny_retention,
        pump_ledger=tiny_pump,
    )

    one_percent_target = replace(
        report.target_reaction,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    complete_retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.02,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    mismatched_pump = _complete_pump(
        target_fractional_gain=0.5,
        measured_energy_ev=1000.0,
    )
    assert mismatched_pump.source_and_pump_gate_pass
    assert not _spin_component_gate_pass(
        target_reaction=one_percent_target,
        source_design_reaction=report.source_design_reaction,
        retention=complete_retention,
        pump_ledger=mismatched_pump,
    )


def test_tensor_state_must_join_source_retention_and_reaction(report) -> None:
    target = replace(
        audit_spin_reaction(deuteron_tensor_polarization=-1.0),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    source = audit_spin_reaction(
        deuteron_polarization=0.5,
        deuteron_tensor_polarization=1.0,
        triton_polarization=0.1,
    )
    mismatched_retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.02,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    measured_transition = audit_polarization_retention(
        source_deuteron_tensor_polarization=1.0,
        target_deuteron_tensor_polarization=-1.0,
        burn_weighted_polarization_product=0.02,
        burn_weighted_deuteron_tensor_polarization=-1.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    pump = _complete_pump()

    assert not _spin_component_gate_pass(
        target_reaction=target,
        source_design_reaction=source,
        retention=mismatched_retention,
        pump_ledger=pump,
    )
    assert measured_transition.burn_weighted_deuteron_tensor_polarization_measured
    assert measured_transition.burn_weighted_deuteron_tensor_matches_required_state
    assert _spin_component_gate_pass(
        target_reaction=target,
        source_design_reaction=source,
        retention=measured_transition,
        pump_ledger=pump,
    )
    assert not physical_spin_branch_gate_pass(
        target_reaction=target,
        source_design_reaction=source,
        retention=measured_transition,
        pump_ledger=pump,
        published_evidence=report.published_evidence,
    )


def test_component_and_evidence_are_locked_to_the_same_temperature_and_plant(report) -> None:
    target_20_kev = replace(
        audit_spin_reaction(temperature_kev=20.0),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    source_20_kev = audit_spin_reaction(
        temperature_kev=20.0,
        deuteron_polarization=0.5,
        triton_polarization=0.1,
    )
    retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.02,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    default_target = replace(
        report.target_reaction,
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    default_pump = _complete_pump()
    alternate_plant_pump = _complete_pump(
        reference_fusion_power_w=1.0e9,
        single_pass_burn_fraction=0.02,
    )
    state_resolved_evidence = replace(
        report.published_evidence,
        reaction=replace(
            report.published_evidence.reaction,
            validated_target_deuteron_vector_polarization=(
                default_target.deuteron_vector_polarization
            ),
            validated_target_deuteron_tensor_polarization=(
                default_target.deuteron_tensor_polarization
            ),
            validated_target_triton_vector_polarization=(default_target.triton_vector_polarization),
            validated_target_state_maxwellian_reactivity_ratio=(
                default_target.maxwellian_reactivity_ratio
            ),
            validated_target_state_maxwellian_reactivity_ratio_std=0.0,
            validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=(
                default_target.maxwellian_reactivity_ratio
            ),
            target_state_resolved_operator_available=True,
            target_state_operator_provenance_digest=OPERATOR_PROVENANCE_DIGEST,
        ),
        source=replace(
            report.published_evidence.source,
            validated_reactor_rate_deuterium_source_rate_s=(
                1.1 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_tritium_source_rate_s=(
                1.1 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_deuterium_source_rate_linearized_std_s=(
                0.01 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_tritium_source_rate_linearized_std_s=(
                0.01 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_deuterium_source_rate_lower_3sigma_s=(
                1.07 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_tritium_source_rate_lower_3sigma_s=(
                1.07 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_deuterium_source_rate_3sigma_bound_model=(
                "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
            ),
            validated_reactor_rate_tritium_source_rate_3sigma_bound_model=(
                "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
            ),
            validated_reactor_rate_deuterium_measurement_duration_s=1.0,
            validated_reactor_rate_tritium_measurement_duration_s=1.0,
            validated_reactor_rate_deuterium_output_count=(
                1.1 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_tritium_output_count=(
                1.1 * default_pump.declared_dt_pair_injection_rate_s
            ),
            validated_reactor_rate_deuteron_vector_polarization=(
                report.source_design_reaction.deuteron_vector_polarization
            ),
            validated_reactor_rate_deuteron_vector_polarization_std=0.0,
            validated_reactor_rate_deuteron_tensor_polarization=(
                report.source_design_reaction.deuteron_tensor_polarization
            ),
            validated_reactor_rate_deuteron_tensor_polarization_std=0.0,
            validated_reactor_rate_triton_vector_polarization=(
                report.source_design_reaction.triton_vector_polarization
            ),
            validated_reactor_rate_triton_vector_polarization_std=0.0,
            validated_reactor_rate_deuterium_source_provenance_digest=(SOURCE_D_PROVENANCE_DIGEST),
            validated_reactor_rate_tritium_source_provenance_digest=(SOURCE_T_PROVENANCE_DIGEST),
            validated_burn_weighted_dt_polarization_product=(
                retention.burn_weighted_polarization_product
            ),
            validated_burn_weighted_dt_polarization_product_std=0.0,
            validated_burn_weighted_deuteron_tensor_polarization=(
                retention.burn_weighted_deuteron_tensor_polarization
            ),
            validated_burn_weighted_deuteron_tensor_polarization_std=0.0,
            validated_burn_retention_measurement_duration_s=1.0,
            validated_burn_retention_reaction_count=1.0e9,
            validated_burn_retention_provenance_digest=BURN_PROVENANCE_DIGEST,
            validated_wall_plug_energy_per_injected_dt_pair_ev=(
                default_pump.measured_wall_plug_energy_per_injected_dt_pair_ev
            ),
            validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev=(
                default_pump.measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev
            ),
            validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev=(
                default_pump.measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev
            ),
            validated_wall_plug_3sigma_bound_model=(
                default_pump.measured_wall_plug_energy_3sigma_bound_model
            ),
            validated_wall_plug_measurement_duration_s=(
                default_pump.wall_plug_measurement_duration_s
            ),
            validated_wall_plug_measurement_injected_dt_pair_count=(
                default_pump.wall_plug_measurement_injected_dt_pair_count
            ),
            validated_wall_plug_measurement_provenance=(
                default_pump.wall_plug_measurement_provenance
            ),
            energy_resolved_burn_operator_sufficient_statistics_available=True,
        ),
        energy_resolved_burn_operator_sufficient_statistics_available=True,
    )
    over_target = replace(
        audit_spin_reaction(deuteron_polarization=0.5, triton_polarization=0.1),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )

    assert alternate_plant_pump.source_and_pump_gate_pass
    assert _spin_evidence_join_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=default_pump,
        published_evidence=state_resolved_evidence,
    )
    assert not _spin_evidence_join_pass(
        target_reaction=over_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=default_pump,
        published_evidence=state_resolved_evidence,
    )
    assert not _spin_evidence_join_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=alternate_plant_pump,
        published_evidence=state_resolved_evidence,
    )
    assert not _spin_evidence_join_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=replace(
            retention,
            burn_weighted_measurement_provenance="sha256:" + "f" * 64,
        ),
        pump_ledger=default_pump,
        published_evidence=state_resolved_evidence,
    )
    evidence_with_burn_uncertainty_below_target = replace(
        state_resolved_evidence,
        source=replace(
            state_resolved_evidence.source,
            validated_burn_weighted_dt_polarization_product_std=0.001,
        ),
    )
    assert not _spin_evidence_join_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=default_pump,
        published_evidence=evidence_with_burn_uncertainty_below_target,
    )
    evidence_with_wrong_operator = replace(
        state_resolved_evidence,
        reaction=replace(
            state_resolved_evidence.reaction,
            target_state_operator_provenance_digest="sha256:" + "f" * 64,
        ),
    )
    assert not _spin_evidence_join_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=default_pump,
        published_evidence=evidence_with_wrong_operator,
    )
    assert not _spin_component_gate_pass(
        target_reaction=over_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=default_pump,
    )

    two_percent_target = replace(
        audit_spin_reaction(
            deuteron_polarization=0.2,
            triton_polarization=0.2,
            target_fractional_gain=0.02,
        ),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    two_percent_source = audit_spin_reaction(
        deuteron_polarization=0.5,
        triton_polarization=0.1,
        target_fractional_gain=0.02,
    )
    two_percent_retention = audit_polarization_retention(
        target_fractional_gain=0.02,
        burn_weighted_polarization_product=two_percent_target.polarization_product,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    two_percent_evidence = replace(
        state_resolved_evidence,
        reaction=replace(
            state_resolved_evidence.reaction,
            target_reactivity_ratio=1.02,
            validated_target_deuteron_vector_polarization=(
                two_percent_target.deuteron_vector_polarization
            ),
            validated_target_triton_vector_polarization=(
                two_percent_target.triton_vector_polarization
            ),
            validated_target_state_maxwellian_reactivity_ratio=(
                two_percent_target.maxwellian_reactivity_ratio
            ),
            validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=(
                two_percent_target.maxwellian_reactivity_ratio
            ),
        ),
        source=replace(
            state_resolved_evidence.source,
            validated_burn_weighted_dt_polarization_product=(
                two_percent_retention.burn_weighted_polarization_product
            ),
            validated_burn_weighted_dt_polarization_product_std=0.001,
        ),
    )
    assert not _spin_evidence_join_pass(
        target_reaction=two_percent_target,
        source_design_reaction=two_percent_source,
        retention=two_percent_retention,
        pump_ledger=_complete_pump(target_fractional_gain=0.02),
        published_evidence=two_percent_evidence,
    )
    assert _spin_evidence_join_pass(
        target_reaction=two_percent_target,
        source_design_reaction=two_percent_source,
        retention=two_percent_retention,
        pump_ledger=_complete_pump(target_fractional_gain=0.02),
        published_evidence=replace(
            two_percent_evidence,
            source=replace(
                two_percent_evidence.source,
                validated_burn_weighted_dt_polarization_product_std=0.0,
            ),
        ),
    )
    assert not _spin_component_gate_pass(
        target_reaction=target_20_kev,
        source_design_reaction=source_20_kev,
        retention=retention,
        pump_ledger=_complete_pump(),
    )
    assert not _spin_component_gate_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=alternate_plant_pump,
    )
    assert not physical_spin_branch_gate_pass(
        target_reaction=default_target,
        source_design_reaction=report.source_design_reaction,
        retention=retention,
        pump_ledger=alternate_plant_pump,
        published_evidence=report.published_evidence,
    )


def test_burn_product_and_tensor_moment_must_share_a_physical_spin_one_state() -> None:
    target = replace(
        audit_spin_reaction(
            deuteron_polarization=0.02,
            deuteron_tensor_polarization=-1.94,
            triton_polarization=1.0,
        ),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
    )
    source = audit_spin_reaction(
        deuteron_polarization=0.5,
        deuteron_tensor_polarization=1.0,
        triton_polarization=0.1,
    )
    impossible_retention = audit_polarization_retention(
        source_deuteron_tensor_polarization=1.0,
        target_deuteron_tensor_polarization=-1.94,
        burn_weighted_polarization_product=0.05,
        burn_weighted_deuteron_tensor_polarization=-1.94,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )

    assert impossible_retention.burn_weighted_product_tensor_physical_upper_bound == (
        pytest.approx(0.02)
    )
    assert not impossible_retention.burn_weighted_product_with_tensor_physical_bound_pass
    assert not impossible_retention.injection_and_retention_gate_pass
    assert not _spin_component_gate_pass(
        target_reaction=target,
        source_design_reaction=source,
        retention=impossible_retention,
        pump_ledger=_complete_pump(),
    )


def test_component_pass_statuses_and_strict_break_even_boundary(report) -> None:
    retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.02,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    pump = _complete_pump(measured_energy_ev=60.0, measured_energy_std_ev=1.0)
    equality = _complete_pump(
        measured_energy_ev=(report.pump_ledger.electrical_break_even_energy_per_injected_pair_ev),
        measured_energy_std_ev=0.0,
    )
    insufficient_retention = audit_polarization_retention(
        burn_weighted_polarization_product=0.019999,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )

    assert retention.injection_and_retention_gate_pass
    assert retention.status == "SOURCE_AND_BURN_WEIGHTED_RETENTION_GATE_PASS"
    assert pump.source_and_pump_gate_pass
    assert pump.status == "MEASURED_SOURCE_AND_PUMP_LEDGER_PASS"
    assert not equality.measured_injected_pair_energy_below_break_even
    assert not equality.net_incremental_energy_positive_demonstrated
    assert not equality.source_and_pump_gate_pass
    assert equality.status == "MEASURED_PUMP_LEDGER_PRESENT_BUT_NET_OR_ACCOUNTING_GATE_FAIL"
    assert not insufficient_retention.injection_and_retention_gate_pass
    assert (
        insufficient_retention.status
        == "MEASURED_OR_DECLARED_RETENTION_INPUT_PRESENT_BUT_GATE_FAIL"
    )


def test_pump_gate_enforces_margin_uncertainty_flow_and_provenance(report) -> None:
    central_only = _complete_pump(measured_energy_ev=699.0, measured_energy_std_ev=0.0)
    exact_margin = _complete_pump(
        measured_energy_ev=report.pump_ledger.engineering_margin_energy_per_injected_pair_ev,
        measured_energy_std_ev=0.0,
    )
    uncertainty_crosses_margin = _complete_pump(
        measured_energy_ev=69.0,
        measured_energy_std_ev=(
            report.pump_ledger.engineering_margin_energy_per_injected_pair_ev - 69.0
        )
        / 3.0
        + 1.0e-6,
    )
    complete = _complete_pump(measured_energy_ev=60.0, measured_energy_std_ev=1.0)
    nonlinear_box_upper_crosses_margin = _complete_pump(
        measured_energy_ev=1.0,
        measured_energy_std_ev=0.0,
        measured_energy_upper_ev=(
            report.pump_ledger.engineering_margin_energy_per_injected_pair_ev + 1.0
        ),
    )

    assert central_only.measured_injected_pair_energy_below_break_even
    assert not central_only.measured_injected_pair_energy_below_engineering_margin
    assert not central_only.source_and_pump_gate_pass
    assert not exact_margin.measured_injected_pair_energy_below_engineering_margin
    assert not exact_margin.source_and_pump_gate_pass
    assert not uncertainty_crosses_margin.uncertainty_upper_below_engineering_margin
    assert not uncertainty_crosses_margin.source_and_pump_gate_pass
    assert (
        nonlinear_box_upper_crosses_margin.measured_wall_plug_energy_per_injected_dt_pair_ev
        + 3.0
        * nonlinear_box_upper_crosses_margin.measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev
        < nonlinear_box_upper_crosses_margin.engineering_margin_energy_per_injected_pair_ev
    )
    assert not (
        nonlinear_box_upper_crosses_margin.uncertainty_upper_below_engineering_margin
    )
    assert not nonlinear_box_upper_crosses_margin.source_and_pump_gate_pass
    assert complete.wall_plug_measurement_pair_flow_rate_s == pytest.approx(
        complete.declared_dt_pair_injection_rate_s
    )
    assert complete.wall_plug_measurement_pair_flow_meets_declared_throughput
    assert complete.wall_plug_measurement_provenance == WALL_PLUG_PROVENANCE_DIGEST
    assert complete.source_and_pump_gate_pass

    no_provenance = replace(
        complete,
        wall_plug_measurement_provenance=None,
        wall_plug_measurement_provenance_available=False,
        source_and_pump_gate_pass=False,
    )
    assert not no_provenance.wall_plug_measurement_provenance_available
    assert not no_provenance.source_and_pump_gate_pass


def test_canonical_recompute_accepts_a_valid_shared_two_percent_control() -> None:
    target = replace(
        audit_spin_reaction(
            deuteron_polarization=0.2,
            triton_polarization=0.2,
            target_fractional_gain=0.02,
        ),
        energy_dependent_polarized_cross_section_integrated=True,
        energy_dependent_polarized_reaction_operator_provenance_pass=True,
        energy_dependent_polarized_reaction_operator_provenance_digest=(OPERATOR_PROVENANCE_DIGEST),
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
        burn_weighted_polarization_product=target.polarization_product,
        burn_weighted_deuteron_tensor_polarization=0.0,
        burn_weighted_measurement_provenance=BURN_PROVENANCE_DIGEST,
        deuteron_source_at_throughput=True,
        tritium_source_at_throughput=True,
        dt_retention_measured=True,
    )
    pump = _complete_pump(
        target_fractional_gain=0.02,
    )

    assert retention.injection_and_retention_gate_pass
    assert pump.source_and_pump_gate_pass
    assert _spin_component_gate_pass(
        target_reaction=target,
        source_design_reaction=source,
        retention=retention,
        pump_ledger=pump,
    )
