from __future__ import annotations

import math

import pytest

from reality_stone.clarus.fusion_equation_iteration_loop import (
    bosch_hale_dt_cross_section_m2,
    current_fusion_equation_iteration_report,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_equation_iteration_report(
        energy_points=181,
        wkb_grid_points=1001,
    )


def test_bosch_hale_dt_cross_section_reference_points() -> None:
    assert bosch_hale_dt_cross_section_m2(10.0) == pytest.approx(2.7022e-30, rel=3.0e-5)
    assert bosch_hale_dt_cross_section_m2(100.0) == pytest.approx(3.4272e-28, rel=3.0e-5)


def test_corrected_equation_loop_closes_without_resonance_multiplier(report) -> None:
    assert report.equation_chain_computationally_closed
    assert report.bosch_hale_numeric_to_closed_ratio == pytest.approx(1.0, rel=6.0e-3)
    assert report.allowed_broken_z2.action_traceable
    assert report.allowed_broken_z2.dt_scalar_charge_product == 6.0
    assert report.allowed_broken_z2.coherent_point_nucleus_upper_bound
    assert not report.allowed_broken_z2.selected_action_contains_interaction
    assert report.allowed_broken_z2.supplied_constraint_pass
    assert report.allowed_broken_z2.thermal_chain_closed_conditionally
    assert report.allowed_broken_z2.thermal_reactivity_ratio_minus_one > 0.0
    assert report.allowed_broken_z2.thermal_reactivity_ratio_minus_one == pytest.approx(
        6.18129e-10,
        rel=8.0e-5,
    )
    assert not report.allowed_broken_z2.engineering_gain_reached
    assert report.allowed_z2_pair.action_traceable
    assert report.allowed_z2_pair.selected_action_contains_interaction
    assert report.allowed_z2_pair.thermal_reactivity_ratio_minus_one > 0.0
    assert report.allowed_z2_pair.thermal_reactivity_ratio_minus_one == pytest.approx(
        3.83348e-18,
        rel=8.0e-5,
    )
    assert not report.allowed_z2_pair.engineering_gain_reached


def test_massless_unit_mixing_is_a_model_class_upper_bound_and_still_fails(report) -> None:
    allowed = report.allowed_broken_z2
    upper = report.massless_unit_mixing_upper_bound

    assert upper.scalar_mass_mev == 0.0
    assert upper.interaction_parameter == 1.0
    assert upper.thermal_reactivity_ratio_minus_one > allowed.thermal_reactivity_ratio_minus_one
    assert upper.thermal_reactivity_ratio_minus_one == pytest.approx(4.01944e-4, rel=8.0e-5)
    assert not upper.engineering_gain_reached
    assert not report.higgs_proportional_model_class_meets_target


def test_direct_coupling_solver_reaches_math_target_but_fails_physical_gate(report) -> None:
    direct = report.direct_coupling_requirement

    assert direct.mathematical_target_reached
    assert direct.required_direct_nucleon_coupling == pytest.approx(0.00569352, rel=3.0e-6)
    assert direct.equivalent_higgs_mixing_sine == pytest.approx(4.97583, rel=5.0e-6)
    assert direct.unit_mixing_bound_exceeded
    assert direct.supplied_mixing_bound_exceeded
    assert not direct.selected_portal_action_contains_direct_operator
    assert not direct.physical_gate_pass
    assert not report.current_selected_action_meets_target
    assert not report.physical_fusion_upgrade_derived
    assert (
        report.direct_coupling_registered_mass_requirement.required_direct_nucleon_coupling
        > direct.required_direct_nucleon_coupling
    )
    assert report.direct_coupling_registered_mass_requirement.equivalent_higgs_mixing_sine > 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature_kev": math.nan}, "finite"),
        ({"engineering_gain_target": 1.0}, "exceed one"),
        ({"energy_points": True}, "integer"),
        ({"energy_points": 40}, "at least 41"),
        ({"wkb_grid_points": 100}, "at least 101"),
    ],
)
def test_iteration_loop_invalid_inputs_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        current_fusion_equation_iteration_report(**kwargs)
