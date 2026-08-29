from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from examples.physics.constant_vacuum_flrw_propagation import (
    propagate_constant_vacuum_flat_flrw,
)
from examples.physics.lattice_scalar_transition_bridge import (
    match_constant_vacuum_on_cauchy_slice,
)


def _vacuum_transition():
    return match_constant_vacuum_on_cauchy_slice(
        battery_energy=25.0,
        spatial_volume=10.0,
        residual_efficiency=0.4,
    )


def test_declared_constant_vacuum_propagates_to_exact_de_sitter_background() -> None:
    transition = _vacuum_transition()
    result = propagate_constant_vacuum_flat_flrw(
        transition,
        newton_constant=3.0 / (8.0 * math.pi),
        evaluation_scale_factor=math.exp(2.0),
        global_constant_vacuum_action_adopted=True,
    )

    assert result.vacuum_energy_density == 1.0
    assert result.pressure == -1.0
    assert result.equation_of_state_parameter == -1.0
    assert result.hubble_rate == pytest.approx(1.0)
    assert result.elapsed_cosmic_time == pytest.approx(2.0)
    assert result.de_sitter_ricci_scalar == pytest.approx(12.0)
    assert result.conditional_constant_vacuum_flrw_propagation_closed
    assert result.background_covariant_conservation_closed
    assert result.status == "CONDITIONAL_CONSTANT_VACUUM_DE_SITTER_PROPAGATION_CLOSED"


@pytest.mark.parametrize("scale_factor", (0.25, 0.5, 1.0, 2.0, 10.0))
def test_constant_density_and_negative_pressure_work_balance(
    scale_factor: float,
) -> None:
    transition = _vacuum_transition()
    result = propagate_constant_vacuum_flat_flrw(
        transition,
        newton_constant=1.0,
        evaluation_scale_factor=scale_factor,
        global_constant_vacuum_action_adopted=True,
    )

    assert result.dimensionless_scale_factor_ratio == scale_factor
    assert result.vacuum_energy_density == transition.vacuum_density
    assert result.physical_volume == pytest.approx(
        transition.spatial_volume * scale_factor**3
    )
    assert result.vacuum_energy_in_comoving_cell == pytest.approx(
        transition.transferred_energy * scale_factor**3
    )
    assert result.continuity_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.friedmann_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.raychaudhuri_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.acceleration_equation_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.negative_pressure_work_residual == pytest.approx(0.0, abs=1.0e-10)
    assert result.scale_factor_solution_residual == pytest.approx(0.0, abs=1.0e-10)


def test_one_slice_energy_match_is_not_silently_promoted_to_global_vacuum() -> None:
    with pytest.raises(ValueError, match="one-slice energy matching is insufficient"):
        propagate_constant_vacuum_flat_flrw(
            _vacuum_transition(),
            newton_constant=1.0,
            evaluation_scale_factor=2.0,
            global_constant_vacuum_action_adopted=False,
        )


def test_wrong_vacuum_stress_is_rejected() -> None:
    transition = _vacuum_transition()
    wrong = replace(
        transition,
        vacuum_stress=np.diag((1.0, 0.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="vacuum stress"):
        propagate_constant_vacuum_flat_flrw(
            wrong,
            newton_constant=1.0,
            evaluation_scale_factor=2.0,
            global_constant_vacuum_action_adopted=True,
        )


def test_de_sitter_propagation_does_not_claim_finite_register_energy_conservation() -> None:
    transition = _vacuum_transition()
    result = propagate_constant_vacuum_flat_flrw(
        transition,
        newton_constant=1.0,
        evaluation_scale_factor=3.0,
        global_constant_vacuum_action_adopted=True,
    )

    assert result.vacuum_energy_change_from_initial_slice > 0.0
    assert result.vacuum_energy_time_derivative > 0.0
    assert not result.one_slice_energy_match_derives_global_vacuum_action
    assert not result.finite_register_supplies_all_later_comoving_vacuum_energy
    assert not result.vacuum_readout_selected_by_ce_dynamics
    assert not result.observed_dark_energy_density_predicted
    assert not result.vacuum_renormalization_and_radiative_stability_derived
    assert not result.perturbations_and_ce_specific_observational_prediction_derived
    assert result.claim_ceiling.endswith("NOT_GLOBAL_ACTION_OR_DENSITY_DERIVATION")


@pytest.mark.parametrize(
    "keyword,value,message",
    (
        ("newton_constant", 0.0, "Newton"),
        ("evaluation_scale_factor", -1.0, "scale factor"),
        ("evaluation_scale_factor", math.nan, "scale factor"),
        ("global_constant_vacuum_action_adopted", 1, "must be boolean"),
        ("tolerance", 0.0, "tolerance"),
    ),
)
def test_invalid_inputs_are_rejected(
    keyword: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "newton_constant": 1.0,
        "evaluation_scale_factor": 2.0,
        "global_constant_vacuum_action_adopted": True,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        propagate_constant_vacuum_flat_flrw(
            _vacuum_transition(), **arguments  # type: ignore[arg-type]
        )
