import math

import pytest

from reality_stone.clarus.warp_drive_reality import (
    audit_alcubierre_tanh_wall,
    warp_pathway_portfolio,
)


def test_tanh_wall_converges_to_thin_wall_energy() -> None:
    audit = audit_alcubierre_tanh_wall(
        bubble_radius_m=10.0,
        wall_thickness_m=0.1,
    )

    assert math.isclose(audit.exact_to_thin_wall_ratio, 1.0, rel_tol=1.0e-4)
    assert audit.total_eulerian_energy_j < 0.0
    assert audit.negative_mass_earth > 6.0e3


def test_warp_energy_scales_exactly_as_speed_squared() -> None:
    slow = audit_alcubierre_tanh_wall(speed_over_c=0.5)
    fast = audit_alcubierre_tanh_wall(speed_over_c=2.0)

    assert math.isclose(
        fast.total_eulerian_energy_j / slow.total_eulerian_energy_j,
        16.0,
        rel_tol=1.0e-14,
    )
    assert not slow.superluminal_shortcut
    assert fast.superluminal_shortcut
    assert fast.front_back_horizons_expected


def test_nontrivial_alcubierre_wall_has_negative_eulerian_energy() -> None:
    audit = audit_alcubierre_tanh_wall(speed_over_c=0.1)

    assert audit.null_energy_condition_violated
    assert not audit.material_source_action_specified
    assert not audit.complete_linear_stability
    assert not audit.realization_pass


def test_positive_energy_subluminal_path_is_not_misreported_as_ftl() -> None:
    portfolio = warp_pathway_portfolio()
    subluminal = portfolio[1]
    positive_ftl = portfolio[2]

    assert subluminal.positive_energy_claim
    assert subluminal.all_observer_nec_gate
    assert not subluminal.superluminal_shortcut
    assert not subluminal.self_propulsion_free

    assert positive_ftl.positive_energy_claim
    assert positive_ftl.superluminal_shortcut
    assert not positive_ftl.all_observer_nec_gate


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bubble_radius_m": 0.0},
        {"wall_thickness_m": 0.0},
        {"speed_over_c": -1.0},
        {"integration_steps": 999},
        {"integration_steps": 1001},
    ],
)
def test_invalid_warp_inputs_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        audit_alcubierre_tanh_wall(**kwargs)
