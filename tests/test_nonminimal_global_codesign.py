from __future__ import annotations

from reality_stone.clarus.nonminimal_global_codesign import (
    global_nonminimal_codesign_audit,
)


def test_simple_local_codesign_fails_at_intermediate_radius() -> None:
    audit = global_nonminimal_codesign_audit(
        adm_shape_limit=2.0 / 3.0,
        shape_second_derivative=-5.0,
        redshift_second_derivative=0.0,
        shape_cubic=0.0,
        shape_quartic=0.0,
        redshift_cubic=0.0,
        redshift_quartic=0.0,
    )

    assert audit.local_kinetic_over_planck_factor > 0.0
    assert audit.minimum_kinetic_over_planck_factor < -2.0
    assert 1.3 < audit.minimum_kinetic_radius < 1.5
    assert audit.positive_adm_mass
    assert audit.asymptotically_flat
    assert not audit.global_healthy_kinetic
    assert not audit.global_codesign_pass


def test_high_order_search_control_still_fails_global_kinetic_gate() -> None:
    audit = global_nonminimal_codesign_audit(
        adm_shape_limit=1.1091603612294227,
        shape_second_derivative=-6.666517860495915,
        redshift_second_derivative=0.9991761531580048,
        shape_cubic=4.197336481457169,
        shape_quartic=-2.0038731811142796,
        redshift_cubic=-0.6064337767724226,
        redshift_quartic=0.034506304210021106,
    )

    assert audit.local_kinetic_over_planck_factor > 0.0
    assert audit.minimum_kinetic_over_planck_factor < -1.0
    assert audit.minimum_shape_gap > 0.0
    assert audit.regular_planck_factor_control
    assert not audit.global_codesign_pass
