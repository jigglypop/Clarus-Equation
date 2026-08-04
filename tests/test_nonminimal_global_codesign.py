from __future__ import annotations

import math

import pytest

from reality_stone.clarus.nonminimal_global_codesign import (
    GlobalCodesignAudit,
    global_nonminimal_codesign_audit,
)


BASE_PARAMETERS = {
    "adm_shape_limit": 2.0 / 3.0,
    "shape_second_derivative": -5.0,
    "redshift_second_derivative": 0.0,
    "shape_cubic": 0.0,
    "shape_quartic": 0.0,
    "redshift_cubic": 0.0,
    "redshift_quartic": 0.0,
    "radial_cutoff": 40.0,
}


def _audit(**overrides: float | int) -> GlobalCodesignAudit:
    parameters = {**BASE_PARAMETERS, **overrides}
    return global_nonminimal_codesign_audit(**parameters)


def test_simple_local_codesign_fails_at_intermediate_radius() -> None:
    audit = _audit()

    assert audit.local_kinetic_over_planck_factor > 0.0
    assert audit.sampled_minimum_kinetic_over_planck_factor < -2.0
    assert 1.3 < audit.sampled_minimum_kinetic_radius < 1.5
    assert audit.positive_adm_mass
    assert audit.sampled_cutoff_flatness_pass
    assert not audit.sampled_healthy_kinetic
    assert not audit.sampled_codesign_pass
    assert audit.continuous_domain_certification == "not established by finite-grid sampling"

    # Legacy names remain read-only aliases, not dataclass certification fields.
    assert audit.minimum_kinetic_over_planck_factor == (
        audit.sampled_minimum_kinetic_over_planck_factor
    )
    assert audit.global_codesign_pass == audit.sampled_codesign_pass
    assert "global_codesign_pass" not in GlobalCodesignAudit.__dataclass_fields__


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
    assert audit.sampled_minimum_kinetic_over_planck_factor < -1.0
    assert audit.sampled_minimum_shape_gap > 0.0
    assert audit.sampled_regular_planck_factor_control
    assert not audit.sampled_codesign_pass


@pytest.mark.parametrize(
    "radial_cutoff",
    [float("nan"), float("inf"), float("-inf"), 2.0, True, "40.0"],
)
def test_radial_cutoff_must_be_a_finite_numeric_value_greater_than_two(
    radial_cutoff: object,
) -> None:
    with pytest.raises(ValueError, match="radial_cutoff"):
        _audit(radial_cutoff=radial_cutoff)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_count", [True, 256.0, 512.0, "256", 255])
def test_sample_count_must_be_a_strict_integer_of_at_least_256(
    sample_count: object,
) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        _audit(sample_count=sample_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("adm_shape_limit", float("nan")),
        ("shape_cubic", float("inf")),
        ("redshift_quartic", float("-inf")),
    ],
)
def test_nonfinite_input_coefficients_are_rejected(parameter: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _audit(**{parameter: value})


@pytest.mark.parametrize("parameter", ["shape_quartic", "redshift_quartic"])
def test_finite_coefficients_that_overflow_a_computed_profile_are_rejected(
    parameter: str,
) -> None:
    with pytest.raises(ValueError, match="computed .* non-finite"):
        _audit(**{parameter: 1.0e308})


def test_n_2n_4n_resolution_deltas_are_reported_without_a_continuum_claim() -> None:
    audit = _audit(sample_count=2400)
    convergence = audit.resolution_convergence

    assert (
        convergence.sample_count_n,
        convergence.sample_count_2n,
        convergence.sample_count_4n,
    ) == (
        2400,
        4800,
        9600,
    )
    deltas = (
        convergence.minimum_kinetic_delta_n_to_2n,
        convergence.minimum_kinetic_delta_2n_to_4n,
        convergence.minimum_kinetic_radius_delta_n_to_2n,
        convergence.minimum_kinetic_radius_delta_2n_to_4n,
        convergence.minimum_shape_gap_delta_n_to_2n,
        convergence.minimum_shape_gap_delta_2n_to_4n,
        convergence.minimum_log_planck_delta_n_to_2n,
        convergence.minimum_log_planck_delta_2n_to_4n,
        convergence.maximum_log_planck_delta_n_to_2n,
        convergence.maximum_log_planck_delta_2n_to_4n,
    )
    assert all(math.isfinite(delta) for delta in deltas)
    assert abs(convergence.minimum_kinetic_delta_2n_to_4n) < 2.0e-6
    assert convergence.sampled_classification_consistent
    assert not convergence.sampled_codesign_pass_n
    assert not convergence.sampled_codesign_pass_2n
    assert not convergence.sampled_codesign_pass_4n
