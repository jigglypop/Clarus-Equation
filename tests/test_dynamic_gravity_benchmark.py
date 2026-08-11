from reality_stone.clarus.dynamic_gravity_benchmark import (
    evaluate_density_contrast,
    evaluate_dynamic_gravity,
    small_dynamic_config,
)


def test_dynamic_field_preflight_and_trace_integrity() -> None:
    result = evaluate_dynamic_gravity(small_dynamic_config())
    assert result["schema"] == "clarus.dynamic-gravity.validation.v1"
    assert result["preflight"]["cfl"] <= 1.0
    assert result["preflight"]["zero_field_error"] == 0.0
    assert result["preflight"]["equal_mass_max_center_force"] <= 1e-10
    assert result["id"]["memory_trace_identical"]
    assert result["ood"]["memory_trace_identical"]


def test_density_contrast_has_zero_equal_prior_field() -> None:
    result = evaluate_density_contrast(small_dynamic_config())
    assert result["schema"] == "clarus.density-contrast-gravity.validation.v1"
    assert result["equal_prior"]["source_max_abs"] <= 1e-12
    assert result["equal_prior"]["field_max_abs"] <= 1e-12
    assert result["id"]["memory_trace_identical"]
    assert result["ood"]["memory_trace_identical"]
