from reality_stone.clarus.gravitational_decision_benchmark import (
    evaluate_gravitational_decision,
    small_gravity_config,
)


def test_small_gravity_field_has_symmetry_and_integrity() -> None:
    result = evaluate_gravitational_decision(small_gravity_config())
    assert result["schema"] == "clarus.gravitational-decision.validation.v1"
    assert result["field"]["plus_residual"] <= 1e-10
    assert result["field"]["minus_residual"] <= 1e-10
    assert abs(result["field"]["equal_mass_central_force"]) <= 1e-12
    assert result["field"]["positive_mass_difference_force"] > 0.0
    assert result["field"]["negative_mass_difference_force"] < 0.0
    assert result["id"]["memory_trace_identical"]
    assert result["ood"]["memory_trace_identical"]
    assert result["future_reads"] == 0
    assert result["environment_clone_calls"] == 0
