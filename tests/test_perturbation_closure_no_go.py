import pytest

from examples.physics.perturbation_closure_no_go import (
    perturbation_closure_no_go,
    subhorizon_acceleration_difference,
)


def test_same_covariant_background_has_different_scalar_sound_speed() -> None:
    audit = perturbation_closure_no_go(
        background_x_over_reference_density=0.2,
        potential_over_reference_density=0.8,
        lambda_times_background_x=0.75,
    )

    assert audit.background_energy_density_over_reference_density == pytest.approx(1.0)
    assert audit.background_pressure_over_reference_density == pytest.approx(-0.6)
    assert audit.same_background_lagrangian
    assert audit.same_background_first_variation
    assert audit.same_background_ward_identity
    assert audit.canonical_sound_speed_squared == 1.0
    assert audit.deformed_sound_speed_squared == pytest.approx(0.25)
    assert not audit.unique_linear_perturbations_follow
    assert audit.status == "BACKGROUND_TO_UNIQUE_PERTURBATIONS_IMPLICATION_DISPROVED"


def test_equal_initial_perturbations_immediately_accelerate_differently() -> None:
    audit = perturbation_closure_no_go(
        background_x_over_reference_density=0.5,
        potential_over_reference_density=0.5,
        lambda_times_background_x=0.25,
    )

    difference = subhorizon_acceleration_difference(
        audit,
        wavenumber_over_a_h=10.0,
        density_contrast=0.01,
    )
    assert difference == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "background_x_over_reference_density": 0.0,
            "potential_over_reference_density": 1.0,
            "lambda_times_background_x": 1.0,
        },
        {
            "background_x_over_reference_density": 1.0,
            "potential_over_reference_density": 0.0,
            "lambda_times_background_x": 0.0,
        },
        {
            "background_x_over_reference_density": 1.0,
            "potential_over_reference_density": float("inf"),
            "lambda_times_background_x": 1.0,
        },
    ],
)
def test_no_go_fails_closed_outside_stable_finite_domain(kwargs) -> None:
    with pytest.raises(ValueError):
        perturbation_closure_no_go(**kwargs)
