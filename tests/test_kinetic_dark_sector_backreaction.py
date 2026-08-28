import pytest

from examples.physics.kinetic_dark_sector_backreaction import (
    ConservedFluid,
    ScalarBackreactionChannel,
    backreaction_closure_receipt,
)


def test_dynamic_clock_backreaction_closes_all_ward_and_constraint_gates() -> None:
    e = 1.2
    planck = 10.0
    theta_n = 0.1
    potential_theta = 0.07
    scalar_seed = ScalarBackreactionChannel(
        degeneracy=3,
        energy_density=0.8,
        pressure=0.2,
        energy_density_d_n=0.0,
        field_squared=0.04,
        mass_squared_d_theta=0.5,
    )
    scalar_force = scalar_seed.clock_force
    scalar = ScalarBackreactionChannel(
        degeneracy=scalar_seed.degeneracy,
        energy_density=scalar_seed.energy_density,
        pressure=scalar_seed.pressure,
        energy_density_d_n=(
            -3.0 * (scalar_seed.energy_density + scalar_seed.pressure)
            + theta_n * scalar_force
        ),
        field_squared=scalar_seed.field_squared,
        mass_squared_d_theta=scalar_seed.mass_squared_d_theta,
    )
    fluid = ConservedFluid(
        energy_density=2.0,
        pressure=2.0 / 3.0,
        energy_density_d_n=-8.0,
    )

    # Choose V from Friedmann, then E_N/E from Raychaudhuri, and theta_NN
    # from the clock equation.  The receipt must recover every identity.
    clock_kinetic = 0.5 * e * e * theta_n * theta_n
    potential = (
        3.0 * planck * planck * e * e
        - scalar.energy_density
        - fluid.energy_density
        - clock_kinetic
    )
    clock_rho = clock_kinetic + potential
    clock_p = clock_kinetic - potential
    total_rho_plus_p = (
        scalar.energy_density
        + scalar.pressure
        + fluid.energy_density
        + fluid.pressure
        + clock_rho
        + clock_p
    )
    d_log_e = -total_rho_plus_p / (2.0 * planck**2 * e**2)
    theta_nn = (
        -(potential_theta + scalar_force) / e**2
        - (3.0 + d_log_e) * theta_n
    )

    receipt = backreaction_closure_receipt(
        e=e,
        d_log_e_d_n=d_log_e,
        reduced_planck_over_h0=planck,
        theta_d_n=theta_n,
        theta_d2_n=theta_nn,
        potential=potential,
        potential_d_theta=potential_theta,
        scalar_channels=(scalar,),
        conserved_fluids=(fluid,),
    )

    assert receipt.scalar_ward_residuals == pytest.approx((0.0,), abs=1.0e-14)
    assert receipt.clock_equation_residual == pytest.approx(0.0, abs=1.0e-14)
    assert receipt.clock_ward_residual == pytest.approx(0.0, abs=1.0e-12)
    assert receipt.total_ward_residual == pytest.approx(0.0, abs=1.0e-12)
    assert receipt.friedmann_constraint_residual == pytest.approx(0.0, abs=1.0e-12)
    assert receipt.raychaudhuri_residual == pytest.approx(0.0, abs=1.0e-14)
    assert receipt.friedmann_constraint_derivative == pytest.approx(0.0, abs=1.0e-12)
    assert receipt.constraint_propagation_identity_residual == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_clock_ward_is_exactly_theta_n_times_clock_equation() -> None:
    receipt = backreaction_closure_receipt(
        e=1.1,
        d_log_e_d_n=-0.2,
        reduced_planck_over_h0=5.0,
        theta_d_n=0.3,
        theta_d2_n=0.4,
        potential=2.0,
        potential_d_theta=-0.1,
        scalar_channels=(
            ScalarBackreactionChannel(2, 0.5, 0.1, -1.7, 0.2, 0.4),
        ),
    )

    assert receipt.clock_ward_factorization_residual == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert receipt.clock_ward_residual == pytest.approx(
        0.3 * receipt.clock_equation_residual
    )


def test_constraint_derivative_identity_holds_off_shell() -> None:
    receipt = backreaction_closure_receipt(
        e=0.9,
        d_log_e_d_n=0.17,
        reduced_planck_over_h0=4.0,
        theta_d_n=-0.2,
        theta_d2_n=0.6,
        potential=1.3,
        potential_d_theta=0.8,
        scalar_channels=(
            ScalarBackreactionChannel(1, 0.7, -0.1, 0.9, 0.3, -0.5),
        ),
        conserved_fluids=(ConservedFluid(0.4, 0.2, -0.1),),
    )

    assert receipt.constraint_propagation_identity_residual == pytest.approx(
        0.0, abs=2.0e-14
    )


def test_field_squared_is_per_component_and_degeneracy_multiplies_force() -> None:
    single = ScalarBackreactionChannel(1, 0.0, 0.0, 0.0, 0.25, 0.8)
    quadruple = ScalarBackreactionChannel(4, 0.0, 0.0, 0.0, 0.25, 0.8)

    assert single.clock_force == pytest.approx(0.1)
    assert quadruple.clock_force == pytest.approx(0.4)


@pytest.mark.parametrize("degeneracy", [0, -1, True])
def test_invalid_degeneracy_fails_closed(degeneracy) -> None:
    with pytest.raises(ValueError, match="degeneracy"):
        ScalarBackreactionChannel(degeneracy, 0.0, 0.0, 0.0, 0.0, 0.0)
