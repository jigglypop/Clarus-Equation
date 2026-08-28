import math

import pytest

from examples.physics.kinetic_dark_sector_adiabatic_stress import (
    CertifiedPowerLawTail,
    MassSquaredJet,
    ModeStress,
    ScaleFactorJet,
    bare_mode_stress,
    fourth_order_adiabatic_initial_state,
    fourth_order_counterterm,
    integrate_isotropic_stress_with_certified_tail,
    renormalized_mode_stress,
    sixth_order_remainder,
    time_dependent_mass_counterterm,
)


def _minkowski_jet() -> ScaleFactorJet:
    return ScaleFactorJet(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _de_sitter_jet() -> ScaleFactorJet:
    # a(x)=-1/x at x=-1: a^(n)=n!.
    return ScaleFactorJet(1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0)


def test_minkowski_counterterm_is_the_zero_point_stress() -> None:
    q = 3.0
    mu = 4.0
    omega = 5.0
    counterterm = fourth_order_counterterm(
        _minkowski_jet(), q=q, mu=mu, xi=0.23
    )

    assert counterterm.w_orders == pytest.approx((omega, 0.0, 0.0), abs=1.0e-14)
    assert counterterm.energy_density_orders == pytest.approx(
        (omega / 2.0, 0.0, 0.0), abs=1.0e-14
    )
    assert counterterm.pressure_orders == pytest.approx(
        (q * q / (6.0 * omega), 0.0, 0.0), abs=1.0e-14
    )


@pytest.mark.parametrize("xi", [0.0, 1.0 / 6.0, 0.31])
def test_explicit_fourth_order_recurrence_matches_two_riccati_iterations(
    xi: float,
) -> None:
    receipt = fourth_order_counterterm(
        _de_sitter_jet(), q=7.0, mu=2.5, xi=xi
    )
    assert receipt.w_orders[2] != 0.0
    assert receipt.max_iterated_recurrence_disagreement < 2.0e-13
    assert receipt.max_riccati_residual_through_order_four < 2.0e-12


@pytest.mark.parametrize(
    ("jet", "q", "mu", "xi"),
    [
        (_de_sitter_jet(), 5.0, 1.75, 0.0),
        (_de_sitter_jet(), 11.0, 3.0, 1.0 / 6.0),
        (ScaleFactorJet(1.3, 0.7, -0.2, 0.4, -0.1, 0.3, -0.5), 4.0, 2.0, 0.27),
    ],
)
def test_projected_counterterm_obeys_modewise_ward_identity(
    jet: ScaleFactorJet,
    q: float,
    mu: float,
    xi: float,
) -> None:
    receipt = fourth_order_counterterm(jet, q=q, mu=mu, xi=xi)
    assert receipt.max_ward_residual_through_order_five < 2.0e-10


def test_minkowski_adiabatic_vacuum_subtracts_to_zero() -> None:
    q = 8.0
    mu = 1.5
    omega = math.sqrt(q * q + mu * mu)
    u = complex(1.0 / math.sqrt(2.0 * omega))
    du_dx = -1.0j * omega * u

    bare = bare_mode_stress(
        _minkowski_jet(), q=q, mu=mu, xi=0.42, u=u, du_dx=du_dx
    )
    renormalized = renormalized_mode_stress(
        _minkowski_jet(), q=q, mu=mu, xi=0.42, u=u, du_dx=du_dx
    )

    assert bare.energy_density_over_h0_four == pytest.approx(omega / 2.0)
    assert renormalized.energy_density_over_h0_four == pytest.approx(0.0, abs=2.0e-14)
    assert renormalized.pressure_over_h0_four == pytest.approx(0.0, abs=2.0e-14)


def test_fourth_order_initial_state_is_canonically_normalized() -> None:
    state = fourth_order_adiabatic_initial_state(
        _de_sitter_jet(), q=9.0, mu=2.0, xi=0.21
    )
    assert state.frequency > 0.0
    assert state.frequency_derivative != 0.0
    assert state.wronskian_residual < 2.0e-16


def test_sixth_order_remainder_has_integrable_large_q_power() -> None:
    jet = _de_sitter_jet()
    low = sixth_order_remainder(jet, q=200.0, mu=2.0, xi=0.13)
    high = sixth_order_remainder(jet, q=400.0, mu=2.0, xi=0.13)
    energy_ratio = abs(high.energy_density_order_six / low.energy_density_order_six)
    pressure_ratio = abs(high.pressure_order_six / low.pressure_order_six)

    assert low.per_mode_large_q_power == -5
    assert low.radial_integrand_large_q_power == -3
    assert low.ultraviolet_integrable
    assert energy_ratio == pytest.approx(2.0**-5, rel=3.0e-3)
    assert pressure_ratio == pytest.approx(2.0**-5, rel=3.0e-3)


def test_certified_power_law_tail_has_exact_integral_bound() -> None:
    q_values = (1.0, 2.0, 4.0)
    stresses = tuple(
        ModeStress(q**-5, -2.0 * q**-5)
        for q in q_values
    )
    result = integrate_isotropic_stress_with_certified_tail(
        q_values,
        stresses,
        energy_tail=CertifiedPowerLawTail(1.0, 5.0, 2.0),
        pressure_tail=CertifiedPowerLawTail(2.0, 5.0, 2.0),
    )

    assert result.energy_tail_absolute_bound == pytest.approx(
        1.0 / (4.0 * math.pi**2 * 4.0**2)
    )
    assert result.pressure_tail_absolute_bound == pytest.approx(
        2.0 / (4.0 * math.pi**2 * 4.0**2)
    )


def test_certified_tail_fails_if_last_sample_violates_bound() -> None:
    with pytest.raises(ValueError, match="last energy sample"):
        integrate_isotropic_stress_with_certified_tail(
            (1.0, 2.0),
            (ModeStress(0.0, 0.0), ModeStress(1.0, 0.0)),
            energy_tail=CertifiedPowerLawTail(1.0, 5.0, 1.0),
            pressure_tail=CertifiedPowerLawTail(1.0, 5.0, 1.0),
        )


def test_time_dependent_mass_counterterm_closes_transfer_ward_identity() -> None:
    rate = 0.2
    mass_squared = 4.0
    mass_jet = MassSquaredJet(
        mass_squared,
        mass_squared * rate,
        mass_squared * rate**2,
        mass_squared * rate**3,
        mass_squared * rate**4,
        mass_squared * rate**5,
        mass_squared * rate**6,
    )
    receipt = time_dependent_mass_counterterm(
        _de_sitter_jet(), mass_jet, q=8.0, xi=0.19
    )

    assert receipt.transfer_orders[0] != 0.0
    assert receipt.max_transfer_ward_residual_through_order_five < 2.0e-11


def test_constant_mass_triplet_reduces_to_constant_mass_counterterm() -> None:
    jet = _de_sitter_jet()
    dynamic = time_dependent_mass_counterterm(
        jet,
        MassSquaredJet(4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        q=6.0,
        xi=0.11,
    )
    constant = fourth_order_counterterm(jet, q=6.0, mu=2.0, xi=0.11)

    assert dynamic.energy_density_orders == pytest.approx(
        constant.energy_density_orders, abs=2.0e-14
    )
    assert dynamic.pressure_orders == pytest.approx(
        constant.pressure_orders, abs=2.0e-14
    )
    assert dynamic.transfer_orders == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-15)


def test_omitting_mass_derivative_counterterms_leaves_a_ward_defect() -> None:
    mass_jet = MassSquaredJet(4.0, 0.8, 0.16, 0.032, 0.0064, 0.00128, 0.000256)
    receipt = time_dependent_mass_counterterm(
        _de_sitter_jet(), mass_jet, q=8.0, xi=0.19
    )
    # A constant-mass subtraction has zero scalar transfer.  Reusing it while
    # the physical mass changes misses at least the leading source below.
    assert abs(receipt.transfer_orders[0]) > 1.0e-6


def test_minimal_coupling_energy_matches_completed_square() -> None:
    jet = _de_sitter_jet()
    q = 2.0
    mu = 3.0
    u = 0.4 + 0.2j
    du_dx = -0.3 + 1.1j

    stress = bare_mode_stress(jet, q=q, mu=mu, xi=0.0, u=u, du_dx=du_dx)
    expected = (
        abs(du_dx - (jet.d1 / jet.a) * u) ** 2
        + (q * q + jet.a * jet.a * mu * mu) * abs(u) ** 2
    ) / (2.0 * jet.a**4)
    assert stress.energy_density_over_h0_four == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q": 0.0, "mu": 1.0, "xi": 0.0},
        {"q": 1.0, "mu": 0.0, "xi": 0.0},
        {"q": 1.0, "mu": 1.0, "xi": math.inf},
    ],
)
def test_counterterm_fails_closed_outside_constant_positive_mass_domain(kwargs) -> None:
    with pytest.raises(ValueError):
        fourth_order_counterterm(_minkowski_jet(), **kwargs)
