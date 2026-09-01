from dataclasses import dataclass, replace
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
    integrate_squeezed_created_stress_with_certified_tail,
    local_isotropic_stress_observer_readout,
    minimal_squeezed_flrw_mode_stress_difference,
    renormalized_mode_stress,
    sixth_order_remainder,
    time_dependent_mass_counterterm,
    trace_squeezed_flrw_mode_stress,
)
from examples.physics.kinetic_dark_sector_flrw_mode import (
    FLRWModeSpec,
    solve_flrw_mode,
)


def _minkowski_jet() -> ScaleFactorJet:
    return ScaleFactorJet(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _de_sitter_jet() -> ScaleFactorJet:
    # a(x)=-1/x at x=-1: a^(n)=n!.
    return ScaleFactorJet(1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0)


@dataclass(frozen=True)
class _TrajectoryBackgroundNode:
    n: float
    e2: float = 1.0


@dataclass(frozen=True)
class _DeSitterTrajectoryBackground:
    nodes: tuple[_TrajectoryBackgroundNode, ...] = (
        _TrajectoryBackgroundNode(-2.1),
        _TrajectoryBackgroundNode(0.1),
    )

    def at_n(self, n: float) -> _TrajectoryBackgroundNode:
        if n < self.nodes[0].n or n > self.nodes[-1].n:
            raise ValueError("outside de Sitter trajectory window")
        return _TrajectoryBackgroundNode(n)


def _de_sitter_trajectory_jet(n: float) -> ScaleFactorJet:
    a = math.exp(n)
    return ScaleFactorJet(
        a,
        a**2,
        2.0 * a**3,
        6.0 * a**4,
        24.0 * a**5,
        120.0 * a**6,
        720.0 * a**7,
    )


def _massive_de_sitter_solution(*, steps: int, mu: float = 40.0):
    background = _DeSitterTrajectoryBackground()
    solution = solve_flrw_mode(
        background,
        FLRWModeSpec(
            comoving_wavenumber_over_h0=0.2,
            mass_over_h0=lambda _n: mu,
            curvature_coupling=0.0,
            initial_n=-2.0,
            final_n=0.0,
            steps=steps,
        ),
    )
    return background, solution


def _massive_de_sitter_stress_trajectory(*, steps: int, beta: complex | None = None):
    background, solution = _massive_de_sitter_solution(steps=steps)
    beta_value = complex(math.sqrt(0.5)) if beta is None else beta
    return trace_squeezed_flrw_mode_stress(
        background,
        solution,
        scale_factor_jet_at_n=_de_sitter_trajectory_jet,
        alpha=math.sqrt(1.0 + abs(beta_value) ** 2),
        beta=beta_value,
        late_window_efolds=1.0,
        canonical_tolerance=1.0e-4,
    )


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


def test_minkowski_squeezed_phase_cancels_from_energy_but_not_pressure() -> None:
    q = 3.0
    mu = 2.0
    omega = math.sqrt(q * q + mu * mu)
    reference_u = complex(1.0 / math.sqrt(2.0 * omega))
    beta_squared = 0.05
    receipt = minimal_squeezed_flrw_mode_stress_difference(
        _minkowski_jet(),
        q=q,
        mu=mu,
        reference_u=reference_u,
        reference_du_dx=-1.0j * omega * reference_u,
        reference_d2u_dx2=-(omega**2) * reference_u,
        alpha=math.sqrt(1.0 + beta_squared),
        beta=math.sqrt(beta_squared),
        initial_occupation=0.1,
    )

    stimulation = 1.2
    assert receipt.created_particle_stress.energy_density_over_h0_four == (
        pytest.approx(stimulation * omega * beta_squared)
    )
    assert receipt.created_particle_stress.pressure_over_h0_four == pytest.approx(
        stimulation * q * q * beta_squared / (3.0 * omega)
    )
    assert receipt.created_anomalous_stress.energy_density_over_h0_four == (
        pytest.approx(0.0, abs=2.0e-15)
    )
    assert abs(receipt.created_anomalous_stress.pressure_over_h0_four) > 0.7
    assert receipt.created_dimensionless_conformal_continuity_residual == (
        pytest.approx(0.0, abs=2.0e-14)
    )
    assert receipt.static_minkowski_anomalous_energy_cancellation_pass
    assert receipt.dimensions_pass
    assert not receipt.universal_planck_tick_assumed
    assert not receipt.physical_dark_matter_dark_energy_identification


def test_supplied_bogoliubov_phase_changes_anomalous_pressure() -> None:
    q = 3.0
    mu = 2.0
    omega = math.sqrt(q * q + mu * mu)
    reference_u = complex(1.0 / math.sqrt(2.0 * omega))
    beta_squared = 0.05

    def receipt_for(beta: complex):
        return minimal_squeezed_flrw_mode_stress_difference(
            _minkowski_jet(),
            q=q,
            mu=mu,
            reference_u=reference_u,
            reference_du_dx=-1.0j * omega * reference_u,
            reference_d2u_dx2=-(omega**2) * reference_u,
            alpha=math.sqrt(1.0 + beta_squared),
            beta=beta,
        )

    real_phase = receipt_for(complex(math.sqrt(beta_squared)))
    quadrature_phase = receipt_for(1.0j * math.sqrt(beta_squared))
    assert real_phase.beta_squared == pytest.approx(quadrature_phase.beta_squared)
    assert real_phase.created_particle_stress == quadrature_phase.created_particle_stress
    assert abs(real_phase.created_anomalous_stress.pressure_over_h0_four) > 0.6
    assert quadrature_phase.created_anomalous_stress.pressure_over_h0_four == (
        pytest.approx(0.0, abs=2.0e-15)
    )
    assert not real_phase.phase_derived_from_quench_profile


def test_expanding_flrw_squeezed_phase_enters_energy_and_closes_ward_identity() -> None:
    jet = _de_sitter_jet()
    q = 9.0
    mu = 2.0
    state = fourth_order_adiabatic_initial_state(jet, q=q, mu=mu, xi=0.0)
    effective_frequency_squared = q * q + jet.a * jet.a * mu * mu - jet.d2 / jet.a
    receipt = minimal_squeezed_flrw_mode_stress_difference(
        jet,
        q=q,
        mu=mu,
        reference_u=state.u,
        reference_du_dx=state.du_dx,
        reference_d2u_dx2=-effective_frequency_squared * state.u,
        alpha=math.sqrt(1.03),
        beta=0.1 + math.sqrt(0.02) * 1.0j,
        initial_occupation=0.2,
    )

    assert not receipt.static_minkowski_background
    assert abs(receipt.created_anomalous_stress.energy_density_over_h0_four) > 0.2
    assert receipt.reference_eom_relative_residual < 1.0e-14
    assert receipt.squeezed_eom_relative_residual < 1.0e-14
    assert abs(receipt.created_dimensionless_conformal_continuity_residual) < 3.0e-13
    assert receipt.comoving_proper_time_rate_per_dimensionless_conformal_time == (
        pytest.approx(jet.a)
    )
    assert receipt.same_local_counterterm_cancels_in_state_difference
    assert not receipt.exact_mode_propagation_verified_by_this_function
    assert not receipt.full_renormalized_flrw_stress


def test_common_adiabatic_counterterm_cancels_in_state_difference() -> None:
    jet = _de_sitter_jet()
    q = 9.0
    mu = 2.0
    state = fourth_order_adiabatic_initial_state(jet, q=q, mu=mu, xi=0.0)
    effective_frequency_squared = q * q + jet.a * jet.a * mu * mu - jet.d2 / jet.a
    receipt = minimal_squeezed_flrw_mode_stress_difference(
        jet,
        q=q,
        mu=mu,
        reference_u=state.u,
        reference_du_dx=state.du_dx,
        reference_d2u_dx2=-effective_frequency_squared * state.u,
        alpha=math.sqrt(1.04),
        beta=0.2j,
    )
    squeezed_renormalized = renormalized_mode_stress(
        jet,
        q=q,
        mu=mu,
        xi=0.0,
        u=receipt.alpha * state.u + receipt.beta * state.u.conjugate(),
        du_dx=(
            receipt.alpha * state.du_dx
            + receipt.beta * state.du_dx.conjugate()
        ),
    )
    reference_renormalized = renormalized_mode_stress(
        jet,
        q=q,
        mu=mu,
        xi=0.0,
        u=state.u,
        du_dx=state.du_dx,
    )

    assert (
        squeezed_renormalized.energy_density_over_h0_four
        - reference_renormalized.energy_density_over_h0_four
    ) == pytest.approx(
        receipt.created_state_dependent_stress.energy_density_over_h0_four,
        abs=3.0e-14,
    )
    assert (
        squeezed_renormalized.pressure_over_h0_four
        - reference_renormalized.pressure_over_h0_four
    ) == pytest.approx(
        receipt.created_state_dependent_stress.pressure_over_h0_four,
        abs=3.0e-14,
    )


def test_zero_squeeze_has_no_created_state_difference() -> None:
    q = 2.0
    mu = 1.0
    omega = math.sqrt(q * q + mu * mu)
    reference_u = complex(1.0 / math.sqrt(2.0 * omega))
    receipt = minimal_squeezed_flrw_mode_stress_difference(
        _minkowski_jet(),
        q=q,
        mu=mu,
        reference_u=reference_u,
        reference_du_dx=-1.0j * omega * reference_u,
        reference_d2u_dx2=-(omega**2) * reference_u,
        alpha=1.0,
        beta=0.0,
        initial_occupation=0.3,
    )

    assert receipt.created_state_dependent_stress == ModeStress(0.0, 0.0)
    assert receipt.created_state_dependent_field_squared_over_h0_two == 0.0
    assert receipt.full_reference_mode_subtracted_stress == ModeStress(
        0.6 * receipt.reference_stress.energy_density_over_h0_four,
        0.6 * receipt.reference_stress.pressure_over_h0_four,
    )
    assert "CALLER_DECLARED" in receipt.initial_occupation_basis_declaration
    assert not receipt.density_matrix_basis_verified_by_this_function
    assert not receipt.v_basis_number_only_input_supported


def test_squeezed_stress_integration_requires_external_uv_tail_certificate() -> None:
    receipts = []
    for q in (1.0, 2.0, 4.0):
        mu = 1.0
        omega = math.sqrt(q * q + mu * mu)
        reference_u = complex(1.0 / math.sqrt(2.0 * omega))
        beta_squared = 1.0e-4 * q**-8
        receipts.append(
            minimal_squeezed_flrw_mode_stress_difference(
                _minkowski_jet(),
                q=q,
                mu=mu,
                reference_u=reference_u,
                reference_du_dx=-1.0j * omega * reference_u,
                reference_d2u_dx2=-(omega**2) * reference_u,
                alpha=math.sqrt(1.0 + beta_squared),
                beta=1.0j * math.sqrt(beta_squared),
            )
        )
    last = receipts[-1].created_state_dependent_stress
    energy_coefficient = 1.01 * abs(last.energy_density_over_h0_four) * 4.0**7
    pressure_coefficient = 1.01 * abs(last.pressure_over_h0_four) * 4.0**7
    integrated = integrate_squeezed_created_stress_with_certified_tail(
        tuple(receipts),
        energy_tail=CertifiedPowerLawTail(energy_coefficient, 7.0, 4.0),
        pressure_tail=CertifiedPowerLawTail(pressure_coefficient, 7.0, 4.0),
    )

    assert integrated.energy_tail_absolute_bound > 0.0
    assert integrated.pressure_tail_absolute_bound > 0.0
    assert "EXTERNALLY_CERTIFIED_UV_TAIL" in integrated.status
    assert integrated.external_tail_certificate_trusted
    assert not integrated.tail_certificate_independently_derived_by_integrator
    assert not integrated.hadamard_state_proved
    assert not integrated.absolute_reference_vacuum_stress_renormalized
    assert not receipts[-1].integrated_uv_tail_certified


def test_squeezed_stress_integration_rejects_mixed_background_jets() -> None:
    def make_receipt(jet: ScaleFactorJet, q: float):
        mu = 1.0
        omega = math.sqrt(q * q + mu * mu)
        reference_u = complex(1.0 / math.sqrt(2.0 * omega))
        effective_frequency_squared = (
            q * q + jet.a * jet.a * mu * mu - jet.d2 / jet.a
        )
        return minimal_squeezed_flrw_mode_stress_difference(
            jet,
            q=q,
            mu=mu,
            reference_u=reference_u,
            reference_du_dx=-1.0j * omega * reference_u,
            reference_d2u_dx2=-effective_frequency_squared * reference_u,
            alpha=math.sqrt(1.01),
            beta=0.1j,
        )

    receipts = (
        make_receipt(_minkowski_jet(), 1.0),
        make_receipt(ScaleFactorJet(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0), 2.0),
    )
    tail = CertifiedPowerLawTail(1.0e6, 5.0, 1.0)
    with pytest.raises(ValueError, match="one background jet"):
        integrate_squeezed_created_stress_with_certified_tail(
            receipts,
            energy_tail=tail,
            pressure_tail=tail,
        )


def test_local_observer_contracts_same_isotropic_stress_without_shared_tick() -> None:
    stress = ModeStress(3.0, 1.0)
    comoving = local_isotropic_stress_observer_readout(stress, relative_speed=0.0)
    moving = local_isotropic_stress_observer_readout(
        stress,
        relative_speed=math.sqrt(3.0) / 2.0,
    )
    vacuum_like = local_isotropic_stress_observer_readout(
        ModeStress(3.0, -3.0),
        relative_speed=0.8,
    )

    assert comoving.observer_energy_density == pytest.approx(3.0)
    assert moving.lorentz_gamma == pytest.approx(2.0)
    assert moving.observer_proper_time_rate_per_comoving_cosmic_time == (
        pytest.approx(0.5)
    )
    assert moving.observer_energy_density == pytest.approx(15.0)
    assert vacuum_like.observer_energy_density == pytest.approx(3.0)
    assert not moving.universal_planck_tick_assumed
    with pytest.raises(ValueError, match="magnitude below one"):
        local_isotropic_stress_observer_readout(stress, relative_speed=1.0)


def test_global_massive_squeezed_trajectory_is_dust_like_not_persistent_de() -> None:
    trajectory = _massive_de_sitter_stress_trajectory(steps=1200)

    assert len(trajectory.nodes) == 1201
    assert trajectory.whole_window.h0_cosmic_time_duration == pytest.approx(
        2.0, rel=2.0e-5
    )
    assert trajectory.late_window.h0_cosmic_time_duration == pytest.approx(
        1.0, rel=2.0e-5
    )
    assert trajectory.max_reference_phase_step < 0.08
    assert trajectory.anomalous_phase_turns > 20.0
    assert trajectory.late_half_cycle_efold_diagnostic < 0.05
    assert trajectory.ward.relative_signed_residual < 2.0e-3
    assert trajectory.ward.relative_absolute_accumulated_residual < 2.0e-3
    assert trajectory.ward.max_finite_difference_relative_residual < 2.0e-2
    assert trajectory.late_cold_adiabatic_gates_pass
    assert abs(trajectory.late_window.particle_equation_of_state) < 2.0e-3
    assert trajectory.late_dm_like_average_diagnostic_pass
    assert trajectory.accelerating_state_difference_span_grid_upper < 0.2
    assert trajectory.de_like_state_difference_span_grid_upper < 0.2
    assert trajectory.grid_diagnostic_excludes_required_de_persistence
    assert trajectory.dimensions_pass
    assert trajectory.global_ward_uses_independent_finite_grid_balance
    assert not trajectory.hadamard_or_uv_admissibility_proved
    assert not trajectory.einstein_backreaction_computed
    assert not trajectory.physical_dark_matter_dark_energy_identification


def test_global_squeezed_stress_and_ward_converge_under_mode_refinement() -> None:
    coarse = _massive_de_sitter_stress_trajectory(steps=600)
    medium = _massive_de_sitter_stress_trajectory(steps=1200)
    fine = _massive_de_sitter_stress_trajectory(steps=2400)

    def endpoint_error(left, right) -> float:
        left_stress = left.nodes[-1].receipt.created_state_dependent_stress
        right_stress = right.nodes[-1].receipt.created_state_dependent_stress
        return math.hypot(
            left_stress.energy_density_over_h0_four
            - right_stress.energy_density_over_h0_four,
            left_stress.pressure_over_h0_four
            - right_stress.pressure_over_h0_four,
        )

    assert endpoint_error(medium, fine) < endpoint_error(coarse, fine) / 8.0
    assert (
        medium.ward.relative_absolute_accumulated_residual
        < coarse.ward.relative_absolute_accumulated_residual / 3.0
    )
    assert (
        fine.ward.max_finite_difference_relative_residual
        < medium.ward.max_finite_difference_relative_residual / 3.0
    )


def test_global_squeezed_trajectory_preserves_zero_control_and_phase_input() -> None:
    background, solution = _massive_de_sitter_solution(steps=600)

    def trace(beta: complex):
        return trace_squeezed_flrw_mode_stress(
            background,
            solution,
            scale_factor_jet_at_n=_de_sitter_trajectory_jet,
            alpha=math.sqrt(1.0 + abs(beta) ** 2),
            beta=beta,
            late_window_efolds=1.0,
            canonical_tolerance=1.0e-4,
        )

    zero = trace(0.0j)
    real_phase = trace(complex(math.sqrt(0.5)))
    quadrature_phase = trace(1.0j * math.sqrt(0.5))

    assert all(
        node.receipt.created_state_dependent_stress == ModeStress(0.0, 0.0)
        for node in zero.nodes
    )
    assert zero.whole_window.created_equation_of_state is None
    assert zero.ward.relative_signed_residual == 0.0
    assert (
        real_phase.nodes[0].receipt.created_anomalous_stress
        != quadrature_phase.nodes[0].receipt.created_anomalous_stress
    )
    assert (
        real_phase.late_window.particle_stress_time_average
        == quadrature_phase.late_window.particle_stress_time_average
    )
    assert not real_phase.phase_derived_from_quench_profile


def test_global_squeezed_trajectory_fails_closed_outside_contract() -> None:
    background = _DeSitterTrajectoryBackground()
    nonminimal = solve_flrw_mode(
        background,
        FLRWModeSpec(
            comoving_wavenumber_over_h0=0.2,
            mass_over_h0=lambda _n: 40.0,
            initial_n=-2.0,
            final_n=0.0,
            steps=100,
        ),
    )
    with pytest.raises(ValueError, match="minimal coupling"):
        trace_squeezed_flrw_mode_stress(
            background,
            nonminimal,
            scale_factor_jet_at_n=_de_sitter_trajectory_jet,
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            canonical_tolerance=1.0e-4,
        )

    changing_mass = solve_flrw_mode(
        background,
        FLRWModeSpec(
            comoving_wavenumber_over_h0=0.2,
            mass_over_h0=lambda n: 40.0 + 0.01 * n,
            curvature_coupling=0.0,
            initial_n=-2.0,
            final_n=0.0,
            steps=100,
        ),
    )
    with pytest.raises(ValueError, match="constant mass"):
        trace_squeezed_flrw_mode_stress(
            background,
            changing_mass,
            scale_factor_jet_at_n=_de_sitter_trajectory_jet,
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            canonical_tolerance=1.0e-4,
        )

    massive_background, massive = _massive_de_sitter_solution(steps=900)
    mismatched_background = _DeSitterTrajectoryBackground(
        nodes=(
            _TrajectoryBackgroundNode(-3.0),
            _TrajectoryBackgroundNode(0.1),
        )
    )
    with pytest.raises(ValueError, match="background window"):
        trace_squeezed_flrw_mode_stress(
            mismatched_background,
            massive,
            scale_factor_jet_at_n=_de_sitter_trajectory_jet,
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            canonical_tolerance=1.0e-4,
        )
    with pytest.raises(ValueError, match="scale factor"):
        trace_squeezed_flrw_mode_stress(
            massive_background,
            massive,
            scale_factor_jet_at_n=lambda n: ScaleFactorJet(
                1.01 * math.exp(n), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ),
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            canonical_tolerance=1.0e-4,
        )
    with pytest.raises(ValueError, match="first scale-factor derivative"):
        trace_squeezed_flrw_mode_stress(
            massive_background,
            massive,
            scale_factor_jet_at_n=lambda n: replace(
                _de_sitter_trajectory_jet(n),
                d1=1.01 * _de_sitter_trajectory_jet(n).d1,
            ),
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            canonical_tolerance=1.0e-4,
        )
    wrong_frequency = replace(
        massive,
        nodes=(
            replace(
                massive.nodes[0],
                omega_squared=1.01 * massive.nodes[0].omega_squared,
            ),
            *massive.nodes[1:],
        ),
    )
    with pytest.raises(ValueError, match="minimal mode frequency"):
        trace_squeezed_flrw_mode_stress(
            massive_background,
            wrong_frequency,
            scale_factor_jet_at_n=_de_sitter_trajectory_jet,
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            canonical_tolerance=1.0e-4,
        )
    with pytest.raises(ValueError, match="sampling bound"):
        trace_squeezed_flrw_mode_stress(
            massive_background,
            massive,
            scale_factor_jet_at_n=_de_sitter_trajectory_jet,
            alpha=math.sqrt(1.5),
            beta=math.sqrt(0.5),
            maximum_reference_phase_step=1.0e-6,
            canonical_tolerance=1.0e-4,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha": 1.0, "beta": 0.2},
        {"reference_du_dx": 0.0j},
        {"reference_d2u_dx2": 0.0j},
    ],
)
def test_squeezed_stress_fails_closed_on_invalid_canonical_data(overrides) -> None:
    q = 2.0
    mu = 1.0
    omega = math.sqrt(q * q + mu * mu)
    reference_u = complex(1.0 / math.sqrt(2.0 * omega))
    kwargs = {
        "q": q,
        "mu": mu,
        "reference_u": reference_u,
        "reference_du_dx": -1.0j * omega * reference_u,
        "reference_d2u_dx2": -(omega**2) * reference_u,
        "alpha": math.sqrt(1.04),
        "beta": 0.2,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        minimal_squeezed_flrw_mode_stress_difference(_minkowski_jet(), **kwargs)


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
