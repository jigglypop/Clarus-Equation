from dataclasses import dataclass, replace
import math

import pytest

from examples.physics.darksector.kinetic_dark_sector_adiabatic_stress import (
    _gaussian_q3_tail_moment,
    CertifiedInfraredPowerLaw,
    CertifiedPowerLawTail,
    GaussianBogoliubovProfile,
    MassSquaredJet,
    ModeStress,
    ScaleFactorJet,
    SqueezedFLRWNodeIntegralCertificate,
    aggregate_squeezed_flrw_stress_ensemble,
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
from examples.physics.darksector.kinetic_dark_sector_backreaction import (
    MeanFieldFLRWBackground,
    MeanFieldFLRWBackgroundNode,
    ModeRecomputedSemiclassicalResponse,
    ReferenceFLRWBaselineNode,
    SemiclassicalReferenceSourceNode,
    _three_point_derivative,
    project_squeezed_ensemble_frozen_constraints,
    solve_squeezed_state_difference_mean_field_fixed_point,
)
from examples.physics.darksector.kinetic_dark_sector_gate import (
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


def _massive_de_sitter_solution(
    *,
    steps: int,
    mu: float = 40.0,
    q: float = 0.2,
):
    background = _DeSitterTrajectoryBackground()
    solution = solve_flrw_mode(
        background,
        FLRWModeSpec(
            comoving_wavenumber_over_h0=q,
            mass_over_h0=lambda _n: mu,
            curvature_coupling=0.0,
            initial_n=-2.0,
            final_n=0.0,
            steps=steps,
        ),
    )
    return background, solution


def _massive_de_sitter_stress_trajectory(
    *,
    steps: int,
    beta: complex | None = None,
    q: float = 0.2,
):
    background, solution = _massive_de_sitter_solution(steps=steps, q=q)
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


def _squeezed_ensemble_trajectories(
    q_values: tuple[float, ...],
    *,
    steps: int,
    zero_squeeze: bool = False,
):
    trajectories = []
    for q in q_values:
        beta = 0.0j if zero_squeeze else 0.15j * math.exp(-8.0 * q * q)
        trajectories.append(
            _massive_de_sitter_stress_trajectory(
                steps=steps,
                q=q,
                beta=beta,
            )
        )
    return tuple(trajectories)


def _squeezed_ensemble_certificates(trajectories):
    """Build caller-asserted endpoint envelopes for contract-plumbing tests.

    These fixtures do not derive an analytic bound on either omitted half-axis.
    The production API therefore keeps the certificates explicitly external.
    """

    q_min = trajectories[0].q
    q_max = trajectories[-1].q
    certificates = []
    for index in range(len(trajectories[0].nodes)):
        first = trajectories[0].nodes[index].receipt.created_state_dependent_stress
        last = trajectories[-1].nodes[index].receipt.created_state_dependent_stress

        def coefficient(value: float, q: float, exponent: float) -> float:
            if value == 0.0:
                return 0.0
            return 1.01 * abs(value) * q ** (-exponent)

        certificates.append(
            SqueezedFLRWNodeIntegralCertificate(
                energy_ir=CertifiedInfraredPowerLaw(
                    coefficient(first.energy_density_over_h0_four, q_min, 0.0),
                    0.0,
                    q_min,
                ),
                pressure_ir=CertifiedInfraredPowerLaw(
                    coefficient(first.pressure_over_h0_four, q_min, 0.0),
                    0.0,
                    q_min,
                ),
                energy_uv=CertifiedPowerLawTail(
                    coefficient(last.energy_density_over_h0_four, q_max, -7.0),
                    7.0,
                    q_max,
                ),
                pressure_uv=CertifiedPowerLawTail(
                    coefficient(last.pressure_over_h0_four, q_max, -7.0),
                    7.0,
                    q_max,
                ),
            )
        )
    return tuple(certificates)


def _gaussian_bogoliubov_profile(*, zero_squeeze: bool = False):
    return GaussianBogoliubovProfile(
        amplitude=0.0 if zero_squeeze else 0.15,
        q_scale=1.0 / math.sqrt(8.0),
        beta_phase=math.pi / 2.0,
    )


def _squeezed_ensemble(
    q_values: tuple[float, ...],
    *,
    steps: int,
    zero_squeeze: bool = False,
):
    trajectories = _squeezed_ensemble_trajectories(
        q_values,
        steps=steps,
        zero_squeeze=zero_squeeze,
    )
    return aggregate_squeezed_flrw_stress_ensemble(
        trajectories,
        node_certificates=_squeezed_ensemble_certificates(trajectories),
        bogoliubov_profile=_gaussian_bogoliubov_profile(
            zero_squeeze=zero_squeeze
        ),
    )


def _de_sitter_reference_baseline(ensemble, *, reduced_planck_over_h0: float):
    energy_density = 3.0 * reduced_planck_over_h0**2
    return tuple(
        ReferenceFLRWBaselineNode(
            n=node.n,
            e=1.0,
            d_log_e_d_n=0.0,
            energy_density=energy_density,
            pressure=-energy_density,
            energy_density_d_n=0.0,
        )
        for node in ensemble.nodes
    )


def _frozen_constraint_projection(
    ensemble,
    *,
    reduced_planck_over_h0: float = 1.0e4,
    maximum_state_difference_ward_relative_residual: float = 1.0,
):
    return project_squeezed_ensemble_frozen_constraints(
        ensemble,
        baseline_nodes=_de_sitter_reference_baseline(
            ensemble,
            reduced_planck_over_h0=reduced_planck_over_h0,
        ),
        reduced_planck_over_h0=reduced_planck_over_h0,
        baseline_reference_sector_declaration=(
            "CLASSICAL_DE_SITTER_PLUS_RENORMALIZED_REFERENCE_STATE"
        ),
        reference_renormalized_sector_included_in_baseline=True,
        maximum_state_difference_ward_relative_residual=(
            maximum_state_difference_ward_relative_residual
        ),
    )


def _mean_field_background(
    *,
    steps: int = 60,
    e2: float = 1.0,
    initial_n: float = -0.2,
    final_n: float = 0.0,
) -> MeanFieldFLRWBackground:
    step = (final_n - initial_n) / steps
    return MeanFieldFLRWBackground(
        nodes=tuple(
            MeanFieldFLRWBackgroundNode(
                n=initial_n + index * step,
                e2=e2,
            )
            for index in range(-2, steps + 3)
        ),
        active_window=(initial_n, final_n),
        curvature_derivative_step_n=step / 4.0,
    )


def _mode_recomputed_mean_field_response(
    background: MeanFieldFLRWBackground,
    *,
    reduced_planck_over_h0: float,
    zero_squeeze: bool,
    reference_target_e2: float = 1.0,
    reference_gain: float = 0.0,
    finite_reference_split: float = 0.0,
    time_dependent_reference_split: bool = False,
    reference_energy_absolute_bound: float = 0.0,
    reference_pressure_absolute_bound: float = 0.0,
) -> ModeRecomputedSemiclassicalResponse:
    active = background.active_nodes
    steps = len(active) - 1
    q_values = (0.1, 0.3, 0.5)
    profile = _gaussian_bogoliubov_profile(zero_squeeze=zero_squeeze)
    trajectories = []
    solutions = []
    for q in q_values:
        solution = solve_flrw_mode(
            background,
            FLRWModeSpec(
                comoving_wavenumber_over_h0=q,
                mass_over_h0=lambda _n: 4.0,
                curvature_coupling=0.0,
                initial_n=active[0].n,
                final_n=active[-1].n,
                steps=steps,
                curvature_derivative_step_n=(
                    background.curvature_derivative_step_n
                ),
                adiabatic_derivative_step_n=(
                    background.curvature_derivative_step_n
                ),
            ),
        )
        solutions.append(solution)
        trajectories.append(
            trace_squeezed_flrw_mode_stress(
                background,
                solution,
                scale_factor_jet_at_n=(
                    background.state_difference_scale_factor_jet_at_n
                ),
                alpha=profile.alpha_at(q),
                beta=profile.beta_at(q),
                late_window_efolds=0.1,
                maximum_reference_phase_step=math.pi / 2.0,
                canonical_tolerance=1.0e-4,
                background_tolerance=1.0e-7,
                required_persistent_de_efolds=0.1,
            )
        )
    trajectories_tuple = tuple(trajectories)
    ensemble = aggregate_squeezed_flrw_stress_ensemble(
        trajectories_tuple,
        node_certificates=_squeezed_ensemble_certificates(trajectories_tuple),
        bogoliubov_profile=profile,
        late_window_efolds=0.1,
        required_persistent_de_efolds=0.1,
    )

    candidate_average_e2 = sum(node.e2 for node in active) / len(active)
    supplied_target_e2 = reference_target_e2 + reference_gain * (
        candidate_average_e2 - reference_target_e2
    )
    reference_energy = (
        3.0 * reduced_planck_over_h0**2 * supplied_target_e2
    )
    reference_nodes = []
    split_nodes = []
    for node in active:
        if time_dependent_reference_split:
            split_energy = finite_reference_split * math.exp(
                -3.0 * (node.n - active[0].n)
            )
            split_pressure = 0.0
            split_energy_d_n = -3.0 * split_energy
        else:
            split_energy = finite_reference_split
            split_pressure = -finite_reference_split
            split_energy_d_n = 0.0
        reference_nodes.append(
            SemiclassicalReferenceSourceNode(
                n=node.n,
                energy_density=reference_energy + split_energy,
                pressure=-reference_energy + split_pressure,
                energy_density_d_n=split_energy_d_n,
                energy_absolute_bound=reference_energy_absolute_bound,
                pressure_absolute_bound=reference_pressure_absolute_bound,
            )
        )
        split_nodes.append(
            SemiclassicalReferenceSourceNode(
                n=node.n,
                energy_density=-split_energy,
                pressure=-split_pressure,
                energy_density_d_n=-split_energy_d_n,
            )
        )
    return ModeRecomputedSemiclassicalResponse(
        ensemble=ensemble,
        reference_source_nodes=tuple(reference_nodes),
        finite_reference_split_adjustment_nodes=(
            tuple(split_nodes) if finite_reference_split != 0.0 else ()
        ),
        maximum_mode_wronskian_residual=max(
            solution.max_wronskian_residual for solution in solutions
        ),
        mode_solution_count=len(solutions),
        renormalization_scheme_declaration=(
            "TEST_EXTERNAL_REFERENCE_AND_COMMON_FINITE_COUNTERTERMS"
        ),
        state_preparation_declaration=(
            "FIXED_GAUSSIAN_BOGOLIUBOV_PROFILE_REBUILT_ON_EACH_BACKGROUND"
        ),
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

    assert result.energy_density_over_h0_four == pytest.approx(
        0.703125 / (2.0 * math.pi**2)
    )
    assert result.energy_tail_absolute_bound == pytest.approx(
        1.0 / (4.0 * math.pi**2 * 4.0**2)
    )
    assert result.pressure_tail_absolute_bound == pytest.approx(
        2.0 / (4.0 * math.pi**2 * 4.0**2)
    )


def test_certified_infrared_power_law_has_exact_isotropic_bound() -> None:
    certificate = CertifiedInfraredPowerLaw(2.0, 0.0, 1.0)

    assert certificate.isotropic_integral_bound_to(0.5) == pytest.approx(
        1.0 / (24.0 * math.pi**2)
    )
    with pytest.raises(ValueError, match="infrared bound"):
        certificate.isotropic_integral_bound_to(2.0)


def test_external_power_law_bounds_fail_closed_on_nonfinite_evaluation() -> None:
    with pytest.raises(ValueError, match="not finite"):
        CertifiedPowerLawTail(1.0e308, 4.0, 1.0e-300).pointwise_bound_at(
            1.0e-300
        )
    with pytest.raises(ValueError, match="not finite"):
        CertifiedInfraredPowerLaw(1.0e308, -2.9, 1.0).pointwise_bound_at(
            1.0e-300
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


def test_squeezed_flrw_ensemble_integrates_hidden_modes_with_explicit_bounds() -> None:
    q_values = tuple(0.05 + 1.15 * index / 16.0 for index in range(17))
    ensemble = _squeezed_ensemble(q_values, steps=1200)

    assert len(ensemble.nodes) == 1201
    assert ensemble.q_values == q_values
    assert ensemble.nodes[-1].created_stress.energy_ir_absolute_bound > 0.0
    assert ensemble.nodes[-1].created_stress.energy_uv_absolute_bound > 0.0
    assert all(node.hubble_over_h0 == pytest.approx(1.0) for node in ensemble.nodes)
    assert all(
        node.background_d_log_h_d_n == pytest.approx(0.0, abs=2.0e-15)
        for node in ensemble.nodes
    )
    assert ensemble.ward.central_grid.relative_signed_residual < 2.0e-3
    assert ensemble.ward.central_grid.relative_absolute_accumulated_residual < 2.0e-3
    assert ensemble.ward.sampled_ir_uv_balance_uncertainty_bound > 0.0
    assert ensemble.late_finite_q_grid_cold_adiabatic_gates_pass
    assert abs(ensemble.late_window.particle_grid_equation_of_state) < 2.0e-3
    assert ensemble.late_particle_grid_dm_like_diagnostic_pass
    assert not ensemble.sampled_nodes_meet_required_de_run_length
    assert ensemble.dimensions_pass
    assert ensemble.analytic_bogoliubov_profile_verified
    assert ensemble.absolute_bogoliubov_amplitude_moments_certified
    assert not ensemble.evolved_mode_stress_tail_derived_from_profile
    assert ensemble.ward.ensemble_ward_recomputed_after_q_integration
    assert not ensemble.ward.mode_ward_receipts_merely_summed
    assert not ensemble.time_global_tail_ward_certified
    assert not ensemble.q_quadrature_error_certified
    assert not ensemble.time_quadrature_error_certified
    assert not ensemble.continuous_de_persistence_certified
    assert not ensemble.full_ir_uv_particle_sector_coldness_proved
    assert not ensemble.hadamard_state_proved
    assert not ensemble.einstein_backreaction_computed
    assert not ensemble.physical_dark_matter_dark_energy_identification


def test_gaussian_bogoliubov_profile_has_exact_q3_amplitude_moments_only() -> None:
    q_values = (0.05, 0.3375, 0.625, 0.9125, 1.2)
    ensemble = _squeezed_ensemble(q_values, steps=900)
    certificate = ensemble.bogoliubov_integrability_certificate
    assert certificate is not None

    profile = certificate.profile
    q_max = q_values[-1]
    u = (q_max / profile.q_scale) ** 2
    anomalous_moment = (
        profile.q_scale**4 * (u + 1.0) * math.exp(-u) / 2.0
    )
    particle_moment = (
        profile.q_scale**4 * (2.0 * u + 1.0) * math.exp(-2.0 * u) / 8.0
    )
    assert certificate.tail_start_q == q_max
    assert certificate.anomalous_q3_amplitude_moment_upper == pytest.approx(
        math.sqrt(1.0 + profile.amplitude**2)
        * profile.amplitude
        * anomalous_moment,
        rel=2.0e-15,
    )
    assert certificate.particle_q3_amplitude_squared_moment_upper == pytest.approx(
        profile.amplitude**2 * particle_moment,
        rel=2.0e-15,
    )
    assert certificate.bogoliubov_normalization_exact_by_construction
    assert certificate.stress_power_counting_moments_finite
    assert certificate.gaussian_exponent_argument_dimensionless
    assert certificate.dimensions_pass
    assert all(
        dimension == 0.0 for _, dimension in certificate.mass_dimension_manifest
    )
    assert not certificate.evolved_mode_stress_tail_bounded
    assert not certificate.time_global_tail_ward_certified
    assert not certificate.reference_state_hadamard_proved
    assert not certificate.full_state_hadamard_proved
    assert not certificate.absolute_renormalized_stress_proved


def test_gaussian_q3_moment_closes_zero_infinite_and_invalid_boundaries() -> None:
    assert _gaussian_q3_tail_moment(
        q_scale=2.0,
        tail_start_q=0.0,
        exponential_rate=1.0,
    ) == pytest.approx(8.0)
    assert _gaussian_q3_tail_moment(
        q_scale=2.0,
        tail_start_q=0.0,
        exponential_rate=2.0,
    ) == pytest.approx(2.0)
    assert _gaussian_q3_tail_moment(
        q_scale=1.0,
        tail_start_q=1.0e200,
        exponential_rate=1.0,
    ) == 0.0
    with pytest.raises(ValueError, match="exponential_rate"):
        _gaussian_q3_tail_moment(
            q_scale=1.0,
            tail_start_q=0.0,
            exponential_rate=0.0,
        )
    with pytest.raises(ValueError, match="not finite"):
        _gaussian_q3_tail_moment(
            q_scale=1.0e100,
            tail_start_q=0.0,
            exponential_rate=1.0,
        )


def test_nonuniform_three_point_derivative_is_exact_on_quadratics_and_conditioned() -> None:
    x_values = (0.0, 0.2, 0.9, 2.0)
    y_values = tuple(x * x + 3.0 * x - 4.0 for x in x_values)
    assert _three_point_derivative(
        x_values,
        y_values,
        maximum_adjacent_step_ratio=6.0,
    ) == pytest.approx(tuple(2.0 * x + 3.0 for x in x_values), abs=2.0e-14)
    with pytest.raises(ValueError, match="adjacent-step ratio"):
        _three_point_derivative(
            (0.0, 1.0e-12, 1.0),
            (0.0, 1.0e-24, 1.0),
            maximum_adjacent_step_ratio=10.0,
        )


def test_frozen_constraint_projection_closes_algebraic_constraints_without_claiming_backreaction() -> None:
    q_values = (0.05, 0.3375, 0.625, 0.9125, 1.2)
    projection = _frozen_constraint_projection(
        _squeezed_ensemble(q_values, steps=1200)
    )

    assert projection.maximum_relative_e_squared_shift_upper < 1.0e-4
    assert projection.maximum_baseline_friedmann_relative_residual == 0.0
    assert projection.maximum_baseline_raychaudhuri_relative_residual == 0.0
    assert projection.maximum_baseline_ward_relative_residual == 0.0
    assert projection.dimensions_pass
    assert all(dimension == 0.0 for _, dimension in projection.mass_dimension_manifest)
    assert projection.fixed_comoving_q_measure_applied_once
    assert projection.degeneracy_applied_once_after_q_integration
    assert projection.initial_occupation_already_in_state_difference
    assert projection.adjacent_n_step_ratio == pytest.approx(1.0)
    assert projection.independent_energy_pressure_tail_bounds_assumed
    assert not projection.joint_rho_p_tail_region_derived
    assert projection.finite_difference_conditioning_pass
    assert not projection.finite_difference_truncation_error_certified
    assert projection.baseline_closure_absolute_tolerance == 1.0e-12
    assert projection.state_difference_ward_absolute_tolerance == 1.0e-12
    assert projection.frozen_constraint_projection_computed
    assert not projection.gaussian_profile_derives_evolved_stress_tail
    assert not projection.tail_time_derivative_certified
    assert not projection.continuous_total_ward_identity_certified
    assert not projection.projected_geometry_evolved
    assert not projection.modes_recomputed_on_projected_geometry
    assert not projection.reference_renormalized_stress_recomputed
    assert not projection.semiclassical_einstein_equation_solved
    assert not projection.einstein_backreaction_computed
    assert not projection.physical_dark_matter_dark_energy_identification

    for node in projection.nodes:
        assert node.projected_e_squared_interval[0] <= node.projected_e_squared
        assert node.projected_e_squared <= node.projected_e_squared_interval[1]
        assert (
            node.projected_d_log_e_d_n_interval[0]
            <= node.projected_d_log_e_d_n
            <= node.projected_d_log_e_d_n_interval[1]
        )
        assert (
            node.projected_acceleration_over_h0_squared_interval[0]
            <= node.projected_acceleration_over_h0_squared
            <= node.projected_acceleration_over_h0_squared_interval[1]
        )
        closure_scale = max(1.0, abs(node.closure.total_energy_density))
        assert abs(node.closure.friedmann_constraint_residual) / closure_scale < 2.0e-15
        assert abs(node.closure.raychaudhuri_residual) < 2.0e-15
        assert node.closure.fluid_ward_residuals == pytest.approx(
            (node.baseline_ward_residual, node.state_difference_ward_residual)
        )


def test_frozen_constraint_projection_zero_squeeze_recovers_reference_exactly() -> None:
    ensemble = _squeezed_ensemble(
        (0.05, 0.625, 1.2),
        steps=900,
        zero_squeeze=True,
    )
    projection = _frozen_constraint_projection(ensemble)

    assert projection.maximum_relative_e_squared_shift_upper == 0.0
    assert projection.maximum_state_difference_ward_relative_residual == 0.0
    assert all(
        node.projected_e == 1.0
        and node.projected_e_squared_interval == (1.0, 1.0)
        and node.projected_d_log_e_d_n == 0.0
        and node.projected_d_log_e_d_n_interval == (0.0, 0.0)
        and node.projected_acceleration_over_h0_squared == 1.0
        and node.state_difference_ward_residual == 0.0
        for node in projection.nodes
    )


def test_frozen_constraint_projection_fails_closed_on_missing_reference_or_nonpositive_bound() -> None:
    ensemble = _squeezed_ensemble((0.05, 0.625, 1.2), steps=900)
    planck = 1.0e4
    baseline = _de_sitter_reference_baseline(
        ensemble,
        reduced_planck_over_h0=planck,
    )
    common = {
        "baseline_nodes": baseline,
        "reduced_planck_over_h0": planck,
        "baseline_reference_sector_declaration": (
            "CLASSICAL_DE_SITTER_PLUS_RENORMALIZED_REFERENCE_STATE"
        ),
    }

    with pytest.raises(ValueError, match="renormalized reference sector"):
        project_squeezed_ensemble_frozen_constraints(
            ensemble,
            reference_renormalized_sector_included_in_baseline=False,
            **common,
        )
    with pytest.raises(ValueError, match="not synchronized"):
        project_squeezed_ensemble_frozen_constraints(
            ensemble,
            baseline_nodes=(replace(baseline[0], e=1.01), *baseline[1:]),
            reduced_planck_over_h0=planck,
            baseline_reference_sector_declaration=(
                "CLASSICAL_DE_SITTER_PLUS_RENORMALIZED_REFERENCE_STATE"
            ),
            reference_renormalized_sector_included_in_baseline=True,
        )

    first = ensemble.nodes[0]
    nonpositive_lower = replace(
        first,
        created_stress=replace(
            first.created_stress,
            energy_ir_absolute_bound=6.0 * planck**2,
        ),
    )
    unsafe_ensemble = replace(
        ensemble,
        nodes=(nonpositive_lower, *ensemble.nodes[1:]),
    )
    with pytest.raises(ValueError, match=r"makes E\^2 non-positive"):
        project_squeezed_ensemble_frozen_constraints(
            unsafe_ensemble,
            reference_renormalized_sector_included_in_baseline=True,
            **common,
        )


def test_frozen_constraint_projection_separates_absolute_and_relative_residual_gates() -> None:
    ensemble = _squeezed_ensemble(
        (0.05, 0.625, 1.2),
        steps=900,
        zero_squeeze=True,
    )
    planck = 0.1
    baseline = _de_sitter_reference_baseline(
        ensemble,
        reduced_planck_over_h0=planck,
    )
    small_scale_but_inconsistent = (
        replace(baseline[0], energy_density_d_n=5.0e-10),
        *baseline[1:],
    )

    with pytest.raises(ValueError, match="does not close"):
        project_squeezed_ensemble_frozen_constraints(
            ensemble,
            baseline_nodes=small_scale_but_inconsistent,
            reduced_planck_over_h0=planck,
            baseline_reference_sector_declaration=(
                "SMALL_SCALE_ABSOLUTE_RELATIVE_GATE_COUNTEREXAMPLE"
            ),
            reference_renormalized_sector_included_in_baseline=True,
            baseline_closure_tolerance=1.0e-9,
            baseline_closure_absolute_tolerance=1.0e-12,
        )


def test_mean_field_background_builds_only_a_second_order_state_difference_jet() -> None:
    background = _mean_field_background(steps=40)
    n = -0.1
    a = math.exp(n)
    jet = background.state_difference_scale_factor_jet_at_n(n)

    assert jet.derivatives == pytest.approx(
        (a, a**2, 2.0 * a**3, 0.0, 0.0, 0.0, 0.0),
        abs=2.0e-14,
    )
    assert background.state_difference_jet_derivative_order == 2
    assert not background.higher_jet_derivatives_certified
    assert not background.suitable_for_absolute_adiabatic_subtraction

    active = background.active_nodes
    sloped = background.with_active_e2(
        tuple(1.0 + 0.02 * (node.n - active[0].n) for node in active)
    )
    left_guard = sloped.nodes[0]
    right_guard = sloped.nodes[-1]
    assert left_guard.e2 == pytest.approx(
        1.0 + 0.02 * (left_guard.n - active[0].n),
        abs=2.0e-14,
    )
    assert right_guard.e2 == pytest.approx(
        1.0 + 0.02 * (right_guard.n - active[0].n),
        abs=2.0e-14,
    )


def test_mode_recomputed_mean_field_zero_squeeze_recovers_reference_exactly() -> None:
    planck = 20.0
    fingerprints = []

    def recompute(background, _iteration):
        fingerprints.append(tuple(node.e2 for node in background.active_nodes))
        return _mode_recomputed_mean_field_response(
            background,
            reduced_planck_over_h0=planck,
            zero_squeeze=True,
        )

    fixed_point = solve_squeezed_state_difference_mean_field_fixed_point(
        _mean_field_background(),
        recompute_response=recompute,
        reduced_planck_over_h0=planck,
        fixed_point_relative_tolerance=1.0e-12,
        constraint_absolute_tolerance=1.0e-10,
        constraint_relative_tolerance=1.0e-6,
        ward_absolute_tolerance=1.0e-10,
        ward_relative_tolerance=1.0e-6,
    )

    assert len(fingerprints) == 2
    assert fingerprints[0] == fingerprints[1]
    assert fixed_point.response_evaluation_count == 2
    assert fixed_point.maximum_final_fixed_point_relative_residual == 0.0
    assert all(node.e2 == 1.0 for node in fixed_point.background.active_nodes)
    assert all(
        node.created_stress.energy_density_over_h0_four == 0.0
        and node.created_stress.pressure_over_h0_four == 0.0
        for node in fixed_point.final_response.ensemble.nodes
    )
    assert fixed_point.modes_recomputed_each_iteration_by_callback_contract
    assert (
        fixed_point.final_modes_recomputed_on_converged_background_by_callback_contract
    )
    assert fixed_point.reference_source_evaluated_each_iteration_by_callback_contract
    assert not fixed_point.callback_recomputation_declarations_independently_proved
    assert fixed_point.aggregate_numeric_response_reproducibility_checked
    assert not fixed_point.complete_internal_mode_trajectory_reproducibility_proved
    assert fixed_point.mode_and_ray_use_same_background_derivative
    assert fixed_point.pressure_tail_propagated_to_raychaudhuri_gate
    assert fixed_point.reference_and_split_derivatives_checked_against_node_grid
    assert fixed_point.independent_energy_pressure_tail_rectangle_used
    assert not fixed_point.joint_energy_pressure_tail_region_derived
    assert not fixed_point.continuous_tail_ward_certified
    assert fixed_point.dimensions_pass
    assert all(dimension == 0.0 for _, dimension in fixed_point.mass_dimension_manifest)
    assert not fixed_point.higher_jet_derivatives_certified
    assert not fixed_point.absolute_reference_renormalization_derived
    assert not fixed_point.full_hadamard_state_proved
    assert not fixed_point.semiclassical_einstein_equation_solved
    assert not fixed_point.physical_dark_matter_dark_energy_identification


def test_mode_recomputed_mean_field_updates_background_and_rebuilds_modes() -> None:
    planck = 30.0
    fingerprints = []

    def recompute(background, _iteration):
        fingerprints.append(tuple(node.e2 for node in background.active_nodes))
        return _mode_recomputed_mean_field_response(
            background,
            reduced_planck_over_h0=planck,
            zero_squeeze=False,
        )

    fixed_point = solve_squeezed_state_difference_mean_field_fixed_point(
        _mean_field_background(e2=1.0002),
        recompute_response=recompute,
        reduced_planck_over_h0=planck,
        damping=0.8,
        maximum_iterations=12,
        fixed_point_relative_tolerance=2.0e-7,
        constraint_absolute_tolerance=2.0e-7,
        constraint_relative_tolerance=0.25,
        ward_absolute_tolerance=2.0e-7,
        ward_relative_tolerance=0.25,
    )

    assert len(fixed_point.iterations) >= 2
    assert len(fingerprints) == fixed_point.response_evaluation_count
    assert fixed_point.response_evaluation_count == len(fixed_point.iterations) + 1
    assert fingerprints[0] != fingerprints[-1]
    assert fingerprints[-2] == fingerprints[-1]
    assert fixed_point.maximum_final_fixed_point_relative_residual <= 2.0e-7
    assert fixed_point.final_response_reproducibility_relative_residual == 0.0
    assert fixed_point.maximum_observed_empirical_residual_ratio < 1.0
    assert fixed_point.maximum_final_geometry_derivative_relative_mismatch < 1.0e-5
    assert fixed_point.maximum_final_raychaudhuri_absolute_uncertainty > 0.0
    assert fixed_point.maximum_final_raychaudhuri_tail_robust_relative_residual < 2.0e-7


def test_mode_recomputed_mean_field_time_grid_and_relaxation_refinement_agree() -> None:
    planck = 30.0

    def solve_refinement(*, steps: int, damping: float):
        return solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(steps=steps, e2=1.0002),
            recompute_response=lambda background, _iteration: (
                _mode_recomputed_mean_field_response(
                    background,
                    reduced_planck_over_h0=planck,
                    zero_squeeze=False,
                )
            ),
            reduced_planck_over_h0=planck,
            damping=damping,
            maximum_iterations=16,
            fixed_point_relative_tolerance=2.0e-7,
            constraint_absolute_tolerance=2.0e-7,
            constraint_relative_tolerance=0.3,
            ward_absolute_tolerance=2.0e-7,
            ward_relative_tolerance=0.3,
        )

    coarse = solve_refinement(steps=40, damping=0.6)
    fine = solve_refinement(steps=80, damping=0.8)
    coarse_end = coarse.background.active_nodes[-1].e2
    fine_end = fine.background.active_nodes[-1].e2

    assert abs(coarse_end - fine_end) < 5.0e-7
    assert coarse.maximum_final_fixed_point_relative_residual <= 2.0e-7
    assert fine.maximum_final_fixed_point_relative_residual <= 2.0e-7
    assert coarse.maximum_observed_empirical_residual_ratio < 1.0
    assert fine.maximum_observed_empirical_residual_ratio < 1.0
    assert (
        fine.maximum_final_geometry_derivative_relative_mismatch
        < coarse.maximum_final_geometry_derivative_relative_mismatch
    )


def test_mode_recomputed_mean_field_rejects_cached_modes_after_geometry_changes() -> None:
    planck = 20.0
    cached = None

    def recompute(background, _iteration):
        nonlocal cached
        if cached is None:
            cached = _mode_recomputed_mean_field_response(
                background,
                reduced_planck_over_h0=planck,
                zero_squeeze=True,
            )
        return cached

    with pytest.raises(ValueError, match="not synchronized to candidate background"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(e2=1.001),
            recompute_response=recompute,
            reduced_planck_over_h0=planck,
            damping=1.0,
            fixed_point_relative_tolerance=1.0e-10,
        )


def test_mode_recomputed_mean_field_is_invariant_under_finite_reference_split() -> None:
    planck = 20.0

    def solve_with_split(split: float):
        return solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(),
            recompute_response=lambda background, _iteration: (
                _mode_recomputed_mean_field_response(
                    background,
                    reduced_planck_over_h0=planck,
                    zero_squeeze=True,
                    finite_reference_split=split,
                )
            ),
            reduced_planck_over_h0=planck,
            fixed_point_relative_tolerance=1.0e-12,
            constraint_absolute_tolerance=1.0e-10,
            constraint_relative_tolerance=1.0e-6,
            ward_absolute_tolerance=1.0e-10,
            ward_relative_tolerance=1.0e-6,
        )

    unsplit = solve_with_split(0.0)
    split = solve_with_split(7.0)
    assert tuple(node.e2 for node in split.background.active_nodes) == pytest.approx(
        tuple(node.e2 for node in unsplit.background.active_nodes),
        abs=0.0,
    )
    assert split.maximum_final_total_ward_relative_residual == 0.0

    time_dependent_split = solve_squeezed_state_difference_mean_field_fixed_point(
        _mean_field_background(),
        recompute_response=lambda background, _iteration: (
            _mode_recomputed_mean_field_response(
                background,
                reduced_planck_over_h0=planck,
                zero_squeeze=True,
                finite_reference_split=0.01,
                time_dependent_reference_split=True,
            )
        ),
        reduced_planck_over_h0=planck,
        fixed_point_relative_tolerance=1.0e-12,
        constraint_absolute_tolerance=1.0e-8,
        constraint_relative_tolerance=1.0e-5,
        ward_absolute_tolerance=1.0e-7,
        ward_relative_tolerance=1.0e-3,
    )
    assert tuple(
        node.e2 for node in time_dependent_split.background.active_nodes
    ) == pytest.approx(
        tuple(node.e2 for node in unsplit.background.active_nodes),
        abs=0.0,
    )
    assert (
        time_dependent_split.maximum_final_split_derivative_consistency_relative_residual
        < 1.0e-5
    )


def test_mode_recomputed_mean_field_rejects_nonpositive_branch_and_runaway() -> None:
    planck = 20.0
    with pytest.raises(ValueError, match=r"positive E\^2"):
        MeanFieldFLRWBackgroundNode(n=0.0, e2=0.0)

    with pytest.raises(ValueError, match="non-positive"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(),
            recompute_response=lambda background, _iteration: (
                _mode_recomputed_mean_field_response(
                    background,
                    reduced_planck_over_h0=planck,
                    zero_squeeze=True,
                    reference_target_e2=-1.0,
                )
            ),
            reduced_planck_over_h0=planck,
        )

    with pytest.raises(ValueError, match="detected a runaway"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(e2=1.001),
            recompute_response=lambda background, _iteration: (
                _mode_recomputed_mean_field_response(
                    background,
                    reduced_planck_over_h0=planck,
                    zero_squeeze=True,
                    reference_gain=2.0,
                )
            ),
            reduced_planck_over_h0=planck,
            damping=1.0,
            maximum_iterations=8,
            fixed_point_relative_tolerance=1.0e-10,
            runaway_patience=2,
        )


def test_mode_recomputed_mean_field_requires_same_background_reproducibility() -> None:
    planck = 20.0

    def drifting_response(background, iteration):
        return _mode_recomputed_mean_field_response(
            background,
            reduced_planck_over_h0=planck,
            zero_squeeze=True,
            reference_target_e2=1.0 + iteration * 1.0e-4,
        )

    with pytest.raises(ValueError, match="final response is not reproducible"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(),
            recompute_response=drifting_response,
            reduced_planck_over_h0=planck,
            fixed_point_relative_tolerance=1.0e-12,
        )


def test_mode_recomputed_mean_field_fingerprint_catches_hidden_response_drift() -> None:
    planck = 20.0

    def drifting_wronskian(background, iteration):
        response = _mode_recomputed_mean_field_response(
            background,
            reduced_planck_over_h0=planck,
            zero_squeeze=True,
        )
        if iteration == 0:
            return response
        return replace(
            response,
            maximum_mode_wronskian_residual=(
                response.maximum_mode_wronskian_residual + 1.0e-6
            ),
        )

    with pytest.raises(ValueError, match="final response is not reproducible"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(),
            recompute_response=drifting_wronskian,
            reduced_planck_over_h0=planck,
            fixed_point_relative_tolerance=1.0e-12,
        )


def test_mode_recomputed_mean_field_rejects_fabricated_source_derivative() -> None:
    planck = 20.0

    def fabricated_derivative(background, _iteration):
        response = _mode_recomputed_mean_field_response(
            background,
            reduced_planck_over_h0=planck,
            zero_squeeze=True,
        )
        first = replace(
            response.reference_source_nodes[0],
            energy_density_d_n=1.0,
        )
        return replace(
            response,
            reference_source_nodes=(first, *response.reference_source_nodes[1:]),
        )

    with pytest.raises(ValueError, match="derivative disagrees with its node grid"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(),
            recompute_response=fabricated_derivative,
            reduced_planck_over_h0=planck,
        )


def test_mode_recomputed_mean_field_pressure_tail_enters_ray_gate() -> None:
    planck = 20.0

    with pytest.raises(ValueError, match="independent tail rectangle"):
        solve_squeezed_state_difference_mean_field_fixed_point(
            _mean_field_background(),
            recompute_response=lambda background, _iteration: (
                _mode_recomputed_mean_field_response(
                    background,
                    reduced_planck_over_h0=planck,
                    zero_squeeze=True,
                    reference_pressure_absolute_bound=1.0,
                )
            ),
            reduced_planck_over_h0=planck,
            fixed_point_relative_tolerance=1.0e-12,
        )


def test_squeezed_flrw_ensemble_zero_control_and_contract_failures() -> None:
    q_values = (0.05, 0.625, 1.2)
    zero_trajectories = _squeezed_ensemble_trajectories(
        q_values,
        steps=900,
        zero_squeeze=True,
    )
    zero_certificates = _squeezed_ensemble_certificates(zero_trajectories)
    zero = aggregate_squeezed_flrw_stress_ensemble(
        zero_trajectories,
        node_certificates=zero_certificates,
        bogoliubov_profile=_gaussian_bogoliubov_profile(zero_squeeze=True),
    )

    assert all(
        node.created_stress.energy_density_over_h0_four == 0.0
        and node.created_stress.pressure_over_h0_four == 0.0
        and node.created_stress.energy_external_ir_uv_remainder_absolute_bound
        == 0.0
        and node.created_stress.pressure_external_ir_uv_remainder_absolute_bound
        == 0.0
        for node in zero.nodes
    )
    assert zero.ward.central_grid.relative_signed_residual == 0.0
    assert zero.ward.sampled_ir_uv_balance_uncertainty_bound == 0.0

    trajectories = _squeezed_ensemble_trajectories(q_values, steps=900)
    certificates = _squeezed_ensemble_certificates(trajectories)
    with pytest.raises(ValueError, match="every time node"):
        aggregate_squeezed_flrw_stress_ensemble(
            trajectories,
            node_certificates=certificates[:-1],
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        aggregate_squeezed_flrw_stress_ensemble(
            tuple(reversed(trajectories)),
            node_certificates=certificates,
        )
    shifted = replace(
        trajectories[-1],
        nodes=(
            replace(
                trajectories[-1].nodes[0],
                x=trajectories[-1].nodes[0].x + 1.0e-3,
            ),
            *trajectories[-1].nodes[1:],
        ),
    )
    with pytest.raises(ValueError, match="synchronized"):
        aggregate_squeezed_flrw_stress_ensemble(
            (*trajectories[:-1], shifted),
            node_certificates=certificates,
        )
    inconsistent_q = replace(
        trajectories[-1],
        nodes=(
            replace(
                trajectories[-1].nodes[0],
                receipt=replace(
                    trajectories[-1].nodes[0].receipt,
                    q=trajectories[-1].q + 1.0e-3,
                ),
            ),
            *trajectories[-1].nodes[1:],
        ),
    )
    with pytest.raises(ValueError, match="q/mass metadata"):
        aggregate_squeezed_flrw_stress_ensemble(
            (*trajectories[:-1], inconsistent_q),
            node_certificates=certificates,
        )
    bad_first_certificate = replace(
        certificates[0],
        energy_ir=CertifiedInfraredPowerLaw(0.0, 0.0, q_values[0]),
    )
    with pytest.raises(ValueError, match="first energy sample"):
        aggregate_squeezed_flrw_stress_ensemble(
            trajectories,
            node_certificates=(bad_first_certificate, *certificates[1:]),
        )
    with pytest.raises(ValueError, match="do not match the Gaussian profile"):
        aggregate_squeezed_flrw_stress_ensemble(
            trajectories,
            node_certificates=certificates,
            bogoliubov_profile=GaussianBogoliubovProfile(
                amplitude=0.16,
                q_scale=1.0 / math.sqrt(8.0),
                beta_phase=math.pi / 2.0,
            ),
        )


def test_squeezed_flrw_ensemble_q_quadrature_converges() -> None:
    def q_grid(intervals: int) -> tuple[float, ...]:
        return tuple(0.05 + 1.15 * index / intervals for index in range(intervals + 1))

    coarse = _squeezed_ensemble(q_grid(4), steps=900)
    medium = _squeezed_ensemble(q_grid(8), steps=900)
    fine = _squeezed_ensemble(q_grid(16), steps=900)

    def endpoint_error(left, right) -> float:
        return math.hypot(
            left.nodes[-1].created_stress.energy_density_over_h0_four
            - right.nodes[-1].created_stress.energy_density_over_h0_four,
            left.nodes[-1].created_stress.pressure_over_h0_four
            - right.nodes[-1].created_stress.pressure_over_h0_four,
        )

    assert endpoint_error(medium, fine) < endpoint_error(coarse, fine) / 3.0
    assert abs(
        medium.late_window.particle_grid_equation_of_state
        - fine.late_window.particle_grid_equation_of_state
    ) < abs(
        coarse.late_window.particle_grid_equation_of_state
        - fine.late_window.particle_grid_equation_of_state
    ) / 3.0


def test_squeezed_flrw_ensemble_time_ward_converges() -> None:
    q_values = (0.05, 0.3375, 0.625, 0.9125, 1.2)
    coarse = _squeezed_ensemble(q_values, steps=600)
    medium = _squeezed_ensemble(q_values, steps=1200)
    fine = _squeezed_ensemble(q_values, steps=2400)
    coarse_projection = _frozen_constraint_projection(
        coarse,
        maximum_state_difference_ward_relative_residual=2.1,
    )
    medium_projection = _frozen_constraint_projection(
        medium,
        maximum_state_difference_ward_relative_residual=2.1,
    )
    fine_projection = _frozen_constraint_projection(
        fine,
        maximum_state_difference_ward_relative_residual=2.1,
    )

    assert (
        medium.ward.central_grid.relative_absolute_accumulated_residual
        < coarse.ward.central_grid.relative_absolute_accumulated_residual / 3.0
    )
    assert (
        fine.ward.central_grid.max_finite_difference_relative_residual
        < medium.ward.central_grid.max_finite_difference_relative_residual / 3.0
    )
    assert (
        medium_projection.maximum_state_difference_ward_relative_residual
        < coarse_projection.maximum_state_difference_ward_relative_residual / 3.0
    )
    assert (
        fine_projection.maximum_state_difference_ward_relative_residual
        < medium_projection.maximum_state_difference_ward_relative_residual / 3.0
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
