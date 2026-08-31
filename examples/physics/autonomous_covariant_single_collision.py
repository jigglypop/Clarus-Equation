"""Autonomous covariant one-collision bridge with an explicit clock cost.

The model in this module is deliberately narrower than a continuous domino.
It asks whether one E12-style local coin can arise from a coordinate-free
matter action once a dynamical clock, a carrier, a detector, and a battery are
all kept in the same energy ledger.

In four spacetime dimensions and natural units, take one real clock ``T`` and
four complex scalars ``H_A, H_D, d, b``.  The two carrier fields have equal
mass, as do the detector and battery modes.  Besides canonical kinetic and
mass terms, the potential contains

    V_stab = lambda / 4 * S**2,
    S = |H_A|**2 + |H_D|**2 + |d|**2 + |b|**2,

    V_P = mu_P**2 r(T/M_T) (H_A* H_D + c.c.),
    V_R = f(T/M_T) (g H_A* H_D d* b + c.c.).

``r`` and ``f`` are smooth compact bumps with disjoint support in the value of
the dynamical scalar clock.  There is no explicit coordinate-time switching.
The first pulse rotates ACTIVE to DEAD; the second pulse mixes

    |F> = |H_D, d=0, b=1>,     |S> = |H_A, d=1, b=0>.

Writing ``Phi=(H_A,H_D,d,b)``, the proposed scalar-density action is

    S = integral sqrt(-g) [M_Pl**2 R / 2 - (grad T)**2 / 2
        - sum_i (|grad Phi_i|**2 + m_i**2 |Phi_i|**2)
        - V_stab - V_P - V_R].

It is local and diffeomorphism invariant by construction.  Its canonical
matter principal symbol is normally hyperbolic on a supplied globally
hyperbolic background, and the interaction is lower order.  The code audits
the potential chain rule, including the non-zero clock source and allocation
ambiguity.  It does not perform the full metric variation or certify the
on-shell Hilbert-stress Ward theorem; that remains a separate formal proof
statement and is deliberately separate from this numerical receipt.

The E12 probability ``p = sin(theta)**2`` appears only after additional,
explicit assumptions: a narrow semiclassical clock trajectory, a hard-core
single-mode projection, rotating-wave/resonance selection, and negligible
leakage.  The projected prep rate is ``mu_P**2 / (2 omega_H)`` and the
projected exchange rate is ``|g|`` times a supplied mode-overlap rate, so both
pulse areas are dimensionless.  The resulting finite-mode reduced channel is
CPTP and has an energy-resolved battery receipt.

This construction also contains its own no-trigger counterexample.  The
action conserves total carrier number, hence ``H_A = H_D = 0`` is an invariant
sector.  A dynamical action does not create a head for free: the initial clock
state and a carrier wavepacket are still supplied.  Repeated cells, a durable
pointer, a continuum QFT instrument, operational no-signalling, and a GR
source matching theorem are not derived.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8
QUADRATURE_ORDER = 160
RANDOM_STABILITY_SAMPLES = 512


@dataclass(frozen=True)
class ProjectedBatteryOutcomeReceipt:
    """One environment outcome of the projected detector channel."""

    environment_label: str
    probability: float
    final_battery_energy: float
    energy_paid_to_detector: float
    conditional_detector_energy: float | None
    relative_branch_energy_residual: float | None


@dataclass(frozen=True)
class AutonomousCovariantCollisionCertificate:
    """Audit receipt for the action and its conditional one-mode projection."""

    head_mass: float
    head_frequency: float
    detector_battery_gap: float
    clock_scale: float
    clock_rate: float
    clock_energy_density: float
    exchange_coupling: float
    quartic_coupling: float
    mode_overlap_rate: float
    trigger_momentum: float
    trigger_group_velocity: float
    prep_clock_support: tuple[float, float]
    coin_clock_support: tuple[float, float]
    prep_proper_time_support: tuple[float, float]
    coin_proper_time_support: tuple[float, float]
    prep_bump_time_integral: float
    coin_bump_time_integral: float
    prep_target_angle: float
    prep_mass_squared: float
    projected_prep_rate: float
    prep_angle: float
    exchange_angle: float
    trigger_probability: float
    quartic_analytic_lower_bound_coefficient: float
    extremal_quartic_potential: float
    minimum_sampled_quartic_potential: float
    minimum_head_mass_squared_eigenvalue: float
    potential_reality_residual: float
    head_phase_symmetry_residual: float
    vacuum_head_force_residual: float
    maximum_relative_chain_rule_residual: float
    potential_gradient_bookkeeping_residual: float
    maximum_allocation_total_current_residual: float
    allocation_current_difference: float
    clock_backreaction_source_norm: float
    projected_unitary_residual: float
    relative_projected_energy_commutator_residual: float
    projected_kraus_completeness_residual: float
    projected_minimum_choi_eigenvalue: float
    projected_output_trace_residual: float
    projected_minimum_output_eigenvalue: float
    projected_detector_activation_probability: float
    one_cell_activation_formula_residual: float
    projected_standard_limit_superoperator_residual: float
    projected_one_cell_channel_residual: float
    projected_initial_detector_energy: float
    projected_final_detector_energy: float
    projected_initial_battery_energy: float
    projected_final_battery_energy: float
    projected_total_energy_balance_residual: float
    projected_expected_battery_energy_paid: float
    projected_reverse_transfer_identity_residual: float
    projected_maximum_branch_energy_residual: float
    projected_battery_outcomes: tuple[ProjectedBatteryOutcomeReceipt, ...]
    dimensionless_core_arguments: tuple[tuple[str, str], ...]
    action_terms_have_mass_dimension_four: bool
    compact_smooth_clock_bumps: bool
    clock_windows_disjoint_and_ordered: bool
    potential_hermitian: bool
    analytic_stability_bound_pass: bool
    sampled_stability_bound_pass: bool
    head_mass_matrix_positive: bool
    head_number_conserved: bool
    vacuum_head_sector_invariant: bool
    spontaneous_trigger_from_vacuum_derived: bool
    explicit_coordinate_switching_present: bool
    dynamic_clock_backreaction_retained: bool
    diffeomorphism_invariant_action_by_construction: bool
    potential_gradient_bookkeeping_within_tolerance: bool
    metric_variation_machine_verified: bool
    unique_sector_exchange_current_derived: bool
    canonical_matter_principal_symbol: bool
    fixed_background_causal_domain_of_dependence: bool
    interacting_qft_microcausality_derived: bool
    operational_no_signalling_instrument_derived: bool
    einstein_hilbert_term_present: bool
    pure_einstein_limit_when_matter_vanishes: bool
    coupled_einstein_hyperbolicity_derived: bool
    projected_single_mode_assumptions_declared: bool
    projected_channel_cptp_within_tolerance: bool
    projected_energy_receipt_within_tolerance: bool
    projected_one_cell_e12_channel_match: bool
    continuum_action_cptp_instrument_derived: bool
    exact_full_qft_to_projected_mode_limit_derived: bool
    full_e12_domino_equivalence_derived: bool
    initial_clock_state_derived: bool
    initial_trigger_wavepacket_derived: bool
    durable_detector_pointer_derived: bool
    gr_source_matching_derived: bool
    cross_dataset_parameter_fixing_derived: bool
    independent_holdout_prediction_derived: bool


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def smooth_compact_bump(
    clock_value_ratio: float,
    *,
    center: float,
    half_width: float,
) -> float:
    """Return a C-infinity compact bump of unit peak in clock-value space."""

    if not all(math.isfinite(value) for value in (clock_value_ratio, center)):
        raise ValueError("clock_value_ratio and center must be finite")
    width = _positive(half_width, "half_width")
    coordinate = (clock_value_ratio - center) / width
    if abs(coordinate) >= 1.0:
        return 0.0
    return math.exp(1.0 - 1.0 / (1.0 - coordinate * coordinate))


def smooth_compact_bump_derivative(
    clock_value_ratio: float,
    *,
    center: float,
    half_width: float,
) -> float:
    """Derivative of :func:`smooth_compact_bump` by its first argument."""

    width = _positive(half_width, "half_width")
    coordinate = (clock_value_ratio - center) / width
    if abs(coordinate) >= 1.0:
        return 0.0
    bump = smooth_compact_bump(
        clock_value_ratio,
        center=center,
        half_width=width,
    )
    return (
        bump
        * (-2.0 * coordinate)
        / (width * (1.0 - coordinate * coordinate) ** 2)
    )


def _bump_integral_in_clock_ratio(center: float, half_width: float) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    values = np.array(
        [
            smooth_compact_bump(
                center + half_width * float(node),
                center=center,
                half_width=half_width,
            )
            for node in nodes
        ],
        dtype=np.float64,
    )
    return half_width * float(np.dot(weights, values))


def _interaction_potential_and_derivatives(
    *,
    fields: np.ndarray,
    clock: float,
    clock_scale: float,
    prep_center: float,
    prep_half_width: float,
    coin_center: float,
    coin_half_width: float,
    prep_mass_squared: float,
    exchange_coupling: complex,
    quartic_coupling: float,
) -> tuple[float, np.ndarray, float]:
    """Return V, its Wirtinger derivatives dV/d(phi*), and dV/dT."""

    active, dead, detector, battery = fields
    clock_ratio = clock / clock_scale
    prep_bump = smooth_compact_bump(
        clock_ratio,
        center=prep_center,
        half_width=prep_half_width,
    )
    prep_derivative = smooth_compact_bump_derivative(
        clock_ratio,
        center=prep_center,
        half_width=prep_half_width,
    )
    coin_bump = smooth_compact_bump(
        clock_ratio,
        center=coin_center,
        half_width=coin_half_width,
    )
    coin_derivative = smooth_compact_bump_derivative(
        clock_ratio,
        center=coin_center,
        half_width=coin_half_width,
    )

    squared_radius = float(np.sum(np.abs(fields) ** 2))
    prep_monomial = np.conj(active) * dead
    exchange_monomial = (
        exchange_coupling
        * np.conj(active)
        * dead
        * np.conj(detector)
        * battery
    )
    stabilizer = 0.25 * quartic_coupling * squared_radius**2
    prep_potential = 2.0 * prep_mass_squared * prep_bump * float(
        np.real(prep_monomial)
    )
    exchange_potential = 2.0 * coin_bump * float(
        np.real(exchange_monomial)
    )
    potential = stabilizer + prep_potential + exchange_potential

    radial_coefficient = 0.5 * quartic_coupling * squared_radius
    derivatives = radial_coefficient * fields.astype(np.complex128)
    derivatives[0] += (
        prep_mass_squared * prep_bump * dead
        + coin_bump
        * exchange_coupling
        * dead
        * np.conj(detector)
        * battery
    )
    derivatives[1] += (
        prep_mass_squared * prep_bump * active
        + coin_bump
        * np.conj(exchange_coupling)
        * active
        * detector
        * np.conj(battery)
    )
    derivatives[2] += (
        coin_bump
        * exchange_coupling
        * np.conj(active)
        * dead
        * battery
    )
    derivatives[3] += (
        coin_bump
        * np.conj(exchange_coupling)
        * active
        * np.conj(dead)
        * detector
    )

    derivative_clock = (
        2.0
        * prep_mass_squared
        * prep_derivative
        * float(np.real(prep_monomial))
        + 2.0
        * coin_derivative
        * float(np.real(exchange_monomial))
    ) / clock_scale
    return potential, derivatives, derivative_clock


def _projected_unitary(
    prep_angle: float,
    exchange_angle: float,
) -> np.ndarray:
    """Return the hard-core one-head unitary R P on H x d x b."""

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    prep_head = (
        math.cos(prep_angle) * np.eye(2, dtype=np.complex128)
        - 1.0j * math.sin(prep_angle) * pauli_x
    )
    prep = np.kron(prep_head, np.eye(4, dtype=np.complex128))

    rotation = np.eye(8, dtype=np.complex128)
    failure = 1 * 4 + 0 * 2 + 1  # H_D, detector 0, battery 1
    success = 0 * 4 + 1 * 2 + 0  # H_A, detector 1, battery 0
    cosine = math.cos(exchange_angle)
    sine = math.sin(exchange_angle)
    rotation[failure, failure] = cosine
    rotation[success, failure] = -1.0j * sine
    rotation[failure, success] = -1.0j * sine
    rotation[success, success] = cosine
    return rotation @ prep


def _projected_kraus(unitary: np.ndarray) -> tuple[tuple[str, np.ndarray, int], ...]:
    """Trace final carrier and battery from initial H_A and excited battery."""

    operators: list[tuple[str, np.ndarray, int]] = []
    for final_head in (0, 1):
        for final_battery in (0, 1):
            operator = np.zeros((2, 2), dtype=np.complex128)
            for output_detector in (0, 1):
                output_index = final_head * 4 + output_detector * 2 + final_battery
                for input_detector in (0, 1):
                    input_index = 0 * 4 + input_detector * 2 + 1
                    operator[output_detector, input_detector] = unitary[
                        output_index, input_index
                    ]
            operators.append(
                (
                    f"head={'A' if final_head == 0 else 'D'};b={final_battery}",
                    operator,
                    final_battery,
                )
            )
    return tuple(operators)


def _apply_channel(kraus: tuple[np.ndarray, ...], density: np.ndarray) -> np.ndarray:
    return sum(
        (operator @ density @ operator.conj().T for operator in kraus),
        start=np.zeros_like(density, dtype=np.complex128),
    )


def _superoperator(kraus: tuple[np.ndarray, ...]) -> np.ndarray:
    return sum(
        (np.kron(operator, operator.conj()) for operator in kraus),
        start=np.zeros((4, 4), dtype=np.complex128),
    )


def _analytic_raising_superoperator(exchange_angle: float) -> np.ndarray:
    cosine = math.cos(exchange_angle)
    sine = math.sin(exchange_angle)
    stay = np.diag([cosine, 1.0]).astype(np.complex128)
    raise_operator = np.array([[0.0, 0.0], [sine, 0.0]], dtype=np.complex128)
    return _superoperator((stay, raise_operator))


def certify_autonomous_covariant_single_collision(
    *,
    head_mass: float,
    detector_battery_gap: float,
    clock_scale: float,
    clock_rate: float,
    exchange_coupling: float,
    quartic_coupling: float,
    mode_overlap_rate: float,
    trigger_momentum: float,
    prep_center: float = -1.0,
    prep_half_width: float = 0.45,
    coin_center: float = 1.0,
    coin_half_width: float = 0.45,
    prep_target_angle: float = 0.5 * math.pi,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AutonomousCovariantCollisionCertificate:
    """Audit one coordinate-free action candidate and its projected coin.

    Dimensions in natural units are part of the contract: ``head_mass``,
    ``detector_battery_gap``, ``clock_scale``, ``mode_overlap_rate``, and
    ``trigger_momentum`` have mass dimension one; ``clock_rate=dT/dtau`` and
    the constructed ``prep_mass_squared`` have mass dimension two; ``g`` and
    ``lambda`` are dimensionless.  The compact-bump arguments and both pulse
    areas are dimensionless.
    """

    head_mass = _positive(head_mass, "head_mass")
    detector_battery_gap = _positive(
        detector_battery_gap, "detector_battery_gap"
    )
    clock_scale = _positive(clock_scale, "clock_scale")
    clock_rate = _positive(clock_rate, "clock_rate")
    mode_overlap_rate = _positive(mode_overlap_rate, "mode_overlap_rate")
    trigger_momentum = _nonnegative(trigger_momentum, "trigger_momentum")
    prep_half_width = _positive(prep_half_width, "prep_half_width")
    coin_half_width = _positive(coin_half_width, "coin_half_width")
    tolerance = _positive(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    for value, name in (
        (exchange_coupling, "exchange_coupling"),
        (quartic_coupling, "quartic_coupling"),
        (prep_center, "prep_center"),
        (coin_center, "coin_center"),
        (prep_target_angle, "prep_target_angle"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if exchange_coupling == 0.0:
        raise ValueError("exchange_coupling must be non-zero for this witness")
    if quartic_coupling < 0.0:
        raise ValueError("quartic_coupling must be non-negative")
    if prep_target_angle <= 0.0:
        raise ValueError("prep_target_angle must be positive")

    prep_clock_support = (
        prep_center - prep_half_width,
        prep_center + prep_half_width,
    )
    coin_clock_support = (
        coin_center - coin_half_width,
        coin_center + coin_half_width,
    )
    if prep_clock_support[1] >= coin_clock_support[0]:
        raise ValueError("prep and coin clock windows must be disjoint and ordered")

    clock_to_time = clock_scale / clock_rate
    prep_proper_time_support = tuple(
        clock_to_time * value for value in prep_clock_support
    )
    coin_proper_time_support = tuple(
        clock_to_time * value for value in coin_clock_support
    )
    prep_bump_time_integral = clock_to_time * _bump_integral_in_clock_ratio(
        prep_center, prep_half_width
    )
    coin_bump_time_integral = clock_to_time * _bump_integral_in_clock_ratio(
        coin_center, coin_half_width
    )

    head_frequency = math.sqrt(head_mass**2 + trigger_momentum**2)
    prep_mass_squared = (
        prep_target_angle
        * 2.0
        * head_frequency
        / prep_bump_time_integral
    )
    projected_prep_rate = prep_mass_squared / (2.0 * head_frequency)
    prep_angle = projected_prep_rate * prep_bump_time_integral
    projected_exchange_rate = abs(exchange_coupling) * mode_overlap_rate
    exchange_angle = projected_exchange_rate * coin_bump_time_integral
    trigger_probability = math.sin(exchange_angle) ** 2

    quartic_lower_bound = (
        2.0 * quartic_coupling - abs(exchange_coupling)
    ) / 8.0
    if quartic_lower_bound < 0.0:
        raise ValueError(
            "quartic stability requires quartic_coupling >= abs(g) / 2"
        )
    minimum_head_mass_squared_eigenvalue = head_mass**2 - abs(
        prep_mass_squared
    )
    if minimum_head_mass_squared_eigenvalue <= 0.0:
        raise ValueError(
            "head mass matrix requires head_mass**2 > abs(prep_mass_squared)"
        )

    rng = np.random.default_rng(1301)
    minimum_sampled_quartic = math.inf
    potential_reality_residual = 0.0
    for _ in range(RANDOM_STABILITY_SAMPLES):
        direction = rng.normal(size=4) + 1.0j * rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        active, dead, detector, battery = direction
        monomial = (
            exchange_coupling
            * np.conj(active)
            * dead
            * np.conj(detector)
            * battery
        )
        for bump_sign in (-1.0, 1.0):
            value = 0.25 * quartic_coupling + 2.0 * bump_sign * float(
                np.real(monomial)
            )
            minimum_sampled_quartic = min(minimum_sampled_quartic, value)
            potential_reality_residual = max(
                potential_reality_residual,
                abs(float(np.imag(value))),
            )

    coupling_phase = complex(exchange_coupling) / abs(exchange_coupling)
    extremal_direction = np.array(
        [0.5, 0.5, 0.5, -0.5 * np.conj(coupling_phase)],
        dtype=np.complex128,
    )
    extremal_active, extremal_dead, extremal_detector, extremal_battery = (
        extremal_direction
    )
    extremal_monomial = (
        exchange_coupling
        * np.conj(extremal_active)
        * extremal_dead
        * np.conj(extremal_detector)
        * extremal_battery
    )
    extremal_quartic_potential = (
        0.25 * quartic_coupling
        + 2.0 * float(np.real(extremal_monomial))
    )

    sample_fields = np.array(
        [0.7 + 0.2j, -0.3 + 0.5j, 0.4 - 0.1j, 0.6 + 0.3j],
        dtype=np.complex128,
    )
    sample_gradients = np.array(
        [
            [0.12 + 0.04j, -0.03 + 0.01j, 0.02 - 0.05j, 0.01 + 0.02j],
            [-0.05 + 0.02j, 0.06 + 0.03j, -0.01 + 0.01j, 0.04 - 0.02j],
            [0.03 - 0.04j, 0.02 + 0.02j, 0.05 + 0.01j, -0.02 + 0.03j],
            [0.07 + 0.01j, -0.04 - 0.02j, 0.01 + 0.03j, 0.02 + 0.01j],
        ],
        dtype=np.complex128,
    )
    clock_gradient = np.array([clock_rate, 0.0, 0.0, 0.0], dtype=np.float64)
    chain_rule_residuals: list[float] = []
    bookkeeping_residuals: list[float] = []
    allocation_total_residuals: list[float] = []
    allocation_differences: list[float] = []
    clock_source_norms: list[float] = []
    for center, half_width, sample_offset in (
        (prep_center, prep_half_width, 0.37),
        (coin_center, coin_half_width, -0.41),
    ):
        clock_ratio = center + sample_offset * half_width
        sample_clock = clock_ratio * clock_scale
        _, derivatives, derivative_clock = _interaction_potential_and_derivatives(
            fields=sample_fields,
            clock=sample_clock,
            clock_scale=clock_scale,
            prep_center=prep_center,
            prep_half_width=prep_half_width,
            coin_center=coin_center,
            coin_half_width=coin_half_width,
            prep_mass_squared=prep_mass_squared,
            exchange_coupling=complex(exchange_coupling),
            quartic_coupling=quartic_coupling,
        )
        field_currents = np.array(
            [
                2.0
                * np.real(
                    derivative * np.conj(sample_gradients[index, :])
                )
                for index, derivative in enumerate(derivatives)
            ],
            dtype=np.float64,
        )
        clock_current = derivative_clock * clock_gradient
        free_sector_currents = np.vstack((field_currents, clock_current))
        analytic_gradient = np.sum(free_sector_currents, axis=0)
        interaction_stress_divergence = -analytic_gradient
        bookkeeping_residuals.append(
            float(
                np.linalg.norm(
                    analytic_gradient + interaction_stress_divergence
                )
            )
        )
        clock_source_norms.append(float(np.linalg.norm(clock_current)))

        alpha_first = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        alpha_second = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        allocated_first = free_sector_currents - np.outer(
            alpha_first, analytic_gradient
        )
        allocated_second = free_sector_currents - np.outer(
            alpha_second, analytic_gradient
        )
        allocation_total_residuals.extend(
            (
                float(np.linalg.norm(np.sum(allocated_first, axis=0))),
                float(np.linalg.norm(np.sum(allocated_second, axis=0))),
            )
        )
        allocation_differences.append(
            float(np.linalg.norm(allocated_first - allocated_second))
        )

        epsilon = 1.0e-6
        finite_difference = np.zeros(4, dtype=np.float64)
        for spacetime_index in range(4):
            plus, _, _ = _interaction_potential_and_derivatives(
                fields=(
                    sample_fields
                    + epsilon * sample_gradients[:, spacetime_index]
                ),
                clock=(
                    sample_clock
                    + epsilon * clock_gradient[spacetime_index]
                ),
                clock_scale=clock_scale,
                prep_center=prep_center,
                prep_half_width=prep_half_width,
                coin_center=coin_center,
                coin_half_width=coin_half_width,
                prep_mass_squared=prep_mass_squared,
                exchange_coupling=complex(exchange_coupling),
                quartic_coupling=quartic_coupling,
            )
            minus, _, _ = _interaction_potential_and_derivatives(
                fields=(
                    sample_fields
                    - epsilon * sample_gradients[:, spacetime_index]
                ),
                clock=(
                    sample_clock
                    - epsilon * clock_gradient[spacetime_index]
                ),
                clock_scale=clock_scale,
                prep_center=prep_center,
                prep_half_width=prep_half_width,
                coin_center=coin_center,
                coin_half_width=coin_half_width,
                prep_mass_squared=prep_mass_squared,
                exchange_coupling=complex(exchange_coupling),
                quartic_coupling=quartic_coupling,
            )
            finite_difference[spacetime_index] = (
                plus - minus
            ) / (2.0 * epsilon)
        chain_rule_residuals.append(
            float(
                np.linalg.norm(finite_difference - analytic_gradient)
                / max(1.0, np.linalg.norm(analytic_gradient))
            )
        )

    phase = 0.731
    phase_rotated = sample_fields.copy()
    phase_rotated[0:2] *= np.exp(1.0j * phase)
    phase_sample_clock = coin_center * clock_scale
    original_potential, _, _ = _interaction_potential_and_derivatives(
        fields=sample_fields,
        clock=phase_sample_clock,
        clock_scale=clock_scale,
        prep_center=prep_center,
        prep_half_width=prep_half_width,
        coin_center=coin_center,
        coin_half_width=coin_half_width,
        prep_mass_squared=prep_mass_squared,
        exchange_coupling=complex(exchange_coupling),
        quartic_coupling=quartic_coupling,
    )
    rotated_potential, _, _ = _interaction_potential_and_derivatives(
        fields=phase_rotated,
        clock=phase_sample_clock,
        clock_scale=clock_scale,
        prep_center=prep_center,
        prep_half_width=prep_half_width,
        coin_center=coin_center,
        coin_half_width=coin_half_width,
        prep_mass_squared=prep_mass_squared,
        exchange_coupling=complex(exchange_coupling),
        quartic_coupling=quartic_coupling,
    )
    head_phase_symmetry_residual = abs(rotated_potential - original_potential)

    vacuum_fields = sample_fields.copy()
    vacuum_fields[0:2] = 0.0
    _, vacuum_derivatives, _ = _interaction_potential_and_derivatives(
        fields=vacuum_fields,
        clock=phase_sample_clock,
        clock_scale=clock_scale,
        prep_center=prep_center,
        prep_half_width=prep_half_width,
        coin_center=coin_center,
        coin_half_width=coin_half_width,
        prep_mass_squared=prep_mass_squared,
        exchange_coupling=complex(exchange_coupling),
        quartic_coupling=quartic_coupling,
    )
    vacuum_head_force_residual = float(np.linalg.norm(vacuum_derivatives[0:2]))

    projected_unitary = _projected_unitary(prep_angle, exchange_angle)
    identity_projected = np.eye(8, dtype=np.complex128)
    projected_unitary_residual = float(
        np.linalg.norm(
            projected_unitary.conj().T @ projected_unitary
            - identity_projected,
            ord="fro",
        )
    )
    projected_energy_diagonal = np.array(
        [
            head_frequency
            + detector * detector_battery_gap
            + battery * detector_battery_gap
            for head in (0, 1)
            for detector in (0, 1)
            for battery in (0, 1)
        ],
        dtype=np.float64,
    )
    projected_hamiltonian = np.diag(projected_energy_diagonal).astype(
        np.complex128
    )
    projected_energy_scale = max(
        float(np.linalg.norm(projected_hamiltonian, ord="fro")),
        detector_battery_gap,
    )
    relative_projected_energy_commutator_residual = float(
        np.linalg.norm(
            projected_unitary @ projected_hamiltonian
            - projected_hamiltonian @ projected_unitary,
            ord="fro",
        )
        / projected_energy_scale
    )

    labelled_kraus = _projected_kraus(projected_unitary)
    kraus = tuple(operator for _, operator, _ in labelled_kraus)
    detector_identity = np.eye(2, dtype=np.complex128)
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        start=np.zeros((2, 2), dtype=np.complex128),
    )
    projected_kraus_completeness_residual = float(
        np.linalg.norm(completeness - detector_identity, ord="fro")
    )
    choi = sum(
        (
            np.outer(
                operator.reshape(-1, order="F"),
                operator.reshape(-1, order="F").conj(),
            )
            for operator in kraus
        ),
        start=np.zeros((4, 4), dtype=np.complex128),
    )
    projected_minimum_choi_eigenvalue = float(np.min(np.linalg.eigvalsh(choi)))

    ground = np.array([1.0, 0.0], dtype=np.complex128)
    seed_density = np.outer(ground, ground.conj())
    projected_output = _apply_channel(kraus, seed_density)
    projected_output_trace_residual = abs(float(np.trace(projected_output).real) - 1.0)
    projected_minimum_output_eigenvalue = float(
        np.min(np.linalg.eigvalsh(projected_output))
    )
    detector_hamiltonian = np.diag([0.0, detector_battery_gap]).astype(
        np.complex128
    )
    projected_detector_activation_probability = float(
        np.real(projected_output[1, 1])
    )
    one_cell_activation_formula_residual = abs(
        projected_detector_activation_probability - trigger_probability
    )
    projected_one_cell_channel_residual = float(
        np.linalg.norm(
            _superoperator(kraus)
            - _analytic_raising_superoperator(exchange_angle),
            ord="fro",
        )
    )

    zero_unitary = _projected_unitary(prep_angle, 0.0)
    zero_kraus = tuple(
        operator for _, operator, _ in _projected_kraus(zero_unitary)
    )
    projected_standard_limit_superoperator_residual = float(
        np.linalg.norm(
            _superoperator(zero_kraus) - np.eye(4, dtype=np.complex128),
            ord="fro",
        )
    )

    projected_initial_detector_energy = 0.0
    projected_final_detector_energy = float(
        np.trace(detector_hamiltonian @ projected_output).real
    )
    projected_initial_battery_energy = detector_battery_gap
    projected_final_battery_energy = 0.0
    projected_expected_battery_energy_paid = 0.0
    reverse_transfer = np.zeros((2, 2), dtype=np.complex128)
    paid_distribution_total = 0.0
    branch_residuals: list[float] = []
    receipts: list[ProjectedBatteryOutcomeReceipt] = []
    for label, operator, final_battery in labelled_kraus:
        final_battery_energy = final_battery * detector_battery_gap
        paid_energy = projected_initial_battery_energy - final_battery_energy
        branch_state = operator @ ground
        probability = float(np.vdot(branch_state, branch_state).real)
        projected_final_battery_energy += probability * final_battery_energy
        projected_expected_battery_energy_paid += probability * paid_energy
        paid_distribution_total += probability
        reverse_transfer += (
            operator.conj().T
            @ (detector_hamiltonian - paid_energy * detector_identity)
            @ operator
        )
        conditional_energy: float | None = None
        branch_residual: float | None = None
        if probability > tolerance:
            conditional_energy = float(
                np.vdot(branch_state, detector_hamiltonian @ branch_state).real
                / probability
            )
            branch_residual = abs(
                conditional_energy
                - projected_initial_detector_energy
                - paid_energy
            ) / detector_battery_gap
            branch_residuals.append(branch_residual)
        receipts.append(
            ProjectedBatteryOutcomeReceipt(
                environment_label=label,
                probability=probability,
                final_battery_energy=final_battery_energy,
                energy_paid_to_detector=paid_energy,
                conditional_detector_energy=conditional_energy,
                relative_branch_energy_residual=branch_residual,
            )
        )
    projected_total_energy_balance_residual = abs(
        projected_final_detector_energy
        + projected_final_battery_energy
        - projected_initial_detector_energy
        - projected_initial_battery_energy
    ) / detector_battery_gap
    projected_reverse_transfer_identity_residual = float(
        np.linalg.norm(reverse_transfer - detector_hamiltonian, ord="fro")
        / detector_battery_gap
    )
    projected_maximum_branch_energy_residual = max(branch_residuals, default=0.0)

    trigger_group_velocity = trigger_momentum / head_frequency
    numerical_limit = tolerance * 8.0
    projected_channel_cptp = max(
        projected_unitary_residual,
        projected_kraus_completeness_residual,
        -projected_minimum_choi_eigenvalue,
        projected_output_trace_residual,
        -projected_minimum_output_eigenvalue,
        abs(paid_distribution_total - 1.0),
    ) <= numerical_limit
    projected_energy_receipt = max(
        relative_projected_energy_commutator_residual,
        projected_total_energy_balance_residual,
        projected_reverse_transfer_identity_residual,
        projected_maximum_branch_energy_residual,
    ) <= numerical_limit
    projected_one_cell_match = max(
        one_cell_activation_formula_residual,
        projected_one_cell_channel_residual,
        projected_standard_limit_superoperator_residual,
    ) <= numerical_limit

    return AutonomousCovariantCollisionCertificate(
        head_mass=head_mass,
        head_frequency=head_frequency,
        detector_battery_gap=detector_battery_gap,
        clock_scale=clock_scale,
        clock_rate=clock_rate,
        clock_energy_density=0.5 * clock_rate**2,
        exchange_coupling=exchange_coupling,
        quartic_coupling=quartic_coupling,
        mode_overlap_rate=mode_overlap_rate,
        trigger_momentum=trigger_momentum,
        trigger_group_velocity=trigger_group_velocity,
        prep_clock_support=prep_clock_support,
        coin_clock_support=coin_clock_support,
        prep_proper_time_support=prep_proper_time_support,
        coin_proper_time_support=coin_proper_time_support,
        prep_bump_time_integral=prep_bump_time_integral,
        coin_bump_time_integral=coin_bump_time_integral,
        prep_target_angle=prep_target_angle,
        prep_mass_squared=prep_mass_squared,
        projected_prep_rate=projected_prep_rate,
        prep_angle=prep_angle,
        exchange_angle=exchange_angle,
        trigger_probability=trigger_probability,
        quartic_analytic_lower_bound_coefficient=quartic_lower_bound,
        extremal_quartic_potential=extremal_quartic_potential,
        minimum_sampled_quartic_potential=minimum_sampled_quartic,
        minimum_head_mass_squared_eigenvalue=(
            minimum_head_mass_squared_eigenvalue
        ),
        potential_reality_residual=potential_reality_residual,
        head_phase_symmetry_residual=head_phase_symmetry_residual,
        vacuum_head_force_residual=vacuum_head_force_residual,
        maximum_relative_chain_rule_residual=max(chain_rule_residuals),
        potential_gradient_bookkeeping_residual=max(bookkeeping_residuals),
        maximum_allocation_total_current_residual=max(
            allocation_total_residuals
        ),
        allocation_current_difference=max(allocation_differences),
        clock_backreaction_source_norm=max(clock_source_norms),
        projected_unitary_residual=projected_unitary_residual,
        relative_projected_energy_commutator_residual=(
            relative_projected_energy_commutator_residual
        ),
        projected_kraus_completeness_residual=(
            projected_kraus_completeness_residual
        ),
        projected_minimum_choi_eigenvalue=projected_minimum_choi_eigenvalue,
        projected_output_trace_residual=projected_output_trace_residual,
        projected_minimum_output_eigenvalue=projected_minimum_output_eigenvalue,
        projected_detector_activation_probability=(
            projected_detector_activation_probability
        ),
        one_cell_activation_formula_residual=(
            one_cell_activation_formula_residual
        ),
        projected_standard_limit_superoperator_residual=(
            projected_standard_limit_superoperator_residual
        ),
        projected_one_cell_channel_residual=projected_one_cell_channel_residual,
        projected_initial_detector_energy=projected_initial_detector_energy,
        projected_final_detector_energy=projected_final_detector_energy,
        projected_initial_battery_energy=projected_initial_battery_energy,
        projected_final_battery_energy=projected_final_battery_energy,
        projected_total_energy_balance_residual=(
            projected_total_energy_balance_residual
        ),
        projected_expected_battery_energy_paid=(
            projected_expected_battery_energy_paid
        ),
        projected_reverse_transfer_identity_residual=(
            projected_reverse_transfer_identity_residual
        ),
        projected_maximum_branch_energy_residual=(
            projected_maximum_branch_energy_residual
        ),
        projected_battery_outcomes=tuple(receipts),
        dimensionless_core_arguments=(
            ("T / M_T", "dimensionless clock-bump argument"),
            ("mu_P^2 Delta tau / (2 omega_H)", "dimensionless prep area"),
            ("g_eff Delta tau", "dimensionless exchange area theta"),
            ("sin(theta)^2", "dimensionless probability"),
        ),
        action_terms_have_mass_dimension_four=True,
        compact_smooth_clock_bumps=True,
        clock_windows_disjoint_and_ordered=True,
        potential_hermitian=(potential_reality_residual <= tolerance),
        analytic_stability_bound_pass=(quartic_lower_bound >= 0.0),
        sampled_stability_bound_pass=(
            minimum_sampled_quartic + tolerance >= quartic_lower_bound
        ),
        head_mass_matrix_positive=(
            minimum_head_mass_squared_eigenvalue > 0.0
        ),
        head_number_conserved=(head_phase_symmetry_residual <= tolerance),
        vacuum_head_sector_invariant=(vacuum_head_force_residual <= tolerance),
        spontaneous_trigger_from_vacuum_derived=False,
        explicit_coordinate_switching_present=False,
        dynamic_clock_backreaction_retained=(
            max(clock_source_norms) > tolerance
        ),
        diffeomorphism_invariant_action_by_construction=True,
        potential_gradient_bookkeeping_within_tolerance=(
            max(bookkeeping_residuals) <= tolerance
            and max(chain_rule_residuals) <= 1.0e-8
        ),
        metric_variation_machine_verified=False,
        unique_sector_exchange_current_derived=False,
        canonical_matter_principal_symbol=True,
        fixed_background_causal_domain_of_dependence=True,
        interacting_qft_microcausality_derived=False,
        operational_no_signalling_instrument_derived=False,
        einstein_hilbert_term_present=True,
        pure_einstein_limit_when_matter_vanishes=True,
        coupled_einstein_hyperbolicity_derived=False,
        projected_single_mode_assumptions_declared=True,
        projected_channel_cptp_within_tolerance=projected_channel_cptp,
        projected_energy_receipt_within_tolerance=projected_energy_receipt,
        projected_one_cell_e12_channel_match=projected_one_cell_match,
        continuum_action_cptp_instrument_derived=False,
        exact_full_qft_to_projected_mode_limit_derived=False,
        full_e12_domino_equivalence_derived=False,
        initial_clock_state_derived=False,
        initial_trigger_wavepacket_derived=False,
        durable_detector_pointer_derived=False,
        gr_source_matching_derived=False,
        cross_dataset_parameter_fixing_derived=False,
        independent_holdout_prediction_derived=False,
    )
