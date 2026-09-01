"""Exact dimensionless FLRW Ward/backreaction closure for a dynamic clock.

The declared action is

    S = integral sqrt(-g) [
        Mpl^2 R/2 - (partial T)^2/2 - V(T)
        - sum_s sum_A ((partial phi_sA)^2
                       + (m_s(T)^2 + xi_s R) phi_sA^2)/2
    ].

Use theta=T/H0, N=log(a), E=H/H0, M=Mpl/H0, r=rho/H0^4,
pi=p/H0^4, v=V/H0^4, and Phi_s^2=<phi_sA^2>/H0^2 for one real
component.  With g_s identical components,

    j_s = g_s (d mu_s^2/d theta) Phi_s^2 / 2,
    r_s,N + 3(r_s+pi_s) = theta_N j_s,

while the canonical clock obeys

    E^2[theta_NN + (3+E_N/E)theta_N] + v_theta + sum_s j_s = 0.

The clock Ward residual is theta_N times this equation.  Consequently the
total Ward identity is algebraic, not a fitted numerical cancellation.  If
the Raychaudhuri equation also holds, the derivative of the Friedmann
constraint is exactly zero:

    C_N = 6 M^2 R_Raychaudhuri - W_total.

This module proves that conditional closure.  The scalar inputs must already
come from a common regulator/counterterm triplet; the module does not create
renormalized stress by ledger arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examples.physics.kinetic_dark_sector_adiabatic_stress import (
        SqueezedFLRWStressEnsemble,
    )


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ScalarBackreactionChannel:
    """One species with explicit single-field versus multiplet conventions.

    ``field_squared`` is for one of ``degeneracy`` identical fields, whereas
    ``energy_density``, ``pressure``, and ``energy_density_d_n`` are already
    degeneracy-summed multiplet totals.
    """

    degeneracy: int
    energy_density: float
    pressure: float
    energy_density_d_n: float
    field_squared: float
    mass_squared_d_theta: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.degeneracy, bool)
            or not isinstance(self.degeneracy, int)
            or self.degeneracy < 1
        ):
            raise ValueError("degeneracy must be a positive integer")
        for name, value in (
            ("energy_density", self.energy_density),
            ("pressure", self.pressure),
            ("energy_density_d_n", self.energy_density_d_n),
            ("field_squared", self.field_squared),
            ("mass_squared_d_theta", self.mass_squared_d_theta),
        ):
            _finite(name, value)

    @property
    def clock_force(self) -> float:
        return (
            0.5
            * self.degeneracy
            * self.mass_squared_d_theta
            * self.field_squared
        )

    def ward_residual(self, theta_d_n: float) -> float:
        theta_d_n = _finite("theta_d_n", theta_d_n)
        return (
            self.energy_density_d_n
            + 3.0 * (self.energy_density + self.pressure)
            - theta_d_n * self.clock_force
        )


@dataclass(frozen=True)
class ConservedFluid:
    energy_density: float
    pressure: float
    energy_density_d_n: float

    def __post_init__(self) -> None:
        for name, value in (
            ("energy_density", self.energy_density),
            ("pressure", self.pressure),
            ("energy_density_d_n", self.energy_density_d_n),
        ):
            _finite(name, value)

    @property
    def ward_residual(self) -> float:
        return self.energy_density_d_n + 3.0 * (
            self.energy_density + self.pressure
        )


@dataclass(frozen=True)
class BackreactionClosureReceipt:
    scalar_ward_residuals: tuple[float, ...]
    fluid_ward_residuals: tuple[float, ...]
    clock_equation_residual: float
    clock_ward_residual: float
    clock_ward_factorization_residual: float
    total_ward_residual: float
    friedmann_constraint_residual: float
    raychaudhuri_residual: float
    friedmann_constraint_derivative: float
    constraint_propagation_identity_residual: float
    clock_energy_density: float
    clock_pressure: float
    total_energy_density: float
    total_pressure: float
    total_clock_force: float
    status: str = "CONDITIONAL_DYNAMIC_CLOCK_FLRW_CLOSURE"


def backreaction_closure_receipt(
    *,
    e: float,
    d_log_e_d_n: float,
    reduced_planck_over_h0: float,
    theta_d_n: float,
    theta_d2_n: float,
    potential: float,
    potential_d_theta: float,
    scalar_channels: tuple[ScalarBackreactionChannel, ...],
    conserved_fluids: tuple[ConservedFluid, ...] = (),
) -> BackreactionClosureReceipt:
    """Audit scalar transfer, clock response, and constraint propagation."""

    e = _finite("e", e)
    d_log_e_d_n = _finite("d_log_e_d_n", d_log_e_d_n)
    reduced_planck_over_h0 = _finite(
        "reduced_planck_over_h0", reduced_planck_over_h0
    )
    theta_d_n = _finite("theta_d_n", theta_d_n)
    theta_d2_n = _finite("theta_d2_n", theta_d2_n)
    potential = _finite("potential", potential)
    potential_d_theta = _finite("potential_d_theta", potential_d_theta)
    if e <= 0.0:
        raise ValueError("e=H/H0 must be positive")
    if reduced_planck_over_h0 <= 0.0:
        raise ValueError("reduced_planck_over_h0 must be positive")
    if not isinstance(scalar_channels, tuple) or not isinstance(
        conserved_fluids, tuple
    ):
        raise ValueError("channels and fluids must be tuples")

    e_squared = e * e
    clock_kinetic = 0.5 * e_squared * theta_d_n * theta_d_n
    clock_energy_density = clock_kinetic + potential
    clock_pressure = clock_kinetic - potential
    total_clock_force = math.fsum(
        channel.clock_force for channel in scalar_channels
    )
    clock_equation_residual = (
        e_squared
        * (theta_d2_n + (3.0 + d_log_e_d_n) * theta_d_n)
        + potential_d_theta
        + total_clock_force
    )

    clock_energy_density_d_n = (
        e_squared * theta_d_n * theta_d2_n
        + e_squared * d_log_e_d_n * theta_d_n * theta_d_n
        + potential_d_theta * theta_d_n
    )
    clock_ward_residual = (
        clock_energy_density_d_n
        + 3.0 * (clock_energy_density + clock_pressure)
        + theta_d_n * total_clock_force
    )
    factored_clock_ward = theta_d_n * clock_equation_residual

    scalar_ward_residuals = tuple(
        channel.ward_residual(theta_d_n) for channel in scalar_channels
    )
    fluid_ward_residuals = tuple(
        fluid.ward_residual for fluid in conserved_fluids
    )
    total_ward_residual = math.fsum(
        (
            *scalar_ward_residuals,
            *fluid_ward_residuals,
            clock_ward_residual,
        )
    )

    total_energy_density = math.fsum(
        (
            clock_energy_density,
            *(channel.energy_density for channel in scalar_channels),
            *(fluid.energy_density for fluid in conserved_fluids),
        )
    )
    total_pressure = math.fsum(
        (
            clock_pressure,
            *(channel.pressure for channel in scalar_channels),
            *(fluid.pressure for fluid in conserved_fluids),
        )
    )
    planck_squared = reduced_planck_over_h0**2
    friedmann_constraint_residual = (
        3.0 * planck_squared * e_squared - total_energy_density
    )
    raychaudhuri_residual = (
        e_squared * d_log_e_d_n
        + (total_energy_density + total_pressure) / (2.0 * planck_squared)
    )
    friedmann_constraint_derivative = (
        6.0 * planck_squared * e_squared * d_log_e_d_n
        - (
            clock_energy_density_d_n
            + math.fsum(
                channel.energy_density_d_n for channel in scalar_channels
            )
            + math.fsum(
                fluid.energy_density_d_n for fluid in conserved_fluids
            )
        )
    )
    propagated_derivative = (
        6.0 * planck_squared * raychaudhuri_residual - total_ward_residual
    )

    return BackreactionClosureReceipt(
        scalar_ward_residuals=scalar_ward_residuals,
        fluid_ward_residuals=fluid_ward_residuals,
        clock_equation_residual=clock_equation_residual,
        clock_ward_residual=clock_ward_residual,
        clock_ward_factorization_residual=(
            clock_ward_residual - factored_clock_ward
        ),
        total_ward_residual=total_ward_residual,
        friedmann_constraint_residual=friedmann_constraint_residual,
        raychaudhuri_residual=raychaudhuri_residual,
        friedmann_constraint_derivative=friedmann_constraint_derivative,
        constraint_propagation_identity_residual=(
            friedmann_constraint_derivative - propagated_derivative
        ),
        clock_energy_density=clock_energy_density,
        clock_pressure=clock_pressure,
        total_energy_density=total_energy_density,
        total_pressure=total_pressure,
        total_clock_force=total_clock_force,
    )


@dataclass(frozen=True)
class ReferenceFLRWBaselineNode:
    """Explicit renormalized reference-plus-classical FLRW source at one N."""

    n: float
    e: float
    d_log_e_d_n: float
    energy_density: float
    pressure: float
    energy_density_d_n: float

    def __post_init__(self) -> None:
        for name, value in (
            ("n", self.n),
            ("e", self.e),
            ("d_log_e_d_n", self.d_log_e_d_n),
            ("energy_density", self.energy_density),
            ("pressure", self.pressure),
            ("energy_density_d_n", self.energy_density_d_n),
        ):
            _finite(name, value)
        if self.e <= 0.0:
            raise ValueError("reference baseline e=H/H0 must be positive")


@dataclass(frozen=True)
class FrozenFLRWConstraintProjectionNode:
    """One frozen-background Friedmann/Raychaudhuri constraint projection."""

    n: float
    background_e: float
    background_d_log_e_d_n: float
    state_difference_energy_density: float
    state_difference_pressure: float
    state_difference_energy_density_d_n: float
    state_difference_energy_external_ir_uv_bound: float
    state_difference_pressure_external_ir_uv_bound: float
    projected_e: float
    projected_e_squared: float
    projected_e_squared_interval: tuple[float, float]
    projected_d_log_e_d_n: float
    projected_d_log_e_d_n_interval: tuple[float, float]
    projected_acceleration_over_h0_squared: float
    projected_acceleration_over_h0_squared_interval: tuple[float, float]
    relative_e_squared_shift_upper: float
    baseline_ward_residual: float
    state_difference_ward_residual: float
    total_ward_residual: float
    closure: BackreactionClosureReceipt


@dataclass(frozen=True)
class FrozenFLRWConstraintProjection:
    r"""Reference-plus-delta algebraic projection, not an evolved geometry."""

    nodes: tuple[FrozenFLRWConstraintProjectionNode, ...]
    reduced_planck_over_h0: float
    degeneracy: int
    baseline_reference_sector_declaration: str
    maximum_relative_e_squared_shift_upper: float
    maximum_state_difference_ward_relative_residual: float
    maximum_baseline_friedmann_relative_residual: float
    maximum_baseline_raychaudhuri_relative_residual: float
    maximum_baseline_ward_relative_residual: float
    baseline_closure_absolute_tolerance: float
    state_difference_ward_absolute_tolerance: float
    adjacent_n_step_ratio: float
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    status: str = "REFERENCE_PLUS_DELTA_FROZEN_FLRW_CONSTRAINT_PROJECTION"
    stress_units: str = "H0^4"
    fixed_comoving_q_measure_applied_once: bool = True
    degeneracy_applied_once_after_q_integration: bool = True
    initial_occupation_already_in_state_difference: bool = True
    reference_renormalized_sector_included_in_baseline: bool = True
    baseline_source_explicitly_supplied: bool = True
    frozen_constraint_projection_computed: bool = True
    gaussian_amplitude_moments_available: bool = True
    gaussian_profile_derives_evolved_stress_tail: bool = False
    external_ir_uv_stress_certificates_trusted: bool = True
    independent_energy_pressure_tail_bounds_assumed: bool = True
    joint_rho_p_tail_region_derived: bool = False
    finite_difference_conditioning_pass: bool = True
    finite_difference_truncation_error_certified: bool = False
    tail_time_derivative_certified: bool = False
    continuous_total_ward_identity_certified: bool = False
    projected_geometry_evolved: bool = False
    modes_recomputed_on_projected_geometry: bool = False
    reference_renormalized_stress_recomputed: bool = False
    full_renormalized_stress_derived: bool = False
    semiclassical_einstein_equation_solved: bool = False
    einstein_backreaction_computed: bool = False
    stochastic_noise_kernel_computed: bool = False
    semiclassical_stability_proved: bool = False
    physical_dark_matter_dark_energy_identification: bool = False
    absolute_abundance_computed: bool = False
    growth_lensing_computed: bool = False


def _three_point_derivative(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    *,
    maximum_adjacent_step_ratio: float = 10.0,
) -> tuple[float, ...]:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        raise ValueError("three-point derivative needs matching grids of length >= 3")
    if any(right <= left for left, right in zip(x_values, x_values[1:])):
        raise ValueError("derivative grid must be strictly increasing")
    maximum_adjacent_step_ratio = _finite(
        "maximum_adjacent_step_ratio", maximum_adjacent_step_ratio
    )
    if maximum_adjacent_step_ratio < 1.0:
        raise ValueError("maximum_adjacent_step_ratio must be at least one")
    adjacent_steps = tuple(
        right - left for left, right in zip(x_values, x_values[1:])
    )
    adjacent_step_ratio = max(adjacent_steps) / min(adjacent_steps)
    if (
        not math.isfinite(adjacent_step_ratio)
        or adjacent_step_ratio > maximum_adjacent_step_ratio
    ):
        raise ValueError("derivative grid adjacent-step ratio exceeds its ceiling")

    derivatives: list[float] = []
    last = len(x_values) - 1
    for index, evaluation_x in enumerate(x_values):
        if index == 0:
            indices = (0, 1, 2)
        elif index == last:
            indices = (last - 2, last - 1, last)
        else:
            indices = (index - 1, index, index + 1)
        xs = tuple(x_values[item] for item in indices)
        ys = tuple(y_values[item] for item in indices)
        derivative = math.fsum(
            value
            * (2.0 * evaluation_x - xs[(position + 1) % 3] - xs[(position + 2) % 3])
            / (
                (xs[position] - xs[(position + 1) % 3])
                * (xs[position] - xs[(position + 2) % 3])
            )
            for position, value in enumerate(ys)
        )
        if not math.isfinite(derivative):
            raise ValueError("three-point derivative is not finite")
        derivatives.append(derivative)
    return tuple(derivatives)


def _relative_residual(residual: float, *scales: float) -> float:
    return abs(residual) / max(1.0, *(abs(value) for value in scales))


def _residual_within_absolute_relative_tolerance(
    residual: float,
    *scales: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    """Apply ``|R| <= eps_abs + eps_rel max_i |term_i|``."""

    residual = _finite("residual", residual)
    absolute_tolerance = _finite("absolute_tolerance", absolute_tolerance)
    relative_tolerance = _finite("relative_tolerance", relative_tolerance)
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("residual tolerances must be non-negative")
    scale = max((abs(_finite("residual scale", value)) for value in scales), default=0.0)
    return abs(residual) <= absolute_tolerance + relative_tolerance * scale


def project_squeezed_ensemble_frozen_constraints(
    ensemble: "SqueezedFLRWStressEnsemble",
    *,
    baseline_nodes: tuple[ReferenceFLRWBaselineNode, ...],
    reduced_planck_over_h0: float,
    baseline_reference_sector_declaration: str,
    reference_renormalized_sector_included_in_baseline: bool,
    degeneracy: int = 1,
    synchronization_tolerance: float = 1.0e-9,
    baseline_closure_tolerance: float = 1.0e-9,
    baseline_closure_absolute_tolerance: float = 1.0e-12,
    maximum_relative_e_squared_shift: float = 0.1,
    maximum_state_difference_ward_relative_residual: float = 0.1,
    maximum_state_difference_ward_absolute_residual: float = 1.0e-12,
    maximum_adjacent_n_step_ratio: float = 4.0,
) -> FrozenFLRWConstraintProjection:
    r"""Project an E51 state difference through frozen FLRW constraints.

    With ``M=Mbar_Pl/H0`` and stresses in ``H0^4``, this computes

    ``E_fr^2 = E_b^2 + delta_r/(3 M^2)`` and
    ``d ln E_fr/dN = -(r_tot+p_tot)/(2 M^2 E_fr^2)``.

    The mode histories, reference subtraction, and external tail certificates
    remain those of the supplied background.  The result is therefore an
    algebraic reference-plus-delta constraint diagnostic, not a solution of
    the semiclassical Einstein equation.
    """

    reduced_planck_over_h0 = _finite(
        "reduced_planck_over_h0", reduced_planck_over_h0
    )
    for name, value in (
        ("synchronization_tolerance", synchronization_tolerance),
        ("baseline_closure_tolerance", baseline_closure_tolerance),
        ("maximum_relative_e_squared_shift", maximum_relative_e_squared_shift),
        (
            "maximum_state_difference_ward_relative_residual",
            maximum_state_difference_ward_relative_residual,
        ),
        ("maximum_adjacent_n_step_ratio", maximum_adjacent_n_step_ratio),
    ):
        value = _finite(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("baseline_closure_absolute_tolerance", baseline_closure_absolute_tolerance),
        (
            "maximum_state_difference_ward_absolute_residual",
            maximum_state_difference_ward_absolute_residual,
        ),
    ):
        value = _finite(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if synchronization_tolerance > 1.0e-4:
        raise ValueError("synchronization_tolerance must not exceed 1e-4")
    if baseline_closure_tolerance > 1.0e-4:
        raise ValueError("baseline_closure_tolerance must not exceed 1e-4")
    if maximum_adjacent_n_step_ratio < 1.0:
        raise ValueError("maximum_adjacent_n_step_ratio must be at least one")
    if reduced_planck_over_h0 <= 0.0:
        raise ValueError("reduced_planck_over_h0 must be positive")
    if isinstance(degeneracy, bool) or not isinstance(degeneracy, int) or degeneracy < 1:
        raise ValueError("degeneracy must be a positive integer")
    if not reference_renormalized_sector_included_in_baseline:
        raise ValueError(
            "the baseline must explicitly include the renormalized reference sector"
        )
    if (
        not isinstance(baseline_reference_sector_declaration, str)
        or not baseline_reference_sector_declaration.strip()
    ):
        raise ValueError("a non-empty baseline reference-sector declaration is required")
    if not isinstance(baseline_nodes, tuple) or len(baseline_nodes) < 3:
        raise ValueError("baseline_nodes must be a tuple with at least three nodes")
    if len(baseline_nodes) != len(ensemble.nodes):
        raise ValueError("one baseline node is required for every ensemble node")
    if not (
        ensemble.dimensions_pass
        and ensemble.pointwise_external_ir_uv_certificates_trusted
        and ensemble.analytic_bogoliubov_profile_verified
        and ensemble.absolute_bogoliubov_amplitude_moments_certified
        and ensemble.bogoliubov_integrability_certificate is not None
    ):
        raise ValueError("ensemble does not satisfy the frozen-projection input contract")
    if ensemble.evolved_mode_stress_tail_derived_from_profile:
        raise ValueError("Gaussian amplitudes must not be mislabeled as a stress-tail proof")

    try:
        planck_squared = reduced_planck_over_h0**2
    except OverflowError as error:
        raise ValueError("reduced Planck ratio squared is not finite") from error
    planck_squared = _finite("reduced Planck ratio squared", planck_squared)
    n_values = tuple(node.n for node in ensemble.nodes)
    adjacent_steps = tuple(
        right - left for left, right in zip(n_values, n_values[1:])
    )
    if any(step <= 0.0 or not math.isfinite(step) for step in adjacent_steps):
        raise ValueError("projection N grid must be finite and strictly increasing")
    adjacent_n_step_ratio = max(adjacent_steps) / min(adjacent_steps)

    def scaled_stress(name: str, value: float) -> float:
        return _finite(name, degeneracy * value)

    delta_energy = tuple(
        scaled_stress(
            "degeneracy-scaled state-difference energy",
            node.created_stress.energy_density_over_h0_four,
        )
        for node in ensemble.nodes
    )
    delta_pressure = tuple(
        scaled_stress(
            "degeneracy-scaled state-difference pressure",
            node.created_stress.pressure_over_h0_four,
        )
        for node in ensemble.nodes
    )
    delta_energy_d_n = _three_point_derivative(
        n_values,
        delta_energy,
        maximum_adjacent_step_ratio=maximum_adjacent_n_step_ratio,
    )

    max_baseline_friedmann_relative = 0.0
    max_baseline_raychaudhuri_relative = 0.0
    max_baseline_ward_relative = 0.0
    maximum_delta_ward_relative = 0.0
    maximum_shift_upper = 0.0
    projected_nodes: list[FrozenFLRWConstraintProjectionNode] = []

    for ensemble_node, baseline, delta_r, delta_p, delta_r_d_n in zip(
        ensemble.nodes,
        baseline_nodes,
        delta_energy,
        delta_pressure,
        delta_energy_d_n,
    ):
        synchronized_values = (
            (baseline.n, ensemble_node.n),
            (baseline.e, ensemble_node.hubble_over_h0),
            (baseline.d_log_e_d_n, ensemble_node.background_d_log_h_d_n),
        )
        if any(
            abs(actual - expected)
            > synchronization_tolerance * max(1.0, abs(actual), abs(expected))
            for actual, expected in synchronized_values
        ):
            raise ValueError("baseline and ensemble background nodes are not synchronized")

        baseline_friedmann = 3.0 * planck_squared * baseline.e**2 - baseline.energy_density
        baseline_raychaudhuri = (
            baseline.e**2 * baseline.d_log_e_d_n
            + (baseline.energy_density + baseline.pressure) / (2.0 * planck_squared)
        )
        baseline_ward = baseline.energy_density_d_n + 3.0 * (
            baseline.energy_density + baseline.pressure
        )
        baseline_friedmann_relative = _relative_residual(
            baseline_friedmann,
            3.0 * planck_squared * baseline.e**2,
            baseline.energy_density,
        )
        baseline_raychaudhuri_relative = _relative_residual(
            baseline_raychaudhuri,
            baseline.e**2 * baseline.d_log_e_d_n,
            (baseline.energy_density + baseline.pressure) / (2.0 * planck_squared),
        )
        baseline_ward_relative = _relative_residual(
            baseline_ward,
            baseline.energy_density_d_n,
            3.0 * (baseline.energy_density + baseline.pressure),
        )
        max_baseline_friedmann_relative = max(
            max_baseline_friedmann_relative,
            baseline_friedmann_relative,
        )
        max_baseline_raychaudhuri_relative = max(
            max_baseline_raychaudhuri_relative,
            baseline_raychaudhuri_relative,
        )
        max_baseline_ward_relative = max(
            max_baseline_ward_relative,
            baseline_ward_relative,
        )
        baseline_friedmann_pass = _residual_within_absolute_relative_tolerance(
            baseline_friedmann,
            3.0 * planck_squared * baseline.e**2,
            baseline.energy_density,
            absolute_tolerance=baseline_closure_absolute_tolerance,
            relative_tolerance=baseline_closure_tolerance,
        )
        baseline_raychaudhuri_pass = _residual_within_absolute_relative_tolerance(
            baseline_raychaudhuri,
            baseline.e**2 * baseline.d_log_e_d_n,
            (baseline.energy_density + baseline.pressure) / (2.0 * planck_squared),
            absolute_tolerance=baseline_closure_absolute_tolerance,
            relative_tolerance=baseline_closure_tolerance,
        )
        baseline_ward_pass = _residual_within_absolute_relative_tolerance(
            baseline_ward,
            baseline.energy_density_d_n,
            3.0 * (baseline.energy_density + baseline.pressure),
            absolute_tolerance=baseline_closure_absolute_tolerance,
            relative_tolerance=baseline_closure_tolerance,
        )
        if not all((
            baseline_friedmann_pass,
            baseline_raychaudhuri_pass,
            baseline_ward_pass,
        )):
            raise ValueError("supplied baseline does not close its FLRW constraints and Ward identity")

        energy_bound = scaled_stress(
            "degeneracy-scaled state-difference energy bound",
            ensemble_node.created_stress.energy_external_ir_uv_remainder_absolute_bound,
        )
        pressure_bound = scaled_stress(
            "degeneracy-scaled state-difference pressure bound",
            ensemble_node.created_stress.pressure_external_ir_uv_remainder_absolute_bound,
        )
        projected_e_squared = baseline.e**2 + delta_r / (3.0 * planck_squared)
        e_squared_interval = (
            baseline.e**2 + (delta_r - energy_bound) / (3.0 * planck_squared),
            baseline.e**2 + (delta_r + energy_bound) / (3.0 * planck_squared),
        )
        if projected_e_squared <= 0.0 or e_squared_interval[0] <= 0.0:
            raise ValueError("state difference or its IR/UV bound makes E^2 non-positive")

        d_log_corners: list[float] = []
        for energy_sign in (-1.0, 1.0):
            varied_delta_r = delta_r + energy_sign * energy_bound
            varied_e_squared = baseline.e**2 + varied_delta_r / (3.0 * planck_squared)
            for pressure_sign in (-1.0, 1.0):
                varied_delta_p = delta_p + pressure_sign * pressure_bound
                d_log_corners.append(
                    -(
                        baseline.energy_density
                        + baseline.pressure
                        + varied_delta_r
                        + varied_delta_p
                    )
                    / (2.0 * planck_squared * varied_e_squared)
                )
        projected_d_log_e_d_n = -(
            baseline.energy_density + baseline.pressure + delta_r + delta_p
        ) / (2.0 * planck_squared * projected_e_squared)
        acceleration = -(
            baseline.energy_density
            + 3.0 * baseline.pressure
            + delta_r
            + 3.0 * delta_p
        ) / (6.0 * planck_squared)
        acceleration_bound = (energy_bound + 3.0 * pressure_bound) / (
            6.0 * planck_squared
        )
        shift_upper = max(
            abs(delta_r - energy_bound),
            abs(delta_r + energy_bound),
        ) / (3.0 * planck_squared * baseline.e**2)
        maximum_shift_upper = max(maximum_shift_upper, shift_upper)
        if shift_upper > maximum_relative_e_squared_shift:
            raise ValueError("frozen E^2 shift exceeds the declared perturbative ceiling")

        delta_ward = delta_r_d_n + 3.0 * (delta_r + delta_p)
        delta_ward_relative = _relative_residual(
            delta_ward,
            delta_r_d_n,
            3.0 * (delta_r + delta_p),
        )
        maximum_delta_ward_relative = max(
            maximum_delta_ward_relative,
            delta_ward_relative,
        )
        if not _residual_within_absolute_relative_tolerance(
            delta_ward,
            delta_r_d_n,
            3.0 * (delta_r + delta_p),
            absolute_tolerance=maximum_state_difference_ward_absolute_residual,
            relative_tolerance=maximum_state_difference_ward_relative_residual,
        ):
            raise ValueError("finite-grid state-difference Ward residual exceeds its ceiling")

        closure = backreaction_closure_receipt(
            e=math.sqrt(projected_e_squared),
            d_log_e_d_n=projected_d_log_e_d_n,
            reduced_planck_over_h0=reduced_planck_over_h0,
            theta_d_n=0.0,
            theta_d2_n=0.0,
            potential=0.0,
            potential_d_theta=0.0,
            scalar_channels=(),
            conserved_fluids=(
                ConservedFluid(
                    energy_density=baseline.energy_density,
                    pressure=baseline.pressure,
                    energy_density_d_n=baseline.energy_density_d_n,
                ),
                ConservedFluid(
                    energy_density=delta_r,
                    pressure=delta_p,
                    energy_density_d_n=delta_r_d_n,
                ),
            ),
        )
        projected_nodes.append(
            FrozenFLRWConstraintProjectionNode(
                n=ensemble_node.n,
                background_e=baseline.e,
                background_d_log_e_d_n=baseline.d_log_e_d_n,
                state_difference_energy_density=delta_r,
                state_difference_pressure=delta_p,
                state_difference_energy_density_d_n=delta_r_d_n,
                state_difference_energy_external_ir_uv_bound=energy_bound,
                state_difference_pressure_external_ir_uv_bound=pressure_bound,
                projected_e=math.sqrt(projected_e_squared),
                projected_e_squared=projected_e_squared,
                projected_e_squared_interval=e_squared_interval,
                projected_d_log_e_d_n=projected_d_log_e_d_n,
                projected_d_log_e_d_n_interval=(
                    min(d_log_corners),
                    max(d_log_corners),
                ),
                projected_acceleration_over_h0_squared=acceleration,
                projected_acceleration_over_h0_squared_interval=(
                    acceleration - acceleration_bound,
                    acceleration + acceleration_bound,
                ),
                relative_e_squared_shift_upper=shift_upper,
                baseline_ward_residual=baseline_ward,
                state_difference_ward_residual=delta_ward,
                total_ward_residual=baseline_ward + delta_ward,
                closure=closure,
            )
        )

    return FrozenFLRWConstraintProjection(
        nodes=tuple(projected_nodes),
        reduced_planck_over_h0=reduced_planck_over_h0,
        degeneracy=degeneracy,
        baseline_reference_sector_declaration=(
            baseline_reference_sector_declaration.strip()
        ),
        maximum_relative_e_squared_shift_upper=maximum_shift_upper,
        maximum_state_difference_ward_relative_residual=(
            maximum_delta_ward_relative
        ),
        maximum_baseline_friedmann_relative_residual=(
            max_baseline_friedmann_relative
        ),
        maximum_baseline_raychaudhuri_relative_residual=(
            max_baseline_raychaudhuri_relative
        ),
        maximum_baseline_ward_relative_residual=max_baseline_ward_relative,
        baseline_closure_absolute_tolerance=baseline_closure_absolute_tolerance,
        state_difference_ward_absolute_tolerance=(
            maximum_state_difference_ward_absolute_residual
        ),
        adjacent_n_step_ratio=adjacent_n_step_ratio,
        mass_dimension_manifest=(
            ("E_and_d_log_E_d_N", 0.0),
            ("Mbar_Pl_over_H0", 0.0),
            ("rho_and_pressure_over_H0_four", 0.0),
            ("friedmann_constraint_over_H0_four", 0.0),
            ("acceleration_over_H0_squared", 0.0),
        ),
        dimensions_pass=True,
    )
