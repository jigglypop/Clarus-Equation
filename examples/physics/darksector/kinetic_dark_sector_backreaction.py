"""동적 시계에 대한 정확한 무차원 FLRW 워드(Ward)/되먹임(backreaction) 닫힘이다.

선언된 작용은

    S = integral sqrt(-g) [
        Mpl^2 R/2 - (partial T)^2/2 - V(T)
        - sum_s sum_A ((partial phi_sA)^2
                       + (m_s(T)^2 + xi_s R) phi_sA^2)/2
    ].

theta=T/H0, N=log(a), E=H/H0, M=Mpl/H0, r=rho/H0^4, pi=p/H0^4, v=V/H0^4 를 쓰고,
실수 성분 하나에 대해 Phi_s^2=<phi_sA^2>/H0^2 로 둔다. 동일한 성분 g_s 개에 대해

    j_s = g_s (d mu_s^2/d theta) Phi_s^2 / 2,
    r_s,N + 3(r_s+pi_s) = theta_N j_s,

이고 정준 시계는

    E^2[theta_NN + (3+E_N/E)theta_N] + v_theta + sum_s j_s = 0

을 따른다. 시계 워드 잔차는 이 방정식에 theta_N 을 곱한 것이다. 따라서 전체 워드
항등식은 대수적이지 적합된 수치 상쇄가 아니다. 레이초두리(Raychaudhuri) 방정식도
성립하면 프리드만 구속의 도함수는 정확히 0이다.

    C_N = 6 M^2 R_Raychaudhuri - W_total.

이 모듈은 그 조건부 닫힘을 증명한다. 스칼라 입력은 이미 공통 조절자/상쇄항 삼중항에서
와야 한다. 이 모듈은 장부 산술로 재규격화 응력을 만들지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examples.physics.darksector.kinetic_dark_sector_adiabatic_stress import (
        SqueezedFLRWStressEnsemble,
    )


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ScalarBackreactionChannel:
    """단일 장 대 다중항 관례를 명시한 종 하나다.

    ``field_squared`` 는 ``degeneracy`` 개의 동일 장 중 하나에 대한 값이고,
    ``energy_density``, ``pressure``, ``energy_density_d_n`` 은 이미 축퇴를 합한 다중항
    총량이다.
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
    """스칼라 전달, 시계 응답, 구속 전파를 감사한다."""

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
    """한 N 에서 명시된 재규격화 기준+고전 FLRW 원천이다."""

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
    """동결 배경 프리드만/레이초두리 구속 사영 하나다."""

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
    """기준+차이 대수 사영이지 진화된 기하가 아니다."""

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
        local_origin = ys[1]
        derivative = math.fsum(
            (value - local_origin)
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
    """``|R| <= eps_abs + eps_rel max_i |term_i|`` 를 적용한다."""

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
    """E51 상태 차이를 동결 FLRW 구속으로 사영한다.

    ``M=Mbar_Pl/H0`` 와 ``H0^4`` 단위의 응력으로

    ``E_fr^2 = E_b^2 + delta_r/(3 M^2)``,
    ``d ln E_fr/dN = -(r_tot+p_tot)/(2 M^2 E_fr^2)``

    을 계산한다. 모드 역사, 기준 빼기, 외부 꼬리 보증서는 주어진 배경의 것 그대로다.
    따라서 결과는 대수적 기준+차이 구속 진단이지 준고전 아인슈타인 방정식의 해가
    아니다.
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


@dataclass(frozen=True)
class MeanFieldFLRWBackgroundNode:
    """E53 반복을 위한 무차원 팽창 배경 노드 하나다."""

    n: float
    e2: float

    def __post_init__(self) -> None:
        _finite("mean-field background n", self.n)
        _finite("mean-field background e2", self.e2)
        if self.e2 <= 0.0:
            raise ValueError("mean-field expanding background requires positive E^2")


@dataclass(frozen=True)
class MeanFieldFLRWBackground:
    """고정된 도함수 보호 노드를 가진 조각별 선형 E^2 배경이다.

    상태 차이 궤적에 대해 ``a``, ``a'``, ``a''`` 만 보증된다. 호환용 ``ScaleFactorJet`` 의
    상위 항목은 0 자리표시자이며 절대 단열 빼기에 써서는 안 된다.
    """

    nodes: tuple[MeanFieldFLRWBackgroundNode, ...]
    active_window: tuple[float, float]
    curvature_derivative_step_n: float = 1.0e-4
    status: str = (
        "EXPANDING_E2_GRID_FIXED_GUARD_POSITIONS_EXTRAPOLATED_VALUES_"
        "STATE_DIFFERENCE_ONLY"
    )
    expanding_branch_only: bool = True
    boundary_guard_grid_fixed: bool = True
    boundary_guard_values_extrapolated_after_update: bool = True
    state_difference_jet_derivative_order: int = 2
    higher_jet_derivatives_certified: bool = False
    suitable_for_absolute_adiabatic_subtraction: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or len(self.nodes) < 5:
            raise ValueError("mean-field background needs at least five nodes")
        n_values = tuple(node.n for node in self.nodes)
        if any(right <= left for left, right in zip(n_values, n_values[1:])):
            raise ValueError("mean-field background nodes must be strictly increasing")
        start, end = self.active_window
        if not all(math.isfinite(value) for value in (start, end)) or end <= start:
            raise ValueError("mean-field active_window must be finite and ordered")
        step = _finite(
            "mean-field curvature_derivative_step_n",
            self.curvature_derivative_step_n,
        )
        if step <= 0.0:
            raise ValueError("curvature_derivative_step_n must be positive")
        if start - step < n_values[0] or end + step > n_values[-1]:
            raise ValueError("active window needs fixed derivative guard nodes")
        active = tuple(node for node in self.nodes if start <= node.n <= end)
        if len(active) < 3:
            raise ValueError("mean-field active window needs at least three nodes")
        tolerance = 32.0 * math.ulp(1.0) * max(1.0, abs(start), abs(end))
        if abs(active[0].n - start) > tolerance or abs(active[-1].n - end) > tolerance:
            raise ValueError("active window endpoints must be explicit background nodes")

    @property
    def active_nodes(self) -> tuple[MeanFieldFLRWBackgroundNode, ...]:
        start, end = self.active_window
        return tuple(node for node in self.nodes if start <= node.n <= end)

    def at_n(self, n: float) -> MeanFieldFLRWBackgroundNode:
        n = _finite("requested mean-field e-fold", n)
        if n < self.nodes[0].n or n > self.nodes[-1].n:
            raise ValueError("requested e-fold is outside the mean-field background")
        lo, hi = 0, len(self.nodes) - 1
        while hi - lo > 1:
            middle = (lo + hi) // 2
            if self.nodes[middle].n <= n:
                lo = middle
            else:
                hi = middle
        left, right = self.nodes[lo], self.nodes[hi]
        weight = (n - left.n) / (right.n - left.n)
        e2 = left.e2 + weight * (right.e2 - left.e2)
        return MeanFieldFLRWBackgroundNode(n=n, e2=e2)

    def d_log_h_d_n(self, n: float) -> float:
        step = self.curvature_derivative_step_n
        left = math.log(self.at_n(n - step).e2)
        right = math.log(self.at_n(n + step).e2)
        return (right - left) / (4.0 * step)

    def state_difference_scale_factor_jet_at_n(self, n: float):
        """상태 차이 추적이 쓰는 2차 제트를 돌려준다."""

        from examples.physics.darksector.kinetic_dark_sector_adiabatic_stress import (
            ScaleFactorJet,
        )

        a = math.exp(_finite("state-difference jet n", n))
        e2 = self.at_n(n).e2
        d_log_h_d_n = self.d_log_h_d_n(n)
        d1 = a * a * math.sqrt(e2)
        d2 = a**3 * e2 * (2.0 + d_log_h_d_n)
        return ScaleFactorJet(a, d1, d2, 0.0, 0.0, 0.0, 0.0)

    def with_active_e2(self, values: tuple[float, ...]) -> "MeanFieldFLRWBackground":
        active = self.active_nodes
        if not isinstance(values, tuple) or len(values) != len(active):
            raise ValueError("one E^2 update is required per active background node")

        left_slope = (values[1] - values[0]) / (active[1].n - active[0].n)
        right_slope = (values[-1] - values[-2]) / (
            active[-1].n - active[-2].n
        )
        active_values = {node.n: value for node, value in zip(active, values)}

        def updated_node(node: MeanFieldFLRWBackgroundNode) -> MeanFieldFLRWBackgroundNode:
            if node.n < active[0].n:
                value = values[0] + left_slope * (node.n - active[0].n)
            elif node.n > active[-1].n:
                value = values[-1] + right_slope * (node.n - active[-1].n)
            else:
                try:
                    value = active_values[node.n]
                except KeyError as error:
                    raise ValueError(
                        "active background grid changed during E^2 update"
                    ) from error
            return MeanFieldFLRWBackgroundNode(node.n, value)

        return MeanFieldFLRWBackground(
            nodes=tuple(updated_node(node) for node in self.nodes),
            active_window=self.active_window,
            curvature_derivative_step_n=self.curvature_derivative_step_n,
        )


@dataclass(frozen=True)
class SemiclassicalReferenceSourceNode:
    """호출자가 준 고전 + 재규격화 기준 원천이다."""

    n: float
    energy_density: float
    pressure: float
    energy_density_d_n: float
    energy_absolute_bound: float = 0.0
    pressure_absolute_bound: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("reference source n", self.n),
            ("reference source energy_density", self.energy_density),
            ("reference source pressure", self.pressure),
            ("reference source energy_density_d_n", self.energy_density_d_n),
            ("reference source energy_absolute_bound", self.energy_absolute_bound),
            ("reference source pressure_absolute_bound", self.pressure_absolute_bound),
        ):
            _finite(name, value)
        if self.energy_absolute_bound < 0.0 or self.pressure_absolute_bound < 0.0:
            raise ValueError("reference source absolute bounds must be non-negative")


@dataclass(frozen=True)
class ModeRecomputedSemiclassicalResponse:
    """E53 고정점 루프에 주는 후보 배경 응답 하나다."""

    ensemble: "SqueezedFLRWStressEnsemble"
    reference_source_nodes: tuple[SemiclassicalReferenceSourceNode, ...]
    maximum_mode_wronskian_residual: float
    mode_solution_count: int
    renormalization_scheme_declaration: str
    state_preparation_declaration: str
    finite_reference_split_adjustment_nodes: (
        tuple[SemiclassicalReferenceSourceNode, ...]
    ) = ()
    status: str = "CALLER_RECOMPUTED_MODES_AND_REFERENCE_SOURCE_RESPONSE"
    modes_recomputed_on_candidate_background: bool = True
    reference_source_evaluated_on_candidate_background: bool = True
    absolute_reference_renormalization_supplied_by_caller: bool = True
    absolute_reference_renormalization_derived_by_fixed_point_solver: bool = False
    same_regulator_and_finite_counterterm_convention_declared: bool = True

    def __post_init__(self) -> None:
        _finite(
            "maximum_mode_wronskian_residual",
            self.maximum_mode_wronskian_residual,
        )
        if self.maximum_mode_wronskian_residual < 0.0:
            raise ValueError("maximum_mode_wronskian_residual must be non-negative")
        if (
            isinstance(self.mode_solution_count, bool)
            or not isinstance(self.mode_solution_count, int)
            or self.mode_solution_count < 1
        ):
            raise ValueError("mode_solution_count must be a positive integer")
        for name, declaration in (
            ("renormalization_scheme_declaration", self.renormalization_scheme_declaration),
            ("state_preparation_declaration", self.state_preparation_declaration),
        ):
            if not isinstance(declaration, str) or not declaration.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class MeanFieldFixedPointIteration:
    iteration: int
    maximum_fixed_point_relative_residual: float
    maximum_damped_update_relative: float
    minimum_target_e_squared_lower: float
    maximum_friedmann_relative_residual: float
    maximum_raychaudhuri_relative_residual: float
    maximum_reference_ward_relative_residual: float
    maximum_state_difference_ward_relative_residual: float
    maximum_total_ward_relative_residual: float
    maximum_constraint_propagation_relative_residual: float
    maximum_geometry_derivative_relative_mismatch: float
    maximum_reference_derivative_consistency_relative_residual: float
    maximum_split_derivative_consistency_relative_residual: float
    maximum_raychaudhuri_tail_robust_relative_residual: float
    maximum_raychaudhuri_absolute_uncertainty: float
    minimum_acceleration_lower_bound: float
    maximum_acceleration_upper_bound: float
    maximum_mode_wronskian_residual: float
    empirical_residual_ratio: float | None


@dataclass(frozen=True)
class ModeRecomputedMeanFieldFixedPoint:
    """외부 기준 원천을 가진 조건부 모드 재계산 고정점이다."""

    background: MeanFieldFLRWBackground
    final_response: ModeRecomputedSemiclassicalResponse
    iterations: tuple[MeanFieldFixedPointIteration, ...]
    reduced_planck_over_h0: float
    damping: float
    response_evaluation_count: int
    maximum_final_fixed_point_relative_residual: float
    maximum_final_friedmann_relative_residual: float
    maximum_final_raychaudhuri_relative_residual: float
    maximum_final_reference_ward_relative_residual: float
    maximum_final_state_difference_ward_relative_residual: float
    maximum_final_total_ward_relative_residual: float
    maximum_final_constraint_propagation_relative_residual: float
    maximum_final_geometry_derivative_relative_mismatch: float
    maximum_final_reference_derivative_consistency_relative_residual: float
    maximum_final_split_derivative_consistency_relative_residual: float
    maximum_final_raychaudhuri_tail_robust_relative_residual: float
    maximum_final_raychaudhuri_absolute_uncertainty: float
    minimum_final_acceleration_lower_bound: float
    maximum_final_acceleration_upper_bound: float
    final_response_reproducibility_relative_residual: float
    maximum_observed_empirical_residual_ratio: float
    geometry_derivative_relative_tolerance: float
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    status: str = (
        "CONDITIONAL_CALLBACK_CONTRACTED_MODE_RECOMPUTED_REFERENCE_SUPPLIED_"
        "CENTRAL_GRID_MEAN_FIELD_FIXED_POINT"
    )
    expanding_branch_only: bool = True
    fixed_boundary_guard_grid_used: bool = True
    boundary_guard_values_extrapolated_after_update: bool = True
    modes_recomputed_each_iteration_by_callback_contract: bool = True
    final_modes_recomputed_on_converged_background_by_callback_contract: bool = True
    reference_source_evaluated_each_iteration_by_callback_contract: bool = True
    callback_recomputation_declarations_independently_proved: bool = False
    aggregate_numeric_response_reproducibility_checked: bool = True
    complete_internal_mode_trajectory_reproducibility_proved: bool = False
    state_difference_mean_field_fixed_point_converged: bool = True
    friedmann_ray_ward_checked_together: bool = True
    finite_reference_split_contract_supported: bool = True
    reference_and_split_derivatives_checked_against_node_grid: bool = True
    mode_and_ray_use_same_background_derivative: bool = True
    pressure_tail_propagated_to_raychaudhuri_gate: bool = True
    independent_energy_pressure_tail_rectangle_used: bool = True
    joint_energy_pressure_tail_region_derived: bool = False
    continuous_tail_ward_certified: bool = False
    state_difference_jet_derivative_order: int = 2
    higher_jet_derivatives_certified: bool = False
    absolute_reference_renormalization_derived: bool = False
    full_hadamard_state_proved: bool = False
    full_renormalized_stress_derived: bool = False
    contraction_mapping_proved: bool = False
    fixed_point_uniqueness_proved: bool = False
    higher_curvature_order_reduction_performed: bool = False
    runaway_absence_proved: bool = False
    semiclassical_einstein_equation_solved: bool = False
    full_einstein_backreaction_computed: bool = False
    stochastic_noise_kernel_computed: bool = False
    semiclassical_stability_proved: bool = False
    universal_planck_tick_assumed: bool = False
    physical_dark_matter_dark_energy_identification: bool = False
    absolute_abundance_computed: bool = False
    growth_lensing_computed: bool = False


@dataclass(frozen=True)
class _MeanFieldResponseEvaluation:
    target_e_squared: tuple[float, ...]
    target_e_squared_lower: tuple[float, ...]
    target_e_squared_upper: tuple[float, ...]
    total_energy_density: tuple[float, ...]
    total_pressure: tuple[float, ...]
    total_energy_density_d_n: tuple[float, ...]
    total_energy_absolute_bounds: tuple[float, ...]
    total_pressure_absolute_bounds: tuple[float, ...]
    raychaudhuri_absolute_uncertainties: tuple[float, ...]
    acceleration_lower_bounds: tuple[float, ...]
    acceleration_upper_bounds: tuple[float, ...]
    reference_derivative_consistency_residuals: tuple[float, ...]
    split_derivative_consistency_residuals: tuple[float, ...]
    reference_ward_residuals: tuple[float, ...]
    state_difference_ward_residuals: tuple[float, ...]
    total_ward_residuals: tuple[float, ...]
    friedmann_residuals: tuple[float, ...]
    raychaudhuri_residuals: tuple[float, ...]
    constraint_propagation_residuals: tuple[float, ...]
    maximum_fixed_point_relative_residual: float
    maximum_friedmann_relative_residual: float
    maximum_raychaudhuri_relative_residual: float
    maximum_reference_ward_relative_residual: float
    maximum_state_difference_ward_relative_residual: float
    maximum_total_ward_relative_residual: float
    maximum_constraint_propagation_relative_residual: float
    maximum_geometry_derivative_relative_mismatch: float
    maximum_reference_derivative_consistency_relative_residual: float
    maximum_split_derivative_consistency_relative_residual: float
    maximum_raychaudhuri_tail_robust_relative_residual: float


def _mean_field_response_evaluation(
    background: MeanFieldFLRWBackground,
    response: ModeRecomputedSemiclassicalResponse,
    *,
    planck_squared: float,
    degeneracy: int,
    synchronization_tolerance: float,
    ward_absolute_tolerance: float,
    ward_relative_tolerance: float,
    maximum_mode_wronskian_residual: float,
    maximum_adjacent_n_step_ratio: float,
) -> _MeanFieldResponseEvaluation:
    ensemble = response.ensemble
    active_nodes = background.active_nodes
    ensemble_nodes = ensemble.nodes
    if len(ensemble_nodes) != len(active_nodes):
        raise ValueError("response ensemble must cover every active background node")
    if len(response.reference_source_nodes) != len(active_nodes):
        raise ValueError("response needs one reference source per active node")
    if response.finite_reference_split_adjustment_nodes and (
        len(response.finite_reference_split_adjustment_nodes) != len(active_nodes)
    ):
        raise ValueError("finite reference-split adjustment must cover every active node")
    if not (
        response.modes_recomputed_on_candidate_background
        and response.reference_source_evaluated_on_candidate_background
        and response.absolute_reference_renormalization_supplied_by_caller
        and response.same_regulator_and_finite_counterterm_convention_declared
    ):
        raise ValueError("response does not satisfy the E53 recomputation contract")
    if not (
        ensemble.dimensions_pass
        and ensemble.analytic_bogoliubov_profile_verified
        and ensemble.absolute_bogoliubov_amplitude_moments_certified
        and ensemble.pointwise_external_ir_uv_certificates_trusted
        and ensemble.bogoliubov_integrability_certificate is not None
    ):
        raise ValueError("response ensemble does not satisfy the E51/E52 contract")
    if ensemble.evolved_mode_stress_tail_derived_from_profile:
        raise ValueError("Gaussian amplitude profile cannot stand in for a stress-tail proof")
    if response.mode_solution_count != len(ensemble.q_values):
        raise ValueError("response mode count must equal the ensemble q-grid size")
    if response.maximum_mode_wronskian_residual > maximum_mode_wronskian_residual:
        raise ValueError("recomputed mode Wronskian residual exceeds its ceiling")

    n_values = tuple(node.n for node in active_nodes)
    candidate_e_squared = tuple(node.e2 for node in active_nodes)
    grid_candidate_e_squared_d_n = _three_point_derivative(
        n_values,
        candidate_e_squared,
        maximum_adjacent_step_ratio=maximum_adjacent_n_step_ratio,
    )
    grid_d_log_h_d_n = tuple(
        0.5 * derivative / e2
        for derivative, e2 in zip(
            grid_candidate_e_squared_d_n,
            candidate_e_squared,
        )
    )
    mode_d_log_h_d_n = tuple(
        background.d_log_h_d_n(node.n) for node in active_nodes
    )
    mode_candidate_e_squared_d_n = tuple(
        2.0 * e2 * d_log
        for e2, d_log in zip(candidate_e_squared, mode_d_log_h_d_n)
    )

    adjustment_nodes = response.finite_reference_split_adjustment_nodes or tuple(
        SemiclassicalReferenceSourceNode(
            n=node.n,
            energy_density=0.0,
            pressure=0.0,
            energy_density_d_n=0.0,
        )
        for node in active_nodes
    )
    reference_grid_energy_d_n = _three_point_derivative(
        n_values,
        tuple(node.energy_density for node in response.reference_source_nodes),
        maximum_adjacent_step_ratio=maximum_adjacent_n_step_ratio,
    )
    adjustment_grid_energy_d_n = _three_point_derivative(
        n_values,
        tuple(node.energy_density for node in adjustment_nodes),
        maximum_adjacent_step_ratio=maximum_adjacent_n_step_ratio,
    )
    reference_derivative_consistency = tuple(
        supplied.energy_density_d_n - grid
        for supplied, grid in zip(
            response.reference_source_nodes,
            reference_grid_energy_d_n,
        )
    )
    split_derivative_consistency = tuple(
        supplied.energy_density_d_n - grid
        for supplied, grid in zip(
            adjustment_nodes,
            adjustment_grid_energy_d_n,
        )
    )
    for name, nodes, grid_values, residuals in (
        (
            "reference source",
            response.reference_source_nodes,
            reference_grid_energy_d_n,
            reference_derivative_consistency,
        ),
        (
            "finite reference-split adjustment",
            adjustment_nodes,
            adjustment_grid_energy_d_n,
            split_derivative_consistency,
        ),
    ):
        for node, grid_value, residual in zip(nodes, grid_values, residuals):
            if not _residual_within_absolute_relative_tolerance(
                residual,
                node.energy_density_d_n,
                grid_value,
                absolute_tolerance=ward_absolute_tolerance,
                relative_tolerance=ward_relative_tolerance,
            ):
                raise ValueError(
                    f"{name} supplied energy derivative disagrees with its node grid"
                )
    for active, ensemble_node, reference, adjustment in zip(
        active_nodes,
        ensemble_nodes,
        response.reference_source_nodes,
        adjustment_nodes,
    ):
        expected_e = math.sqrt(active.e2)
        expected_d_log_h_d_n = background.d_log_h_d_n(active.n)
        synchronized = (
            (ensemble_node.n, active.n),
            (reference.n, active.n),
            (adjustment.n, active.n),
            (ensemble_node.hubble_over_h0, expected_e),
            (ensemble_node.background_d_log_h_d_n, expected_d_log_h_d_n),
        )
        if any(
            abs(actual - expected)
            > synchronization_tolerance * max(1.0, abs(actual), abs(expected))
            for actual, expected in synchronized
        ):
            raise ValueError(
                "response ensemble/reference source is not synchronized to candidate background"
            )

    def scaled(name: str, value: float) -> float:
        return _finite(name, degeneracy * value)

    central_delta_energy = tuple(
        scaled(
            "mean-field degeneracy-scaled state-difference energy",
            node.created_stress.energy_density_over_h0_four,
        )
        for node in ensemble_nodes
    )
    central_delta_pressure = tuple(
        scaled(
            "mean-field degeneracy-scaled state-difference pressure",
            node.created_stress.pressure_over_h0_four,
        )
        for node in ensemble_nodes
    )
    central_delta_energy_d_n = _three_point_derivative(
        n_values,
        central_delta_energy,
        maximum_adjacent_step_ratio=maximum_adjacent_n_step_ratio,
    )

    total_energy: list[float] = []
    total_pressure: list[float] = []
    total_energy_d_n: list[float] = []
    total_energy_bounds: list[float] = []
    total_pressure_bounds: list[float] = []
    target_e_squared: list[float] = []
    target_e_squared_lower: list[float] = []
    target_e_squared_upper: list[float] = []
    ray_uncertainties: list[float] = []
    acceleration_lower: list[float] = []
    acceleration_upper: list[float] = []
    reference_ward: list[float] = []
    state_ward: list[float] = []
    total_ward: list[float] = []
    friedmann: list[float] = []
    raychaudhuri: list[float] = []
    propagation: list[float] = []
    geometry_mismatch: list[float] = []

    for (
        active,
        ensemble_node,
        reference,
        adjustment,
        delta_r,
        delta_p,
        delta_r_d_n,
        reference_r_d_n,
        adjustment_r_d_n,
        mode_d_log,
        grid_d_log,
        mode_e2_d_n,
    ) in zip(
        active_nodes,
        ensemble_nodes,
        response.reference_source_nodes,
        adjustment_nodes,
        central_delta_energy,
        central_delta_pressure,
        central_delta_energy_d_n,
        reference_grid_energy_d_n,
        adjustment_grid_energy_d_n,
        mode_d_log_h_d_n,
        grid_d_log_h_d_n,
        mode_candidate_e_squared_d_n,
    ):
        state_r = delta_r + adjustment.energy_density
        state_p = delta_p + adjustment.pressure
        state_r_d_n = delta_r_d_n + adjustment_r_d_n
        total_r = reference.energy_density + state_r
        total_p = reference.pressure + state_p
        total_r_d_n = reference_r_d_n + state_r_d_n
        energy_bound = (
            reference.energy_absolute_bound
            + adjustment.energy_absolute_bound
            + scaled(
                "mean-field degeneracy-scaled state-difference energy bound",
                ensemble_node.created_stress.energy_external_ir_uv_remainder_absolute_bound,
            )
        )
        pressure_bound = (
            reference.pressure_absolute_bound
            + adjustment.pressure_absolute_bound
            + scaled(
                "mean-field degeneracy-scaled state-difference pressure bound",
                ensemble_node.created_stress.pressure_external_ir_uv_remainder_absolute_bound,
            )
        )
        target = total_r / (3.0 * planck_squared)
        target_lower = (total_r - energy_bound) / (3.0 * planck_squared)
        target_upper = (total_r + energy_bound) / (3.0 * planck_squared)
        ray_uncertainty = (energy_bound + pressure_bound) / (
            2.0 * planck_squared
        )
        acceleration = -(total_r + 3.0 * total_p) / (6.0 * planck_squared)
        acceleration_uncertainty = (energy_bound + 3.0 * pressure_bound) / (
            6.0 * planck_squared
        )
        if not all(
            math.isfinite(value)
            for value in (
                target,
                target_lower,
                target_upper,
                ray_uncertainty,
                acceleration,
                acceleration_uncertainty,
            )
        ):
            raise ValueError("mean-field Friedmann target is not finite")
        if target <= 0.0 or target_lower <= 0.0:
            raise ValueError("mean-field Friedmann target or tail lower bound is non-positive")

        reference_residual = reference_r_d_n + 3.0 * (
            reference.energy_density + reference.pressure
        )
        state_residual = state_r_d_n + 3.0 * (state_r + state_p)
        total_residual = total_r_d_n + 3.0 * (total_r + total_p)
        for name, residual, derivative_term, pressure_term in (
            (
                "reference source",
                reference_residual,
                reference_r_d_n,
                3.0 * (reference.energy_density + reference.pressure),
            ),
            (
                "state-difference source",
                state_residual,
                state_r_d_n,
                3.0 * (state_r + state_p),
            ),
            (
                "total source",
                total_residual,
                total_r_d_n,
                3.0 * (total_r + total_p),
            ),
        ):
            if not _residual_within_absolute_relative_tolerance(
                residual,
                derivative_term,
                pressure_term,
                absolute_tolerance=ward_absolute_tolerance,
                relative_tolerance=ward_relative_tolerance,
            ):
                raise ValueError(f"{name} finite-grid Ward residual exceeds its ceiling")

        friedmann_residual = 3.0 * planck_squared * active.e2 - total_r
        ray_residual = active.e2 * mode_d_log + (
            total_r + total_p
        ) / (2.0 * planck_squared)
        propagation_residual = 3.0 * planck_squared * mode_e2_d_n - total_r_d_n

        total_energy.append(total_r)
        total_pressure.append(total_p)
        total_energy_d_n.append(total_r_d_n)
        total_energy_bounds.append(energy_bound)
        total_pressure_bounds.append(pressure_bound)
        target_e_squared.append(target)
        target_e_squared_lower.append(target_lower)
        target_e_squared_upper.append(target_upper)
        ray_uncertainties.append(ray_uncertainty)
        acceleration_lower.append(acceleration - acceleration_uncertainty)
        acceleration_upper.append(acceleration + acceleration_uncertainty)
        reference_ward.append(reference_residual)
        state_ward.append(state_residual)
        total_ward.append(total_residual)
        friedmann.append(friedmann_residual)
        raychaudhuri.append(ray_residual)
        propagation.append(propagation_residual)
        geometry_mismatch.append(mode_d_log - grid_d_log)

    fixed_relative = max(
        abs(target - active.e2) / max(1.0, abs(target), abs(active.e2))
        for target, active in zip(target_e_squared, active_nodes)
    )
    friedmann_relative = max(
        _relative_residual(
            residual,
            3.0 * planck_squared * active.e2,
            total_r,
        )
        for residual, active, total_r in zip(
            friedmann,
            active_nodes,
            total_energy,
        )
    )
    ray_relative = max(
        _relative_residual(
            residual,
            active.e2 * mode_d_log,
            (total_r + total_p) / (2.0 * planck_squared),
        )
        for residual, active, mode_d_log, total_r, total_p in zip(
            raychaudhuri,
            active_nodes,
            mode_d_log_h_d_n,
            total_energy,
            total_pressure,
        )
    )
    ray_tail_robust_relative = max(
        (abs(residual) + uncertainty)
        / max(
            1.0,
            abs(active.e2 * mode_d_log),
            abs((total_r + total_p) / (2.0 * planck_squared)),
        )
        for residual, uncertainty, active, mode_d_log, total_r, total_p in zip(
            raychaudhuri,
            ray_uncertainties,
            active_nodes,
            mode_d_log_h_d_n,
            total_energy,
            total_pressure,
        )
    )
    reference_ward_relative = max(
        _relative_residual(
            residual,
            derivative,
            3.0 * (node.energy_density + node.pressure),
        )
        for residual, node, derivative in zip(
            reference_ward,
            response.reference_source_nodes,
            reference_grid_energy_d_n,
        )
    )
    state_ward_relative = max(
        _relative_residual(residual, derivative, 3.0 * (energy + pressure))
        for residual, derivative, energy, pressure in zip(
            state_ward,
            tuple(
                derivative + adjustment_derivative
                for derivative, adjustment_derivative in zip(
                    central_delta_energy_d_n,
                    adjustment_grid_energy_d_n,
                )
            ),
            tuple(
                energy + adjustment.energy_density
                for energy, adjustment in zip(central_delta_energy, adjustment_nodes)
            ),
            tuple(
                pressure + adjustment.pressure
                for pressure, adjustment in zip(central_delta_pressure, adjustment_nodes)
            ),
        )
    )
    total_ward_relative = max(
        _relative_residual(
            residual,
            derivative,
            3.0 * (energy + pressure),
        )
        for residual, derivative, energy, pressure in zip(
            total_ward,
            total_energy_d_n,
            total_energy,
            total_pressure,
        )
    )
    propagation_relative = max(
        _relative_residual(
            residual,
            3.0 * planck_squared * derivative,
            total_derivative,
        )
        for residual, derivative, total_derivative in zip(
            propagation,
            mode_candidate_e_squared_d_n,
            total_energy_d_n,
        )
    )
    geometry_relative = max(
        _relative_residual(mismatch, local, grid)
        for mismatch, local, grid in zip(
            geometry_mismatch,
            mode_d_log_h_d_n,
            grid_d_log_h_d_n,
        )
    )
    reference_derivative_relative = max(
        _relative_residual(residual, node.energy_density_d_n, grid)
        for residual, node, grid in zip(
            reference_derivative_consistency,
            response.reference_source_nodes,
            reference_grid_energy_d_n,
        )
    )
    split_derivative_relative = max(
        _relative_residual(residual, node.energy_density_d_n, grid)
        for residual, node, grid in zip(
            split_derivative_consistency,
            adjustment_nodes,
            adjustment_grid_energy_d_n,
        )
    )
    return _MeanFieldResponseEvaluation(
        target_e_squared=tuple(target_e_squared),
        target_e_squared_lower=tuple(target_e_squared_lower),
        target_e_squared_upper=tuple(target_e_squared_upper),
        total_energy_density=tuple(total_energy),
        total_pressure=tuple(total_pressure),
        total_energy_density_d_n=tuple(total_energy_d_n),
        total_energy_absolute_bounds=tuple(total_energy_bounds),
        total_pressure_absolute_bounds=tuple(total_pressure_bounds),
        raychaudhuri_absolute_uncertainties=tuple(ray_uncertainties),
        acceleration_lower_bounds=tuple(acceleration_lower),
        acceleration_upper_bounds=tuple(acceleration_upper),
        reference_derivative_consistency_residuals=(
            reference_derivative_consistency
        ),
        split_derivative_consistency_residuals=split_derivative_consistency,
        reference_ward_residuals=tuple(reference_ward),
        state_difference_ward_residuals=tuple(state_ward),
        total_ward_residuals=tuple(total_ward),
        friedmann_residuals=tuple(friedmann),
        raychaudhuri_residuals=tuple(raychaudhuri),
        constraint_propagation_residuals=tuple(propagation),
        maximum_fixed_point_relative_residual=fixed_relative,
        maximum_friedmann_relative_residual=friedmann_relative,
        maximum_raychaudhuri_relative_residual=ray_relative,
        maximum_reference_ward_relative_residual=reference_ward_relative,
        maximum_state_difference_ward_relative_residual=state_ward_relative,
        maximum_total_ward_relative_residual=total_ward_relative,
        maximum_constraint_propagation_relative_residual=propagation_relative,
        maximum_geometry_derivative_relative_mismatch=geometry_relative,
        maximum_reference_derivative_consistency_relative_residual=(
            reference_derivative_relative
        ),
        maximum_split_derivative_consistency_relative_residual=(
            split_derivative_relative
        ),
        maximum_raychaudhuri_tail_robust_relative_residual=(
            ray_tail_robust_relative
        ),
    )


def _mean_field_response_numeric_fingerprint(
    response: ModeRecomputedSemiclassicalResponse,
    evaluation: _MeanFieldResponseEvaluation,
) -> tuple[float, ...]:
    """E53 영수증이 쓰는 모든 수치 응답 채널의 지문을 만든다."""

    values: list[float] = [response.maximum_mode_wronskian_residual]
    for nodes in (
        response.reference_source_nodes,
        response.finite_reference_split_adjustment_nodes,
    ):
        for node in nodes:
            values.extend(
                (
                    node.n,
                    node.energy_density,
                    node.pressure,
                    node.energy_density_d_n,
                    node.energy_absolute_bound,
                    node.pressure_absolute_bound,
                )
            )
    for node in response.ensemble.nodes:
        stress = node.created_stress
        values.extend(
            (
                node.n,
                node.hubble_over_h0,
                node.background_d_log_h_d_n,
                stress.energy_density_over_h0_four,
                stress.pressure_over_h0_four,
                stress.energy_external_ir_uv_remainder_absolute_bound,
                stress.pressure_external_ir_uv_remainder_absolute_bound,
            )
        )
    for sequence in (
        evaluation.target_e_squared,
        evaluation.target_e_squared_lower,
        evaluation.target_e_squared_upper,
        evaluation.total_energy_density,
        evaluation.total_pressure,
        evaluation.total_energy_density_d_n,
        evaluation.total_energy_absolute_bounds,
        evaluation.total_pressure_absolute_bounds,
        evaluation.raychaudhuri_absolute_uncertainties,
        evaluation.acceleration_lower_bounds,
        evaluation.acceleration_upper_bounds,
        evaluation.reference_derivative_consistency_residuals,
        evaluation.split_derivative_consistency_residuals,
        evaluation.reference_ward_residuals,
        evaluation.state_difference_ward_residuals,
        evaluation.total_ward_residuals,
        evaluation.friedmann_residuals,
        evaluation.raychaudhuri_residuals,
        evaluation.constraint_propagation_residuals,
    ):
        values.extend(sequence)
    return tuple(_finite("mean-field response fingerprint value", value) for value in values)


def _maximum_relative_fingerprint_difference(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    if len(first) != len(second):
        raise ValueError("mean-field response fingerprint length changed")
    return max(
        (
            abs(left - right) / max(1.0, abs(left), abs(right))
            for left, right in zip(first, second)
        ),
        default=0.0,
    )


def solve_squeezed_state_difference_mean_field_fixed_point(
    initial_background: MeanFieldFLRWBackground,
    *,
    recompute_response: Callable[
        [MeanFieldFLRWBackground, int],
        ModeRecomputedSemiclassicalResponse,
    ],
    reduced_planck_over_h0: float,
    degeneracy: int = 1,
    damping: float = 0.7,
    maximum_iterations: int = 24,
    fixed_point_relative_tolerance: float = 1.0e-8,
    constraint_absolute_tolerance: float = 1.0e-8,
    constraint_relative_tolerance: float = 0.1,
    ward_absolute_tolerance: float = 1.0e-8,
    ward_relative_tolerance: float = 0.1,
    synchronization_tolerance: float = 1.0e-8,
    response_reproducibility_tolerance: float = 1.0e-10,
    geometry_derivative_relative_tolerance: float = 1.0e-5,
    maximum_mode_wronskian_residual: float = 1.0e-4,
    maximum_adjacent_n_step_ratio: float = 4.0,
    runaway_growth_factor: float = 1.05,
    runaway_patience: int = 3,
) -> ModeRecomputedMeanFieldFixedPoint:
    """조건부 배경-모드 평균장 고정점을 푼다.

    콜백은 후보 배경마다 모든 모드를 다시 만들고 외부에서 지정한 고전 + 재규격화 기준
    원천을 계산해야 한다. 풀이기 자체는 그 절대 기준 응력도, 4차 빼기에 필요한 빠진
    상위 척도인자 제트도 유도하지 않는다.
    """

    if not callable(recompute_response):
        raise ValueError("recompute_response must be callable")
    reduced_planck_over_h0 = _finite(
        "reduced_planck_over_h0",
        reduced_planck_over_h0,
    )
    if reduced_planck_over_h0 <= 0.0:
        raise ValueError("reduced_planck_over_h0 must be positive")
    if isinstance(degeneracy, bool) or not isinstance(degeneracy, int) or degeneracy < 1:
        raise ValueError("degeneracy must be a positive integer")
    damping = _finite("mean-field damping", damping)
    if damping <= 0.0 or damping > 1.0:
        raise ValueError("mean-field damping must lie in (0, 1]")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    if (
        isinstance(runaway_patience, bool)
        or not isinstance(runaway_patience, int)
        or runaway_patience < 1
    ):
        raise ValueError("runaway_patience must be a positive integer")
    positive_controls = (
        ("fixed_point_relative_tolerance", fixed_point_relative_tolerance),
        ("constraint_relative_tolerance", constraint_relative_tolerance),
        ("ward_relative_tolerance", ward_relative_tolerance),
        ("synchronization_tolerance", synchronization_tolerance),
        ("response_reproducibility_tolerance", response_reproducibility_tolerance),
        (
            "geometry_derivative_relative_tolerance",
            geometry_derivative_relative_tolerance,
        ),
        ("maximum_mode_wronskian_residual", maximum_mode_wronskian_residual),
        ("maximum_adjacent_n_step_ratio", maximum_adjacent_n_step_ratio),
        ("runaway_growth_factor", runaway_growth_factor),
    )
    for name, value in positive_controls:
        value = _finite(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if maximum_adjacent_n_step_ratio < 1.0:
        raise ValueError("maximum_adjacent_n_step_ratio must be at least one")
    if runaway_growth_factor <= 1.0:
        raise ValueError("runaway_growth_factor must exceed one")
    for name, value in (
        ("constraint_absolute_tolerance", constraint_absolute_tolerance),
        ("ward_absolute_tolerance", ward_absolute_tolerance),
    ):
        value = _finite(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    try:
        planck_squared = reduced_planck_over_h0**2
    except OverflowError as error:
        raise ValueError("reduced Planck ratio squared is not finite") from error
    planck_squared = _finite("reduced Planck ratio squared", planck_squared)

    background = initial_background
    iterations: list[MeanFieldFixedPointIteration] = []
    previous_fixed_residual: float | None = None
    consecutive_growth = 0
    response_evaluation_count = 0
    contract: tuple[object, ...] | None = None
    observed_ratios: list[float] = []

    def evaluate(at_background: MeanFieldFLRWBackground, iteration: int):
        nonlocal response_evaluation_count, contract
        response = recompute_response(at_background, iteration)
        response_evaluation_count += 1
        if not isinstance(response, ModeRecomputedSemiclassicalResponse):
            raise ValueError("recompute_response returned the wrong response type")
        evaluation = _mean_field_response_evaluation(
            at_background,
            response,
            planck_squared=planck_squared,
            degeneracy=degeneracy,
            synchronization_tolerance=synchronization_tolerance,
            ward_absolute_tolerance=ward_absolute_tolerance,
            ward_relative_tolerance=ward_relative_tolerance,
            maximum_mode_wronskian_residual=maximum_mode_wronskian_residual,
            maximum_adjacent_n_step_ratio=maximum_adjacent_n_step_ratio,
        )
        certificate = response.ensemble.bogoliubov_integrability_certificate
        assert certificate is not None
        current_contract = (
            response.ensemble.q_values,
            response.ensemble.mu,
            certificate.profile,
            response.mode_solution_count,
            response.renormalization_scheme_declaration.strip(),
            response.state_preparation_declaration.strip(),
        )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError("mode/state/renormalization response contract changed during iteration")
        return response, evaluation

    for iteration_index in range(maximum_iterations):
        response, evaluation = evaluate(background, iteration_index)
        fixed_residual = evaluation.maximum_fixed_point_relative_residual
        residual_ratio = None
        if previous_fixed_residual is not None and previous_fixed_residual > 0.0:
            residual_ratio = fixed_residual / previous_fixed_residual
            if math.isfinite(residual_ratio):
                observed_ratios.append(residual_ratio)
            if (
                residual_ratio > runaway_growth_factor
                and fixed_residual > fixed_point_relative_tolerance
            ):
                consecutive_growth += 1
            else:
                consecutive_growth = 0
            if consecutive_growth >= runaway_patience:
                raise ValueError("mean-field fixed-point iteration detected a runaway")

        current_e2 = tuple(node.e2 for node in background.active_nodes)
        updated_e2 = tuple(
            (1.0 - damping) * current + damping * target
            for current, target in zip(current_e2, evaluation.target_e_squared)
        )
        update_relative = max(
            abs(updated - current) / max(1.0, abs(updated), abs(current))
            for updated, current in zip(updated_e2, current_e2)
        )
        iterations.append(
            MeanFieldFixedPointIteration(
                iteration=iteration_index,
                maximum_fixed_point_relative_residual=fixed_residual,
                maximum_damped_update_relative=update_relative,
                minimum_target_e_squared_lower=min(
                    evaluation.target_e_squared_lower
                ),
                maximum_friedmann_relative_residual=(
                    evaluation.maximum_friedmann_relative_residual
                ),
                maximum_raychaudhuri_relative_residual=(
                    evaluation.maximum_raychaudhuri_relative_residual
                ),
                maximum_reference_ward_relative_residual=(
                    evaluation.maximum_reference_ward_relative_residual
                ),
                maximum_state_difference_ward_relative_residual=(
                    evaluation.maximum_state_difference_ward_relative_residual
                ),
                maximum_total_ward_relative_residual=(
                    evaluation.maximum_total_ward_relative_residual
                ),
                maximum_constraint_propagation_relative_residual=(
                    evaluation.maximum_constraint_propagation_relative_residual
                ),
                maximum_geometry_derivative_relative_mismatch=(
                    evaluation.maximum_geometry_derivative_relative_mismatch
                ),
                maximum_reference_derivative_consistency_relative_residual=(
                    evaluation.maximum_reference_derivative_consistency_relative_residual
                ),
                maximum_split_derivative_consistency_relative_residual=(
                    evaluation.maximum_split_derivative_consistency_relative_residual
                ),
                maximum_raychaudhuri_tail_robust_relative_residual=(
                    evaluation.maximum_raychaudhuri_tail_robust_relative_residual
                ),
                maximum_raychaudhuri_absolute_uncertainty=max(
                    evaluation.raychaudhuri_absolute_uncertainties
                ),
                minimum_acceleration_lower_bound=min(
                    evaluation.acceleration_lower_bounds
                ),
                maximum_acceleration_upper_bound=max(
                    evaluation.acceleration_upper_bounds
                ),
                maximum_mode_wronskian_residual=(
                    response.maximum_mode_wronskian_residual
                ),
                empirical_residual_ratio=residual_ratio,
            )
        )

        if fixed_residual <= fixed_point_relative_tolerance:
            final_response, final_evaluation = evaluate(
                background,
                iteration_index + 1,
            )
            reproducibility = _maximum_relative_fingerprint_difference(
                _mean_field_response_numeric_fingerprint(response, evaluation),
                _mean_field_response_numeric_fingerprint(
                    final_response,
                    final_evaluation,
                ),
            )
            if reproducibility > response_reproducibility_tolerance:
                raise ValueError("final response is not reproducible on the same background")
            if (
                final_evaluation.maximum_fixed_point_relative_residual
                > fixed_point_relative_tolerance
            ):
                raise ValueError("final recomputation left the fixed-point tolerance")

            active_nodes = background.active_nodes
            mode_d_log = tuple(
                background.d_log_h_d_n(node.n) for node in active_nodes
            )
            mode_e2_d_n = tuple(
                2.0 * node.e2 * d_log
                for node, d_log in zip(active_nodes, mode_d_log)
            )
            for (
                node,
                derivative,
                d_log,
                total_r,
                total_p,
                total_r_d_n,
                ray_uncertainty,
            ) in zip(
                active_nodes,
                mode_e2_d_n,
                mode_d_log,
                final_evaluation.total_energy_density,
                final_evaluation.total_pressure,
                final_evaluation.total_energy_density_d_n,
                final_evaluation.raychaudhuri_absolute_uncertainties,
            ):
                friedmann_residual = 3.0 * planck_squared * node.e2 - total_r
                ray_residual = node.e2 * d_log + (
                    total_r + total_p
                ) / (2.0 * planck_squared)
                propagation_residual = 3.0 * planck_squared * derivative - total_r_d_n
                for name, residual, scales in (
                    (
                        "final Friedmann",
                        friedmann_residual,
                        (3.0 * planck_squared * node.e2, total_r),
                    ),
                    (
                        "final Raychaudhuri",
                        ray_residual,
                        (
                            node.e2 * d_log,
                            (total_r + total_p) / (2.0 * planck_squared),
                        ),
                    ),
                    (
                        "final constraint propagation",
                        propagation_residual,
                        (3.0 * planck_squared * derivative, total_r_d_n),
                    ),
                ):
                    if not _residual_within_absolute_relative_tolerance(
                        residual,
                        *scales,
                        absolute_tolerance=constraint_absolute_tolerance,
                        relative_tolerance=constraint_relative_tolerance,
                    ):
                        raise ValueError(f"{name} residual exceeds its ceiling")
                if not _residual_within_absolute_relative_tolerance(
                    abs(ray_residual) + ray_uncertainty,
                    node.e2 * d_log,
                    (total_r + total_p) / (2.0 * planck_squared),
                    absolute_tolerance=constraint_absolute_tolerance,
                    relative_tolerance=constraint_relative_tolerance,
                ):
                    raise ValueError(
                        "final Raychaudhuri residual plus independent tail rectangle "
                        "exceeds its ceiling"
                    )
            if (
                final_evaluation.maximum_geometry_derivative_relative_mismatch
                > geometry_derivative_relative_tolerance
            ):
                raise ValueError("final background derivative representations disagree")

            return ModeRecomputedMeanFieldFixedPoint(
                background=background,
                final_response=final_response,
                iterations=tuple(iterations),
                reduced_planck_over_h0=reduced_planck_over_h0,
                damping=damping,
                response_evaluation_count=response_evaluation_count,
                maximum_final_fixed_point_relative_residual=(
                    final_evaluation.maximum_fixed_point_relative_residual
                ),
                maximum_final_friedmann_relative_residual=(
                    final_evaluation.maximum_friedmann_relative_residual
                ),
                maximum_final_raychaudhuri_relative_residual=(
                    final_evaluation.maximum_raychaudhuri_relative_residual
                ),
                maximum_final_reference_ward_relative_residual=(
                    final_evaluation.maximum_reference_ward_relative_residual
                ),
                maximum_final_state_difference_ward_relative_residual=(
                    final_evaluation.maximum_state_difference_ward_relative_residual
                ),
                maximum_final_total_ward_relative_residual=(
                    final_evaluation.maximum_total_ward_relative_residual
                ),
                maximum_final_constraint_propagation_relative_residual=(
                    final_evaluation.maximum_constraint_propagation_relative_residual
                ),
                maximum_final_geometry_derivative_relative_mismatch=(
                    final_evaluation.maximum_geometry_derivative_relative_mismatch
                ),
                maximum_final_reference_derivative_consistency_relative_residual=(
                    final_evaluation.maximum_reference_derivative_consistency_relative_residual
                ),
                maximum_final_split_derivative_consistency_relative_residual=(
                    final_evaluation.maximum_split_derivative_consistency_relative_residual
                ),
                maximum_final_raychaudhuri_tail_robust_relative_residual=(
                    final_evaluation.maximum_raychaudhuri_tail_robust_relative_residual
                ),
                maximum_final_raychaudhuri_absolute_uncertainty=max(
                    final_evaluation.raychaudhuri_absolute_uncertainties
                ),
                minimum_final_acceleration_lower_bound=min(
                    final_evaluation.acceleration_lower_bounds
                ),
                maximum_final_acceleration_upper_bound=max(
                    final_evaluation.acceleration_upper_bounds
                ),
                final_response_reproducibility_relative_residual=reproducibility,
                maximum_observed_empirical_residual_ratio=(
                    max(observed_ratios) if observed_ratios else 0.0
                ),
                geometry_derivative_relative_tolerance=(
                    geometry_derivative_relative_tolerance
                ),
                mass_dimension_manifest=(
                    ("N_and_E_squared", 0.0),
                    ("q_mu_and_bogoliubov_profile", 0.0),
                    ("Mbar_Pl_over_H0", 0.0),
                    ("rho_pressure_and_source_over_H0_four", 0.0),
                    ("fixed_point_and_constraint_residuals", 0.0),
                ),
                dimensions_pass=True,
            )

        background = background.with_active_e2(updated_e2)
        previous_fixed_residual = fixed_residual

    raise ValueError("mean-field fixed-point iteration did not converge")
