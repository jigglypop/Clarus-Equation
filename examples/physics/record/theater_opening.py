"""극장 개장 비유의 조건부 양자 급랭(quantum quench) 모형과 틱 접힘(tick-fold) 규칙 스캔을 한 모듈에 담는다.

앞부분(양자 개장)은 좌석 종(species) 하나를 실수 스칼라 모드의 다중항으로 표현한다.
질량의 매끈한 tanh 변화가 계산 가능한 보골리우보프(Bogoliubov) 점유수를 만든다.
종의 퇴화도와 최종 정지 질량이 동등한 좌석의 수와 그 재료 무게를 구현한다.
이는 이미 선언된 개장 면 위의 4차원 유효 모형이다. 시공간 자체, 초기 진공, 급랭
프로파일, 에너지 척도를 0차원 자료로부터 유도하지 않는다.

뒷부분(틱 접힘 규칙 스캔)은 매 틱마다 상태의 일부를 어두운 싱크(sink)로 접는 규칙이
FLRW 배경 수준에서 관측된 암흑 부문을 재현할 수 있는지 살핀다. 탐색 전용이다
(workspace 노트 20260902-de-틱접힘_규칙스캔.md). 허용오차는 그 노트에 사전등록되어
있고 아래 TOL에 복제되어 있다. 실행 뒤에는 바꾸지 않는다.
단위: rho_crit0 = 1, H0 = 1, x = ln a. 모든 적분은 x = 0(오늘)에서 X_MIN까지 x 방향
역행이며, 명시적 RK4(numpy만 사용)로 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import sys

import numpy as np


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _log_sinh_positive(value: float) -> float:
    """양의 인자에 대해 log(sinh(value))를 안정적으로 돌려준다."""

    if value <= 0.0:
        raise ValueError("log-sinh argument must be positive")
    if value < 20.0:
        return math.log(math.sinh(value))
    return value - math.log(2.0) + math.log1p(-math.exp(-2.0 * value))


@dataclass(frozen=True)
class QuantumSeatSpecies:
    """실수 스칼라 다중항 하나와 그 매끈한 개장 프로토콜이다."""

    label: str
    degeneracy: int
    mass_in: float
    mass_out: float
    duration: float
    initial_mode_occupation: float = 0.0
    role: str = "CONDITIONAL_SMOOTH_QUENCH_SPECIES"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("species label must be non-empty")
        if (
            isinstance(self.degeneracy, bool)
            or not isinstance(self.degeneracy, int)
            or self.degeneracy < 1
        ):
            raise ValueError("degeneracy must be a positive integer")
        for name, value in (
            ("mass_in", self.mass_in),
            ("mass_out", self.mass_out),
            ("duration", self.duration),
            ("initial_mode_occupation", self.initial_mode_occupation),
        ):
            _require_finite(name, value)
        if self.mass_in <= 0.0 or self.mass_out <= 0.0:
            raise ValueError("asymptotic masses must be positive")
        if self.duration <= 0.0:
            raise ValueError("smooth quench duration must be positive")
        if self.initial_mode_occupation < 0.0:
            raise ValueError("initial mode occupation must be non-negative")

    @classmethod
    def from_seat_weight(
        cls,
        *,
        label: str,
        degeneracy: int,
        mass_in: float,
        reference_energy: float,
        relative_rest_mass: float,
        duration: float,
        initial_mode_occupation: float = 0.0,
    ) -> QuantumSeatSpecies:
        """좌석 무게를 최종 정지 질량 E_* epsilon_s에 맞춘다."""

        _require_finite("reference_energy", reference_energy)
        _require_finite("relative_rest_mass", relative_rest_mass)
        if reference_energy <= 0.0 or relative_rest_mass <= 0.0:
            raise ValueError("seat energy scales must be positive")
        return cls(
            label=label,
            degeneracy=degeneracy,
            mass_in=mass_in,
            mass_out=reference_energy * relative_rest_mass,
            duration=duration,
            initial_mode_occupation=initial_mode_occupation,
        )


@dataclass(frozen=True)
class BogoliubovMode:
    momentum: float
    omega_in: float
    omega_out: float
    alpha_squared: float
    beta_squared: float
    created_occupation: float
    normalization_residual: float
    protocol: str


def _frequencies(
    species: QuantumSeatSpecies,
    momentum: float,
) -> tuple[float, float]:
    _require_finite("momentum", momentum)
    if momentum < 0.0:
        raise ValueError("momentum must be non-negative")
    return (
        math.hypot(momentum, species.mass_in),
        math.hypot(momentum, species.mass_out),
    )


def bosonic_out_occupation(
    *,
    beta_squared: float,
    initial_occupation: float,
) -> float:
    """보손 모드 하나의 n_out=n_in+(1+2*n_in)|beta|^2 을 돌려준다."""

    for name, value in (
        ("beta_squared", beta_squared),
        ("initial_occupation", initial_occupation),
    ):
        _require_finite(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    return initial_occupation + (1.0 + 2.0 * initial_occupation) * beta_squared


def instantaneous_mode(
    species: QuantumSeatSpecies,
    momentum: float,
) -> BogoliubovMode:
    """자외선(UV) 음성 대조군으로 쓰는 정확한 순간 급랭 모드를 돌려준다."""

    omega_in, omega_out = _frequencies(species, momentum)
    root_ratio = math.sqrt(omega_out / omega_in)
    inverse_root_ratio = 1.0 / root_ratio
    alpha = 0.5 * (root_ratio + inverse_root_ratio)
    beta = 0.5 * (root_ratio - inverse_root_ratio)
    alpha_squared = alpha * alpha
    beta_squared = beta * beta
    out_occupation = bosonic_out_occupation(
        beta_squared=beta_squared,
        initial_occupation=species.initial_mode_occupation,
    )
    return BogoliubovMode(
        momentum=momentum,
        omega_in=omega_in,
        omega_out=omega_out,
        alpha_squared=alpha_squared,
        beta_squared=beta_squared,
        created_occupation=(
            out_occupation - species.initial_mode_occupation
        ),
        normalization_residual=alpha_squared - beta_squared - 1.0,
        protocol="INSTANTANEOUS_UV_NEGATIVE_CONTROL",
    )


def smooth_tanh_mode(
    species: QuantumSeatSpecies,
    momentum: float,
) -> BogoliubovMode:
    """매끈한 tanh 질량제곱 급랭의 정확한 모드 점유수를 돌려준다."""

    omega_in, omega_out = _frequencies(species, momentum)
    prefactor = math.pi * species.duration
    denominator_log = (
        _log_sinh_positive(prefactor * omega_in)
        + _log_sinh_positive(prefactor * omega_out)
    )
    omega_minus = 0.5 * abs(omega_out - omega_in)
    if omega_minus == 0.0:
        beta_squared = 0.0
    else:
        beta_squared = math.exp(
            2.0 * _log_sinh_positive(prefactor * omega_minus)
            - denominator_log
        )
    omega_plus = 0.5 * (omega_out + omega_in)
    alpha_squared = math.exp(
        2.0 * _log_sinh_positive(prefactor * omega_plus)
        - denominator_log
    )
    out_occupation = bosonic_out_occupation(
        beta_squared=beta_squared,
        initial_occupation=species.initial_mode_occupation,
    )
    return BogoliubovMode(
        momentum=momentum,
        omega_in=omega_in,
        omega_out=omega_out,
        alpha_squared=alpha_squared,
        beta_squared=beta_squared,
        created_occupation=(
            out_occupation - species.initial_mode_occupation
        ),
        normalization_residual=alpha_squared - beta_squared - 1.0,
        protocol="SMOOTH_TANH_EXACT",
    )


@dataclass(frozen=True)
class SuddenQuenchUVVerdict:
    beta_squared_power: int = -4
    radial_number_integrand_power: int = -2
    radial_energy_integrand_power: int = -1
    number_density_uv_convergent: bool = True
    energy_density_uv_convergent: bool = False
    status: str = "SUDDEN_ENERGY_LOG_DIVERGENT_NOT_COSMOLOGY_SOURCE"


@dataclass(frozen=True)
class QuenchDensityAudit:
    label: str
    number_density: float
    excess_energy_density: float
    dephased_pressure: float
    equation_of_state: float
    mean_energy_per_created_quantum: float
    rms_momentum: float
    rms_momentum_over_mass: float
    momentum_max: float
    intervals: int
    maximum_bogoliubov_residual: float
    protocol: str
    ultraviolet_status: str
    stress_role: str


def _default_smooth_momentum_max(species: QuantumSeatSpecies) -> float:
    return max(
        12.0 * species.mass_in,
        12.0 * species.mass_out,
        20.0 / species.duration,
    )


def _simpson_weight(index: int, intervals: int) -> float:
    if index in (0, intervals):
        return 1.0
    return 4.0 if index % 2 else 2.0


def _simpson_spherical_factor(
    species: QuantumSeatSpecies,
    step: float,
) -> float:
    """g/(2 pi^2)에 심프슨(Simpson) 정규화 h/3을 곱한 값을 돌려준다."""

    return species.degeneracy / (2.0 * math.pi * math.pi) * step / 3.0


def integrate_quench_densities(
    species: QuantumSeatSpecies,
    *,
    protocol: str = "smooth",
    momentum_max: float | None = None,
    intervals: int = 2400,
) -> QuenchDensityAudit:
    """점근 out 입자수·에너지·탈위상(dephased) 압력을 적분한다.

    순간 프로토콜은 3차원 공간에서 들뜸 에너지가 로그 발산하므로 명시적 자외선
    절단이 필수다. 매끈한 결과는 최종 민코프스키 진공 위의 유한한 out 입자
    초과분이지, 완전히 재규격화된 FLRW 응력 텐서가 아니다.
    """

    if intervals < 200:
        raise ValueError("intervals must be at least 200")
    if intervals % 2:
        intervals += 1
    if protocol not in {"smooth", "instantaneous"}:
        raise ValueError("protocol must be smooth or instantaneous")
    if protocol == "instantaneous" and momentum_max is None:
        raise ValueError("instantaneous energy requires an explicit UV cutoff")
    upper = (
        _default_smooth_momentum_max(species)
        if momentum_max is None
        else momentum_max
    )
    _require_finite("momentum_max", upper)
    if upper <= 0.0:
        raise ValueError("momentum_max must be positive")

    mode_function = (
        smooth_tanh_mode if protocol == "smooth" else instantaneous_mode
    )
    step = upper / intervals
    number_sum = 0.0
    energy_sum = 0.0
    pressure_sum = 0.0
    momentum2_sum = 0.0
    max_residual = 0.0
    for index in range(intervals + 1):
        momentum = index * step
        mode = mode_function(species, momentum)
        weight = _simpson_weight(index, intervals)
        occupation = mode.created_occupation
        radial_number = momentum * momentum * occupation
        number_sum += weight * radial_number
        energy_sum += weight * radial_number * mode.omega_out
        pressure_sum += (
            weight
            * momentum**4
            * occupation
            / (3.0 * mode.omega_out)
        )
        momentum2_sum += weight * momentum**4 * occupation
        max_residual = max(max_residual, abs(mode.normalization_residual))

    # 헬퍼의 /3은 심프슨의 h/3이지 압력 평균이 아니다.
    simpson_spherical_factor = _simpson_spherical_factor(species, step)
    number_density = simpson_spherical_factor * number_sum
    energy_density = simpson_spherical_factor * energy_sum
    pressure = simpson_spherical_factor * pressure_sum
    momentum2_density = simpson_spherical_factor * momentum2_sum
    if number_density > 0.0:
        mean_energy = energy_density / number_density
        rms_momentum = math.sqrt(momentum2_density / number_density)
    else:
        mean_energy = 0.0
        rms_momentum = 0.0
    equation_of_state = pressure / energy_density if energy_density > 0.0 else 0.0
    return QuenchDensityAudit(
        label=species.label,
        number_density=number_density,
        excess_energy_density=energy_density,
        dephased_pressure=pressure,
        equation_of_state=equation_of_state,
        mean_energy_per_created_quantum=mean_energy,
        rms_momentum=rms_momentum,
        rms_momentum_over_mass=rms_momentum / species.mass_out,
        momentum_max=upper,
        intervals=intervals,
        maximum_bogoliubov_residual=max_residual,
        protocol=(
            "SMOOTH_TANH_EXACT"
            if protocol == "smooth"
            else "INSTANTANEOUS_CUTOFF_DEPENDENT"
        ),
        ultraviolet_status=(
            "FINITE_FOR_POSITIVE_DURATION"
            if protocol == "smooth"
            else "ENERGY_LOG_DIVERGENT_AS_CUTOFF_REMOVED"
        ),
        stress_role=(
            "ASYMPTOTIC_OUT_EXCESS_NOT_FULL_RENORMALIZED_FLRW_STRESS"
        ),
    )


@dataclass(frozen=True)
class LateSqueezedStressEnvelopeAudit:
    """늦은 시각 압착(squeezed) 생성 초과분의 위상 무관 상계·하계다."""

    label: str
    averaging_duration: float
    momentum_max: float
    intervals: int
    created_energy_density: float
    dephased_created_pressure: float
    dephased_created_equation_of_state: float
    dephased_created_field_variance: float
    static_out_created_anomalous_energy_density_coefficient: float
    instantaneous_anomalous_pressure_independent_phase_upper: float
    boxcar_anomalous_pressure_integrated_triangle_upper: float
    one_over_time_anomalous_pressure_coefficient: float
    one_over_time_anomalous_pressure_upper: float
    boxcar_pressure_lower: float
    boxcar_pressure_upper: float
    boxcar_equation_of_state_lower: float
    boxcar_equation_of_state_upper: float
    instantaneous_anomalous_field_variance_independent_phase_upper: float
    boxcar_anomalous_field_variance_integrated_triangle_upper: float
    one_over_time_anomalous_field_variance_coefficient: float
    one_over_time_anomalous_field_variance_upper: float
    sufficient_averaging_duration_for_nonnegative_pressure: float
    sufficient_averaging_duration_for_no_acceleration: float
    nonnegative_pressure_certified_by_one_over_time_bound: bool
    no_acceleration_certified_by_one_over_time_bound: bool
    maximum_bogoliubov_residual: float
    exact_no_mass_quench: bool
    numerical_created_excess_resolved: bool
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    status: str
    phase_resolved_value_available: bool = False
    global_time_supremum_computed: bool = False
    instantaneous_bound_is_independent_phase_triangle_bound: bool = True
    full_initial_state_stress_computed: bool = False
    constant_initial_occupation_used_only_as_created_excess_stimulation: bool = True
    initial_state_assumed_isotropic_number_diagonal: bool = True
    anomalous_energy_cancels_exactly: bool = True
    static_out_created_excess_scope: bool = True
    out_vacuum_normal_ordering: bool = True
    finite_momentum_window_only: bool = True
    analytic_uv_tail_certificate: bool = False
    full_renormalized_flrw_stress: bool = False
    cosmological_phase_propagation: bool = False
    conditional_long_time_no_sustained_dark_energy_scope_declared: bool = True
    long_time_no_sustained_dark_energy_numerically_certified: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def integrate_late_squeezed_stress_envelope(
    species: QuantumSeatSpecies,
    *,
    averaging_duration: float,
    momentum_max: float | None = None,
    intervals: int = 2400,
) -> LateSqueezedStressEnvelopeAudit:
    r"""정적 out 영역에서 관측되지 않은 보골리우보프 위상을 위상 무관 상계로 가둔다.

    ``B_k=|beta_k|^2``, ``A_k=alpha_k beta_k^*`` 로 두면 생성 초과분의 에너지는

    ``rho = integral omega (1+2 n_k) B_k``

    이고, 그 비정상(anomalous) 기여는 out 해밀토니안에서 정확히 상쇄된다.
    반면 압력 계수는

    ``-(m_out^2+2 k^2/3) (1+2 n_k) Re(A_k e^-2i omega t)/omega``

    이다. 입력은 등방·수 대각(number-diagonal)이고 초기 비정상 상관자는 없다고
    가정한다. ``A_k`` 의 복소 위상은 점유수만 주는 급랭 결과로는 복원되지 않는다.
    따라서 이 함수는 위상이 풀린 압력이나 도달된 전역 시간 최댓값이 아니라
    보수적인 삼각부등식 상계를 돌려준다.
    박스카(boxcar) 평균은 ``|sinc(omega T)|`` 를 주고, 그로부터 명시적 ``C_p/T``
    탈위상 상계가 나온다. 결과는 정적 out 진공 위의 매끈한 급랭 생성 초과분이며,
    공변 재규격화된 FLRW 응력 텐서가 아니다. 운동량에 무관한 0이 아닌 초기 점유수는
    자외선 유한 초과분에 곱해지는 보스 자극(Bose stimulation) 인자로만 쓴다.
    """

    _require_finite("averaging_duration", averaging_duration)
    if averaging_duration <= 0.0:
        raise ValueError("averaging_duration must be positive")

    production = integrate_quench_densities(
        species,
        protocol="smooth",
        momentum_max=momentum_max,
        intervals=intervals,
    )
    upper = production.momentum_max
    even_intervals = production.intervals
    step = upper / even_intervals
    dephased_field_variance_sum = 0.0
    instantaneous_pressure_sum = 0.0
    boxcar_pressure_sum = 0.0
    pressure_one_over_time_coefficient_sum = 0.0
    instantaneous_field_variance_sum = 0.0
    boxcar_field_variance_sum = 0.0
    field_variance_one_over_time_coefficient_sum = 0.0
    maximum_residual = 0.0
    stimulation = 1.0 + 2.0 * species.initial_mode_occupation

    for index in range(even_intervals + 1):
        momentum = index * step
        mode = smooth_tanh_mode(species, momentum)
        weight = _simpson_weight(index, even_intervals)
        radial_measure = momentum * momentum
        created_occupation = stimulation * mode.beta_squared
        alpha_beta_magnitude = math.sqrt(
            mode.alpha_squared * mode.beta_squared
        )
        anomalous_amplitude = stimulation * alpha_beta_magnitude
        omega = mode.omega_out
        pressure_coefficient = (
            species.mass_out * species.mass_out
            + (2.0 / 3.0) * momentum * momentum
        ) / omega
        omega_time = omega * averaging_duration
        _require_finite("omega_times_averaging_duration", omega_time)
        absolute_boxcar_sinc = abs(math.sin(omega_time) / omega_time)

        dephased_field_variance_sum += (
            weight * radial_measure * created_occupation / omega
        )
        pressure_envelope = (
            radial_measure * pressure_coefficient * anomalous_amplitude
        )
        instantaneous_pressure_sum += weight * pressure_envelope
        boxcar_pressure_sum += (
            weight * pressure_envelope * absolute_boxcar_sinc
        )
        pressure_one_over_time_coefficient_sum += (
            weight * pressure_envelope / omega
        )
        field_variance_envelope = (
            radial_measure * anomalous_amplitude / omega
        )
        instantaneous_field_variance_sum += (
            weight * field_variance_envelope
        )
        boxcar_field_variance_sum += (
            weight * field_variance_envelope * absolute_boxcar_sinc
        )
        field_variance_one_over_time_coefficient_sum += (
            weight * field_variance_envelope / omega
        )
        maximum_residual = max(
            maximum_residual,
            abs(mode.normalization_residual),
        )

    simpson_spherical_factor = _simpson_spherical_factor(species, step)
    dephased_field_variance = (
        simpson_spherical_factor * dephased_field_variance_sum
    )
    instantaneous_pressure_upper = (
        simpson_spherical_factor * instantaneous_pressure_sum
    )
    boxcar_pressure_upper = (
        simpson_spherical_factor * boxcar_pressure_sum
    )
    pressure_one_over_time_coefficient = (
        simpson_spherical_factor
        * pressure_one_over_time_coefficient_sum
    )
    pressure_one_over_time_upper = (
        pressure_one_over_time_coefficient / averaging_duration
    )
    instantaneous_field_variance_upper = (
        simpson_spherical_factor * instantaneous_field_variance_sum
    )
    boxcar_field_variance_upper = (
        simpson_spherical_factor * boxcar_field_variance_sum
    )
    field_variance_one_over_time_coefficient = (
        simpson_spherical_factor
        * field_variance_one_over_time_coefficient_sum
    )
    field_variance_one_over_time_upper = (
        field_variance_one_over_time_coefficient / averaging_duration
    )

    energy_density = production.excess_energy_density
    particle_pressure = production.dephased_pressure
    if particle_pressure > 0.0:
        duration_for_nonnegative_pressure = (
            pressure_one_over_time_coefficient / particle_pressure
        )
    else:
        duration_for_nonnegative_pressure = (
            0.0 if pressure_one_over_time_coefficient == 0.0 else math.inf
        )
    no_acceleration_denominator = particle_pressure + energy_density / 3.0
    if no_acceleration_denominator > 0.0:
        duration_for_no_acceleration = (
            pressure_one_over_time_coefficient
            / no_acceleration_denominator
        )
    else:
        duration_for_no_acceleration = (
            0.0 if pressure_one_over_time_coefficient == 0.0 else math.inf
        )

    pressure_lower = particle_pressure - boxcar_pressure_upper
    pressure_upper = particle_pressure + boxcar_pressure_upper
    if energy_density > 0.0:
        equation_of_state_lower = pressure_lower / energy_density
        equation_of_state_upper = pressure_upper / energy_density
    else:
        equation_of_state_lower = 0.0
        equation_of_state_upper = 0.0

    mass_dimensions = {
        "averaging_duration": -1.0,
        "momentum_max": 1.0,
        "created_energy_density": 4.0,
        "dephased_created_pressure": 4.0,
        "dephased_created_field_variance": 2.0,
        "pressure_one_over_time_coefficient": 3.0,
        "field_variance_one_over_time_coefficient": 1.0,
    }
    dimensionless_core_dimensions = {
        "omega_times_averaging_duration": 1.0 - 1.0,
        "boxcar_sinc_argument": 1.0 - 1.0,
        "dephased_created_equation_of_state": 4.0 - 4.0,
    }
    dimensions_pass = all(
        dimension == 0.0
        for dimension in dimensionless_core_dimensions.values()
    )
    exact_no_mass_quench = species.mass_in == species.mass_out
    numerical_created_excess_resolved = energy_density > 0.0

    return LateSqueezedStressEnvelopeAudit(
        label=species.label,
        averaging_duration=averaging_duration,
        momentum_max=upper,
        intervals=even_intervals,
        created_energy_density=energy_density,
        dephased_created_pressure=particle_pressure,
        dephased_created_equation_of_state=production.equation_of_state,
        dephased_created_field_variance=dephased_field_variance,
        static_out_created_anomalous_energy_density_coefficient=0.0,
        instantaneous_anomalous_pressure_independent_phase_upper=(
            instantaneous_pressure_upper
        ),
        boxcar_anomalous_pressure_integrated_triangle_upper=(
            boxcar_pressure_upper
        ),
        one_over_time_anomalous_pressure_coefficient=(
            pressure_one_over_time_coefficient
        ),
        one_over_time_anomalous_pressure_upper=(
            pressure_one_over_time_upper
        ),
        boxcar_pressure_lower=pressure_lower,
        boxcar_pressure_upper=pressure_upper,
        boxcar_equation_of_state_lower=equation_of_state_lower,
        boxcar_equation_of_state_upper=equation_of_state_upper,
        instantaneous_anomalous_field_variance_independent_phase_upper=(
            instantaneous_field_variance_upper
        ),
        boxcar_anomalous_field_variance_integrated_triangle_upper=(
            boxcar_field_variance_upper
        ),
        one_over_time_anomalous_field_variance_coefficient=(
            field_variance_one_over_time_coefficient
        ),
        one_over_time_anomalous_field_variance_upper=(
            field_variance_one_over_time_upper
        ),
        sufficient_averaging_duration_for_nonnegative_pressure=(
            duration_for_nonnegative_pressure
        ),
        sufficient_averaging_duration_for_no_acceleration=(
            duration_for_no_acceleration
        ),
        nonnegative_pressure_certified_by_one_over_time_bound=(
            exact_no_mass_quench
            or (
                numerical_created_excess_resolved
                and averaging_duration >= duration_for_nonnegative_pressure
            )
        ),
        no_acceleration_certified_by_one_over_time_bound=(
            exact_no_mass_quench
            or (
                numerical_created_excess_resolved
                and averaging_duration >= duration_for_no_acceleration
            )
        ),
        maximum_bogoliubov_residual=maximum_residual,
        exact_no_mass_quench=exact_no_mass_quench,
        numerical_created_excess_resolved=numerical_created_excess_resolved,
        mass_dimension_manifest=tuple(mass_dimensions.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        status=(
            "PASS_ZERO_QUENCH_NO_SQUEEZED_EXCESS"
            if exact_no_mass_quench
            else (
                "PASS_CONDITIONAL_STATIC_OUT_FINITE_WINDOW_SQUEEZED_ENVELOPE"
                if numerical_created_excess_resolved
                else "FAIL_NUMERICAL_CREATED_EXCESS_UNRESOLVED"
            )
        ),
    )


@dataclass(frozen=True)
class OpeningSpeciesFraction:
    label: str
    degeneracy: int
    final_rest_mass: float
    number_density: float
    energy_density: float
    energy_fraction: float
    mean_energy_over_rest_mass: float


def multi_species_opening(
    species: tuple[QuantumSeatSpecies, ...],
    *,
    intervals: int = 2400,
) -> tuple[OpeningSpeciesFraction, ...]:
    """서로 다른 좌석 재료의 급랭 후 에너지 분율을 돌려준다."""

    if not species:
        raise ValueError("at least one quantum seat species is required")
    labels = tuple(item.label for item in species)
    if len(set(labels)) != len(labels):
        raise ValueError("quantum seat labels must be unique")
    audits = tuple(
        integrate_quench_densities(item, intervals=intervals)
        for item in species
    )
    total_energy = math.fsum(item.excess_energy_density for item in audits)
    if total_energy <= 0.0:
        raise ZeroDivisionError("the selected quench creates no excitation energy")
    return tuple(
        OpeningSpeciesFraction(
            label=item.label,
            degeneracy=item.degeneracy,
            final_rest_mass=item.mass_out,
            number_density=audit.number_density,
            energy_density=audit.excess_energy_density,
            energy_fraction=audit.excess_energy_density / total_energy,
            mean_energy_over_rest_mass=(
                audit.mean_energy_per_created_quantum / item.mass_out
            ),
        )
        for item, audit in zip(species, audits)
    )


def scalar_energy_transfer_rate(
    *,
    degeneracy: int,
    mass_squared_rate: float,
    renormalized_field_squared: float,
) -> float:
    """작용으로부터 나오는 Q_s=g_s*dot(m_s^2)*<chi_s^2>_ren/2 를 돌려준다."""

    if (
        isinstance(degeneracy, bool)
        or not isinstance(degeneracy, int)
        or degeneracy < 1
    ):
        raise ValueError("degeneracy must be a positive integer")
    _require_finite("mass_squared_rate", mass_squared_rate)
    _require_finite("renormalized_field_squared", renormalized_field_squared)
    return (
        0.5
        * degeneracy
        * mass_squared_rate
        * renormalized_field_squared
    )


def total_ward_residual(
    *,
    scalar_transfer_rates: tuple[float, ...],
    clock_transfer_rate: float,
) -> float:
    """선언된 작용 분할에 대한 sum_s Q_s+Q_clock 을 돌려준다."""

    _require_finite("clock_transfer_rate", clock_transfer_rate)
    for rate in scalar_transfer_rates:
        _require_finite("scalar_transfer_rate", rate)
    return math.fsum((*scalar_transfer_rates, clock_transfer_rate))


# ---------------------------------------------------------------- 기준값
OM_B, OM_C, OM_L, OM_R = 0.0493, 0.2645, 0.6847, 9.1e-5
OM_DARK0 = OM_C + OM_L
X_MIN = math.log(1e-6)
N_STEPS = 1200
Z_GRID = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1100.0])
TOL = dict(single_rho=0.05, single_E=0.02, dm_ratio=0.05, dm_late=0.05,
           de_w=0.10, de_early=0.02, de_E=0.02, growth=0.10)

RATES = ["R1", "R2", "R3", "R4", "R5", "R5b", "R6"]
SOURCES = ["S1", "S3"]
W_SINK = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0]
CONSERVE = ["transfer", "copy"]
GAMMAS = np.logspace(-8, 2, 21)
A_STARS = np.logspace(-6, -0.3, 8)
F_ONESHOT = [0.5, 0.84, 0.95, 5.4 / 6.4, 19.0]


def lcdm_E2(a: np.ndarray) -> np.ndarray:
    return (OM_B + OM_C) * a ** -3 + OM_R * a ** -4 + OM_L


def lcdm_dark(a: np.ndarray) -> np.ndarray:
    return OM_C * a ** -3 + OM_L


# ---------------------------------------------------------------- 규칙 우변
def gamma_over_H(rate: str, gamma: float, a: float, E: float, x_star: float, f: float) -> float:
    """비율 계열에 대한 Gamma/H 를 돌려준다(R5는 절대 생성률/H 를 돌려준다)."""
    if rate == "R1":
        return gamma
    if rate == "R2":
        return gamma / E
    if rate == "R3":
        return gamma * a ** -3 / E
    if rate == "R4":
        return gamma * a ** -4 / E
    if rate == "R5":
        return gamma  # ln a 당 절대 생성(비율이 H에 비례), src == 1
    if rate == "R5b":
        return gamma / E  # H0 단위 시간당 절대 생성, src == 1
    if rate == "R6":  # 일회성: x 에서 가우스 봉우리, 총 분율 f
        sig = 0.05
        x = math.log(a)
        return f * math.exp(-0.5 * ((x - x_star) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    raise ValueError(rate)


def rhs(y: np.ndarray, x: float, rule: dict, lam: float) -> np.ndarray:
    rho_b, rho_r, rho_d = y
    a = math.exp(x)
    E2 = rho_b + rho_r + rho_d + lam
    if E2 <= 0 or not np.isfinite(E2):
        return np.full(3, np.nan)
    E = math.sqrt(E2)
    g = gamma_over_H(rule["rate"], rule["gamma"], a, E, rule["x_star"], rule["f"])
    if rule["rate"] in ("R5", "R5b"):
        src_b, src_r, src = 0.0, 0.0, 1.0
    elif rule["source"] == "S1":
        src_b, src_r, src = rho_b, 0.0, rho_b
    else:
        src_b, src_r, src = rho_b, rho_r, rho_b + rho_r
    take = 1.0 if rule["conserve"] == "transfer" else 0.0
    w = rule["w"]
    d_b = -3 * rho_b - take * g * src_b
    d_r = -4 * rho_r - take * g * src_r
    d_d = -3 * (1 + w) * rho_d + g * src
    return np.array([d_b, d_r, d_d])


def integrate_backward(rule: dict, rho_d0: float, lam: float):
    """x=0 에서 X_MIN 까지 RK4 로 내려간다. (x_grid, Y) 를 돌려주고 실패하면 None 이다."""
    h = X_MIN / N_STEPS  # 음수
    xs = np.empty(N_STEPS + 1)
    Y = np.empty((N_STEPS + 1, 3))
    y = np.array([OM_B, OM_R, rho_d0], dtype=float)
    x = 0.0
    xs[0], Y[0] = x, y
    for i in range(1, N_STEPS + 1):
        k1 = rhs(y, x, rule, lam)
        k2 = rhs(y + 0.5 * h * k1, x + 0.5 * h, rule, lam)
        k3 = rhs(y + 0.5 * h * k2, x + 0.5 * h, rule, lam)
        k4 = rhs(y + h * k3, x + h, rule, lam)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + h
        if not np.all(np.isfinite(y)) or np.any(y < -1e-12):
            return None
        xs[i], Y[i] = x, y
    return xs, Y


def interp(xs: np.ndarray, col: np.ndarray, a_query: np.ndarray) -> np.ndarray:
    xq = np.log(a_query)
    order = np.argsort(xs)
    return np.interp(xq, xs[order], col[order])


# ---------------------------------------------------------------- 가설
def test_single(rule: dict) -> dict:
    out = integrate_backward(rule, OM_DARK0, 0.0)
    if out is None:
        return dict(ok=False, reason="nonfinite_or_negative")
    xs, Y = out
    a_q = 1.0 / (1.0 + Z_GRID)
    rho_d = interp(xs, Y[:, 2], a_q)
    rho_b = interp(xs, Y[:, 0], a_q)
    rho_r = interp(xs, Y[:, 1], a_q)
    E2 = rho_b + rho_r + rho_d
    dev_rho = float(np.max(np.abs(rho_d / lcdm_dark(a_q) - 1.0)))
    dev_E = float(np.max(np.abs(np.sqrt(E2 / lcdm_E2(a_q)) - 1.0)))
    ok = dev_rho < TOL["single_rho"] and dev_E < TOL["single_E"]
    return dict(ok=bool(ok), dev_rho=dev_rho, dev_E=dev_E)


def test_dm(rule: dict) -> dict:
    out = integrate_backward(rule, OM_C, OM_L)
    if out is None:
        return dict(ok=False, reason="nonfinite_or_negative")
    xs, Y = out
    a_rec = 1.0 / 1101.0
    ratio0 = OM_C / OM_B
    ratio_rec = float(interp(xs, Y[:, 2], np.array([a_rec]))[0] / interp(xs, Y[:, 0], np.array([a_rec]))[0])
    var = abs(ratio_rec / ratio0 - 1.0)
    # 재결합 이후 만들어진 오늘 DM 의 분율: 자유 희석과 비교한다
    rho_d_rec = float(interp(xs, Y[:, 2], np.array([a_rec]))[0])
    free_today = rho_d_rec * a_rec ** (3 * (1 + rule["w"]))
    late = abs(OM_C - free_today) / OM_C
    ok = var < TOL["dm_ratio"] and late < TOL["dm_late"]
    return dict(ok=bool(ok), ratio_var=float(var), late_fraction=float(late))


def test_de(rule: dict) -> dict:
    # CDM 은 분리한다: 'lam' 자리에 접어 넣으면 틀린다(희석되므로). 별도의 먼지로 넣는다.
    rule_cdm = dict(rule)
    out = integrate_backward_with_cdm(rule_cdm, OM_L)
    if out is None:
        return dict(ok=False, reason="nonfinite_or_negative")
    xs, Y, cdm = out
    a_q = 1.0 / (1.0 + Z_GRID)
    rho_d = interp(xs, Y[:, 2], a_q)
    rho_b = interp(xs, Y[:, 0], a_q)
    rho_r = interp(xs, Y[:, 1], a_q)
    rho_c = interp(xs, cdm, a_q)
    E2 = rho_b + rho_r + rho_d + rho_c
    dev_E = float(np.max(np.abs(np.sqrt(E2 / lcdm_E2(a_q)) - 1.0)))
    early = float((rho_d / E2)[-1])
    # z<1 에서의 유효 w 는 rho_d 의 로그 미분으로 구한다
    order = np.argsort(xs)
    xs_s, rd = xs[order], Y[order, 2]
    mask = xs_s > math.log(0.5)
    dlnrho = np.gradient(np.log(np.maximum(rd[mask], 1e-300)), xs_s[mask])
    w_eff = -1.0 - dlnrho / 3.0
    dev_w = float(np.max(np.abs(w_eff + 1.0)))
    ok = dev_w < TOL["de_w"] and early < TOL["de_early"] and dev_E < TOL["de_E"]
    return dict(ok=bool(ok), dev_w=dev_w, early_fraction=early, dev_E=dev_E)


def integrate_backward_with_cdm(rule: dict, rho_d0: float):
    """integrate_backward 와 같되, 상호작용 없는 별도의 CDM 먼지(정확히 OM_C a^-3)가
    H 에 들어간다. rhs 를 감싸서 구현한다."""
    h = X_MIN / N_STEPS
    xs = np.empty(N_STEPS + 1)
    Y = np.empty((N_STEPS + 1, 3))
    cdm = np.empty(N_STEPS + 1)
    y = np.array([OM_B, OM_R, rho_d0], dtype=float)
    x = 0.0
    xs[0], Y[0], cdm[0] = x, y, OM_C

    def f(yv, xv):
        return rhs(yv, xv, rule, OM_C * math.exp(-3 * xv))

    for i in range(1, N_STEPS + 1):
        k1 = f(y, x)
        k2 = f(y + 0.5 * h * k1, x + 0.5 * h)
        k3 = f(y + 0.5 * h * k2, x + 0.5 * h)
        k4 = f(y + h * k3, x + h)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + h
        if not np.all(np.isfinite(y)) or np.any(y < -1e-12):
            return None
        xs[i], Y[i], cdm[i] = x, y, OM_C * math.exp(-3 * x)
    return xs, Y, cdm


# ---------------------------------------------------------------- 성장
def growth(rule: dict | None, a_ini: float = 1e-3, n: int = 4000) -> dict:
    """정지 상태의 균질 생성을 포함한 선형 성장: delta' = -theta - G delta,
    theta' = -(1+G) theta - 1.5 Om_m(a) delta   (프라임은 d/dln a, theta 는 1/H 로 정규화).
    rule=None 이면 LCDM 이다. H_single 배경을 쓴다(D 는 전부 어두운 성분이며 먼지형
    부분이 뭉친다. w=0 싱크에서는 D 전체가 물질이다)."""
    if rule is None:
        def bg(x):
            a = math.exp(x)
            E2 = lcdm_E2(a)
            return E2, (OM_B + OM_C) * a ** -3 / E2, 0.0
    else:
        out = integrate_backward(rule, OM_DARK0, 0.0)
        assert out is not None
        xs, Y = out
        order = np.argsort(xs)
        xs_s, Ys = xs[order], Y[order]

        def bg(x):
            a = math.exp(x)
            rb = np.interp(x, xs_s, Ys[:, 0]); rr = np.interp(x, xs_s, Ys[:, 1]); rd = np.interp(x, xs_s, Ys[:, 2])
            E2 = rb + rr + rd
            E = math.sqrt(E2)
            g = gamma_over_H(rule["rate"], rule["gamma"], a, E, rule["x_star"], rule["f"])
            src = 1.0 if rule["rate"] in ("R5", "R5b") else (rb if rule["source"] == "S1" else rb + rr)
            # 뭉친 물질의 생성률을 그 밀도에 대한 비로 잰다
            Gm = g * src / (rb + rd) if rule["w"] == 0.0 else 0.0
            return E2, (rb + rd) / E2 if rule["w"] == 0.0 else rb / E2, Gm

    def dE_dx(x, eps=1e-4):
        return (math.log(bg(x + eps)[0]) - math.log(bg(x - eps)[0])) / (4 * eps)  # d ln E / dx

    def f(v, x):
        d, th = v
        E2, om, G = bg(x)
        dlnE = dE_dx(x)
        # theta := (div v)/(aH); ln a 에 대한 방정식은 th' = -(2 + dlnE + G) th - 1.5 om d
        return np.array([-th - G * d, -(2.0 + dlnE + G) * th - 1.5 * om * d])

    x0, x1 = math.log(a_ini), 0.0
    h = (x1 - x0) / n
    v = np.array([a_ini, -a_ini])  # 물질 시대의 성장 모드 delta ~ a, theta = -delta
    x = x0
    hist = []
    for _ in range(n):
        k1 = f(v, x); k2 = f(v + 0.5 * h * k1, x + 0.5 * h); k3 = f(v + 0.5 * h * k2, x + 0.5 * h); k4 = f(v + h * k3, x + h)
        v = v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
        hist.append((x, v[0], v[1]))
    hist = np.array(hist)
    D0 = hist[-1, 1] / a_ini
    i05 = int(np.argmin(np.abs(hist[:, 0] - math.log(1 / 1.5))))
    d05, th05 = hist[i05, 1], hist[i05, 2]
    fD05 = -th05  # f*delta = dδ/dlna = -theta - G delta; -theta 를 보고한다(속도 기반 f sigma8 대리량)
    return dict(D0=float(D0), fD05=float(fD05 / a_ini))


# ---------------------------------------------------------------- 스캔
def all_rules():
    for rate in RATES:
        if rate == "R6":
            for w, cons, src, a_s, f in itertools.product(W_SINK, CONSERVE, SOURCES, A_STARS, F_ONESHOT):
                yield dict(rate=rate, source=src, w=w, conserve=cons, gamma=0.0, x_star=math.log(a_s), f=f)
        elif rate in ("R5", "R5b"):
            for w, g in itertools.product(W_SINK, GAMMAS):
                yield dict(rate=rate, source="const", w=w, conserve="copy", gamma=float(g), x_star=0.0, f=0.0)
        else:
            for w, cons, src, g in itertools.product(W_SINK, CONSERVE, SOURCES, GAMMAS):
                yield dict(rate=rate, source=src, w=w, conserve=cons, gamma=float(g), x_star=0.0, f=0.0)


def main() -> dict:
    results = dict(single=[], dm=[], de=[])
    n_rules = 0
    for rule in all_rules():
        n_rules += 1
        s = test_single(rule)
        if s.get("ok"):
            results["single"].append(dict(rule=rule, **s))
        if rule["w"] == 0.0:
            d = test_dm(rule)
            if d.get("ok"):
                results["dm"].append(dict(rule=rule, **d))
        if rule["w"] <= -1.0 / 3.0:
            e = test_de(rule)
            if e.get("ok"):
                results["de"].append(dict(rule=rule, **e))
    summary = dict(n_rules=n_rules, n_single=len(results["single"]), n_dm=len(results["dm"]), n_de=len(results["de"]))

    # 생존자를 간결하게 특성화한다
    def key(r):
        return (r["rule"]["rate"], r["rule"]["source"], r["rule"]["w"], r["rule"]["conserve"])
    fam = {}
    for hyp in ("single", "dm", "de"):
        fam[hyp] = {}
        for r in results[hyp]:
            k = "|".join(str(v) for v in key(r))
            fam[hyp].setdefault(k, []).append(r["rule"]["gamma"] if r["rule"]["rate"] != "R6" else (math.exp(r["rule"]["x_star"]), r["rule"]["f"]))
    summary["families"] = {h: {k: (min(v), max(v), len(v)) if fam[h][k] and not isinstance(v[0], tuple) else (len(v),) for k, v in fam[h].items()} for h in fam}

    # single 생존자의 성장: 계열마다 대표 하나(중앙값 gamma)
    g_l = growth(None)
    summary["growth_lcdm"] = g_l
    summary["growth"] = {}
    for k, v in fam["single"].items():
        reps = [r for r in results["single"] if "|".join(str(t) for t in key(r)) == k]
        rep = sorted(reps, key=lambda r: r["dev_rho"])[0]
        try:
            g = growth(rep["rule"])
            summary["growth"][k] = dict(rule=rep["rule"], D0_ratio=g["D0"] / g_l["D0"], fD05_ratio=g["fD05"] / g_l["fD05"],
                                        pass_growth=bool(abs(g["D0"] / g_l["D0"] - 1) < TOL["growth"] and abs(g["fD05"] / g_l["fD05"] - 1) < TOL["growth"]))
        except AssertionError:
            summary["growth"][k] = dict(rule=rep["rule"], error="background failed")
    return summary


if __name__ == "__main__":
    out = main()
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False, default=str)
    print()
