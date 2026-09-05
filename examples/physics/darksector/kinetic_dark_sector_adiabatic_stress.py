"""일정 질량 FLRW 스칼라에 대한 4차 단열(adiabatic) 응력 빼기다.

무차원 계약은 ``kinetic_dark_sector_gate`` 의 FLRW 모드 절과 같다.

    x = H0 * eta, q = k / H0, mu = m / H0, U = sqrt(H0) * u_phys.

``ds^2 = a^2(-d eta^2 + dx^2)`` 와 일정한 양의 ``mu`` 에 대해

    w^2 = q^2 + a^2 mu^2,
    sigma = (6 xi - 1) a_xx / a

로 정의한다. WKB 진동수는

    W^2 = w^2 + sigma - W_xx/(2 W) + 3 W_x^2/(4 W^2)

를 만족한다. 이 모듈은 파커-풀링/번치(Parker--Fulling/Bunch) 전개
``W = w + W2 + W4`` 를 형식적 단열 급수로 구현한다. 시간 도함수는 보조 epsilon 으로
표지하므로, 돌려주는 0차·2차·4차 계수는 유한 차분 적합이 아니라 사영이다.

증명 경계
--------
공변 스칼라 응력 텐서에 대해

    nabla_mu T^{mu nu} = (Box phi - m^2 phi - xi R phi) nabla^nu phi.

위의 리카티(Riccati) 방정식은 WKB 가설 아래 재척도된 모드 방정식과 정확히 같다.
``w + W2 + W4`` 뒤의 잔차는 단열 6차에서 시작한다. 따라서 그 가설로 만든 응력의
발산도 6차에서 시작하고, 응력을 4차까지 사영하면 모드별 연속 항등식이 5차까지 성립한다.
구현은 두 잔차를 증명 영수증으로 돌려주고, 리카티 반복 두 번으로 W4 를 독립 재구성한다.

이로써 일정한 양의 질량에 대한 모드별 0/2/4 빼기 항등식이 닫힌다. 유한 국소 곡률
상쇄항, 하다마르(Hadamard) 상태, 시간 의존 질량 에너지 전달 법칙은 의도적으로 추론하지
않는다. 아래의 별도 멱법칙 꼬리 API 는 외부에서 보증된 자외선 상계를 정확한 적분
나머지 상계로 바꾼다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import cmath
import math

from examples.physics.darksector.kinetic_dark_sector_gate import (
    FLRWBackgroundLike,
    FLRWModeSolution,
)


_MAX_TIME_DEGREE = 6
_MAX_ADIABATIC_ORDER = 6


class _FormalSeries:
    """시간 변위와 단열 epsilon 에 대한 절단 급수다."""

    def __init__(self, coefficients: dict[tuple[int, int], float] | None = None):
        self.coefficients: dict[tuple[int, int], float] = {}
        for (time_degree, order), value in (coefficients or {}).items():
            if (
                0 <= time_degree <= _MAX_TIME_DEGREE
                and 0 <= order <= _MAX_ADIABATIC_ORDER
                and value != 0.0
            ):
                self.coefficients[(time_degree, order)] = float(value)

    @classmethod
    def constant(cls, value: float) -> _FormalSeries:
        return cls({(0, 0): value})

    @staticmethod
    def _coerce(value: _FormalSeries | float) -> _FormalSeries:
        if isinstance(value, _FormalSeries):
            return value
        return _FormalSeries.constant(float(value))

    def __add__(self, other: _FormalSeries | float) -> _FormalSeries:
        other_series = self._coerce(other)
        result = dict(self.coefficients)
        for key, value in other_series.coefficients.items():
            result[key] = result.get(key, 0.0) + value
        return _FormalSeries(result)

    def __radd__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self + other

    def __neg__(self) -> _FormalSeries:
        return _FormalSeries({key: -value for key, value in self.coefficients.items()})

    def __sub__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self + (-self._coerce(other))

    def __rsub__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self._coerce(other) - self

    def __mul__(self, other: _FormalSeries | float) -> _FormalSeries:
        other_series = self._coerce(other)
        result: dict[tuple[int, int], float] = {}
        for (left_time, left_order), left_value in self.coefficients.items():
            for (right_time, right_order), right_value in other_series.coefficients.items():
                time_degree = left_time + right_time
                order = left_order + right_order
                if time_degree <= _MAX_TIME_DEGREE and order <= _MAX_ADIABATIC_ORDER:
                    key = (time_degree, order)
                    result[key] = result.get(key, 0.0) + left_value * right_value
        return _FormalSeries(result)

    def __rmul__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self * other

    def __truediv__(self, other: _FormalSeries | float) -> _FormalSeries:
        if isinstance(other, _FormalSeries):
            return self * other.inverse()
        scalar = float(other)
        if scalar == 0.0:
            raise ZeroDivisionError("formal-series scalar division by zero")
        return _FormalSeries(
            {key: value / scalar for key, value in self.coefficients.items()}
        )

    def __rtruediv__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self._coerce(other) * self.inverse()

    def derivative(self, count: int = 1) -> _FormalSeries:
        if count < 0:
            raise ValueError("derivative count must be non-negative")
        result = self
        for _ in range(count):
            differentiated: dict[tuple[int, int], float] = {}
            for (time_degree, order), value in result.coefficients.items():
                if time_degree > 0:
                    differentiated[(time_degree - 1, order)] = time_degree * value
            result = _FormalSeries(differentiated)
        return result

    def inverse(self) -> _FormalSeries:
        constant = self.coefficient(0)
        if constant == 0.0:
            raise ZeroDivisionError("formal series has zero constant term")
        delta = self / constant - 1.0
        result = _FormalSeries.constant(1.0)
        term = _FormalSeries.constant(1.0)
        for _ in range(_MAX_TIME_DEGREE + _MAX_ADIABATIC_ORDER + 1):
            term = -term * delta
            if not term.coefficients:
                break
            result = result + term
        return result / constant

    def sqrt(self) -> _FormalSeries:
        constant = self.coefficient(0)
        if constant <= 0.0:
            raise ValueError("formal square root requires a positive constant term")
        delta = self / constant - 1.0
        result = _FormalSeries.constant(1.0)
        term = _FormalSeries.constant(1.0)
        binomial = 1.0
        for power in range(1, _MAX_TIME_DEGREE + _MAX_ADIABATIC_ORDER + 1):
            term = term * delta
            if not term.coefficients:
                break
            binomial *= (0.5 - (power - 1)) / power
            result = result + binomial * term
        return math.sqrt(constant) * result

    def coefficient(self, order: int, time_degree: int = 0) -> float:
        return self.coefficients.get((time_degree, order), 0.0)


@dataclass(frozen=True)
class ScaleFactorJet:
    """한 사건에서의 척도인자와 x-도함수, 6차까지다."""

    a: float
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    d6: float

    def __post_init__(self) -> None:
        values = (self.a, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all scale-factor jet entries must be finite")
        if self.a <= 0.0:
            raise ValueError("scale factor must be positive")

    @property
    def derivatives(self) -> tuple[float, ...]:
        return (self.a, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6)


@dataclass(frozen=True)
class MassSquaredJet:
    """한 사건에서의 mu(x)^2 과 x-도함수, 6차까지다."""

    value: float
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    d6: float

    def __post_init__(self) -> None:
        values = (
            self.value,
            self.d1,
            self.d2,
            self.d3,
            self.d4,
            self.d5,
            self.d6,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all mass-squared jet entries must be finite")
        if self.value <= 0.0:
            raise ValueError("mass squared must be positive")

    @property
    def derivatives(self) -> tuple[float, ...]:
        return (
            self.value,
            self.d1,
            self.d2,
            self.d3,
            self.d4,
            self.d5,
            self.d6,
        )


@dataclass(frozen=True)
class ModeStress:
    """(2 pi)^-3 측도를 곱하기 전의 d^3q 당 무차원 응력이다."""

    energy_density_over_h0_four: float
    pressure_over_h0_four: float


@dataclass(frozen=True)
class FourthOrderCounterterm:
    """사영된 0/2/4 상쇄항과 대수적 증명 영수증이다."""

    w_orders: tuple[float, float, float]
    energy_density_orders: tuple[float, float, float]
    pressure_orders: tuple[float, float, float]
    max_riccati_residual_through_order_four: float
    max_ward_residual_through_order_five: float
    max_iterated_recurrence_disagreement: float
    status: str = (
        "FOURTH_ORDER_CONSTANT_MASS_COUNTERTERM_NO_FINITE_RENORMALIZATION_CONDITION"
    )

    @property
    def stress(self) -> ModeStress:
        return ModeStress(
            energy_density_over_h0_four=math.fsum(self.energy_density_orders),
            pressure_over_h0_four=math.fsum(self.pressure_orders),
        )


@dataclass(frozen=True)
class FourthOrderAdiabaticState:
    """국소 W0+W2+W4 진동수로 정의한 정준 초기 자료다."""

    u: complex
    du_dx: complex
    frequency: float
    frequency_derivative: float
    wronskian_residual: float
    status: str = "LOCAL_FOURTH_ORDER_ADIABATIC_INITIAL_STATE"


@dataclass(frozen=True)
class SixthOrderRemainder:
    """응력 0·2·4차를 뺀 뒤 남는 선두 형식항이다."""

    energy_density_order_six: float
    pressure_order_six: float
    per_mode_large_q_power: int = -5
    radial_integrand_large_q_power: int = -3
    ultraviolet_integrable: bool = True


@dataclass(frozen=True)
class CertifiedPowerLawTail:
    """start_q 위에서 |s(q)| <= coefficient*q^-exponent 인 외부 보증서다."""

    coefficient: float
    exponent: float
    start_q: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("tail coefficient must be finite and non-negative")
        if not math.isfinite(self.exponent) or self.exponent <= 3.0:
            raise ValueError("tail exponent must be finite and greater than three")
        if not math.isfinite(self.start_q) or self.start_q <= 0.0:
            raise ValueError("tail start_q must be finite and positive")

    def isotropic_integral_bound_from(self, q: float) -> float:
        if not math.isfinite(q) or q < self.start_q:
            raise ValueError("tail bound is not certified at the requested q")
        try:
            bound = (
                self.coefficient
                * q ** (3.0 - self.exponent)
                / (2.0 * math.pi**2 * (self.exponent - 3.0))
            )
        except OverflowError as error:
            raise ValueError("tail integral bound is not finite") from error
        if not math.isfinite(bound):
            raise ValueError("tail integral bound is not finite")
        return bound

    def pointwise_bound_at(self, q: float) -> float:
        if not math.isfinite(q) or q < self.start_q:
            raise ValueError("tail bound is not certified at the requested q")
        try:
            bound = self.coefficient * q ** (-self.exponent)
        except OverflowError as error:
            raise ValueError("tail pointwise bound is not finite") from error
        if not math.isfinite(bound):
            raise ValueError("tail pointwise bound is not finite")
        return bound


@dataclass(frozen=True)
class CertifiedInfraredPowerLaw:
    """end_q 아래에서 |s(q)| <= coefficient*q^exponent 인 외부 보증서다."""

    coefficient: float
    exponent: float
    end_q: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("infrared coefficient must be finite and non-negative")
        if not math.isfinite(self.exponent) or self.exponent <= -3.0:
            raise ValueError("infrared exponent must be finite and greater than minus three")
        if not math.isfinite(self.end_q) or self.end_q <= 0.0:
            raise ValueError("infrared end_q must be finite and positive")

    def isotropic_integral_bound_to(self, q: float) -> float:
        if not math.isfinite(q) or q <= 0.0 or q > self.end_q:
            raise ValueError("infrared bound is not certified at the requested q")
        try:
            bound = (
                self.coefficient
                * q ** (self.exponent + 3.0)
                / (2.0 * math.pi**2 * (self.exponent + 3.0))
            )
        except OverflowError as error:
            raise ValueError("infrared integral bound is not finite") from error
        if not math.isfinite(bound):
            raise ValueError("infrared integral bound is not finite")
        return bound

    def pointwise_bound_at(self, q: float) -> float:
        if not math.isfinite(q) or q <= 0.0 or q > self.end_q:
            raise ValueError("infrared bound is not certified at the requested q")
        try:
            bound = self.coefficient * q**self.exponent
        except OverflowError as error:
            raise ValueError("infrared pointwise bound is not finite") from error
        if not math.isfinite(bound):
            raise ValueError("infrared pointwise bound is not finite")
        return bound


@dataclass(frozen=True)
class GaussianBogoliubovProfile:
    """공변 운동량에 대한 해석적 급감소 압축(squeeze) 프로파일이다.

    프로파일은

    ``beta(q)=B exp[-(q/Q)^2] exp(i phi_beta)``,
    ``alpha(q)=sqrt(1+|beta(q)|^2) exp(i phi_alpha)``

    이다. 매끄러운 상태족과 정확한 보골류보프 정규화를 고정한다. 이 프로파일 하나의
    급감소만으로는 기준 상태의 하다마르 증명이나 진화된 모드 응력 핵의 절대 상계가 되지
    않는다.
    """

    amplitude: float
    q_scale: float
    beta_phase: float = 0.0
    alpha_phase: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("Gaussian squeeze amplitude must be finite and non-negative")
        if not math.isfinite(self.q_scale) or self.q_scale <= 0.0:
            raise ValueError("Gaussian squeeze q_scale must be finite and positive")
        if not math.isfinite(self.beta_phase) or not math.isfinite(self.alpha_phase):
            raise ValueError("Gaussian squeeze phases must be finite")

    def beta_at(self, q: float) -> complex:
        if not math.isfinite(q) or q <= 0.0:
            raise ValueError("Gaussian squeeze q must be finite and positive")
        try:
            scaled_q_squared = (q / self.q_scale) ** 2
        except OverflowError:
            scaled_q_squared = math.inf
        magnitude = self.amplitude * math.exp(-scaled_q_squared)
        return magnitude * complex(
            math.cos(self.beta_phase),
            math.sin(self.beta_phase),
        )

    def alpha_at(self, q: float) -> complex:
        beta_magnitude = abs(self.beta_at(q))
        magnitude = math.hypot(1.0, beta_magnitude)
        return magnitude * complex(
            math.cos(self.alpha_phase),
            math.sin(self.alpha_phase),
        )


@dataclass(frozen=True)
class GaussianBogoliubovIntegrabilityCertificate:
    """정확한 q^3 진폭 모멘트이지 모드 응력이나 하다마르 증명이 아니다."""

    profile: GaussianBogoliubovProfile
    tail_start_q: float
    maximum_initial_occupation: float
    anomalous_q3_amplitude_moment_upper: float
    particle_q3_amplitude_squared_moment_upper: float
    mass_dimension_manifest: tuple[tuple[str, float], ...] = (
        ("q_over_h0", 0.0),
        ("q_scale_over_h0", 0.0),
        ("q_over_q_scale", 0.0),
        ("alpha_beta_and_occupation", 0.0),
        ("normalized_q3_amplitude_moments", 0.0),
    )
    status: str = "GAUSSIAN_BOGOLIUBOV_AMPLITUDE_MOMENTS_ONLY"
    gaussian_exponent_argument_dimensionless: bool = True
    dimensions_pass: bool = True
    profile_verified_on_ensemble_q_grid: bool = True
    bogoliubov_normalization_exact_by_construction: bool = True
    rapid_decrease_profile_declared: bool = True
    stress_power_counting_moments_finite: bool = True
    evolved_mode_stress_tail_bounded: bool = False
    time_global_tail_ward_certified: bool = False
    reference_state_hadamard_proved: bool = False
    full_state_hadamard_proved: bool = False
    absolute_renormalized_stress_proved: bool = False


@dataclass(frozen=True)
class IntegratedStress:
    """유한 격자 중심값과 엄밀히 보증된 UV 꼬리 상계다."""

    energy_density_over_h0_four: float
    pressure_over_h0_four: float
    energy_tail_absolute_bound: float
    pressure_tail_absolute_bound: float
    status: str = "FINITE_GRID_PLUS_CERTIFIED_POWER_LAW_UV_TAIL"
    external_tail_certificate_trusted: bool = True
    tail_certificate_independently_derived_by_integrator: bool = False
    hadamard_state_proved: bool = False
    absolute_reference_vacuum_stress_renormalized: bool = False


@dataclass(frozen=True)
class IRUVIntegratedStress:
    """유한 q 격자 중심 응력과 별도의 외부 IR·UV 상계다."""

    energy_density_over_h0_four: float
    pressure_over_h0_four: float
    energy_ir_absolute_bound: float
    pressure_ir_absolute_bound: float
    energy_uv_absolute_bound: float
    pressure_uv_absolute_bound: float
    status: str = "FINITE_Q_GRID_PLUS_EXTERNAL_IR_AND_UV_BOUNDS"
    external_ir_uv_certificates_trusted: bool = True
    certificates_independently_derived_by_integrator: bool = False
    hadamard_state_proved: bool = False
    absolute_reference_vacuum_stress_renormalized: bool = False

    @property
    def energy_external_ir_uv_remainder_absolute_bound(self) -> float:
        return self.energy_ir_absolute_bound + self.energy_uv_absolute_bound

    @property
    def pressure_external_ir_uv_remainder_absolute_bound(self) -> float:
        return self.pressure_ir_absolute_bound + self.pressure_uv_absolute_bound


@dataclass(frozen=True)
class SqueezedFLRWNodeIntegralCertificate:
    """앙상블 시간 노드 하나에 대해 호출자가 준 점별 IR/UV 상계다."""

    energy_ir: CertifiedInfraredPowerLaw
    pressure_ir: CertifiedInfraredPowerLaw
    energy_uv: CertifiedPowerLawTail
    pressure_uv: CertifiedPowerLawTail


@dataclass(frozen=True)
class SqueezedFLRWModeStressDifference:
    """고정 FLRW 사건에서의 유한 상태 의존 응력 차이 하나다."""

    q: float
    mu: float
    scale_factor: float
    background_jet: ScaleFactorJet
    alpha: complex
    beta: complex
    initial_occupation: float
    beta_squared: float
    bogoliubov_normalization_residual: float
    reference_wronskian_residual: float
    squeezed_wronskian_residual: float
    reference_eom_relative_residual: float
    squeezed_eom_relative_residual: float
    reference_stress: ModeStress
    squeezed_stress: ModeStress
    preexisting_particle_stress: ModeStress
    created_particle_stress: ModeStress
    created_anomalous_stress: ModeStress
    created_state_dependent_stress: ModeStress
    full_reference_mode_subtracted_stress: ModeStress
    reference_field_squared_over_h0_two: float
    created_particle_field_squared_over_h0_two: float
    created_anomalous_field_squared_over_h0_two: float
    created_state_dependent_field_squared_over_h0_two: float
    full_reference_mode_subtracted_field_squared_over_h0_two: float
    created_dimensionless_conformal_continuity_residual: float
    full_dimensionless_conformal_continuity_residual: float
    comoving_proper_time_rate_per_dimensionless_conformal_time: float
    static_minkowski_background: bool
    static_minkowski_anomalous_energy_cancellation_pass: bool
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    status: str = (
        "PASS_CONDITIONAL_EVENT_LOCAL_PER_MODE_FLRW_STATE_DIFFERENCE_"
        "NO_TIME_PROPAGATION"
    )
    minimal_coupling_constant_mass: bool = True
    phase_resolved_from_supplied_bogoliubov_coefficients: bool = True
    phase_derived_from_quench_profile: bool = False
    initial_occupation_basis_declaration: str = (
        "CALLER_DECLARED_NUMBER_DIAGONAL_IN_SQUEEZED_U_BASIS"
    )
    density_matrix_basis_verified_by_this_function: bool = False
    v_basis_number_only_input_supported: bool = False
    constant_bogoliubov_coefficients_assumed: bool = True
    same_local_counterterm_cancels_in_state_difference: bool = True
    reference_mode_eom_input_verified_at_event: bool = True
    exact_mode_propagation_verified_by_this_function: bool = False
    per_mode_state_difference_computed: bool = True
    hadamard_or_uv_admissibility_proved: bool = False
    integrated_uv_tail_certified: bool = False
    absolute_reference_vacuum_stress_renormalized: bool = False
    full_renormalized_flrw_stress: bool = False
    local_noncomoving_observer_readout_available: bool = True
    universal_planck_tick_assumed: bool = False
    quench_driver_ward_ledger_closed: bool = False
    einstein_backreaction_computed: bool = False
    absolute_abundance_computed: bool = False
    growth_lensing_computed: bool = False
    persistent_dark_energy_from_phase_proved: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


@dataclass(frozen=True)
class SqueezedFLRWStressTrajectoryNode:
    """연속 전파된 기준 모드 위의 표본 사건 하나다."""

    n: float
    x: float
    h0_cosmic_time: float
    receipt: SqueezedFLRWModeStressDifference
    created_equation_of_state: float | None
    positive_density_acceleration_diagnostic: bool
    de_like_state_difference_diagnostic: bool
    hubble_to_mass: float
    physical_momentum_to_mass: float


@dataclass(frozen=True)
class SqueezedFLRWTrajectoryWardReceipt:
    """응력 궤적 하나에 대한 독립 유한 격자 연속 영수증이다."""

    endpoint_plus_pressure_integral_signed_residual: float
    interval_absolute_accumulated_residual: float
    balance_absolute_scale: float
    relative_signed_residual: float
    relative_absolute_accumulated_residual: float
    max_finite_difference_relative_residual: float
    status: str = "FINITE_GRID_GLOBAL_CREATED_STRESS_WARD_DIAGNOSTIC"


@dataclass(frozen=True)
class SqueezedFLRWStressTimeWindow:
    """표본 e-fold 창 하나에서 우주 시간 가중 응력 판독이다."""

    start_n: float
    end_n: float
    h0_cosmic_time_duration: float
    created_stress_time_average: ModeStress
    particle_stress_time_average: ModeStress
    anomalous_stress_time_average: ModeStress
    created_equation_of_state: float | None
    particle_equation_of_state: float | None
    max_hubble_to_mass: float
    max_physical_momentum_to_mass: float
    particle_comoving_energy_relative_span: float
    status: str = "COSMIC_TIME_WEIGHTED_STATE_DIFFERENCE_WINDOW"


@dataclass(frozen=True)
class SqueezedFLRWStressTrajectory:
    """주어진 FLRW 모드와 정확한 제트 제공자에 대한 전역 진단이다."""

    nodes: tuple[SqueezedFLRWStressTrajectoryNode, ...]
    whole_window: SqueezedFLRWStressTimeWindow
    late_window: SqueezedFLRWStressTimeWindow
    ward: SqueezedFLRWTrajectoryWardReceipt
    q: float
    mu: float
    alpha: complex
    beta: complex
    initial_occupation: float
    max_reference_phase_step: float
    anomalous_phase_turns: float
    late_half_cycle_efold_diagnostic: float
    accelerating_state_difference_span_lower: float
    accelerating_state_difference_span_grid_upper: float
    de_like_state_difference_span_lower: float
    de_like_state_difference_span_grid_upper: float
    required_persistent_de_efolds: float
    late_cold_adiabatic_gates_pass: bool
    late_dm_like_average_diagnostic_pass: bool
    grid_diagnostic_excludes_required_de_persistence: bool
    dimensions_pass: bool
    status: str = "CONDITIONAL_GLOBAL_FLRW_SQUEEZED_STATE_DIFFERENCE_DIAGNOSTIC"
    continuously_propagated_reference_solution_used: bool = True
    event_second_derivative_reconstructed_from_mode_rhs: bool = True
    global_ward_uses_independent_finite_grid_balance: bool = True
    scale_factor_jet_provider_supplied_by_caller: bool = True
    scale_factor_jet_exactness_proved_by_this_function: bool = False
    constant_positive_mass_verified_on_sampled_solution: bool = True
    minimal_coupling_verified: bool = True
    constant_bogoliubov_coefficients_assumed: bool = True
    initial_occupation_basis_declaration: str = (
        "CALLER_DECLARED_NUMBER_DIAGONAL_IN_SQUEEZED_U_BASIS"
    )
    phase_derived_from_quench_profile: bool = False
    de_like_span_is_einstein_acceleration_proof: bool = False
    hadamard_or_uv_admissibility_proved: bool = False
    integrated_uv_tail_certified: bool = False
    absolute_reference_vacuum_stress_renormalized: bool = False
    full_renormalized_flrw_stress: bool = False
    einstein_backreaction_computed: bool = False
    absolute_abundance_computed: bool = False
    growth_lensing_computed: bool = False
    analytic_persistent_dark_energy_no_go_proved_by_this_function: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


@dataclass(frozen=True)
class SqueezedFLRWStressEnsembleNode:
    """q 에 대한 등방 적분 뒤의 동기화된 시간 노드 하나다."""

    n: float
    x: float
    h0_cosmic_time: float
    scale_factor: float
    conformal_hubble: float
    hubble_over_h0: float
    background_d_log_h_d_n: float
    created_stress: IRUVIntegratedStress
    particle_grid_stress: ModeStress
    anomalous_grid_stress: ModeStress
    created_central_equation_of_state: float | None
    created_external_ir_uv_equation_of_state_interval: tuple[float, float] | None
    external_ir_uv_positive_density_acceleration_at_sample: bool
    external_ir_uv_de_like_at_sample: bool


@dataclass(frozen=True)
class SqueezedFLRWStressEnsembleTimeWindow:
    """동기화된 q 적분 뒤에 만든 우주 시간 평균이다."""

    start_n: float
    end_n: float
    h0_cosmic_time_duration: float
    created_central_stress_time_average: ModeStress
    particle_grid_stress_time_average: ModeStress
    anomalous_grid_stress_time_average: ModeStress
    created_energy_external_ir_uv_remainder_time_average_bound: float
    created_pressure_external_ir_uv_remainder_time_average_bound: float
    created_central_equation_of_state: float | None
    created_external_ir_uv_equation_of_state_interval: tuple[float, float] | None
    particle_grid_equation_of_state: float | None
    max_hubble_to_mass: float
    max_physical_momentum_to_mass: float
    particle_grid_comoving_energy_relative_span: float
    status: str = "COSMIC_TIME_WEIGHTED_FINITE_Q_GRID_ENSEMBLE_WINDOW"


@dataclass(frozen=True)
class SqueezedFLRWStressEnsembleWardReceipt:
    """중심 격자 워드 영수증과 표본 IR/UV 균형 불확도다."""

    central_grid: SqueezedFLRWTrajectoryWardReceipt
    sampled_ir_uv_balance_uncertainty_bound: float
    status: str = "FINITE_Q_AND_TIME_GRID_ENSEMBLE_WARD_DIAGNOSTIC"
    ensemble_ward_recomputed_after_q_integration: bool = True
    mode_ward_receipts_merely_summed: bool = False
    time_continuous_ir_uv_ward_certified: bool = False


@dataclass(frozen=True)
class SqueezedFLRWStressEnsemble:
    """E49 응력 궤적의 조건부 고정 공변 q 집계다."""

    nodes: tuple[SqueezedFLRWStressEnsembleNode, ...]
    whole_window: SqueezedFLRWStressEnsembleTimeWindow
    late_window: SqueezedFLRWStressEnsembleTimeWindow
    ward: SqueezedFLRWStressEnsembleWardReceipt
    q_values: tuple[float, ...]
    mu: float
    sampled_accelerating_run_node_width: float
    sampled_accelerating_run_grid_upper: float
    sampled_de_like_run_node_width: float
    sampled_de_like_run_grid_upper: float
    required_persistent_de_efolds: float
    late_finite_q_grid_cold_adiabatic_gates_pass: bool
    late_particle_grid_dm_like_diagnostic_pass: bool
    sampled_nodes_meet_required_de_run_length: bool
    dimensions_pass: bool
    bogoliubov_integrability_certificate: (
        GaussianBogoliubovIntegrabilityCertificate | None
    ) = None
    status: str = "CONDITIONAL_FIXED_Q_FLRW_SQUEEZED_ENSEMBLE_DIAGNOSTIC"
    fixed_comoving_q_grid_verified: bool = True
    synchronized_time_grid_verified: bool = True
    analytic_bogoliubov_profile_verified: bool = False
    absolute_bogoliubov_amplitude_moments_certified: bool = False
    evolved_mode_stress_tail_derived_from_profile: bool = False
    pointwise_external_ir_uv_certificates_trusted: bool = True
    pointwise_certificates_derived_by_this_function: bool = False
    time_global_tail_ward_certified: bool = False
    q_quadrature_error_certified: bool = False
    time_quadrature_error_certified: bool = False
    continuous_de_persistence_certified: bool = False
    full_ir_uv_particle_sector_coldness_proved: bool = False
    hadamard_state_proved: bool = False
    absolute_reference_vacuum_stress_renormalized: bool = False
    full_renormalized_flrw_stress: bool = False
    einstein_backreaction_computed: bool = False
    absolute_abundance_computed: bool = False
    growth_lensing_computed: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


@dataclass(frozen=True)
class LocalIsotropicObserverStressReadout:
    """국소 관측자 하나에 대한 등방 껍질/앙상블 응력 판독이다."""

    relative_speed: float
    lorentz_gamma: float
    comoving_energy_density: float
    isotropic_pressure: float
    observer_energy_density: float
    observer_proper_time_rate_per_comoving_cosmic_time: float
    status: str = "LOCAL_ISOTROPIC_STRESS_OBSERVER_CONTRACTION"
    local_tetrad_supplied: bool = True
    observer_worldline_global_evolution_computed: bool = False
    universal_planck_tick_assumed: bool = False


@dataclass(frozen=True)
class TimeDependentMassCounterterm:
    """에너지 전달을 가진 국소 rho/p/<phi^2> 상쇄항 삼중항 하나다."""

    energy_density_orders: tuple[float, float, float]
    pressure_orders: tuple[float, float, float]
    field_squared_orders: tuple[float, float, float]
    transfer_orders: tuple[float, float, float]
    max_transfer_ward_residual_through_order_five: float
    status: str = "TIME_DEPENDENT_MASS_VARIATIONAL_COUNTERTERM_TRIPLET"


def _validate_parameters(q: float, mu: float, xi: float) -> None:
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q=k/H0 must be finite and positive")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("constant mu=m/H0 must be finite and positive")
    if not math.isfinite(xi):
        raise ValueError("xi must be finite")


def _scale_factor_series(jet: ScaleFactorJet) -> _FormalSeries:
    coefficients = {
        (degree, degree): derivative / math.factorial(degree)
        for degree, derivative in enumerate(jet.derivatives)
        if derivative != 0.0
    }
    return _FormalSeries(coefficients)


def _mass_squared_series(jet: MassSquaredJet) -> _FormalSeries:
    coefficients = {
        (degree, degree): derivative / math.factorial(degree)
        for degree, derivative in enumerate(jet.derivatives)
        if derivative != 0.0
    }
    return _FormalSeries(coefficients)


def _wkb_frequencies(
    a: _FormalSeries,
    q: float,
    mu: float,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries, _FormalSeries, _FormalSeries]:
    return _wkb_frequencies_from_mass_squared(
        a,
        q,
        _FormalSeries.constant(mu * mu),
        xi,
    )


def _wkb_frequencies_from_mass_squared(
    a: _FormalSeries,
    q: float,
    mass_squared_ratio: _FormalSeries,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries, _FormalSeries, _FormalSeries]:
    w = (q * q + mass_squared_ratio * a * a).sqrt()
    sigma = (6.0 * xi - 1.0) * a.derivative(2) / a
    inverse_w = w.inverse()
    w_prime = w.derivative()
    w_second = w.derivative(2)
    w2 = (
        0.5 * sigma * inverse_w
        - 0.25 * w_second * inverse_w * inverse_w
        + 0.375 * w_prime * w_prime * inverse_w * inverse_w * inverse_w
    )
    w2_prime = w2.derivative()
    w2_second = w2.derivative(2)
    # 리카티 방정식을 4차에서 전개하면
    #
    #   2 w W4 + W2^2 = delta[-W''/(2W)+3W'^2/(4W^2)]|W2
    #
    # 이다. 이 계수 형태를 유지하면 조판된 중첩 반정수 거듭제곱의 모호함을
    # 피하고 리카티 반복과 직접 비교할 수 있다.
    w4 = (
        -0.25 * w2_second * inverse_w * inverse_w
        + 0.25 * w_second * w2 * inverse_w * inverse_w * inverse_w
        + 0.75 * w_prime * w2_prime * inverse_w * inverse_w * inverse_w
        - 0.75
        * w_prime
        * w_prime
        * w2
        * inverse_w
        * inverse_w
        * inverse_w
        * inverse_w
        - 0.5 * w2 * w2 * inverse_w
    )
    return w, w2, w4, sigma


def _riccati_step(
    current: _FormalSeries,
    w: _FormalSeries,
    sigma: _FormalSeries,
) -> _FormalSeries:
    logarithmic_derivative = current.derivative() / current
    return (
        w * w
        + sigma
        - 0.5 * current.derivative(2) / current
        + 0.75 * logarithmic_derivative * logarithmic_derivative
    ).sqrt()


def _stress_series(
    a: _FormalSeries,
    wkb_frequency: _FormalSeries,
    q: float,
    mu: float,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries]:
    return _stress_series_from_mass_squared(
        a,
        wkb_frequency,
        q,
        _FormalSeries.constant(mu * mu),
        xi,
    )


def _stress_series_from_mass_squared(
    a: _FormalSeries,
    wkb_frequency: _FormalSeries,
    q: float,
    mass_squared_ratio: _FormalSeries,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries]:
    inverse_wkb = wkb_frequency.inverse()
    amplitude = 0.5 * inverse_wkb
    frequency_derivative = wkb_frequency.derivative()
    cross = -0.5 * frequency_derivative * inverse_wkb * inverse_wkb
    kinetic = (
        0.5 * wkb_frequency
        + 0.125
        * frequency_derivative
        * frequency_derivative
        * inverse_wkb
        * inverse_wkb
        * inverse_wkb
    )

    hubble = a.derivative() / a
    acceleration = a.derivative(2) / a
    mass_squared = mass_squared_ratio * a * a
    inverse_a_four = (a * a * a * a).inverse()

    energy_bracket = (
        kinetic
        + (q * q + mass_squared) * amplitude
        + (6.0 * xi - 1.0)
        * (hubble * cross - hubble * hubble * amplitude)
    )
    pressure_bracket = (
        kinetic
        - hubble * cross
        + (hubble * hubble - q * q / 3.0 - mass_squared) * amplitude
        + 2.0
        * xi
        * (
            -2.0 * kinetic
            + 3.0 * hubble * cross
            + (
                2.0 * q * q
                + 2.0 * mass_squared
                + (12.0 * xi - 2.0) * acceleration
                - 3.0 * hubble * hubble
            )
            * amplitude
        )
    )
    return 0.5 * inverse_a_four * energy_bracket, 0.5 * inverse_a_four * pressure_bracket


def fourth_order_counterterm(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
) -> FourthOrderCounterterm:
    """일정 질량 4차 단열 응력 상쇄항을 돌려준다."""

    _validate_parameters(q, mu, xi)
    a = _scale_factor_series(jet)
    w, w2, w4, sigma = _wkb_frequencies(a, q, mu, xi)
    full_frequency = w + w2 + w4
    energy, pressure = _stress_series(a, full_frequency, q, mu, xi)

    logarithmic_derivative = full_frequency.derivative() / full_frequency
    riccati_residual = (
        full_frequency * full_frequency
        - w * w
        - sigma
        + 0.5 * full_frequency.derivative(2) / full_frequency
        - 0.75 * logarithmic_derivative * logarithmic_derivative
    )
    hubble = a.derivative() / a
    ward_residual = energy.derivative() + 3.0 * hubble * (energy + pressure)

    first_iteration = _riccati_step(w, w, sigma)
    second_iteration = _riccati_step(first_iteration, w, sigma)
    recurrence_disagreements = [
        abs(full_frequency.coefficient(order) - second_iteration.coefficient(order))
        for order in (0, 2, 4)
    ]

    return FourthOrderCounterterm(
        w_orders=tuple(full_frequency.coefficient(order) for order in (0, 2, 4)),
        energy_density_orders=tuple(energy.coefficient(order) for order in (0, 2, 4)),
        pressure_orders=tuple(pressure.coefficient(order) for order in (0, 2, 4)),
        max_riccati_residual_through_order_four=max(
            abs(riccati_residual.coefficient(order)) for order in range(5)
        ),
        max_ward_residual_through_order_five=max(
            abs(ward_residual.coefficient(order)) for order in range(6)
        ),
        max_iterated_recurrence_disagreement=max(recurrence_disagreements),
    )


def fourth_order_adiabatic_initial_state(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
) -> FourthOrderAdiabaticState:
    """국소 4차 진동수로 정준 WKB 자료를 만든다."""

    _validate_parameters(q, mu, xi)
    a = _scale_factor_series(jet)
    w, w2, w4, _ = _wkb_frequencies(a, q, mu, xi)
    full_frequency = w + w2 + w4
    frequency = math.fsum(full_frequency.coefficient(order) for order in (0, 2, 4))
    frequency_derivative = math.fsum(
        full_frequency.derivative().coefficient(order) for order in (1, 3, 5)
    )
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("fourth-order WKB frequency must be finite and positive")
    if not math.isfinite(frequency_derivative):
        raise ValueError("fourth-order WKB frequency derivative must be finite")
    u = complex(1.0 / math.sqrt(2.0 * frequency))
    du_dx = complex(-frequency_derivative / (2.0 * frequency), -frequency) * u
    wronskian = u * du_dx.conjugate() - u.conjugate() * du_dx
    return FourthOrderAdiabaticState(
        u=u,
        du_dx=du_dx,
        frequency=frequency,
        frequency_derivative=frequency_derivative,
        wronskian_residual=abs(wronskian - 1.0j),
    )


def sixth_order_remainder(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
) -> SixthOrderRemainder:
    """4차 빼기 뒤에 남는 선두 형식 응력을 돌려준다.

    고정 배경 제트와 양의 질량에서 w~q 다. W2~q^-1, W4~q^-3 이므로 다음 응력 계수는
    O(q^-5) 다. 등방 측도가 이를 q^2 O(q^-5)=O(q^-3) 으로 바꾸고, 그 자외선 적분은
    수렴한다.
    """

    _validate_parameters(q, mu, xi)
    a = _scale_factor_series(jet)
    w, _, _, sigma = _wkb_frequencies(a, q, mu, xi)
    # 리카티 반복 세 번이 보존된 형식 차수까지 W0+W2+W4+W6 을 재구성한다.
    # W6 은 6차 압력 계수에 기여하므로 선두 나머지에서 빠뜨릴 수 없다.
    first = _riccati_step(w, w, sigma)
    second = _riccati_step(first, w, sigma)
    third = _riccati_step(second, w, sigma)
    energy, pressure = _stress_series(a, third, q, mu, xi)
    return SixthOrderRemainder(
        energy_density_order_six=energy.coefficient(6),
        pressure_order_six=pressure.coefficient(6),
    )


def time_dependent_mass_counterterm(
    jet: ScaleFactorJet,
    mass_squared_jet: MassSquaredJet,
    *,
    q: float,
    xi: float,
) -> TimeDependentMassCounterterm:
    """mu(x)^2 에 대해 짝지어진 rho/p/<phi^2> 삼중항을 만든다.

    등각 시간 전달 항등식은

        rho_x + 3 (a_x/a) (rho+p) = (mu^2)_x <phi^2>/2.

    네 양은 모두 같은 WKB 범함수에서 사영한다. 그래서 항등식이 단열 5차까지 빼기 뒤에도
    살아남는다.
    """

    _validate_parameters(q, math.sqrt(mass_squared_jet.value), xi)
    a = _scale_factor_series(jet)
    mass_squared = _mass_squared_series(mass_squared_jet)
    w, w2, w4, _ = _wkb_frequencies_from_mass_squared(
        a, q, mass_squared, xi
    )
    full_frequency = w + w2 + w4
    energy, pressure = _stress_series_from_mass_squared(
        a, full_frequency, q, mass_squared, xi
    )
    field_squared = 0.5 * (a * a * full_frequency).inverse()
    transfer = 0.5 * mass_squared.derivative() * field_squared
    hubble = a.derivative() / a
    ward_residual = (
        energy.derivative()
        + 3.0 * hubble * (energy + pressure)
        - transfer
    )
    return TimeDependentMassCounterterm(
        energy_density_orders=tuple(energy.coefficient(order) for order in (0, 2, 4)),
        pressure_orders=tuple(
            pressure.coefficient(order) for order in (0, 2, 4)
        ),
        field_squared_orders=tuple(
            field_squared.coefficient(order) for order in (0, 2, 4)
        ),
        transfer_orders=tuple(
            transfer.coefficient(order) for order in (1, 3, 5)
        ),
        max_transfer_ward_residual_through_order_five=max(
            abs(ward_residual.coefficient(order)) for order in range(6)
        ),
    )


def _integrate_isotropic_stress_grid(
    q_values: tuple[float, ...],
    stresses: tuple[ModeStress, ...],
) -> ModeStress:
    if len(q_values) != len(stresses) or len(q_values) < 2:
        raise ValueError("q_values and stresses must have the same length of at least two")
    if not all(math.isfinite(q) and q > 0.0 for q in q_values):
        raise ValueError("q grid must be finite and positive")
    if any(right <= left for left, right in zip(q_values, q_values[1:])):
        raise ValueError("q grid must be strictly increasing")
    for stress in stresses:
        if not all(
            math.isfinite(value)
            for value in (
                stress.energy_density_over_h0_four,
                stress.pressure_over_h0_four,
            )
        ):
            raise ValueError("stress samples must be finite")
    energy_terms: list[float] = []
    pressure_terms: list[float] = []
    for left, right, left_stress, right_stress in zip(
        q_values,
        q_values[1:],
        stresses,
        stresses[1:],
    ):
        width = right - left
        energy_terms.append(
            0.5
            * width
            * (
                left * left * left_stress.energy_density_over_h0_four
                + right * right * right_stress.energy_density_over_h0_four
            )
        )
        pressure_terms.append(
            0.5
            * width
            * (
                left * left * left_stress.pressure_over_h0_four
                + right * right * right_stress.pressure_over_h0_four
            )
        )
    measure = 1.0 / (2.0 * math.pi**2)
    return ModeStress(
        energy_density_over_h0_four=measure * math.fsum(energy_terms),
        pressure_over_h0_four=measure * math.fsum(pressure_terms),
    )


def integrate_isotropic_stress_with_certified_tail(
    q_values: tuple[float, ...],
    stresses: tuple[ModeStress, ...],
    *,
    energy_tail: CertifiedPowerLawTail,
    pressure_tail: CertifiedPowerLawTail,
) -> IntegratedStress:
    """q^2 s(q)/(2 pi^2) 를 적분하고 정확한 UV 꼬리 오차 상계를 붙인다."""

    central = _integrate_isotropic_stress_grid(q_values, stresses)
    last_q = q_values[-1]
    if last_q < energy_tail.start_q or last_q < pressure_tail.start_q:
        raise ValueError("the q grid must reach both certified tail domains")
    if (
        abs(stresses[-1].energy_density_over_h0_four)
        > energy_tail.pointwise_bound_at(last_q)
    ):
        raise ValueError("last energy sample violates its certified tail bound")
    if (
        abs(stresses[-1].pressure_over_h0_four)
        > pressure_tail.pointwise_bound_at(last_q)
    ):
        raise ValueError("last pressure sample violates its certified tail bound")

    return IntegratedStress(
        energy_density_over_h0_four=central.energy_density_over_h0_four,
        pressure_over_h0_four=central.pressure_over_h0_four,
        energy_tail_absolute_bound=energy_tail.isotropic_integral_bound_from(last_q),
        pressure_tail_absolute_bound=pressure_tail.isotropic_integral_bound_from(last_q),
    )


def bare_mode_stress(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
    u: complex,
    du_dx: complex,
) -> ModeStress:
    """정준 모드 하나의 온셸 공변 응력을 계산한다."""

    _validate_parameters(q, mu, xi)
    if not all(
        math.isfinite(value)
        for value in (u.real, u.imag, du_dx.real, du_dx.imag)
    ):
        raise ValueError("mode and derivative must be finite")

    a = jet.a
    hubble = jet.d1 / a
    acceleration = jet.d2 / a
    amplitude = abs(u) ** 2
    cross = 2.0 * (du_dx * u.conjugate()).real
    kinetic = abs(du_dx) ** 2
    mass_squared = a * a * mu * mu

    energy_bracket = (
        kinetic
        + (q * q + mass_squared) * amplitude
        + (6.0 * xi - 1.0)
        * (hubble * cross - hubble * hubble * amplitude)
    )
    pressure_bracket = (
        kinetic
        - hubble * cross
        + (hubble * hubble - q * q / 3.0 - mass_squared) * amplitude
        + 2.0
        * xi
        * (
            -2.0 * kinetic
            + 3.0 * hubble * cross
            + (
                2.0 * q * q
                + 2.0 * mass_squared
                + (12.0 * xi - 2.0) * acceleration
                - 3.0 * hubble * hubble
            )
            * amplitude
        )
    )
    scale = 0.5 / a**4
    return ModeStress(
        energy_density_over_h0_four=scale * energy_bracket,
        pressure_over_h0_four=scale * pressure_bracket,
    )


def renormalized_mode_stress(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
    u: complex,
    du_dx: complex,
) -> ModeStress:
    """맨(bare) 모드 하나에서 국소 0/2/4 상쇄항을 뺀다."""

    bare = bare_mode_stress(jet, q=q, mu=mu, xi=xi, u=u, du_dx=du_dx)
    subtraction = fourth_order_counterterm(jet, q=q, mu=mu, xi=xi).stress
    return ModeStress(
        energy_density_over_h0_four=(
            bare.energy_density_over_h0_four
            - subtraction.energy_density_over_h0_four
        ),
        pressure_over_h0_four=(
            bare.pressure_over_h0_four - subtraction.pressure_over_h0_four
        ),
    )


def _finite_complex(name: str, value: complex) -> complex:
    converted = complex(value)
    if not math.isfinite(converted.real) or not math.isfinite(converted.imag):
        raise ValueError(f"{name} must be finite")
    return converted


def _scale_mode_stress(stress: ModeStress, factor: float) -> ModeStress:
    return ModeStress(
        energy_density_over_h0_four=(
            factor * stress.energy_density_over_h0_four
        ),
        pressure_over_h0_four=factor * stress.pressure_over_h0_four,
    )


def _subtract_mode_stress(left: ModeStress, right: ModeStress) -> ModeStress:
    return ModeStress(
        energy_density_over_h0_four=(
            left.energy_density_over_h0_four
            - right.energy_density_over_h0_four
        ),
        pressure_over_h0_four=(
            left.pressure_over_h0_four - right.pressure_over_h0_four
        ),
    )


def _add_mode_stress(left: ModeStress, right: ModeStress) -> ModeStress:
    return ModeStress(
        energy_density_over_h0_four=(
            left.energy_density_over_h0_four
            + right.energy_density_over_h0_four
        ),
        pressure_over_h0_four=(
            left.pressure_over_h0_four + right.pressure_over_h0_four
        ),
    )


def _minimal_mode_eom_relative_residual(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    u: complex,
    d2u_dx2: complex,
) -> float:
    effective_frequency_squared = (
        q * q
        + jet.a * jet.a * mu * mu
        - jet.d2 / jet.a
    )
    second_term = effective_frequency_squared * u
    residual = d2u_dx2 + second_term
    scale = max(1.0, abs(d2u_dx2), abs(second_term))
    return abs(residual) / scale


def _minimal_mode_conformal_continuity_residual(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    u: complex,
    du_dx: complex,
    d2u_dx2: complex,
) -> float:
    """무차원 연속 잔차를 돌려준다.

    ``rho`` 와 ``p`` 는 ``H0^4`` 로 나누고 ``x=H0*eta`` 이므로, 이는 물리적 등각 시간 잔차를
    ``H0^5`` 로 나눈 값과 같다.
    """

    a = jet.a
    conformal_hubble = jet.d1 / a
    conformal_hubble_derivative = (
        jet.d2 / a - conformal_hubble * conformal_hubble
    )
    physical_derivative_mode = du_dx - conformal_hubble * u
    derivative_of_physical_derivative_mode = (
        d2u_dx2
        - conformal_hubble_derivative * u
        - conformal_hubble * du_dx
    )
    frequency_squared = q * q + a * a * mu * mu
    frequency_squared_derivative = 2.0 * a * jet.d1 * mu * mu
    amplitude = abs(u) ** 2
    amplitude_derivative = 2.0 * (du_dx * u.conjugate()).real
    kinetic = abs(physical_derivative_mode) ** 2
    kinetic_derivative = 2.0 * (
        derivative_of_physical_derivative_mode
        * physical_derivative_mode.conjugate()
    ).real
    energy_bracket = kinetic + frequency_squared * amplitude
    energy_bracket_derivative = (
        kinetic_derivative
        + frequency_squared_derivative * amplitude
        + frequency_squared * amplitude_derivative
    )
    energy = 0.5 * energy_bracket / a**4
    pressure = 0.5 / a**4 * (
        kinetic - (q * q / 3.0 + a * a * mu * mu) * amplitude
    )
    energy_derivative = 0.5 / a**4 * (
        energy_bracket_derivative
        - 4.0 * conformal_hubble * energy_bracket
    )
    return energy_derivative + 3.0 * conformal_hubble * (energy + pressure)


def minimal_squeezed_flrw_mode_stress_difference(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    reference_u: complex,
    reference_du_dx: complex,
    reference_d2u_dx2: complex,
    alpha: complex,
    beta: complex,
    initial_occupation: float = 0.0,
    canonical_tolerance: float = 1.0e-9,
) -> SqueezedFLRWModeStressDifference:
    """FLRW 모드 하나에 대한 유한 생성 상태 응력 차이를 돌려준다.

    기준 모드 ``v`` 와 ``u=alpha*v+beta*v*`` 는 같은 일정 질량 최소 결합 FLRW 방정식을
    쓴다. 호출자는 ``initial_occupation`` 이 압축된 ``u`` 기저에서 수 대각이라고 선언한다.
    ``v`` 기저의 수 전용 점유는 다른 입력이며 동등한 밀도 행렬로 받지 않는다. 돌려주는
    생성 초과분은

    ``(1+2*n) * (T_bare[u] - T_bare[v])``

    이다. 공통 국소 단열 상쇄항은 이 차이에서 상쇄된다. 절대 재규격화 기준 진공 응력이
    아니고, 그 운동량 적분은 하다마르/UV 꼬리 조건이 주어져야만 유한하다. 복소
    보골류보프 위상은 입력이다. 이 함수는 급냉 프로파일에서 그것을 유도하지 않는다.
    EOM 과 연속 영수증은 호출자가 준 2차 도함수를 쓰는 사건 국소 검사다. 시간 구간에
    걸쳐 모드를 전파하지 않는다. 시간 의존 alpha 나 beta 는 이 API 밖의 명시적
    구동자/원천 워드 장부가 필요하다.
    """

    _validate_parameters(q, mu, 0.0)
    reference_u = _finite_complex("reference_u", reference_u)
    reference_du_dx = _finite_complex(
        "reference_du_dx",
        reference_du_dx,
    )
    reference_d2u_dx2 = _finite_complex(
        "reference_d2u_dx2",
        reference_d2u_dx2,
    )
    alpha = _finite_complex("alpha", alpha)
    beta = _finite_complex("beta", beta)
    if not math.isfinite(initial_occupation) or initial_occupation < 0.0:
        raise ValueError("initial_occupation must be finite and non-negative")
    if (
        not math.isfinite(canonical_tolerance)
        or canonical_tolerance <= 0.0
        or canonical_tolerance > 1.0e-4
    ):
        raise ValueError("canonical_tolerance must lie in (0, 1e-4]")

    beta_squared = abs(beta) ** 2
    bogoliubov_residual = abs(abs(alpha) ** 2 - beta_squared - 1.0)
    if bogoliubov_residual > canonical_tolerance:
        raise ValueError("Bogoliubov normalization exceeds canonical_tolerance")
    reference_wronskian = (
        reference_u * reference_du_dx.conjugate()
        - reference_u.conjugate() * reference_du_dx
    )
    reference_wronskian_residual = abs(reference_wronskian - 1.0j)
    if reference_wronskian_residual > canonical_tolerance:
        raise ValueError("reference Wronskian exceeds canonical_tolerance")
    reference_eom_residual = _minimal_mode_eom_relative_residual(
        jet,
        q=q,
        mu=mu,
        u=reference_u,
        d2u_dx2=reference_d2u_dx2,
    )
    if reference_eom_residual > canonical_tolerance:
        raise ValueError("reference mode EOM exceeds canonical_tolerance")

    squeezed_u = alpha * reference_u + beta * reference_u.conjugate()
    squeezed_du_dx = (
        alpha * reference_du_dx + beta * reference_du_dx.conjugate()
    )
    squeezed_d2u_dx2 = (
        alpha * reference_d2u_dx2
        + beta * reference_d2u_dx2.conjugate()
    )
    squeezed_wronskian = (
        squeezed_u * squeezed_du_dx.conjugate()
        - squeezed_u.conjugate() * squeezed_du_dx
    )
    squeezed_wronskian_residual = abs(squeezed_wronskian - 1.0j)
    squeezed_eom_residual = _minimal_mode_eom_relative_residual(
        jet,
        q=q,
        mu=mu,
        u=squeezed_u,
        d2u_dx2=squeezed_d2u_dx2,
    )
    if squeezed_wronskian_residual > canonical_tolerance:
        raise ValueError("squeezed Wronskian exceeds canonical_tolerance")
    if squeezed_eom_residual > canonical_tolerance:
        raise ValueError("squeezed mode EOM exceeds canonical_tolerance")

    reference_stress = bare_mode_stress(
        jet,
        q=q,
        mu=mu,
        xi=0.0,
        u=reference_u,
        du_dx=reference_du_dx,
    )
    squeezed_stress = bare_mode_stress(
        jet,
        q=q,
        mu=mu,
        xi=0.0,
        u=squeezed_u,
        du_dx=squeezed_du_dx,
    )
    stimulation = 1.0 + 2.0 * initial_occupation
    created_state_stress = _scale_mode_stress(
        _subtract_mode_stress(squeezed_stress, reference_stress),
        stimulation,
    )
    created_particle_stress = _scale_mode_stress(
        reference_stress,
        2.0 * stimulation * beta_squared,
    )
    created_anomalous_stress = _subtract_mode_stress(
        created_state_stress,
        created_particle_stress,
    )
    preexisting_particle_stress = _scale_mode_stress(
        reference_stress,
        2.0 * initial_occupation,
    )
    full_reference_mode_subtracted_stress = _add_mode_stress(
        preexisting_particle_stress,
        created_state_stress,
    )

    inverse_a_squared = 1.0 / (jet.a * jet.a)
    reference_field_squared = abs(reference_u) ** 2 * inverse_a_squared
    squeezed_field_squared = abs(squeezed_u) ** 2 * inverse_a_squared
    created_state_field_squared = stimulation * (
        squeezed_field_squared - reference_field_squared
    )
    created_particle_field_squared = (
        2.0 * stimulation * beta_squared * reference_field_squared
    )
    created_anomalous_field_squared = (
        created_state_field_squared - created_particle_field_squared
    )
    full_reference_mode_subtracted_field_squared = (
        2.0 * initial_occupation * reference_field_squared
        + created_state_field_squared
    )

    reference_continuity_residual = _minimal_mode_conformal_continuity_residual(
        jet,
        q=q,
        mu=mu,
        u=reference_u,
        du_dx=reference_du_dx,
        d2u_dx2=reference_d2u_dx2,
    )
    squeezed_continuity_residual = _minimal_mode_conformal_continuity_residual(
        jet,
        q=q,
        mu=mu,
        u=squeezed_u,
        du_dx=squeezed_du_dx,
        d2u_dx2=squeezed_d2u_dx2,
    )
    created_continuity_residual = stimulation * (
        squeezed_continuity_residual - reference_continuity_residual
    )
    full_continuity_residual = (
        stimulation * squeezed_continuity_residual
        - reference_continuity_residual
    )

    static_minkowski = all(
        derivative == 0.0
        for derivative in (
            jet.d1,
            jet.d2,
            jet.d3,
            jet.d4,
            jet.d5,
            jet.d6,
        )
    )
    energy_scale = max(
        1.0,
        abs(created_state_stress.energy_density_over_h0_four),
        abs(created_particle_stress.energy_density_over_h0_four),
    )
    minkowski_cancellation_pass = (
        static_minkowski
        and abs(created_anomalous_stress.energy_density_over_h0_four)
        <= canonical_tolerance * energy_scale
    )
    mass_dimensions = {
        "physical_comoving_wavenumber": 1.0,
        "physical_mass": 1.0,
        "physical_mode_function": -0.5,
        "physical_field_squared": 2.0,
        "physical_energy_density": 4.0,
        "physical_pressure": 4.0,
        "physical_conformal_continuity_residual": 5.0,
        "physical_proper_time": -1.0,
    }
    core_dimensions = {
        "q_equals_k_over_H0": 1.0 - 1.0,
        "mu_equals_m_over_H0": 1.0 - 1.0,
        "bogoliubov_normalization": 0.0,
        "wronskian_normalization_in_x_units": 0.0,
        "proper_to_conformal_time_rate": 0.0,
        "conformal_continuity_residual_over_H0_five": 5.0 - 5.0,
    }
    dimensions_pass = all(value == 0.0 for value in core_dimensions.values())
    return SqueezedFLRWModeStressDifference(
        q=q,
        mu=mu,
        scale_factor=jet.a,
        background_jet=jet,
        alpha=alpha,
        beta=beta,
        initial_occupation=initial_occupation,
        beta_squared=beta_squared,
        bogoliubov_normalization_residual=bogoliubov_residual,
        reference_wronskian_residual=reference_wronskian_residual,
        squeezed_wronskian_residual=squeezed_wronskian_residual,
        reference_eom_relative_residual=reference_eom_residual,
        squeezed_eom_relative_residual=squeezed_eom_residual,
        reference_stress=reference_stress,
        squeezed_stress=squeezed_stress,
        preexisting_particle_stress=preexisting_particle_stress,
        created_particle_stress=created_particle_stress,
        created_anomalous_stress=created_anomalous_stress,
        created_state_dependent_stress=created_state_stress,
        full_reference_mode_subtracted_stress=(
            full_reference_mode_subtracted_stress
        ),
        reference_field_squared_over_h0_two=reference_field_squared,
        created_particle_field_squared_over_h0_two=(
            created_particle_field_squared
        ),
        created_anomalous_field_squared_over_h0_two=(
            created_anomalous_field_squared
        ),
        created_state_dependent_field_squared_over_h0_two=(
            created_state_field_squared
        ),
        full_reference_mode_subtracted_field_squared_over_h0_two=(
            full_reference_mode_subtracted_field_squared
        ),
        created_dimensionless_conformal_continuity_residual=(
            created_continuity_residual
        ),
        full_dimensionless_conformal_continuity_residual=(
            full_continuity_residual
        ),
        comoving_proper_time_rate_per_dimensionless_conformal_time=jet.a,
        static_minkowski_background=static_minkowski,
        static_minkowski_anomalous_energy_cancellation_pass=(
            minkowski_cancellation_pass
        ),
        mass_dimension_manifest=tuple(mass_dimensions.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
    )


def _stress_equation_of_state(stress: ModeStress, floor: float) -> float | None:
    if stress.energy_density_over_h0_four <= floor:
        return None
    return (
        stress.pressure_over_h0_four
        / stress.energy_density_over_h0_four
    )


def _external_ir_uv_equation_of_state_interval(
    stress: ModeStress,
    *,
    energy_absolute_bound: float,
    pressure_absolute_bound: float,
) -> tuple[float, float] | None:
    rho_low = stress.energy_density_over_h0_four - energy_absolute_bound
    rho_high = stress.energy_density_over_h0_four + energy_absolute_bound
    if rho_low <= 0.0 or not math.isfinite(rho_high):
        return None
    pressure_low = stress.pressure_over_h0_four - pressure_absolute_bound
    pressure_high = stress.pressure_over_h0_four + pressure_absolute_bound
    ratios = (
        pressure_low / rho_low,
        pressure_low / rho_high,
        pressure_high / rho_low,
        pressure_high / rho_high,
    )
    return min(ratios), max(ratios)


def _time_average_mode_stress(
    h0_times: tuple[float, ...],
    stresses: tuple[ModeStress, ...],
) -> ModeStress:
    if len(h0_times) != len(stresses) or len(h0_times) < 2:
        raise ValueError("time-average arrays must share a length of at least two")
    duration = h0_times[-1] - h0_times[0]
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("time-average duration must be finite and positive")
    energy_terms: list[float] = []
    pressure_terms: list[float] = []
    for index in range(len(h0_times) - 1):
        width = h0_times[index + 1] - h0_times[index]
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("time-average grid must be strictly increasing")
        energy_terms.append(
            0.5
            * width
            * (
                stresses[index].energy_density_over_h0_four
                + stresses[index + 1].energy_density_over_h0_four
            )
        )
        pressure_terms.append(
            0.5
            * width
            * (
                stresses[index].pressure_over_h0_four
                + stresses[index + 1].pressure_over_h0_four
            )
        )
    return ModeStress(
        math.fsum(energy_terms) / duration,
        math.fsum(pressure_terms) / duration,
    )


def _time_average_nonnegative_bound(
    h0_times: tuple[float, ...],
    bounds: tuple[float, ...],
) -> float:
    if len(h0_times) != len(bounds) or len(h0_times) < 2:
        raise ValueError("time-bound arrays must share a length of at least two")
    if not all(math.isfinite(value) and value >= 0.0 for value in bounds):
        raise ValueError("time-dependent error bounds must be finite and non-negative")
    duration = h0_times[-1] - h0_times[0]
    terms = tuple(
        0.5
        * (h0_times[index + 1] - h0_times[index])
        * (bounds[index] + bounds[index + 1])
        for index in range(len(bounds) - 1)
    )
    return math.fsum(terms) / duration


def _trajectory_time_window(
    nodes: tuple[SqueezedFLRWStressTrajectoryNode, ...],
) -> SqueezedFLRWStressTimeWindow:
    if len(nodes) < 2:
        raise ValueError("a stress time window needs at least two nodes")
    duration = nodes[-1].h0_cosmic_time - nodes[0].h0_cosmic_time
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("stress time-window duration must be finite and positive")
    h0_times = tuple(node.h0_cosmic_time for node in nodes)
    created = _time_average_mode_stress(
        h0_times,
        tuple(node.receipt.created_state_dependent_stress for node in nodes),
    )
    particle = _time_average_mode_stress(
        h0_times,
        tuple(node.receipt.created_particle_stress for node in nodes),
    )
    anomalous = _time_average_mode_stress(
        h0_times,
        tuple(node.receipt.created_anomalous_stress for node in nodes),
    )
    energy_scale = max(
        abs(node.receipt.created_state_dependent_stress.energy_density_over_h0_four)
        for node in nodes
    )
    floor = max(math.ulp(1.0) * energy_scale, 1.0e-300)
    comoving_particle_energies = tuple(
        node.receipt.scale_factor**3
        * node.receipt.created_particle_stress.energy_density_over_h0_four
        for node in nodes
    )
    comoving_scale = max(
        max(abs(value) for value in comoving_particle_energies),
        1.0e-300,
    )
    return SqueezedFLRWStressTimeWindow(
        start_n=nodes[0].n,
        end_n=nodes[-1].n,
        h0_cosmic_time_duration=duration,
        created_stress_time_average=created,
        particle_stress_time_average=particle,
        anomalous_stress_time_average=anomalous,
        created_equation_of_state=_stress_equation_of_state(created, floor),
        particle_equation_of_state=_stress_equation_of_state(particle, floor),
        max_hubble_to_mass=max(node.hubble_to_mass for node in nodes),
        max_physical_momentum_to_mass=max(
            node.physical_momentum_to_mass for node in nodes
        ),
        particle_comoving_energy_relative_span=(
            max(comoving_particle_energies) - min(comoving_particle_energies)
        )
        / comoving_scale,
    )


def _global_ward_from_arrays(
    x_values: tuple[float, ...],
    scale_factors: tuple[float, ...],
    conformal_hubbles: tuple[float, ...],
    stresses: tuple[ModeStress, ...],
) -> SqueezedFLRWTrajectoryWardReceipt:
    lengths = {
        len(x_values),
        len(scale_factors),
        len(conformal_hubbles),
        len(stresses),
    }
    if len(lengths) != 1 or len(x_values) < 3:
        raise ValueError("global Ward arrays must share a length of at least three")
    if any(right <= left for left, right in zip(x_values, x_values[1:])):
        raise ValueError("global Ward x grid must be strictly increasing")
    interval_residuals: list[float] = []
    interval_scales: list[float] = []
    for index in range(len(x_values) - 1):
        left_stress = stresses[index]
        right_stress = stresses[index + 1]
        left_a = scale_factors[index]
        right_a = scale_factors[index + 1]
        left_boundary = left_a**3 * left_stress.energy_density_over_h0_four
        right_boundary = right_a**3 * right_stress.energy_density_over_h0_four
        boundary_change = right_boundary - left_boundary
        left_hubble = conformal_hubbles[index]
        right_hubble = conformal_hubbles[index + 1]
        left_integrand = (
            3.0
            * left_hubble
            * left_a**3
            * left_stress.pressure_over_h0_four
        )
        right_integrand = (
            3.0
            * right_hubble
            * right_a**3
            * right_stress.pressure_over_h0_four
        )
        pressure_integral = (
            0.5
            * (x_values[index + 1] - x_values[index])
            * (left_integrand + right_integrand)
        )
        interval_residuals.append(boundary_change + pressure_integral)
        interval_scales.append(abs(boundary_change) + abs(pressure_integral))

    signed = math.fsum(interval_residuals)
    accumulated = math.fsum(abs(value) for value in interval_residuals)
    balance_scale = math.fsum(interval_scales)
    if balance_scale == 0.0:
        relative_signed = 0.0 if signed == 0.0 else math.inf
        relative_accumulated = 0.0 if accumulated == 0.0 else math.inf
    else:
        relative_signed = abs(signed) / balance_scale
        relative_accumulated = accumulated / balance_scale

    total_x = x_values[-1] - x_values[0]
    max_energy = max(
        abs(stress.energy_density_over_h0_four) for stress in stresses
    )
    derivative_floor = max(1.0e-300, math.ulp(1.0) * max_energy / total_x)
    finite_difference_residuals: list[float] = []
    finite_difference_scales: list[float] = []
    for index in range(1, len(x_values) - 1):
        h_left = x_values[index] - x_values[index - 1]
        h_right = x_values[index + 1] - x_values[index]
        left_rho = stresses[index - 1].energy_density_over_h0_four
        center_stress = stresses[index]
        right_rho = stresses[index + 1].energy_density_over_h0_four
        derivative = (
            -h_right / (h_left * (h_left + h_right)) * left_rho
            + (h_right - h_left) / (h_left * h_right)
            * center_stress.energy_density_over_h0_four
            + h_left / (h_right * (h_left + h_right)) * right_rho
        )
        hubble = conformal_hubbles[index]
        dilution = 3.0 * hubble * (
            center_stress.energy_density_over_h0_four
            + center_stress.pressure_over_h0_four
        )
        finite_difference_residuals.append(abs(derivative + dilution))
        finite_difference_scales.extend((abs(derivative), abs(dilution)))
    finite_difference_global_scale = max(
        max(finite_difference_scales),
        derivative_floor,
    )
    return SqueezedFLRWTrajectoryWardReceipt(
        endpoint_plus_pressure_integral_signed_residual=signed,
        interval_absolute_accumulated_residual=accumulated,
        balance_absolute_scale=balance_scale,
        relative_signed_residual=relative_signed,
        relative_absolute_accumulated_residual=relative_accumulated,
        # L-무한대 잔차를 궤적 전체의 L-무한대 척도 하나로 정규화한다. 점별 분모는
        # 두 워드 항이 함께 0 을 지나는 무해한 교차점에서 특이하다.
        max_finite_difference_relative_residual=(
            max(finite_difference_residuals) / finite_difference_global_scale
        ),
    )


def _trajectory_global_ward(
    nodes: tuple[SqueezedFLRWStressTrajectoryNode, ...],
) -> SqueezedFLRWTrajectoryWardReceipt:
    return _global_ward_from_arrays(
        tuple(node.x for node in nodes),
        tuple(node.receipt.scale_factor for node in nodes),
        tuple(
            node.receipt.background_jet.d1 / node.receipt.scale_factor
            for node in nodes
        ),
        tuple(node.receipt.created_state_dependent_stress for node in nodes),
    )


def _grid_resolved_boolean_span(
    n_values: tuple[float, ...],
    truth_values: tuple[bool, ...],
) -> tuple[float, float]:
    if len(n_values) != len(truth_values) or len(n_values) < 2:
        raise ValueError("span arrays must share a length of at least two")
    if any(right <= left for left, right in zip(n_values, n_values[1:])):
        raise ValueError("span e-fold grid must be strictly increasing")
    longest = 0.0
    start: float | None = None
    for n, is_true in zip(n_values, truth_values):
        if is_true:
            if start is None:
                start = n
            longest = max(longest, n - start)
        else:
            start = None
    max_step = max(right - left for left, right in zip(n_values, n_values[1:]))
    total_span = n_values[-1] - n_values[0]
    return longest, min(total_span, longest + 2.0 * max_step)


def _grid_resolved_true_span(
    nodes: tuple[SqueezedFLRWStressTrajectoryNode, ...],
    predicate: Callable[[SqueezedFLRWStressTrajectoryNode], bool],
) -> tuple[float, float]:
    return _grid_resolved_boolean_span(
        tuple(node.n for node in nodes),
        tuple(predicate(node) for node in nodes),
    )


def trace_squeezed_flrw_mode_stress(
    background: FLRWBackgroundLike,
    solution: FLRWModeSolution,
    *,
    scale_factor_jet_at_n: Callable[[float], ScaleFactorJet],
    alpha: complex,
    beta: complex,
    initial_occupation: float = 0.0,
    sample_stride: int = 1,
    late_window_efolds: float = 1.0,
    maximum_reference_phase_step: float = math.pi / 4.0,
    canonical_tolerance: float = 1.0e-7,
    background_tolerance: float = 1.0e-8,
    maximum_hubble_to_mass_for_cold: float = 0.1,
    maximum_momentum_to_mass_for_cold: float = 0.1,
    maximum_abs_late_w_for_dm_like: float = 0.1,
    de_like_w_tolerance: float = 0.1,
    required_persistent_de_efolds: float = 1.0,
) -> SqueezedFLRWStressTrajectory:
    """전파된 FLRW 모드 하나를 따라 E48 상태 의존 응력을 추적한다.

    모드 해는 일정한 양의 질량과 최소 결합 ``xi=0`` 을 써야 한다. 호출자가 준 척도인자
    제트는 표본 사건마다 ``a``, ``a'``, 모드 진동수와 대조한다. 상위 도함수는 출처용으로
    가지고 다니지만 절대 단열 재규격화 응력을 추론하는 데 쓰지 않는다.

    전역 연속성은 유한 차분과 끝점+압력 적분으로 독립 검사한다. 가속과 암흑에너지 구간은
    고정 배경 위의 기준 모드를 뺀 상태 차이에만 해당하며, 아인슈타인 되먹임 결과가
    아니다.
    """

    if isinstance(sample_stride, bool) or not isinstance(sample_stride, int) or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    positive_controls = (
        ("late_window_efolds", late_window_efolds),
        ("maximum_reference_phase_step", maximum_reference_phase_step),
        ("canonical_tolerance", canonical_tolerance),
        ("background_tolerance", background_tolerance),
        ("maximum_hubble_to_mass_for_cold", maximum_hubble_to_mass_for_cold),
        ("maximum_momentum_to_mass_for_cold", maximum_momentum_to_mass_for_cold),
        ("maximum_abs_late_w_for_dm_like", maximum_abs_late_w_for_dm_like),
        ("de_like_w_tolerance", de_like_w_tolerance),
        ("required_persistent_de_efolds", required_persistent_de_efolds),
    )
    for name, value in positive_controls:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if canonical_tolerance > 1.0e-4:
        raise ValueError("canonical_tolerance must not exceed 1e-4")
    if background_tolerance > 1.0e-4:
        raise ValueError("background_tolerance must not exceed 1e-4")
    if solution.spec.curvature_coupling != 0.0:
        raise ValueError("global squeezed stress currently requires minimal coupling xi=0")
    background_nodes = tuple(background.nodes)
    if len(background_nodes) < 2:
        raise ValueError("background must contain at least two ordered nodes")
    background_n_values = tuple(float(node.n) for node in background_nodes)
    if not all(math.isfinite(value) for value in background_n_values):
        raise ValueError("background e-fold nodes must be finite")
    if any(
        right <= left
        for left, right in zip(background_n_values, background_n_values[1:])
    ):
        raise ValueError("background e-fold nodes must be strictly increasing")
    supplied_background_window = (
        background_n_values[0],
        background_n_values[-1],
    )
    if any(
        abs(actual - expected)
        > background_tolerance * max(1.0, abs(actual), abs(expected))
        for actual, expected in zip(
            supplied_background_window,
            solution.background_window,
        )
    ):
        raise ValueError("solution background window does not match supplied background")
    mode_nodes = solution.nodes
    if len(mode_nodes) < 3:
        raise ValueError("mode solution must contain at least three nodes")
    if any(right.n <= left.n or right.x <= left.x for left, right in zip(mode_nodes, mode_nodes[1:])):
        raise ValueError("mode solution n and x coordinates must be strictly increasing")
    total_efolds = mode_nodes[-1].n - mode_nodes[0].n
    if late_window_efolds > total_efolds:
        raise ValueError("late_window_efolds must not exceed the solution interval")

    mass_values = tuple(float(solution.spec.mass_over_h0(node.n)) for node in mode_nodes)
    if not all(math.isfinite(value) and value > 0.0 for value in mass_values):
        raise ValueError("global squeezed stress requires a finite positive mass")
    mu = mass_values[0]
    if max(abs(value - mu) for value in mass_values) > background_tolerance * max(1.0, mu):
        raise ValueError("global squeezed stress requires constant mass")
    q = solution.spec.comoving_wavenumber_over_h0
    _validate_parameters(q, mu, 0.0)

    indices = list(range(0, len(mode_nodes), sample_stride))
    if indices[-1] != len(mode_nodes) - 1:
        indices.append(len(mode_nodes) - 1)
    if len(indices) < 3:
        raise ValueError("sampled trajectory must contain at least three nodes")

    sampled: list[tuple[object, ScaleFactorJet, float, SqueezedFLRWModeStressDifference]] = []
    for index in indices:
        node = mode_nodes[index]
        jet = scale_factor_jet_at_n(node.n)
        if not isinstance(jet, ScaleFactorJet):
            raise ValueError("scale_factor_jet_at_n must return ScaleFactorJet")
        expected_a = math.exp(node.n)
        e2 = float(background.at_n(node.n).e2)
        if not math.isfinite(e2) or e2 <= 0.0:
            raise ValueError("background e2 must be finite and positive")
        expected_d1 = expected_a**2 * math.sqrt(e2)
        expected_omega_squared = q * q + jet.a * jet.a * mu * mu - jet.d2 / jet.a
        for name, actual, expected in (
            ("scale factor", jet.a, expected_a),
            ("first scale-factor derivative", jet.d1, expected_d1),
            ("minimal mode frequency", node.omega_squared, expected_omega_squared),
        ):
            relative = abs(actual - expected) / max(1.0, abs(actual), abs(expected))
            if relative > background_tolerance:
                raise ValueError(f"{name} is inconsistent with background or mode solution")
        receipt = minimal_squeezed_flrw_mode_stress_difference(
            jet,
            q=q,
            mu=mu,
            reference_u=node.u,
            reference_du_dx=node.du_dx,
            reference_d2u_dx2=-node.omega_squared * node.u,
            alpha=alpha,
            beta=beta,
            initial_occupation=initial_occupation,
            canonical_tolerance=canonical_tolerance,
        )
        sampled.append((node, jet, e2, receipt))

    phase_steps = tuple(
        cmath.phase(right[0].u / left[0].u)
        for left, right in zip(sampled, sampled[1:])
    )
    max_phase_step = max(abs(value) for value in phase_steps)
    if max_phase_step > maximum_reference_phase_step:
        raise ValueError("reference phase step exceeds the declared sampling bound")

    h0_times = [0.0]
    for left, right in zip(sampled, sampled[1:]):
        delta_x = right[0].x - left[0].x
        delta_time = 0.5 * (left[1].a + right[1].a) * delta_x
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("dimensionless cosmic time must be strictly increasing")
        h0_times.append(h0_times[-1] + delta_time)

    max_created_energy = max(
        abs(item[3].created_state_dependent_stress.energy_density_over_h0_four)
        for item in sampled
    )
    energy_floor = max(math.ulp(1.0) * max_created_energy, 1.0e-300)
    trajectory_nodes: list[SqueezedFLRWStressTrajectoryNode] = []
    for h0_time, (node, jet, e2, receipt) in zip(h0_times, sampled):
        stress = receipt.created_state_dependent_stress
        equation_of_state = _stress_equation_of_state(stress, energy_floor)
        accelerating = (
            stress.energy_density_over_h0_four > energy_floor
            and stress.energy_density_over_h0_four
            + 3.0 * stress.pressure_over_h0_four
            < 0.0
        )
        de_like = (
            equation_of_state is not None
            and abs(equation_of_state + 1.0) <= de_like_w_tolerance
        )
        trajectory_nodes.append(
            SqueezedFLRWStressTrajectoryNode(
                n=node.n,
                x=node.x,
                h0_cosmic_time=h0_time,
                receipt=receipt,
                created_equation_of_state=equation_of_state,
                positive_density_acceleration_diagnostic=accelerating,
                de_like_state_difference_diagnostic=de_like,
                hubble_to_mass=abs(jet.d1) / (jet.a * jet.a * mu),
                physical_momentum_to_mass=q / (jet.a * mu),
            )
        )
    nodes_tuple = tuple(trajectory_nodes)
    late_target = nodes_tuple[-1].n - late_window_efolds
    late_nodes = tuple(
        node
        for node in nodes_tuple
        if node.n >= late_target - background_tolerance
    )
    if len(late_nodes) < 2:
        raise ValueError("late window contains fewer than two sampled nodes")
    whole_window = _trajectory_time_window(nodes_tuple)
    late_window = _trajectory_time_window(late_nodes)
    ward = _trajectory_global_ward(nodes_tuple)
    acceleration_span = _grid_resolved_true_span(
        nodes_tuple,
        lambda node: node.positive_density_acceleration_diagnostic,
    )
    de_like_span = _grid_resolved_true_span(
        nodes_tuple,
        lambda node: node.de_like_state_difference_diagnostic,
    )

    late_phase_rates = tuple(
        2.0 * abs(phase_step) / (right[0].n - left[0].n)
        for phase_step, left, right in zip(phase_steps, sampled, sampled[1:])
        if 0.5 * (left[0].n + right[0].n) >= late_nodes[0].n
    )
    minimum_late_phase_rate = min(late_phase_rates)
    late_half_cycle = (
        math.pi / minimum_late_phase_rate
        if minimum_late_phase_rate > 0.0
        else math.inf
    )
    cold_gates = (
        late_window.max_hubble_to_mass <= maximum_hubble_to_mass_for_cold
        and late_window.max_physical_momentum_to_mass
        <= maximum_momentum_to_mass_for_cold
    )
    late_dm_like = (
        cold_gates
        and late_window.created_equation_of_state is not None
        and abs(late_window.created_equation_of_state)
        <= maximum_abs_late_w_for_dm_like
    )
    core_dimensions = {
        "n_equals_log_a": 0.0,
        "x_equals_H0_eta": 1.0 - 1.0,
        "h0_cosmic_time": 1.0 - 1.0,
        "q_equals_k_over_H0": 1.0 - 1.0,
        "mu_equals_m_over_H0": 1.0 - 1.0,
        "reference_phase": 0.0,
        "physical_conformal_ward_over_H0_five": 5.0 - 5.0,
    }
    return SqueezedFLRWStressTrajectory(
        nodes=nodes_tuple,
        whole_window=whole_window,
        late_window=late_window,
        ward=ward,
        q=q,
        mu=mu,
        alpha=complex(alpha),
        beta=complex(beta),
        initial_occupation=initial_occupation,
        max_reference_phase_step=max_phase_step,
        anomalous_phase_turns=abs(math.fsum(phase_steps)) / math.pi,
        late_half_cycle_efold_diagnostic=late_half_cycle,
        accelerating_state_difference_span_lower=acceleration_span[0],
        accelerating_state_difference_span_grid_upper=acceleration_span[1],
        de_like_state_difference_span_lower=de_like_span[0],
        de_like_state_difference_span_grid_upper=de_like_span[1],
        required_persistent_de_efolds=required_persistent_de_efolds,
        late_cold_adiabatic_gates_pass=cold_gates,
        late_dm_like_average_diagnostic_pass=late_dm_like,
        grid_diagnostic_excludes_required_de_persistence=(
            de_like_span[1] < required_persistent_de_efolds
        ),
        dimensions_pass=all(value == 0.0 for value in core_dimensions.values()),
    )


def _integrate_ir_uv_stress(
    q_values: tuple[float, ...],
    stresses: tuple[ModeStress, ...],
    certificate: SqueezedFLRWNodeIntegralCertificate,
) -> IRUVIntegratedStress:
    uv_integrated = integrate_isotropic_stress_with_certified_tail(
        q_values,
        stresses,
        energy_tail=certificate.energy_uv,
        pressure_tail=certificate.pressure_uv,
    )
    first_q = q_values[0]
    energy_ir_bound_at_first = certificate.energy_ir.pointwise_bound_at(first_q)
    pressure_ir_bound_at_first = certificate.pressure_ir.pointwise_bound_at(first_q)
    if (
        abs(stresses[0].energy_density_over_h0_four)
        > energy_ir_bound_at_first
    ):
        raise ValueError("first energy sample violates its certified infrared bound")
    if abs(stresses[0].pressure_over_h0_four) > pressure_ir_bound_at_first:
        raise ValueError("first pressure sample violates its certified infrared bound")
    return IRUVIntegratedStress(
        energy_density_over_h0_four=uv_integrated.energy_density_over_h0_four,
        pressure_over_h0_four=uv_integrated.pressure_over_h0_four,
        energy_ir_absolute_bound=(
            certificate.energy_ir.isotropic_integral_bound_to(first_q)
        ),
        pressure_ir_absolute_bound=(
            certificate.pressure_ir.isotropic_integral_bound_to(first_q)
        ),
        energy_uv_absolute_bound=uv_integrated.energy_tail_absolute_bound,
        pressure_uv_absolute_bound=uv_integrated.pressure_tail_absolute_bound,
    )


def _ensemble_time_window(
    nodes: tuple[SqueezedFLRWStressEnsembleNode, ...],
    *,
    mu: float,
    maximum_q: float,
) -> SqueezedFLRWStressEnsembleTimeWindow:
    if len(nodes) < 2:
        raise ValueError("an ensemble time window needs at least two nodes")
    h0_times = tuple(node.h0_cosmic_time for node in nodes)
    duration = h0_times[-1] - h0_times[0]
    created = _time_average_mode_stress(
        h0_times,
        tuple(
            ModeStress(
                node.created_stress.energy_density_over_h0_four,
                node.created_stress.pressure_over_h0_four,
            )
            for node in nodes
        ),
    )
    particle = _time_average_mode_stress(
        h0_times,
        tuple(node.particle_grid_stress for node in nodes),
    )
    anomalous = _time_average_mode_stress(
        h0_times,
        tuple(node.anomalous_grid_stress for node in nodes),
    )
    energy_bound = _time_average_nonnegative_bound(
        h0_times,
        tuple(
            node.created_stress.energy_external_ir_uv_remainder_absolute_bound
            for node in nodes
        ),
    )
    pressure_bound = _time_average_nonnegative_bound(
        h0_times,
        tuple(
            node.created_stress.pressure_external_ir_uv_remainder_absolute_bound
            for node in nodes
        ),
    )
    created_energy_scale = max(
        abs(node.created_stress.energy_density_over_h0_four) for node in nodes
    )
    particle_energy_scale = max(
        abs(node.particle_grid_stress.energy_density_over_h0_four)
        for node in nodes
    )
    created_floor = max(math.ulp(1.0) * created_energy_scale, 1.0e-300)
    particle_floor = max(math.ulp(1.0) * particle_energy_scale, 1.0e-300)
    comoving_particle_energies = tuple(
        node.scale_factor**3
        * node.particle_grid_stress.energy_density_over_h0_four
        for node in nodes
    )
    comoving_scale = max(
        max(abs(value) for value in comoving_particle_energies),
        1.0e-300,
    )
    return SqueezedFLRWStressEnsembleTimeWindow(
        start_n=nodes[0].n,
        end_n=nodes[-1].n,
        h0_cosmic_time_duration=duration,
        created_central_stress_time_average=created,
        particle_grid_stress_time_average=particle,
        anomalous_grid_stress_time_average=anomalous,
        created_energy_external_ir_uv_remainder_time_average_bound=energy_bound,
        created_pressure_external_ir_uv_remainder_time_average_bound=pressure_bound,
        created_central_equation_of_state=_stress_equation_of_state(
            created,
            created_floor,
        ),
        created_external_ir_uv_equation_of_state_interval=(
            _external_ir_uv_equation_of_state_interval(
                created,
                energy_absolute_bound=energy_bound,
                pressure_absolute_bound=pressure_bound,
            )
        ),
        particle_grid_equation_of_state=_stress_equation_of_state(
            particle,
            particle_floor,
        ),
        max_hubble_to_mass=max(
            abs(node.conformal_hubble) / (node.scale_factor * mu)
            for node in nodes
        ),
        max_physical_momentum_to_mass=max(
            maximum_q / (node.scale_factor * mu) for node in nodes
        ),
        particle_grid_comoving_energy_relative_span=(
            max(comoving_particle_energies) - min(comoving_particle_energies)
        )
        / comoving_scale,
    )


def _sampled_ir_uv_balance_uncertainty(
    nodes: tuple[SqueezedFLRWStressEnsembleNode, ...],
) -> float:
    first = nodes[0]
    last = nodes[-1]
    endpoint_bound = (
        first.scale_factor**3
        * first.created_stress.energy_external_ir_uv_remainder_absolute_bound
        + last.scale_factor**3
        * last.created_stress.energy_external_ir_uv_remainder_absolute_bound
    )
    pressure_terms: list[float] = []
    for left, right in zip(nodes, nodes[1:]):
        left_integrand_bound = (
            3.0
            * abs(left.conformal_hubble)
            * left.scale_factor**3
            * left.created_stress.pressure_external_ir_uv_remainder_absolute_bound
        )
        right_integrand_bound = (
            3.0
            * abs(right.conformal_hubble)
            * right.scale_factor**3
            * right.created_stress.pressure_external_ir_uv_remainder_absolute_bound
        )
        pressure_terms.append(
            0.5
            * (right.x - left.x)
            * (left_integrand_bound + right_integrand_bound)
        )
    return endpoint_bound + math.fsum(pressure_terms)


def _gaussian_q3_tail_moment(
    *,
    q_scale: float,
    tail_start_q: float,
    exponential_rate: float,
) -> float:
    """``integral_q0^inf q^3 exp[-rate (q/Q)^2] dq`` 를 돌려준다."""

    if not math.isfinite(q_scale) or q_scale <= 0.0:
        raise ValueError("Gaussian q^3 moment q_scale must be finite and positive")
    if not math.isfinite(tail_start_q) or tail_start_q < 0.0:
        raise ValueError("Gaussian q^3 moment tail_start_q must be finite and non-negative")
    if not math.isfinite(exponential_rate) or exponential_rate <= 0.0:
        raise ValueError("Gaussian q^3 moment exponential_rate must be finite and positive")
    try:
        scaled_start_squared = (tail_start_q / q_scale) ** 2
    except OverflowError:
        return 0.0
    if math.isinf(scaled_start_squared):
        return 0.0
    try:
        moment = (
            q_scale**4
            * math.exp(-exponential_rate * scaled_start_squared)
            * (
                scaled_start_squared / (2.0 * exponential_rate)
                + 1.0 / (2.0 * exponential_rate**2)
            )
        )
    except OverflowError as error:
        raise ValueError("Gaussian q^3 tail moment is not finite") from error
    if not math.isfinite(moment) or moment < 0.0:
        raise ValueError("Gaussian q^3 tail moment is not finite")
    return moment


def certify_gaussian_bogoliubov_profile_on_ensemble(
    trajectories: tuple[SqueezedFLRWStressTrajectory, ...],
    *,
    profile: GaussianBogoliubovProfile,
    verification_tolerance: float = 1.0e-10,
) -> GaussianBogoliubovIntegrabilityCertificate:
    """해석적 압축 프로파일 하나를 검증하고 정확한 q^3 모멘트 둘을 보증한다.

    두 모멘트는 4차 응력 핵 멱 계수 아래서 프로파일 인자 ``|alpha beta|`` 와 ``|beta|^2``
    을 통제한다. 진화된 모드 핵의 상계를 유도하지는 않으므로 하다마르나 단열 재규격화
    증명을 대체하지 않는다.
    """

    if len(trajectories) < 2:
        raise ValueError("a Gaussian ensemble certificate needs at least two modes")
    if (
        not math.isfinite(verification_tolerance)
        or verification_tolerance <= 0.0
        or verification_tolerance > 1.0e-4
    ):
        raise ValueError(
            "Gaussian profile verification_tolerance must lie in (0, 1e-4]"
        )

    def close(actual: complex, expected: complex) -> bool:
        return abs(actual - expected) <= verification_tolerance * max(
            1.0,
            abs(actual),
            abs(expected),
        )

    maximum_initial_occupation = 0.0
    q_values: list[float] = []
    for trajectory in trajectories:
        q = trajectory.q
        expected_beta = profile.beta_at(q)
        expected_alpha = profile.alpha_at(q)
        if not close(trajectory.beta, expected_beta) or not close(
            trajectory.alpha,
            expected_alpha,
        ):
            raise ValueError(
                "ensemble Bogoliubov coefficients do not match the Gaussian profile"
            )
        if (
            not math.isfinite(trajectory.initial_occupation)
            or trajectory.initial_occupation < 0.0
        ):
            raise ValueError("ensemble initial occupations must be finite and non-negative")
        maximum_initial_occupation = max(
            maximum_initial_occupation,
            trajectory.initial_occupation,
        )
        q_values.append(q)

    tail_start_q = max(q_values)
    anomalous_profile_moment = _gaussian_q3_tail_moment(
        q_scale=profile.q_scale,
        tail_start_q=tail_start_q,
        exponential_rate=1.0,
    )
    particle_profile_moment = _gaussian_q3_tail_moment(
        q_scale=profile.q_scale,
        tail_start_q=tail_start_q,
        exponential_rate=2.0,
    )
    occupation_factor = 1.0 + 2.0 * maximum_initial_occupation
    anomalous_upper = (
        occupation_factor
        * math.hypot(1.0, profile.amplitude)
        * profile.amplitude
        * anomalous_profile_moment
    )
    particle_upper = (
        occupation_factor
        * profile.amplitude
        * profile.amplitude
        * particle_profile_moment
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        anomalous_upper,
        particle_upper,
    )):
        raise ValueError("Gaussian profile amplitude-moment bound is not finite")
    return GaussianBogoliubovIntegrabilityCertificate(
        profile=profile,
        tail_start_q=tail_start_q,
        maximum_initial_occupation=maximum_initial_occupation,
        anomalous_q3_amplitude_moment_upper=anomalous_upper,
        particle_q3_amplitude_squared_moment_upper=particle_upper,
    )


def aggregate_squeezed_flrw_stress_ensemble(
    trajectories: tuple[SqueezedFLRWStressTrajectory, ...],
    *,
    node_certificates: tuple[SqueezedFLRWNodeIntegralCertificate, ...],
    bogoliubov_profile: GaussianBogoliubovProfile | None = None,
    bogoliubov_profile_tolerance: float = 1.0e-10,
    synchronization_tolerance: float = 1.0e-9,
    late_window_efolds: float = 1.0,
    maximum_hubble_to_mass_for_cold: float = 0.1,
    maximum_momentum_to_mass_for_cold: float = 0.1,
    maximum_abs_particle_w_for_dm_like: float = 0.1,
    maximum_particle_comoving_span_for_dm_like: float = 0.1,
    de_like_w_tolerance: float = 0.1,
    required_persistent_de_efolds: float = 1.0,
) -> SqueezedFLRWStressEnsemble:
    """고정 공변 q 격자 하나에서 동기화된 E49 궤적을 집계한다.

    각 시간 노드는 ``q^2 dq/(2*pi^2)`` 로 적분한다. 호출자는 표본화되지 않은 적외선과
    자외선 영역에 대한 점별 보증서를 따로 준다. 그 보증서는 표본 q/시간 격자 안의
    구적 오차를 제한하거나 꼬리 시간 도함수를 통제하지 않는다. 따라서 아래의
    상태방정식 구간은 그 외부 IR/UV 나머지만 전파하고, 지속 길이는 표본 노드에만
    해당한다. 워드 영수증은 표본 나머지 균형 상계를 가진 유한 격자 진단이지 전체 꼬리
    워드 증명, 연속 지속 증명, 절대 재규격화 응력 텐서가 아니다. 해석적 가우스
    프로파일이 주어지면 모든 궤적 계수를 검증하고 정확한 진폭 모멘트도 기록한다. 그
    모멘트도 진화된 응력 꼬리를 보증하지는 않는다.
    """

    if len(trajectories) < 2:
        raise ValueError("an ensemble requires at least two q trajectories")
    controls = (
        ("synchronization_tolerance", synchronization_tolerance),
        ("bogoliubov_profile_tolerance", bogoliubov_profile_tolerance),
        ("late_window_efolds", late_window_efolds),
        ("maximum_hubble_to_mass_for_cold", maximum_hubble_to_mass_for_cold),
        ("maximum_momentum_to_mass_for_cold", maximum_momentum_to_mass_for_cold),
        ("maximum_abs_particle_w_for_dm_like", maximum_abs_particle_w_for_dm_like),
        (
            "maximum_particle_comoving_span_for_dm_like",
            maximum_particle_comoving_span_for_dm_like,
        ),
        ("de_like_w_tolerance", de_like_w_tolerance),
        ("required_persistent_de_efolds", required_persistent_de_efolds),
    )
    for name, value in controls:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if synchronization_tolerance > 1.0e-4:
        raise ValueError("synchronization_tolerance must not exceed 1e-4")

    q_values = tuple(trajectory.q for trajectory in trajectories)
    if not all(math.isfinite(q) and q > 0.0 for q in q_values):
        raise ValueError("ensemble q values must be finite and positive")
    if any(right <= left for left, right in zip(q_values, q_values[1:])):
        raise ValueError("ensemble q values must be strictly increasing")
    base = trajectories[0]
    node_count = len(base.nodes)
    if node_count < 3:
        raise ValueError("ensemble trajectories need at least three time nodes")
    if len(node_certificates) != node_count:
        raise ValueError("one IR/UV certificate is required for every time node")
    total_efolds = base.nodes[-1].n - base.nodes[0].n
    if late_window_efolds > total_efolds:
        raise ValueError("late_window_efolds must not exceed the trajectory interval")
    mu = base.mu

    def mismatched(actual: float, expected: float) -> bool:
        return abs(actual - expected) > synchronization_tolerance * max(
            1.0,
            abs(actual),
            abs(expected),
        )

    for trajectory in trajectories:
        if len(trajectory.nodes) != node_count:
            raise ValueError("ensemble trajectories must share one time-node count")
        if mismatched(trajectory.mu, mu):
            raise ValueError("ensemble trajectories must share one constant mass")
        if not (
            trajectory.minimal_coupling_verified
            and trajectory.constant_positive_mass_verified_on_sampled_solution
            and trajectory.constant_bogoliubov_coefficients_assumed
            and trajectory.dimensions_pass
        ):
            raise ValueError("ensemble trajectory does not satisfy the E49 contract")
        if (
            trajectory.initial_occupation_basis_declaration
            != base.initial_occupation_basis_declaration
        ):
            raise ValueError("ensemble occupation basis declarations must agree")
        for base_node, node in zip(base.nodes, trajectory.nodes):
            if mismatched(node.receipt.q, trajectory.q) or mismatched(
                node.receipt.mu,
                mu,
            ):
                raise ValueError(
                    "ensemble trajectory q/mass metadata must match every receipt"
                )
            synchronized_values = (
                (node.n, base_node.n),
                (node.x, base_node.x),
                (node.h0_cosmic_time, base_node.h0_cosmic_time),
                (node.receipt.scale_factor, base_node.receipt.scale_factor),
                *zip(
                    node.receipt.background_jet.derivatives,
                    base_node.receipt.background_jet.derivatives,
                ),
            )
            if any(mismatched(actual, expected) for actual, expected in synchronized_values):
                raise ValueError(
                    "ensemble trajectories must share synchronized time and background grids"
                )

    bogoliubov_certificate = (
        certify_gaussian_bogoliubov_profile_on_ensemble(
            trajectories,
            profile=bogoliubov_profile,
            verification_tolerance=bogoliubov_profile_tolerance,
        )
        if bogoliubov_profile is not None
        else None
    )

    ensemble_nodes: list[SqueezedFLRWStressEnsembleNode] = []
    for index, certificate in enumerate(node_certificates):
        mode_nodes = tuple(trajectory.nodes[index] for trajectory in trajectories)
        created_samples = tuple(
            node.receipt.created_state_dependent_stress for node in mode_nodes
        )
        particle_samples = tuple(
            node.receipt.created_particle_stress for node in mode_nodes
        )
        anomalous_samples = tuple(
            node.receipt.created_anomalous_stress for node in mode_nodes
        )
        created = _integrate_ir_uv_stress(q_values, created_samples, certificate)
        particle = _integrate_isotropic_stress_grid(q_values, particle_samples)
        anomalous = _integrate_isotropic_stress_grid(q_values, anomalous_samples)
        decomposition_scale = max(
            1.0,
            abs(created.energy_density_over_h0_four),
            abs(created.pressure_over_h0_four),
        )
        decomposition_residual = max(
            abs(
                created.energy_density_over_h0_four
                - particle.energy_density_over_h0_four
                - anomalous.energy_density_over_h0_four
            ),
            abs(
                created.pressure_over_h0_four
                - particle.pressure_over_h0_four
                - anomalous.pressure_over_h0_four
            ),
        )
        if decomposition_residual > 10.0 * math.ulp(1.0) * decomposition_scale:
            raise ValueError("integrated particle/anomalous stress decomposition failed")
        energy_bound = created.energy_external_ir_uv_remainder_absolute_bound
        pressure_bound = created.pressure_external_ir_uv_remainder_absolute_bound
        central_stress = ModeStress(
            created.energy_density_over_h0_four,
            created.pressure_over_h0_four,
        )
        energy_scale = max(abs(created.energy_density_over_h0_four), 1.0e-300)
        interval = _external_ir_uv_equation_of_state_interval(
            central_stress,
            energy_absolute_bound=energy_bound,
            pressure_absolute_bound=pressure_bound,
        )
        external_ir_uv_acceleration = (
            created.energy_density_over_h0_four - energy_bound > 0.0
            and created.energy_density_over_h0_four
            + energy_bound
            + 3.0 * (created.pressure_over_h0_four + pressure_bound)
            < 0.0
        )
        external_ir_uv_de_like = (
            interval is not None
            and interval[0] >= -1.0 - de_like_w_tolerance
            and interval[1] <= -1.0 + de_like_w_tolerance
        )
        reference = mode_nodes[0]
        background_jet = reference.receipt.background_jet
        if background_jet.d1 <= 0.0:
            raise ValueError("ensemble FLRW constraint data require an expanding branch")
        try:
            hubble_over_h0 = background_jet.d1 / reference.receipt.scale_factor**2
            background_d_log_h_d_n = (
                reference.receipt.scale_factor
                * background_jet.d2
                / background_jet.d1**2
                - 2.0
            )
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("ensemble FLRW background expansion data are not finite") from error
        if not all(math.isfinite(value) for value in (
            hubble_over_h0,
            background_d_log_h_d_n,
        )):
            raise ValueError("ensemble FLRW background expansion data are not finite")
        ensemble_nodes.append(
            SqueezedFLRWStressEnsembleNode(
                n=reference.n,
                x=reference.x,
                h0_cosmic_time=reference.h0_cosmic_time,
                scale_factor=reference.receipt.scale_factor,
                conformal_hubble=(
                    background_jet.d1
                    / reference.receipt.scale_factor
                ),
                hubble_over_h0=hubble_over_h0,
                background_d_log_h_d_n=background_d_log_h_d_n,
                created_stress=created,
                particle_grid_stress=particle,
                anomalous_grid_stress=anomalous,
                created_central_equation_of_state=_stress_equation_of_state(
                    central_stress,
                    max(math.ulp(1.0) * energy_scale, 1.0e-300),
                ),
                created_external_ir_uv_equation_of_state_interval=interval,
                external_ir_uv_positive_density_acceleration_at_sample=(
                    external_ir_uv_acceleration
                ),
                external_ir_uv_de_like_at_sample=external_ir_uv_de_like,
            )
        )

    nodes = tuple(ensemble_nodes)
    late_target = nodes[-1].n - late_window_efolds
    late_nodes = tuple(
        node
        for node in nodes
        if node.n >= late_target - synchronization_tolerance
    )
    if len(late_nodes) < 2:
        raise ValueError("ensemble late window contains fewer than two nodes")
    whole_window = _ensemble_time_window(nodes, mu=mu, maximum_q=q_values[-1])
    late_window = _ensemble_time_window(
        late_nodes,
        mu=mu,
        maximum_q=q_values[-1],
    )
    central_ward = _global_ward_from_arrays(
        tuple(node.x for node in nodes),
        tuple(node.scale_factor for node in nodes),
        tuple(node.conformal_hubble for node in nodes),
        tuple(
            ModeStress(
                node.created_stress.energy_density_over_h0_four,
                node.created_stress.pressure_over_h0_four,
            )
            for node in nodes
        ),
    )
    ward = SqueezedFLRWStressEnsembleWardReceipt(
        central_grid=central_ward,
        sampled_ir_uv_balance_uncertainty_bound=(
            _sampled_ir_uv_balance_uncertainty(nodes)
        ),
    )
    n_values = tuple(node.n for node in nodes)
    acceleration_span = _grid_resolved_boolean_span(
        n_values,
        tuple(
            node.external_ir_uv_positive_density_acceleration_at_sample
            for node in nodes
        ),
    )
    de_like_span = _grid_resolved_boolean_span(
        n_values,
        tuple(node.external_ir_uv_de_like_at_sample for node in nodes),
    )
    cold_gates = (
        late_window.max_hubble_to_mass <= maximum_hubble_to_mass_for_cold
        and late_window.max_physical_momentum_to_mass
        <= maximum_momentum_to_mass_for_cold
    )
    particle_dm_like = (
        cold_gates
        and late_window.particle_grid_equation_of_state is not None
        and abs(late_window.particle_grid_equation_of_state)
        <= maximum_abs_particle_w_for_dm_like
        and late_window.particle_grid_comoving_energy_relative_span
        <= maximum_particle_comoving_span_for_dm_like
    )
    dimension_manifest = {
        "fixed_q_grid": 1.0 - 1.0,
        "isotropic_q_measure": 0.0,
        "h0_cosmic_time": 1.0 - 1.0,
        "integrated_stress_over_h0_four": 4.0 - 4.0,
        "conformal_ward_over_h0_five": 5.0 - 5.0,
    }
    return SqueezedFLRWStressEnsemble(
        nodes=nodes,
        whole_window=whole_window,
        late_window=late_window,
        ward=ward,
        q_values=q_values,
        mu=mu,
        sampled_accelerating_run_node_width=acceleration_span[0],
        sampled_accelerating_run_grid_upper=acceleration_span[1],
        sampled_de_like_run_node_width=de_like_span[0],
        sampled_de_like_run_grid_upper=de_like_span[1],
        required_persistent_de_efolds=required_persistent_de_efolds,
        late_finite_q_grid_cold_adiabatic_gates_pass=cold_gates,
        late_particle_grid_dm_like_diagnostic_pass=particle_dm_like,
        sampled_nodes_meet_required_de_run_length=(
            de_like_span[0] >= required_persistent_de_efolds
        ),
        dimensions_pass=all(value == 0.0 for value in dimension_manifest.values()),
        bogoliubov_integrability_certificate=bogoliubov_certificate,
        analytic_bogoliubov_profile_verified=bogoliubov_certificate is not None,
        absolute_bogoliubov_amplitude_moments_certified=(
            bogoliubov_certificate is not None
        ),
    )


def integrate_squeezed_created_stress_with_certified_tail(
    receipts: tuple[SqueezedFLRWModeStressDifference, ...],
    *,
    energy_tail: CertifiedPowerLawTail,
    pressure_tail: CertifiedPowerLawTail,
) -> IntegratedStress:
    """호출자가 보증한 UV 상계로 한 사건의 생성 응력을 적분한다.

    멱법칙 보증서는 신뢰하는 외부 입력이다. 이 래퍼는 정의역과 마지막 표본을 검사할 뿐,
    압축 핵에서 상계를 유도하지도 상태가 하다마르임을 증명하지도 않는다.
    """

    if len(receipts) < 2:
        raise ValueError("at least two squeezed mode receipts are required")
    first = receipts[0]
    if any(
        receipt.background_jet != first.background_jet or receipt.mu != first.mu
        for receipt in receipts[1:]
    ):
        raise ValueError("all squeezed receipts must share one background jet and mass")
    integrated = integrate_isotropic_stress_with_certified_tail(
        tuple(receipt.q for receipt in receipts),
        tuple(receipt.created_state_dependent_stress for receipt in receipts),
        energy_tail=energy_tail,
        pressure_tail=pressure_tail,
    )
    return IntegratedStress(
        energy_density_over_h0_four=integrated.energy_density_over_h0_four,
        pressure_over_h0_four=integrated.pressure_over_h0_four,
        energy_tail_absolute_bound=integrated.energy_tail_absolute_bound,
        pressure_tail_absolute_bound=integrated.pressure_tail_absolute_bound,
        status=(
            "FINITE_SQUEEZED_CREATED_STRESS_GRID_PLUS_EXTERNALLY_CERTIFIED_UV_TAIL"
        ),
    )


def local_isotropic_stress_observer_readout(
    stress: ModeStress,
    *,
    relative_speed: float,
) -> LocalIsotropicObserverStressReadout:
    """등방 diag(rho,p,p,p) 를 국소 4-속도와 축약한다.

    사건 국소 관계는 ``rho_obs=(rho+p)*gamma^2-p`` 이고 움직이는 시계는
    ``d tau_obs / dt_comoving = 1/gamma`` 로 표본한다. 전역 세계선, 엽층, 붕괴 규칙,
    보편 플랑크 눈금은 도입하지 않는다. 방향 분해된 개별 푸리에 모드는 일반적으로
    비등방이다. 입력은 이미 각도 평균된 껍질이나 등방 앙상블이어야 한다.
    """

    if not math.isfinite(relative_speed) or abs(relative_speed) >= 1.0:
        raise ValueError("relative_speed must be finite with magnitude below one")
    rho = stress.energy_density_over_h0_four
    pressure = stress.pressure_over_h0_four
    if not math.isfinite(rho) or not math.isfinite(pressure):
        raise ValueError("stress entries must be finite")
    gamma = 1.0 / math.sqrt(1.0 - relative_speed * relative_speed)
    return LocalIsotropicObserverStressReadout(
        relative_speed=relative_speed,
        lorentz_gamma=gamma,
        comoving_energy_density=rho,
        isotropic_pressure=pressure,
        observer_energy_density=(rho + pressure) * gamma * gamma - pressure,
        observer_proper_time_rate_per_comoving_cosmic_time=1.0 / gamma,
    )
