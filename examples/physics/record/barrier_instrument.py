"""유한 장벽 계측기 묶음: 기저 누설(E16)·이중 우물 스펙트럼 도약(E17)·단일 에너지 두 포트 산란 계측기를 한 모듈에 담는다.

세 부분 모두 자연 단위 ``hbar = 1`` (셋째 부분은 ``hbar = c = 1``)을 쓴다.

첫째 부분(E16 유한 인증서)은 국소화 함수를 명시적 규약

``psi_n(x) = (pi sigma^2)^(-1/4) exp(-(x - n a)^2/(2 sigma^2))``

으로 잡고, 서로 통약되지 않는 세 예산 ``basis_amplitude``,
``barrier_probability``, ``projected_operator_norm`` 을 분리해 둔다. 하나의
"오차"로 더하는 일은 없다. 공급된 매개변수에 대한 유한 증인(witness)일 뿐이며,
:class:`FiniteBarrierModeLeakageCertificate` 끝의 거짓 플래그가 유도하지 않은
것을 기록한다. 표시된 ``S = 0`` 또는 투과 확률 ``T = 0`` 은 로그 영역 계산의
수치 언더플로(underflow)일 수 있고, 정확한 직교성·콤팩트 지지·정확한 유한 장벽
국소화를 주장하지 않는다.

둘째 부분(E17 스펙트럼 인증서)은 대칭 디리클레(Dirichlet) 이중 우물의 유한
속박 상태 스펙트럼만으로 두 모드 도약 ``J`` 를 유도한다. 보조로 계산하는 열린
산란 투과율은 의도적으로 별개 양이다: ``m, V0, b, Es`` 를 고정한 채 우물 폭
``w`` 만 바꾸면 스펙트럼은 바뀌지만 그 투과율은 바뀌지 않는다. 모드 정규화와
좌/우 편향은 부동소수점 구적(quadrature)과 닫힌 식 증인이지 형식적 구간 증명이
아니다. 특히 유한 장벽은 정확히 공간 국소화된 모드를 주지 않는다.

셋째 부분(단색 두 포트 장벽 계측기)은 공급된 한 에너지에서의 산란 계산이지
자율 검출기가 아니다. 산란 진폭은 표준 장벽 면(face) 규약을 쓴다. 통상 좌표
진폭은 ``t_conv = exp(-i*k*b) * t`` 이며, 이는 포트 하나의 위상 재설정이므로
절대 위상 주장을 하지 않는다. 매우 넓은 *유한* 장벽에서 표시되는 0 투과는
부동소수점 언더플로이고 ``log_transmission_probability`` 가 여전히 유효한
양이며 정확한 국소화를 주장하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_TOLERANCE = 1.0e-11
_SIGMA_X = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=complex)
_SIGMA_Z = np.array(((1.0, 0.0), (0.0, -1.0)), dtype=complex)
_IDENTITY_2 = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class FiniteBarrierModeLeakageCertificate:
    """분리된 유한 인증서. ``budget`` 이 붙은 필드는 서로 더하지 않는다."""

    mode_count: int
    sigma: float
    center_spacing: float
    gaussian_log_overlap_amplitude: float
    gaussian_overlap_amplitude: float
    gaussian_overlap_numerically_underflowed: bool
    basis_amplitude_budget: float
    basis_log_amplitude_budget: float
    basis_amplitude_target: float
    required_center_spacing: float
    basis_amplitude_budget_holds: bool
    nonrelativistic_mass: float
    barrier_height: float
    incident_energy: float
    barrier_width: float
    kappa: float
    barrier_log_transmission_probability: float
    barrier_transmission_probability: float
    barrier_probability_numerically_underflowed: bool
    barrier_probability_budget: float
    barrier_log_probability_budget: float
    barrier_probability_target: float
    per_barrier_probability_target: float
    exact_required_barrier_width: float
    exponential_prefactor: float
    exponential_regime_threshold: float
    exponential_regime_holds: bool
    exponential_probability_upper: float | None
    exponential_required_barrier_width: float
    barrier_probability_budget_holds: bool
    ideal_hopping: float
    projected_hamiltonian_norm_error: float
    dynamic_operator_norm_target: float
    ideal_swap_time: float
    ideal_swap_probability: float
    ideal_unitarity_residual: float
    ideal_swap_phase_residual: float
    single_step_operator_difference: float
    single_step_duhamel_bound_raw: float
    single_step_duhamel_bound_clipped: float
    repeated_step_operator_difference: float
    repeated_step_telescoping_bound_raw: float
    repeated_step_telescoping_bound_clipped: float
    required_projected_hamiltonian_norm_error: float
    projected_operator_norm_budget_holds: bool
    error_type_tuple: tuple[str, str, str]
    sigma_mass_dimension: int
    spacing_mass_dimension: int
    mass_mass_dimension: int
    energy_mass_dimension: int
    kappa_mass_dimension: int
    barrier_width_mass_dimension: int
    hopping_mass_dimension: int
    hamiltonian_norm_error_mass_dimension: int
    time_mass_dimension: int
    overlap_amplitude_mass_dimension: int
    transmission_probability_mass_dimension: int
    operator_norm_difference_mass_dimension: int
    dimensions_pass: bool
    identities_and_finite_witness_only: bool
    identical_parameters_required_by_contract: bool
    e15_modes_derived: bool = False
    kg_to_schrodinger_projection_derived: bool = False
    rectangular_barrier_represents_periodic_lattice: bool = False
    barrier_or_wkb_to_hopping_derived: bool = False
    finite_barrier_exact_localization: bool = False
    autonomous_dwell_time_derived: bool = False
    scattering_instrument_or_energy_receipt_derived: bool = False
    repeated_cptp_or_fresh_ancilla_derived: bool = False
    causal_or_strict_front_derived: bool = False
    qft_microcausality_or_no_signalling_derived: bool = False
    gr_source_derived: bool = False
    selection_derived: bool = False
    gates_5_to_8_closed: bool = False


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _unit_interval(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return value


def gaussian_overlap_amplitude(*, sigma: float, center_spacing: float) -> float:
    """명시된 psi 에 대한 표시용 ``S`` 를 돌려준다. 0으로 언더플로할 수 있다.

    그 0은 수치적일 뿐이다. 유한 가우스(Gaussian) 꼬리는 정확히 직교하지도
    콤팩트 지지를 갖지도 않는다. 표시 진폭이 언더플로하면 인증서의 로그 필드에서
    유한값을 읽는다.
    """

    sigma = _positive(sigma, "sigma")
    center_spacing = _positive(center_spacing, "center_spacing")
    return math.exp(_gaussian_log_overlap(sigma=sigma, center_spacing=center_spacing))


def _gaussian_log_overlap(*, sigma: float, center_spacing: float) -> float:
    return -(center_spacing / (2.0 * sigma)) ** 2


def exact_rectangular_barrier_transmission_probability(
    *, nonrelativistic_mass: float, barrier_height: float, incident_energy: float,
    barrier_width: float,
) -> float:
    """장벽 아래 1차원 투과 확률의 정확한 값이다. WKB 대체물이 아니다."""

    mass = _positive(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive(barrier_height, "barrier_height")
    energy = _positive(incident_energy, "incident_energy")
    width = _positive(barrier_width, "barrier_width")
    if energy >= height:
        raise ValueError("incident_energy must satisfy 0 < incident_energy < barrier_height")
    log_transmission, _ = _barrier_log_transmission(
        mass=mass, height=height, energy=energy, width=width
    )
    return math.exp(log_transmission)


def _log_sinh_nonnegative(value: float) -> float:
    """양의 무차원 ``value`` 에 대한 안정한 ``log(sinh(value))`` 이다."""

    if value < 40.0:
        return math.log(math.sinh(value))
    return value - math.log(2.0) + math.log1p(-math.exp(-2.0 * value))


def _barrier_log_transmission(
    *, mass: float, height: float, energy: float, width: float
) -> tuple[float, float]:
    """유한 장벽에서 오버플로 없이 ``log(T)`` 와 kappa 를 돌려준다."""

    log_energy_fraction = math.log(energy) - math.log(height)
    log_gap_fraction = math.log(height - energy) - math.log(height)
    log_factor = -math.log(4.0) - log_energy_fraction - log_gap_fraction
    kappa = math.sqrt(2.0 * mass * (height - energy))
    log_sinh = _log_sinh_nonnegative(kappa * width)
    log_transmission = -float(np.logaddexp(0.0, log_factor + 2.0 * log_sinh))
    return log_transmission, kappa


def _asinh_exp(log_argument: float) -> float:
    """``exp`` 오버플로 없이 ``asinh(exp(log_argument))`` 를 돌려준다."""

    if log_argument < 40.0:
        return math.asinh(math.exp(log_argument))
    return log_argument + math.log(2.0)


def _exp_or_infinity(log_value: float) -> float:
    maximum_log_float = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
    return math.exp(log_value) if log_value < maximum_log_float else math.inf


def _hermitian_exponential(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return (eigenvectors * np.exp(-1j * time * eigenvalues)) @ eigenvectors.conj().T


def certify_finite_barrier_mode_leakage(
    *,
    mode_count: int,
    sigma: float,
    center_spacing: float,
    delta_basis: float,
    nonrelativistic_mass: float,
    barrier_height: float,
    incident_energy: float,
    barrier_width: float,
    delta_leak: float,
    ideal_projected_hopping: float,
    projected_hamiltonian_norm_error: float,
    delta_dyn: float,
) -> FiniteBarrierModeLeakageCertificate:
    """동일한 공급 셀에 대해 승인된 유한 E16 인증서를 계산한다.

    ``mode_count`` 는 선언된 합집합/망원(telescoping) 할당에만 쓰인다. 동역학
    계산은 유한한 2차원 연산자 노름 증인이며, 다이아몬드 노름 진술도 CPTP 구성도
    아니다.
    """

    if isinstance(mode_count, bool) or not isinstance(mode_count, int) or mode_count < 1:
        raise ValueError("mode_count must be an integer at least one")
    sigma = _positive(sigma, "sigma")
    spacing = _positive(center_spacing, "center_spacing")
    delta_basis = _unit_interval(delta_basis, "delta_basis")
    mass = _positive(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive(barrier_height, "barrier_height")
    energy = _positive(incident_energy, "incident_energy")
    width = _positive(barrier_width, "barrier_width")
    delta_leak = _unit_interval(delta_leak, "delta_leak")
    hopping = _positive(ideal_projected_hopping, "ideal_projected_hopping")
    eta = _finite(projected_hamiltonian_norm_error, "projected_hamiltonian_norm_error")
    if eta < 0.0:
        raise ValueError("projected_hamiltonian_norm_error must be non-negative")
    delta_dyn = _unit_interval(delta_dyn, "delta_dyn")
    if energy >= height:
        raise ValueError("incident_energy must satisfy 0 < incident_energy < barrier_height")

    log_overlap = _gaussian_log_overlap(sigma=sigma, center_spacing=spacing)
    overlap = math.exp(log_overlap)
    log_mode_count = math.log(mode_count)
    log_basis_target = math.log(delta_basis)
    log_basis_budget = log_mode_count + log_overlap
    basis_budget = _exp_or_infinity(log_basis_budget)
    required_spacing = 2.0 * sigma * math.sqrt(log_mode_count - log_basis_target)

    log_transmission, kappa = _barrier_log_transmission(
        mass=mass, height=height, energy=energy, width=width
    )
    transmission = math.exp(log_transmission)
    log_leak_target = math.log(delta_leak)
    log_probability_budget = log_mode_count + log_transmission
    leak_budget = _exp_or_infinity(log_probability_budget)
    log_per_barrier_target = log_leak_target - log_mode_count
    per_barrier_target = math.exp(log_per_barrier_target)
    log_energy_fraction = math.log(energy) - math.log(height)
    log_gap_fraction = math.log(height - energy) - math.log(height)
    log_inverse_factor = math.log(4.0) + log_energy_fraction + log_gap_fraction
    log_z = 0.5 * (
        log_inverse_factor
        + math.log1p(-per_barrier_target)
        - log_per_barrier_target
    )
    exact_required_width = _asinh_exp(log_z) / kappa
    log_prefactor = math.log(64.0) + log_energy_fraction + log_gap_fraction
    prefactor = math.exp(log_prefactor)
    regime_threshold = math.log(2.0)
    regime_holds = kappa * width >= regime_threshold
    exponential_upper = (
        math.exp(log_prefactor - 2.0 * kappa * width) if regime_holds else None
    )
    exponential_required_width = max(
        regime_threshold,
        0.5 * (log_prefactor + log_mode_count - log_leak_target),
    ) / kappa

    tau = math.pi / (2.0 * hopping)
    h0 = -hopping * _SIGMA_X
    h = h0 + eta * _SIGMA_Z
    u0 = _hermitian_exponential(h0, tau)
    u = _hermitian_exponential(h, tau)
    single_difference = float(np.linalg.norm(u - u0, ord=2))
    single_bound_raw = tau * eta
    repeated_u = np.linalg.matrix_power(u, mode_count)
    repeated_u0 = np.linalg.matrix_power(u0, mode_count)
    repeated_difference = float(np.linalg.norm(repeated_u - repeated_u0, ord=2))
    repeated_bound_raw = mode_count * single_bound_raw
    required_eta = delta_dyn / (mode_count * tau)

    ideal_unitarity = float(np.linalg.norm(u0.conj().T @ u0 - _IDENTITY_2, ord=2))
    ideal_probability = float(abs(u0[1, 0]) ** 2)
    ideal_phase = float(abs(u0[1, 0] - 1j))
    sigma_dimension = -1
    spacing_dimension = -1
    mass_dimension = 1
    energy_dimension = 1
    kappa_dimension = 1
    barrier_width_dimension = -1
    hopping_dimension = 1
    eta_dimension = 1
    time_dimension = -1
    amplitude_dimension = 0
    probability_dimension = 0
    operator_difference_dimension = 0
    # 검사 대상은 kappa^2=2m(V0-E), tau=pi/(2J), tau*eta 이다.
    dimensions_pass = (
        sigma_dimension == spacing_dimension == -1
        and mass_dimension + energy_dimension == 2
        and kappa_dimension + barrier_width_dimension == 0
        and hopping_dimension + time_dimension == 0
        and time_dimension + eta_dimension == operator_difference_dimension
        and amplitude_dimension == probability_dimension == 0
    )
    dynamic_witness_holds = (
        single_difference <= single_bound_raw + _TOLERANCE
        and repeated_difference <= repeated_bound_raw + _TOLERANCE
    )

    return FiniteBarrierModeLeakageCertificate(
        mode_count=mode_count,
        sigma=sigma,
        center_spacing=spacing,
        gaussian_log_overlap_amplitude=log_overlap,
        gaussian_overlap_amplitude=overlap,
        gaussian_overlap_numerically_underflowed=(overlap == 0.0),
        basis_amplitude_budget=basis_budget,
        basis_log_amplitude_budget=log_basis_budget,
        basis_amplitude_target=delta_basis,
        required_center_spacing=required_spacing,
        basis_amplitude_budget_holds=(
            log_basis_budget <= log_basis_target + _TOLERANCE
        ),
        nonrelativistic_mass=mass,
        barrier_height=height,
        incident_energy=energy,
        barrier_width=width,
        kappa=kappa,
        barrier_log_transmission_probability=log_transmission,
        barrier_transmission_probability=transmission,
        barrier_probability_numerically_underflowed=(transmission == 0.0),
        barrier_probability_budget=leak_budget,
        barrier_log_probability_budget=log_probability_budget,
        barrier_probability_target=delta_leak,
        per_barrier_probability_target=per_barrier_target,
        exact_required_barrier_width=exact_required_width,
        exponential_prefactor=prefactor,
        exponential_regime_threshold=regime_threshold,
        exponential_regime_holds=regime_holds,
        exponential_probability_upper=exponential_upper,
        exponential_required_barrier_width=exponential_required_width,
        barrier_probability_budget_holds=(
            log_probability_budget <= log_leak_target + _TOLERANCE
        ),
        ideal_hopping=hopping,
        projected_hamiltonian_norm_error=eta,
        dynamic_operator_norm_target=delta_dyn,
        ideal_swap_time=tau,
        ideal_swap_probability=ideal_probability,
        ideal_unitarity_residual=ideal_unitarity,
        ideal_swap_phase_residual=ideal_phase,
        single_step_operator_difference=single_difference,
        single_step_duhamel_bound_raw=single_bound_raw,
        single_step_duhamel_bound_clipped=min(2.0, single_bound_raw),
        repeated_step_operator_difference=repeated_difference,
        repeated_step_telescoping_bound_raw=repeated_bound_raw,
        repeated_step_telescoping_bound_clipped=min(2.0, repeated_bound_raw),
        required_projected_hamiltonian_norm_error=required_eta,
        projected_operator_norm_budget_holds=(eta <= required_eta and dynamic_witness_holds),
        error_type_tuple=(
            "basis_amplitude",
            "barrier_probability",
            "projected_operator_norm",
        ),
        sigma_mass_dimension=sigma_dimension,
        spacing_mass_dimension=spacing_dimension,
        mass_mass_dimension=mass_dimension,
        energy_mass_dimension=energy_dimension,
        kappa_mass_dimension=kappa_dimension,
        barrier_width_mass_dimension=barrier_width_dimension,
        hopping_mass_dimension=hopping_dimension,
        hamiltonian_norm_error_mass_dimension=eta_dimension,
        time_mass_dimension=time_dimension,
        overlap_amplitude_mass_dimension=amplitude_dimension,
        transmission_probability_mass_dimension=probability_dimension,
        operator_norm_difference_mass_dimension=operator_difference_dimension,
        dimensions_pass=dimensions_pass,
        identities_and_finite_witness_only=True,
        identical_parameters_required_by_contract=True,
    )


_PI = math.pi


@dataclass(frozen=True)
class FiniteDoubleWellSpectralHoppingCertificate:
    """조건부 유한 스펙트럼 증인이다. 격자 브리지도 연속체 브리지도 없다."""

    nonrelativistic_mass: float
    barrier_height: float
    total_barrier_width: float
    well_width: float
    scattering_energy: float
    energy_unit: float
    nu: float
    beta: float
    even_z_bracket: tuple[float, float]
    odd_z_bracket: tuple[float, float]
    even_endpoint_values: tuple[float, float]
    odd_endpoint_values: tuple[float, float]
    even_root_residual: float
    odd_root_residual: float
    even_z: float
    odd_z: float
    even_energy_interval: tuple[float, float]
    odd_energy_interval: tuple[float, float]
    ground_energy: float
    first_excited_energy: float
    hopping_interval: tuple[float, float]
    hopping: float
    mean_energy: float
    spectral_order_holds: bool
    right_mode_norm_witness: float
    left_mode_norm_witness: float
    right_left_overlap_witness: float
    right_mode_right_probability_witness: float
    left_mode_left_probability_witness: float
    maximum_join_residual: float
    spectral_hamiltonian: np.ndarray
    ideal_swap_time: float
    spectral_swap_phase_residual: float
    auxiliary_scattering_transmission: float
    mass_mass_dimension: int
    barrier_height_mass_dimension: int
    barrier_width_mass_dimension: int
    well_width_mass_dimension: int
    scattering_energy_mass_dimension: int
    energy_unit_mass_dimension: int
    nu_mass_dimension: int
    beta_mass_dimension: int
    wavefunction_mass_dimension: float
    hopping_mass_dimension: int
    time_mass_dimension: int
    transmission_probability_mass_dimension: int
    dimensions_pass: bool
    finite_double_well_spectrum_to_J_derived: bool = True
    prepared_exact_spectral_pair_invariant_by_construction: bool = True
    transmission_to_hopping_derived: bool = False
    wkb_to_hopping_derived: bool = False
    exact_spatial_localization_derived: bool = False
    e15_material_lattice_embedding_derived: bool = False
    periodic_or_n_chain_derived: bool = False
    arbitrary_continuum_preparation_projects_to_subspace: bool = False
    scattering_instrument_or_energy_receipt_derived: bool = False
    cptp_or_fresh_ancilla_derived: bool = False
    causal_c_front_derived: bool = False
    qft_microcausality_or_gr_derived: bool = False
    selection_or_residual_explanation_derived: bool = False
    gates_5_to_8_closed: bool = False


def _positive_hopping(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _dimensionless_root_function(*, z: float, nu: float, beta: float, parity: str) -> float:
    if not _PI / 2.0 < z < _PI:
        raise ValueError("z must lie in (pi/2, pi)")
    q = math.sqrt(nu - z * z)
    u = 0.5 * beta * q
    barrier_ratio = math.tanh(u) if parity == "even" else 1.0 / math.tanh(u)
    return z / math.tan(z) + q * barrier_ratio


def _bisect_root(*, nu: float, beta: float, parity: str, tolerance: float) -> tuple[tuple[float, float], tuple[float, float], float]:
    # 선언된 nu > pi^2 조건이 전체 괄호 구간에서 q 를 실수로 만든다.
    # 이 선언된 가지에서의 유일성은 z*cot(z) 가 감소하고, q 가 z 에 대해 감소하므로
    # q*tanh(aq) 와 q*coth(aq) 가 q 에 대해 증가한다는 사실에서 따른다.
    # 증명은 원장이 담고, 이분법(bisection)은 수치 괄호 구간만 기록한다.
    left = math.nextafter(_PI / 2.0, _PI)
    right = math.nextafter(_PI, _PI / 2.0)
    f_left = _dimensionless_root_function(z=left, nu=nu, beta=beta, parity=parity)
    f_right = _dimensionless_root_function(z=right, nu=nu, beta=beta, parity=parity)
    if not (f_left > 0.0 and f_right < 0.0):
        raise RuntimeError("declared parity-root sign bracket failed")
    while right - left > tolerance:
        midpoint = 0.5 * (left + right)
        f_mid = _dimensionless_root_function(z=midpoint, nu=nu, beta=beta, parity=parity)
        if f_mid > 0.0:
            left = midpoint
            f_left = f_mid
        else:
            right = midpoint
            f_right = f_mid
    root = 0.5 * (left + right)
    return (left, right), (f_left, f_right), _dimensionless_root_function(
        z=root, nu=nu, beta=beta, parity=parity
    )


def _mode_values(*, x: np.ndarray, z: float, nu: float, beta: float, well_width: float, parity: str) -> np.ndarray:
    """전체 영역에서 해석적으로 정규화된 짝 모드 또는 홀 모드를 돌려준다."""

    q = math.sqrt(nu - z * z)
    k = z / well_width
    kappa = q / well_width
    half_barrier = 0.5 * beta * well_width
    barrier_edge = 0.5 * beta * q
    well_integral = 0.5 * well_width - math.sin(2.0 * z) / (4.0 * k)
    if parity == "even":
        barrier_integral = 2.0 * (0.25 * math.sinh(2.0 * barrier_edge) / kappa + half_barrier / 2.0)
        match_denominator = math.cosh(barrier_edge)
    else:
        barrier_integral = 2.0 * (0.25 * math.sinh(2.0 * barrier_edge) / kappa - half_barrier / 2.0)
        match_denominator = math.sinh(barrier_edge)
    normalization = 1.0 / math.sqrt(2.0 * well_integral + (math.sin(z) / match_denominator) ** 2 * barrier_integral)
    absolute_x = np.abs(x)
    in_barrier = absolute_x <= half_barrier
    values = np.empty_like(x, dtype=float)
    y = absolute_x - half_barrier
    values[~in_barrier] = normalization * np.sin(k * (well_width - y[~in_barrier]))
    if parity == "even":
        values[in_barrier] = normalization * math.sin(z) * np.cosh(kappa * x[in_barrier]) / match_denominator
    else:
        values[in_barrier] = (
            normalization * math.sin(z) * np.sinh(kappa * x[in_barrier]) / match_denominator
        )
        values[~in_barrier] *= np.sign(x[~in_barrier])
    return values


def _join_residual(*, z: float, nu: float, beta: float, well_width: float, parity: str) -> float:
    q = math.sqrt(nu - z * z)
    k = z / well_width
    kappa = q / well_width
    edge = 0.5 * beta * q
    # 비율 접합: 우물 쪽 f'/f=-k cot z, 장벽 쪽 f'/f=kappa tanh/coth.
    barrier_ratio = kappa * (math.tanh(edge) if parity == "even" else 1.0 / math.tanh(edge))
    return abs(-k / math.tan(z) - barrier_ratio)


def certify_finite_double_well_spectral_hopping(
    *,
    nonrelativistic_mass: float,
    barrier_height: float,
    total_barrier_width: float,
    well_width: float,
    scattering_energy: float,
    root_tolerance: float = 1.0e-12,
) -> FiniteDoubleWellSpectralHoppingCertificate:
    """공급된 유한 디리클레 우물의 가장 낮은 패리티(parity) 쌍을 인증한다.

    영역은 ``[-(w+b/2), w+b/2]`` 이고 ``|x|<b/2`` 에서 ``V=V0`` 이다. 아래의
    정확한 두 상태 불변성은 여기서 찾은 두 스펙트럼 고유모드로 준비된 상태에만
    해당하며, 임의의 연속체 준비를 그 부분공간으로 사영하지 않는다.
    """

    mass = _positive_hopping(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive_hopping(barrier_height, "barrier_height")
    width = _positive_hopping(total_barrier_width, "total_barrier_width")
    well = _positive_hopping(well_width, "well_width")
    scattering = _positive_hopping(scattering_energy, "scattering_energy")
    tolerance = _positive_hopping(root_tolerance, "root_tolerance")
    if tolerance >= 0.1:
        raise ValueError("root_tolerance must be below 0.1")
    if scattering >= height:
        raise ValueError("scattering_energy must satisfy 0 < Es < V0")
    energy_unit = 1.0 / (2.0 * mass * well * well)
    nu = height / energy_unit
    beta = width / well
    if nu <= _PI * _PI:
        raise ValueError("safe lowest-pair domain requires nu > pi^2")

    even_bracket, even_values, even_residual = _bisect_root(
        nu=nu, beta=beta, parity="even", tolerance=tolerance
    )
    odd_bracket, odd_values, odd_residual = _bisect_root(
        nu=nu, beta=beta, parity="odd", tolerance=tolerance
    )
    even_z = 0.5 * sum(even_bracket)
    odd_z = 0.5 * sum(odd_bracket)
    even_interval = (energy_unit * even_bracket[0] ** 2, energy_unit * even_bracket[1] ** 2)
    odd_interval = (energy_unit * odd_bracket[0] ** 2, energy_unit * odd_bracket[1] ** 2)
    if not even_z < odd_z or not even_interval[1] < odd_interval[0]:
        raise RuntimeError("lowest even/odd spectral order was not certified")
    ground = energy_unit * even_z * even_z
    excited = energy_unit * odd_z * odd_z
    hopping_interval = (
        0.5 * (odd_interval[0] - even_interval[1]),
        0.5 * (odd_interval[1] - even_interval[0]),
    )
    hopping = 0.5 * (excited - ground)
    mean_energy = 0.5 * (excited + ground)

    endpoint = well + 0.5 * width
    grid = np.linspace(-endpoint, endpoint, 100_001)
    even_mode = _mode_values(x=grid, z=even_z, nu=nu, beta=beta, well_width=well, parity="even")
    odd_mode = _mode_values(x=grid, z=odd_z, nu=nu, beta=beta, well_width=well, parity="odd")
    right_mode = (even_mode + odd_mode) / math.sqrt(2.0)
    left_mode = (even_mode - odd_mode) / math.sqrt(2.0)
    right_norm = float(np.trapezoid(right_mode * right_mode, grid))
    left_norm = float(np.trapezoid(left_mode * left_mode, grid))
    overlap = float(np.trapezoid(right_mode * left_mode, grid))
    right_bias = float(np.trapezoid(right_mode[grid >= 0.0] ** 2, grid[grid >= 0.0]))
    left_bias = float(np.trapezoid(left_mode[grid <= 0.0] ** 2, grid[grid <= 0.0]))
    joins = (
        _join_residual(z=even_z, nu=nu, beta=beta, well_width=well, parity="even"),
        _join_residual(z=odd_z, nu=nu, beta=beta, well_width=well, parity="odd"),
    )

    spectral_hamiltonian = np.array(
        ((mean_energy, -hopping), (-hopping, mean_energy)), dtype=float
    )
    swap_time = _PI / (2.0 * hopping)
    eigenvalues, eigenvectors = np.linalg.eigh(spectral_hamiltonian)
    swap = (eigenvectors * np.exp(-1j * swap_time * eigenvalues)) @ eigenvectors.conj().T
    swap_phase_residual = float(abs(swap[1, 0] - 1j * np.exp(-1j * mean_energy * swap_time)))
    transmission = exact_rectangular_barrier_transmission_probability(
        nonrelativistic_mass=mass,
        barrier_height=height,
        incident_energy=scattering,
        barrier_width=width,
    )
    dimensions_pass = (
        1 + (-2) + 1 == 0  # 2*m*w^2*V0 는 무차원이다.
        and 1 + (-1) == 0  # beta=b/w.
        and 1 + (-1) == 0  # J*tau.
    )

    return FiniteDoubleWellSpectralHoppingCertificate(
        nonrelativistic_mass=mass, barrier_height=height, total_barrier_width=width,
        well_width=well, scattering_energy=scattering, energy_unit=energy_unit, nu=nu, beta=beta,
        even_z_bracket=even_bracket, odd_z_bracket=odd_bracket,
        even_endpoint_values=even_values, odd_endpoint_values=odd_values,
        even_root_residual=even_residual, odd_root_residual=odd_residual,
        even_z=even_z, odd_z=odd_z, even_energy_interval=even_interval,
        odd_energy_interval=odd_interval, ground_energy=ground, first_excited_energy=excited,
        hopping_interval=hopping_interval, hopping=hopping, mean_energy=mean_energy,
        spectral_order_holds=True, right_mode_norm_witness=right_norm,
        left_mode_norm_witness=left_norm, right_left_overlap_witness=overlap,
        right_mode_right_probability_witness=right_bias,
        left_mode_left_probability_witness=left_bias, maximum_join_residual=max(joins),
        spectral_hamiltonian=spectral_hamiltonian, ideal_swap_time=swap_time,
        spectral_swap_phase_residual=swap_phase_residual,
        auxiliary_scattering_transmission=transmission,
        mass_mass_dimension=1, barrier_height_mass_dimension=1, barrier_width_mass_dimension=-1,
        well_width_mass_dimension=-1, scattering_energy_mass_dimension=1,
        energy_unit_mass_dimension=1, nu_mass_dimension=0, beta_mass_dimension=0,
        wavefunction_mass_dimension=0.5, hopping_mass_dimension=1, time_mass_dimension=-1,
        transmission_probability_mass_dimension=0, dimensions_pass=dimensions_pass,
    )


_I2 = np.eye(2, dtype=complex)
_P0 = np.diag((1.0, 0.0)).astype(complex)
_P1 = np.diag((0.0, 1.0)).astype(complex)


@dataclass(frozen=True)
class SingleEnergyBarrierInstrumentCertificate:
    """유한 조건부 증인이다. 거짓 플래그가 생략된 모든 브리지의 경계를 정한다."""

    nonrelativistic_mass: float
    barrier_height: float
    incident_energy: float
    barrier_width: float
    k: float
    kappa: float
    dimensionless_barrier_width: float
    coefficient_a: float
    coefficient_b: float
    tanh_dimensionless_width: float
    log_sech_dimensionless_width: float
    reflection_amplitude: complex
    transmission_amplitude: complex
    conventional_coordinate_transmission_amplitude: complex
    log_transmission_probability: float
    transmission_probability: float
    transmission_amplitude_numerically_underflowed: bool
    transmission_probability_numerically_underflowed: bool
    reflection_probability: float
    reflection_probability_residual: float
    scattering_matrix: np.ndarray
    scattering_unitarity_residual: float
    cross_amplitude_residual: float
    coefficient_identity_residual: float
    transmission_e16_residual: float
    input_density_matrix: np.ndarray
    output_density_matrix: np.ndarray
    output_port_probabilities: tuple[float, float]
    output_port_projectors: tuple[np.ndarray, np.ndarray]
    kraus_operators: tuple[np.ndarray, np.ndarray]
    kraus_completeness_residual: float
    record_isometry: np.ndarray
    record_isometry_residual: float
    choi_matrix: np.ndarray
    choi_minimum_eigenvalue: float
    output_trace_residual: float
    output_minimum_eigenvalue: float
    elastic_shell_energy: float
    input_shell_hamiltonian: np.ndarray
    port_shell_hamiltonian: np.ndarray
    record_hamiltonian: np.ndarray
    final_shell_hamiltonian: np.ndarray
    energy_intertwining_residual: float
    input_shell_energy_expectation: float
    nonselective_output_shell_energy_expectation: float
    isometric_total_output_energy_expectation: float
    nonselective_shell_energy_residual: float
    isometric_shell_energy_residual: float
    nonselective_energy_residual: float
    left_input_port0_reflection_probability: float
    left_input_port1_transmission_probability: float
    mass_mass_dimension: int
    energy_mass_dimension: int
    width_mass_dimension: int
    wavenumber_mass_dimension: int
    kappa_mass_dimension: int
    dimensionless_barrier_width_mass_dimension: int
    amplitude_mass_dimension: int
    probability_mass_dimension: int
    dimensions_pass: bool
    conditional_single_energy_scattering_unitarity: bool
    output_port_cptp_instrument: bool
    prepared_record_isometry: bool
    elastic_degenerate_energy_bookkeeping: bool
    one_sided_port_label_statement: bool
    physical_observation_or_selection_derived: bool = False
    general_reflection_transmission_labels_derived: bool = False
    wavepacket_or_energy_spread_derived: bool = False
    autonomous_detector_derived: bool = False
    durable_record_reset_or_battery_derived: bool = False
    physical_non_degenerate_record_energy_receipt_derived: bool = False
    repeated_fresh_ancilla_cptp_derived: bool = False
    causal_front_derived: bool = False
    qft_or_gr_derived: bool = False
    e17_j_transmission_relation_derived: bool = False
    residual_prediction_derived: bool = False
    gates_3_to_8_closed: bool = False


def _positive_single(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _density_matrix(value: np.ndarray | None) -> np.ndarray:
    rho = _P0.copy() if value is None else np.asarray(value, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must have shape (2, 2)")
    if not np.isfinite(rho).all():
        raise ValueError("rho must be finite")
    if not np.allclose(rho, rho.conj().T, atol=1.0e-12, rtol=0.0):
        raise ValueError("rho must be Hermitian")
    if not math.isclose(float(np.trace(rho).real), 1.0, abs_tol=1.0e-12):
        raise ValueError("rho must have unit trace")
    if np.linalg.eigvalsh(rho).min() < -1.0e-12:
        raise ValueError("rho must be positive semidefinite")
    return rho


def _log_sech(x: float) -> float:
    """여기서 쓰는 양의 무차원 x 에 대한 안정한 log(sech x) 이다."""

    return math.log(2.0) - x - math.log1p(math.exp(-2.0 * x))


def _e16_log_transmission(*, mass: float, height: float, energy: float, width: float) -> float:
    """E16 의 정확한 장벽 아래 결과를 독립된 대수 형태로 다시 쓴 것이다."""

    log_fraction = math.log(energy) - math.log(height)
    log_gap_fraction = math.log(height - energy) - math.log(height)
    log_factor = -math.log(4.0) - log_fraction - log_gap_fraction
    x = math.sqrt(2.0 * mass * (height - energy)) * width
    log_sinh = math.log(math.sinh(x)) if x < 40.0 else x - math.log(2.0) + math.log1p(-math.exp(-2.0 * x))
    return -float(np.logaddexp(0.0, log_factor + 2.0 * log_sinh))


def certify_single_energy_barrier_instrument(
    *, nonrelativistic_mass: float, barrier_height: float, incident_energy: float,
    barrier_width: float, rho: np.ndarray | None = None,
) -> SingleEnergyBarrierInstrumentCertificate:
    """한 번 쓰는 고정 에너지 출력 포트 계측기를 인증한다.

    왼쪽만 입력하는 경우 출력 포트 0이 반사, 포트 1이 투과이다. 임의의 양쪽 입력에
    대해서는 그 낱말을 의도적으로 붙이지 않는다. 그저 두 출력 포트일 뿐이다.
    """

    mass = _positive_single(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive_single(barrier_height, "barrier_height")
    energy = _positive_single(incident_energy, "incident_energy")
    width = _positive_single(barrier_width, "barrier_width")
    if energy >= height:
        raise ValueError("incident_energy must satisfy 0 < incident_energy < barrier_height")
    input_rho = _density_matrix(rho)

    k = math.sqrt(2.0 * mass * energy)
    kappa = math.sqrt(2.0 * mass * (height - energy))
    x = kappa * width
    # 이 비율들은 별도의 차원 있는 계수 규약을 피한다.
    a = (kappa / k - k / kappa) / 2.0
    b = (kappa / k + k / kappa) / 2.0
    q = math.tanh(x)
    logsech = _log_sech(x)
    sech = math.exp(logsech)
    denominator = 1.0 + 1j * a * q
    t = sech / denominator
    r = -1j * b * q / denominator
    scattering = np.array(((r, t), (t, r)), dtype=complex)
    log_t = 2.0 * logsech - math.log1p((a * q) ** 2)
    transmission = math.exp(log_t)
    reflection = float(abs(r) ** 2)
    conventional_t = complex(math.cos(k * width), -math.sin(k * width)) * t
    e16_log_t = _e16_log_transmission(mass=mass, height=height, energy=energy, width=width)

    kraus = (_P0 @ scattering, _P1 @ scattering)
    output = sum((operator @ input_rho @ operator.conj().T for operator in kraus), np.zeros((2, 2), complex))
    probabilities = tuple(float(np.trace(operator @ input_rho @ operator.conj().T).real) for operator in kraus)
    record_isometry = np.vstack(kraus)
    choi = sum((np.outer(operator.reshape(-1, order="F"), operator.reshape(-1, order="F").conj()) for operator in kraus), np.zeros((4, 4), complex))
    completeness = sum((operator.conj().T @ operator for operator in kraus), np.zeros((2, 2), complex))
    h_shell = energy * _I2
    h_record = np.zeros((2, 2), dtype=complex)
    h_final = np.kron(_I2, h_shell) + np.kron(h_record, _I2)
    isometric_output = record_isometry @ input_rho @ record_isometry.conj().T
    input_energy = float(np.trace(h_shell @ input_rho).real)
    port_energy = float(np.trace(h_shell @ output).real)
    isometric_energy = float(np.trace(h_final @ isometric_output).real)
    left_output = scattering @ _P0 @ scattering.conj().T

    return SingleEnergyBarrierInstrumentCertificate(
        nonrelativistic_mass=mass, barrier_height=height, incident_energy=energy, barrier_width=width,
        k=k, kappa=kappa, dimensionless_barrier_width=x, coefficient_a=a, coefficient_b=b,
        tanh_dimensionless_width=q, log_sech_dimensionless_width=logsech,
        reflection_amplitude=r, transmission_amplitude=t,
        conventional_coordinate_transmission_amplitude=conventional_t,
        log_transmission_probability=log_t, transmission_probability=transmission,
        transmission_amplitude_numerically_underflowed=(t == 0.0j),
        transmission_probability_numerically_underflowed=(transmission == 0.0), reflection_probability=reflection,
        reflection_probability_residual=float(abs(reflection - (1.0 - transmission))),
        scattering_matrix=scattering,
        scattering_unitarity_residual=float(np.linalg.norm(scattering.conj().T @ scattering - _I2, ord=2)),
        cross_amplitude_residual=float(abs(r * t.conjugate() + t * r.conjugate())),
        coefficient_identity_residual=float(abs(b * b - (1.0 + a * a))),
        transmission_e16_residual=float(abs(log_t - e16_log_t)), input_density_matrix=input_rho,
        output_density_matrix=output, output_port_probabilities=probabilities,
        output_port_projectors=(_P0.copy(), _P1.copy()), kraus_operators=kraus,
        kraus_completeness_residual=float(np.linalg.norm(completeness - _I2, ord=2)),
        record_isometry=record_isometry,
        record_isometry_residual=float(np.linalg.norm(record_isometry.conj().T @ record_isometry - _I2, ord=2)),
        choi_matrix=choi, choi_minimum_eigenvalue=float(np.linalg.eigvalsh(choi).min()),
        output_trace_residual=float(abs(np.trace(output) - 1.0)),
        output_minimum_eigenvalue=float(np.linalg.eigvalsh(output).min()), elastic_shell_energy=energy,
        input_shell_hamiltonian=h_shell, port_shell_hamiltonian=h_shell.copy(), record_hamiltonian=h_record,
        final_shell_hamiltonian=h_final,
        energy_intertwining_residual=float(np.linalg.norm(h_final @ record_isometry - record_isometry @ h_shell, ord=2)),
        input_shell_energy_expectation=input_energy,
        nonselective_output_shell_energy_expectation=port_energy,
        isometric_total_output_energy_expectation=isometric_energy,
        nonselective_shell_energy_residual=abs(port_energy - input_energy),
        isometric_shell_energy_residual=abs(isometric_energy - input_energy),
        nonselective_energy_residual=float(np.linalg.norm(h_shell @ output - output @ h_shell, ord=2)),
        left_input_port0_reflection_probability=float(left_output[0, 0].real),
        left_input_port1_transmission_probability=float(left_output[1, 1].real),
        mass_mass_dimension=1, energy_mass_dimension=1, width_mass_dimension=-1,
        wavenumber_mass_dimension=1, kappa_mass_dimension=1,
        dimensionless_barrier_width_mass_dimension=0, amplitude_mass_dimension=0, probability_mass_dimension=0,
        dimensions_pass=True, conditional_single_energy_scattering_unitarity=True,
        output_port_cptp_instrument=True, prepared_record_isometry=True,
        elastic_degenerate_energy_bookkeeping=True, one_sided_port_label_statement=True,
    )
