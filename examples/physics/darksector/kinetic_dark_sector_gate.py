"""자기비동일성 운동 암흑부문의 배경 게이트, Pantheon 초신성 형상 게이트, FLRW 모드 진화를 한 모듈에 모은다.

세 절로 구성한다.

1. 배경 게이트(background gate): 선언된 균질 P(T, X) 가지 자체를 구현한다.
   ``ce_residual_forward_model`` 이 쓰는 w0-wa 대리 모형이 아니고, 양의 초기 흐름에
   대한 미시적 유도를 주장하지도 않는다. 현재 밀도 분율을 경계 보정한 뒤, 얻은 궤적을
   가지 건전성과 압축 DESI DR2 BAO 공분산에 대해 검사한다.

2. Pantheon 40구간 초신성 형상 게이트(supernova shape gate): 공식 40구간 Pantheon
   벡터와 계통 공분산을 해시로 고정한다. 절대등급/H0 절편은 해석적으로 프로파일하므로
   상대 광도거리 형상만 검사한다. Pantheon+ 는 주장하지 않는다. 1701개 천체의 전체
   공분산은 이 압축 게이트의 범위 밖이다.

3. FLRW 스칼라 모드 진화(scalar-mode evolution): 선언된 CE 배경 위에서 무차원 모드를
   푼다. c=1, a0=1 에서 독립변수와 장은

       N = log(a),  x = H0 * eta,  q = k / H0,  mu = m / H0,
       U = sqrt(H0) * u_phys,  V = dU / dx

   이고 다음을 만족한다.

       dU/dN = V / (a E),
       dV/dN = -Omega^2 U / (a E),

   여기서 E=H/H0 이고

       Omega^2 = q^2 + a^2 mu^2 + (xi - 1/6) a^2 R/H0^2.

   이 절은 모드 함수와 정준 브론스키안(Wronskian) 감사만 제공한다. 4차 단열 빼기,
   재규격화 응력 텐서, 워드 항등식, 되먹임, 우주론 가능도는 구현하지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from examples.physics.darksector.ce_residual_forward_model import (
    BAODataset,
    CEForwardParams,
    assess_bao_fit,
    C_KM_S,
    invert_matrix,
    named_bao_dataset,
    quadratic_form,
    sound_horizon_at_redshift_mpc,
)


# ---------------------------------------------------------------------------
# 1. 배경 게이트
# ---------------------------------------------------------------------------


OMEGA_B0 = 0.049
OMEGA_R0 = 9.0e-5
OMEGA_V0 = 0.687
OMEGA_K0 = 1.0 - OMEGA_B0 - OMEGA_R0 - OMEGA_V0
N_INITIAL = math.log(1.0e-4)
SPEED_OF_LIGHT_KM_S = 299_792.458
REFERENCE_RD_MPC = 147.09
SH0ES_H0_KM_S_MPC = 73.04
SH0ES_H0_SIGMA = 1.04
PLANCK_Z_STAR = 1089.92
PLANCK_100_THETA_STAR = 1.04109
PLANCK_100_THETA_STAR_SIGMA = 0.00030


@dataclass(frozen=True)
class KineticClockConfig:
    gamma: float = 10.0
    kappa: float = 1.0e17
    x_star: float = 0.5
    steps: int = 2400

    def __post_init__(self) -> None:
        if self.gamma <= 0.0 or self.kappa <= 0.0 or self.x_star <= 0.0:
            raise ValueError("gamma, kappa, and x_star must be positive")
        if self.steps < 200:
            raise ValueError("steps must be at least 200")


@dataclass(frozen=True)
class BackgroundNode:
    n: float
    tau: float
    u: float
    e2: float
    cs2: float
    q_s_over_mpl2: float
    current_shape: float


@dataclass(frozen=True)
class BackgroundSolution:
    config: KineticClockConfig
    b: float
    amplitude: float
    nodes: tuple[BackgroundNode, ...]

    @property
    def min_u(self) -> float:
        return min(node.u for node in self.nodes)

    @property
    def min_cs2(self) -> float:
        return min(node.cs2 for node in self.nodes)

    @property
    def min_q_s_over_mpl2(self) -> float:
        return min(node.q_s_over_mpl2 for node in self.nodes)

    def at_n(self, n: float) -> BackgroundNode:
        if n < self.nodes[0].n or n > self.nodes[-1].n:
            raise ValueError("requested e-fold is outside the solved observation window")
        lo, hi = 0, len(self.nodes) - 1
        while hi - lo > 1:
            middle = (lo + hi) // 2
            if self.nodes[middle].n <= n:
                lo = middle
            else:
                hi = middle
        left, right = self.nodes[lo], self.nodes[hi]
        weight = (n - left.n) / (right.n - left.n)
        return BackgroundNode(
            n=n,
            tau=left.tau + weight * (right.tau - left.tau),
            u=left.u + weight * (right.u - left.u),
            e2=left.e2 + weight * (right.e2 - left.e2),
            cs2=left.cs2 + weight * (right.cs2 - left.cs2),
            q_s_over_mpl2=(
                left.q_s_over_mpl2
                + weight * (right.q_s_over_mpl2 - left.q_s_over_mpl2)
            ),
            current_shape=(
                left.current_shape + weight * (right.current_shape - left.current_shape)
            ),
        )


@dataclass(frozen=True)
class ProfiledBAOFit:
    scale: float
    scale_sigma: float
    chi2: float
    dof: int
    aic: float
    bic: float
    dataset: str
    role: str = "posthoc_boundary_calibrated_shape_test"


@dataclass(frozen=True)
class GammaScanComparison:
    gamma_values: tuple[float, ...]
    chi2_values: tuple[float, ...]
    best_gamma: float
    best_chi2: float
    best_scale: float
    best_h0_at_reference_rd: float
    lcdm_chi2: float
    delta_aic_vs_lcdm: float
    delta_bic_vs_lcdm: float
    best_is_upper_boundary: bool
    tail_delta_chi2: float
    tail_is_numerically_saturated: bool
    cmb_raw_pull_values: tuple[float, ...]
    minimum_abs_cmb_pull: float
    cmb_least_discrepant_gamma: float
    cmb_delta_pull_vs_lcdm_values: tuple[float, ...]
    minimum_abs_cmb_delta_pull_vs_lcdm: float
    role: str = "posthoc_seen_data_model_comparison"


@dataclass(frozen=True)
class CompressedCMBDiagnostic:
    z_star: float
    h0_km_s_mpc: float
    sound_horizon_mpc: float
    transverse_distance_mpc: float
    predicted_100_theta_star: float
    observed_100_theta_star: float
    observational_sigma: float
    raw_observational_pull: float
    lcdm_boundary_100_theta_star: float
    lcdm_boundary_raw_pull: float
    delta_100_theta_star_vs_lcdm_boundary: float
    delta_pull_vs_lcdm_boundary: float
    role: str = "APPROXIMATE_EARLY_PHYSICS_NOT_PRECISION_CMB_LIKELIHOOD"


@dataclass(frozen=True)
class BAOCMBAcousticClosure:
    h0_km_s_mpc: float
    rd_mpc: float
    sh0es_offset_sigma: float
    bao_profiled_scale: float
    role: str = "PLANCK_CALIBRATED_JOINT_DIAGNOSTIC_NOT_PREDICTION"


def positive_u_from_density(target_over_amplitude: float, kappa: float) -> float:
    """2u+3u^2/(2 kappa)=target/A 의 양근을 소거 오차 없이 돌려준다."""

    return target_over_amplitude / (
        math.sqrt(1.0 + 1.5 * target_over_amplitude / kappa) + 1.0
    )


def amplitude_from_b(b: float) -> float:
    denominator = -math.expm1(-b)
    if denominator <= 0.0:
        raise ValueError("the saturated vacuum readout must be positive")
    return OMEGA_V0 / denominator


def _densities(n: float, tau: float, u: float, config: KineticClockConfig, amplitude: float) -> tuple[float, ...]:
    rho_b = OMEGA_B0 * math.exp(-3.0 * n)
    rho_r = OMEGA_R0 * math.exp(-4.0 * n)
    rho_v = amplitude * (1.0 - math.exp(-config.gamma * tau))
    rho_k = amplitude * (2.0 * u + 1.5 * u * u / config.kappa)
    p_k = amplitude * u * u / (2.0 * config.kappa)
    e2 = rho_b + rho_r + rho_v + rho_k
    if not math.isfinite(e2) or e2 <= 0.0:
        raise ArithmeticError("kinetic-clock trajectory reached non-positive E^2")
    return rho_b, rho_r, rho_v, rho_k, p_k, e2


def _rhs(n: float, state: tuple[float, float], config: KineticClockConfig, amplitude: float) -> tuple[float, float]:
    tau, u = state
    if u <= -0.99 * config.kappa:
        raise ArithmeticError("kinetic-clock trajectory left the square-root domain")
    e = math.sqrt(_densities(n, tau, u, config, amplitude)[-1])
    root = math.sqrt(1.0 + u / config.kappa)
    current_shape = u * root
    current_derivative = (1.0 + 1.5 * u / config.kappa) / root
    tau_prime = root / e
    u_prime = (
        -3.0 * current_shape
        - config.gamma * math.exp(-config.gamma * tau) / (2.0 * e)
    ) / current_derivative
    return tau_prime, u_prime


def _rk4(n: float, state: tuple[float, float], step: float, config: KineticClockConfig, amplitude: float) -> tuple[float, float]:
    def add(base: tuple[float, float], delta: tuple[float, float], factor: float) -> tuple[float, float]:
        return base[0] + factor * delta[0], base[1] + factor * delta[1]

    k1 = _rhs(n, state, config, amplitude)
    k2 = _rhs(n + 0.5 * step, add(state, k1, 0.5 * step), config, amplitude)
    k3 = _rhs(n + 0.5 * step, add(state, k2, 0.5 * step), config, amplitude)
    k4 = _rhs(n + step, add(state, k3, step), config, amplitude)
    return (
        state[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _integrate_backward(config: KineticClockConfig, b: float, keep: bool) -> tuple[float, float, list[tuple[float, float, float]]]:
    amplitude = amplitude_from_b(b)
    state = (b / config.gamma, positive_u_from_density(OMEGA_K0 / amplitude, config.kappa))
    n = 0.0
    step = N_INITIAL / config.steps
    grid = [(n, *state)] if keep else []
    for _ in range(config.steps):
        state = _rk4(n, state, step, config, amplitude)
        n += step
        if keep:
            grid.append((n, *state))
    return state[0], amplitude, grid


def _shoot_b(config: KineticClockConfig) -> float:
    low, high = 0.5 * config.gamma, 1.2 * config.gamma

    def endpoint(value: float) -> float:
        try:
            return _integrate_backward(config, value, False)[0]
        except (ArithmeticError, OverflowError, ValueError):
            return -math.inf

    while endpoint(high) <= 0.0:
        high *= 1.25
        if high >= 10.0 * config.gamma:
            raise ArithmeticError("could not bracket the kinetic-clock shooting root")
    if endpoint(low) >= 0.0:
        raise ArithmeticError("kinetic-clock lower shooting bracket is invalid")
    for _ in range(52):
        middle = 0.5 * (low + high)
        if endpoint(middle) > 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def solve_background(config: KineticClockConfig = KineticClockConfig()) -> BackgroundSolution:
    b = _shoot_b(config)
    _, amplitude, descending = _integrate_backward(config, b, True)
    nodes: list[BackgroundNode] = []
    for n, tau, u in reversed(descending):
        e2 = _densities(n, tau, u, config, amplitude)[-1]
        delta = u / config.kappa
        cs2 = delta / (2.0 + 3.0 * delta)
        q_s = 3.0 * config.kappa * amplitude * (1.0 + delta) * (2.0 + 3.0 * delta) / e2
        nodes.append(
            BackgroundNode(
                n=n,
                tau=tau,
                u=u,
                e2=e2,
                cs2=cs2,
                q_s_over_mpl2=q_s,
                current_shape=u * math.sqrt(1.0 + delta),
            )
        )
    solution = BackgroundSolution(config=config, b=b, amplitude=amplitude, nodes=tuple(nodes))
    if solution.min_u <= 0.0 or solution.min_cs2 <= 0.0 or solution.min_q_s_over_mpl2 <= 0.0:
        raise ArithmeticError("the calibrated observation-window branch is not healthy")
    return solution


def zero_current_local_verdict() -> dict[str, object]:
    """원천 없는 영-흐름 출발에 대한 정확한 국소 반례를 돌려준다."""

    return {
        "initial_current": 0.0,
        "current_derivative_sign": -1,
        "delta_sign_immediately_after": -1,
        "cs2_sign_immediately_after": -1,
        "healthy_branch": False,
        "reason": "P_T<0 makes d(a^3 J)/dt<0; positive Pi_fold is required",
    }


def _dimensionless_distance(z: float, solution: BackgroundSolution, intervals: int = 512) -> float:
    if intervals % 2:
        intervals += 1
    dz = z / intervals

    def inverse_e(redshift: float) -> float:
        node = solution.at_n(-math.log1p(redshift))
        return 1.0 / math.sqrt(node.e2)

    total = inverse_e(0.0) + inverse_e(z)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * inverse_e(index * dz)
    return total * dz / 3.0


def _lcdm_dimensionless_distance(z: float, intervals: int = 512) -> float:
    """같은 경계의 평탄 LCDM+복사 음성 대조군이다."""

    if intervals % 2:
        intervals += 1
    dz = z / intervals

    def inverse_e(redshift: float) -> float:
        zp1 = 1.0 + redshift
        e2 = (
            OMEGA_R0 * zp1**4
            + (OMEGA_B0 + OMEGA_K0) * zp1**3
            + OMEGA_V0
        )
        return 1.0 / math.sqrt(e2)

    total = inverse_e(0.0) + inverse_e(z)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * inverse_e(index * dz)
    return total * dz / 3.0


def profile_desi_bao(
    solution: BackgroundSolution,
    dataset: BAODataset | None = None,
) -> ProfiledBAOFit:
    """공통 척도 c/(H0 rd) 를 압축 DESI DR2 BAO 에 대해 프로파일한다.

    프로파일된 척도는 외부 보정 진단이지 CE 예측이 아니다. 현재 밀도 분율과 gamma 는
    이미 선택되어 있다.
    """

    selected = dataset or named_bao_dataset("desi-dr2-all")
    inverse = invert_matrix(selected.covariance)
    shapes: list[float] = []
    values: list[float] = []
    for point in selected.data:
        node = solution.at_n(-math.log1p(point.z))
        dh = 1.0 / math.sqrt(node.e2)
        dm = _dimensionless_distance(point.z, solution)
        dv = (point.z * dm * dm * dh) ** (1.0 / 3.0)
        shapes.append({"dh": dh, "dm": dm, "dv": dv}[point.kind])
        values.append(point.value)
    shape_tuple, value_tuple = tuple(shapes), tuple(values)
    gg = quadratic_form(shape_tuple, inverse)
    gd = sum(
        shape_tuple[i] * inverse[i][j] * value_tuple[j]
        for i in range(len(shape_tuple))
        for j in range(len(shape_tuple))
    )
    scale = gd / gg
    residual = tuple(scale * shape - value for shape, value in zip(shape_tuple, value_tuple))
    chi2 = quadratic_form(residual, inverse)
    fitted_parameters = 1
    dof = len(values) - fitted_parameters
    return ProfiledBAOFit(
        scale=scale,
        scale_sigma=1.0 / math.sqrt(gg),
        chi2=chi2,
        dof=dof,
        aic=chi2 + 2.0 * fitted_parameters,
        bic=chi2 + fitted_parameters * math.log(len(values)),
        dataset=selected.name,
    )


def compressed_cmb_acoustic_diagnostic(
    solution: BackgroundSolution,
    *,
    h0_km_s_mpc: float,
    z_star: float = PLANCK_Z_STAR,
    distance_intervals: int = 4096,
) -> CompressedCMBDiagnostic:
    """선언된 표준 초기 물리로 음향각(acoustic angle)을 저용량으로 검사한다.

    재결합 적색이동과 광자-바리온 음속은 외부 표준 물리 입력이다. 원시 풀(raw pull)은
    Planck 관측 오차만 쓰므로 불일치 표지이지 정밀 가능도가 아니다.
    """

    if h0_km_s_mpc <= 0.0 or z_star <= 0.0:
        raise ValueError("CMB diagnostic inputs must be positive")
    params = CEForwardParams(
        omega_b0=OMEGA_B0,
        omega_dm0=OMEGA_K0,
        omega_lambda0=OMEGA_V0,
        h0=h0_km_s_mpc,
    )
    sound_horizon = sound_horizon_at_redshift_mpc(params, z_star)
    distance = (
        C_KM_S
        / h0_km_s_mpc
        * _dimensionless_distance(z_star, solution, intervals=distance_intervals)
    )
    predicted = 100.0 * sound_horizon / distance
    lcdm_distance = (
        C_KM_S
        / h0_km_s_mpc
        * _lcdm_dimensionless_distance(z_star, intervals=distance_intervals)
    )
    lcdm_theta = 100.0 * sound_horizon / lcdm_distance
    raw_pull = (predicted - PLANCK_100_THETA_STAR) / PLANCK_100_THETA_STAR_SIGMA
    lcdm_pull = (lcdm_theta - PLANCK_100_THETA_STAR) / PLANCK_100_THETA_STAR_SIGMA
    return CompressedCMBDiagnostic(
        z_star=z_star,
        h0_km_s_mpc=h0_km_s_mpc,
        sound_horizon_mpc=sound_horizon,
        transverse_distance_mpc=distance,
        predicted_100_theta_star=predicted,
        observed_100_theta_star=PLANCK_100_THETA_STAR,
        observational_sigma=PLANCK_100_THETA_STAR_SIGMA,
        raw_observational_pull=raw_pull,
        lcdm_boundary_100_theta_star=lcdm_theta,
        lcdm_boundary_raw_pull=lcdm_pull,
        delta_100_theta_star_vs_lcdm_boundary=predicted - lcdm_theta,
        delta_pull_vs_lcdm_boundary=raw_pull - lcdm_pull,
    )


def solve_bao_cmb_acoustic_closure(
    solution: BackgroundSolution,
    fit: ProfiledBAOFit,
    *,
    lower_h0: float = 45.0,
    upper_h0: float = 95.0,
) -> BAOCMBAcousticClosure:
    """Planck theta* 로 BAO H0*rd 퇴화 방향을 따라 H0 를 보정한다."""

    dimensionless_distance = _dimensionless_distance(
        PLANCK_Z_STAR, solution, intervals=4096
    )

    def residual(h0: float) -> float:
        params = CEForwardParams(
            omega_b0=OMEGA_B0,
            omega_dm0=OMEGA_K0,
            omega_lambda0=OMEGA_V0,
            h0=h0,
        )
        sound_horizon = sound_horizon_at_redshift_mpc(params, PLANCK_Z_STAR)
        predicted = 100.0 * sound_horizon * h0 / (
            C_KM_S * dimensionless_distance
        )
        return predicted - PLANCK_100_THETA_STAR

    low_value, high_value = residual(lower_h0), residual(upper_h0)
    if low_value * high_value >= 0.0:
        raise ArithmeticError("CMB acoustic root is not bracketed in H0")
    low, high = lower_h0, upper_h0
    for _ in range(52):
        middle = 0.5 * (low + high)
        if residual(middle) * low_value > 0.0:
            low = middle
            low_value = residual(low)
        else:
            high = middle
    h0 = 0.5 * (low + high)
    rd = SPEED_OF_LIGHT_KM_S / (fit.scale * h0)
    return BAOCMBAcousticClosure(
        h0_km_s_mpc=h0,
        rd_mpc=rd,
        sh0es_offset_sigma=(SH0ES_H0_KM_S_MPC - h0) / SH0ES_H0_SIGMA,
        bao_profiled_scale=fit.scale,
    )


def scan_gamma_against_lcdm(
    gamma_values: tuple[float, ...] = (3.5, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 300.0),
    *,
    steps: int = 1200,
    dataset: BAODataset | None = None,
) -> GammaScanComparison:
    """추가 파라미터 벌점을 명시한 관측 자료(seen data) gamma 훑기다.

    두 모형 모두 같은 BAO 척도를 프로파일한다. 이 자료에서 gamma 를 고르는 것은
    운동 모형에 적합 파라미터 하나를 더한다. 탐색적 모형 비교이지 홀드아웃이나
    CE 예측이 아니다.
    """

    if not gamma_values or any(value <= 0.0 for value in gamma_values):
        raise ValueError("gamma scan values must be non-empty and positive")
    selected = dataset or named_bao_dataset("desi-dr2-all")
    solutions = tuple(
        solve_background(KineticClockConfig(gamma=value, steps=steps))
        for value in gamma_values
    )
    fits = tuple(profile_desi_bao(solution, selected) for solution in solutions)
    cmb_diagnostics = tuple(
        compressed_cmb_acoustic_diagnostic(
            solution,
            h0_km_s_mpc=(
                SPEED_OF_LIGHT_KM_S / (fit.scale * REFERENCE_RD_MPC)
            ),
            distance_intervals=4096,
        )
        for solution, fit in zip(solutions, fits)
    )
    best_index = min(range(len(fits)), key=lambda index: fits[index].chi2)
    cmb_best_index = min(
        range(len(cmb_diagnostics)),
        key=lambda index: abs(cmb_diagnostics[index].raw_observational_pull),
    )
    lcdm = assess_bao_fit(
        selected.data,
        CEForwardParams(),
        covariance=selected.covariance,
    ).scale_fit_diagnostic
    if lcdm is None:
        raise ArithmeticError("LCDM BAO scale profile was not available")
    count = len(selected.data)
    # 운동 모형: 공통 척도 + 선택한 gamma, LCDM: 공통 척도만.
    delta_aic = (fits[best_index].chi2 + 4.0) - (lcdm.chi2 + 2.0)
    delta_bic = (
        fits[best_index].chi2 + 2.0 * math.log(count)
        - lcdm.chi2
        - math.log(count)
    )
    return GammaScanComparison(
        gamma_values=gamma_values,
        chi2_values=tuple(fit.chi2 for fit in fits),
        best_gamma=gamma_values[best_index],
        best_chi2=fits[best_index].chi2,
        best_scale=fits[best_index].scale,
        best_h0_at_reference_rd=(
            SPEED_OF_LIGHT_KM_S / (fits[best_index].scale * REFERENCE_RD_MPC)
        ),
        lcdm_chi2=lcdm.chi2,
        delta_aic_vs_lcdm=delta_aic,
        delta_bic_vs_lcdm=delta_bic,
        best_is_upper_boundary=best_index == len(fits) - 1,
        tail_delta_chi2=(
            abs(fits[-1].chi2 - fits[-2].chi2) if len(fits) > 1 else math.inf
        ),
        tail_is_numerically_saturated=(
            len(fits) > 1 and abs(fits[-1].chi2 - fits[-2].chi2) < 1.0e-8
        ),
        cmb_raw_pull_values=tuple(
            diagnostic.raw_observational_pull for diagnostic in cmb_diagnostics
        ),
        minimum_abs_cmb_pull=abs(
            cmb_diagnostics[cmb_best_index].raw_observational_pull
        ),
        cmb_least_discrepant_gamma=gamma_values[cmb_best_index],
        cmb_delta_pull_vs_lcdm_values=tuple(
            diagnostic.delta_pull_vs_lcdm_boundary
            for diagnostic in cmb_diagnostics
        ),
        minimum_abs_cmb_delta_pull_vs_lcdm=min(
            abs(diagnostic.delta_pull_vs_lcdm_boundary)
            for diagnostic in cmb_diagnostics
        ),
    )


def main() -> int:
    solution = solve_background()
    fit = profile_desi_bao(solution)
    comparison = scan_gamma_against_lcdm()
    print("# CE self-nonidentity kinetic dark-sector background gate")
    print(f"gamma {solution.config.gamma:.9g}")
    print(f"kappa {solution.config.kappa:.9g}")
    print(f"amplitude {solution.amplitude:.12g}")
    print(f"min_u {solution.min_u:.12g}")
    print(f"min_cs2 {solution.min_cs2:.12g}")
    print(f"min_Qs_over_Mpl2 {solution.min_q_s_over_mpl2:.12g}")
    print(f"desi_profiled_scale {fit.scale:.12g}")
    print(f"desi_profiled_scale_sigma {fit.scale_sigma:.12g}")
    print(f"desi_profiled_chi2 {fit.chi2:.12g}")
    print(f"desi_dof {fit.dof}")
    equivalent_h0 = SPEED_OF_LIGHT_KM_S / (fit.scale * REFERENCE_RD_MPC)
    equivalent_h0_sigma = equivalent_h0 * fit.scale_sigma / fit.scale
    cmb = compressed_cmb_acoustic_diagnostic(
        solution, h0_km_s_mpc=equivalent_h0
    )
    joint = solve_bao_cmb_acoustic_closure(solution, fit)
    print(f"diagnostic_h0_at_rd_147p09 {equivalent_h0:.12g}")
    print(f"diagnostic_h0_bao_stat_sigma {equivalent_h0_sigma:.12g}")
    print(
        "diagnostic_sh0es_point_offset_sigma",
        f"{(SH0ES_H0_KM_S_MPC - equivalent_h0) / SH0ES_H0_SIGMA:.12g}",
    )
    print("h0_role DIAGNOSTIC_ONLY_RD_FIXED_NO_MODEL_UNCERTAINTY")
    for name, value in cmb.__dict__.items():
        print(f"cmb_{name}", value)
    for name, value in joint.__dict__.items():
        print(f"bao_cmb_joint_{name}", value)
    print(f"gamma_scan_best_gamma {comparison.best_gamma:.12g}")
    print(f"gamma_scan_best_chi2 {comparison.best_chi2:.12g}")
    print(f"gamma_scan_best_scale {comparison.best_scale:.12g}")
    print(
        "gamma_scan_best_h0_at_rd_147p09",
        f"{comparison.best_h0_at_reference_rd:.12g}",
    )
    print(f"lcdm_profiled_chi2 {comparison.lcdm_chi2:.12g}")
    print(f"gamma_scan_delta_aic_vs_lcdm {comparison.delta_aic_vs_lcdm:.12g}")
    print(f"gamma_scan_delta_bic_vs_lcdm {comparison.delta_bic_vs_lcdm:.12g}")
    print(f"gamma_scan_best_is_upper_boundary {comparison.best_is_upper_boundary}")
    print(f"gamma_scan_tail_delta_chi2 {comparison.tail_delta_chi2:.12g}")
    print(
        "gamma_scan_tail_is_numerically_saturated",
        comparison.tail_is_numerically_saturated,
    )
    print(
        "gamma_scan_minimum_abs_cmb_raw_pull",
        f"{comparison.minimum_abs_cmb_pull:.12g}",
    )
    print(
        "gamma_scan_cmb_least_discrepant_gamma",
        f"{comparison.cmb_least_discrepant_gamma:.12g}",
    )
    print(
        "gamma_scan_minimum_abs_cmb_delta_pull_vs_lcdm",
        f"{comparison.minimum_abs_cmb_delta_pull_vs_lcdm:.12g}",
    )
    print(f"gamma_scan_role {comparison.role}")
    print("status CONDITIONAL_BACKGROUND_BRANCH_ONLY")
    print("growth_likelihood NOT_IMPLEMENTED")
    print("microscopic_Pi_fold NOT_DERIVED")
    return 0


# ---------------------------------------------------------------------------
# 2. Pantheon 40구간 초신성 형상 게이트
# ---------------------------------------------------------------------------


DATA_DIR = Path(__file__).resolve().parents[3] / "benchmarks/cosmology/pantheon_binned_v1"
VECTOR_SHA256 = "085daafcc4ae19ece72e69d69ac84fb0a4a1f52626ac4782e46571e6d679b000"
COVARIANCE_SHA256 = "642391b0a56ee4f0c3275e85376fbdb880c1c289503520fd32b3920c19f4d7d9"


@dataclass(frozen=True)
class SupernovaDataset:
    redshift: tuple[float, ...]
    apparent_magnitude: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    source: str = "Pantheon official 40-bin DS17 release"


@dataclass(frozen=True)
class SupernovaShapeFit:
    intercept: float
    chi2: float
    dof: int
    dataset: str
    role: str = "POSTHOC_PROFILED_INTERCEPT_SHAPE_TEST_NOT_PANTHEON_PLUS"


@dataclass(frozen=True)
class SupernovaModelComparison:
    kinetic: SupernovaShapeFit
    lcdm: SupernovaShapeFit

    @property
    def delta_chi2_kinetic_minus_lcdm(self) -> float:
        return self.kinetic.chi2 - self.lcdm.chi2


@dataclass(frozen=True)
class SupernovaHoldoutFit:
    training_intercept: float
    training_chi2: float
    predictive_chi2: float
    predictive_log_determinant: float
    training_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    dataset: str
    role: str = (
        "RETROSPECTIVE_DETERMINISTIC_COVARIANCE_CONDITIONAL_HOLDOUT_"
        "NOT_PREREGISTERED"
    )


@dataclass(frozen=True)
class SupernovaHoldoutComparison:
    kinetic: SupernovaHoldoutFit
    lcdm: SupernovaHoldoutFit

    @property
    def delta_predictive_chi2_kinetic_minus_lcdm(self) -> float:
        return self.kinetic.predictive_chi2 - self.lcdm.predictive_chi2


def _checked_text(path: Path, expected_sha256: str) -> str:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Pantheon input hash mismatch for {path.name}: {actual}")
    return payload.decode("utf-8")


def load_pantheon_binned() -> SupernovaDataset:
    vector_lines = _checked_text(DATA_DIR / "lcparam_DS17f.txt", VECTOR_SHA256)
    covariance_lines = _checked_text(DATA_DIR / "sys_DS17f.txt", COVARIANCE_SHA256)
    rows = [line.split() for line in vector_lines.splitlines() if line and not line.startswith("#")]
    redshift = tuple(float(row[1]) for row in rows)
    magnitude = tuple(float(row[4]) for row in rows)
    statistical_sigma = tuple(float(row[5]) for row in rows)
    covariance_values = covariance_lines.split()
    size = int(covariance_values[0])
    flat = tuple(float(value) for value in covariance_values[1:])
    if size != len(rows) or len(flat) != size * size:
        raise ValueError("Pantheon vector/covariance dimensions do not match")
    covariance = tuple(
        tuple(
            flat[i * size + j] + (statistical_sigma[i] ** 2 if i == j else 0.0)
            for j in range(size)
        )
        for i in range(size)
    )
    return SupernovaDataset(redshift, magnitude, covariance)


def _profile_intercept(
    shapes: tuple[float, ...], dataset: SupernovaDataset, label: str
) -> SupernovaShapeFit:
    inverse = invert_matrix(dataset.covariance)
    ones = tuple(1.0 for _ in shapes)
    target = tuple(obs - shape for obs, shape in zip(dataset.apparent_magnitude, shapes))
    denominator = quadratic_form(ones, inverse)
    numerator = sum(
        ones[i] * inverse[i][j] * target[j]
        for i in range(len(ones))
        for j in range(len(ones))
    )
    intercept = numerator / denominator
    residual = tuple(
        shape + intercept - observed
        for shape, observed in zip(shapes, dataset.apparent_magnitude)
    )
    return SupernovaShapeFit(
        intercept=intercept,
        chi2=quadratic_form(residual, inverse),
        dof=len(shapes) - 1,
        dataset=label,
    )


def _lcdm_distance(z: float, intervals: int = 512) -> float:
    if intervals % 2:
        intervals += 1
    step = z / intervals

    def inverse_e(redshift: float) -> float:
        zp1 = 1.0 + redshift
        e2 = OMEGA_R0 * zp1**4 + (OMEGA_B0 + OMEGA_K0) * zp1**3 + OMEGA_V0
        return 1.0 / math.sqrt(e2)

    total = inverse_e(0.0) + inverse_e(z)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * inverse_e(index * step)
    return total * step / 3.0


def _model_shapes(
    solution: BackgroundSolution,
    dataset: SupernovaDataset,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    def magnitude_shape(z: float, distance: float) -> float:
        if z <= 0.0 or distance <= 0.0:
            raise ValueError("supernova distance shape must be positive")
        return 5.0 * math.log10((1.0 + z) * distance)

    kinetic_shapes = tuple(
        magnitude_shape(z, _dimensionless_distance(z, solution))
        for z in dataset.redshift
    )
    lcdm_shapes = tuple(
        magnitude_shape(z, _lcdm_distance(z)) for z in dataset.redshift
    )
    return kinetic_shapes, lcdm_shapes


def compare_pantheon_binned(
    solution: BackgroundSolution | None = None,
    dataset: SupernovaDataset | None = None,
) -> SupernovaModelComparison:
    selected_solution = solution or solve_background()
    selected_data = dataset or load_pantheon_binned()
    kinetic_shapes, lcdm_shapes = _model_shapes(selected_solution, selected_data)
    return SupernovaModelComparison(
        kinetic=_profile_intercept(kinetic_shapes, selected_data, "Pantheon-40 kinetic"),
        lcdm=_profile_intercept(lcdm_shapes, selected_data, "Pantheon-40 LCDM"),
    )


def profiled_intercept_holdout_fit(
    shapes: tuple[float, ...],
    dataset: SupernovaDataset,
    *,
    holdout_indices: tuple[int, ...],
    label: str,
) -> SupernovaHoldoutFit:
    """훈련 구간으로 절편을 적합하고 상관된 홀드아웃 구간을 예측한다.

    가우스 조건부 분포는 훈련-홀드아웃 공분산과, 평탄 사전분포 아래 프로파일된 절편의
    사후 불확도를 모두 포함한다.
    """

    size = len(dataset.redshift)
    if len(shapes) != size or len(dataset.apparent_magnitude) != size:
        raise ValueError("shape and dataset vectors must have equal length")
    if not holdout_indices or len(set(holdout_indices)) != len(holdout_indices):
        raise ValueError("holdout_indices must be non-empty and unique")
    if tuple(sorted(holdout_indices)) != holdout_indices:
        raise ValueError("holdout_indices must be strictly increasing")
    if holdout_indices[0] < 0 or holdout_indices[-1] >= size:
        raise ValueError("holdout index lies outside the dataset")
    holdout_set = set(holdout_indices)
    training_indices = tuple(index for index in range(size) if index not in holdout_set)
    if not training_indices:
        raise ValueError("holdout split leaves no training bins")

    covariance = np.asarray(dataset.covariance, dtype=float)
    observed = np.asarray(dataset.apparent_magnitude, dtype=float)
    model = np.asarray(shapes, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError("dataset covariance has the wrong shape")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(observed)):
        raise ValueError("dataset must be finite")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("dataset covariance must be symmetric")

    train = np.asarray(training_indices, dtype=int)
    held = np.asarray(holdout_indices, dtype=int)
    c_tt = covariance[np.ix_(train, train)]
    c_hh = covariance[np.ix_(held, held)]
    c_ht = covariance[np.ix_(held, train)]
    ones_t = np.ones(len(train))
    ones_h = np.ones(len(held))
    target_t = observed[train] - model[train]
    solved_ones = np.linalg.solve(c_tt, ones_t)
    solved_target = np.linalg.solve(c_tt, target_t)
    denominator = float(ones_t @ solved_ones)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("training covariance is not positive definite")
    intercept = float(ones_t @ solved_target / denominator)
    training_residual = model[train] + intercept - observed[train]
    training_chi2 = float(
        training_residual @ np.linalg.solve(c_tt, training_residual)
    )

    solved_cross_transpose = np.linalg.solve(c_tt, c_ht.T)
    conditional_covariance = c_hh - c_ht @ solved_cross_transpose
    intercept_response = ones_h - c_ht @ solved_ones
    predictive_covariance = (
        conditional_covariance
        + np.outer(intercept_response, intercept_response) / denominator
    )
    predictive_mean = (
        model[held]
        + c_ht @ solved_target
        + intercept_response * intercept
    )
    predictive_residual = predictive_mean - observed[held]
    sign, log_determinant = np.linalg.slogdet(predictive_covariance)
    if sign <= 0.0:
        raise ValueError("predictive covariance is not positive definite")
    predictive_chi2 = float(
        predictive_residual
        @ np.linalg.solve(predictive_covariance, predictive_residual)
    )
    return SupernovaHoldoutFit(
        training_intercept=intercept,
        training_chi2=training_chi2,
        predictive_chi2=predictive_chi2,
        predictive_log_determinant=float(log_determinant),
        training_indices=training_indices,
        holdout_indices=holdout_indices,
        dataset=label,
    )


def compare_pantheon_binned_holdout(
    solution: BackgroundSolution | None = None,
    dataset: SupernovaDataset | None = None,
    *,
    holdout_indices: tuple[int, ...] | None = None,
) -> SupernovaHoldoutComparison:
    """네 구간마다 하나를 떼는 결정론적 회고 홀드아웃을 돌린다."""

    selected_solution = solution or solve_background()
    selected_data = dataset or load_pantheon_binned()
    selected_holdout = (
        tuple(range(3, len(selected_data.redshift), 4))
        if holdout_indices is None
        else holdout_indices
    )
    kinetic_shapes, lcdm_shapes = _model_shapes(selected_solution, selected_data)
    return SupernovaHoldoutComparison(
        kinetic=profiled_intercept_holdout_fit(
            kinetic_shapes,
            selected_data,
            holdout_indices=selected_holdout,
            label="Pantheon-40 kinetic holdout",
        ),
        lcdm=profiled_intercept_holdout_fit(
            lcdm_shapes,
            selected_data,
            holdout_indices=selected_holdout,
            label="Pantheon-40 LCDM holdout",
        ),
    )


def sn_gate_main() -> int:
    """Pantheon 40구간 형상·홀드아웃 비교를 출력하는 진입점이다(옛 kinetic_dark_sector_sn_gate.main)."""

    result = compare_pantheon_binned()
    print("kinetic_chi2", result.kinetic.chi2)
    print("lcdm_chi2", result.lcdm.chi2)
    print("dof", result.kinetic.dof)
    print("delta_chi2_kinetic_minus_lcdm", result.delta_chi2_kinetic_minus_lcdm)
    print("role", result.kinetic.role)
    holdout = compare_pantheon_binned_holdout()
    print("kinetic_holdout_predictive_chi2", holdout.kinetic.predictive_chi2)
    print("lcdm_holdout_predictive_chi2", holdout.lcdm.predictive_chi2)
    print(
        "delta_holdout_predictive_chi2_kinetic_minus_lcdm",
        holdout.delta_predictive_chi2_kinetic_minus_lcdm,
    )
    print("holdout_role", holdout.kinetic.role)
    return 0


# ---------------------------------------------------------------------------
# 3. FLRW 스칼라 모드 진화
# ---------------------------------------------------------------------------


class BackgroundNodeLike(Protocol):
    n: float
    e2: float


class FLRWBackgroundLike(Protocol):
    nodes: tuple[BackgroundNodeLike, ...]

    def at_n(self, n: float) -> BackgroundNodeLike: ...


@dataclass(frozen=True)
class FLRWModeSpec:
    """모든 진동수 입력을 무차원 H0 비로 표현한다."""

    comoving_wavenumber_over_h0: float
    mass_over_h0: Callable[[float], float]
    curvature_coupling: float = 1.0 / 6.0
    initial_n: float | None = None
    final_n: float | None = None
    steps: int = 1200
    curvature_derivative_step_n: float = 1.0e-4
    adiabatic_derivative_step_n: float = 1.0e-4
    max_initial_adiabaticity: float | None = None

    def __post_init__(self) -> None:
        q = self.comoving_wavenumber_over_h0
        if not math.isfinite(q) or q <= 0.0:
            raise ValueError("comoving_wavenumber_over_h0 must be finite and positive")
        if not callable(self.mass_over_h0):
            raise ValueError("mass_over_h0 must be callable")
        if not math.isfinite(self.curvature_coupling):
            raise ValueError("curvature_coupling must be finite")
        for name, value in (("initial_n", self.initial_n), ("final_n", self.final_n)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 20:
            raise ValueError("steps must be an integer of at least 20")
        for name, derivative_step in (
            ("curvature_derivative_step_n", self.curvature_derivative_step_n),
            ("adiabatic_derivative_step_n", self.adiabatic_derivative_step_n),
        ):
            if not math.isfinite(derivative_step) or derivative_step <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        maximum = self.max_initial_adiabaticity
        if maximum is not None and (not math.isfinite(maximum) or maximum <= 0.0):
            raise ValueError(
                "max_initial_adiabaticity must be finite and positive when provided"
            )


@dataclass(frozen=True)
class FLRWAdiabaticInitialState:
    n: float
    omega: float
    u: complex
    du_dx: complex
    adiabaticity: float
    wronskian_residual: float
    amplitude_residual: float


@dataclass(frozen=True)
class FLRWModeNode:
    n: float
    x: float
    omega_squared: float
    u: complex
    du_dx: complex
    wronskian_residual: float


@dataclass(frozen=True)
class FLRWModeSolution:
    spec: FLRWModeSpec
    nodes: tuple[FLRWModeNode, ...]
    background_window: tuple[float, float]
    initial_adiabaticity: float
    initial_amplitude_residual: float
    max_wronskian_residual: float
    status: str = "MODE_ONLY_NO_RENORMALIZED_STRESS_OR_BACKREACTION"
    dimensionless_contract: str = (
        "N=log(a); x=H0*eta; q=k/H0; mu=m/H0; U=sqrt(H0)*u_phys"
    )


def _background_bounds(background: FLRWBackgroundLike) -> tuple[float, float]:
    nodes = tuple(background.nodes)
    if len(nodes) < 2:
        raise ValueError("background must contain at least two ordered nodes")
    n_values = tuple(float(node.n) for node in nodes)
    if not all(math.isfinite(value) for value in n_values):
        raise ValueError("background e-fold nodes must be finite")
    if any(right <= left for left, right in zip(n_values, n_values[1:])):
        raise ValueError("background e-fold nodes must be strictly increasing")
    return n_values[0], n_values[-1]


def _resolved_interval(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
) -> tuple[float, float, tuple[float, float]]:
    bounds = _background_bounds(background)
    initial_n = bounds[0] if spec.initial_n is None else spec.initial_n
    final_n = bounds[1] if spec.final_n is None else spec.final_n
    assert initial_n is not None and final_n is not None
    if initial_n < bounds[0] or final_n > bounds[1]:
        raise ValueError("mode interval lies outside the solved background window")
    if final_n <= initial_n:
        raise ValueError("final_n must be greater than initial_n")
    return initial_n, final_n, bounds


def _e2_at_n(background: FLRWBackgroundLike, n: float) -> float:
    e2 = float(background.at_n(n).e2)
    if not math.isfinite(e2) or e2 <= 0.0:
        raise ValueError("background e2 must be finite and positive")
    return e2


def _mass_ratio_at_n(spec: FLRWModeSpec, n: float) -> float:
    mass_ratio = float(spec.mass_over_h0(n))
    if not math.isfinite(mass_ratio) or mass_ratio < 0.0:
        raise ValueError("mass_over_h0(n) must be finite and non-negative")
    return mass_ratio


def _dimensionless_ricci_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    bounds: tuple[float, float],
) -> float:
    step = spec.curvature_derivative_step_n
    if n - step < bounds[0] or n + step > bounds[1]:
        raise ValueError("curvature derivative stencil leaves the background window")
    e2 = _e2_at_n(background, n)
    log_e2_left = math.log(_e2_at_n(background, n - step))
    log_e2_right = math.log(_e2_at_n(background, n + step))
    d_log_h_d_n = (log_e2_right - log_e2_left) / (4.0 * step)
    return 6.0 * e2 * (2.0 + d_log_h_d_n)


def _omega_squared_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    bounds: tuple[float, float],
) -> float:
    if n < bounds[0] or n > bounds[1]:
        raise ValueError("requested e-fold is outside the solved background window")
    scale_factor = math.exp(n)
    mass_ratio = _mass_ratio_at_n(spec, n)
    curvature_coefficient = spec.curvature_coupling - 1.0 / 6.0
    curvature_term = 0.0
    if curvature_coefficient != 0.0:
        curvature_term = (
            curvature_coefficient
            * scale_factor**2
            * _dimensionless_ricci_at_n(background, spec, n, bounds)
        )
    omega_squared = (
        spec.comoving_wavenumber_over_h0**2
        + scale_factor**2 * mass_ratio**2
        + curvature_term
    )
    if not math.isfinite(omega_squared) or omega_squared <= 0.0:
        raise ValueError("dimensionless omega_squared must be finite and positive")
    return omega_squared


def omega_squared_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
) -> float:
    """무차원 Omega^2 를 돌려주고, 정의역 밖이면 닫힌 실패(fail-closed)로 예외를 낸다."""

    return _omega_squared_at_n(background, spec, n, _background_bounds(background))


def _wronskian_residual(u: complex, du_dx: complex) -> float:
    wronskian = u * du_dx.conjugate() - u.conjugate() * du_dx
    return abs(wronskian - 1.0j)


def _omega_derivative_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    difference_step: float,
    bounds: tuple[float, float],
) -> float:
    def omega(at_n: float) -> float:
        return math.sqrt(_omega_squared_at_n(background, spec, at_n, bounds))

    if n - difference_step >= bounds[0] and n + difference_step <= bounds[1]:
        return (omega(n + difference_step) - omega(n - difference_step)) / (
            2.0 * difference_step
        )
    if n + 2.0 * difference_step <= bounds[1]:
        return (
            -3.0 * omega(n)
            + 4.0 * omega(n + difference_step)
            - omega(n + 2.0 * difference_step)
        ) / (2.0 * difference_step)
    if n - 2.0 * difference_step >= bounds[0]:
        return (
            3.0 * omega(n)
            - 4.0 * omega(n - difference_step)
            + omega(n - 2.0 * difference_step)
        ) / (2.0 * difference_step)
    raise ValueError("omega derivative stencil leaves the background window")


def _adiabatic_initial_mode(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    bounds: tuple[float, float],
) -> FLRWAdiabaticInitialState:
    omega = math.sqrt(_omega_squared_at_n(background, spec, n, bounds))
    d_omega_d_n = _omega_derivative_at_n(
        background,
        spec,
        n,
        spec.adiabatic_derivative_step_n,
        bounds,
    )
    scale_factor_times_e = math.exp(n) * math.sqrt(_e2_at_n(background, n))
    d_omega_dx = scale_factor_times_e * d_omega_d_n
    u = complex(1.0 / math.sqrt(2.0 * omega))
    logarithmic_amplitude_rate = -d_omega_dx / (2.0 * omega)
    du_dx = complex(logarithmic_amplitude_rate, -omega) * u
    adiabaticity = abs(d_omega_dx) / omega**2
    amplitude_residual = abs(2.0 * omega * abs(u) ** 2 - 1.0)
    wronskian_residual = _wronskian_residual(u, du_dx)
    maximum = spec.max_initial_adiabaticity
    if maximum is not None and adiabaticity > maximum:
        raise ValueError(
            "initial adiabaticity exceeds max_initial_adiabaticity"
        )
    return FLRWAdiabaticInitialState(
        n=n,
        omega=omega,
        u=u,
        du_dx=du_dx,
        adiabaticity=adiabaticity,
        wronskian_residual=wronskian_residual,
        amplitude_residual=amplitude_residual,
    )


def adiabatic_initial_mode(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float | None = None,
) -> FLRWAdiabaticInitialState:
    """0차/1차 도함수 단열(adiabatic) 정준 상태를 만든다."""

    initial_n, final_n, bounds = _resolved_interval(background, spec)
    target_n = initial_n if n is None else n
    if target_n < initial_n or target_n > final_n:
        raise ValueError("initial-state e-fold lies outside the mode interval")
    return _adiabatic_initial_mode(
        background,
        spec,
        target_n,
        bounds,
    )


def _mode_rhs(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    bounds: tuple[float, float],
    n: float,
    state: tuple[float, complex, complex],
) -> tuple[float, complex, complex]:
    edge_tolerance = (
        16.0
        * math.ulp(1.0)
        * max(1.0, abs(bounds[0]), abs(bounds[1]))
    )
    if bounds[0] - edge_tolerance <= n < bounds[0]:
        n = bounds[0]
    elif bounds[1] < n <= bounds[1] + edge_tolerance:
        n = bounds[1]
    _, u, du_dx = state
    inverse_a_e = 1.0 / (math.exp(n) * math.sqrt(_e2_at_n(background, n)))
    omega_squared = _omega_squared_at_n(background, spec, n, bounds)
    return (
        inverse_a_e,
        du_dx * inverse_a_e,
        -omega_squared * u * inverse_a_e,
    )


def _rk4_mode_step(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    bounds: tuple[float, float],
    n: float,
    state: tuple[float, complex, complex],
    step: float,
) -> tuple[float, complex, complex]:
    k1 = _mode_rhs(background, spec, bounds, n, state)
    second = (
        state[0] + 0.5 * step * k1[0],
        state[1] + 0.5 * step * k1[1],
        state[2] + 0.5 * step * k1[2],
    )
    k2 = _mode_rhs(background, spec, bounds, n + 0.5 * step, second)
    third = (
        state[0] + 0.5 * step * k2[0],
        state[1] + 0.5 * step * k2[1],
        state[2] + 0.5 * step * k2[2],
    )
    k3 = _mode_rhs(background, spec, bounds, n + 0.5 * step, third)
    fourth = (
        state[0] + step * k3[0],
        state[1] + step * k3[1],
        state[2] + step * k3[2],
    )
    k4 = _mode_rhs(background, spec, bounds, n + step, fourth)
    return (
        state[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + step * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )


def solve_flrw_mode(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
) -> FLRWModeSolution:
    """정준 스칼라 모드 하나를 풀고 브론스키안 영수증을 보존한다."""

    initial_n, final_n, bounds = _resolved_interval(background, spec)
    step = (final_n - initial_n) / spec.steps
    initial = _adiabatic_initial_mode(
        background,
        spec,
        initial_n,
        bounds,
    )
    state = (0.0, initial.u, initial.du_dx)
    nodes: list[FLRWModeNode] = [
        FLRWModeNode(
            n=initial_n,
            x=state[0],
            omega_squared=initial.omega**2,
            u=state[1],
            du_dx=state[2],
            wronskian_residual=initial.wronskian_residual,
        )
    ]
    for index in range(spec.steps):
        n = initial_n + index * step
        state = _rk4_mode_step(background, spec, bounds, n, state, step)
        next_n = initial_n + (index + 1) * step
        nodes.append(
            FLRWModeNode(
                n=next_n,
                x=state[0],
                omega_squared=_omega_squared_at_n(
                    background,
                    spec,
                    next_n,
                    bounds,
                ),
                u=state[1],
                du_dx=state[2],
                wronskian_residual=_wronskian_residual(state[1], state[2]),
            )
        )
    max_wronskian_residual = max(node.wronskian_residual for node in nodes)
    return FLRWModeSolution(
        spec=spec,
        nodes=tuple(nodes),
        background_window=bounds,
        initial_adiabaticity=initial.adiabaticity,
        initial_amplitude_residual=initial.amplitude_residual,
        max_wronskian_residual=max_wronskian_residual,
    )


if __name__ == "__main__":
    raise SystemExit(main())
