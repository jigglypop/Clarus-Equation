"""SNKC 극장 개막 급냉(quench) 생산 배경 게이트, 해석적 자외선 꼬리 상계, 봉인 홀드아웃 평가를 한 모듈에 모은다.

세 절로 구성한다.

1. 급냉 생산 배경 게이트(보정 단계). SNKC 급냉 계약의 사전 선언 배경 결합 규칙을
   구현한다. 동결된 경계 조합 (Omega_b, Omega_r, Omega_K, Omega_V) =
   (0.049, 9.0e-5, 0.26391, 0.687) 은 바꾸지 않는다. 보정 파라미터 하나
   Omega_prod,0 in [0, Omega_K] 가 정확한 a^-3 생산항 Omega_prod,0*(1+z)^3 을 E^2(a)
   에 더하고, 운동 부문 사격(shooting) 경계는 Omega_K' = 0.26391 - Omega_prod,0 로
   재해석되어 평탄한 현재 예산의 합은 여전히 1이다. 검사 대상은 정확한 a^-3 희석과
   운동 u-동역학 사이의 형상 차이다.

   보정 규칙(사전 선언, 관측·표적 인지 자료만):

   * 목적함수 = 저장소 내부 DESI DR2 13점 전체 공분산 chi^2 에 프로파일된 공통 척도
     c/(H0 rd) 하나, 그 외 없음;
   * 탐색 격자 = {0} 과 [1e-6, 0.05] 의 로그 균등 40점의 합집합, argmin 채택;
   * Pantheon-40 은 argmin 에서만 보고하고 보정에는 쓰지 않는다;
   * 홀드아웃 자료(eBOSS DR16, 우주 시계)는 여기서 접촉하지 않는다. 그런 자료를
     인자로만 받는 순수 평가 함수만 제공한다.

   기본 게이트 모듈 ``kinetic_dark_sector_gate`` 는 가져다 쓰기만 하고 수정하지 않는다.
   아래 급냉 풀이기는 그 적분기를 연산 단위로 그대로 따르며 Omega_prod,0 -> 0
   극한에서 비트 단위로 동일하게 환원된다.

2. 매끄러운 tanh 생성 초과분의 해석적 자외선 꼬리 상계(ultraviolet tail bound).
   ``p >= P > 0`` 에 대해

       omega_i,o = sqrt(p^2 + m_i,o^2),
       x = pi*tau*abs(omega_o-omega_i)/2

   로 두고, ``abs(omega_o-omega_i) <= abs(m_o^2-m_i^2)/(2p)``, ``sinh(x) <= x exp(x)``,
   그리고 분모 sinh 각각의 지수 하계를 쓰면

       f_created(p) <= A(P) p^-2 exp(-2*pi*tau*p)

   를 얻는다. 여기서 ``f_created=(1+2*n_in)|beta_p|^2`` 이다. 남은 수밀도와 무충돌
   현재 에너지 꼬리는 초등적인 지수적분 상계를 가진다. 채택한 점근 민코프스키 tanh
   스펙트럼에 대한 정확한 실수 산술 부등식이며, 재규격화된 FLRW 응력 텐서나 ``P``
   아래 유한 창 구적법의 보증서는 아니다.

3. SNKC 급냉 배경 봉인 홀드아웃 평가(sealed holdout evaluation).
   사전등록: experiments/preregistration/cosmology_snkc_quench_bg_v1.json
   (원장 SNKC-R2-THEATER-QUENCH-BG-PREREG-10). 이 절은 평가 프로토콜을 구현한다.
   프로토콜의 출처: 매니페스트(모형, 홀드아웃 신원, 프로파일 성가신 모수 둘, kill
   문턱 +9)는 홀드아웃 접근 전에 동결됐다. P/X 분해와 더 엄격한 P-또는-X kill 규칙은
   매니페스트 동결 후, 홀드아웃 중심값 취득 후, 그러나 모형 대 자료 계산 전에 선언된
   명확화다. 이 접근 후 명확화는 감사에 문서화된 이탈(보수적 방향, 판정 불변)로
   기록한다.

   모형(동일 취급):

   * M1 -- 등록 가지: ``kinetic_dark_sector_quench`` 의 동결 운동+급냉 배경,
     Omega_prod,0 = 0 (홀드아웃에서 적합한 파라미터 0개).
   * M2 -- 대조군: 정확한 동결 경계 조합 (Omega_b, Omega_r, Omega_m_extra, Omega_V) =
     (0.049, 9e-5, 0.26391, 0.687) 의 평탄 LambdaCDM, 곧
     ``same_frozen_boundary_lcdm_chi2`` 와 같은 배경.

   1차 홀드아웃(eBOSS/BOSS 합의 BAO):

   * P (1차 판정): 가우스 블록만 -- DR12 LRG (관측 4, 4x4 covtot) + DR16 LRG (관측 2)
     + DR16 QSO (관측 2), 블록 대각 8x8. 모형 관측량 DM/rd = s * dc(z),
     DH/rd = s / E(z), 공통 프로파일 척도 s = c/(H0 rd) 하나; s 는 선형으로 들어가므로
     프로파일은 정확한 GLS(해석적)이고 수치 프로파일과 교차 확인한다.
   * X (확장 변형, 함께 보고): chi2_P(s) 에 공식 정규화 ELG DV/rd 표와 LYAUTO / LYxQSO
     DM-DH 가능도비 격자의 -2 ln L 을 더한다(선형 / 쌍선형 보간; 표 범위 밖에서는
     경계값에 선언된 큰 이차 벌점을 더함). s 는 수치로 프로파일한다.
   * kill 규칙(사전 선언): Delta chi2 = M1 - M2 가 P 또는 X 에서 +9 를 넘으면 이 가지의
     배경 주장은 REJECTED 다.

   2차 홀드아웃(우주 시계, Moresco 15점 균질 부분집합):

   * H_model(z) = H0 * E(z), 프로파일 절편 H0 하나(정확한 GLS). 공분산 =
     diag(errHz^2) + ``data_MM20.dat`` 의 IMF + stlib + SPS 성분을 자료 적색이동에
     보간한 값(퍼센트 / 100)의 외적. 공식 CCcovariance 노트북 조리법을 따른다.
     주 채택 = ``mod`` 열; 민감도 변형 = ``mod_ooo`` 를 함께 보고. 참고 보고:
     풀 벡터와 chi2/dof, dof = 15 - 1.

   자료는 ``benchmarks/cosmology/snkc_quench_bg_holdout_v1/`` 에서만 읽는다(출처와
   sha256 은 그 README.md). 이 절은 보정(DESI) 자료를 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

from examples.physics.darksector.ce_residual_forward_model import (
    BAODataset,
    CEForwardParams,
    assess_bao_fit,
    invert_matrix,
    named_bao_dataset,
    quadratic_form,
)
from examples.physics.darksector.kinetic_dark_sector_gate import (
    BackgroundNode,
    BackgroundSolution,
    KineticClockConfig,
    N_INITIAL,
    OMEGA_B0,
    OMEGA_K0,
    OMEGA_R0,
    OMEGA_V0,
    _dimensionless_distance,
    amplitude_from_b,
    compare_pantheon_binned,
    positive_u_from_density,
    profile_desi_bao,
)
from examples.physics.record.theater_opening import (
    QuantumSeatSpecies,
)


# ---------------------------------------------------------------------------
# 1. 급냉 생산 배경 게이트
# ---------------------------------------------------------------------------


FROZEN_BASE_DESI_CHI2 = 16.18945
FROZEN_BASE_DESI_SCALE = 29.91640809
FROZEN_LCDM_CONTROL_CHI2 = 12.60835
LCDM_CONTROL_NOTE = (
    "control = assess_bao_fit(CEForwardParams()) as in the base gate scan; "
    "its boundary is the CE_RATIOS tuple (0.0487, 0.2623, 0.6891), not the "
    "frozen kinetic tuple.  The strict frozen-tuple flat LCDM chi^2 is "
    "reported separately as same_frozen_boundary_lcdm_chi2."
)

BAO_KIND_FAMILY = frozenset({"dm", "dh", "dv"})
HZ_KIND_FAMILY = frozenset({"hz"})
_KIND_ALIASES = {
    "dm": "dm",
    "dm/rd": "dm",
    "dh": "dh",
    "dh/rd": "dh",
    "dv": "dv",
    "dv/rd": "dv",
    "hz": "hz",
    "h(z)": "hz",
}


def _quench_densities(
    n: float,
    tau: float,
    u: float,
    config: KineticClockConfig,
    amplitude: float,
    omega_prod0: float,
) -> tuple[float, ...]:
    """기본 게이트 밀도에 정확한 a^-3 급냉 생산항을 더한다."""

    rho_b = OMEGA_B0 * math.exp(-3.0 * n)
    rho_r = OMEGA_R0 * math.exp(-4.0 * n)
    rho_v = amplitude * (1.0 - math.exp(-config.gamma * tau))
    rho_k = amplitude * (2.0 * u + 1.5 * u * u / config.kappa)
    p_k = amplitude * u * u / (2.0 * config.kappa)
    rho_prod = omega_prod0 * math.exp(-3.0 * n)
    e2 = rho_b + rho_r + rho_v + rho_k + rho_prod
    if not math.isfinite(e2) or e2 <= 0.0:
        raise ArithmeticError("quench trajectory reached non-positive E^2")
    return rho_b, rho_r, rho_v, rho_k, p_k, rho_prod, e2


def _quench_rhs(
    n: float,
    state: tuple[float, float],
    config: KineticClockConfig,
    amplitude: float,
    omega_prod0: float,
) -> tuple[float, float]:
    tau, u = state
    if u <= -0.99 * config.kappa:
        raise ArithmeticError("quench trajectory left the square-root domain")
    e = math.sqrt(_quench_densities(n, tau, u, config, amplitude, omega_prod0)[-1])
    root = math.sqrt(1.0 + u / config.kappa)
    current_shape = u * root
    current_derivative = (1.0 + 1.5 * u / config.kappa) / root
    tau_prime = root / e
    u_prime = (
        -3.0 * current_shape
        - config.gamma * math.exp(-config.gamma * tau) / (2.0 * e)
    ) / current_derivative
    return tau_prime, u_prime


def _quench_rk4(
    n: float,
    state: tuple[float, float],
    step: float,
    config: KineticClockConfig,
    amplitude: float,
    omega_prod0: float,
) -> tuple[float, float]:
    def add(base: tuple[float, float], delta: tuple[float, float], factor: float) -> tuple[float, float]:
        return base[0] + factor * delta[0], base[1] + factor * delta[1]

    k1 = _quench_rhs(n, state, config, amplitude, omega_prod0)
    k2 = _quench_rhs(n + 0.5 * step, add(state, k1, 0.5 * step), config, amplitude, omega_prod0)
    k3 = _quench_rhs(n + 0.5 * step, add(state, k2, 0.5 * step), config, amplitude, omega_prod0)
    k4 = _quench_rhs(n + step, add(state, k3, step), config, amplitude, omega_prod0)
    return (
        state[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _quench_integrate_backward(
    config: KineticClockConfig,
    b: float,
    keep: bool,
    omega_prod0: float,
) -> tuple[float, float, list[tuple[float, float, float]]]:
    amplitude = amplitude_from_b(b)
    state = (
        b / config.gamma,
        positive_u_from_density((OMEGA_K0 - omega_prod0) / amplitude, config.kappa),
    )
    n = 0.0
    step = N_INITIAL / config.steps
    grid = [(n, *state)] if keep else []
    for _ in range(config.steps):
        state = _quench_rk4(n, state, step, config, amplitude, omega_prod0)
        n += step
        if keep:
            grid.append((n, *state))
    return state[0], amplitude, grid


def _quench_shoot_b(config: KineticClockConfig, omega_prod0: float) -> float:
    low, high = 0.5 * config.gamma, 1.2 * config.gamma

    def endpoint(value: float) -> float:
        try:
            return _quench_integrate_backward(config, value, False, omega_prod0)[0]
        except (ArithmeticError, OverflowError, ValueError):
            return -math.inf

    while endpoint(high) <= 0.0:
        high *= 1.25
        if high >= 10.0 * config.gamma:
            raise ArithmeticError("could not bracket the quench shooting root")
    if endpoint(low) >= 0.0:
        raise ArithmeticError("quench lower shooting bracket is invalid")
    for _ in range(52):
        middle = 0.5 * (low + high)
        if endpoint(middle) > 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def solve_quench_background(
    omega_prod0: float,
    config: KineticClockConfig = KineticClockConfig(),
) -> BackgroundSolution:
    """정확한 a^-3 급냉 성분을 더한 동결 운동 배경을 푼다.

    ``omega_prod0`` 는 운동 부문 재고 예산의 일부를 대체한다. 운동 사격 경계는
    Omega_K' = OMEGA_K0 - omega_prod0 이고 E^2 는 omega_prod0 * (1+z)^3 항을 추가로
    가지므로 평탄한 현재 합은 1로 유지된다. ``omega_prod0 = 0.0`` 은 기본 게이트의
    ``solve_background`` 궤적으로 비트 단위 동일하게 환원된다.
    """

    if not 0.0 <= omega_prod0 <= OMEGA_K0:
        raise ValueError("omega_prod0 must lie in [0, OMEGA_K0]")
    b = _quench_shoot_b(config, omega_prod0)
    _, amplitude, descending = _quench_integrate_backward(config, b, True, omega_prod0)
    nodes: list[BackgroundNode] = []
    for n, tau, u in reversed(descending):
        e2 = _quench_densities(n, tau, u, config, amplitude, omega_prod0)[-1]
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
        raise ArithmeticError("the quench observation-window branch is not healthy")
    return solution


def e_of_z(solution: BackgroundSolution, z: float) -> float:
    return math.sqrt(solution.at_n(-math.log1p(z)).e2)


def calibration_grid(
    minimum: float = 1.0e-6,
    maximum: float = 0.05,
    count: int = 40,
) -> tuple[float, ...]:
    """사전 선언 탐색 격자: {0} 과 ``count`` 개의 로그 균등점의 합집합이다."""

    if not 0.0 < minimum < maximum or count < 2:
        raise ValueError("invalid calibration grid specification")
    ratio = maximum / minimum
    return (0.0,) + tuple(
        minimum * ratio ** (index / (count - 1)) for index in range(count)
    )


@dataclass(frozen=True)
class QuenchGridPoint:
    omega_prod0: float
    chi2: float
    scale: float
    status: str


@dataclass(frozen=True)
class QuenchCalibration:
    grid: tuple[QuenchGridPoint, ...]
    argmin_omega_prod0: float
    argmin_chi2: float
    argmin_scale: float
    lcdm_control_chi2: float
    delta_chi2_argmin_minus_lcdm_control: float
    same_frozen_boundary_lcdm_chi2: float
    dataset: str
    dof: int
    lcdm_control_note: str = LCDM_CONTROL_NOTE
    role: str = "SEEN_DATA_TARGET_AWARE_CALIBRATION_NOT_HOLDOUT"


def lcdm_control_chi2(dataset: BAODataset | None = None) -> float:
    """기본 게이트의 LCDM 대조군(12.60835 로 동결): CEForwardParams 기본값이다."""

    selected = dataset or named_bao_dataset("desi-dr2-all")
    diagnostic = assess_bao_fit(
        selected.data, CEForwardParams(), covariance=selected.covariance
    ).scale_fit_diagnostic
    if diagnostic is None:
        raise ArithmeticError("LCDM BAO scale profile was not available")
    return diagnostic.chi2


def same_frozen_boundary_lcdm_chi2(dataset: BAODataset | None = None) -> float:
    """동결 운동 경계 조합에서 평탄 LCDM 의 프로파일된 DESI chi^2 이다."""

    selected = dataset or named_bao_dataset("desi-dr2-all")
    inverse = invert_matrix(selected.covariance)

    def dimensionless_distance(z: float, intervals: int = 512) -> float:
        step = z / intervals

        def inverse_e(redshift: float) -> float:
            zp1 = 1.0 + redshift
            return 1.0 / math.sqrt(
                OMEGA_R0 * zp1**4 + (OMEGA_B0 + OMEGA_K0) * zp1**3 + OMEGA_V0
            )

        total = inverse_e(0.0) + inverse_e(z)
        for index in range(1, intervals):
            total += (4.0 if index % 2 else 2.0) * inverse_e(index * step)
        return total * step / 3.0

    shapes: list[float] = []
    values: list[float] = []
    for point in selected.data:
        zp1 = 1.0 + point.z
        e2 = OMEGA_R0 * zp1**4 + (OMEGA_B0 + OMEGA_K0) * zp1**3 + OMEGA_V0
        dh = 1.0 / math.sqrt(e2)
        dm = dimensionless_distance(point.z)
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
    return quadratic_form(residual, inverse)


def calibrate_omega_prod(
    grid: tuple[float, ...] | None = None,
    *,
    config: KineticClockConfig = KineticClockConfig(),
    dataset: BAODataset | None = None,
) -> QuenchCalibration:
    """사전 선언된 DESI 전용 보정 훑기를 돌리고 argmin 을 채택한다.

    실패한 격자점은 조용히 버리지 않고 상태와 무한대 chi^2 로 기록한다. argmin 은
    성공한 점들에서만 취한다.
    """

    selected = dataset or named_bao_dataset("desi-dr2-all")
    values = calibration_grid() if grid is None else grid
    points: list[QuenchGridPoint] = []
    for omega_prod0 in values:
        try:
            solution = solve_quench_background(omega_prod0, config)
            fit = profile_desi_bao(solution, selected)
            points.append(
                QuenchGridPoint(
                    omega_prod0=omega_prod0,
                    chi2=fit.chi2,
                    scale=fit.scale,
                    status="OK",
                )
            )
        except (ArithmeticError, OverflowError, ValueError) as error:
            points.append(
                QuenchGridPoint(
                    omega_prod0=omega_prod0,
                    chi2=math.inf,
                    scale=math.nan,
                    status=f"FAILED: {error}",
                )
            )
    successful = [point for point in points if point.status == "OK"]
    if not successful:
        raise ArithmeticError("no calibration grid point produced a healthy branch")
    best = min(successful, key=lambda point: point.chi2)
    control = lcdm_control_chi2(selected)
    dof = len(selected.data) - 1
    return QuenchCalibration(
        grid=tuple(points),
        argmin_omega_prod0=best.omega_prod0,
        argmin_chi2=best.chi2,
        argmin_scale=best.scale,
        lcdm_control_chi2=control,
        delta_chi2_argmin_minus_lcdm_control=best.chi2 - control,
        same_frozen_boundary_lcdm_chi2=same_frozen_boundary_lcdm_chi2(selected),
        dataset=selected.name,
        dof=dof,
    )


@dataclass(frozen=True)
class HoldoutEvaluation:
    scale: float
    scale_sigma: float
    chi2: float
    dof: int
    n_points: int
    kinds: tuple[str, ...]
    family: str
    role: str = "PURE_HOLDOUT_EVALUATOR_DATA_PASSED_AS_ARGUMENTS_ONLY"


def holdout_profiled_chi2(
    solution: BackgroundSolution,
    z_bins: tuple[float, ...],
    observed: tuple[float, ...],
    covariance: tuple[tuple[float, ...], ...],
    kinds: tuple[str, ...],
) -> HoldoutEvaluation:
    """홀드아웃 자료에 대한 순수 프로파일 절편, 공분산 조건부 chi^2 이다.

    관측 벡터, 공분산, 점별 관측량 종류(rd 단위의 ``dm``/``dh``/``dv``, 또는 직접
    H(z) 의 ``hz``)는 인자로만 받는다. 이 함수는 파일이나 네트워크에 접근하지 않고
    자료를 내장하지 않으므로, 봉인 평가 단계가 공급하기 전에는 홀드아웃 자료를
    접촉하지 않는다.

    곱셈 절편 하나를 전체 공분산에 대해 해석적으로 프로파일한다. BAO 계열은
    c/(H0 rd), H(z) 계열은 H0 다. 두 계열은 절편을 공유하지 않으므로 한 호출에
    섞으면 거부한다. chi^2 는 프로파일된 절편에서의 일반화 최소제곱값이고
    dof = n - 1 이다.
    """

    size = len(z_bins)
    if size == 0:
        raise ValueError("holdout evaluation requires at least one point")
    if len(observed) != size or len(kinds) != size:
        raise ValueError("z_bins, observed, and kinds must have equal length")
    if len(covariance) != size or any(len(row) != size for row in covariance):
        raise ValueError("covariance must be square and match the data length")
    for i in range(size):
        for j in range(size):
            entry = covariance[i][j]
            if not math.isfinite(entry):
                raise ValueError("covariance must be finite")
            if abs(entry - covariance[j][i]) > 1.0e-12 * max(1.0, abs(entry)):
                raise ValueError("covariance must be symmetric")
    normalized = []
    for kind in kinds:
        key = kind.strip().lower()
        if key not in _KIND_ALIASES:
            raise ValueError(f"unsupported observable kind: {kind!r}")
        normalized.append(_KIND_ALIASES[key])
    kind_set = set(normalized)
    if kind_set <= BAO_KIND_FAMILY:
        family = "bao_c_over_h0rd"
    elif kind_set <= HZ_KIND_FAMILY:
        family = "hz_h0"
    else:
        raise ValueError(
            "observable kinds mix the BAO (dm/dh/dv) and H(z) intercept "
            "families; profile them in separate calls"
        )

    shapes: list[float] = []
    for z, kind in zip(z_bins, normalized):
        if z <= 0.0:
            raise ValueError("holdout redshifts must be positive")
        e_value = e_of_z(solution, z)
        if kind == "hz":
            shapes.append(e_value)
        else:
            dh = 1.0 / e_value
            dm = _dimensionless_distance(z, solution)
            dv = (z * dm * dm * dh) ** (1.0 / 3.0)
            shapes.append({"dh": dh, "dm": dm, "dv": dv}[kind])
    inverse = invert_matrix(tuple(tuple(row) for row in covariance))
    shape_tuple = tuple(shapes)
    value_tuple = tuple(observed)
    gg = quadratic_form(shape_tuple, inverse)
    if not math.isfinite(gg) or gg <= 0.0:
        raise ValueError("covariance is not positive definite for these shapes")
    gd = sum(
        shape_tuple[i] * inverse[i][j] * value_tuple[j]
        for i in range(size)
        for j in range(size)
    )
    scale = gd / gg
    residual = tuple(scale * shape - value for shape, value in zip(shape_tuple, value_tuple))
    return HoldoutEvaluation(
        scale=scale,
        scale_sigma=1.0 / math.sqrt(gg),
        chi2=quadratic_form(residual, inverse),
        dof=size - 1,
        n_points=size,
        kinds=tuple(normalized),
        family=family,
    )


E_TABLE_REDSHIFTS: tuple[float, ...] = (0.07, 0.1, 0.5, 1.0, 1.5)


def e_of_z_table(
    solution: BackgroundSolution,
    baseline: BackgroundSolution,
    redshifts: tuple[float, ...] = E_TABLE_REDSHIFTS,
) -> tuple[dict[str, float], ...]:
    rows = []
    for z in redshifts:
        e_model = e_of_z(solution, z)
        e_base = e_of_z(baseline, z)
        rows.append(
            {
                "z": z,
                "e_argmin": e_model,
                "e_omega_prod0_zero": e_base,
                "relative_difference": e_model / e_base - 1.0,
            }
        )
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the full calibration curve and argmin report as JSON",
    )
    args = parser.parse_args()

    calibration = calibrate_omega_prod()
    baseline = solve_quench_background(0.0)
    best_solution = (
        baseline
        if calibration.argmin_omega_prod0 == 0.0
        else solve_quench_background(calibration.argmin_omega_prod0)
    )
    table = e_of_z_table(best_solution, baseline)
    pantheon = compare_pantheon_binned(solution=best_solution)

    print("# SNKC quench-production background calibration gate")
    print(f"dataset {calibration.dataset}")
    print(f"role {calibration.role}")
    print(f"grid_points {len(calibration.grid)}")
    print(f"argmin_omega_prod0 {calibration.argmin_omega_prod0:.12g}")
    print(f"argmin_desi_chi2 {calibration.argmin_chi2:.12g}")
    print(f"argmin_desi_dof {calibration.dof}")
    print(f"argmin_profiled_scale {calibration.argmin_scale:.12g}")
    print(f"lcdm_control_chi2 {calibration.lcdm_control_chi2:.12g}")
    print(
        "delta_chi2_argmin_minus_lcdm_control",
        f"{calibration.delta_chi2_argmin_minus_lcdm_control:.12g}",
    )
    print(
        "same_frozen_boundary_lcdm_chi2",
        f"{calibration.same_frozen_boundary_lcdm_chi2:.12g}",
    )
    print(f"lcdm_control_note {calibration.lcdm_control_note}")
    print(
        "pantheon40_kinetic_chi2_report_only",
        f"{pantheon.kinetic.chi2:.12g}",
    )
    print("pantheon40_lcdm_chi2_report_only", f"{pantheon.lcdm.chi2:.12g}")
    print(
        "pantheon40_delta_chi2_kinetic_minus_lcdm",
        f"{pantheon.delta_chi2_kinetic_minus_lcdm:.12g}",
    )
    print("pantheon40_role NOT_USED_FOR_CALIBRATION_REPORT_ONLY")
    for row in table:
        print(
            f"e_of_z z={row['z']:.3g}",
            f"E_argmin={row['e_argmin']:.12g}",
            f"E_zero={row['e_omega_prod0_zero']:.12g}",
            f"rel_diff={row['relative_difference']:.6g}",
        )
    print("holdout_contact NONE_EVALUATOR_INTERFACE_ONLY")
    print("status SEEN_DATA_CALIBRATION_STAGE_ONLY")

    if args.json_out is not None:
        payload = {
            "calibration": asdict(calibration),
            "e_of_z_table": list(table),
            "pantheon40_report_only": {
                "kinetic_chi2": pantheon.kinetic.chi2,
                "lcdm_chi2": pantheon.lcdm.chi2,
                "delta_chi2_kinetic_minus_lcdm": pantheon.delta_chi2_kinetic_minus_lcdm,
                "dof": pantheon.kinetic.dof,
                "role": "NOT_USED_FOR_CALIBRATION_REPORT_ONLY",
            },
            "frozen_references": {
                "base_desi_chi2": FROZEN_BASE_DESI_CHI2,
                "base_desi_scale": FROZEN_BASE_DESI_SCALE,
                "lcdm_control_chi2": FROZEN_LCDM_CONTROL_CHI2,
            },
            "holdout_contact": "NONE_EVALUATOR_INTERFACE_ONLY",
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"json_written {args.json_out}")
    return 0


# ---------------------------------------------------------------------------
# 2. 해석적 자외선 꼬리 상계
# ---------------------------------------------------------------------------


_LOG_FLOAT_MAX = math.log(sys.float_info.max)
_MIN_SUBNORMAL = math.nextafter(0.0, 1.0)
_LOG_MIN_SUBNORMAL = math.log(_MIN_SUBNORMAL)


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _exp_upward(log_value: float) -> float:
    """양의 해석적 상계를 계산하고 마지막 exp 를 위쪽으로 한 ulp 민다.

    마지막 변환과 비정규(subnormal) 경우를 보호하지만 구간 산술은 아니다. 앞선
    log/expm1 연산은 어느 방향으로도 반올림될 수 있다.
    """

    if log_value == -math.inf:
        return 0.0
    if not math.isfinite(log_value) or log_value >= _LOG_FLOAT_MAX:
        raise ValueError("tail bound is outside the finite numerical domain")
    if log_value <= _LOG_MIN_SUBNORMAL:
        return _MIN_SUBNORMAL
    value = math.exp(log_value)
    upper = math.nextafter(value, math.inf)
    if not math.isfinite(upper):
        raise ValueError("tail bound is outside the finite numerical domain")
    return upper


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


@dataclass(frozen=True)
class SmoothQuenchTailCertificate:
    """선언된 운동량 문턱 하나에 대한 닫힌 형태 꼬리 영수증이다."""

    momentum_start: float
    exponential_decay_rate: float
    log_occupation_coefficient: float
    present_number_density_upper: float
    present_energy_density_upper: float
    present_pressure_upper: float
    omega_produced_upper: float
    scale_factor_at_production: float
    critical_density_today: float
    proof_assumptions: tuple[str, ...]
    numerical_status: str = (
        "FLOAT_EVALUATION_OF_ANALYTIC_BOUND_NOT_INTERVAL_CERTIFIED"
    )
    role: str = (
        "ANALYTIC_CREATED_EXCESS_UV_TAIL_BOUND_NOT_RENORMALIZED_FLRW_STRESS"
    )


def _log_occupation_coefficient(
    species: QuantumSeatSpecies,
    momentum_start: float,
) -> tuple[float, float]:
    rate = 2.0 * math.pi * species.duration
    mass_square_difference = abs(
        (species.mass_out - species.mass_in)
        * (species.mass_out + species.mass_in)
    )
    if mass_square_difference == 0.0:
        return rate, -math.inf
    if not math.isfinite(rate) or not math.isfinite(mass_square_difference):
        raise ValueError("quench tail parameters are outside the finite domain")
    c_value = math.pi * species.duration * mass_square_difference / 4.0
    rate_times_start = rate * momentum_start
    denominator_factor = -math.expm1(-rate_times_start)
    stimulation = 1.0 + 2.0 * species.initial_mode_occupation
    log_coefficient = (
        math.log(stimulation)
        + math.log(4.0)
        + 2.0 * math.log(c_value)
        + 2.0 * c_value / momentum_start
        - 2.0 * math.log(denominator_factor)
    )
    if not math.isfinite(log_coefficient):
        raise ValueError("occupation tail coefficient is not finite")
    return rate, log_coefficient


def smooth_quench_created_occupation_tail_upper(
    species: QuantumSeatSpecies,
    *,
    momentum: object,
    momentum_start: object,
) -> float:
    """``p >= P`` 에 대한 해석적 생성 점유수 상계를 돌려준다."""

    if not isinstance(species, QuantumSeatSpecies):
        raise ValueError("species must be a QuantumSeatSpecies")
    p_value = _positive_finite(momentum, "momentum")
    start = _positive_finite(momentum_start, "momentum_start")
    if p_value < start:
        raise ValueError("momentum must be >= momentum_start")
    rate, log_coefficient = _log_occupation_coefficient(species, start)
    if log_coefficient == -math.inf:
        return 0.0
    return _exp_upward(
        log_coefficient - 2.0 * math.log(p_value) - rate * p_value
    )


def smooth_quench_present_tail_certificate(
    species: QuantumSeatSpecies,
    *,
    momentum_start: object,
    scale_factor_at_production: object,
    critical_density_today: object,
) -> SmoothQuenchTailCertificate:
    """``momentum_start`` 위에서 생략된 생성 초과 모드 전부를 상계한다."""

    if not isinstance(species, QuantumSeatSpecies):
        raise ValueError("species must be a QuantumSeatSpecies")
    start = _positive_finite(momentum_start, "momentum_start")
    scale_factor = _positive_finite(
        scale_factor_at_production,
        "scale_factor_at_production",
    )
    if scale_factor > 1.0:
        raise ValueError("scale_factor_at_production must be <= 1")
    critical_density = _positive_finite(
        critical_density_today,
        "critical_density_today",
    )
    rate, log_coefficient = _log_occupation_coefficient(species, start)
    if log_coefficient == -math.inf:
        number_upper = 0.0
        energy_upper = 0.0
        pressure_upper = 0.0
        omega_upper = 0.0
    else:
        log_common = log_coefficient - rate * start
        log_prefactor = (
            math.log(species.degeneracy)
            - math.log(2.0 * math.pi * math.pi)
            + 3.0 * math.log(scale_factor)
        )
        log_number_upper = log_prefactor + log_common - math.log(rate)
        log_rest_term = math.log(species.mass_out) - math.log(rate)
        log_momentum_term = (
            math.log(scale_factor)
            - math.log(rate)
            + math.log(start + 1.0 / rate)
        )
        log_energy_bracket = _logaddexp(log_rest_term, log_momentum_term)
        log_energy_upper = log_prefactor + log_common + log_energy_bracket
        number_upper = _exp_upward(log_number_upper)
        energy_upper = _exp_upward(log_energy_upper)
        pressure_upper = math.nextafter(energy_upper / 3.0, math.inf)
        omega_upper = math.nextafter(
            energy_upper / critical_density,
            math.inf,
        )
        if not math.isfinite(omega_upper):
            raise ValueError("omega tail bound is outside the finite domain")

    return SmoothQuenchTailCertificate(
        momentum_start=start,
        exponential_decay_rate=rate,
        log_occupation_coefficient=log_coefficient,
        present_number_density_upper=number_upper,
        present_energy_density_upper=energy_upper,
        present_pressure_upper=pressure_upper,
        omega_produced_upper=omega_upper,
        scale_factor_at_production=scale_factor,
        critical_density_today=critical_density,
        proof_assumptions=(
            "exact asymptotic Minkowski smooth-tanh Bogoliubov spectrum",
            "created excess f=(1+2*n_in)|beta|^2",
            "stable decoupled constant-mass propagation after production",
            "physical momentum redshifts as p0=a_star*p_star",
            "bound certifies p>=momentum_start only",
        ),
    )


# ---------------------------------------------------------------------------
# 3. 봉인 홀드아웃 평가 (사전등록 프로토콜, 숫자·판정 논리 불변)
# ---------------------------------------------------------------------------


DATA_DIR = Path("benchmarks/cosmology/snkc_quench_bg_holdout_v1")

Z_ELG = 0.845
Z_LYA = 2.334
LIKELIHOOD_FLOOR = 1.0e-300
OUT_OF_GRID_PENALTY = 1.0e6  # 축마다 정규화 초과량의 제곱에 곱한다.
KILL_RULE_DELTA_CHI2 = 9.0

GAUSSIAN_BLOCK_FILES = (
    ("sdss_DR12_LRG_BAO_DMDH.dat", "sdss_DR12_LRG_BAO_DMDH_covtot.txt"),
    ("sdss_DR16_LRG_BAO_DMDH.dat", "sdss_DR16_LRG_BAO_DMDH_covtot.txt"),
    ("sdss_DR16_QSO_BAO_DMDH.txt", "sdss_DR16_QSO_BAO_DMDH_covtot.txt"),
)

_OBS_KIND = {"DM_over_rs": "dm", "DH_over_rs": "dh", "DV_over_rs": "dv"}


# --------------------------------------------------------------------------
# 배경 모형(동일 취급, E(z) 의 출처만 다름)
# --------------------------------------------------------------------------


class KineticQuenchModel:
    """M1: Omega_prod,0 = 0 의 동결 운동+급냉 배경이다."""

    name = "kinetic_quench_omega_prod0_zero"

    def __init__(self) -> None:
        self.solution = solve_quench_background(0.0)

    def e(self, z: float) -> float:
        return e_of_z(self.solution, z)

    def dc(self, z: float) -> float:
        return _dimensionless_distance(z, self.solution)


class FrozenBoundaryLCDMModel:
    """M2: 정확한 동결 경계 조합의 평탄 LambdaCDM 이다."""

    name = "same_frozen_boundary_flat_lcdm"

    def e(self, z: float) -> float:
        zp1 = 1.0 + z
        return math.sqrt(
            OMEGA_R0 * zp1**4 + (OMEGA_B0 + OMEGA_K0) * zp1**3 + OMEGA_V0
        )

    def dc(self, z: float, intervals: int = 512) -> float:
        # 운동 가지와 같은 512 구간 해상도의 심프슨 적분이다
        # (same_frozen_boundary_lcdm_chi2 를 그대로 따른다).
        if intervals % 2:
            intervals += 1
        step = z / intervals
        total = 1.0 / self.e(0.0) + 1.0 / self.e(z)
        for index in range(1, intervals):
            total += (4.0 if index % 2 else 2.0) / self.e(index * step)
        return total * step / 3.0


def model_shape(model, z: float, kind: str) -> float:
    if kind == "hz":
        return model.e(z)
    dh = 1.0 / model.e(z)
    if kind == "dh":
        return dh
    dm = model.dc(z)
    if kind == "dm":
        return dm
    if kind == "dv":
        return (z * dm * dm * dh) ** (1.0 / 3.0)
    raise ValueError(f"unsupported observable kind: {kind!r}")


# --------------------------------------------------------------------------
# 로더 (DATA_DIR 로 제한)
# --------------------------------------------------------------------------


def _read_matrix(path: Path) -> tuple[tuple[float, ...], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(tuple(float(token) for token in stripped.split()))
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError(f"{path.name}: covariance file is not square")
    return tuple(rows)


def _read_bao_points(path: Path) -> tuple[tuple[float, float, str], ...]:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 3 or tokens[2] not in _OBS_KIND:
            raise ValueError(f"{path.name}: unrecognized BAO row {stripped!r}")
        points.append((float(tokens[0]), float(tokens[1]), _OBS_KIND[tokens[2]]))
    if not points:
        raise ValueError(f"{path.name}: no BAO rows found")
    return tuple(points)


@dataclass(frozen=True)
class GaussianBlock:
    z: tuple[float, ...]
    observed: tuple[float, ...]
    kinds: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    block_sizes: tuple[int, ...]
    block_names: tuple[str, ...]


def load_gaussian_block(data_dir: Path = DATA_DIR) -> GaussianBlock:
    """DR12 LRG + DR16 LRG + DR16 QSO 를 블록 대각 가우스 블록 하나로 읽는다."""

    z: list[float] = []
    observed: list[float] = []
    kinds: list[str] = []
    matrices: list[tuple[tuple[float, ...], ...]] = []
    names: list[str] = []
    for data_name, cov_name in GAUSSIAN_BLOCK_FILES:
        points = _read_bao_points(data_dir / data_name)
        matrix = _read_matrix(data_dir / cov_name)
        if len(matrix) != len(points):
            raise ValueError(f"{cov_name}: covariance size does not match {data_name}")
        for redshift, value, kind in points:
            z.append(redshift)
            observed.append(value)
            kinds.append(kind)
        matrices.append(matrix)
        names.append(data_name)
    size = len(z)
    covariance = [[0.0] * size for _ in range(size)]
    offset = 0
    for matrix in matrices:
        for i, row in enumerate(matrix):
            for j, entry in enumerate(row):
                covariance[offset + i][offset + j] = entry
        offset += len(matrix)
    return GaussianBlock(
        z=tuple(z),
        observed=tuple(observed),
        kinds=tuple(kinds),
        covariance=tuple(tuple(row) for row in covariance),
        block_sizes=tuple(len(matrix) for matrix in matrices),
        block_names=tuple(names),
    )


@dataclass(frozen=True)
class LikelihoodTable1D:
    """1차원 격자에 표로 놓인 관측량 하나의 정규화 가능도다."""

    grid: tuple[float, ...]
    likelihood: tuple[float, ...]


def load_elg_dv_table(data_dir: Path = DATA_DIR) -> LikelihoodTable1D:
    grid: list[float] = []
    likelihood: list[float] = []
    for line in (data_dir / "sdss_DR16_ELG_BAO_DVtable.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first, second = stripped.split()
        grid.append(float(first))
        likelihood.append(float(second))
    if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("ELG DV table must be strictly increasing")
    return LikelihoodTable1D(grid=tuple(grid), likelihood=tuple(likelihood))


@dataclass(frozen=True)
class LikelihoodGrid2D:
    """직사각 격자에 표로 놓인 (DM/rd, DH/rd) 의 가능도(비)다."""

    dm_axis: tuple[float, ...]
    dh_axis: tuple[float, ...]
    likelihood: tuple[tuple[float, ...], ...]  # [dm_index][dh_index]


def load_lya_grid(file_name: str, data_dir: Path = DATA_DIR) -> LikelihoodGrid2D:
    entries: dict[tuple[float, float], float] = {}
    dm_values: list[float] = []
    dh_values: list[float] = []
    for line in (data_dir / file_name).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        dm_token, dh_token, like_token = stripped.split()
        dm, dh, like = float(dm_token), float(dh_token), float(like_token)
        entries[(dm, dh)] = like
        dm_values.append(dm)
        dh_values.append(dh)
    dm_axis = tuple(sorted(set(dm_values)))
    dh_axis = tuple(sorted(set(dh_values)))
    if len(entries) != len(dm_axis) * len(dh_axis):
        raise ValueError(f"{file_name}: grid is not rectangular")
    likelihood = tuple(
        tuple(entries[(dm, dh)] for dh in dh_axis) for dm in dm_axis
    )
    return LikelihoodGrid2D(dm_axis=dm_axis, dh_axis=dh_axis, likelihood=likelihood)


@dataclass(frozen=True)
class CCData:
    z: tuple[float, ...]
    hz: tuple[float, ...]
    err_hz: tuple[float, ...]


def load_cc_data(data_dir: Path = DATA_DIR) -> CCData:
    z: list[float] = []
    hz: list[float] = []
    err: list[float] = []
    for line in (data_dir / "HzTable_MM_BC03.dat").read_text(
        encoding="utf-8"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split(",")
        z.append(float(tokens[0]))
        hz.append(float(tokens[1]))
        err.append(float(tokens[2]))
    if len(z) != 15:
        raise ValueError(f"expected the 15-point homogeneous CC subset, got {len(z)}")
    return CCData(z=tuple(z), hz=tuple(hz), err_hz=tuple(err))


def load_mm20_table(data_dir: Path = DATA_DIR) -> dict[str, tuple[float, ...]]:
    columns: dict[str, list[float]] = {
        "z": [],
        "imf": [],
        "stlib": [],
        "mod": [],
        "mod_ooo": [],
    }
    for line in (data_dir / "data_MM20.dat").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = [float(token) for token in stripped.split()]
        if len(tokens) != 5:
            raise ValueError("data_MM20.dat rows must have 5 columns")
        for key, value in zip(("z", "imf", "stlib", "mod", "mod_ooo"), tokens):
            columns[key].append(value)
    return {key: tuple(values) for key, values in columns.items()}


# --------------------------------------------------------------------------
# 곱셈 절편 하나를 프로파일하는 일반화 최소제곱(GLS)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GLSProfile:
    scale: float
    scale_sigma: float
    chi2: float
    dof: int
    pulls: tuple[float, ...]


def gls_profile(
    shapes: tuple[float, ...],
    observed: tuple[float, ...],
    covariance: tuple[tuple[float, ...], ...],
) -> GLSProfile:
    """전체 공분산에 대해 곱셈 척도 하나를 정확히 해석적으로 프로파일한다.

    chi2(s) = (s*g - d)^T C^-1 (s*g - d) 는 s 의 이차식이고 최소점은
    s = (g^T C^-1 d) / (g^T C^-1 g) 다. 풀(pull)은 프로파일된 척도에서 점별 잔차를
    sqrt(diag C) 로 나눈 값이다(참고 보고 전용).
    """

    size = len(shapes)
    if size == 0 or len(observed) != size or len(covariance) != size:
        raise ValueError("shapes, observed, and covariance sizes must agree")
    inverse = invert_matrix(tuple(tuple(row) for row in covariance))
    gg = quadratic_form(tuple(shapes), inverse)
    if not math.isfinite(gg) or gg <= 0.0:
        raise ValueError("covariance is not positive definite for these shapes")
    gd = sum(
        shapes[i] * inverse[i][j] * observed[j]
        for i in range(size)
        for j in range(size)
    )
    scale = gd / gg
    residual = tuple(scale * g - d for g, d in zip(shapes, observed))
    pulls = tuple(
        (observed[i] - scale * shapes[i]) / math.sqrt(covariance[i][i])
        for i in range(size)
    )
    return GLSProfile(
        scale=scale,
        scale_sigma=1.0 / math.sqrt(gg),
        chi2=quadratic_form(residual, inverse),
        dof=size - 1,
        pulls=pulls,
    )


def chi2_at_scale(
    scale: float,
    shapes: tuple[float, ...],
    observed: tuple[float, ...],
    inverse: tuple[tuple[float, ...], ...],
) -> float:
    residual = tuple(scale * g - d for g, d in zip(shapes, observed))
    return quadratic_form(residual, inverse)


# --------------------------------------------------------------------------
# 표로 주어진 -2 ln L 항 (선언된 보간과 격자 밖 규칙)
# --------------------------------------------------------------------------


def _interp_1d(x: float, grid: tuple[float, ...], values: tuple[float, ...]) -> tuple[float, float]:
    """선형 보간이다. (고정된 x 에서의 값, 벌점) 을 돌려준다."""

    lo, hi = grid[0], grid[-1]
    span = hi - lo
    penalty = 0.0
    if x < lo:
        penalty = OUT_OF_GRID_PENALTY * ((lo - x) / span) ** 2
        x = lo
    elif x > hi:
        penalty = OUT_OF_GRID_PENALTY * ((x - hi) / span) ** 2
        x = hi
    left, right = 0, len(grid) - 1
    while right - left > 1:
        middle = (left + right) // 2
        if grid[middle] <= x:
            left = middle
        else:
            right = middle
    weight = (x - grid[left]) / (grid[right] - grid[left])
    return values[left] + weight * (values[right] - values[left]), penalty


def neg2_log_like_1d(table: LikelihoodTable1D, value: float) -> float:
    like, penalty = _interp_1d(value, table.grid, table.likelihood)
    return -2.0 * math.log(max(like, LIKELIHOOD_FLOOR)) + penalty


def neg2_log_like_2d(grid: LikelihoodGrid2D, dm: float, dh: float) -> float:
    def clamp(x: float, axis: tuple[float, ...]) -> tuple[float, float]:
        lo, hi = axis[0], axis[-1]
        span = hi - lo
        if x < lo:
            return lo, OUT_OF_GRID_PENALTY * ((lo - x) / span) ** 2
        if x > hi:
            return hi, OUT_OF_GRID_PENALTY * ((x - hi) / span) ** 2
        return x, 0.0

    dm_clamped, penalty_dm = clamp(dm, grid.dm_axis)
    dh_clamped, penalty_dh = clamp(dh, grid.dh_axis)

    def bracket(x: float, axis: tuple[float, ...]) -> tuple[int, float]:
        left, right = 0, len(axis) - 1
        while right - left > 1:
            middle = (left + right) // 2
            if axis[middle] <= x:
                left = middle
            else:
                right = middle
        return left, (x - axis[left]) / (axis[left + 1] - axis[left])

    i, wx = bracket(dm_clamped, grid.dm_axis)
    j, wy = bracket(dh_clamped, grid.dh_axis)
    like = (
        grid.likelihood[i][j] * (1.0 - wx) * (1.0 - wy)
        + grid.likelihood[i + 1][j] * wx * (1.0 - wy)
        + grid.likelihood[i][j + 1] * (1.0 - wx) * wy
        + grid.likelihood[i + 1][j + 1] * wx * wy
    )
    return -2.0 * math.log(max(like, LIKELIHOOD_FLOOR)) + penalty_dm + penalty_dh


# --------------------------------------------------------------------------
# 수치 1차원 프로파일 (선언: 괄호 훑기 + 황금분할)
# --------------------------------------------------------------------------


def minimize_scalar(
    objective,
    center: float,
    half_width_fraction: float = 0.2,
    scan_points: int = 401,
    tolerance: float = 1.0e-12,
    max_widenings: int = 4,
) -> tuple[float, float]:
    """``center`` 주위를 괄호 훑기한 뒤 황금분할(golden section)로 정련한다.

    훑기 최소점이 훑기 경계에 놓이면 정련 전에 괄호를 두 배로 넓힌다(선언된
    ``max_widenings`` 회까지). (argmin, 최소값) 을 돌려준다.
    """

    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for widening in range(max_widenings + 1):
        width = half_width_fraction * (2.0**widening) * abs(center)
        low, high = center - width, center + width
        step = (high - low) / (scan_points - 1)
        values = [objective(low + k * step) for k in range(scan_points)]
        best = min(range(scan_points), key=values.__getitem__)
        if 0 < best < scan_points - 1:
            a, b = low + (best - 1) * step, low + (best + 1) * step
            break
    else:
        raise ArithmeticError("numerical scale profile did not bracket a minimum")
    x1 = b - golden * (b - a)
    x2 = a + golden * (b - a)
    f1, f2 = objective(x1), objective(x2)
    while (b - a) > tolerance * max(1.0, abs(center)):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - golden * (b - a)
            f1 = objective(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + golden * (b - a)
            f2 = objective(x2)
    argmin = 0.5 * (a + b)
    return argmin, objective(argmin)


# --------------------------------------------------------------------------
# 평가 프로토콜
# --------------------------------------------------------------------------


def evaluate_primary_p(model, block: GaussianBlock) -> GLSProfile:
    shapes = tuple(model_shape(model, z, kind) for z, kind in zip(block.z, block.kinds))
    return gls_profile(shapes, block.observed, block.covariance)


def evaluate_primary_x(
    model,
    block: GaussianBlock,
    elg: LikelihoodTable1D,
    lyauto: LikelihoodGrid2D,
    lyxqso: LikelihoodGrid2D,
) -> dict:
    shapes = tuple(model_shape(model, z, kind) for z, kind in zip(block.z, block.kinds))
    inverse = invert_matrix(block.covariance)
    dv_shape = model_shape(model, Z_ELG, "dv")
    dm_shape = model_shape(model, Z_LYA, "dm")
    dh_shape = model_shape(model, Z_LYA, "dh")

    def objective(scale: float) -> float:
        return (
            chi2_at_scale(scale, shapes, block.observed, inverse)
            + neg2_log_like_1d(elg, scale * dv_shape)
            + neg2_log_like_2d(lyauto, scale * dm_shape, scale * dh_shape)
            + neg2_log_like_2d(lyxqso, scale * dm_shape, scale * dh_shape)
        )

    center = gls_profile(shapes, block.observed, block.covariance).scale
    scale, total = minimize_scalar(objective, center)
    return {
        "profiled_scale": scale,
        "chi2_total": total,
        "components_at_profiled_scale": {
            "gaussian_block_chi2": chi2_at_scale(scale, shapes, block.observed, inverse),
            "elg_dv_neg2lnl": neg2_log_like_1d(elg, scale * dv_shape),
            "lyauto_neg2lnl": neg2_log_like_2d(
                lyauto, scale * dm_shape, scale * dh_shape
            ),
            "lyxqso_neg2lnl": neg2_log_like_2d(
                lyxqso, scale * dm_shape, scale * dh_shape
            ),
        },
        "model_observables_at_profiled_scale": {
            "elg_dv_over_rd": scale * dv_shape,
            "lya_dm_over_rd": scale * dm_shape,
            "lya_dh_over_rd": scale * dh_shape,
        },
    }


def cc_covariance(
    data: CCData,
    mm20: dict[str, tuple[float, ...]],
    sps_column: str,
) -> tuple[tuple[float, ...], ...]:
    """diag(errHz^2) + IMF + stlib + SPS 퍼센트 성분의 외적이다."""

    if sps_column not in ("mod", "mod_ooo"):
        raise ValueError("sps_column must be 'mod' or 'mod_ooo'")
    size = len(data.z)
    covariance = [[0.0] * size for _ in range(size)]
    for i in range(size):
        covariance[i][i] = data.err_hz[i] ** 2

    def interp_percent(column: str) -> tuple[float, ...]:
        grid, values = mm20["z"], mm20[column]
        out = []
        for z in data.z:
            value, _ = _interp_1d(z, grid, values)
            out.append(value)
        return tuple(out)

    for column in ("imf", "stlib", sps_column):
        fractions = interp_percent(column)
        component = tuple(
            data.hz[i] * fractions[i] / 100.0 for i in range(size)
        )
        for i in range(size):
            for j in range(size):
                covariance[i][j] += component[i] * component[j]
    return tuple(tuple(row) for row in covariance)


def evaluate_cc(model, data: CCData, mm20: dict[str, tuple[float, ...]], sps_column: str) -> GLSProfile:
    shapes = tuple(model.e(z) for z in data.z)
    covariance = cc_covariance(data, mm20, sps_column)
    return gls_profile(shapes, data.hz, covariance)


def run_sealed_evaluation(data_dir: Path = DATA_DIR) -> dict:
    block = load_gaussian_block(data_dir)
    elg = load_elg_dv_table(data_dir)
    lyauto = load_lya_grid("sdss_DR16_LYAUTO_BAO_DMDHgrid.txt", data_dir)
    lyxqso = load_lya_grid("sdss_DR16_LYxQSO_BAO_DMDHgrid.txt", data_dir)
    cc_data = load_cc_data(data_dir)
    mm20 = load_mm20_table(data_dir)

    models = {"M1": KineticQuenchModel(), "M2": FrozenBoundaryLCDMModel()}
    report: dict = {
        "manifest": "experiments/preregistration/cosmology_snkc_quench_bg_v1.json",
        "ledger": "SNKC-R2-THEATER-QUENCH-BG-PREREG-10",
        "data_dir": str(data_dir),
        "fitted_parameter_count_on_holdout": 0,
        "profiled_nuisances": {
            "primary": "common scale s = c/(H0*rd)",
            "secondary": "H0 intercept",
        },
        "models": {key: model.name for key, model in models.items()},
        "primary_P": {},
        "primary_X": {},
        "secondary_CC": {},
    }

    # 교차 확인: M1 의 1차 P 를 사전등록 평가기 인터페이스로도 계산한다.
    m1 = models["M1"]
    crosscheck = holdout_profiled_chi2(
        m1.solution, block.z, block.observed, block.covariance, block.kinds
    )

    for key, model in models.items():
        p = evaluate_primary_p(model, block)
        report["primary_P"][key] = {
            "chi2": p.chi2,
            "dof": p.dof,
            "profiled_scale": p.scale,
            "scale_sigma": p.scale_sigma,
            "pulls": list(p.pulls),
        }
        report["primary_X"][key] = evaluate_primary_x(model, block, elg, lyauto, lyxqso)
        report["secondary_CC"][key] = {}
        for sps_column, label in (("mod", "main"), ("mod_ooo", "sensitivity")):
            cc = evaluate_cc(model, cc_data, mm20, sps_column)
            report["secondary_CC"][key][sps_column] = {
                "role": label,
                "chi2": cc.chi2,
                "dof": cc.dof,
                "chi2_over_dof": cc.chi2 / cc.dof,
                "profiled_h0_km_s_mpc": cc.scale,
                "h0_sigma_km_s_mpc": cc.scale_sigma,
                "pulls": list(cc.pulls),
            }

    report["primary_P"]["m1_evaluator_interface_crosscheck"] = {
        "chi2": crosscheck.chi2,
        "profiled_scale": crosscheck.scale,
        "matches_generic_gls": (
            abs(crosscheck.chi2 - report["primary_P"]["M1"]["chi2"]) < 1.0e-9
            and abs(crosscheck.scale - report["primary_P"]["M1"]["profiled_scale"])
            < 1.0e-9
        ),
    }

    delta_p = report["primary_P"]["M1"]["chi2"] - report["primary_P"]["M2"]["chi2"]
    delta_x = (
        report["primary_X"]["M1"]["chi2_total"]
        - report["primary_X"]["M2"]["chi2_total"]
    )
    rejected = delta_p > KILL_RULE_DELTA_CHI2 or delta_x > KILL_RULE_DELTA_CHI2
    report["verdict"] = {
        "kill_rule": f"delta_chi2 (M1 - M2) > +{KILL_RULE_DELTA_CHI2:g} on P or X",
        "delta_chi2_P": delta_p,
        "delta_chi2_X": delta_x,
        "rejected": rejected,
        "status": "REJECTED" if rejected else "NOT_REJECTED_SHAPE_CONSISTENCY_ONLY",
        "no_superiority_claim": (
            "equal-or-better results are shape consistency only, never "
            "superiority or confirmation (manifest acceptance_criteria)"
        ),
    }
    report["secondary_CC"]["delta_chi2_mod"] = (
        report["secondary_CC"]["M1"]["mod"]["chi2"]
        - report["secondary_CC"]["M2"]["mod"]["chi2"]
    )
    report["secondary_CC"]["delta_chi2_mod_ooo"] = (
        report["secondary_CC"]["M1"]["mod_ooo"]["chi2"]
        - report["secondary_CC"]["M2"]["mod_ooo"]["chi2"]
    )
    return report


def holdout_eval_main() -> int:
    """봉인 홀드아웃 평가를 실행해 JSON 보고서를 출력하는 진입점이다(옛 kinetic_dark_sector_quench_holdout_eval.main)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None, help="write the sealed report JSON")
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR), help="holdout data directory"
    )
    args = parser.parse_args()
    report = run_sealed_evaluation(Path(args.data_dir))
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
