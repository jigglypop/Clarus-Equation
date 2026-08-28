"""Low-data background gate for the self-nonidentity kinetic dark sector.

This module implements the declared homogeneous P(T, X) branch itself.  It is
not the w0-wa surrogate used by ``ce_residual_forward_model`` and it does not
claim a microscopic derivation of the positive initial current.  Present-day
density fractions are boundary calibrated, then the resulting trajectory is
tested for branch health and against the compact DESI DR2 BAO covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.ce_residual_forward_model import (
    BAODataset,
    CEForwardParams,
    assess_bao_fit,
    C_KM_S,
    invert_matrix,
    named_bao_dataset,
    quadratic_form,
    sound_horizon_at_redshift_mpc,
)


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
    """Cancellation-safe positive root of 2u+3u^2/(2 kappa)=target/A."""

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
    """Exact local counterexample for a source-free zero-current start."""

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
    """Same-boundary flat LCDM+radiation negative control."""

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
    """Profile the common c/(H0 rd) scale against compact DESI DR2 BAO.

    The profiled scale is an external calibration diagnostic, not a CE
    prediction.  Present density fractions and gamma are already selected.
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
    """Low-download acoustic-angle check with declared standard early physics.

    The recombination redshift and photon-baryon sound speed are external
    standard-physics inputs.  The raw pull uses only Planck's observational
    error and therefore is a discrepancy flag, not a precision likelihood.
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
    """Use Planck theta* to calibrate H0 along the BAO H0*rd degeneracy."""

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
    """Seen-data gamma scan with an explicit extra-parameter penalty.

    Both models profile the same BAO scale.  Selecting gamma from these data
    adds one further fitted parameter to the kinetic model.  This is an
    exploratory model comparison, never a holdout or CE prediction.
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
    # kinetic: common scale + selected gamma; LCDM: common scale only
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


if __name__ == "__main__":
    raise SystemExit(main())
