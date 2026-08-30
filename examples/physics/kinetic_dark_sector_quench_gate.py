"""SNKC theater-opening quench-production background gate (calibration stage).

Implements the pre-declared background coupling rule of the SNKC quench
contract: the frozen boundary tuple (Omega_b, Omega_r, Omega_K, Omega_V) =
(0.049, 9.0e-5, 0.26391, 0.687) is unchanged; a single calibration parameter
Omega_prod,0 in [0, Omega_K] adds an exact-a^-3 production term
Omega_prod,0*(1+z)^3 to E^2(a) while the kinetic-sector shooting boundary is
reinterpreted as Omega_K' = 0.26391 - Omega_prod,0, so the flat present-day
budget still sums to one.  What is being tested is the shape difference
between exact a^-3 dilution and kinetic u-dynamics.

Calibration rule (pre-declared, seen/target-aware data only):

* objective = repo-internal DESI DR2 13-point full-covariance chi^2 with one
  profiled common scale c/(H0 rd), nothing else;
* search grid = {0} union 40 log-uniform points in [1e-6, 0.05], argmin taken;
* Pantheon-40 is reported at the argmin only, never used for calibration;
* holdout data (eBOSS DR16, cosmic chronometers) are NOT contacted here --
  only a pure evaluation function is provided that receives any such data
  strictly as arguments.

The base gate module ``kinetic_dark_sector_gate`` is imported, not modified.
The quench solver below mirrors its integrator operation-for-operation and
reduces bit-identically to it in the Omega_prod,0 -> 0 limit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

from examples.physics.ce_residual_forward_model import (
    BAODataset,
    CEForwardParams,
    assess_bao_fit,
    invert_matrix,
    named_bao_dataset,
    quadratic_form,
)
from examples.physics.kinetic_dark_sector_gate import (
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
    positive_u_from_density,
    profile_desi_bao,
)
from examples.physics.kinetic_dark_sector_sn_gate import compare_pantheon_binned


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
    """Base-gate densities plus the exact-a^-3 quench production term."""

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
    """Solve the frozen kinetic background with an exact-a^-3 quench component.

    ``omega_prod0`` replaces part of the kinetic inventory budget: the kinetic
    shooting boundary is Omega_K' = OMEGA_K0 - omega_prod0 while E^2 carries an
    additional omega_prod0 * (1+z)^3 term, keeping the flat present-day sum at
    one.  ``omega_prod0 = 0.0`` reduces bit-identically to the base gate's
    ``solve_background`` trajectory.
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
    """Pre-declared search grid: {0} union ``count`` log-uniform points."""

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
    """The base-gate LCDM control (frozen at 12.60835): CEForwardParams defaults."""

    selected = dataset or named_bao_dataset("desi-dr2-all")
    diagnostic = assess_bao_fit(
        selected.data, CEForwardParams(), covariance=selected.covariance
    ).scale_fit_diagnostic
    if diagnostic is None:
        raise ArithmeticError("LCDM BAO scale profile was not available")
    return diagnostic.chi2


def same_frozen_boundary_lcdm_chi2(dataset: BAODataset | None = None) -> float:
    """Profiled DESI chi^2 for flat LCDM at the frozen kinetic boundary tuple."""

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
    """Run the pre-declared DESI-only calibration scan and take the argmin.

    Failed grid points are recorded with status and an infinite chi^2 rather
    than silently dropped; the argmin is taken over successful points only.
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
    """Pure profiled-intercept, covariance-conditional chi^2 for holdout data.

    The observation vector, covariance, and per-point observable kinds
    (``dm``/``dh``/``dv`` in rd units, or ``hz`` for direct H(z)) are received
    strictly as arguments -- this function performs no file or network access
    and bundles no data, so holdout material is not contacted until the sealed
    evaluation stage supplies it.

    One multiplicative intercept is profiled analytically against the full
    covariance: c/(H0 rd) for the BAO family, H0 for the H(z) family.  Mixing
    the two families in one call is rejected because they do not share an
    intercept.  chi^2 is the generalized least-squares value at the profiled
    intercept, dof = n - 1.
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


if __name__ == "__main__":
    raise SystemExit(main())
