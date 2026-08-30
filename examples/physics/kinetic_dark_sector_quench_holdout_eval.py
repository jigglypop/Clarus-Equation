"""SNKC quench background sealed holdout evaluation.

Preregistration: experiments/preregistration/cosmology_snkc_quench_bg_v1.json
(ledger SNKC-R2-THEATER-QUENCH-BG-PREREG-10).  This module implements the
evaluation protocol.  Provenance of the protocol: the manifest (model,
holdout identity, two profiled nuisances, kill threshold +9) was frozen
before holdout access; the P/X decomposition and the stricter either-P-or-X
kill rule were clarifications declared AFTER the manifest freeze and AFTER
holdout central values had been acquired, but BEFORE any model-vs-data
computation.  This post-access clarification is recorded as a documented
deviation in the audit (conservative direction, verdict-invariant):

Models (identical treatment):

* M1 -- registered branch: the frozen kinetic+quench background of
  ``kinetic_dark_sector_quench_gate`` at Omega_prod,0 = 0 (zero parameters
  fitted on holdout).
* M2 -- control: flat LambdaCDM at the exact frozen boundary tuple
  (Omega_b, Omega_r, Omega_m_extra, Omega_V) = (0.049, 9e-5, 0.26391, 0.687),
  i.e. the same background as ``same_frozen_boundary_lcdm_chi2``.

Primary holdout (eBOSS/BOSS consensus BAO):

* P (primary verdict): Gaussian blocks only -- DR12 LRG (4 obs, 4x4 covtot) +
  DR16 LRG (2 obs) + DR16 QSO (2 obs), block-diagonal 8x8.  Model observables
  DM/rd = s * dc(z), DH/rd = s / E(z) with one common profiled scale
  s = c/(H0 rd); s enters linearly, so the profile is exact GLS (analytic),
  cross-checked against a numerical profile.
* X (extended variant, reported alongside): chi2_P(s) plus -2 ln L from the
  official normalized ELG DV/rd table and the LYAUTO / LYxQSO DM-DH
  likelihood-ratio grids (linear / bilinear interpolation; outside the
  tabulated range the boundary value is used plus a declared large quadratic
  penalty), with s profiled numerically.
* Kill rule (pre-declared): if Delta chi2 = M1 - M2 exceeds +9 on either P or
  X, the background claim of this branch is REJECTED.

Secondary holdout (cosmic chronometers, Moresco 15-point homogeneous subset):

* H_model(z) = H0 * E(z), one profiled intercept H0 (exact GLS).  Covariance =
  diag(errHz^2) + outer products of the IMF + stlib + SPS components of
  ``data_MM20.dat`` interpolated to the data redshifts (percent / 100),
  following the official CCcovariance notebook recipe.  Main adoption = ``mod``
  column; sensitivity variant = ``mod_ooo`` reported alongside.  Reference
  report: pull vector and chi2/dof with dof = 15 - 1.

Data are loaded exclusively from
``benchmarks/cosmology/snkc_quench_bg_holdout_v1/`` (provenance and sha256 in
its README.md).  This module never touches the calibration (DESI) data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from examples.physics.ce_residual_forward_model import invert_matrix, quadratic_form
from examples.physics.kinetic_dark_sector_gate import (
    OMEGA_B0,
    OMEGA_K0,
    OMEGA_R0,
    OMEGA_V0,
    _dimensionless_distance,
)
from examples.physics.kinetic_dark_sector_quench_gate import (
    e_of_z,
    holdout_profiled_chi2,
    solve_quench_background,
)


DATA_DIR = Path("benchmarks/cosmology/snkc_quench_bg_holdout_v1")

Z_ELG = 0.845
Z_LYA = 2.334
LIKELIHOOD_FLOOR = 1.0e-300
OUT_OF_GRID_PENALTY = 1.0e6  # times squared normalized overshoot, per axis
KILL_RULE_DELTA_CHI2 = 9.0

GAUSSIAN_BLOCK_FILES = (
    ("sdss_DR12_LRG_BAO_DMDH.dat", "sdss_DR12_LRG_BAO_DMDH_covtot.txt"),
    ("sdss_DR16_LRG_BAO_DMDH.dat", "sdss_DR16_LRG_BAO_DMDH_covtot.txt"),
    ("sdss_DR16_QSO_BAO_DMDH.txt", "sdss_DR16_QSO_BAO_DMDH_covtot.txt"),
)

_OBS_KIND = {"DM_over_rs": "dm", "DH_over_rs": "dh", "DV_over_rs": "dv"}


# --------------------------------------------------------------------------
# background models (identical treatment; only the E(z) source differs)
# --------------------------------------------------------------------------


class KineticQuenchModel:
    """M1: frozen kinetic+quench background at Omega_prod,0 = 0."""

    name = "kinetic_quench_omega_prod0_zero"

    def __init__(self) -> None:
        self.solution = solve_quench_background(0.0)

    def e(self, z: float) -> float:
        return e_of_z(self.solution, z)

    def dc(self, z: float) -> float:
        return _dimensionless_distance(z, self.solution)


class FrozenBoundaryLCDMModel:
    """M2: flat LambdaCDM at the exact frozen boundary tuple."""

    name = "same_frozen_boundary_flat_lcdm"

    def e(self, z: float) -> float:
        zp1 = 1.0 + z
        return math.sqrt(
            OMEGA_R0 * zp1**4 + (OMEGA_B0 + OMEGA_K0) * zp1**3 + OMEGA_V0
        )

    def dc(self, z: float, intervals: int = 512) -> float:
        # Simpson with the same 512-interval resolution as the kinetic branch
        # (mirrors same_frozen_boundary_lcdm_chi2).
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
# loaders (restricted to DATA_DIR)
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
    """DR12 LRG + DR16 LRG + DR16 QSO as one block-diagonal Gaussian block."""

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
    """Normalized likelihood of one observable tabulated on a 1D grid."""

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
    """Likelihood(-ratio) of (DM/rd, DH/rd) tabulated on a rectangular grid."""

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
# generalized least squares with one profiled multiplicative intercept
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
    """Exact analytic profile of one multiplicative scale against a full cov.

    chi2(s) = (s*g - d)^T C^-1 (s*g - d) is quadratic in s; the minimizer is
    s = (g^T C^-1 d) / (g^T C^-1 g).  Pulls are per-point residuals divided by
    sqrt(diag C) at the profiled scale (reference report only).
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
# tabulated -2 ln L terms (declared interpolation and out-of-grid rule)
# --------------------------------------------------------------------------


def _interp_1d(x: float, grid: tuple[float, ...], values: tuple[float, ...]) -> tuple[float, float]:
    """Linear interpolation; returns (value_at_clamped_x, penalty)."""

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
# numerical 1D profile (declared: bracket scan + golden section)
# --------------------------------------------------------------------------


def minimize_scalar(
    objective,
    center: float,
    half_width_fraction: float = 0.2,
    scan_points: int = 401,
    tolerance: float = 1.0e-12,
    max_widenings: int = 4,
) -> tuple[float, float]:
    """Bracket scan around ``center`` then golden-section refinement.

    If the scan minimum sits on the scan boundary the bracket is doubled (at
    most ``max_widenings`` times, declared) before refinement.  Returns
    (argmin, minimum value).
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
# evaluation protocol
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
    """diag(errHz^2) + outer products of IMF + stlib + SPS percent components."""

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

    # cross-check: M1 primary P through the pre-registered evaluator interface
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


def main() -> int:
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
