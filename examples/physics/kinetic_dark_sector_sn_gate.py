"""Low-download Pantheon binned supernova shape gate.

The official 40-bin Pantheon vector and systematic covariance are hash-pinned.
The absolute magnitude/H0 intercept is analytically profiled, so this tests the
relative luminosity-distance shape only.  Pantheon+ is deliberately not
claimed: its full 1701-object covariance is outside this compact gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from examples.physics.ce_residual_forward_model import invert_matrix, quadratic_form
from examples.physics.kinetic_dark_sector_gate import (
    BackgroundSolution,
    OMEGA_B0,
    OMEGA_K0,
    OMEGA_R0,
    OMEGA_V0,
    _dimensionless_distance,
    solve_background,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "benchmarks/cosmology/pantheon_binned_v1"
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
    """Fit the intercept on training bins and predict correlated holdout bins.

    The Gaussian conditional includes both train--holdout covariance and the
    posterior uncertainty of the profiled intercept under a flat prior.
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
    """Run a deterministic every-fourth-bin retrospective holdout."""

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


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
