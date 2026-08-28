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


def compare_pantheon_binned(
    solution: BackgroundSolution | None = None,
    dataset: SupernovaDataset | None = None,
) -> SupernovaModelComparison:
    selected_solution = solution or solve_background()
    selected_data = dataset or load_pantheon_binned()

    def magnitude_shape(z: float, distance: float) -> float:
        if z <= 0.0 or distance <= 0.0:
            raise ValueError("supernova distance shape must be positive")
        return 5.0 * math.log10((1.0 + z) * distance)

    kinetic_shapes = tuple(
        magnitude_shape(z, _dimensionless_distance(z, selected_solution))
        for z in selected_data.redshift
    )
    lcdm_shapes = tuple(
        magnitude_shape(z, _lcdm_distance(z)) for z in selected_data.redshift
    )
    return SupernovaModelComparison(
        kinetic=_profile_intercept(kinetic_shapes, selected_data, "Pantheon-40 kinetic"),
        lcdm=_profile_intercept(lcdm_shapes, selected_data, "Pantheon-40 LCDM"),
    )


def main() -> int:
    result = compare_pantheon_binned()
    print("kinetic_chi2", result.kinetic.chi2)
    print("lcdm_chi2", result.lcdm.chi2)
    print("dof", result.kinetic.dof)
    print("delta_chi2_kinetic_minus_lcdm", result.delta_chi2_kinetic_minus_lcdm)
    print("role", result.kinetic.role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
