"""Global ansatz search for a healthy nonminimal-scalar throat."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GlobalCodesignAudit:
    parameters: tuple[float, ...]
    local_kinetic_over_planck_factor: float
    minimum_kinetic_over_planck_factor: float
    minimum_kinetic_radius: float
    minimum_shape_gap: float
    minimum_log_planck_factor: float
    maximum_log_planck_factor: float
    positive_adm_mass: bool
    asymptotically_flat: bool
    global_healthy_kinetic: bool
    regular_planck_factor_control: bool
    global_codesign_pass: bool
    potential_reconstructed: bool
    perturbative_stability_derived: bool


def _exp_polynomial(
    z: np.ndarray,
    coefficients: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficient_array = np.asarray(coefficients, dtype=float)
    first_coefficients = np.arange(1, len(coefficients)) * coefficient_array[1:]
    second_coefficients = (
        np.arange(2, len(coefficients))
        * np.arange(1, len(coefficients) - 1)
        * coefficient_array[2:]
    )
    polynomial = np.polynomial.polynomial.polyval(z, coefficient_array)
    first = np.polynomial.polynomial.polyval(z, first_coefficients)
    second = np.polynomial.polynomial.polyval(z, second_coefficients)
    exponential = np.exp(-z)
    return (
        polynomial * exponential,
        (first - polynomial) * exponential,
        (second - 2.0 * first + polynomial) * exponential,
    )


def global_nonminimal_codesign_audit(
    *,
    adm_shape_limit: float,
    shape_second_derivative: float,
    redshift_second_derivative: float,
    shape_cubic: float,
    shape_quartic: float,
    redshift_cubic: float,
    redshift_quartic: float,
    radial_cutoff: float = 40.0,
    sample_count: int = 2400,
) -> GlobalCodesignAudit:
    """Audit a polynomial-times-exponential global co-design family."""

    values = (
        adm_shape_limit,
        shape_second_derivative,
        redshift_second_derivative,
        shape_cubic,
        shape_quartic,
        redshift_cubic,
        redshift_quartic,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("all co-design parameters must be finite")
    if radial_cutoff <= 2.0 or sample_count < 256:
        raise ValueError("radial_cutoff and sample_count are too small")

    shape_limit, gamma, redshift_second, p3, p4, q3, q4 = map(float, values)
    z = np.geomspace(1.0e-4, radial_cutoff - 1.0, sample_count)
    x = 1.0 + z

    p0 = 1.0 - shape_limit
    p1 = p0 - 1.0 / 3.0
    p2 = (gamma + 2.0 * p1 - p0) / 2.0
    shape_tail, shape_tail_first, _ = _exp_polynomial(z, (p0, p1, p2, p3, p4))
    shape = shape_limit + shape_tail
    shape_first = shape_tail_first

    redshift, redshift_first, redshift_second_profile = _exp_polynomial(
        z,
        (1.0, 0.5, redshift_second / 2.0, q3, q4),
    )
    metric_factor = 1.0 - shape / x
    metric_factor_first = -shape_first / x + shape / x**2
    density = shape_first / x**2
    radial_pressure = (
        -shape / x**3 + 2.0 * metric_factor * redshift_first / x
    )
    difference = shape_first * x - shape
    tangential_pressure = (
        metric_factor
        * (
            redshift_second_profile
            + redshift_first**2
            + redshift_first / x
        )
        - difference * redshift_first / (2.0 * x**2)
        - difference / (2.0 * x**3)
    )

    denominator = metric_factor * (1.0 / x - redshift_first)
    logarithmic_slope = (density + tangential_pressure) / denominator
    slope_first = np.gradient(logarithmic_slope, x, edge_order=2)
    kinetic_over_planck = (
        density
        + radial_pressure
        - 0.5 * metric_factor_first * logarithmic_slope
        - metric_factor * (logarithmic_slope**2 + slope_first)
        + metric_factor * redshift_first * logarithmic_slope
    )

    dx = np.diff(x)
    local_log_slope = (3.0 * gamma + 8.0 * redshift_second - 4.0) / 8.0
    log_planck = np.r_[
        local_log_slope * z[0],
        local_log_slope * z[0]
        + np.cumsum((logarithmic_slope[1:] + logarithmic_slope[:-1]) * dx / 2.0),
    ]
    local_kinetic = -(3.0 * gamma + 8.0 * redshift_second + 12.0) / 12.0
    minimum_index = int(np.nanargmin(kinetic_over_planck))
    minimum_gap = float(np.min(x - shape))
    minimum_kinetic = min(local_kinetic, float(kinetic_over_planck[minimum_index]))
    finite_arrays = bool(
        np.all(np.isfinite(logarithmic_slope))
        and np.all(np.isfinite(kinetic_over_planck))
    )
    regular_planck = bool(
        finite_arrays and np.min(log_planck) > math.log(0.1) and np.max(log_planck) < math.log(10.0)
    )
    geometry_pass = minimum_gap > 0.0 and shape_limit > 0.0
    kinetic_pass = minimum_kinetic >= -1.0e-6
    asymptotically_flat = abs(shape[-1] / x[-1]) < 0.05 and abs(redshift[-1]) < 1.0e-10
    return GlobalCodesignAudit(
        parameters=tuple(map(float, values)),
        local_kinetic_over_planck_factor=local_kinetic,
        minimum_kinetic_over_planck_factor=minimum_kinetic,
        minimum_kinetic_radius=float(x[minimum_index]),
        minimum_shape_gap=minimum_gap,
        minimum_log_planck_factor=float(np.min(log_planck)),
        maximum_log_planck_factor=float(np.max(log_planck)),
        positive_adm_mass=shape_limit > 0.0,
        asymptotically_flat=asymptotically_flat,
        global_healthy_kinetic=kinetic_pass,
        regular_planck_factor_control=regular_planck,
        global_codesign_pass=(
            geometry_pass and kinetic_pass and regular_planck and asymptotically_flat
        ),
        potential_reconstructed=False,
        perturbative_stability_derived=False,
    )
