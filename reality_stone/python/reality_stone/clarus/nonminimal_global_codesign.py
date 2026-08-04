"""Global ansatz search for a healthy nonminimal-scalar throat."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class CodesignResolutionConvergenceAudit:
    """Raw N/2N/4N changes; this is not a continuous-domain error bound."""

    sample_count_n: int
    sample_count_2n: int
    sample_count_4n: int
    minimum_kinetic_delta_n_to_2n: float
    minimum_kinetic_delta_2n_to_4n: float
    minimum_kinetic_radius_delta_n_to_2n: float
    minimum_kinetic_radius_delta_2n_to_4n: float
    minimum_shape_gap_delta_n_to_2n: float
    minimum_shape_gap_delta_2n_to_4n: float
    minimum_log_planck_delta_n_to_2n: float
    minimum_log_planck_delta_2n_to_4n: float
    maximum_log_planck_delta_n_to_2n: float
    maximum_log_planck_delta_2n_to_4n: float
    sampled_codesign_pass_n: bool
    sampled_codesign_pass_2n: bool
    sampled_codesign_pass_4n: bool
    sampled_classification_consistent: bool


@dataclass(frozen=True)
class GlobalCodesignAudit:
    """Finite-grid co-design diagnostics with explicit sampling semantics.

    The legacy properties without a ``sampled_`` prefix are compatibility
    aliases only.  They must not be read as a proof over the full radial
    continuum.
    """

    parameters: tuple[float, ...]
    radial_cutoff: float
    sample_count: int
    local_kinetic_over_planck_factor: float
    sampled_minimum_kinetic_over_planck_factor: float
    sampled_minimum_kinetic_radius: float
    sampled_minimum_shape_gap: float
    sampled_minimum_log_planck_factor: float
    sampled_maximum_log_planck_factor: float
    positive_adm_mass: bool
    sampled_cutoff_flatness_pass: bool
    sampled_geometry_pass: bool
    sampled_healthy_kinetic: bool
    sampled_regular_planck_factor_control: bool
    sampled_codesign_pass: bool
    resolution_convergence: CodesignResolutionConvergenceAudit
    continuous_domain_certification: str
    potential_reconstructed: bool
    perturbative_stability_derived: bool

    @property
    def minimum_kinetic_over_planck_factor(self) -> float:
        """Compatibility alias for ``sampled_minimum_kinetic_over_planck_factor``."""

        return self.sampled_minimum_kinetic_over_planck_factor

    @property
    def minimum_kinetic_radius(self) -> float:
        """Compatibility alias for ``sampled_minimum_kinetic_radius``."""

        return self.sampled_minimum_kinetic_radius

    @property
    def minimum_shape_gap(self) -> float:
        """Compatibility alias for ``sampled_minimum_shape_gap``."""

        return self.sampled_minimum_shape_gap

    @property
    def minimum_log_planck_factor(self) -> float:
        """Compatibility alias for ``sampled_minimum_log_planck_factor``."""

        return self.sampled_minimum_log_planck_factor

    @property
    def maximum_log_planck_factor(self) -> float:
        """Compatibility alias for ``sampled_maximum_log_planck_factor``."""

        return self.sampled_maximum_log_planck_factor

    @property
    def asymptotically_flat(self) -> bool:
        """Compatibility alias for the finite-cutoff flatness tolerance."""

        return self.sampled_cutoff_flatness_pass

    @property
    def global_healthy_kinetic(self) -> bool:
        """Compatibility alias; the returned classification is sampled only."""

        return self.sampled_healthy_kinetic

    @property
    def regular_planck_factor_control(self) -> bool:
        """Compatibility alias; the returned classification is sampled only."""

        return self.sampled_regular_planck_factor_control

    @property
    def global_codesign_pass(self) -> bool:
        """Compatibility alias; no continuous global proof is implied."""

        return self.sampled_codesign_pass


@dataclass(frozen=True)
class _SampledCodesignProfile:
    local_kinetic_over_planck_factor: float
    minimum_kinetic_over_planck_factor: float
    minimum_kinetic_radius: float
    minimum_shape_gap: float
    minimum_log_planck_factor: float
    maximum_log_planck_factor: float
    cutoff_flatness_pass: bool
    geometry_pass: bool
    healthy_kinetic: bool
    regular_planck_factor_control: bool
    codesign_pass: bool


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


def _require_finite_arrays(**arrays: np.ndarray) -> None:
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"computed {name} profile contains non-finite values")


def _sample_codesign_profile(
    *,
    shape_limit: float,
    gamma: float,
    redshift_second: float,
    p3: float,
    p4: float,
    q3: float,
    q4: float,
    radial_cutoff: float,
    sample_count: int,
) -> _SampledCodesignProfile:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
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
        shape_gap = x - shape
        metric_factor = 1.0 - shape / x
        metric_factor_first = -shape_first / x + shape / x**2
        density = shape_first / x**2
        radial_pressure = -shape / x**3 + 2.0 * metric_factor * redshift_first / x
        difference = shape_first * x - shape
        tangential_pressure = (
            metric_factor * (redshift_second_profile + redshift_first**2 + redshift_first / x)
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

    _require_finite_arrays(
        radial_coordinate=x,
        shape=shape,
        shape_gap=shape_gap,
        shape_first_derivative=shape_first,
        redshift=redshift,
        redshift_first_derivative=redshift_first,
        redshift_second_derivative=redshift_second_profile,
        metric_factor=metric_factor,
        metric_factor_first_derivative=metric_factor_first,
        density=density,
        radial_pressure=radial_pressure,
        tangential_pressure=tangential_pressure,
        radial_pressure_denominator=denominator,
        logarithmic_slope=logarithmic_slope,
        logarithmic_slope_first_derivative=slope_first,
        kinetic_over_planck_factor=kinetic_over_planck,
        log_planck_factor=log_planck,
    )
    if not math.isfinite(local_kinetic):
        raise ValueError("computed local kinetic coefficient is non-finite")

    # Reductions are deliberately after the complete finite-array gate above.
    minimum_index = int(np.argmin(kinetic_over_planck))
    minimum_gap = float(np.min(shape_gap))
    minimum_kinetic = min(local_kinetic, float(kinetic_over_planck[minimum_index]))
    minimum_log_planck = float(np.min(log_planck))
    maximum_log_planck = float(np.max(log_planck))
    regular_planck = bool(
        minimum_log_planck > math.log(0.1) and maximum_log_planck < math.log(10.0)
    )
    geometry_pass = minimum_gap > 0.0 and shape_limit > 0.0
    kinetic_pass = minimum_kinetic >= -1.0e-6
    cutoff_flatness_pass = bool(abs(shape[-1] / x[-1]) < 0.05 and abs(redshift[-1]) < 1.0e-10)
    codesign_pass = bool(geometry_pass and kinetic_pass and regular_planck and cutoff_flatness_pass)
    return _SampledCodesignProfile(
        local_kinetic_over_planck_factor=local_kinetic,
        minimum_kinetic_over_planck_factor=minimum_kinetic,
        minimum_kinetic_radius=float(x[minimum_index]),
        minimum_shape_gap=minimum_gap,
        minimum_log_planck_factor=minimum_log_planck,
        maximum_log_planck_factor=maximum_log_planck,
        cutoff_flatness_pass=cutoff_flatness_pass,
        geometry_pass=geometry_pass,
        healthy_kinetic=kinetic_pass,
        regular_planck_factor_control=regular_planck,
        codesign_pass=codesign_pass,
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
    """Audit a polynomial-times-exponential family on finite radial grids.

    The requested grid is accompanied by 2N and 4N controls.  Their raw
    changes expose resolution sensitivity, but do not constitute a rigorous
    error bound between sample points.
    """

    values = (
        adm_shape_limit,
        shape_second_derivative,
        redshift_second_derivative,
        shape_cubic,
        shape_quartic,
        redshift_cubic,
        redshift_quartic,
    )
    try:
        normalized_values = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("all co-design parameters must be finite real values") from error
    if not all(math.isfinite(value) for value in normalized_values):
        raise ValueError("all co-design parameters must be finite")
    if isinstance(radial_cutoff, bool) or not isinstance(radial_cutoff, Real):
        raise ValueError("radial_cutoff must be a finite real value greater than 2")
    cutoff = float(radial_cutoff)
    if not math.isfinite(cutoff) or cutoff <= 2.0:
        raise ValueError("radial_cutoff must be a finite real value greater than 2")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count < 256
    ):
        raise ValueError("sample_count must be an integer (not bool) of at least 256")
    count = int(sample_count)

    shape_limit, gamma, redshift_second, p3, p4, q3, q4 = normalized_values
    sample_arguments = dict(
        shape_limit=shape_limit,
        gamma=gamma,
        redshift_second=redshift_second,
        p3=p3,
        p4=p4,
        q3=q3,
        q4=q4,
        radial_cutoff=cutoff,
    )
    sampled_n = _sample_codesign_profile(**sample_arguments, sample_count=count)
    sampled_2n = _sample_codesign_profile(**sample_arguments, sample_count=2 * count)
    sampled_4n = _sample_codesign_profile(**sample_arguments, sample_count=4 * count)

    classification_n = (
        sampled_n.geometry_pass,
        sampled_n.healthy_kinetic,
        sampled_n.regular_planck_factor_control,
        sampled_n.cutoff_flatness_pass,
        sampled_n.codesign_pass,
    )
    classification_2n = (
        sampled_2n.geometry_pass,
        sampled_2n.healthy_kinetic,
        sampled_2n.regular_planck_factor_control,
        sampled_2n.cutoff_flatness_pass,
        sampled_2n.codesign_pass,
    )
    classification_4n = (
        sampled_4n.geometry_pass,
        sampled_4n.healthy_kinetic,
        sampled_4n.regular_planck_factor_control,
        sampled_4n.cutoff_flatness_pass,
        sampled_4n.codesign_pass,
    )
    convergence = CodesignResolutionConvergenceAudit(
        sample_count_n=count,
        sample_count_2n=2 * count,
        sample_count_4n=4 * count,
        minimum_kinetic_delta_n_to_2n=(
            sampled_2n.minimum_kinetic_over_planck_factor
            - sampled_n.minimum_kinetic_over_planck_factor
        ),
        minimum_kinetic_delta_2n_to_4n=(
            sampled_4n.minimum_kinetic_over_planck_factor
            - sampled_2n.minimum_kinetic_over_planck_factor
        ),
        minimum_kinetic_radius_delta_n_to_2n=(
            sampled_2n.minimum_kinetic_radius - sampled_n.minimum_kinetic_radius
        ),
        minimum_kinetic_radius_delta_2n_to_4n=(
            sampled_4n.minimum_kinetic_radius - sampled_2n.minimum_kinetic_radius
        ),
        minimum_shape_gap_delta_n_to_2n=(
            sampled_2n.minimum_shape_gap - sampled_n.minimum_shape_gap
        ),
        minimum_shape_gap_delta_2n_to_4n=(
            sampled_4n.minimum_shape_gap - sampled_2n.minimum_shape_gap
        ),
        minimum_log_planck_delta_n_to_2n=(
            sampled_2n.minimum_log_planck_factor - sampled_n.minimum_log_planck_factor
        ),
        minimum_log_planck_delta_2n_to_4n=(
            sampled_4n.minimum_log_planck_factor - sampled_2n.minimum_log_planck_factor
        ),
        maximum_log_planck_delta_n_to_2n=(
            sampled_2n.maximum_log_planck_factor - sampled_n.maximum_log_planck_factor
        ),
        maximum_log_planck_delta_2n_to_4n=(
            sampled_4n.maximum_log_planck_factor - sampled_2n.maximum_log_planck_factor
        ),
        sampled_codesign_pass_n=sampled_n.codesign_pass,
        sampled_codesign_pass_2n=sampled_2n.codesign_pass,
        sampled_codesign_pass_4n=sampled_4n.codesign_pass,
        sampled_classification_consistent=(
            classification_n == classification_2n == classification_4n
        ),
    )
    return GlobalCodesignAudit(
        parameters=normalized_values,
        radial_cutoff=cutoff,
        sample_count=count,
        local_kinetic_over_planck_factor=sampled_n.local_kinetic_over_planck_factor,
        sampled_minimum_kinetic_over_planck_factor=(sampled_n.minimum_kinetic_over_planck_factor),
        sampled_minimum_kinetic_radius=sampled_n.minimum_kinetic_radius,
        sampled_minimum_shape_gap=sampled_n.minimum_shape_gap,
        sampled_minimum_log_planck_factor=sampled_n.minimum_log_planck_factor,
        sampled_maximum_log_planck_factor=sampled_n.maximum_log_planck_factor,
        positive_adm_mass=shape_limit > 0.0,
        sampled_cutoff_flatness_pass=sampled_n.cutoff_flatness_pass,
        sampled_geometry_pass=sampled_n.geometry_pass,
        sampled_healthy_kinetic=sampled_n.healthy_kinetic,
        sampled_regular_planck_factor_control=sampled_n.regular_planck_factor_control,
        sampled_codesign_pass=sampled_n.codesign_pass,
        resolution_convergence=convergence,
        continuous_domain_certification="not established by finite-grid sampling",
        potential_reconstructed=False,
        perturbative_stability_derived=False,
    )
