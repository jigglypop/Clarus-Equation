"""Finite-dimensional single-branch normal form for a curved vertex.

Curved Chern--Simons spin-foam proposals recover a constant-curvature Regge
phase at geometric critical flat connections, but the published construction
does not supply a global proper-sector projector and a fully controlled
functional integration cycle.  This module isolates the exact local Gaussian
template associated with one branch after a finite-dimensional regulator,
gauge fixing, and thimble/cutoff have been supplied.  It does not transfer the
template to a regulated curved block: that requires an actual integrand and
measure, an isolated critical point, a thimble chart and Jacobian, and
derivative/remainder estimates.

The exact result here is a Gaussian steepest-descent normal form.  Flags keep
it separate from a defined global ``SL(2,C)`` Chern--Simons path integral,
regulator removal, equivalence to Engle's proper projector, and multi-block
gluing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import cmath
import math

import numpy as np


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DimensionlessCurvedReggePhaseAudit:
    triangle_areas_over_reference_area: tuple[float, ...]
    oriented_boost_angles: tuple[float, ...]
    cosmological_constant_times_reference_length_squared: float
    four_volume_over_reference_length_fourth: float
    inverse_dimensionless_gravitational_coupling: float
    area_angle_sum: float
    cosmological_volume_term: float
    dimensionless_curved_regge_phase: float
    all_input_normalizations_declared_dimensionless: bool
    status: str


def dimensionless_curved_regge_phase(
    triangle_areas_over_reference_area: Sequence[float],
    oriented_boost_angles: Sequence[float],
    *,
    cosmological_constant_times_reference_length_squared: float,
    four_volume_over_reference_length_fourth: float,
    inverse_dimensionless_gravitational_coupling: float,
) -> DimensionlessCurvedReggePhaseAudit:
    """Return ``gbar^-1 [sum_f a_f Theta_f - lambdabar v_4]``.

    In units ``c = hbar = 1`` the convention used here is
    ``gbar^-1 = L_ref^2/(8 pi G)``.  The caller declares
    ``a_f=A_f/L_ref^2``, ``lambdabar=Lambda L_ref^2`` and
    ``v_4=V_4/L_ref^4``; Python can verify the algebra but not physical units.
    """

    areas = tuple(
        _finite("triangle area over reference area", value)
        for value in triangle_areas_over_reference_area
    )
    angles = tuple(
        _finite("oriented boost angle", value) for value in oriented_boost_angles
    )
    if not areas or len(areas) != len(angles):
        raise ValueError("areas and angles must be nonempty and have equal length")
    if any(area <= 0.0 for area in areas):
        raise ValueError("normalized triangle areas must be positive")
    lambda_bar = _finite(
        "cosmological_constant_times_reference_length_squared",
        cosmological_constant_times_reference_length_squared,
    )
    volume_bar = _finite(
        "four_volume_over_reference_length_fourth",
        four_volume_over_reference_length_fourth,
    )
    inverse_coupling = _finite(
        "inverse_dimensionless_gravitational_coupling",
        inverse_dimensionless_gravitational_coupling,
    )
    if volume_bar <= 0.0 or inverse_coupling <= 0.0:
        raise ValueError("normalized volume and inverse coupling must be positive")
    area_angle = math.fsum(area * angle for area, angle in zip(areas, angles))
    cosmological_volume = lambda_bar * volume_bar
    phase = inverse_coupling * (area_angle - cosmological_volume)
    return DimensionlessCurvedReggePhaseAudit(
        triangle_areas_over_reference_area=areas,
        oriented_boost_angles=angles,
        cosmological_constant_times_reference_length_squared=lambda_bar,
        four_volume_over_reference_length_fourth=volume_bar,
        inverse_dimensionless_gravitational_coupling=inverse_coupling,
        area_angle_sum=area_angle,
        cosmological_volume_term=cosmological_volume,
        dimensionless_curved_regge_phase=phase,
        all_input_normalizations_declared_dimensionless=True,
        status="DIMENSIONLESS_CURVED_REGGE_PHASE_CLOSED",
    )


@dataclass(frozen=True)
class ProjectedCurvedThimbleNormalFormAudit:
    target_branch_label: str
    variable_count: int
    large_dimensionless_parameter: float
    transverse_hessian_eigenvalues: tuple[float, ...]
    transverse_hessian_determinant: float
    dimensionless_target_phase: float
    exact_quadratic_normal_form_amplitude: complex
    exact_quadratic_prefactor_magnitude: float
    large_parameter_power: float
    finite_dimensional_regulator_fixed: bool
    gauge_zero_modes_removed: bool
    admissible_middle_dimensional_cycle_fixed: bool
    cutoff_equal_to_one_near_target: bool
    other_critical_points_excluded_from_support: bool
    uniform_nonstationary_remainder_bound_supplied: bool
    numerically_well_conditioned_positive_definite: bool
    quadratic_gaussian_local_template_exact: bool
    conditional_local_single_branch_stationary_phase_template: bool
    regulated_curved_block_single_branch_asymptotic_proved: bool
    global_chern_simons_functional_integral_defined: bool
    regulator_removal_proved: bool
    equivalent_to_engle_proper_projector: bool
    multi_block_gluing_proved: bool
    status: str
    claim_ceiling: str = (
        "EXACT_LOCAL_GAUSSIAN_TEMPLATE_NOT_A_REGULATED_CURVED_BLOCK"
    )


def projected_curved_thimble_normal_form(
    transverse_hessian: Sequence[Sequence[float]],
    *,
    dimensionless_target_phase: float,
    large_dimensionless_parameter: float,
    target_branch_label: str = "positive Einstein-Hilbert orientation",
    finite_dimensional_regulator_fixed: bool,
    gauge_zero_modes_removed: bool,
    admissible_middle_dimensional_cycle_fixed: bool,
    cutoff_equal_to_one_near_target: bool,
    other_critical_points_excluded_from_support: bool,
    uniform_nonstationary_remainder_bound_supplied: bool,
    tolerance: float = 1.0e-12,
) -> ProjectedCurvedThimbleNormalFormAudit:
    """Evaluate an exact ``R^n`` Gaussian template for a target branch.

    The normal form is

    ``exp(i rho S_+) integral_R^n exp[-rho x^T Q x/2] dx``.

    Exactness is conditional on the supplied local coordinates having the
    displayed flat measure.  A numerically well-conditioned positive-definite
    ``Q`` is the transverse Hessian of this exact quadratic model.  The six
    booleans describe necessary
    parts of a candidate local construction.  Even when all hold, this function
    does *not* transfer the model to a regulated Chern--Simons block.  Such a
    transfer additionally needs a smooth finite-dimensional integrand and
    measure, an isolated nondegenerate target critical point, a local thimble
    parametrization (including orientation and Jacobian), and uniform
    higher-derivative/remainder bounds.
    """

    if not isinstance(target_branch_label, str) or not target_branch_label.strip():
        raise ValueError("target_branch_label must be nonempty")
    phase = _finite("dimensionless_target_phase", dimensionless_target_phase)
    scale = _finite("large_dimensionless_parameter", large_dimensionless_parameter)
    tolerance = _finite("tolerance", tolerance)
    if scale <= 0.0 or tolerance <= 0.0:
        raise ValueError("large parameter and tolerance must be positive")
    matrix = np.asarray(transverse_hessian, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("transverse_hessian must be a finite nonempty square matrix")
    if np.linalg.norm(matrix - matrix.T) > tolerance * max(1.0, np.linalg.norm(matrix)):
        raise ValueError("transverse_hessian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = bool(
        np.all(eigenvalues > tolerance * max(1.0, np.linalg.norm(matrix)))
    )
    determinant = float(np.linalg.det(matrix))
    variable_count = matrix.shape[0]
    prefactor = (
        (2.0 * math.pi / scale) ** (variable_count / 2.0)
        / math.sqrt(determinant)
        if positive
        else math.inf
    )
    amplitude = (
        prefactor * cmath.exp(1j * scale * phase)
        if positive
        else complex(math.nan, math.nan)
    )
    supplied_hypotheses = (
        finite_dimensional_regulator_fixed,
        gauge_zero_modes_removed,
        admissible_middle_dimensional_cycle_fixed,
        cutoff_equal_to_one_near_target,
        other_critical_points_excluded_from_support,
        uniform_nonstationary_remainder_bound_supplied,
    )
    if any(not isinstance(value, bool) for value in supplied_hypotheses):
        raise ValueError("all thimble hypotheses must be boolean")
    hypotheses = supplied_hypotheses
    local_template = positive and all(hypotheses)
    return ProjectedCurvedThimbleNormalFormAudit(
        target_branch_label=target_branch_label,
        variable_count=variable_count,
        large_dimensionless_parameter=scale,
        transverse_hessian_eigenvalues=tuple(float(value) for value in eigenvalues),
        transverse_hessian_determinant=determinant,
        dimensionless_target_phase=phase,
        exact_quadratic_normal_form_amplitude=amplitude,
        exact_quadratic_prefactor_magnitude=prefactor,
        large_parameter_power=-0.5 * variable_count,
        finite_dimensional_regulator_fixed=hypotheses[0],
        gauge_zero_modes_removed=hypotheses[1],
        admissible_middle_dimensional_cycle_fixed=hypotheses[2],
        cutoff_equal_to_one_near_target=hypotheses[3],
        other_critical_points_excluded_from_support=hypotheses[4],
        uniform_nonstationary_remainder_bound_supplied=hypotheses[5],
        numerically_well_conditioned_positive_definite=positive,
        quadratic_gaussian_local_template_exact=positive,
        conditional_local_single_branch_stationary_phase_template=local_template,
        regulated_curved_block_single_branch_asymptotic_proved=False,
        global_chern_simons_functional_integral_defined=False,
        regulator_removal_proved=False,
        equivalent_to_engle_proper_projector=False,
        multi_block_gluing_proved=False,
        status=(
            "FINITE_SINGLE_BRANCH_GAUSSIAN_TEMPLATE_CLOSED"
            if local_template
            else "LOCAL_GAUSSIAN_TEMPLATE_HYPOTHESES_INCOMPLETE"
        ),
    )
