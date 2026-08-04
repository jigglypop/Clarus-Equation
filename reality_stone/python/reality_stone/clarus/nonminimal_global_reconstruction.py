"""Throat reconstruction gate for healthy nonminimally coupled CE scalars."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NonminimalThroatReconstructionAudit:
    shape_derivative: float
    target_rho_plus_radial_pressure: float
    logarithmic_planck_factor_radial_slope: float
    proper_planck_factor_second_derivative: float
    required_positive_metric_scalar_kinetic: float
    positive_effective_planck_mass_assumed: bool
    positive_field_space_metric_assumed: bool
    healthy_single_scalar_possible: bool
    healthy_multiscalar_modes_possible: bool
    potential_reconstruction_reached: bool
    target_refuted_for_healthy_nonminimal_scalars: bool


@dataclass(frozen=True)
class NonminimalThroatCodesignAudit:
    shape_second_derivative: float
    redshift_second_derivative: float
    logarithmic_planck_factor_radial_slope: float
    required_scalar_kinetic_over_planck_factor: float
    positive_kinetic_gate: bool
    exact_casimir_throat_values_retained: bool
    local_codesign_survives: bool
    global_solution_derived: bool
    perturbative_stability_derived: bool


def nonminimal_throat_reconstruction_audit() -> NonminimalThroatReconstructionAudit:
    """Reconstruct the scalar kinetic sign at the global target's throat.

    Consider any scalar-tensor action whose metric equation has the form

    ``F G_ab = kappa T_ab + nabla_a nabla_b F - g_ab box F``

    with a positive-definite scalar field-space kinetic metric.  In proper
    radial distance ``s=l/r0``, the target's ``tt+angular`` equation fixes
    ``d ln(F)/dx = 1/8`` at the throat.  Since
    ``d(1-b/r)/dx = 4/3``, this gives ``F_ss/F=1/12`` there.

    The ``tt+radial`` equation then requires

    ``kappa G_IJ phi_s^I phi_s^J / F = -4/3 - 1/12 = -17/12``.

    Its sign is incompatible with both ``F>0`` and a healthy positive field-
    space metric.  Adding more canonical radial modes only adds non-negative
    terms and cannot repair the sign.
    """

    shape_derivative = -1.0 / 3.0
    rho_plus_radial = -4.0 / 3.0
    log_planck_slope = 1.0 / 8.0
    proper_second = 1.0 / 12.0
    required_kinetic = rho_plus_radial - proper_second
    impossible = required_kinetic < 0.0
    return NonminimalThroatReconstructionAudit(
        shape_derivative=shape_derivative,
        target_rho_plus_radial_pressure=rho_plus_radial,
        logarithmic_planck_factor_radial_slope=log_planck_slope,
        proper_planck_factor_second_derivative=proper_second,
        required_positive_metric_scalar_kinetic=required_kinetic,
        positive_effective_planck_mass_assumed=True,
        positive_field_space_metric_assumed=True,
        healthy_single_scalar_possible=not impossible,
        healthy_multiscalar_modes_possible=not impossible,
        potential_reconstruction_reached=False,
        target_refuted_for_healthy_nonminimal_scalars=impossible,
    )


def nonminimal_throat_codesign_audit(
    *,
    shape_second_derivative: float,
    redshift_second_derivative: float,
) -> NonminimalThroatCodesignAudit:
    """Test second-order throat data while retaining exact Casimir values.

    Keep ``b'=-1/3`` and ``Phi'=-1/2`` at the throat, but write
    ``b/r0=1-z/3+gamma*z^2/2+...`` and
    ``Phi=Phi0-z/2+v*z^2/2+...``.  Reconstruction gives

    ``(ln F)_x=(3 gamma+8 v-4)/8`` and
    ``kappa G_IJ phi_s^I phi_s^J/F=-(3 gamma+8 v+12)/12``.

    Hence healthy local kinetic data exist exactly when
    ``3 gamma+8 v+12 <= 0``.  This is a local co-design gate only.
    """

    gamma = float(shape_second_derivative)
    redshift_second = float(redshift_second_derivative)
    if not math.isfinite(gamma) or not math.isfinite(redshift_second):
        raise ValueError("second-order throat coefficients must be finite")
    log_slope = (3.0 * gamma + 8.0 * redshift_second - 4.0) / 8.0
    kinetic = -(3.0 * gamma + 8.0 * redshift_second + 12.0) / 12.0
    positive_kinetic = kinetic >= 0.0
    return NonminimalThroatCodesignAudit(
        shape_second_derivative=gamma,
        redshift_second_derivative=redshift_second,
        logarithmic_planck_factor_radial_slope=log_slope,
        required_scalar_kinetic_over_planck_factor=kinetic,
        positive_kinetic_gate=positive_kinetic,
        exact_casimir_throat_values_retained=True,
        local_codesign_survives=positive_kinetic,
        global_solution_derived=False,
        perturbative_stability_derived=False,
    )
