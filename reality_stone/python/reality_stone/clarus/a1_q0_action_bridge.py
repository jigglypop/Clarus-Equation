"""Conditional A1-to-Q0 algebra gates for the CE+SM action bridge.

This module checks two local algebraic facts that are easy to overstate:

* an ordinary action Hessian is not a tensor under a nonlinear field
  reparameterization away from a stationary point;
* the unbroken-``Z2`` Higgs portal has no vacuum ``h``-``phi`` quadratic
  mixing even though its cubic and quartic interactions are nonzero.

It also exposes a separate tree-level invisible-width audit for supplied
portal benchmarks.  That audit can reject a benchmark without deriving the
portal, its coupling, or its scalar pole from CE.

Passing these gates does **not** construct a field-space covariant effective
action, derive a renormalized stress tensor, or compute a physical spectral
density.  Those claims remain explicitly locked in :class:`A1Q0ActionReport`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import factorial, pi, sqrt
from typing import Any

import numpy as np


A1_Q0_SCOPE = "conditional_a1_q0_local_algebra_only"
CONDITIONAL_PASS = "A1_Q0_CONDITIONAL_PASS"
CONDITIONAL_FAIL = "A1_Q0_CONDITIONAL_FAIL"

_PASS_CONCLUSION = (
    "The supplied local coordinate jet and Z2 portal parameters pass the "
    "conditional algebra gates. This is not a covariant CE+SM action, stress-"
    "tensor derivation, or spectral-density derivation."
)
_FAIL_CONCLUSION = (
    "At least one supplied local algebra gate failed. No conclusion about a "
    "covariant CE+SM action, stress tensor, or spectral density follows."
)


@dataclass(frozen=True)
class HessianCoordinateAudit:
    """One-dimensional scalar-action Hessian transformation diagnostics."""

    action_gradient_x: float
    action_hessian_x: float
    dx_dy: float
    d2x_dy2: float
    action_gradient_y: float
    tensor_pullback_hessian_y: float
    ordinary_hessian_y: float
    non_tensor_extra_term: float
    induced_connection_y: float
    connection_correction: float
    covariant_hessian_y: float
    chain_rule_residual: float
    covariance_residual: float
    tolerance: float
    stationary: bool
    locally_nonlinear: bool
    ordinary_tensorial: bool
    covariant_tensorial: bool
    structural_pass: bool


@dataclass(frozen=True)
class PortalVacuumDerivativeAudit:
    """Exact polynomial derivatives of the unbroken-Z2 Higgs portal."""

    lambda_hp: float
    higgs_vev: float
    h_phi_cross_hessian: float
    phi_mass_shift: float
    h_phi_phi_cubic: float
    h_h_phi_phi_quartic: float
    expected_phi_mass_shift: float
    expected_h_phi_phi_cubic: float
    expected_h_h_phi_phi_quartic: float
    maximum_identity_residual: float
    tolerance: float
    cross_hessian_zero: bool
    algebraic_pass: bool


@dataclass(frozen=True)
class HiggsInvisibleWidthAudit:
    """Tree-level ``h -> phi phi`` audit for a supplied portal benchmark."""

    lambda_hp: float
    higgs_vev: float
    higgs_mass: float
    scalar_mass: float
    sm_higgs_width: float
    branching_fraction_upper_limit: float
    kinematically_open: bool
    phase_space_factor: float
    partial_width: float
    branching_fraction: float
    benchmark_allowed: bool


@dataclass(frozen=True)
class A1Q0ActionReport:
    """Serializable report with physical-completion claims locked off."""

    schema_version: str
    scope: str
    conditional_status: str
    covariant_action_complete: bool
    stress_tensor_derived: bool
    spectral_density_derived: bool
    hessian_coordinate_audit: HessianCoordinateAudit
    portal_vacuum_audit: PortalVacuumDerivativeAudit
    assumptions_not_audited: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _as_finite_real(value: float, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a real scalar")
    if np.iscomplexobj(raw) and float(abs(np.imag(raw))) > 0.0:
        raise ValueError(f"{name} must be real")
    result = float(np.real(raw))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_tolerance(tolerance: float) -> float:
    value = _as_finite_real(tolerance, name="tolerance")
    if value <= 0.0:
        raise ValueError("tolerance must be positive")
    return value


def _close(left: float, right: float, tolerance: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def audit_nonlinear_hessian_transform(
    *,
    action_gradient_x: float,
    action_hessian_x: float,
    dx_dy: float,
    d2x_dy2: float,
    tolerance: float = 1.0e-12,
) -> HessianCoordinateAudit:
    """Audit an ordinary versus covariant Hessian under ``x = x(y)``.

    The reference ``x`` coordinate is assigned a zero connection.  For a scalar
    action ``S``, the ordinary chain rule is

    ``S_yy = x_y**2 S_xx + x_yy S_x``.

    The second term is the non-tensor contribution.  At a stationary point it
    vanishes.  Away from stationarity, the induced one-dimensional connection
    ``Gamma^y_yy = x_yy / x_y`` removes it from
    ``S_yy - Gamma^y_yy S_y``.

    This is a local-jet audit.  ``dx_dy`` must be nonzero, and ``d2x_dy2`` must
    be nonzero at the audited point so that the example is locally nonlinear.
    """
    threshold = _validate_tolerance(tolerance)
    gradient_x = _as_finite_real(action_gradient_x, name="action_gradient_x")
    hessian_x = _as_finite_real(action_hessian_x, name="action_hessian_x")
    jacobian = _as_finite_real(dx_dy, name="dx_dy")
    curvature = _as_finite_real(d2x_dy2, name="d2x_dy2")
    if abs(jacobian) <= threshold:
        raise ValueError("dx_dy must be nonzero at the audited point")
    if abs(curvature) <= threshold:
        raise ValueError(
            "d2x_dy2 must be nonzero for a locally nonlinear audit"
        )

    gradient_y = jacobian * gradient_x
    tensor_pullback = jacobian**2 * hessian_x
    extra_term = curvature * gradient_x
    ordinary_hessian = tensor_pullback + extra_term

    induced_connection = curvature / jacobian
    connection_correction = induced_connection * gradient_y
    covariant_hessian = ordinary_hessian - connection_correction

    chain_rule_residual = abs(
        ordinary_hessian - tensor_pullback - extra_term
    )
    covariance_residual = abs(covariant_hessian - tensor_pullback)
    stationary = abs(gradient_x) <= threshold
    ordinary_tensorial = _close(ordinary_hessian, tensor_pullback, threshold)
    covariant_tensorial = _close(covariant_hessian, tensor_pullback, threshold)
    structural_pass = (
        chain_rule_residual <= threshold
        and covariance_residual <= threshold
        and covariant_tensorial
    )
    return HessianCoordinateAudit(
        action_gradient_x=gradient_x,
        action_hessian_x=hessian_x,
        dx_dy=jacobian,
        d2x_dy2=curvature,
        action_gradient_y=gradient_y,
        tensor_pullback_hessian_y=tensor_pullback,
        ordinary_hessian_y=ordinary_hessian,
        non_tensor_extra_term=extra_term,
        induced_connection_y=induced_connection,
        connection_correction=connection_correction,
        covariant_hessian_y=covariant_hessian,
        chain_rule_residual=chain_rule_residual,
        covariance_residual=covariance_residual,
        tolerance=threshold,
        stationary=stationary,
        locally_nonlinear=True,
        ordinary_tensorial=ordinary_tensorial,
        covariant_tensorial=covariant_tensorial,
        structural_pass=structural_pass,
    )


def _portal_polynomial_coefficients(
    lambda_hp: float,
    higgs_vev: float,
) -> dict[tuple[int, int], float]:
    """Expand ``(lambda_hp / 2) (v + h)^2 phi^2`` in ``(h, phi)``."""
    return {
        (0, 2): 0.5 * lambda_hp * higgs_vev**2,
        (1, 2): lambda_hp * higgs_vev,
        (2, 2): 0.5 * lambda_hp,
    }


def _polynomial_derivative_at_origin(
    coefficients: dict[tuple[int, int], float],
    *,
    h_order: int,
    phi_order: int,
) -> float:
    coefficient = coefficients.get((h_order, phi_order), 0.0)
    return float(coefficient * factorial(h_order) * factorial(phi_order))


def audit_z2_higgs_portal(
    *,
    lambda_hp: float,
    higgs_vev: float,
    tolerance: float = 1.0e-12,
) -> PortalVacuumDerivativeAudit:
    """Audit vacuum derivatives of ``V=(lambda_hp/2)(v+h)^2 phi^2``.

    All derivatives are evaluated at ``h = phi = 0``.  The mass-shift naming
    follows the potential convention ``V contains (1/2) delta_m_phi^2 phi^2``.
    """
    threshold = _validate_tolerance(tolerance)
    coupling = _as_finite_real(lambda_hp, name="lambda_hp")
    vev = _as_finite_real(higgs_vev, name="higgs_vev")
    coefficients = _portal_polynomial_coefficients(coupling, vev)

    cross_hessian = _polynomial_derivative_at_origin(
        coefficients,
        h_order=1,
        phi_order=1,
    )
    mass_shift = _polynomial_derivative_at_origin(
        coefficients,
        h_order=0,
        phi_order=2,
    )
    cubic = _polynomial_derivative_at_origin(
        coefficients,
        h_order=1,
        phi_order=2,
    )
    quartic = _polynomial_derivative_at_origin(
        coefficients,
        h_order=2,
        phi_order=2,
    )

    expected_mass_shift = coupling * vev**2
    expected_cubic = 2.0 * coupling * vev
    expected_quartic = 2.0 * coupling
    residuals = (
        abs(cross_hessian),
        abs(mass_shift - expected_mass_shift),
        abs(cubic - expected_cubic),
        abs(quartic - expected_quartic),
    )
    maximum_residual = max(residuals)
    scale = max(
        1.0,
        abs(expected_mass_shift),
        abs(expected_cubic),
        abs(expected_quartic),
    )
    cross_hessian_zero = abs(cross_hessian) <= threshold
    algebraic_pass = cross_hessian_zero and maximum_residual <= threshold * scale
    return PortalVacuumDerivativeAudit(
        lambda_hp=coupling,
        higgs_vev=vev,
        h_phi_cross_hessian=cross_hessian,
        phi_mass_shift=mass_shift,
        h_phi_phi_cubic=cubic,
        h_h_phi_phi_quartic=quartic,
        expected_phi_mass_shift=expected_mass_shift,
        expected_h_phi_phi_cubic=expected_cubic,
        expected_h_h_phi_phi_quartic=expected_quartic,
        maximum_identity_residual=maximum_residual,
        tolerance=threshold,
        cross_hessian_zero=cross_hessian_zero,
        algebraic_pass=algebraic_pass,
    )


def audit_higgs_invisible_width(
    *,
    lambda_hp: float,
    higgs_vev: float,
    higgs_mass: float,
    scalar_mass: float,
    sm_higgs_width: float,
    branching_fraction_upper_limit: float,
    tolerance: float = 1.0e-12,
) -> HiggsInvisibleWidthAudit:
    """Audit a real-singlet portal benchmark against a supplied BR limit.

    The convention is ``L_portal = -lambda_hp |H|^2 phi^2`` with
    ``H = (0, (v + h) / sqrt(2))``.  Hence the ``h phi phi`` vertex magnitude
    is ``2 lambda_hp v`` and, when the channel is open,

    ``Gamma(h -> phi phi) = lambda_hp**2 v**2 / (8 pi m_h)
    * sqrt(1 - 4 m_phi**2 / m_h**2)``.

    This is a conditional tree-level EFT check.  It neither derives the portal
    from CE nor substitutes for loop, running, detector, or global-fit audits.
    """
    threshold = _validate_tolerance(tolerance)
    coupling = _as_finite_real(lambda_hp, name="lambda_hp")
    vev = _as_finite_real(higgs_vev, name="higgs_vev")
    mass_h = _as_finite_real(higgs_mass, name="higgs_mass")
    mass_phi = _as_finite_real(scalar_mass, name="scalar_mass")
    width_sm = _as_finite_real(sm_higgs_width, name="sm_higgs_width")
    upper_limit = _as_finite_real(
        branching_fraction_upper_limit,
        name="branching_fraction_upper_limit",
    )
    if mass_h <= 0.0:
        raise ValueError("higgs_mass must be positive")
    if mass_phi < 0.0:
        raise ValueError("scalar_mass must be nonnegative")
    if width_sm <= 0.0:
        raise ValueError("sm_higgs_width must be positive")
    if not 0.0 <= upper_limit <= 1.0:
        raise ValueError(
            "branching_fraction_upper_limit must be between 0 and 1"
        )

    kinematically_open = 2.0 * mass_phi < mass_h
    if kinematically_open:
        phase_space = sqrt(1.0 - 4.0 * mass_phi**2 / mass_h**2)
        partial_width = (
            coupling**2 * vev**2 / (8.0 * pi * mass_h) * phase_space
        )
    else:
        phase_space = 0.0
        partial_width = 0.0
    branching_fraction = partial_width / (width_sm + partial_width)
    benchmark_allowed = branching_fraction <= upper_limit + threshold
    return HiggsInvisibleWidthAudit(
        lambda_hp=coupling,
        higgs_vev=vev,
        higgs_mass=mass_h,
        scalar_mass=mass_phi,
        sm_higgs_width=width_sm,
        branching_fraction_upper_limit=upper_limit,
        kinematically_open=kinematically_open,
        phase_space_factor=phase_space,
        partial_width=partial_width,
        branching_fraction=branching_fraction,
        benchmark_allowed=benchmark_allowed,
    )


def a1_q0_action_report(
    *,
    action_gradient_x: float,
    action_hessian_x: float,
    dx_dy: float,
    d2x_dy2: float,
    lambda_hp: float,
    higgs_vev: float,
    tolerance: float = 1.0e-12,
) -> A1Q0ActionReport:
    """Run both local gates while preserving their strictly conditional scope."""
    hessian_audit = audit_nonlinear_hessian_transform(
        action_gradient_x=action_gradient_x,
        action_hessian_x=action_hessian_x,
        dx_dy=dx_dy,
        d2x_dy2=d2x_dy2,
        tolerance=tolerance,
    )
    portal_audit = audit_z2_higgs_portal(
        lambda_hp=lambda_hp,
        higgs_vev=higgs_vev,
        tolerance=tolerance,
    )
    conditional_pass = (
        hessian_audit.structural_pass and portal_audit.algebraic_pass
    )
    return A1Q0ActionReport(
        schema_version="1.0",
        scope=A1_Q0_SCOPE,
        conditional_status=(
            CONDITIONAL_PASS if conditional_pass else CONDITIONAL_FAIL
        ),
        covariant_action_complete=False,
        stress_tensor_derived=False,
        spectral_density_derived=False,
        hessian_coordinate_audit=hessian_audit,
        portal_vacuum_audit=portal_audit,
        assumptions_not_audited=(
            "full CE+SM action, background equations, and boundary conditions",
            "global CE field-space metric, connection, and gauge quotient",
            "gauge fixing, Faddeev-Popov ghosts, and path-integral measure",
            "regularization, counterterms, and renormalization conditions",
            "metric variation and conservation of a renormalized stress tensor",
            "physical poles, cuts, absorptive self-energy, and spectral density",
        ),
        conclusion=_PASS_CONCLUSION if conditional_pass else _FAIL_CONCLUSION,
    )
