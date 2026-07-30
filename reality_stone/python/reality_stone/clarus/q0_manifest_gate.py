"""Strict Q0.0--Q0.3 structural gates for one Abelian-Higgs control slice.

This module deliberately separates a small, exactly auditable toy truncation
from the full CE+SM quantum-action problem.  Only the explicitly prefixed
``control_q0_*_pass`` flags can become true:

* ``control_q0_0_manifest_pass``: the toy manifest has every declaration;
* ``control_q0_1_field_space_pass``: one nonlinear one-dimensional local jet
  is covariant after its induced field-space connection is included;
* ``control_q0_2_background_pass``: the supplied toy background solves its
  scalar tadpole equation within tolerance;
* ``control_q0_3_gauge_pass``: the Abelian ``R_xi`` mixing cancellation and
  Faddeev--Popov ghost-mass identities hold.

The unprefixed ``q0_0_scope_complete`` through ``q0_3_gauge_complete`` flags
refer to the full Q0/CE+SM obligations and are therefore locked false.  So are
``full_q0_pass``, ``full_ce_sm_complete``, stress-tensor derivation, and
spectral-density derivation.

Conventions are fixed to Lorentzian signature ``(-,+,+,+)``,
``L_scalar=-|D H|^2``, ``D=partial-i g A``, and
``F=partial.A-xi g v chi``.  With these choices the scalar kinetic term gives
``+m_A A.d(chi)`` and the gauge-fixing cross term cancels it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


TOY_SCOPE = "q0_minimal_abelian_higgs_rxi_control_slice_only"
TOY_CONDITIONAL_PASS = "Q0_ABELIAN_CONTROL_SLICE_PASS"
TOY_CONDITIONAL_FAIL = "Q0_ABELIAN_CONTROL_SLICE_FAIL"

SIGNATURE_CONVENTION = "(-,+,+,+)"
ACTION_CONVENTION = (
    "S=int d^4x L;L_gauge=-F_mu_nu*F^mu_nu/4;"
    "L_scalar=-|D_mu H|^2-V(H)"
)
COVARIANT_DERIVATIVE_CONVENTION = "D_mu=partial_mu-i*g*A_mu"
GAUGE_TRANSFORMATION_CONVENTION = (
    "A_mu->A_mu+partial_mu(alpha);H->exp(i*g*alpha)*H"
)
R_XI_CONVENTION = (
    "F=partial_dot_A-xi*g*v*chi;L_gf=-F^2/(2*xi)"
)
GHOST_CONVENTION = (
    "L_gh=-c_bar*(delta_F/delta_alpha)*c;"
    "m_ghost^2=xi*g^2*v^2"
)

_REQUIRED_FIELDS = frozenset({"A_mu", "h", "chi", "c", "c_bar"})
_REQUIRED_ACTION_TERMS = frozenset(
    {
        "abelian_gauge_kinetic",
        "complex_scalar_kinetic",
        "quartic_symmetry_breaking_potential",
        "r_xi_gauge_fixing",
        "faddeev_popov_ghost",
        "counterterms",
    }
)
_REQUIRED_EXCLUDED_SECTORS = frozenset(
    {
        "non_abelian_gauge_sector",
        "fermion_yukawa_sector",
        "gravity_and_ce_sector",
        "global_gauge_orbit_and_anomaly_analysis",
    }
)
_REQUIRED_MANIFEST_SECTIONS = (
    "schema_version",
    "scope_id",
    "model_label",
    "field_declarations",
    "action_terms",
    "spacetime_signature",
    "action_convention",
    "covariant_derivative",
    "gauge_transformation",
    "background_declaration",
    "boundary_conditions",
    "field_space_metric",
    "field_space_connection",
    "gauge_fixing",
    "ghost_action",
    "regularization",
    "counterterms",
    "renormalization_conditions",
    "excluded_sectors",
    "full_ce_sm_complete",
)


@dataclass(frozen=True)
class Q0StructuralManifest:
    """Declarations required by the minimal Abelian-Higgs control slice."""

    schema_version: str
    scope_id: str
    model_label: str
    field_declarations: tuple[str, ...]
    action_terms: tuple[str, ...]
    spacetime_signature: str
    action_convention: str
    covariant_derivative: str
    gauge_transformation: str
    background_declaration: str
    boundary_conditions: tuple[str, ...]
    field_space_metric: str
    field_space_connection: str
    gauge_fixing: str
    ghost_action: str
    regularization: str
    counterterms: tuple[str, ...]
    renormalization_conditions: tuple[str, ...]
    excluded_sectors: tuple[str, ...]
    full_ce_sm_complete: bool


@dataclass(frozen=True)
class Q0ControlInputs:
    """Numerical local-jet, background, and gauge inputs for the toy gate."""

    action_gradient_x: float
    action_hessian_x: float
    dx_dy: float
    d2x_dy2: float
    field_metric_x: float
    mu_squared: float
    scalar_self_coupling: float
    scalar_vev: float
    gauge_coupling: float
    xi: float
    gauge_fixing_goldstone_coefficient: float
    declared_ghost_mass_squared: float
    tolerance: float = 1.0e-12


@dataclass(frozen=True)
class Q0ControlBenchmark:
    """Loaded benchmark bundle with provenance supplied by the caller."""

    schema_version: str
    manifest: Q0StructuralManifest
    control_inputs: Q0ControlInputs


@dataclass(frozen=True)
class ManifestCompletenessAudit:
    """Syntactic and convention-lock diagnostics for Q0.0."""

    required_sections: tuple[str, ...]
    present_sections: tuple[str, ...]
    missing_sections: tuple[str, ...]
    invalid_sections: tuple[str, ...]
    convention_issues: tuple[str, ...]
    scope_locked: bool
    signature_locked: bool
    required_fields_declared: bool
    required_action_terms_declared: bool
    excluded_sectors_explicit: bool
    full_claim_locked_false: bool
    complete: bool


@dataclass(frozen=True)
class FieldSpaceJetAudit:
    """One-dimensional covariant-Hessian and metric-connection identities."""

    action_gradient_x: float
    action_hessian_x: float
    dx_dy: float
    d2x_dy2: float
    field_metric_x: float
    action_gradient_y: float
    ordinary_hessian_y: float
    tensor_pullback_hessian_y: float
    non_tensor_extra_term: float
    field_metric_y: float
    field_metric_derivative_y: float
    induced_connection_y: float
    levi_civita_connection_y: float
    covariant_hessian_y: float
    chain_rule_residual: float
    metric_connection_residual: float
    covariance_residual: float
    stationary: bool
    ordinary_tensorial: bool
    covariant_tensorial: bool
    locally_nonlinear: bool
    structural_pass: bool


@dataclass(frozen=True)
class BackgroundTadpoleAudit:
    """Broken-phase tadpole audit for ``V=-mu^2 rho^2/2+lambda rho^4/4``."""

    mu_squared: float
    scalar_self_coupling: float
    scalar_vev: float
    tadpole: float
    goldstone_curvature: float
    radial_curvature: float
    tadpole_goldstone_identity_residual: float
    normalized_tadpole_residual: float
    on_shell_background: bool


@dataclass(frozen=True)
class AbelianRXiAudit:
    """Quadratic Abelian-Higgs ``R_xi`` mixing and ghost identities."""

    gauge_coupling: float
    scalar_vev: float
    xi: float
    vector_mass: float
    vector_mass_squared: float
    gauge_fixing_goldstone_coefficient: float
    kinetic_a_dot_dchi_coefficient: float
    gauge_fixing_a_dot_dchi_coefficient: float
    net_a_dot_dchi_coefficient: float
    gauge_fixing_goldstone_mass_squared: float
    fp_operator_ghost_mass_squared: float
    declared_ghost_mass_squared: float
    expected_r_xi_mass_squared: float
    mixing_cancellation_residual: float
    fp_operator_residual: float
    goldstone_ghost_mass_identity_residual: float
    expected_ghost_mass_residual: float
    mixing_cancelled: bool
    fp_operator_consistent: bool
    goldstone_ghost_masses_match: bool
    structural_pass: bool


@dataclass(frozen=True)
class Q0ManifestGateReport:
    """Cumulative toy flags with every full-theory claim locked false."""

    schema_version: str
    scope: str
    control_scope: str
    structural_status: str
    control_q0_0_manifest_pass: bool
    control_q0_1_field_space_pass: bool
    control_q0_2_background_pass: bool
    control_q0_3_gauge_pass: bool
    abelian_control_slice_pass: bool
    q0_0_scope_complete: bool
    q0_1_field_space_complete: bool
    q0_2_background_complete: bool
    q0_3_gauge_complete: bool
    full_q0_pass: bool
    full_ce_sm_complete: bool
    stress_tensor_derived: bool
    spectral_density_derived: bool
    excluded_sectors: tuple[str, ...]
    manifest_audit: ManifestCompletenessAudit
    field_space_audit: FieldSpaceJetAudit
    background_audit: BackgroundTadpoleAudit
    gauge_audit: AbelianRXiAudit
    assumptions_not_audited: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _as_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _as_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _close(left: float, right: float, tolerance: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_string_tuple(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(_is_nonempty_string(item) for item in value)
        and len(set(value)) == len(value)
    )


def _section_is_present(manifest: Q0StructuralManifest, name: str) -> bool:
    value = getattr(manifest, name)
    if name == "full_ce_sm_complete":
        return isinstance(value, bool)
    if isinstance(value, tuple):
        return bool(value)
    return _is_nonempty_string(value)


def audit_q0_manifest(
    manifest: Q0StructuralManifest,
) -> ManifestCompletenessAudit:
    """Validate completeness and exact convention locks for the toy manifest."""
    if not isinstance(manifest, Q0StructuralManifest):
        raise TypeError("manifest must be a Q0StructuralManifest")

    present = tuple(
        name
        for name in _REQUIRED_MANIFEST_SECTIONS
        if _section_is_present(manifest, name)
    )
    missing = tuple(
        name for name in _REQUIRED_MANIFEST_SECTIONS if name not in present
    )
    tuple_sections = (
        "field_declarations",
        "action_terms",
        "boundary_conditions",
        "counterterms",
        "renormalization_conditions",
        "excluded_sectors",
    )
    invalid = tuple(
        name
        for name in tuple_sections
        if _section_is_present(manifest, name)
        and not _valid_string_tuple(getattr(manifest, name))
    )

    scope_locked = manifest.scope_id == TOY_SCOPE
    signature_locked = manifest.spacetime_signature == SIGNATURE_CONVENTION
    fields_valid = _valid_string_tuple(manifest.field_declarations)
    terms_valid = _valid_string_tuple(manifest.action_terms)
    excluded_valid = _valid_string_tuple(manifest.excluded_sectors)
    fields_declared = fields_valid and _REQUIRED_FIELDS.issubset(
        manifest.field_declarations
    )
    action_terms_declared = terms_valid and _REQUIRED_ACTION_TERMS.issubset(
        manifest.action_terms
    )
    excluded_explicit = excluded_valid and _REQUIRED_EXCLUDED_SECTORS.issubset(
        manifest.excluded_sectors
    )
    full_claim_locked_false = manifest.full_ce_sm_complete is False

    convention_checks = (
        ("scope_id", scope_locked),
        ("spacetime_signature", signature_locked),
        ("action_convention", manifest.action_convention == ACTION_CONVENTION),
        (
            "covariant_derivative",
            manifest.covariant_derivative
            == COVARIANT_DERIVATIVE_CONVENTION,
        ),
        (
            "gauge_transformation",
            manifest.gauge_transformation
            == GAUGE_TRANSFORMATION_CONVENTION,
        ),
        ("gauge_fixing", manifest.gauge_fixing == R_XI_CONVENTION),
        ("ghost_action", manifest.ghost_action == GHOST_CONVENTION),
        ("required_fields", fields_declared),
        ("required_action_terms", action_terms_declared),
        ("excluded_sectors", excluded_explicit),
        ("full_ce_sm_complete", full_claim_locked_false),
    )
    convention_issues = tuple(
        name for name, passed in convention_checks if not passed
    )
    complete = not missing and not invalid and not convention_issues
    return ManifestCompletenessAudit(
        required_sections=_REQUIRED_MANIFEST_SECTIONS,
        present_sections=present,
        missing_sections=missing,
        invalid_sections=invalid,
        convention_issues=convention_issues,
        scope_locked=scope_locked,
        signature_locked=signature_locked,
        required_fields_declared=fields_declared,
        required_action_terms_declared=action_terms_declared,
        excluded_sectors_explicit=excluded_explicit,
        full_claim_locked_false=full_claim_locked_false,
        complete=complete,
    )


def audit_field_space_local_jet(
    *,
    action_gradient_x: float,
    action_hessian_x: float,
    dx_dy: float,
    d2x_dy2: float,
    field_metric_x: float,
    tolerance: float = 1.0e-12,
) -> FieldSpaceJetAudit:
    """Audit a nonlinear local field-coordinate jet ``x=x(y)``.

    The reference coordinate has constant positive metric and zero connection.
    The transformed metric gives the same induced connection that cancels the
    non-tensor term in the ordinary action Hessian.
    """
    threshold = _positive(tolerance, name="tolerance")
    gradient_x = _as_finite_real(
        action_gradient_x,
        name="action_gradient_x",
    )
    hessian_x = _as_finite_real(
        action_hessian_x,
        name="action_hessian_x",
    )
    jacobian = _as_finite_real(dx_dy, name="dx_dy")
    curvature = _as_finite_real(d2x_dy2, name="d2x_dy2")
    metric_x = _positive(field_metric_x, name="field_metric_x")
    if abs(jacobian) <= threshold:
        raise ValueError("dx_dy must be nonzero at the audited point")
    if abs(curvature) <= threshold:
        raise ValueError(
            "d2x_dy2 must be nonzero for a nonlinear local jet"
        )

    gradient_y = jacobian * gradient_x
    tensor_pullback = jacobian**2 * hessian_x
    extra_term = curvature * gradient_x
    ordinary_hessian = tensor_pullback + extra_term

    metric_y = metric_x * jacobian**2
    metric_derivative_y = 2.0 * metric_x * jacobian * curvature
    levi_civita_connection = 0.5 * metric_derivative_y / metric_y
    induced_connection = curvature / jacobian
    covariant_hessian = (
        ordinary_hessian - levi_civita_connection * gradient_y
    )

    chain_rule_residual = abs(
        ordinary_hessian - tensor_pullback - extra_term
    )
    metric_connection_residual = abs(
        levi_civita_connection - induced_connection
    )
    covariance_residual = abs(covariant_hessian - tensor_pullback)
    stationary = abs(gradient_x) <= threshold
    ordinary_tensorial = _close(
        ordinary_hessian,
        tensor_pullback,
        threshold,
    )
    covariant_tensorial = _close(
        covariant_hessian,
        tensor_pullback,
        threshold,
    )
    structural_pass = (
        chain_rule_residual <= threshold
        and metric_connection_residual <= threshold
        and covariance_residual <= threshold
        and covariant_tensorial
    )
    return FieldSpaceJetAudit(
        action_gradient_x=gradient_x,
        action_hessian_x=hessian_x,
        dx_dy=jacobian,
        d2x_dy2=curvature,
        field_metric_x=metric_x,
        action_gradient_y=gradient_y,
        ordinary_hessian_y=ordinary_hessian,
        tensor_pullback_hessian_y=tensor_pullback,
        non_tensor_extra_term=extra_term,
        field_metric_y=metric_y,
        field_metric_derivative_y=metric_derivative_y,
        induced_connection_y=induced_connection,
        levi_civita_connection_y=levi_civita_connection,
        covariant_hessian_y=covariant_hessian,
        chain_rule_residual=chain_rule_residual,
        metric_connection_residual=metric_connection_residual,
        covariance_residual=covariance_residual,
        stationary=stationary,
        ordinary_tensorial=ordinary_tensorial,
        covariant_tensorial=covariant_tensorial,
        locally_nonlinear=True,
        structural_pass=structural_pass,
    )


def audit_background_tadpole(
    *,
    mu_squared: float,
    scalar_self_coupling: float,
    scalar_vev: float,
    tolerance: float = 1.0e-12,
) -> BackgroundTadpoleAudit:
    """Audit the supplied broken-phase scalar background."""
    threshold = _positive(tolerance, name="tolerance")
    mass_parameter = _positive(mu_squared, name="mu_squared")
    coupling = _positive(
        scalar_self_coupling,
        name="scalar_self_coupling",
    )
    vev = _positive(scalar_vev, name="scalar_vev")

    goldstone_curvature = -mass_parameter + coupling * vev**2
    tadpole = vev * goldstone_curvature
    radial_curvature = -mass_parameter + 3.0 * coupling * vev**2
    identity_residual = abs(tadpole - vev * goldstone_curvature)
    scale = max(
        1.0,
        abs(mass_parameter * vev),
        abs(coupling * vev**3),
    )
    normalized_residual = abs(tadpole) / scale
    on_shell = (
        identity_residual <= threshold * scale
        and normalized_residual <= threshold
    )
    return BackgroundTadpoleAudit(
        mu_squared=mass_parameter,
        scalar_self_coupling=coupling,
        scalar_vev=vev,
        tadpole=tadpole,
        goldstone_curvature=goldstone_curvature,
        radial_curvature=radial_curvature,
        tadpole_goldstone_identity_residual=identity_residual,
        normalized_tadpole_residual=normalized_residual,
        on_shell_background=on_shell,
    )


def audit_abelian_higgs_r_xi(
    *,
    gauge_coupling: float,
    scalar_vev: float,
    xi: float,
    gauge_fixing_goldstone_coefficient: float,
    declared_ghost_mass_squared: float,
    tolerance: float = 1.0e-12,
) -> AbelianRXiAudit:
    """Audit the exact quadratic ``R_xi`` identities in the fixed convention.

    Write ``F=partial.A-kappa chi``.  The requested ``R_xi`` choice is
    ``kappa=xi*m_A``.  After integration by parts the gauge-fixing cross term
    contributes ``-(kappa/xi) A.d(chi)``, cancelling the scalar-kinetic
    coefficient ``+m_A``.  Linearized Faddeev--Popov variation gives the ghost
    mass magnitude ``kappa*m_A``.
    """
    threshold = _positive(tolerance, name="tolerance")
    coupling = _positive(gauge_coupling, name="gauge_coupling")
    vev = _positive(scalar_vev, name="scalar_vev")
    gauge_parameter = _positive(xi, name="xi")
    kappa = _as_finite_real(
        gauge_fixing_goldstone_coefficient,
        name="gauge_fixing_goldstone_coefficient",
    )
    declared_ghost_mass = _as_finite_real(
        declared_ghost_mass_squared,
        name="declared_ghost_mass_squared",
    )

    vector_mass = coupling * vev
    vector_mass_squared = vector_mass**2
    kinetic_mixing = vector_mass
    gauge_fixing_mixing = -kappa / gauge_parameter
    net_mixing = kinetic_mixing + gauge_fixing_mixing

    gauge_fixing_goldstone_mass = kappa**2 / gauge_parameter
    fp_ghost_mass = kappa * vector_mass
    expected_mass = gauge_parameter * vector_mass_squared
    mixing_residual = abs(net_mixing)
    fp_residual = abs(declared_ghost_mass - fp_ghost_mass)
    identity_residual = abs(
        gauge_fixing_goldstone_mass - declared_ghost_mass
    )
    expected_residual = abs(declared_ghost_mass - expected_mass)
    scale = max(
        1.0,
        abs(vector_mass),
        abs(gauge_fixing_goldstone_mass),
        abs(fp_ghost_mass),
        abs(declared_ghost_mass),
        abs(expected_mass),
    )
    mixing_cancelled = mixing_residual <= threshold * scale
    fp_consistent = fp_residual <= threshold * scale
    ghost_masses_match = (
        identity_residual <= threshold * scale
        and expected_residual <= threshold * scale
    )
    structural_pass = (
        mixing_cancelled and fp_consistent and ghost_masses_match
    )
    return AbelianRXiAudit(
        gauge_coupling=coupling,
        scalar_vev=vev,
        xi=gauge_parameter,
        vector_mass=vector_mass,
        vector_mass_squared=vector_mass_squared,
        gauge_fixing_goldstone_coefficient=kappa,
        kinetic_a_dot_dchi_coefficient=kinetic_mixing,
        gauge_fixing_a_dot_dchi_coefficient=gauge_fixing_mixing,
        net_a_dot_dchi_coefficient=net_mixing,
        gauge_fixing_goldstone_mass_squared=gauge_fixing_goldstone_mass,
        fp_operator_ghost_mass_squared=fp_ghost_mass,
        declared_ghost_mass_squared=declared_ghost_mass,
        expected_r_xi_mass_squared=expected_mass,
        mixing_cancellation_residual=mixing_residual,
        fp_operator_residual=fp_residual,
        goldstone_ghost_mass_identity_residual=identity_residual,
        expected_ghost_mass_residual=expected_residual,
        mixing_cancelled=mixing_cancelled,
        fp_operator_consistent=fp_consistent,
        goldstone_ghost_masses_match=ghost_masses_match,
        structural_pass=structural_pass,
    )


def q0_manifest_gate_report(
    manifest: Q0StructuralManifest,
    control_inputs: Q0ControlInputs,
) -> Q0ManifestGateReport:
    """Run cumulative Q0.0--Q0.3 gates for the declared toy scope only."""
    if not isinstance(control_inputs, Q0ControlInputs):
        raise TypeError("control_inputs must be Q0ControlInputs")
    tolerance = _positive(control_inputs.tolerance, name="tolerance")
    manifest_audit = audit_q0_manifest(manifest)
    field_space_audit = audit_field_space_local_jet(
        action_gradient_x=control_inputs.action_gradient_x,
        action_hessian_x=control_inputs.action_hessian_x,
        dx_dy=control_inputs.dx_dy,
        d2x_dy2=control_inputs.d2x_dy2,
        field_metric_x=control_inputs.field_metric_x,
        tolerance=tolerance,
    )
    background_audit = audit_background_tadpole(
        mu_squared=control_inputs.mu_squared,
        scalar_self_coupling=control_inputs.scalar_self_coupling,
        scalar_vev=control_inputs.scalar_vev,
        tolerance=tolerance,
    )
    gauge_audit = audit_abelian_higgs_r_xi(
        gauge_coupling=control_inputs.gauge_coupling,
        scalar_vev=control_inputs.scalar_vev,
        xi=control_inputs.xi,
        gauge_fixing_goldstone_coefficient=(
            control_inputs.gauge_fixing_goldstone_coefficient
        ),
        declared_ghost_mass_squared=(
            control_inputs.declared_ghost_mass_squared
        ),
        tolerance=tolerance,
    )

    control_q0_0 = manifest_audit.complete
    control_q0_1 = control_q0_0 and field_space_audit.structural_pass
    control_q0_2 = (
        control_q0_1 and background_audit.on_shell_background
    )
    control_q0_3 = control_q0_2 and gauge_audit.structural_pass
    conclusion = (
        "The declared minimal Abelian-Higgs R_xi control slice passes its "
        "cumulative control Q0.0--Q0.3 structural gates. The excluded sectors "
        "keep full Q0, full CE+SM, stress-tensor, and spectral claims open."
        if control_q0_3
        else
        "At least one cumulative control Q0.0--Q0.3 gate failed. No full-Q0 "
        "or full-CE+SM conclusion follows."
    )
    return Q0ManifestGateReport(
        schema_version="1.0",
        scope=TOY_SCOPE,
        control_scope=TOY_SCOPE,
        structural_status=(
            TOY_CONDITIONAL_PASS
            if control_q0_3
            else TOY_CONDITIONAL_FAIL
        ),
        control_q0_0_manifest_pass=control_q0_0,
        control_q0_1_field_space_pass=control_q0_1,
        control_q0_2_background_pass=control_q0_2,
        control_q0_3_gauge_pass=control_q0_3,
        abelian_control_slice_pass=control_q0_3,
        q0_0_scope_complete=False,
        q0_1_field_space_complete=False,
        q0_2_background_complete=False,
        q0_3_gauge_complete=False,
        full_q0_pass=False,
        full_ce_sm_complete=False,
        stress_tensor_derived=False,
        spectral_density_derived=False,
        excluded_sectors=manifest.excluded_sectors,
        manifest_audit=manifest_audit,
        field_space_audit=field_space_audit,
        background_audit=background_audit,
        gauge_audit=gauge_audit,
        assumptions_not_audited=(
            "the complete CE field content and CE-to-SM identification",
            "the non-Abelian SU(3)xSU(2) gauge sectors",
            "fermions, Yukawa couplings, flavor, and chiral anomalies",
            "gravity, metric variation, and boundary counterterms",
            "global gauge-orbit geometry, Gribov copies, and BRST cohomology",
            "regulator-preserved Ward identities and a full counterterm basis",
            "renormalized stress tensor, poles, cuts, and spectral density",
        ),
        conclusion=conclusion,
    )


def _mapping(payload: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _required(payload: Mapping[str, Any], key: str, *, parent: str) -> Any:
    if key not in payload:
        raise ValueError(f"{parent}.{key} is required")
    return payload[key]


def _string(payload: Mapping[str, Any], key: str, *, parent: str) -> str:
    value = _required(payload, key, parent=parent)
    if not _is_nonempty_string(value):
        raise ValueError(f"{parent}.{key} must be a non-empty string")
    return value


def _string_tuple(
    payload: Mapping[str, Any],
    key: str,
    *,
    parent: str,
) -> tuple[str, ...]:
    value = _required(payload, key, parent=parent)
    if (
        not isinstance(value, list)
        or not value
        or not all(_is_nonempty_string(item) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{parent}.{key} must be a non-empty list of unique strings"
        )
    return tuple(value)


def manifest_from_mapping(payload: Mapping[str, Any]) -> Q0StructuralManifest:
    """Parse a strict manifest mapping loaded from JSON."""
    parent = "manifest"
    full_claim = _required(payload, "full_ce_sm_complete", parent=parent)
    if not isinstance(full_claim, bool):
        raise ValueError("manifest.full_ce_sm_complete must be a boolean")
    return Q0StructuralManifest(
        schema_version=_string(payload, "schema_version", parent=parent),
        scope_id=_string(payload, "scope_id", parent=parent),
        model_label=_string(payload, "model_label", parent=parent),
        field_declarations=_string_tuple(
            payload,
            "field_declarations",
            parent=parent,
        ),
        action_terms=_string_tuple(payload, "action_terms", parent=parent),
        spacetime_signature=_string(
            payload,
            "spacetime_signature",
            parent=parent,
        ),
        action_convention=_string(
            payload,
            "action_convention",
            parent=parent,
        ),
        covariant_derivative=_string(
            payload,
            "covariant_derivative",
            parent=parent,
        ),
        gauge_transformation=_string(
            payload,
            "gauge_transformation",
            parent=parent,
        ),
        background_declaration=_string(
            payload,
            "background_declaration",
            parent=parent,
        ),
        boundary_conditions=_string_tuple(
            payload,
            "boundary_conditions",
            parent=parent,
        ),
        field_space_metric=_string(
            payload,
            "field_space_metric",
            parent=parent,
        ),
        field_space_connection=_string(
            payload,
            "field_space_connection",
            parent=parent,
        ),
        gauge_fixing=_string(payload, "gauge_fixing", parent=parent),
        ghost_action=_string(payload, "ghost_action", parent=parent),
        regularization=_string(payload, "regularization", parent=parent),
        counterterms=_string_tuple(payload, "counterterms", parent=parent),
        renormalization_conditions=_string_tuple(
            payload,
            "renormalization_conditions",
            parent=parent,
        ),
        excluded_sectors=_string_tuple(
            payload,
            "excluded_sectors",
            parent=parent,
        ),
        full_ce_sm_complete=full_claim,
    )


def control_inputs_from_mapping(
    payload: Mapping[str, Any],
) -> Q0ControlInputs:
    """Parse strict numerical control inputs from a JSON mapping."""
    names = (
        "action_gradient_x",
        "action_hessian_x",
        "dx_dy",
        "d2x_dy2",
        "field_metric_x",
        "mu_squared",
        "scalar_self_coupling",
        "scalar_vev",
        "gauge_coupling",
        "xi",
        "gauge_fixing_goldstone_coefficient",
        "declared_ghost_mass_squared",
    )
    values = {
        name: _as_finite_real(
            _required(payload, name, parent="control_inputs"),
            name=f"control_inputs.{name}",
        )
        for name in names
    }
    tolerance = _as_finite_real(
        payload.get("tolerance", 1.0e-12),
        name="control_inputs.tolerance",
    )
    return Q0ControlInputs(**values, tolerance=tolerance)


def load_q0_control_benchmark(
    path: str | Path,
) -> Q0ControlBenchmark:
    """Load a manifest and fixed control inputs from a JSON benchmark."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load Q0 benchmark {source}: {error}") from error
    root = _mapping(payload, name="benchmark")
    schema_version = _string(root, "schema_version", parent="benchmark")
    manifest_payload = _mapping(
        _required(root, "manifest", parent="benchmark"),
        name="benchmark.manifest",
    )
    inputs_payload = _mapping(
        _required(root, "control_inputs", parent="benchmark"),
        name="benchmark.control_inputs",
    )
    return Q0ControlBenchmark(
        schema_version=schema_version,
        manifest=manifest_from_mapping(manifest_payload),
        control_inputs=control_inputs_from_mapping(inputs_payload),
    )
