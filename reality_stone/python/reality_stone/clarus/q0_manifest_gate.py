"""Strict Q0.0--Q0.3 structural gates for one Abelian-Higgs control slice.

This module deliberately separates a small, exactly auditable toy truncation
from the full CE+SM quantum-action problem.  Only the explicitly prefixed
``control_q0_*_pass`` flags can become true as independent local diagnostics:

* ``control_q0_0_pass``: the toy manifest has every declaration;
* ``control_q0_1_pass``: one nonlinear one-dimensional local jet
  is covariant after its induced field-space connection is included;
* ``control_q0_2_pass``: the supplied toy background solves its
  scalar tadpole equation within tolerance;
* ``control_q0_3_pass``: the Abelian ``R_xi`` mixing cancellation and
  Faddeev--Popov ghost-mass identities hold.

``control_through_q0_3_pass`` is their conjunction.  The
``full_q0_0_complete`` through ``full_q0_3_complete`` flags refer to the full
Q0/CE+SM obligations and are therefore locked false.  So are
``full_q0_pass``, ``full_ce_sm_complete``, stress-tensor derivation, and
spectral-density derivation.

The control action is classical, bare, and tree level.  It includes a
``Z2``-odd real singlet ``phi`` with an independent Higgs-portal coupling;
counterterms and renormalization are declared but explicitly not applied.
Conventions are fixed to Lorentzian signature ``(-,+,+,+)``,
``L_scalar=-|D H|^2``, ``D=partial-i g A``, and
``F=partial.A-xi g v chi``.  With these choices the scalar kinetic term gives
``+m_A A.d(chi)`` and the gauge-fixing cross term cancels it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


TOY_SCOPE = "q0_minimal_abelian_higgs_z2_singlet_rxi_control_slice_only"
TOY_CONDITIONAL_PASS = "Q0_ABELIAN_Z2_CONTROL_SLICE_PASS"
TOY_CONDITIONAL_FAIL = "Q0_ABELIAN_Z2_CONTROL_SLICE_FAIL"

SIGNATURE_CONVENTION = "(-,+,+,+)"
ACTION_KIND = "classical_bare_tree_level_control"
FIXED_BACKGROUND_METRIC = "eta_mu_nu=diag(-1,+1,+1,+1);not_varied"
NATURAL_UNITS = "natural_units_hbar=c=1"
ACTION_CONVENTION = (
    "S=int d^4x L;L_gauge=-F_mu_nu*F^mu_nu/4;L_scalar=-|D_mu H|^2-(partial_mu phi)^2/2-V(H,phi)"
)
POTENTIAL_CONVENTION = (
    "V=-mu_H^2*|H|^2+lambda_H*|H|^4+(m0_phi^2/2)*phi^2+(lambda_phi/4)*phi^4+lambda_HP*|H|^2*phi^2"
)
COVARIANT_DERIVATIVE_CONVENTION = "D_mu=partial_mu-i*g*A_mu"
GAUGE_TRANSFORMATION_CONVENTION = "A_mu->A_mu+partial_mu(alpha);H->exp(i*g*alpha)*H"
R_XI_CONVENTION = "F=partial_dot_A-xi*g*v*chi;L_gf=-F^2/(2*xi)"
GHOST_CONVENTION = "L_gh=-c_bar*(delta_F/delta_alpha)*c;m_ghost^2=xi*g^2*v^2"
BACKGROUND_CONVENTION = "H=(v+h+i*chi)/sqrt(2);phi=0;A_mu=0;h=chi=0"
FIELD_SPACE_METRIC_CONVENTION = (
    "G_(h,chi,phi)=diag(1,1,1);local_jet_audits_one_scalar_direction_only"
)
FIELD_SPACE_CONNECTION_CONVENTION = (
    "Gamma=0_in_Cartesian_(h,chi,phi);Gamma_y=(d2x/dy2)/(dx/dy)_on_one_direction_local_jet"
)
NOT_APPLIED_STATUS = "not_applied_excluded_from_control_scope"


def q0_control_action_definition_payload() -> dict[str, object]:
    """Return the canonical declarations that define the Q0 control action.

    Numerical benchmark values are deliberately excluded: this digest identifies
    the selected action, field basis, background, and conventions rather than one
    parameter point.  Full CE and renormalized-action claims remain excluded by
    the payload itself.
    """

    return {
        "scope_id": TOY_SCOPE,
        "action_kind": ACTION_KIND,
        "spacetime_signature": SIGNATURE_CONVENTION,
        "fixed_background_metric": FIXED_BACKGROUND_METRIC,
        "units": NATURAL_UNITS,
        "action_convention": ACTION_CONVENTION,
        "potential_convention": POTENTIAL_CONVENTION,
        "covariant_derivative": COVARIANT_DERIVATIVE_CONVENTION,
        "gauge_transformation": GAUGE_TRANSFORMATION_CONVENTION,
        "background_declaration": BACKGROUND_CONVENTION,
        "field_space_metric": FIELD_SPACE_METRIC_CONVENTION,
        "field_space_connection": FIELD_SPACE_CONNECTION_CONVENTION,
        "gauge_fixing": R_XI_CONVENTION,
        "ghost_action": GHOST_CONVENTION,
        "counterterm_status": NOT_APPLIED_STATUS,
        "renormalization_status": NOT_APPLIED_STATUS,
        "field_declarations": sorted(_REQUIRED_FIELDS),
        "action_terms": sorted(_REQUIRED_ACTION_TERMS),
        "excluded_sectors": sorted(_REQUIRED_EXCLUDED_SECTORS),
        "full_ce_sm_complete": False,
    }


def q0_control_action_definition_sha256() -> str:
    """Hash the canonical Q0 action declarations without filesystem state."""

    canonical = json.dumps(
        q0_control_action_definition_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


_REQUIRED_FIELDS = frozenset({"A_mu", "h", "chi", "phi", "c", "c_bar"})
_REQUIRED_ACTION_TERMS = frozenset(
    {
        "abelian_gauge_kinetic",
        "complex_scalar_kinetic",
        "z2_singlet_scalar_kinetic",
        "higgs_z2_singlet_portal_potential",
        "r_xi_gauge_fixing",
        "faddeev_popov_ghost",
    }
)
_REQUIRED_EXCLUDED_SECTORS = frozenset(
    {
        "non_abelian_gauge_sector",
        "fermion_yukawa_sector",
        "gravity_and_ce_sector",
        "global_gauge_orbit_and_anomaly_analysis",
        "loop_counterterms_and_renormalization",
    }
)
_REQUIRED_BOUNDARY_CONDITIONS = frozenset(
    {
        "fluctuations_and_first_derivatives_vanish_at_spatial_infinity",
        "integration_by_parts_surface_terms_vanish",
    }
)
_REQUIRED_MANIFEST_SECTIONS = (
    "schema_version",
    "scope_id",
    "model_label",
    "field_declarations",
    "action_terms",
    "spacetime_signature",
    "action_kind",
    "fixed_background_metric",
    "units",
    "action_convention",
    "potential_convention",
    "covariant_derivative",
    "gauge_transformation",
    "background_declaration",
    "boundary_conditions",
    "field_space_metric",
    "field_space_connection",
    "gauge_fixing",
    "ghost_action",
    "counterterm_status",
    "renormalization_status",
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
    action_kind: str
    fixed_background_metric: str
    units: str
    action_convention: str
    potential_convention: str
    covariant_derivative: str
    gauge_transformation: str
    background_declaration: str
    boundary_conditions: tuple[str, ...]
    field_space_metric: str
    field_space_connection: str
    gauge_fixing: str
    ghost_action: str
    counterterm_status: str
    renormalization_status: str
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
    higgs_self_coupling: float
    higgs_vev: float
    singlet_bare_mass_squared: float
    singlet_self_coupling: float
    lambda_hp: float
    singlet_background: float
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
    cartesian_unit_metric: bool
    locally_nonlinear: bool
    structural_pass: bool


@dataclass(frozen=True)
class BackgroundTadpoleAudit:
    """Higgs and ``Z2``-singlet background identities for the toy potential."""

    mu_squared: float
    higgs_self_coupling: float
    higgs_vev: float
    singlet_bare_mass_squared: float
    singlet_self_coupling: float
    lambda_hp: float
    singlet_background: float
    higgs_tadpole: float
    singlet_tadpole: float
    goldstone_curvature: float
    radial_curvature: float
    singlet_curvature: float
    singlet_effective_mass_squared: float
    higgs_tadpole_identity_residual: float
    singlet_tadpole_identity_residual: float
    normalized_higgs_tadpole_residual: float
    normalized_singlet_tadpole_residual: float
    z2_symmetric_background: bool
    portal_coupling_is_independent_input: bool
    on_shell_background: bool


@dataclass(frozen=True)
class AbelianRXiAudit:
    """Quadratic Abelian-Higgs ``R_xi`` mixing and ghost identities."""

    gauge_coupling: float
    higgs_vev: float
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
    control_q0_0_pass: bool
    control_q0_1_pass: bool
    control_q0_2_pass: bool
    control_q0_3_pass: bool
    control_through_q0_3_pass: bool
    abelian_control_slice_pass: bool
    full_q0_0_complete: bool
    full_q0_1_complete: bool
    full_q0_2_complete: bool
    full_q0_3_complete: bool
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
        name for name in _REQUIRED_MANIFEST_SECTIONS if _section_is_present(manifest, name)
    )
    missing = tuple(name for name in _REQUIRED_MANIFEST_SECTIONS if name not in present)
    tuple_sections = (
        "field_declarations",
        "action_terms",
        "boundary_conditions",
        "excluded_sectors",
    )
    invalid = tuple(
        name
        for name in tuple_sections
        if _section_is_present(manifest, name) and not _valid_string_tuple(getattr(manifest, name))
    )

    scope_locked = manifest.scope_id == TOY_SCOPE
    signature_locked = manifest.spacetime_signature == SIGNATURE_CONVENTION
    fields_valid = _valid_string_tuple(manifest.field_declarations)
    terms_valid = _valid_string_tuple(manifest.action_terms)
    excluded_valid = _valid_string_tuple(manifest.excluded_sectors)
    fields_declared = fields_valid and set(manifest.field_declarations) == _REQUIRED_FIELDS
    action_terms_declared = terms_valid and set(manifest.action_terms) == _REQUIRED_ACTION_TERMS
    boundary_conditions_declared = _valid_string_tuple(
        manifest.boundary_conditions
    ) and _REQUIRED_BOUNDARY_CONDITIONS.issubset(manifest.boundary_conditions)
    excluded_explicit = excluded_valid and _REQUIRED_EXCLUDED_SECTORS.issubset(
        manifest.excluded_sectors
    )
    full_claim_locked_false = manifest.full_ce_sm_complete is False

    convention_checks = (
        ("scope_id", scope_locked),
        ("spacetime_signature", signature_locked),
        ("action_kind", manifest.action_kind == ACTION_KIND),
        (
            "fixed_background_metric",
            manifest.fixed_background_metric == FIXED_BACKGROUND_METRIC,
        ),
        ("units", manifest.units == NATURAL_UNITS),
        ("action_convention", manifest.action_convention == ACTION_CONVENTION),
        (
            "potential_convention",
            manifest.potential_convention == POTENTIAL_CONVENTION,
        ),
        (
            "covariant_derivative",
            manifest.covariant_derivative == COVARIANT_DERIVATIVE_CONVENTION,
        ),
        (
            "gauge_transformation",
            manifest.gauge_transformation == GAUGE_TRANSFORMATION_CONVENTION,
        ),
        (
            "background_declaration",
            manifest.background_declaration == BACKGROUND_CONVENTION,
        ),
        ("boundary_conditions", boundary_conditions_declared),
        (
            "field_space_metric",
            manifest.field_space_metric == FIELD_SPACE_METRIC_CONVENTION,
        ),
        (
            "field_space_connection",
            manifest.field_space_connection == FIELD_SPACE_CONNECTION_CONVENTION,
        ),
        ("gauge_fixing", manifest.gauge_fixing == R_XI_CONVENTION),
        ("ghost_action", manifest.ghost_action == GHOST_CONVENTION),
        (
            "counterterm_status",
            manifest.counterterm_status == NOT_APPLIED_STATUS,
        ),
        (
            "renormalization_status",
            manifest.renormalization_status == NOT_APPLIED_STATUS,
        ),
        ("required_fields", fields_declared),
        ("required_action_terms", action_terms_declared),
        ("excluded_sectors", excluded_explicit),
        ("full_ce_sm_complete", full_claim_locked_false),
    )
    convention_issues = tuple(name for name, passed in convention_checks if not passed)
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

    The reference coordinate is one direction of the manifest's Cartesian
    scalar metric ``G_(h,chi,phi)=diag(1,1,1)``.  The transformed metric gives
    the same induced connection that cancels the non-tensor term in the
    ordinary action Hessian.
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
        raise ValueError("d2x_dy2 must be nonzero for a nonlinear local jet")

    gradient_y = jacobian * gradient_x
    tensor_pullback = jacobian**2 * hessian_x
    extra_term = curvature * gradient_x
    ordinary_hessian = tensor_pullback + extra_term

    metric_y = metric_x * jacobian**2
    metric_derivative_y = 2.0 * metric_x * jacobian * curvature
    levi_civita_connection = 0.5 * metric_derivative_y / metric_y
    induced_connection = curvature / jacobian
    covariant_hessian = ordinary_hessian - levi_civita_connection * gradient_y

    chain_rule_residual = abs(ordinary_hessian - tensor_pullback - extra_term)
    metric_connection_residual = abs(levi_civita_connection - induced_connection)
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
    cartesian_unit_metric = _close(metric_x, 1.0, threshold)
    structural_pass = (
        chain_rule_residual <= threshold
        and metric_connection_residual <= threshold
        and covariance_residual <= threshold
        and covariant_tensorial
        and cartesian_unit_metric
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
        cartesian_unit_metric=cartesian_unit_metric,
        locally_nonlinear=True,
        structural_pass=structural_pass,
    )


def audit_background_tadpole(
    *,
    mu_squared: float,
    higgs_self_coupling: float,
    higgs_vev: float,
    singlet_bare_mass_squared: float,
    singlet_self_coupling: float,
    lambda_hp: float,
    singlet_background: float,
    tolerance: float = 1.0e-12,
) -> BackgroundTadpoleAudit:
    """Audit the Higgs and ``Z2``-singlet background equations.

    With ``H=(v+h+i chi)/sqrt(2)`` and singlet background ``s``, the potential
    is

    ``-mu_H^2 r^2/2 + lambda_H r^4/4 + m0_phi^2 s^2/2
    + lambda_phi s^4/4 + lambda_HP r^2 s^2/2``.

    The control background requires ``s=0``.  Therefore the singlet tadpole
    vanishes by ``Z2`` and its effective mass is
    ``m0_phi^2 + lambda_HP v^2``.  ``lambda_HP`` is supplied independently;
    this gate does not derive it from CE.
    """
    threshold = _positive(tolerance, name="tolerance")
    mass_parameter = _positive(mu_squared, name="mu_squared")
    higgs_coupling = _positive(
        higgs_self_coupling,
        name="higgs_self_coupling",
    )
    vev = _positive(higgs_vev, name="higgs_vev")
    singlet_mass = _as_finite_real(
        singlet_bare_mass_squared,
        name="singlet_bare_mass_squared",
    )
    singlet_coupling = _positive(
        singlet_self_coupling,
        name="singlet_self_coupling",
    )
    portal_coupling = _as_finite_real(lambda_hp, name="lambda_hp")
    singlet_vev = _as_finite_real(
        singlet_background,
        name="singlet_background",
    )

    goldstone_curvature = (
        -mass_parameter + higgs_coupling * vev**2 + portal_coupling * singlet_vev**2
    )
    higgs_tadpole = vev * goldstone_curvature
    radial_curvature = (
        -mass_parameter + 3.0 * higgs_coupling * vev**2 + portal_coupling * singlet_vev**2
    )
    singlet_effective_mass = singlet_mass + portal_coupling * vev**2
    singlet_bracket = singlet_effective_mass + singlet_coupling * singlet_vev**2
    singlet_tadpole = singlet_vev * singlet_bracket
    singlet_curvature = singlet_effective_mass + 3.0 * singlet_coupling * singlet_vev**2
    higgs_identity_residual = abs(higgs_tadpole - vev * goldstone_curvature)
    singlet_identity_residual = abs(singlet_tadpole - singlet_vev * singlet_bracket)
    higgs_scale = max(
        1.0,
        abs(mass_parameter * vev),
        abs(higgs_coupling * vev**3),
        abs(portal_coupling * vev * singlet_vev**2),
    )
    singlet_scale = max(
        1.0,
        abs(singlet_mass * singlet_vev),
        abs(portal_coupling * vev**2 * singlet_vev),
        abs(singlet_coupling * singlet_vev**3),
    )
    normalized_higgs_residual = abs(higgs_tadpole) / higgs_scale
    normalized_singlet_residual = abs(singlet_tadpole) / singlet_scale
    z2_symmetric = abs(singlet_vev) <= threshold
    on_shell = (
        higgs_identity_residual <= threshold * higgs_scale
        and singlet_identity_residual <= threshold * singlet_scale
        and normalized_higgs_residual <= threshold
        and normalized_singlet_residual <= threshold
        and z2_symmetric
    )
    return BackgroundTadpoleAudit(
        mu_squared=mass_parameter,
        higgs_self_coupling=higgs_coupling,
        higgs_vev=vev,
        singlet_bare_mass_squared=singlet_mass,
        singlet_self_coupling=singlet_coupling,
        lambda_hp=portal_coupling,
        singlet_background=singlet_vev,
        higgs_tadpole=higgs_tadpole,
        singlet_tadpole=singlet_tadpole,
        goldstone_curvature=goldstone_curvature,
        radial_curvature=radial_curvature,
        singlet_curvature=singlet_curvature,
        singlet_effective_mass_squared=singlet_effective_mass,
        higgs_tadpole_identity_residual=higgs_identity_residual,
        singlet_tadpole_identity_residual=singlet_identity_residual,
        normalized_higgs_tadpole_residual=normalized_higgs_residual,
        normalized_singlet_tadpole_residual=normalized_singlet_residual,
        z2_symmetric_background=z2_symmetric,
        portal_coupling_is_independent_input=True,
        on_shell_background=on_shell,
    )


def audit_abelian_higgs_r_xi(
    *,
    gauge_coupling: float,
    higgs_vev: float,
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
    vev = _positive(higgs_vev, name="higgs_vev")
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
    identity_residual = abs(gauge_fixing_goldstone_mass - declared_ghost_mass)
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
        identity_residual <= threshold * scale and expected_residual <= threshold * scale
    )
    structural_pass = mixing_cancelled and fp_consistent and ghost_masses_match
    return AbelianRXiAudit(
        gauge_coupling=coupling,
        higgs_vev=vev,
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
        higgs_self_coupling=control_inputs.higgs_self_coupling,
        higgs_vev=control_inputs.higgs_vev,
        singlet_bare_mass_squared=(control_inputs.singlet_bare_mass_squared),
        singlet_self_coupling=control_inputs.singlet_self_coupling,
        lambda_hp=control_inputs.lambda_hp,
        singlet_background=control_inputs.singlet_background,
        tolerance=tolerance,
    )
    gauge_audit = audit_abelian_higgs_r_xi(
        gauge_coupling=control_inputs.gauge_coupling,
        higgs_vev=control_inputs.higgs_vev,
        xi=control_inputs.xi,
        gauge_fixing_goldstone_coefficient=(control_inputs.gauge_fixing_goldstone_coefficient),
        declared_ghost_mass_squared=(control_inputs.declared_ghost_mass_squared),
        tolerance=tolerance,
    )

    control_q0_0 = manifest_audit.complete
    control_q0_1 = field_space_audit.structural_pass
    control_q0_2 = background_audit.on_shell_background
    control_q0_3 = gauge_audit.structural_pass
    control_through_q0_3 = all((control_q0_0, control_q0_1, control_q0_2, control_q0_3))
    conclusion = (
        "The declared minimal Abelian-Higgs plus Z2-singlet R_xi control "
        "slice passes all control Q0.0--Q0.3 structural gates. The excluded "
        "sectors keep full Q0, full CE+SM, stress-tensor, and spectral claims "
        "open."
        if control_through_q0_3
        else "At least one local control Q0.0--Q0.3 gate failed. The independent "
        "control flags identify which one; no full-Q0 or full-CE+SM "
        "conclusion follows."
    )
    return Q0ManifestGateReport(
        schema_version="1.0",
        scope=TOY_SCOPE,
        control_scope=TOY_SCOPE,
        structural_status=(TOY_CONDITIONAL_PASS if control_through_q0_3 else TOY_CONDITIONAL_FAIL),
        control_q0_0_pass=control_q0_0,
        control_q0_1_pass=control_q0_1,
        control_q0_2_pass=control_q0_2,
        control_q0_3_pass=control_q0_3,
        control_through_q0_3_pass=control_through_q0_3,
        abelian_control_slice_pass=control_through_q0_3,
        full_q0_0_complete=False,
        full_q0_1_complete=False,
        full_q0_2_complete=False,
        full_q0_3_complete=False,
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
        raise ValueError(f"{parent}.{key} must be a non-empty list of unique strings")
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
        action_kind=_string(payload, "action_kind", parent=parent),
        fixed_background_metric=_string(
            payload,
            "fixed_background_metric",
            parent=parent,
        ),
        units=_string(payload, "units", parent=parent),
        action_convention=_string(
            payload,
            "action_convention",
            parent=parent,
        ),
        potential_convention=_string(
            payload,
            "potential_convention",
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
        counterterm_status=_string(
            payload,
            "counterterm_status",
            parent=parent,
        ),
        renormalization_status=_string(
            payload,
            "renormalization_status",
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
        "higgs_self_coupling",
        "higgs_vev",
        "singlet_bare_mass_squared",
        "singlet_self_coupling",
        "lambda_hp",
        "singlet_background",
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
