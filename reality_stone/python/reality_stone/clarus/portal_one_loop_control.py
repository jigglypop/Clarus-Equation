"""One-loop scalar-sector control for the optional ``Z2`` Higgs portal.

This module does not claim a full Standard Model, CE, or pole-mass
calculation.  It evaluates the finite ``MSbar`` pieces of the three scalar
one-loop diagrams that occur in the explicitly selected two-real-scalar
truncation

``L_int = -(g/2) h phi^2 - (lambda_HP/2) h^2 phi^2
         - (lambda_phi/4) phi^4``, ``g=2 lambda_HP v``.

With ``Gamma_R^(2)(s)=s-m_R^2+Sigma_R(s)``, the displayed finite pieces are

``Pi_fin = [lambda_HP A0(m_h^2) + 3 lambda_phi A0(m_phi^2)
            + g^2 B0(s;m_h^2,m_phi^2)]/(16 pi^2)``.

They are regulator/subtraction controls, not an observable mass shift:
counterterms, a renormalization condition, the full gauge/Goldstone sector,
and RG improvement are absent.  Their scale drift is retained precisely to
prevent the finite sum from being promoted to a physical prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Callable

from .ce_two_point_vertex_certificate import DEFAULT_LAMBDA_HP
from .q0_manifest_gate import q0_control_action_definition_sha256


LOOP_NORMALIZATION = 16.0 * math.pi**2
CONTROL_SCOPE = "optional_z2_portal_two_real_scalar_one_loop_msbar_finite_control"
CONTROL_STATUS = "UNRENORMALIZED_ONE_LOOP_SCALAR_TRUNCATION_ONLY"


@dataclass(frozen=True)
class PortalOneLoopControlAudit:
    """Serializable finite one-loop control with physical claims locked off."""

    schema_version: str
    scope: str
    status: str
    action_definition_sha256: str
    inverse_kernel_convention: str
    subtraction_convention: str
    target_mass_gev: float
    target_mass_squared_gev2: float
    higgs_mass_gev: float
    lambda_hp: float
    lambda_phi: float
    higgs_vev_gev: float
    renormalization_scale_gev: float
    cubic_h_phi_phi_coupling_gev: float
    a0_higgs_finite_gev2: float
    a0_singlet_finite_gev2: float
    b0_mixed_finite: float
    b0_mixed_derivative_gev_minus2: float
    higgs_tadpole_finite_gev2: float
    singlet_tadpole_finite_gev2: float
    mixed_bubble_finite_gev2: float
    portal_only_finite_sum_gev2: float
    scalar_truncation_finite_sum_gev2: float
    finite_sum_to_target_mass_squared_ratio: float
    target_mass_squared_to_finite_sum_ratio: float
    bubble_self_energy_derivative: float
    linearized_residue_control: float
    first_mixed_cut_energy_gev: float
    below_first_mixed_cut: bool
    mixed_bubble_imaginary_part_gev2: float
    half_scale_finite_sum_gev2: float
    double_scale_finite_sum_gev2: float
    finite_sum_changes_sign_over_scale_holdout: bool
    scalar_loop_expansion_parameter: float
    perturbative_coupling_control_pass: bool
    raw_finite_piece_small_against_light_target: bool
    counterterm_basis_included: bool
    renormalization_condition_supplied: bool
    full_gauge_goldstone_sector_included: bool
    rg_improvement_included: bool
    renormalized_pole_mass_predicted: bool
    spectral_density_derived: bool
    lsz_particle_derived: bool
    ce_field_identity_derived: bool
    maximum_supported_stage: str
    blockers: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible payload."""

        return asdict(self)


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _adaptive_simpson(
    function: Callable[[float], float],
    *,
    tolerance: float = 1.0e-11,
    maximum_depth: int = 24,
) -> float:
    """Integrate one smooth real Feynman-parameter function on ``[0,1]``."""

    left = 0.0
    right = 1.0
    midpoint = 0.5
    f_left = function(left)
    f_midpoint = function(midpoint)
    f_right = function(right)
    whole = (f_left + 4.0 * f_midpoint + f_right) / 6.0

    def recurse(
        a: float,
        b: float,
        f_a: float,
        f_b: float,
        f_mid: float,
        estimate: float,
        local_tolerance: float,
        depth: int,
    ) -> float:
        midpoint_local = 0.5 * (a + b)
        left_midpoint = 0.5 * (a + midpoint_local)
        right_midpoint = 0.5 * (midpoint_local + b)
        f_left_midpoint = function(left_midpoint)
        f_right_midpoint = function(right_midpoint)
        left_estimate = (midpoint_local - a) * (f_a + 4.0 * f_left_midpoint + f_mid) / 6.0
        right_estimate = (b - midpoint_local) * (f_mid + 4.0 * f_right_midpoint + f_b) / 6.0
        refined = left_estimate + right_estimate
        if depth <= 0 or abs(refined - estimate) <= 15.0 * local_tolerance:
            return refined + (refined - estimate) / 15.0
        return recurse(
            a,
            midpoint_local,
            f_a,
            f_mid,
            f_left_midpoint,
            left_estimate,
            local_tolerance / 2.0,
            depth - 1,
        ) + recurse(
            midpoint_local,
            b,
            f_mid,
            f_b,
            f_right_midpoint,
            right_estimate,
            local_tolerance / 2.0,
            depth - 1,
        )

    return recurse(
        left,
        right,
        f_left,
        f_right,
        f_midpoint,
        whole,
        tolerance,
        maximum_depth,
    )


def a0_msbar_finite(*, mass_squared_gev2: Real, scale_squared_gev2: Real) -> float:
    """Return ``A0_fin=m^2[1-log(m^2/mu^2)]`` in the declared convention."""

    mass_squared = _nonnegative(mass_squared_gev2, name="mass_squared_gev2")
    scale_squared = _positive(scale_squared_gev2, name="scale_squared_gev2")
    if mass_squared == 0.0:
        return 0.0
    return mass_squared * (1.0 - math.log(mass_squared / scale_squared))


def b0_msbar_finite_below_threshold(
    *,
    invariant_gev2: Real,
    first_mass_squared_gev2: Real,
    second_mass_squared_gev2: Real,
    scale_squared_gev2: Real,
) -> float:
    """Return the real finite ``B0`` Feynman-parameter integral below cut."""

    invariant = _nonnegative(invariant_gev2, name="invariant_gev2")
    first_mass_squared = _positive(
        first_mass_squared_gev2,
        name="first_mass_squared_gev2",
    )
    second_mass_squared = _positive(
        second_mass_squared_gev2,
        name="second_mass_squared_gev2",
    )
    scale_squared = _positive(scale_squared_gev2, name="scale_squared_gev2")
    threshold = (math.sqrt(first_mass_squared) + math.sqrt(second_mass_squared)) ** 2
    if invariant >= threshold:
        raise ValueError("invariant_gev2 must lie below the two-particle threshold")

    def integrand(x: float) -> float:
        denominator = (
            x * first_mass_squared + (1.0 - x) * second_mass_squared - x * (1.0 - x) * invariant
        )
        if denominator <= 0.0:
            raise ValueError("Feynman denominator must remain positive below threshold")
        return -math.log(denominator / scale_squared)

    return _adaptive_simpson(integrand)


def b0_derivative_below_threshold(
    *,
    invariant_gev2: Real,
    first_mass_squared_gev2: Real,
    second_mass_squared_gev2: Real,
) -> float:
    """Return ``dB0/ds=int_0^1 x(1-x)/Delta(x,s) dx`` below cut."""

    invariant = _nonnegative(invariant_gev2, name="invariant_gev2")
    first_mass_squared = _positive(
        first_mass_squared_gev2,
        name="first_mass_squared_gev2",
    )
    second_mass_squared = _positive(
        second_mass_squared_gev2,
        name="second_mass_squared_gev2",
    )
    threshold = (math.sqrt(first_mass_squared) + math.sqrt(second_mass_squared)) ** 2
    if invariant >= threshold:
        raise ValueError("invariant_gev2 must lie below the two-particle threshold")

    def integrand(x: float) -> float:
        denominator = (
            x * first_mass_squared + (1.0 - x) * second_mass_squared - x * (1.0 - x) * invariant
        )
        if denominator <= 0.0:
            raise ValueError("Feynman denominator must remain positive below threshold")
        return x * (1.0 - x) / denominator

    return _adaptive_simpson(integrand)


def audit_portal_one_loop_scalar_control(
    *,
    target_mass_gev: Real = 0.02964757,
    lambda_hp: Real = DEFAULT_LAMBDA_HP,
    lambda_phi: Real = 0.1,
    higgs_vev_gev: Real = 246.22,
    higgs_mass_gev: Real = 125.25,
    renormalization_scale_gev: Real = 125.25,
) -> PortalOneLoopControlAudit:
    """Evaluate the finite scalar-only one-loop control at ``s=m_target^2``."""

    target_mass = _positive(target_mass_gev, name="target_mass_gev")
    portal = _nonnegative(lambda_hp, name="lambda_hp")
    self_coupling = _nonnegative(lambda_phi, name="lambda_phi")
    vev = _positive(higgs_vev_gev, name="higgs_vev_gev")
    higgs_mass = _positive(higgs_mass_gev, name="higgs_mass_gev")
    scale = _positive(renormalization_scale_gev, name="renormalization_scale_gev")
    target_squared = target_mass**2
    higgs_squared = higgs_mass**2
    cubic = 2.0 * portal * vev

    def finite_components(local_scale: float) -> tuple[float, float, float, float, float]:
        local_scale_squared = local_scale**2
        a0_higgs = a0_msbar_finite(
            mass_squared_gev2=higgs_squared,
            scale_squared_gev2=local_scale_squared,
        )
        a0_singlet = a0_msbar_finite(
            mass_squared_gev2=target_squared,
            scale_squared_gev2=local_scale_squared,
        )
        b0_mixed = b0_msbar_finite_below_threshold(
            invariant_gev2=target_squared,
            first_mass_squared_gev2=higgs_squared,
            second_mass_squared_gev2=target_squared,
            scale_squared_gev2=local_scale_squared,
        )
        higgs_tadpole = portal * a0_higgs / LOOP_NORMALIZATION
        singlet_tadpole = 3.0 * self_coupling * a0_singlet / LOOP_NORMALIZATION
        mixed_bubble = cubic**2 * b0_mixed / LOOP_NORMALIZATION
        return (
            a0_higgs,
            a0_singlet,
            b0_mixed,
            higgs_tadpole + mixed_bubble,
            higgs_tadpole + singlet_tadpole + mixed_bubble,
        )

    a0_higgs, a0_singlet, b0_mixed, portal_sum, scalar_sum = finite_components(scale)
    higgs_tadpole = portal * a0_higgs / LOOP_NORMALIZATION
    singlet_tadpole = 3.0 * self_coupling * a0_singlet / LOOP_NORMALIZATION
    mixed_bubble = cubic**2 * b0_mixed / LOOP_NORMALIZATION
    b0_derivative = b0_derivative_below_threshold(
        invariant_gev2=target_squared,
        first_mass_squared_gev2=higgs_squared,
        second_mass_squared_gev2=target_squared,
    )
    self_energy_derivative = cubic**2 * b0_derivative / LOOP_NORMALIZATION
    linearized_residue = 1.0 / (1.0 + self_energy_derivative)
    cut_energy = higgs_mass + target_mass
    below_cut = target_mass < cut_energy
    _, _, _, _, half_scale_sum = finite_components(scale / 2.0)
    _, _, _, _, double_scale_sum = finite_components(scale * 2.0)
    changes_sign = (
        min(half_scale_sum, scalar_sum, double_scale_sum)
        < 0.0
        < max(
            half_scale_sum,
            scalar_sum,
            double_scale_sum,
        )
    )
    ratio = abs(scalar_sum) / target_squared
    inverse_ratio = math.inf if scalar_sum == 0.0 else target_squared / abs(scalar_sum)
    expansion_parameter = portal / LOOP_NORMALIZATION

    return PortalOneLoopControlAudit(
        schema_version="1.0",
        scope=CONTROL_SCOPE,
        status=CONTROL_STATUS,
        action_definition_sha256=q0_control_action_definition_sha256(),
        inverse_kernel_convention="Gamma_R^(2)(s)=s-m_R^2+Sigma_R(s)",
        subtraction_convention=(
            "A0_fin=m^2[1-ln(m^2/mu^2)];B0_fin=-int_0^1 dx ln(Delta(x,s)/mu^2)"
        ),
        target_mass_gev=target_mass,
        target_mass_squared_gev2=target_squared,
        higgs_mass_gev=higgs_mass,
        lambda_hp=portal,
        lambda_phi=self_coupling,
        higgs_vev_gev=vev,
        renormalization_scale_gev=scale,
        cubic_h_phi_phi_coupling_gev=cubic,
        a0_higgs_finite_gev2=a0_higgs,
        a0_singlet_finite_gev2=a0_singlet,
        b0_mixed_finite=b0_mixed,
        b0_mixed_derivative_gev_minus2=b0_derivative,
        higgs_tadpole_finite_gev2=higgs_tadpole,
        singlet_tadpole_finite_gev2=singlet_tadpole,
        mixed_bubble_finite_gev2=mixed_bubble,
        portal_only_finite_sum_gev2=portal_sum,
        scalar_truncation_finite_sum_gev2=scalar_sum,
        finite_sum_to_target_mass_squared_ratio=ratio,
        target_mass_squared_to_finite_sum_ratio=inverse_ratio,
        bubble_self_energy_derivative=self_energy_derivative,
        linearized_residue_control=linearized_residue,
        first_mixed_cut_energy_gev=cut_energy,
        below_first_mixed_cut=below_cut,
        mixed_bubble_imaginary_part_gev2=0.0 if below_cut else math.nan,
        half_scale_finite_sum_gev2=half_scale_sum,
        double_scale_finite_sum_gev2=double_scale_sum,
        finite_sum_changes_sign_over_scale_holdout=changes_sign,
        scalar_loop_expansion_parameter=expansion_parameter,
        perturbative_coupling_control_pass=expansion_parameter < 0.01,
        raw_finite_piece_small_against_light_target=ratio < 1.0,
        counterterm_basis_included=False,
        renormalization_condition_supplied=False,
        full_gauge_goldstone_sector_included=False,
        rg_improvement_included=False,
        renormalized_pole_mass_predicted=False,
        spectral_density_derived=False,
        lsz_particle_derived=False,
        ce_field_identity_derived=False,
        maximum_supported_stage="FINITE_ONE_LOOP_SCALAR_TRUNCATION_DIAGNOSTIC",
        blockers=(
            "the scalar finite sum changes with the subtraction scale",
            "no mass or field counterterm basis and no renormalization condition were supplied",
            "the gauge, Goldstone, fermion, and complete Standard Model sectors are absent",
            "the optional portal field is not identified with the CE Hessian readout",
            "no spectral-density positivity or asymptotic-state certificate was supplied",
        ),
        conclusion=(
            "The optional scalar truncation is perturbative in coupling, but its displayed "
            "finite one-loop mass-squared piece is much larger than the 29.65 MeV target "
            "and is strongly subtraction-scale dependent. This diagnoses radiative tuning; "
            "it is not a renormalized pole-mass prediction."
        ),
    )
