"""Identifiability gates from a resonance correlation length to null stress."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class PoleFamilyCountermodel:
    requested_stress_exponent: float
    correlation_length_gain: float
    residue_gain: float
    dimensional_stress_proxy_gain: float


@dataclass(frozen=True)
class StressIdentifiabilityAudit:
    resonance_q: float
    stress_mass_dimension: float
    countermodels: tuple[PoleFamilyCountermodel, ...]
    all_countermodels_have_same_correlation_length: bool
    stress_scaling_unique_from_correlation_length: bool
    pole_residue_required: bool
    spectral_density_required: bool
    renormalization_required: bool
    physical_null_stress_derived: bool


@dataclass(frozen=True)
class CEResonanceBridgeAudit:
    correlation_length_ansatz_documented: bool
    correlation_length_scaling_derived: bool
    isolated_positive_pole_derived: bool
    pole_residue_scaling_derived: bool
    reflection_positivity_and_lsz_passed: bool
    spectral_density_derived: bool
    renormalized_stress_tensor_derived: bool
    metric_backreaction_solved: bool
    maximum_supported_stage: str


def pole_family_countermodel_audit(
    *,
    resonance_q: float,
    requested_stress_exponents: Iterable[float] = (0.0, 1.0, 2.0),
    stress_mass_dimension: float = 4.0,
) -> StressIdentifiabilityAudit:
    """Construct equal-correlation-length pole families with arbitrary scaling.

    For ``xi(Q)=Q*xi(1)``, a dimensional proxy of the form ``Z/xi**d``
    scales as ``Z(Q)/Q**d``.  Choosing ``Z(Q)=Q**(d+p)`` realizes any requested
    proxy exponent ``p`` without changing the correlation length.  The proxy is
    not asserted to be a physical stress tensor; it is a counterexample showing
    that the correlation-length exponent alone cannot identify stress scaling.
    """

    q_value = float(resonance_q)
    dimension = float(stress_mass_dimension)
    exponents = tuple(float(value) for value in requested_stress_exponents)
    if not math.isfinite(q_value) or q_value <= 1.0:
        raise ValueError("resonance_q must be finite and greater than one")
    if not math.isfinite(dimension) or dimension <= 0.0:
        raise ValueError("stress_mass_dimension must be finite and positive")
    if not exponents or not all(math.isfinite(value) for value in exponents):
        raise ValueError("requested_stress_exponents must be finite and non-empty")

    countermodels = tuple(
        PoleFamilyCountermodel(
            requested_stress_exponent=exponent,
            correlation_length_gain=q_value,
            residue_gain=q_value ** (dimension + exponent),
            dimensional_stress_proxy_gain=q_value**exponent,
        )
        for exponent in exponents
    )
    return StressIdentifiabilityAudit(
        resonance_q=q_value,
        stress_mass_dimension=dimension,
        countermodels=countermodels,
        all_countermodels_have_same_correlation_length=True,
        # Asking the constructor for only one exponent does not make that
        # exponent identifiable.  The construction above works for every
        # finite exponent, so correlation length alone never fixes the stress
        # scaling within this countermodel family.
        stress_scaling_unique_from_correlation_length=False,
        pole_residue_required=True,
        spectral_density_required=True,
        renormalization_required=True,
        physical_null_stress_derived=False,
    )


def ce_resonance_bridge_audit() -> CEResonanceBridgeAudit:
    """Encode the currently documented CE bridge state without promoting it."""

    return CEResonanceBridgeAudit(
        correlation_length_ansatz_documented=True,
        correlation_length_scaling_derived=False,
        isolated_positive_pole_derived=False,
        pole_residue_scaling_derived=False,
        reflection_positivity_and_lsz_passed=False,
        spectral_density_derived=False,
        renormalized_stress_tensor_derived=False,
        metric_backreaction_solved=False,
        maximum_supported_stage="KINEMATIC_CORRELATION_ANSATZ",
    )
