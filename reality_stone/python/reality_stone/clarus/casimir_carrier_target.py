"""Canonical carrier targets for the inverse-designed Casimir throat.

The repository historically used two nearby but physically different controls:

* a ``b'(r0)=-1`` null-scale control near 169 GeV; and
* the current ``b'(r0)=-1/3`` full throat tensor near 152.93 GeV.

Keeping them in one typed module prevents a legacy wavelength from silently
becoming the default target for resonant-matter calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import (
    ELECTRON_VOLT_J,
    NEWTON_G_M3_KG_S2,
    SPEED_OF_LIGHT_M_S,
)


CURRENT_SHAPE_DERIVATIVE = -1.0 / 3.0
CURRENT_TARGET_LABEL = "bprime_minus_one_third_full_casimir_tensor"
LEGACY_CONTROL_LABEL = "bprime_minus_one_null_scale_control"
DEFAULT_CE_INVERSE_CORRELATION_SCALE_MEV = 29.64757
# Backward-compatible conditional-EFT name.  The value is not a physical pole
# until a separate two-point/residue/LSZ certificate exists.
DEFAULT_CE_POLE_ENERGY_MEV = DEFAULT_CE_INVERSE_CORRELATION_SCALE_MEV
_LEGACY_SEPARATION_AT_ONE_METRE_M = 3.662808556063564e-18


@dataclass(frozen=True)
class CasimirCarrierTarget:
    """A wavelength target with its geometric and spectral provenance."""

    target_definition: str
    is_current_full_tensor_target: bool
    throat_radius_m: float
    shape_derivative: float
    target_rho_over_curvature_scale: float | None
    target_radial_pressure_over_curvature_scale: float | None
    target_tangential_pressure_over_curvature_scale: float | None
    required_density_magnitude_j_m3: float | None
    required_null_magnitude_j_m3: float | None
    separation_m: float
    wavelength_m: float
    frequency_hz: float
    carrier_energy_ev: float
    ce_pole_energy_ev: float
    carrier_to_ce_pole_ratio: float
    nearest_integer_harmonic: int
    nearest_harmonic_energy_ev: float
    nearest_harmonic_detuning_ev: float
    nearest_harmonic_relative_detuning: float
    wavelength_equals_twice_separation_is_planar_mode_choice: bool
    single_mode_determines_casimir_stress: bool
    throat_boundary_eigenmode_derived: bool
    quality_factor_changes_carrier_frequency: bool
    harmonic_vertex_derived: bool


def _finite_positive(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _spectral_fields(
    *,
    wavelength_m: float,
    ce_pole_energy_mev: float,
) -> tuple[float, float, float, float, int, float, float, float]:
    c = SPEED_OF_LIGHT_M_S
    frequency = c / wavelength_m
    carrier_energy = 2.0 * math.pi * HBAR_J_S * frequency / ELECTRON_VOLT_J
    pole_energy = ce_pole_energy_mev * 1.0e6
    ratio = carrier_energy / pole_energy
    nearest_harmonic = max(1, int(round(ratio)))
    nearest_energy = nearest_harmonic * pole_energy
    detuning = carrier_energy - nearest_energy
    relative_detuning = abs(detuning) / carrier_energy
    return (
        frequency,
        carrier_energy,
        pole_energy,
        ratio,
        nearest_harmonic,
        nearest_energy,
        detuning,
        relative_detuning,
    )


def exact_casimir_carrier_target(
    *,
    throat_radius_m: Real = 1.0,
    ce_pole_energy_mev: Real = DEFAULT_CE_POLE_ENERGY_MEV,
) -> CasimirCarrierTarget:
    """Return the current ``b'(r0)=-1/3`` full-tensor carrier target.

    At the throat the normalized target is
    ``(rho, p_r, p_t)/C=(-1/3, -1, +1/3)`` with
    ``C=c^4/(8*pi*G*r0^2)``.  Equating ``|rho|=C/3`` to the ideal
    electromagnetic Casimir density fixes the separation.  ``wavelength=2a``
    then selects the lowest normal mode of an ideal planar cavity; it is not a
    derived eigenmode of the spherical throat and no single mode determines
    the Casimir stress.  This calibrates a formal scale only; it does not derive
    a reflector or a renormalized stress.
    """

    radius = _finite_positive(throat_radius_m, name="throat_radius_m")
    pole_mev = _finite_positive(ce_pole_energy_mev, name="ce_pole_energy_mev")
    c = SPEED_OF_LIGHT_M_S
    curvature_scale = c**4 / (8.0 * math.pi * NEWTON_G_M3_KG_S2 * radius**2)
    density = curvature_scale / 3.0
    null_magnitude = 4.0 * curvature_scale / 3.0
    separation = (math.pi**2 * HBAR_J_S * c / (720.0 * density)) ** 0.25
    wavelength = 2.0 * separation
    spectral = _spectral_fields(
        wavelength_m=wavelength,
        ce_pole_energy_mev=pole_mev,
    )
    return CasimirCarrierTarget(
        target_definition=CURRENT_TARGET_LABEL,
        is_current_full_tensor_target=True,
        throat_radius_m=radius,
        shape_derivative=CURRENT_SHAPE_DERIVATIVE,
        target_rho_over_curvature_scale=-1.0 / 3.0,
        target_radial_pressure_over_curvature_scale=-1.0,
        target_tangential_pressure_over_curvature_scale=1.0 / 3.0,
        required_density_magnitude_j_m3=density,
        required_null_magnitude_j_m3=null_magnitude,
        separation_m=separation,
        wavelength_m=wavelength,
        frequency_hz=spectral[0],
        carrier_energy_ev=spectral[1],
        ce_pole_energy_ev=spectral[2],
        carrier_to_ce_pole_ratio=spectral[3],
        nearest_integer_harmonic=spectral[4],
        nearest_harmonic_energy_ev=spectral[5],
        nearest_harmonic_detuning_ev=spectral[6],
        nearest_harmonic_relative_detuning=spectral[7],
        wavelength_equals_twice_separation_is_planar_mode_choice=True,
        single_mode_determines_casimir_stress=False,
        throat_boundary_eigenmode_derived=False,
        quality_factor_changes_carrier_frequency=False,
        harmonic_vertex_derived=False,
    )


def legacy_bprime_minus_one_null_control(
    *,
    throat_radius_m: Real = 1.0,
    ce_pole_energy_mev: Real = DEFAULT_CE_POLE_ENERGY_MEV,
) -> CasimirCarrierTarget:
    """Return the historical 169 GeV control without promoting it.

    The stored one-metre separation scales as ``sqrt(r0)``.  The control did
    not match the current full ``(-1/3,-1,+1/3)`` throat tensor, so its stress
    fields are intentionally ``None`` rather than copied from the current target.
    """

    radius = _finite_positive(throat_radius_m, name="throat_radius_m")
    pole_mev = _finite_positive(ce_pole_energy_mev, name="ce_pole_energy_mev")
    separation = _LEGACY_SEPARATION_AT_ONE_METRE_M * math.sqrt(radius)
    wavelength = 2.0 * separation
    spectral = _spectral_fields(
        wavelength_m=wavelength,
        ce_pole_energy_mev=pole_mev,
    )
    return CasimirCarrierTarget(
        target_definition=LEGACY_CONTROL_LABEL,
        is_current_full_tensor_target=False,
        throat_radius_m=radius,
        shape_derivative=-1.0,
        target_rho_over_curvature_scale=None,
        target_radial_pressure_over_curvature_scale=None,
        target_tangential_pressure_over_curvature_scale=None,
        required_density_magnitude_j_m3=None,
        required_null_magnitude_j_m3=None,
        separation_m=separation,
        wavelength_m=wavelength,
        frequency_hz=spectral[0],
        carrier_energy_ev=spectral[1],
        ce_pole_energy_ev=spectral[2],
        carrier_to_ce_pole_ratio=spectral[3],
        nearest_integer_harmonic=spectral[4],
        nearest_harmonic_energy_ev=spectral[5],
        nearest_harmonic_detuning_ev=spectral[6],
        nearest_harmonic_relative_detuning=spectral[7],
        wavelength_equals_twice_separation_is_planar_mode_choice=True,
        single_mode_determines_casimir_stress=False,
        throat_boundary_eigenmode_derived=False,
        quality_factor_changes_carrier_frequency=False,
        harmonic_vertex_derived=False,
    )
