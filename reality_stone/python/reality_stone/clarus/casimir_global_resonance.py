"""Global asymptotic and wavelength gates for a resonant Casimir throat."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import ELECTRON_VOLT_J, SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class FixedEOSAsymptoticAudit:
    density_tail_power: float
    redshift_log_coefficient: float
    spatial_metric_asymptotically_flat: bool
    finite_redshift_at_infinity: bool
    finite_total_source_energy: bool
    finite_adm_mass_falloff: bool
    all_global_conditions_met: bool
    fixed_casimir_eos_global_no_go: bool


@dataclass(frozen=True)
class EngineeredTailAudit:
    throat_log_density_slope: float
    required_throat_log_density_slope: float
    asymptotic_density_power: float
    asymptotic_redshift_finite: bool
    shape_over_radius_tends_to_zero: bool
    total_source_energy_finite: bool
    standard_finite_mass_asymptotics: bool


@dataclass(frozen=True)
class WavelengthResonanceAudit:
    cavity_separation_m: float
    fundamental_wavelength_m: float
    fundamental_frequency_hz: float
    fundamental_quantum_energy_ev: float
    ce_pole_energy_ev: float
    required_harmonic_ratio: float
    same_as_ce_light_pole: bool
    quality_factor_changes_carrier_frequency: bool
    negative_vacuum_stress_from_driven_resonance_derived: bool


def fixed_casimir_eos_asymptotic_audit(
    *,
    density_tail_power: float,
) -> FixedEOSAsymptoticAudit:
    """Audit a tail ``|rho| ~ r^-n`` with ``p_r=3rho, p_t=-rho``.

    Conservation gives ``Phi ~ (3n/4 - 2) log r``.  Finite redshift therefore
    fixes ``n=8/3``, while finite integrated source energy and standard finite
    ADM mass require ``n>3``.  No fixed-EOS power tail satisfies both.
    """

    power = float(density_tail_power)
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("density_tail_power must be finite and positive")

    redshift_coefficient = 3.0 * power / 4.0 - 2.0
    finite_redshift = math.isclose(redshift_coefficient, 0.0, abs_tol=1e-12)
    spatial_flat = power > 2.0
    finite_energy = power > 3.0
    finite_mass = power > 3.0
    return FixedEOSAsymptoticAudit(
        density_tail_power=power,
        redshift_log_coefficient=redshift_coefficient,
        spatial_metric_asymptotically_flat=spatial_flat,
        finite_redshift_at_infinity=finite_redshift,
        finite_total_source_energy=finite_energy,
        finite_adm_mass_falloff=finite_mass,
        all_global_conditions_met=(
            spatial_flat and finite_redshift and finite_energy and finite_mass
        ),
        fixed_casimir_eos_global_no_go=True,
    )


def engineered_eight_thirds_tail_audit() -> EngineeredTailAudit:
    """Audit a smooth tail matching the throat slope and finite redshift.

    ``f(x)=x^(-8/3) exp[(2/3)(1-1/x)]`` has ``f'(1)/f(1)=-2`` and the
    required ``x^-8/3`` asymptotic behavior.  It gives finite redshift and
    ``b/r -> 0``, but its integrated exotic energy and ADM mass diverge.
    """

    return EngineeredTailAudit(
        throat_log_density_slope=-2.0,
        required_throat_log_density_slope=-2.0,
        asymptotic_density_power=8.0 / 3.0,
        asymptotic_redshift_finite=True,
        shape_over_radius_tends_to_zero=True,
        total_source_energy_finite=False,
        standard_finite_mass_asymptotics=False,
    )


def wavelength_resonance_audit(
    *,
    cavity_separation_m: float,
    ce_pole_energy_mev: float = 29.65,
) -> WavelengthResonanceAudit:
    """Compare the required cavity wavelength with the documented CE pole."""

    separation = float(cavity_separation_m)
    pole_mev = float(ce_pole_energy_mev)
    if not math.isfinite(separation) or separation <= 0.0:
        raise ValueError("cavity_separation_m must be finite and positive")
    if not math.isfinite(pole_mev) or pole_mev <= 0.0:
        raise ValueError("ce_pole_energy_mev must be finite and positive")

    wavelength = 2.0 * separation
    frequency = SPEED_OF_LIGHT_M_S / wavelength
    energy_j = 2.0 * math.pi * HBAR_J_S * frequency
    energy_ev = energy_j / ELECTRON_VOLT_J
    pole_ev = pole_mev * 1e6
    ratio = energy_ev / pole_ev
    return WavelengthResonanceAudit(
        cavity_separation_m=separation,
        fundamental_wavelength_m=wavelength,
        fundamental_frequency_hz=frequency,
        fundamental_quantum_energy_ev=energy_ev,
        ce_pole_energy_ev=pole_ev,
        required_harmonic_ratio=ratio,
        same_as_ce_light_pole=math.isclose(ratio, 1.0, rel_tol=1e-3),
        quality_factor_changes_carrier_frequency=False,
        negative_vacuum_stress_from_driven_resonance_derived=False,
    )
