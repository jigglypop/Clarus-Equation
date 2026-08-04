"""Reality gates for magnetic-flux charged-fermion wormhole support."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import ELECTRON_VOLT_J, SPEED_OF_LIGHT_M_S


ELECTRON_MASS_ENERGY_EV = 510_998.95


@dataclass(frozen=True)
class ChargedFermionTopologyAudit:
    channel_length_m: float
    effectively_massless_energy_bound_ev: float
    electron_mass_energy_ev: float
    electron_to_massless_bound_ratio: float
    electron_effectively_massless: bool
    standard_model_macroscopic_charged_mode_available: bool
    ce_charged_massless_fermion_specified: bool
    ce_quantized_magnetic_flux_sector_specified: bool
    negative_casimir_stress_derived_in_external_control: bool
    external_control_is_long_wormhole: bool
    ambient_space_shortcut: bool
    human_scale_ce_mapping_pass: bool


@dataclass(frozen=True)
class FluxMultiplicityControl:
    wormhole_length_m: float
    magnetic_radius_m: float
    flux_zero_mode_count: int
    dsnec_scale_lower_bound: float
    scale_bound_satisfied: bool
    exact_integer_flux_action_specified: bool


def charged_fermion_topology_audit(
    *,
    channel_length_m: float = 1.0,
) -> ChargedFermionTopologyAudit:
    """Check whether known charged fermions are effectively massless in a channel."""

    length = float(channel_length_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("channel_length_m must be finite and positive")
    massless_bound = (
        HBAR_J_S * SPEED_OF_LIGHT_M_S / length / ELECTRON_VOLT_J
    )
    ratio = ELECTRON_MASS_ENERGY_EV / massless_bound
    electron_massless = ratio < 1.0
    return ChargedFermionTopologyAudit(
        channel_length_m=length,
        effectively_massless_energy_bound_ev=massless_bound,
        electron_mass_energy_ev=ELECTRON_MASS_ENERGY_EV,
        electron_to_massless_bound_ratio=ratio,
        electron_effectively_massless=electron_massless,
        standard_model_macroscopic_charged_mode_available=False,
        ce_charged_massless_fermion_specified=False,
        ce_quantized_magnetic_flux_sector_specified=False,
        negative_casimir_stress_derived_in_external_control=True,
        external_control_is_long_wormhole=True,
        ambient_space_shortcut=False,
        human_scale_ce_mapping_pass=False,
    )


def flux_multiplicity_control(
    *,
    wormhole_length_m: float,
    magnetic_radius_m: float,
    flux_zero_mode_count: int,
) -> FluxMultiplicityControl:
    """Apply the parametric DSNEC control ``q >= length / magnetic_radius``."""

    length = float(wormhole_length_m)
    radius = float(magnetic_radius_m)
    count = int(flux_zero_mode_count)
    if not all(math.isfinite(value) and value > 0.0 for value in (length, radius)):
        raise ValueError("length scales must be finite and positive")
    if count <= 0:
        raise ValueError("flux_zero_mode_count must be positive")
    lower_bound = length / radius
    return FluxMultiplicityControl(
        wormhole_length_m=length,
        magnetic_radius_m=radius,
        flux_zero_mode_count=count,
        dsnec_scale_lower_bound=lower_bound,
        scale_bound_satisfied=count >= lower_bound,
        exact_integer_flux_action_specified=False,
    )
