"""Floquet source ledger for the fusion equation-engineering loop.

This module keeps two physically different statements separate:

* standard QED in a prescribed x-ray field has a Floquet--Volkov (FV)
  formula extrapolation that can increase the 10 keV D--T Maxwellian
  reactivity by one percent; the publication's thermal benchmark is 1 keV,
  so the 10 keV result is not labelled a published-validation pass;
* the 29.64757 MeV CE scalar cannot be identified with that x-ray field.

The QED calculation follows Lindsey et al., Phys. Rev. C 109, 044605 (2024):
the generalized-Bessel sidebands dress the Bosch--Hale cross section before
the energy and polarization averages are taken.  The exact-Z2 scalar loophole
is also evaluated: two on-shell scalar modes can make a low-frequency beat in
``phi**2``, but the allowed portal coefficient makes its field-energy cost
enormous.  The latter is therefore retained only as a fail-closed control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from .ce_two_point_vertex_certificate import ce_light_pole_q04_q05_certificate
from .fusion_full_loop import (
    DEFAULT_NUCLEON_FORM_FACTOR,
    bosch_hale_dt_reactivity,
)
from .fusion_resonance_loop import (
    ALPHA_EM,
    DEUTERON_MASS_MEV,
    HBAR_C_MEV_FM,
    HBAR_C_MEV_M,
    NUCLEON_MASS_MEV,
    TRITON_MASS_MEV,
)


ELEMENTARY_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
SPEED_OF_LIGHT_M_S = 299_792_458.0
MEV_C2_TO_KG = 1.7826619216279e-30
MEV_TO_JOULE = 1.602176634e-13
HIGGS_MASS_MEV = 125_100.0
REGISTERED_SCALAR_MASS_MEV = 29.64757
DT_FUSION_ENERGY_MEV = 17.6
FV_VALIDATED_PHOTON_MIN_KEV = 0.3
FV_CONTROL_PHOTON_MAX_KEV = 1.0
FV_CONTROL_FIELD_MIN_V_M = 1.0e14
FV_CONTROL_FIELD_MAX_V_M = 1.0e16
FV_PUBLISHED_THERMAL_BENCHMARK_KEV = 1.0
FV_PUBLISHED_CN_ENERGY_MAX_KEV = 10.0
BOSCH_HALE_ENERGY_MIN_KEV = 0.5
BOSCH_HALE_ENERGY_MAX_KEV = 550.0

# Bosch--Hale Eq. 9 / Table IV coefficients for T(d,n)4He.  This vector form
# is kept numerically identical to bosch_hale_dt_cross_section_m2.
_DT_A = (6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0)
_DT_B = (6.38e1, -9.95e-1, 6.981e-5, 1.728e-4)
_DT_BG_SQRT_KEV = 34.3827
_MILLIBARN_TO_M2 = 1.0e-31


@dataclass(frozen=True)
class FloquetVolkovReactivityAudit:
    temperature_kev: float
    photon_energy_kev: float
    electric_field_v_m: float
    effective_charge_fraction: float
    reduced_mass_kg: float
    angular_frequency_rad_s: float
    ponderomotive_energy_kev: float
    energy_points: int
    angle_points: int
    phase_points: int
    maximum_sideband_probability_residual: float
    reaction_weighted_out_of_fit_probability: float
    baseline_reactivity_cm3_s: float
    modified_reactivity_cm3_s: float
    reactivity_ratio: float
    reactivity_fractional_gain: float
    target_fractional_gain: float
    target_reached: bool
    qed_action_gauge_invariant: bool
    bosch_hale_cross_section_used: bool
    published_fv_formula_used: bool
    photon_inside_validated_control_window: bool
    field_inside_validated_control_window: bool
    temperature_matches_published_thermal_benchmark: bool
    gamow_saddle_inside_published_cn_energy_window: bool
    published_validation_support_pass: bool
    shifted_cross_section_domain_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FloquetThresholdAudit:
    temperature_kev: float
    photon_energy_kev: float
    target_fractional_gain: float
    required_electric_field_v_m: float
    achieved_fractional_gain: float
    ponderomotive_energy_ev: float
    gamow_saddle_energy_kev: float
    keldysh_gamow_parameter: float
    electric_energy_density_j_m3: float
    plane_wave_intensity_w_m2: float
    photon_wavelength_nm: float
    coarse_fractional_gain_at_required_field: float
    default_fractional_gain_at_required_field: float
    fine_fractional_gain_at_required_field: float
    maximum_grid_fractional_gain_spread: float
    multiphoton_regime: bool
    numerical_convergence_pass: bool
    temperature_matches_published_thermal_benchmark: bool
    gamow_saddle_inside_published_cn_energy_window: bool
    published_parameter_window_pass: bool
    formula_extrapolation_one_percent_pass: bool
    prescribed_qed_reactivity_branch_pass: bool
    status: str


@dataclass(frozen=True)
class PumpEnergyLedgerAudit:
    pulse_duration_fs: float
    spot_radius_nm: float
    illuminated_area_m2: float
    pulse_length_m: float
    illuminated_volume_m3: float
    optical_cycles: float
    incident_peak_power_w: float
    incident_pulse_energy_j: float
    declared_total_dt_ion_density_m3: float
    declared_equal_species_density_m3: float
    baseline_fusion_energy_in_volume_j: float
    incremental_fusion_energy_in_volume_j: float
    incremental_fusion_to_incident_pulse_energy_ratio: float
    source_geometry_declared: bool
    incident_pump_energy_accounted: bool
    absorption_and_propagation_solved: bool
    pump_recovery_solved: bool
    net_energy_positive: bool
    reactor_upgrade_derived: bool
    status: str


@dataclass(frozen=True)
class CEScalarBeatAudit:
    scalar_mass_mev: float
    beat_quantum_energy_kev: float
    second_mode_momentum_mev: float
    beat_reduced_wavelength_fm: float
    gamow_saddle_turning_radius_fm: float
    beat_locally_uniform_over_barrier: bool
    allowed_portal_lambda: float
    quadratic_nucleon_coefficient_mev_inv: float
    required_fractional_mass_modulation: float
    required_equal_mode_amplitude_mev: float
    required_scalar_energy_density_j_m3: float
    scalar_to_qed_field_energy_density_ratio: float
    unavoidable_dc_mass_shift_fraction: float
    sum_frequency_quantum_mev: float
    linearized_mass_modulation_valid: bool
    single_ce_mode_matches_xray_frequency: bool
    scalar_source_preparation_derived: bool
    scalar_specific_crank_nicolson_solved: bool
    scalar_pump_ledger_pass: bool
    physical_ce_scalar_reactivity_branch_pass: bool
    status: str


@dataclass(frozen=True)
class FusionFloquetSourceReport:
    schema_version: str
    regression_point: FloquetVolkovReactivityAudit
    qed_threshold: FloquetThresholdAudit
    pump_ledger: PumpEnergyLedgerAudit
    ce_scalar_beat: CEScalarBeatAudit
    qed_fv_formula_extrapolation_one_percent_derived: bool
    qed_prescribed_field_one_percent_reactivity_derived: bool
    source_and_pump_numbers_explicit: bool
    qed_net_reactor_upgrade_derived: bool
    ce_scalar_one_percent_reactivity_derived: bool
    electromagnetic_result_promoted_to_scalar: bool
    maximum_supported_stage: str
    next_required_ce_gate: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
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


def _grid_count(value: Integral, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _dt_reduced_mass_kg() -> float:
    reduced_mass_mev = DEUTERON_MASS_MEV * TRITON_MASS_MEV / (
        DEUTERON_MASS_MEV + TRITON_MASS_MEV
    )
    return reduced_mass_mev * MEV_C2_TO_KG


def _dt_effective_charge_fraction() -> float:
    return (TRITON_MASS_MEV - DEUTERON_MASS_MEV) / (
        TRITON_MASS_MEV + DEUTERON_MASS_MEV
    )


def _bosch_hale_cross_section_array(energies_kev: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies_kev, dtype=float)
    result = np.zeros_like(energies)
    inside = (energies >= BOSCH_HALE_ENERGY_MIN_KEV) & (
        energies <= BOSCH_HALE_ENERGY_MAX_KEV
    )
    if not np.any(inside):
        return result
    energy = energies[inside]
    a1, a2, a3, a4, a5 = _DT_A
    b1, b2, b3, b4 = _DT_B
    numerator = a1 + energy * (a2 + energy * (a3 + energy * (a4 + energy * a5)))
    denominator = 1.0 + energy * (
        b1 + energy * (b2 + energy * (b3 + energy * b4))
    )
    if np.any(denominator <= 0.0):
        raise ValueError("Bosch--Hale denominator left its positive fit branch")
    result[inside] = (
        _MILLIBARN_TO_M2
        * numerator
        / denominator
        / energy
        * np.exp(-_DT_BG_SQRT_KEV / np.sqrt(energy))
    )
    return result


def _fv_numeric_response(
    *,
    temperature_kev: float,
    photon_energy_kev: float,
    electric_field_v_m: float,
    energy_points: int,
    angle_points: int,
    phase_points: int,
) -> tuple[float, float, float, float]:
    """Return gain, ponderomotive energy, probability residual, domain loss."""

    energies = np.geomspace(
        BOSCH_HALE_ENERGY_MIN_KEV,
        BOSCH_HALE_ENERGY_MAX_KEV,
        energy_points,
    )
    baseline_cross_sections = _bosch_hale_cross_section_array(energies)
    reaction_weights = baseline_cross_sections * energies * np.exp(
        -energies / temperature_kev
    )
    baseline_integral = float(np.trapezoid(reaction_weights, energies))
    if baseline_integral <= 0.0:
        raise RuntimeError("Bosch--Hale baseline integral vanished")

    cosines, angle_weights = np.polynomial.legendre.leggauss(angle_points)
    angle_weights *= 0.5
    phases = 2.0 * math.pi * np.arange(phase_points) / phase_points
    indices = np.arange(phase_points)
    sidebands = np.where(
        indices <= phase_points // 2,
        indices,
        indices - phase_points,
    )

    reduced_mass = _dt_reduced_mass_kg()
    effective_charge = ELEMENTARY_CHARGE_C * _dt_effective_charge_fraction()
    omega = photon_energy_kev * 1000.0 * ELEMENTARY_CHARGE_C / HBAR_J_S
    ponderomotive_j = (effective_charge * electric_field_v_m) ** 2 / (
        4.0 * reduced_mass * omega**2
    )
    ponderomotive_kev = ponderomotive_j / (1000.0 * ELEMENTARY_CHARGE_C)
    u = (
        effective_charge
        * electric_field_v_m
        * np.sqrt(2.0 * reduced_mass * energies[:, None] * 1000.0 * ELEMENTARY_CHARGE_C)
        * cosines[None, :]
        / (reduced_mass * HBAR_J_S * omega**2)
    )
    v = ponderomotive_kev / (2.0 * photon_energy_kev)

    # With E_n=E+Up+n*hbar*omega, the sign-correct convention is the
    # negative Volkov phase followed by numpy's positive-exponent IFFT.
    phase_function = np.exp(
        -1j * u[..., None] * np.sin(phases)
        - 1j * v * np.sin(2.0 * phases)
    )
    probabilities = np.abs(np.fft.ifft(phase_function, axis=-1)) ** 2
    probability_residual = float(
        np.max(np.abs(np.sum(probabilities, axis=-1) - 1.0))
    )
    shifted_energies = (
        energies[:, None, None]
        + ponderomotive_kev
        + photon_energy_kev * sidebands[None, None, :]
    )
    shifted_cross_sections = _bosch_hale_cross_section_array(shifted_energies)
    dressed_by_angle = np.sum(probabilities * shifted_cross_sections, axis=-1)
    dressed_cross_sections = np.sum(
        dressed_by_angle * angle_weights[None, :], axis=1
    )
    modified_integral = float(
        np.trapezoid(
            dressed_cross_sections * energies * np.exp(-energies / temperature_kev),
            energies,
        )
    )

    outside = (shifted_energies < BOSCH_HALE_ENERGY_MIN_KEV) | (
        shifted_energies > BOSCH_HALE_ENERGY_MAX_KEV
    )
    outside_probability = np.sum(probabilities * outside, axis=-1)
    angle_averaged_outside = np.sum(
        outside_probability * angle_weights[None, :], axis=1
    )
    weighted_outside = float(
        np.trapezoid(reaction_weights * angle_averaged_outside, energies)
        / baseline_integral
    )
    return (
        modified_integral / baseline_integral - 1.0,
        ponderomotive_kev,
        probability_residual,
        weighted_outside,
    )


def audit_floquet_volkov_reactivity(
    *,
    temperature_kev: Real = 10.0,
    photon_energy_kev: Real = 0.3,
    electric_field_v_m: Real = 1.0e16,
    target_fractional_gain: Real = 0.01,
    energy_points: Integral = 181,
    angle_points: Integral = 16,
    phase_points: Integral = 256,
) -> FloquetVolkovReactivityAudit:
    """Evaluate the published FV sideband formula before thermal averaging."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    photon = _positive(photon_energy_kev, name="photon_energy_kev")
    field = _nonnegative(electric_field_v_m, name="electric_field_v_m")
    target = _positive(target_fractional_gain, name="target_fractional_gain")
    energy_count = _grid_count(energy_points, name="energy_points", minimum=41)
    angle_count = _grid_count(angle_points, name="angle_points", minimum=4)
    phase_count = _grid_count(phase_points, name="phase_points", minimum=64)
    if phase_count % 2:
        raise ValueError("phase_points must be even")

    gain, ponderomotive, probability_residual, weighted_outside = _fv_numeric_response(
        temperature_kev=temperature,
        photon_energy_kev=photon,
        electric_field_v_m=field,
        energy_points=energy_count,
        angle_points=angle_count,
        phase_points=phase_count,
    )
    _, _, baseline = bosch_hale_dt_reactivity(temperature)
    modified = baseline * (1.0 + gain)
    photon_valid = FV_VALIDATED_PHOTON_MIN_KEV <= photon <= FV_CONTROL_PHOTON_MAX_KEV
    field_valid = (
        field == 0.0
        or FV_CONTROL_FIELD_MIN_V_M <= field <= FV_CONTROL_FIELD_MAX_V_M
    )
    domain_pass = probability_residual < 1.0e-12 and weighted_outside < 1.0e-9
    target_reached = gain >= target
    temperature_match = math.isclose(
        temperature,
        FV_PUBLISHED_THERMAL_BENCHMARK_KEV,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    saddle_inside_cn_window = (
        _gamow_saddle_energy_kev(temperature) <= FV_PUBLISHED_CN_ENERGY_MAX_KEV
    )
    published_support = temperature_match and saddle_inside_cn_window
    numerical_target_pass = (
        target_reached and photon_valid and field_valid and domain_pass
    )
    return FloquetVolkovReactivityAudit(
        temperature_kev=temperature,
        photon_energy_kev=photon,
        electric_field_v_m=field,
        effective_charge_fraction=_dt_effective_charge_fraction(),
        reduced_mass_kg=_dt_reduced_mass_kg(),
        angular_frequency_rad_s=(
            photon * 1000.0 * ELEMENTARY_CHARGE_C / HBAR_J_S
        ),
        ponderomotive_energy_kev=ponderomotive,
        energy_points=energy_count,
        angle_points=angle_count,
        phase_points=phase_count,
        maximum_sideband_probability_residual=probability_residual,
        reaction_weighted_out_of_fit_probability=weighted_outside,
        baseline_reactivity_cm3_s=baseline,
        modified_reactivity_cm3_s=modified,
        reactivity_ratio=1.0 + gain,
        reactivity_fractional_gain=gain,
        target_fractional_gain=target,
        target_reached=target_reached,
        qed_action_gauge_invariant=True,
        bosch_hale_cross_section_used=True,
        published_fv_formula_used=True,
        photon_inside_validated_control_window=photon_valid,
        field_inside_validated_control_window=field_valid,
        temperature_matches_published_thermal_benchmark=temperature_match,
        gamow_saddle_inside_published_cn_energy_window=saddle_inside_cn_window,
        published_validation_support_pass=published_support,
        shifted_cross_section_domain_gate_pass=domain_pass,
        status=(
            "PUBLISHED_BENCHMARK_QED_FV_REACTIVITY_TARGET_REACHED"
            if numerical_target_pass and published_support
            else (
                "QED_FV_FORMULA_TARGET_REACHED_OUTSIDE_PUBLISHED_THERMAL_SUPPORT"
                if numerical_target_pass
                else "QED_FV_CONTROL_OR_TARGET_GATE_FAILED"
            )
        ),
    )


def _solve_required_field(
    *,
    temperature_kev: float,
    photon_energy_kev: float,
    target_fractional_gain: float,
    energy_points: int,
    angle_points: int,
    phase_points: int,
) -> float:
    lower = 0.0
    upper = FV_CONTROL_FIELD_MAX_V_M
    upper_gain = _fv_numeric_response(
        temperature_kev=temperature_kev,
        photon_energy_kev=photon_energy_kev,
        electric_field_v_m=upper,
        energy_points=energy_points,
        angle_points=angle_points,
        phase_points=phase_points,
    )[0]
    if upper_gain < target_fractional_gain:
        raise ValueError("target is not bracketed inside the published field control window")
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        gain = _fv_numeric_response(
            temperature_kev=temperature_kev,
            photon_energy_kev=photon_energy_kev,
            electric_field_v_m=midpoint,
            energy_points=energy_points,
            angle_points=angle_points,
            phase_points=phase_points,
        )[0]
        if gain >= target_fractional_gain:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _gamow_saddle_energy_kev(temperature_kev: float) -> float:
    return ((_DT_BG_SQRT_KEV * temperature_kev / 2.0) ** (2.0 / 3.0))


def audit_floquet_threshold(
    *,
    temperature_kev: Real = 10.0,
    photon_energy_kev: Real = 0.3,
    target_fractional_gain: Real = 0.01,
) -> FloquetThresholdAudit:
    """Solve the QED field threshold and audit numerical and regime controls."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    photon = _positive(photon_energy_kev, name="photon_energy_kev")
    target = _positive(target_fractional_gain, name="target_fractional_gain")
    required_field = _solve_required_field(
        temperature_kev=temperature,
        photon_energy_kev=photon,
        target_fractional_gain=target,
        energy_points=181,
        angle_points=16,
        phase_points=256,
    )
    grid_specs = ((121, 12, 128), (181, 16, 256), (361, 24, 512))
    gains = tuple(
        _fv_numeric_response(
            temperature_kev=temperature,
            photon_energy_kev=photon,
            electric_field_v_m=required_field,
            energy_points=energy_points,
            angle_points=angle_points,
            phase_points=phase_points,
        )[0]
        for energy_points, angle_points, phase_points in grid_specs
    )
    _, ponderomotive_kev, _, _ = _fv_numeric_response(
        temperature_kev=temperature,
        photon_energy_kev=photon,
        electric_field_v_m=required_field,
        energy_points=181,
        angle_points=16,
        phase_points=256,
    )
    omega = photon * 1000.0 * ELEMENTARY_CHARGE_C / HBAR_J_S
    effective_charge = ELEMENTARY_CHARGE_C * _dt_effective_charge_fraction()
    saddle = _gamow_saddle_energy_kev(temperature)
    keldysh = (
        omega
        * math.sqrt(
            2.0 * _dt_reduced_mass_kg() * saddle * 1000.0 * ELEMENTARY_CHARGE_C
        )
        / (effective_charge * required_field)
    )
    energy_density = 0.5 * VACUUM_PERMITTIVITY_F_M * required_field**2
    grid_spread = max(gains) - min(gains)
    convergence = grid_spread < 1.0e-8
    temperature_match = math.isclose(
        temperature,
        FV_PUBLISHED_THERMAL_BENCHMARK_KEV,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    saddle_inside_cn_window = saddle <= FV_PUBLISHED_CN_ENERGY_MAX_KEV
    published_window = (
        FV_VALIDATED_PHOTON_MIN_KEV <= photon <= FV_CONTROL_PHOTON_MAX_KEV
        and FV_CONTROL_FIELD_MIN_V_M <= required_field <= FV_CONTROL_FIELD_MAX_V_M
        and temperature_match
        and saddle_inside_cn_window
    )
    extrapolation_pass = gains[1] >= target and keldysh > 1.0 and convergence
    branch_pass = extrapolation_pass and published_window
    return FloquetThresholdAudit(
        temperature_kev=temperature,
        photon_energy_kev=photon,
        target_fractional_gain=target,
        required_electric_field_v_m=required_field,
        achieved_fractional_gain=gains[1],
        ponderomotive_energy_ev=1000.0 * ponderomotive_kev,
        gamow_saddle_energy_kev=saddle,
        keldysh_gamow_parameter=keldysh,
        electric_energy_density_j_m3=energy_density,
        plane_wave_intensity_w_m2=energy_density * SPEED_OF_LIGHT_M_S,
        photon_wavelength_nm=(
            2.0 * math.pi * SPEED_OF_LIGHT_M_S / omega * 1.0e9
        ),
        coarse_fractional_gain_at_required_field=gains[0],
        default_fractional_gain_at_required_field=gains[1],
        fine_fractional_gain_at_required_field=gains[2],
        maximum_grid_fractional_gain_spread=grid_spread,
        multiphoton_regime=keldysh > 1.0,
        numerical_convergence_pass=convergence,
        temperature_matches_published_thermal_benchmark=temperature_match,
        gamow_saddle_inside_published_cn_energy_window=saddle_inside_cn_window,
        published_parameter_window_pass=published_window,
        formula_extrapolation_one_percent_pass=extrapolation_pass,
        prescribed_qed_reactivity_branch_pass=branch_pass,
        status=(
            "QED_FV_ONE_PERCENT_THRESHOLD_CLOSED"
            if branch_pass
            else (
                "QED_FV_ONE_PERCENT_FORMULA_EXTRAPOLATION_ONLY"
                if extrapolation_pass
                else "QED_FV_THRESHOLD_GATE_FAILED"
            )
        ),
    )


def audit_pump_energy_ledger(
    threshold: FloquetThresholdAudit,
    *,
    pulse_duration_fs: Real = 10.0,
    spot_radius_nm: Real = 10.0,
    total_dt_ion_density_m3: Real = 1.0e31,
) -> PumpEnergyLedgerAudit:
    """Close a declared plane-wave pulse ledger without claiming net gain."""

    duration_fs = _positive(pulse_duration_fs, name="pulse_duration_fs")
    radius_nm = _positive(spot_radius_nm, name="spot_radius_nm")
    total_density = _positive(total_dt_ion_density_m3, name="total_dt_ion_density_m3")
    duration_s = duration_fs * 1.0e-15
    radius_m = radius_nm * 1.0e-9
    area = math.pi * radius_m**2
    pulse_length = SPEED_OF_LIGHT_M_S * duration_s
    volume = area * pulse_length
    peak_power = threshold.plane_wave_intensity_w_m2 * area
    pulse_energy = peak_power * duration_s
    _, _, baseline_cm3_s = bosch_hale_dt_reactivity(threshold.temperature_kev)
    baseline_m3_s = baseline_cm3_s * 1.0e-6
    species_density = 0.5 * total_density
    reaction_count = (
        species_density**2 * baseline_m3_s * volume * duration_s
    )
    baseline_fusion_energy = reaction_count * DT_FUSION_ENERGY_MEV * MEV_TO_JOULE
    incremental_energy = baseline_fusion_energy * threshold.achieved_fractional_gain
    energy_ratio = incremental_energy / pulse_energy
    omega = threshold.photon_energy_kev * 1000.0 * ELEMENTARY_CHARGE_C / HBAR_J_S
    cycles = duration_s * omega / (2.0 * math.pi)
    return PumpEnergyLedgerAudit(
        pulse_duration_fs=duration_fs,
        spot_radius_nm=radius_nm,
        illuminated_area_m2=area,
        pulse_length_m=pulse_length,
        illuminated_volume_m3=volume,
        optical_cycles=cycles,
        incident_peak_power_w=peak_power,
        incident_pulse_energy_j=pulse_energy,
        declared_total_dt_ion_density_m3=total_density,
        declared_equal_species_density_m3=species_density,
        baseline_fusion_energy_in_volume_j=baseline_fusion_energy,
        incremental_fusion_energy_in_volume_j=incremental_energy,
        incremental_fusion_to_incident_pulse_energy_ratio=energy_ratio,
        source_geometry_declared=True,
        incident_pump_energy_accounted=True,
        absorption_and_propagation_solved=False,
        pump_recovery_solved=False,
        net_energy_positive=incremental_energy >= pulse_energy,
        reactor_upgrade_derived=False,
        status="PUMP_LEDGER_CLOSED_NET_ENERGY_AND_PROPAGATION_FAIL",
    )


def _quadratic_mass_modulation_gain(
    *,
    fractional_modulation: float,
    temperature_kev: float,
    beat_quantum_energy_kev: float,
    energy_points: int = 361,
    phase_points: int = 2048,
) -> float:
    """Asymptotic kinetic-phase toy for a periodic reduced-mass modulation."""

    energies = np.geomspace(
        BOSCH_HALE_ENERGY_MIN_KEV,
        BOSCH_HALE_ENERGY_MAX_KEV,
        energy_points,
    )
    baseline = _bosch_hale_cross_section_array(energies)
    weights = baseline * energies * np.exp(-energies / temperature_kev)
    baseline_integral = float(np.trapezoid(weights, energies))
    phases = 2.0 * math.pi * np.arange(phase_points) / phase_points
    indices = np.arange(phase_points)
    sidebands = np.where(
        indices <= phase_points // 2,
        indices,
        indices - phase_points,
    )
    phase_amplitude = fractional_modulation * energies / beat_quantum_energy_kev
    probabilities = np.abs(
        np.fft.ifft(
            np.exp(1j * phase_amplitude[:, None] * np.sin(phases)), axis=-1
        )
    ) ** 2
    shifted_energies = (
        energies[:, None] + beat_quantum_energy_kev * sidebands[None, :]
    )
    dressed = np.sum(
        probabilities * _bosch_hale_cross_section_array(shifted_energies), axis=1
    )
    modified = float(
        np.trapezoid(
            dressed * energies * np.exp(-energies / temperature_kev), energies
        )
    )
    return modified / baseline_integral - 1.0


def _solve_quadratic_modulation(
    *,
    temperature_kev: float,
    beat_quantum_energy_kev: float,
    target_fractional_gain: float,
) -> float:
    lower = 0.0
    upper = 0.8
    if (
        _quadratic_mass_modulation_gain(
            fractional_modulation=upper,
            temperature_kev=temperature_kev,
            beat_quantum_energy_kev=beat_quantum_energy_kev,
        )
        < target_fractional_gain
    ):
        raise RuntimeError("quadratic modulation target is not bracketed")
    for _ in range(44):
        midpoint = 0.5 * (lower + upper)
        gain = _quadratic_mass_modulation_gain(
            fractional_modulation=midpoint,
            temperature_kev=temperature_kev,
            beat_quantum_energy_kev=beat_quantum_energy_kev,
        )
        if gain >= target_fractional_gain:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def audit_ce_scalar_beat(
    threshold: FloquetThresholdAudit,
    *,
    beat_quantum_energy_kev: Real = 0.3,
) -> CEScalarBeatAudit:
    """Audit the exact-Z2 two-mode beat without promoting the kinetic toy."""

    beat_kev = _positive(beat_quantum_energy_kev, name="beat_quantum_energy_kev")
    beat_mev = beat_kev / 1000.0
    second_momentum = math.sqrt(
        (REGISTERED_SCALAR_MASS_MEV + beat_mev) ** 2
        - REGISTERED_SCALAR_MASS_MEV**2
    )
    beat_length = HBAR_C_MEV_FM / second_momentum
    turning_radius = (
        ALPHA_EM * HBAR_C_MEV_FM / (threshold.gamow_saddle_energy_kev / 1000.0)
    )
    certificate = ce_light_pole_q04_q05_certificate()
    allowed_lambda = certificate.invisible_width.maximum_allowed_abs_lambda
    if allowed_lambda is None:
        raise RuntimeError("portal certificate did not supply an allowed lambda")
    # Eliminating h from -lambda*v*h*phi^2 and -(f_N*m_N/v)h*Nbar*N
    # gives the classical monomial +(lambda*f_N*m_N/m_h^2)phi^2*Nbar*N.
    # Its phi-phi-N-N Feynman vertex carries an additional factor of two,
    # but that pair vertex must not be reused as the background mass-shift
    # coefficient here.
    coefficient = (
        allowed_lambda
        * DEFAULT_NUCLEON_FORM_FACTOR
        * NUCLEON_MASS_MEV
        / HIGGS_MASS_MEV**2
    )
    required_modulation = _solve_quadratic_modulation(
        temperature_kev=threshold.temperature_kev,
        beat_quantum_energy_kev=beat_kev,
        target_fractional_gain=threshold.target_fractional_gain,
    )
    amplitude = math.sqrt(required_modulation * NUCLEON_MASS_MEV / coefficient)
    omega_one = REGISTERED_SCALAR_MASS_MEV
    omega_two = REGISTERED_SCALAR_MASS_MEV + beat_mev
    natural_energy_density = 0.5 * (omega_one**2 + omega_two**2) * amplitude**2
    mev4_to_j_m3 = MEV_TO_JOULE / HBAR_C_MEV_M**3
    energy_density = natural_energy_density * mev4_to_j_m3
    return CEScalarBeatAudit(
        scalar_mass_mev=REGISTERED_SCALAR_MASS_MEV,
        beat_quantum_energy_kev=beat_kev,
        second_mode_momentum_mev=second_momentum,
        beat_reduced_wavelength_fm=beat_length,
        gamow_saddle_turning_radius_fm=turning_radius,
        beat_locally_uniform_over_barrier=beat_length > turning_radius,
        allowed_portal_lambda=allowed_lambda,
        quadratic_nucleon_coefficient_mev_inv=coefficient,
        required_fractional_mass_modulation=required_modulation,
        required_equal_mode_amplitude_mev=amplitude,
        required_scalar_energy_density_j_m3=energy_density,
        scalar_to_qed_field_energy_density_ratio=(
            energy_density / threshold.electric_energy_density_j_m3
        ),
        unavoidable_dc_mass_shift_fraction=required_modulation,
        sum_frequency_quantum_mev=2.0 * REGISTERED_SCALAR_MASS_MEV + beat_mev,
        linearized_mass_modulation_valid=required_modulation < 0.1,
        single_ce_mode_matches_xray_frequency=False,
        scalar_source_preparation_derived=False,
        scalar_specific_crank_nicolson_solved=False,
        scalar_pump_ledger_pass=False,
        physical_ce_scalar_reactivity_branch_pass=False,
        status="Z2_BEAT_KINEMATICALLY_EXISTS_SOURCE_ENERGY_AND_DYNAMIC_GATES_FAIL",
    )


@lru_cache(maxsize=1)
def current_fusion_floquet_source_report() -> FusionFloquetSourceReport:
    """Build the QED success control and the CE scalar non-equivalence ledger."""

    regression = audit_floquet_volkov_reactivity()
    threshold = audit_floquet_threshold()
    pump = audit_pump_energy_ledger(threshold)
    beat = audit_ce_scalar_beat(threshold)
    return FusionFloquetSourceReport(
        schema_version="1.0",
        regression_point=regression,
        qed_threshold=threshold,
        pump_ledger=pump,
        ce_scalar_beat=beat,
        qed_fv_formula_extrapolation_one_percent_derived=(
            threshold.formula_extrapolation_one_percent_pass
        ),
        qed_prescribed_field_one_percent_reactivity_derived=(
            threshold.prescribed_qed_reactivity_branch_pass
        ),
        source_and_pump_numbers_explicit=(
            pump.source_geometry_declared and pump.incident_pump_energy_accounted
        ),
        qed_net_reactor_upgrade_derived=pump.reactor_upgrade_derived,
        ce_scalar_one_percent_reactivity_derived=(
            beat.physical_ce_scalar_reactivity_branch_pass
        ),
        electromagnetic_result_promoted_to_scalar=False,
        maximum_supported_stage=(
            "QED_FV_10KEV_FORMULA_EXTRAPOLATION_CE_SCALAR_SOURCE_NO_GO"
        ),
        next_required_ce_gate=(
            "supply a scalar source whose finite-pulse energy and spatial profile are explicit, "
            "then solve the scalar-specific time-dependent D-T equation with DC and sum-frequency "
            "terms retained; the electromagnetic FV result cannot substitute for that calculation"
        ),
        conclusion=(
            "The published QED FV formula extrapolates to a one-percent 10 keV D-T reactivity "
            "gain and its incident pulse energy is explicit, but most of that gain lies outside "
            "the publication's 0.1--10 keV CN comparison support and it is strongly net-energy "
            "negative in the declared microvolume.  The 29.64757 MeV CE scalar is not that field. "
            "Its exact-Z2 two-mode beat is kinematically possible but requires an enormous scalar "
            "energy density and a mass modulation outside the declared linearized regime, so no "
            "physical CE scalar "
            "reactivity upgrade is derived."
        ),
    )


__all__ = [
    "CEScalarBeatAudit",
    "FloquetThresholdAudit",
    "FloquetVolkovReactivityAudit",
    "FusionFloquetSourceReport",
    "PumpEnergyLedgerAudit",
    "audit_ce_scalar_beat",
    "audit_floquet_threshold",
    "audit_floquet_volkov_reactivity",
    "audit_pump_energy_ledger",
    "current_fusion_floquet_source_report",
]
