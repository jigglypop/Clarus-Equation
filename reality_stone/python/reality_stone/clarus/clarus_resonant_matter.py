"""Loop-engineering gates for Clarus resonance to material production.

This module deliberately separates five statements that were previously easy
to conflate:

1. coherent Clarus modes have a quadratic sum/difference spectrum;
2. a spectral line has enough invariant four-momentum for a daughter pair;
3. a specified finite-pulse toy EFT excites an asymptotic daughter mode;
4. the daughters form a causal material boundary; and
5. the full renormalized net stress supports the target throat.

Only the first two are kinematic identities.  The finite-pulse solver is a
conditional bosonic EFT control.  It does not derive a propagating Clarus pole,
an interaction vertex, a material phase, or a negative stress tensor from CE.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal

from .casimir_carrier_target import (
    DEFAULT_CE_POLE_ENERGY_MEV,
    CasimirCarrierTarget,
    exact_casimir_carrier_target,
)
from .clarus_negative_source_search import HBAR_J_S
from .spatial_folding import ELECTRON_VOLT_J, SPEED_OF_LIGHT_M_S


HBAR_EV_S = HBAR_J_S / ELECTRON_VOLT_J
HC_EV_M = 2.0 * math.pi * HBAR_J_S * SPEED_OF_LIGHT_M_S / ELECTRON_VOLT_J


@dataclass(frozen=True)
class ClarusPumpMode:
    """One real cosine pump mode in a one-axis natural-unit reduction.

    ``energy_ev`` is ``hbar*omega`` and ``axial_momentum_ev`` is ``c*p_z``.
    The field amplitude has mass dimension one, so it is expressed in eV.
    A mode represents ``A*cos(E*t - p*z + phase)`` in natural units.
    """

    energy_ev: float
    axial_momentum_ev: float
    amplitude_ev: float
    phase_rad: float = 0.0
    linewidth_ev: float = 0.0
    physical_pole_derived: bool = False


@dataclass(frozen=True)
class CoherentPumpMode:
    energy_ev: float
    axial_momentum_ev: float
    amplitude_ev: float
    phase_rad: float
    linewidth_ev: float
    source_count: int
    all_source_poles_derived: bool


@dataclass(frozen=True)
class SquaredFieldSpectralLine:
    energy_transfer_ev: float
    axial_momentum_transfer_ev: float
    invariant_mass_squared_ev2: float
    cosine_amplitude_ev2: float
    phase_rad: float
    combined_linewidth_ev: float | None
    combined_linewidth_model_derived: bool
    origins: tuple[str, ...]


@dataclass(frozen=True)
class SquaredFieldSpectrumAudit:
    input_mode_count: int
    coherent_mode_count: int
    coherent_modes: tuple[CoherentPumpMode, ...]
    dc_field_squared_ev2: float
    spectral_lines: tuple[SquaredFieldSpectralLine, ...]
    zero_after_coherent_cancellation: bool
    exact_quadratic_identity_used: bool
    exact_fourier_key_grouping_used: bool
    phase_coherent_aggregation_used: bool
    all_supplied_physical_pole_flags_true: bool
    full_spacetime_normalization_derived: bool


@dataclass(frozen=True)
class SpectralTargetAudit:
    target_energy_ev: float
    target_axial_momentum_ev: float
    energy_linewidth_ev: float
    momentum_tolerance_ev: float
    nearest_line: SquaredFieldSpectralLine | None
    energy_detuning_ev: float | None
    momentum_detuning_ev: float | None
    within_supplied_resolution: bool
    coherent_excitation_conditionally_matched: bool
    particle_production_implied: bool
    negative_stress_implied: bool


@dataclass(frozen=True)
class PairKinematicsAudit:
    line: SquaredFieldSpectralLine
    daughter_mass_ev: float
    positive_energy_transfer: bool
    timelike_or_null_transfer: bool
    invariant_pair_threshold_open: bool
    threshold_within_numerical_tolerance_only: bool
    centre_of_mass_frame_exists: bool
    centre_of_mass_energy_per_daughter_ev: float | None
    centre_of_mass_momentum_per_daughter_ev: float | None
    temporal_frequency_only_would_be_misleading: bool
    particle_production_dynamics_derived: bool


@dataclass(frozen=True)
class StandingWaveTargetAudit:
    target: CasimirCarrierTarget
    clarus_pole_mass_ev: float
    pump_axial_momentum_ev: float
    pump_energy_ev: float
    pump_wavelength_m: float
    static_grating_period_m: float
    target_separation_m: float
    grating_matches_target_separation: bool
    pair_line_total_energy_ev: float
    pump_energy_detuning_from_boundary_carrier_ev: float
    pair_line_detuning_from_twice_boundary_carrier_ev: float
    supplied_pair_linewidth_ev: float | None
    pair_line_within_supplied_linewidth: bool | None
    maximum_rest_mass_per_identical_daughter_ev: float
    pair_line_is_twice_boundary_carrier: bool
    target_carrier_is_not_daughter_mass: bool


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _strict_integral(value: Integral, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_bool(value: bool, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _normal_phase(phase: float) -> float:
    return math.remainder(phase, 2.0 * math.pi)


def _phasor(amplitude: float, phase: float) -> complex:
    angle = _normal_phase(phase)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    snap = 8.0 * math.ulp(1.0)
    if abs(cosine) <= snap:
        cosine = 0.0
    elif abs(abs(cosine) - 1.0) <= snap:
        cosine = math.copysign(1.0, cosine)
    if abs(sine) <= snap:
        sine = 0.0
    elif abs(abs(sine) - 1.0) <= snap:
        sine = math.copysign(1.0, sine)
    return amplitude * complex(cosine, sine)


def _same_number(
    left: float,
    right: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def _same_fourier_key(
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return _same_number(
        left[0],
        right[0],
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    ) and _same_number(
        left[1],
        right[1],
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def _canonical_transfer(
    energy: float,
    momentum: float,
    phase: float,
    *,
    zero_tolerance: float,
) -> tuple[float, float, float]:
    if energy < -zero_tolerance or (abs(energy) <= zero_tolerance and momentum < -zero_tolerance):
        energy = -energy
        momentum = -momentum
        phase = -phase
    if abs(energy) <= zero_tolerance:
        energy = 0.0
    if abs(momentum) <= zero_tolerance:
        momentum = 0.0
    return energy, momentum, _normal_phase(phase)


def _validated_pump_modes(
    modes: Iterable[ClarusPumpMode],
) -> tuple[ClarusPumpMode, ...]:
    supplied = tuple(modes)
    if not supplied:
        raise ValueError("at least one Clarus pump mode is required")
    validated: list[ClarusPumpMode] = []
    for index, mode in enumerate(supplied):
        if not isinstance(mode, ClarusPumpMode):
            raise ValueError("all modes must be ClarusPumpMode instances")
        energy = _finite_positive(mode.energy_ev, name=f"modes[{index}].energy_ev")
        momentum = _finite_real(
            mode.axial_momentum_ev,
            name=f"modes[{index}].axial_momentum_ev",
        )
        amplitude = _finite_positive(
            mode.amplitude_ev,
            name=f"modes[{index}].amplitude_ev",
        )
        phase = _finite_real(mode.phase_rad, name=f"modes[{index}].phase_rad")
        linewidth = _finite_nonnegative(
            mode.linewidth_ev,
            name=f"modes[{index}].linewidth_ev",
        )
        validated.append(
            ClarusPumpMode(
                energy_ev=energy,
                axial_momentum_ev=momentum,
                amplitude_ev=amplitude,
                phase_rad=_normal_phase(phase),
                linewidth_ev=linewidth,
                physical_pole_derived=_strict_bool(
                    mode.physical_pole_derived,
                    name=f"modes[{index}].physical_pole_derived",
                ),
            )
        )
    return tuple(validated)


def _coherently_merge_pumps(
    modes: tuple[ClarusPumpMode, ...],
    *,
    relative_tolerance: float,
    absolute_tolerance_ev: float,
    cancellation_relative_tolerance: float,
) -> tuple[CoherentPumpMode, ...]:
    remaining = list(sorted(modes, key=lambda mode: (mode.energy_ev, mode.axial_momentum_ev)))
    groups: list[list[ClarusPumpMode]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        keep: list[ClarusPumpMode] = []
        for candidate in remaining:
            if _same_fourier_key(
                (seed.energy_ev, seed.axial_momentum_ev),
                (candidate.energy_ev, candidate.axial_momentum_ev),
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance_ev,
            ):
                group.append(candidate)
            else:
                keep.append(candidate)
        groups.append(group)
        remaining = keep

    merged: list[CoherentPumpMode] = []
    for group in groups:
        vector = sum(
            (_phasor(mode.amplitude_ev, mode.phase_rad) for mode in group),
            start=0j,
        )
        raw_amplitude = sum(mode.amplitude_ev for mode in group)
        amplitude = abs(vector)
        if amplitude <= cancellation_relative_tolerance * raw_amplitude:
            continue
        weight = sum(mode.amplitude_ev for mode in group)
        energy = sum(mode.energy_ev * mode.amplitude_ev for mode in group) / weight
        momentum = sum(mode.axial_momentum_ev * mode.amplitude_ev for mode in group) / weight
        merged.append(
            CoherentPumpMode(
                energy_ev=energy,
                axial_momentum_ev=momentum,
                amplitude_ev=amplitude,
                phase_rad=_normal_phase(math.atan2(vector.imag, vector.real)),
                linewidth_ev=max(mode.linewidth_ev for mode in group),
                source_count=len(group),
                all_source_poles_derived=all(mode.physical_pole_derived for mode in group),
            )
        )
    return tuple(sorted(merged, key=lambda mode: (mode.energy_ev, mode.axial_momentum_ev)))


def squared_field_spectrum_audit(
    modes: Iterable[ClarusPumpMode],
    *,
    grouping_relative_tolerance: Real = 0.0,
    grouping_absolute_tolerance_ev: Real = 0.0,
    cancellation_relative_tolerance: Real = 0.0,
) -> SquaredFieldSpectrumAudit:
    """Build the exact phase-aware Fourier lines of ``Phi**2``.

    For real cosine modes, the non-DC lines are ``2K_i``, ``K_i+K_j`` and
    ``K_i-K_j``.  Degenerate lines are combined as complex phasors.  This is
    essential: line amplitudes can cancel even though every individual power
    contribution is positive.
    """

    supplied = _validated_pump_modes(modes)
    rel_tol = _finite_nonnegative(
        grouping_relative_tolerance,
        name="grouping_relative_tolerance",
    )
    abs_tol = _finite_nonnegative(
        grouping_absolute_tolerance_ev,
        name="grouping_absolute_tolerance_ev",
    )
    cancel_tol = _finite_nonnegative(
        cancellation_relative_tolerance,
        name="cancellation_relative_tolerance",
    )
    coherent = _coherently_merge_pumps(
        supplied,
        relative_tolerance=rel_tol,
        absolute_tolerance_ev=abs_tol,
        cancellation_relative_tolerance=cancel_tol,
    )
    dc = math.fsum(mode.amplitude_ev**2 / 2.0 for mode in coherent)

    raw_lines: list[tuple[float, float, complex, float, str]] = []
    for index, mode in enumerate(coherent):
        raw_lines.append(
            (
                2.0 * mode.energy_ev,
                2.0 * mode.axial_momentum_ev,
                _phasor(mode.amplitude_ev**2 / 2.0, 2.0 * mode.phase_rad),
                2.0 * mode.linewidth_ev,
                f"self:{index}",
            )
        )
    for left_index, left in enumerate(coherent):
        for right_index in range(left_index + 1, len(coherent)):
            right = coherent[right_index]
            raw_lines.append(
                (
                    left.energy_ev + right.energy_ev,
                    left.axial_momentum_ev + right.axial_momentum_ev,
                    _phasor(
                        left.amplitude_ev * right.amplitude_ev,
                        left.phase_rad + right.phase_rad,
                    ),
                    left.linewidth_ev + right.linewidth_ev,
                    f"sum:{left_index},{right_index}",
                )
            )
            difference = _canonical_transfer(
                left.energy_ev - right.energy_ev,
                left.axial_momentum_ev - right.axial_momentum_ev,
                left.phase_rad - right.phase_rad,
                zero_tolerance=abs_tol,
            )
            raw_lines.append(
                (
                    difference[0],
                    difference[1],
                    _phasor(
                        left.amplitude_ev * right.amplitude_ev,
                        difference[2],
                    ),
                    left.linewidth_ev + right.linewidth_ev,
                    f"difference:{left_index},{right_index}",
                )
            )

    remaining = list(sorted(raw_lines, key=lambda item: (item[0], item[1])))
    lines: list[SquaredFieldSpectralLine] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        keep: list[tuple[float, float, complex, float, str]] = []
        for candidate in remaining:
            if _same_fourier_key(
                (seed[0], seed[1]),
                (candidate[0], candidate[1]),
                relative_tolerance=rel_tol,
                absolute_tolerance=abs_tol,
            ):
                group.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        vector = sum((item[2] for item in group), start=0j)
        raw_amplitude = sum(abs(item[2]) for item in group)
        amplitude = abs(vector)
        if amplitude <= cancel_tol * raw_amplitude:
            continue
        weight = sum(abs(item[2]) for item in group)
        energy = sum(item[0] * abs(item[2]) for item in group) / weight
        momentum = sum(item[1] * abs(item[2]) for item in group) / weight
        lines.append(
            SquaredFieldSpectralLine(
                energy_transfer_ev=energy,
                axial_momentum_transfer_ev=momentum,
                invariant_mass_squared_ev2=(energy - momentum) * (energy + momentum),
                cosine_amplitude_ev2=amplitude,
                phase_rad=_normal_phase(math.atan2(vector.imag, vector.real)),
                combined_linewidth_ev=None,
                combined_linewidth_model_derived=False,
                origins=tuple(item[4] for item in group),
            )
        )

    return SquaredFieldSpectrumAudit(
        input_mode_count=len(supplied),
        coherent_mode_count=len(coherent),
        coherent_modes=coherent,
        dc_field_squared_ev2=dc,
        spectral_lines=tuple(
            sorted(
                lines, key=lambda line: (line.energy_transfer_ev, line.axial_momentum_transfer_ev)
            )
        ),
        zero_after_coherent_cancellation=not coherent,
        exact_quadratic_identity_used=True,
        exact_fourier_key_grouping_used=(rel_tol == 0.0 and abs_tol == 0.0 and cancel_tol == 0.0),
        phase_coherent_aggregation_used=True,
        all_supplied_physical_pole_flags_true=all(mode.physical_pole_derived for mode in supplied),
        full_spacetime_normalization_derived=False,
    )


def spectral_target_audit(
    spectrum: SquaredFieldSpectrumAudit,
    *,
    target_energy_ev: Real,
    target_axial_momentum_ev: Real = 0.0,
    energy_linewidth_ev: Real,
    momentum_tolerance_ev: Real,
) -> SpectralTargetAudit:
    """Check a supplied spectral target without promoting it to particles."""

    if not isinstance(spectrum, SquaredFieldSpectrumAudit):
        raise ValueError("spectrum must be a SquaredFieldSpectrumAudit")
    energy = _finite_nonnegative(target_energy_ev, name="target_energy_ev")
    momentum = _finite_real(
        target_axial_momentum_ev,
        name="target_axial_momentum_ev",
    )
    linewidth = _finite_nonnegative(energy_linewidth_ev, name="energy_linewidth_ev")
    momentum_tolerance = _finite_nonnegative(
        momentum_tolerance_ev,
        name="momentum_tolerance_ev",
    )
    if not spectrum.spectral_lines:
        return SpectralTargetAudit(
            target_energy_ev=energy,
            target_axial_momentum_ev=momentum,
            energy_linewidth_ev=linewidth,
            momentum_tolerance_ev=momentum_tolerance,
            nearest_line=None,
            energy_detuning_ev=None,
            momentum_detuning_ev=None,
            within_supplied_resolution=False,
            coherent_excitation_conditionally_matched=False,
            particle_production_implied=False,
            negative_stress_implied=False,
        )
    matching_lines = tuple(
        line
        for line in spectrum.spectral_lines
        if abs(line.energy_transfer_ev - energy) <= 0.5 * linewidth
        and abs(line.axial_momentum_transfer_ev - momentum) <= momentum_tolerance
    )
    search_pool = matching_lines or spectrum.spectral_lines
    nearest = min(
        search_pool,
        key=lambda line: math.hypot(
            line.energy_transfer_ev - energy,
            line.axial_momentum_transfer_ev - momentum,
        ),
    )
    energy_detuning = nearest.energy_transfer_ev - energy
    momentum_detuning = nearest.axial_momentum_transfer_ev - momentum
    matched = bool(matching_lines)
    return SpectralTargetAudit(
        target_energy_ev=energy,
        target_axial_momentum_ev=momentum,
        energy_linewidth_ev=linewidth,
        momentum_tolerance_ev=momentum_tolerance,
        nearest_line=nearest,
        energy_detuning_ev=energy_detuning,
        momentum_detuning_ev=momentum_detuning,
        within_supplied_resolution=matched,
        coherent_excitation_conditionally_matched=matched,
        particle_production_implied=False,
        negative_stress_implied=False,
    )


def pair_kinematics_scan(
    spectrum: SquaredFieldSpectrumAudit,
    *,
    daughter_mass_ev: Real,
    threshold_tolerance_ev2: Real = 1.0e-9,
) -> tuple[PairKinematicsAudit, ...]:
    """Apply the invariant ``Q0>0`` and ``Q^2>=4m_chi^2`` pair gate."""

    if not isinstance(spectrum, SquaredFieldSpectrumAudit):
        raise ValueError("spectrum must be a SquaredFieldSpectrumAudit")
    mass = _finite_nonnegative(daughter_mass_ev, name="daughter_mass_ev")
    tolerance = _finite_nonnegative(
        threshold_tolerance_ev2,
        name="threshold_tolerance_ev2",
    )
    audits: list[PairKinematicsAudit] = []
    for line in spectrum.spectral_lines:
        invariant = line.invariant_mass_squared_ev2
        positive_energy = line.energy_transfer_ev > 0.0
        threshold_value = 4.0 * mass**2
        causal = invariant >= 0.0
        threshold = positive_energy and causal and invariant >= threshold_value
        tolerance_only = bool(
            positive_energy
            and not threshold
            and invariant + tolerance >= threshold_value
            and invariant >= -tolerance
        )
        centre_frame = threshold and invariant > 0.0
        if centre_frame:
            centre_energy = math.sqrt(invariant) / 2.0
            centre_momentum = math.sqrt(max(centre_energy**2 - mass**2, 0.0))
        else:
            centre_energy = None
            centre_momentum = None
        temporal_only_misleading = line.energy_transfer_ev >= 2.0 * mass and not threshold
        audits.append(
            PairKinematicsAudit(
                line=line,
                daughter_mass_ev=mass,
                positive_energy_transfer=positive_energy,
                timelike_or_null_transfer=causal,
                invariant_pair_threshold_open=threshold,
                threshold_within_numerical_tolerance_only=tolerance_only,
                centre_of_mass_frame_exists=centre_frame,
                centre_of_mass_energy_per_daughter_ev=centre_energy,
                centre_of_mass_momentum_per_daughter_ev=centre_momentum,
                temporal_frequency_only_would_be_misleading=temporal_only_misleading,
                particle_production_dynamics_derived=False,
            )
        )
    return tuple(audits)


def standing_wave_target_audit(
    *,
    throat_radius_m: Real = 1.0,
    clarus_pole_mass_mev: Real = DEFAULT_CE_POLE_ENERGY_MEV,
    pair_linewidth_ev: Real | None = None,
) -> StandingWaveTargetAudit:
    """Audit a counter-propagating pump that also forms a static grating.

    ``Phi=A*cos(omega*t)*cos(k*z)`` produces a timelike ``2*omega`` line and
    a static ``2*k`` grating.  Choosing the pump wavelength ``2a`` makes the
    grating period exactly ``a``; its pair line is then ``2E_pump`` rather than
    the boundary carrier ``E_*``.
    """

    target = exact_casimir_carrier_target(throat_radius_m=throat_radius_m)
    pole_mass = 1.0e6 * _finite_nonnegative(
        clarus_pole_mass_mev,
        name="clarus_pole_mass_mev",
    )
    pump_momentum = target.carrier_energy_ev
    pump_energy = math.hypot(pump_momentum, pole_mass)
    pump_wavelength = HC_EV_M / pump_momentum
    grating_period = HC_EV_M / (2.0 * pump_momentum)
    pair_energy = 2.0 * pump_energy
    pump_detuning = pole_mass**2 / (pump_energy + pump_momentum) if pole_mass > 0.0 else 0.0
    pair_detuning = 2.0 * pump_detuning
    if pair_linewidth_ev is None:
        supplied_linewidth = None
        within_linewidth = None
    else:
        supplied_linewidth = _finite_nonnegative(
            pair_linewidth_ev,
            name="pair_linewidth_ev",
        )
        within_linewidth = abs(pair_detuning) <= supplied_linewidth / 2.0
    return StandingWaveTargetAudit(
        target=target,
        clarus_pole_mass_ev=pole_mass,
        pump_axial_momentum_ev=pump_momentum,
        pump_energy_ev=pump_energy,
        pump_wavelength_m=pump_wavelength,
        static_grating_period_m=grating_period,
        target_separation_m=target.separation_m,
        grating_matches_target_separation=math.isclose(
            grating_period,
            target.separation_m,
            rel_tol=1.0e-12,
        ),
        pair_line_total_energy_ev=pair_energy,
        pump_energy_detuning_from_boundary_carrier_ev=pump_detuning,
        pair_line_detuning_from_twice_boundary_carrier_ev=pair_detuning,
        supplied_pair_linewidth_ev=supplied_linewidth,
        pair_line_within_supplied_linewidth=within_linewidth,
        maximum_rest_mass_per_identical_daughter_ev=pump_energy,
        pair_line_is_twice_boundary_carrier=(pole_mass == 0.0),
        target_carrier_is_not_daughter_mass=True,
    )


@dataclass(frozen=True)
class BogoliubovModeAudit:
    daughter_statistics: str
    daughter_mass_ev: float
    daughter_momentum_ev: float
    asymptotic_mode_energy_ev: float
    drive_energy_ev: float
    global_pair_phase_space_open: bool
    central_drive_matches_selected_pair_energy: bool
    selected_mode_detuning_ev: float
    leading_order_first_resonance_band_estimate: bool
    floquet_monodromy_trace_n: float
    floquet_monodromy_trace_2n: float
    floquet_monodromy_trace_4n: float
    floquet_trace_refinement_delta: float
    floquet_determinant_residual_4n: float
    numerical_periodic_floquet_instability_resolved: bool
    pulse_fourier_resolution_ev: float
    finite_pulse_off_resonant_excitation_only: bool
    modulation_mass_squared_ev2: float
    dimensionless_mode_frequency: float
    dimensionless_modulation: float
    pulse_cycles: int
    ramp_cycles: int
    smooth_in_out_switching: bool
    instantaneous_frequency_squared_lower_bound_ev2: float
    tachyon_free_certified_by_lower_bound: bool
    tachyonic_during_pulse_derived: bool
    occupation_n: float
    occupation_2n: float
    occupation_4n: float
    no_drive_occupation_4n: float
    refinement_delta_n_2n: float
    refinement_delta_2n_4n: float
    wronskian_residual_4n: float
    occupation_above_no_drive_control: bool
    occupation_numerically_resolved: bool
    conditional_asymptotic_daughter_excitation: bool
    physical_clarus_pole_derived: bool
    action_vertex_derived: bool
    pump_backreaction_solved: bool
    pump_work_energy_accounted: bool
    all_physical_prerequisites_self_reported: bool
    physical_particle_production_derived: bool
    maximum_supported_stage: str


@dataclass(frozen=True)
class CanonicalDaughterStressAudit:
    null_directional_derivatives: tuple[float, ...]
    classical_null_projection: float
    classical_null_projection_nonnegative: bool
    dephased_particle_null_projection_nonnegative: bool
    directly_supplies_negative_throat_source: bool
    occupation_determines_quantum_stress_sign: bool
    anomalous_correlator_and_phase_required_to_infer_from_occupation: bool
    boundary_subtraction_required_for_casimir_route: bool


@dataclass(frozen=True)
class BoundaryResponseAudit:
    state_kind: str
    target_energy_ev: float
    reflectivity_at_target: float
    imaginary_frequency_grid_complete: bool
    transverse_momentum_grid_complete: bool
    polarization_response_complete: bool
    retarded_susceptibility_derived: bool
    kramers_kronig_residual: float
    gain_balance_residual: float
    nonequilibrium_keldysh_stress_derived: bool
    nonzero_target_reflectivity: bool
    single_real_frequency_reflectivity_sufficient: bool
    equilibrium_lifshitz_applicable: bool
    causal_broadband_response_pass: bool
    conditional_response_metadata_gate_pass: bool
    physical_boundary_response_pass: bool


@dataclass(frozen=True)
class BoundaryStressMatchAudit:
    rho_over_curvature_scale: float
    radial_pressure_over_curvature_scale: float
    tangential_pressure_over_curvature_scale: float
    target_rho_over_curvature_scale: float
    target_radial_pressure_over_curvature_scale: float
    target_tangential_pressure_over_curvature_scale: float
    maximum_component_residual: float
    radial_null_projection_over_curvature_scale: float
    component_match: bool
    full_net_stress_includes_boundary_matter_pump_and_vacuum: bool
    renormalized_stress_derived: bool
    conservation_derived: bool
    finite_tail_certified: bool
    physical_affine_anec_negative: bool
    quantum_inequality_pass: bool
    backreaction_solved: bool
    perturbative_stability_pass: bool
    all_realization_prerequisites_self_reported: bool
    throat_realization_pass: bool


@dataclass(frozen=True)
class StageGate:
    name: str
    status: str
    evidence: str


@dataclass(frozen=True)
class ClarusResonantMatterReport:
    schema_version: str
    target: CasimirCarrierTarget
    stages: tuple[StageGate, ...]
    maximum_supported_stage: str
    maximum_conditional_toy_stage: str
    maximum_ce_physical_stage: str
    conditional_toy_excitation: bool
    physical_particle_production_derived: bool
    physical_boundary_derived: bool
    renormalized_negative_stress_derived: bool
    stable_backreacted_throat_derived: bool
    wormhole_realization_derived: bool


def _pulse_envelope(
    phase: float,
    *,
    total_phase: float,
    ramp_phase: float,
) -> float:
    if ramp_phase == 0.0:
        return 1.0
    if phase <= ramp_phase:
        return math.sin(0.5 * math.pi * phase / ramp_phase) ** 2
    if phase >= total_phase - ramp_phase:
        return math.sin(0.5 * math.pi * (total_phase - phase) / ramp_phase) ** 2
    return 1.0


def _integrate_bogoliubov_mode(
    *,
    dimensionless_frequency: float,
    dimensionless_modulation: float,
    phase_offset: float,
    pulse_cycles: int,
    ramp_cycles: int,
    steps_per_cycle: int,
) -> tuple[float, float]:
    """Integrate one normalized complex mode through a finite smooth pulse."""

    total_phase = 2.0 * math.pi * pulse_cycles
    ramp_phase = 2.0 * math.pi * ramp_cycles
    total_steps = pulse_cycles * steps_per_cycle
    step = total_phase / total_steps
    nu = dimensionless_frequency
    mode = complex(1.0 / math.sqrt(2.0 * nu), 0.0)
    velocity = -1j * nu * mode

    def derivative(
        phase: float,
        value: complex,
        slope: complex,
    ) -> tuple[complex, complex]:
        envelope = _pulse_envelope(
            phase,
            total_phase=total_phase,
            ramp_phase=ramp_phase,
        )
        frequency_squared = nu**2 + dimensionless_modulation * envelope * math.cos(
            phase + phase_offset
        )
        return slope, -frequency_squared * value

    phase = 0.0
    for _ in range(total_steps):
        k1_u, k1_v = derivative(phase, mode, velocity)
        k2_u, k2_v = derivative(
            phase + 0.5 * step,
            mode + 0.5 * step * k1_u,
            velocity + 0.5 * step * k1_v,
        )
        k3_u, k3_v = derivative(
            phase + 0.5 * step,
            mode + 0.5 * step * k2_u,
            velocity + 0.5 * step * k2_v,
        )
        k4_u, k4_v = derivative(
            phase + step,
            mode + step * k3_u,
            velocity + step * k3_v,
        )
        mode += step * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u) / 6.0
        velocity += step * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
        phase += step

    raw_occupation = (abs(velocity) ** 2 + nu**2 * abs(mode) ** 2) / (2.0 * nu) - 0.5
    occupation = max(float(raw_occupation), 0.0)
    wronskian = 1j * (mode.conjugate() * velocity - mode * velocity.conjugate())
    wronskian_residual = abs(wronskian - 1.0)
    return occupation, float(wronskian_residual)


def _integrate_periodic_monodromy(
    *,
    dimensionless_frequency: float,
    dimensionless_modulation: float,
    phase_offset: float,
    steps_per_cycle: int,
) -> tuple[float, float]:
    """Integrate the full-amplitude periodic oscillator over one cycle."""

    step = 2.0 * math.pi / steps_per_cycle
    state = (1.0, 0.0, 0.0, 1.0)

    def derivative(
        phase: float,
        values: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        first_u, first_v, second_u, second_v = values
        frequency_squared = dimensionless_frequency**2 + dimensionless_modulation * math.cos(
            phase + phase_offset
        )
        return (
            first_v,
            -frequency_squared * first_u,
            second_v,
            -frequency_squared * second_u,
        )

    def shifted(
        values: tuple[float, float, float, float],
        slope: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return tuple(
            value + scale * gradient for value, gradient in zip(values, slope, strict=True)
        )  # type: ignore[return-value]

    phase = 0.0
    for _ in range(steps_per_cycle):
        k1 = derivative(phase, state)
        k2 = derivative(phase + 0.5 * step, shifted(state, k1, 0.5 * step))
        k3 = derivative(phase + 0.5 * step, shifted(state, k2, 0.5 * step))
        k4 = derivative(phase + step, shifted(state, k3, step))
        state = tuple(
            value + step * (slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4) / 6.0
            for value, slope_1, slope_2, slope_3, slope_4 in zip(
                state,
                k1,
                k2,
                k3,
                k4,
                strict=True,
            )
        )  # type: ignore[assignment]
        phase += step

    first_u, first_v, second_u, second_v = state
    trace = first_u + second_v
    determinant = first_u * second_v - second_u * first_v
    return trace, abs(determinant - 1.0)


def finite_pulse_bogoliubov_audit(
    *,
    daughter_mass_ev: Real,
    daughter_momentum_ev: Real,
    drive_energy_ev: Real,
    modulation_mass_squared_ev2: Real,
    pulse_cycles: Integral = 8,
    ramp_cycles: Integral = 2,
    phase_offset_rad: Real = 0.0,
    integration_steps_per_cycle: Integral = 256,
    convergence_relative_tolerance: Real = 1.0e-5,
    convergence_absolute_tolerance: Real = 1.0e-12,
    wronskian_tolerance: Real = 1.0e-8,
    physical_clarus_pole_derived: bool = False,
    action_vertex_derived: bool = False,
    pump_backreaction_solved: bool = False,
    pump_work_energy_accounted: bool = False,
    daughter_statistics: Literal["boson"] = "boson",
) -> BogoliubovModeAudit:
    """Evolve a bosonic in-vacuum through a smooth finite mass modulation.

    The dimensionless equation is

    ``u'' + [nu**2 + q*s(tau)*cos(tau+phase)] u = 0``.

    The pulse is zero at both ends when ``ramp_cycles>=1``, so the in/out
    particle basis is defined.  N, 2N, and 4N integration plus a no-drive control are
    recorded.  The cosine term is a generic mean-subtracted mass modulation;
    mapping it to ``Phi**2`` also requires the associated DC mass shift.
    This remains a prescribed-pump toy EFT: daughter energy comes from external
    pump work until the coupled backreaction problem is solved.
    """

    if daughter_statistics != "boson":
        raise ValueError("finite_pulse_bogoliubov_audit currently supports bosons only")
    mass = _finite_nonnegative(daughter_mass_ev, name="daughter_mass_ev")
    momentum = _finite_nonnegative(daughter_momentum_ev, name="daughter_momentum_ev")
    drive = _finite_positive(drive_energy_ev, name="drive_energy_ev")
    modulation = _finite_real(
        modulation_mass_squared_ev2,
        name="modulation_mass_squared_ev2",
    )
    phase_offset = _finite_real(phase_offset_rad, name="phase_offset_rad")
    cycles = _strict_integral(pulse_cycles, name="pulse_cycles", minimum=1)
    if isinstance(ramp_cycles, bool) or not isinstance(ramp_cycles, Integral):
        raise ValueError("ramp_cycles must be an integer")
    ramps = int(ramp_cycles)
    if ramps < 0 or 2 * ramps > cycles:
        raise ValueError("ramp_cycles must be nonnegative and at most half the pulse")
    steps = _strict_integral(
        integration_steps_per_cycle,
        name="integration_steps_per_cycle",
        minimum=64,
    )
    convergence_tolerance = _finite_positive(
        convergence_relative_tolerance,
        name="convergence_relative_tolerance",
    )
    convergence_absolute = _finite_nonnegative(
        convergence_absolute_tolerance,
        name="convergence_absolute_tolerance",
    )
    wronskian_limit = _finite_positive(
        wronskian_tolerance,
        name="wronskian_tolerance",
    )
    pole_claim = _strict_bool(
        physical_clarus_pole_derived,
        name="physical_clarus_pole_derived",
    )
    vertex_claim = _strict_bool(action_vertex_derived, name="action_vertex_derived")
    backreaction_claim = _strict_bool(
        pump_backreaction_solved,
        name="pump_backreaction_solved",
    )
    work_claim = _strict_bool(
        pump_work_energy_accounted,
        name="pump_work_energy_accounted",
    )

    mode_energy = math.hypot(mass, momentum)
    if mode_energy <= 0.0:
        raise ValueError("the daughter mode energy must be positive")
    nu = mode_energy / drive
    q_value = modulation / drive**2
    frequency_squared_lower_bound = mode_energy**2 - abs(modulation)
    tachyon_free_certified = frequency_squared_lower_bound > 0.0
    smooth = ramps >= 1

    occupation_n, _ = _integrate_bogoliubov_mode(
        dimensionless_frequency=nu,
        dimensionless_modulation=q_value,
        phase_offset=phase_offset,
        pulse_cycles=cycles,
        ramp_cycles=ramps,
        steps_per_cycle=steps,
    )
    occupation_2n, _ = _integrate_bogoliubov_mode(
        dimensionless_frequency=nu,
        dimensionless_modulation=q_value,
        phase_offset=phase_offset,
        pulse_cycles=cycles,
        ramp_cycles=ramps,
        steps_per_cycle=2 * steps,
    )
    occupation_4n, wronskian_residual = _integrate_bogoliubov_mode(
        dimensionless_frequency=nu,
        dimensionless_modulation=q_value,
        phase_offset=phase_offset,
        pulse_cycles=cycles,
        ramp_cycles=ramps,
        steps_per_cycle=4 * steps,
    )
    no_drive, _ = _integrate_bogoliubov_mode(
        dimensionless_frequency=nu,
        dimensionless_modulation=0.0,
        phase_offset=phase_offset,
        pulse_cycles=cycles,
        ramp_cycles=ramps,
        steps_per_cycle=4 * steps,
    )
    floquet_trace_n, _ = _integrate_periodic_monodromy(
        dimensionless_frequency=nu,
        dimensionless_modulation=q_value,
        phase_offset=phase_offset,
        steps_per_cycle=steps,
    )
    floquet_trace_2n, _ = _integrate_periodic_monodromy(
        dimensionless_frequency=nu,
        dimensionless_modulation=q_value,
        phase_offset=phase_offset,
        steps_per_cycle=2 * steps,
    )
    floquet_trace_4n, floquet_determinant_residual = _integrate_periodic_monodromy(
        dimensionless_frequency=nu,
        dimensionless_modulation=q_value,
        phase_offset=phase_offset,
        steps_per_cycle=4 * steps,
    )

    delta_n_2n = abs(occupation_n - occupation_2n)
    delta_2n_4n = abs(occupation_2n - occupation_4n)
    scale = max(occupation_2n, occupation_4n)
    converged = (
        delta_2n_4n <= convergence_absolute + convergence_tolerance * scale
        and wronskian_residual <= wronskian_limit
    )
    numerical_floor = max(
        10.0 * convergence_absolute,
        10.0 * no_drive,
        5.0 * delta_2n_4n,
    )
    above_control = occupation_4n > numerical_floor
    global_pair_phase_space_open = drive >= 2.0 * mass
    selected_detuning = drive - 2.0 * mode_energy
    central_match = math.isclose(
        drive,
        2.0 * mode_energy,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )
    leading_band_estimate = abs(4.0 * mode_energy**2 - drive**2) <= 2.0 * abs(modulation)
    floquet_trace_delta = abs(floquet_trace_2n - floquet_trace_4n)
    floquet_instability_margin = abs(floquet_trace_4n) - 2.0
    floquet_instability_resolved = (
        floquet_instability_margin > max(10.0 * floquet_trace_delta, convergence_absolute)
        and floquet_determinant_residual <= wronskian_limit
    )
    pulse_fourier_resolution = drive / cycles
    numerically_excited = smooth and tachyon_free_certified and converged and above_control
    off_resonant_only = numerically_excited and not floquet_instability_resolved
    conditional = (
        global_pair_phase_space_open
        and floquet_instability_resolved
        and smooth
        and tachyon_free_certified
        and converged
        and above_control
    )
    prerequisites_self_reported = all((pole_claim, vertex_claim, backreaction_claim, work_claim))
    physical = False
    if conditional:
        maximum_stage = "CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION"
    elif not tachyon_free_certified:
        maximum_stage = "TACHYON_STATUS_UNRESOLVED_BY_CONSERVATIVE_BOUND"
    elif off_resonant_only:
        maximum_stage = "FINITE_PULSE_OFF_RESONANT_EXCITATION_ONLY"
    else:
        maximum_stage = "FINITE_PULSE_MODE_CONTROL_ONLY"
    return BogoliubovModeAudit(
        daughter_statistics=daughter_statistics,
        daughter_mass_ev=mass,
        daughter_momentum_ev=momentum,
        asymptotic_mode_energy_ev=mode_energy,
        drive_energy_ev=drive,
        global_pair_phase_space_open=global_pair_phase_space_open,
        central_drive_matches_selected_pair_energy=central_match,
        selected_mode_detuning_ev=selected_detuning,
        leading_order_first_resonance_band_estimate=leading_band_estimate,
        floquet_monodromy_trace_n=floquet_trace_n,
        floquet_monodromy_trace_2n=floquet_trace_2n,
        floquet_monodromy_trace_4n=floquet_trace_4n,
        floquet_trace_refinement_delta=floquet_trace_delta,
        floquet_determinant_residual_4n=floquet_determinant_residual,
        numerical_periodic_floquet_instability_resolved=(floquet_instability_resolved),
        pulse_fourier_resolution_ev=pulse_fourier_resolution,
        finite_pulse_off_resonant_excitation_only=off_resonant_only,
        modulation_mass_squared_ev2=modulation,
        dimensionless_mode_frequency=nu,
        dimensionless_modulation=q_value,
        pulse_cycles=cycles,
        ramp_cycles=ramps,
        smooth_in_out_switching=smooth,
        instantaneous_frequency_squared_lower_bound_ev2=frequency_squared_lower_bound,
        tachyon_free_certified_by_lower_bound=tachyon_free_certified,
        tachyonic_during_pulse_derived=False,
        occupation_n=occupation_n,
        occupation_2n=occupation_2n,
        occupation_4n=occupation_4n,
        no_drive_occupation_4n=no_drive,
        refinement_delta_n_2n=delta_n_2n,
        refinement_delta_2n_4n=delta_2n_4n,
        wronskian_residual_4n=wronskian_residual,
        occupation_above_no_drive_control=above_control,
        occupation_numerically_resolved=converged,
        conditional_asymptotic_daughter_excitation=conditional,
        physical_clarus_pole_derived=pole_claim,
        action_vertex_derived=vertex_claim,
        pump_backreaction_solved=backreaction_claim,
        pump_work_energy_accounted=work_claim,
        all_physical_prerequisites_self_reported=prerequisites_self_reported,
        physical_particle_production_derived=physical,
        maximum_supported_stage=maximum_stage,
    )


def canonical_daughter_stress_audit(
    null_directional_derivatives: Iterable[Real],
) -> CanonicalDaughterStressAudit:
    """Prove that canonical produced matter is not a direct negative source."""

    derivatives = tuple(
        _finite_real(value, name=f"null_directional_derivatives[{index}]")
        for index, value in enumerate(null_directional_derivatives)
    )
    if not derivatives:
        raise ValueError("at least one null directional derivative is required")
    projection = math.fsum(value**2 for value in derivatives)
    return CanonicalDaughterStressAudit(
        null_directional_derivatives=derivatives,
        classical_null_projection=projection,
        classical_null_projection_nonnegative=projection >= 0.0,
        dephased_particle_null_projection_nonnegative=True,
        directly_supplies_negative_throat_source=False,
        occupation_determines_quantum_stress_sign=False,
        anomalous_correlator_and_phase_required_to_infer_from_occupation=True,
        boundary_subtraction_required_for_casimir_route=True,
    )


def boundary_response_audit(
    *,
    state_kind: Literal["passive_equilibrium", "active_driven"],
    target_energy_ev: Real,
    reflectivity_at_target: Real,
    imaginary_frequency_grid_complete: bool = False,
    transverse_momentum_grid_complete: bool = False,
    polarization_response_complete: bool = False,
    retarded_susceptibility_derived: bool = False,
    kramers_kronig_residual: Real = 1.0,
    gain_balance_residual: Real = 1.0,
    nonequilibrium_keldysh_stress_derived: bool = False,
    residual_tolerance: Real = 1.0e-8,
) -> BoundaryResponseAudit:
    """Gate a material response before any Casimir-stress promotion.

    A single real-frequency reflectivity is never sufficient.  Passive
    equilibrium boundaries require causal TE/TM response on the imaginary
    frequency and transverse-momentum domain.  A continuously pumped boundary
    is non-equilibrium and additionally requires gain/noise balance and a
    Keldysh stress calculation; equilibrium Lifshitz formulas are then locked.
    """

    if state_kind not in ("passive_equilibrium", "active_driven"):
        raise ValueError("state_kind must be passive_equilibrium or active_driven")
    energy = _finite_positive(target_energy_ev, name="target_energy_ev")
    reflectivity = _finite_nonnegative(
        reflectivity_at_target,
        name="reflectivity_at_target",
    )
    if state_kind == "passive_equilibrium" and reflectivity > 1.0:
        raise ValueError("passive-equilibrium reflectivity cannot exceed one")
    kk_residual = _finite_nonnegative(
        kramers_kronig_residual,
        name="kramers_kronig_residual",
    )
    gain_residual = _finite_nonnegative(
        gain_balance_residual,
        name="gain_balance_residual",
    )
    tolerance = _finite_positive(residual_tolerance, name="residual_tolerance")
    imaginary_claim = _strict_bool(
        imaginary_frequency_grid_complete,
        name="imaginary_frequency_grid_complete",
    )
    transverse_claim = _strict_bool(
        transverse_momentum_grid_complete,
        name="transverse_momentum_grid_complete",
    )
    polarization_claim = _strict_bool(
        polarization_response_complete,
        name="polarization_response_complete",
    )
    susceptibility_claim = _strict_bool(
        retarded_susceptibility_derived,
        name="retarded_susceptibility_derived",
    )
    keldysh_claim = _strict_bool(
        nonequilibrium_keldysh_stress_derived,
        name="nonequilibrium_keldysh_stress_derived",
    )
    metadata_gate = (
        imaginary_claim
        and transverse_claim
        and polarization_claim
        and susceptibility_claim
        and kk_residual <= tolerance
    )
    nonzero_target_reflectivity = reflectivity > 0.0
    if state_kind == "active_driven":
        metadata_gate = metadata_gate and gain_residual <= tolerance and keldysh_claim
    conditional_metadata_gate = metadata_gate and nonzero_target_reflectivity
    causal = False
    equilibrium_applicable = False
    physical_pass = False
    return BoundaryResponseAudit(
        state_kind=state_kind,
        target_energy_ev=energy,
        reflectivity_at_target=reflectivity,
        imaginary_frequency_grid_complete=imaginary_claim,
        transverse_momentum_grid_complete=transverse_claim,
        polarization_response_complete=polarization_claim,
        retarded_susceptibility_derived=susceptibility_claim,
        kramers_kronig_residual=kk_residual,
        gain_balance_residual=gain_residual,
        nonequilibrium_keldysh_stress_derived=keldysh_claim,
        nonzero_target_reflectivity=nonzero_target_reflectivity,
        single_real_frequency_reflectivity_sufficient=False,
        equilibrium_lifshitz_applicable=equilibrium_applicable,
        causal_broadband_response_pass=causal,
        conditional_response_metadata_gate_pass=conditional_metadata_gate,
        physical_boundary_response_pass=physical_pass,
    )


def boundary_stress_match_audit(
    *,
    rho_over_curvature_scale: Real,
    radial_pressure_over_curvature_scale: Real,
    tangential_pressure_over_curvature_scale: Real,
    throat_radius_m: Real = 1.0,
    component_tolerance: Real = 1.0e-8,
    full_net_stress_includes_boundary_matter_pump_and_vacuum: bool = False,
    renormalized_stress_derived: bool = False,
    conservation_derived: bool = False,
    finite_tail_certified: bool = False,
    physical_affine_anec_negative: bool = False,
    quantum_inequality_pass: bool = False,
    backreaction_solved: bool = False,
    perturbative_stability_pass: bool = False,
) -> BoundaryStressMatchAudit:
    """Compare a supplied full stress to the current throat target.

    Matching three numbers is only the component gate.  A realization pass is
    impossible unless provenance, conservation, tail, ANEC, quantum-inequality,
    backreaction, and perturbative-stability gates are also supplied.
    """

    rho = _finite_real(rho_over_curvature_scale, name="rho_over_curvature_scale")
    radial = _finite_real(
        radial_pressure_over_curvature_scale,
        name="radial_pressure_over_curvature_scale",
    )
    tangential = _finite_real(
        tangential_pressure_over_curvature_scale,
        name="tangential_pressure_over_curvature_scale",
    )
    tolerance = _finite_positive(component_tolerance, name="component_tolerance")
    target = exact_casimir_carrier_target(throat_radius_m=throat_radius_m)
    residuals = (
        abs(rho - target.target_rho_over_curvature_scale),
        abs(radial - target.target_radial_pressure_over_curvature_scale),
        abs(tangential - target.target_tangential_pressure_over_curvature_scale),
    )
    maximum_residual = max(residuals)
    component_match = maximum_residual <= tolerance
    provenance = _strict_bool(
        full_net_stress_includes_boundary_matter_pump_and_vacuum,
        name="full_net_stress_includes_boundary_matter_pump_and_vacuum",
    )
    renormalized_claim = _strict_bool(
        renormalized_stress_derived,
        name="renormalized_stress_derived",
    )
    conservation_claim = _strict_bool(
        conservation_derived,
        name="conservation_derived",
    )
    finite_tail_claim = _strict_bool(
        finite_tail_certified,
        name="finite_tail_certified",
    )
    anec_claim = _strict_bool(
        physical_affine_anec_negative,
        name="physical_affine_anec_negative",
    )
    qi_claim = _strict_bool(quantum_inequality_pass, name="quantum_inequality_pass")
    backreaction_claim = _strict_bool(
        backreaction_solved,
        name="backreaction_solved",
    )
    stability_claim = _strict_bool(
        perturbative_stability_pass,
        name="perturbative_stability_pass",
    )
    prerequisites_self_reported = all(
        (
            component_match,
            provenance,
            renormalized_claim,
            conservation_claim,
            finite_tail_claim,
            anec_claim,
            qi_claim,
            backreaction_claim,
            stability_claim,
        )
    )
    realization_pass = False
    return BoundaryStressMatchAudit(
        rho_over_curvature_scale=rho,
        radial_pressure_over_curvature_scale=radial,
        tangential_pressure_over_curvature_scale=tangential,
        target_rho_over_curvature_scale=target.target_rho_over_curvature_scale,
        target_radial_pressure_over_curvature_scale=(
            target.target_radial_pressure_over_curvature_scale
        ),
        target_tangential_pressure_over_curvature_scale=(
            target.target_tangential_pressure_over_curvature_scale
        ),
        maximum_component_residual=maximum_residual,
        radial_null_projection_over_curvature_scale=rho + radial,
        component_match=component_match,
        full_net_stress_includes_boundary_matter_pump_and_vacuum=provenance,
        renormalized_stress_derived=renormalized_claim,
        conservation_derived=conservation_claim,
        finite_tail_certified=finite_tail_claim,
        physical_affine_anec_negative=anec_claim,
        quantum_inequality_pass=qi_claim,
        backreaction_solved=backreaction_claim,
        perturbative_stability_pass=stability_claim,
        all_realization_prerequisites_self_reported=prerequisites_self_reported,
        throat_realization_pass=realization_pass,
    )


def current_clarus_resonant_matter_report(
    *,
    throat_radius_m: Real = 1.0,
    spectrum: SquaredFieldSpectrumAudit | None = None,
    pair_channels: tuple[PairKinematicsAudit, ...] = (),
    bogoliubov: BogoliubovModeAudit | None = None,
    boundary_response: BoundaryResponseAudit | None = None,
    stress_match: BoundaryStressMatchAudit | None = None,
) -> ClarusResonantMatterReport:
    """Assemble a monotone report that cannot skip unresolved bridge stages."""

    target = exact_casimir_carrier_target(throat_radius_m=throat_radius_m)
    stages: list[StageGate] = [
        StageGate(
            name="TARGET_CALIBRATED",
            status="PASS",
            evidence=(
                "current b'=-1/3 target fixes a; lambda=2a then defines a "
                "formal ideal-planar 152.93 GeV scale"
            ),
        ),
        StageGate(
            name="PHYSICAL_CLARUS_POLE",
            status="OPEN",
            evidence="29.64757 MeV remains an inverse-correlation bridge, not a derived LSZ pole",
        ),
        StageGate(
            name="NONLINEAR_PRODUCTION_VERTEX",
            status="OPEN",
            evidence="the Z2-compatible Phi^2 chi^2 toy vertex is not derived from a complete CE action",
        ),
    ]

    spectrum_nonzero = bool(
        spectrum is not None
        and not spectrum.zero_after_coherent_cancellation
        and spectrum.spectral_lines
    )
    if spectrum is None:
        stages.append(
            StageGate(
                name="PHASE_AWARE_FOURIER_OVERLAP",
                status="NOT_REACHED",
                evidence="no pump spectrum supplied",
            )
        )
    elif not spectrum_nonzero:
        stages.append(
            StageGate(
                name="PHASE_AWARE_FOURIER_OVERLAP",
                status="NULL_CONTROL",
                evidence="the supplied coherent field cancels or has no non-DC quadratic line",
            )
        )
    else:
        stages.append(
            StageGate(
                name="PHASE_AWARE_FOURIER_OVERLAP",
                status="CONDITIONAL",
                evidence="exact Phi^2 identity evaluated for supplied classical pump modes",
            )
        )

    open_pair_channel = bool(
        spectrum is not None
        and spectrum_nonzero
        and spectrum.exact_fourier_key_grouping_used
        and any(
            channel.invariant_pair_threshold_open and channel.line in spectrum.spectral_lines
            for channel in pair_channels
        )
    )
    stages.append(
        StageGate(
            name="INVARIANT_PAIR_KINEMATICS",
            status="CONDITIONAL" if open_pair_channel else "NOT_REACHED",
            evidence=(
                "at least one supplied Q satisfies Q^2>=4m_chi^2"
                if open_pair_channel
                else "no supplied physical four-momentum channel clears threshold"
            ),
        )
    )

    linked_pair_channel = bool(
        open_pair_channel
        and bogoliubov is not None
        and any(
            channel.invariant_pair_threshold_open
            and channel.line in spectrum.spectral_lines
            and math.isclose(
                channel.daughter_mass_ev,
                bogoliubov.daughter_mass_ev,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                channel.line.energy_transfer_ev,
                bogoliubov.drive_energy_ev,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                channel.line.axial_momentum_transfer_ev,
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and channel.centre_of_mass_frame_exists
            and channel.centre_of_mass_momentum_per_daughter_ev is not None
            and math.isclose(
                channel.centre_of_mass_momentum_per_daughter_ev,
                bogoliubov.daughter_momentum_ev,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                channel.line.energy_transfer_ev,
                2.0 * bogoliubov.asymptotic_mode_energy_ev,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for channel in pair_channels
        )
    )
    toy_excitation = bool(
        linked_pair_channel
        and bogoliubov is not None
        and bogoliubov.conditional_asymptotic_daughter_excitation
    )
    physical_particles = False
    stages.append(
        StageGate(
            name="BOGOLIUBOV_PARTICLE_PRODUCTION",
            status="CONDITIONAL" if toy_excitation else "NOT_REACHED",
            evidence=(
                bogoliubov.maximum_supported_stage
                if toy_excitation and bogoliubov is not None
                else "the oscillator audit is not linked to the same exact "
                "spectrum line, zero-transfer momentum, and daughter mass"
                if bogoliubov is not None
                else "no smooth finite-pulse in/out audit supplied"
            ),
        )
    )

    physical_boundary = False
    stages.append(
        StageGate(
            name="CAUSAL_BROADBAND_BOUNDARY_RESPONSE",
            status="PASS" if physical_boundary else "NOT_REACHED",
            evidence=(
                "single-frequency reflectivity never substitutes for the Lifshitz/Keldysh response integral"
            ),
        )
    )

    stress_derived = False
    stable_throat = False
    stages.extend(
        (
            StageGate(
                name="RENORMALIZED_NEGATIVE_NET_STRESS",
                status="PASS" if stress_derived else "NOT_REACHED",
                evidence="vacuum, produced matter, pump, and apparatus stress must all be included",
            ),
            StageGate(
                name="BACKREACTED_STABLE_THROAT",
                status="PASS" if stable_throat else "NOT_REACHED",
                evidence="requires conserved finite-tail stress, ANEC/QI, backreaction, and perturbations",
            ),
        )
    )

    if stable_throat:
        maximum_stage = "BACKREACTED_STABLE_THROAT_CONTROL"
    elif stress_derived:
        maximum_stage = "RENORMALIZED_STRESS_CONTROL"
    elif physical_boundary:
        maximum_stage = "PHYSICAL_BOUNDARY_RESPONSE_CONTROL"
    elif physical_particles:
        maximum_stage = "PHYSICAL_PARTICLE_PRODUCTION_CONTROL"
    elif toy_excitation:
        maximum_stage = "CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION"
    elif spectrum_nonzero:
        maximum_stage = "CONDITIONAL_NONLINEAR_SPECTRUM"
    elif spectrum is not None:
        maximum_stage = "COHERENT_FIELD_CANCELLATION_NULL_CONTROL"
    else:
        maximum_stage = "KINEMATIC_CORRELATION_ANSATZ"

    return ClarusResonantMatterReport(
        schema_version="1.1",
        target=target,
        stages=tuple(stages),
        maximum_supported_stage=maximum_stage,
        maximum_conditional_toy_stage=maximum_stage,
        maximum_ce_physical_stage="TARGET_SCALE_CALIBRATION_ONLY",
        conditional_toy_excitation=toy_excitation,
        physical_particle_production_derived=physical_particles,
        physical_boundary_derived=physical_boundary,
        renormalized_negative_stress_derived=stress_derived,
        stable_backreacted_throat_derived=stable_throat,
        wormhole_realization_derived=False,
    )
