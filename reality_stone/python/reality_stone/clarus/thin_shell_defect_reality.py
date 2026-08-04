"""Reality gates for a Schwarzschild cut-and-paste thin-shell wormhole.

The geometry is exact, but the Darmois--Israel junction condition specifies a
surface stress tensor rather than a microscopic material.  This module keeps
that distinction explicit and tests the simplest proposed completion: an
isotropic scale-free (2+1)-dimensional edge QFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


C = 299_792_458.0
G = 6.67430e-11
HBAR = 1.054_571_817e-34
EARTH_MASS_KG = 5.9722e24
PLANCK_LENGTH_M = math.sqrt(G * HBAR / C**3)
ELECTRON_VOLT_J = 1.602_176_634e-19


@dataclass(frozen=True)
class ThinShellDefectAudit:
    radius_m: float
    lapse: float
    schwarzschild_mass_kg: float
    surface_energy_j_m2: float
    tangential_pressure_n_m: float
    surface_nec_j_m2: float
    total_shell_energy_j: float
    shell_mass_earth: float
    pressure_to_abs_energy_ratio: float
    conformal_edge_pressure_n_m: float
    conformal_pressure_residual_n_m: float
    conformal_eos_match: bool
    casimir_coefficient: float
    required_effective_degrees: float
    species_cutoff_m: float
    species_cutoff_to_radius: float
    explicit_defect_action_present: bool
    microscopic_stability_closed: bool

    @property
    def reality_pass(self) -> bool:
        return (
            self.conformal_eos_match
            and self.explicit_defect_action_present
            and self.microscopic_stability_closed
        )


@dataclass(frozen=True)
class BarotropicStabilityAudit:
    lapse: float
    sound_speed_squared: float
    potential_curvature_times_radius_squared: float
    radially_stable: bool
    causal_gradient_stable: bool
    stability_threshold: float | None
    required_inequality: str
    causal_stable_overlap_exists: bool


@dataclass(frozen=True)
class ElasticDefectAudit:
    lapse: float
    bulk_sound_speed_squared: float
    shear_sound_speed_squared: float
    spherical_mode_shear_strain_norm: float
    shear_changes_radial_mode: bool
    radial_potential_curvature_times_radius_squared: float
    radial_mode_stable: bool
    elastic_characteristics_causal: bool
    negative_tension_eos_match: bool
    negative_tension_bending_kinetic_sign: int
    negative_tension_bending_ghost: bool
    minimal_elastic_reality_pass: bool


@dataclass(frozen=True)
class QuantumLayerAudit:
    radius_m: float
    lapse: float
    effective_species: float
    qei_coefficient: float
    required_surface_energy_j_m2: float
    maximum_negative_layer_thickness_m: float
    sampling_time_s: float
    ultraviolet_energy_ev: float
    thickness_to_radius: float
    flat_space_qei_is_direct_boundary_proof: bool
    boundary_completion_required: bool
    one_layer_reality_pass: bool


@dataclass(frozen=True)
class InternalModeStabilityAudit:
    bare_radial_curvature: float
    internal_mode_curvature: float
    mixing: float
    direct_radial_stiffness: float
    relaxed_effective_radial_curvature: float
    passive_mixing_stiffens_radial_mode: bool
    radially_stable_after_relaxation: bool
    minimum_direct_stiffness_required: float
    active_or_nonadiabatic_control_required: bool


@dataclass(frozen=True)
class FloquetRadialControlAudit:
    instability_to_drive_frequency: float
    modulation_over_drive_squared: float
    averaged_curvature: float
    monodromy_trace: float
    monodromy_determinant: float
    exact_floquet_stable: bool
    high_frequency_control_regime: bool
    drive_is_continuously_required: bool
    stable_after_drive_loss: bool
    supplies_static_negative_stress: bool
    control_pass: bool


@dataclass(frozen=True)
class FloquetActuatorAudit:
    radius_m: float
    lapse: float
    radial_potential_curvature_times_radius_squared: float
    displacement_fraction: float
    growth_rate_s_inv: float
    drive_angular_frequency_s_inv: float
    drive_frequency_hz: float
    modulation_coefficient_s_inv2: float
    pressure_stiffness_n_m2: float
    pressure_modulation_n_m: float
    background_pressure_n_m: float
    pressure_modulation_fraction: float
    peak_reactive_mechanical_power_w: float
    drive_loss_efold_s: float
    exact_floquet_control_pass: bool
    actuator_action_specified: bool
    supplies_required_negative_surface_energy: bool
    realization_pass: bool


@dataclass(frozen=True)
class RigidNegativeTensionBraneAudit:
    tension_kinetic_coefficient: float
    rigidity_coefficient: float
    massless_pole_residue: float
    additional_pole_residue: float
    additional_pole_scale_squared: float
    infrared_bending_ghost: bool
    residues_have_opposite_sign: bool
    rigidity_removes_all_ghosts: bool
    minimal_rigid_brane_reality_pass: bool


@dataclass(frozen=True)
class InducedGravityDefectAudit:
    worldvolume_spacetime_dimensions: int
    pure_worldvolume_eh_local_graviton_dof: int
    localized_gravity_coupling_nonnegative: bool
    explicit_modified_junction_solution: bool
    negative_tension_bending_mode_cured: bool
    full_bulk_brane_spectrum_closed: bool
    localized_eh_coefficient_in_ce_action: bool
    bulk_boundary_conditions_specified: bool
    current_reality_pass: bool
    verdict: str


def barotropic_radial_stability(
    lapse: float,
    sound_speed_squared: float,
) -> BarotropicStabilityAudit:
    """Test radial stability for a barotropic Schwarzschild thin shell.

    Conservation on the shell and ``eta = dp/dsigma`` give

      a^2 V'' = 2 eta (1 - 3 f) - (1 + 3 f^2)/(2 f).

    Stability requires ``V'' > 0``.  A conventional local fluid additionally
    needs ``0 <= eta <= 1`` to avoid gradient instability and superluminal
    sound.  Those intervals never overlap for ``0 < f <= 1``.
    """

    if not math.isfinite(lapse) or not 0.0 < lapse <= 1.0:
        raise ValueError("lapse must be finite and lie in (0, 1]")
    if not math.isfinite(sound_speed_squared):
        raise ValueError("sound_speed_squared must be finite")

    curvature = (
        2.0 * sound_speed_squared * (1.0 - 3.0 * lapse)
        - (1.0 + 3.0 * lapse**2) / (2.0 * lapse)
    )
    causal = 0.0 <= sound_speed_squared <= 1.0

    if math.isclose(lapse, 1.0 / 3.0, rel_tol=0.0, abs_tol=1.0e-15):
        threshold = None
        inequality = "no finite eta stabilizes f=1/3"
    else:
        threshold = (1.0 + 3.0 * lapse**2) / (
            4.0 * lapse * (1.0 - 3.0 * lapse)
        )
        operator = ">" if lapse < 1.0 / 3.0 else "<"
        inequality = f"eta {operator} {threshold:.16g}"

    return BarotropicStabilityAudit(
        lapse=lapse,
        sound_speed_squared=sound_speed_squared,
        potential_curvature_times_radius_squared=curvature,
        radially_stable=curvature > 0.0,
        causal_gradient_stable=causal,
        stability_threshold=threshold,
        required_inequality=inequality,
        causal_stable_overlap_exists=False,
    )


def audit_minimal_elastic_defect(
    lapse: float,
    bulk_sound_speed_squared: float = 1.0,
    shear_sound_speed_squared: float = 1.0,
) -> ElasticDefectAudit:
    """Audit the minimal local isotropic elastic completion.

    The spherical ``l=0`` perturbation has purely trace strain, so its shear
    strain is identically zero.  A shear modulus can change nonspherical modes
    but cannot repair the radial stability gate.  The special negative-tension
    Nambu--Goto equation of state ``p=-sigma`` matches the junction background
    only at ``f=1/3``; there the radial curvature is ``-2`` and the negative
    tension also reverses the transverse bending-mode kinetic term.
    """

    for name, value in (
        ("bulk_sound_speed_squared", bulk_sound_speed_squared),
        ("shear_sound_speed_squared", shear_sound_speed_squared),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    radial = barotropic_radial_stability(lapse, bulk_sound_speed_squared)
    causal = (
        0.0 <= bulk_sound_speed_squared <= 1.0
        and 0.0 <= shear_sound_speed_squared <= 1.0
    )
    tension_match = math.isclose(
        lapse,
        1.0 / 3.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )

    return ElasticDefectAudit(
        lapse=lapse,
        bulk_sound_speed_squared=bulk_sound_speed_squared,
        shear_sound_speed_squared=shear_sound_speed_squared,
        spherical_mode_shear_strain_norm=0.0,
        shear_changes_radial_mode=False,
        radial_potential_curvature_times_radius_squared=(
            radial.potential_curvature_times_radius_squared
        ),
        radial_mode_stable=radial.radially_stable,
        elastic_characteristics_causal=causal,
        negative_tension_eos_match=tension_match,
        negative_tension_bending_kinetic_sign=-1 if tension_match else 0,
        negative_tension_bending_ghost=tension_match,
        minimal_elastic_reality_pass=False,
    )


def audit_quantum_negative_layer(
    radius_m: float = 1.0,
    lapse: float = 1.0,
    *,
    effective_species: float = 1.0,
    qei_coefficient: float = 3.0 / (32.0 * math.pi**2),
) -> QuantumLayerAudit:
    """Translate the shell energy into a free-field QEI thickness control.

    A four-dimensional massless-field quantum inequality scales as
    ``|rho| <= K N hbar/(c^3 tau^4)``. Sampling a layer for ``tau=d/c`` and
    integrating across it gives ``|sigma| <= K N hbar c/d^3``. This is a
    flat-space smooth-layer control; a material boundary changes the reference
    state and must be audited together with its own stress tensor.
    """

    for name, value in (
        ("effective_species", effective_species),
        ("qei_coefficient", qei_coefficient),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    shell = audit_static_schwarzschild_thin_shell(radius_m, lapse)
    thickness = (
        qei_coefficient
        * effective_species
        * HBAR
        * C
        / abs(shell.surface_energy_j_m2)
    ) ** (1.0 / 3.0)
    uv_energy = HBAR * C / thickness / ELECTRON_VOLT_J

    return QuantumLayerAudit(
        radius_m=radius_m,
        lapse=lapse,
        effective_species=effective_species,
        qei_coefficient=qei_coefficient,
        required_surface_energy_j_m2=shell.surface_energy_j_m2,
        maximum_negative_layer_thickness_m=thickness,
        sampling_time_s=thickness / C,
        ultraviolet_energy_ev=uv_energy,
        thickness_to_radius=thickness / radius_m,
        flat_space_qei_is_direct_boundary_proof=False,
        boundary_completion_required=True,
        one_layer_reality_pass=False,
    )


def audit_relaxed_internal_mode(
    bare_radial_curvature: float,
    internal_mode_curvature: float,
    mixing: float,
    *,
    direct_radial_stiffness: float = 0.0,
) -> InternalModeStabilityAudit:
    """Integrate out one stable passive internal mode at quadratic order.

    For ``V2 = (Krr+D)x^2/2 + B*x*y + C*y^2/2`` with ``C>0``, relaxation of
    ``y`` gives the Schur complement ``Keff=Krr+D-B^2/C``.  Passive resonance
    mixing alone therefore cannot raise a negative radial eigenvalue.
    """

    values = (
        bare_radial_curvature,
        internal_mode_curvature,
        mixing,
        direct_radial_stiffness,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("internal-mode inputs must be finite")
    if internal_mode_curvature <= 0.0:
        raise ValueError("internal_mode_curvature must be positive")

    mixing_softening = mixing**2 / internal_mode_curvature
    effective = bare_radial_curvature + direct_radial_stiffness - mixing_softening
    minimum_direct = max(0.0, -bare_radial_curvature + mixing_softening)

    return InternalModeStabilityAudit(
        bare_radial_curvature=bare_radial_curvature,
        internal_mode_curvature=internal_mode_curvature,
        mixing=mixing,
        direct_radial_stiffness=direct_radial_stiffness,
        relaxed_effective_radial_curvature=effective,
        passive_mixing_stiffens_radial_mode=False,
        radially_stable_after_relaxation=effective > 0.0,
        minimum_direct_stiffness_required=minimum_direct,
        active_or_nonadiabatic_control_required=(
            bare_radial_curvature < 0.0 and direct_radial_stiffness <= minimum_direct
        ),
    )


def _floquet_monodromy(
    instability_to_drive_frequency: float,
    modulation_over_drive_squared: float,
    steps: int,
) -> tuple[float, float]:
    """Integrate two fundamental solutions over one dimensionless period."""

    state = [1.0, 0.0, 0.0, 1.0]
    step = 2.0 * math.pi / steps

    def derivative(time: float, values: list[float]) -> list[float]:
        coefficient = (
            instability_to_drive_frequency**2
            - modulation_over_drive_squared * math.cos(time)
        )
        return [values[1], coefficient * values[0], values[3], coefficient * values[2]]

    time = 0.0
    for _ in range(steps):
        k1 = derivative(time, state)
        k2_state = [value + 0.5 * step * slope for value, slope in zip(state, k1)]
        k2 = derivative(time + 0.5 * step, k2_state)
        k3_state = [value + 0.5 * step * slope for value, slope in zip(state, k2)]
        k3 = derivative(time + 0.5 * step, k3_state)
        k4_state = [value + step * slope for value, slope in zip(state, k3)]
        k4 = derivative(time + step, k4_state)
        state = [
            value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(state, k1, k2, k3, k4)
        ]
        time += step

    trace = state[0] + state[3]
    determinant = state[0] * state[3] - state[1] * state[2]
    return trace, determinant


def audit_floquet_radial_control(
    instability_to_drive_frequency: float,
    modulation_over_drive_squared: float,
    *,
    integration_steps: int = 4096,
) -> FloquetRadialControlAudit:
    """Test a driven inverted radial mode using its exact monodromy matrix.

    The dimensionless equation is
    ``x'' + [-r^2 + epsilon*cos(tau)] x = 0``.  Averaging predicts curvature
    ``-r^2 + epsilon^2/2``; the monodromy trace performs the actual Floquet
    gate and detects instability tongues missed by averaging.
    """

    values = (instability_to_drive_frequency, modulation_over_drive_squared)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Floquet ratios must be finite and non-negative")
    if integration_steps < 256:
        raise ValueError("integration_steps must be at least 256")

    trace, determinant = _floquet_monodromy(*values, integration_steps)
    exact_stable = abs(trace) < 2.0
    high_frequency = (
        instability_to_drive_frequency <= 0.1
        and modulation_over_drive_squared <= 0.3
    )
    averaged_curvature = (
        -instability_to_drive_frequency**2
        + 0.5 * modulation_over_drive_squared**2
    )

    return FloquetRadialControlAudit(
        instability_to_drive_frequency=instability_to_drive_frequency,
        modulation_over_drive_squared=modulation_over_drive_squared,
        averaged_curvature=averaged_curvature,
        monodromy_trace=trace,
        monodromy_determinant=determinant,
        exact_floquet_stable=exact_stable,
        high_frequency_control_regime=high_frequency,
        drive_is_continuously_required=True,
        stable_after_drive_loss=False,
        supplies_static_negative_stress=False,
        control_pass=exact_stable and high_frequency,
    )


def audit_static_schwarzschild_thin_shell(
    radius_m: float = 1.0,
    lapse: float = 1.0,
    *,
    casimir_coefficient: float = 1.0,
) -> ThinShellDefectAudit:
    """Audit a symmetric static shell at ``r=a`` with ``f(a)=lapse``.

    For two identical Schwarzschild exteriors, the junction equations give

      sigma = -c^4 sqrt(f)/(2 pi G a),
      p     =  c^4 (1+f)/(8 pi G a sqrt(f)).

    A scale-free isotropic 2+1 dimensional QFT has a traceless surface stress,
    hence ``p = sigma/2``.  The comparison is an equation-of-state gate, not a
    claim that every possible defect theory is conformal.
    """

    values = (radius_m, lapse, casimir_coefficient)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("thin-shell inputs must be finite")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if not 0.0 < lapse <= 1.0:
        raise ValueError("lapse must lie in (0, 1]")
    if casimir_coefficient <= 0.0:
        raise ValueError("casimir_coefficient must be positive")

    root_f = math.sqrt(lapse)
    surface_energy = -(C**4) * root_f / (2.0 * math.pi * G * radius_m)
    pressure = (C**4) * (1.0 + lapse) / (
        8.0 * math.pi * G * radius_m * root_f
    )
    surface_nec = surface_energy + pressure
    total_energy = 4.0 * math.pi * radius_m**2 * surface_energy
    conformal_pressure = surface_energy / 2.0

    required_degrees = (
        abs(surface_energy)
        * radius_m**3
        / (casimir_coefficient * HBAR * C)
    )
    species_cutoff = math.sqrt(required_degrees) * PLANCK_LENGTH_M

    mass_geom_m = 0.5 * radius_m * (1.0 - lapse)
    schwarzschild_mass = mass_geom_m * C**2 / G

    return ThinShellDefectAudit(
        radius_m=radius_m,
        lapse=lapse,
        schwarzschild_mass_kg=schwarzschild_mass,
        surface_energy_j_m2=surface_energy,
        tangential_pressure_n_m=pressure,
        surface_nec_j_m2=surface_nec,
        total_shell_energy_j=total_energy,
        shell_mass_earth=abs(total_energy) / C**2 / EARTH_MASS_KG,
        pressure_to_abs_energy_ratio=pressure / abs(surface_energy),
        conformal_edge_pressure_n_m=conformal_pressure,
        conformal_pressure_residual_n_m=pressure - conformal_pressure,
        conformal_eos_match=math.isclose(
            pressure,
            conformal_pressure,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ),
        casimir_coefficient=casimir_coefficient,
        required_effective_degrees=required_degrees,
        species_cutoff_m=species_cutoff,
        species_cutoff_to_radius=species_cutoff / radius_m,
        explicit_defect_action_present=False,
        microscopic_stability_closed=False,
    )


def audit_floquet_junction_actuator(
    radius_m: float = 1.0,
    lapse: float = 1.0 / 3.0,
    *,
    radial_potential_curvature_times_radius_squared: float = -2.0,
    instability_to_drive_frequency: float = 0.05,
    modulation_over_drive_squared: float = 0.1,
    displacement_fraction: float = 1.0e-6,
) -> FloquetActuatorAudit:
    """Map dimensionless Floquet control to a junction-pressure actuator.

    Linearising the dynamic Israel pressure gives an actuator acceleration
    ``delta a_ddot = 4 pi G sqrt(f) delta p / c^2``.  Matching the parametric
    term ``-h cos(Omega t) delta a`` therefore fixes the pressure stiffness.
    The reported power is a peak reactive mechanical bound ``area*dp*v``;
    it is not a claim about cycle-averaged power consumption.
    """

    values = (
        radius_m,
        lapse,
        radial_potential_curvature_times_radius_squared,
        instability_to_drive_frequency,
        modulation_over_drive_squared,
        displacement_fraction,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Floquet actuator inputs must be finite")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if not 0.0 < lapse <= 1.0:
        raise ValueError("lapse must lie in (0, 1]")
    if radial_potential_curvature_times_radius_squared >= 0.0:
        raise ValueError("radial potential curvature must be negative")
    if instability_to_drive_frequency <= 0.0:
        raise ValueError("instability_to_drive_frequency must be positive")
    if modulation_over_drive_squared < 0.0:
        raise ValueError("modulation_over_drive_squared must be non-negative")
    if displacement_fraction <= 0.0:
        raise ValueError("displacement_fraction must be positive")

    control = audit_floquet_radial_control(
        instability_to_drive_frequency,
        modulation_over_drive_squared,
    )
    shell = audit_static_schwarzschild_thin_shell(radius_m, lapse)
    growth_rate = (
        C
        / radius_m
        * math.sqrt(-0.5 * radial_potential_curvature_times_radius_squared)
    )
    drive_angular_frequency = growth_rate / instability_to_drive_frequency
    modulation_coefficient = (
        modulation_over_drive_squared * drive_angular_frequency**2
    )
    pressure_stiffness = (
        C**2 * modulation_coefficient / (4.0 * math.pi * G * math.sqrt(lapse))
    )
    displacement = displacement_fraction * radius_m
    pressure_modulation = pressure_stiffness * displacement
    area = 4.0 * math.pi * radius_m**2
    peak_velocity = drive_angular_frequency * displacement
    peak_reactive_power = area * pressure_modulation * peak_velocity

    return FloquetActuatorAudit(
        radius_m=radius_m,
        lapse=lapse,
        radial_potential_curvature_times_radius_squared=(
            radial_potential_curvature_times_radius_squared
        ),
        displacement_fraction=displacement_fraction,
        growth_rate_s_inv=growth_rate,
        drive_angular_frequency_s_inv=drive_angular_frequency,
        drive_frequency_hz=drive_angular_frequency / (2.0 * math.pi),
        modulation_coefficient_s_inv2=modulation_coefficient,
        pressure_stiffness_n_m2=pressure_stiffness,
        pressure_modulation_n_m=pressure_modulation,
        background_pressure_n_m=shell.tangential_pressure_n_m,
        pressure_modulation_fraction=(
            pressure_modulation / shell.tangential_pressure_n_m
        ),
        peak_reactive_mechanical_power_w=peak_reactive_power,
        drive_loss_efold_s=1.0 / growth_rate,
        exact_floquet_control_pass=control.control_pass,
        actuator_action_specified=False,
        supplies_required_negative_surface_energy=False,
        realization_pass=False,
    )


def audit_rigid_negative_tension_brane(
    tension_kinetic_coefficient: float = -1.0,
    rigidity_coefficient: float = 1.0,
) -> RigidNegativeTensionBraneAudit:
    """Audit whether a local ``K^2`` rigidity term cures negative tension.

    The quadratic inverse propagator is ``P(z)=T*z+alpha*z^2``.  Its inverse
    decomposes as ``1/P=(1/T)/z-(1/T)/(z+T/alpha)``.  For ``T<0`` the original
    massless bending pole keeps negative residue, while the new pole has the
    opposite residue.  Higher derivatives do not turn both poles healthy.
    """

    if not math.isfinite(tension_kinetic_coefficient):
        raise ValueError("tension_kinetic_coefficient must be finite")
    if not math.isfinite(rigidity_coefficient) or rigidity_coefficient == 0.0:
        raise ValueError("rigidity_coefficient must be finite and nonzero")
    if tension_kinetic_coefficient >= 0.0:
        raise ValueError("this audit requires a negative tension coefficient")

    massless_residue = 1.0 / tension_kinetic_coefficient
    additional_residue = -massless_residue
    return RigidNegativeTensionBraneAudit(
        tension_kinetic_coefficient=tension_kinetic_coefficient,
        rigidity_coefficient=rigidity_coefficient,
        massless_pole_residue=massless_residue,
        additional_pole_residue=additional_residue,
        additional_pole_scale_squared=abs(
            tension_kinetic_coefficient / rigidity_coefficient
        ),
        infrared_bending_ghost=massless_residue < 0.0,
        residues_have_opposite_sign=(massless_residue * additional_residue < 0.0),
        rigidity_removes_all_ghosts=False,
        minimal_rigid_brane_reality_pass=False,
    )


def audit_induced_gravity_defect_frontier() -> InducedGravityDefectAudit:
    """Keep a DGP-like shell extension separate from the current CE action.

    A 2+1 dimensional Einstein--Hilbert term has no local graviton by itself.
    Any new kinetic response comes from its mixing with the ambient bulk and
    therefore requires a modified junction solution and the complete coupled
    spectrum.  No such coefficient or boundary problem is specified in CE.
    """

    return InducedGravityDefectAudit(
        worldvolume_spacetime_dimensions=3,
        pure_worldvolume_eh_local_graviton_dof=0,
        localized_gravity_coupling_nonnegative=True,
        explicit_modified_junction_solution=False,
        negative_tension_bending_mode_cured=False,
        full_bulk_brane_spectrum_closed=False,
        localized_eh_coefficient_in_ce_action=False,
        bulk_boundary_conditions_specified=False,
        current_reality_pass=False,
        verdict="EXTERNAL INDUCED-GRAVITY FRONTIER / TARGET SOLUTION ABSENT",
    )
