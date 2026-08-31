"""Finite material-lattice cost certificate in 3+1 natural units.

The candidate uses three compact scalar phases ``Theta^I`` on a supplied
flat, isotropic background, ``Theta^I = q x^I``.  Their winding and proper
spacing are inputs, not predictions of an action.  This module keeps the
free rod stress ledger distinct from the batteries: a one-dimensional guide
can contain ``N`` successive batteries while a finite three-dimensional cube
contains ``N**3`` cells.  Neither capacity is added to the rod energy.

The candidate rod sector is ``-f_X**2/2 sum_I (grad Theta^I)^2``.  It can be
supplemented by static dimensionless material profiles ``W_cell`` and
``W_perp`` multiplied by supplied mass-squared coefficients.  The exact
onsite normalization audited here is
``g |H|**2 (D* B + B* D) + lambda/4 S**2``, where
``S=|H|**2+|D|**2+|B|**2``.  Stored well coefficients audit dimensions only;
they do not prove periodic localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CovariantMaterialLatticeCostCertificate:
    """A bounded bookkeeping receipt, not a lattice dynamics solution."""

    cells_per_axis: int
    proper_cell_spacing: float
    rod_scale: float
    battery_energy_per_cell: float
    carrier_mass: float
    carrier_momentum: float
    onsite_exchange_coupling: float
    quartic_coupling: float
    guide_well_mass_squared: float
    cell_well_mass_squared: float
    wave_number: float
    cube_side_length: float
    material_gram_diagonal: tuple[float, float, float]
    material_gram_determinant: float
    normalized_gram_determinant: float
    proper_cell_volume: float
    winding_per_axis: float
    rod_energy_density: float
    rod_pressure: float
    rod_equation_of_state: float
    finite_rod_energy: float
    guide_all_success_battery_count: int
    guide_battery_capacity: float
    full_volume_cell_count: int
    full_volume_battery_capacity: float
    carrier_frequency: float
    carrier_group_velocity: float
    quartic_lower_bound_coefficient: float
    extremal_quartic_potential: float
    quartic_saturation_residual: float
    dimensionless_core_arguments: tuple[tuple[str, str], ...]
    action_terms_have_mass_dimension_four: bool
    compact_phase_period_is_two_pi: bool
    clock_field_used: bool
    diffeomorphism_covariant_scalar_candidate_by_construction: bool
    supplied_finite_free_rod_background_bookkeeping: bool
    invariant_gram_and_nondegenerate_conditional_geometry: bool
    finite_rod_receipt: bool
    rod_and_battery_ledgers_kept_separate: bool
    dimension_closure: bool
    static_common_coupling_without_coordinate_time_schedule: bool
    canonical_fixed_background_classical_principal_symbol: bool
    fixed_background_classical_domain_of_dependence: bool
    spacing_action_winding_derived: bool
    interacting_backreacted_theta_solution_derived: bool
    background_stability_or_caustic_freedom_derived: bool
    periodic_well_localized_modes_derived: bool
    action_to_projected_rates_or_resonance_derived: bool
    scattering_energy_transfer_receipt_derived: bool
    durable_record_or_selection_derived: bool
    repeated_cptp_fresh_ancilla_derived: bool
    band_or_front_speed_derived: bool
    qft_microcausality_or_no_signalling_derived: bool
    coupled_gr_source_derived: bool
    infinite_isolated_lattice_finite_total_energy_derived: bool
    gates_five_to_eight_derived: bool


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def certify_covariant_material_lattice_cost(
    *,
    cells_per_axis: int,
    proper_cell_spacing: float,
    rod_scale: float,
    battery_energy_per_cell: float,
    carrier_mass: float,
    carrier_momentum: float,
    onsite_exchange_coupling: float,
    quartic_coupling: float,
    guide_well_mass_squared: float = 0.0,
    cell_well_mass_squared: float = 0.0,
) -> CovariantMaterialLatticeCostCertificate:
    """Certify supplied finite-lattice costs and explicit claim boundaries.

    ``Theta^I`` are dimensionless compact phases of period ``2 pi`` and
    ``f_X`` has mass dimension one.  Thus ``f_X**2 (d Theta)**2`` has mass
    dimension four.  ``g`` and ``lambda`` are dimensionless while both well
    coefficients have mass dimension two.  No clock field or time schedule is
    present.
    """

    if isinstance(cells_per_axis, bool) or not isinstance(cells_per_axis, int):
        raise ValueError("cells_per_axis must be an integer")
    if cells_per_axis < 1:
        raise ValueError("cells_per_axis must be at least one")
    a = _positive(proper_cell_spacing, "proper_cell_spacing")
    f_x = _positive(rod_scale, "rod_scale")
    e_b = _positive(battery_energy_per_cell, "battery_energy_per_cell")
    mass = _positive(carrier_mass, "carrier_mass")
    momentum = _nonnegative(carrier_momentum, "carrier_momentum")
    g = float(onsite_exchange_coupling)
    lam = _positive(quartic_coupling, "quartic_coupling")
    if not math.isfinite(g) or g == 0.0:
        raise ValueError("onsite_exchange_coupling must be finite and non-zero")
    guide_well = _nonnegative(guide_well_mass_squared, "guide_well_mass_squared")
    cell_well = _nonnegative(cell_well_mass_squared, "cell_well_mass_squared")
    if lam < abs(g):
        raise ValueError("quartic stability requires quartic_coupling >= abs(g)")

    q = 2.0 * math.pi / a
    length = cells_per_axis * a
    gram_diagonal = (q * q, q * q, q * q)
    determinant = q**6
    normalized_determinant = determinant / q**6
    cell_volume = (2.0 * math.pi) ** 3 / math.sqrt(determinant)
    winding = q * length / (2.0 * math.pi)

    rho = 1.5 * f_x * f_x * q * q
    pressure = -0.5 * f_x * f_x * q * q
    rod_energy = rho * length**3
    guide_count = cells_per_axis
    volume_count = cells_per_axis**3
    omega = math.sqrt(momentum * momentum + mass * mass)
    velocity = momentum / omega
    lower_coefficient = (lam - abs(g)) / 4.0

    # H=1/sqrt(2), D=1/2, B=-sign(g)/2 gives S=1.  Therefore the
    # stabilizer is lambda/4 and g |H|^2 (D*B+B*D) is -|g|/4.
    h_extremal = 1.0 / math.sqrt(2.0)
    d_extremal = 0.5
    b_extremal = -math.copysign(0.5, g)
    extremal_s = h_extremal**2 + d_extremal**2 + b_extremal**2
    extremal_stabilizer = 0.25 * lam * extremal_s**2
    extremal_interaction = g * h_extremal**2 * (
        d_extremal * b_extremal + b_extremal * d_extremal
    )
    extremal_potential = extremal_stabilizer + extremal_interaction
    saturation_residual = abs(extremal_potential - lower_coefficient)

    return CovariantMaterialLatticeCostCertificate(
        cells_per_axis=cells_per_axis,
        proper_cell_spacing=a,
        rod_scale=f_x,
        battery_energy_per_cell=e_b,
        carrier_mass=mass,
        carrier_momentum=momentum,
        onsite_exchange_coupling=g,
        quartic_coupling=lam,
        guide_well_mass_squared=guide_well,
        cell_well_mass_squared=cell_well,
        wave_number=q,
        cube_side_length=length,
        material_gram_diagonal=gram_diagonal,
        material_gram_determinant=determinant,
        normalized_gram_determinant=normalized_determinant,
        proper_cell_volume=cell_volume,
        winding_per_axis=winding,
        rod_energy_density=rho,
        rod_pressure=pressure,
        rod_equation_of_state=pressure / rho,
        finite_rod_energy=rod_energy,
        guide_all_success_battery_count=guide_count,
        guide_battery_capacity=guide_count * e_b,
        full_volume_cell_count=volume_count,
        full_volume_battery_capacity=volume_count * e_b,
        carrier_frequency=omega,
        carrier_group_velocity=velocity,
        quartic_lower_bound_coefficient=lower_coefficient,
        extremal_quartic_potential=extremal_potential,
        quartic_saturation_residual=saturation_residual,
        dimensionless_core_arguments=(
            ("q a = 2 pi", "compact phase winding"),
            ("q L / (2 pi) = N", "supplied integer winding"),
            ("v_g = |k| / sqrt(|k|^2 + m_H^2)", "free-particle sample"),
        ),
        action_terms_have_mass_dimension_four=True,
        compact_phase_period_is_two_pi=True,
        clock_field_used=False,
        diffeomorphism_covariant_scalar_candidate_by_construction=True,
        supplied_finite_free_rod_background_bookkeeping=True,
        invariant_gram_and_nondegenerate_conditional_geometry=True,
        finite_rod_receipt=True,
        rod_and_battery_ledgers_kept_separate=True,
        dimension_closure=True,
        static_common_coupling_without_coordinate_time_schedule=True,
        canonical_fixed_background_classical_principal_symbol=True,
        fixed_background_classical_domain_of_dependence=True,
        spacing_action_winding_derived=False,
        interacting_backreacted_theta_solution_derived=False,
        background_stability_or_caustic_freedom_derived=False,
        periodic_well_localized_modes_derived=False,
        action_to_projected_rates_or_resonance_derived=False,
        scattering_energy_transfer_receipt_derived=False,
        durable_record_or_selection_derived=False,
        repeated_cptp_fresh_ancilla_derived=False,
        band_or_front_speed_derived=False,
        qft_microcausality_or_no_signalling_derived=False,
        coupled_gr_source_derived=False,
        infinite_isolated_lattice_finite_total_energy_derived=False,
        gates_five_to_eight_derived=False,
    )
