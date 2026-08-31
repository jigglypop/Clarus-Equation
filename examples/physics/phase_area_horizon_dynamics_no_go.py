'''No-go audit for the phase-area horizon dark-energy interpretation.'''

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

try:
    from examples.physics.cosmological_constant_holographic_gate import (
        N_GAUGE,
        OMEGA_LAMBDA,
        derive_entropy,
        rho_lambda_quarter_mev,
        true_de_sitter_vacuum_quarter_mev,
    )
except ModuleNotFoundError:
    from cosmological_constant_holographic_gate import (
        N_GAUGE,
        OMEGA_LAMBDA,
        derive_entropy,
        rho_lambda_quarter_mev,
        true_de_sitter_vacuum_quarter_mev,
    )


PHASE_AREA_COEFFICIENT = math.pi**2 / 2.0


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f'{name} must be positive')
    return value


def apparent_horizon_log_entropy_relative(hubble_ratio: float) -> float:
    '''Return ln[S(H)/S(H_ref)] for a flat apparent horizon.'''

    return -2.0 * math.log(_positive(hubble_ratio, 'hubble_ratio'))


def phase_label_for_hubble_ratio(
    hubble_ratio: float,
    *,
    phase_area_coefficient: float = PHASE_AREA_COEFFICIENT,
) -> float:
    coefficient = _positive(phase_area_coefficient, 'phase_area_coefficient')
    return apparent_horizon_log_entropy_relative(hubble_ratio) / coefficient


def hubble_ratio_from_phase_label(
    phase_label: float,
    *,
    phase_area_coefficient: float = PHASE_AREA_COEFFICIENT,
) -> float:
    phase_label = _finite(phase_label, 'phase_label')
    coefficient = _positive(phase_area_coefficient, 'phase_area_coefficient')
    return math.exp(-0.5 * coefficient * phase_label)


@dataclass(frozen=True)
class PhysicalEfoldPhaseAreaAudit:
    phase_area_coefficient: float
    entropy_slope_per_physical_efold: float
    epsilon_h: float
    effective_w_flat_gr: float
    deceleration_parameter: float
    power_law_scale_factor_exponent: float
    exact_de_sitter_entropy_slope: float
    accelerates: bool
    compatible_with_exact_de_sitter: bool
    compatible_with_late_dark_energy_acceleration: bool
    entropy_growth_law_is_adopted_axiom: bool = True
    apparent_horizon_area_law_is_conditional_input: bool = True
    unique_dark_energy_prediction: bool = False
    status: str = (
        'PHYSICAL_EFOLD_PHASE_AREA_IMPLIES_DECELERATION_NOT_DARK_ENERGY'
    )


def audit_physical_efold_phase_area(
    *,
    phase_area_coefficient: float = PHASE_AREA_COEFFICIENT,
) -> PhysicalEfoldPhaseAreaAudit:
    '''Audit the adopted law d ln(S_A)/d ln(a)=xi in flat GR.'''

    coefficient = _positive(phase_area_coefficient, 'phase_area_coefficient')
    epsilon_h = 0.5 * coefficient
    effective_w = -1.0 + coefficient / 3.0
    deceleration = epsilon_h - 1.0
    accelerates = epsilon_h < 1.0
    return PhysicalEfoldPhaseAreaAudit(
        phase_area_coefficient=coefficient,
        entropy_slope_per_physical_efold=coefficient,
        epsilon_h=epsilon_h,
        effective_w_flat_gr=effective_w,
        deceleration_parameter=deceleration,
        power_law_scale_factor_exponent=1.0 / epsilon_h,
        exact_de_sitter_entropy_slope=0.0,
        accelerates=accelerates,
        compatible_with_exact_de_sitter=math.isclose(
            coefficient, 0.0, rel_tol=0.0, abs_tol=1.0e-15
        ),
        compatible_with_late_dark_energy_acceleration=accelerates,
    )


@dataclass(frozen=True)
class BoundaryHistoryWitness:
    name: str
    z: float
    hubble_ratio: float
    phase_label: float
    reconstructed_hubble_ratio: float
    reconstruction_residual: float


@dataclass(frozen=True)
class BoundaryLabelPhaseAreaAudit:
    witnesses: tuple[BoundaryHistoryWitness, ...]
    all_histories_reconstructed: bool
    histories_are_distinct: bool
    phase_relation_selects_one_history: bool
    physical_efold_map_derived: bool
    unique_hubble_history: bool
    status: str = 'BOUNDARY_PHASE_LABEL_REPARAMETRIZES_ARBITRARY_H_OF_Z'


def audit_boundary_label_phase_area(
    *,
    z: float = 1.0,
    omega_m: float = 0.3,
    omega_lambda: float = 0.7,
    phase_area_coefficient: float = PHASE_AREA_COEFFICIENT,
) -> BoundaryLabelPhaseAreaAudit:
    '''Show that a boundary label can encode two expansion histories.'''

    z = _positive(z, 'z')
    omega_m = _finite(omega_m, 'omega_m')
    omega_lambda = _finite(omega_lambda, 'omega_lambda')
    if omega_m < 0.0 or omega_lambda < 0.0:
        raise ValueError('density fractions must be non-negative')
    if not math.isclose(omega_m + omega_lambda, 1.0, abs_tol=1.0e-12):
        raise ValueError('flat density fractions must sum to one')
    coefficient = _positive(phase_area_coefficient, 'phase_area_coefficient')
    histories = (
        ('Einstein_de_Sitter', (1.0 + z) ** 1.5),
        (
            'flat_LambdaCDM_0p3_0p7',
            math.sqrt(omega_m * (1.0 + z) ** 3 + omega_lambda),
        ),
    )
    witnesses = []
    for name, hubble_ratio in histories:
        label = phase_label_for_hubble_ratio(
            hubble_ratio, phase_area_coefficient=coefficient
        )
        reconstructed = hubble_ratio_from_phase_label(
            label, phase_area_coefficient=coefficient
        )
        witnesses.append(
            BoundaryHistoryWitness(
                name=name,
                z=z,
                hubble_ratio=hubble_ratio,
                phase_label=label,
                reconstructed_hubble_ratio=reconstructed,
                reconstruction_residual=reconstructed - hubble_ratio,
            )
        )
    all_reconstructed = all(
        abs(item.reconstruction_residual) <= 1.0e-14 for item in witnesses
    )
    distinct = not math.isclose(
        witnesses[0].hubble_ratio,
        witnesses[1].hubble_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    return BoundaryLabelPhaseAreaAudit(
        witnesses=tuple(witnesses),
        all_histories_reconstructed=all_reconstructed,
        histories_are_distinct=distinct,
        phase_relation_selects_one_history=False,
        physical_efold_map_derived=False,
        unique_hubble_history=False,
    )


@dataclass(frozen=True)
class PhaseAreaInputAudit:
    d_eff: float
    n_gauge: float
    n_e: float
    phase_area_coefficient: float
    defect_boundary: float
    log_entropy: float
    dln_h_d_d_eff: float
    dln_density_d_d_eff: float
    dln_quarter_scale_d_d_eff: float
    density_ratio_for_delta_d_eff_0p01: float
    true_de_sitter_quarter_scale_mev: float
    legacy_mixed_quarter_scale_mev: float
    supplied_omega_lambda: float
    mixed_over_true_quarter_scale: float
    consistent_planck_convention_coefficient_residual: float
    wrong_reduced_mass_in_unreduced_formula_factor: float
    n_gauge_is_supplied: bool = True
    n_e_relation_is_supplied: bool = True
    omega_lambda_is_supplied: bool = True
    legacy_mixed_value_is_target_aware: bool = True
    absolute_scale_unique: bool = False
    status: str = 'PHASE_AREA_SCALE_RETAINS_COUNT_EPOCH_AND_OMEGA_INPUTS'


def audit_phase_area_inputs() -> PhaseAreaInputAudit:
    '''Audit supplied counts, sensitivity, and mixed-epoch input use.'''

    entropy = derive_entropy()
    coefficient = PHASE_AREA_COEFFICIENT
    n_gauge = float(N_GAUGE)
    dln_h = -0.75 * coefficient * n_gauge
    dln_density = -1.5 * coefficient * n_gauge
    dln_quarter = 0.25 * dln_density
    true_ds = true_de_sitter_vacuum_quarter_mev(entropy['log_s'])
    mixed = rho_lambda_quarter_mev(entropy['log_s'], OMEGA_LAMBDA)

    # Correct reduced and unreduced Planck conventions agree exactly:
    # pi*(sqrt(8*pi) Mbar)^2 = 8*pi^2*Mbar^2.
    correct_reduced_coefficient = 8.0 * math.pi**2
    converted_unreduced_coefficient = math.pi * (8.0 * math.pi)
    return PhaseAreaInputAudit(
        d_eff=entropy['d_eff'],
        n_gauge=n_gauge,
        n_e=entropy['n_e'],
        phase_area_coefficient=coefficient,
        defect_boundary=math.pi * entropy['delta'] * entropy['sigma'],
        log_entropy=entropy['log_s'],
        dln_h_d_d_eff=dln_h,
        dln_density_d_d_eff=dln_density,
        dln_quarter_scale_d_d_eff=dln_quarter,
        density_ratio_for_delta_d_eff_0p01=math.exp(0.01 * dln_density),
        true_de_sitter_quarter_scale_mev=true_ds,
        legacy_mixed_quarter_scale_mev=mixed,
        supplied_omega_lambda=OMEGA_LAMBDA,
        mixed_over_true_quarter_scale=mixed / true_ds,
        consistent_planck_convention_coefficient_residual=(
            converted_unreduced_coefficient - correct_reduced_coefficient
        ),
        wrong_reduced_mass_in_unreduced_formula_factor=8.0 * math.pi,
    )


@dataclass(frozen=True)
class PhaseAreaHorizonEndToEndAudit:
    physical_efold: PhysicalEfoldPhaseAreaAudit
    boundary_label: BoundaryLabelPhaseAreaAudit
    inputs: PhaseAreaInputAudit
    universal_entropy_growth_parent_refuted: bool = True
    physical_efold_dark_energy_parent_refuted: bool = True
    boundary_label_unique_hubble_parent_refuted: bool = True
    unique_absolute_dark_energy_prediction: bool = False
    maximum_true_claims: tuple[str, ...] = (
        'flat_apparent_horizon_relative_entropy_is_minus_two_log_hubble_ratio',
        'adopted_physical_entropy_slope_gives_a_conditional_power_law',
        'boundary_phase_labels_can_reencode_a_supplied_positive_hubble_history',
    )
    dimensionless_arguments: tuple[tuple[str, str], ...] = (
        ('log(H/H_ref)', 'H/H_ref is dimensionless'),
        ('xi * N', 'xi and the phase label N are dimensionless'),
        ('Omega_Lambda**(1/4)', 'Omega_Lambda is dimensionless'),
    )
    status: str = (
        'PHASE_AREA_DARK_ENERGY_ROUTE_REFUTED_KINEMATIC_SUBCLAIM_RETAINED'
    )


def audit_phase_area_horizon_end_to_end() -> PhaseAreaHorizonEndToEndAudit:
    return PhaseAreaHorizonEndToEndAudit(
        physical_efold=audit_physical_efold_phase_area(),
        boundary_label=audit_boundary_label_phase_area(),
        inputs=audit_phase_area_inputs(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='phase_area_horizon_dynamics_no_go'
    )
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args(argv)
    print(
        json.dumps(
            asdict(audit_phase_area_horizon_end_to_end()),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
