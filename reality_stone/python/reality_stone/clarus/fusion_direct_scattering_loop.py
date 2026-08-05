"""Low-energy nuclear diagnostics for the direct fusion scalar operator.

The calculations here deliberately stop short of an exclusion.  They evaluate
the free Born scattering-length shift and the first-order expectation value in
a normalized Hulthen deuteron control wavefunction.  A physical constraint
still requires a distorted-wave NN phase-shift and few-body refit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .fusion_equation_iteration_loop import current_fusion_equation_iteration_report
from .fusion_resonance_loop import HBAR_C_MEV_FM, NUCLEON_MASS_MEV


REGISTERED_SCALAR_MASS_MEV = 29.64757
DEUTERON_BINDING_ENERGY_MEV = 2.224566
DEFAULT_HULTHEN_BETA_FM_INV = 1.4
NP_TRIPLET_SCATTERING_LENGTH_FM = 5.4112
NP_TRIPLET_SCATTERING_LENGTH_UNCERTAINTY_FM = 0.0015
NP_SINGLET_SCATTERING_LENGTH_FM = -23.7148
NP_SINGLET_SCATTERING_LENGTH_UNCERTAINTY_FM = 0.0043


@dataclass(frozen=True)
class DirectNuclearScatteringAudit:
    required_nucleon_coupling: float
    scalar_mass_mev: float
    scalar_range_fm: float
    scalar_fine_structure: float
    free_born_scattering_length_shift_fm: float
    np_triplet_scattering_length_fm: float
    np_triplet_reported_uncertainty_fm: float
    born_shift_to_triplet_uncertainty: float
    np_singlet_scattering_length_fm: float
    np_singlet_reported_uncertainty_fm: float
    born_shift_to_singlet_uncertainty: float
    deuteron_binding_momentum_fm_inv: float
    hulthen_beta_fm_inv: float
    hulthen_normalization_fm_inv_sqrt: float
    hulthen_yukawa_expectation_mev: float
    absolute_deuteron_shift_kev: float
    deuteron_shift_to_binding_fraction: float
    free_born_shift_resolved_by_reported_precision: bool
    fixed_hamiltonian_deuteron_shift_nonzero: bool
    strong_potential_refit_performed: bool
    distorted_wave_born_calculation_performed: bool
    few_body_binding_refit_performed: bool
    experimental_exclusion_derived: bool
    physical_direct_operator_gate_pass: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_direct_nuclear_scattering() -> DirectNuclearScatteringAudit:
    """Evaluate Born and Hulthen controls for the registered-mass one-percent fit."""

    iteration = current_fusion_equation_iteration_report()
    direct = iteration.direct_coupling_registered_mass_requirement
    coupling = direct.required_direct_nucleon_coupling
    mass = direct.scalar_mass_mev
    alpha_scalar = coupling**2 / (4.0 * math.pi)
    reduced_np_mass = 0.5 * NUCLEON_MASS_MEV
    born_shift = -2.0 * reduced_np_mass * alpha_scalar * HBAR_C_MEV_FM / mass**2

    binding_momentum = math.sqrt(2.0 * reduced_np_mass * DEUTERON_BINDING_ENERGY_MEV) / (
        HBAR_C_MEV_FM
    )
    beta = DEFAULT_HULTHEN_BETA_FM_INV
    normalization_squared = (
        2.0
        * binding_momentum
        * beta
        * (binding_momentum + beta)
        / (beta - binding_momentum) ** 2
    )
    inverse_range = mass / HBAR_C_MEV_FM
    p = 2.0 * binding_momentum + inverse_range
    q = 2.0 * beta + inverse_range
    s = binding_momentum + beta + inverse_range
    radial_integral_fm_inv = normalization_squared * math.log(s**2 / (p * q))
    deuteron_shift = -alpha_scalar * HBAR_C_MEV_FM * radial_integral_fm_inv

    triplet_ratio = abs(born_shift) / NP_TRIPLET_SCATTERING_LENGTH_UNCERTAINTY_FM
    singlet_ratio = abs(born_shift) / NP_SINGLET_SCATTERING_LENGTH_UNCERTAINTY_FM
    return DirectNuclearScatteringAudit(
        required_nucleon_coupling=coupling,
        scalar_mass_mev=mass,
        scalar_range_fm=HBAR_C_MEV_FM / mass,
        scalar_fine_structure=alpha_scalar,
        free_born_scattering_length_shift_fm=born_shift,
        np_triplet_scattering_length_fm=NP_TRIPLET_SCATTERING_LENGTH_FM,
        np_triplet_reported_uncertainty_fm=NP_TRIPLET_SCATTERING_LENGTH_UNCERTAINTY_FM,
        born_shift_to_triplet_uncertainty=triplet_ratio,
        np_singlet_scattering_length_fm=NP_SINGLET_SCATTERING_LENGTH_FM,
        np_singlet_reported_uncertainty_fm=NP_SINGLET_SCATTERING_LENGTH_UNCERTAINTY_FM,
        born_shift_to_singlet_uncertainty=singlet_ratio,
        deuteron_binding_momentum_fm_inv=binding_momentum,
        hulthen_beta_fm_inv=beta,
        hulthen_normalization_fm_inv_sqrt=math.sqrt(normalization_squared),
        hulthen_yukawa_expectation_mev=deuteron_shift,
        absolute_deuteron_shift_kev=1000.0 * abs(deuteron_shift),
        deuteron_shift_to_binding_fraction=(abs(deuteron_shift) / DEUTERON_BINDING_ENERGY_MEV),
        free_born_shift_resolved_by_reported_precision=(triplet_ratio > 1.0),
        fixed_hamiltonian_deuteron_shift_nonzero=(abs(deuteron_shift) > 0.0),
        strong_potential_refit_performed=False,
        distorted_wave_born_calculation_performed=False,
        few_body_binding_refit_performed=False,
        experimental_exclusion_derived=False,
        physical_direct_operator_gate_pass=False,
        status="FREE_BORN_AND_HULTHEN_TENSION_CONTROL_FULL_NUCLEAR_REFIT_REQUIRED",
    )


__all__ = ["DirectNuclearScatteringAudit", "audit_direct_nuclear_scattering"]
