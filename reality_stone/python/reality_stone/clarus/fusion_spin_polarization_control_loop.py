"""Standard-Model spin-polarized D--T control for the fusion loop.

This is deliberately not a CE/new-particle success branch.  It asks whether a
known reaction-specific control can clear the same one-percent, Maxwellian,
and source-ledger gates used for CE candidates.

For vector polarizations ``P_D`` and ``P_T``, dominance of the low-energy
``J^pi=3/2+`` entrance channel gives the conditional ideal-projector control

    sigma_pol(E) / sigma_unpol(E) = 1 + P_D P_T / 2.

The factor is energy independent in that approximation, so it commutes with
the Maxwellian integral.  A general spin-1 deuteron state additionally needs
tensor polarization ``p_zz``; this module now validates the corresponding
three diagonal populations.  Ab-initio NCSMC calculations support the
underlying channel dominance, but the public energy-dependent curve and source
evidence are audited separately.  The ideal-projector benchmark is therefore
kept separate from the physical reaction and engineering gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any

from .fusion_full_loop import (
    BOSCH_HALE_DT_TEMPERATURE_MAX_KEV,
    BOSCH_HALE_DT_TEMPERATURE_MIN_KEV,
    bosch_hale_dt_reactivity,
)
from .fusion_polarized_evidence_loop import (
    FusionPolarizedEvidenceReport,
    current_fusion_polarized_evidence_report,
    physical_polarized_evidence_gate_pass,
)
from .fusion_resonance_loop import DEUTERON_MASS_MEV, TRITON_MASS_MEV


EV_TO_JOULE = 1.602176634e-19
NUCLEAR_MAGNETON_EV_T = 3.15245125844e-8
VACUUM_PERMEABILITY_H_M = 1.25663706212e-6
DEUTERON_MAGNETIC_MOMENT_NM = 0.8574382338
TRITON_MAGNETIC_MOMENT_NM = 2.978962465
DT_FUSION_ENERGY_MEV = 17.6
MEV_C2_TO_KG = 1.7826619216279e-30
QED_CRITICAL_MAGNETIC_FIELD_T = 4.414e9
DEFAULT_TEMPERATURE_KEV = 10.0
DEFAULT_TARGET_FRACTIONAL_GAIN = 0.01
DEFAULT_DESIGN_DEUTERON_POLARIZATION = 0.50
DEFAULT_DESIGN_DEUTERON_TENSOR_POLARIZATION = 0.0
DEFAULT_DESIGN_TRITON_POLARIZATION = 0.10
DEFAULT_TOTAL_DT_ION_DENSITY_M3 = 1.0e20
DEFAULT_REFERENCE_FUSION_POWER_W = 5.0e8
DEFAULT_SINGLE_PASS_BURN_FRACTION = 0.01
DEFAULT_ELECTRIC_CONVERSION_EFFICIENCY = 0.40
DEFAULT_ENGINEERING_MARGIN_FACTOR = 10.0
REFERENCE_IDEAL_POLARIZED_DEUTERIUM_BEAM_TARGET_RATE_S = 1.0e18


@dataclass(frozen=True)
class SpinReactionAudit:
    temperature_kev: float
    deuteron_vector_polarization: float
    deuteron_tensor_polarization: float
    deuteron_mplus_population: float
    deuteron_mzero_population: float
    deuteron_mminus_population: float
    deuteron_spin_density_matrix_physical: bool
    triton_vector_polarization: float
    polarization_product: float
    quartet_probability: float
    unpolarized_quartet_probability: float
    cross_section_multiplier: float
    baseline_maxwellian_reactivity_cm3_s: float
    modified_maxwellian_reactivity_cm3_s: float
    maxwellian_reactivity_ratio: float
    fractional_reactivity_gain: float
    target_fractional_gain: float
    required_polarization_product: float
    ideal_projector_arithmetic_target_reached: bool
    standard_model_spin_projector_derived: bool
    j32_s_wave_dominance_assumed: bool
    ab_initio_channel_support_available: bool
    spin_correlation_czz_directly_measured: bool
    polarized_dt_rate_directly_validated: bool
    energy_dependent_polarized_reaction_operator_provenance_pass: bool
    energy_dependent_polarized_reaction_operator_provenance_digest: str | None
    constant_ideal_projector_multiplier_commuted_with_maxwellian_average: bool
    energy_dependent_polarized_cross_section_integrated: bool
    conditional_ideal_projector_target_reached: bool
    status: str


@dataclass(frozen=True)
class PolarizationRetentionAudit:
    source_deuteron_polarization: float
    source_deuteron_tensor_polarization: float
    source_triton_polarization: float
    source_polarization_product: float
    source_fractional_reactivity_gain: float
    required_reaction_point_product: float
    required_reaction_point_deuteron_tensor_polarization: float
    burn_weighted_product_required: float
    source_polarization_product_meets_required_threshold: bool
    minimum_common_species_retention_fraction: float
    minimum_product_retention_fraction: float
    deuteron_dnp_benchmark_polarization: float
    deuteron_dnp_benchmark_field_t: float
    deuteron_dnp_benchmark_temperature_k: float
    deuteron_dnp_buildup_hours_representative: float
    dnp_buildup_time_is_representative_not_exact: bool
    high_frequency_wave_depolarization_small_in_simulation: bool
    partial_helium3_plasma_retention_evidence_available: bool
    burn_weighted_polarization_product: float | None
    burn_weighted_deuteron_tensor_polarization: float | None
    burn_weighted_product_tensor_physical_upper_bound: float | None
    burn_weighted_measurement_provenance: str | None
    burn_weighted_product_measured: bool
    burn_weighted_deuteron_tensor_polarization_measured: bool
    burn_weighted_deuteron_tensor_matches_required_state: bool
    burn_weighted_product_with_tensor_physical_bound_pass: bool
    burn_weighted_measurement_provenance_available: bool
    burn_weighted_product_meets_required_threshold: bool
    burn_weighted_product_not_above_source_product: bool
    deuteron_pellet_source_demonstrated_at_reactor_throughput: bool
    tritium_polarization_source_demonstrated_at_reactor_throughput: bool
    dt_in_plasma_retention_measured: bool
    injection_and_retention_gate_pass: bool
    status: str


@dataclass(frozen=True)
class ThermalEquilibriumPolarizationAudit:
    temperature_kev: float
    target_polarization_product: float
    required_uniform_magnetic_field_t: float
    equilibrium_deuteron_polarization: float
    equilibrium_triton_polarization: float
    magnetic_field_energy_density_j_m3: float
    qed_critical_magnetic_field_t: float
    required_to_qed_critical_field_ratio: float
    declared_total_dt_ion_density_m3: float
    plasma_ion_plus_electron_thermal_energy_density_j_m3: float
    field_to_plasma_thermal_energy_density_ratio: float
    polarization_product_residual: float
    adaptive_field_bracket_pass: bool
    finite_temperature_brillouin_solved: bool
    equilibrium_magnetization_route_pass: bool
    status: str


@dataclass(frozen=True)
class PolarizedFuelPumpLedgerAudit:
    declared_reference_fusion_power_w: float
    required_fusion_reaction_rate_s: float
    declared_dt_pair_injection_rate_s: float
    declared_dt_mass_throughput_kg_s: float
    declared_single_pass_burn_fraction: float
    target_fractional_reactivity_gain: float
    fusion_energy_per_reaction_mev: float
    expected_baseline_fusion_energy_per_injected_pair_kev: float
    fixed_profile_linear_rate_response_assumed: bool
    linearized_incremental_fusion_energy_budget_per_injected_pair_kev: float
    linearized_raw_incremental_fusion_power_w: float
    fixed_exposure_burn_probability_model_evaluated: bool
    fixed_exposure_incremental_burn_fraction: float
    conservative_fixed_exposure_budget_used: bool
    incremental_fusion_energy_budget_per_injected_pair_kev: float
    raw_incremental_fusion_power_w: float
    declared_electric_conversion_efficiency: float
    electrical_break_even_energy_per_injected_pair_ev: float
    electrical_break_even_energy_per_baseline_reacted_pair_ev: float
    electrical_break_even_energy_per_incremental_fusion_reaction_ev: float
    maximum_wall_plug_pump_power_for_net_gain_w: float
    declared_engineering_margin_factor: float
    engineering_margin_energy_per_injected_pair_ev: float
    engineering_margin_wall_plug_power_w: float
    reference_ideal_polarized_deuterium_beam_target_rate_s: float
    reactor_per_species_to_reference_deuterium_beam_target_ratio: float
    dnp_source_temperature_k: float
    dnp_source_magnetic_field_t: float
    energy_ledger_denominator: str
    reference_throughput_calculation_declared: bool
    measured_wall_plug_energy_per_injected_dt_pair_ev: float | None
    measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev: float | None
    measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev: float | None
    measured_wall_plug_energy_3sigma_bound_model: str | None
    measured_wall_plug_energy_per_injected_dt_pair_available: bool
    wall_plug_measurement_uncertainty_available: bool
    wall_plug_measurement_sigma_multiplier: float
    measured_injected_pair_energy_below_break_even: bool
    measured_injected_pair_energy_below_engineering_margin: bool
    uncertainty_upper_below_engineering_margin: bool
    wall_plug_measurement_duration_s: float | None
    wall_plug_measurement_injected_dt_pair_count: float | None
    wall_plug_measurement_pair_flow_rate_s: float | None
    wall_plug_measurement_pair_flow_meets_declared_throughput: bool
    wall_plug_measurement_provenance: str | None
    wall_plug_measurement_provenance_available: bool
    cryogenic_and_microwave_power_accounted: bool
    tritium_handling_power_accounted: bool
    recycle_depolarization_and_repolarization_accounted: bool
    net_incremental_energy_positive_demonstrated: bool
    source_and_pump_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionSpinPolarizationControlReport:
    schema_version: str
    target_reaction: SpinReactionAudit
    source_design_reaction: SpinReactionAudit
    retention: PolarizationRetentionAudit
    thermal_equilibrium: ThermalEquilibriumPolarizationAudit
    pump_ledger: PolarizedFuelPumpLedgerAudit
    published_evidence: FusionPolarizedEvidenceReport
    standard_model_non_ce_mechanism_identified: bool
    conditional_ideal_projector_one_percent_control_derived: bool
    full_source_and_pump_ledger_closed: bool
    physical_spin_polarized_branch_pass: bool
    ce_new_physics_used: bool
    physical_ce_one_percent_branch_pass: bool
    maximum_supported_stage: str
    next_required_gate: str
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


def _unit_interval(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _signed_unit_interval(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < -1.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return result


def _deuteron_spin_populations(
    *, vector_polarization: float, tensor_polarization: float
) -> tuple[float, float, float]:
    """Return diagonal spin-1 populations and reject an unphysical density matrix.

    The convention is ``p_z = n_+ - n_-`` and
    ``p_zz = n_+ + n_- - 2 n_0``.  It follows that
    ``n_0=(1-p_zz)/3`` and ``n_±=(2+p_zz±3p_z)/6``.
    """

    if tensor_polarization < -2.0 or tensor_polarization > 1.0:
        raise ValueError("deuteron_tensor_polarization must lie in [-2, 1]")
    populations = (
        (2.0 + tensor_polarization + 3.0 * vector_polarization) / 6.0,
        (1.0 - tensor_polarization) / 3.0,
        (2.0 + tensor_polarization - 3.0 * vector_polarization) / 6.0,
    )
    if any(population < 0.0 or population > 1.0 for population in populations):
        raise ValueError("deuteron vector/tensor polarizations define a nonphysical density matrix")
    if not math.isclose(math.fsum(populations), 1.0, rel_tol=0.0, abs_tol=2.0e-16):
        raise RuntimeError("deuteron spin populations failed normalization")
    return populations


def _boolean(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def audit_spin_reaction(
    *,
    temperature_kev: Real = DEFAULT_TEMPERATURE_KEV,
    deuteron_polarization: Real = math.sqrt(0.02),
    deuteron_tensor_polarization: Real = 0.0,
    triton_polarization: Real = math.sqrt(0.02),
    target_fractional_gain: Real = DEFAULT_TARGET_FRACTIONAL_GAIN,
) -> SpinReactionAudit:
    """Apply the spin projector before the Bosch--Hale Maxwellian average."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    p_d = _signed_unit_interval(
        deuteron_polarization,
        name="deuteron_polarization",
    )
    p_zz = _finite_real(
        deuteron_tensor_polarization,
        name="deuteron_tensor_polarization",
    )
    n_plus, n_zero, n_minus = _deuteron_spin_populations(
        vector_polarization=p_d,
        tensor_polarization=p_zz,
    )
    p_t = _signed_unit_interval(triton_polarization, name="triton_polarization")
    target = _positive(target_fractional_gain, name="target_fractional_gain")
    if target > 0.5:
        raise ValueError("target_fractional_gain cannot exceed the spin limit 0.5")

    product = p_d * p_t
    quartet_probability = (2.0 + product) / 3.0
    unpolarized_quartet_probability = 2.0 / 3.0
    multiplier = quartet_probability / unpolarized_quartet_probability
    _, _, baseline = bosch_hale_dt_reactivity(temperature)
    modified = multiplier * baseline
    ratio = multiplier
    gain = 0.5 * product
    required_product = 2.0 * target
    target_reached = product >= required_product
    conditional_control_pass = target_reached and product > 0.0
    return SpinReactionAudit(
        temperature_kev=temperature,
        deuteron_vector_polarization=p_d,
        deuteron_tensor_polarization=p_zz,
        deuteron_mplus_population=n_plus,
        deuteron_mzero_population=n_zero,
        deuteron_mminus_population=n_minus,
        deuteron_spin_density_matrix_physical=True,
        triton_vector_polarization=p_t,
        polarization_product=product,
        quartet_probability=quartet_probability,
        unpolarized_quartet_probability=unpolarized_quartet_probability,
        cross_section_multiplier=multiplier,
        baseline_maxwellian_reactivity_cm3_s=baseline,
        modified_maxwellian_reactivity_cm3_s=modified,
        maxwellian_reactivity_ratio=ratio,
        fractional_reactivity_gain=gain,
        target_fractional_gain=target,
        required_polarization_product=required_product,
        ideal_projector_arithmetic_target_reached=target_reached,
        standard_model_spin_projector_derived=True,
        j32_s_wave_dominance_assumed=True,
        ab_initio_channel_support_available=True,
        spin_correlation_czz_directly_measured=False,
        polarized_dt_rate_directly_validated=False,
        energy_dependent_polarized_reaction_operator_provenance_pass=False,
        energy_dependent_polarized_reaction_operator_provenance_digest=None,
        constant_ideal_projector_multiplier_commuted_with_maxwellian_average=True,
        energy_dependent_polarized_cross_section_integrated=False,
        conditional_ideal_projector_target_reached=conditional_control_pass,
        status=(
            "IDEAL_QUARTET_PROJECTOR_TARGET_REACHED_CZZ_UNMEASURED"
            if conditional_control_pass
            else "SPIN_PRODUCT_BELOW_REACTIVITY_TARGET"
        ),
    )


def audit_polarization_retention(
    *,
    source_deuteron_polarization: Real = DEFAULT_DESIGN_DEUTERON_POLARIZATION,
    source_deuteron_tensor_polarization: Real = (DEFAULT_DESIGN_DEUTERON_TENSOR_POLARIZATION),
    source_triton_polarization: Real = DEFAULT_DESIGN_TRITON_POLARIZATION,
    target_fractional_gain: Real = DEFAULT_TARGET_FRACTIONAL_GAIN,
    target_deuteron_tensor_polarization: Real = 0.0,
    burn_weighted_polarization_product: Real | None = None,
    burn_weighted_deuteron_tensor_polarization: Real | None = None,
    burn_weighted_measurement_provenance: str | None = None,
    deuteron_source_at_throughput: bool = False,
    tritium_source_at_throughput: bool = False,
    dt_retention_measured: bool = False,
) -> PolarizationRetentionAudit:
    """Quantify source margin while leaving unmeasured D--T gates closed."""

    p_d = _unit_interval(
        source_deuteron_polarization,
        name="source_deuteron_polarization",
    )
    p_t = _unit_interval(
        source_triton_polarization,
        name="source_triton_polarization",
    )
    source_p_zz = _finite_real(
        source_deuteron_tensor_polarization,
        name="source_deuteron_tensor_polarization",
    )
    if not -2.0 <= source_p_zz <= 1.0:
        raise ValueError("source_deuteron_tensor_polarization must lie in [-2, 1]")
    _deuteron_spin_populations(
        vector_polarization=p_d,
        tensor_polarization=source_p_zz,
    )
    target_p_zz = _finite_real(
        target_deuteron_tensor_polarization,
        name="target_deuteron_tensor_polarization",
    )
    if not -2.0 <= target_p_zz <= 1.0:
        raise ValueError("target_deuteron_tensor_polarization must lie in [-2, 1]")
    target = _positive(target_fractional_gain, name="target_fractional_gain")
    if target > 0.5:
        raise ValueError("target_fractional_gain cannot exceed the spin limit 0.5")
    product = p_d * p_t
    required = 2.0 * target
    source_product_meets_threshold = product >= required
    if product < required:
        common_retention = math.inf
        product_retention = math.inf
    else:
        product_retention = required / product
        common_retention = math.sqrt(product_retention)
    if burn_weighted_polarization_product is None:
        measured_product = None
        burn_weighted_product_measured = False
        burn_weighted_product_meets_threshold = False
        burn_weighted_product_not_above_source = False
    else:
        measured_product = _unit_interval(
            burn_weighted_polarization_product,
            name="burn_weighted_polarization_product",
        )
        burn_weighted_product_measured = True
        burn_weighted_product_meets_threshold = measured_product >= required
        burn_weighted_product_not_above_source = measured_product <= product
    if burn_weighted_deuteron_tensor_polarization is None:
        measured_p_zz = None
        burn_weighted_p_zz_measured = False
        burn_weighted_p_zz_matches_target = False
    else:
        measured_p_zz = _finite_real(
            burn_weighted_deuteron_tensor_polarization,
            name="burn_weighted_deuteron_tensor_polarization",
        )
        if not -2.0 <= measured_p_zz <= 1.0:
            raise ValueError("burn_weighted_deuteron_tensor_polarization must lie in [-2, 1]")
        burn_weighted_p_zz_measured = True
        burn_weighted_p_zz_matches_target = measured_p_zz == target_p_zz
    if measured_product is not None and measured_p_zz is not None:
        product_tensor_upper_bound = (2.0 + measured_p_zz) / 3.0
        product_tensor_physical_bound_pass = measured_product <= product_tensor_upper_bound
    else:
        product_tensor_upper_bound = None
        product_tensor_physical_bound_pass = False
    if burn_weighted_measurement_provenance is None:
        measurement_provenance = None
        measurement_provenance_available = False
    elif not isinstance(burn_weighted_measurement_provenance, str):
        raise ValueError("burn_weighted_measurement_provenance must be a string or None")
    else:
        measurement_provenance = burn_weighted_measurement_provenance.strip()
        measurement_provenance_available = _sha256_provenance_digest_available(
            measurement_provenance
        )
    deuteron_source_at_throughput = _boolean(
        deuteron_source_at_throughput,
        name="deuteron_source_at_throughput",
    )
    tritium_source_at_throughput = _boolean(
        tritium_source_at_throughput,
        name="tritium_source_at_throughput",
    )
    dt_retention_measured = _boolean(
        dt_retention_measured,
        name="dt_retention_measured",
    )
    retention_gate = all(
        (
            burn_weighted_product_measured,
            burn_weighted_product_meets_threshold,
            burn_weighted_product_not_above_source,
            burn_weighted_p_zz_measured,
            burn_weighted_p_zz_matches_target,
            product_tensor_physical_bound_pass,
            measurement_provenance_available,
            source_product_meets_threshold,
            deuteron_source_at_throughput,
            tritium_source_at_throughput,
            dt_retention_measured,
        )
    )
    return PolarizationRetentionAudit(
        source_deuteron_polarization=p_d,
        source_deuteron_tensor_polarization=source_p_zz,
        source_triton_polarization=p_t,
        source_polarization_product=product,
        source_fractional_reactivity_gain=0.5 * product,
        required_reaction_point_product=required,
        required_reaction_point_deuteron_tensor_polarization=target_p_zz,
        burn_weighted_product_required=required,
        source_polarization_product_meets_required_threshold=(source_product_meets_threshold),
        minimum_common_species_retention_fraction=common_retention,
        minimum_product_retention_fraction=product_retention,
        deuteron_dnp_benchmark_polarization=0.50,
        deuteron_dnp_benchmark_field_t=5.0,
        deuteron_dnp_benchmark_temperature_k=1.0,
        deuteron_dnp_buildup_hours_representative=3.0,
        dnp_buildup_time_is_representative_not_exact=True,
        high_frequency_wave_depolarization_small_in_simulation=True,
        partial_helium3_plasma_retention_evidence_available=True,
        burn_weighted_polarization_product=measured_product,
        burn_weighted_deuteron_tensor_polarization=measured_p_zz,
        burn_weighted_product_tensor_physical_upper_bound=(product_tensor_upper_bound),
        burn_weighted_measurement_provenance=measurement_provenance,
        burn_weighted_product_measured=burn_weighted_product_measured,
        burn_weighted_deuteron_tensor_polarization_measured=(burn_weighted_p_zz_measured),
        burn_weighted_deuteron_tensor_matches_required_state=(burn_weighted_p_zz_matches_target),
        burn_weighted_product_with_tensor_physical_bound_pass=(product_tensor_physical_bound_pass),
        burn_weighted_measurement_provenance_available=(measurement_provenance_available),
        burn_weighted_product_meets_required_threshold=(burn_weighted_product_meets_threshold),
        burn_weighted_product_not_above_source_product=(burn_weighted_product_not_above_source),
        deuteron_pellet_source_demonstrated_at_reactor_throughput=(deuteron_source_at_throughput),
        tritium_polarization_source_demonstrated_at_reactor_throughput=(
            tritium_source_at_throughput
        ),
        dt_in_plasma_retention_measured=dt_retention_measured,
        injection_and_retention_gate_pass=retention_gate,
        status=(
            "SOURCE_AND_BURN_WEIGHTED_RETENTION_GATE_PASS"
            if retention_gate
            else (
                "MEASURED_OR_DECLARED_RETENTION_INPUT_PRESENT_BUT_GATE_FAIL"
                if (
                    burn_weighted_product_measured
                    or burn_weighted_p_zz_measured
                    or measurement_provenance_available
                    or deuteron_source_at_throughput
                    or tritium_source_at_throughput
                    or dt_retention_measured
                )
                else "SOURCE_MARGIN_EXISTS_BUT_TRITIUM_AND_RETENTION_GATES_ARE_UNMEASURED"
            )
        ),
    )


def _equilibrium_deuteron_polarization(field_t: float, temperature_ev: float) -> float:
    x = DEUTERON_MAGNETIC_MOMENT_NM * NUCLEAR_MAGNETON_EV_T * field_t / temperature_ev
    half_argument_tanh = math.tanh(0.5 * x)
    return 4.0 * half_argument_tanh / (3.0 + half_argument_tanh**2)


def _equilibrium_triton_polarization(field_t: float, temperature_ev: float) -> float:
    x = TRITON_MAGNETIC_MOMENT_NM * NUCLEAR_MAGNETON_EV_T * field_t / temperature_ev
    return math.tanh(x)


def audit_thermal_equilibrium_polarization(
    *,
    temperature_kev: Real = DEFAULT_TEMPERATURE_KEV,
    target_fractional_gain: Real = DEFAULT_TARGET_FRACTIONAL_GAIN,
    total_dt_ion_density_m3: Real = DEFAULT_TOTAL_DT_ION_DENSITY_M3,
) -> ThermalEquilibriumPolarizationAudit:
    """Solve the ideal noninteracting spin-1/spin-1/2 Zeeman model."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    if not (BOSCH_HALE_DT_TEMPERATURE_MIN_KEV <= temperature <= BOSCH_HALE_DT_TEMPERATURE_MAX_KEV):
        raise ValueError("temperature_kev must lie in the 0.2--100 keV D-T fusion-fit domain")
    target = _positive(target_fractional_gain, name="target_fractional_gain")
    density = _positive(total_dt_ion_density_m3, name="total_dt_ion_density_m3")
    required_product = 2.0 * target
    if required_product >= 1.0:
        raise ValueError("target_fractional_gain must be below 0.5")
    temperature_ev = 1.0e3 * temperature

    def equilibrium_product(field_t: float) -> float:
        return _equilibrium_deuteron_polarization(
            field_t,
            temperature_ev,
        ) * _equilibrium_triton_polarization(field_t, temperature_ev)

    low = 0.0
    high = 1.0
    while equilibrium_product(high) < required_product:
        high *= 2.0
        if high > 1.0e14:
            raise RuntimeError("failed to bracket the thermal-polarization field")
    bracket_pass = equilibrium_product(high) >= required_product
    for _ in range(200):
        middle = 0.5 * (low + high)
        product = equilibrium_product(middle)
        if product < required_product:
            low = middle
        else:
            high = middle
    field = 0.5 * (low + high)
    p_d = _equilibrium_deuteron_polarization(field, temperature_ev)
    p_t = _equilibrium_triton_polarization(field, temperature_ev)
    residual = abs(p_d * p_t - required_product)
    if residual > max(1.0e-13, required_product * 1.0e-12):
        raise RuntimeError("thermal-polarization field solve did not converge")
    field_density = field * field / (2.0 * VACUUM_PERMEABILITY_H_M)
    # Equal-temperature D+T ions and their quasineutral electrons give
    # (3/2 n_i + 3/2 n_e) kT = 3 n_i kT.
    plasma_density = 3.0 * density * temperature_ev * EV_TO_JOULE
    equilibrium_route_pass = (
        field <= QED_CRITICAL_MAGNETIC_FIELD_T and field_density <= plasma_density
    )
    return ThermalEquilibriumPolarizationAudit(
        temperature_kev=temperature,
        target_polarization_product=required_product,
        required_uniform_magnetic_field_t=field,
        equilibrium_deuteron_polarization=p_d,
        equilibrium_triton_polarization=p_t,
        magnetic_field_energy_density_j_m3=field_density,
        qed_critical_magnetic_field_t=QED_CRITICAL_MAGNETIC_FIELD_T,
        required_to_qed_critical_field_ratio=(field / QED_CRITICAL_MAGNETIC_FIELD_T),
        declared_total_dt_ion_density_m3=density,
        plasma_ion_plus_electron_thermal_energy_density_j_m3=plasma_density,
        field_to_plasma_thermal_energy_density_ratio=field_density / plasma_density,
        polarization_product_residual=residual,
        adaptive_field_bracket_pass=bracket_pass,
        finite_temperature_brillouin_solved=True,
        equilibrium_magnetization_route_pass=equilibrium_route_pass,
        status=(
            "FUSION_DOMAIN_EQUILIBRIUM_MAGNETIZATION_CONTROL_PASS"
            if equilibrium_route_pass
            else "FUSION_DOMAIN_EQUILIBRIUM_FIELD_ENERGY_OR_QED_NO_GO"
        ),
    )


def audit_polarized_fuel_pump_ledger(
    *,
    reference_fusion_power_w: Real = DEFAULT_REFERENCE_FUSION_POWER_W,
    single_pass_burn_fraction: Real = DEFAULT_SINGLE_PASS_BURN_FRACTION,
    target_fractional_gain: Real = DEFAULT_TARGET_FRACTIONAL_GAIN,
    electric_conversion_efficiency: Real = DEFAULT_ELECTRIC_CONVERSION_EFFICIENCY,
    engineering_margin_factor: Real = DEFAULT_ENGINEERING_MARGIN_FACTOR,
    measured_wall_plug_energy_per_injected_dt_pair_ev: Real | None = None,
    measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev: Real | None = None,
    measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev: Real | None = None,
    measured_wall_plug_energy_3sigma_bound_model: str | None = None,
    wall_plug_measurement_duration_s: Real | None = None,
    wall_plug_measurement_injected_dt_pair_count: Real | None = None,
    wall_plug_measurement_provenance: str | None = None,
    cryogenic_and_microwave_power_accounted: bool = False,
    tritium_handling_power_accounted: bool = False,
    recycle_repolarization_accounted: bool = False,
    net_incremental_energy_positive_demonstrated: bool = False,
) -> PolarizedFuelPumpLedgerAudit:
    """Set the wall-plug ceiling that a real polarized source must beat."""

    fusion_power = _positive(
        reference_fusion_power_w,
        name="reference_fusion_power_w",
    )
    burn_fraction = _unit_interval(
        single_pass_burn_fraction,
        name="single_pass_burn_fraction",
    )
    if burn_fraction == 0.0 or burn_fraction == 1.0:
        raise ValueError("single_pass_burn_fraction must lie strictly between zero and one")
    target = _positive(target_fractional_gain, name="target_fractional_gain")
    if target > 0.5:
        raise ValueError("target_fractional_gain cannot exceed the spin limit 0.5")
    efficiency = _unit_interval(
        electric_conversion_efficiency,
        name="electric_conversion_efficiency",
    )
    if efficiency == 0.0:
        raise ValueError("electric_conversion_efficiency must be positive")
    margin = _positive(engineering_margin_factor, name="engineering_margin_factor")
    if margin < 1.0:
        raise ValueError("engineering_margin_factor must be at least one")

    fusion_energy_j = DT_FUSION_ENERGY_MEV * 1.0e6 * EV_TO_JOULE
    reaction_rate = fusion_power / fusion_energy_j
    pair_rate = reaction_rate / burn_fraction
    pair_mass_kg = (DEUTERON_MASS_MEV + TRITON_MASS_MEV) * MEV_C2_TO_KG
    baseline_kev = burn_fraction * DT_FUSION_ENERGY_MEV * 1.0e3
    linearized_incremental_kev = target * baseline_kev
    linearized_raw_incremental_power = target * fusion_power
    incremental_burn_fraction = -(1.0 - burn_fraction) * math.expm1(
        target * math.log1p(-burn_fraction)
    )
    incremental_kev = incremental_burn_fraction * DT_FUSION_ENERGY_MEV * 1.0e3
    raw_incremental_power = fusion_power * incremental_burn_fraction / burn_fraction
    wall_plug_power = efficiency * raw_incremental_power
    electrical_break_even_ev = efficiency * incremental_kev * 1.0e3
    engineering_margin_ev = electrical_break_even_ev / margin
    if measured_wall_plug_energy_per_injected_dt_pair_ev is None:
        measured_pair_energy_ev = None
        measured_pair_energy_available = False
        measured_pair_energy_below_break_even = False
        measured_pair_energy_below_margin = False
    else:
        measured_pair_energy_ev = _finite_real(
            measured_wall_plug_energy_per_injected_dt_pair_ev,
            name="measured_wall_plug_energy_per_injected_dt_pair_ev",
        )
        if measured_pair_energy_ev < 0.0:
            raise ValueError(
                "measured_wall_plug_energy_per_injected_dt_pair_ev must be nonnegative"
            )
        measured_pair_energy_available = True
        measured_pair_energy_below_break_even = measured_pair_energy_ev < electrical_break_even_ev
        measured_pair_energy_below_margin = measured_pair_energy_ev < engineering_margin_ev
    if measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev is None:
        measured_pair_energy_std_ev = None
    else:
        measured_pair_energy_std_ev = _finite_real(
            measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev,
            name="measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev",
        )
        if measured_pair_energy_std_ev < 0.0:
            raise ValueError(
                "measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev "
                "must be nonnegative"
            )
    if measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev is None:
        uncertainty_upper_ev = None
    else:
        uncertainty_upper_ev = _finite_real(
            measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev,
            name=(
                "measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev"
            ),
        )
        if uncertainty_upper_ev < 0.0:
            raise ValueError(
                "measured wall-plug 3sigma upper energy must be nonnegative"
            )
    exact_bound_model = (
        "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
    )
    bound_model = (
        measured_wall_plug_energy_3sigma_bound_model.strip()
        if isinstance(measured_wall_plug_energy_3sigma_bound_model, str)
        else None
    )
    measurement_uncertainty_available = all(
        (
            measured_pair_energy_std_ev is not None,
            uncertainty_upper_ev is not None,
            bound_model == exact_bound_model,
        )
    )
    sigma_multiplier = 3.0
    if measured_pair_energy_available and measurement_uncertainty_available:
        if uncertainty_upper_ev < measured_pair_energy_ev:
            raise ValueError(
                "measured wall-plug 3sigma upper energy must not be below nominal"
            )
        uncertainty_upper_below_margin = uncertainty_upper_ev < engineering_margin_ev
    else:
        uncertainty_upper_below_margin = False
    if wall_plug_measurement_duration_s is None:
        measurement_duration = None
    else:
        measurement_duration = _positive(
            wall_plug_measurement_duration_s,
            name="wall_plug_measurement_duration_s",
        )
    if wall_plug_measurement_injected_dt_pair_count is None:
        measurement_pair_count = None
    else:
        measurement_pair_count = _positive(
            wall_plug_measurement_injected_dt_pair_count,
            name="wall_plug_measurement_injected_dt_pair_count",
        )
    if measurement_duration is not None and measurement_pair_count is not None:
        measurement_pair_flow_rate = measurement_pair_count / measurement_duration
        measurement_pair_flow_meets_throughput = measurement_pair_flow_rate >= pair_rate
    else:
        measurement_pair_flow_rate = None
        measurement_pair_flow_meets_throughput = False
    if wall_plug_measurement_provenance is None:
        measurement_provenance = None
        measurement_provenance_available = False
    elif not isinstance(wall_plug_measurement_provenance, str):
        raise ValueError("wall_plug_measurement_provenance must be a string or None")
    else:
        measurement_provenance = wall_plug_measurement_provenance.strip()
        measurement_provenance_available = _sha256_provenance_digest_available(
            measurement_provenance
        )
    cryogenic_and_microwave_accounted = _boolean(
        cryogenic_and_microwave_power_accounted,
        name="cryogenic_and_microwave_power_accounted",
    )
    tritium_handling_accounted = _boolean(
        tritium_handling_power_accounted,
        name="tritium_handling_power_accounted",
    )
    recycle_repolarization_accounted = _boolean(
        recycle_repolarization_accounted,
        name="recycle_repolarization_accounted",
    )
    positive_net_increment_demonstrated = (
        _boolean(
            net_incremental_energy_positive_demonstrated,
            name="net_incremental_energy_positive_demonstrated",
        )
        and measured_pair_energy_below_break_even
    )
    pump_gate = all(
        (
            measured_pair_energy_available,
            measured_pair_energy_below_break_even,
            measured_pair_energy_below_margin,
            measurement_uncertainty_available,
            uncertainty_upper_below_margin,
            measurement_pair_flow_meets_throughput,
            measurement_provenance_available,
            cryogenic_and_microwave_accounted,
            tritium_handling_accounted,
            recycle_repolarization_accounted,
            positive_net_increment_demonstrated,
        )
    )
    return PolarizedFuelPumpLedgerAudit(
        declared_reference_fusion_power_w=fusion_power,
        required_fusion_reaction_rate_s=reaction_rate,
        declared_dt_pair_injection_rate_s=pair_rate,
        declared_dt_mass_throughput_kg_s=pair_rate * pair_mass_kg,
        declared_single_pass_burn_fraction=burn_fraction,
        target_fractional_reactivity_gain=target,
        fusion_energy_per_reaction_mev=DT_FUSION_ENERGY_MEV,
        expected_baseline_fusion_energy_per_injected_pair_kev=baseline_kev,
        fixed_profile_linear_rate_response_assumed=True,
        linearized_incremental_fusion_energy_budget_per_injected_pair_kev=(
            linearized_incremental_kev
        ),
        linearized_raw_incremental_fusion_power_w=linearized_raw_incremental_power,
        fixed_exposure_burn_probability_model_evaluated=True,
        fixed_exposure_incremental_burn_fraction=incremental_burn_fraction,
        conservative_fixed_exposure_budget_used=True,
        incremental_fusion_energy_budget_per_injected_pair_kev=incremental_kev,
        raw_incremental_fusion_power_w=raw_incremental_power,
        declared_electric_conversion_efficiency=efficiency,
        electrical_break_even_energy_per_injected_pair_ev=electrical_break_even_ev,
        electrical_break_even_energy_per_baseline_reacted_pair_ev=(
            electrical_break_even_ev / burn_fraction
        ),
        electrical_break_even_energy_per_incremental_fusion_reaction_ev=(
            electrical_break_even_ev / incremental_burn_fraction
        ),
        maximum_wall_plug_pump_power_for_net_gain_w=wall_plug_power,
        declared_engineering_margin_factor=margin,
        engineering_margin_energy_per_injected_pair_ev=engineering_margin_ev,
        engineering_margin_wall_plug_power_w=wall_plug_power / margin,
        reference_ideal_polarized_deuterium_beam_target_rate_s=(
            REFERENCE_IDEAL_POLARIZED_DEUTERIUM_BEAM_TARGET_RATE_S
        ),
        reactor_per_species_to_reference_deuterium_beam_target_ratio=(
            pair_rate / REFERENCE_IDEAL_POLARIZED_DEUTERIUM_BEAM_TARGET_RATE_S
        ),
        dnp_source_temperature_k=1.0,
        dnp_source_magnetic_field_t=5.0,
        energy_ledger_denominator="injected D-T pair",
        reference_throughput_calculation_declared=True,
        measured_wall_plug_energy_per_injected_dt_pair_ev=measured_pair_energy_ev,
        measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev=(
            measured_pair_energy_std_ev
        ),
        measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev=(
            uncertainty_upper_ev
        ),
        measured_wall_plug_energy_3sigma_bound_model=bound_model,
        measured_wall_plug_energy_per_injected_dt_pair_available=(measured_pair_energy_available),
        wall_plug_measurement_uncertainty_available=measurement_uncertainty_available,
        wall_plug_measurement_sigma_multiplier=sigma_multiplier,
        measured_injected_pair_energy_below_break_even=(measured_pair_energy_below_break_even),
        measured_injected_pair_energy_below_engineering_margin=(measured_pair_energy_below_margin),
        uncertainty_upper_below_engineering_margin=uncertainty_upper_below_margin,
        wall_plug_measurement_duration_s=measurement_duration,
        wall_plug_measurement_injected_dt_pair_count=measurement_pair_count,
        wall_plug_measurement_pair_flow_rate_s=measurement_pair_flow_rate,
        wall_plug_measurement_pair_flow_meets_declared_throughput=(
            measurement_pair_flow_meets_throughput
        ),
        wall_plug_measurement_provenance=measurement_provenance,
        wall_plug_measurement_provenance_available=measurement_provenance_available,
        cryogenic_and_microwave_power_accounted=cryogenic_and_microwave_accounted,
        tritium_handling_power_accounted=tritium_handling_accounted,
        recycle_depolarization_and_repolarization_accounted=(recycle_repolarization_accounted),
        net_incremental_energy_positive_demonstrated=positive_net_increment_demonstrated,
        source_and_pump_gate_pass=pump_gate,
        status=(
            "MEASURED_SOURCE_AND_PUMP_LEDGER_PASS"
            if pump_gate
            else (
                "MEASURED_PUMP_LEDGER_PRESENT_BUT_NET_OR_ACCOUNTING_GATE_FAIL"
                if measured_pair_energy_available
                else "CONSERVATIVE_FIXED_EXPOSURE_PUMP_CEILING_COMPUTED_MEASURED_LEDGER_ABSENT"
            )
        ),
    )


def _finite_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _sha256_provenance_digest_available(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _reaction_numeric_integrity(
    reaction: SpinReactionAudit,
    *,
    expected_target: float | None = None,
) -> bool:
    numeric_values = (
        reaction.temperature_kev,
        reaction.deuteron_vector_polarization,
        reaction.deuteron_tensor_polarization,
        reaction.deuteron_mplus_population,
        reaction.deuteron_mzero_population,
        reaction.deuteron_mminus_population,
        reaction.triton_vector_polarization,
        reaction.polarization_product,
        reaction.quartet_probability,
        reaction.unpolarized_quartet_probability,
        reaction.cross_section_multiplier,
        reaction.baseline_maxwellian_reactivity_cm3_s,
        reaction.modified_maxwellian_reactivity_cm3_s,
        reaction.maxwellian_reactivity_ratio,
        reaction.fractional_reactivity_gain,
        reaction.target_fractional_gain,
        reaction.required_polarization_product,
    )
    if not all(_finite_scalar(value) for value in numeric_values):
        return False
    if not (
        BOSCH_HALE_DT_TEMPERATURE_MIN_KEV
        <= reaction.temperature_kev
        <= BOSCH_HALE_DT_TEMPERATURE_MAX_KEV
        and -1.0 <= reaction.deuteron_vector_polarization <= 1.0
        and -2.0 <= reaction.deuteron_tensor_polarization <= 1.0
        and -1.0 <= reaction.triton_vector_polarization <= 1.0
        and 0.0 < reaction.target_fractional_gain <= 0.5
    ):
        return False
    if expected_target is not None and reaction.target_fractional_gain != expected_target:
        return False
    _, _, recomputed_baseline = bosch_hale_dt_reactivity(reaction.temperature_kev)
    try:
        n_plus, n_zero, n_minus = _deuteron_spin_populations(
            vector_polarization=reaction.deuteron_vector_polarization,
            tensor_polarization=reaction.deuteron_tensor_polarization,
        )
    except (ValueError, RuntimeError):
        return False
    product = reaction.deuteron_vector_polarization * reaction.triton_vector_polarization
    quartet_probability = (2.0 + product) / 3.0
    unpolarized_quartet_probability = 2.0 / 3.0
    multiplier = quartet_probability / unpolarized_quartet_probability
    return all(
        (
            reaction.polarization_product == product,
            reaction.deuteron_mplus_population == n_plus,
            reaction.deuteron_mzero_population == n_zero,
            reaction.deuteron_mminus_population == n_minus,
            reaction.deuteron_spin_density_matrix_physical is True,
            reaction.quartet_probability == quartet_probability,
            reaction.unpolarized_quartet_probability == unpolarized_quartet_probability,
            reaction.cross_section_multiplier == multiplier,
            reaction.baseline_maxwellian_reactivity_cm3_s == recomputed_baseline,
            reaction.modified_maxwellian_reactivity_cm3_s == multiplier * recomputed_baseline,
            reaction.maxwellian_reactivity_ratio == multiplier,
            reaction.fractional_reactivity_gain == 0.5 * product,
            reaction.required_polarization_product == 2.0 * reaction.target_fractional_gain,
            product >= reaction.required_polarization_product,
            reaction.fractional_reactivity_gain >= reaction.target_fractional_gain,
        )
    )


def _retention_numeric_integrity(
    retention: PolarizationRetentionAudit,
    *,
    expected_target: float,
    expected_source_deuteron_polarization: float,
    expected_source_deuteron_tensor_polarization: float,
    expected_source_triton_polarization: float,
    expected_target_deuteron_tensor_polarization: float,
) -> bool:
    measured = retention.burn_weighted_polarization_product
    measured_p_zz = retention.burn_weighted_deuteron_tensor_polarization
    product_tensor_upper_bound = retention.burn_weighted_product_tensor_physical_upper_bound
    numeric_values = (
        retention.source_deuteron_polarization,
        retention.source_deuteron_tensor_polarization,
        retention.source_triton_polarization,
        retention.source_polarization_product,
        retention.source_fractional_reactivity_gain,
        retention.required_reaction_point_product,
        retention.required_reaction_point_deuteron_tensor_polarization,
        retention.burn_weighted_product_required,
        retention.minimum_common_species_retention_fraction,
        retention.minimum_product_retention_fraction,
        measured,
        measured_p_zz,
        product_tensor_upper_bound,
    )
    if not all(_finite_scalar(value) for value in numeric_values):
        return False
    source_product = retention.source_deuteron_polarization * retention.source_triton_polarization
    required = 2.0 * expected_target
    if not (
        0.0 <= retention.source_deuteron_polarization <= 1.0
        and 0.0 <= retention.source_triton_polarization <= 1.0
        and -2.0 <= retention.source_deuteron_tensor_polarization <= 1.0
        and source_product >= required
        and 0.0 <= measured <= 1.0
        and required <= measured <= source_product
        and -2.0 <= measured_p_zz <= 1.0
        and retention.source_deuteron_tensor_polarization
        == expected_source_deuteron_tensor_polarization
        and retention.source_deuteron_polarization == expected_source_deuteron_polarization
        and retention.source_triton_polarization == expected_source_triton_polarization
        and retention.required_reaction_point_deuteron_tensor_polarization
        == expected_target_deuteron_tensor_polarization
        and measured_p_zz == expected_target_deuteron_tensor_polarization
        and product_tensor_upper_bound == (2.0 + measured_p_zz) / 3.0
        and measured <= product_tensor_upper_bound
        and _sha256_provenance_digest_available(retention.burn_weighted_measurement_provenance)
    ):
        return False
    try:
        _deuteron_spin_populations(
            vector_polarization=retention.source_deuteron_polarization,
            tensor_polarization=retention.source_deuteron_tensor_polarization,
        )
    except (ValueError, RuntimeError):
        return False
    product_retention = required / source_product
    return all(
        (
            retention.source_polarization_product == source_product,
            retention.source_fractional_reactivity_gain == 0.5 * source_product,
            retention.required_reaction_point_product == required,
            retention.required_reaction_point_deuteron_tensor_polarization
            == expected_target_deuteron_tensor_polarization,
            retention.burn_weighted_product_required == required,
            retention.minimum_product_retention_fraction == product_retention,
            retention.minimum_common_species_retention_fraction == math.sqrt(product_retention),
            retention.burn_weighted_deuteron_tensor_polarization_measured is True,
            retention.burn_weighted_deuteron_tensor_matches_required_state is True,
            retention.burn_weighted_product_with_tensor_physical_bound_pass is True,
            retention.burn_weighted_measurement_provenance_available is True,
        )
    )


def _pump_numeric_integrity(
    pump: PolarizedFuelPumpLedgerAudit,
    *,
    expected_target: float,
) -> bool:
    measured = pump.measured_wall_plug_energy_per_injected_dt_pair_ev
    measured_std = (
        pump.measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev
    )
    uncertainty_upper = (
        pump.measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev
    )
    duration = pump.wall_plug_measurement_duration_s
    pair_count = pump.wall_plug_measurement_injected_dt_pair_count
    measured_flow = pump.wall_plug_measurement_pair_flow_rate_s
    numeric_values = (
        pump.declared_reference_fusion_power_w,
        pump.required_fusion_reaction_rate_s,
        pump.declared_dt_pair_injection_rate_s,
        pump.declared_dt_mass_throughput_kg_s,
        pump.declared_single_pass_burn_fraction,
        pump.target_fractional_reactivity_gain,
        pump.fusion_energy_per_reaction_mev,
        pump.expected_baseline_fusion_energy_per_injected_pair_kev,
        pump.linearized_incremental_fusion_energy_budget_per_injected_pair_kev,
        pump.linearized_raw_incremental_fusion_power_w,
        pump.fixed_exposure_incremental_burn_fraction,
        pump.incremental_fusion_energy_budget_per_injected_pair_kev,
        pump.raw_incremental_fusion_power_w,
        pump.declared_electric_conversion_efficiency,
        pump.electrical_break_even_energy_per_injected_pair_ev,
        pump.electrical_break_even_energy_per_baseline_reacted_pair_ev,
        pump.electrical_break_even_energy_per_incremental_fusion_reaction_ev,
        pump.maximum_wall_plug_pump_power_for_net_gain_w,
        pump.declared_engineering_margin_factor,
        pump.engineering_margin_energy_per_injected_pair_ev,
        pump.engineering_margin_wall_plug_power_w,
        pump.reference_ideal_polarized_deuterium_beam_target_rate_s,
        pump.reactor_per_species_to_reference_deuterium_beam_target_ratio,
        pump.dnp_source_temperature_k,
        pump.dnp_source_magnetic_field_t,
        measured,
        measured_std,
        pump.wall_plug_measurement_sigma_multiplier,
        uncertainty_upper,
        duration,
        pair_count,
        measured_flow,
    )
    if not all(_finite_scalar(value) for value in numeric_values):
        return False
    burn_fraction = pump.declared_single_pass_burn_fraction
    efficiency = pump.declared_electric_conversion_efficiency
    margin = pump.declared_engineering_margin_factor
    if not (
        pump.declared_reference_fusion_power_w == DEFAULT_REFERENCE_FUSION_POWER_W
        and burn_fraction == DEFAULT_SINGLE_PASS_BURN_FRACTION
        and pump.target_fractional_reactivity_gain == expected_target
        and pump.fusion_energy_per_reaction_mev == DT_FUSION_ENERGY_MEV
        and 0.0 < efficiency <= 1.0
        and margin >= 1.0
        and pump.dnp_source_temperature_k > 0.0
        and pump.dnp_source_magnetic_field_t > 0.0
        and measured >= 0.0
        and measured_std >= 0.0
        and uncertainty_upper >= measured
        and pump.measured_wall_plug_energy_3sigma_bound_model
        == "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
        and duration > 0.0
        and pair_count > 0.0
        and measured_flow > 0.0
        and pump.energy_ledger_denominator == "injected D-T pair"
        and _sha256_provenance_digest_available(pump.wall_plug_measurement_provenance)
    ):
        return False
    fusion_energy_j = DT_FUSION_ENERGY_MEV * 1.0e6 * EV_TO_JOULE
    reaction_rate = pump.declared_reference_fusion_power_w / fusion_energy_j
    pair_rate = reaction_rate / burn_fraction
    pair_mass = (DEUTERON_MASS_MEV + TRITON_MASS_MEV) * MEV_C2_TO_KG
    baseline_kev = burn_fraction * DT_FUSION_ENERGY_MEV * 1.0e3
    linearized_incremental_kev = expected_target * baseline_kev
    incremental_burn_fraction = -(1.0 - burn_fraction) * math.expm1(
        expected_target * math.log1p(-burn_fraction)
    )
    incremental_kev = incremental_burn_fraction * DT_FUSION_ENERGY_MEV * 1.0e3
    raw_incremental_power = (
        pump.declared_reference_fusion_power_w * incremental_burn_fraction / burn_fraction
    )
    electrical_break_even_ev = efficiency * incremental_kev * 1.0e3
    engineering_margin_ev = electrical_break_even_ev / margin
    wall_plug_power = efficiency * raw_incremental_power
    recomputed_flow = pair_count / duration
    return all(
        (
            pump.required_fusion_reaction_rate_s == reaction_rate,
            pump.declared_dt_pair_injection_rate_s == pair_rate,
            pump.declared_dt_mass_throughput_kg_s == pair_rate * pair_mass,
            pump.expected_baseline_fusion_energy_per_injected_pair_kev == baseline_kev,
            pump.linearized_incremental_fusion_energy_budget_per_injected_pair_kev
            == linearized_incremental_kev,
            pump.linearized_raw_incremental_fusion_power_w
            == expected_target * pump.declared_reference_fusion_power_w,
            pump.fixed_exposure_incremental_burn_fraction == incremental_burn_fraction,
            pump.incremental_fusion_energy_budget_per_injected_pair_kev == incremental_kev,
            pump.raw_incremental_fusion_power_w == raw_incremental_power,
            pump.electrical_break_even_energy_per_injected_pair_ev == electrical_break_even_ev,
            pump.electrical_break_even_energy_per_baseline_reacted_pair_ev
            == electrical_break_even_ev / burn_fraction,
            pump.electrical_break_even_energy_per_incremental_fusion_reaction_ev
            == electrical_break_even_ev / incremental_burn_fraction,
            pump.maximum_wall_plug_pump_power_for_net_gain_w == wall_plug_power,
            pump.engineering_margin_energy_per_injected_pair_ev == engineering_margin_ev,
            pump.engineering_margin_wall_plug_power_w == wall_plug_power / margin,
            pump.reference_ideal_polarized_deuterium_beam_target_rate_s
            == REFERENCE_IDEAL_POLARIZED_DEUTERIUM_BEAM_TARGET_RATE_S,
            pump.reactor_per_species_to_reference_deuterium_beam_target_ratio
            == pair_rate / REFERENCE_IDEAL_POLARIZED_DEUTERIUM_BEAM_TARGET_RATE_S,
            pump.reference_throughput_calculation_declared is True,
            pump.measured_wall_plug_energy_per_injected_dt_pair_available is True,
            pump.wall_plug_measurement_uncertainty_available is True,
            pump.wall_plug_measurement_sigma_multiplier == 3.0,
            measured_flow == recomputed_flow,
            pump.wall_plug_measurement_pair_flow_meets_declared_throughput
            == (recomputed_flow >= pair_rate),
            pump.wall_plug_measurement_provenance_available is True,
            measured < electrical_break_even_ev,
            measured < engineering_margin_ev,
            uncertainty_upper < engineering_margin_ev,
            pump.measured_injected_pair_energy_below_break_even is True,
            pump.measured_injected_pair_energy_below_engineering_margin is True,
            pump.uncertainty_upper_below_engineering_margin is True,
        )
    )


def _spin_component_gate_pass(
    *,
    target_reaction: SpinReactionAudit,
    source_design_reaction: SpinReactionAudit,
    retention: PolarizationRetentionAudit,
    pump_ledger: PolarizedFuelPumpLedgerAudit,
) -> bool:
    """Check internally supplied reaction/source/pump components only."""

    target = target_reaction.target_fractional_gain
    if not (
        _reaction_numeric_integrity(target_reaction)
        and _reaction_numeric_integrity(source_design_reaction, expected_target=target)
        and _retention_numeric_integrity(
            retention,
            expected_target=target,
            expected_source_deuteron_polarization=(
                source_design_reaction.deuteron_vector_polarization
            ),
            expected_source_deuteron_tensor_polarization=(
                source_design_reaction.deuteron_tensor_polarization
            ),
            expected_source_triton_polarization=(source_design_reaction.triton_vector_polarization),
            expected_target_deuteron_tensor_polarization=(
                target_reaction.deuteron_tensor_polarization
            ),
        )
        and _pump_numeric_integrity(pump_ledger, expected_target=target)
        and target >= DEFAULT_TARGET_FRACTIONAL_GAIN
        and target_reaction.temperature_kev == DEFAULT_TEMPERATURE_KEV
        and source_design_reaction.temperature_kev == target_reaction.temperature_kev
        and source_design_reaction.fractional_reactivity_gain >= target
        and retention.burn_weighted_polarization_product == target_reaction.polarization_product
        and _sha256_provenance_digest_available(
            target_reaction.energy_dependent_polarized_reaction_operator_provenance_digest
        )
    ):
        return False
    required_true_flags = (
        target_reaction.ideal_projector_arithmetic_target_reached,
        target_reaction.deuteron_spin_density_matrix_physical,
        target_reaction.conditional_ideal_projector_target_reached,
        target_reaction.standard_model_spin_projector_derived,
        target_reaction.j32_s_wave_dominance_assumed,
        target_reaction.ab_initio_channel_support_available,
        target_reaction.constant_ideal_projector_multiplier_commuted_with_maxwellian_average,
        target_reaction.energy_dependent_polarized_cross_section_integrated,
        target_reaction.energy_dependent_polarized_reaction_operator_provenance_pass,
        target_reaction.spin_correlation_czz_directly_measured,
        target_reaction.polarized_dt_rate_directly_validated,
        source_design_reaction.ideal_projector_arithmetic_target_reached,
        source_design_reaction.deuteron_spin_density_matrix_physical,
        source_design_reaction.conditional_ideal_projector_target_reached,
        retention.burn_weighted_product_measured,
        retention.burn_weighted_product_meets_required_threshold,
        retention.burn_weighted_product_not_above_source_product,
        retention.burn_weighted_deuteron_tensor_polarization_measured,
        retention.burn_weighted_deuteron_tensor_matches_required_state,
        retention.burn_weighted_product_with_tensor_physical_bound_pass,
        retention.burn_weighted_measurement_provenance_available,
        retention.source_polarization_product_meets_required_threshold,
        retention.deuteron_pellet_source_demonstrated_at_reactor_throughput,
        retention.tritium_polarization_source_demonstrated_at_reactor_throughput,
        retention.dt_in_plasma_retention_measured,
        retention.injection_and_retention_gate_pass,
        pump_ledger.reference_throughput_calculation_declared,
        pump_ledger.fixed_exposure_burn_probability_model_evaluated,
        pump_ledger.conservative_fixed_exposure_budget_used,
        pump_ledger.measured_wall_plug_energy_per_injected_dt_pair_available,
        pump_ledger.wall_plug_measurement_uncertainty_available,
        pump_ledger.measured_injected_pair_energy_below_break_even,
        pump_ledger.measured_injected_pair_energy_below_engineering_margin,
        pump_ledger.uncertainty_upper_below_engineering_margin,
        pump_ledger.wall_plug_measurement_pair_flow_meets_declared_throughput,
        pump_ledger.wall_plug_measurement_provenance_available,
        pump_ledger.cryogenic_and_microwave_power_accounted,
        pump_ledger.tritium_handling_power_accounted,
        pump_ledger.recycle_depolarization_and_repolarization_accounted,
        pump_ledger.net_incremental_energy_positive_demonstrated,
        pump_ledger.source_and_pump_gate_pass,
    )
    return all(flag is True for flag in required_true_flags)


def _spin_evidence_join_pass(
    *,
    target_reaction: SpinReactionAudit,
    source_design_reaction: SpinReactionAudit,
    retention: PolarizationRetentionAudit,
    pump_ledger: PolarizedFuelPumpLedgerAudit,
    published_evidence: FusionPolarizedEvidenceReport,
) -> bool:
    """Join reaction, source, burn retention, and plant records exactly."""

    return all(
        (
            published_evidence.reaction.temperature_kev == target_reaction.temperature_kev,
            published_evidence.reaction.target_reactivity_ratio
            == 1.0 + target_reaction.target_fractional_gain,
            published_evidence.reaction.validated_target_deuteron_vector_polarization
            == target_reaction.deuteron_vector_polarization,
            published_evidence.reaction.validated_target_deuteron_tensor_polarization
            == target_reaction.deuteron_tensor_polarization,
            published_evidence.reaction.validated_target_triton_vector_polarization
            == target_reaction.triton_vector_polarization,
            published_evidence.reaction.validated_target_state_maxwellian_reactivity_ratio
            == target_reaction.maxwellian_reactivity_ratio,
            published_evidence.reaction.target_state_resolved_operator_available is True,
            published_evidence.reaction.target_state_operator_provenance_digest
            == target_reaction.energy_dependent_polarized_reaction_operator_provenance_digest,
            _sha256_provenance_digest_available(
                target_reaction.energy_dependent_polarized_reaction_operator_provenance_digest
            ),
            published_evidence.source.declared_reference_fusion_power_w
            == pump_ledger.declared_reference_fusion_power_w,
            published_evidence.source.declared_single_pass_burn_fraction
            == pump_ledger.declared_single_pass_burn_fraction,
            published_evidence.source.required_fusion_reaction_rate_s
            == pump_ledger.required_fusion_reaction_rate_s,
            published_evidence.source.required_per_species_fuel_rate_s
            == pump_ledger.declared_dt_pair_injection_rate_s,
            _finite_scalar(published_evidence.source.validated_reactor_rate_deuterium_source_rate_s)
            and _finite_scalar(
                published_evidence.source.validated_reactor_rate_deuterium_source_rate_linearized_std_s
            )
            and _finite_scalar(
                published_evidence.source.validated_reactor_rate_deuterium_source_rate_lower_3sigma_s
            )
            and published_evidence.source.validated_reactor_rate_deuterium_source_rate_linearized_std_s
            >= 0.0
            and published_evidence.source.validated_reactor_rate_deuterium_source_rate_lower_3sigma_s
            >= pump_ledger.declared_dt_pair_injection_rate_s,
            published_evidence.source.validated_reactor_rate_deuterium_source_rate_3sigma_bound_model
            == "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1",
            _finite_scalar(published_evidence.source.validated_reactor_rate_tritium_source_rate_s)
            and _finite_scalar(
                published_evidence.source.validated_reactor_rate_tritium_source_rate_linearized_std_s
            )
            and _finite_scalar(
                published_evidence.source.validated_reactor_rate_tritium_source_rate_lower_3sigma_s
            )
            and published_evidence.source.validated_reactor_rate_tritium_source_rate_linearized_std_s
            >= 0.0
            and published_evidence.source.validated_reactor_rate_tritium_source_rate_lower_3sigma_s
            >= pump_ledger.declared_dt_pair_injection_rate_s,
            published_evidence.source.validated_reactor_rate_tritium_source_rate_3sigma_bound_model
            == "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1",
            published_evidence.source.validated_reactor_rate_deuteron_vector_polarization
            == source_design_reaction.deuteron_vector_polarization,
            published_evidence.source.validated_reactor_rate_deuteron_tensor_polarization
            == source_design_reaction.deuteron_tensor_polarization,
            published_evidence.source.validated_reactor_rate_triton_vector_polarization
            == source_design_reaction.triton_vector_polarization,
            _sha256_provenance_digest_available(
                published_evidence.source.validated_reactor_rate_deuterium_source_provenance_digest
            ),
            _sha256_provenance_digest_available(
                published_evidence.source.validated_reactor_rate_tritium_source_provenance_digest
            ),
            published_evidence.source.validated_burn_weighted_dt_polarization_product
            == retention.burn_weighted_polarization_product,
            published_evidence.source.validated_burn_weighted_deuteron_tensor_polarization
            == retention.burn_weighted_deuteron_tensor_polarization,
            _finite_scalar(
                published_evidence.source.validated_burn_weighted_dt_polarization_product
            )
            and _finite_scalar(
                published_evidence.source.validated_burn_weighted_dt_polarization_product_std
            )
            and published_evidence.source.validated_burn_weighted_dt_polarization_product_std >= 0.0
            and published_evidence.source.validated_burn_weighted_dt_polarization_product
            - 3.0 * published_evidence.source.validated_burn_weighted_dt_polarization_product_std
            >= target_reaction.required_polarization_product,
            published_evidence.source.validated_burn_retention_provenance_digest
            == retention.burn_weighted_measurement_provenance,
            published_evidence.source.validated_wall_plug_energy_per_injected_dt_pair_ev
            == pump_ledger.measured_wall_plug_energy_per_injected_dt_pair_ev,
            published_evidence.source.validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev
            == pump_ledger.measured_wall_plug_energy_linearized_std_per_injected_dt_pair_ev,
            published_evidence.source.validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev
            == pump_ledger.measured_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev,
            published_evidence.source.validated_wall_plug_3sigma_bound_model
            == pump_ledger.measured_wall_plug_energy_3sigma_bound_model,
            published_evidence.source.validated_wall_plug_measurement_duration_s
            == pump_ledger.wall_plug_measurement_duration_s,
            published_evidence.source.validated_wall_plug_measurement_injected_dt_pair_count
            == pump_ledger.wall_plug_measurement_injected_dt_pair_count,
            published_evidence.source.validated_wall_plug_measurement_provenance
            == pump_ledger.wall_plug_measurement_provenance,
            published_evidence.energy_resolved_burn_operator_sufficient_statistics_available
            is True,
            published_evidence.source.energy_resolved_burn_operator_sufficient_statistics_available
            is True,
            _sha256_provenance_digest_available(pump_ledger.wall_plug_measurement_provenance),
        )
    )


def physical_spin_branch_gate_pass(
    *,
    target_reaction: SpinReactionAudit,
    source_design_reaction: SpinReactionAudit,
    retention: PolarizationRetentionAudit,
    pump_ledger: PolarizedFuelPumpLedgerAudit,
    published_evidence: FusionPolarizedEvidenceReport,
) -> bool:
    """Require component integrity and a fresh canonical primary-evidence report."""

    if not (
        type(target_reaction) is SpinReactionAudit
        and type(source_design_reaction) is SpinReactionAudit
        and type(retention) is PolarizationRetentionAudit
        and type(pump_ledger) is PolarizedFuelPumpLedgerAudit
        and type(published_evidence) is FusionPolarizedEvidenceReport
    ):
        return False
    try:
        component_pass = _spin_component_gate_pass(
            target_reaction=target_reaction,
            source_design_reaction=source_design_reaction,
            retention=retention,
            pump_ledger=pump_ledger,
        )
        evidence_join = _spin_evidence_join_pass(
            target_reaction=target_reaction,
            source_design_reaction=source_design_reaction,
            retention=retention,
            pump_ledger=pump_ledger,
            published_evidence=published_evidence,
        )
        return (
            component_pass
            and evidence_join
            and physical_polarized_evidence_gate_pass(published_evidence)
        )
    except Exception:
        return False


def current_fusion_spin_polarization_control_report() -> FusionSpinPolarizationControlReport:
    """Return the fail-closed standard-control report."""

    target_reaction = audit_spin_reaction()
    source_design_reaction = audit_spin_reaction(
        deuteron_polarization=DEFAULT_DESIGN_DEUTERON_POLARIZATION,
        triton_polarization=DEFAULT_DESIGN_TRITON_POLARIZATION,
    )
    retention = audit_polarization_retention()
    equilibrium = audit_thermal_equilibrium_polarization()
    pump = audit_polarized_fuel_pump_ledger()
    published_evidence = current_fusion_polarized_evidence_report()
    physical_pass = physical_spin_branch_gate_pass(
        target_reaction=target_reaction,
        source_design_reaction=source_design_reaction,
        retention=retention,
        pump_ledger=pump,
        published_evidence=published_evidence,
    )
    full_source_and_pump_ledger_closed = (
        retention.injection_and_retention_gate_pass and pump.source_and_pump_gate_pass
    )
    ce_new_physics_used = False
    return FusionSpinPolarizationControlReport(
        schema_version="fusion-spin-polarization-control-v10",
        target_reaction=target_reaction,
        source_design_reaction=source_design_reaction,
        retention=retention,
        thermal_equilibrium=equilibrium,
        pump_ledger=pump,
        published_evidence=published_evidence,
        standard_model_non_ce_mechanism_identified=True,
        conditional_ideal_projector_one_percent_control_derived=(
            target_reaction.conditional_ideal_projector_target_reached
        ),
        full_source_and_pump_ledger_closed=full_source_and_pump_ledger_closed,
        physical_spin_polarized_branch_pass=physical_pass,
        ce_new_physics_used=ce_new_physics_used,
        physical_ce_one_percent_branch_pass=(physical_pass and ce_new_physics_used),
        maximum_supported_stage=(
            "STANDARD_MODEL_ENERGY_DEPENDENT_FIGURE_DIGITIZED_CONTROL_"
            "SOURCE_THROUGHPUT_AND_RETENTION_FAIL_CLOSED"
        ),
        next_required_gate=(
            "AUTHOR_NUMERIC_SPIN_GRID_COVARIANCE_AND_CZZ_PLUS_MEASURED_"
            "POLARIZED_DT_SOURCE_RETENTION_WALL_PLUG_LEDGER"
        ),
        conclusion=(
            "Spin polarization clears the non-CE ideal-quartet control, and the published "
            "full-alignment figure clears a separate digitized energy-dependent Maxwellian "
            "control. Neither certifies the partial target state: author numeric spin data, "
            "covariance, Czz validation, tritium source, retention, and wall-plug ledger are "
            "still absent. No physical engineering or CE branch is promoted."
        ),
    )
