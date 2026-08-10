"""Primary-evidence audit for spin-polarized D--T fusion.

This loop closes a narrower reproducibility gap left by the ideal quartet
projector control.  It digitizes the published *energy-dependent* maximum
polarization enhancement in Hupin, Quaglioni, and Navratil (2019), multiplies
the Bosch--Hale D--T cross section by that curve, and performs the 10 keV
Maxwellian integral locally.

The result is deliberately a figure-digitized control, not a certified
reaction data evaluation.  The paper exposes no numeric energy grid, S-matrix,
or nuclear-model covariance, and no D--T spin-correlation measurement closes
those omissions.  Source evidence is audited separately against a declared
500 MW, one-percent single-pass-burn reference plant.  Every physical gate
therefore remains fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from typing import Any

import numpy as np

from .fusion_equation_iteration_loop import bosch_hale_dt_cross_section_m2
from .fusion_polarized_artifact_payloads import (
    ArtifactPayloadError,
    ENGINEERING_AUXILIARY_PAYLOAD_ROLES,
    ENGINEERING_PRIMARY_PAYLOAD_ROLES,
    PYTHON_CANONICAL_JSON_CONVENTION,
    derive_artifact_claims,
)
from .fusion_full_loop import (
    BOSCH_HALE_DT_TEMPERATURE_MAX_KEV,
    BOSCH_HALE_DT_TEMPERATURE_MIN_KEV,
    bosch_hale_dt_reactivity,
)
from .fusion_resonance_loop import DEUTERON_MASS_MEV, TRITON_MASS_MEV
from .fusion_sciencedb_payload_loop import (
    ScienceDBV1PayloadAudit,
    current_sciencedb_v1_payload_audit,
)
from .fusion_sciencedb_reactivity_loop import (
    ScienceDBDTReactivityAudit,
    current_sciencedb_dt_reactivity_audit,
)


EV_TO_JOULE = 1.602176634e-19
SPEED_OF_LIGHT_M_S = 299_792_458.0
DT_FUSION_ENERGY_MEV = 17.6
DEFAULT_TEMPERATURE_KEV = 10.0
DEFAULT_TARGET_REACTIVITY_RATIO = 1.01
DEFAULT_REQUIRED_BURN_POLARIZATION_PRODUCT = 0.02
DEFAULT_REFERENCE_FUSION_POWER_W = 5.0e8
DEFAULT_SINGLE_PASS_BURN_FRACTION = 0.01
DEFAULT_ENERGY_GRID_POINTS = 2_001
DEFAULT_DIGITIZATION_HALF_WIDTH = 0.003
BURN_STATE_OPERATOR_PHASE_RELATIVE_TOLERANCE = 1.0e-3
BURN_STATE_OPERATOR_CHANNEL_PROJECTION = (
    (1.0 / 6.0, 1.0 / 4.0, 1.0 / 12.0),
    (1.0 / 6.0, -1.0 / 4.0, 1.0 / 12.0),
    (1.0 / 6.0, 0.0, -1.0 / 6.0),
    (1.0 / 6.0, 0.0, -1.0 / 6.0),
    (1.0 / 6.0, -1.0 / 4.0, 1.0 / 12.0),
    (1.0 / 6.0, 1.0 / 4.0, 1.0 / 12.0),
)
BH_CROSS_SECTION_MIN_ENERGY_KEV = 0.5
BH_CROSS_SECTION_MAX_ENERGY_KEV = 550.0

HUPIN_DOI = "https://doi.org/10.1038/s41467-018-08052-6"
HUPIN_PMCID = "PMC6341121"
HUPIN_FIGURE_4_IMAGE_SHA256 = "7765d1e5a92fc90377972a8dd2aa885299d87f4a3e5cc9ee32847cc2fcd04fbf"
HUPIN_FIGURE_4_IMAGE_SIZE_PX = (781, 296)
HUPIN_FIGURE_4_REPOSITORY_RELATIVE_PATH = ".research-tmp/PMC6341121/41467_2018_8052_Fig4_HTML.jpg"
MAX_HUPIN_FIGURE_BYTES = 250_000
HUPIN_FIGURE_HASH_CHUNK_BYTES = 64 * 1024

# Figure 4b calibration for the 781 x 296 pixel CC BY image in the PMC OA
# package.  E_D[MeV] = (x - 460) / 187.5 and
# delta = 1.5 + (45 - y) / 1800.  The solid NCSMC-pheno full curve was sampled
# every 0.05 MeV at x = round(460 + 187.5 E_D).  At the dashed-curve crossing,
# continuity of the solid centerline fixes the selected pixel.  Keeping the
# source pixels rather than rounded delta values makes the transformation
# auditable.
HUPIN_FIG4B_DEUTERON_LAB_ENERGIES_MEV = tuple(0.05 * index for index in range(33))
HUPIN_FIG4B_FULL_CURVE_Y_PIXELS = (
    45,
    39,
    35,
    30,
    25,
    22,
    20,
    19,
    19,
    20,
    22,
    25,
    29,
    34,
    40,
    47,
    55,
    63,
    73,
    82,
    93,
    103,
    114,
    126,
    138,
    149,
    162,
    175,
    187,
    200,
    213,
    226,
    238,
)

COULTER_DOI = "https://doi.org/10.1103/PhysRevLett.68.174"
COULTER_CONTINUOUS_DEUTERIUM_RATE_S = 2.1e17
COULTER_DEUTERIUM_POLARIZATION = 0.73
COULTER_DEUTERIUM_POLARIZATION_STD = 0.03
ANKE_DOI = "https://doi.org/10.1016/j.nima.2013.03.043"
ANKE_DEUTERIUM_RATE_S = 3.9e16
ANKE_DEUTERIUM_RATE_STD_S = 0.2e16
ANKE_DEUTERON_VECTOR_POLARIZATION_ABS = 0.90
ANKE_DEUTERON_TENSOR_POLARIZATION = 0.86
ANKE_RF_DISSOCIATOR_POWER_W = 300.0
SOFIKITIS_DOI = "https://doi.org/10.1103/PhysRevLett.118.233401"
SOFIKITIS_NUCLEAR_DEUTERIUM_POLARIZATION = 0.60
SOFIKITIS_HYPERFINE_TRANSFER_TIME_NS = 1.6
UTSURO_DOI = "https://doi.org/10.1080/15361055.2022.2062098"
COOK_DOI = "https://doi.org/10.1088/1741-4326/ae5ab8"
COOK_SPARC_DEUTERIUM_MPLUS_RETENTION_PROBABILITY = 0.8905
COOK_SPARC_TRITIUM_DEPOLARIZATION_FRACTION = 0.13
COOK_SPARC_AGGREGATE_DEPOLARIZATION_FRACTION = 0.12
COOK_SPARC_AGGREGATE_DEPOLARIZATION_STD = 0.01
COOK_QUASI_STEADY_TIME_MS = 0.03
SOUERS_SOLID_DT_RELAXATION_DOI = "https://doi.org/10.13182/FST88-A25242"
COLLINS_SOLID_DT_NMR_DOI = "https://doi.org/10.1103/PhysRevB.46.695"
DRIES_DOI = "https://doi.org/10.1103/PhysRevC.21.475"
DRIES_EXFOR_URL = "https://www.jcprg.org/master/exfor/C/c0475.txt"
ARNOLD_DOI = "https://doi.org/10.1103/PhysRev.93.483"
ARNOLD_EXFOR_URL = "https://www.jcprg.org/master/exfor/C/c0018.txt"
HAN_RMATRIX_DOI = "https://doi.org/10.1007/s41365-025-01874-2"
HAN_SCIENCEDB_DOI = "https://doi.org/10.57760/sciencedb.j00186.00813"

EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION = "fusion-evidence-artifact-manifest-v1"
EVIDENCE_ARTIFACT_CLAIMS_SCHEMA_VERSION = "fusion-evidence-artifact-claims-v1"
POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS = (
    "polarized_dt_state_resolved_operator_v1"
)
REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS = (
    "reactor_rate_polarized_deuterium_source_v1"
)
REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS = (
    "reactor_rate_polarized_tritium_source_v1"
)
BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS = "burn_weighted_dt_retention_v1"
POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS = "polarized_dt_wall_plug_ledger_v1"
KNOWN_EVIDENCE_ARTIFACT_CLASSES = frozenset(
    (
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
    )
)
MAX_EVIDENCE_MANIFEST_BYTES = 1_000_000
MAX_EVIDENCE_CLAIMS_BYTES = 1_000_000
MAX_EVIDENCE_TYPED_PAYLOAD_BYTES = 16_000_000
MAX_EVIDENCE_MANIFEST_PAYLOADS = 256
EXPECTED_EVIDENCE_ARTIFACT_PAYLOAD_ROLES = {
    POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS: frozenset(
        ("claims", "operator_grid")
    ),
    REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS: frozenset(
        ("claims", "measurement", "event_counts", "calibration", "pipeline")
    ),
    REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS: frozenset(
        ("claims", "measurement", "event_counts", "calibration", "pipeline")
    ),
    BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS: frozenset(
        ("claims", "measurement", "burn_events", "calibration", "estimator")
    ),
    POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS: frozenset(
        (
            "claims",
            "component_ledger",
            "meter_records",
            "calibrations",
            "uncertainty_pipeline",
        )
    ),
}
# Success inputs are pinned here, never accepted from a submitted report.  The
# entries intentionally remain absent until reviewed primary evidence exists.
CANONICAL_EVIDENCE_ARTIFACT_MANIFESTS: tuple[
    tuple[str, str | None, str | None], ...
] = (
    (POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS, None, None),
    (REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS, None, None),
    (REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS, None, None),
    (BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS, None, None),
    (POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS, None, None),
)


@dataclass(frozen=True)
class EvidenceArtifactAudit:
    """Runtime verification record for one typed evidence manifest."""

    manifest_schema_version: str
    evidence_class: str
    manifest_repository_relative_path: str | None
    expected_manifest_sha256: str | None
    runtime_manifest_sha256: str | None
    manifest_file_available: bool
    manifest_sha256_verified: bool
    manifest_claim_keys: tuple[str, ...]
    manifest_claims_sha256: str | None
    typed_payload_derived: bool
    derived_claim_keys: tuple[str, ...]
    derived_claims_sha256: str | None
    derived_claims_canonical_json: str | None
    derived_plant_configuration_id: str | None
    derived_linked_manifest_sha256s: tuple[tuple[str, str], ...]
    derived_typed_payload_sha256s: tuple[tuple[str, str], ...]
    derived_claims_canonical_json_convention: str | None
    payload_roles: tuple[str, ...]
    payload_repository_relative_paths: tuple[str, ...]
    expected_payload_sha256_digests: tuple[str, ...]
    runtime_payload_sha256_digests: tuple[str, ...]
    payload_hashes_verified: bool
    runtime_artifact_gate_pass: bool
    status: str


@dataclass(frozen=True)
class PublishedPolarizedDTReactivityAudit:
    source_doi: str
    source_pmcid: str
    source_license: str
    source_figure: str
    expected_source_image_sha256: str
    expected_source_image_width_px: int
    expected_source_image_height_px: int
    source_image_repository_relative_path: str
    source_image_bundled_in_repository: bool
    source_image_runtime_sha256: str | None
    source_image_sha256_verified: bool
    source_image_runtime_width_px: int | None
    source_image_runtime_height_px: int | None
    source_image_dimensions_verified: bool
    digitization_method: str
    digitized_deuteron_lab_energies_mev: tuple[float, ...]
    digitized_full_alignment_enhancement_factors: tuple[float, ...]
    digitization_enhancement_half_width: float
    temperature_kev: float
    target_reactivity_ratio: float
    cross_section_energy_min_kev: float
    cross_section_energy_max_kev: float
    energy_grid_points: int
    deuteron_lab_to_centre_of_mass_energy_factor: float
    numerical_baseline_reactivity_cm3_s: float
    bosch_hale_closed_fit_reactivity_cm3_s: float
    numerical_to_closed_fit_ratio: float
    maxwellian_kernel_mode_energy_kev: float
    maxwellian_kernel_central_90_low_energy_kev: float
    maxwellian_kernel_central_90_high_energy_kev: float
    maxwellian_kernel_0p1_percent_energy_kev: float
    maxwellian_kernel_99p9_percent_energy_kev: float
    maxwellian_kernel_fraction_below_100_kev: float
    full_alignment_maxwellian_reactivity_ratio: float
    digitization_lower_maxwellian_reactivity_ratio: float
    grid_refinement_absolute_residual: float
    figure_digitized_energy_dependent_cross_section_integrated: bool
    figure_digitized_one_percent_control_pass: bool
    authors_published_energy_dependent_maxwellian_calculation: bool
    published_partial_deuteron_vector_polarization: float
    published_partial_deuteron_tensor_polarization: float
    published_partial_triton_vector_polarization: float
    published_approximate_partial_rate_ratio: float
    validated_target_deuteron_vector_polarization: float | None
    validated_target_deuteron_tensor_polarization: float | None
    validated_target_triton_vector_polarization: float | None
    validated_target_state_maxwellian_reactivity_ratio: float | None
    validated_target_state_maxwellian_reactivity_ratio_std: float | None
    validated_target_state_maxwellian_reactivity_ratio_lower_3sigma: float | None
    target_state_resolved_operator_available: bool
    target_state_operator_provenance_digest: str | None
    target_state_operator_artifact: EvidenceArtifactAudit
    author_machine_readable_energy_grid_available: bool
    author_s_matrix_or_angle_integrated_coefficients_available: bool
    nuclear_model_systematic_covariance_available: bool
    dries_source_doi: str
    dries_exfor_url: str
    dries_exfor_machine_readable_single_angle_azz_available: bool
    dries_exfor_point_count: int
    dries_initial_tritium_target_unpolarized: bool
    dries_record_is_not_double_polarized_czz_or_total_rate: bool
    arnold_source_doi: str
    arnold_exfor_url: str
    arnold_exfor_unpolarized_total_dt_cross_section_available: bool
    arnold_exfor_point_count: int
    han_rmatrix_source_doi: str
    han_sciencedb_doi: str
    han_sciencedb_public_metadata_available: bool
    han_sciencedb_license: str
    han_sciencedb_version: str
    han_sciencedb_metadata_declares_unpolarized_rmatrix_covariance: bool
    han_sciencedb_numeric_files_locally_verified: bool
    han_sciencedb_payload_integrity_gate_pass: bool
    han_sciencedb_numeric_covariance_matrix_available: bool
    han_sciencedb_initial_state_spin_operator_available: bool
    han_sciencedb_payload_audit: ScienceDBV1PayloadAudit
    han_sciencedb_unpolarized_reactivity_audit: ScienceDBDTReactivityAudit
    han_sciencedb_unpolarized_interpolation_spread_below_one_percent: bool
    han_sciencedb_unpolarized_sub_one_percent_certification_gate_pass: bool
    han_initial_double_polarized_state_operator_available: bool
    spin_correlation_czz_directly_measured: bool
    polarized_dt_rate_directly_validated: bool
    local_author_data_reproduction_pass: bool
    physical_reaction_evidence_gate_pass: bool
    status: str


@dataclass(frozen=True)
class PolarizedFuelPrimaryEvidenceAudit:
    declared_reference_fusion_power_w: float
    declared_single_pass_burn_fraction: float
    required_fusion_reaction_rate_s: float
    required_per_species_fuel_rate_s: float
    reference_plant_minimum_per_species_fuel_rate_s: float
    reference_plant_scale_invariant_pass: bool
    validated_integrated_plant_configuration_id: str | None
    coulter_source_doi: str
    coulter_measured_continuous_deuterium_rate_s: float
    coulter_measured_deuterium_polarization: float
    coulter_measured_deuterium_polarization_std: float
    required_to_coulter_rate_ratio: float
    coulter_record_meets_reactor_deuterium_rate: bool
    anke_source_doi: str
    anke_measured_deuterium_rate_s: float
    anke_measured_deuterium_rate_std_s: float
    anke_measured_deuteron_vector_polarization_abs: float
    anke_measured_deuteron_tensor_polarization: float
    anke_rf_dissociator_power_w: float
    anke_rf_power_is_partial_component_not_complete_wall_plug: bool
    anke_record_meets_reactor_deuterium_rate: bool
    sofikitis_source_doi: str
    sofikitis_nuclear_deuterium_polarization: float
    sofikitis_hyperfine_transfer_time_ns: float
    sofikitis_reactor_continuous_throughput_measured: bool
    utsuro_source_doi: str
    utsuro_hd_proxy_proof_of_concept_performed: bool
    utsuro_actual_polarized_tritium_source_demonstrated: bool
    utsuro_reactor_continuous_throughput_measured: bool
    cook_source_doi: str
    cook_sparc_deuterium_mplus_retention_probability: float
    cook_sparc_tritium_depolarization_fraction: float
    cook_sparc_aggregate_depolarization_fraction: float
    cook_sparc_aggregate_depolarization_std: float
    cook_quasi_steady_time_ms: float
    cook_result_is_simulation_not_burn_measurement: bool
    souers_source_doi: str
    souers_actual_solid_dt_triton_relaxation_measured: bool
    souers_result_is_preburn_not_burn_weighted_plasma_retention: bool
    collins_source_doi: str
    collins_actual_solid_dt_deuteron_nmr_relaxation_measured: bool
    collins_result_is_preburn_not_burn_weighted_plasma_retention: bool
    reactor_rate_polarized_deuterium_source_demonstrated: bool
    reactor_rate_polarized_tritium_source_demonstrated: bool
    validated_reactor_rate_deuterium_source_rate_s: float | None
    validated_reactor_rate_tritium_source_rate_s: float | None
    validated_reactor_rate_deuterium_source_rate_linearized_std_s: float | None
    validated_reactor_rate_tritium_source_rate_linearized_std_s: float | None
    validated_reactor_rate_deuterium_source_rate_lower_3sigma_s: float | None
    validated_reactor_rate_tritium_source_rate_lower_3sigma_s: float | None
    validated_reactor_rate_deuterium_source_rate_3sigma_bound_model: str | None
    validated_reactor_rate_tritium_source_rate_3sigma_bound_model: str | None
    validated_reactor_rate_deuterium_measurement_duration_s: float | None
    validated_reactor_rate_tritium_measurement_duration_s: float | None
    validated_reactor_rate_deuterium_measurement_duration_std_s: float | None
    validated_reactor_rate_tritium_measurement_duration_std_s: float | None
    validated_reactor_rate_deuterium_output_count: float | None
    validated_reactor_rate_tritium_output_count: float | None
    validated_reactor_rate_deuterium_output_count_std: float | None
    validated_reactor_rate_tritium_output_count_std: float | None
    validated_reactor_rate_deuterium_background_count: float | None
    validated_reactor_rate_tritium_background_count: float | None
    validated_reactor_rate_deuterium_background_count_std: float | None
    validated_reactor_rate_tritium_background_count_std: float | None
    validated_reactor_rate_deuterium_detection_efficiency: float | None
    validated_reactor_rate_tritium_detection_efficiency: float | None
    validated_reactor_rate_deuterium_detection_efficiency_std: float | None
    validated_reactor_rate_tritium_detection_efficiency_std: float | None
    validated_reactor_rate_deuterium_uncertainty_correlation_matrix: (
        tuple[tuple[float, ...], ...] | None
    )
    validated_reactor_rate_tritium_uncertainty_correlation_matrix: (
        tuple[tuple[float, ...], ...] | None
    )
    validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix: (
        tuple[tuple[float, ...], ...] | None
    )
    validated_reactor_rate_deuteron_vector_polarization: float | None
    validated_reactor_rate_deuteron_vector_polarization_std: float | None
    validated_reactor_rate_deuteron_tensor_polarization: float | None
    validated_reactor_rate_deuteron_tensor_polarization_std: float | None
    validated_reactor_rate_triton_vector_polarization: float | None
    validated_reactor_rate_triton_vector_polarization_std: float | None
    validated_reactor_rate_deuterium_source_provenance_digest: str | None
    validated_reactor_rate_tritium_source_provenance_digest: str | None
    validated_reactor_rate_deuterium_source_artifact: EvidenceArtifactAudit
    validated_reactor_rate_tritium_source_artifact: EvidenceArtifactAudit
    burn_weighted_dt_polarization_product_measured: bool
    validated_burn_weighted_dt_polarization_product: float | None
    validated_burn_weighted_dt_polarization_product_std: float | None
    validated_burn_weighted_deuteron_tensor_polarization: float | None
    validated_burn_weighted_deuteron_tensor_polarization_std: float | None
    validated_burn_retention_measurement_duration_s: float | None
    validated_burn_retention_measurement_duration_std_s: float | None
    validated_burn_retention_reaction_count: float | None
    validated_burn_retention_uncertainty_correlation_matrix: (
        tuple[tuple[float, ...], ...] | None
    )
    validated_burn_retention_observable_semantics: str | None
    validated_burn_reaction_operator_provenance_digest: str | None
    validated_burn_operator_uncertainty_relationship: str | None
    validated_burn_state_operator_reactivity_ratio: float | None
    validated_burn_state_operator_reactivity_ratio_robust_margin_per_sigma_equivalent: (
        float | None
    )
    validated_burn_state_operator_reactivity_ratio_robust_lower_3sigma: float | None
    validated_burn_state_operator_max_phase_relative_residual: float | None
    validated_burn_state_operator_contraction_semantics: str | None
    burn_state_operator_componentwise_3sigma_stress_envelope_control_pass: bool
    energy_resolved_burn_operator_sufficient_statistics_available: bool
    validated_burn_retention_provenance_digest: str | None
    validated_burn_retention_artifact: EvidenceArtifactAudit
    complete_wall_plug_energy_per_injected_pair_measured: bool
    complete_wall_plug_uncertainty_flow_and_provenance_available: bool
    validated_wall_plug_energy_per_injected_dt_pair_ev: float | None
    validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev: float | None
    validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev: float | None
    validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev: float | None
    validated_wall_plug_total_energy_j: float | None
    validated_wall_plug_total_energy_linearized_std_j: float | None
    validated_wall_plug_total_energy_lower_3sigma_j: float | None
    validated_wall_plug_total_energy_upper_3sigma_j: float | None
    validated_wall_plug_measurement_duration_s: float | None
    validated_wall_plug_measurement_duration_std_s: float | None
    validated_wall_plug_measurement_injected_dt_pair_count: float | None
    validated_wall_plug_measurement_injected_dt_pair_count_std: float | None
    validated_wall_plug_measurement_injected_dt_pair_flow_rate_s: float | None
    validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s: (
        float | None
    )
    validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s: float | None
    validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s: float | None
    validated_wall_plug_3sigma_bound_model: str | None
    validated_wall_plug_uncertainty_correlation_matrix: (
        tuple[tuple[float, ...], ...] | None
    )
    validated_wall_plug_measurement_provenance: str | None
    validated_wall_plug_artifact: EvidenceArtifactAudit
    source_retention_and_wall_plug_gate_pass: bool
    status: str


@dataclass(frozen=True)
class FusionPolarizedEvidenceReport:
    schema_version: str
    reaction: PublishedPolarizedDTReactivityAudit
    source: PolarizedFuelPrimaryEvidenceAudit
    energy_dependent_figure_control_reproduced: bool
    physical_reaction_evidence_gate_pass: bool
    physical_source_evidence_gate_pass: bool
    burn_state_operator_componentwise_3sigma_stress_envelope_control_pass: bool
    energy_resolved_burn_operator_sufficient_statistics_available: bool
    validated_burn_state_operator_reactivity_ratio: float | None
    validated_burn_state_operator_reactivity_ratio_robust_margin_per_sigma_equivalent: (
        float | None
    )
    validated_burn_state_operator_reactivity_ratio_robust_lower_3sigma: float | None
    validated_burn_state_operator_max_phase_relative_residual: float | None
    physical_spin_fusion_evidence_gate_pass: bool
    maximum_supported_stage: str
    next_required_input: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _BurnStateOperatorContraction:
    central_reactivity_ratio: float
    robust_margin_per_sigma_equivalent: float
    robust_lower_3sigma: float
    full_grid_robust_lower_3sigma: float
    coarsened_robust_lower_3sigma: float
    max_phase_relative_residual: float
    evaluated_model_count: int
    evaluated_corner_count: int
    semantics: str
    control_passes_one_percent: bool


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


def _grid_count(value: Integral, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 101:
        raise ValueError(f"{name} must be at least 101")
    return result


def _sha256_provenance_digest_available(value: object) -> bool:
    """Require a content-addressed evidence artifact, not a citation label."""

    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


class _DuplicateEvidenceManifestKey(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateEvidenceManifestKey(key)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json_sha256(value: object) -> str | None:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (
        TypeError,
        ValueError,
        UnicodeEncodeError,
        RecursionError,
        OverflowError,
        MemoryError,
    ):
        return None
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _current_repository_root() -> Path | None:
    """Locate this checkout without trusting the process working directory."""

    for ancestor in Path(__file__).resolve().parents:
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "reality_stone").is_dir():
            return ancestor.resolve()
    return None


def _path_is_link_or_reparse_point(path: Path) -> bool:
    try:
        is_junction = getattr(path, "is_junction", lambda: False)
        file_attributes = getattr(os.lstat(path), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return path.is_symlink() or is_junction() or bool(file_attributes & reparse_flag)
    except OSError:
        return True


def _strict_repository_relative_file(
    repository_root: Path,
    repository_relative_path: object,
) -> Path | None:
    """Resolve one canonical POSIX-style repository path without following links."""

    if not isinstance(repository_relative_path, str) or not repository_relative_path:
        return None
    if "\\" in repository_relative_path or "\x00" in repository_relative_path:
        return None
    posix_path = PurePosixPath(repository_relative_path)
    windows_path = PureWindowsPath(repository_relative_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or posix_path.as_posix() != repository_relative_path
        or any(part in ("", ".", "..") or ":" in part for part in posix_path.parts)
    ):
        return None

    try:
        root = repository_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    raw_candidate = root.joinpath(*posix_path.parts)
    cursor = root
    for part in posix_path.parts:
        cursor = cursor / part
        if _path_is_link_or_reparse_point(cursor):
            return None
    try:
        candidate = raw_candidate.resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _missing_evidence_artifact(evidence_class: str) -> EvidenceArtifactAudit:
    return EvidenceArtifactAudit(
        manifest_schema_version=EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        evidence_class=evidence_class,
        manifest_repository_relative_path=None,
        expected_manifest_sha256=None,
        runtime_manifest_sha256=None,
        manifest_file_available=False,
        manifest_sha256_verified=False,
        manifest_claim_keys=(),
        manifest_claims_sha256=None,
        typed_payload_derived=False,
        derived_claim_keys=(),
        derived_claims_sha256=None,
        derived_claims_canonical_json=None,
        derived_plant_configuration_id=None,
        derived_linked_manifest_sha256s=(),
        derived_typed_payload_sha256s=(),
        derived_claims_canonical_json_convention=None,
        payload_roles=(),
        payload_repository_relative_paths=(),
        expected_payload_sha256_digests=(),
        runtime_payload_sha256_digests=(),
        payload_hashes_verified=False,
        runtime_artifact_gate_pass=False,
        status="MANIFEST_NOT_DECLARED",
    )


def _canonical_evidence_artifact(evidence_class: str) -> EvidenceArtifactAudit:
    for pinned_class, manifest_path, manifest_sha256 in CANONICAL_EVIDENCE_ARTIFACT_MANIFESTS:
        if pinned_class == evidence_class:
            if manifest_path is None and manifest_sha256 is None:
                return _missing_evidence_artifact(evidence_class)
            return audit_repository_evidence_artifact(
                evidence_class=evidence_class,
                manifest_repository_relative_path=manifest_path,
                expected_manifest_sha256=manifest_sha256,
            )
    return _missing_evidence_artifact(evidence_class)


def audit_repository_evidence_artifact(
    *,
    evidence_class: str,
    manifest_repository_relative_path: str | None,
    expected_manifest_sha256: str | None,
    repository_root: str | Path | None = None,
) -> EvidenceArtifactAudit:
    """Hash a typed manifest and every payload from a contained repository path."""

    recorded_class = evidence_class if isinstance(evidence_class, str) else ""
    recorded_path = (
        manifest_repository_relative_path
        if isinstance(manifest_repository_relative_path, str)
        else None
    )
    recorded_expected_hash = (
        expected_manifest_sha256 if isinstance(expected_manifest_sha256, str) else None
    )
    runtime_manifest_sha256: str | None = None
    manifest_file_available = False
    manifest_sha256_verified = False
    manifest_claim_keys: tuple[str, ...] = ()
    manifest_claims_sha256: str | None = None
    typed_payload_derived = False
    derived_claim_keys: tuple[str, ...] = ()
    derived_claims_sha256: str | None = None
    derived_claims_canonical_json: str | None = None
    derived_plant_configuration_id: str | None = None
    derived_linked_manifest_sha256s: tuple[tuple[str, str], ...] = ()
    derived_typed_payload_sha256s: tuple[tuple[str, str], ...] = ()
    derived_claims_canonical_json_convention: str | None = None
    payload_roles: list[str] = []
    payload_paths: list[str] = []
    expected_payload_hashes: list[str] = []
    runtime_payload_hashes: list[str] = []

    def result(status: str, *, passed: bool = False) -> EvidenceArtifactAudit:
        return EvidenceArtifactAudit(
            manifest_schema_version=EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            evidence_class=recorded_class,
            manifest_repository_relative_path=recorded_path,
            expected_manifest_sha256=recorded_expected_hash,
            runtime_manifest_sha256=runtime_manifest_sha256,
            manifest_file_available=manifest_file_available,
            manifest_sha256_verified=manifest_sha256_verified,
            manifest_claim_keys=manifest_claim_keys,
            manifest_claims_sha256=manifest_claims_sha256,
            typed_payload_derived=typed_payload_derived,
            derived_claim_keys=derived_claim_keys,
            derived_claims_sha256=derived_claims_sha256,
            derived_claims_canonical_json=derived_claims_canonical_json,
            derived_plant_configuration_id=derived_plant_configuration_id,
            derived_linked_manifest_sha256s=derived_linked_manifest_sha256s,
            derived_typed_payload_sha256s=derived_typed_payload_sha256s,
            derived_claims_canonical_json_convention=(
                derived_claims_canonical_json_convention
            ),
            payload_roles=tuple(payload_roles),
            payload_repository_relative_paths=tuple(payload_paths),
            expected_payload_sha256_digests=tuple(expected_payload_hashes),
            runtime_payload_sha256_digests=tuple(runtime_payload_hashes),
            payload_hashes_verified=passed,
            runtime_artifact_gate_pass=passed,
            status=status,
        )

    if recorded_class not in KNOWN_EVIDENCE_ARTIFACT_CLASSES:
        return result("UNKNOWN_EVIDENCE_CLASS")
    if recorded_path is None or recorded_expected_hash is None:
        return result("MANIFEST_NOT_DECLARED")
    if not _sha256_provenance_digest_available(recorded_expected_hash):
        return result("INVALID_EXPECTED_MANIFEST_SHA256")

    if repository_root is None:
        root = _current_repository_root()
    else:
        try:
            root = Path(repository_root)
        except TypeError:
            root = None
    if root is None:
        return result("REPOSITORY_ROOT_NOT_FOUND")

    manifest_path = _strict_repository_relative_file(root, recorded_path)
    if manifest_path is None:
        return result("MANIFEST_PATH_INVALID_OR_MISSING")
    manifest_file_available = True
    try:
        if manifest_path.stat().st_size > MAX_EVIDENCE_MANIFEST_BYTES:
            return result("MANIFEST_TOO_LARGE")
        manifest_bytes = manifest_path.read_bytes()
        runtime_manifest_sha256 = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    except OSError:
        return result("MANIFEST_READ_FAILED")
    manifest_sha256_verified = hmac.compare_digest(
        runtime_manifest_sha256,
        recorded_expected_hash,
    )
    if not manifest_sha256_verified:
        return result("MANIFEST_SHA256_MISMATCH")

    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError, MemoryError):
        return result("MANIFEST_JSON_INVALID_OR_DUPLICATE_KEY")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "evidence_class",
        "claims_payload_repository_relative_path",
        "payloads",
    }:
        return result("MANIFEST_SCHEMA_KEYS_INVALID")
    if manifest.get("schema_version") != EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        return result("MANIFEST_SCHEMA_VERSION_MISMATCH")
    if manifest.get("evidence_class") != recorded_class:
        return result("MANIFEST_EVIDENCE_CLASS_MISMATCH")
    claims_payload_path_value = manifest.get("claims_payload_repository_relative_path")
    if not isinstance(claims_payload_path_value, str):
        return result("CLAIMS_PAYLOAD_PATH_INVALID")
    payloads = manifest.get("payloads")
    if (
        not isinstance(payloads, list)
        or not payloads
        or len(payloads) > MAX_EVIDENCE_MANIFEST_PAYLOADS
    ):
        return result("MANIFEST_PAYLOAD_COUNT_INVALID")

    resolved_payloads: set[str] = set()
    claims_payload_bytes: bytes | None = None
    typed_payload_bytes_by_role: dict[str, bytes] = {}
    typed_payload_sha256s_by_role: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, dict) or set(payload) != {
            "role",
            "repository_relative_path",
            "sha256",
        }:
            return result("PAYLOAD_SCHEMA_KEYS_INVALID")
        payload_role = payload.get("role")
        payload_path_value = payload.get("repository_relative_path")
        expected_payload_hash = payload.get("sha256")
        if not isinstance(payload_role, str) or not payload_role:
            return result("PAYLOAD_ROLE_INVALID")
        if payload_role in payload_roles:
            return result("DUPLICATE_PAYLOAD_ROLE")
        payload_roles.append(payload_role)
        if not isinstance(payload_path_value, str):
            return result("PAYLOAD_PATH_INVALID_OR_MISSING")
        payload_paths.append(payload_path_value)
        if not isinstance(expected_payload_hash, str):
            return result("PAYLOAD_SHA256_INVALID")
        if not _sha256_provenance_digest_available(expected_payload_hash):
            return result("PAYLOAD_SHA256_INVALID")
        if expected_payload_hash in expected_payload_hashes:
            return result("DUPLICATE_PAYLOAD_SHA256")
        expected_payload_hashes.append(expected_payload_hash)
        payload_path = _strict_repository_relative_file(root, payload_path_value)
        if payload_path is None or payload_path == manifest_path:
            return result("PAYLOAD_PATH_INVALID_OR_MISSING")
        resolved_key = os.path.normcase(str(payload_path))
        if resolved_key in resolved_payloads:
            return result("DUPLICATE_PAYLOAD_PATH")
        resolved_payloads.add(resolved_key)
        try:
            if payload_role == "claims":
                if payload_path.stat().st_size > MAX_EVIDENCE_CLAIMS_BYTES:
                    return result("CLAIMS_PAYLOAD_TOO_LARGE")
                claims_payload_bytes = payload_path.read_bytes()
                runtime_payload_hash = (
                    "sha256:" + hashlib.sha256(claims_payload_bytes).hexdigest()
                )
            else:
                if payload_path.stat().st_size > MAX_EVIDENCE_TYPED_PAYLOAD_BYTES:
                    return result("TYPED_PAYLOAD_TOO_LARGE")
                typed_payload_bytes = payload_path.read_bytes()
                runtime_payload_hash = (
                    "sha256:" + hashlib.sha256(typed_payload_bytes).hexdigest()
                )
                typed_payload_bytes_by_role[payload_role] = typed_payload_bytes
                typed_payload_sha256s_by_role[payload_role] = runtime_payload_hash
        except OSError:
            return result("PAYLOAD_READ_FAILED")
        runtime_payload_hashes.append(runtime_payload_hash)
        if not hmac.compare_digest(runtime_payload_hash, expected_payload_hash):
            return result("PAYLOAD_SHA256_MISMATCH")

    expected_roles = EXPECTED_EVIDENCE_ARTIFACT_PAYLOAD_ROLES[recorded_class]
    if frozenset(payload_roles) != expected_roles:
        return result("PAYLOAD_ROLE_SET_MISMATCH")
    if claims_payload_path_value not in payload_paths:
        return result("CLAIMS_PAYLOAD_LINK_MISMATCH")
    claims_payload_index = payload_paths.index(claims_payload_path_value)
    if payload_roles[claims_payload_index] != "claims":
        return result("CLAIMS_PAYLOAD_LINK_MISMATCH")
    if claims_payload_bytes is None:
        return result("CLAIMS_PAYLOAD_MISSING")
    expected_typed_roles = expected_roles - frozenset(("claims",))
    if frozenset(typed_payload_bytes_by_role) != expected_typed_roles:
        return result("TYPED_PAYLOAD_MISSING")
    try:
        claims_document = json.loads(
            claims_payload_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError, MemoryError):
        return result("CLAIMS_PAYLOAD_JSON_INVALID_OR_DUPLICATE_KEY")
    if not isinstance(claims_document, dict) or set(claims_document) != {
        "schema_version",
        "evidence_class",
        "claims",
    }:
        return result("CLAIMS_PAYLOAD_SCHEMA_KEYS_INVALID")
    if claims_document.get("schema_version") != EVIDENCE_ARTIFACT_CLAIMS_SCHEMA_VERSION:
        return result("CLAIMS_PAYLOAD_SCHEMA_VERSION_MISMATCH")
    if claims_document.get("evidence_class") != recorded_class:
        return result("CLAIMS_PAYLOAD_EVIDENCE_CLASS_MISMATCH")
    claims = claims_document.get("claims")
    if not isinstance(claims, dict) or not claims:
        return result("CLAIMS_PAYLOAD_CLAIMS_INVALID")
    manifest_claim_keys = tuple(sorted(claims))
    manifest_claims_sha256 = _canonical_json_sha256(claims)
    if manifest_claims_sha256 is None:
        return result("CLAIMS_PAYLOAD_CLAIMS_INVALID")

    typed_payloads: dict[str, dict[str, Any]] = {}
    for role, typed_payload_bytes in typed_payload_bytes_by_role.items():
        try:
            typed_payload = json.loads(
                typed_payload_bytes.decode("utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, ValueError, RecursionError, OverflowError, MemoryError):
            return result("TYPED_PAYLOAD_JSON_INVALID_OR_DUPLICATE_KEY")
        if not isinstance(typed_payload, dict):
            return result("TYPED_PAYLOAD_SCHEMA_INVALID")
        if typed_payload.get("payload_role") != role:
            return result("TYPED_PAYLOAD_ROLE_MISMATCH")
        typed_payloads[role] = typed_payload
    primary_role = ENGINEERING_PRIMARY_PAYLOAD_ROLES.get(recorded_class, "operator_grid")
    if primary_role not in typed_payloads:
        return result("TYPED_PAYLOAD_MISSING")
    primary_payload = typed_payloads[primary_role]
    auxiliary_payloads = {
        role: document
        for role, document in typed_payloads.items()
        if role != primary_role
    }
    try:
        derived = derive_artifact_claims(
            recorded_class,
            primary_payload,
            manifest_sha256=recorded_expected_hash,
            auxiliary_payloads=auxiliary_payloads,
            typed_payload_sha256s=typed_payload_sha256s_by_role,
        )
    except (ArtifactPayloadError, ArithmeticError, MemoryError, np.linalg.LinAlgError):
        return result("TYPED_PAYLOAD_DERIVATION_FAILED")
    typed_payload_derived = True
    derived_claim_keys = tuple(sorted(derived.claims))
    derived_claims_sha256 = derived.claims_sha256
    derived_claims_canonical_json = derived.claims_canonical_json
    derived_plant_configuration_id = derived.plant_configuration_id
    derived_linked_manifest_sha256s = derived.linked_manifest_sha256s
    derived_typed_payload_sha256s = tuple(sorted(typed_payload_sha256s_by_role.items()))
    derived_claims_canonical_json_convention = derived.canonical_json_convention
    if recorded_class in ENGINEERING_AUXILIARY_PAYLOAD_ROLES and (
        derived.typed_payload_sha256s != derived_typed_payload_sha256s
    ):
        return result("TYPED_PAYLOAD_SHA256_MAP_MISMATCH")
    if derived_claims_canonical_json_convention != PYTHON_CANONICAL_JSON_CONVENTION:
        return result("DERIVED_CLAIMS_CANONICAL_CONVENTION_MISMATCH")
    if manifest_claim_keys != derived_claim_keys:
        return result("CLAIMS_PAYLOAD_NOT_DERIVED")
    if claims != derived.claims or not hmac.compare_digest(
        manifest_claims_sha256,
        derived_claims_sha256,
    ):
        return result("CLAIMS_PAYLOAD_NOT_DERIVED")

    return result("VERIFIED", passed=True)


def repository_evidence_artifact_integrity_pass(
    audit: EvidenceArtifactAudit,
    *,
    repository_root: str | Path | None = None,
) -> bool:
    """Re-read caller-scoped bytes; this is integrity, not canonical evidence."""

    if type(audit) is not EvidenceArtifactAudit or not audit.runtime_artifact_gate_pass:
        return False
    refreshed = audit_repository_evidence_artifact(
        evidence_class=audit.evidence_class,
        manifest_repository_relative_path=audit.manifest_repository_relative_path,
        expected_manifest_sha256=audit.expected_manifest_sha256,
        repository_root=repository_root,
    )
    return refreshed.runtime_artifact_gate_pass and refreshed == audit


def physical_canonical_repository_evidence_artifact_gate_pass(
    audit: EvidenceArtifactAudit,
) -> bool:
    """Accept only an artifact reloaded from a code-pinned canonical slot."""

    if type(audit) is not EvidenceArtifactAudit:
        return False
    pinned = tuple(
        entry for entry in CANONICAL_EVIDENCE_ARTIFACT_MANIFESTS if entry[0] == audit.evidence_class
    )
    if len(pinned) != 1:
        return False
    _, manifest_path, manifest_sha256 = pinned[0]
    if manifest_path is None or manifest_sha256 is None:
        return False
    canonical = audit_repository_evidence_artifact(
        evidence_class=audit.evidence_class,
        manifest_repository_relative_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
    )
    return canonical.runtime_artifact_gate_pass and audit == canonical


def _typed_evidence_artifact_gate_pass(
    artifact: object,
    *,
    evidence_class: str,
    provenance_digest: object,
    expected_claims: dict[str, object],
    repository_root: str | Path | None = None,
) -> bool:
    if type(artifact) is not EvidenceArtifactAudit:
        return False
    try:
        derived_claims = json.loads(
            artifact.derived_claims_canonical_json
            if isinstance(artifact.derived_claims_canonical_json, str)
            else "null",
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (ValueError, TypeError, RecursionError, OverflowError, MemoryError):
        return False
    if not isinstance(derived_claims, dict):
        return False
    if len(artifact.payload_roles) != len(artifact.expected_payload_sha256_digests):
        return False
    manifest_role_hashes = {
        role: digest
        for role, digest in zip(
            artifact.payload_roles, artifact.expected_payload_sha256_digests
        )
        if role != "claims"
    }
    if (
        artifact.derived_typed_payload_sha256s
        != tuple(sorted(manifest_role_hashes.items()))
        or artifact.derived_claims_canonical_json_convention
        != PYTHON_CANONICAL_JSON_CONVENTION
    ):
        return False
    unsupported_direct_claims: set[str] = set()
    if evidence_class == POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS:
        # These are the only reaction fields proved by the numeric operator
        # payload.  In particular, a grid cannot prove authorship, a direct
        # Czz measurement, or a direct polarized-rate measurement.
        key_map = {
            "temperature_kev": "temperature_kev",
            "target_reactivity_ratio": "target_reactivity_ratio",
            "validated_target_deuteron_vector_polarization": (
                "validated_target_deuteron_vector_polarization"
            ),
            "validated_target_deuteron_tensor_polarization": (
                "validated_target_deuteron_tensor_polarization"
            ),
            "validated_target_triton_vector_polarization": (
                "validated_target_triton_vector_polarization"
            ),
            "validated_target_state_maxwellian_reactivity_ratio": (
                "validated_target_state_maxwellian_reactivity_ratio"
            ),
            "validated_target_state_maxwellian_reactivity_ratio_std": (
                "validated_target_state_maxwellian_reactivity_ratio_std"
            ),
            "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma": (
                "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"
            ),
            "target_state_resolved_operator_available": (
                "target_state_resolved_operator_available"
            ),
            "author_machine_readable_energy_grid_available": (
                "machine_readable_energy_grid_available"
            ),
            "nuclear_model_systematic_covariance_available": (
                "systematic_covariance_available"
            ),
        }
        unsupported_direct_claims = {
            "author_s_matrix_or_angle_integrated_coefficients_available",
            "spin_correlation_czz_directly_measured",
            "polarized_dt_rate_directly_validated",
            "local_author_data_reproduction_pass",
        }
        if any(key in derived_claims for key in unsupported_direct_claims):
            return False
    else:
        key_map = {key: key for key in expected_claims}
    if any(key not in key_map for key in expected_claims if key not in unsupported_direct_claims):
        return False
    expected_subset = {
        derived_key: expected_claims[audit_key]
        for audit_key, derived_key in key_map.items()
        if audit_key in expected_claims
    }
    if any(key not in derived_claims for key in expected_subset):
        return False
    derived_subset = {key: derived_claims[key] for key in expected_subset}
    subset_matches = (
        _canonical_json_sha256(expected_subset) is not None
        and _canonical_json_sha256(expected_subset) == _canonical_json_sha256(derived_subset)
    )
    return (
        artifact.evidence_class == evidence_class
        and artifact.expected_manifest_sha256 == provenance_digest
        and artifact.typed_payload_derived
        and artifact.manifest_claim_keys == artifact.derived_claim_keys
        and artifact.manifest_claims_sha256 == artifact.derived_claims_sha256
        and artifact.derived_claims_sha256 == _canonical_json_sha256(derived_claims)
        and subset_matches
        and _sha256_provenance_digest_available(provenance_digest)
        and repository_evidence_artifact_integrity_pass(
            artifact,
            repository_root=repository_root,
        )
    )


def _plant_configuration_id_available(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all(character.isascii() and (character.isalnum() or character in "._-") for character in value)
    )


def _distinct_artifact_storage_pass(
    *artifacts: object,
    repository_root: str | Path | None = None,
) -> bool:
    if not artifacts or not all(type(artifact) is EvidenceArtifactAudit for artifact in artifacts):
        return False
    typed_artifacts = tuple(artifact for artifact in artifacts if type(artifact) is EvidenceArtifactAudit)
    manifest_paths = tuple(artifact.manifest_repository_relative_path for artifact in typed_artifacts)
    manifest_hashes = tuple(artifact.expected_manifest_sha256 for artifact in typed_artifacts)
    payload_paths = tuple(
        path
        for artifact in typed_artifacts
        for path in artifact.payload_repository_relative_paths
    )
    payload_hashes = tuple(
        digest
        for artifact in typed_artifacts
        for digest in artifact.expected_payload_sha256_digests
    )
    logical_identity_pass = all(
        (
            all(isinstance(path, str) and path for path in manifest_paths),
            all(_sha256_provenance_digest_available(value) for value in manifest_hashes),
            len(set(manifest_paths)) == len(manifest_paths),
            len(set(manifest_hashes)) == len(manifest_hashes),
            bool(payload_paths),
            len(set(payload_paths)) == len(payload_paths),
            len(payload_hashes) == len(payload_paths),
            all(_sha256_provenance_digest_available(value) for value in payload_hashes),
            len(set(payload_hashes)) == len(payload_hashes),
        )
    )
    if not logical_identity_pass:
        return False
    if repository_root is None:
        root = _current_repository_root()
    else:
        try:
            root = Path(repository_root)
        except TypeError:
            root = None
    if root is None:
        return False
    all_paths = tuple(manifest_paths) + payload_paths
    resolved_paths = tuple(
        _strict_repository_relative_file(root, relative_path) for relative_path in all_paths
    )
    if any(path is None for path in resolved_paths):
        return False
    physical_paths = tuple(path for path in resolved_paths if path is not None)
    normalized_paths = tuple(os.path.normcase(str(path)) for path in physical_paths)
    if len(set(normalized_paths)) != len(normalized_paths):
        return False
    try:
        return not any(
            os.path.samefile(left, right)
            for index, left in enumerate(physical_paths)
            for right in physical_paths[index + 1 :]
        )
    except OSError:
        return False


def _digitized_full_curve() -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(HUPIN_FIG4B_DEUTERON_LAB_ENERGIES_MEV, dtype=float)
    y_pixels = np.asarray(HUPIN_FIG4B_FULL_CURVE_Y_PIXELS, dtype=float)
    enhancements = 1.5 + (45.0 - y_pixels) / 1800.0
    return energies, enhancements


def _integrate_digitized_curve(
    *,
    temperature_kev: float,
    energy_grid_points: int,
    digitization_half_width: float,
) -> tuple[float, float, float, tuple[float, float, float, float, float, float]]:
    relative_energies = np.geomspace(
        BH_CROSS_SECTION_MIN_ENERGY_KEV,
        BH_CROSS_SECTION_MAX_ENERGY_KEV,
        energy_grid_points,
    )
    cross_sections = np.asarray(
        [bosch_hale_dt_cross_section_m2(float(energy)) for energy in relative_energies]
    )
    weights = cross_sections * relative_energies * np.exp(-relative_energies / temperature_kev)
    baseline_integral = float(np.trapezoid(weights, relative_energies))
    if not math.isfinite(baseline_integral) or baseline_integral <= 0.0:
        raise RuntimeError("Bosch-Hale Maxwellian baseline integral vanished")

    lab_to_cm = TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
    deuteron_lab_energies_mev = relative_energies / (1.0e3 * lab_to_cm)
    curve_energies, curve_enhancements = _digitized_full_curve()
    if (
        deuteron_lab_energies_mev[0] < curve_energies[0]
        or deuteron_lab_energies_mev[-1] > curve_energies[-1]
    ):
        raise RuntimeError("Maxwellian grid leaves the digitized Figure 4b energy domain")
    enhancements = np.interp(
        deuteron_lab_energies_mev,
        curve_energies,
        curve_enhancements,
    )
    lower_enhancements = enhancements - digitization_half_width
    if np.any(lower_enhancements <= 0.0):
        raise ValueError("digitization_half_width makes the enhancement nonpositive")
    full_ratio = float(np.trapezoid(weights * enhancements, relative_energies) / baseline_integral)
    lower_ratio = float(
        np.trapezoid(weights * lower_enhancements, relative_energies) / baseline_integral
    )

    reduced_mass_kev = (
        1.0e3 * DEUTERON_MASS_MEV * TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
    )
    numerical_reactivity_m3_s = (
        math.sqrt(8.0 / (math.pi * reduced_mass_kev))
        * baseline_integral
        / temperature_kev**1.5
        * SPEED_OF_LIGHT_M_S
    )
    trapezoids = 0.5 * (weights[1:] + weights[:-1]) * np.diff(relative_energies)
    cumulative = np.concatenate(([0.0], np.cumsum(trapezoids))) / baseline_integral

    def quantile(probability: float) -> float:
        return float(np.interp(probability, cumulative, relative_energies))

    kernel_summary = (
        float(relative_energies[int(np.argmax(weights))]),
        quantile(0.05),
        quantile(0.95),
        quantile(0.001),
        quantile(0.999),
        float(np.interp(100.0, relative_energies, cumulative)),
    )
    return (
        full_ratio,
        lower_ratio,
        numerical_reactivity_m3_s * 1.0e6,
        kernel_summary,
    )


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read JPEG SOF dimensions without adding an image-library dependency."""

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    position = 2
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while position + 3 < len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            return None
        marker = data[position]
        position += 1
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            return None
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            return None
        if marker in start_of_frame_markers:
            if segment_length < 7:
                return None
            height = int.from_bytes(data[position + 3 : position + 5], "big")
            width = int.from_bytes(data[position + 5 : position + 7], "big")
            return (width, height) if width > 0 and height > 0 else None
        position += segment_length
    return None


def _bundled_hupin_figure_sha256(
) -> tuple[bool, str | None, bool, int | None, int | None, bool]:
    """Hash and dimension-check the tracked source image inside this checkout."""

    root = _current_repository_root()
    if root is None:
        return False, None, False, None, None, False
    candidate = _strict_repository_relative_file(root, HUPIN_FIGURE_4_REPOSITORY_RELATIVE_PATH)
    if candidate is None:
        return False, None, False, None, None, False
    try:
        expected_size = candidate.stat().st_size
        if expected_size <= 0 or expected_size > MAX_HUPIN_FIGURE_BYTES:
            return True, None, False, None, None, False
        digest_state = hashlib.sha256()
        image_buffer = bytearray()
        with candidate.open("rb") as stream:
            while chunk := stream.read(HUPIN_FIGURE_HASH_CHUNK_BYTES):
                if len(image_buffer) + len(chunk) > MAX_HUPIN_FIGURE_BYTES:
                    return True, None, False, None, None, False
                digest_state.update(chunk)
                image_buffer.extend(chunk)
        if len(image_buffer) != expected_size:
            return True, None, False, None, None, False
        image_bytes = bytes(image_buffer)
    except (OSError, OverflowError, MemoryError):
        return True, None, False, None, None, False
    digest = digest_state.hexdigest()
    dimensions = _jpeg_dimensions(image_bytes)
    width, height = dimensions if dimensions is not None else (None, None)
    return (
        True,
        digest,
        hmac.compare_digest(digest, HUPIN_FIGURE_4_IMAGE_SHA256),
        width,
        height,
        dimensions == HUPIN_FIGURE_4_IMAGE_SIZE_PX,
    )


def _fresh_hupin_image_gate_pass(audit: PublishedPolarizedDTReactivityAudit) -> bool:
    bundled, digest, hash_verified, width, height, dimensions_verified = (
        _bundled_hupin_figure_sha256()
    )
    return all(
        (
            bundled is audit.source_image_bundled_in_repository,
            digest == audit.source_image_runtime_sha256,
            hash_verified is audit.source_image_sha256_verified,
            width == audit.source_image_runtime_width_px,
            height == audit.source_image_runtime_height_px,
            dimensions_verified is audit.source_image_dimensions_verified,
            hash_verified,
            dimensions_verified,
        )
    )


def audit_published_polarized_dt_reactivity(
    *,
    temperature_kev: Real = DEFAULT_TEMPERATURE_KEV,
    target_reactivity_ratio: Real = DEFAULT_TARGET_REACTIVITY_RATIO,
    energy_grid_points: Integral = DEFAULT_ENERGY_GRID_POINTS,
    digitization_half_width: Real = DEFAULT_DIGITIZATION_HALF_WIDTH,
) -> PublishedPolarizedDTReactivityAudit:
    """Digitize Figure 4b and repeat its full-alignment Maxwellian control."""

    temperature = _positive(temperature_kev, name="temperature_kev")
    if not (BOSCH_HALE_DT_TEMPERATURE_MIN_KEV <= temperature <= BOSCH_HALE_DT_TEMPERATURE_MAX_KEV):
        raise ValueError("temperature_kev must lie in the Bosch-Hale D-T fit domain")
    target = _positive(target_reactivity_ratio, name="target_reactivity_ratio")
    if target < DEFAULT_TARGET_REACTIVITY_RATIO:
        raise ValueError(
            "target_reactivity_ratio must be at least the fixed one-percent threshold"
        )
    points = _grid_count(energy_grid_points, name="energy_grid_points")
    half_width = _finite_real(
        digitization_half_width,
        name="digitization_half_width",
    )
    if half_width < 0.0 or half_width >= 0.1:
        raise ValueError("digitization_half_width must lie in [0, 0.1)")

    full_ratio, lower_ratio, numerical_baseline, kernel_summary = _integrate_digitized_curve(
        temperature_kev=temperature,
        energy_grid_points=points,
        digitization_half_width=half_width,
    )
    refined_ratio, _, _, _ = _integrate_digitized_curve(
        temperature_kev=temperature,
        energy_grid_points=2 * points - 1,
        digitization_half_width=half_width,
    )
    _, _, closed_baseline = bosch_hale_dt_reactivity(temperature)
    curve_energies, curve_enhancements = _digitized_full_curve()
    refinement_residual = abs(refined_ratio - full_ratio)
    (
        image_bundled,
        image_runtime_sha256,
        image_sha256_verified,
        image_runtime_width,
        image_runtime_height,
        image_dimensions_verified,
    ) = _bundled_hupin_figure_sha256()
    digitized_control_pass = all(
        (
            temperature == DEFAULT_TEMPERATURE_KEV,
            lower_ratio >= target,
            refinement_residual < 1.0e-8,
            image_sha256_verified,
            image_dimensions_verified,
        )
    )
    sciencedb_payload = current_sciencedb_v1_payload_audit()
    sciencedb_reactivity = current_sciencedb_dt_reactivity_audit()
    return PublishedPolarizedDTReactivityAudit(
        source_doi=HUPIN_DOI,
        source_pmcid=HUPIN_PMCID,
        source_license="CC BY 4.0",
        source_figure="Figure 4b, solid NCSMC-pheno full curve",
        expected_source_image_sha256=HUPIN_FIGURE_4_IMAGE_SHA256,
        expected_source_image_width_px=HUPIN_FIGURE_4_IMAGE_SIZE_PX[0],
        expected_source_image_height_px=HUPIN_FIGURE_4_IMAGE_SIZE_PX[1],
        source_image_repository_relative_path=(HUPIN_FIGURE_4_REPOSITORY_RELATIVE_PATH),
        source_image_bundled_in_repository=image_bundled,
        source_image_runtime_sha256=image_runtime_sha256,
        source_image_sha256_verified=image_sha256_verified,
        source_image_runtime_width_px=image_runtime_width,
        source_image_runtime_height_px=image_runtime_height,
        source_image_dimensions_verified=image_dimensions_verified,
        digitization_method=(
            "E_D=(x-460)/187.5 MeV; delta=1.5+(45-y)/1800; solid centerline sampled every 0.05 MeV"
        ),
        digitized_deuteron_lab_energies_mev=tuple(float(value) for value in curve_energies),
        digitized_full_alignment_enhancement_factors=tuple(
            float(value) for value in curve_enhancements
        ),
        digitization_enhancement_half_width=half_width,
        temperature_kev=temperature,
        target_reactivity_ratio=target,
        cross_section_energy_min_kev=BH_CROSS_SECTION_MIN_ENERGY_KEV,
        cross_section_energy_max_kev=BH_CROSS_SECTION_MAX_ENERGY_KEV,
        energy_grid_points=points,
        deuteron_lab_to_centre_of_mass_energy_factor=(
            TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
        ),
        numerical_baseline_reactivity_cm3_s=numerical_baseline,
        bosch_hale_closed_fit_reactivity_cm3_s=closed_baseline,
        numerical_to_closed_fit_ratio=numerical_baseline / closed_baseline,
        maxwellian_kernel_mode_energy_kev=kernel_summary[0],
        maxwellian_kernel_central_90_low_energy_kev=kernel_summary[1],
        maxwellian_kernel_central_90_high_energy_kev=kernel_summary[2],
        maxwellian_kernel_0p1_percent_energy_kev=kernel_summary[3],
        maxwellian_kernel_99p9_percent_energy_kev=kernel_summary[4],
        maxwellian_kernel_fraction_below_100_kev=kernel_summary[5],
        full_alignment_maxwellian_reactivity_ratio=full_ratio,
        digitization_lower_maxwellian_reactivity_ratio=lower_ratio,
        grid_refinement_absolute_residual=refinement_residual,
        figure_digitized_energy_dependent_cross_section_integrated=True,
        figure_digitized_one_percent_control_pass=digitized_control_pass,
        authors_published_energy_dependent_maxwellian_calculation=True,
        published_partial_deuteron_vector_polarization=0.8,
        published_partial_deuteron_tensor_polarization=0.8,
        published_partial_triton_vector_polarization=0.8,
        published_approximate_partial_rate_ratio=1.32,
        validated_target_deuteron_vector_polarization=None,
        validated_target_deuteron_tensor_polarization=None,
        validated_target_triton_vector_polarization=None,
        validated_target_state_maxwellian_reactivity_ratio=None,
        validated_target_state_maxwellian_reactivity_ratio_std=None,
        validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=None,
        target_state_resolved_operator_available=False,
        target_state_operator_provenance_digest=None,
        target_state_operator_artifact=_canonical_evidence_artifact(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS
        ),
        author_machine_readable_energy_grid_available=False,
        author_s_matrix_or_angle_integrated_coefficients_available=False,
        nuclear_model_systematic_covariance_available=False,
        dries_source_doi=DRIES_DOI,
        dries_exfor_url=DRIES_EXFOR_URL,
        dries_exfor_machine_readable_single_angle_azz_available=True,
        dries_exfor_point_count=29,
        dries_initial_tritium_target_unpolarized=True,
        dries_record_is_not_double_polarized_czz_or_total_rate=True,
        arnold_source_doi=ARNOLD_DOI,
        arnold_exfor_url=ARNOLD_EXFOR_URL,
        arnold_exfor_unpolarized_total_dt_cross_section_available=True,
        arnold_exfor_point_count=64,
        han_rmatrix_source_doi=HAN_RMATRIX_DOI,
        han_sciencedb_doi=HAN_SCIENCEDB_DOI,
        han_sciencedb_public_metadata_available=True,
        han_sciencedb_license="CC BY-SA 4.0",
        han_sciencedb_version="V1",
        han_sciencedb_metadata_declares_unpolarized_rmatrix_covariance=True,
        han_sciencedb_numeric_files_locally_verified=(
            sciencedb_payload.payload_integrity_gate_pass
        ),
        han_sciencedb_payload_integrity_gate_pass=(
            sciencedb_payload.payload_integrity_gate_pass
        ),
        han_sciencedb_numeric_covariance_matrix_available=(
            sciencedb_payload.numeric_covariance_matrix_or_correlation_payload_available
        ),
        han_sciencedb_initial_state_spin_operator_available=(
            sciencedb_payload.initial_state_spin_columns_or_operator_available
        ),
        han_sciencedb_payload_audit=sciencedb_payload,
        han_sciencedb_unpolarized_reactivity_audit=sciencedb_reactivity,
        han_sciencedb_unpolarized_interpolation_spread_below_one_percent=(
            sciencedb_reactivity.interpolation_spread_below_one_percent
        ),
        han_sciencedb_unpolarized_sub_one_percent_certification_gate_pass=(
            sciencedb_reactivity.unpolarized_sub_one_percent_certification_gate_pass
        ),
        han_initial_double_polarized_state_operator_available=False,
        spin_correlation_czz_directly_measured=False,
        polarized_dt_rate_directly_validated=False,
        local_author_data_reproduction_pass=False,
        physical_reaction_evidence_gate_pass=False,
        status=(
            "FIGURE_DIGITIZED_ENERGY_DEPENDENT_MAXWELLIAN_CONTROL_PASS_"
            "AUTHOR_GRID_COVARIANCE_AND_CZZ_VALIDATION_MISSING"
            if digitized_control_pass
            else "FIGURE_DIGITIZED_MAXWELLIAN_CONTROL_FAIL"
        ),
    )


def audit_polarized_fuel_primary_evidence(
    *,
    reference_fusion_power_w: Real = DEFAULT_REFERENCE_FUSION_POWER_W,
    single_pass_burn_fraction: Real = DEFAULT_SINGLE_PASS_BURN_FRACTION,
) -> PolarizedFuelPrimaryEvidenceAudit:
    """Compare primary source/retention records to a declared plant throughput."""

    fusion_power = _positive(reference_fusion_power_w, name="reference_fusion_power_w")
    burn_fraction = _positive(single_pass_burn_fraction, name="single_pass_burn_fraction")
    if burn_fraction >= 1.0:
        raise ValueError("single_pass_burn_fraction must be below one")
    fusion_energy_j = DT_FUSION_ENERGY_MEV * 1.0e6 * EV_TO_JOULE
    reaction_rate = fusion_power / fusion_energy_j
    per_species_rate = reaction_rate / burn_fraction
    reference_minimum_rate = (
        DEFAULT_REFERENCE_FUSION_POWER_W / fusion_energy_j / DEFAULT_SINGLE_PASS_BURN_FRACTION
    )
    reference_plant_scale_pass = per_species_rate >= reference_minimum_rate
    rate_ratio = per_species_rate / COULTER_CONTINUOUS_DEUTERIUM_RATE_S
    coulter_meets_rate = COULTER_CONTINUOUS_DEUTERIUM_RATE_S >= per_species_rate
    coulter_meets_reference_reactor_rate = coulter_meets_rate and reference_plant_scale_pass
    anke_meets_reference_reactor_rate = (
        ANKE_DEUTERIUM_RATE_S - 3.0 * ANKE_DEUTERIUM_RATE_STD_S >= per_species_rate
        and reference_plant_scale_pass
    )
    return PolarizedFuelPrimaryEvidenceAudit(
        declared_reference_fusion_power_w=fusion_power,
        declared_single_pass_burn_fraction=burn_fraction,
        required_fusion_reaction_rate_s=reaction_rate,
        required_per_species_fuel_rate_s=per_species_rate,
        reference_plant_minimum_per_species_fuel_rate_s=reference_minimum_rate,
        reference_plant_scale_invariant_pass=reference_plant_scale_pass,
        validated_integrated_plant_configuration_id=None,
        coulter_source_doi=COULTER_DOI,
        coulter_measured_continuous_deuterium_rate_s=(COULTER_CONTINUOUS_DEUTERIUM_RATE_S),
        coulter_measured_deuterium_polarization=COULTER_DEUTERIUM_POLARIZATION,
        coulter_measured_deuterium_polarization_std=(COULTER_DEUTERIUM_POLARIZATION_STD),
        required_to_coulter_rate_ratio=rate_ratio,
        coulter_record_meets_reactor_deuterium_rate=(coulter_meets_reference_reactor_rate),
        anke_source_doi=ANKE_DOI,
        anke_measured_deuterium_rate_s=ANKE_DEUTERIUM_RATE_S,
        anke_measured_deuterium_rate_std_s=ANKE_DEUTERIUM_RATE_STD_S,
        anke_measured_deuteron_vector_polarization_abs=(ANKE_DEUTERON_VECTOR_POLARIZATION_ABS),
        anke_measured_deuteron_tensor_polarization=(ANKE_DEUTERON_TENSOR_POLARIZATION),
        anke_rf_dissociator_power_w=ANKE_RF_DISSOCIATOR_POWER_W,
        anke_rf_power_is_partial_component_not_complete_wall_plug=True,
        anke_record_meets_reactor_deuterium_rate=anke_meets_reference_reactor_rate,
        sofikitis_source_doi=SOFIKITIS_DOI,
        sofikitis_nuclear_deuterium_polarization=(SOFIKITIS_NUCLEAR_DEUTERIUM_POLARIZATION),
        sofikitis_hyperfine_transfer_time_ns=SOFIKITIS_HYPERFINE_TRANSFER_TIME_NS,
        sofikitis_reactor_continuous_throughput_measured=False,
        utsuro_source_doi=UTSURO_DOI,
        utsuro_hd_proxy_proof_of_concept_performed=True,
        utsuro_actual_polarized_tritium_source_demonstrated=False,
        utsuro_reactor_continuous_throughput_measured=False,
        cook_source_doi=COOK_DOI,
        cook_sparc_deuterium_mplus_retention_probability=(
            COOK_SPARC_DEUTERIUM_MPLUS_RETENTION_PROBABILITY
        ),
        cook_sparc_tritium_depolarization_fraction=(COOK_SPARC_TRITIUM_DEPOLARIZATION_FRACTION),
        cook_sparc_aggregate_depolarization_fraction=(COOK_SPARC_AGGREGATE_DEPOLARIZATION_FRACTION),
        cook_sparc_aggregate_depolarization_std=(COOK_SPARC_AGGREGATE_DEPOLARIZATION_STD),
        cook_quasi_steady_time_ms=COOK_QUASI_STEADY_TIME_MS,
        cook_result_is_simulation_not_burn_measurement=True,
        souers_source_doi=SOUERS_SOLID_DT_RELAXATION_DOI,
        souers_actual_solid_dt_triton_relaxation_measured=True,
        souers_result_is_preburn_not_burn_weighted_plasma_retention=True,
        collins_source_doi=COLLINS_SOLID_DT_NMR_DOI,
        collins_actual_solid_dt_deuteron_nmr_relaxation_measured=True,
        collins_result_is_preburn_not_burn_weighted_plasma_retention=True,
        reactor_rate_polarized_deuterium_source_demonstrated=(coulter_meets_reference_reactor_rate),
        reactor_rate_polarized_tritium_source_demonstrated=False,
        validated_reactor_rate_deuterium_source_rate_s=None,
        validated_reactor_rate_tritium_source_rate_s=None,
        validated_reactor_rate_deuterium_source_rate_linearized_std_s=None,
        validated_reactor_rate_tritium_source_rate_linearized_std_s=None,
        validated_reactor_rate_deuterium_source_rate_lower_3sigma_s=None,
        validated_reactor_rate_tritium_source_rate_lower_3sigma_s=None,
        validated_reactor_rate_deuterium_source_rate_3sigma_bound_model=None,
        validated_reactor_rate_tritium_source_rate_3sigma_bound_model=None,
        validated_reactor_rate_deuterium_measurement_duration_s=None,
        validated_reactor_rate_tritium_measurement_duration_s=None,
        validated_reactor_rate_deuterium_measurement_duration_std_s=None,
        validated_reactor_rate_tritium_measurement_duration_std_s=None,
        validated_reactor_rate_deuterium_output_count=None,
        validated_reactor_rate_tritium_output_count=None,
        validated_reactor_rate_deuterium_output_count_std=None,
        validated_reactor_rate_tritium_output_count_std=None,
        validated_reactor_rate_deuterium_background_count=None,
        validated_reactor_rate_tritium_background_count=None,
        validated_reactor_rate_deuterium_background_count_std=None,
        validated_reactor_rate_tritium_background_count_std=None,
        validated_reactor_rate_deuterium_detection_efficiency=None,
        validated_reactor_rate_tritium_detection_efficiency=None,
        validated_reactor_rate_deuterium_detection_efficiency_std=None,
        validated_reactor_rate_tritium_detection_efficiency_std=None,
        validated_reactor_rate_deuterium_uncertainty_correlation_matrix=None,
        validated_reactor_rate_tritium_uncertainty_correlation_matrix=None,
        validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix=None,
        validated_reactor_rate_deuteron_vector_polarization=None,
        validated_reactor_rate_deuteron_vector_polarization_std=None,
        validated_reactor_rate_deuteron_tensor_polarization=None,
        validated_reactor_rate_deuteron_tensor_polarization_std=None,
        validated_reactor_rate_triton_vector_polarization=None,
        validated_reactor_rate_triton_vector_polarization_std=None,
        validated_reactor_rate_deuterium_source_provenance_digest=None,
        validated_reactor_rate_tritium_source_provenance_digest=None,
        validated_reactor_rate_deuterium_source_artifact=_canonical_evidence_artifact(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS
        ),
        validated_reactor_rate_tritium_source_artifact=_canonical_evidence_artifact(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS
        ),
        burn_weighted_dt_polarization_product_measured=False,
        validated_burn_weighted_dt_polarization_product=None,
        validated_burn_weighted_dt_polarization_product_std=None,
        validated_burn_weighted_deuteron_tensor_polarization=None,
        validated_burn_weighted_deuteron_tensor_polarization_std=None,
        validated_burn_retention_measurement_duration_s=None,
        validated_burn_retention_measurement_duration_std_s=None,
        validated_burn_retention_reaction_count=None,
        validated_burn_retention_uncertainty_correlation_matrix=None,
        validated_burn_retention_observable_semantics=None,
        validated_burn_reaction_operator_provenance_digest=None,
        validated_burn_operator_uncertainty_relationship=None,
        validated_burn_state_operator_reactivity_ratio=None,
        validated_burn_state_operator_reactivity_ratio_robust_margin_per_sigma_equivalent=None,
        validated_burn_state_operator_reactivity_ratio_robust_lower_3sigma=None,
        validated_burn_state_operator_max_phase_relative_residual=None,
        validated_burn_state_operator_contraction_semantics=None,
        burn_state_operator_componentwise_3sigma_stress_envelope_control_pass=False,
        energy_resolved_burn_operator_sufficient_statistics_available=False,
        validated_burn_retention_provenance_digest=None,
        validated_burn_retention_artifact=_canonical_evidence_artifact(
            BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS
        ),
        complete_wall_plug_energy_per_injected_pair_measured=False,
        complete_wall_plug_uncertainty_flow_and_provenance_available=False,
        validated_wall_plug_energy_per_injected_dt_pair_ev=None,
        validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev=None,
        validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev=None,
        validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev=None,
        validated_wall_plug_total_energy_j=None,
        validated_wall_plug_total_energy_linearized_std_j=None,
        validated_wall_plug_total_energy_lower_3sigma_j=None,
        validated_wall_plug_total_energy_upper_3sigma_j=None,
        validated_wall_plug_measurement_duration_s=None,
        validated_wall_plug_measurement_duration_std_s=None,
        validated_wall_plug_measurement_injected_dt_pair_count=None,
        validated_wall_plug_measurement_injected_dt_pair_count_std=None,
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_s=None,
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s=None,
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s=None,
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s=None,
        validated_wall_plug_3sigma_bound_model=None,
        validated_wall_plug_uncertainty_correlation_matrix=None,
        validated_wall_plug_measurement_provenance=None,
        validated_wall_plug_artifact=_canonical_evidence_artifact(
            POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS
        ),
        source_retention_and_wall_plug_gate_pass=False,
        status=(
            "REFERENCE_PLANT_SCALE_BELOW_500MW_ONE_PERCENT_THROUGHPUT"
            if not reference_plant_scale_pass
            else (
                "MEASURED_D_SOURCE_MEETS_DECLARED_REFERENCE_RATE_"
                "TRITIUM_BURN_RETENTION_AND_WALL_PLUG_ABSENT"
                if coulter_meets_reference_reactor_rate
                else "MEASURED_D_SOURCE_BELOW_DECLARED_REFERENCE_RATE_"
                "TRITIUM_BURN_RETENTION_AND_WALL_PLUG_ABSENT"
            )
        ),
    )


def _reaction_operator_artifact_claims(
    audit: PublishedPolarizedDTReactivityAudit,
) -> dict[str, object]:
    return {
        "temperature_kev": audit.temperature_kev,
        "target_reactivity_ratio": audit.target_reactivity_ratio,
        "validated_target_deuteron_vector_polarization": (
            audit.validated_target_deuteron_vector_polarization
        ),
        "validated_target_deuteron_tensor_polarization": (
            audit.validated_target_deuteron_tensor_polarization
        ),
        "validated_target_triton_vector_polarization": (
            audit.validated_target_triton_vector_polarization
        ),
        "validated_target_state_maxwellian_reactivity_ratio": (
            audit.validated_target_state_maxwellian_reactivity_ratio
        ),
        "validated_target_state_maxwellian_reactivity_ratio_std": (
            audit.validated_target_state_maxwellian_reactivity_ratio_std
        ),
        "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma": (
            audit.validated_target_state_maxwellian_reactivity_ratio_lower_3sigma
        ),
        "target_state_resolved_operator_available": audit.target_state_resolved_operator_available,
        "author_machine_readable_energy_grid_available": (
            audit.author_machine_readable_energy_grid_available
        ),
        "author_s_matrix_or_angle_integrated_coefficients_available": (
            audit.author_s_matrix_or_angle_integrated_coefficients_available
        ),
        "nuclear_model_systematic_covariance_available": (
            audit.nuclear_model_systematic_covariance_available
        ),
        "spin_correlation_czz_directly_measured": audit.spin_correlation_czz_directly_measured,
        "polarized_dt_rate_directly_validated": audit.polarized_dt_rate_directly_validated,
        "local_author_data_reproduction_pass": audit.local_author_data_reproduction_pass,
    }


def _source_common_artifact_claims(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> dict[str, object]:
    return {
        "declared_reference_fusion_power_w": audit.declared_reference_fusion_power_w,
        "declared_single_pass_burn_fraction": audit.declared_single_pass_burn_fraction,
        "required_per_species_fuel_rate_s": audit.required_per_species_fuel_rate_s,
        "validated_integrated_plant_configuration_id": (
            audit.validated_integrated_plant_configuration_id
        ),
    }


def _source_plant_ledger_integrity_pass(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> bool:
    numeric_values = (
        audit.declared_reference_fusion_power_w,
        audit.declared_single_pass_burn_fraction,
        audit.required_fusion_reaction_rate_s,
        audit.required_per_species_fuel_rate_s,
        audit.reference_plant_minimum_per_species_fuel_rate_s,
        audit.required_to_coulter_rate_ratio,
    )
    if not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in numeric_values
    ):
        return False
    fusion_power = float(audit.declared_reference_fusion_power_w)
    burn_fraction = float(audit.declared_single_pass_burn_fraction)
    if fusion_power <= 0.0 or not 0.0 < burn_fraction < 1.0:
        return False
    fusion_energy_j = DT_FUSION_ENERGY_MEV * 1.0e6 * EV_TO_JOULE
    reaction_rate = fusion_power / fusion_energy_j
    per_species_rate = reaction_rate / burn_fraction
    reference_minimum_rate = (
        DEFAULT_REFERENCE_FUSION_POWER_W / fusion_energy_j / DEFAULT_SINGLE_PASS_BURN_FRACTION
    )
    reference_scale_pass = per_species_rate >= reference_minimum_rate
    coulter_meets = (
        COULTER_CONTINUOUS_DEUTERIUM_RATE_S >= per_species_rate and reference_scale_pass
    )
    anke_meets = (
        ANKE_DEUTERIUM_RATE_S - 3.0 * ANKE_DEUTERIUM_RATE_STD_S >= per_species_rate
        and reference_scale_pass
    )
    return all(
        (
            audit.required_fusion_reaction_rate_s == reaction_rate,
            audit.required_per_species_fuel_rate_s == per_species_rate,
            audit.reference_plant_minimum_per_species_fuel_rate_s == reference_minimum_rate,
            audit.reference_plant_scale_invariant_pass is reference_scale_pass,
            audit.required_to_coulter_rate_ratio
            == per_species_rate / COULTER_CONTINUOUS_DEUTERIUM_RATE_S,
            audit.coulter_record_meets_reactor_deuterium_rate is coulter_meets,
            audit.anke_record_meets_reactor_deuterium_rate is anke_meets,
        )
    )


def _deuterium_source_artifact_claims(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> dict[str, object]:
    return {
        **_source_common_artifact_claims(audit),
        "species": "D",
        "reactor_rate_source_demonstrated": (
            audit.reactor_rate_polarized_deuterium_source_demonstrated
        ),
        "source_rate_s": audit.validated_reactor_rate_deuterium_source_rate_s,
        "source_rate_linearized_std_s": (
            audit.validated_reactor_rate_deuterium_source_rate_linearized_std_s
        ),
        "source_rate_lower_3sigma_s": (
            audit.validated_reactor_rate_deuterium_source_rate_lower_3sigma_s
        ),
        "source_rate_3sigma_bound_model": (
            audit.validated_reactor_rate_deuterium_source_rate_3sigma_bound_model
        ),
        "measurement_duration_s": audit.validated_reactor_rate_deuterium_measurement_duration_s,
        "measurement_duration_std_s": (
            audit.validated_reactor_rate_deuterium_measurement_duration_std_s
        ),
        "output_count": audit.validated_reactor_rate_deuterium_output_count,
        "output_count_std": audit.validated_reactor_rate_deuterium_output_count_std,
        "background_count": audit.validated_reactor_rate_deuterium_background_count,
        "background_count_std": (
            audit.validated_reactor_rate_deuterium_background_count_std
        ),
        "detection_efficiency": (
            audit.validated_reactor_rate_deuterium_detection_efficiency
        ),
        "detection_efficiency_std": (
            audit.validated_reactor_rate_deuterium_detection_efficiency_std
        ),
        "source_rate_uncertainty_correlation_matrix": (
            audit.validated_reactor_rate_deuterium_uncertainty_correlation_matrix
        ),
        "deuteron_polarization_uncertainty_correlation_matrix": (
            audit.validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix
        ),
        "deuteron_vector_polarization": (
            audit.validated_reactor_rate_deuteron_vector_polarization
        ),
        "deuteron_vector_polarization_std": (
            audit.validated_reactor_rate_deuteron_vector_polarization_std
        ),
        "deuteron_tensor_polarization": (
            audit.validated_reactor_rate_deuteron_tensor_polarization
        ),
        "deuteron_tensor_polarization_std": (
            audit.validated_reactor_rate_deuteron_tensor_polarization_std
        ),
    }


def _tritium_source_artifact_claims(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> dict[str, object]:
    return {
        **_source_common_artifact_claims(audit),
        "species": "T",
        "reactor_rate_source_demonstrated": (
            audit.reactor_rate_polarized_tritium_source_demonstrated
        ),
        "source_rate_s": audit.validated_reactor_rate_tritium_source_rate_s,
        "source_rate_linearized_std_s": (
            audit.validated_reactor_rate_tritium_source_rate_linearized_std_s
        ),
        "source_rate_lower_3sigma_s": (
            audit.validated_reactor_rate_tritium_source_rate_lower_3sigma_s
        ),
        "source_rate_3sigma_bound_model": (
            audit.validated_reactor_rate_tritium_source_rate_3sigma_bound_model
        ),
        "measurement_duration_s": audit.validated_reactor_rate_tritium_measurement_duration_s,
        "measurement_duration_std_s": (
            audit.validated_reactor_rate_tritium_measurement_duration_std_s
        ),
        "output_count": audit.validated_reactor_rate_tritium_output_count,
        "output_count_std": audit.validated_reactor_rate_tritium_output_count_std,
        "background_count": audit.validated_reactor_rate_tritium_background_count,
        "background_count_std": (
            audit.validated_reactor_rate_tritium_background_count_std
        ),
        "detection_efficiency": audit.validated_reactor_rate_tritium_detection_efficiency,
        "detection_efficiency_std": (
            audit.validated_reactor_rate_tritium_detection_efficiency_std
        ),
        "source_rate_uncertainty_correlation_matrix": (
            audit.validated_reactor_rate_tritium_uncertainty_correlation_matrix
        ),
        "triton_vector_polarization": audit.validated_reactor_rate_triton_vector_polarization,
        "triton_vector_polarization_std": (
            audit.validated_reactor_rate_triton_vector_polarization_std
        ),
    }


def _burn_retention_artifact_claims(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> dict[str, object]:
    return {
        **_source_common_artifact_claims(audit),
        "burn_weighted_dt_polarization_product_measured": (
            audit.burn_weighted_dt_polarization_product_measured
        ),
        "burn_weighted_dt_polarization_product": (
            audit.validated_burn_weighted_dt_polarization_product
        ),
        "burn_weighted_dt_polarization_product_std": (
            audit.validated_burn_weighted_dt_polarization_product_std
        ),
        "burn_weighted_deuteron_tensor_polarization": (
            audit.validated_burn_weighted_deuteron_tensor_polarization
        ),
        "burn_weighted_deuteron_tensor_polarization_std": (
            audit.validated_burn_weighted_deuteron_tensor_polarization_std
        ),
        "measurement_duration_s": audit.validated_burn_retention_measurement_duration_s,
        "measurement_duration_std_s": (
            audit.validated_burn_retention_measurement_duration_std_s
        ),
        "reaction_count": audit.validated_burn_retention_reaction_count,
        "burn_uncertainty_correlation_matrix": (
            audit.validated_burn_retention_uncertainty_correlation_matrix
        ),
        "burn_observable_semantics": (
            audit.validated_burn_retention_observable_semantics
        ),
        "reaction_operator_manifest_sha256": (
            audit.validated_burn_reaction_operator_provenance_digest
        ),
        "operator_burn_uncertainty_relationship": (
            audit.validated_burn_operator_uncertainty_relationship
        ),
    }


def _wall_plug_artifact_claims(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> dict[str, object]:
    return {
        **_source_common_artifact_claims(audit),
        "complete_wall_plug_energy_per_injected_pair_measured": (
            audit.complete_wall_plug_energy_per_injected_pair_measured
        ),
        "complete_wall_plug_uncertainty_flow_and_provenance_available": (
            audit.complete_wall_plug_uncertainty_flow_and_provenance_available
        ),
        "wall_plug_energy_per_injected_dt_pair_ev": (
            audit.validated_wall_plug_energy_per_injected_dt_pair_ev
        ),
        "wall_plug_energy_linearized_std_per_injected_dt_pair_ev": (
            audit.validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev
        ),
        "wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev": (
            audit.validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev
        ),
        "wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev": (
            audit.validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev
        ),
        "wall_plug_total_energy_j": audit.validated_wall_plug_total_energy_j,
        "wall_plug_total_energy_linearized_std_j": (
            audit.validated_wall_plug_total_energy_linearized_std_j
        ),
        "wall_plug_total_energy_lower_3sigma_j": (
            audit.validated_wall_plug_total_energy_lower_3sigma_j
        ),
        "wall_plug_total_energy_upper_3sigma_j": (
            audit.validated_wall_plug_total_energy_upper_3sigma_j
        ),
        "measurement_duration_s": audit.validated_wall_plug_measurement_duration_s,
        "measurement_duration_std_s": (
            audit.validated_wall_plug_measurement_duration_std_s
        ),
        "measurement_injected_dt_pair_count": (
            audit.validated_wall_plug_measurement_injected_dt_pair_count
        ),
        "measurement_injected_dt_pair_count_std": (
            audit.validated_wall_plug_measurement_injected_dt_pair_count_std
        ),
        "measurement_injected_dt_pair_flow_rate_s": (
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_s
        ),
        "measurement_injected_dt_pair_flow_rate_linearized_std_s": (
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s
        ),
        "measurement_injected_dt_pair_flow_rate_lower_3sigma_s": (
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s
        ),
        "measurement_injected_dt_pair_flow_rate_upper_3sigma_s": (
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s
        ),
        "wall_plug_3sigma_bound_model": (
            audit.validated_wall_plug_3sigma_bound_model
        ),
        "wall_plug_uncertainty_correlation_matrix": (
            audit.validated_wall_plug_uncertainty_correlation_matrix
        ),
    }


def _published_reaction_evidence_record_integrity_pass(
    audit: PublishedPolarizedDTReactivityAudit,
    *,
    repository_root: str | Path | None = None,
) -> bool:
    """Validate a complete reaction record without granting canonical status."""
    validated_states = (
        audit.validated_target_deuteron_vector_polarization,
        audit.validated_target_deuteron_tensor_polarization,
        audit.validated_target_triton_vector_polarization,
    )
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in validated_states
    ):
        return False
    p_d, p_zz, p_t = (float(value) for value in validated_states)
    validated_ratios = (
        audit.validated_target_state_maxwellian_reactivity_ratio,
        audit.validated_target_state_maxwellian_reactivity_ratio_std,
        audit.validated_target_state_maxwellian_reactivity_ratio_lower_3sigma,
    )
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in validated_ratios
    ):
        return False
    validated_ratio, validated_ratio_std, validated_ratio_lower = (
        float(value) for value in validated_ratios
    )
    operator_artifact_pass = _typed_evidence_artifact_gate_pass(
        audit.target_state_operator_artifact,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        provenance_digest=audit.target_state_operator_provenance_digest,
        expected_claims=_reaction_operator_artifact_claims(audit),
        repository_root=repository_root,
    )
    state_resolved_operator_pass = all(
        (
            -1.0 <= p_d <= 1.0,
            -2.0 <= p_zz <= 1.0,
            -1.0 <= p_t <= 1.0,
            (1.0 - p_zz) / 3.0 >= 0.0,
            (2.0 + p_zz + 3.0 * p_d) / 6.0 >= 0.0,
            (2.0 + p_zz - 3.0 * p_d) / 6.0 >= 0.0,
            validated_ratio_std >= 0.0,
            validated_ratio_lower == validated_ratio - 3.0 * validated_ratio_std,
            audit.target_reactivity_ratio >= DEFAULT_TARGET_REACTIVITY_RATIO,
            validated_ratio_lower >= DEFAULT_TARGET_REACTIVITY_RATIO,
            validated_ratio_lower >= audit.target_reactivity_ratio,
            audit.target_state_resolved_operator_available is True,
            operator_artifact_pass,
        )
    )
    reference_classification_pass = all(
        (
            audit.dries_source_doi == DRIES_DOI,
            audit.dries_exfor_url == DRIES_EXFOR_URL,
            audit.dries_exfor_machine_readable_single_angle_azz_available is True,
            audit.dries_exfor_point_count == 29,
            audit.dries_initial_tritium_target_unpolarized is True,
            audit.dries_record_is_not_double_polarized_czz_or_total_rate is True,
            audit.arnold_source_doi == ARNOLD_DOI,
            audit.arnold_exfor_url == ARNOLD_EXFOR_URL,
            audit.arnold_exfor_unpolarized_total_dt_cross_section_available is True,
            audit.arnold_exfor_point_count == 64,
            audit.han_rmatrix_source_doi == HAN_RMATRIX_DOI,
            audit.han_sciencedb_doi == HAN_SCIENCEDB_DOI,
            audit.han_sciencedb_public_metadata_available is True,
            audit.han_sciencedb_license == "CC BY-SA 4.0",
            audit.han_sciencedb_version == "V1",
            audit.han_sciencedb_metadata_declares_unpolarized_rmatrix_covariance is True,
            audit.han_sciencedb_numeric_files_locally_verified is True,
            audit.han_sciencedb_payload_integrity_gate_pass is True,
            audit.han_sciencedb_payload_audit == current_sciencedb_v1_payload_audit(),
            audit.han_sciencedb_unpolarized_reactivity_audit
            == current_sciencedb_dt_reactivity_audit(),
            audit.han_sciencedb_unpolarized_interpolation_spread_below_one_percent is False,
            audit.han_sciencedb_unpolarized_sub_one_percent_certification_gate_pass is False,
            audit.han_sciencedb_numeric_covariance_matrix_available is False,
            audit.han_sciencedb_initial_state_spin_operator_available is False,
            audit.han_initial_double_polarized_state_operator_available is False,
        )
    )
    return all(
        flag is True
        for flag in (
            state_resolved_operator_pass,
            reference_classification_pass,
            audit.source_image_sha256_verified,
            audit.source_image_dimensions_verified,
            audit.source_image_runtime_width_px == audit.expected_source_image_width_px,
            audit.source_image_runtime_height_px == audit.expected_source_image_height_px,
            audit.figure_digitized_energy_dependent_cross_section_integrated,
            audit.figure_digitized_one_percent_control_pass,
            audit.authors_published_energy_dependent_maxwellian_calculation,
            audit.author_machine_readable_energy_grid_available,
            audit.author_s_matrix_or_angle_integrated_coefficients_available,
            audit.nuclear_model_systematic_covariance_available,
            audit.spin_correlation_czz_directly_measured,
            audit.polarized_dt_rate_directly_validated,
            audit.local_author_data_reproduction_pass,
            audit.physical_reaction_evidence_gate_pass,
        )
    )


def physical_published_reaction_evidence_gate_pass(
    audit: PublishedPolarizedDTReactivityAudit,
) -> bool:
    """Accept only a semantically complete, fresh canonical reaction audit."""

    if type(audit) is not PublishedPolarizedDTReactivityAudit:
        return False
    try:
        expected = audit_published_polarized_dt_reactivity(
            temperature_kev=audit.temperature_kev,
            target_reactivity_ratio=audit.target_reactivity_ratio,
            energy_grid_points=audit.energy_grid_points,
            digitization_half_width=audit.digitization_enhancement_half_width,
        )
        return (
            audit == expected
            and _fresh_hupin_image_gate_pass(audit)
            and _published_reaction_evidence_record_integrity_pass(audit)
            and physical_canonical_repository_evidence_artifact_gate_pass(
                audit.target_state_operator_artifact
            )
        )
    except Exception:
        return False


def _artifact_derived_claims_object(artifact: EvidenceArtifactAudit) -> dict[str, Any] | None:
    try:
        claims = json.loads(
            artifact.derived_claims_canonical_json
            if isinstance(artifact.derived_claims_canonical_json, str)
            else "null",
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, RecursionError, OverflowError, MemoryError):
        return None
    return claims if isinstance(claims, dict) else None


def _semantic_correlation_matrix(value: object, *, size: int) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != size:
            return None
        converted: list[float] = []
        for item in row:
            if (
                not isinstance(item, Real)
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or (float(item) == 0.0 and math.copysign(1.0, float(item)) < 0.0)
            ):
                return None
            converted.append(float(item))
        rows.append(converted)
    matrix = np.asarray(rows, dtype=np.float64)
    if (
        not np.array_equal(matrix, matrix.T)
        or not np.array_equal(np.diag(matrix), np.ones(size, dtype=np.float64))
        or np.any(matrix < -1.0)
        or np.any(matrix > 1.0)
    ):
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigen_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * size
        * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    )
    if (
        not np.all(np.isfinite(eigenvalues))
        or float(np.min(eigenvalues)) < -eigen_tolerance
    ):
        return None
    return matrix


def _semantic_psd_matrix(value: object, *, size: int) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != size:
            return None
        converted: list[float] = []
        for item in row:
            if (
                not isinstance(item, Real)
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or (float(item) == 0.0 and math.copysign(1.0, float(item)) < 0.0)
            ):
                return None
            converted.append(float(item))
        rows.append(converted)
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.array_equal(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigen_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * size
        * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    )
    if (
        not np.all(np.isfinite(eigenvalues))
        or float(np.min(eigenvalues)) < -eigen_tolerance
    ):
        return None
    return matrix


def _semantic_gradient_standard_deviation(
    gradient: tuple[float, ...],
    standard_deviations: tuple[float, ...],
    correlation: np.ndarray,
) -> float | None:
    gradient_array = np.asarray(gradient, dtype=np.float64)
    std_array = np.asarray(standard_deviations, dtype=np.float64)
    if (
        gradient_array.shape != std_array.shape
        or correlation.shape != (gradient_array.size, gradient_array.size)
        or not np.all(np.isfinite(gradient_array))
        or not np.all(np.isfinite(std_array))
        or np.any(std_array <= 0.0)
    ):
        return None
    covariance = correlation * np.outer(std_array, std_array)
    variance = float(gradient_array @ covariance @ gradient_array)
    if not math.isfinite(variance) or variance <= 0.0:
        return None
    result = math.sqrt(variance)
    return result if math.isfinite(result) and result > 0.0 else None


def _polarized_source_evidence_record_integrity_pass(
    audit: PolarizedFuelPrimaryEvidenceAudit,
    *,
    repository_root: str | Path | None = None,
) -> bool:
    """Validate a complete source record without granting canonical status."""
    deuterium_source_artifact_pass = _typed_evidence_artifact_gate_pass(
        audit.validated_reactor_rate_deuterium_source_artifact,
        evidence_class=REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        provenance_digest=audit.validated_reactor_rate_deuterium_source_provenance_digest,
        expected_claims=_deuterium_source_artifact_claims(audit),
        repository_root=repository_root,
    )
    tritium_source_artifact_pass = _typed_evidence_artifact_gate_pass(
        audit.validated_reactor_rate_tritium_source_artifact,
        evidence_class=REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        provenance_digest=audit.validated_reactor_rate_tritium_source_provenance_digest,
        expected_claims=_tritium_source_artifact_claims(audit),
        repository_root=repository_root,
    )
    burn_retention_artifact_pass = _typed_evidence_artifact_gate_pass(
        audit.validated_burn_retention_artifact,
        evidence_class=BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        provenance_digest=audit.validated_burn_retention_provenance_digest,
        expected_claims=_burn_retention_artifact_claims(audit),
        repository_root=repository_root,
    )
    wall_plug_artifact_pass = _typed_evidence_artifact_gate_pass(
        audit.validated_wall_plug_artifact,
        evidence_class=POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        provenance_digest=audit.validated_wall_plug_measurement_provenance,
        expected_claims=_wall_plug_artifact_claims(audit),
        repository_root=repository_root,
    )
    integrated_artifact_storage_pass = _distinct_artifact_storage_pass(
        audit.validated_reactor_rate_deuterium_source_artifact,
        audit.validated_reactor_rate_tritium_source_artifact,
        audit.validated_burn_retention_artifact,
        audit.validated_wall_plug_artifact,
        repository_root=repository_root,
    )
    deuterium_artifact = audit.validated_reactor_rate_deuterium_source_artifact
    tritium_artifact = audit.validated_reactor_rate_tritium_source_artifact
    burn_artifact = audit.validated_burn_retention_artifact
    wall_artifact = audit.validated_wall_plug_artifact
    expected_source_links = (
        ("deuterium_source", deuterium_artifact.expected_manifest_sha256),
        ("tritium_source", tritium_artifact.expected_manifest_sha256),
    )
    expected_burn_links = (
        *expected_source_links,
        ("reaction_operator", audit.validated_burn_reaction_operator_provenance_digest),
    )
    integrated_artifact_cross_join_pass = all(
        (
            isinstance(audit.validated_integrated_plant_configuration_id, str),
            deuterium_artifact.derived_plant_configuration_id
            == audit.validated_integrated_plant_configuration_id,
            tritium_artifact.derived_plant_configuration_id
            == audit.validated_integrated_plant_configuration_id,
            burn_artifact.derived_plant_configuration_id
            == audit.validated_integrated_plant_configuration_id,
            wall_artifact.derived_plant_configuration_id
            == audit.validated_integrated_plant_configuration_id,
            _sha256_provenance_digest_available(
                audit.validated_burn_reaction_operator_provenance_digest
            ),
            burn_artifact.derived_linked_manifest_sha256s == expected_burn_links,
            wall_artifact.derived_linked_manifest_sha256s == expected_source_links,
        )
    )
    source_numeric_records = (
        (
            audit.validated_reactor_rate_deuterium_source_rate_s,
            audit.validated_reactor_rate_deuterium_source_rate_linearized_std_s,
            audit.validated_reactor_rate_deuterium_source_rate_lower_3sigma_s,
            audit.validated_reactor_rate_deuterium_source_rate_3sigma_bound_model,
            audit.validated_reactor_rate_deuterium_measurement_duration_s,
            audit.validated_reactor_rate_deuterium_measurement_duration_std_s,
            audit.validated_reactor_rate_deuterium_output_count,
            audit.validated_reactor_rate_deuterium_output_count_std,
            audit.validated_reactor_rate_deuterium_background_count,
            audit.validated_reactor_rate_deuterium_background_count_std,
            audit.validated_reactor_rate_deuterium_detection_efficiency,
            audit.validated_reactor_rate_deuterium_detection_efficiency_std,
            audit.validated_reactor_rate_deuterium_uncertainty_correlation_matrix,
            deuterium_source_artifact_pass,
            audit.reactor_rate_polarized_deuterium_source_demonstrated,
        ),
        (
            audit.validated_reactor_rate_tritium_source_rate_s,
            audit.validated_reactor_rate_tritium_source_rate_linearized_std_s,
            audit.validated_reactor_rate_tritium_source_rate_lower_3sigma_s,
            audit.validated_reactor_rate_tritium_source_rate_3sigma_bound_model,
            audit.validated_reactor_rate_tritium_measurement_duration_s,
            audit.validated_reactor_rate_tritium_measurement_duration_std_s,
            audit.validated_reactor_rate_tritium_output_count,
            audit.validated_reactor_rate_tritium_output_count_std,
            audit.validated_reactor_rate_tritium_background_count,
            audit.validated_reactor_rate_tritium_background_count_std,
            audit.validated_reactor_rate_tritium_detection_efficiency,
            audit.validated_reactor_rate_tritium_detection_efficiency_std,
            audit.validated_reactor_rate_tritium_uncertainty_correlation_matrix,
            tritium_source_artifact_pass,
            audit.reactor_rate_polarized_tritium_source_demonstrated,
        ),
    )
    source_rate_record_flags: list[bool] = []
    for record in source_numeric_records:
        raw_values = (*record[:3], *record[4:12])
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in raw_values
        ):
            return False
        (
            rate,
            rate_linearized_std,
            rate_lower,
            duration,
            duration_std,
            count,
            count_std,
            background,
            background_std,
            efficiency,
            efficiency_std,
        ) = (float(value) for value in raw_values)
        bound_model = record[3]
        correlation = _semantic_correlation_matrix(record[12], size=4)
        if (
            correlation is None
            or duration <= 0.0
            or efficiency <= 0.0
            or count <= background
        ):
            return False
        calculated_rate = (count - background) / efficiency / duration
        calculated_rate_std = _semantic_gradient_standard_deviation(
            (
                1.0 / efficiency / duration,
                -1.0 / efficiency / duration,
                -calculated_rate / efficiency,
                -calculated_rate / duration,
            ),
            (count_std, background_std, efficiency_std, duration_std),
            correlation,
        )
        calculated_rate_lower = (
            count - 3.0 * count_std - background - 3.0 * background_std
        ) / min(1.0, efficiency + 3.0 * efficiency_std) / (
            duration + 3.0 * duration_std
        )
        calculated_demonstrated = (
            calculated_rate_lower >= audit.required_per_species_fuel_rate_s
        )
        source_rate_record_flags.append(
            all(
                (
                    record[13] is True,
                    record[14] is calculated_demonstrated,
                    bound_model
                    == "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1",
                    count >= 1_000_000.0,
                    count.is_integer(),
                    count_std > 0.0,
                    background >= 0.0,
                    background.is_integer(),
                    background_std > 0.0,
                    background - 3.0 * background_std >= 0.0,
                    count
                    - 3.0 * count_std
                    - background
                    - 3.0 * background_std
                    >= 1_000_000.0,
                    0.0 < efficiency <= 1.0,
                    efficiency_std > 0.0,
                    efficiency - 3.0 * efficiency_std > 0.0,
                    duration > 0.0,
                    duration_std > 0.0,
                    duration - 3.0 * duration_std > 0.0,
                    calculated_rate_std is not None,
                    math.isclose(rate, calculated_rate, rel_tol=2.0e-15, abs_tol=0.0),
                    calculated_rate_std is not None
                    and math.isclose(
                        rate_linearized_std,
                        calculated_rate_std,
                        rel_tol=2.0e-15,
                        abs_tol=0.0,
                    ),
                    math.isclose(
                        rate_lower,
                        calculated_rate_lower,
                        rel_tol=2.0e-15,
                        abs_tol=0.0,
                    ),
                    calculated_demonstrated,
                )
            )
        )

    source_polarization_values = (
        audit.validated_reactor_rate_deuteron_vector_polarization,
        audit.validated_reactor_rate_deuteron_vector_polarization_std,
        audit.validated_reactor_rate_deuteron_tensor_polarization,
        audit.validated_reactor_rate_deuteron_tensor_polarization_std,
        audit.validated_reactor_rate_triton_vector_polarization,
        audit.validated_reactor_rate_triton_vector_polarization_std,
    )
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in source_polarization_values
    ):
        return False
    source_p_d, source_p_d_std, source_p_zz, source_p_zz_std, source_p_t, source_p_t_std = (
        float(value) for value in source_polarization_values
    )
    source_p_d_low = source_p_d - 3.0 * source_p_d_std
    source_p_d_high = source_p_d + 3.0 * source_p_d_std
    source_p_zz_low = source_p_zz - 3.0 * source_p_zz_std
    source_p_zz_high = source_p_zz + 3.0 * source_p_zz_std
    source_p_t_low = source_p_t - 3.0 * source_p_t_std
    source_p_t_high = source_p_t + 3.0 * source_p_t_std
    source_product_lower = source_p_d_low * source_p_t_low
    deuteron_polarization_correlation = _semantic_correlation_matrix(
        audit.validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix,
        size=2,
    )
    source_rate_record_pass = all(
        (
            all(source_rate_record_flags),
            deuteron_polarization_correlation is not None,
            source_p_d_std > 0.0,
            source_p_zz_std > 0.0,
            source_p_t_std > 0.0,
            0.0 <= source_p_d_low <= source_p_d_high <= 1.0,
            -2.0 <= source_p_zz_low <= source_p_zz_high <= 1.0,
            0.0 <= source_p_t_low <= source_p_t_high <= 1.0,
            3.0 * source_p_d_high <= 2.0 + source_p_zz_low,
        )
    )
    burn_values = (
        audit.validated_burn_weighted_dt_polarization_product,
        audit.validated_burn_weighted_dt_polarization_product_std,
        audit.validated_burn_weighted_deuteron_tensor_polarization,
        audit.validated_burn_weighted_deuteron_tensor_polarization_std,
        audit.validated_burn_retention_measurement_duration_s,
        audit.validated_burn_retention_measurement_duration_std_s,
        audit.validated_burn_retention_reaction_count,
    )
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in burn_values
    ):
        return False
    (
        burn_product,
        burn_product_std,
        burn_p_zz,
        burn_p_zz_std,
        burn_duration,
        burn_duration_std,
        burn_reaction_count,
    ) = (float(value) for value in burn_values)
    burn_correlation = _semantic_correlation_matrix(
        audit.validated_burn_retention_uncertainty_correlation_matrix,
        size=2,
    )
    burn_observable_semantics = (
        "burn_weighted_dt_vector_product_and_deuteron_tensor_joint_estimator_v1"
    )
    burn_product_low = burn_product - 3.0 * burn_product_std
    burn_product_high = burn_product + 3.0 * burn_product_std
    burn_p_zz_low = burn_p_zz - 3.0 * burn_p_zz_std
    burn_p_zz_high = burn_p_zz + 3.0 * burn_p_zz_std
    burn_retention_record_pass = all(
        (
            burn_product_std > 0.0,
            burn_p_zz_std > 0.0,
            burn_duration > 0.0,
            burn_duration_std > 0.0,
            burn_duration - 3.0 * burn_duration_std > 0.0,
            burn_reaction_count >= 1_000_000.0,
            burn_reaction_count.is_integer(),
            burn_correlation is not None,
            audit.validated_burn_retention_observable_semantics
            == burn_observable_semantics,
            burn_product_low >= DEFAULT_REQUIRED_BURN_POLARIZATION_PRODUCT,
            burn_product_high <= source_product_lower,
            0.0 <= burn_product_low <= burn_product_high <= 1.0,
            -2.0 <= burn_p_zz_low <= burn_p_zz_high <= 1.0,
            burn_product_high <= (2.0 + burn_p_zz_low) / 3.0,
            burn_retention_artifact_pass,
        )
    )

    wall_claims = _artifact_derived_claims_object(wall_artifact)
    wall_values = (
        audit.validated_wall_plug_energy_per_injected_dt_pair_ev,
        audit.validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev,
        audit.validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev,
        audit.validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev,
        audit.validated_wall_plug_total_energy_j,
        audit.validated_wall_plug_total_energy_linearized_std_j,
        audit.validated_wall_plug_total_energy_lower_3sigma_j,
        audit.validated_wall_plug_total_energy_upper_3sigma_j,
        audit.validated_wall_plug_measurement_duration_s,
        audit.validated_wall_plug_measurement_duration_std_s,
        audit.validated_wall_plug_measurement_injected_dt_pair_count,
        audit.validated_wall_plug_measurement_injected_dt_pair_count_std,
        audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_s,
        audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s,
        audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s,
        audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s,
    )
    if wall_claims is None or not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in wall_values
    ):
        return False
    (
        energy,
        energy_linearized_std,
        energy_lower,
        energy_upper,
        total_energy_j,
        total_energy_linearized_std_j,
        total_energy_lower,
        total_energy_upper,
        duration,
        duration_std,
        pair_count,
        pair_count_std,
        pair_flow,
        pair_flow_linearized_std,
        pair_flow_lower,
        pair_flow_upper,
    ) = (float(value) for value in wall_values)
    component_energies = wall_claims.get("wall_plug_component_energies_j")
    component_stds = wall_claims.get("wall_plug_component_energy_stds_j")
    component_lowers = wall_claims.get("wall_plug_component_energy_lower_3sigma_j")
    component_uppers = wall_claims.get("wall_plug_component_energy_upper_3sigma_j")
    categories = wall_claims.get("wall_plug_component_categories")
    if (
        not isinstance(component_energies, dict)
        or not isinstance(component_stds, dict)
        or not isinstance(component_lowers, dict)
        or not isinstance(component_uppers, dict)
        or not isinstance(categories, list)
        or len(categories) != 11
        or categories != sorted(categories)
        or set(component_energies) != set(categories)
        or set(component_stds) != set(categories)
        or set(component_lowers) != set(categories)
        or set(component_uppers) != set(categories)
    ):
        return False
    component_values = tuple(component_energies[category] for category in categories)
    component_std_values = tuple(component_stds[category] for category in categories)
    component_lower_values = tuple(component_lowers[category] for category in categories)
    component_upper_values = tuple(component_uppers[category] for category in categories)
    if not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in (
            *component_values,
            *component_std_values,
            *component_lower_values,
            *component_upper_values,
        )
    ):
        return False
    wall_correlation = _semantic_correlation_matrix(
        audit.validated_wall_plug_uncertainty_correlation_matrix,
        size=13,
    )
    if wall_correlation is None:
        return False
    if duration <= 0.0 or pair_count <= 0.0:
        return False
    wall_stds = tuple(float(value) for value in component_std_values) + (
        pair_count_std,
        duration_std,
    )
    calculated_total_energy = math.fsum(float(value) for value in component_values)
    calculated_component_lowers = tuple(
        float(value) - 3.0 * float(std)
        for value, std in zip(component_values, component_std_values, strict=True)
    )
    calculated_component_uppers = tuple(
        float(value) + 3.0 * float(std)
        for value, std in zip(component_values, component_std_values, strict=True)
    )
    if any(value <= 0.0 for value in calculated_component_lowers):
        return False
    calculated_total_std = _semantic_gradient_standard_deviation(
        (*([1.0] * 11), 0.0, 0.0), wall_stds, wall_correlation
    )
    calculated_energy = calculated_total_energy / pair_count / EV_TO_JOULE
    calculated_energy_std = _semantic_gradient_standard_deviation(
        (
            *([1.0 / pair_count / EV_TO_JOULE] * 11),
            -calculated_total_energy / (pair_count * pair_count) / EV_TO_JOULE,
            0.0,
        ),
        wall_stds,
        wall_correlation,
    )
    calculated_pair_flow = pair_count / duration
    calculated_pair_flow_std = _semantic_gradient_standard_deviation(
        (*([0.0] * 11), 1.0 / duration, -pair_count / (duration * duration)),
        wall_stds,
        wall_correlation,
    )
    calculated_total_lower = math.fsum(calculated_component_lowers)
    calculated_total_upper = math.fsum(calculated_component_uppers)
    pair_count_lower = pair_count - 3.0 * pair_count_std
    pair_count_upper = pair_count + 3.0 * pair_count_std
    duration_lower = duration - 3.0 * duration_std
    duration_upper = duration + 3.0 * duration_std
    calculated_energy_lower = (
        calculated_total_lower / pair_count_upper / EV_TO_JOULE
    )
    calculated_energy_upper = (
        calculated_total_upper / pair_count_lower / EV_TO_JOULE
    )
    calculated_pair_flow_lower = pair_count_lower / duration_upper
    calculated_pair_flow_upper = pair_count_upper / duration_lower
    wall_plug_record_pass = all(
        (
            wall_plug_artifact_pass,
            energy > 0.0,
            energy_linearized_std > 0.0,
            total_energy_j > 0.0,
            total_energy_linearized_std_j > 0.0,
            total_energy_lower > 0.0,
            energy_lower > 0.0,
            duration > 0.0,
            duration_std > 0.0,
            duration - 3.0 * duration_std > 0.0,
            pair_count >= 1_000_000.0,
            pair_count.is_integer(),
            pair_count_std > 0.0,
            pair_count_lower > 0.0,
            audit.validated_wall_plug_3sigma_bound_model
            == "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1",
            tuple(float(value) for value in component_lower_values)
            == calculated_component_lowers,
            tuple(float(value) for value in component_upper_values)
            == calculated_component_uppers,
            calculated_total_std is not None,
            calculated_energy_std is not None,
            calculated_pair_flow_std is not None,
            math.isclose(total_energy_j, calculated_total_energy, rel_tol=2.0e-15),
            calculated_total_std is not None
            and math.isclose(
                total_energy_linearized_std_j,
                calculated_total_std,
                rel_tol=2.0e-15,
            ),
            math.isclose(
                total_energy_lower,
                calculated_total_lower,
                rel_tol=2.0e-15,
            ),
            math.isclose(total_energy_upper, calculated_total_upper, rel_tol=2.0e-15),
            math.isclose(energy, calculated_energy, rel_tol=2.0e-15),
            calculated_energy_std is not None
            and math.isclose(
                energy_linearized_std,
                calculated_energy_std,
                rel_tol=2.0e-15,
            ),
            math.isclose(
                energy_lower,
                calculated_energy_lower,
                rel_tol=2.0e-15,
            ),
            math.isclose(energy_upper, calculated_energy_upper, rel_tol=2.0e-15),
            math.isclose(pair_flow, calculated_pair_flow, rel_tol=2.0e-15),
            calculated_pair_flow_std is not None
            and math.isclose(
                pair_flow_linearized_std,
                calculated_pair_flow_std,
                rel_tol=2.0e-15,
            ),
            math.isclose(
                pair_flow_lower,
                calculated_pair_flow_lower,
                rel_tol=2.0e-15,
            ),
            math.isclose(
                pair_flow_upper,
                calculated_pair_flow_upper,
                rel_tol=2.0e-15,
            ),
            pair_flow_lower >= audit.required_per_species_fuel_rate_s,
        )
    )
    historical_proxy_classification_pass = all(
        (
            audit.coulter_record_meets_reactor_deuterium_rate is False,
            audit.anke_record_meets_reactor_deuterium_rate is False,
            audit.anke_rf_power_is_partial_component_not_complete_wall_plug is True,
            audit.sofikitis_reactor_continuous_throughput_measured is False,
            audit.utsuro_hd_proxy_proof_of_concept_performed is True,
            audit.utsuro_actual_polarized_tritium_source_demonstrated is False,
            audit.utsuro_reactor_continuous_throughput_measured is False,
            audit.cook_result_is_simulation_not_burn_measurement is True,
            audit.souers_actual_solid_dt_triton_relaxation_measured is True,
            audit.souers_result_is_preburn_not_burn_weighted_plasma_retention is True,
            audit.collins_actual_solid_dt_deuteron_nmr_relaxation_measured is True,
            audit.collins_result_is_preburn_not_burn_weighted_plasma_retention is True,
        )
    )
    return all(
        flag is True
        for flag in (
            wall_plug_record_pass,
            source_rate_record_pass,
            burn_retention_record_pass,
            _plant_configuration_id_available(
                audit.validated_integrated_plant_configuration_id
            ),
            _source_plant_ledger_integrity_pass(audit),
            integrated_artifact_storage_pass,
            integrated_artifact_cross_join_pass,
            historical_proxy_classification_pass,
            audit.reference_plant_scale_invariant_pass,
            audit.reactor_rate_polarized_deuterium_source_demonstrated,
            audit.reactor_rate_polarized_tritium_source_demonstrated,
            audit.burn_weighted_dt_polarization_product_measured,
            audit.complete_wall_plug_energy_per_injected_pair_measured,
            audit.complete_wall_plug_uncertainty_flow_and_provenance_available,
            audit.source_retention_and_wall_plug_gate_pass,
        )
    )


def _derive_burn_state_operator_contraction(
    reaction: PublishedPolarizedDTReactivityAudit,
    source: PolarizedFuelPrimaryEvidenceAudit,
    *,
    repository_root: str | Path | None = None,
) -> _BurnStateOperatorContraction | None:
    if (
        type(reaction) is not PublishedPolarizedDTReactivityAudit
        or type(source) is not PolarizedFuelPrimaryEvidenceAudit
        or not _published_reaction_evidence_record_integrity_pass(
            reaction, repository_root=repository_root
        )
        or not _polarized_source_evidence_record_integrity_pass(
            source, repository_root=repository_root
        )
    ):
        return None
    operator_artifact = reaction.target_state_operator_artifact
    burn_artifact = source.validated_burn_retention_artifact
    if (
        operator_artifact.expected_manifest_sha256
        != source.validated_burn_reaction_operator_provenance_digest
    ):
        return None
    operator_claims = _artifact_derived_claims_object(operator_artifact)
    burn_claims = _artifact_derived_claims_object(burn_artifact)
    if operator_claims is None or burn_claims is None:
        return None
    relationship = (
        "unknown_cross_artifact_correlation_componentwise_3sigma_robust_box_v1"
    )
    semantics = (
        "engineering_control_componentwise_3sigma_burn_box_affine_operator_with_phase_stress_v2"
    )
    if (
        operator_claims.get("operator_variant")
        != "six_channel_longitudinal_initial_spin_v1"
        or operator_claims.get("burn_state_operator_contraction_model")
        != "maxwellian_affine_intercept_vector_product_and_deuteron_tensor_v2"
        or operator_claims.get("burn_state_operator_contraction_variables")
        != [
            "intercept_one",
            "burn_weighted_dt_polarization_product",
            "burn_weighted_deuteron_tensor_polarization",
        ]
        or operator_claims.get("burn_state_operator_channel_projection_matrix")
        != [list(row) for row in BURN_STATE_OPERATOR_CHANNEL_PROJECTION]
        or operator_claims.get("simultaneous_spin_reversal_exact") is not True
        or operator_claims.get("covariance_mode_spin_reversal_exact") is not True
        or operator_claims.get("burn_state_operator_response_uncertainty_scope")
        != "nuclear_model_systematic"
        or operator_claims.get("burn_state_operator_response_uncertainty_level")
        != "1sigma"
        or operator_claims.get("maxwellian_quadrature_relative_tolerance")
        != BURN_STATE_OPERATOR_PHASE_RELATIVE_TOLERANCE
        or burn_claims.get("operator_burn_uncertainty_relationship") != relationship
        or burn_claims.get("reaction_operator_manifest_sha256")
        != operator_artifact.expected_manifest_sha256
        or burn_claims.get("burn_quantization_axis")
        != operator_claims.get("quantization_axis")
        or burn_claims.get("burn_initial_state_basis")
        != operator_claims.get("initial_state_basis")
        or burn_claims.get("burn_deuteron_tensor_polarization_convention")
        != "Pzz=p_plus+p_minus-2*p_zero"
        or burn_claims.get("burn_collision_ensemble_temperature_kev")
        != operator_claims.get("temperature_kev")
        or burn_claims.get("burn_spin_state_energy_dependence")
        != "declared_energy_independent_diagonal_spin_state_engineering_control_v1"
    ):
        return None
    burn_values = (
        source.validated_burn_weighted_dt_polarization_product,
        source.validated_burn_weighted_dt_polarization_product_std,
        source.validated_burn_weighted_deuteron_tensor_polarization,
        source.validated_burn_weighted_deuteron_tensor_polarization_std,
    )
    if not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in burn_values
    ):
        return None
    product, product_std, tensor, tensor_std = (
        float(value) for value in burn_values
    )
    if product_std <= 0.0 or tensor_std <= 0.0:
        return None
    product_bounds = (product - 3.0 * product_std, product + 3.0 * product_std)
    tensor_bounds = (tensor - 3.0 * tensor_std, tensor + 3.0 * tensor_std)
    if (
        not 0.0 <= product_bounds[0] <= product_bounds[1] <= 1.0
        or not -2.0 <= tensor_bounds[0] <= tensor_bounds[1] <= 1.0
        or product_bounds[1] > (2.0 + tensor_bounds[0]) / 3.0
    ):
        return None
    corners = tuple(
        np.asarray((1.0, product_value, tensor_value), dtype=np.float64)
        for product_value in product_bounds
        for tensor_value in tensor_bounds
    )

    phase_summaries = operator_claims.get("maxwellian_quadrature_phase_summaries")
    if not isinstance(phase_summaries, list) or len(phase_summaries) != 6:
        return None
    expected_phase_keys = {
        (2, 0),
        (2, 1),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    }
    observed_phase_keys: set[tuple[int, int]] = set()
    model_documents: list[tuple[object, object]] = [
        (
            operator_claims.get("burn_state_operator_response_coefficients"),
            operator_claims.get(
                "burn_state_operator_response_coefficient_covariance"
            ),
        )
    ]
    for summary in phase_summaries:
        if not isinstance(summary, dict):
            return None
        stride = summary.get("stride")
        phase = summary.get("phase")
        if (
            not isinstance(stride, int)
            or isinstance(stride, bool)
            or not isinstance(phase, int)
            or isinstance(phase, bool)
            or (stride, phase) in observed_phase_keys
        ):
            return None
        observed_phase_keys.add((stride, phase))
        model_documents.append(
            (
                summary.get("burn_state_operator_response_coefficients"),
                summary.get(
                    "burn_state_operator_response_coefficient_covariance"
                ),
            )
        )
    if observed_phase_keys != expected_phase_keys:
        return None

    models: list[tuple[np.ndarray, np.ndarray]] = []
    for coefficient_document, covariance_document in model_documents:
        if not isinstance(coefficient_document, (list, tuple)) or len(
            coefficient_document
        ) != 3:
            return None
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in coefficient_document
        ):
            return None
        coefficients = np.asarray(coefficient_document, dtype=np.float64)
        covariance = _semantic_psd_matrix(covariance_document, size=3)
        if covariance is None:
            return None
        models.append((coefficients, covariance))

    def _evaluate_model(
        coefficients: np.ndarray,
        covariance: np.ndarray,
        state: np.ndarray,
    ) -> tuple[float, float, float] | None:
        covariance_scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
        variance_tolerance = 64.0 * np.finfo(np.float64).eps * covariance_scale
        ratio = float(coefficients @ state)
        variance = float(state @ covariance @ state)
        if (
            not math.isfinite(ratio)
            or ratio <= 0.0
            or not math.isfinite(variance)
            or variance < -variance_tolerance
        ):
            return None
        operator_std = math.sqrt(max(0.0, variance))
        lower = ratio - 3.0 * operator_std
        if not math.isfinite(lower):
            return None
        return ratio, operator_std, lower

    model_lower_bounds: list[list[float]] = [[] for _ in models]
    max_phase_relative_residual = 0.0
    for corner in corners:
        full_evaluation = _evaluate_model(*models[0], corner)
        if full_evaluation is None:
            return None
        full_ratio, full_std, full_lower_at_corner = full_evaluation
        model_lower_bounds[0].append(full_lower_at_corner)
        for model_index, model in enumerate(models[1:], start=1):
            phase_evaluation = _evaluate_model(*model, corner)
            if phase_evaluation is None:
                return None
            phase_ratio, phase_std, phase_lower_at_corner = phase_evaluation
            model_lower_bounds[model_index].append(phase_lower_at_corner)
            ratio_residual = abs(phase_ratio - full_ratio) / abs(full_ratio)
            if full_std == 0.0:
                std_residual = 0.0 if phase_std == 0.0 else math.inf
            else:
                std_residual = abs(phase_std - full_std) / full_std
            phase_residual = max(ratio_residual, std_residual)
            if not math.isfinite(phase_residual):
                return None
            max_phase_relative_residual = max(
                max_phase_relative_residual, phase_residual
            )
    if max_phase_relative_residual > BURN_STATE_OPERATOR_PHASE_RELATIVE_TOLERANCE:
        return None

    model_minima = [min(lower_bounds) for lower_bounds in model_lower_bounds]
    central_state = np.asarray((1.0, product, tensor), dtype=np.float64)
    central_evaluation = _evaluate_model(*models[0], central_state)
    if central_evaluation is None:
        return None
    central_ratio = central_evaluation[0]
    full_lower = model_minima[0]
    coarsened_lower = min(model_minima[1:])
    robust_lower = min(model_minima)
    robust_margin_per_sigma_equivalent = (central_ratio - robust_lower) / 3.0
    if (
        not math.isfinite(central_ratio)
        or not math.isfinite(robust_margin_per_sigma_equivalent)
        or robust_margin_per_sigma_equivalent < 0.0
    ):
        return None
    return _BurnStateOperatorContraction(
        central_reactivity_ratio=central_ratio,
        robust_margin_per_sigma_equivalent=robust_margin_per_sigma_equivalent,
        robust_lower_3sigma=robust_lower,
        full_grid_robust_lower_3sigma=full_lower,
        coarsened_robust_lower_3sigma=coarsened_lower,
        max_phase_relative_residual=max_phase_relative_residual,
        evaluated_model_count=len(model_documents),
        evaluated_corner_count=len(model_documents) * len(corners),
        semantics=semantics,
        control_passes_one_percent=(
            robust_lower >= DEFAULT_TARGET_REACTIVITY_RATIO
        ),
    )


def _burn_state_operator_evidence_record_integrity_pass(
    reaction: PublishedPolarizedDTReactivityAudit,
    source: PolarizedFuelPrimaryEvidenceAudit,
    *,
    repository_root: str | Path | None = None,
) -> bool:
    contraction = _derive_burn_state_operator_contraction(
        reaction, source, repository_root=repository_root
    )
    if contraction is None:
        return False
    recorded_values = (
        source.validated_burn_state_operator_reactivity_ratio,
        source.validated_burn_state_operator_reactivity_ratio_robust_margin_per_sigma_equivalent,
        source.validated_burn_state_operator_reactivity_ratio_robust_lower_3sigma,
        source.validated_burn_state_operator_max_phase_relative_residual,
    )
    if not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in recorded_values
    ):
        return False
    ratio, robust_margin, lower, phase_residual = (
        float(value) for value in recorded_values
    )
    return all(
        (
            math.isclose(
                ratio,
                contraction.central_reactivity_ratio,
                rel_tol=2.0e-15,
                abs_tol=0.0,
            ),
            math.isclose(
                robust_margin,
                contraction.robust_margin_per_sigma_equivalent,
                rel_tol=2.0e-15,
                abs_tol=0.0,
            ),
            math.isclose(
                lower,
                contraction.robust_lower_3sigma,
                rel_tol=2.0e-15,
                abs_tol=0.0,
            ),
            math.isclose(
                phase_residual,
                contraction.max_phase_relative_residual,
                rel_tol=2.0e-15,
                abs_tol=0.0,
            ),
            source.validated_burn_state_operator_contraction_semantics
            == contraction.semantics,
            source.burn_state_operator_componentwise_3sigma_stress_envelope_control_pass
            is contraction.control_passes_one_percent,
            contraction.control_passes_one_percent,
        )
    )


def physical_polarized_source_evidence_gate_pass(
    audit: PolarizedFuelPrimaryEvidenceAudit,
) -> bool:
    """Accept only a semantically complete, fresh canonical source audit."""

    if type(audit) is not PolarizedFuelPrimaryEvidenceAudit:
        return False
    try:
        expected = audit_polarized_fuel_primary_evidence(
            reference_fusion_power_w=audit.declared_reference_fusion_power_w,
            single_pass_burn_fraction=audit.declared_single_pass_burn_fraction,
        )
        return (
            audit == expected
            and _polarized_source_evidence_record_integrity_pass(audit)
            and all(
                physical_canonical_repository_evidence_artifact_gate_pass(artifact)
                for artifact in (
                    audit.validated_reactor_rate_deuterium_source_artifact,
                    audit.validated_reactor_rate_tritium_source_artifact,
                    audit.validated_burn_retention_artifact,
                    audit.validated_wall_plug_artifact,
                )
            )
        )
    except Exception:
        return False


def current_fusion_polarized_evidence_report() -> FusionPolarizedEvidenceReport:
    """Compose the reaction-curve and source evidence audits."""

    reaction = audit_published_polarized_dt_reactivity()
    source = audit_polarized_fuel_primary_evidence()
    reaction_pass = physical_published_reaction_evidence_gate_pass(reaction)
    source_pass = physical_polarized_source_evidence_gate_pass(source)
    contraction = _derive_burn_state_operator_contraction(reaction, source)
    combined_pass = _burn_state_operator_evidence_record_integrity_pass(
        reaction, source
    )
    return FusionPolarizedEvidenceReport(
        schema_version="fusion-polarized-evidence-v7",
        reaction=reaction,
        source=source,
        energy_dependent_figure_control_reproduced=(
            reaction.figure_digitized_one_percent_control_pass
        ),
        physical_reaction_evidence_gate_pass=reaction_pass,
        physical_source_evidence_gate_pass=source_pass,
        burn_state_operator_componentwise_3sigma_stress_envelope_control_pass=(
            combined_pass
        ),
        energy_resolved_burn_operator_sufficient_statistics_available=(
            source.energy_resolved_burn_operator_sufficient_statistics_available
        ),
        validated_burn_state_operator_reactivity_ratio=(
            contraction.central_reactivity_ratio if contraction is not None else None
        ),
        validated_burn_state_operator_reactivity_ratio_robust_margin_per_sigma_equivalent=(
            contraction.robust_margin_per_sigma_equivalent
            if contraction is not None
            else None
        ),
        validated_burn_state_operator_reactivity_ratio_robust_lower_3sigma=(
            contraction.robust_lower_3sigma if contraction is not None else None
        ),
        validated_burn_state_operator_max_phase_relative_residual=(
            contraction.max_phase_relative_residual
            if contraction is not None
            else None
        ),
        physical_spin_fusion_evidence_gate_pass=(
            reaction_pass
            and source_pass
            and combined_pass
            and source.energy_resolved_burn_operator_sufficient_statistics_available
        ),
        maximum_supported_stage=(
            "PUBLISHED_ENERGY_DEPENDENT_FIGURE_DIGITIZED_MAXWELLIAN_CONTROL_"
            "PHYSICAL_EVIDENCE_FAIL_CLOSED"
        ),
        next_required_input=(
            "author numeric NCSMC spin energy grid/S-matrix with systematic covariance "
            "(the verified Han V1 files contain pointwise ERR but no numeric covariance); "
            "direct Czz or polarized D-T rate validation with target-state ratio uncertainty "
            "and a byte-verified operator manifest; reactor-rate polarized T and D sources "
            "with species-separated 3-sigma rates and byte-verified manifests; burn-weighted "
            "energy-resolved vector/tensor sufficient statistics with cross-bin covariance "
            "on the operator grid (the scalar constant-state contraction is control-only); "
            "complete measured wall-plug eV/injected-pair with "
            "uncertainty, flow, and byte-verified provenance"
        ),
        conclusion=(
            "The published energy-dependent full-alignment curve independently clears the "
            "one-percent 10 keV Maxwellian control after digitization uncertainty. All six "
            "Han ScienceDB V1 files are byte-verified, but they are unpolarized point tables "
            "without the numeric covariance matrix. The spin operator and every reactor "
            "source/retention/wall-plug gate remain open; scalar burn-state/operator coupling "
            "is retained only as an engineering stress-envelope control until energy-resolved "
            "sufficient statistics exist. No physical or CE branch is promoted."
        ),
    )


def physical_polarized_evidence_gate_pass(
    report: FusionPolarizedEvidenceReport,
) -> bool:
    """Require a fresh canonical report and both underlying physical audits."""

    if type(report) is not FusionPolarizedEvidenceReport:
        return False
    try:
        expected = current_fusion_polarized_evidence_report()
        if report != expected:
            return False
        reaction_pass = physical_published_reaction_evidence_gate_pass(report.reaction)
        source_pass = physical_polarized_source_evidence_gate_pass(report.source)
        combined_pass = _burn_state_operator_evidence_record_integrity_pass(
            report.reaction, report.source
        )
        distinct_artifact_bundle = _distinct_artifact_storage_pass(
            report.reaction.target_state_operator_artifact,
            report.source.validated_reactor_rate_deuterium_source_artifact,
            report.source.validated_reactor_rate_tritium_source_artifact,
            report.source.validated_burn_retention_artifact,
            report.source.validated_wall_plug_artifact,
        )
        return all(
            flag is True
            for flag in (
                reaction_pass,
                source_pass,
                distinct_artifact_bundle,
                report.physical_reaction_evidence_gate_pass,
                report.physical_source_evidence_gate_pass,
                combined_pass,
                report.burn_state_operator_componentwise_3sigma_stress_envelope_control_pass,
                report.energy_resolved_burn_operator_sufficient_statistics_available,
                report.physical_spin_fusion_evidence_gate_pass,
            )
        )
    except Exception:
        return False


__all__ = [
    "BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS",
    "EVIDENCE_ARTIFACT_CLAIMS_SCHEMA_VERSION",
    "EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "EvidenceArtifactAudit",
    "FusionPolarizedEvidenceReport",
    "POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS",
    "POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS",
    "PolarizedFuelPrimaryEvidenceAudit",
    "PublishedPolarizedDTReactivityAudit",
    "REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS",
    "REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS",
    "audit_polarized_fuel_primary_evidence",
    "audit_published_polarized_dt_reactivity",
    "audit_repository_evidence_artifact",
    "current_fusion_polarized_evidence_report",
    "physical_polarized_source_evidence_gate_pass",
    "physical_polarized_evidence_gate_pass",
    "physical_canonical_repository_evidence_artifact_gate_pass",
    "physical_published_reaction_evidence_gate_pass",
    "repository_evidence_artifact_integrity_pass",
]
