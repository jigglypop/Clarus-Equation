from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import fusion_polarized_evidence_loop as evidence_loop
from reality_stone.clarus.fusion_polarized_artifact_payloads import (
    BASELINE_CROSS_SECTION_MODEL,
    CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER,
    COVARIANCE_FACTOR_REPRESENTATION,
    COVARIANCE_UNCERTAINTY_LEVEL,
    COVARIANCE_UNCERTAINTY_SCOPE,
    ENGINEERING_PRIMARY_PAYLOAD_ROLES,
    INITIAL_STATE_BASIS,
    LONGITUDINAL_CHANNEL_ORDER,
    LONGITUDINAL_QUANTIZATION_AXIS,
    PAYLOAD_SCHEMA_VERSION,
    REACTION_ENERGY_FRAME,
    REACTION_OBSERVABLE,
    SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT,
    WALL_PLUG_COMPONENT_CATEGORIES,
    derive_artifact_claims,
)
from reality_stone.clarus.fusion_polarized_evidence_loop import (
    BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
    EVIDENCE_ARTIFACT_CLAIMS_SCHEMA_VERSION,
    EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    HUPIN_FIGURE_4_IMAGE_SHA256,
    HUPIN_FIGURE_4_IMAGE_SIZE_PX,
    POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
    POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
    REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
    REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
    audit_repository_evidence_artifact,
    audit_polarized_fuel_primary_evidence,
    audit_published_polarized_dt_reactivity,
    current_fusion_polarized_evidence_report,
    physical_canonical_repository_evidence_artifact_gate_pass,
    physical_polarized_evidence_gate_pass,
    physical_polarized_source_evidence_gate_pass,
    physical_published_reaction_evidence_gate_pass,
    repository_evidence_artifact_integrity_pass,
)


OPERATOR_PROVENANCE_DIGEST = "sha256:" + "a" * 64
SOURCE_D_PROVENANCE_DIGEST = "sha256:" + "b" * 64
SOURCE_T_PROVENANCE_DIGEST = "sha256:" + "c" * 64
BURN_PROVENANCE_DIGEST = "sha256:" + "d" * 64
WALL_PLUG_PROVENANCE_DIGEST = "sha256:" + "e" * 64


class _AlwaysEqualProxy:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _plant_payload(audit, *, configuration_id: str | None = None) -> dict[str, object]:
    return {
        "configuration_id": (
            configuration_id
            if configuration_id is not None
            else audit.validated_integrated_plant_configuration_id
        ),
        "fusion_power_w": audit.declared_reference_fusion_power_w,
        "fusion_power_unit": "W",
        "single_pass_burn_fraction": audit.declared_single_pass_burn_fraction,
        "burn_fraction_unit": "1",
    }


def _reaction_raw_payload(audit=None) -> dict[str, object]:
    if audit is None:
        p_d, p_zz, p_t = 0.2, 0.0, 0.2
        ratio, ratio_std, target = 1.02, 0.002, 1.01
    else:
        p_d = audit.validated_target_deuteron_vector_polarization
        p_zz = audit.validated_target_deuteron_tensor_polarization
        p_t = audit.validated_target_triton_vector_polarization
        ratio = audit.validated_target_state_maxwellian_reactivity_ratio
        ratio_std = audit.validated_target_state_maxwellian_reactivity_ratio_std
        target = audit.target_reactivity_ratio
    point_count = 101
    energy = np.geomspace(0.5, 550.0, point_count).tolist()
    correlation = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    p_product = p_d * p_t
    if p_product == 0.0 or ratio_std <= 0.0:
        raise ValueError("channel fixture requires nonzero pD*pT and uncertainty")
    ratio_scale = (ratio - 1.0) / p_product
    uncertainty_scale = ratio_std / abs(p_product)
    p_plus = (2.0 + p_zz + 3.0 * p_d) / 6.0
    p_zero = (1.0 - p_zz) / 3.0
    p_minus = (2.0 + p_zz - 3.0 * p_d) / 6.0
    t_minus = (1.0 - p_t) / 2.0
    t_plus = (1.0 + p_t) / 2.0
    weights = [
        p_minus * t_minus,
        p_minus * t_plus,
        p_zero * t_minus,
        p_zero * t_plus,
        p_plus * t_minus,
        p_plus * t_plus,
    ]
    payload: dict[str, object] = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        "payload_role": "operator_grid",
        "operator_variant": SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT,
        "energy_frame": REACTION_ENERGY_FRAME,
        "observable": REACTION_OBSERVABLE,
        "baseline_cross_section_model": BASELINE_CROSS_SECTION_MODEL,
        "quantization_axis": LONGITUDINAL_QUANTIZATION_AXIS,
        "initial_state_basis": INITIAL_STATE_BASIS,
        "temperature_kev": 10.0,
        "temperature_unit": "keV",
        "target_reactivity_ratio": target,
        "energy_grid_kev": energy,
        "energy_unit": "keV",
        "channel_order": list(LONGITUDINAL_CHANNEL_ORDER),
        "channel_reactivity_ratio_grids": [],
        "target_channel_weights": weights,
        "target_state_reactivity_ratio_grid": [],
        "ratio_unit": "1",
        "reactivity_ratio_covariance_factor": {
            "representation": COVARIANCE_FACTOR_REPRESENTATION,
            "flatten_order": CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER,
            "factor_unit": "1",
            "uncertainty_scope": COVARIANCE_UNCERTAINTY_SCOPE,
            "uncertainty_level": COVARIANCE_UNCERTAINTY_LEVEL,
            "modes": [],
        },
        "target_deuteron_vector_polarization": p_d,
        "target_deuteron_tensor_polarization": p_zz,
        "target_triton_vector_polarization": p_t,
        "polarization_unit": "1",
    }
    for _ in range(4):
        channel_constants = [1.0 + ratio_scale * value for value in correlation]
        payload["channel_reactivity_ratio_grids"] = [
            [value] * point_count for value in channel_constants
        ]
        payload["target_state_reactivity_ratio_grid"] = [
            math.fsum(
                weight * value
                for weight, value in zip(weights, channel_constants, strict=True)
            )
        ] * point_count
        payload["reactivity_ratio_covariance_factor"]["modes"] = [  # type: ignore[index]
            [
                uncertainty_scale * correlation[channel]
                for channel in range(6)
                for _ in range(point_count)
            ]
        ]
        derived = derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256="sha256:" + "0" * 64,
        ).claims
        derived_ratio = float(
            derived["validated_target_state_maxwellian_reactivity_ratio"]
        )
        derived_std = float(
            derived["validated_target_state_maxwellian_reactivity_ratio_std"]
        )
        if derived_ratio != ratio:
            ratio_scale *= (ratio - 1.0) / (derived_ratio - 1.0)
        if derived_std != ratio_std:
            uncertainty_scale *= ratio_std / derived_std
    return payload


def _standalone_deuterium_raw_payload() -> dict[str, object]:
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        "payload_role": "measurement",
        "plant_configuration": {
            "configuration_id": "standalone-500mw-v1",
            "fusion_power_w": 5.0e8,
            "fusion_power_unit": "W",
            "single_pass_burn_fraction": 0.01,
            "burn_fraction_unit": "1",
        },
        "measurement": {
            "species": "D",
            "output_count": 20_000_000_000_000_000_000_000,
            "output_count_unit": "particles",
            "output_count_std": 1.0e12,
            "measurement_duration_s": 1.0,
            "duration_unit": "s",
            "polarization_unit": "1",
            "uncertainty_level": "1sigma",
            "deuteron_vector_polarization": 0.5,
            "deuteron_vector_polarization_std": 0.001,
            "deuteron_tensor_polarization": 0.0,
            "deuteron_tensor_polarization_std": 0.001,
        },
    }


def _source_raw_payload(audit, species: str) -> dict[str, object]:
    if species == "D":
        evidence_class = REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS
        count = audit.validated_reactor_rate_deuterium_output_count
        duration = audit.validated_reactor_rate_deuterium_measurement_duration_s
        count_std = audit.validated_reactor_rate_deuterium_output_count_std
        polarization = {
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
    else:
        evidence_class = REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS
        count = audit.validated_reactor_rate_tritium_output_count
        duration = audit.validated_reactor_rate_tritium_measurement_duration_s
        count_std = audit.validated_reactor_rate_tritium_output_count_std
        polarization = {
            "triton_vector_polarization": audit.validated_reactor_rate_triton_vector_polarization,
            "triton_vector_polarization_std": (
                audit.validated_reactor_rate_triton_vector_polarization_std
            ),
        }
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": evidence_class,
        "payload_role": "measurement",
        "plant_configuration": _plant_payload(audit),
        "measurement": {
            "species": species,
            "output_count": count,
            "output_count_unit": "particles",
            "output_count_std": count_std,
            "measurement_duration_s": duration,
            "duration_unit": "s",
            "polarization_unit": "1",
            "uncertainty_level": "1sigma",
            **polarization,
        },
    }


def _burn_raw_payload(
    audit,
    *,
    deuterium_hash: str,
    tritium_hash: str,
    reaction_operator_hash: str,
    configuration_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        "payload_role": "measurement",
        "plant_configuration": _plant_payload(audit, configuration_id=configuration_id),
        "measurement": {
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
            "polarization_unit": "1",
            "uncertainty_level": "1sigma",
            "reaction_count": audit.validated_burn_retention_reaction_count,
            "reaction_count_unit": "reactions",
            "measurement_duration_s": audit.validated_burn_retention_measurement_duration_s,
            "duration_unit": "s",
            "deuterium_source_manifest_sha256": deuterium_hash,
            "tritium_source_manifest_sha256": tritium_hash,
            "reaction_operator_manifest_sha256": reaction_operator_hash,
            "quantization_axis": "longitudinal_beam_axis",
            "initial_state_basis": "diagonal_in_product_spin_projection_basis",
            "deuteron_tensor_polarization_convention": (
                "Pzz=p_plus+p_minus-2*p_zero"
            ),
            "collision_ensemble_temperature_kev": 10.0,
            "temperature_unit": "keV",
            "spin_state_energy_dependence": (
                "declared_energy_independent_diagonal_spin_state_engineering_control_v1"
            ),
        },
    }


def _wall_raw_payload(
    audit, *, deuterium_hash: str, tritium_hash: str, configuration_id: str | None = None
) -> dict[str, object]:
    categories = sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    weight_sum = math.fsum(range(1, len(categories) + 1))
    components = {
        category: {
            "energy_j": audit.validated_wall_plug_total_energy_j * index / weight_sum,
            "energy_std_j": (
                0.01
                * audit.validated_wall_plug_total_energy_j
                * index
                / weight_sum
            ),
        }
        for index, category in enumerate(categories, start=1)
    }
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        "payload_role": "component_ledger",
        "plant_configuration": _plant_payload(audit, configuration_id=configuration_id),
        "ledger": {
            "uncertainty_model": "manifest_bound_correlated_1sigma",
            "energy_unit": "J",
            "components": components,
            "injected_dt_pair_count": (
                audit.validated_wall_plug_measurement_injected_dt_pair_count
            ),
            "pair_count_unit": "pairs",
            "measurement_duration_s": audit.validated_wall_plug_measurement_duration_s,
            "duration_unit": "s",
            "deuterium_source_manifest_sha256": deuterium_hash,
            "tritium_source_manifest_sha256": tritium_hash,
        },
    }


def _artifact_auxiliary_payloads(
    raw_payload: dict[str, object],
) -> dict[str, dict[str, object]]:
    evidence_class = str(raw_payload["evidence_class"])
    if evidence_class in {
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
    }:
        measurement = raw_payload["measurement"]
        species = measurement["species"]  # type: ignore[index]
        pipeline: dict[str, object] = {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "payload_role": "pipeline",
            "species": species,
            "pipeline_id": f"source-{species}-pipeline-v1",
            "rate_estimator": (
                "background_subtracted_efficiency_corrected_count_over_duration_v1"
            ),
            "measurement_duration_s": measurement["measurement_duration_s"],  # type: ignore[index]
            "measurement_duration_std_s": 1.0e-6,
            "duration_unit": "s",
            "uncertainty_variables": [
                "observed_output_count",
                "background_count",
                "detection_efficiency",
                "measurement_duration_s",
            ],
            "correlation_matrix": np.eye(4).tolist(),
            "uncertainty_level": "1sigma",
        }
        if species == "D":
            pipeline.update(
                {
                    "polarization_uncertainty_variables": [
                        "deuteron_vector_polarization",
                        "deuteron_tensor_polarization",
                    ],
                    "polarization_correlation_matrix": [[1.0, 0.2], [0.2, 1.0]],
                }
            )
        return {
            "event_counts": {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "evidence_class": evidence_class,
                "payload_role": "event_counts",
                "species": species,
                "observed_output_count": measurement["output_count"],  # type: ignore[index]
                "observed_output_count_unit": "particles",
                "observed_output_count_std": measurement["output_count_std"],  # type: ignore[index]
            },
            "calibration": {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "evidence_class": evidence_class,
                "payload_role": "calibration",
                "species": species,
                "calibration_id": f"source-{species}-calibration-v1",
                "calibration_model": "absolute_count_efficiency_and_background_v1",
                "detection_efficiency": 1.0,
                "detection_efficiency_std": 1.0e-6,
                "background_count": 3_000_000_000,
                "background_count_std": 1.0e9,
                "background_count_unit": "particles",
            },
            "pipeline": pipeline,
        }
    if evidence_class == BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS:
        measurement = raw_payload["measurement"]
        return {
            "burn_events": {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "evidence_class": evidence_class,
                "payload_role": "burn_events",
                "event_dataset_id": "burn-event-dataset-v1",
                "reaction_count": measurement["reaction_count"],  # type: ignore[index]
                "reaction_count_unit": "reactions",
                "measurement_duration_s": measurement["measurement_duration_s"],  # type: ignore[index]
                "measurement_duration_std_s": 0.01,
                "duration_unit": "s",
            },
            "calibration": {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "evidence_class": evidence_class,
                "payload_role": "calibration",
                "calibration_id": "burn-calibration-v1",
                "calibration_model": (
                    "absolute_burn_product_and_tensor_polarimetry_v1"
                ),
                "product_observable": "burn_weighted_dt_vector_polarization_product",
                "tensor_observable": "burn_weighted_deuteron_tensor_polarization",
            },
            "estimator": {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "evidence_class": evidence_class,
                "payload_role": "estimator",
                "estimator_id": "burn-joint-estimator-v1",
                "observable_semantics": (
                    "burn_weighted_dt_vector_product_and_deuteron_tensor_joint_estimator_v1"
                ),
                "uncertainty_variables": [
                    "burn_weighted_dt_polarization_product",
                    "burn_weighted_deuteron_tensor_polarization",
                ],
                "correlation_matrix": [[1.0, 0.25], [0.25, 1.0]],
                "uncertainty_level": "1sigma",
                "operator_burn_uncertainty_relationship": (
                    "unknown_cross_artifact_correlation_componentwise_3sigma_robust_box_v1"
                ),
                "quantization_axis": measurement["quantization_axis"],
                "initial_state_basis": measurement["initial_state_basis"],
                "deuteron_tensor_polarization_convention": measurement[
                    "deuteron_tensor_polarization_convention"
                ],
                "collision_ensemble_temperature_kev": measurement[
                    "collision_ensemble_temperature_kev"
                ],
                "temperature_unit": measurement["temperature_unit"],
                "spin_state_energy_dependence": measurement[
                    "spin_state_energy_dependence"
                ],
            },
        }
    ledger = raw_payload["ledger"]
    components = ledger["components"]  # type: ignore[index]
    categories = sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    return {
        "meter_records": {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "payload_role": "meter_records",
            "energy_unit": "J",
            "components": {
                category: {
                    "energy_j": components[category]["energy_j"],  # type: ignore[index]
                    "energy_std_j": components[category]["energy_std_j"],  # type: ignore[index]
                    "meter_id": f"wall-meter-{index:02d}-v1",
                }
                for index, category in enumerate(categories, start=1)
            },
            "injected_dt_pair_count": ledger["injected_dt_pair_count"],  # type: ignore[index]
            "injected_dt_pair_count_std": 1.0e12,
            "pair_count_unit": "pairs",
            "measurement_duration_s": ledger["measurement_duration_s"],  # type: ignore[index]
            "measurement_duration_std_s": 1.0e-6,
            "duration_unit": "s",
        },
        "calibrations": {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "payload_role": "calibrations",
            "calibration_model": "traceable_absolute_energy_count_and_time_v1",
            "energy_meter_calibration_ids": {
                category: f"wall-energy-cal-{index:02d}-v1"
                for index, category in enumerate(categories, start=1)
            },
            "pair_counter_calibration_id": "wall-pair-counter-cal-v1",
            "clock_calibration_id": "wall-clock-cal-v1",
        },
        "uncertainty_pipeline": {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "payload_role": "uncertainty_pipeline",
            "pipeline_id": "wall-uncertainty-pipeline-v1",
            "uncertainty_propagation": "first_order_gradient_covariance_v1",
            "uncertainty_variables": [
                *categories,
                "injected_dt_pair_count",
                "measurement_duration_s",
            ],
            "correlation_matrix": np.eye(13).tolist(),
            "uncertainty_level": "1sigma",
        },
    }


def _write_verified_artifact(
    root: Path,
    *,
    evidence_class: str,
    stem: str,
    raw_payload: dict[str, object] | None = None,
    raw_data_bytes: bytes | None = None,
    claims_override: dict[str, object] | None = None,
):
    artifact_directory = root / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    claims_relative_path = f"artifacts/{stem}.claims.json"
    data_relative_path = f"artifacts/{stem}.data.txt"
    manifest_relative_path = f"artifacts/{stem}.manifest.json"
    if raw_payload is not None and raw_data_bytes is not None:
        raise ValueError("provide raw_payload or raw_data_bytes, not both")
    if raw_payload is None and raw_data_bytes is None:
        raw_payload = _reaction_raw_payload()
    primary_role = ENGINEERING_PRIMARY_PAYLOAD_ROLES.get(evidence_class, "operator_grid")
    auxiliary_payloads: dict[str, dict[str, object]] = {}
    typed_payload_bytes: dict[str, bytes] = {}
    typed_payload_hashes: dict[str, str] = {}
    if raw_payload is not None and evidence_class in ENGINEERING_PRIMARY_PAYLOAD_ROLES:
        auxiliary_payloads = _artifact_auxiliary_payloads(raw_payload)
        for role, document in auxiliary_payloads.items():
            document_bytes = json.dumps(
                document,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            typed_payload_bytes[role] = document_bytes
            typed_payload_hashes[role] = _sha256(document_bytes)
        raw_payload["auxiliary_payload_sha256s"] = dict(typed_payload_hashes)
    data_bytes = raw_data_bytes
    if data_bytes is None and raw_payload is not None:
        data_bytes = json.dumps(
            raw_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    if data_bytes is None:  # pragma: no cover - input invariant
        raise RuntimeError("typed payload bytes are missing")
    typed_payload_bytes[primary_role] = data_bytes
    typed_payload_hashes[primary_role] = _sha256(data_bytes)
    if raw_payload is not None:
        derived_claims = derive_artifact_claims(
            evidence_class,
            raw_payload,
            manifest_sha256="sha256:" + "0" * 64,
            auxiliary_payloads=auxiliary_payloads,
            typed_payload_sha256s=typed_payload_hashes,
        ).claims
    else:
        derived_claims = {}
    sidecar_claims = derived_claims if claims_override is None else claims_override
    claims_bytes = json.dumps(
        {
            "schema_version": EVIDENCE_ARTIFACT_CLAIMS_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "claims": sidecar_claims,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    (root / claims_relative_path).write_bytes(claims_bytes)
    typed_manifest_entries: list[dict[str, str]] = []
    ordered_typed_roles = (primary_role, *sorted(set(typed_payload_bytes) - {primary_role}))
    for role in ordered_typed_roles:
        role_relative_path = (
            data_relative_path if role == primary_role else f"artifacts/{stem}.{role}.data.txt"
        )
        (root / role_relative_path).write_bytes(typed_payload_bytes[role])
        typed_manifest_entries.append(
            {
                "role": role,
                "repository_relative_path": role_relative_path,
                "sha256": typed_payload_hashes[role],
            }
        )
    manifest_bytes = json.dumps(
        {
            "schema_version": EVIDENCE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "claims_payload_repository_relative_path": claims_relative_path,
            "payloads": [
                {
                    "role": "claims",
                    "repository_relative_path": claims_relative_path,
                    "sha256": _sha256(claims_bytes),
                },
                *typed_manifest_entries,
            ],
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    (root / manifest_relative_path).write_bytes(manifest_bytes)
    manifest_digest = _sha256(manifest_bytes)
    artifact = audit_repository_evidence_artifact(
        evidence_class=evidence_class,
        manifest_repository_relative_path=manifest_relative_path,
        expected_manifest_sha256=manifest_digest,
        repository_root=root,
    )
    return artifact, manifest_digest


@pytest.fixture(scope="module")
def report():
    return current_fusion_polarized_evidence_report()


def test_hupin_figure_digitization_is_traceable(report) -> None:
    audit = report.reaction

    assert audit.source_pmcid == "PMC6341121"
    assert audit.source_license == "CC BY 4.0"
    assert audit.expected_source_image_sha256 == HUPIN_FIGURE_4_IMAGE_SHA256
    assert (
        audit.expected_source_image_width_px,
        audit.expected_source_image_height_px,
    ) == HUPIN_FIGURE_4_IMAGE_SIZE_PX
    assert audit.source_image_repository_relative_path.endswith("41467_2018_8052_Fig4_HTML.jpg")
    assert audit.source_image_bundled_in_repository
    assert audit.source_image_runtime_sha256 == HUPIN_FIGURE_4_IMAGE_SHA256
    assert audit.source_image_sha256_verified
    assert (
        audit.source_image_runtime_width_px,
        audit.source_image_runtime_height_px,
    ) == HUPIN_FIGURE_4_IMAGE_SIZE_PX
    assert audit.source_image_dimensions_verified
    assert len(audit.digitized_deuteron_lab_energies_mev) == 33
    assert len(audit.digitized_full_alignment_enhancement_factors) == 33
    assert audit.digitized_deuteron_lab_energies_mev[0] == 0.0
    assert audit.digitized_deuteron_lab_energies_mev[-1] == pytest.approx(1.6)
    assert audit.digitized_full_alignment_enhancement_factors[0] == 1.5
    assert max(audit.digitized_full_alignment_enhancement_factors) == pytest.approx(
        1.5144444444444445
    )


def test_missing_or_unverified_source_image_closes_the_figure_control(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_loop,
        "HUPIN_FIGURE_4_REPOSITORY_RELATIVE_PATH",
        ".research-tmp/missing-hupin-figure.jpg",
    )

    audit = audit_published_polarized_dt_reactivity()
    assert not audit.source_image_bundled_in_repository
    assert audit.source_image_runtime_sha256 is None
    assert not audit.source_image_sha256_verified
    assert audit.source_image_runtime_width_px is None
    assert audit.source_image_runtime_height_px is None
    assert not audit.source_image_dimensions_verified
    assert not audit.figure_digitized_one_percent_control_pass


def test_oversized_hupin_source_image_is_rejected_before_open(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path.joinpath(
        *Path(evidence_loop.HUPIN_FIGURE_4_REPOSITORY_RELATIVE_PATH).parts
    )
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"x" * (evidence_loop.MAX_HUPIN_FIGURE_BYTES + 1))
    monkeypatch.setattr(evidence_loop, "_current_repository_root", lambda: tmp_path)
    original_open = Path.open

    def reject_image_open(path: Path, *args, **kwargs):
        if path == image_path:
            raise AssertionError("oversized Hupin image must be rejected before open")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_image_open)

    assert evidence_loop._bundled_hupin_figure_sha256() == (
        True,
        None,
        False,
        None,
        None,
        False,
    )


def test_repository_artifact_gate_hashes_claims_and_payload_bytes(tmp_path: Path) -> None:
    raw_payload = _reaction_raw_payload()
    claims = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        raw_payload,
        manifest_sha256="sha256:" + "0" * 64,
    ).claims
    artifact, manifest_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="valid",
        raw_payload=raw_payload,
    )

    assert artifact.expected_manifest_sha256 == manifest_digest
    assert artifact.runtime_manifest_sha256 == manifest_digest
    assert artifact.manifest_sha256_verified
    assert artifact.manifest_claim_keys == tuple(sorted(claims))
    assert artifact.manifest_claims_sha256 == evidence_loop._canonical_json_sha256(claims)
    assert artifact.typed_payload_derived
    assert artifact.derived_claim_keys == artifact.manifest_claim_keys
    assert artifact.derived_claims_sha256 == artifact.manifest_claims_sha256
    assert artifact.derived_claims_canonical_json is not None
    assert artifact.derived_plant_configuration_id is None
    assert artifact.derived_linked_manifest_sha256s == ()
    assert artifact.payload_roles == ("claims", "operator_grid")
    assert artifact.payload_hashes_verified
    assert artifact.runtime_artifact_gate_pass
    assert artifact.status == "VERIFIED"
    assert repository_evidence_artifact_integrity_pass(
        artifact,
        repository_root=tmp_path,
    )
    assert not physical_canonical_repository_evidence_artifact_gate_pass(artifact)

    data_path = tmp_path / "artifacts/valid.data.txt"
    data_path.write_bytes(data_path.read_bytes() + b"tamper")
    assert not repository_evidence_artifact_integrity_pass(
        artifact,
        repository_root=tmp_path,
    )


def test_opaque_raw_payload_and_claims_only_sidecar_cannot_verify(tmp_path: Path) -> None:
    raw_payload = _reaction_raw_payload()
    derived_claims = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        raw_payload,
        manifest_sha256="sha256:" + "0" * 64,
    ).claims
    artifact, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="opaque",
        raw_data_bytes=b"opaque operator bytes\n",
        claims_override=derived_claims,
    )

    assert artifact.status == "TYPED_PAYLOAD_JSON_INVALID_OR_DUPLICATE_KEY"
    assert not artifact.typed_payload_derived
    assert not artifact.runtime_artifact_gate_pass


def test_stale_or_injected_claims_sidecar_must_equal_full_derived_claims(tmp_path: Path) -> None:
    raw_payload = _reaction_raw_payload()
    stale_claims = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        raw_payload,
        manifest_sha256="sha256:" + "0" * 64,
    ).claims
    stale_claims["validated_target_state_maxwellian_reactivity_ratio"] = 99.0
    artifact, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="stale",
        raw_payload=raw_payload,
        claims_override=stale_claims,
    )

    assert artifact.typed_payload_derived
    assert artifact.status == "CLAIMS_PAYLOAD_NOT_DERIVED"
    assert artifact.manifest_claims_sha256 != artifact.derived_claims_sha256
    assert not artifact.runtime_artifact_gate_pass


@pytest.mark.parametrize(
    "raw_data_bytes",
    (
        b'{"schema_version":"fusion-polarized-artifact-raw-v1",'
        b'"schema_version":"fusion-polarized-artifact-raw-v1"}',
        b'{"value":NaN}',
    ),
)
def test_typed_raw_json_duplicate_keys_and_nonfinite_values_fail_closed(
    tmp_path: Path,
    raw_data_bytes: bytes,
) -> None:
    artifact, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem=_sha256(raw_data_bytes)[7:15],
        raw_data_bytes=raw_data_bytes,
        claims_override={"placeholder": 1.0},
    )
    assert artifact.status == "TYPED_PAYLOAD_JSON_INVALID_OR_DUPLICATE_KEY"
    assert not artifact.runtime_artifact_gate_pass


def test_deeply_nested_json_and_canonical_hashing_fail_closed(tmp_path: Path) -> None:
    deep_json = b"[" * 200_000 + b"0" + b"]" * 200_000
    artifact, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="deep-typed",
        raw_data_bytes=deep_json,
        claims_override={"placeholder": 1.0},
    )
    assert artifact.status == "TYPED_PAYLOAD_JSON_INVALID_OR_DUPLICATE_KEY"
    assert not artifact.runtime_artifact_gate_pass

    nested: object = 0
    for _ in range(200_000):
        nested = [nested]
    assert evidence_loop._canonical_json_sha256(nested) is None


def test_typed_raw_payload_has_a_hard_size_ceiling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(evidence_loop, "MAX_EVIDENCE_TYPED_PAYLOAD_BYTES", 32)
    artifact, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="oversized",
        raw_data_bytes=b"{" + b" " * 64 + b"}",
        claims_override={"placeholder": 1.0},
    )
    assert artifact.status == "TYPED_PAYLOAD_TOO_LARGE"
    assert not artifact.runtime_artifact_gate_pass


@pytest.mark.parametrize(
    "manifest_path",
    (
        "/absolute.json",
        "C:/absolute.json",
        "C:relative.json",
        "//server/share.json",
        "../escape.json",
        "a/../escape.json",
        "a\\manifest.json",
        "a//manifest.json",
    ),
)
def test_repository_artifact_gate_rejects_ambiguous_or_escaping_paths(
    tmp_path: Path,
    manifest_path: str,
) -> None:
    audit = audit_repository_evidence_artifact(
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        manifest_repository_relative_path=manifest_path,
        expected_manifest_sha256="sha256:" + "0" * 64,
        repository_root=tmp_path,
    )

    assert not audit.runtime_artifact_gate_pass
    assert audit.status == "MANIFEST_PATH_INVALID_OR_MISSING"


def test_repository_artifact_gate_rejects_duplicate_keys_and_nonfinite_claims(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    duplicate_manifest = (
        b'{"schema_version":"fusion-evidence-artifact-manifest-v1",'
        b'"schema_version":"fusion-evidence-artifact-manifest-v1",'
        b'"evidence_class":"polarized_dt_state_resolved_operator_v1",'
        b'"claims_payload_repository_relative_path":"artifacts/claims.json",'
        b'"payloads":[]}'
    )
    duplicate_path = artifact_directory / "duplicate.json"
    duplicate_path.write_bytes(duplicate_manifest)
    duplicate_audit = audit_repository_evidence_artifact(
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        manifest_repository_relative_path="artifacts/duplicate.json",
        expected_manifest_sha256=_sha256(duplicate_manifest),
        repository_root=tmp_path,
    )
    assert duplicate_audit.status == "MANIFEST_JSON_INVALID_OR_DUPLICATE_KEY"
    assert not duplicate_audit.runtime_artifact_gate_pass

    valid_artifact, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="nonfinite",
        raw_payload=_reaction_raw_payload(),
    )
    claims_path = tmp_path / "artifacts/nonfinite.claims.json"
    claims_document = json.loads(claims_path.read_text())
    claims_document["claims"]["temperature_kev"] = math.nan
    claims_bytes = json.dumps(
        claims_document,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    claims_path.write_bytes(claims_bytes)
    manifest_path = tmp_path / "artifacts/nonfinite.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payloads"][0]["sha256"] = _sha256(claims_bytes)
    manifest_bytes = json.dumps(
        manifest,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    nonfinite_audit = audit_repository_evidence_artifact(
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        manifest_repository_relative_path="artifacts/nonfinite.manifest.json",
        expected_manifest_sha256=_sha256(manifest_bytes),
        repository_root=tmp_path,
    )
    assert valid_artifact.runtime_artifact_gate_pass
    assert nonfinite_audit.status == "CLAIMS_PAYLOAD_JSON_INVALID_OR_DUPLICATE_KEY"
    assert not nonfinite_audit.runtime_artifact_gate_pass


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    (
        ("payload_traversal", "PAYLOAD_PATH_INVALID_OR_MISSING"),
        ("duplicate_role", "DUPLICATE_PAYLOAD_ROLE"),
        ("duplicate_hash", "DUPLICATE_PAYLOAD_SHA256"),
    ),
)
def test_repository_artifact_gate_rejects_payload_aliases(
    tmp_path: Path,
    mutation: str,
    expected_status: str,
) -> None:
    _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="aliases",
        raw_payload=_reaction_raw_payload(),
    )
    manifest_path = tmp_path / "artifacts/aliases.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "payload_traversal":
        manifest["payloads"][1]["repository_relative_path"] = "../escape.txt"
    elif mutation == "duplicate_role":
        manifest["payloads"][1]["role"] = "claims"
    else:
        manifest["payloads"][1]["sha256"] = manifest["payloads"][0]["sha256"]
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)

    audit = audit_repository_evidence_artifact(
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        manifest_repository_relative_path="artifacts/aliases.manifest.json",
        expected_manifest_sha256=_sha256(manifest_bytes),
        repository_root=tmp_path,
    )
    assert audit.status == expected_status
    assert not audit.runtime_artifact_gate_pass


def test_repository_artifact_gate_rejects_symlink_payload_when_supported(
    tmp_path: Path,
) -> None:
    artifact, manifest_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="symlink",
        raw_payload=_reaction_raw_payload(),
    )
    data_path = tmp_path / "artifacts/symlink.data.txt"
    outside_path = tmp_path / "outside.data.txt"
    outside_path.write_bytes(data_path.read_bytes())
    data_path.unlink()
    try:
        os.symlink(outside_path, data_path)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are not available to this test process")

    refreshed = audit_repository_evidence_artifact(
        evidence_class=artifact.evidence_class,
        manifest_repository_relative_path=artifact.manifest_repository_relative_path,
        expected_manifest_sha256=manifest_digest,
        repository_root=tmp_path,
    )
    assert refreshed.status == "PAYLOAD_PATH_INVALID_OR_MISSING"
    assert not refreshed.runtime_artifact_gate_pass


def test_distinct_artifact_storage_rejects_copied_or_hardlinked_payloads(
    tmp_path: Path,
) -> None:
    shared_payload = _standalone_deuterium_raw_payload()
    deuterium, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        stem="same-d",
        raw_payload=shared_payload,
    )
    tritium, _ = _write_verified_artifact(
        tmp_path,
        evidence_class=REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        stem="same-t",
        raw_payload=shared_payload,
    )
    assert repository_evidence_artifact_integrity_pass(
        deuterium,
        repository_root=tmp_path,
    )
    assert repository_evidence_artifact_integrity_pass(
        tritium,
        repository_root=tmp_path,
    )
    assert not evidence_loop._distinct_artifact_storage_pass(
        deuterium,
        tritium,
        repository_root=tmp_path,
    )

    tritium_data = tmp_path / "artifacts/same-t.data.txt"
    tritium_data.unlink()
    try:
        os.link(tmp_path / "artifacts/same-d.data.txt", tritium_data)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are not available to this test process")
    assert not evidence_loop._distinct_artifact_storage_pass(
        deuterium,
        tritium,
        repository_root=tmp_path,
    )


def test_energy_dependent_curve_is_locally_maxwellian_averaged_at_ten_kev(report) -> None:
    audit = report.reaction

    assert audit.temperature_kev == 10.0
    assert audit.figure_digitized_energy_dependent_cross_section_integrated
    assert audit.full_alignment_maxwellian_reactivity_ratio == pytest.approx(
        1.5040404019090645,
        rel=2.0e-11,
    )
    assert audit.digitization_lower_maxwellian_reactivity_ratio == pytest.approx(
        1.5010404019090646,
        rel=2.0e-11,
    )
    assert audit.digitization_lower_maxwellian_reactivity_ratio > 1.01
    assert audit.grid_refinement_absolute_residual < 1.0e-8
    assert audit.figure_digitized_one_percent_control_pass
    assert audit.numerical_to_closed_fit_ratio == pytest.approx(
        1.0049704525353218,
        rel=2.0e-10,
    )


def test_ten_kev_kernel_coverage_is_explicit(report) -> None:
    audit = report.reaction

    assert audit.maxwellian_kernel_mode_energy_kev == pytest.approx(34.8381, rel=2.0e-5)
    assert audit.maxwellian_kernel_central_90_low_energy_kev == pytest.approx(
        18.5452779048,
        rel=2.0e-10,
    )
    assert audit.maxwellian_kernel_central_90_high_energy_kev == pytest.approx(
        67.8635973994,
        rel=2.0e-10,
    )
    assert audit.maxwellian_kernel_0p1_percent_energy_kev == pytest.approx(
        8.96532118143,
        rel=2.0e-10,
    )
    assert audit.maxwellian_kernel_99p9_percent_energy_kev == pytest.approx(
        105.855103092,
        rel=2.0e-10,
    )
    assert audit.maxwellian_kernel_fraction_below_100_kev == pytest.approx(
        0.99814961212,
        rel=2.0e-10,
    )


def test_figure_control_never_becomes_author_data_or_measurement(report) -> None:
    audit = report.reaction

    assert audit.authors_published_energy_dependent_maxwellian_calculation
    assert audit.published_approximate_partial_rate_ratio == pytest.approx(1.32)
    assert audit.validated_target_deuteron_vector_polarization is None
    assert audit.validated_target_deuteron_tensor_polarization is None
    assert audit.validated_target_triton_vector_polarization is None
    assert audit.validated_target_state_maxwellian_reactivity_ratio is None
    assert audit.validated_target_state_maxwellian_reactivity_ratio_std is None
    assert audit.validated_target_state_maxwellian_reactivity_ratio_lower_3sigma is None
    assert not audit.target_state_resolved_operator_available
    assert audit.target_state_operator_provenance_digest is None
    assert not audit.target_state_operator_artifact.runtime_artifact_gate_pass
    assert audit.target_state_operator_artifact.status == "MANIFEST_NOT_DECLARED"
    assert not audit.author_machine_readable_energy_grid_available
    assert not audit.author_s_matrix_or_angle_integrated_coefficients_available
    assert not audit.nuclear_model_systematic_covariance_available
    assert audit.dries_exfor_machine_readable_single_angle_azz_available
    assert audit.dries_exfor_point_count == 29
    assert audit.dries_initial_tritium_target_unpolarized
    assert audit.dries_record_is_not_double_polarized_czz_or_total_rate
    assert audit.arnold_exfor_unpolarized_total_dt_cross_section_available
    assert audit.arnold_exfor_point_count == 64
    assert audit.han_sciencedb_public_metadata_available
    assert audit.han_sciencedb_license == "CC BY-SA 4.0"
    assert audit.han_sciencedb_version == "V1"
    assert audit.han_sciencedb_metadata_declares_unpolarized_rmatrix_covariance
    assert audit.han_sciencedb_numeric_files_locally_verified
    assert audit.han_sciencedb_payload_integrity_gate_pass
    assert audit.han_sciencedb_payload_audit.payload_integrity_gate_pass
    assert not audit.han_sciencedb_numeric_covariance_matrix_available
    assert not audit.han_sciencedb_initial_state_spin_operator_available
    assert audit.han_sciencedb_unpolarized_reactivity_audit.dt_table_row_count == 54
    assert audit.han_sciencedb_unpolarized_reactivity_audit.sigma_interpolation_relative_spread > 0.02
    assert not audit.han_sciencedb_unpolarized_interpolation_spread_below_one_percent
    assert not audit.han_sciencedb_unpolarized_sub_one_percent_certification_gate_pass
    assert not audit.han_initial_double_polarized_state_operator_available
    assert not audit.spin_correlation_czz_directly_measured
    assert not audit.polarized_dt_rate_directly_validated
    assert not audit.local_author_data_reproduction_pass
    assert not audit.physical_reaction_evidence_gate_pass
    assert not physical_published_reaction_evidence_gate_pass(audit)


def test_primary_source_throughput_gap_is_computed_from_plant_ledger(report) -> None:
    audit = report.source

    assert audit.required_fusion_reaction_rate_s == pytest.approx(1.773155987e20)
    assert audit.required_per_species_fuel_rate_s == pytest.approx(1.773155987e22)
    assert audit.reference_plant_minimum_per_species_fuel_rate_s == pytest.approx(1.773155987e22)
    assert audit.reference_plant_scale_invariant_pass
    assert audit.validated_integrated_plant_configuration_id is None
    assert audit.coulter_measured_continuous_deuterium_rate_s == pytest.approx(2.1e17)
    assert audit.coulter_measured_deuterium_polarization == pytest.approx(0.73)
    assert audit.coulter_measured_deuterium_polarization_std == pytest.approx(0.03)
    assert audit.required_to_coulter_rate_ratio == pytest.approx(84435.9993839389)
    assert not audit.coulter_record_meets_reactor_deuterium_rate
    assert audit.anke_source_doi == "https://doi.org/10.1016/j.nima.2013.03.043"
    assert audit.anke_measured_deuterium_rate_s == pytest.approx(3.9e16)
    assert audit.anke_measured_deuterium_rate_std_s == pytest.approx(0.2e16)
    assert audit.anke_measured_deuteron_vector_polarization_abs == pytest.approx(0.90)
    assert audit.anke_measured_deuteron_tensor_polarization == pytest.approx(0.86)
    assert audit.anke_rf_dissociator_power_w == pytest.approx(300.0)
    assert audit.anke_rf_power_is_partial_component_not_complete_wall_plug
    assert not audit.anke_record_meets_reactor_deuterium_rate
    assert (
        audit.required_per_species_fuel_rate_s
        / (audit.anke_measured_deuterium_rate_s - 3.0 * audit.anke_measured_deuterium_rate_std_s)
        > 5.3e5
    )
    assert not audit.sofikitis_reactor_continuous_throughput_measured
    assert audit.utsuro_source_doi == "https://doi.org/10.1080/15361055.2022.2062098"
    assert audit.utsuro_hd_proxy_proof_of_concept_performed
    assert not audit.utsuro_actual_polarized_tritium_source_demonstrated
    assert not audit.utsuro_reactor_continuous_throughput_measured
    assert audit.cook_result_is_simulation_not_burn_measurement
    assert audit.cook_sparc_deuterium_mplus_retention_probability == pytest.approx(0.8905)
    assert audit.cook_sparc_tritium_depolarization_fraction == pytest.approx(0.13)
    assert audit.souers_actual_solid_dt_triton_relaxation_measured
    assert audit.souers_result_is_preburn_not_burn_weighted_plasma_retention
    assert audit.collins_actual_solid_dt_deuteron_nmr_relaxation_measured
    assert audit.collins_result_is_preburn_not_burn_weighted_plasma_retention
    assert not audit.reactor_rate_polarized_tritium_source_demonstrated
    assert audit.validated_reactor_rate_deuterium_source_rate_s is None
    assert audit.validated_reactor_rate_tritium_source_rate_s is None
    assert audit.validated_reactor_rate_deuterium_source_rate_linearized_std_s is None
    assert audit.validated_reactor_rate_tritium_source_rate_linearized_std_s is None
    assert audit.validated_reactor_rate_deuterium_source_rate_lower_3sigma_s is None
    assert audit.validated_reactor_rate_tritium_source_rate_lower_3sigma_s is None
    assert audit.validated_reactor_rate_deuterium_source_rate_3sigma_bound_model is None
    assert audit.validated_reactor_rate_tritium_source_rate_3sigma_bound_model is None
    assert audit.validated_reactor_rate_deuterium_measurement_duration_s is None
    assert audit.validated_reactor_rate_tritium_measurement_duration_s is None
    assert audit.validated_reactor_rate_deuterium_output_count is None
    assert audit.validated_reactor_rate_tritium_output_count is None
    assert all(
        value is None
        for value in (
            audit.validated_reactor_rate_deuterium_measurement_duration_std_s,
            audit.validated_reactor_rate_tritium_measurement_duration_std_s,
            audit.validated_reactor_rate_deuterium_output_count_std,
            audit.validated_reactor_rate_tritium_output_count_std,
            audit.validated_reactor_rate_deuterium_background_count,
            audit.validated_reactor_rate_tritium_background_count,
            audit.validated_reactor_rate_deuterium_background_count_std,
            audit.validated_reactor_rate_tritium_background_count_std,
            audit.validated_reactor_rate_deuterium_detection_efficiency,
            audit.validated_reactor_rate_tritium_detection_efficiency,
            audit.validated_reactor_rate_deuterium_detection_efficiency_std,
            audit.validated_reactor_rate_tritium_detection_efficiency_std,
            audit.validated_reactor_rate_deuterium_uncertainty_correlation_matrix,
            audit.validated_reactor_rate_tritium_uncertainty_correlation_matrix,
            audit.validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix,
        )
    )
    assert audit.validated_reactor_rate_deuteron_vector_polarization is None
    assert audit.validated_reactor_rate_deuteron_vector_polarization_std is None
    assert audit.validated_reactor_rate_deuteron_tensor_polarization is None
    assert audit.validated_reactor_rate_deuteron_tensor_polarization_std is None
    assert audit.validated_reactor_rate_triton_vector_polarization is None
    assert audit.validated_reactor_rate_triton_vector_polarization_std is None
    assert audit.validated_reactor_rate_deuterium_source_provenance_digest is None
    assert audit.validated_reactor_rate_tritium_source_provenance_digest is None
    assert not audit.validated_reactor_rate_deuterium_source_artifact.runtime_artifact_gate_pass
    assert not audit.validated_reactor_rate_tritium_source_artifact.runtime_artifact_gate_pass
    assert not audit.burn_weighted_dt_polarization_product_measured
    assert audit.validated_burn_weighted_dt_polarization_product is None
    assert audit.validated_burn_weighted_dt_polarization_product_std is None
    assert audit.validated_burn_weighted_deuteron_tensor_polarization is None
    assert audit.validated_burn_weighted_deuteron_tensor_polarization_std is None
    assert audit.validated_burn_retention_measurement_duration_s is None
    assert audit.validated_burn_retention_measurement_duration_std_s is None
    assert audit.validated_burn_retention_reaction_count is None
    assert audit.validated_burn_retention_uncertainty_correlation_matrix is None
    assert audit.validated_burn_retention_observable_semantics is None
    assert audit.validated_burn_retention_provenance_digest is None
    assert not audit.validated_burn_retention_artifact.runtime_artifact_gate_pass
    assert not audit.complete_wall_plug_energy_per_injected_pair_measured
    assert not audit.complete_wall_plug_uncertainty_flow_and_provenance_available
    assert audit.validated_wall_plug_energy_per_injected_dt_pair_ev is None
    assert audit.validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev is None
    assert audit.validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev is None
    assert audit.validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev is None
    assert audit.validated_wall_plug_total_energy_j is None
    assert audit.validated_wall_plug_total_energy_linearized_std_j is None
    assert audit.validated_wall_plug_total_energy_upper_3sigma_j is None
    assert audit.validated_wall_plug_total_energy_lower_3sigma_j is None
    assert audit.validated_wall_plug_measurement_duration_s is None
    assert audit.validated_wall_plug_measurement_injected_dt_pair_count is None
    assert all(
        value is None
        for value in (
            audit.validated_wall_plug_measurement_duration_std_s,
            audit.validated_wall_plug_measurement_injected_dt_pair_count_std,
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_s,
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s,
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s,
            audit.validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s,
            audit.validated_wall_plug_uncertainty_correlation_matrix,
        )
    )
    assert audit.validated_wall_plug_3sigma_bound_model is None
    assert not audit.energy_resolved_burn_operator_sufficient_statistics_available
    assert not (
        audit.burn_state_operator_componentwise_3sigma_stress_envelope_control_pass
    )
    assert audit.validated_wall_plug_measurement_provenance is None
    assert not audit.validated_wall_plug_artifact.runtime_artifact_gate_pass
    assert not physical_polarized_source_evidence_gate_pass(audit)


def test_subscale_source_audit_cannot_be_relabeled_as_reactor_rate() -> None:
    audit = audit_polarized_fuel_primary_evidence(reference_fusion_power_w=5.0e3)

    assert audit.coulter_measured_continuous_deuterium_rate_s >= (
        audit.required_per_species_fuel_rate_s
    )
    assert not audit.reference_plant_scale_invariant_pass
    assert not audit.coulter_record_meets_reactor_deuterium_rate
    assert not audit.reactor_rate_polarized_deuterium_source_demonstrated
    assert audit.status == "REFERENCE_PLANT_SCALE_BELOW_500MW_ONE_PERCENT_THROUGHPUT"
    assert not physical_polarized_source_evidence_gate_pass(audit)


def test_private_semantic_records_require_typed_artifacts_but_public_gates_stay_closed(
    report,
    tmp_path: Path,
) -> None:
    required_count = int(1.1 * report.source.required_per_species_fuel_rate_s) + 1
    forged_reaction = replace(
        report.reaction,
        validated_target_deuteron_vector_polarization=math.sqrt(0.02),
        validated_target_deuteron_tensor_polarization=0.0,
        validated_target_triton_vector_polarization=math.sqrt(0.02),
        validated_target_state_maxwellian_reactivity_ratio=1.019,
        validated_target_state_maxwellian_reactivity_ratio_std=0.002,
        validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=1.013,
        target_state_resolved_operator_available=True,
        target_state_operator_provenance_digest=OPERATOR_PROVENANCE_DIGEST,
        author_machine_readable_energy_grid_available=True,
        author_s_matrix_or_angle_integrated_coefficients_available=True,
        nuclear_model_systematic_covariance_available=True,
        spin_correlation_czz_directly_measured=True,
        polarized_dt_rate_directly_validated=True,
        local_author_data_reproduction_pass=True,
        physical_reaction_evidence_gate_pass=True,
    )
    forged_source = replace(
        report.source,
        validated_integrated_plant_configuration_id="synthetic-plant-v1",
        reactor_rate_polarized_deuterium_source_demonstrated=True,
        reactor_rate_polarized_tritium_source_demonstrated=True,
        validated_reactor_rate_deuterium_source_rate_s=(
            float(required_count)
        ),
        validated_reactor_rate_tritium_source_rate_s=(
            float(required_count)
        ),
        validated_reactor_rate_deuterium_source_rate_linearized_std_s=1.0,
        validated_reactor_rate_tritium_source_rate_linearized_std_s=1.0,
        validated_reactor_rate_deuterium_source_rate_lower_3sigma_s=(
            float(required_count) - 3.0
        ),
        validated_reactor_rate_tritium_source_rate_lower_3sigma_s=(
            float(required_count) - 3.0
        ),
        validated_reactor_rate_deuterium_source_rate_3sigma_bound_model=(
            "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
        ),
        validated_reactor_rate_tritium_source_rate_3sigma_bound_model=(
            "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
        ),
        validated_reactor_rate_deuterium_measurement_duration_s=1.0,
        validated_reactor_rate_tritium_measurement_duration_s=1.0,
        validated_reactor_rate_deuterium_measurement_duration_std_s=1.0e-6,
        validated_reactor_rate_tritium_measurement_duration_std_s=1.0e-6,
        validated_reactor_rate_deuterium_output_count=required_count,
        validated_reactor_rate_tritium_output_count=required_count,
        validated_reactor_rate_deuterium_output_count_std=1.0e12,
        validated_reactor_rate_tritium_output_count_std=1.0e12,
        validated_reactor_rate_deuterium_background_count=3_000_000_000,
        validated_reactor_rate_tritium_background_count=3_000_000_000,
        validated_reactor_rate_deuterium_background_count_std=1.0e9,
        validated_reactor_rate_tritium_background_count_std=1.0e9,
        validated_reactor_rate_deuterium_detection_efficiency=1.0,
        validated_reactor_rate_tritium_detection_efficiency=1.0,
        validated_reactor_rate_deuterium_detection_efficiency_std=1.0e-6,
        validated_reactor_rate_tritium_detection_efficiency_std=1.0e-6,
        validated_reactor_rate_deuterium_uncertainty_correlation_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        validated_reactor_rate_tritium_uncertainty_correlation_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix=(
            (1.0, 0.2),
            (0.2, 1.0),
        ),
        validated_reactor_rate_deuteron_vector_polarization=0.5,
        validated_reactor_rate_deuteron_vector_polarization_std=0.001,
        validated_reactor_rate_deuteron_tensor_polarization=0.0,
        validated_reactor_rate_deuteron_tensor_polarization_std=0.001,
        validated_reactor_rate_triton_vector_polarization=0.1,
        validated_reactor_rate_triton_vector_polarization_std=0.001,
        validated_reactor_rate_deuterium_source_provenance_digest=(SOURCE_D_PROVENANCE_DIGEST),
        validated_reactor_rate_tritium_source_provenance_digest=(SOURCE_T_PROVENANCE_DIGEST),
        burn_weighted_dt_polarization_product_measured=True,
        validated_burn_weighted_dt_polarization_product=0.03,
        validated_burn_weighted_dt_polarization_product_std=0.001,
        validated_burn_weighted_deuteron_tensor_polarization=0.0,
        validated_burn_weighted_deuteron_tensor_polarization_std=0.01,
        validated_burn_retention_measurement_duration_s=2.0,
        validated_burn_retention_measurement_duration_std_s=0.01,
        validated_burn_retention_reaction_count=1_000_000,
        validated_burn_retention_uncertainty_correlation_matrix=(
            (1.0, 0.25),
            (0.25, 1.0),
        ),
        validated_burn_retention_observable_semantics=(
            "burn_weighted_dt_vector_product_and_deuteron_tensor_joint_estimator_v1"
        ),
        validated_burn_retention_provenance_digest=BURN_PROVENANCE_DIGEST,
        complete_wall_plug_energy_per_injected_pair_measured=True,
        complete_wall_plug_uncertainty_flow_and_provenance_available=True,
        validated_wall_plug_energy_per_injected_dt_pair_ev=1.0,
        validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev=0.1,
        validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev=0.7,
        validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev=1.3,
        validated_wall_plug_total_energy_j=(
            required_count * evidence_loop.EV_TO_JOULE
        ),
        validated_wall_plug_total_energy_linearized_std_j=(
            0.1 * required_count * evidence_loop.EV_TO_JOULE
        ),
        validated_wall_plug_total_energy_lower_3sigma_j=(
            0.7 * required_count * evidence_loop.EV_TO_JOULE
        ),
        validated_wall_plug_total_energy_upper_3sigma_j=(
            1.3 * required_count * evidence_loop.EV_TO_JOULE
        ),
        validated_wall_plug_measurement_duration_s=1.0,
        validated_wall_plug_measurement_duration_std_s=1.0e-6,
        validated_wall_plug_measurement_injected_dt_pair_count=required_count,
        validated_wall_plug_measurement_injected_dt_pair_count_std=1.0e12,
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_s=float(required_count),
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s=1.0,
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s=(
            float(required_count) - 3.0
        ),
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s=(
            float(required_count) + 3.0
        ),
        validated_wall_plug_3sigma_bound_model=(
            "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
        ),
        validated_wall_plug_uncertainty_correlation_matrix=tuple(
            tuple(1.0 if row == column else 0.0 for column in range(13))
            for row in range(13)
        ),
        validated_wall_plug_measurement_provenance=WALL_PLUG_PROVENANCE_DIGEST,
        source_retention_and_wall_plug_gate_pass=True,
    )

    assert not evidence_loop._published_reaction_evidence_record_integrity_pass(
        forged_reaction
    )
    assert not evidence_loop._polarized_source_evidence_record_integrity_pass(forged_source)

    operator_artifact, operator_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="reaction",
        raw_payload=_reaction_raw_payload(forged_reaction),
    )
    operator_claims = json.loads(operator_artifact.derived_claims_canonical_json)
    forged_reaction = replace(
        forged_reaction,
        validated_target_state_maxwellian_reactivity_ratio=(
            operator_claims["validated_target_state_maxwellian_reactivity_ratio"]
        ),
        validated_target_state_maxwellian_reactivity_ratio_std=(
            operator_claims["validated_target_state_maxwellian_reactivity_ratio_std"]
        ),
        validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=(
            operator_claims[
                "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"
            ]
        ),
    )
    forged_source = replace(
        forged_source,
        validated_burn_reaction_operator_provenance_digest=operator_digest,
        validated_burn_operator_uncertainty_relationship=(
            "unknown_cross_artifact_correlation_componentwise_3sigma_robust_box_v1"
        ),
    )
    deuterium_artifact, deuterium_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        stem="deuterium",
        raw_payload=_source_raw_payload(forged_source, "D"),
    )
    tritium_artifact, tritium_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        stem="tritium",
        raw_payload=_source_raw_payload(forged_source, "T"),
    )
    deuterium_claims = json.loads(deuterium_artifact.derived_claims_canonical_json)
    tritium_claims = json.loads(tritium_artifact.derived_claims_canonical_json)
    forged_source = replace(
        forged_source,
        validated_reactor_rate_deuterium_source_rate_s=deuterium_claims["source_rate_s"],
        validated_reactor_rate_tritium_source_rate_s=tritium_claims["source_rate_s"],
        validated_reactor_rate_deuterium_source_rate_linearized_std_s=(
            deuterium_claims["source_rate_linearized_std_s"]
        ),
        validated_reactor_rate_tritium_source_rate_linearized_std_s=(
            tritium_claims["source_rate_linearized_std_s"]
        ),
        validated_reactor_rate_deuterium_source_rate_lower_3sigma_s=(
            deuterium_claims["source_rate_lower_3sigma_s"]
        ),
        validated_reactor_rate_tritium_source_rate_lower_3sigma_s=(
            tritium_claims["source_rate_lower_3sigma_s"]
        ),
        validated_reactor_rate_deuterium_source_rate_3sigma_bound_model=(
            deuterium_claims["source_rate_3sigma_bound_model"]
        ),
        validated_reactor_rate_tritium_source_rate_3sigma_bound_model=(
            tritium_claims["source_rate_3sigma_bound_model"]
        ),
        validated_reactor_rate_deuterium_measurement_duration_std_s=(
            deuterium_claims["measurement_duration_std_s"]
        ),
        validated_reactor_rate_tritium_measurement_duration_std_s=(
            tritium_claims["measurement_duration_std_s"]
        ),
        validated_reactor_rate_deuterium_background_count=(
            deuterium_claims["background_count"]
        ),
        validated_reactor_rate_tritium_background_count=(
            tritium_claims["background_count"]
        ),
        validated_reactor_rate_deuterium_background_count_std=(
            deuterium_claims["background_count_std"]
        ),
        validated_reactor_rate_tritium_background_count_std=(
            tritium_claims["background_count_std"]
        ),
        validated_reactor_rate_deuterium_detection_efficiency=(
            deuterium_claims["detection_efficiency"]
        ),
        validated_reactor_rate_tritium_detection_efficiency=(
            tritium_claims["detection_efficiency"]
        ),
        validated_reactor_rate_deuterium_detection_efficiency_std=(
            deuterium_claims["detection_efficiency_std"]
        ),
        validated_reactor_rate_tritium_detection_efficiency_std=(
            tritium_claims["detection_efficiency_std"]
        ),
        validated_reactor_rate_deuterium_uncertainty_correlation_matrix=tuple(
            tuple(row)
            for row in deuterium_claims[
                "source_rate_uncertainty_correlation_matrix"
            ]
        ),
        validated_reactor_rate_tritium_uncertainty_correlation_matrix=tuple(
            tuple(row)
            for row in tritium_claims[
                "source_rate_uncertainty_correlation_matrix"
            ]
        ),
        validated_reactor_rate_deuteron_polarization_uncertainty_correlation_matrix=tuple(
            tuple(row)
            for row in deuterium_claims[
                "deuteron_polarization_uncertainty_correlation_matrix"
            ]
        ),
    )
    burn_artifact, burn_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        stem="burn",
        raw_payload=_burn_raw_payload(
            forged_source,
            deuterium_hash=deuterium_digest,
            tritium_hash=tritium_digest,
            reaction_operator_hash=operator_digest,
        ),
    )
    burn_claims = json.loads(burn_artifact.derived_claims_canonical_json)
    forged_source = replace(
        forged_source,
        validated_burn_retention_measurement_duration_std_s=(
            burn_claims["measurement_duration_std_s"]
        ),
        validated_burn_retention_uncertainty_correlation_matrix=tuple(
            tuple(row) for row in burn_claims["burn_uncertainty_correlation_matrix"]
        ),
        validated_burn_retention_observable_semantics=(
            burn_claims["burn_observable_semantics"]
        ),
    )
    wall_artifact, wall_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        stem="wall",
        raw_payload=_wall_raw_payload(
            forged_source,
            deuterium_hash=deuterium_digest,
            tritium_hash=tritium_digest,
        ),
    )
    wall_claims = json.loads(wall_artifact.derived_claims_canonical_json)
    forged_source = replace(
        forged_source,
        validated_wall_plug_energy_per_injected_dt_pair_ev=(
            wall_claims["wall_plug_energy_per_injected_dt_pair_ev"]
        ),
        validated_wall_plug_energy_linearized_std_per_injected_dt_pair_ev=(
            wall_claims["wall_plug_energy_linearized_std_per_injected_dt_pair_ev"]
        ),
        validated_wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev=(
            wall_claims[
                "wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev"
            ]
        ),
        validated_wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev=(
            wall_claims[
                "wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev"
            ]
        ),
        validated_wall_plug_total_energy_j=wall_claims["wall_plug_total_energy_j"],
        validated_wall_plug_total_energy_linearized_std_j=(
            wall_claims["wall_plug_total_energy_linearized_std_j"]
        ),
        validated_wall_plug_total_energy_lower_3sigma_j=(
            wall_claims["wall_plug_total_energy_lower_3sigma_j"]
        ),
        validated_wall_plug_total_energy_upper_3sigma_j=(
            wall_claims["wall_plug_total_energy_upper_3sigma_j"]
        ),
        validated_wall_plug_measurement_duration_std_s=(
            wall_claims["measurement_duration_std_s"]
        ),
        validated_wall_plug_measurement_injected_dt_pair_count_std=(
            wall_claims["measurement_injected_dt_pair_count_std"]
        ),
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_s=(
            wall_claims["measurement_injected_dt_pair_flow_rate_s"]
        ),
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_linearized_std_s=(
            wall_claims[
                "measurement_injected_dt_pair_flow_rate_linearized_std_s"
            ]
        ),
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_lower_3sigma_s=(
            wall_claims[
                "measurement_injected_dt_pair_flow_rate_lower_3sigma_s"
            ]
        ),
        validated_wall_plug_measurement_injected_dt_pair_flow_rate_upper_3sigma_s=(
            wall_claims[
                "measurement_injected_dt_pair_flow_rate_upper_3sigma_s"
            ]
        ),
        validated_wall_plug_3sigma_bound_model=(
            wall_claims["wall_plug_3sigma_bound_model"]
        ),
        validated_wall_plug_uncertainty_correlation_matrix=tuple(
            tuple(row)
            for row in wall_claims["wall_plug_uncertainty_correlation_matrix"]
        ),
    )
    forged_reaction = replace(
        forged_reaction,
        target_state_operator_provenance_digest=operator_digest,
        target_state_operator_artifact=operator_artifact,
    )
    forged_source = replace(
        forged_source,
        validated_reactor_rate_deuterium_source_provenance_digest=deuterium_digest,
        validated_reactor_rate_tritium_source_provenance_digest=tritium_digest,
        validated_reactor_rate_deuterium_source_artifact=deuterium_artifact,
        validated_reactor_rate_tritium_source_artifact=tritium_artifact,
        validated_burn_retention_provenance_digest=burn_digest,
        validated_burn_retention_artifact=burn_artifact,
        validated_wall_plug_measurement_provenance=wall_digest,
        validated_wall_plug_artifact=wall_artifact,
    )
    contraction = evidence_loop._derive_burn_state_operator_contraction(
        forged_reaction,
        forged_source,
        repository_root=tmp_path,
    )
    assert contraction is not None
    assert contraction.control_passes_one_percent
    forged_source = replace(
        forged_source,
        validated_burn_state_operator_reactivity_ratio=(
            contraction.central_reactivity_ratio
        ),
        validated_burn_state_operator_reactivity_ratio_robust_margin_per_sigma_equivalent=(
            contraction.robust_margin_per_sigma_equivalent
        ),
        validated_burn_state_operator_reactivity_ratio_robust_lower_3sigma=(
            contraction.robust_lower_3sigma
        ),
        validated_burn_state_operator_max_phase_relative_residual=(
            contraction.max_phase_relative_residual
        ),
        validated_burn_state_operator_contraction_semantics=contraction.semantics,
        burn_state_operator_componentwise_3sigma_stress_envelope_control_pass=True,
    )

    assert operator_artifact.runtime_artifact_gate_pass
    assert set(deuterium_artifact.payload_roles) == {
        "claims",
        "measurement",
        "event_counts",
        "calibration",
        "pipeline",
    }
    assert dict(deuterium_artifact.derived_typed_payload_sha256s) == {
        role: digest
        for role, digest in zip(
            deuterium_artifact.payload_roles,
            deuterium_artifact.expected_payload_sha256_digests,
        )
        if role != "claims"
    }
    assert json.loads(operator_artifact.derived_claims_canonical_json)[
        "target_state_resolved_operator_available"
    ]
    assert evidence_loop._published_reaction_evidence_record_integrity_pass(
        forged_reaction,
        repository_root=tmp_path,
    )
    assert not evidence_loop._published_reaction_evidence_record_integrity_pass(
        replace(forged_reaction, target_reactivity_ratio=1.0001),
        repository_root=tmp_path,
    )
    assert evidence_loop._polarized_source_evidence_record_integrity_pass(
        forged_source,
        repository_root=tmp_path,
    )
    assert evidence_loop._burn_state_operator_evidence_record_integrity_pass(
        forged_reaction,
        forged_source,
        repository_root=tmp_path,
    )

    target_only_reaction = replace(
        forged_reaction,
        validated_target_deuteron_vector_polarization=0.5,
        validated_target_deuteron_tensor_polarization=0.0,
        validated_target_triton_vector_polarization=1.0,
    )
    target_only_operator, target_only_operator_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="target-only-reaction",
        raw_payload=_reaction_raw_payload(target_only_reaction),
    )
    target_only_claims = json.loads(
        target_only_operator.derived_claims_canonical_json
    )
    target_only_reaction = replace(
        target_only_reaction,
        validated_target_state_maxwellian_reactivity_ratio=target_only_claims[
            "validated_target_state_maxwellian_reactivity_ratio"
        ],
        validated_target_state_maxwellian_reactivity_ratio_std=target_only_claims[
            "validated_target_state_maxwellian_reactivity_ratio_std"
        ],
        validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=(
            target_only_claims[
                "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"
            ]
        ),
        target_state_operator_provenance_digest=target_only_operator_digest,
        target_state_operator_artifact=target_only_operator,
    )
    target_only_source = replace(
        forged_source,
        validated_burn_reaction_operator_provenance_digest=(
            target_only_operator_digest
        ),
    )
    target_only_burn, target_only_burn_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        stem="target-only-burn",
        raw_payload=_burn_raw_payload(
            target_only_source,
            deuterium_hash=deuterium_digest,
            tritium_hash=tritium_digest,
            reaction_operator_hash=target_only_operator_digest,
        ),
    )
    target_only_source = replace(
        target_only_source,
        validated_burn_retention_provenance_digest=target_only_burn_digest,
        validated_burn_retention_artifact=target_only_burn,
    )
    target_only_contraction = evidence_loop._derive_burn_state_operator_contraction(
        target_only_reaction,
        target_only_source,
        repository_root=tmp_path,
    )
    assert target_only_claims[
        "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"
    ] >= 1.01
    assert target_only_contraction is not None
    assert target_only_contraction.central_reactivity_ratio < 1.01
    assert not target_only_contraction.control_passes_one_percent

    phase_alias_reaction = replace(
        forged_reaction,
        validated_target_deuteron_vector_polarization=0.2,
        validated_target_deuteron_tensor_polarization=0.2,
        validated_target_triton_vector_polarization=0.2,
    )
    phase_alias_payload = _reaction_raw_payload(phase_alias_reaction)
    phase_alias_channels = phase_alias_payload["channel_reactivity_ratio_grids"]
    for energy_index in range(101):
        delta_beta = 0.04 if energy_index % 2 == 0 else -0.04
        delta_tensor = -0.2 * delta_beta
        increments = (
            delta_tensor + delta_beta,
            delta_tensor - delta_beta,
            -2.0 * delta_tensor,
            -2.0 * delta_tensor,
            delta_tensor - delta_beta,
            delta_tensor + delta_beta,
        )
        for channel, increment in enumerate(increments):
            phase_alias_channels[channel][energy_index] += increment
    phase_alias_weights = phase_alias_payload["target_channel_weights"]
    phase_alias_payload["target_state_reactivity_ratio_grid"] = [
        math.fsum(
            phase_alias_weights[channel] * phase_alias_channels[channel][energy_index]
            for channel in range(6)
        )
        for energy_index in range(101)
    ]
    phase_alias_operator, phase_alias_operator_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="phase-alias-reaction",
        raw_payload=phase_alias_payload,
    )
    phase_alias_claims = json.loads(phase_alias_operator.derived_claims_canonical_json)
    phase_alias_reaction = replace(
        phase_alias_reaction,
        validated_target_state_maxwellian_reactivity_ratio=phase_alias_claims[
            "validated_target_state_maxwellian_reactivity_ratio"
        ],
        validated_target_state_maxwellian_reactivity_ratio_std=phase_alias_claims[
            "validated_target_state_maxwellian_reactivity_ratio_std"
        ],
        validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=(
            phase_alias_claims[
                "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"
            ]
        ),
        target_state_operator_provenance_digest=phase_alias_operator_digest,
        target_state_operator_artifact=phase_alias_operator,
    )
    phase_alias_source = replace(
        forged_source,
        validated_burn_reaction_operator_provenance_digest=(
            phase_alias_operator_digest
        ),
    )
    phase_alias_burn, phase_alias_burn_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        stem="phase-alias-burn",
        raw_payload=_burn_raw_payload(
            phase_alias_source,
            deuterium_hash=deuterium_digest,
            tritium_hash=tritium_digest,
            reaction_operator_hash=phase_alias_operator_digest,
        ),
    )
    phase_alias_source = replace(
        phase_alias_source,
        validated_burn_retention_provenance_digest=phase_alias_burn_digest,
        validated_burn_retention_artifact=phase_alias_burn,
    )
    assert evidence_loop._published_reaction_evidence_record_integrity_pass(
        phase_alias_reaction,
        repository_root=tmp_path,
    )
    assert evidence_loop._polarized_source_evidence_record_integrity_pass(
        phase_alias_source,
        repository_root=tmp_path,
    )
    assert (
        evidence_loop._derive_burn_state_operator_contraction(
            phase_alias_reaction,
            phase_alias_source,
            repository_root=tmp_path,
        )
        is None
    )

    wrong_link_burn, wrong_link_burn_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        stem="wrong-link-burn",
        raw_payload=_burn_raw_payload(
            forged_source,
            deuterium_hash=deuterium_digest,
            tritium_hash="sha256:" + "f" * 64,
            reaction_operator_hash=operator_digest,
        ),
    )
    assert wrong_link_burn.runtime_artifact_gate_pass
    assert not evidence_loop._polarized_source_evidence_record_integrity_pass(
        replace(
            forged_source,
            validated_burn_retention_artifact=wrong_link_burn,
            validated_burn_retention_provenance_digest=wrong_link_burn_digest,
        ),
        repository_root=tmp_path,
    )

    wrong_config_wall, wrong_config_wall_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        stem="wrong-config-wall",
        raw_payload=_wall_raw_payload(
            forged_source,
            deuterium_hash=deuterium_digest,
            tritium_hash=tritium_digest,
            configuration_id="different-plant-v1",
        ),
    )
    assert wrong_config_wall.runtime_artifact_gate_pass
    assert not evidence_loop._polarized_source_evidence_record_integrity_pass(
        replace(
            forged_source,
            validated_wall_plug_artifact=wrong_config_wall,
            validated_wall_plug_measurement_provenance=wrong_config_wall_digest,
        ),
        repository_root=tmp_path,
    )

    lowered_required_rate = 0.5 * forged_source.required_per_species_fuel_rate_s
    lowered_count = int(lowered_required_rate)
    lowered_source = replace(
        forged_source,
        required_fusion_reaction_rate_s=(
            0.5 * forged_source.required_fusion_reaction_rate_s
        ),
        required_per_species_fuel_rate_s=lowered_required_rate,
        reference_plant_minimum_per_species_fuel_rate_s=lowered_required_rate,
        required_to_coulter_rate_ratio=(
            lowered_required_rate / forged_source.coulter_measured_continuous_deuterium_rate_s
        ),
        validated_reactor_rate_deuterium_source_rate_s=lowered_required_rate,
        validated_reactor_rate_tritium_source_rate_s=lowered_required_rate,
        validated_reactor_rate_deuterium_output_count=lowered_count,
        validated_reactor_rate_tritium_output_count=lowered_count,
        validated_wall_plug_measurement_injected_dt_pair_count=lowered_count,
        validated_wall_plug_total_energy_j=(
            lowered_count * evidence_loop.EV_TO_JOULE
        ),
        validated_wall_plug_total_energy_linearized_std_j=(
            0.1 * lowered_count * evidence_loop.EV_TO_JOULE
        ),
    )
    lowered_deuterium = _write_verified_artifact(
        tmp_path,
        evidence_class=REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        stem="lowered-deuterium",
        raw_payload=_source_raw_payload(lowered_source, "D"),
    )
    lowered_tritium = _write_verified_artifact(
        tmp_path,
        evidence_class=REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        stem="lowered-tritium",
        raw_payload=_source_raw_payload(lowered_source, "T"),
    )
    lowered_burn = _write_verified_artifact(
        tmp_path,
        evidence_class=BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        stem="lowered-burn",
        raw_payload=_burn_raw_payload(
            lowered_source,
            deuterium_hash=lowered_deuterium[1],
            tritium_hash=lowered_tritium[1],
            reaction_operator_hash=operator_digest,
        ),
    )
    lowered_wall = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        stem="lowered-wall",
        raw_payload=_wall_raw_payload(
            lowered_source,
            deuterium_hash=lowered_deuterium[1],
            tritium_hash=lowered_tritium[1],
        ),
    )
    lowered_artifacts = (
        lowered_deuterium,
        lowered_tritium,
        lowered_burn,
        lowered_wall,
    )
    lowered_source = replace(
        lowered_source,
        validated_reactor_rate_deuterium_source_artifact=lowered_artifacts[0][0],
        validated_reactor_rate_deuterium_source_provenance_digest=lowered_artifacts[0][1],
        validated_reactor_rate_tritium_source_artifact=lowered_artifacts[1][0],
        validated_reactor_rate_tritium_source_provenance_digest=lowered_artifacts[1][1],
        validated_burn_retention_artifact=lowered_artifacts[2][0],
        validated_burn_retention_provenance_digest=lowered_artifacts[2][1],
        validated_wall_plug_artifact=lowered_artifacts[3][0],
        validated_wall_plug_measurement_provenance=lowered_artifacts[3][1],
    )
    assert not evidence_loop._polarized_source_evidence_record_integrity_pass(
        lowered_source,
        repository_root=tmp_path,
    )
    assert not physical_published_reaction_evidence_gate_pass(forged_reaction)
    assert not physical_polarized_source_evidence_gate_pass(forged_source)

    reaction_with_uncertainty_margin = replace(
        forged_reaction,
        validated_target_state_maxwellian_reactivity_ratio=1.019,
        validated_target_state_maxwellian_reactivity_ratio_std=0.0015,
        validated_target_state_maxwellian_reactivity_ratio_lower_3sigma=1.0145,
    )
    assert not evidence_loop._published_reaction_evidence_record_integrity_pass(
        reaction_with_uncertainty_margin,
        repository_root=tmp_path,
    )
    margin_artifact, margin_digest = _write_verified_artifact(
        tmp_path,
        evidence_class=POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        stem="reaction-margin",
        raw_payload=_reaction_raw_payload(reaction_with_uncertainty_margin),
    )
    reaction_with_uncertainty_margin = replace(
        reaction_with_uncertainty_margin,
        target_state_operator_provenance_digest=margin_digest,
        target_state_operator_artifact=margin_artifact,
    )
    assert margin_artifact.runtime_artifact_gate_pass
    assert evidence_loop._published_reaction_evidence_record_integrity_pass(
        reaction_with_uncertainty_margin,
        repository_root=tmp_path,
    )
    assert not evidence_loop._published_reaction_evidence_record_integrity_pass(
        replace(
            forged_reaction,
            han_initial_double_polarized_state_operator_available=True,
        ),
        repository_root=tmp_path,
    )
    assert not evidence_loop._published_reaction_evidence_record_integrity_pass(
        replace(
            forged_reaction,
            dries_record_is_not_double_polarized_czz_or_total_rate=False,
        ),
        repository_root=tmp_path,
    )

    for detached_source in (
        replace(
            forged_source,
            validated_reactor_rate_deuterium_source_provenance_digest=(
                forged_source.coulter_source_doi
            ),
        ),
        replace(
            forged_source,
            validated_reactor_rate_tritium_source_provenance_digest=(
                forged_source.utsuro_source_doi
            ),
        ),
        replace(
            forged_source,
            validated_reactor_rate_deuterium_source_provenance_digest=(
                forged_source.anke_source_doi
            ),
        ),
        replace(
            forged_source,
            validated_reactor_rate_tritium_source_provenance_digest=(
                forged_source.souers_source_doi
            ),
        ),
        replace(
            forged_source,
            validated_burn_retention_provenance_digest=forged_source.cook_source_doi,
        ),
        replace(
            forged_source,
            validated_burn_retention_provenance_digest=forged_source.collins_source_doi,
        ),
        replace(forged_source, anke_record_meets_reactor_deuterium_rate=True),
        replace(
            forged_source,
            anke_rf_power_is_partial_component_not_complete_wall_plug=False,
        ),
        replace(
            forged_source,
            souers_result_is_preburn_not_burn_weighted_plasma_retention=False,
        ),
        replace(
            forged_source,
            collins_result_is_preburn_not_burn_weighted_plasma_retention=False,
        ),
        replace(
            forged_source,
            validated_reactor_rate_deuterium_source_rate_linearized_std_s=(
                forged_source.required_per_species_fuel_rate_s
            ),
        ),
        replace(
            forged_source,
            validated_burn_weighted_dt_polarization_product_std=0.01,
        ),
        replace(
            forged_source,
            validated_wall_plug_total_energy_j=(
                forged_source.validated_wall_plug_total_energy_j * 1.001
            ),
        ),
        replace(
            forged_source,
            validated_reactor_rate_tritium_source_provenance_digest=deuterium_digest,
            validated_reactor_rate_tritium_source_artifact=deuterium_artifact,
        ),
    ):
        assert not evidence_loop._polarized_source_evidence_record_integrity_pass(
            detached_source,
            repository_root=tmp_path,
        )

    forged_report = replace(
        report,
        reaction=forged_reaction,
        source=forged_source,
        physical_reaction_evidence_gate_pass=True,
        physical_source_evidence_gate_pass=True,
        physical_spin_fusion_evidence_gate_pass=True,
    )
    assert not physical_polarized_evidence_gate_pass(forged_report)


def test_fresh_canonical_report_cannot_be_poisoned_through_a_cached_alias() -> None:
    poisoned = current_fusion_polarized_evidence_report()
    object.__setattr__(poisoned, "physical_reaction_evidence_gate_pass", True)
    object.__setattr__(poisoned, "physical_source_evidence_gate_pass", True)
    object.__setattr__(poisoned, "physical_spin_fusion_evidence_gate_pass", True)
    object.__setattr__(
        poisoned,
        "energy_resolved_burn_operator_sufficient_statistics_available",
        True,
    )
    object.__setattr__(
        poisoned.source,
        "energy_resolved_burn_operator_sufficient_statistics_available",
        True,
    )

    fresh = current_fusion_polarized_evidence_report()
    assert fresh is not poisoned
    assert not fresh.physical_reaction_evidence_gate_pass
    assert not fresh.physical_source_evidence_gate_pass
    assert not fresh.physical_spin_fusion_evidence_gate_pass
    assert not fresh.energy_resolved_burn_operator_sufficient_statistics_available
    assert not (
        fresh.source.energy_resolved_burn_operator_sufficient_statistics_available
    )
    assert not physical_polarized_evidence_gate_pass(poisoned)


def test_public_gates_reject_always_equal_foreign_objects(report) -> None:
    assert not physical_published_reaction_evidence_gate_pass(
        _AlwaysEqualProxy(report.reaction)
    )
    assert not physical_polarized_source_evidence_gate_pass(_AlwaysEqualProxy(report.source))
    assert not physical_polarized_evidence_gate_pass(_AlwaysEqualProxy(report))


def test_public_gates_return_false_for_malformed_exact_dataclasses(report) -> None:
    malformed_reaction = replace(report.reaction, temperature_kev="not-a-temperature")
    malformed_source = replace(
        report.source,
        declared_reference_fusion_power_w="not-a-power",
    )
    malformed_report = replace(report, reaction="not-a-reaction-audit")

    assert not physical_published_reaction_evidence_gate_pass(malformed_reaction)
    assert not physical_polarized_source_evidence_gate_pass(malformed_source)
    assert not physical_polarized_evidence_gate_pass(malformed_report)


def test_report_keeps_physical_and_ce_claims_closed(report) -> None:
    assert report.schema_version == "fusion-polarized-evidence-v7"
    assert report.energy_dependent_figure_control_reproduced
    assert not report.physical_reaction_evidence_gate_pass
    assert not report.physical_source_evidence_gate_pass
    assert not (
        report.burn_state_operator_componentwise_3sigma_stress_envelope_control_pass
    )
    assert not report.energy_resolved_burn_operator_sufficient_statistics_available
    assert not report.physical_spin_fusion_evidence_gate_pass
    assert report.maximum_supported_stage == (
        "PUBLISHED_ENERGY_DEPENDENT_FIGURE_DIGITIZED_MAXWELLIAN_CONTROL_"
        "PHYSICAL_EVIDENCE_FAIL_CLOSED"
    )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (audit_published_polarized_dt_reactivity, {"temperature_kev": 0.1}),
        (audit_published_polarized_dt_reactivity, {"temperature_kev": math.nan}),
        (audit_published_polarized_dt_reactivity, {"target_reactivity_ratio": 1.0}),
        (audit_published_polarized_dt_reactivity, {"target_reactivity_ratio": 1.0001}),
        (audit_published_polarized_dt_reactivity, {"energy_grid_points": 100}),
        (audit_published_polarized_dt_reactivity, {"energy_grid_points": True}),
        (audit_published_polarized_dt_reactivity, {"digitization_half_width": -1.0}),
        (audit_published_polarized_dt_reactivity, {"digitization_half_width": True}),
        (audit_polarized_fuel_primary_evidence, {"reference_fusion_power_w": 0.0}),
        (audit_polarized_fuel_primary_evidence, {"single_pass_burn_fraction": 1.0}),
    ],
)
def test_invalid_evidence_inputs_fail_closed(function, kwargs) -> None:
    with pytest.raises(ValueError):
        function(**kwargs)
