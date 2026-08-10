from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import math

import numpy as np
import pytest

from reality_stone.clarus.fusion_polarized_artifact_payloads import (
    ArtifactPayloadError,
    BASELINE_CROSS_SECTION_MODEL,
    BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
    CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER,
    COVARIANCE_FACTOR_REPRESENTATION,
    COVARIANCE_UNCERTAINTY_LEVEL,
    COVARIANCE_UNCERTAINTY_SCOPE,
    DERIVED_CLAIMS_SCHEMA_VERSION,
    INITIAL_STATE_BASIS,
    LONGITUDINAL_CHANNEL_ORDER,
    LONGITUDINAL_QUANTIZATION_AXIS,
    PAYLOAD_SCHEMA_VERSION,
    POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
    POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
    PYTHON_CANONICAL_JSON_CONVENTION,
    REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
    REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
    REACTION_ENERGY_FRAME,
    REACTION_OBSERVABLE,
    SCALAR_TARGET_RATIO_CONTROL_VARIANT,
    SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT,
    WALL_PLUG_COMPONENT_CATEGORIES,
    derive_artifact_claims as _raw_derive_artifact_claims,
)


REACTION_HASH = "sha256:" + "c" * 64
DEUTERIUM_HASH = "sha256:" + "a" * 64
TRITIUM_HASH = "sha256:" + "b" * 64
BURN_HASH = "sha256:" + "d" * 64
WALL_HASH = "sha256:" + "e" * 64
PLANT_ID = "dt-500mw-burn-0p01-v1"


def _plant() -> dict[str, object]:
    return {
        "configuration_id": PLANT_ID,
        "fusion_power_w": 5.0e8,
        "fusion_power_unit": "W",
        "single_pass_burn_fraction": 0.01,
        "burn_fraction_unit": "1",
    }


def _reaction_payload() -> dict[str, object]:
    point_count = 101
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        "payload_role": "operator_grid",
        "operator_variant": SCALAR_TARGET_RATIO_CONTROL_VARIANT,
        "energy_frame": REACTION_ENERGY_FRAME,
        "observable": REACTION_OBSERVABLE,
        "baseline_cross_section_model": BASELINE_CROSS_SECTION_MODEL,
        "quantization_axis": LONGITUDINAL_QUANTIZATION_AXIS,
        "initial_state_basis": INITIAL_STATE_BASIS,
        "temperature_kev": 10.0,
        "temperature_unit": "keV",
        "target_reactivity_ratio": 1.01,
        "energy_grid_kev": np.geomspace(0.5, 550.0, point_count).tolist(),
        "energy_unit": "keV",
        "reactivity_ratio_grid": [1.02] * point_count,
        "ratio_unit": "1",
        "reactivity_ratio_covariance": [[4.0e-6] * point_count for _ in range(point_count)],
        "covariance_unit": "1",
        "target_deuteron_vector_polarization": 0.2,
        "target_deuteron_tensor_polarization": 0.0,
        "target_triton_vector_polarization": 0.2,
        "polarization_unit": "1",
    }


def _six_channel_reaction_payload(
    *,
    p_d: float = 0.2,
    p_zz: float = 0.0,
    p_t: float = 0.2,
    integrated_ratio: float = 1.02,
    integrated_std: float = 0.002,
    target_ratio: float = 1.01,
) -> dict[str, object]:
    point_count = 101
    energy = np.geomspace(0.5, 550.0, point_count).tolist()
    correlation = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    polarization_product = p_d * p_t
    if polarization_product == 0.0:
        raise ValueError("test fixture requires nonzero p_d*p_t")
    ratio_scale = (integrated_ratio - 1.0) / polarization_product
    uncertainty_scale = integrated_std / abs(polarization_product)
    channel_constants = [1.0 + ratio_scale * value for value in correlation]
    channel_grids = [[value] * point_count for value in channel_constants]

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
    target_grid = [
        math.fsum(weight * value for weight, value in zip(weights, channel_constants))
    ] * point_count
    factor_mode = [
        uncertainty_scale * correlation[channel]
        for channel in range(6)
        for _ in range(point_count)
    ]
    return {
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
        "target_reactivity_ratio": target_ratio,
        "energy_grid_kev": energy,
        "energy_unit": "keV",
        "channel_order": list(LONGITUDINAL_CHANNEL_ORDER),
        "channel_reactivity_ratio_grids": channel_grids,
        "target_channel_weights": weights,
        "target_state_reactivity_ratio_grid": target_grid,
        "ratio_unit": "1",
        "reactivity_ratio_covariance_factor": {
            "representation": COVARIANCE_FACTOR_REPRESENTATION,
            "flatten_order": CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER,
            "factor_unit": "1",
            "uncertainty_scope": COVARIANCE_UNCERTAINTY_SCOPE,
            "uncertainty_level": COVARIANCE_UNCERTAINTY_LEVEL,
            "modes": [factor_mode],
        },
        "target_deuteron_vector_polarization": p_d,
        "target_deuteron_tensor_polarization": p_zz,
        "target_triton_vector_polarization": p_t,
        "polarization_unit": "1",
    }


def _set_longitudinal_shapes(
    payload: dict[str, object],
    *,
    ratio_amplitudes: list[float],
    factor_amplitudes: list[float],
) -> None:
    if len(ratio_amplitudes) != 101 or len(factor_amplitudes) != 101:
        raise ValueError("test shapes must match the 101-point fixture")
    correlation = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    channel_grids = [
        [1.0 + correlation[channel] * amplitude for amplitude in ratio_amplitudes]
        for channel in range(6)
    ]
    weights = payload["target_channel_weights"]
    payload["channel_reactivity_ratio_grids"] = channel_grids
    payload["target_state_reactivity_ratio_grid"] = [
        math.fsum(
            weight * channel_grids[channel][energy_index]
            for channel, weight in enumerate(weights)  # type: ignore[arg-type]
        )
        for energy_index in range(101)
    ]
    payload["reactivity_ratio_covariance_factor"]["modes"] = [  # type: ignore[index]
        [
            correlation[channel] * factor_amplitudes[energy_index]
            for channel in range(6)
            for energy_index in range(101)
        ]
    ]


def _source_payload(species: str) -> dict[str, object]:
    evidence_class = (
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS
        if species == "D"
        else REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS
    )
    measurement: dict[str, object] = {
        "species": species,
        "output_count": 20_000_000_000_000_000_000_000,
        "output_count_unit": "particles",
        "output_count_std": 1.0e18,
        "measurement_duration_s": 1.0,
        "duration_unit": "s",
        "polarization_unit": "1",
        "uncertainty_level": "1sigma",
    }
    if species == "D":
        measurement.update(
            {
                "deuteron_vector_polarization": 0.5,
                "deuteron_vector_polarization_std": 0.001,
                "deuteron_tensor_polarization": 0.0,
                "deuteron_tensor_polarization_std": 0.001,
            }
        )
    else:
        measurement.update(
            {
                "triton_vector_polarization": 0.1,
                "triton_vector_polarization_std": 0.001,
            }
        )
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": evidence_class,
        "payload_role": "measurement",
        "plant_configuration": _plant(),
        "measurement": measurement,
    }


def _burn_payload() -> dict[str, object]:
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        "payload_role": "measurement",
        "plant_configuration": _plant(),
        "measurement": {
            "burn_weighted_dt_polarization_product": 0.03,
            "burn_weighted_dt_polarization_product_std": 0.001,
            "burn_weighted_deuteron_tensor_polarization": 0.0,
            "burn_weighted_deuteron_tensor_polarization_std": 0.01,
            "polarization_unit": "1",
            "uncertainty_level": "1sigma",
            "reaction_count": 1_000_000,
            "reaction_count_unit": "reactions",
            "measurement_duration_s": 2.0,
            "duration_unit": "s",
            "deuterium_source_manifest_sha256": DEUTERIUM_HASH,
            "tritium_source_manifest_sha256": TRITIUM_HASH,
            "reaction_operator_manifest_sha256": REACTION_HASH,
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


def _wall_payload() -> dict[str, object]:
    components = {
        category: {"energy_j": float(index), "energy_std_j": float(index) / 10.0}
        for index, category in enumerate(sorted(WALL_PLUG_COMPONENT_CATEGORIES), start=1)
    }
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "evidence_class": POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        "payload_role": "component_ledger",
        "plant_configuration": _plant(),
        "ledger": {
            "uncertainty_model": "manifest_bound_correlated_1sigma",
            "energy_unit": "J",
            "components": components,
            "injected_dt_pair_count": 20_000_000_000_000_000_000_000,
            "pair_count_unit": "pairs",
            "measurement_duration_s": 1.0,
            "duration_unit": "s",
            "deuterium_source_manifest_sha256": DEUTERIUM_HASH,
            "tritium_source_manifest_sha256": TRITIUM_HASH,
        },
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _engineering_auxiliary_payloads(
    payload: dict[str, object],
) -> dict[str, dict[str, object]]:
    evidence_class = str(payload["evidence_class"])
    if evidence_class in {
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
    }:
        measurement = payload["measurement"]
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
                "background_count": 3_000_000_000_000_000,
                "background_count_std": 1.0e15,
                "background_count_unit": "particles",
            },
            "pipeline": pipeline,
        }
    if evidence_class == BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS:
        measurement = payload["measurement"]
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
    ledger = payload["ledger"]
    primary_components = ledger["components"]  # type: ignore[index]
    categories = sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    present_categories = sorted(primary_components)
    return {
        "meter_records": {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "evidence_class": evidence_class,
            "payload_role": "meter_records",
            "energy_unit": "J",
            "components": {
                category: {
                    "energy_j": primary_components[category]["energy_j"],  # type: ignore[index]
                    "energy_std_j": primary_components[category]["energy_std_j"],  # type: ignore[index]
                    "meter_id": f"meter-{index:02d}-v1",
                }
                for index, category in enumerate(present_categories, start=1)
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
                category: f"energy-calibration-{index:02d}-v1"
                for index, category in enumerate(categories, start=1)
            },
            "pair_counter_calibration_id": "pair-counter-calibration-v1",
            "clock_calibration_id": "clock-calibration-v1",
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


def _engineering_derivation_inputs(
    payload: dict[str, object],
    auxiliary_payloads: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    auxiliary = (
        _engineering_auxiliary_payloads(payload)
        if auxiliary_payloads is None
        else auxiliary_payloads
    )
    auxiliary_hashes = {
        role: _payload_sha256(document) for role, document in auxiliary.items()
    }
    payload["auxiliary_payload_sha256s"] = auxiliary_hashes
    typed_hashes = {
        str(payload["payload_role"]): _payload_sha256(payload),
        **auxiliary_hashes,
    }
    return auxiliary, typed_hashes


def derive_artifact_claims(
    evidence_class: str,
    payload: dict[str, object],
    *,
    manifest_sha256: str,
    auxiliary_payloads: dict[str, dict[str, object]] | None = None,
    typed_payload_sha256s: dict[str, str] | None = None,
):
    if (
        evidence_class == POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS
        or evidence_class
        not in {
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
            POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        }
        or type(payload) is not dict
    ):
        return _raw_derive_artifact_claims(
            evidence_class, payload, manifest_sha256=manifest_sha256
        )
    if typed_payload_sha256s is None:
        auxiliary_payloads, typed_payload_sha256s = _engineering_derivation_inputs(
            payload, auxiliary_payloads
        )
    return _raw_derive_artifact_claims(
        evidence_class,
        payload,
        manifest_sha256=manifest_sha256,
        auxiliary_payloads=auxiliary_payloads,
        typed_payload_sha256s=typed_payload_sha256s,
    )


def test_reaction_grid_recomputes_maxwellian_ratio_covariance_and_spin_state() -> None:
    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        _reaction_payload(),
        manifest_sha256=REACTION_HASH,
    )
    claims = result.claims

    assert result.schema_version == DERIVED_CLAIMS_SCHEMA_VERSION
    assert result.plant_configuration_id is None
    assert result.linked_manifest_sha256s == ()
    assert claims["validated_target_state_maxwellian_reactivity_ratio"] == pytest.approx(1.02)
    assert claims["validated_target_state_maxwellian_reactivity_ratio_std"] == pytest.approx(
        0.002
    )
    assert claims["validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"] == (
        pytest.approx(1.014)
    )
    assert claims["target_ratio_pass_at_3sigma"] is True
    assert claims["target_state_resolved_operator_available"] is False
    assert claims["validated_target_deuteron_mplus_population"] == pytest.approx(13.0 / 30.0)
    assert claims["validated_target_deuteron_mzero_population"] == pytest.approx(1.0 / 3.0)
    assert claims["validated_target_deuteron_mminus_population"] == pytest.approx(7.0 / 30.0)


def test_six_channel_operator_contracts_physical_spin_weights_and_factor_covariance() -> None:
    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        _six_channel_reaction_payload(),
        manifest_sha256=REACTION_HASH,
    )
    claims = result.claims

    assert claims["operator_variant"] == SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT
    assert claims["energy_frame"] == REACTION_ENERGY_FRAME
    assert claims["observable"] == REACTION_OBSERVABLE
    assert claims["baseline_cross_section_model"] == BASELINE_CROSS_SECTION_MODEL
    assert claims["quantization_axis"] == LONGITUDINAL_QUANTIZATION_AXIS
    assert claims["initial_state_basis"] == INITIAL_STATE_BASIS
    assert claims["operator_channel_order"] == list(LONGITUDINAL_CHANNEL_ORDER)
    assert claims["covariance_factor_representation"] == COVARIANCE_FACTOR_REPRESENTATION
    assert claims["covariance_factor_flatten_order"] == (
        CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER
    )
    assert claims["covariance_uncertainty_scope"] == COVARIANCE_UNCERTAINTY_SCOPE
    assert claims["covariance_uncertainty_level"] == COVARIANCE_UNCERTAINTY_LEVEL
    assert claims["validated_target_state_maxwellian_reactivity_ratio"] == pytest.approx(1.02)
    assert claims["validated_target_state_maxwellian_reactivity_ratio_std"] == pytest.approx(
        0.002
    )
    assert claims["validated_target_state_maxwellian_reactivity_ratio_lower_3sigma"] == (
        pytest.approx(1.014)
    )
    assert claims["target_state_resolved_operator_available"] is True
    assert claims["target_ratio_pass_at_3sigma"] is True
    assert claims["unpolarized_channel_mean_max_abs_residual"] == pytest.approx(0.0)
    assert claims["simultaneous_spin_reversal_max_abs_residual"] == pytest.approx(0.0)
    assert claims["target_channel_weight_max_abs_residual"] == pytest.approx(0.0)
    assert claims["target_ratio_grid_max_abs_residual"] == pytest.approx(0.0)
    assert claims["covariance_mode_channel_sum_max_abs_residual"] == pytest.approx(0.0)
    assert claims["covariance_mode_spin_reversal_max_abs_residual"] == pytest.approx(0.0)
    assert claims["simultaneous_spin_reversal_exact"] is True
    assert claims["covariance_mode_spin_reversal_exact"] is True
    assert claims["burn_state_operator_contraction_variables"] == [
        "intercept_one",
        "burn_weighted_dt_polarization_product",
        "burn_weighted_deuteron_tensor_polarization",
    ]
    assert claims["burn_state_operator_response_coefficients"] == pytest.approx(
        [1.0, 0.5, 0.0]
    )
    assert np.asarray(
        claims["burn_state_operator_response_coefficient_covariance"]
    ) == pytest.approx(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.05**2, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    assert claims["covariance_factor_mode_count"] == 1
    assert claims["covariance_factor_rank"] == 1
    assert claims["flattened_derivative_weight_count"] == 606
    assert claims["maxwellian_quadrature_convergence_pass"] is True
    assert claims["maxwellian_quadrature_coarsening_strides"] == [2, 4]
    assert claims["maxwellian_quadrature_phase_count"] == 6
    assert len(claims["maxwellian_quadrature_phase_summaries"]) == 6
    assert claims["operator_local_adjacent_max_relative_change"] == pytest.approx(0.0)
    assert claims["operator_local_second_difference_max_relative"] == pytest.approx(0.0)
    assert claims["covariance_local_adjacent_max_relative_change"] == pytest.approx(0.0)
    assert claims["covariance_local_second_difference_max_relative"] == pytest.approx(0.0)
    assert claims["coarsened_target_ratio_pass_at_3sigma"] is True
    assert claims["coarsened_target_state_maxwellian_reactivity_ratio"] == pytest.approx(
        1.02
    )
    assert claims["coarsened_target_state_maxwellian_reactivity_ratio_std"] == pytest.approx(
        0.002
    )
    assert "spin_correlation_czz_directly_measured" not in claims
    assert "polarized_dt_rate_directly_validated" not in claims
    assert "author_s_matrix_or_angle_integrated_coefficients_available" not in claims


def test_six_channel_nonzero_tensor_pattern_matches_manual_spin_one_contraction() -> None:
    p_d, p_zz, p_t = 0.2, 0.2, 0.2
    vector_amplitude, tensor_amplitude = 0.25, 0.05
    payload = _six_channel_reaction_payload(p_d=p_d, p_zz=p_zz, p_t=p_t)
    vector_pattern = (1.0, -1.0, 0.0, 0.0, -1.0, 1.0)
    tensor_pattern = (1.0, 1.0, -2.0, -2.0, 1.0, 1.0)
    channel_constants = [
        1.0
        + vector_amplitude * vector_value
        + tensor_amplitude * tensor_value
        for vector_value, tensor_value in zip(
            vector_pattern, tensor_pattern, strict=True
        )
    ]
    payload["channel_reactivity_ratio_grids"] = [
        [value] * 101 for value in channel_constants
    ]
    weights = payload["target_channel_weights"]
    manual_ratio = math.fsum(
        weight * value
        for weight, value in zip(weights, channel_constants, strict=True)  # type: ignore[arg-type]
    )
    payload["target_state_reactivity_ratio_grid"] = [manual_ratio] * 101

    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        payload,
        manifest_sha256=REACTION_HASH,
    )

    assert manual_ratio == pytest.approx(
        1.0 + vector_amplitude * p_d * p_t + tensor_amplitude * p_zz
    )
    assert result.claims[
        "validated_target_state_maxwellian_reactivity_ratio"
    ] == pytest.approx(manual_ratio)
    assert result.claims["validated_target_deuteron_tensor_polarization"] == p_zz


def test_energy_dependent_symmetric_operator_and_factor_pass_coarsened_quadrature() -> None:
    payload = _six_channel_reaction_payload()
    log_coordinate = np.linspace(0.0, 1.0, 101)
    _set_longitudinal_shapes(
        payload,
        ratio_amplitudes=(0.45 + 0.10 * log_coordinate).tolist(),
        factor_amplitudes=(0.045 + 0.010 * log_coordinate).tolist(),
    )
    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        payload,
        manifest_sha256=REACTION_HASH,
    )
    claims = result.claims

    assert payload["channel_reactivity_ratio_grids"][0][0] != (  # type: ignore[index]
        payload["channel_reactivity_ratio_grids"][0][-1]  # type: ignore[index]
    )
    assert claims["target_state_resolved_operator_available"] is True
    assert claims["maxwellian_quadrature_convergence_pass"] is True
    assert claims["maxwellian_quadrature_max_relative_residual"] < 1.0e-3
    assert claims["coarsened_target_ratio_pass_at_3sigma"] is True


def test_smooth_log_energy_resonance_passes_local_regularity_and_all_phases() -> None:
    payload = _six_channel_reaction_payload()
    energy = np.asarray(payload["energy_grid_kev"], dtype=float)
    log_energy = np.log(energy)
    resonance = np.exp(-0.5 * ((log_energy - math.log(70.0)) / 0.35) ** 2)
    _set_longitudinal_shapes(
        payload,
        ratio_amplitudes=(0.45 + 0.10 * resonance).tolist(),
        factor_amplitudes=(0.045 + 0.010 * resonance).tolist(),
    )

    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        payload,
        manifest_sha256=REACTION_HASH,
    )
    claims = result.claims

    assert claims["maxwellian_quadrature_phase_count"] == 6
    assert claims["maxwellian_quadrature_convergence_pass"] is True
    assert claims["operator_local_adjacent_max_relative_change"] < claims[
        "local_adjacent_relative_change_limit"
    ]
    assert claims["operator_local_second_difference_max_relative"] < claims[
        "local_second_difference_relative_limit"
    ]
    assert claims["covariance_local_adjacent_max_relative_change"] < claims[
        "local_adjacent_relative_change_limit"
    ]
    assert claims["covariance_local_second_difference_max_relative"] < claims[
        "local_second_difference_relative_limit"
    ]


def test_multi_phase_quadrature_rejects_stride_four_alias_hidden_from_even_phase() -> None:
    payload = _six_channel_reaction_payload()
    phase_offsets = (0.0, 0.04, 0.0, -0.04)
    _set_longitudinal_shapes(
        payload,
        ratio_amplitudes=[
            0.5 + phase_offsets[index % 4]
            for index in range(101)
        ],
        factor_amplitudes=[0.05] * 101,
    )

    with pytest.raises(ArtifactPayloadError, match="multi-phase stride=4"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_local_regularity_rejects_grid_scale_hundred_percent_alternation() -> None:
    payload = _six_channel_reaction_payload()
    ratio_amplitudes = [
        0.5
        if index % 2 == 0
        else (0.0 if (index // 2) % 2 == 0 else 1.0)
        for index in range(101)
    ]
    factor_amplitudes = [
        0.05
        if index % 2 == 0
        else (0.0 if (index // 2) % 2 == 0 else 0.10)
        for index in range(101)
    ]
    _set_longitudinal_shapes(
        payload,
        ratio_amplitudes=ratio_amplitudes,
        factor_amplitudes=factor_amplitudes,
    )

    with pytest.raises(ArtifactPayloadError, match="local .*regularity"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_direct_svd_rank_keeps_duplicated_identical_modes_rank_one() -> None:
    payload = _six_channel_reaction_payload()
    factor = payload["reactivity_ratio_covariance_factor"]
    original_mode = list(factor["modes"][0])  # type: ignore[index]
    mode_count = 50
    scaled_mode = [value / math.sqrt(mode_count) for value in original_mode]
    factor["modes"] = [scaled_mode[:] for _ in range(mode_count)]  # type: ignore[index]

    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        payload,
        manifest_sha256=REACTION_HASH,
    )

    assert result.claims["covariance_factor_mode_count"] == mode_count
    assert result.claims["covariance_factor_rank"] == 1
    assert result.claims[
        "validated_target_state_maxwellian_reactivity_ratio_std"
    ] == pytest.approx(0.002)


def test_coarsened_lower_bound_must_independently_preserve_one_percent_target() -> None:
    payload = _six_channel_reaction_payload()
    ratio_amplitudes = [
        (0.0159 if index % 2 == 0 else 0.0165) / 0.04
        for index in range(101)
    ]
    _set_longitudinal_shapes(
        payload,
        ratio_amplitudes=ratio_amplitudes,
        factor_amplitudes=[0.05] * 101,
    )

    with pytest.raises(ArtifactPayloadError, match="coarsened Maxwellian lower"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_exact_reaction_variant_discriminator_rejects_missing_unknown_and_schema_swap() -> None:
    missing = _reaction_payload()
    missing.pop("operator_variant")
    unknown = _reaction_payload()
    unknown["operator_variant"] = "six_channel_longitudinal_initial_spin_v1 "
    schema_swap = _reaction_payload()
    schema_swap["operator_variant"] = SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT

    for payload in (missing, unknown, schema_swap):
        with pytest.raises(ArtifactPayloadError, match="operator_variant|keys are invalid"):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_six_channel_rejects_all_100_wrong_order_and_normalization_drift() -> None:
    all_100 = _six_channel_reaction_payload()
    all_100["channel_reactivity_ratio_grids"] = [[100.0] * 101 for _ in range(6)]
    all_100["target_state_reactivity_ratio_grid"] = [100.0] * 101
    wrong_order = _six_channel_reaction_payload()
    wrong_order["channel_order"][0], wrong_order["channel_order"][1] = (  # type: ignore[index]
        wrong_order["channel_order"][1],  # type: ignore[index]
        wrong_order["channel_order"][0],  # type: ignore[index]
    )
    normalization_drift = _six_channel_reaction_payload()
    normalization_drift["channel_reactivity_ratio_grids"][0][50] += 1.0e-8  # type: ignore[index]

    for payload in (all_100, wrong_order, normalization_drift):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_six_channel_rejects_spin_reversal_asymmetry_even_when_mean_and_target_pass() -> None:
    payload = _six_channel_reaction_payload()
    channel_constants = [0.9, 1.0, 1.0, 1.0, 1.0, 1.1]
    payload["channel_reactivity_ratio_grids"] = [
        [value] * 101 for value in channel_constants
    ]
    weights = payload["target_channel_weights"]
    payload["target_state_reactivity_ratio_grid"] = [  # type: ignore[index]
        math.fsum(
            weight * value
            for weight, value in zip(weights, channel_constants, strict=True)  # type: ignore[arg-type]
        )
    ] * 101

    with pytest.raises(ArtifactPayloadError, match="simultaneous spin reversal"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_six_channel_requires_exact_reversal_below_former_tolerances() -> None:
    central = _six_channel_reaction_payload()
    central["channel_reactivity_ratio_grids"][0][50] += 5.0e-13  # type: ignore[index]
    factor = _six_channel_reaction_payload()
    factor["reactivity_ratio_covariance_factor"]["modes"][0][50] += 5.0e-15  # type: ignore[index]

    for payload in (central, factor):
        with pytest.raises(ArtifactPayloadError, match="spin reversal exactly"):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_six_channel_rejects_covariance_normalization_drift_and_zero_uncertainty() -> None:
    drift = _six_channel_reaction_payload()
    drift["reactivity_ratio_covariance_factor"]["modes"][0][0] += 1.0e-8  # type: ignore[index]
    zero = _six_channel_reaction_payload()
    zero["reactivity_ratio_covariance_factor"]["modes"][0] = [0.0] * 606  # type: ignore[index]

    with pytest.raises(ArtifactPayloadError, match="preserve the unpolarized"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            drift,
            manifest_sha256=REACTION_HASH,
        )
    with pytest.raises(ArtifactPayloadError, match="nonzero uncertainty"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            zero,
            manifest_sha256=REACTION_HASH,
        )


def test_six_channel_rejects_numerically_meaningless_nonzero_covariance() -> None:
    payload = _six_channel_reaction_payload()
    pattern = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    payload["reactivity_ratio_covariance_factor"]["modes"][0] = [  # type: ignore[index]
        1.0e-100 * pattern[channel]
        for channel in range(6)
        for _ in range(101)
    ]

    with pytest.raises(ArtifactPayloadError, match="numerical-meaning floor"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_six_channel_rejects_covariance_spin_reversal_drift_with_zero_channel_sum() -> None:
    payload = _six_channel_reaction_payload()
    mode = payload["reactivity_ratio_covariance_factor"]["modes"][0]  # type: ignore[index]
    mode[0] += 1.0e-8
    mode[101] -= 1.0e-8

    with pytest.raises(ArtifactPayloadError, match="simultaneous spin reversal"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("uncertainty_scope", "counting_statistical"),
        ("uncertainty_level", "95_percent_interval"),
    ),
)
def test_six_channel_rejects_wrong_covariance_scope_or_level(
    field: str,
    bad_value: str,
) -> None:
    payload = _six_channel_reaction_payload()
    payload["reactivity_ratio_covariance_factor"][field] = bad_value  # type: ignore[index]

    with pytest.raises(ArtifactPayloadError, match=field):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


@pytest.mark.parametrize("mutation", ("flatten", "shape", "nonfinite", "too_many_modes"))
def test_six_channel_rejects_malformed_covariance_factor(mutation: str) -> None:
    payload = _six_channel_reaction_payload()
    factor = payload["reactivity_ratio_covariance_factor"]
    if mutation == "flatten":
        factor["flatten_order"] = "energy_major_channel_minor"  # type: ignore[index]
    elif mutation == "shape":
        factor["modes"][0] = factor["modes"][0][:-1]  # type: ignore[index]
    elif mutation == "nonfinite":
        factor["modes"][0][0] = math.inf  # type: ignore[index]
    else:
        factor["modes"] = [[] for _ in range(607)]  # type: ignore[index]

    with pytest.raises(ArtifactPayloadError):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_six_channel_rejects_spin_weight_and_target_grid_tampering() -> None:
    spin = _six_channel_reaction_payload()
    spin["target_deuteron_vector_polarization"] = 1.0
    spin["target_deuteron_tensor_polarization"] = -2.0
    weight = _six_channel_reaction_payload()
    weight["target_channel_weights"][0] += 1.0e-6  # type: ignore[index]
    target_grid = _six_channel_reaction_payload()
    target_grid["target_state_reactivity_ratio_grid"][50] += 1.0e-6  # type: ignore[index]

    for payload in (spin, weight, target_grid):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_six_channel_point_and_factor_value_ceilings_fail_before_dense_covariance(
    monkeypatch,
) -> None:
    oversized_grid = _six_channel_reaction_payload()
    oversized_grid["energy_grid_kev"] = [object()] * 2_002
    with pytest.raises(ArtifactPayloadError, match="101..2001"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            oversized_grid,
            manifest_sha256=REACTION_HASH,
        )

    import reality_stone.clarus.fusion_polarized_artifact_payloads as payload_module

    monkeypatch.setattr(payload_module, "MAX_COVARIANCE_FACTOR_VALUES", 605)
    factor_too_large = _six_channel_reaction_payload()
    with pytest.raises(ArtifactPayloadError, match="numeric-value ceiling"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            factor_too_large,
            manifest_sha256=REACTION_HASH,
        )


def test_six_channel_rejects_overlong_factor_row_before_numeric_copy() -> None:
    payload = _six_channel_reaction_payload()
    payload["reactivity_ratio_covariance_factor"]["modes"][0] = [  # type: ignore[index]
        object()
    ] * 607

    with pytest.raises(ArtifactPayloadError, match="exactly 6N flattened values"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_scalar_grid_ceiling_rejects_dense_covariance_dos_shape_early() -> None:
    payload = _reaction_payload()
    payload["energy_grid_kev"] = np.geomspace(0.5, 550.0, 502).tolist()

    with pytest.raises(ArtifactPayloadError, match="scalar energy_grid_kev"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_covariance_factor_normalization_is_scale_stable_and_overflow_fails_closed() -> None:
    scaled = _six_channel_reaction_payload()
    pattern = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    scaled["reactivity_ratio_covariance_factor"]["modes"][0] = [  # type: ignore[index]
        0.075 * pattern[channel]
        for channel in range(6)
        for _ in range(101)
    ]
    scaled_result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        scaled,
        manifest_sha256=REACTION_HASH,
    )
    assert scaled_result.claims["covariance_mode_channel_sum_max_abs_residual"] == 0.0
    assert scaled_result.claims[
        "validated_target_state_maxwellian_reactivity_ratio_std"
    ] == pytest.approx(0.003)

    target_failure = _six_channel_reaction_payload()
    target_failure["reactivity_ratio_covariance_factor"]["modes"][0] = [  # type: ignore[index]
        1.0e4 * pattern[channel]
        for channel in range(6)
        for _ in range(101)
    ]
    with pytest.raises(ArtifactPayloadError, match="lower 3sigma"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            target_failure,
            manifest_sha256=REACTION_HASH,
        )

    overflow = _six_channel_reaction_payload()
    overflow["reactivity_ratio_covariance_factor"]["modes"][0] = [  # type: ignore[index]
        1.0e308 * pattern[channel]
        for channel in range(6)
        for _ in range(101)
    ]
    with pytest.raises(ArtifactPayloadError):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            overflow,
            manifest_sha256=REACTION_HASH,
        )


def test_derived_claims_are_canonical_and_caller_mutation_cannot_change_them() -> None:
    result = derive_artifact_claims(
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        _source_payload("T"),
        manifest_sha256=TRITIUM_HASH,
    )
    expected_canonical = json.dumps(
        result.claims, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    assert result.claims_canonical_json == expected_canonical
    assert result.claims_sha256 == (
        "sha256:" + hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()
    )
    assert result.canonical_json_convention == PYTHON_CANONICAL_JSON_CONVENTION
    first = result.claims
    first["source_rate_s"] = 1.0
    assert result.claims["source_rate_s"] != 1.0
    with pytest.raises(FrozenInstanceError):
        result.manifest_sha256 = DEUTERIUM_HASH  # type: ignore[misc]


def test_engineering_derivation_requires_exact_auxiliary_role_to_byte_hash_map() -> None:
    missing = _source_payload("T")
    with pytest.raises(ArtifactPayloadError):
        _raw_derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            missing,
            manifest_sha256=TRITIUM_HASH,
        )

    payload = _source_payload("T")
    auxiliary, typed_hashes = _engineering_derivation_inputs(payload)
    result = _raw_derive_artifact_claims(
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        payload,
        manifest_sha256=TRITIUM_HASH,
        auxiliary_payloads=auxiliary,
        typed_payload_sha256s=typed_hashes,
    )
    assert result.typed_payload_sha256s == tuple(sorted(typed_hashes.items()))

    payload["auxiliary_payload_sha256s"]["pipeline"] = "sha256:" + "f" * 64  # type: ignore[index]
    with pytest.raises(ArtifactPayloadError, match="manifest-bound payload bytes"):
        _raw_derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            payload,
            manifest_sha256=TRITIUM_HASH,
            auxiliary_payloads=auxiliary,
            typed_payload_sha256s=typed_hashes,
        )


def test_negative_zero_is_not_a_distinct_canonical_measurement_value() -> None:
    payload = _source_payload("T")
    payload["measurement"]["triton_vector_polarization"] = -0.0  # type: ignore[index]
    with pytest.raises(ArtifactPayloadError, match="negative zero"):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            payload,
            manifest_sha256=TRITIUM_HASH,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("temperature_kev", 9.999),
        ("temperature_unit", "eV"),
        ("target_reactivity_ratio", 1.0),
        ("energy_unit", "MeV"),
        ("ratio_unit", "%"),
        ("covariance_unit", "percent2"),
        ("polarization_unit", "%"),
    ),
)
def test_reaction_rejects_wrong_fixed_values_and_units(field: str, bad_value: object) -> None:
    payload = _reaction_payload()
    payload[field] = bad_value
    with pytest.raises(ArtifactPayloadError):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("energy_frame", "laboratory"),
        ("observable", "differential_cross_section_ratio"),
        ("baseline_cross_section_model", "arbitrary_fit"),
        ("quantization_axis", "transverse_axis"),
        ("initial_state_basis", "coherent_unspecified_basis"),
    ),
)
def test_reaction_variants_reject_ambiguous_operator_semantics(
    field: str,
    bad_value: object,
) -> None:
    for payload in (_reaction_payload(), _six_channel_reaction_payload()):
        payload[field] = bad_value
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_reaction_variants_reject_sub_one_percent_target_or_lower_bound() -> None:
    for payload in (_reaction_payload(), _six_channel_reaction_payload()):
        payload["target_reactivity_ratio"] = 1.0001
        with pytest.raises(ArtifactPayloadError, match="at least 1.01"):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )

    lower_failure = _six_channel_reaction_payload(
        integrated_ratio=1.0101,
        integrated_std=0.0001,
    )
    with pytest.raises(ArtifactPayloadError, match="lower 3sigma"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            lower_failure,
            manifest_sha256=REACTION_HASH,
        )


def test_reaction_rejects_short_nonspanning_nonincreasing_and_nonpositive_grids() -> None:
    short = _reaction_payload()
    short["energy_grid_kev"] = list(short["energy_grid_kev"])[:100]  # type: ignore[arg-type]
    short["reactivity_ratio_grid"] = list(short["reactivity_ratio_grid"])[:100]  # type: ignore[arg-type]
    short["reactivity_ratio_covariance"] = [
        list(row)[:100] for row in list(short["reactivity_ratio_covariance"])[:100]  # type: ignore[arg-type]
    ]
    nonspanning = _reaction_payload()
    nonspanning["energy_grid_kev"][0] = 0.6  # type: ignore[index]
    nonincreasing = _reaction_payload()
    nonincreasing["energy_grid_kev"][50] = nonincreasing["energy_grid_kev"][49]  # type: ignore[index]
    nonpositive = _reaction_payload()
    nonpositive["reactivity_ratio_grid"][10] = 0.0  # type: ignore[index]

    for payload in (short, nonspanning, nonincreasing, nonpositive):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_reaction_rejects_bool_nonfinite_asymmetric_and_non_psd_covariance() -> None:
    boolean = _reaction_payload()
    boolean["reactivity_ratio_grid"][0] = True  # type: ignore[index]
    nonfinite = _reaction_payload()
    nonfinite["reactivity_ratio_grid"][0] = math.nan  # type: ignore[index]
    asymmetric = _reaction_payload()
    asymmetric["reactivity_ratio_covariance"][0][1] = 1.0e-3  # type: ignore[index]
    non_psd = _reaction_payload()
    non_psd["reactivity_ratio_covariance"] = np.eye(101).tolist()
    non_psd["reactivity_ratio_covariance"][0][0] = -1.0  # type: ignore[index]

    for payload in (boolean, nonfinite, asymmetric, non_psd):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


def test_reaction_rejects_scale_hidden_negative_covariance_and_does_not_clamp_it() -> None:
    payload = _reaction_payload()
    covariance = np.zeros((101, 101))
    covariance[0, 0] = 1.0
    covariance[50, 50] = -5.0e-11
    payload["reactivity_ratio_covariance"] = covariance.tolist()

    with pytest.raises(ArtifactPayloadError, match="diagonal must be nonnegative"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_scalar_ratio_grid_cannot_claim_a_state_resolved_spin_operator() -> None:
    payload = _reaction_payload()
    payload["reactivity_ratio_grid"] = [100.0] * 101
    payload["reactivity_ratio_covariance"] = np.zeros((101, 101)).tolist()
    payload["target_deuteron_vector_polarization"] = 0.0
    payload["target_deuteron_tensor_polarization"] = 0.0
    payload["target_triton_vector_polarization"] = 0.0

    result = derive_artifact_claims(
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        payload,
        manifest_sha256=REACTION_HASH,
    )
    assert result.claims["validated_target_state_maxwellian_reactivity_ratio"] == pytest.approx(
        100.0
    )
    assert result.claims["target_state_resolved_operator_available"] is False
    assert "spin_correlation_czz_directly_measured" not in result.claims
    assert "polarized_dt_rate_directly_validated" not in result.claims


def test_reaction_rejects_unresolved_log_energy_gap() -> None:
    payload = _reaction_payload()
    energy = np.geomspace(0.5, 550.0, 101)
    energy[1] = 50.0
    energy[2:] = np.geomspace(50.1, 550.0, 99)
    payload["energy_grid_kev"] = energy.tolist()

    with pytest.raises(ArtifactPayloadError, match="unresolved logarithmic gap"):
        derive_artifact_claims(
            POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
            payload,
            manifest_sha256=REACTION_HASH,
        )


def test_reaction_rejects_covariance_shape_and_unphysical_spin_states() -> None:
    shape = _reaction_payload()
    shape["reactivity_ratio_covariance"][0] = [0.0] * 100  # type: ignore[index]
    spin_one = _reaction_payload()
    spin_one["target_deuteron_vector_polarization"] = 1.0
    spin_one["target_deuteron_tensor_polarization"] = -2.0
    triton = _reaction_payload()
    triton["target_triton_vector_polarization"] = 1.01

    for payload in (shape, spin_one, triton):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
                payload,
                manifest_sha256=REACTION_HASH,
            )


@pytest.mark.parametrize(
    ("species", "evidence_class", "manifest_hash"),
    (
        ("D", REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS, DEUTERIUM_HASH),
        ("T", REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS, TRITIUM_HASH),
    ),
)
def test_source_rate_and_three_sigma_bound_are_derived_from_count_and_duration(
    species: str, evidence_class: str, manifest_hash: str
) -> None:
    result = derive_artifact_claims(
        evidence_class, _source_payload(species), manifest_sha256=manifest_hash
    )
    claims = result.claims
    assert result.plant_configuration_id == PLANT_ID
    assert result.manifest_sha256 == manifest_hash
    corrected_count = 2.0e22 - 3.0e15
    expected_rate = corrected_count
    expected_std = math.sqrt(
        (1.0e18) ** 2
        + (1.0e15) ** 2
        + 2.0 * (expected_rate * 1.0e-6) ** 2
    )
    expected_lower = (2.0e22 - 3.0e18 - 6.0e15) / 1.000003
    assert claims["source_rate_s"] == pytest.approx(expected_rate)
    assert claims["source_rate_linearized_std_s"] == pytest.approx(expected_std)
    assert claims["source_rate_lower_3sigma_s"] == pytest.approx(
        expected_lower
    )
    assert claims["source_rate_3sigma_bound_model"] == (
        "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
    )
    assert claims["required_per_species_fuel_rate_s"] == pytest.approx(1.773155987e22)
    assert claims["reactor_rate_source_demonstrated"] is True


def test_source_below_required_flow_is_derived_as_failure_not_promoted() -> None:
    payload = _source_payload("T")
    payload["measurement"]["measurement_duration_s"] = 1.0e9  # type: ignore[index]
    result = derive_artifact_claims(
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        payload,
        manifest_sha256=TRITIUM_HASH,
    )
    assert result.claims["reactor_rate_source_demonstrated"] is False


def test_source_numeric_overflow_fails_closed() -> None:
    payload = _source_payload("T")
    payload["measurement"]["output_count"] = 10**400  # type: ignore[index]

    with pytest.raises(ArtifactPayloadError, match="failed closed"):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            payload,
            manifest_sha256=TRITIUM_HASH,
        )


def test_source_rejects_claim_injection_bool_counts_wrong_units_and_wrong_species() -> None:
    injected = _source_payload("T")
    injected["measurement"]["source_rate_s"] = 1.0e99  # type: ignore[index]
    boolean = _source_payload("T")
    boolean["measurement"]["output_count"] = True  # type: ignore[index]
    wrong_unit = _source_payload("T")
    wrong_unit["measurement"]["output_count_unit"] = "mol"  # type: ignore[index]
    wrong_species = _source_payload("T")
    wrong_species["measurement"]["species"] = "D"  # type: ignore[index]

    for payload in (injected, boolean, wrong_unit, wrong_species):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
                payload,
                manifest_sha256=TRITIUM_HASH,
            )


def test_source_rejects_unphysical_three_sigma_polarization_boxes() -> None:
    deuterium = _source_payload("D")
    deuterium["measurement"]["deuteron_vector_polarization_std"] = 0.3  # type: ignore[index]
    tritium = _source_payload("T")
    tritium["measurement"]["triton_vector_polarization_std"] = -0.01  # type: ignore[index]

    with pytest.raises(ArtifactPayloadError):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            deuterium,
            manifest_sha256=DEUTERIUM_HASH,
        )
    with pytest.raises(ArtifactPayloadError):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            tritium,
            manifest_sha256=TRITIUM_HASH,
        )


def test_source_rejects_background_minimum_sample_and_zero_uncertainty_attacks() -> None:
    near_cancelled = _source_payload("T")
    auxiliary = _engineering_auxiliary_payloads(near_cancelled)
    auxiliary["calibration"]["background_count"] = (
        near_cancelled["measurement"]["output_count"] - 999_999  # type: ignore[index,operator]
    )
    auxiliary["calibration"]["background_count_std"] = 1.0
    near_cancelled["measurement"]["output_count_std"] = 1.0  # type: ignore[index]
    auxiliary["event_counts"]["observed_output_count_std"] = 1.0
    with pytest.raises(ArtifactPayloadError, match="at least"):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            near_cancelled,
            manifest_sha256=TRITIUM_HASH,
            auxiliary_payloads=auxiliary,
        )

    zero_count_std = _source_payload("T")
    zero_count_std["measurement"]["output_count_std"] = 0.0  # type: ignore[index]
    zero_duration_std = _source_payload("T")
    zero_duration_auxiliary = _engineering_auxiliary_payloads(zero_duration_std)
    zero_duration_auxiliary["pipeline"]["measurement_duration_std_s"] = 0.0
    zero_polarization_std = _source_payload("D")
    zero_polarization_std["measurement"]["deuteron_tensor_polarization_std"] = 0.0  # type: ignore[index]
    cases = (
        (zero_count_std, None, REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS),
        (
            zero_duration_std,
            zero_duration_auxiliary,
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        ),
        (zero_polarization_std, None, REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS),
    )
    for payload, auxiliary_payloads, evidence_class in cases:
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                evidence_class,
                payload,
                manifest_sha256=(
                    DEUTERIUM_HASH
                    if evidence_class == REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS
                    else TRITIUM_HASH
                ),
                auxiliary_payloads=auxiliary_payloads,
            )


def test_source_rate_and_deuteron_joint_correlations_are_strict_and_recomputed() -> None:
    payload = _source_payload("D")
    auxiliary = _engineering_auxiliary_payloads(payload)
    auxiliary["pipeline"]["correlation_matrix"] = [
        [1.0, 0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result = derive_artifact_claims(
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        payload,
        manifest_sha256=DEUTERIUM_HASH,
        auxiliary_payloads=auxiliary,
    )
    claims = result.claims
    gradient = np.asarray(claims["source_rate_uncertainty_gradient"], dtype=float)
    stds = np.asarray((1.0e18, 1.0e15, 1.0e-6, 1.0e-6), dtype=float)
    correlation = np.asarray(
        claims["source_rate_uncertainty_correlation_matrix"], dtype=float
    )
    manual_std = math.sqrt(float(gradient @ (correlation * np.outer(stds, stds)) @ gradient))
    assert claims["source_rate_linearized_std_s"] == pytest.approx(manual_std)

    singular_joint = _source_payload("D")
    singular_auxiliary = _engineering_auxiliary_payloads(singular_joint)
    singular_auxiliary["pipeline"]["polarization_correlation_matrix"] = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    singular_result = derive_artifact_claims(
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        singular_joint,
        manifest_sha256=DEUTERIUM_HASH,
        auxiliary_payloads=singular_auxiliary,
    )
    assert singular_result.claims[
        "deuteron_polarization_uncertainty_covariance_determinant"
    ] == pytest.approx(0.0)

    non_psd = _source_payload("D")
    non_psd_auxiliary = _engineering_auxiliary_payloads(non_psd)
    non_psd_auxiliary["pipeline"]["correlation_matrix"] = [
        [1.0, 0.9, 0.9, 0.0],
        [0.9, 1.0, -0.9, 0.0],
        [0.9, -0.9, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ArtifactPayloadError, match="positive semidefinite"):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            non_psd,
            manifest_sha256=DEUTERIUM_HASH,
            auxiliary_payloads=non_psd_auxiliary,
        )


def test_source_exact_box_rejects_nonlinear_rank_one_cancellation_false_pass() -> None:
    payload = _source_payload("T")
    auxiliary = _engineering_auxiliary_payloads(payload)
    count = 12_000_000
    background = 1_000
    efficiency = 0.5
    nominal_rate = 2.2e22
    duration = (count - background) / efficiency / nominal_rate
    payload["measurement"]["output_count"] = count  # type: ignore[index]
    payload["measurement"]["output_count_std"] = 3_000_000.0  # type: ignore[index]
    payload["measurement"]["measurement_duration_s"] = duration  # type: ignore[index]
    auxiliary["event_counts"]["observed_output_count"] = count
    auxiliary["event_counts"]["observed_output_count_std"] = 3_000_000.0
    auxiliary["calibration"]["background_count"] = background
    auxiliary["calibration"]["background_count_std"] = 1.0
    auxiliary["calibration"]["detection_efficiency"] = efficiency
    auxiliary["calibration"]["detection_efficiency_std"] = 0.05
    auxiliary["pipeline"]["measurement_duration_s"] = duration
    auxiliary["pipeline"]["measurement_duration_std_s"] = 0.1 * duration
    auxiliary["pipeline"]["correlation_matrix"] = [
        [1.0] * 4 for _ in range(4)
    ]

    claims = derive_artifact_claims(
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        payload,
        manifest_sha256=TRITIUM_HASH,
        auxiliary_payloads=auxiliary,
    ).claims
    required = claims["required_per_species_fuel_rate_s"]
    assert claims["source_rate_s"] - 3.0 * claims[
        "source_rate_linearized_std_s"
    ] > required
    assert claims["source_rate_lower_3sigma_s"] < required
    assert claims["reactor_rate_source_demonstrated"] is False


def test_burn_retention_derives_three_sigma_bounds_and_preserves_typed_links() -> None:
    result = derive_artifact_claims(
        BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        _burn_payload(),
        manifest_sha256=BURN_HASH,
    )
    claims = result.claims
    assert result.plant_configuration_id == PLANT_ID
    assert result.linked_manifest_sha256s == (
        ("deuterium_source", DEUTERIUM_HASH),
        ("tritium_source", TRITIUM_HASH),
        ("reaction_operator", REACTION_HASH),
    )
    assert claims["burn_weighted_dt_polarization_product_lower_3sigma"] == pytest.approx(0.027)
    assert claims["burn_weighted_dt_polarization_product_upper_3sigma"] == pytest.approx(0.033)
    assert claims["burn_weighted_deuteron_tensor_polarization_lower_3sigma"] == pytest.approx(
        -0.03
    )
    assert claims["reaction_count"] == 1_000_000
    assert claims["burn_quantization_axis"] == "longitudinal_beam_axis"
    assert claims["burn_initial_state_basis"] == (
        "diagonal_in_product_spin_projection_basis"
    )
    assert claims["burn_collision_ensemble_temperature_kev"] == 10.0
    assert claims["burn_spin_state_energy_dependence"] == (
        "declared_energy_independent_diagonal_spin_state_engineering_control_v1"
    )


def test_burn_rejects_same_source_hash_bool_count_and_unphysical_uncertainty() -> None:
    same_hash = _burn_payload()
    same_hash["measurement"]["tritium_source_manifest_sha256"] = DEUTERIUM_HASH  # type: ignore[index]
    boolean = _burn_payload()
    boolean["measurement"]["reaction_count"] = True  # type: ignore[index]
    negative_std = _burn_payload()
    negative_std["measurement"]["burn_weighted_dt_polarization_product_std"] = -0.1  # type: ignore[index]
    positivity = _burn_payload()
    positivity["measurement"]["burn_weighted_dt_polarization_product"] = 0.9  # type: ignore[index]
    positivity["measurement"]["burn_weighted_deuteron_tensor_polarization"] = -1.8  # type: ignore[index]

    for payload in (same_hash, boolean, negative_std, positivity):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
                payload,
                manifest_sha256=BURN_HASH,
            )


def test_burn_requires_large_sample_duration_uncertainty_and_joint_semantics() -> None:
    too_few = _burn_payload()
    too_few["measurement"]["reaction_count"] = 999_999  # type: ignore[index]
    zero_std = _burn_payload()
    zero_std["measurement"]["burn_weighted_deuteron_tensor_polarization_std"] = 0.0  # type: ignore[index]
    excessive_duration = _burn_payload()
    excessive_auxiliary = _engineering_auxiliary_payloads(excessive_duration)
    excessive_auxiliary["burn_events"]["measurement_duration_std_s"] = 1.0
    wrong_semantics = _burn_payload()
    wrong_auxiliary = _engineering_auxiliary_payloads(wrong_semantics)
    wrong_auxiliary["estimator"]["observable_semantics"] = "time_weighted_proxy"
    wrong_axis = _burn_payload()
    wrong_axis["measurement"]["quantization_axis"] = "transverse_axis"  # type: ignore[index]
    unspecified_energy = _burn_payload()
    unspecified_energy["measurement"]["spin_state_energy_dependence"] = (  # type: ignore[index]
        "unspecified"
    )
    for payload, auxiliary_payloads in (
        (too_few, None),
        (zero_std, None),
        (excessive_duration, excessive_auxiliary),
        (wrong_semantics, wrong_auxiliary),
        (wrong_axis, None),
        (unspecified_energy, None),
    ):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
                payload,
                manifest_sha256=BURN_HASH,
                auxiliary_payloads=auxiliary_payloads,
            )

    singular = _burn_payload()
    singular_auxiliary = _engineering_auxiliary_payloads(singular)
    singular_auxiliary["estimator"]["correlation_matrix"] = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    singular_result = derive_artifact_claims(
        BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        singular,
        manifest_sha256=BURN_HASH,
        auxiliary_payloads=singular_auxiliary,
    )
    assert singular_result.claims["burn_uncertainty_covariance_determinant"] == pytest.approx(
        0.0
    )


def test_wall_ledger_sums_components_and_propagates_correlated_gradients() -> None:
    payload = _wall_payload()
    result = derive_artifact_claims(
        POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        payload,
        manifest_sha256=WALL_HASH,
    )
    claims = result.claims
    energies = [float(index) for index in range(1, 12)]
    standard_deviations = [value / 10.0 for value in energies]
    expected_total = math.fsum(energies)
    expected_std = math.sqrt(math.fsum(value * value for value in standard_deviations))
    pair_count = 20_000_000_000_000_000_000_000
    pair_count_std = 1.0e12
    expected_energy_std = math.sqrt(
        (expected_std / pair_count / 1.602176634e-19) ** 2
        + (
            expected_total
            * pair_count_std
            / (pair_count * pair_count)
            / 1.602176634e-19
        )
        ** 2
    )
    expected_flow_std = math.sqrt(
        pair_count_std**2 + (pair_count * 1.0e-6) ** 2
    )

    assert result.plant_configuration_id == PLANT_ID
    assert result.linked_manifest_sha256s == (
        ("deuterium_source", DEUTERIUM_HASH),
        ("tritium_source", TRITIUM_HASH),
    )
    assert claims["wall_plug_component_categories"] == sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    assert claims["wall_plug_total_energy_j"] == pytest.approx(expected_total)
    expected_total_lower = math.fsum(
        value - 3.0 * std
        for value, std in zip(energies, standard_deviations, strict=True)
    )
    expected_total_upper = math.fsum(
        value + 3.0 * std
        for value, std in zip(energies, standard_deviations, strict=True)
    )
    assert claims["wall_plug_total_energy_linearized_std_j"] == pytest.approx(
        expected_std
    )
    assert claims["wall_plug_total_energy_lower_3sigma_j"] == pytest.approx(
        expected_total_lower
    )
    assert claims["wall_plug_total_energy_upper_3sigma_j"] == pytest.approx(
        expected_total_upper
    )
    assert claims["wall_plug_energy_per_injected_dt_pair_ev"] == pytest.approx(
        expected_total / pair_count / 1.602176634e-19
    )
    assert claims[
        "wall_plug_energy_linearized_std_per_injected_dt_pair_ev"
    ] == pytest.approx(expected_energy_std)
    assert claims[
        "wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev"
    ] == pytest.approx(
        expected_total_lower / (pair_count + 3.0 * pair_count_std) / 1.602176634e-19
    )
    assert claims[
        "wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev"
    ] == pytest.approx(
        expected_total_upper / (pair_count - 3.0 * pair_count_std) / 1.602176634e-19
    )
    assert claims[
        "measurement_injected_dt_pair_flow_rate_linearized_std_s"
    ] == pytest.approx(expected_flow_std)
    assert claims["measurement_injected_dt_pair_flow_rate_lower_3sigma_s"] == pytest.approx(
        (pair_count - 3.0 * pair_count_std) / 1.000003
    )
    assert claims["measurement_injected_dt_pair_flow_rate_upper_3sigma_s"] == pytest.approx(
        (pair_count + 3.0 * pair_count_std) / 0.999997
    )
    assert claims["wall_plug_3sigma_bound_model"] == (
        "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
    )
    assert claims["measurement_pair_flow_meets_required_throughput"] is True


def test_wall_exact_box_rejects_rank_one_flow_cancellation_and_binds_component_boxes() -> None:
    payload = _wall_payload()
    auxiliary = _engineering_auxiliary_payloads(payload)
    required_rate = 1.773155987e22
    pair_count = int(2.0 * required_rate) + 1
    pair_count_std = 0.25 * pair_count
    duration = 1.0
    duration_std = 0.1
    payload["ledger"]["injected_dt_pair_count"] = pair_count  # type: ignore[index]
    payload["ledger"]["measurement_duration_s"] = duration  # type: ignore[index]
    auxiliary["meter_records"]["injected_dt_pair_count"] = pair_count
    auxiliary["meter_records"]["injected_dt_pair_count_std"] = pair_count_std
    auxiliary["meter_records"]["measurement_duration_s"] = duration
    auxiliary["meter_records"]["measurement_duration_std_s"] = duration_std
    auxiliary["uncertainty_pipeline"]["correlation_matrix"] = [
        [1.0] * 13 for _ in range(13)
    ]

    claims = derive_artifact_claims(
        POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
        payload,
        manifest_sha256=WALL_HASH,
        auxiliary_payloads=auxiliary,
    ).claims
    required = claims["required_per_species_fuel_rate_s"]
    assert claims["measurement_injected_dt_pair_flow_rate_s"] - 3.0 * claims[
        "measurement_injected_dt_pair_flow_rate_linearized_std_s"
    ] > required
    assert claims["measurement_injected_dt_pair_flow_rate_lower_3sigma_s"] < required
    assert claims["measurement_pair_flow_meets_required_throughput"] is False
    for category in claims["wall_plug_component_categories"]:
        energy = claims["wall_plug_component_energies_j"][category]
        std = claims["wall_plug_component_energy_stds_j"][category]
        assert claims["wall_plug_component_energy_lower_3sigma_j"][category] == (
            energy - 3.0 * std
        )
        assert claims["wall_plug_component_energy_upper_3sigma_j"][category] == (
            energy + 3.0 * std
        )


def test_wall_rejects_any_component_not_positive_over_its_three_sigma_box() -> None:
    payload = _wall_payload()
    payload["ledger"]["components"]["balance_of_plant"]["energy_std_j"] = 1.0  # type: ignore[index]
    with pytest.raises(ArtifactPayloadError, match="component balance_of_plant"):
        derive_artifact_claims(
            POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
            payload,
            manifest_sha256=WALL_HASH,
        )


def test_wall_ledger_rejects_missing_or_extra_component_and_wrong_model() -> None:
    missing = _wall_payload()
    missing["ledger"]["components"].pop("balance_of_plant")  # type: ignore[union-attr,index]
    extra = _wall_payload()
    extra["ledger"]["components"]["catch_all"] = {  # type: ignore[index]
        "energy_j": 0.0,
        "energy_std_j": 0.0,
    }
    model = _wall_payload()
    model["ledger"]["uncertainty_model"] = "fully_correlated"  # type: ignore[index]

    for payload in (missing, extra, model):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
                payload,
                manifest_sha256=WALL_HASH,
            )


def test_wall_ledger_rejects_negative_energy_bool_count_duplicate_links_and_extra_rows() -> None:
    negative = _wall_payload()
    negative["ledger"]["components"]["balance_of_plant"]["energy_j"] = -1.0  # type: ignore[index]
    boolean = _wall_payload()
    boolean["ledger"]["injected_dt_pair_count"] = True  # type: ignore[index]
    same_hash = _wall_payload()
    same_hash["ledger"]["tritium_source_manifest_sha256"] = DEUTERIUM_HASH  # type: ignore[index]
    extra_row = _wall_payload()
    extra_row["ledger"]["components"]["balance_of_plant"]["claimed_total_j"] = 0.0  # type: ignore[index]

    for payload in (negative, boolean, same_hash, extra_row):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
                payload,
                manifest_sha256=WALL_HASH,
            )


def test_wall_component_sum_overflow_fails_closed() -> None:
    payload = _wall_payload()
    for component in payload["ledger"]["components"].values():  # type: ignore[union-attr,index]
        component["energy_j"] = 1.0e308

    with pytest.raises(ArtifactPayloadError, match="overflowed"):
        derive_artifact_claims(
            POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
            payload,
            manifest_sha256=WALL_HASH,
        )


def test_wall_rejects_zero_or_unphysical_pair_duration_and_non_psd_pipeline() -> None:
    zero_energy = _wall_payload()
    zero_energy["ledger"]["components"]["balance_of_plant"]["energy_j"] = 0.0  # type: ignore[index]
    zero_std = _wall_payload()
    zero_std["ledger"]["components"]["balance_of_plant"]["energy_std_j"] = 0.0  # type: ignore[index]
    zero_pair_std = _wall_payload()
    zero_pair_auxiliary = _engineering_auxiliary_payloads(zero_pair_std)
    zero_pair_auxiliary["meter_records"]["injected_dt_pair_count_std"] = 0.0
    excessive_duration = _wall_payload()
    excessive_duration_auxiliary = _engineering_auxiliary_payloads(excessive_duration)
    excessive_duration_auxiliary["meter_records"]["measurement_duration_std_s"] = 1.0
    non_psd = _wall_payload()
    non_psd_auxiliary = _engineering_auxiliary_payloads(non_psd)
    bad_correlation = np.eye(13)
    bad_correlation[:3, :3] = np.asarray(
        ((1.0, 0.9, 0.9), (0.9, 1.0, -0.9), (0.9, -0.9, 1.0))
    )
    non_psd_auxiliary["uncertainty_pipeline"]["correlation_matrix"] = (
        bad_correlation.tolist()
    )
    non_robust_energy = _wall_payload()
    for component in non_robust_energy["ledger"]["components"].values():  # type: ignore[index,union-attr]
        component["energy_j"] = 1.0
        component["energy_std_j"] = 1.0e6
    for payload, auxiliary_payloads in (
        (zero_energy, None),
        (zero_std, None),
        (zero_pair_std, zero_pair_auxiliary),
        (excessive_duration, excessive_duration_auxiliary),
        (non_psd, non_psd_auxiliary),
        (non_robust_energy, None),
    ):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
                payload,
                manifest_sha256=WALL_HASH,
                auxiliary_payloads=auxiliary_payloads,
            )


def test_all_engineering_payloads_preserve_the_same_plant_id_for_cross_join() -> None:
    records = (
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            _source_payload("D"),
            manifest_sha256=DEUTERIUM_HASH,
        ),
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
            _source_payload("T"),
            manifest_sha256=TRITIUM_HASH,
        ),
        derive_artifact_claims(
            BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
            _burn_payload(),
            manifest_sha256=BURN_HASH,
        ),
        derive_artifact_claims(
            POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
            _wall_payload(),
            manifest_sha256=WALL_HASH,
        ),
    )
    assert {record.plant_configuration_id for record in records} == {PLANT_ID}


def test_dispatch_rejects_unknown_classes_bad_hashes_non_objects_and_class_mismatch() -> None:
    with pytest.raises(ArtifactPayloadError, match="unknown evidence_class"):
        derive_artifact_claims("unknown", {}, manifest_sha256=REACTION_HASH)
    with pytest.raises(ArtifactPayloadError, match="lowercase sha256"):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            _source_payload("D"),
            manifest_sha256="sha256:" + "A" * 64,
        )
    with pytest.raises(ArtifactPayloadError, match="plain object"):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            [],  # type: ignore[arg-type]
            manifest_sha256=DEUTERIUM_HASH,
        )
    mismatch = _source_payload("D")
    mismatch["evidence_class"] = REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS
    with pytest.raises(ArtifactPayloadError):
        derive_artifact_claims(
            REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
            mismatch,
            manifest_sha256=DEUTERIUM_HASH,
        )


def test_exact_plant_schema_rejects_bool_numbers_bad_units_and_extra_keys() -> None:
    boolean = _source_payload("D")
    boolean["plant_configuration"]["fusion_power_w"] = True  # type: ignore[index]
    unit = _source_payload("D")
    unit["plant_configuration"]["fusion_power_unit"] = "MW"  # type: ignore[index]
    extra = _source_payload("D")
    extra["plant_configuration"]["operator_note"] = "trust me"  # type: ignore[index]

    for payload in (boolean, unit, extra):
        with pytest.raises(ArtifactPayloadError):
            derive_artifact_claims(
                REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
                payload,
                manifest_sha256=DEUTERIUM_HASH,
            )


def test_fixture_builders_do_not_share_mutable_nested_state() -> None:
    first = _wall_payload()
    second = deepcopy(first)
    first["ledger"]["components"]["balance_of_plant"]["energy_j"] = 999.0  # type: ignore[index]
    assert second["ledger"]["components"]["balance_of_plant"]["energy_j"] != 999.0  # type: ignore[index]
