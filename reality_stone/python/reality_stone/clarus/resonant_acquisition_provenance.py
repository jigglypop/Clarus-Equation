"""Fail-closed acquisition-to-cluster provenance for resonant-mask audits.

Repeated measurements from one randomization unit are not independent trials.
This module binds an acquisition ledger, averages paired matched-minus-sham
responses once per declared cluster, and only then calls the frozen
spatiotemporal-mask gate.  Its highest stage remains conditional on caller
declarations: hashes and identifiers cannot authenticate a clock, a device, or
the truth of the cluster partition without an external signed record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
from numbers import Integral, Real
import re
from typing import Any, Sequence

import numpy as np

from .resonant_spatiotemporal_mask import (
    ArrayLike,
    ResonantSpatiotemporalMaskAudit,
    _block_ids,
    _bool_mask,
    _finite_real,
    _hash_text,
    _hex_digest,
    _numeric_array,
    _strict_bool,
    _strict_integer,
    resonant_mask_manifest_sha256,
    resonant_spatiotemporal_mask_audit,
    validate_resonant_spatiotemporal_mask_audit,
)


_AGGREGATION_RULE = "paired_difference_then_unweighted_cluster_mean/v1"
_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{9})Z$"
)


class ResonantAcquisitionStage(str, Enum):
    """Monotone controls; none authenticates physical provenance."""

    INPUT_VALIDATION_ONLY = "INPUT_VALIDATION_ONLY"
    DECLARED_ACQUISITION_LEDGER_CONTROL = "DECLARED_ACQUISITION_LEDGER_CONTROL"
    DECLARED_CLUSTER_PREAGGREGATION_CONTROL = (
        "DECLARED_CLUSTER_PREAGGREGATION_CONTROL"
    )
    CONDITIONAL_DECLARED_CLUSTER_RESPONSE_MASK = (
        "CONDITIONAL_DECLARED_CLUSTER_RESPONSE_MASK"
    )


@dataclass(frozen=True)
class ResonantAcquisitionClaimLocks:
    """Claims that an internally self-consistent ledger cannot establish."""

    raw_artifact_bytes_independently_verified: bool = False
    external_preregistration_timestamp_verified: bool = False
    external_ledger_signature_verified: bool = False
    independent_acquisition_provenance_verified: bool = False
    relativistic_causality_derived: bool = False
    ce_coupling_derived: bool = False
    material_phase_derived: bool = False
    new_matter_derived: bool = False
    renormalized_stress_tensor_derived: bool = False


@dataclass(frozen=True)
class ResonantAcquisitionRawInputs:
    """Canonical immutable inputs retained for complete recomputation."""

    cell_shape: tuple[int, ...]
    matched_response_flat: tuple[tuple[float, ...], ...]
    sham_response_flat: tuple[tuple[float, ...], ...]
    design_flat: tuple[float, ...]
    training_mask_flat: tuple[bool, ...]
    heldout_mask_flat: tuple[bool, ...]
    prearrival_mask_flat: tuple[bool, ...]
    off_support_mask_flat: tuple[bool, ...]
    target_mask_flat: tuple[bool, ...]
    acquisition_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    device_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    acquired_at_utc: tuple[str, ...]
    matched_raw_payload_sha256: tuple[str, ...]
    sham_raw_payload_sha256: tuple[str, ...]
    preregistration_recorded_at_utc: str
    declared_acquisition_ledger_sha256: str
    preprocessing_artifact_sha256: str
    design_calibration_artifact_sha256: str
    declared_manifest_sha256: str
    payload_hashes_recomputed_from_raw_artifacts: bool
    timestamps_from_acquisition_system_declared: bool
    cluster_mapping_frozen_before_outcome_analysis: bool
    clusters_declared_independent: bool
    cluster_means_iid_gaussian_declared: bool
    manifest_frozen_before_data: bool
    masks_fixed_before_holdout: bool
    expected_response_sign: int
    familywise_alpha: float
    equivalence_bound: float
    minimum_target_response: float
    maximum_covariance_condition_number: float
    covariance_rank_relative_tolerance: float
    minimum_paired_covariance_eigenvalue: float
    minimum_residual_mean_variance: float
    minimum_clusters: int
    minimum_pairs_per_cluster: int


@dataclass(frozen=True)
class ClusteredResonantMaskAudit:
    """Acquisition ledger, cluster reduction, and downstream mask certificate."""

    schema_version: str
    raw_inputs: ResonantAcquisitionRawInputs
    aggregation_rule: str
    raw_acquisition_count: int
    cluster_count: int
    cluster_degrees_of_freedom: int
    ordered_cluster_ids: tuple[str, ...]
    cluster_sizes: tuple[int, ...]
    minimum_cluster_size: int
    maximum_cluster_size: int
    acquisition_ids_unique: bool
    payload_hashes_unique: bool
    declared_chronology_pass: bool
    acquisition_ledger_sha256: str
    computed_acquisition_ledger_sha256: str
    acquisition_ledger_hash_matches: bool
    payload_hash_recomputation_declared: bool
    acquisition_clock_origin_declared: bool
    cluster_mapping_frozen_before_outcome_analysis: bool
    cluster_partition_complete: bool
    balanced_clusters: bool
    minimum_pairs_per_cluster_met: bool
    minimum_clusters_met: bool
    cluster_means_iid_gaussian_declared: bool
    clusters_declared_independent: bool
    exact_t_eligible: bool
    aggregated_matched_response_flat: tuple[tuple[float, ...], ...]
    aggregated_sham_response_flat: tuple[tuple[float, ...], ...]
    downstream_mask_audit: ResonantSpatiotemporalMaskAudit
    conditional_declared_cluster_response_mask: bool
    maximum_supported_stage: ResonantAcquisitionStage
    first_blocker: str
    blockers: tuple[str, ...]
    claim_locks: ResonantAcquisitionClaimLocks

    @property
    def external_acquisition_provenance_verified(self) -> bool:
        return self.claim_locks.independent_acquisition_provenance_verified

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["maximum_supported_stage"] = self.maximum_supported_stage.value
        payload["downstream_mask_audit"] = self.downstream_mask_audit.to_dict()
        payload["external_acquisition_provenance_verified"] = (
            self.external_acquisition_provenance_verified
        )
        return payload


def _canonical_timestamps(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of UTC timestamps")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    for timestamp in result:
        _timestamp_key(timestamp, name=name)
    return result


def _timestamp_key(value: str, *, name: str) -> tuple[int, ...]:
    if type(value) is not str or not value.isascii():
        raise ValueError(f"{name} must use canonical ASCII UTC timestamps")
    match = _TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"{name} must use YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ UTC format"
        )
    year, month, day, hour, minute, second, nanoseconds = (
        int(part) for part in match.groups()
    )
    try:
        datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            nanoseconds // 1000,
            tzinfo=timezone.utc,
        )
    except ValueError as error:
        raise ValueError(f"{name} contains an invalid UTC timestamp") from error
    return year, month, day, hour, minute, second, nanoseconds


def _hex_sequence(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of SHA-256 digests")
    result = tuple(_hex_digest(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _same_length(expected: int, **sequences: Sequence[object]) -> None:
    mismatched = tuple(name for name, value in sequences.items() if len(value) != expected)
    if mismatched:
        raise ValueError(
            "acquisition metadata must match response row count: " + ", ".join(mismatched)
        )


def _canonical_acquisition_inputs(
    *,
    matched_response: ArrayLike,
    sham_response: ArrayLike,
    acquisition_ids: Sequence[str],
    cluster_ids: Sequence[str],
    device_ids: Sequence[str],
    clock_ids: Sequence[str],
    acquired_at_utc: Sequence[str],
    matched_raw_payload_sha256: Sequence[str],
    sham_raw_payload_sha256: Sequence[str],
    preregistration_recorded_at_utc: str,
    payload_hashes_recomputed_from_raw_artifacts: bool,
    timestamps_from_acquisition_system_declared: bool,
    cluster_mapping_frozen_before_outcome_analysis: bool,
    clusters_declared_independent: bool,
    cluster_means_iid_gaussian_declared: bool,
    minimum_clusters: Integral,
    minimum_pairs_per_cluster: Integral,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
    bool,
    bool,
    bool,
    bool,
    bool,
    int,
    int,
]:
    matched = _numeric_array(matched_response, name="matched_response")
    sham = _numeric_array(sham_response, name="sham_response")
    if matched.shape != sham.shape or matched.ndim < 2 or matched.shape[0] < 2:
        raise ValueError(
            "matched_response and sham_response must share shape (acquisition, *cells)"
        )
    row_count = matched.shape[0]
    acquisitions = _block_ids(acquisition_ids, name="acquisition_ids")
    clusters = _block_ids(cluster_ids, name="cluster_ids")
    devices = _block_ids(device_ids, name="device_ids")
    clocks = _block_ids(clock_ids, name="clock_ids")
    timestamps = _canonical_timestamps(acquired_at_utc, name="acquired_at_utc")
    matched_hashes = _hex_sequence(
        matched_raw_payload_sha256, name="matched_raw_payload_sha256"
    )
    sham_hashes = _hex_sequence(
        sham_raw_payload_sha256, name="sham_raw_payload_sha256"
    )
    _same_length(
        row_count,
        acquisition_ids=acquisitions,
        cluster_ids=clusters,
        device_ids=devices,
        clock_ids=clocks,
        acquired_at_utc=timestamps,
        matched_raw_payload_sha256=matched_hashes,
        sham_raw_payload_sha256=sham_hashes,
    )
    preregistration = preregistration_recorded_at_utc
    _timestamp_key(preregistration, name="preregistration_recorded_at_utc")
    payloads_recomputed = _strict_bool(
        payload_hashes_recomputed_from_raw_artifacts,
        name="payload_hashes_recomputed_from_raw_artifacts",
    )
    clock_declared = _strict_bool(
        timestamps_from_acquisition_system_declared,
        name="timestamps_from_acquisition_system_declared",
    )
    mapping_frozen = _strict_bool(
        cluster_mapping_frozen_before_outcome_analysis,
        name="cluster_mapping_frozen_before_outcome_analysis",
    )
    independent = _strict_bool(
        clusters_declared_independent, name="clusters_declared_independent"
    )
    gaussian = _strict_bool(
        cluster_means_iid_gaussian_declared,
        name="cluster_means_iid_gaussian_declared",
    )
    min_clusters = _strict_integer(
        minimum_clusters, name="minimum_clusters", minimum=64
    )
    min_pairs = _strict_integer(
        minimum_pairs_per_cluster, name="minimum_pairs_per_cluster", minimum=1
    )
    return (
        matched,
        sham,
        acquisitions,
        clusters,
        devices,
        clocks,
        timestamps,
        matched_hashes,
        sham_hashes,
        preregistration,
        payloads_recomputed,
        clock_declared,
        mapping_frozen,
        independent,
        gaussian,
        min_clusters,
        min_pairs,
    )


def resonant_acquisition_ledger_sha256(
    *,
    matched_response: ArrayLike,
    sham_response: ArrayLike,
    acquisition_ids: Sequence[str],
    cluster_ids: Sequence[str],
    device_ids: Sequence[str],
    clock_ids: Sequence[str],
    acquired_at_utc: Sequence[str],
    matched_raw_payload_sha256: Sequence[str],
    sham_raw_payload_sha256: Sequence[str],
    preregistration_recorded_at_utc: str,
    payload_hashes_recomputed_from_raw_artifacts: bool,
    timestamps_from_acquisition_system_declared: bool,
    cluster_mapping_frozen_before_outcome_analysis: bool,
    clusters_declared_independent: bool,
    cluster_means_iid_gaussian_declared: bool,
    minimum_clusters: Integral = 64,
    minimum_pairs_per_cluster: Integral = 1,
) -> str:
    """Hash response rows, the declared partition, and every cluster-gate setting."""

    values = _canonical_acquisition_inputs(
        matched_response=matched_response,
        sham_response=sham_response,
        acquisition_ids=acquisition_ids,
        cluster_ids=cluster_ids,
        device_ids=device_ids,
        clock_ids=clock_ids,
        acquired_at_utc=acquired_at_utc,
        matched_raw_payload_sha256=matched_raw_payload_sha256,
        sham_raw_payload_sha256=sham_raw_payload_sha256,
        preregistration_recorded_at_utc=preregistration_recorded_at_utc,
        payload_hashes_recomputed_from_raw_artifacts=(
            payload_hashes_recomputed_from_raw_artifacts
        ),
        timestamps_from_acquisition_system_declared=(
            timestamps_from_acquisition_system_declared
        ),
        cluster_mapping_frozen_before_outcome_analysis=(
            cluster_mapping_frozen_before_outcome_analysis
        ),
        clusters_declared_independent=clusters_declared_independent,
        cluster_means_iid_gaussian_declared=cluster_means_iid_gaussian_declared,
        minimum_clusters=minimum_clusters,
        minimum_pairs_per_cluster=minimum_pairs_per_cluster,
    )
    (
        matched,
        sham,
        acquisitions,
        clusters,
        devices,
        clocks,
        timestamps,
        matched_hashes,
        sham_hashes,
        preregistration,
        payloads_recomputed,
        clock_declared,
        mapping_frozen,
        independent,
        gaussian,
        min_clusters,
        min_pairs,
    ) = values
    digest = hashlib.sha256()
    digest.update(b"resonant-acquisition-ledger/v1\0")
    _hash_text(digest, "response_shape", repr(matched.shape))
    digest.update(np.asarray(matched, dtype="<f8", order="C").tobytes())
    digest.update(np.asarray(sham, dtype="<f8", order="C").tobytes())
    for row in zip(
        acquisitions,
        clusters,
        devices,
        clocks,
        timestamps,
        matched_hashes,
        sham_hashes,
        strict=True,
    ):
        for label, value in zip(
            (
                "acquisition_id",
                "cluster_id",
                "device_id",
                "clock_id",
                "acquired_at_utc",
                "matched_raw_payload_sha256",
                "sham_raw_payload_sha256",
            ),
            row,
            strict=True,
        ):
            _hash_text(digest, label, value)
    _hash_text(digest, "preregistration_recorded_at_utc", preregistration)
    _hash_text(digest, "aggregation_rule", _AGGREGATION_RULE)
    for label, value in (
        ("payload_hashes_recomputed_from_raw_artifacts", payloads_recomputed),
        ("timestamps_from_acquisition_system_declared", clock_declared),
        ("cluster_mapping_frozen_before_outcome_analysis", mapping_frozen),
        ("clusters_declared_independent", independent),
        ("cluster_means_iid_gaussian_declared", gaussian),
        ("minimum_clusters", min_clusters),
        ("minimum_pairs_per_cluster", min_pairs),
    ):
        _hash_text(digest, label, repr(value))
    return digest.hexdigest()


def _ordered_unique_clusters(cluster_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(cluster_ids)))


def clustered_resonant_mask_manifest_sha256(
    *,
    design_tensor: ArrayLike,
    training_mask: object,
    heldout_mask: object,
    prearrival_mask: object,
    off_support_mask: object,
    target_mask: object,
    cluster_ids: Sequence[str],
    preprocessing_artifact_sha256: str,
    design_calibration_artifact_sha256: str,
    manifest_frozen_before_data: bool,
    masks_fixed_before_holdout: bool,
    clusters_declared_independent: bool,
    cluster_means_iid_gaussian_declared: bool,
    expected_response_sign: Integral = 1,
    familywise_alpha: Real = 0.05,
    equivalence_bound: Real = 0.05,
    minimum_target_response: Real = 0.5,
    maximum_covariance_condition_number: Real = 1.0e8,
    covariance_rank_relative_tolerance: Real = 1.0e-10,
    minimum_paired_covariance_eigenvalue: Real = 1.0e-12,
    minimum_residual_mean_variance: Real = 1.0e-12,
    minimum_clusters: Integral = 64,
) -> str:
    """Build the existing v5 mask manifest for one row per declared cluster."""

    clusters = _block_ids(cluster_ids, name="cluster_ids")
    ordered = _ordered_unique_clusters(clusters)
    return resonant_mask_manifest_sha256(
        design_tensor=design_tensor,
        training_mask=training_mask,
        heldout_mask=heldout_mask,
        prearrival_mask=prearrival_mask,
        off_support_mask=off_support_mask,
        target_mask=target_mask,
        matched_block_ids=ordered,
        sham_block_ids=ordered,
        preprocessing_artifact_sha256=preprocessing_artifact_sha256,
        design_calibration_artifact_sha256=design_calibration_artifact_sha256,
        manifest_frozen_before_data=manifest_frozen_before_data,
        masks_fixed_before_holdout=masks_fixed_before_holdout,
        observations_are_independent_blocks=clusters_declared_independent,
        gaussian_mean_model_declared=cluster_means_iid_gaussian_declared,
        expected_response_sign=expected_response_sign,
        familywise_alpha=familywise_alpha,
        equivalence_bound=equivalence_bound,
        minimum_target_response=minimum_target_response,
        maximum_covariance_condition_number=maximum_covariance_condition_number,
        covariance_rank_relative_tolerance=covariance_rank_relative_tolerance,
        minimum_paired_covariance_eigenvalue=minimum_paired_covariance_eigenvalue,
        minimum_residual_mean_variance=minimum_residual_mean_variance,
        minimum_trials=minimum_clusters,
    )


def _raw_arrays(
    raw: ResonantAcquisitionRawInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    shape = raw.cell_shape
    matched = np.asarray(raw.matched_response_flat, dtype=float).reshape((-1, *shape))
    sham = np.asarray(raw.sham_response_flat, dtype=float).reshape((-1, *shape))
    design = np.asarray(raw.design_flat, dtype=float).reshape(shape)
    masks = tuple(
        np.asarray(values, dtype=bool).reshape(shape)
        for values in (
            raw.training_mask_flat,
            raw.heldout_mask_flat,
            raw.prearrival_mask_flat,
            raw.off_support_mask_flat,
            raw.target_mask_flat,
        )
    )
    return matched, sham, design, masks


def _aggregate_clusters(
    matched: np.ndarray,
    sham: np.ndarray,
    cluster_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    ordered = _ordered_unique_clusters(cluster_ids)
    matched_rows = []
    sham_rows = []
    sizes = []
    cluster_array = np.asarray(cluster_ids, dtype=object)
    for cluster in ordered:
        selector = cluster_array == cluster
        sizes.append(int(np.count_nonzero(selector)))
        matched_rows.append(np.mean(matched[selector], axis=0))
        sham_rows.append(np.mean(sham[selector], axis=0))
    return (
        ordered,
        tuple(sizes),
        np.asarray(matched_rows, dtype=float),
        np.asarray(sham_rows, dtype=float),
    )


def _stage(
    *,
    ledger_pass: bool,
    exact_t_eligible: bool,
    conditional_mask: bool,
) -> ResonantAcquisitionStage:
    if not ledger_pass:
        return ResonantAcquisitionStage.INPUT_VALIDATION_ONLY
    if not exact_t_eligible:
        return ResonantAcquisitionStage.DECLARED_ACQUISITION_LEDGER_CONTROL
    if not conditional_mask:
        return ResonantAcquisitionStage.DECLARED_CLUSTER_PREAGGREGATION_CONTROL
    return ResonantAcquisitionStage.CONDITIONAL_DECLARED_CLUSTER_RESPONSE_MASK


def _build_report(raw: ResonantAcquisitionRawInputs) -> ClusteredResonantMaskAudit:
    matched, sham, design, masks = _raw_arrays(raw)
    computed_ledger = resonant_acquisition_ledger_sha256(
        matched_response=matched,
        sham_response=sham,
        acquisition_ids=raw.acquisition_ids,
        cluster_ids=raw.cluster_ids,
        device_ids=raw.device_ids,
        clock_ids=raw.clock_ids,
        acquired_at_utc=raw.acquired_at_utc,
        matched_raw_payload_sha256=raw.matched_raw_payload_sha256,
        sham_raw_payload_sha256=raw.sham_raw_payload_sha256,
        preregistration_recorded_at_utc=raw.preregistration_recorded_at_utc,
        payload_hashes_recomputed_from_raw_artifacts=(
            raw.payload_hashes_recomputed_from_raw_artifacts
        ),
        timestamps_from_acquisition_system_declared=(
            raw.timestamps_from_acquisition_system_declared
        ),
        cluster_mapping_frozen_before_outcome_analysis=(
            raw.cluster_mapping_frozen_before_outcome_analysis
        ),
        clusters_declared_independent=raw.clusters_declared_independent,
        cluster_means_iid_gaussian_declared=(
            raw.cluster_means_iid_gaussian_declared
        ),
        minimum_clusters=raw.minimum_clusters,
        minimum_pairs_per_cluster=raw.minimum_pairs_per_cluster,
    )
    ledger_matches = hmac.compare_digest(
        raw.declared_acquisition_ledger_sha256, computed_ledger
    )
    acquisitions_unique = len(set(raw.acquisition_ids)) == len(raw.acquisition_ids)
    all_payload_hashes = (*raw.matched_raw_payload_sha256, *raw.sham_raw_payload_sha256)
    payload_hashes_unique = len(set(all_payload_hashes)) == len(all_payload_hashes)
    preregistration_key = _timestamp_key(
        raw.preregistration_recorded_at_utc,
        name="preregistration_recorded_at_utc",
    )
    chronology_pass = all(
        preregistration_key < _timestamp_key(value, name="acquired_at_utc")
        for value in raw.acquired_at_utc
    )
    ordered, sizes, aggregated_matched, aggregated_sham = _aggregate_clusters(
        matched, sham, raw.cluster_ids
    )
    cluster_count = len(ordered)
    balanced = len(set(sizes)) == 1
    minimum_pairs_met = min(sizes) >= raw.minimum_pairs_per_cluster
    minimum_clusters_met = cluster_count >= raw.minimum_clusters
    partition_complete = sum(sizes) == matched.shape[0] and all(size > 0 for size in sizes)
    ledger_pass = bool(
        ledger_matches
        and acquisitions_unique
        and payload_hashes_unique
        and chronology_pass
        and raw.payload_hashes_recomputed_from_raw_artifacts
        and raw.timestamps_from_acquisition_system_declared
    )
    exact_t_eligible = bool(
        ledger_pass
        and raw.cluster_mapping_frozen_before_outcome_analysis
        and partition_complete
        and balanced
        and minimum_pairs_met
        and minimum_clusters_met
        and raw.clusters_declared_independent
        and raw.cluster_means_iid_gaussian_declared
    )
    downstream = resonant_spatiotemporal_mask_audit(
        matched_response=aggregated_matched,
        sham_response=aggregated_sham,
        design_tensor=design,
        training_mask=masks[0],
        heldout_mask=masks[1],
        prearrival_mask=masks[2],
        off_support_mask=masks[3],
        target_mask=masks[4],
        matched_block_ids=ordered,
        sham_block_ids=ordered,
        preprocessing_artifact_sha256=raw.preprocessing_artifact_sha256,
        design_calibration_artifact_sha256=(
            raw.design_calibration_artifact_sha256
        ),
        declared_manifest_sha256=raw.declared_manifest_sha256,
        manifest_frozen_before_data=raw.manifest_frozen_before_data,
        masks_fixed_before_holdout=raw.masks_fixed_before_holdout,
        observations_are_independent_blocks=raw.clusters_declared_independent,
        gaussian_mean_model_declared=raw.cluster_means_iid_gaussian_declared,
        expected_response_sign=raw.expected_response_sign,
        familywise_alpha=raw.familywise_alpha,
        equivalence_bound=raw.equivalence_bound,
        minimum_target_response=raw.minimum_target_response,
        maximum_covariance_condition_number=(
            raw.maximum_covariance_condition_number
        ),
        covariance_rank_relative_tolerance=(
            raw.covariance_rank_relative_tolerance
        ),
        minimum_paired_covariance_eigenvalue=(
            raw.minimum_paired_covariance_eigenvalue
        ),
        minimum_residual_mean_variance=raw.minimum_residual_mean_variance,
        minimum_trials=raw.minimum_clusters,
    )
    conditional = bool(
        exact_t_eligible
        and downstream.conditional_declared_block_spatiotemporal_response_mask
    )
    blockers: list[str] = []
    if not ledger_matches:
        blockers.append("acquisition ledger hash does not match canonical recomputation")
    if not acquisitions_unique:
        blockers.append("acquisition identifiers are not unique")
    if not payload_hashes_unique:
        blockers.append("raw payload digests are reused across acquisition arms")
    if not chronology_pass:
        blockers.append("declared preregistration time does not precede every acquisition")
    if not raw.payload_hashes_recomputed_from_raw_artifacts:
        blockers.append("raw artifact payload-hash recomputation was not declared")
    if not raw.timestamps_from_acquisition_system_declared:
        blockers.append("timestamps were not declared to originate from the acquisition system")
    if not raw.cluster_mapping_frozen_before_outcome_analysis:
        blockers.append("cluster mapping was not frozen before outcome analysis")
    if not partition_complete:
        blockers.append("acquisitions do not form one complete cluster partition")
    if not balanced:
        blockers.append("cluster sizes are unequal, so the current ordinary exact-t tier is invalid")
    if not minimum_pairs_met:
        blockers.append("at least one cluster has too few paired acquisitions")
    if not minimum_clusters_met:
        blockers.append("too few independent clusters for the preregistered minimum")
    if not raw.clusters_declared_independent:
        blockers.append("cluster independence was not declared")
    if not raw.cluster_means_iid_gaussian_declared:
        blockers.append("iid Gaussian cluster-mean model was not declared")
    if exact_t_eligible and not downstream.conditional_declared_block_spatiotemporal_response_mask:
        blockers.append("cluster-level tensor fails the frozen spatiotemporal-mask gate")
    if not blockers:
        blockers.append(
            "obtain external timestamp/signature verification and actual resonant raw artifacts"
        )
    return ClusteredResonantMaskAudit(
        schema_version="resonant-acquisition-cluster/v1",
        raw_inputs=raw,
        aggregation_rule=_AGGREGATION_RULE,
        raw_acquisition_count=matched.shape[0],
        cluster_count=cluster_count,
        cluster_degrees_of_freedom=cluster_count - 1,
        ordered_cluster_ids=ordered,
        cluster_sizes=sizes,
        minimum_cluster_size=min(sizes),
        maximum_cluster_size=max(sizes),
        acquisition_ids_unique=acquisitions_unique,
        payload_hashes_unique=payload_hashes_unique,
        declared_chronology_pass=chronology_pass,
        acquisition_ledger_sha256=raw.declared_acquisition_ledger_sha256,
        computed_acquisition_ledger_sha256=computed_ledger,
        acquisition_ledger_hash_matches=ledger_matches,
        payload_hash_recomputation_declared=(
            raw.payload_hashes_recomputed_from_raw_artifacts
        ),
        acquisition_clock_origin_declared=(
            raw.timestamps_from_acquisition_system_declared
        ),
        cluster_mapping_frozen_before_outcome_analysis=(
            raw.cluster_mapping_frozen_before_outcome_analysis
        ),
        cluster_partition_complete=partition_complete,
        balanced_clusters=balanced,
        minimum_pairs_per_cluster_met=minimum_pairs_met,
        minimum_clusters_met=minimum_clusters_met,
        cluster_means_iid_gaussian_declared=(
            raw.cluster_means_iid_gaussian_declared
        ),
        clusters_declared_independent=raw.clusters_declared_independent,
        exact_t_eligible=exact_t_eligible,
        aggregated_matched_response_flat=tuple(
            tuple(float(value) for value in row)
            for row in aggregated_matched.reshape(cluster_count, -1)
        ),
        aggregated_sham_response_flat=tuple(
            tuple(float(value) for value in row)
            for row in aggregated_sham.reshape(cluster_count, -1)
        ),
        downstream_mask_audit=downstream,
        conditional_declared_cluster_response_mask=conditional,
        maximum_supported_stage=_stage(
            ledger_pass=ledger_pass,
            exact_t_eligible=exact_t_eligible,
            conditional_mask=conditional,
        ),
        first_blocker=blockers[0],
        blockers=tuple(blockers),
        claim_locks=ResonantAcquisitionClaimLocks(),
    )


def _validate_raw_structure(raw: ResonantAcquisitionRawInputs) -> None:
    if type(raw) is not ResonantAcquisitionRawInputs:
        raise ValueError("raw_inputs must be exactly ResonantAcquisitionRawInputs")
    if type(raw.cell_shape) is not tuple or any(
        type(value) is not int for value in raw.cell_shape
    ):
        raise ValueError("raw cell_shape must be an immutable tuple of built-in ints")
    for name in ("matched_response_flat", "sham_response_flat"):
        rows = getattr(raw, name)
        if type(rows) is not tuple or any(
            type(row) is not tuple or any(type(value) is not float for value in row)
            for row in rows
        ):
            raise ValueError(f"raw {name} must be an immutable tuple of float tuples")
    if type(raw.design_flat) is not tuple or any(
        type(value) is not float for value in raw.design_flat
    ):
        raise ValueError("raw design_flat must be an immutable tuple of floats")
    for name in (
        "training_mask_flat",
        "heldout_mask_flat",
        "prearrival_mask_flat",
        "off_support_mask_flat",
        "target_mask_flat",
    ):
        value = getattr(raw, name)
        if type(value) is not tuple or any(type(item) is not bool for item in value):
            raise ValueError(f"raw {name} must be an immutable tuple of bools")
    for name in (
        "acquisition_ids",
        "cluster_ids",
        "device_ids",
        "clock_ids",
        "acquired_at_utc",
        "matched_raw_payload_sha256",
        "sham_raw_payload_sha256",
    ):
        value = getattr(raw, name)
        if type(value) is not tuple or any(type(item) is not str for item in value):
            raise ValueError(f"raw {name} must be an immutable tuple of strings")
    for name in (
        "preregistration_recorded_at_utc",
        "declared_acquisition_ledger_sha256",
        "preprocessing_artifact_sha256",
        "design_calibration_artifact_sha256",
        "declared_manifest_sha256",
    ):
        if type(getattr(raw, name)) is not str:
            raise ValueError(f"raw {name} must be a built-in string")
    for name in (
        "payload_hashes_recomputed_from_raw_artifacts",
        "timestamps_from_acquisition_system_declared",
        "cluster_mapping_frozen_before_outcome_analysis",
        "clusters_declared_independent",
        "cluster_means_iid_gaussian_declared",
        "manifest_frozen_before_data",
        "masks_fixed_before_holdout",
    ):
        if type(getattr(raw, name)) is not bool:
            raise ValueError(f"raw {name} must be a built-in bool")
    for name in ("expected_response_sign", "minimum_clusters", "minimum_pairs_per_cluster"):
        if type(getattr(raw, name)) is not int:
            raise ValueError(f"raw {name} must be a built-in int")
    for name in (
        "familywise_alpha",
        "equivalence_bound",
        "minimum_target_response",
        "maximum_covariance_condition_number",
        "covariance_rank_relative_tolerance",
        "minimum_paired_covariance_eigenvalue",
        "minimum_residual_mean_variance",
    ):
        if type(getattr(raw, name)) is not float:
            raise ValueError(f"raw {name} must be a built-in float")


def _validate_report_structure(report: ClusteredResonantMaskAudit) -> None:
    if type(report) is not ClusteredResonantMaskAudit:
        raise ValueError("report must be exactly ClusteredResonantMaskAudit")
    _validate_raw_structure(report.raw_inputs)
    if type(report.claim_locks) is not ResonantAcquisitionClaimLocks or any(
        type(getattr(report.claim_locks, item.name)) is not bool
        for item in fields(report.claim_locks)
    ):
        raise ValueError("claim_locks must contain only built-in bool fields")
    if type(report.downstream_mask_audit) is not ResonantSpatiotemporalMaskAudit:
        raise ValueError("downstream_mask_audit has the wrong concrete type")
    for name in (
        "raw_acquisition_count",
        "cluster_count",
        "cluster_degrees_of_freedom",
        "minimum_cluster_size",
        "maximum_cluster_size",
    ):
        if type(getattr(report, name)) is not int:
            raise ValueError(f"report {name} must be a built-in int")
    for name in (
        "acquisition_ids_unique",
        "payload_hashes_unique",
        "declared_chronology_pass",
        "acquisition_ledger_hash_matches",
        "payload_hash_recomputation_declared",
        "acquisition_clock_origin_declared",
        "cluster_mapping_frozen_before_outcome_analysis",
        "cluster_partition_complete",
        "balanced_clusters",
        "minimum_pairs_per_cluster_met",
        "minimum_clusters_met",
        "cluster_means_iid_gaussian_declared",
        "clusters_declared_independent",
        "exact_t_eligible",
        "conditional_declared_cluster_response_mask",
    ):
        if type(getattr(report, name)) is not bool:
            raise ValueError(f"report {name} must be a built-in bool")
    for name in (
        "schema_version",
        "aggregation_rule",
        "acquisition_ledger_sha256",
        "computed_acquisition_ledger_sha256",
        "first_blocker",
    ):
        if type(getattr(report, name)) is not str:
            raise ValueError(f"report {name} must be a built-in string")
    if type(report.ordered_cluster_ids) is not tuple or any(
        type(value) is not str for value in report.ordered_cluster_ids
    ):
        raise ValueError("ordered_cluster_ids must be an immutable string tuple")
    if type(report.cluster_sizes) is not tuple or any(
        type(value) is not int for value in report.cluster_sizes
    ):
        raise ValueError("cluster_sizes must be an immutable integer tuple")
    for name in (
        "aggregated_matched_response_flat",
        "aggregated_sham_response_flat",
    ):
        rows = getattr(report, name)
        if type(rows) is not tuple or any(
            type(row) is not tuple or any(type(value) is not float for value in row)
            for row in rows
        ):
            raise ValueError(f"{name} must be an immutable tuple of float tuples")
    if type(report.maximum_supported_stage) is not ResonantAcquisitionStage:
        raise ValueError("maximum_supported_stage must be ResonantAcquisitionStage")
    if type(report.blockers) is not tuple or any(
        type(value) is not str for value in report.blockers
    ):
        raise ValueError("blockers must be an immutable tuple of strings")


def clustered_resonant_mask_audit(
    *,
    matched_response: ArrayLike,
    sham_response: ArrayLike,
    design_tensor: ArrayLike,
    training_mask: object,
    heldout_mask: object,
    prearrival_mask: object,
    off_support_mask: object,
    target_mask: object,
    acquisition_ids: Sequence[str],
    cluster_ids: Sequence[str],
    device_ids: Sequence[str],
    clock_ids: Sequence[str],
    acquired_at_utc: Sequence[str],
    matched_raw_payload_sha256: Sequence[str],
    sham_raw_payload_sha256: Sequence[str],
    preregistration_recorded_at_utc: str,
    declared_acquisition_ledger_sha256: str,
    preprocessing_artifact_sha256: str,
    design_calibration_artifact_sha256: str,
    declared_manifest_sha256: str,
    payload_hashes_recomputed_from_raw_artifacts: bool,
    timestamps_from_acquisition_system_declared: bool,
    cluster_mapping_frozen_before_outcome_analysis: bool,
    clusters_declared_independent: bool,
    cluster_means_iid_gaussian_declared: bool,
    manifest_frozen_before_data: bool,
    masks_fixed_before_holdout: bool,
    expected_response_sign: Integral = 1,
    familywise_alpha: Real = 0.05,
    equivalence_bound: Real = 0.05,
    minimum_target_response: Real = 0.5,
    maximum_covariance_condition_number: Real = 1.0e8,
    covariance_rank_relative_tolerance: Real = 1.0e-10,
    minimum_paired_covariance_eigenvalue: Real = 1.0e-12,
    minimum_residual_mean_variance: Real = 1.0e-12,
    minimum_clusters: Integral = 64,
    minimum_pairs_per_cluster: Integral = 1,
) -> ClusteredResonantMaskAudit:
    """Reduce paired acquisitions to declared clusters and audit the mask."""

    values = _canonical_acquisition_inputs(
        matched_response=matched_response,
        sham_response=sham_response,
        acquisition_ids=acquisition_ids,
        cluster_ids=cluster_ids,
        device_ids=device_ids,
        clock_ids=clock_ids,
        acquired_at_utc=acquired_at_utc,
        matched_raw_payload_sha256=matched_raw_payload_sha256,
        sham_raw_payload_sha256=sham_raw_payload_sha256,
        preregistration_recorded_at_utc=preregistration_recorded_at_utc,
        payload_hashes_recomputed_from_raw_artifacts=(
            payload_hashes_recomputed_from_raw_artifacts
        ),
        timestamps_from_acquisition_system_declared=(
            timestamps_from_acquisition_system_declared
        ),
        cluster_mapping_frozen_before_outcome_analysis=(
            cluster_mapping_frozen_before_outcome_analysis
        ),
        clusters_declared_independent=clusters_declared_independent,
        cluster_means_iid_gaussian_declared=cluster_means_iid_gaussian_declared,
        minimum_clusters=minimum_clusters,
        minimum_pairs_per_cluster=minimum_pairs_per_cluster,
    )
    (
        matched,
        sham,
        acquisitions,
        clusters,
        devices,
        clocks,
        timestamps,
        matched_hashes,
        sham_hashes,
        preregistration,
        payloads_recomputed,
        clock_declared,
        mapping_frozen,
        independent,
        gaussian,
        min_clusters,
        min_pairs,
    ) = values
    design = _numeric_array(design_tensor, name="design_tensor")
    if design.ndim < 2 or design.size < 4 or matched.shape[1:] != design.shape:
        raise ValueError("responses and design_tensor must share a multi-axis cell shape")
    masks = tuple(
        _bool_mask(value, name=name, shape=design.shape)
        for name, value in (
            ("training_mask", training_mask),
            ("heldout_mask", heldout_mask),
            ("prearrival_mask", prearrival_mask),
            ("off_support_mask", off_support_mask),
            ("target_mask", target_mask),
        )
    )
    ledger_hash = _hex_digest(
        declared_acquisition_ledger_sha256,
        name="declared_acquisition_ledger_sha256",
    )
    preprocessing_hash = _hex_digest(
        preprocessing_artifact_sha256, name="preprocessing_artifact_sha256"
    )
    calibration_hash = _hex_digest(
        design_calibration_artifact_sha256,
        name="design_calibration_artifact_sha256",
    )
    mask_manifest_hash = _hex_digest(
        declared_manifest_sha256, name="declared_manifest_sha256"
    )
    frozen = _strict_bool(
        manifest_frozen_before_data, name="manifest_frozen_before_data"
    )
    fixed = _strict_bool(
        masks_fixed_before_holdout, name="masks_fixed_before_holdout"
    )
    sign = _strict_integer(expected_response_sign, name="expected_response_sign", minimum=-1)
    if sign not in {-1, 1}:
        raise ValueError("expected_response_sign must be -1 or +1")
    floats = tuple(
        _finite_real(value, name=name)
        for name, value in (
            ("familywise_alpha", familywise_alpha),
            ("equivalence_bound", equivalence_bound),
            ("minimum_target_response", minimum_target_response),
            ("maximum_covariance_condition_number", maximum_covariance_condition_number),
            ("covariance_rank_relative_tolerance", covariance_rank_relative_tolerance),
            ("minimum_paired_covariance_eigenvalue", minimum_paired_covariance_eigenvalue),
            ("minimum_residual_mean_variance", minimum_residual_mean_variance),
        )
    )
    raw = ResonantAcquisitionRawInputs(
        cell_shape=design.shape,
        matched_response_flat=tuple(
            tuple(float(value) for value in row)
            for row in matched.reshape(matched.shape[0], -1)
        ),
        sham_response_flat=tuple(
            tuple(float(value) for value in row)
            for row in sham.reshape(sham.shape[0], -1)
        ),
        design_flat=tuple(float(value) for value in design.reshape(-1)),
        training_mask_flat=tuple(bool(value) for value in masks[0].reshape(-1)),
        heldout_mask_flat=tuple(bool(value) for value in masks[1].reshape(-1)),
        prearrival_mask_flat=tuple(bool(value) for value in masks[2].reshape(-1)),
        off_support_mask_flat=tuple(bool(value) for value in masks[3].reshape(-1)),
        target_mask_flat=tuple(bool(value) for value in masks[4].reshape(-1)),
        acquisition_ids=acquisitions,
        cluster_ids=clusters,
        device_ids=devices,
        clock_ids=clocks,
        acquired_at_utc=timestamps,
        matched_raw_payload_sha256=matched_hashes,
        sham_raw_payload_sha256=sham_hashes,
        preregistration_recorded_at_utc=preregistration,
        declared_acquisition_ledger_sha256=ledger_hash,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        declared_manifest_sha256=mask_manifest_hash,
        payload_hashes_recomputed_from_raw_artifacts=payloads_recomputed,
        timestamps_from_acquisition_system_declared=clock_declared,
        cluster_mapping_frozen_before_outcome_analysis=mapping_frozen,
        clusters_declared_independent=independent,
        cluster_means_iid_gaussian_declared=gaussian,
        manifest_frozen_before_data=frozen,
        masks_fixed_before_holdout=fixed,
        expected_response_sign=sign,
        familywise_alpha=floats[0],
        equivalence_bound=floats[1],
        minimum_target_response=floats[2],
        maximum_covariance_condition_number=floats[3],
        covariance_rank_relative_tolerance=floats[4],
        minimum_paired_covariance_eigenvalue=floats[5],
        minimum_residual_mean_variance=floats[6],
        minimum_clusters=min_clusters,
        minimum_pairs_per_cluster=min_pairs,
    )
    return validate_clustered_resonant_mask_audit(_build_report(raw))


def validate_clustered_resonant_mask_audit(
    report: ClusteredResonantMaskAudit,
) -> ClusteredResonantMaskAudit:
    """Recompute the full certificate and reject field-level tampering."""

    _validate_report_structure(report)
    if any(asdict(report.claim_locks).values()):
        raise ValueError("acquisition physical/provenance claim locks must remain false")
    validate_resonant_spatiotemporal_mask_audit(report.downstream_mask_audit)
    expected = _build_report(report.raw_inputs)
    if report != expected:
        mismatches = tuple(
            item.name
            for item in fields(report)
            if getattr(report, item.name) != getattr(expected, item.name)
        )
        detail = ", ".join(mismatches[:4]) or "unknown field"
        raise ValueError(
            "clustered resonant-mask report differs from canonical recomputation: "
            + detail
        )
    return report


__all__ = [
    "ClusteredResonantMaskAudit",
    "ResonantAcquisitionClaimLocks",
    "ResonantAcquisitionRawInputs",
    "ResonantAcquisitionStage",
    "clustered_resonant_mask_audit",
    "clustered_resonant_mask_manifest_sha256",
    "resonant_acquisition_ledger_sha256",
    "validate_clustered_resonant_mask_audit",
]
