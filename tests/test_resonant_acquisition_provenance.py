from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import numpy as np
import pytest

from reality_stone.clarus.resonant_acquisition_provenance import (
    ResonantAcquisitionStage,
    clustered_resonant_mask_audit,
    clustered_resonant_mask_manifest_sha256,
    resonant_acquisition_ledger_sha256,
    validate_clustered_resonant_mask_audit,
)


DESIGN = np.array(
    [
        [1.0, 0.8, 1.2, 0.9],
        [1.1, 0.0, 0.0, 1.3],
        [0.7, 0.0, 0.0, 0.0],
    ]
)
TRAINING = np.zeros(DESIGN.shape, dtype=bool)
TRAINING.flat[[0, 1, 2, 3, 4, 7]] = True
HELDOUT = ~TRAINING
PREARRIVAL = np.zeros(DESIGN.shape, dtype=bool)
PREARRIVAL.flat[[6, 11]] = True
OFF_SUPPORT = np.zeros(DESIGN.shape, dtype=bool)
OFF_SUPPORT.flat[[5, 9, 10]] = True
TARGET = np.zeros(DESIGN.shape, dtype=bool)
TARGET.flat[8] = True
PREPROCESSING_HASH = hashlib.sha256(b"synthetic cluster aggregation v1").hexdigest()
CALIBRATION_HASH = hashlib.sha256(b"synthetic frozen cluster design v1").hexdigest()

CONFIG = {
    "clusters_declared_independent": True,
    "cluster_means_iid_gaussian_declared": True,
    "expected_response_sign": 1,
    "familywise_alpha": 0.05,
    "equivalence_bound": 0.05,
    "minimum_target_response": 0.5,
    "maximum_covariance_condition_number": 1.0e8,
    "covariance_rank_relative_tolerance": 1.0e-10,
    "minimum_paired_covariance_eigenvalue": 1.0e-10,
    "minimum_residual_mean_variance": 1.0e-12,
    "minimum_clusters": 64,
}


def _timestamp(index: int) -> str:
    value = datetime(2026, 1, 2, tzinfo=timezone.utc) + timedelta(seconds=index)
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _payload_hash(row: np.ndarray, arm: str) -> str:
    digest = hashlib.sha256()
    digest.update(arm.encode("ascii"))
    digest.update(np.asarray(row, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()


def _responses(
    cluster_sizes: tuple[int, ...] = (2,) * 64,
    *,
    seed: int = 810,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    cluster_noise = rng.normal(0.0, 0.018, size=(len(cluster_sizes), *DESIGN.shape))
    matched_rows = []
    sham_rows = []
    cluster_ids = []
    for cluster_index, cluster_size in enumerate(cluster_sizes):
        for _ in range(cluster_size):
            baseline = rng.normal(3.0, 0.1, size=DESIGN.shape)
            repeat_noise = rng.normal(0.0, 0.004, size=DESIGN.shape)
            sham_rows.append(baseline)
            matched_rows.append(
                baseline + 2.0 * DESIGN + cluster_noise[cluster_index] + repeat_noise
            )
            cluster_ids.append(f"cluster-{cluster_index:04d}")
    return np.asarray(matched_rows), np.asarray(sham_rows), tuple(cluster_ids)


def _metadata(
    matched: np.ndarray,
    sham: np.ndarray,
) -> dict[str, tuple[str, ...] | str | bool]:
    count = matched.shape[0]
    return {
        "acquisition_ids": tuple(f"acquisition-{index:05d}" for index in range(count)),
        "device_ids": ("device-A",) * count,
        "clock_ids": ("clock-A",) * count,
        "acquired_at_utc": tuple(_timestamp(index) for index in range(count)),
        "matched_raw_payload_sha256": tuple(
            _payload_hash(row, f"matched-{index}") for index, row in enumerate(matched)
        ),
        "sham_raw_payload_sha256": tuple(
            _payload_hash(row, f"sham-{index}") for index, row in enumerate(sham)
        ),
        "preregistration_recorded_at_utc": "2026-01-01T00:00:00.000000000Z",
        "payload_hashes_recomputed_from_raw_artifacts": True,
        "timestamps_from_acquisition_system_declared": True,
        "cluster_mapping_frozen_before_outcome_analysis": True,
    }


def _audit(
    *,
    cluster_sizes: tuple[int, ...] = (2,) * 64,
    seed: int = 810,
    cluster_ids_override: tuple[str, ...] | None = None,
    declared_ledger_override: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
    config_overrides: dict[str, object] | None = None,
):
    matched, sham, cluster_ids = _responses(cluster_sizes, seed=seed)
    if cluster_ids_override is not None:
        cluster_ids = cluster_ids_override
    metadata = {**_metadata(matched, sham), **(metadata_overrides or {})}
    config = {**CONFIG, **(config_overrides or {})}
    ledger = resonant_acquisition_ledger_sha256(
        matched_response=matched,
        sham_response=sham,
        cluster_ids=cluster_ids,
        minimum_pairs_per_cluster=2,
        **metadata,
        **{
            key: config[key]
            for key in (
                "clusters_declared_independent",
                "cluster_means_iid_gaussian_declared",
                "minimum_clusters",
            )
        },
    )
    manifest = clustered_resonant_mask_manifest_sha256(
        design_tensor=DESIGN,
        training_mask=TRAINING,
        heldout_mask=HELDOUT,
        prearrival_mask=PREARRIVAL,
        off_support_mask=OFF_SUPPORT,
        target_mask=TARGET,
        cluster_ids=cluster_ids,
        preprocessing_artifact_sha256=PREPROCESSING_HASH,
        design_calibration_artifact_sha256=CALIBRATION_HASH,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        **config,
    )
    return clustered_resonant_mask_audit(
        matched_response=matched,
        sham_response=sham,
        design_tensor=DESIGN,
        training_mask=TRAINING,
        heldout_mask=HELDOUT,
        prearrival_mask=PREARRIVAL,
        off_support_mask=OFF_SUPPORT,
        target_mask=TARGET,
        cluster_ids=cluster_ids,
        declared_acquisition_ledger_sha256=declared_ledger_override or ledger,
        preprocessing_artifact_sha256=PREPROCESSING_HASH,
        design_calibration_artifact_sha256=CALIBRATION_HASH,
        declared_manifest_sha256=manifest,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        minimum_pairs_per_cluster=2,
        **metadata,
        **config,
    )


def test_balanced_repeats_reduce_to_cluster_count_before_exact_t() -> None:
    report = _audit()

    assert report.raw_acquisition_count == 128
    assert report.cluster_count == 64
    assert report.cluster_degrees_of_freedom == 63
    assert report.cluster_sizes == (2,) * 64
    assert report.balanced_clusters
    assert report.exact_t_eligible
    assert report.downstream_mask_audit.trial_count == 64
    assert report.conditional_declared_cluster_response_mask
    assert report.maximum_supported_stage is (
        ResonantAcquisitionStage.CONDITIONAL_DECLARED_CLUSTER_RESPONSE_MASK
    )
    assert report.first_blocker.startswith("obtain external timestamp")
    assert not report.external_acquisition_provenance_verified
    assert not any(vars(report.claim_locks).values())


def test_cluster_mean_is_computed_once_per_declared_randomization_unit() -> None:
    report = _audit()
    raw = report.raw_inputs
    matched = np.asarray(raw.matched_response_flat).reshape((128, *DESIGN.shape))
    sham = np.asarray(raw.sham_response_flat).reshape((128, *DESIGN.shape))
    aggregated_matched = np.asarray(report.aggregated_matched_response_flat).reshape(
        (64, *DESIGN.shape)
    )
    aggregated_sham = np.asarray(report.aggregated_sham_response_flat).reshape(
        (64, *DESIGN.shape)
    )

    assert np.allclose(aggregated_matched[0], np.mean(matched[:2], axis=0))
    assert np.allclose(aggregated_sham[0], np.mean(sham[:2], axis=0))
    assert np.allclose(
        aggregated_matched[0] - aggregated_sham[0],
        np.mean(matched[:2] - sham[:2], axis=0),
    )


def test_260_rows_from_20_clusters_remain_n20_and_fail_minimum_clusters() -> None:
    report = _audit(cluster_sizes=(13,) * 20)

    assert report.raw_acquisition_count == 260
    assert report.cluster_count == 20
    assert report.cluster_degrees_of_freedom == 19
    assert not report.minimum_clusters_met
    assert not report.exact_t_eligible
    assert not report.conditional_declared_cluster_response_mask
    assert report.maximum_supported_stage is (
        ResonantAcquisitionStage.DECLARED_ACQUISITION_LEDGER_CONTROL
    )


def test_unequal_cluster_sizes_are_not_exact_t_eligible() -> None:
    report = _audit(cluster_sizes=(2,) * 63 + (3,))

    assert report.cluster_count == 64
    assert report.minimum_clusters_met
    assert not report.balanced_clusters
    assert not report.exact_t_eligible
    assert "cluster sizes are unequal" in report.first_blocker


def test_cluster_mapping_change_breaks_the_bound_acquisition_ledger() -> None:
    matched, sham, original_ids = _responses()
    metadata = _metadata(matched, sham)
    original_ledger = resonant_acquisition_ledger_sha256(
        matched_response=matched,
        sham_response=sham,
        cluster_ids=original_ids,
        minimum_pairs_per_cluster=2,
        clusters_declared_independent=True,
        cluster_means_iid_gaussian_declared=True,
        minimum_clusters=64,
        **metadata,
    )
    changed_ids = list(original_ids)
    changed_ids[1] = changed_ids[2]
    report = _audit(
        cluster_ids_override=tuple(changed_ids),
        declared_ledger_override=original_ledger,
    )

    assert not report.acquisition_ledger_hash_matches
    assert not report.exact_t_eligible
    assert report.maximum_supported_stage is ResonantAcquisitionStage.INPUT_VALIDATION_ONLY


def test_reused_payload_hash_and_postdated_preregistration_fail_ledger_control() -> None:
    matched, sham, _ = _responses()
    metadata = _metadata(matched, sham)
    hashes = metadata["matched_raw_payload_sha256"]
    assert isinstance(hashes, tuple)
    duplicated = (hashes[0], hashes[0], *hashes[2:])
    report = _audit(
        metadata_overrides={
            "matched_raw_payload_sha256": duplicated,
            "preregistration_recorded_at_utc": "2026-02-01T00:00:00.000000000Z",
        }
    )

    assert not report.payload_hashes_unique
    assert not report.declared_chronology_pass
    assert not report.exact_t_eligible
    assert report.maximum_supported_stage is ResonantAcquisitionStage.INPUT_VALIDATION_ONLY


def test_fresh_cluster_ids_can_only_pass_a_declared_not_verified_tier() -> None:
    matched, sham, _ = _responses(cluster_sizes=(1,) * 128, seed=991)
    fake_ids = tuple(f"fresh-cluster-{index:04d}" for index in range(128))
    metadata = _metadata(matched, sham)
    config = {**CONFIG, "minimum_clusters": 64}
    ledger = resonant_acquisition_ledger_sha256(
        matched_response=matched,
        sham_response=sham,
        cluster_ids=fake_ids,
        minimum_pairs_per_cluster=1,
        clusters_declared_independent=True,
        cluster_means_iid_gaussian_declared=True,
        minimum_clusters=64,
        **metadata,
    )
    manifest = clustered_resonant_mask_manifest_sha256(
        design_tensor=DESIGN,
        training_mask=TRAINING,
        heldout_mask=HELDOUT,
        prearrival_mask=PREARRIVAL,
        off_support_mask=OFF_SUPPORT,
        target_mask=TARGET,
        cluster_ids=fake_ids,
        preprocessing_artifact_sha256=PREPROCESSING_HASH,
        design_calibration_artifact_sha256=CALIBRATION_HASH,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        **config,
    )
    report = clustered_resonant_mask_audit(
        matched_response=matched,
        sham_response=sham,
        design_tensor=DESIGN,
        training_mask=TRAINING,
        heldout_mask=HELDOUT,
        prearrival_mask=PREARRIVAL,
        off_support_mask=OFF_SUPPORT,
        target_mask=TARGET,
        cluster_ids=fake_ids,
        declared_acquisition_ledger_sha256=ledger,
        preprocessing_artifact_sha256=PREPROCESSING_HASH,
        design_calibration_artifact_sha256=CALIBRATION_HASH,
        declared_manifest_sha256=manifest,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        minimum_pairs_per_cluster=1,
        **metadata,
        **config,
    )

    assert report.exact_t_eligible
    assert report.conditional_declared_cluster_response_mask
    assert not report.external_acquisition_provenance_verified
    assert not report.claim_locks.external_ledger_signature_verified


def test_validator_recomputes_report_and_rejects_type_confusion() -> None:
    report = _audit()

    with pytest.raises(ValueError, match="exact_t_eligible must be a built-in bool"):
        validate_clustered_resonant_mask_audit(replace(report, exact_t_eligible=1))
    with pytest.raises(ValueError, match="canonical recomputation"):
        validate_clustered_resonant_mask_audit(
            replace(report, cluster_degrees_of_freedom=62)
        )
    with pytest.raises(ValueError, match="claim locks must remain false"):
        validate_clustered_resonant_mask_audit(
            replace(
                report,
                claim_locks=replace(
                    report.claim_locks,
                    independent_acquisition_provenance_verified=True,
                ),
            )
        )


def test_timestamp_and_integer_inputs_are_strict_and_canonical() -> None:
    matched, sham, cluster_ids = _responses()
    metadata = _metadata(matched, sham)

    with pytest.raises(ValueError, match="nnnnnnnnnZ"):
        resonant_acquisition_ledger_sha256(
            matched_response=matched,
            sham_response=sham,
            cluster_ids=cluster_ids,
            minimum_pairs_per_cluster=2,
            clusters_declared_independent=True,
            cluster_means_iid_gaussian_declared=True,
            minimum_clusters=64,
            **{**metadata, "acquired_at_utc": ("2026-01-02T00:00:00Z",) * 128},
        )
    with pytest.raises(ValueError, match="minimum_pairs_per_cluster must be an integer"):
        resonant_acquisition_ledger_sha256(
            matched_response=matched,
            sham_response=sham,
            cluster_ids=cluster_ids,
            minimum_pairs_per_cluster=True,
            clusters_declared_independent=True,
            cluster_means_iid_gaussian_declared=True,
            minimum_clusters=64,
            **metadata,
        )
