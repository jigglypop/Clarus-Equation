"""Synthetic acquisition-to-cluster preaggregation control.

The 128 acquisition rows below are deliberately only 64 statistical rows:
two paired repeats are averaged inside each declared cluster before the frozen
spatiotemporal mask is tested.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib

import numpy as np

from reality_stone.clarus.resonant_acquisition_provenance import (
    clustered_resonant_mask_audit,
    clustered_resonant_mask_manifest_sha256,
    resonant_acquisition_ledger_sha256,
)


def _sha256_array(row: np.ndarray) -> str:
    canonical = np.asarray(row, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _timestamp(index: int) -> str:
    value = datetime(2026, 1, 2, tzinfo=timezone.utc) + timedelta(seconds=index)
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def clustered_control():
    design = np.array(
        [
            [1.0, 0.8, 1.2, 0.9],
            [1.1, 0.0, 0.0, 1.3],
            [0.7, 0.0, 0.0, 0.0],
        ]
    )
    training = np.zeros(design.shape, dtype=bool)
    training.flat[[0, 1, 2, 3, 4, 7]] = True
    heldout = ~training
    prearrival = np.zeros(design.shape, dtype=bool)
    prearrival.flat[[6, 11]] = True
    off_support = np.zeros(design.shape, dtype=bool)
    off_support.flat[[5, 9, 10]] = True
    target = np.zeros(design.shape, dtype=bool)
    target.flat[8] = True

    cluster_count = 64
    repeats_per_cluster = 2
    acquisition_count = cluster_count * repeats_per_cluster
    acquisition_ids = tuple(
        f"synthetic-acquisition-{index:04d}" for index in range(acquisition_count)
    )
    cluster_ids = tuple(
        f"synthetic-cluster-{index // repeats_per_cluster:04d}"
        for index in range(acquisition_count)
    )
    device_ids = ("synthetic-device-A",) * acquisition_count
    clock_ids = ("synthetic-clock-A",) * acquisition_count
    acquired_at_utc = tuple(_timestamp(index) for index in range(acquisition_count))
    preregistration_recorded_at_utc = "2026-01-01T00:00:00.000000000Z"

    rng = np.random.default_rng(905)
    shared_baseline = rng.normal(
        3.0,
        0.1,
        size=(acquisition_count, *design.shape),
    )
    paired_noise = rng.normal(
        0.0,
        0.02,
        size=(acquisition_count, *design.shape),
    )
    matched = shared_baseline + 2.0 * design + paired_noise
    sham = shared_baseline
    matched_payload_hashes = tuple(_sha256_array(row) for row in matched)
    sham_payload_hashes = tuple(_sha256_array(row) for row in sham)

    preprocessing_hash = hashlib.sha256(
        b"synthetic cluster paired subtraction v1"
    ).hexdigest()
    calibration_hash = hashlib.sha256(
        b"synthetic cluster mask calibration v1"
    ).hexdigest()
    ledger_sha256 = resonant_acquisition_ledger_sha256(
        matched_response=matched,
        sham_response=sham,
        acquisition_ids=acquisition_ids,
        cluster_ids=cluster_ids,
        device_ids=device_ids,
        clock_ids=clock_ids,
        acquired_at_utc=acquired_at_utc,
        matched_raw_payload_sha256=matched_payload_hashes,
        sham_raw_payload_sha256=sham_payload_hashes,
        preregistration_recorded_at_utc=preregistration_recorded_at_utc,
        payload_hashes_recomputed_from_raw_artifacts=True,
        timestamps_from_acquisition_system_declared=True,
        cluster_mapping_frozen_before_outcome_analysis=True,
        clusters_declared_independent=True,
        cluster_means_iid_gaussian_declared=True,
        minimum_clusters=64,
        minimum_pairs_per_cluster=2,
    )

    config = {
        "clusters_declared_independent": True,
        "cluster_means_iid_gaussian_declared": True,
        "expected_response_sign": 1,
        "familywise_alpha": 0.05,
        "equivalence_bound": 0.05,
        "minimum_target_response": 0.5,
        "maximum_covariance_condition_number": 1.0e8,
        "covariance_rank_relative_tolerance": 1.0e-10,
        "minimum_paired_covariance_eigenvalue": 1.0e-8,
        "minimum_residual_mean_variance": 1.0e-10,
        "minimum_clusters": 64,
    }
    manifest_sha256 = clustered_resonant_mask_manifest_sha256(
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        cluster_ids=cluster_ids,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        **config,
    )

    return clustered_resonant_mask_audit(
        matched_response=matched,
        sham_response=sham,
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        acquisition_ids=acquisition_ids,
        cluster_ids=cluster_ids,
        device_ids=device_ids,
        clock_ids=clock_ids,
        acquired_at_utc=acquired_at_utc,
        matched_raw_payload_sha256=matched_payload_hashes,
        sham_raw_payload_sha256=sham_payload_hashes,
        preregistration_recorded_at_utc=preregistration_recorded_at_utc,
        declared_acquisition_ledger_sha256=ledger_sha256,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        declared_manifest_sha256=manifest_sha256,
        payload_hashes_recomputed_from_raw_artifacts=True,
        timestamps_from_acquisition_system_declared=True,
        cluster_mapping_frozen_before_outcome_analysis=True,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        minimum_pairs_per_cluster=2,
        **config,
    )


def main() -> None:
    report = clustered_control()
    print("synthetic balanced acquisition-to-cluster control")
    print(f"  raw acquisition rows       {report.raw_acquisition_count}")
    print(f"  independent cluster rows   {report.cluster_count}")
    print(f"  exact-t degrees of freedom {report.cluster_degrees_of_freedom}")
    print(f"  balanced clusters          {report.balanced_clusters}")
    print(f"  maximum supported stage    {report.maximum_supported_stage.value}")
    print(
        "  external provenance       "
        f"{report.external_acquisition_provenance_verified} (claim locked)"
    )


if __name__ == "__main__":
    main()
