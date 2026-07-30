from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus.tafazoli_processed_audit import (
    PROCESSED_FIGURE_AUDIT_SCOPE,
    SCHEMA_VERSION,
    DynamicCorrelationSpec,
    RequiredFile,
    audit_required_files,
    load_tafazoli_processed_audit_manifest,
    run_tafazoli_processed_audit,
    summarize_decoder_curve,
    summarize_dynamic_correlation,
)


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "tafazoli_processed_audit_v1.json"
)


@pytest.fixture
def manifest():
    return load_tafazoli_processed_audit_manifest(MANIFEST)


def test_manifest_is_narrow_and_all_scientific_claims_are_locked(
    manifest,
) -> None:
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.scope == PROCESSED_FIGURE_AUDIT_SCOPE
    assert len(manifest.decoder_curves) == 9
    assert len(manifest.dynamic_correlations) == 4
    assert not manifest.claim_locks.raw_trial_or_spike_reanalysis
    assert not manifest.claim_locks.classifier_refit
    assert not manifest.claim_locks.independent_replication
    assert not manifest.claim_locks.independent_statistical_reanalysis
    assert not manifest.claim_locks.transfer_entropy_validated
    assert not manifest.claim_locks.causal_information_flow_validated
    assert not manifest.claim_locks.unseen_composition_tested
    assert not manifest.claim_locks.neural_clarus_assembly_validated
    assert not manifest.claim_locks.causal_instruction_set_validated
    assert not manifest.claim_locks.full_brain_language_identified


def test_decoder_summary_reports_descriptive_values_not_fake_replicates(
    manifest,
) -> None:
    spec = manifest.decoder_curves[0]
    time = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])
    values = np.asarray(
        [
            [0.50, 0.55, 0.60, 0.75, 0.90],
            [0.50, 0.55, 0.65, 0.80, 0.85],
            [0.50, 0.55, 0.70, 0.85, 0.80],
        ]
    )

    audit = summarize_decoder_curve(
        spec,
        time,
        values,
        expected_resample_count=3,
        expected_timepoint_count=5,
        moving_mean_width=3,
        author_cluster_indices_one_based=np.asarray([3, 4, 5]),
        author_cluster_reported_p=np.asarray([0.02, 0.01, 0.01]),
    )

    assert audit.classifier_resample_count == 3
    assert audit.timepoint_count == 5
    assert audit.raw_peak_accuracy == pytest.approx(0.85)
    assert audit.raw_peak_time_seconds == pytest.approx(0.2)
    assert audit.raw_post_event_mean_accuracy == pytest.approx(0.7666666667)
    assert audit.plotted_smoothed_peak_accuracy == pytest.approx(0.825)
    assert audit.plotted_smoothed_peak_time_seconds == pytest.approx(0.2)
    assert audit.plotted_smoothed_post_event_mean_accuracy == pytest.approx(
        0.7527777778
    )
    assert audit.author_cluster_index_start_one_based == 3
    assert audit.author_cluster_index_end_one_based == 5
    assert audit.author_cluster_time_start_seconds == pytest.approx(0.0)
    assert audit.author_cluster_time_end_seconds == pytest.approx(0.2)
    assert audit.author_cluster_minimum_reported_p == pytest.approx(0.01)


@pytest.mark.parametrize(
    "bad_values",
    [
        np.zeros((2, 5)),
        np.full((3, 5), np.nan),
        np.full((3, 5), 1.1),
    ],
)
def test_decoder_summary_rejects_wrong_shape_or_invalid_accuracy(
    manifest,
    bad_values: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        summarize_decoder_curve(
            manifest.decoder_curves[0],
            np.linspace(-0.2, 0.2, 5),
            bad_values,
            expected_resample_count=3,
            expected_timepoint_count=5,
            moving_mean_width=3,
            author_cluster_indices_one_based=[],
            author_cluster_reported_p=[],
        )


def test_dynamic_summary_uses_embedded_window_and_positive_diagonals() -> None:
    spec = DynamicCorrelationSpec(
        name="test_dynamic",
        figure_panel="3",
        correlation_variable="a",
        pvalue_variable="p",
    )
    time = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])
    correlation = np.eye(5, dtype=np.float64) * 0.1
    correlation += np.diag(np.full(4, 0.05), k=1)
    p_values = np.full((5, 5), 0.5)
    p_values[1:4, 1:4] = 0.0005

    audit = summarize_dynamic_correlation(
        spec,
        time,
        correlation,
        p_values,
        expected_timepoint_count=5,
        declared_window_start_seconds=-0.1,
        declared_window_end_seconds=0.1,
        bin_shift_seconds=0.1,
    )

    assert audit.grid_window_timepoint_count == 3
    assert audit.grid_window_start_seconds == pytest.approx(-0.1)
    assert audit.grid_window_end_seconds == pytest.approx(0.1)
    assert audit.pointwise_p_below_0_05_fraction == pytest.approx(1.0)
    assert audit.pointwise_p_below_0_001_fraction == pytest.approx(1.0)
    assert audit.positive_diagonal_projection_peak_lag_seconds == pytest.approx(
        0.0
    )
    assert audit.zero_lag_positive_diagonal_mean == pytest.approx(0.1)


def test_required_file_audit_is_checksum_locked(tmp_path: Path) -> None:
    payload = b"processed-neural-fixture"
    path = tmp_path / "fixture.mat"
    path.write_bytes(payload)
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    audit = audit_required_files(
        tmp_path,
        (RequiredFile(filename=path.name, md5=expected),),
    )[0]

    assert audit.checksum_matches
    assert audit.byte_count == len(payload)
    assert audit.observed_md5 == expected


def test_loader_rejects_unlocking_any_scientific_claim(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["claim_locks"]["full_brain_language_identified"] = True
    path = tmp_path / "unlocked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="claim lock"):
        load_tafazoli_processed_audit_manifest(path)


def test_loader_rejects_unknown_mapping_key(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["decoder_curves"][0]["oracle_truth"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_tafazoli_processed_audit_manifest(path)


def test_real_runner_rejects_wrong_manifest_container_before_scipy() -> None:
    with pytest.raises(
        TypeError,
        match="TafazoliProcessedAuditManifest",
    ):
        run_tafazoli_processed_audit({}, ".")  # type: ignore[arg-type]
