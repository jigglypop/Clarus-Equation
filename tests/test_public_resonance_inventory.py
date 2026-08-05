from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from reality_stone.clarus.public_resonance_inventory import (
    PublicResonanceInventoryError,
    audit_public_resonance_inventory,
    verify_public_resonance_inventory,
)


def _digests(content: bytes) -> tuple[str, str]:
    return hashlib.md5(content).hexdigest(), hashlib.sha256(content).hexdigest()


def _manifest(content: bytes) -> dict[str, object]:
    source_md5, local_sha256 = _digests(content)
    return {
        "schema": "clarus.public_resonance_source_manifest.v1",
        "data_root": "data/public_resonance",
        "raw_files_committed": False,
        "claim_scope": {
            "public_measurements_are_not_clarus_specific_evidence": True,
            "new_matter_created": False,
            "clarus_field_detected": False,
        },
        "datasets": [
            {
                "dataset_id": "selected-public-record",
                "status": "selected_files_verified",
                "gate_readiness": {"clarus_specific_evidence": False},
                "files": [
                    {
                        "relative_path": "selected/source.bin",
                        "source_url": "https://example.org/source.bin",
                        "bytes": len(content),
                        "source_checksum": {
                            "algorithm": "md5",
                            "digest": source_md5,
                        },
                        "local_sha256": local_sha256,
                        "downloaded": True,
                    }
                ],
            },
            {
                "dataset_id": "metadata-record",
                "status": "metadata_only",
                "gate_readiness": {"clarus_specific_evidence": False},
                "files": [
                    {
                        "relative_path": "not-downloaded/large-source.bin",
                        "source_url": "https://example.org/large-source.bin",
                        "bytes": 1_000_000,
                        "source_checksum": {
                            "algorithm": "sha256",
                            "digest": "1" * 64,
                        },
                        "local_sha256": "1" * 64,
                        "downloaded": False,
                    }
                ],
            },
        ],
    }


@pytest.fixture
def inventory(tmp_path: Path) -> tuple[Path, bytes, dict[str, object]]:
    content = b"immutable public resonance source\n"
    source = tmp_path / "data" / "public_resonance" / "selected" / "source.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    return source, content, _manifest(content)


def test_downloaded_file_is_verified_and_metadata_only_file_may_be_absent(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    _, _, manifest = inventory

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert report.valid
    assert report.claim_lock_valid
    assert not report.clarus_specific_evidence
    assert report.downloaded_file_count == 1
    assert report.verified_file_count == 1
    assert report.analysis_ready_dataset_count == 1
    assert report.metadata_only_dataset_count == 1
    assert report.all_downloaded_files_verified
    report.assert_valid()


def test_same_size_content_tamper_breaks_both_checksum_locks(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    source, content, manifest = inventory
    source.write_bytes(content[:-2] + b"x\n")

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert not report.all_downloaded_files_verified
    assert any("source_checksum mismatch" in error for error in report.errors)
    assert any("local_sha256 mismatch" in error for error in report.errors)
    with pytest.raises(PublicResonanceInventoryError):
        report.assert_valid()


@pytest.mark.parametrize(
    "escaped_path",
    ["../outside.bin", "nested/../../outside.bin"],
)
def test_relative_path_cannot_escape_data_root(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
    escaped_path: str,
) -> None:
    _, _, manifest = inventory
    manifest["datasets"][0]["files"][0]["relative_path"] = escaped_path

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert any("must stay under data_root" in error for error in report.errors)


@pytest.mark.parametrize("claim", [True, None, 0, "false"])
def test_clarus_specific_evidence_is_exactly_false_claim_lock(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
    claim: object,
) -> None:
    _, _, manifest = inventory
    manifest["datasets"][0]["gate_readiness"]["clarus_specific_evidence"] = claim

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert not report.claim_lock_valid
    assert any("must be exactly false" in error for error in report.errors)


@pytest.mark.parametrize(
    ("claim_name", "value", "expected_text"),
    [
        (
            "public_measurements_are_not_clarus_specific_evidence",
            False,
            "must be exactly true",
        ),
        ("new_matter_created", True, "must be exactly false"),
        ("clarus_field_detected", True, "must be exactly false"),
    ],
)
def test_top_level_physical_claims_are_locked(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
    claim_name: str,
    value: bool,
    expected_text: str,
) -> None:
    _, _, manifest = inventory
    manifest["claim_scope"][claim_name] = value

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert not report.claim_lock_valid
    assert any(expected_text in error for error in report.errors)


def test_downloaded_true_requires_file_to_exist(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    source, _, manifest = inventory
    source.unlink()

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert any("downloaded file is missing" in error for error in report.errors)


def test_complete_record_count_must_match_all_downloaded_files(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    _, _, manifest = inventory
    dataset = manifest["datasets"][0]
    dataset["status"] = "complete_record_verified"
    dataset["record_file_count"] = 2

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert any("record_file_count must equal" in error for error in report.errors)


def test_verified_status_cannot_pass_without_a_download(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    _, _, manifest = inventory
    manifest["datasets"][0]["files"][0]["downloaded"] = False

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert any("requires at least one downloaded file" in error for error in report.errors)


def test_duplicate_dataset_and_file_paths_are_rejected(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    _, _, manifest = inventory
    duplicate = deepcopy(manifest["datasets"][0])
    manifest["datasets"].append(duplicate)

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert any("dataset_id duplicates" in error for error in report.errors)
    assert any("relative_path duplicates" in error for error in report.errors)


def test_manifest_loader_and_cli_audit_preserve_integrity_result(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
) -> None:
    _, _, manifest = inventory
    manifest_path = tmp_path / "inventory.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_public_resonance_inventory(manifest_path, repo_root=tmp_path)

    assert report.valid
    assert report.analysis_ready_dataset_count == 1


@pytest.mark.parametrize(
    ("field", "value", "error_text"),
    [
        ("schema", "clarus.public_resonance_source_manifest.v2", "schema must be exactly"),
        ("raw_files_committed", 0, "raw_files_committed must be exactly false"),
        ("raw_files_committed", True, "raw_files_committed must be exactly false"),
    ],
)
def test_top_level_schema_and_raw_commit_lock_are_exact(
    tmp_path: Path,
    inventory: tuple[Path, bytes, dict[str, object]],
    field: str,
    value: object,
    error_text: str,
) -> None:
    _, _, manifest = inventory
    manifest[field] = value

    report = verify_public_resonance_inventory(manifest, repo_root=tmp_path)

    assert not report.valid
    assert any(error_text in error for error in report.errors)
