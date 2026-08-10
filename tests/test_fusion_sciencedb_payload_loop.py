from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from reality_stone.clarus.fusion_sciencedb_payload_loop import (
    LEGENDRE_ANGULAR_DISTRIBUTION_A1_A12,
    SCALAR_CROSS_SECTION_WITH_ERR,
    SCIENCEDB_EXPECTED_FILE_COUNT,
    SCIENCEDB_EXPECTED_TOTAL_BYTES,
    SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY,
    SCIENCEDB_V1_FILE_SPECS,
    audit_sciencedb_v1_payload,
    current_sciencedb_v1_payload_audit,
    sciencedb_v1_physical_polarized_reaction_gate_pass,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAYLOAD = REPOSITORY_ROOT / SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY


@pytest.fixture(scope="module")
def audit():
    return current_sciencedb_v1_payload_audit()


def _copy_payload(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    destination = root / SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY
    destination.parent.mkdir(parents=True)
    shutil.copytree(SOURCE_PAYLOAD, destination)
    return root, destination


def test_pinned_manifest_is_exactly_six_files_and_8602_bytes() -> None:
    assert SCIENCEDB_EXPECTED_FILE_COUNT == 6
    assert SCIENCEDB_EXPECTED_TOTAL_BYTES == 8_602
    assert len(SCIENCEDB_V1_FILE_SPECS) == 6
    assert sum(spec.expected_size_bytes for spec in SCIENCEDB_V1_FILE_SPECS) == 8_602
    assert len({spec.filename for spec in SCIENCEDB_V1_FILE_SPECS}) == 6
    assert len({spec.file_id for spec in SCIENCEDB_V1_FILE_SPECS}) == 6
    assert all(spec.file_id in spec.official_download_url for spec in SCIENCEDB_V1_FILE_SPECS)
    assert all(len(spec.expected_md5) == 32 for spec in SCIENCEDB_V1_FILE_SPECS)
    assert all(len(spec.expected_sha256) == 64 for spec in SCIENCEDB_V1_FILE_SPECS)


def test_local_v1_raw_bytes_match_all_pinned_digests(audit) -> None:
    assert audit.source_dataset_id == "3a7535ebc6094d4fba445d104f7f2b96"
    assert audit.source_dataset_license == "CC BY-SA 4.0"
    assert audit.directory_available
    assert audit.directory_path_containment_pass
    assert audit.directory_symlink_free
    assert audit.file_count_matches
    assert audit.runtime_entry_count == 6
    assert audit.runtime_total_bytes == 8_602
    assert audit.total_bytes_match
    assert audit.exact_file_set_and_total_size_pass
    assert not audit.missing_file_names
    assert not audit.unexpected_entry_names
    assert audit.all_file_hashes_match
    assert audit.payload_integrity_gate_pass

    for file_audit in audit.file_audits:
        assert file_audit.hashes_computed_from_raw_file_bytes
        assert file_audit.runtime_size_bytes == file_audit.expected_size_bytes
        assert file_audit.runtime_md5 == file_audit.expected_md5
        assert file_audit.runtime_sha256 == file_audit.expected_sha256
        assert file_audit.exact_file_gate_pass


def test_all_headers_row_counts_and_numeric_shapes_are_parsed(audit) -> None:
    expected_rows = {
        "4He(n,d)T-CS.txt": 9,
        "4He(n,el)-CS.txt": 44,
        "4He(n,tot)-CS.txt": 59,
        "T(d,n)4He-CS.txt": 54,
        "T(d,n)4He-DA.txt": 10,
        "T(d,n)4He-L1-CS.txt": 34,
    }
    assert audit.all_table_structures_match
    for file_audit in audit.file_audits:
        assert file_audit.runtime_columns == file_audit.expected_columns
        assert file_audit.header_matches
        assert file_audit.runtime_row_count == expected_rows[file_audit.filename]
        assert file_audit.row_count_matches
        assert file_audit.rows_have_exact_column_count
        assert file_audit.rows_are_finite_numeric
        assert file_audit.energy_grid_strictly_increasing
        assert file_audit.cross_sections_nonnegative
        assert file_audit.table_structure_pass


def test_payload_classes_are_scalar_err_and_legendre_a1_through_a12(audit) -> None:
    scalar = [
        item for item in audit.file_audits if item.payload_class == SCALAR_CROSS_SECTION_WITH_ERR
    ]
    legendre = [
        item
        for item in audit.file_audits
        if item.payload_class == LEGENDRE_ANGULAR_DISTRIBUTION_A1_A12
    ]
    assert len(scalar) == 5
    assert all(item.runtime_columns[-1] == "ERR(mb)" for item in scalar)
    assert all(item.pointwise_err_values_nonnegative for item in scalar)
    assert len(legendre) == 1
    assert legendre[0].filename == "T(d,n)4He-DA.txt"
    assert legendre[0].runtime_columns[2:] == tuple(f"A{index}" for index in range(1, 13))
    assert legendre[0].pointwise_err_values_nonnegative is None


def test_integrity_does_not_promote_covariance_spin_or_polarized_reaction(audit) -> None:
    assert audit.payload_integrity_gate_pass
    assert audit.pointwise_scalar_err_columns_only
    assert not audit.numeric_covariance_matrix_or_correlation_payload_available
    assert not audit.initial_state_spin_columns_or_operator_available
    assert audit.legendre_coefficients_are_not_initial_state_spin_evidence
    assert not audit.physical_polarized_reaction_evidence_gate_pass
    assert not sciencedb_v1_physical_polarized_reaction_gate_pass()
    assert audit.maximum_supported_stage == "unpolarized point-table payload integrity"


def test_one_byte_tamper_fails_raw_hash_and_integrity(tmp_path: Path) -> None:
    root, payload_dir = _copy_payload(tmp_path)
    target = payload_dir / "T(d,n)4He-CS.txt"
    payload = target.read_bytes()
    target.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))

    tampered = audit_sciencedb_v1_payload(repository_root=root)
    target_audit = next(item for item in tampered.file_audits if item.filename == target.name)
    assert target_audit.runtime_size_bytes == target_audit.expected_size_bytes
    assert not target_audit.md5_matches
    assert not target_audit.sha256_matches
    assert not target_audit.exact_file_gate_pass
    assert not tampered.all_file_hashes_match
    assert not tampered.payload_integrity_gate_pass
    assert not tampered.physical_polarized_reaction_evidence_gate_pass


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_exact_file_set_is_fail_closed(tmp_path: Path, mutation: str) -> None:
    root, payload_dir = _copy_payload(tmp_path)
    if mutation == "missing":
        (payload_dir / SCIENCEDB_V1_FILE_SPECS[0].filename).unlink()
    else:
        (payload_dir / "unexpected.txt").write_bytes(b"unexpected")

    changed = audit_sciencedb_v1_payload(repository_root=root)
    assert not changed.exact_file_set_and_total_size_pass
    assert not changed.payload_integrity_gate_pass


def test_header_or_row_structure_change_fails_closed(tmp_path: Path) -> None:
    root, payload_dir = _copy_payload(tmp_path)
    target = payload_dir / "T(d,n)4He-DA.txt"
    lines = target.read_text(encoding="ascii").splitlines()
    lines[0] = lines[0].replace("A12", "SPIN")
    target.write_text("\n".join(lines) + "\n", encoding="ascii", newline="")

    changed = audit_sciencedb_v1_payload(repository_root=root)
    target_audit = next(item for item in changed.file_audits if item.filename == target.name)
    assert not target_audit.header_matches
    assert not target_audit.table_structure_pass
    assert not changed.payload_integrity_gate_pass


@pytest.mark.parametrize(
    "relative_path",
    ["../outside", "C:/outside", "C:outside", "a\\b", "a//b", "a/../b"],
)
def test_directory_escape_is_rejected(tmp_path: Path, relative_path: str) -> None:
    root, _ = _copy_payload(tmp_path)
    changed = audit_sciencedb_v1_payload(
        repository_root=root,
        repository_relative_directory=relative_path,
    )
    assert not changed.directory_available
    assert not changed.directory_path_containment_pass
    assert not changed.payload_integrity_gate_pass


def test_payload_file_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    root, payload_dir = _copy_payload(tmp_path)
    target = payload_dir / SCIENCEDB_V1_FILE_SPECS[0].filename
    outside = root / "outside-copy.txt"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    try:
        os.symlink(outside, target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are not available to this test process")

    changed = audit_sciencedb_v1_payload(repository_root=root)
    target_audit = next(item for item in changed.file_audits if item.filename == target.name)
    assert not target_audit.symlink_free
    assert not target_audit.exact_file_gate_pass
    assert not changed.payload_integrity_gate_pass
