"""Fail-closed audit of the local ScienceDB D--T R-matrix V1 payload.

The ScienceDB record associated with Han et al. contains six small text
tables.  This module pins their byte-level identity and parses their table
shape without promoting the payload beyond what it actually contains:

* five cross-section tables expose one scalar ``ERR`` value per row;
* one angular-distribution table exposes Legendre coefficients ``A1`` ...
  ``A12``;
* none of the files contains a numeric covariance matrix, an initial-state
  spin label/density operator, or a double-polarized D--T reaction operator.

Consequently, a fully intact local payload passes the reproducibility gate
but always fails the physical polarized-reaction evidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from typing import Sequence


HAN_RMATRIX_DOI = "https://doi.org/10.1007/s41365-025-01874-2"
HAN_SCIENCEDB_DOI = "https://doi.org/10.57760/sciencedb.j00186.00813"
SCIENCEDB_DATASET_ID = "3a7535ebc6094d4fba445d104f7f2b96"
SCIENCEDB_VERSION = "V1"
SCIENCEDB_LICENSE = "CC BY-SA 4.0"
SCIENCEDB_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SCIENCEDB_FILE_TREE_API_URL = (
    "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
)
SCIENCEDB_DOWNLOAD_URL_TEMPLATE = (
    "https://www.scidb.cn/api/sdb-download-service/downloadFileMole?fileId={file_id}"
)
SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY = (
    ".research-tmp/ScienceDB-j00186-00813-V1"
)
SCIENCEDB_EXPECTED_FILE_COUNT = 6
SCIENCEDB_EXPECTED_TOTAL_BYTES = 8_602

SCALAR_CROSS_SECTION_WITH_ERR = "scalar_cross_section_with_pointwise_err"
LEGENDRE_ANGULAR_DISTRIBUTION_A1_A12 = "legendre_angular_distribution_a1_a12"


@dataclass(frozen=True)
class ScienceDBV1FileSpec:
    """Pinned byte identity and table contract for one ScienceDB V1 file."""

    filename: str
    file_id: str
    official_download_url: str
    expected_size_bytes: int
    expected_md5: str
    expected_sha256: str
    expected_columns: tuple[str, ...]
    expected_row_count: int
    payload_class: str


@dataclass(frozen=True)
class ScienceDBV1FileAudit:
    """Runtime byte and table-shape audit for one pinned file."""

    filename: str
    payload_class: str
    available: bool
    regular_file: bool
    symlink_free: bool
    repository_path_containment_pass: bool
    expected_size_bytes: int
    runtime_size_bytes: int | None
    size_matches: bool
    expected_md5: str
    runtime_md5: str | None
    md5_matches: bool
    expected_sha256: str
    runtime_sha256: str | None
    sha256_matches: bool
    hashes_computed_from_raw_file_bytes: bool
    expected_columns: tuple[str, ...]
    runtime_columns: tuple[str, ...]
    header_matches: bool
    expected_row_count: int
    runtime_row_count: int
    row_count_matches: bool
    rows_have_exact_column_count: bool
    rows_are_finite_numeric: bool
    energy_grid_strictly_increasing: bool
    cross_sections_nonnegative: bool
    pointwise_err_values_nonnegative: bool | None
    table_structure_pass: bool
    exact_file_gate_pass: bool
    status: str


@dataclass(frozen=True)
class ScienceDBV1PayloadAudit:
    """Aggregate fail-closed audit of the six-file local payload."""

    source_paper_doi: str
    source_dataset_doi: str
    source_dataset_id: str
    source_dataset_version: str
    source_dataset_license: str
    source_dataset_license_url: str
    source_file_tree_api_url: str
    repository_relative_directory: str
    directory_available: bool
    directory_path_containment_pass: bool
    directory_symlink_free: bool
    expected_file_names: tuple[str, ...]
    runtime_entry_names: tuple[str, ...]
    missing_file_names: tuple[str, ...]
    unexpected_entry_names: tuple[str, ...]
    expected_file_count: int
    runtime_entry_count: int
    file_count_matches: bool
    expected_total_bytes: int
    runtime_total_bytes: int | None
    total_bytes_match: bool
    exact_file_set_and_total_size_pass: bool
    file_audits: tuple[ScienceDBV1FileAudit, ...]
    all_file_hashes_match: bool
    all_table_structures_match: bool
    payload_integrity_gate_pass: bool
    scalar_err_table_names: tuple[str, ...]
    legendre_a1_a12_table_names: tuple[str, ...]
    pointwise_scalar_err_columns_only: bool
    numeric_covariance_matrix_or_correlation_payload_available: bool
    initial_state_spin_columns_or_operator_available: bool
    legendre_coefficients_are_not_initial_state_spin_evidence: bool
    physical_polarized_reaction_evidence_gate_pass: bool
    maximum_supported_stage: str
    status: str


_SCALAR_COLUMNS_EN = ("En(MeV)", "CS(mb)", "ERR(mb)")
_SCALAR_COLUMNS_ED = ("Ed(MeV)", "CS(mb)", "ERR(mb)")
_LEGENDRE_COLUMNS = (
    "Ed(MeV)",
    "CS(mb)",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
)

SCIENCEDB_V1_FILE_SPECS = (
    ScienceDBV1FileSpec(
        filename="4He(n,d)T-CS.txt",
        file_id="251be4c159cafd31c99a48519b09393c",
        official_download_url=SCIENCEDB_DOWNLOAD_URL_TEMPLATE.format(
            file_id="251be4c159cafd31c99a48519b09393c"
        ),
        expected_size_bytes=339,
        expected_md5="4c6eec036e9335f24e44e550793c17a8",
        expected_sha256="51fb599f20d88b4c6ed672f839e05b659a431b16db1d9faab6ae65421cef9e77",
        expected_columns=_SCALAR_COLUMNS_EN,
        expected_row_count=9,
        payload_class=SCALAR_CROSS_SECTION_WITH_ERR,
    ),
    ScienceDBV1FileSpec(
        filename="4He(n,el)-CS.txt",
        file_id="c5a6b5203c77431e74cb084c18f3e846",
        official_download_url=SCIENCEDB_DOWNLOAD_URL_TEMPLATE.format(
            file_id="c5a6b5203c77431e74cb084c18f3e846"
        ),
        expected_size_bytes=1_564,
        expected_md5="bc74acf592a482b71e802a1ab124f827",
        expected_sha256="7a4167ade0352cfa73aafb10629fdedec996848ccde97300a331daea168eba17",
        expected_columns=_SCALAR_COLUMNS_EN,
        expected_row_count=44,
        payload_class=SCALAR_CROSS_SECTION_WITH_ERR,
    ),
    ScienceDBV1FileSpec(
        filename="4He(n,tot)-CS.txt",
        file_id="3b0447162d530714f7d48173a3b1b318",
        official_download_url=SCIENCEDB_DOWNLOAD_URL_TEMPLATE.format(
            file_id="3b0447162d530714f7d48173a3b1b318"
        ),
        expected_size_bytes=2_086,
        expected_md5="c7cc9c5b61afa6dabc9b2b935dd961b7",
        expected_sha256="54442b15c2fbc7ad9bf0fe3874ac5816a7027aa1da0d18b64e00d87107e69037",
        expected_columns=_SCALAR_COLUMNS_EN,
        expected_row_count=59,
        payload_class=SCALAR_CROSS_SECTION_WITH_ERR,
    ),
    ScienceDBV1FileSpec(
        filename="T(d,n)4He-CS.txt",
        file_id="a309e0fa39c6abfdf4caf524e7eb703a",
        official_download_url=SCIENCEDB_DOWNLOAD_URL_TEMPLATE.format(
            file_id="a309e0fa39c6abfdf4caf524e7eb703a"
        ),
        expected_size_bytes=1_921,
        expected_md5="22029223a7bbd4ab2257dee00df442a2",
        expected_sha256="3cdc7242f7e135832865057bb522f986d839b10afee40e995b36fcd16f9f996e",
        expected_columns=_SCALAR_COLUMNS_ED,
        expected_row_count=54,
        payload_class=SCALAR_CROSS_SECTION_WITH_ERR,
    ),
    ScienceDBV1FileSpec(
        filename="T(d,n)4He-DA.txt",
        file_id="61318cfc3faa44476171fa1a1928d96e",
        official_download_url=SCIENCEDB_DOWNLOAD_URL_TEMPLATE.format(
            file_id="61318cfc3faa44476171fa1a1928d96e"
        ),
        expected_size_bytes=1_479,
        expected_md5="591e8db4e004c3d386c0314e785018fa",
        expected_sha256="0fc805c14fda768b4d6e8d9af75ae0f0e94d966af6df13895e25b8c51a6936a6",
        expected_columns=_LEGENDRE_COLUMNS,
        expected_row_count=10,
        payload_class=LEGENDRE_ANGULAR_DISTRIBUTION_A1_A12,
    ),
    ScienceDBV1FileSpec(
        filename="T(d,n)4He-L1-CS.txt",
        file_id="d2d3fa11c36d80aedc15e682b6f36806",
        official_download_url=SCIENCEDB_DOWNLOAD_URL_TEMPLATE.format(
            file_id="d2d3fa11c36d80aedc15e682b6f36806"
        ),
        expected_size_bytes=1_213,
        expected_md5="f0217b2842cb63bc59c7e38cebf5a06a",
        expected_sha256="0a997636569780f882cce5a34662019957f9a860089b2d937374a9cca08f4033",
        expected_columns=_SCALAR_COLUMNS_ED,
        expected_row_count=34,
        payload_class=SCALAR_CROSS_SECTION_WITH_ERR,
    ),
)


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "reality_stone").is_dir():
            return parent
    return None


def _safe_relative_parts(value: str) -> tuple[str, ...] | None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        return None
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    if (
        windows_path.is_absolute()
        or windows_path.drive
        or posix_path.is_absolute()
        or posix_path.as_posix() != value
    ):
        return None
    if not posix_path.parts or any(
        part in {"", ".", ".."} or ":" in part for part in posix_path.parts
    ):
        return None
    return tuple(posix_path.parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_is_link_or_reparse_point(path: Path) -> bool:
    try:
        is_junction = getattr(path, "is_junction", lambda: False)
        file_attributes = getattr(os.lstat(path), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return path.is_symlink() or is_junction() or bool(file_attributes & reparse_flag)
    except OSError:
        return True


def _safe_payload_directory(
    repository_root: Path | None,
    repository_relative_directory: str,
) -> tuple[Path | None, bool, bool, bool]:
    parts = _safe_relative_parts(repository_relative_directory)
    if repository_root is None or parts is None:
        return None, False, False, False
    try:
        root = repository_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None, False, False, False
    if not root.is_dir():
        return None, False, False, False

    raw_path = root.joinpath(*parts)
    cursor = root
    symlink_free = True
    for part in parts:
        cursor = cursor / part
        if _path_is_link_or_reparse_point(cursor):
            symlink_free = False
            break
    try:
        resolved = raw_path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        unresolved = raw_path.resolve(strict=False)
        return None, _is_relative_to(unresolved, root), symlink_free, False
    contained = _is_relative_to(resolved, root)
    available = contained and symlink_free and resolved.is_dir()
    return (resolved if available else None), contained, symlink_free, available


def _missing_file_audit(
    spec: ScienceDBV1FileSpec,
    *,
    symlink_free: bool = False,
    containment_pass: bool = False,
    status: str = "missing or unsafe file",
) -> ScienceDBV1FileAudit:
    return ScienceDBV1FileAudit(
        filename=spec.filename,
        payload_class=spec.payload_class,
        available=False,
        regular_file=False,
        symlink_free=symlink_free,
        repository_path_containment_pass=containment_pass,
        expected_size_bytes=spec.expected_size_bytes,
        runtime_size_bytes=None,
        size_matches=False,
        expected_md5=spec.expected_md5,
        runtime_md5=None,
        md5_matches=False,
        expected_sha256=spec.expected_sha256,
        runtime_sha256=None,
        sha256_matches=False,
        hashes_computed_from_raw_file_bytes=False,
        expected_columns=spec.expected_columns,
        runtime_columns=(),
        header_matches=False,
        expected_row_count=spec.expected_row_count,
        runtime_row_count=0,
        row_count_matches=False,
        rows_have_exact_column_count=False,
        rows_are_finite_numeric=False,
        energy_grid_strictly_increasing=False,
        cross_sections_nonnegative=False,
        pointwise_err_values_nonnegative=None,
        table_structure_pass=False,
        exact_file_gate_pass=False,
        status=status,
    )


def _parse_numeric_rows(
    payload: bytes,
    spec: ScienceDBV1FileSpec,
) -> tuple[tuple[str, ...], int, bool, bool, bool, bool, bool | None]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError:
        return (), 0, False, False, False, False, None
    lines = text.splitlines()
    if not lines:
        return (), 0, False, False, False, False, None
    runtime_columns = tuple(lines[0].split())
    data_lines = lines[1:]
    rows_have_exact_columns = bool(data_lines)
    rows_are_finite = bool(data_lines)
    parsed_rows: list[tuple[float, ...]] = []
    for line in data_lines:
        tokens = line.split()
        if len(tokens) != len(spec.expected_columns):
            rows_have_exact_columns = False
            continue
        try:
            values = tuple(float(token) for token in tokens)
        except ValueError:
            rows_are_finite = False
            continue
        if not all(math.isfinite(value) for value in values):
            rows_are_finite = False
            continue
        parsed_rows.append(values)

    all_rows_parsed = (
        rows_have_exact_columns
        and rows_are_finite
        and len(parsed_rows) == len(data_lines)
    )
    energies = [row[0] for row in parsed_rows]
    energy_increasing = all_rows_parsed and all(
        right > left for left, right in zip(energies, energies[1:])
    )
    cross_sections_nonnegative = all_rows_parsed and all(row[1] >= 0.0 for row in parsed_rows)
    err_nonnegative: bool | None
    if spec.payload_class == SCALAR_CROSS_SECTION_WITH_ERR:
        err_nonnegative = all_rows_parsed and all(row[2] >= 0.0 for row in parsed_rows)
    else:
        err_nonnegative = None
    return (
        runtime_columns,
        len(data_lines),
        rows_have_exact_columns,
        rows_are_finite,
        energy_increasing,
        cross_sections_nonnegative,
        err_nonnegative,
    )


def _audit_file(directory: Path, root: Path, spec: ScienceDBV1FileSpec) -> ScienceDBV1FileAudit:
    raw_path = directory / spec.filename
    symlink_free = not _path_is_link_or_reparse_point(raw_path)
    if not symlink_free:
        return _missing_file_audit(
            spec,
            symlink_free=False,
            containment_pass=False,
            status="symlink rejected",
        )
    try:
        resolved = raw_path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return _missing_file_audit(spec)
    containment = _is_relative_to(resolved, root)
    regular = containment and resolved.is_file()
    if not regular:
        return _missing_file_audit(
            spec,
            symlink_free=True,
            containment_pass=containment,
            status="non-regular or out-of-root file rejected",
        )
    try:
        payload = resolved.read_bytes()
    except OSError:
        return _missing_file_audit(
            spec,
            symlink_free=True,
            containment_pass=True,
            status="file could not be read",
        )

    runtime_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    runtime_sha256 = hashlib.sha256(payload).hexdigest()
    runtime_size = len(payload)
    size_matches = runtime_size == spec.expected_size_bytes
    md5_matches = hmac.compare_digest(runtime_md5, spec.expected_md5)
    sha256_matches = hmac.compare_digest(runtime_sha256, spec.expected_sha256)
    (
        runtime_columns,
        runtime_rows,
        exact_columns,
        finite_numeric,
        energy_increasing,
        cross_sections_nonnegative,
        err_nonnegative,
    ) = _parse_numeric_rows(payload, spec)
    header_matches = runtime_columns == spec.expected_columns
    row_count_matches = runtime_rows == spec.expected_row_count
    error_contract_pass = err_nonnegative is not False
    structure_pass = (
        header_matches
        and row_count_matches
        and exact_columns
        and finite_numeric
        and energy_increasing
        and cross_sections_nonnegative
        and error_contract_pass
    )
    exact_pass = size_matches and md5_matches and sha256_matches and structure_pass
    return ScienceDBV1FileAudit(
        filename=spec.filename,
        payload_class=spec.payload_class,
        available=True,
        regular_file=True,
        symlink_free=True,
        repository_path_containment_pass=True,
        expected_size_bytes=spec.expected_size_bytes,
        runtime_size_bytes=runtime_size,
        size_matches=size_matches,
        expected_md5=spec.expected_md5,
        runtime_md5=runtime_md5,
        md5_matches=md5_matches,
        expected_sha256=spec.expected_sha256,
        runtime_sha256=runtime_sha256,
        sha256_matches=sha256_matches,
        hashes_computed_from_raw_file_bytes=True,
        expected_columns=spec.expected_columns,
        runtime_columns=runtime_columns,
        header_matches=header_matches,
        expected_row_count=spec.expected_row_count,
        runtime_row_count=runtime_rows,
        row_count_matches=row_count_matches,
        rows_have_exact_column_count=exact_columns,
        rows_are_finite_numeric=finite_numeric,
        energy_grid_strictly_increasing=energy_increasing,
        cross_sections_nonnegative=cross_sections_nonnegative,
        pointwise_err_values_nonnegative=err_nonnegative,
        table_structure_pass=structure_pass,
        exact_file_gate_pass=exact_pass,
        status="exact pinned file verified" if exact_pass else "file identity or structure mismatch",
    )


def _regular_entry_total_bytes(entries: Sequence[Path]) -> int | None:
    if any(_path_is_link_or_reparse_point(entry) or not entry.is_file() for entry in entries):
        return None
    try:
        return sum(entry.stat().st_size for entry in entries)
    except OSError:
        return None


def audit_sciencedb_v1_payload(
    *,
    repository_root: str | Path | None = None,
    repository_relative_directory: str = SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY,
) -> ScienceDBV1PayloadAudit:
    """Audit the exact six-file V1 payload from raw bytes and parsed rows."""

    requested_root = Path(repository_root) if repository_root is not None else _repository_root()
    directory, contained, symlink_free, available = _safe_payload_directory(
        requested_root,
        repository_relative_directory,
    )
    expected_names = tuple(spec.filename for spec in SCIENCEDB_V1_FILE_SPECS)
    expected_name_set = frozenset(expected_names)

    entries: tuple[Path, ...] = ()
    if directory is not None:
        try:
            entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
        except OSError:
            available = False
            directory = None
    runtime_names = tuple(entry.name for entry in entries)
    runtime_name_set = frozenset(runtime_names)
    missing_names = tuple(sorted(expected_name_set - runtime_name_set))
    unexpected_names = tuple(sorted(runtime_name_set - expected_name_set))
    runtime_total_bytes = _regular_entry_total_bytes(entries) if directory is not None else None
    file_count_matches = len(entries) == SCIENCEDB_EXPECTED_FILE_COUNT
    total_bytes_match = runtime_total_bytes == SCIENCEDB_EXPECTED_TOTAL_BYTES
    exact_set_and_size = (
        available
        and runtime_name_set == expected_name_set
        and len(runtime_names) == len(expected_names)
        and file_count_matches
        and total_bytes_match
    )

    if directory is not None and requested_root is not None:
        root = Path(requested_root).resolve(strict=True)
        file_audits = tuple(
            _audit_file(directory, root, spec) for spec in SCIENCEDB_V1_FILE_SPECS
        )
    else:
        file_audits = tuple(_missing_file_audit(spec) for spec in SCIENCEDB_V1_FILE_SPECS)

    all_hashes = all(
        audit.size_matches and audit.md5_matches and audit.sha256_matches
        for audit in file_audits
    )
    all_structures = all(audit.table_structure_pass for audit in file_audits)
    integrity_pass = (
        contained
        and symlink_free
        and exact_set_and_size
        and all_hashes
        and all_structures
        and all(audit.exact_file_gate_pass for audit in file_audits)
    )
    scalar_names = tuple(
        spec.filename
        for spec in SCIENCEDB_V1_FILE_SPECS
        if spec.payload_class == SCALAR_CROSS_SECTION_WITH_ERR
    )
    legendre_names = tuple(
        spec.filename
        for spec in SCIENCEDB_V1_FILE_SPECS
        if spec.payload_class == LEGENDRE_ANGULAR_DISTRIBUTION_A1_A12
    )

    # These are schema facts, not conclusions inferred from numerical values.
    # A scalar ERR column provides no off-diagonal covariance/correlation
    # structure, and A_l angular coefficients provide no initial-state spin
    # density/operator labels.
    numeric_covariance_available = False
    initial_state_spin_available = False
    physical_polarized_gate = (
        integrity_pass and numeric_covariance_available and initial_state_spin_available
    )
    return ScienceDBV1PayloadAudit(
        source_paper_doi=HAN_RMATRIX_DOI,
        source_dataset_doi=HAN_SCIENCEDB_DOI,
        source_dataset_id=SCIENCEDB_DATASET_ID,
        source_dataset_version=SCIENCEDB_VERSION,
        source_dataset_license=SCIENCEDB_LICENSE,
        source_dataset_license_url=SCIENCEDB_LICENSE_URL,
        source_file_tree_api_url=SCIENCEDB_FILE_TREE_API_URL,
        repository_relative_directory=repository_relative_directory,
        directory_available=available,
        directory_path_containment_pass=contained,
        directory_symlink_free=symlink_free,
        expected_file_names=expected_names,
        runtime_entry_names=runtime_names,
        missing_file_names=missing_names,
        unexpected_entry_names=unexpected_names,
        expected_file_count=SCIENCEDB_EXPECTED_FILE_COUNT,
        runtime_entry_count=len(entries),
        file_count_matches=file_count_matches,
        expected_total_bytes=SCIENCEDB_EXPECTED_TOTAL_BYTES,
        runtime_total_bytes=runtime_total_bytes,
        total_bytes_match=total_bytes_match,
        exact_file_set_and_total_size_pass=exact_set_and_size,
        file_audits=file_audits,
        all_file_hashes_match=all_hashes,
        all_table_structures_match=all_structures,
        payload_integrity_gate_pass=integrity_pass,
        scalar_err_table_names=scalar_names,
        legendre_a1_a12_table_names=legendre_names,
        pointwise_scalar_err_columns_only=True,
        numeric_covariance_matrix_or_correlation_payload_available=numeric_covariance_available,
        initial_state_spin_columns_or_operator_available=initial_state_spin_available,
        legendre_coefficients_are_not_initial_state_spin_evidence=True,
        physical_polarized_reaction_evidence_gate_pass=physical_polarized_gate,
        maximum_supported_stage=(
            "unpolarized point-table payload integrity"
            if integrity_pass
            else "ScienceDB payload identity unverified"
        ),
        status=(
            "local V1 bytes and table shapes verified; polarized evidence remains absent"
            if integrity_pass
            else "fail-closed: local V1 payload identity or structure mismatch"
        ),
    )


def current_sciencedb_v1_payload_audit() -> ScienceDBV1PayloadAudit:
    """Return the audit for the repository's canonical local payload path."""

    return audit_sciencedb_v1_payload()


def sciencedb_v1_physical_polarized_reaction_gate_pass() -> bool:
    """Recompute the canonical physical gate instead of trusting caller data."""

    return current_sciencedb_v1_payload_audit().physical_polarized_reaction_evidence_gate_pass


__all__ = [
    "HAN_RMATRIX_DOI",
    "HAN_SCIENCEDB_DOI",
    "LEGENDRE_ANGULAR_DISTRIBUTION_A1_A12",
    "SCALAR_CROSS_SECTION_WITH_ERR",
    "SCIENCEDB_DATASET_ID",
    "SCIENCEDB_DOWNLOAD_URL_TEMPLATE",
    "SCIENCEDB_EXPECTED_FILE_COUNT",
    "SCIENCEDB_EXPECTED_TOTAL_BYTES",
    "SCIENCEDB_FILE_TREE_API_URL",
    "SCIENCEDB_LICENSE",
    "SCIENCEDB_LICENSE_URL",
    "SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY",
    "SCIENCEDB_V1_FILE_SPECS",
    "SCIENCEDB_VERSION",
    "ScienceDBV1FileAudit",
    "ScienceDBV1FileSpec",
    "ScienceDBV1PayloadAudit",
    "audit_sciencedb_v1_payload",
    "current_sciencedb_v1_payload_audit",
    "sciencedb_v1_physical_polarized_reaction_gate_pass",
]
