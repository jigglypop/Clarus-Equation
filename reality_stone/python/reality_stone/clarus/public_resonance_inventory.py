"""Fail-closed verification of the public resonance source inventory.

The inventory records external files without treating their presence as
Clarus-specific evidence.  A downloaded file is accepted only when its path
stays below the declared repository-local data root and its byte count, source
checksum, and local SHA-256 digest all match the manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA = "clarus.public_resonance_source_manifest.v1"
ALLOWED_DATASET_STATUSES = frozenset(
    {"complete_record_verified", "selected_files_verified", "metadata_only"}
)
VERIFIED_DATASET_STATUSES = frozenset(
    {"complete_record_verified", "selected_files_verified"}
)
_DIGEST_LENGTHS = {"md5": 32, "sha256": 64}
_LOWER_HEX = re.compile(r"^[0-9a-f]+$")


class PublicResonanceInventoryError(ValueError):
    """Raised when a public source inventory fails a locked invariant."""


@dataclass(frozen=True)
class PublicFileAudit:
    """Verification result for one declared source file."""

    dataset_id: str
    relative_path: str
    downloaded: bool
    exists: bool
    expected_bytes: int | None
    observed_bytes: int | None
    size_matches: bool
    source_checksum_algorithm: str
    source_checksum_matches: bool
    local_sha256_matches: bool
    verified: bool


@dataclass(frozen=True)
class PublicDatasetAudit:
    """Verification result for one public dataset record."""

    dataset_id: str
    status: str
    declared_file_count: int
    downloaded_file_count: int
    verified_file_count: int
    record_file_count: int | None
    clarus_specific_evidence: bool
    analysis_ready: bool
    files: tuple[PublicFileAudit, ...]


@dataclass(frozen=True)
class PublicResonanceInventoryReport:
    """Aggregated fail-closed inventory result."""

    schema: str
    data_root: str
    raw_files_committed: bool | None
    dataset_count: int
    downloaded_file_count: int
    verified_file_count: int
    analysis_ready_dataset_count: int
    metadata_only_dataset_count: int
    claim_lock_valid: bool
    clarus_specific_evidence: bool
    all_downloaded_files_verified: bool
    valid: bool
    errors: tuple[str, ...]
    datasets: tuple[PublicDatasetAudit, ...]

    def assert_valid(self) -> None:
        """Raise when any schema, claim, path, or checksum invariant failed."""

        if not self.valid:
            raise PublicResonanceInventoryError("; ".join(self.errors))


def load_public_resonance_manifest(path: str | Path) -> dict[str, Any]:
    """Load a public source manifest and require a JSON object at its root."""

    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicResonanceInventoryError(
            f"could not load public resonance manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PublicResonanceInventoryError("manifest root must be a JSON object")
    return value


def _mapping(value: object, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _list(value: object, name: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{name} must be an array")
        return []
    return value


def _nonempty_text(value: object, name: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string")
        return ""
    return value


def _positive_integer(value: object, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{name} must be a positive integer")
        return None
    return value


def _nonnegative_integer(value: object, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{name} must be a non-negative integer")
        return None
    return value


def _digest(
    value: object,
    *,
    algorithm: str,
    name: str,
    errors: list[str],
) -> str:
    text = _nonempty_text(value, name, errors)
    expected_length = _DIGEST_LENGTHS.get(algorithm)
    if text and (
        expected_length is None
        or len(text) != expected_length
        or _LOWER_HEX.fullmatch(text) is None
    ):
        label = algorithm.upper() if algorithm else "checksum"
        errors.append(f"{name} must be lowercase {label} hex")
        return ""
    return text


def _source_url(value: object, name: str, errors: list[str]) -> str:
    text = _nonempty_text(value, name, errors)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        errors.append(f"{name} must be an absolute HTTP(S) URL")
        return ""
    return text


def _safe_child(root: Path, raw_path: str) -> Path | None:
    """Resolve a non-empty relative path that remains strictly below root."""

    relative = Path(raw_path)
    if not raw_path or relative.is_absolute() or relative.anchor or ".." in relative.parts:
        return None
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate == resolved_root:
        return None
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _file_hashes(path: Path, source_algorithm: str) -> tuple[str, str]:
    source_digest = hashlib.new(source_algorithm)
    local_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            source_digest.update(block)
            local_digest.update(block)
    return source_digest.hexdigest(), local_digest.hexdigest()


def _empty_file_audit(
    *,
    dataset_id: str,
    relative_path: str,
    downloaded: bool,
    expected_bytes: int | None,
    source_algorithm: str,
) -> PublicFileAudit:
    return PublicFileAudit(
        dataset_id=dataset_id,
        relative_path=relative_path,
        downloaded=downloaded,
        exists=False,
        expected_bytes=expected_bytes,
        observed_bytes=None,
        size_matches=False,
        source_checksum_algorithm=source_algorithm,
        source_checksum_matches=False,
        local_sha256_matches=False,
        verified=False,
    )


def verify_public_resonance_inventory(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> PublicResonanceInventoryReport:
    """Verify schema, claim locks, paths, and every downloaded file.

    Missing files are permitted only when their entry declares
    ``downloaded=false``.  A verified dataset status nevertheless requires at
    least one downloaded and fully verified file.  ``metadata_only`` entries
    remain useful provenance records but never become analysis-ready.
    """

    errors: list[str] = []
    root = Path(repo_root).resolve()
    if not root.is_dir():
        errors.append(f"repo_root must be an existing directory: {root}")

    schema_value = manifest.get("schema")
    schema = schema_value if isinstance(schema_value, str) else ""
    if schema_value != SCHEMA:
        errors.append(f"schema must be exactly {SCHEMA!r}")

    data_root = _nonempty_text(manifest.get("data_root"), "data_root", errors)
    resolved_data_root = _safe_child(root, data_root) if data_root else None
    if data_root and resolved_data_root is None:
        errors.append("data_root must be a safe repository-relative path")

    raw_files_value = manifest.get("raw_files_committed")
    if raw_files_value is not False:
        errors.append("raw_files_committed must be exactly false")
    raw_files_committed = raw_files_value if isinstance(raw_files_value, bool) else None

    claim_lock_valid = True
    claim_scope = _mapping(manifest.get("claim_scope"), "claim_scope", errors)
    if claim_scope.get("public_measurements_are_not_clarus_specific_evidence") is not True:
        claim_lock_valid = False
        errors.append(
            "claim_scope.public_measurements_are_not_clarus_specific_evidence "
            "must be exactly true"
        )
    for claim_name in ("new_matter_created", "clarus_field_detected"):
        if claim_scope.get(claim_name) is not False:
            claim_lock_valid = False
            errors.append(f"claim_scope.{claim_name} must be exactly false")

    raw_datasets = _list(manifest.get("datasets"), "datasets", errors)
    if not raw_datasets:
        errors.append("datasets must not be empty")

    dataset_ids: set[str] = set()
    file_paths: set[str] = set()
    dataset_audits: list[PublicDatasetAudit] = []

    for dataset_index, raw_dataset in enumerate(raw_datasets):
        prefix = f"datasets[{dataset_index}]"
        dataset = _mapping(raw_dataset, prefix, errors)
        dataset_id = _nonempty_text(dataset.get("dataset_id"), f"{prefix}.dataset_id", errors)
        if dataset_id:
            if dataset_id in dataset_ids:
                errors.append(f"{prefix}.dataset_id duplicates {dataset_id!r}")
            dataset_ids.add(dataset_id)

        status = _nonempty_text(dataset.get("status"), f"{prefix}.status", errors)
        if status and status not in ALLOWED_DATASET_STATUSES:
            errors.append(f"{prefix}.status is not an allowed status")

        gate = _mapping(dataset.get("gate_readiness"), f"{prefix}.gate_readiness", errors)
        clarus_value = gate.get("clarus_specific_evidence")
        if clarus_value is not False:
            claim_lock_valid = False
            errors.append(
                f"{prefix}.gate_readiness.clarus_specific_evidence must be exactly false"
            )

        raw_files = _list(dataset.get("files"), f"{prefix}.files", errors)
        record_file_count: int | None = None
        if status == "complete_record_verified":
            record_file_count = _positive_integer(
                dataset.get("record_file_count"),
                f"{prefix}.record_file_count",
                errors,
            )
        elif "record_file_count" in dataset:
            record_file_count = _nonnegative_integer(
                dataset.get("record_file_count"),
                f"{prefix}.record_file_count",
                errors,
            )

        file_audits: list[PublicFileAudit] = []
        downloaded_count = 0
        verified_count = 0
        for file_index, raw_file in enumerate(raw_files):
            file_prefix = f"{prefix}.files[{file_index}]"
            file_entry = _mapping(raw_file, file_prefix, errors)
            relative_path = _nonempty_text(
                file_entry.get("relative_path"),
                f"{file_prefix}.relative_path",
                errors,
            )
            _source_url(file_entry.get("source_url"), f"{file_prefix}.source_url", errors)
            expected_bytes = _positive_integer(
                file_entry.get("bytes"), f"{file_prefix}.bytes", errors
            )

            checksum = _mapping(
                file_entry.get("source_checksum"),
                f"{file_prefix}.source_checksum",
                errors,
            )
            algorithm_value = checksum.get("algorithm")
            algorithm = algorithm_value if isinstance(algorithm_value, str) else ""
            if algorithm not in _DIGEST_LENGTHS:
                errors.append(
                    f"{file_prefix}.source_checksum.algorithm must be md5 or sha256"
                )
            source_digest = _digest(
                checksum.get("digest"),
                algorithm=algorithm,
                name=f"{file_prefix}.source_checksum.digest",
                errors=errors,
            )
            local_sha256 = _digest(
                file_entry.get("local_sha256"),
                algorithm="sha256",
                name=f"{file_prefix}.local_sha256",
                errors=errors,
            )

            downloaded_value = file_entry.get("downloaded")
            if not isinstance(downloaded_value, bool):
                errors.append(f"{file_prefix}.downloaded must be exactly boolean")
                downloaded = False
            else:
                downloaded = downloaded_value

            candidate = (
                _safe_child(resolved_data_root, relative_path)
                if resolved_data_root is not None and relative_path
                else None
            )
            if relative_path and candidate is None:
                errors.append(f"{file_prefix}.relative_path must stay under data_root")

            if candidate is not None:
                normalized_path = str(candidate).casefold()
                if normalized_path in file_paths:
                    errors.append(f"{file_prefix}.relative_path duplicates another file")
                file_paths.add(normalized_path)

            if not downloaded:
                file_audits.append(
                    _empty_file_audit(
                        dataset_id=dataset_id,
                        relative_path=relative_path,
                        downloaded=False,
                        expected_bytes=expected_bytes,
                        source_algorithm=algorithm,
                    )
                )
                continue

            downloaded_count += 1
            if candidate is None:
                file_audits.append(
                    _empty_file_audit(
                        dataset_id=dataset_id,
                        relative_path=relative_path,
                        downloaded=True,
                        expected_bytes=expected_bytes,
                        source_algorithm=algorithm,
                    )
                )
                continue
            if not candidate.is_file():
                errors.append(f"{file_prefix} downloaded file is missing: {relative_path}")
                file_audits.append(
                    _empty_file_audit(
                        dataset_id=dataset_id,
                        relative_path=relative_path,
                        downloaded=True,
                        expected_bytes=expected_bytes,
                        source_algorithm=algorithm,
                    )
                )
                continue

            observed_bytes = candidate.stat().st_size
            size_matches = expected_bytes is not None and observed_bytes == expected_bytes
            if not size_matches:
                errors.append(
                    f"{file_prefix}.bytes mismatch: expected {expected_bytes}, "
                    f"observed {observed_bytes}"
                )

            source_matches = False
            local_matches = False
            if algorithm in _DIGEST_LENGTHS:
                observed_source, observed_local = _file_hashes(candidate, algorithm)
                source_matches = bool(source_digest) and observed_source == source_digest
                local_matches = bool(local_sha256) and observed_local == local_sha256
                if not source_matches:
                    errors.append(f"{file_prefix}.source_checksum mismatch")
                if not local_matches:
                    errors.append(f"{file_prefix}.local_sha256 mismatch")

            verified = size_matches and source_matches and local_matches
            verified_count += int(verified)
            file_audits.append(
                PublicFileAudit(
                    dataset_id=dataset_id,
                    relative_path=relative_path,
                    downloaded=True,
                    exists=True,
                    expected_bytes=expected_bytes,
                    observed_bytes=observed_bytes,
                    size_matches=size_matches,
                    source_checksum_algorithm=algorithm,
                    source_checksum_matches=source_matches,
                    local_sha256_matches=local_matches,
                    verified=verified,
                )
            )

        if status in VERIFIED_DATASET_STATUSES and downloaded_count == 0:
            errors.append(f"{prefix} verified status requires at least one downloaded file")
        if status == "complete_record_verified":
            if record_file_count is not None and record_file_count != downloaded_count:
                errors.append(
                    f"{prefix}.record_file_count must equal the number of downloaded files"
                )
            if downloaded_count != len(raw_files):
                errors.append(f"{prefix} complete record requires every listed file downloaded")

        analysis_ready = (
            status in VERIFIED_DATASET_STATUSES
            and downloaded_count > 0
            and verified_count == downloaded_count
            and (status != "complete_record_verified" or downloaded_count == len(raw_files))
        )
        dataset_audits.append(
            PublicDatasetAudit(
                dataset_id=dataset_id,
                status=status,
                declared_file_count=len(raw_files),
                downloaded_file_count=downloaded_count,
                verified_file_count=verified_count,
                record_file_count=record_file_count,
                clarus_specific_evidence=False,
                analysis_ready=analysis_ready,
                files=tuple(file_audits),
            )
        )

    downloaded_total = sum(item.downloaded_file_count for item in dataset_audits)
    verified_total = sum(item.verified_file_count for item in dataset_audits)
    all_downloaded_verified = downloaded_total == verified_total
    valid = not errors and claim_lock_valid and all_downloaded_verified
    return PublicResonanceInventoryReport(
        schema=schema,
        data_root=data_root,
        raw_files_committed=raw_files_committed,
        dataset_count=len(dataset_audits),
        downloaded_file_count=downloaded_total,
        verified_file_count=verified_total,
        analysis_ready_dataset_count=sum(item.analysis_ready for item in dataset_audits),
        metadata_only_dataset_count=sum(
            item.status == "metadata_only" for item in dataset_audits
        ),
        claim_lock_valid=claim_lock_valid,
        clarus_specific_evidence=False,
        all_downloaded_files_verified=all_downloaded_verified,
        valid=valid,
        errors=tuple(errors),
        datasets=tuple(dataset_audits),
    )


def audit_public_resonance_inventory(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
) -> PublicResonanceInventoryReport:
    """Load and verify one inventory in a single call."""

    return verify_public_resonance_inventory(
        load_public_resonance_manifest(manifest_path),
        repo_root=repo_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-analysis-ready", action="store_true")
    args = parser.parse_args(argv)

    report = audit_public_resonance_inventory(
        args.manifest,
        repo_root=args.repo_root,
    )
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(
        not report.valid
        or (args.require_analysis_ready and report.analysis_ready_dataset_count == 0)
    )


__all__ = [
    "ALLOWED_DATASET_STATUSES",
    "PublicDatasetAudit",
    "PublicFileAudit",
    "PublicResonanceInventoryError",
    "PublicResonanceInventoryReport",
    "SCHEMA",
    "audit_public_resonance_inventory",
    "load_public_resonance_manifest",
    "main",
    "verify_public_resonance_inventory",
]
