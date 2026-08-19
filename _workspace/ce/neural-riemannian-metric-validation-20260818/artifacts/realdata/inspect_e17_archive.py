"""Create a byte-level manifest and extract the E17 files used for eligibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


DEFAULT_PREFIXES = ("figure2/", "figure3/", "figure4/", "figure5/")
DEFAULT_NAMES = {"readme", "readme.md", "license", "license.md", "datacite.yml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_selected(name: str) -> bool:
    normalized = name.replace("\\", "/").lower().lstrip("./")
    parts = PurePosixPath(normalized).parts
    return (
        any(f"/{prefix}" in f"/{normalized}" for prefix in DEFAULT_PREFIXES)
        or (parts and parts[-1] in DEFAULT_NAMES)
    )


def safe_destination(root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe ZIP member: {member_name}")
    destination = root.joinpath(*relative.parts).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise ValueError(f"ZIP member escapes extraction root: {member_name}")
    return destination


def inspect_archive(archive: Path, extract_root: Path, output: Path) -> dict[str, object]:
    extract_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    extracted: list[dict[str, object]] = []

    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC failure in {bad_member}")

        for info in bundle.infolist():
            selected = not info.is_dir() and is_selected(info.filename)
            entries.append(
                {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "selected": selected,
                }
            )
            if not selected:
                continue

            destination = safe_destination(extract_root, info.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with bundle.open(info) as source, destination.open("wb") as sink:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    sink.write(block)
            extracted.append(
                {
                    "archive_name": info.filename,
                    "local_path": destination.relative_to(output.parent.parent.parent).as_posix(),
                    "bytes": info.file_size,
                    "sha256": digest.hexdigest(),
                }
            )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": "NRM-E17",
            "title": "MaristanyEtAl_2025_Dataset",
            "official_doi": "https://doi.org/10.12751/g-node.etlk5k",
            "download_url": "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip",
            "license": "CC BY 4.0",
        },
        "archive": {
            "local_path": archive.relative_to(output.parent.parent.parent).as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "zip_crc_check": "PASS",
            "entry_count": len(entries),
        },
        "selection": {
            "rule": "all Figure2-Figure5 files plus repository README/LICENSE/DataCite metadata",
            "outcome_blind_reason": "eligibility and schema inspection for connection, intervention, longitudinal identity, and trajectory fields",
        },
        "entries": entries,
        "extracted": extracted,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("extract_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = inspect_archive(args.archive.resolve(), args.extract_root.resolve(), args.output.resolve())
    archive_info = manifest["archive"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "archive_bytes": archive_info["bytes"],
                "archive_sha256": archive_info["sha256"],
                "entry_count": archive_info["entry_count"],
                "extracted_count": len(manifest["extracted"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
