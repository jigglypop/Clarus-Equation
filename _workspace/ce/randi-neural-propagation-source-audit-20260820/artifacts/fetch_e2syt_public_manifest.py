"""Fetch a compact, metadata-only receipt for published DANDI:001075.

No NWB payload is downloaded.  The receipt freezes the published version,
asset UUID/path/size/SHA-256 values, and the deterministic smallest
segmentation asset used by the subsequent schema-only inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DANDISET = "001075"
VERSION = "0.240920.1434"
VERSION_URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/"
ASSETS_URL = VERSION_URL + "assets/?page_size=1000&metadata=true"

EXPECTED = {
    "version_id": f"DANDI:{DANDISET}/{VERSION}",
    "doi": f"10.48324/dandi.{DANDISET}/{VERSION}",
    "asset_count": 223,
    "subject_count": 113,
    "total_bytes": 4_073_427_051_047,
    "segmentation_count": 110,
    "segmentation_bytes": 893_457_040,
    "full_count": 113,
    "full_bytes": 4_072_533_594_007,
    "missing_segmentation_subjects": ["sub-20", "sub-23", "sub-33"],
}


def _fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "Clarus-Equation-E2SYT-source-audit/1"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subject_from_path(path: str) -> str:
    subject = path.split("/", 1)[0]
    if not subject.startswith("sub-"):
        raise ValueError(f"unexpected asset path: {path}")
    return subject


def _normalize_asset(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = raw.get("metadata") or {}
    digest = metadata.get("digest") or {}
    participants = metadata.get("wasAttributedTo") or []
    sessions = [
        item
        for item in metadata.get("wasGeneratedBy") or []
        if item.get("schemaKey") == "Session"
    ]
    path = str(raw["path"])
    sha256 = digest.get("dandi:sha2-256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"missing SHA-256 for {path}")
    return {
        "asset_id": str(raw["asset_id"]),
        "blob_id": raw.get("blob"),
        "path": path,
        "subject": _subject_from_path(path),
        "asset_class": "segmentation" if "_desc-segmentation_" in path else "full",
        "bytes": int(raw["size"]),
        "sha256": sha256,
        "dandi_etag": digest.get("dandi:dandi-etag"),
        "encoding_format": metadata.get("encodingFormat"),
        "genotype": participants[0].get("genotype") if participants else None,
        "session_start": sessions[0].get("startDate") if sessions else None,
        "content_urls": list(metadata.get("contentUrl") or []),
        "created": raw.get("created"),
        "modified": raw.get("modified"),
    }


def build_receipt() -> dict[str, Any]:
    version = _fetch_json(VERSION_URL)
    page = _fetch_json(ASSETS_URL)
    if page.get("next") is not None:
        raise ValueError("asset pagination was not exhausted")
    raw_assets = page.get("results") or []
    assets = sorted((_normalize_asset(item) for item in raw_assets), key=lambda x: x["path"])
    segmentation = [item for item in assets if item["asset_class"] == "segmentation"]
    full = [item for item in assets if item["asset_class"] == "full"]
    full_subjects = sorted({item["subject"] for item in full})
    segmentation_subjects = sorted({item["subject"] for item in segmentation})
    missing_segmentation = sorted(set(full_subjects) - set(segmentation_subjects))
    smallest = min(segmentation, key=lambda x: (x["bytes"], x["path"]))

    summary = {
        "asset_count": len(assets),
        "subject_count": len(full_subjects),
        "total_bytes": sum(item["bytes"] for item in assets),
        "segmentation_count": len(segmentation),
        "segmentation_bytes": sum(item["bytes"] for item in segmentation),
        "full_count": len(full),
        "full_bytes": sum(item["bytes"] for item in full),
        "missing_segmentation_subjects": missing_segmentation,
    }
    observed = {
        "version_id": version.get("id"),
        "doi": version.get("doi"),
        **summary,
    }
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"published manifest changed: {mismatches}")

    assets_sha256 = _canonical_sha256(assets)
    return {
        "schema": "clarus.e2syt.public-manifest.v1",
        "status": "PASS_DANDI_MANIFEST",
        "scope": "metadata_only_no_nwb_payload_downloaded",
        "accessed_date": "2026-08-20",
        "source_urls": {
            "version": VERSION_URL,
            "assets": ASSETS_URL,
            "published_assets_yaml": (
                f"https://dandiarchive.s3.amazonaws.com/dandisets/{DANDISET}/{VERSION}/assets.yaml"
            ),
        },
        "dandiset": {
            "id": version.get("id"),
            "identifier": version.get("identifier"),
            "version": version.get("version"),
            "doi": version.get("doi"),
            "name": version.get("name"),
            "url": version.get("url"),
            "license": version.get("license"),
            "access": version.get("access"),
            "date_published": version.get("datePublished"),
            "schema_version": version.get("schemaVersion"),
            "manifest_location": version.get("manifestLocation"),
        },
        "summary": summary,
        "assets_canonical_sha256": assets_sha256,
        "deterministic_schema_exemplar": {
            "selection_rule": "minimum(bytes,path) among asset_class=segmentation",
            **smallest,
        },
        "assets": assets,
        "claim_boundary": {
            "event_schema": "UNINSPECTED",
            "nwb_payload": "NOT_DOWNLOADED",
            "empirical_effect": "NOT_COMPUTED",
            "osf_native_manifest": "BLOCKED_OSF_MANIFEST",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt()
    rendered = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(args.output)
    print(f"assets_sha256={receipt['assets_canonical_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
