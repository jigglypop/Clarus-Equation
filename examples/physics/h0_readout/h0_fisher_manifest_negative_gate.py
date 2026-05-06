"""Negative validation tests for H0 Fisher/covariance source manifests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h0_fisher_manifest_validate_gate import validate_manifest


def write_manifest(directory: Path, payload: dict) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def base_payload() -> dict:
    return {
        "dataset_bundle": "negative_test",
        "version": "0.0.0",
        "channels": [
            {
                "file": "channel.json",
                "source": "synthetic",
                "source_url": None,
                "matrix_role": "fisher",
                "channel_class": "test",
                "notes": "negative test",
            }
        ],
    }


BAD_CASES = [
    (
        "missing channel file",
        lambda payload: payload,
        "manifest channel file does not exist",
    ),
    (
        "non-synthetic missing source url",
        lambda payload: (
            payload["channels"][0].update({"source": "paper", "source_url": None}) or payload
        ),
        "non-synthetic channel requires source_url",
    ),
    (
        "duplicate channel file",
        lambda payload: (
            payload["channels"].append(dict(payload["channels"][0])) or payload
        ),
        "duplicate channel file in manifest",
    ),
    (
        "missing channel field",
        lambda payload: (payload["channels"][0].pop("matrix_role"), payload)[1],
        "missing fields",
    ),
]


def main() -> int:
    print("# H0 Fisher Manifest Negative Gate")
    print()
    print("| case | expected fragment | status | validator notes |")
    print("|---|---|---|---|")
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, mutate, expected in BAD_CASES:
            payload = mutate(base_payload())
            manifest = write_manifest(root, payload)
            errors = validate_manifest(manifest)
            joined = "; ".join(errors)
            passed = bool(errors) and expected in joined
            if not passed:
                failed += 1
            print(f"| {name} | {expected} | {'PASS' if passed else 'FAIL'} | {joined or 'no error'} |")
    print()
    print(f"negative cases = {len(BAD_CASES)}")
    print(f"failed = {failed}")
    if failed:
        raise SystemExit(1)
    print("Verdict: manifest validator rejects malformed source metadata as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
