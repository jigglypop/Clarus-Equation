"""Validate a manifest for H0 Fisher/covariance channel bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CHANNEL_FIELDS = {
    "file",
    "source",
    "matrix_role",
    "channel_class",
    "notes",
}


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ["manifest top-level JSON must be an object"]

    for field in ["dataset_bundle", "version", "channels"]:
        if field not in payload:
            errors.append(f"missing required manifest field: {field}")

    channels: Any = payload.get("channels")
    if not isinstance(channels, list) or not channels:
        errors.append("channels must be a nonempty list")
        return errors

    seen_files: set[str] = set()
    for index, channel in enumerate(channels):
        if not isinstance(channel, dict):
            errors.append(f"channels[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_CHANNEL_FIELDS - set(channel))
        if missing:
            errors.append(f"channels[{index}] missing fields: {', '.join(missing)}")
        file_name = str(channel.get("file", ""))
        if not file_name.endswith(".json"):
            errors.append(f"channels[{index}].file must point to a .json file")
        if file_name in seen_files:
            errors.append(f"duplicate channel file in manifest: {file_name}")
        seen_files.add(file_name)
        channel_path = path.parent / file_name
        if not channel_path.exists():
            errors.append(f"manifest channel file does not exist: {file_name}")
        source = channel.get("source")
        source_url = channel.get("source_url")
        if source != "synthetic" and not source_url:
            errors.append(f"non-synthetic channel requires source_url: {file_name}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples") / "manifest.json"),
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    errors = validate_manifest(manifest)

    print("# H0 Fisher Manifest Validate Gate")
    print()
    print(f"manifest = {manifest}")
    if errors:
        print("status = FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("status = PASS")
    print("Verdict: source manifest is complete and channel files exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
