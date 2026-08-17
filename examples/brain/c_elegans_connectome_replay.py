"""Offline replay of a user-supplied frozen C. elegans structural CSV."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from reality_stone.clarus.connectome_replay import (
    ReplayValidationError,
    canonical_bytes,
    replay_source_file,
)


def _aliases(left: Path, right: Path) -> bool:
    if left == right:
        return True
    return left.exists() and os.path.samefile(left, right)


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest_path = args.manifest.resolve(strict=True)
        source_path = args.source.resolve(strict=True)
        output_path = args.output.resolve(strict=False)
        if _aliases(output_path, manifest_path) or _aliases(output_path, source_path):
            raise ReplayValidationError("output must not alias manifest or source")
        artifact, digest = replay_source_file(manifest_path, source_path)
        _write_atomic(output_path, canonical_bytes(artifact))
    except (OSError, ReplayValidationError) as exc:
        parser.error(str(exc))
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
