"""Verify all locally downloaded files in the public resonance inventory."""

from __future__ import annotations

from pathlib import Path

from reality_stone.clarus.public_resonance_inventory import main


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    default_manifest = (
        repository_root / "artifacts" / "physics" / "public_resonance_source_manifest.json"
    )
    raise SystemExit(
        main(
            [
                str(default_manifest),
                "--repo-root",
                str(repository_root),
                "--require-analysis-ready",
            ]
        )
    )
