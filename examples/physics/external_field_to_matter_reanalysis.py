"""Recompute the public optical-pair and massive-pair measurements."""

from __future__ import annotations

from pathlib import Path

from reality_stone.clarus.external_field_to_matter import main


if __name__ == "__main__":
    default_snapshot = (
        Path(__file__).parents[2] / "benchmarks" / "external_field_to_matter_v1.json"
    )
    raise SystemExit(main([str(default_snapshot), "--require-external-reproduction"]))

