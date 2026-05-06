"""Promotion decision audit for H0 real covariance channels.

This gate applies the real-covariance requirements to the current manifest.
Synthetic examples may pass IO, but they must not be promoted to real
covariance evidence.  Future non-synthetic channels must carry source metadata
and pass JSON ingestion before they can become real-ready.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from h0_fisher_io_validate_gate import validate_payload


BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BASE_DIR / "h0_fisher_io_examples"
MANIFEST = EXAMPLE_DIR / "manifest.json"
RESULT_JSON = BASE_DIR / "h0_real_covariance_promotion_decision_results.json"
REPORT_MD = BASE_DIR / "h0_real_covariance_promotion_decision_report.md"


@dataclass(frozen=True)
class DecisionRow:
    file: str
    source: str
    channel_class: str
    io_valid: bool
    has_source_url: bool
    has_version_pin: bool
    promotion_status: str
    blockers: tuple[str, ...]


def manifest_payload() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest top-level JSON must be an object")
    return payload


def version_pinned(channel: dict[str, Any], manifest: dict[str, Any]) -> bool:
    del manifest
    return bool(
        channel.get("commit")
        or channel.get("release")
        or channel.get("dataset_date")
        or channel.get("version")
    )


def decide(channel: dict[str, Any], manifest: dict[str, Any]) -> DecisionRow:
    file_name = str(channel.get("file", ""))
    source = str(channel.get("source", ""))
    channel_path = EXAMPLE_DIR / file_name
    blockers: list[str] = []

    if not channel_path.exists():
        blockers.append("channel file missing")
        io_valid = False
    else:
        payload = json.loads(channel_path.read_text(encoding="utf-8"))
        errors = validate_payload(payload) if isinstance(payload, dict) else ["top-level JSON must be an object"]
        io_valid = not errors
        if errors:
            blockers.append("IO validation failed: " + "; ".join(errors))

    if source == "synthetic":
        blockers.append("synthetic source")

    has_source_url = bool(channel.get("source_url"))
    if source != "synthetic" and not has_source_url:
        blockers.append("missing source_url")

    has_version_pin = version_pinned(channel, manifest)
    if source != "synthetic" and not has_version_pin:
        blockers.append("missing version pin")

    if source != "synthetic" and not channel.get("matrix_role"):
        blockers.append("missing matrix_role")

    promotion_status = "real-ready" if source != "synthetic" and io_valid and not blockers else "not-promoted"
    return DecisionRow(
        file=file_name,
        source=source,
        channel_class=str(channel.get("channel_class", "")),
        io_valid=io_valid,
        has_source_url=has_source_url,
        has_version_pin=has_version_pin,
        promotion_status=promotion_status,
        blockers=tuple(blockers),
    )


def main() -> int:
    manifest = manifest_payload()
    channels = [channel for channel in manifest.get("channels", []) if isinstance(channel, dict)]
    rows = [decide(channel, manifest) for channel in channels]
    real_ready = [row for row in rows if row.promotion_status == "real-ready"]
    synthetic_promoted = [
        row for row in rows if row.source == "synthetic" and row.promotion_status == "real-ready"
    ]
    passed = not synthetic_promoted and all(row.io_valid for row in rows)

    payload = {
        "gate": "h0_real_covariance_promotion_decision",
        "passed": passed,
        "channel_count": len(rows),
        "real_ready_count": len(real_ready),
        "rows": [asdict(row) for row in rows],
        "verdict": (
            "Current channels pass IO but are not promoted as real covariance evidence because they are synthetic."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# H0 real covariance promotion decision gate",
        "",
        f"- passed: `{passed}`",
        f"- channels: {len(rows)}",
        f"- real-ready: {len(real_ready)}",
        "",
        "| file | source | class | IO valid | source URL | version pin | promotion | blockers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        blockers = ", ".join(row.blockers) if row.blockers else ""
        lines.append(
            f"| `{row.file}` | `{row.source}` | `{row.channel_class}` | `{row.io_valid}` | "
            f"`{row.has_source_url}` | `{row.has_version_pin}` | `{row.promotion_status}` | {blockers} |"
        )
    lines.extend(["", "## Verdict", "", payload["verdict"], ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"passed": passed, "real_ready_count": len(real_ready)}, indent=2))
    if not passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
