"""Readiness audit for moving the H0 selector from synthetic IO to real covariance.

The Fisher/covariance JSON interface is now reproducible, but the tracked
example bundle is synthetic.  This gate prevents over-promotion by separating:

* IO readiness: manifest/schema/regression/batch examples exist.
* Real-data readiness: at least one non-synthetic covariance/Fisher channel is
  tracked with provenance metadata and source URL.

The gate passes as an audit when the boundary is explicit; it does not claim
real covariance closure until the real-ready count is nonzero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BASE_DIR / "h0_fisher_io_examples"
MANIFEST = EXAMPLE_DIR / "manifest.json"
RESULT_JSON = BASE_DIR / "h0_fisher_real_readiness_results.json"
REPORT_MD = BASE_DIR / "h0_fisher_real_readiness_report.md"


@dataclass(frozen=True)
class ReadinessRow:
    channel_class: str
    expected_role: str
    required_source: str
    current_status: str
    next_action: str
    priority: int


ROADMAP = [
    ReadinessRow(
        channel_class="BAO+SN inverse distance ladder",
        expected_role="global standard-ruler closure",
        required_source="public covariance/compressed likelihood with ruler and SN nuisance roles",
        current_status="not tracked in Fisher JSON bundle",
        next_action="convert labelled BAO+SN covariance into observable/local/global role graph",
        priority=1,
    ),
    ReadinessRow(
        channel_class="SH0ES-style local distance ladder",
        expected_role="local calibrator endpoint closure",
        required_source="public ladder covariance with Cepheid/TRGB/SN calibration blocks",
        current_status="not tracked in Fisher JSON bundle",
        next_action="recover calibration graph instead of final scalar H0 only",
        priority=2,
    ),
    ReadinessRow(
        channel_class="GW standard sirens",
        expected_role="mixed distance-redshift bridge",
        required_source="event-level distance-redshift posterior or population covariance",
        current_status="synthetic smoke channel only",
        next_action="ingest event/posterior covariance and split distance vs redshift anchors",
        priority=3,
    ),
    ReadinessRow(
        channel_class="CMB acoustic-scale inference",
        expected_role="early global horizon closure",
        required_source="public parameter covariance/likelihood with acoustic-scale roles",
        current_status="synthetic global-horizon smoke channel only",
        next_action="map acoustic-scale covariance nodes to global horizon role",
        priority=4,
    ),
]


def load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest top-level JSON must be an object")
    return payload


def channel_file_ready(channel: dict[str, Any]) -> bool:
    file_name = str(channel.get("file", ""))
    return bool(file_name.endswith(".json") and (EXAMPLE_DIR / file_name).exists())


def main() -> int:
    manifest = load_manifest()
    channels = [channel for channel in manifest.get("channels", []) if isinstance(channel, dict)]
    synthetic = [channel for channel in channels if channel.get("source") == "synthetic"]
    real = [channel for channel in channels if channel.get("source") != "synthetic"]
    real_ready = [
        channel
        for channel in real
        if channel_file_ready(channel) and channel.get("source_url") and channel.get("matrix_role")
    ]
    files_ready = sum(channel_file_ready(channel) for channel in channels)

    payload = {
        "gate": "h0_fisher_real_readiness",
        "manifest": str(MANIFEST),
        "tracked_channels": len(channels),
        "tracked_channel_files_ready": files_ready,
        "synthetic_channels": len(synthetic),
        "real_channels": len(real),
        "real_ready_channels": len(real_ready),
        "io_ready": files_ready == len(channels) and len(channels) > 0,
        "real_data_boundary": len(real_ready) == 0,
        "roadmap": [asdict(row) for row in ROADMAP],
        "verdict": (
            "Fisher/covariance IO is ready, but real covariance closure is still a data boundary."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# H0 Fisher Real Readiness Gate",
        "",
        "## Manifest status",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| tracked channels | {payload['tracked_channels']} |",
        f"| tracked channel files ready | {payload['tracked_channel_files_ready']} |",
        f"| synthetic channels | {payload['synthetic_channels']} |",
        f"| real channels | {payload['real_channels']} |",
        f"| real-ready channels | {payload['real_ready_channels']} |",
        "",
        "## Real covariance roadmap",
        "",
        "| priority | channel class | expected role | required source | current status | next action |",
        "|---:|---|---|---|---|---|",
    ]
    for row in sorted(ROADMAP, key=lambda item: item.priority):
        lines.append(
            f"| {row.priority} | {row.channel_class} | {row.expected_role} | "
            f"{row.required_source} | {row.current_status} | {row.next_action} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
            "Do not promote the q-selector to a real covariance result until `real_ready_channels > 0`.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("# H0 Fisher Real Readiness Gate")
    print()
    print(f"tracked channels = {payload['tracked_channels']}")
    print(f"synthetic channels = {payload['synthetic_channels']}")
    print(f"real-ready channels = {payload['real_ready_channels']}")
    print(f"io_ready = {payload['io_ready']}")
    print(f"real_data_boundary = {payload['real_data_boundary']}")
    print()
    print("Verdict:", payload["verdict"])
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {RESULT_JSON}")

    if not payload["io_ready"]:
        raise SystemExit("Fisher/covariance IO bundle is not ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
