"""Audit whether IBL/OpenAlyx has a small mouse Neuropixels bridge.

This script does not download spike arrays.  It checks the public OpenAlyx
metadata for sessions that have the minimum datasets needed for the next
evolution gate:

    trials.table + wheel.timestamps + spikes.times + spikes.clusters
    + cluster or channel brain-region metadata

The result is a bridge audit, not a neural decoding result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


OPENALYX_URL = "https://openalyx.internationalbrainlab.org"
PROJECT = "ibl_neuropixel_brainwide_01"
RESULT_JSON = Path(__file__).with_name("mouse_ibl_neuropixels_audit_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_neuropixels_audit_report.md")

CORE_DATASET_TYPES = [
    "trials.table",
    "wheel.timestamps",
    "spikes.times",
    "spikes.clusters",
]

STRICT_REGION_DATASET_TYPES = [
    "clusters.brainLocationAcronyms_ccf_2017",
]

CHANNEL_REGION_DATASET_TYPES = [
    "clusters.channels",
    "channels.brainLocationIds_ccf_2017",
]


@dataclass
class Candidate:
    eid: str
    session_ref: str
    trial_table: list[str]
    wheel_timestamps: list[str]
    spikes_times: list[str]
    spikes_clusters: list[str]
    cluster_region_acronyms: list[str]
    cluster_channels: list[str]
    channel_region_ids: list[str]
    probe_collections: list[str]
    strict_ready: bool
    channel_region_ready: bool


def contains_any(dataset: str, patterns: Iterable[str]) -> bool:
    return any(pattern in dataset for pattern in patterns)


def find_datasets(datasets: list[str], *patterns: str) -> list[str]:
    return [dataset for dataset in datasets if contains_any(dataset, patterns)]


def probe_collection(dataset: str) -> str | None:
    parts = dataset.split("/")
    if len(parts) >= 3 and parts[0] == "alf" and parts[1].startswith("probe"):
        if len(parts) >= 4 and parts[2] == "pykilosort":
            return "/".join(parts[:3])
    return None


def session_reference(one, eid) -> str:
    path = str(one.eid2path(eid))
    marker = "openalyx.internationalbrainlab.org/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path


def candidate_from_eid(one, eid) -> Candidate:
    datasets = [str(item) for item in one.list_datasets(eid, query_type="remote")]
    trial_table = find_datasets(datasets, "_ibl_trials.table.pqt")
    wheel_timestamps = find_datasets(datasets, "_ibl_wheel.timestamps.npy")
    spikes_times = find_datasets(datasets, "spikes.times.npy")
    spikes_clusters = find_datasets(datasets, "spikes.clusters.npy")
    cluster_region_acronyms = find_datasets(
        datasets, "clusters.brainLocationAcronyms_ccf_2017"
    )
    cluster_channels = find_datasets(datasets, "clusters.channels")
    channel_region_ids = find_datasets(
        datasets, "channels.brainLocationIds_ccf_2017"
    )
    probe_collections = sorted(
        {
            collection
            for dataset in (
                spikes_times
                + spikes_clusters
                + cluster_region_acronyms
                + cluster_channels
                + channel_region_ids
            )
            for collection in [probe_collection(dataset)]
            if collection is not None
        }
    )
    strict_ready = bool(
        trial_table
        and wheel_timestamps
        and spikes_times
        and spikes_clusters
        and cluster_region_acronyms
    )
    channel_region_ready = bool(
        trial_table
        and wheel_timestamps
        and spikes_times
        and spikes_clusters
        and cluster_channels
        and channel_region_ids
    )
    return Candidate(
        eid=str(eid),
        session_ref=session_reference(one, eid),
        trial_table=trial_table,
        wheel_timestamps=wheel_timestamps,
        spikes_times=spikes_times,
        spikes_clusters=spikes_clusters,
        cluster_region_acronyms=cluster_region_acronyms,
        cluster_channels=cluster_channels,
        channel_region_ids=channel_region_ids,
        probe_collections=probe_collections,
        strict_ready=strict_ready,
        channel_region_ready=channel_region_ready,
    )


def make_report(output: dict) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx Neuropixels audit",
        "",
        "목표는 mouse Neuropixels/IBL 단계에서 region-level decision/action gate를 열 수 있는 최소 공개 데이터 bridge가 있는지 확인하는 것이다.",
        "이 audit는 spike array를 내려받지 않고 OpenAlyx metadata만 확인한다.",
        "",
        "## source",
        "",
        f"- OpenAlyx: `{output['openalyx_url']}`",
        f"- project: `{output['project']}`",
        "- ONE dataset type names are used for search; file names may include `_ibl_` prefixes and revisions.",
        "",
        "## bridge requirements",
        "",
        "| tier | required dataset types | meaning |",
        "|---|---|---|",
        f"| core | `{', '.join(output['core_dataset_types'])}` | trial, wheel, spike time/cluster bridge |",
        f"| strict region | `{', '.join(output['strict_region_dataset_types'])}` | cluster-level CCF acronym labels |",
        f"| channel fallback | `{', '.join(output['channel_region_dataset_types'])}` | channel CCF ids plus cluster-channel map |",
        "",
        "## search results",
        "",
        "| query | sessions found |",
        "|---|---:|",
        f"| core + strict cluster region | {output['strict_session_count']} |",
        f"| core + channel-region fallback | {output['channel_region_session_count']} |",
        "",
        "## first strict candidates",
        "",
        "| eid | session | probes | strict ready | channel fallback ready |",
        "|---|---|---|---|---|",
    ]
    for candidate in output["strict_candidates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate['eid']}`",
                    f"`{candidate['session_ref']}`",
                    ", ".join(f"`{probe}`" for probe in candidate["probe_collections"]),
                    str(candidate["strict_ready"]),
                    str(candidate["channel_region_ready"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## first candidate dataset coverage",
            "",
        ]
    )
    if output["strict_candidates"]:
        first = output["strict_candidates"][0]
        coverage_rows = [
            ("trial table", first["trial_table"]),
            ("wheel timestamps", first["wheel_timestamps"]),
            ("spikes times", first["spikes_times"]),
            ("spikes clusters", first["spikes_clusters"]),
            ("cluster region acronyms", first["cluster_region_acronyms"]),
            ("cluster channels", first["cluster_channels"]),
            ("channel region ids", first["channel_region_ids"]),
        ]
        lines.extend(["| item | files |", "|---|---|"])
        for name, files in coverage_rows:
            shown = "<br>".join(f"`{file}`" for file in files[:8])
            if len(files) > 8:
                shown += f"<br>... +{len(files) - 8} more"
            lines.append(f"| {name} | {shown} |")
    lines.extend(
        [
            "",
            "## verdict",
            "",
            f"- metadata bridge ready: `{output['metadata_bridge_ready']}`",
            f"- next gate: `{output['next_gate']}`",
            "",
            "이 결과는 mouse 단계의 방정식 항을 닫은 것이 아니라, 다음 region-binned decision/action decoding gate를 열 수 있음을 뜻한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates", type=int, default=5)
    args = parser.parse_args()

    from one.api import ONE

    ONE.setup(base_url=OPENALYX_URL, silent=True)
    one = ONE(password="international")

    strict_types = CORE_DATASET_TYPES + STRICT_REGION_DATASET_TYPES
    channel_types = CORE_DATASET_TYPES + CHANNEL_REGION_DATASET_TYPES

    strict_eids = one.search(
        query_type="remote",
        project=PROJECT,
        dataset_types=strict_types,
    )
    channel_eids = one.search(
        query_type="remote",
        project=PROJECT,
        dataset_types=channel_types,
    )

    strict_candidates = [
        candidate_from_eid(one, eid) for eid in strict_eids[: args.max_candidates]
    ]
    output = {
        "openalyx_url": OPENALYX_URL,
        "project": PROJECT,
        "core_dataset_types": CORE_DATASET_TYPES,
        "strict_region_dataset_types": STRICT_REGION_DATASET_TYPES,
        "channel_region_dataset_types": CHANNEL_REGION_DATASET_TYPES,
        "strict_session_count": len(strict_eids),
        "channel_region_session_count": len(channel_eids),
        "strict_candidates": [asdict(candidate) for candidate in strict_candidates],
        "metadata_bridge_ready": bool(strict_eids),
        "next_gate": "region_binned_decision_action_decoding"
        if strict_eids
        else "find_mouse_neuropixels_session_with_region_labels",
    }

    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(make_report(output))

    print("Mouse IBL/OpenAlyx Neuropixels audit")
    print(f"  strict sessions         = {len(strict_eids)}")
    print(f"  channel fallback        = {len(channel_eids)}")
    print(f"  metadata_bridge_ready   = {output['metadata_bridge_ready']}")
    if strict_candidates:
        print(f"  first eid               = {strict_candidates[0].eid}")
        print(f"  first session           = {strict_candidates[0].session_ref}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
