"""Audit a larger IBL/OpenAlyx registered mouse panel.

The previous mouse gates leave a sharper equation:

    y ~ X_task + R_hybrid + U_current

is supported at the 5-candidate level, while

    U_lag | X_task, R_hybrid, U_current

does not pass the strict temporal GLM gate.  Before adding a more elaborate
directed coupling model, this audit builds a larger registered candidate panel
that can repeat the current mixed-readout tests.

This script is metadata-only.  It does not download spike times or trial arrays.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from mouse_ibl_neuropixels_audit import (
    CHANNEL_REGION_DATASET_TYPES,
    CORE_DATASET_TYPES,
    OPENALYX_URL,
    PROJECT,
    STRICT_REGION_DATASET_TYPES,
)
from mouse_ibl_cross_session_region_generalization_gate import CANDIDATES as SEEDED_CANDIDATES


RESULT_JSON = Path(__file__).with_name("mouse_ibl_larger_registered_panel_audit_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_larger_registered_panel_audit_report.md")


@dataclass
class RegisteredCandidate:
    eid: str
    session_ref: str
    lab: str
    subject: str
    date: str
    probe_collections: list[str]
    strict_probe_collections: list[str]
    channel_probe_collections: list[str]
    metrics_probe_collections: list[str]
    has_trial_table: bool
    has_wheel_timestamps: bool
    has_wheel_position: bool
    probe_count: int
    strict_probe_count: int
    channel_probe_count: int
    metrics_probe_count: int
    mixed_readout_ready: bool
    temporal_glm_ready: bool
    panel_score: int


def probe_collection(dataset: str) -> str | None:
    parts = dataset.split("/")
    if len(parts) >= 3 and parts[0] == "alf" and parts[1].startswith("probe"):
        if parts[2] == "pykilosort":
            return "/".join(parts[:3])
    return None


def collections_with(datasets: list[str], pattern: str) -> set[str]:
    return {
        collection
        for dataset in datasets
        if pattern in dataset
        for collection in [probe_collection(dataset)]
        if collection is not None
    }


def has_any(datasets: list[str], pattern: str) -> bool:
    return any(pattern in dataset for dataset in datasets)


def session_reference(one, eid) -> str:
    path = str(one.eid2path(eid))
    marker = "openalyx.internationalbrainlab.org/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path


def parse_session_ref(session_ref: str) -> tuple[str, str, str]:
    parts = session_ref.split("/")
    lab = parts[0] if len(parts) > 0 else "unknown"
    subject = parts[2] if len(parts) > 2 else "unknown"
    date = parts[3] if len(parts) > 3 else "unknown"
    return lab, subject, date


def candidate_from_eid(one, eid) -> RegisteredCandidate | None:
    datasets = [str(item) for item in one.list_datasets(eid, query_type="remote")]

    spike_times = collections_with(datasets, "spikes.times.npy")
    spike_clusters = collections_with(datasets, "spikes.clusters.npy")
    strict_regions = collections_with(datasets, "clusters.brainLocationAcronyms_ccf_2017")
    cluster_channels = collections_with(datasets, "clusters.channels")
    channel_regions = collections_with(datasets, "channels.brainLocationIds_ccf_2017")
    metrics = collections_with(datasets, "clusters.metrics.pqt")

    strict_probe_collections = sorted(spike_times & spike_clusters & strict_regions)
    channel_probe_collections = sorted(
        spike_times & spike_clusters & cluster_channels & channel_regions
    )
    probe_collections = sorted(set(strict_probe_collections) | set(channel_probe_collections))
    metrics_probe_collections = sorted(set(probe_collections) & metrics)

    has_trial_table = has_any(datasets, "_ibl_trials.table.pqt")
    has_wheel_timestamps = has_any(datasets, "_ibl_wheel.timestamps.npy")
    has_wheel_position = has_any(datasets, "_ibl_wheel.position.npy")

    mixed_readout_ready = bool(
        has_trial_table
        and has_wheel_timestamps
        and has_wheel_position
        and channel_probe_collections
    )
    temporal_glm_ready = bool(mixed_readout_ready and len(channel_probe_collections) >= 1)
    if not mixed_readout_ready:
        return None

    session_ref = session_reference(one, eid)
    lab, subject, date = parse_session_ref(session_ref)
    panel_score = (
        100 * len(channel_probe_collections)
        + 40 * len(strict_probe_collections)
        + 10 * len(metrics_probe_collections)
        + 5 * int(len(channel_probe_collections) >= 2)
        + 3 * int(has_wheel_position)
    )
    return RegisteredCandidate(
        eid=str(eid),
        session_ref=session_ref,
        lab=lab,
        subject=subject,
        date=date,
        probe_collections=probe_collections,
        strict_probe_collections=strict_probe_collections,
        channel_probe_collections=channel_probe_collections,
        metrics_probe_collections=metrics_probe_collections,
        has_trial_table=has_trial_table,
        has_wheel_timestamps=has_wheel_timestamps,
        has_wheel_position=has_wheel_position,
        probe_count=len(probe_collections),
        strict_probe_count=len(strict_probe_collections),
        channel_probe_count=len(channel_probe_collections),
        metrics_probe_count=len(metrics_probe_collections),
        mixed_readout_ready=mixed_readout_ready,
        temporal_glm_ready=temporal_glm_ready,
        panel_score=panel_score,
    )


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx larger registered panel audit",
        "",
        "## equation reset",
        "",
        "The current mouse equation is not a region-only or unit-only claim.",
        "The supported term is the mixed current-window readout:",
        "",
        "$$",
        "y_t^{(s)}",
        "\\sim",
        "X_t^{(s,\\mathrm{task})}",
        "+R_t^{(s,\\mathrm{hybrid})}",
        "+U_t^{(s)}\\mid X_t^{(s,\\mathrm{task})},R_t^{(s,\\mathrm{hybrid})}.",
        "$$",
        "",
        "The strict temporal extension remains a counterexample, not a promoted term:",
        "",
        "$$",
        "U_{t-\\ell}^{(s)}\\mid",
        "X_t^{(s,\\mathrm{task})},R_t^{(s,\\mathrm{hybrid})},U_t^{(s)}",
        "\\quad\\mathrm{not\\ yet\\ supported}.",
        "$$",
        "",
        "So the next step is a larger registered panel that can repeat the mixed-readout tests before fitting heavier directed/state-space coupling models.",
        "",
        "## metadata search",
        "",
        f"- OpenAlyx: `{output['openalyx_url']}`",
        f"- project: `{output['project']}`",
        f"- scanned sessions: {output['scanned_session_count']}",
        f"- mixed-readout ready sessions: {output['mixed_readout_ready_count']}",
        f"- temporal-GLM metadata ready sessions: {output['temporal_glm_ready_count']}",
        "",
        "Required metadata/files:",
        "",
        "| block | requirement |",
        "|---|---|",
        "| task/history \(X_t\) | `_ibl_trials.table.pqt` |",
        "| movement targets | `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy` |",
        "| current unit \(U_t\) | `spikes.times.npy`, `spikes.clusters.npy` |",
        "| hybrid region \(R_t\) | `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`; strict acronyms are kept when present |",
        "",
        "## top registered candidates",
        "",
        "| rank | eid | session | probes | strict probes | channel probes | metrics probes | score |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, candidate in enumerate(output["top_candidates"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{candidate['eid']}`",
                    f"`{candidate['session_ref']}`",
                    str(candidate["probe_count"]),
                    str(candidate["strict_probe_count"]),
                    str(candidate["channel_probe_count"]),
                    str(candidate["metrics_probe_count"]),
                    str(candidate["panel_score"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## lab and subject spread",
            "",
            "| item | count |",
            "|---|---:|",
        ]
    )
    for lab, count in output["lab_counts"]:
        lines.append(f"| lab `{lab}` | {count} |")
    for subject, count in output["subject_counts"][:20]:
        lines.append(f"| subject `{subject}` | {count} |")

    lines.extend(
        [
            "",
            "## next gate",
            "",
            f"- panel audit passed: `{output['panel_audit_passed']}`",
            f"- recommended candidate count: {output['recommended_candidate_count']}",
            "- next executable gate: repeat block-regularized mixed readout on this larger panel, then rerun strict temporal GLM only if the mixed term remains stable.",
            "",
        ]
    )
    return "\n".join(lines)


def seeded_eids() -> list[str]:
    seen = set()
    eids = []
    for candidate in SEEDED_CANDIDATES:
        eid = str(candidate["eid"])
        if eid not in seen:
            seen.add(eid)
            eids.append(eid)
    return eids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument(
        "--seeded-only",
        action="store_true",
        help="Audit the already validated 5-candidate panel without a remote session search.",
    )
    args = parser.parse_args()

    from one.api import ONE

    ONE.setup(base_url=OPENALYX_URL, silent=True)
    one = ONE(password="international")
    if args.seeded_only:
        eids = seeded_eids()
    else:
        search_types = CORE_DATASET_TYPES + CHANNEL_REGION_DATASET_TYPES
        eids = one.search(query_type="remote", project=PROJECT, dataset_types=search_types)

    candidates = []
    for eid in eids[: args.max_sessions]:
        candidate = candidate_from_eid(one, eid)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.panel_score,
            item.channel_probe_count,
            item.strict_probe_count,
            item.session_ref,
        ),
        reverse=True,
    )
    top_candidates = candidates[: args.max_candidates]
    lab_counts = Counter(candidate.lab for candidate in candidates).most_common()
    subject_counts = Counter(candidate.subject for candidate in candidates).most_common()

    output = {
        "openalyx_url": OPENALYX_URL,
        "project": PROJECT,
        "core_dataset_types": CORE_DATASET_TYPES,
        "strict_region_dataset_types": STRICT_REGION_DATASET_TYPES,
        "channel_region_dataset_types": CHANNEL_REGION_DATASET_TYPES,
        "searched_session_count": len(eids),
        "scanned_session_count": min(len(eids), args.max_sessions),
        "mixed_readout_ready_count": len(candidates),
        "temporal_glm_ready_count": sum(
            1 for candidate in candidates if candidate.temporal_glm_ready
        ),
        "recommended_candidate_count": len(top_candidates),
        "panel_audit_passed": len(top_candidates) >= 12,
        "top_candidates": [asdict(candidate) for candidate in top_candidates],
        "lab_counts": lab_counts,
        "subject_counts": subject_counts,
    }

    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(make_report(output))

    print("Mouse IBL/OpenAlyx larger registered panel audit")
    print(f"  searched sessions       = {len(eids)}")
    print(f"  scanned sessions        = {output['scanned_session_count']}")
    print(f"  mixed-readout ready     = {len(candidates)}")
    print(f"  recommended candidates  = {len(top_candidates)}")
    print(f"  panel_audit_passed      = {output['panel_audit_passed']}")
    if top_candidates:
        print(f"  top eid                 = {top_candidates[0].eid}")
        print(f"  top session             = {top_candidates[0].session_ref}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
