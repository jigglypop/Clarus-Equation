"""Audit IBL/OpenAlyx strict sessions for motor/striatal probe coverage.

The first mouse decision/action gate passed on a thalamic/visual/hippocampal
probe.  This audit scans the strict OpenAlyx sessions with small cluster-level
files only, looking for probes that contain motor cortex or striatal-complex
labels before downloading more spike arrays.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from mouse_ibl_neuropixels_audit import (
    CORE_DATASET_TYPES,
    OPENALYX_URL,
    PROJECT,
    STRICT_REGION_DATASET_TYPES,
)
from mouse_ibl_region_decision_action_gate import region_family


RESULT_JSON = Path(__file__).with_name("mouse_ibl_motor_striatum_audit_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_motor_striatum_audit_report.md")
TARGET_FAMILIES = {
    "motor_cortex",
    "striatal_complex",
    "septal_subpallium",
    "basal_ganglia_output",
}


@dataclass
class ProbeCandidate:
    eid: str
    session_ref: str
    collection: str
    target_cluster_count: int
    target_spike_count_from_metrics: int
    family_cluster_counts: dict[str, int]
    family_spike_counts_from_metrics: dict[str, int]
    top_acronym_cluster_counts: list[tuple[str, int]]
    top_acronym_spike_counts_from_metrics: list[tuple[str, int]]


def probe_collection(dataset: str) -> str | None:
    parts = dataset.split("/")
    if len(parts) >= 3 and parts[0] == "alf" and parts[1].startswith("probe"):
        if parts[2] == "pykilosort":
            return "/".join(parts[:3])
    return None


def session_reference(one, eid) -> str:
    path = str(one.eid2path(eid))
    marker = "openalyx.internationalbrainlab.org/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path


def family_counts_from_acronyms(acronyms) -> Counter:
    return Counter(region_family(acronym) for acronym in acronyms)


def spike_counts_from_metrics(metrics, acronyms) -> tuple[Counter, Counter]:
    family_counts = Counter()
    acronym_counts = Counter()
    if "cluster_id" not in metrics or "spike_count" not in metrics:
        return family_counts, acronym_counts
    for _, row in metrics.iterrows():
        cluster_id = int(row["cluster_id"])
        spike_count = int(row["spike_count"])
        acronym = str(acronyms[cluster_id]) if cluster_id < len(acronyms) else "unknown"
        family_counts[region_family(acronym)] += spike_count
        acronym_counts[acronym] += spike_count
    return family_counts, acronym_counts


def strict_probe_collections(one, eid) -> list[str]:
    datasets = [str(item) for item in one.list_datasets(eid, query_type="remote")]
    collections = {
        collection
        for dataset in datasets
        for collection in [probe_collection(dataset)]
        if collection and "clusters.brainLocationAcronyms_ccf_2017" in dataset
    }
    return sorted(collections)


def scan(max_candidates: int) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    strict_eids = one.search(
        query_type="remote",
        project=PROJECT,
        dataset_types=CORE_DATASET_TYPES + STRICT_REGION_DATASET_TYPES,
    )

    candidates: list[ProbeCandidate] = []
    for eid in strict_eids:
        session_ref = session_reference(one, eid)
        for collection in strict_probe_collections(one, eid):
            acronyms = one.load_dataset(
                eid,
                "clusters.brainLocationAcronyms_ccf_2017.npy",
                collection=collection,
                query_type="remote",
            )
            family_cluster_counts = family_counts_from_acronyms(acronyms)
            acronym_cluster_counts = Counter(str(acronym) for acronym in acronyms)
            family_spike_counts = Counter()
            acronym_spike_counts = Counter()
            try:
                metrics = one.load_dataset(
                    eid,
                    "clusters.metrics.pqt",
                    collection=collection,
                    query_type="remote",
                )
                family_spike_counts, acronym_spike_counts = spike_counts_from_metrics(
                    metrics,
                    acronyms,
                )
            except Exception:
                pass

            target_cluster_count = sum(
                family_cluster_counts[family] for family in TARGET_FAMILIES
            )
            target_spike_count = sum(
                family_spike_counts[family] for family in TARGET_FAMILIES
            )
            if target_cluster_count == 0 and target_spike_count == 0:
                continue
            candidates.append(
                ProbeCandidate(
                    eid=str(eid),
                    session_ref=session_ref,
                    collection=collection,
                    target_cluster_count=int(target_cluster_count),
                    target_spike_count_from_metrics=int(target_spike_count),
                    family_cluster_counts=dict(family_cluster_counts),
                    family_spike_counts_from_metrics=dict(family_spike_counts),
                    top_acronym_cluster_counts=acronym_cluster_counts.most_common(12),
                    top_acronym_spike_counts_from_metrics=acronym_spike_counts.most_common(12),
                )
            )

    candidates.sort(
        key=lambda candidate: (
            candidate.target_spike_count_from_metrics,
            candidate.target_cluster_count,
        ),
        reverse=True,
    )
    selected = candidates[:max_candidates]
    multi_probe_sessions = {}
    for candidate in candidates:
        entry = multi_probe_sessions.setdefault(
            candidate.eid,
            {
                "eid": candidate.eid,
                "session_ref": candidate.session_ref,
                "collections": [],
                "target_spike_count_from_metrics": 0,
                "target_cluster_count": 0,
                "families": set(),
            },
        )
        entry["collections"].append(candidate.collection)
        entry["target_spike_count_from_metrics"] += candidate.target_spike_count_from_metrics
        entry["target_cluster_count"] += candidate.target_cluster_count
        for family in TARGET_FAMILIES:
            if candidate.family_cluster_counts.get(family, 0):
                entry["families"].add(family)

    multi_probe_ranked = []
    for entry in multi_probe_sessions.values():
        if len(entry["collections"]) < 2:
            continue
        entry["families"] = sorted(entry["families"])
        multi_probe_ranked.append(entry)
    multi_probe_ranked.sort(
        key=lambda item: (
            len(item["families"]),
            item["target_spike_count_from_metrics"],
            item["target_cluster_count"],
        ),
        reverse=True,
    )

    return {
        "openalyx_url": OPENALYX_URL,
        "project": PROJECT,
        "strict_session_count": int(len(strict_eids)),
        "target_families": sorted(TARGET_FAMILIES),
        "candidate_probe_count": int(len(candidates)),
        "candidates": [asdict(candidate) for candidate in selected],
        "multi_probe_candidates": multi_probe_ranked[:max_candidates],
        "selected_next_gate": multi_probe_ranked[0] if multi_probe_ranked else None,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx motor-striatum audit",
        "",
        "첫 mouse strict-session gate가 thalamus/visual/hippocampus 중심이었으므로, 이번 audit는 strict session 안에서 motor cortex와 striatal-complex coverage를 찾는다.",
        "Spike arrays는 내려받지 않고 cluster acronym과 cluster metrics만 사용한다.",
        "",
        "## source",
        "",
        f"- OpenAlyx: `{output['openalyx_url']}`",
        f"- project: `{output['project']}`",
        f"- strict sessions scanned: {output['strict_session_count']}",
        f"- target families: `{', '.join(output['target_families'])}`",
        "",
        "## candidate probes",
        "",
        "| eid | session | collection | target clusters | target spikes from metrics | target family spike counts |",
        "|---|---|---|---:|---:|---|",
    ]
    for candidate in output["candidates"]:
        family_counts = {
            family: count
            for family, count in candidate["family_spike_counts_from_metrics"].items()
            if family in output["target_families"]
        }
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate['eid']}`",
                    f"`{candidate['session_ref']}`",
                    f"`{candidate['collection']}`",
                    str(candidate["target_cluster_count"]),
                    str(candidate["target_spike_count_from_metrics"]),
                    "`" + json.dumps(family_counts, sort_keys=True) + "`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## multi-probe candidates",
            "",
            "| eid | session | collections | target families | target clusters | target spikes from metrics |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for candidate in output["multi_probe_candidates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate['eid']}`",
                    f"`{candidate['session_ref']}`",
                    ", ".join(f"`{collection}`" for collection in candidate["collections"]),
                    "`" + ", ".join(candidate["families"]) + "`",
                    str(candidate["target_cluster_count"]),
                    str(candidate["target_spike_count_from_metrics"]),
                ]
            )
            + " |"
        )

    selected = output["selected_next_gate"]
    lines.extend(["", "## verdict", ""])
    if selected:
        lines.extend(
            [
                f"- selected next gate eid: `{selected['eid']}`",
                f"- session: `{selected['session_ref']}`",
                f"- collections: {', '.join(f'`{collection}`' for collection in selected['collections'])}",
                f"- families: `{', '.join(selected['families'])}`",
                "",
                "이 세션은 같은 behavioral trial table에서 motor cortex probe와 striatal-complex probe를 동시에 걸 수 있으므로, 다음 gate는 multi-probe region/action decoder다.",
            ]
        )
    else:
        lines.append("- no multi-probe motor/striatal candidate found")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()

    output = scan(args.max_candidates)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx motor-striatum audit")
    print(f"  strict sessions scanned={output['strict_session_count']}")
    print(f"  candidate probes={output['candidate_probe_count']}")
    selected = output["selected_next_gate"]
    if selected:
        print(f"  selected eid={selected['eid']}")
        print(f"  selected session={selected['session_ref']}")
        print(f"  selected collections={', '.join(selected['collections'])}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
